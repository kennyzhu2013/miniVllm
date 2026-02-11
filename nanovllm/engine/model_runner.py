import pickle
import platform
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.memory import get_gpu_memory

# 延迟导入模型，避免在仅导入包时因外部依赖缺失而失败
try:
    from nanovllm.models.qwen3 import Qwen3ForCausalLM  # type: ignore
except Exception:
    Qwen3ForCausalLM = None  # 在 __init__ 中再尝试加载并提供友好错误提示

from nanovllm.layers.sampler import Sampler
from nanovllm.utils.loader import load_model


# 用于在分布式环境中运行一个语言模型（这里主要针对Qwen3）。
# 这个类负责初始化模型、管理分布式进程、分配KV缓存、运行模型的前向传播（包括预填充和解码阶段）以及采样生成下一个token。
class ModelRunner: 

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size  # 默认256，一个block块大小(通常是一组多头组合，一层有多个block).
        self.enforce_eager = config.enforce_eager  # 是否强制使用 eager 模式，默认 False,为ture则调用`capture_cudagraph`方法捕获CUDA图，以优化推理速度
        self.world_size = config.tensor_parallel_size  # 进程数，如果>1，则创建或附加共享内存（SharedMemory）。主进程（rank0）创建共享内存，其他进程（rank>0）则进入循环等待命令。
        self.rank = rank
        self.event = event

        #  # 分布式后端选择：Windows 使用 gloo；Linux/Mac 使用 nccl
        backend = "nccl" if platform.system() != "Windows" else "gloo"
        if platform.system() == "Windows" and self.world_size > 1:  # windows上单进程避免windows报错
            raise RuntimeError(
                "Tensor parallel (>1) is not supported on Windows. "
                "Please set tensor_parallel_size=1 or run under WSL/Linux with NCCL."
            )

        # 初始化进程组：单卡无需分布式初始化，避免 Windows 上 libuv 相关问题，实际上写成1不会进入if语句，所以直接init_method直接传入tcp
        if self.world_size > 1:
            # init_process_group需要一个 rendezvous（会合点） 机制，让所有 rank 在同一个地方登记、交换地址信息，然后再建立通信。
            # init_method 就是在指定这个 rendezvous 机制，使用 env://（从环境变量读取 rendezvous 信息） 或 file:// 以提高跨平台兼容性，如- MASTER_ADDR （rank0 的地址）
            # - MASTER_PORT （端口），你需要在启动各个进程前把 MASTER_ADDR/MASTER_PORT 设好（很多 launcher： torchrun 、SLURM、K8s 等会自动设置）
            # 说明：Windows 官方 PyTorch 多数未启用 libuv，tcp:// 可能触发 TCPStore/use_libuv 报错
            # dist.init_process_group(backend, init_method="env://", world_size=self.world_size, rank=rank)
            # 参数为tcp://localhost:2333， 含义：显式指定一个 TCPStore 地址（ host:port ），rank0 通常作为 store server，其它 rank 连接过去
            dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
            torch.cuda.set_device(rank)
        else:
            torch.cuda.set_device(0)

        default_dtype = torch.get_default_dtype()
        dtype = resolve_torch_dtype(getattr(hf_config, "dtype", getattr(hf_config, "torch_dtype", torch.float16)))
        prop = torch.cuda.get_device_properties(rank)
        if dtype == torch.bfloat16 and prop.major < 8:
            dtype = torch.float16
        torch.set_default_dtype(dtype)
        torch.set_default_device("cuda")

        # 加载模型（`Qwen3ForCausalLM`），并调用`load_model`加载预训练权重, 报错更精准
        global Qwen3ForCausalLM
        if Qwen3ForCausalLM is None:
            try:
                from nanovllm.models.qwen3 import Qwen3ForCausalLM as _Qwen3ForCausalLM
                Qwen3ForCausalLM = _Qwen3ForCausalLM
            except Exception as e:
                raise ImportError(
                    "Failed to import Qwen3 model. Ensure your Transformers version supports Qwen3Config "
                    "and that all optional GPU dependencies (flash-attn, triton) are installed."
                ) from e
        
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # 模拟最大负载的运行，来测量模型在推理过程中所需的显存峰值，确保系统知道模型计算本身（权重+激活）最多需要吃掉多少显存，从而避免在后续分配 KV Cache 时分配过多导致 OOM（显存溢出）。
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()  # 调用`capture_cudagraph`方法捕获CUDA图，以优化推理速度
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # 主进程创建或附加共享内存（SharedMemory）
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()

    # 在非主进程中，循环读取共享内存中的命令（方法名和参数），并调用相应的方法。当收到"exit"命令时退出循环
    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    # 非主进程从共享内存中读取方法名和参数（使用pickle序列化）            
    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    # 主进程：将方法名和参数写入共享内存（使用pickle序列化），并通知其他进程
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # 主进程将方法调用写入共享内存，然后所有进程（包括主进程）执行该方法
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        assert callable(method)  # 更安全确保是函数.
        return method(*args)

    # - 虽然这个函数本身没有返回值，但它产生的副作用（更新了 CUDA 的 Peak Memory 统计）至关重要。
    #- 紧接着调用的 allocate_kv_cache 方法（ model_runner.py:L145 ）会读取 torch.cuda.memory_stats()["allocated_bytes.all.peak"] ，用总显存减去这个峰值，剩下的显存才会被安全地划分为 KV Cache Block。
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    # 根据GPU内存利用率计算可用的显存
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # 兼容无 num_key_value_heads 配置的模型
        total_kv_heads = getattr(hf_config, "num_key_value_heads", getattr(hf_config, "num_attention_heads"))
        num_kv_heads = total_kv_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // total_kv_heads)
        dtype = resolve_torch_dtype(getattr(hf_config, "dtype", getattr(hf_config, "torch_dtype", torch.float16)))
        # 这里2 代表 key 和 value，blocks × block_size 就是能缓存的最大上下文长度
        # hf_config.torch_dtype.itemsize为dtype_size，数据类型的大小，通常以字节为单位。例如，float32 数据类型的大小为 4 字节
        # 计算每个KV缓存块的大小（`block_bytes`），然后计算可分配的块数（`num_kvcache_blocks`）
        # 把同一大块 kv_cache 索引分片后，mounted 到不同层，避免重复 alloc，提高缓存命中
        # itemsize = torch.zeros((), dtype=dtype).element_size()，
        # 直接使用 dtype.itemsize 更高效，避免了创建临时张量的开销
        itemsize = dtype.itemsize
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * itemsize

        # 精准控制确保 num_kvcache_blocks 大于 0 ，避免缓存为空： 
        # 计算每个KV缓存块的大小（`block_bytes`），然后计算可分配的块数（`num_kvcache_blocks`）
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        #  把同一大块 kv_cache 索引分片后，mounted 到不同层，避免重复 alloc，提高缓存命中，直接empty不初始化
        # hf_config.num_hidden_layers为隐藏层数
        # num_kvcache_blocks这个参数表示键值缓存（key-value cache）块的数量
        # 创建的张量的形状为：
        # Shape=(2,num_hidden_layers,num_kvcache_blocks,block_size,num_kv_heads,head_dim)
        # 维度索引	维度大小	含义描述
        # 0	2	K/V 分离：0 通常代表 Key 缓存，1 代表 Value 缓存。
        # 1	hf_config.num_hidden_layers	层数：模型每一层都有自己独立的 KV Cache。
        # 2	config.num_kvcache_blocks	块数量：这是 PagedAttention（分块注意力机制）的核心，表示预分配的总物理块数。
        # 3	self.block_size	块大小：每个物理块能容纳的 Token 数量（如 8 或 16）。
        # 4	num_kv_heads	KV 头数：对应 Multi-Query 或 Grouped-Query Attention 中的头数。
        # 5	head_dim	每个头的维度：每个 Attention Head 的向量长度（如 128）。
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim, dtype=dtype)
        layer_id = 0
        # 绑定到每个 attention 模块：遍历模型的所有模块，将每个注意力层的`k_cache`和`v_cache`设置为KV缓存的相应切片
        # 最终缓存的是历史所有 token 的 每层 attention 的 Key + Value 向量
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # 决定“计算 Attention 时去哪里读历史 KV”
    # 高效做批处理推理（batch inference）： 为seqs分配block table，每个seq的block table长度相同，不足的用-1填充
    def prepare_block_tables(self, seqs: list[Sequence]):
        # 计算所有序列的块表（block_table）的最大长度，每个 Sequence 对象都有一个 block_table 属性
        max_len = max(len(seq.block_table) for seq in seqs)
        # 为每个序列的块表（block_table）填充 -1，使所有块表长度相同
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # 将填充后的块表转换为 PyTorch 的 CUDA 张量， non_blocking=True 允许在数据传输时进行其他计算，进一步提高效率
        # pin_memory=True，在 CPU 内存中分配 页锁定内存（pinned memory），页锁定内存不会被操作系统换出到磁盘，因此数据传输到 GPU 更快，常用于 DataLoader 或需要频繁把数据拷贝到 GPU 的场景。
        # 要把 多个不同长度的 block_table 拼成一个二维 Tensor，送给 GPU kernel 批量处理，走统一的并行 kernel。
        # 但 PyTorch/TensorCore 的内核要求张量是规则的二维结构——也就是：[ batch_size, max_num_blocks_per_seq ]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # 处理预填充阶段（即第一次处理输入提示）的输入
    # 处理新输入的序列：
    #   拼接所有序列的未处理 token (input_ids)
    #   生成位置索引 (positions)
    #   计算序列长度信息 (cu_seqlens_q/k, max_seqlen_q/k)
    # 槽位映射：
    #   slot_mapping：将 token 映射到 KV 缓存的物理位置
    #   对已缓存的块建立块表 (block_tables)
    # 收集多个 Sequence 中尚未缓存的 token，构造模型输入张量，并设置上下文信息供 CUDA Kernel 使用。
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])  # num_cached_tokens表示前面已经被缓存（prefill）部分的长度
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))  # 位置索引
            seqlen_q = seqlen - seq.num_cached_tokens  # q只考虑新增token
            seqlen_k = seqlen  # k考虑所有token
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup，因为 warmup 阶段 block_table 为空
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):  # 遍历当前序列的 未缓存 block，建立 token → KV Cache 位置的映射：
                start = seq.block_table[i] * self.block_size  #  seq.block_table[i] 是 block 在全局 KV cache 中的位置 
                if i != seq.num_blocks - 1:
                    end = start + self.block_size   # 每块 block 有 block_size 个槽，每个槽对应一个 token
                else:
                    end = start + seq.last_block_num_tokens   # 倒数第二块可能是部分填充，需要去掉填充位置
                slot_mapping.extend(list(range(start, end)))    # 每个seq的token到kv cache完整填充位置，extend 会展开加入变成一维
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    #  prefix cache，有缓存历史，需要建立 block table，将每个seq的block table填充到最大长度
            block_tables = self.prepare_block_tables(seqs)  # 记录了每个序列的所有 Block（包括历史缓存的和新分配的）在显存池中的 ID。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 这一步类似于为 GPU kernel 提供“调度信息表”： 哪个 token 该写入哪里、对应哪个序列、使用哪些 block 等等。
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    # 解码准备阶段： 单步处理：构建解码（Decode）输入，在每轮生成一个 token 时使用，用于增量生成阶段
    #   每个序列仅最后 1 个 token (input_ids = [seq.last_token])
    #   位置 = 序列长度 (positions = [len(seq)])
    #   槽位 = 最后一个 token 的缓存位置
    # 块表维护：每个序列的块索引列表
    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        # is_prefill=False	表示当前是 decode 模式
        # 此上下文信息会绑定到 thread-local CUDA kernel 配置中，在后续的 run_model() 调用时起作用
        # 最后生成的数据结构：所有结构都为 CUDA batch decode 设计（匹配 tensor 并行 & CUDAGraph）
        # +---------------------+     +----------------------+
        # |  input_ids          | --> |  [token_t0, token_t1, ...]    |
        # +---------------------+     +----------------------+
        # |  positions          | --> |  [pos_t0,   pos_t1,  ...]     |
        # +---------------------+     +----------------------+
        # |  slot_mapping       | --> |  [kvcache_idx0, ...]          |
        # |  context_lens       | --> |  [len0, len1, ...]            |
        # |  block_tables       | --> |  [[b0, b1, ...], [...]]       |
        # +---------------------+     +----------------------+
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    # 为每个序列准备温度参数（浮点值，用于采样），并转换为CUDA张量
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"].fill_(-1)  # 建议显式初始化为-1表示无效位置
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    # 解码阶段的数据流与 prepare_* 模块协作（如果prefill为false（rank=0））
    # ┌──────────────┐
    # │ Sequence 对象│
    # └────┬─────────┘
    #      │ 包含：last_token、block_table、block_size、last_block_num_tokens、temperature 等
    #      ▼
    # ┌──────────────────────────────┐
    # │    prepare_decode(seqs)      │
    # │ ───────────────              │
    # │ 获取 input_ids ← last_token  │
    # │ 获取 positions  ← len(seq)   │
    # │ 计算 slot_mapping            │
    # │ 构建 block_table（prefix）   │
    # │ 设置上下文 set_context       │
    # └────┬─────────────────────────┘
    #      ▼
    # ┌──────────────────────────────┐
    # │    prepare_sample(seqs)      │
    # │ ───────────────              │
    # │ 收集每条 sequence 的温度参数 │
    # │ 构建 temperature 张量         │
    # └────┬─────────────────────────┘
    #      ▼
    # ┌──────────────────────────────┐
    # │       run_model(...)         │ ◄────────────┐
    # │ ───────────────              │              │
    # │ 若使用 CUDA 图 → 执行 Graph Replay         │
    # │ 生成 logits（每个 token 的概率分布）       │
    # └────┬─────────────────────────┘              │
    #      ▼                                         │
    # ┌──────────────────────────────┐              │
    # │        Sampler(logits)       │              │
    # │ ───────────────              │              │
    # │ 使用 temperature 采样出下个 token ID ←──────┘
    # └────┬─────────────────────────┘
    #      ▼
    # ┌──────────────────────────────┐
    # │ 写入 Sequence 对象：         │
    # │ - token 追加                 │
    # │ - block_table 扩容（如有需要）│
    # │ - last_block_num_tokens += 1 │
    # └──────────────────────────────┘
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)  
        # Rank 0 使用 Sampler 选择下一个 token，其他 Rank 返回 None（仅主进程负责采样）
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    # - 关键限制： 图捕获时用到的张量对象（地址/shape/stride）会被“绑定” 。回放时不能换一套新 tensor，只能往同一块预分配的 tensor 里写新数据
    # 避免运行时内核启动开销，尤其加速小 batch 推理
    # 静态图重放： 在模型初始化阶段捕获一条计算路径（例如 decode 批大小为 8），封装成一张“图”，之后每次 decode 都只需传入数据 → 直接播放图 → 得到结果。
    # 再也不需要 Python kernel 调用、PyTorch 动态构图或调度器调参与干预。
    #
    # 场景	是否适合 Graph	原因
    # prefill()	❌ 各序列长度差异大，kernel 输入 shape 动态	无法复用图结构
    # decode()	✅ 每次输入都是 1 token，shape 固定，可建静态图	完美适配 replay 模式
    @torch.inference_mode()
    def capture_cudagraph(self):  
        # 这段代码是在 capture CUDA Graph 期间“冻结 CUDA 随机数状态（RNG state）” ，避免随机性相关的 CUDA 算子在录图/回放时出现不一致或直接报错。
        get_rng_state = torch.cuda.get_rng_state
        set_rng_state = torch.cuda.set_rng_state
        rng_state = torch.cuda.get_rng_state()
        torch.cuda.get_rng_state = lambda: rng_state
        torch.cuda.set_rng_state = lambda _: None

        # 主要准备/绑定了两类东西： (A) decode 路径需要的“静态输入/上下文张量” ，以及 (B) 模型前向的输出缓冲和图对象本身 。录完之后，它们基本都会 常驻（至少在本进程生命周期内）
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # 把 slot_mapping/context_lens/block_tables （切到 [:bs] ）塞进线程上下文，后续模型前向调用时会从这里读取
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 静态图绑定结构：
        # ┌────────────┐      ┌──────────────┐
        # │ input_ids  │◄────▶input_ids     │
        # │ positions  │◄────▶positions     │
        # │ ...        │      │ ...          │
        # │ outputs    │─────▶outputs       │
        # └────────────┘      └──────────────┘
        #  （graph_vars）         （图内部使用）
        #
        # 每轮调用：
        # graph_vars["input_ids"][:] ← 新输入值
        # graph_vars["positions"][:] ← ...
        # graph.replay()            ← 使用旧张量执行新数据
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        torch.cuda.get_rng_state = get_rng_state
        torch.cuda.set_rng_state = set_rng_state


# 将可能为字符串的 dtype（来自原始 config.json）解析为 torch.dtype
def resolve_torch_dtype(v):
    if isinstance(v, torch.dtype):
        return v
    if isinstance(v, str):
        m = {
            "float16": torch.float16,
            "half": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
        }
        return m.get(v.lower(), torch.float16)
    return torch.float16