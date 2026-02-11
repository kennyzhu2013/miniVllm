import torch
from torch import nn
import torch.nn.functional as F

# 可选依赖：triton 与 flash-attn，若不存在则走纯 PyTorch 回退实现
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    _HAS_FLASH_ATTN = True
except Exception:
    _HAS_FLASH_ATTN = False

try:
    import xformers.ops as xops
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
    _HAS_XFORMERS = True
except ImportError:
    _HAS_XFORMERS = False
from nanovllm.utils.context import get_context

# 支持 HAS_TRITON 时，这段的区别几乎都体现在 “把 (key,value) 写入 KVCache 的实现方式” 上：
# 同样的逻辑（按 slot_mapping 把每个 token 的 K/V 写到 KVCache 的对应槽位），Triton 路径用自定义 GPU kernel，回退路径用 PyTorch 的张量操作。
if HAS_TRITON:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        # 更快 ：写 KVCache 这一步通常是 decode 热路径，Triton kernel 往往比 index_copy_ 更高效（更少框架开销、更可控的内存访问）。
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


    def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
else:
    def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
        # 纯 PyTorch 回退：按 slot_mapping 将 (N, H, D) 写入缓存的行视图 (slots, H*D)
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert slot_mapping.numel() == N
        key_flat = key.reshape(N, D)
        value_flat = value.reshape(N, D)
        k_flat = k_cache.reshape(-1, D)
        v_flat = v_cache.reshape(-1, D)
        if key_flat.dtype != k_flat.dtype:
            key_flat = key_flat.to(k_flat.dtype)
            value_flat = value_flat.to(v_flat.dtype)
        k_flat.index_copy_(0, slot_mapping.to(torch.long), key_flat)
        v_flat.index_copy_(0, slot_mapping.to(torch.long), value_flat)

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def _repeat_kv(self, x: torch.Tensor):
        if self.num_heads == self.num_kv_heads:
            return x
        g = self.num_heads // self.num_kv_heads
        return x.unsqueeze(2).repeat(1, 1, g, 1).view(x.size(0), self.num_heads, self.head_dim)

    def _run_xformers_varlen(self, q, k, v, context):
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        
        # Convert cumulative seqlens to individual seqlens list for xformers
        q_seqlen = (context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]).cpu().tolist()
        kv_seqlen = (context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]).cpu().tolist()
        
        attn_bias = BlockDiagonalCausalMask.from_seqlens(
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen
        )
        o = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=self.scale)
        return o.squeeze(0)

    def _prefill_xformers(self, q, k, v, context):
        k_cache, v_cache = self.k_cache, self.v_cache
        block_tables = context.block_tables
        block_size = k_cache.size(1)
        o = torch.empty_like(q)
        for i in range(context.cu_seqlens_q.size(0) - 1):
            sq0 = int(context.cu_seqlens_q[i].item())
            sq1 = int(context.cu_seqlens_q[i + 1].item())
            sk0 = int(context.cu_seqlens_k[i].item())
            sk1 = int(context.cu_seqlens_k[i + 1].item())
            qi = q[sq0:sq1]
            bt = block_tables[i]
            nb = int((bt >= 0).sum().item())
            last_tokens = sk1 - sk0 - (nb - 1) * block_size
            ks = []
            vs = []
            for j in range(nb):
                bidx = int(bt[j].item())
                if j == nb - 1:
                    ks.append(k_cache[bidx, :last_tokens])
                    vs.append(v_cache[bidx, :last_tokens])
                else:
                    ks.append(k_cache[bidx])
                    vs.append(v_cache[bidx])
            ki = torch.cat(ks, dim=0)
            vi = torch.cat(vs, dim=0)
            ki = self._repeat_kv(ki)
            vi = self._repeat_kv(vi)
            Lq = qi.size(0)
            Lk = ki.size(0)
            cached = Lk - Lq
            q_abs = torch.arange(cached, cached + Lq, device=q.device)
            k_abs = torch.arange(Lk, device=q.device)
            mask = (k_abs[None, :] > q_abs[:, None]).unsqueeze(0)
            attn_bias = torch.zeros((1, Lq, Lk), device=q.device, dtype=q.dtype)
            attn_bias.masked_fill_(mask, float('-inf'))
            oi = xops.memory_efficient_attention(qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0), attn_bias=attn_bias, scale=self.scale)
            o[sq0:sq1] = oi.squeeze(0)
        return o

    def _decode_xformers(self, q, context):
        k_cache, v_cache = self.k_cache, self.v_cache
        block_tables = context.block_tables
        block_size = k_cache.size(1)
        bs = block_tables.size(0)
        o = torch.empty(q.size(0), self.num_heads, self.head_dim, device=q.device, dtype=q.dtype)
        for i in range(bs):
            nb = int((block_tables[i] >= 0).sum().item())
            Lk = int(context.context_lens[i].item())
            last_tokens = Lk - (nb - 1) * block_size
            ks = []
            vs = []
            for j in range(nb):
                bidx = int(block_tables[i, j].item())
                if j == nb - 1:
                    ks.append(k_cache[bidx, :last_tokens])
                    vs.append(v_cache[bidx, :last_tokens])
                else:
                    ks.append(k_cache[bidx])
                    vs.append(v_cache[bidx])
            ki = torch.cat(ks, dim=0)
            vi = torch.cat(vs, dim=0)
            ki = self._repeat_kv(ki)
            vi = self._repeat_kv(vi)
            qi = q[i].unsqueeze(0).unsqueeze(0)
            oi = xops.memory_efficient_attention(qi, ki.unsqueeze(0), vi.unsqueeze(0), scale=self.scale)
            o[i] = oi.squeeze(0).squeeze(0)
        return o

    # 增加是否支撑FLASH_ATTN或XFORMERS的判断
    # 返回的是三维 [tokens_or_batch, num_heads, head_dim]，调用地方flatten 后为 [tokens_or_batch, num_heads * head_dim]
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        def _gather_cache_for_seq(cache: torch.Tensor, blocks_row: torch.Tensor, seqlen: int) -> torch.Tensor:
            # cache: [num_blocks, block_size, num_kv_heads, head_dim]
            num_blocks, block_size, _, _ = cache.shape
            full_blocks = seqlen // block_size
            last_tokens = seqlen % block_size
            chunks = []
            for b in range(full_blocks):
                block_id = int(blocks_row[b].item())
                if block_id < 0:
                    break
                chunks.append(cache[block_id, :block_size])
            if last_tokens > 0:
                block_id = int(blocks_row[full_blocks].item())
                if block_id >= 0:
                    chunks.append(cache[block_id, :last_tokens])
            if len(chunks) == 0:
                return cache.new_zeros((0, cache.size(2), cache.size(3)))
            return torch.cat(chunks, dim=0)  # [seqlen, num_kv_heads, head_dim]

        def _attend(q_seq: torch.Tensor, k_seq: torch.Tensor, v_seq: torch.Tensor) -> torch.Tensor:
            # q_seq: [n_q, num_heads, head_dim]
            # k_seq/v_seq: [n_k, num_kv_heads, head_dim]
            if k_seq.dtype != q_seq.dtype:
                k_seq = k_seq.to(q_seq.dtype)
            if v_seq.dtype != q_seq.dtype:
                v_seq = v_seq.to(q_seq.dtype)
            n_q = q_seq.size(0)
            num_heads = q_seq.size(1)
            num_kv = k_seq.size(1)
            head_dim = k_seq.size(2)
            assert num_heads % max(1, num_kv) == 0
            group = max(1, num_heads // max(1, num_kv))
            outs = []
            for g in range(num_kv):
                q_g = q_seq[:, g * group:(g + 1) * group, :]  # [n_q, group, D]
                k_g = k_seq[:, g, :]  # [n_k, D]
                v_g = v_seq[:, g, :]  # [n_k, D]
                # [n_q*group, D] x [D, n_k] -> [n_q*group, n_k]
                q_flat = q_g.reshape(-1, head_dim)
                logits = (q_flat @ k_g.t()) * self.scale
                attn = F.softmax(logits, dim=-1)
                out = attn @ v_g  # [n_q*group, D]
                outs.append(out.reshape(n_q, group, head_dim))
            # 拼回所有头
            o_seq = torch.cat(outs, dim=1)  # [n_q, num_heads, head_dim]
            return o_seq    

        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            if _HAS_FLASH_ATTN:
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            elif _HAS_XFORMERS:
                if context.block_tables is None:
                    o = self._run_xformers_varlen(q, k, v, context)
                else:
                    o = self._prefill_xformers(q, k, v, context)
            else:
                # 逐序列切分计算（支持有/无 prefix cache 两种场景）
                cu_q = context.cu_seqlens_q.tolist()
                cu_k = context.cu_seqlens_k.tolist()
                blocks = context.block_tables
                outputs = []
                for i in range(len(cu_q) - 1):
                    q_i = q[cu_q[i]:cu_q[i + 1]]  # [n_q_i, H, D]
                    n_q_i = q_i.size(0)
                    # 收集 K/V：若存在 prefix cache（block_tables 非空），则从缓存聚合；否则直接使用当前 K/V 切片
                    if blocks is not None:
                        seqlen_k_i = cu_k[i + 1] - cu_k[i]
                        k_i = _gather_cache_for_seq(k_cache, blocks[i], seqlen_k_i)
                        v_i = _gather_cache_for_seq(v_cache, blocks[i], seqlen_k_i)
                        # 允许每个查询看到的上下文长度：prefix_len + j + 1（因果掩码）
                        prefix_len = seqlen_k_i - n_q_i
                        prefix_len = max(0, int(prefix_len))
                    else:
                        k_i = k[cu_k[i]:cu_k[i + 1]]
                        v_i = v[cu_k[i]:cu_k[i + 1]]
                        prefix_len = 0
                    # 逐查询应用因果掩码：第 j 个查询仅关注到 prefix_len + j + 1 个键值
                    out_tokens = []
                    for j in range(n_q_i):
                        allowed = prefix_len + j + 1
                        k_slice = k_i[:allowed]
                        v_slice = v_i[:allowed]
                        o_j = _attend(q_i[j:j + 1], k_slice, v_slice)  # [1, H, D]
                        out_tokens.append(o_j)
                    o_i = torch.cat(out_tokens, dim=0)  # [n_q_i, H, D]
                    outputs.append(o_i)
                o = torch.cat(outputs, dim=0)  # [sum n_q_i, H, D]
        else:    # decode
            if _HAS_FLASH_ATTN:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
            elif _HAS_XFORMERS:
                o = self._decode_xformers(q, context)
            else:
                # decode：每个序列一个 token，按缓存长度收集后计算
                lens = context.context_lens.tolist()
                blocks = context.block_tables
                batch = len(lens)
                outputs = []
                # q 当前为 [batch, H, D]
                for i in range(batch):
                    seqlen_i = lens[i]
                    k_i = _gather_cache_for_seq(k_cache, blocks[i], seqlen_i)
                    v_i = _gather_cache_for_seq(v_cache, blocks[i], seqlen_i)
                    q_i = q[i:i + 1]  # [1, H, D]
                    o_i = _attend(q_i, k_i, v_i)
                    outputs.append(o_i)
                o = torch.cat(outputs, dim=0)  # [batch, H, D]
        return o
