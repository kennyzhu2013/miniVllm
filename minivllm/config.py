import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        # 尝试通过 Transformers 读取配置；若失败（如未支持 qwen3），回退到直接解析 config.json
        try:
            self.hf_config = AutoConfig.from_pretrained(self.model)
        except Exception:
            cfg_path = os.path.join(self.model, "config.json")
            if not os.path.isfile(cfg_path):
                raise
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 轻量封装为对象以支持属性访问
            class _HFConfig:
                pass

            hf = _HFConfig()
            for k, v in data.items():
                setattr(hf, k, v)

            # 必要字段兜底：max_position_embeddings
            if not hasattr(hf, "max_position_embeddings"):
                max_pos = data.get("max_position_embeddings") or data.get("max_sequence_length") or 2048
                setattr(hf, "max_position_embeddings", max_pos)

            self.hf_config = hf
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
