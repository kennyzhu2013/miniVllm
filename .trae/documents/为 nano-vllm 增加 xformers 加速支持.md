我将修改 `nanovllm/layers/attention.py` 以增加 `xformers` 加速支持。

**修改方案：**
1.  **引入 xformers 库**：在文件头部尝试导入 `xformers.ops`，并设置 `_HAS_XFORMERS` 标志。
2.  **新增 Xformers 专用方法**：
    *   `_run_xformers_varlen`：用于无前缀缓存的预填充阶段。利用 `BlockDiagonalCausalMask` 一次性处理所有序列，无需循环，性能接近 Flash-Attn。
    *   `_prefill_xformers`：用于带前缀缓存的预填充阶段。基于现有的 `_prefill_fallback` 逻辑改造，在 gather KV 后调用 `xformers` 进行计算，替代原本的 `torch.einsum`。
    *   `_decode_xformers`：用于解码阶段。基于 `_decode_fallback` 改造，在 gather KV 后调用 `xformers`。
3.  **修改 `forward` 调度逻辑**：
    *   优先级：`Flash-Attn` (首选) -> `Xformers` (次选) -> `PyTorch Fallback` (保底)。
    *   在 `_HAS_FLASH_ATTN` 为 `False` 但 `_HAS_XFORMERS` 为 `True` 时，根据是否是预填充及是否有 `block_tables` 路由到上述新增方法。

**代码变更点：**
*   `nanovllm/layers/attention.py`:
    *   导入 `xformers.ops`。
    *   实现 `_run_xformers_varlen` (高效路径)。
    *   实现 `_prefill_xformers` (兼容路径)。
    *   实现 `_decode_xformers` (兼容路径)。
    *   更新 `forward` 方法。

这一改动将显著提升在不支持 Flash-Attn 的显卡（如 V100/T4）上的推理性能，同时保持对 PagedAttention 功能的完整支持。