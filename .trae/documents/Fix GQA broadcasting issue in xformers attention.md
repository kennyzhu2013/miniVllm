I will fix the `ValueError: Incompatible shapes for attention inputs` caused by GQA (Grouped Query Attention) where the number of query heads differs from key/value heads. The `xformers` `memory_efficient_attention` does not automatically broadcast key/value tensors to match query heads.

I will update `nanovllm/layers/attention.py` to manually expand the key and value tensors using the existing `_repeat_kv` method in the following functions:
1.  `_run_xformers_varlen`: Expand `k` and `v` before calling `xops.memory_efficient_attention`.
2.  `_prefill_xformers`: Expand `ki` and `vi` before calling `xops.memory_efficient_attention`.
3.  `_decode_xformers`: Expand `ki` and `vi` before calling `xops.memory_efficient_attention`.

This ensures that the attention operation receives tensors with compatible shapes, matching the manual expansion done in the fallback implementations.