I will fix the `AttributeError` caused by `xformers.ops.fmha.BlockDiagonalCausalMask` not being found. This class is located in `xformers.ops.fmha.attn_bias` and might not be exposed directly under `xformers.ops.fmha` in the installed version of `xformers`.

I will modify `nanovllm/layers/attention.py` to:
1.  Explicitly import `BlockDiagonalCausalMask` from `xformers.ops.fmha.attn_bias` inside the `try-except` block for xformers imports.
2.  Update the usage in `_run_xformers_varlen` to use the imported class directly instead of accessing it via `xops.fmha`.

This change ensures the code uses the correct path to the class, which is consistent with xformers documentation and common usage patterns.