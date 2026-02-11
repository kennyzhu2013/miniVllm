import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from nanovllm.layers.attention import Attention
from nanovllm.utils.context import set_context, reset_context


def test_prefill_dtype_cast():
    H = 4
    Kv = 2
    D = 8
    Lq = 5
    Lk = 7
    attn = Attention(num_heads=H, head_dim=D, scale=1.0, num_kv_heads=Kv)
    q = torch.randn(Lq, H, D, dtype=torch.float16)
    k = torch.randn(Lk, Kv, D, dtype=torch.bfloat16)
    v = torch.randn(Lk, Kv, D, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, Lq], dtype=torch.int32)
    cu_k = torch.tensor([0, Lk], dtype=torch.int32)
    set_context(True, cu_q, cu_k, Lq, Lk, torch.tensor([], dtype=torch.int32), None, None)
    o = attn(q, k, v)
    assert o.shape == (Lq, H, D)
    assert o.dtype == torch.float16
    reset_context()


def test_decode_dtype_cast():
    H = 4
    Kv = 2
    D = 8
    block_size = 4
    num_blocks = 2
    seqlen = 7
    bs = 1
    attn = Attention(num_heads=H, head_dim=D, scale=1.0, num_kv_heads=Kv)
    # Prepare KV cache: [num_blocks, block_size, Kv, D]
    k_cache = torch.randn(num_blocks, block_size, Kv, D, dtype=torch.bfloat16)
    v_cache = torch.randn(num_blocks, block_size, Kv, D, dtype=torch.bfloat16)
    attn.k_cache = k_cache
    attn.v_cache = v_cache
    # q of shape [bs, H, D] in half
    q = torch.randn(bs, H, D, dtype=torch.float16)
    # block table: two blocks, indices [0, 1]
    blocks = torch.tensor([[0, 1]], dtype=torch.int32)
    context_lens = torch.tensor([seqlen], dtype=torch.int32)
    set_context(False, slot_mapping=torch.tensor([0], dtype=torch.int32), context_lens=context_lens, block_tables=blocks)
    o = attn(q, torch.empty(0), torch.empty(0))
    assert o.shape == (bs, H, D)
    assert o.dtype == torch.float16
    reset_context()


if __name__ == "__main__":
    test_prefill_dtype_cast()
    test_decode_dtype_cast()
    print("OK")
