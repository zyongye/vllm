# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for fuse_mla_decode_rope_q_concat_kv_insert CUDA kernel.

Tests the fused kernel against an unfused reference:
  - Manual RoPE (matching DeepseekScalingRotaryEmbedding NeoX / GPT-J styles)
  - torch.cat for Q concat
  - ops.concat_and_cache_mla for KV insert
"""

import pytest
import torch

try:
    from vllm import _custom_ops as ops
except ImportError:
    pytest.skip("Run `pip install -e .` first.", allow_module_level=True)

# DeepSeekV3 dimensions
KV_LORA_RANK = 512  # L
QK_ROPE_DIM = 64  # R
MAX_POSITION = 4096
BLOCK_SIZE = 16

NUM_TOKENS = [1, 4, 16, 64, 128]
DTYPES = [torch.bfloat16, torch.float16]
FP8_Q = [False, True]
TP_HEADS = [128, 16]  # num_heads / TP_size
IS_NEOX = [True, False]
KV_DTYPES = ["auto", "fp8"]


def make_cos_sin_cache(rope_dim: int, max_pos: int, device: str) -> torch.Tensor:
    """Build a float32 cos/sin cache [max_pos, rope_dim].

    Kept in float32 to match the production path where FlashInfer is enabled
    (base.py skips the dtype cast when use_flashinfer=True).
    """
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [max_pos, rope_dim/2]
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat([cos, sin], dim=-1).to(device)  # [max_pos, rope_dim], float32
    return cache


def apply_rope_ref(
    x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor, is_neox: bool
) -> torch.Tensor:
    """Apply RoPE to x in float32 (matching the kernel's internal precision).

    x can be [B, R] or [B, N, R].  cos_sin_cache is float32.
    Returns float32 so callers can choose when to downcast.
    """
    embed_dim = x.shape[-1] // 2
    cos = cos_sin_cache[positions, :embed_dim]  # [B, R/2], float32
    sin = cos_sin_cache[positions, embed_dim:]  # [B, R/2], float32
    x_f = x.float()
    if x_f.dim() == 3:
        cos = cos.unsqueeze(1)  # [B, 1, R/2]
        sin = sin.unsqueeze(1)
    if is_neox:
        x1, x2 = x_f[..., :embed_dim], x_f[..., embed_dim:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    else:
        x1 = x_f[..., 0::2]
        x2 = x_f[..., 1::2]
        rot = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return rot.flatten(-2)


def unfused_reference(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    kv_scale: torch.Tensor,
    q_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Reference: separate RoPE → concat → kv_insert operations."""
    L = kv.shape[-1] - QK_ROPE_DIM

    # RoPE on q_pe and k_pe — results are float32.
    q_pe_rot = apply_rope_ref(q_pe, positions, cos_sin_cache, is_neox)
    k_pe_src = kv[:, L:].clone()
    k_pe_rot = apply_rope_ref(k_pe_src, positions, cos_sin_cache, is_neox)

    # Q concat — stay in float32 until final cast to match kernel precision.
    q_nope_f = q_nope.float()
    q_out_f = torch.cat([q_nope_f, q_pe_rot], dim=-1).contiguous()  # [B, N, L+R] f32

    # Optional FP8 quant of q (quantize from float32, same as kernel).
    if q_scale is not None:
        FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
        q_out = (
            (q_out_f / q_scale.item()).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        )
    else:
        q_out = q_out_f.to(q_pe.dtype)

    # KV insert via existing kernel
    kv_c = kv[:, :L].contiguous()
    ops.concat_and_cache_mla(
        kv_c,
        k_pe_rot.to(kv.dtype).contiguous(),
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        kv_scale,
    )

    return q_out


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", TP_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("do_fp8_q", FP8_Q)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@pytest.mark.parametrize("kv_cache_dtype", KV_DTYPES)
def test_fuse_mla_decode_rope_q_concat_kv_insert(
    num_tokens, num_heads, dtype, do_fp8_q, is_neox, kv_cache_dtype
):
    if kv_cache_dtype == "fp8" and not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("FP8 not supported on this torch version")

    torch.manual_seed(42)
    device = "cuda"
    B, N, L, R = num_tokens, num_heads, KV_LORA_RANK, QK_ROPE_DIM

    # q_nope: non-contiguous, simulating transpose(0,1) from BMM output
    q_nope_raw = torch.randn(N, B, L, dtype=dtype, device=device)
    q_nope = q_nope_raw.transpose(0, 1).contiguous()  # [B, N, L]
    q_pe = torch.randn(B, N, R, dtype=dtype, device=device)
    kv = torch.randn(B, L + R, dtype=dtype, device=device)
    positions = torch.randint(0, MAX_POSITION, (B,), device=device)

    cos_sin_cache = make_cos_sin_cache(R, MAX_POSITION, device)  # float32

    num_blocks = (B + BLOCK_SIZE - 1) // BLOCK_SIZE + 4
    cache_dtype = torch.float8_e4m3fn if kv_cache_dtype == "fp8" else dtype
    kv_cache_fused = torch.zeros(
        num_blocks, BLOCK_SIZE, L + R, dtype=cache_dtype, device=device
    )
    kv_cache_ref = torch.zeros_like(kv_cache_fused)

    slot_mapping = torch.arange(B, dtype=torch.long, device=device)
    kv_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    q_scale = (
        torch.tensor([1.0], dtype=torch.float32, device=device) if do_fp8_q else None
    )

    # --- Fused kernel ---
    out_dtype = torch.float8_e4m3fn if do_fp8_q else dtype
    q_out_fused = torch.empty(B, N, L + R, dtype=out_dtype, device=device)
    ops.fuse_mla_decode_rope_q_concat_kv_insert(
        positions,
        q_nope,
        q_pe.clone(),  # q_pe is now read-only; clone kept for safety
        kv.clone(),
        cos_sin_cache,
        is_neox,
        slot_mapping,
        kv_cache_fused,
        kv_cache_dtype,
        kv_scale,
        q_scale,
        q_out_fused,
    )

    # --- Reference ---
    q_out_ref = unfused_reference(
        q_nope,
        q_pe,
        kv,
        positions,
        cos_sin_cache,
        is_neox,
        slot_mapping,
        kv_cache_ref,
        kv_cache_dtype,
        kv_scale,
        q_scale,
    )

    # Compare q_out.  FP8 e4m3fn has ~0.0625 spacing at |val|~1, so we need
    # a looser tolerance.  For bf16/fp16, 1e-2 covers the 1-ULP (1/128) diff
    # from CUDA FMA vs PyTorch float32 ordering.
    atol_q = 0.1 if do_fp8_q else 1e-2
    torch.testing.assert_close(
        q_out_fused.float(),
        q_out_ref.float(),
        atol=atol_q,
        rtol=0,
        msg=f"q_out mismatch (num_tokens={num_tokens}, num_heads={num_heads}, "
        f"dtype={dtype}, fp8_q={do_fp8_q}, neox={is_neox})",
    )

    # Compare kv_cache
    atol_kv = 1e-2 if kv_cache_dtype == "fp8" else 1e-2
    torch.testing.assert_close(
        kv_cache_fused.float(),
        kv_cache_ref.float(),
        atol=atol_kv,
        rtol=0,
        msg=f"kv_cache mismatch (num_tokens={num_tokens}, kv_dtype={kv_cache_dtype})",
    )
