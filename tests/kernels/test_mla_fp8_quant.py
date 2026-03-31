# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the mla_fp8_quantize_qkv CUDA kernel.

Tests the fused FP8 quantization of Q, K, V tensors used in MLA prefill
attention. Fixed for DeepSeek V3 shapes: QK_NOPE=128, QK_ROPE=64, V=128,
num_heads=128.
"""

import pytest
import torch

try:
    from vllm import _custom_ops as ops
except ImportError:
    pytest.skip(
        "Could not import vllm._custom_ops. Run `pip install -e .` first.",
        allow_module_level=True,
    )

# DeepSeek V3 MLA fixed dimensions (must match kernel template parameters).
NUM_HEADS = 128
QK_NOPE_DIM = 128
QK_ROPE_DIM = 64
QK_HEAD_DIM = QK_NOPE_DIM + QK_ROPE_DIM  # 192
V_HEAD_DIM = 128
COMBINED_DIM = QK_NOPE_DIM + V_HEAD_DIM  # 256 (kv_nope last dim)

FP8_MAX = 448.0  # max representable value of float8_e4m3fn

NUM_TOKENS = [1, 7, 8, 9, 16, 64, 128]
DTYPES = [torch.bfloat16, torch.float16]
SCALES = [1.0, 0.5, 2.0]


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_qkv(n_tokens: int, dtype: torch.dtype, device: str = "cuda"):
    """Construct q, k, v matching the actual mla_attention.py call path.

    v is non-contiguous — it is a split view of kv_nope with shape
    [n_tokens, num_heads, COMBINED_DIM], so v.stride(0) = NUM_HEADS * COMBINED_DIM.
    """
    torch.manual_seed(42)
    q = torch.randn(n_tokens, NUM_HEADS, QK_HEAD_DIM, dtype=dtype, device=device)
    # k is built by concat of k_nope (from split) and k_pe (rope, contiguous).
    kv_nope = torch.randn(
        n_tokens, NUM_HEADS, COMBINED_DIM, dtype=dtype, device=device
    )
    k_nope, v = kv_nope.split([QK_NOPE_DIM, V_HEAD_DIM], dim=-1)
    k_pe = torch.randn(n_tokens, NUM_HEADS, QK_ROPE_DIM, dtype=dtype, device=device)
    k = torch.cat([k_nope, k_pe], dim=-1)
    return q, k, v


def _ref_fp8_quant(x: torch.Tensor, scale: float) -> torch.Tensor:
    """PyTorch reference: scale, clamp, convert to fp8_e4m3fn."""
    return (x.float() * scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


# ── shape / dtype tests ───────────────────────────────────────────────────────


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1, reason="Requires a CUDA device"
)
@pytest.mark.parametrize("n_tokens", NUM_TOKENS)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_output_shapes_and_dtype(n_tokens: int, dtype: torch.dtype):
    """Output tensors have correct shapes, dtype, and are contiguous."""
    q, k, v = _make_qkv(n_tokens, dtype)

    q_fp8, k_fp8, v_fp8 = ops.mla_fp8_quantize_qkv(q, k, v)

    assert q_fp8.shape == (n_tokens, NUM_HEADS, QK_HEAD_DIM)
    assert k_fp8.shape == (n_tokens, NUM_HEADS, QK_HEAD_DIM)
    assert v_fp8.shape == (n_tokens, NUM_HEADS, V_HEAD_DIM)
    for t in (q_fp8, k_fp8, v_fp8):
        assert t.dtype == torch.float8_e4m3fn
        assert t.is_contiguous()


# ── correctness vs PyTorch reference ─────────────────────────────────────────


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1, reason="Requires a CUDA device"
)
@pytest.mark.parametrize("n_tokens", NUM_TOKENS)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_q_k_values_match_reference(n_tokens: int, dtype: torch.dtype):
    """Q and K fp8 values match a simple PyTorch reference (scale=1.0)."""
    q, k, v = _make_qkv(n_tokens, dtype)

    q_fp8, k_fp8, _ = ops.mla_fp8_quantize_qkv(q, k, v, scale=1.0)
    q_ref = _ref_fp8_quant(q, 1.0)
    k_ref = _ref_fp8_quant(k, 1.0)

    # Compare as uint8 (bit-exact) — both paths use round-to-nearest-even.
    torch.testing.assert_close(
        q_fp8.view(torch.uint8), q_ref.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        k_fp8.view(torch.uint8), k_ref.view(torch.uint8), atol=0, rtol=0
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1, reason="Requires a CUDA device"
)
@pytest.mark.parametrize("n_tokens", NUM_TOKENS)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_v_non_contiguous_values_match_reference(n_tokens: int, dtype: torch.dtype):
    """V is quantized correctly despite being a non-contiguous split view."""
    q, k, v = _make_qkv(n_tokens, dtype)

    assert not v.is_contiguous(), "V must be non-contiguous for this test to be meaningful"
    assert v.stride(0) == NUM_HEADS * COMBINED_DIM, (
        f"Expected V token stride {NUM_HEADS * COMBINED_DIM}, got {v.stride(0)}"
    )

    _, _, v_fp8 = ops.mla_fp8_quantize_qkv(q, k, v, scale=1.0)
    v_ref = _ref_fp8_quant(v, 1.0)

    torch.testing.assert_close(
        v_fp8.view(torch.uint8), v_ref.view(torch.uint8), atol=0, rtol=0
    )


# ── scale tests ───────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1, reason="Requires a CUDA device"
)
@pytest.mark.parametrize("scale", SCALES)
@torch.inference_mode()
def test_scale_applied_to_q_k_v(scale: float):
    """Non-unit scale is correctly applied to Q, K, and V."""
    n_tokens = 16
    dtype = torch.bfloat16
    q, k, v = _make_qkv(n_tokens, dtype)

    q_fp8, k_fp8, v_fp8 = ops.mla_fp8_quantize_qkv(q, k, v, scale=scale)

    q_ref = _ref_fp8_quant(q, scale)
    k_ref = _ref_fp8_quant(k, scale)
    v_ref = _ref_fp8_quant(v, scale)

    torch.testing.assert_close(
        q_fp8.view(torch.uint8), q_ref.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        k_fp8.view(torch.uint8), k_ref.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        v_fp8.view(torch.uint8), v_ref.view(torch.uint8), atol=0, rtol=0
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1, reason="Requires a CUDA device"
)
@torch.inference_mode()
def test_scale_one_preserves_small_exact_values():
    """Values exactly representable in fp8 survive scale=1.0 round-trip."""
    n_tokens = 8
    dtype = torch.bfloat16

    # Use values that are exactly representable in fp8_e4m3fn: powers of 2.
    exact_vals = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, -1.0, -2.0]
    val_t = torch.tensor(exact_vals, dtype=torch.float32)
    # Verify they are exact in fp8 (round-trip through fp8 == identity).
    assert torch.equal(val_t.to(torch.float8_e4m3fn).float(), val_t)

    q = torch.full((n_tokens, NUM_HEADS, QK_HEAD_DIM), 1.0, dtype=dtype, device="cuda")
    kv_nope = torch.full(
        (n_tokens, NUM_HEADS, COMBINED_DIM), 1.0, dtype=dtype, device="cuda"
    )
    _, v = kv_nope.split([QK_NOPE_DIM, V_HEAD_DIM], dim=-1)
    k = torch.full((n_tokens, NUM_HEADS, QK_HEAD_DIM), 1.0, dtype=dtype, device="cuda")

    q_fp8, k_fp8, v_fp8 = ops.mla_fp8_quantize_qkv(q, k, v, scale=1.0)

    expected_fp8_val = torch.tensor([1.0]).to(torch.float8_e4m3fn).view(torch.uint8).item()
    assert torch.all(q_fp8.view(torch.uint8) == expected_fp8_val)
    assert torch.all(k_fp8.view(torch.uint8) == expected_fp8_val)
    assert torch.all(v_fp8.view(torch.uint8) == expected_fp8_val)


# ── saturation / clamping ─────────────────────────────────────────────────────


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1, reason="Requires a CUDA device"
)
@torch.inference_mode()
def test_saturation_at_fp8_max():
    """Values exceeding FP8_MAX are saturated, not wrapped or NaN'd."""
    n_tokens = 4
    dtype = torch.bfloat16

    large_val = 1000.0  # > FP8_MAX = 448
    q = torch.full((n_tokens, NUM_HEADS, QK_HEAD_DIM), large_val, dtype=dtype, device="cuda")
    kv_nope = torch.full(
        (n_tokens, NUM_HEADS, COMBINED_DIM), large_val, dtype=dtype, device="cuda"
    )
    _, v = kv_nope.split([QK_NOPE_DIM, V_HEAD_DIM], dim=-1)
    k = torch.full((n_tokens, NUM_HEADS, QK_HEAD_DIM), large_val, dtype=dtype, device="cuda")

    q_fp8, k_fp8, v_fp8 = ops.mla_fp8_quantize_qkv(q, k, v, scale=1.0)

    fp8_max_val = torch.tensor([FP8_MAX]).to(torch.float8_e4m3fn).float().item()
    assert torch.all(q_fp8.float() == fp8_max_val), "Q not saturated at FP8_MAX"
    assert torch.all(k_fp8.float() == fp8_max_val), "K not saturated at FP8_MAX"
    assert torch.all(v_fp8.float() == fp8_max_val), "V not saturated at FP8_MAX"


# ── edge cases ────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1, reason="Requires a CUDA device"
)
@torch.inference_mode()
def test_single_token():
    """Single token: kernel handles n < QK_TOKENS_PER_BLOCK (=8) correctly."""
    q, k, v = _make_qkv(1, torch.bfloat16)
    q_fp8, k_fp8, v_fp8 = ops.mla_fp8_quantize_qkv(q, k, v)

    assert q_fp8.shape == (1, NUM_HEADS, QK_HEAD_DIM)
    assert k_fp8.shape == (1, NUM_HEADS, QK_HEAD_DIM)
    assert v_fp8.shape == (1, NUM_HEADS, V_HEAD_DIM)

    # Values must match reference.
    torch.testing.assert_close(
        q_fp8.view(torch.uint8), _ref_fp8_quant(q, 1.0).view(torch.uint8),
        atol=0, rtol=0,
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1, reason="Requires a CUDA device"
)
@torch.inference_mode()
def test_token_count_not_multiple_of_block_size():
    """Non-power-of-2 token count: boundary handling is correct."""
    # QK_TOKENS_PER_BLOCK=8, V_TOKENS_PER_BLOCK=12. Use a count that is
    # a multiple of neither.
    n_tokens = 100
    q, k, v = _make_qkv(n_tokens, torch.bfloat16)

    q_fp8, k_fp8, v_fp8 = ops.mla_fp8_quantize_qkv(q, k, v, scale=1.0)

    torch.testing.assert_close(
        q_fp8.view(torch.uint8), _ref_fp8_quant(q, 1.0).view(torch.uint8),
        atol=0, rtol=0,
    )
    torch.testing.assert_close(
        k_fp8.view(torch.uint8), _ref_fp8_quant(k, 1.0).view(torch.uint8),
        atol=0, rtol=0,
    )
    torch.testing.assert_close(
        v_fp8.view(torch.uint8), _ref_fp8_quant(v, 1.0).view(torch.uint8),
        atol=0, rtol=0,
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1, reason="Requires a CUDA device"
)
@torch.inference_mode()
def test_large_batch():
    """Large sequence: kernel loop-over-tokens path is exercised."""
    n_tokens = 4096
    q, k, v = _make_qkv(n_tokens, torch.bfloat16)

    q_fp8, k_fp8, v_fp8 = ops.mla_fp8_quantize_qkv(q, k, v, scale=1.0)

    assert q_fp8.shape == (n_tokens, NUM_HEADS, QK_HEAD_DIM)
    assert k_fp8.shape == (n_tokens, NUM_HEADS, QK_HEAD_DIM)
    assert v_fp8.shape == (n_tokens, NUM_HEADS, V_HEAD_DIM)

    # Spot-check a slice to keep memory usage reasonable.
    sl = slice(0, 128)
    torch.testing.assert_close(
        q_fp8[sl].view(torch.uint8),
        _ref_fp8_quant(q[sl], 1.0).view(torch.uint8),
        atol=0, rtol=0,
    )
    torch.testing.assert_close(
        v_fp8[sl].view(torch.uint8),
        _ref_fp8_quant(v[sl], 1.0).view(torch.uint8),
        atol=0, rtol=0,
    )
