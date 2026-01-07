# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark: fused vs unfused compressor + FP8 quant + KV cache insert.

Unfused (2 kernels):
  1. _fused_compress_kv_rmsnorm_rope_kernel → bf16 intermediate
  2. quantize_and_insert_k_cache → KV cache

Fused (1 kernel):
  _fused_kv_compress_norm_rope_insert_sparse_attn /
  _fused_kv_compress_norm_rope_insert_indexer_attn → KV cache directly

Measures both DeepseekV4 Attention (head=512) and Indexer (head=128) paths
across a range of num_tokens (batch sizes).

Usage:
  .venv/bin/python benchmarks/kernels/bench_compressor_fusion.py
"""

import statistics

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.deepseek_v4_attention import (
    quantize_and_insert_k_cache,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_indexer_attn,
    _fused_kv_compress_norm_rope_insert_sparse_attn,
)

# ── Reconstruct the OLD compress+rmsnorm+rope kernel (removed from codebase) ─


@triton.jit
def _old_compress_kv_rmsnorm_rope_kernel(
    compressed_kv_ptr,
    compressed_kv_stride,
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return
    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0
    block_indices = pos // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    block_offsets = pos % block_size
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE
    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    block_numbers_i64 = block_numbers.to(tl.int64)
    score = tl.load(
        state_cache_ptr
        + block_numbers_i64[:, None] * state_cache_stride0
        + block_offsets[:, None] * state_cache_stride1
        + STATE_WIDTH
        + head_offset[:, None]
        + block[None, :],
        mask=mask_pos[:, None] & mask[None, :],
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)
    kv = tl.load(
        state_cache_ptr
        + block_numbers_i64[:, None] * state_cache_stride0
        + block_offsets[:, None] * state_cache_stride1
        + head_offset[:, None]
        + block[None, :],
        mask=mask_pos[:, None] & mask[None, :],
        other=0.0,
    )
    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    compressed_kv = tl.sum(kv * score, axis=0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = (compressed_kv * rrms).to(tl.bfloat16) * rms_w.to(tl.bfloat16)
    out_base = compressed_kv_ptr + token_idx * compressed_kv_stride
    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    is_rope = (block >= NOPE_HEAD_DIM) & mask
    tl.store(out_base + block, normed, mask=is_rope)
    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    rope_local = block - NOPE_HEAD_DIM
    x_partner = tl.load(out_base + (block ^ 1), mask=is_rope, other=0.0)
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = normed * cos_v + x_partner * sin_v
    x_sub = normed * cos_v - x_partner * sin_v
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_sub, x_add)
    result = tl.where(is_rope, rotated, normed)
    tl.store(out_base + block, result, mask=mask)


# ── Benchmark harness ────────────────────────────────────────────────────


def make_test_data(
    num_tokens: int,
    head_dim: int,
    compress_ratio: int,
    rope_head_dim: int = 64,
    device: str = "cuda",
):
    """Create all tensors needed by both fused and unfused kernels."""
    nope_dim = head_dim - rope_head_dim
    overlap = compress_ratio == 4
    coff = 1 + int(overlap)
    state_dim = 2 * coff * head_dim
    state_width = state_dim // 2

    # Compressor state cache
    state_block_size = 4 if compress_ratio == 4 else 8

    # Positions: all at compression boundaries for maximum kernel work
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    positions = positions * compress_ratio + (compress_ratio - 1)

    # The kernel window extends back (1+overlap)*compress_ratio positions.
    # Block table must cover all position-derived block indices.
    max_position = int(positions[-1].item()) if num_tokens > 0 else 0
    num_state_blocks = max_position // state_block_size + 2
    state_cache = torch.randn(
        num_state_blocks,
        state_block_size,
        state_dim,
        dtype=torch.float32,
        device=device,
    )

    # Slot mapping (sequential, fits within state_cache)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # Block table: identity mapping, must cover all block indices the kernel touches
    block_table = torch.arange(
        num_state_blocks, dtype=torch.int32, device=device
    ).unsqueeze(0)

    # Token-to-request mapping
    token_to_req_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)

    # RMSNorm weight
    rms_norm_weight = torch.ones(head_dim, dtype=torch.float32, device=device)
    rms_norm_eps = 1e-6

    # RoPE cos_sin_cache
    max_pos = int(positions.max().item()) + 1
    cos_sin_cache = torch.randn(
        max_pos,
        rope_head_dim,
        dtype=torch.float32,
        device=device,
    )

    # KV cache (DeepseekV4 attention layout)
    if head_dim == 512:
        is_attention = True
        quant_block = 64
        token_stride = nope_dim + rope_head_dim * 2  # 576
        scale_dim = nope_dim // 64 + 1  # 8
        head_bytes = 584
    else:
        is_attention = False
        quant_block = 128
        token_stride = head_dim  # 128
        scale_dim = 4
        head_bytes = 132

    kv_block_size = 64
    num_kv_blocks = (num_tokens + kv_block_size - 1) // kv_block_size + 1
    kv_cache = torch.zeros(
        num_kv_blocks,
        kv_block_size,
        head_bytes,
        dtype=torch.uint8,
        device=device,
    )
    kv_slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # Intermediate bf16 tensor (for unfused path)
    compressed_kv = torch.empty(
        num_tokens,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    return dict(
        state_cache=state_cache,
        state_width=state_width,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=rms_norm_eps,
        cos_sin_cache=cos_sin_cache,
        kv_cache=kv_cache,
        kv_slot_mapping=kv_slot_mapping,
        kv_block_size=kv_block_size,
        compressed_kv=compressed_kv,
        head_dim=head_dim,
        nope_dim=nope_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        is_attention=is_attention,
        quant_block=quant_block,
        token_stride=token_stride,
        scale_dim=scale_dim,
        state_block_size=state_block_size,
    )


def run_unfused(d):
    """Old 2-kernel pipeline: compress+rmsnorm+rope → quant+cache."""
    num_tokens = d["positions"].shape[0]
    state_cache = d["state_cache"]

    # Kernel 1: compress + rmsnorm + rope → bf16 intermediate
    _old_compress_kv_rmsnorm_rope_kernel[(num_tokens,)](
        d["compressed_kv"],
        d["compressed_kv"].stride(0),
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        d["token_to_req_indices"],
        d["positions"],
        d["slot_mapping"],
        d["block_table"],
        d["block_table"].stride(0),
        d["state_block_size"],
        d["rms_norm_weight"],
        d["rms_norm_eps"],
        d["cos_sin_cache"],
        d["cos_sin_cache"].stride(0),
        HEAD_SIZE=d["head_dim"],
        TRITON_BLOCK_SIZE=triton.next_power_of_2(d["head_dim"]),
        STATE_WIDTH=d["state_width"],
        COMPRESS_RATIO=d["compress_ratio"],
        OVERLAP=d["overlap"],
        ROPE_HEAD_DIM=d["rope_head_dim"],
    )

    # Kernel 2: quant + cache insert
    if d["is_attention"]:
        kv_cache_2d = d["kv_cache"].view(d["kv_cache"].shape[0], -1)
        quantize_and_insert_k_cache(
            d["compressed_kv"],
            kv_cache_2d,
            d["kv_slot_mapping"],
            block_size=d["kv_block_size"],
        )
    else:
        ops.indexer_k_quant_and_cache(
            d["compressed_kv"],
            d["kv_cache"],
            d["kv_slot_mapping"],
            d["quant_block"],
            "ue8m0",
        )


def run_fused(d):
    """New 1-kernel pipeline: compress+rmsnorm+rope+quant+cache."""
    num_tokens = d["positions"].shape[0]
    state_cache = d["state_cache"]
    kv_cache = d["kv_cache"]

    # Dispatch directly to the specialized kernel (DeepseekV4 or indexer)
    # num_warps tuned per path: DeepseekV4 (512 elements) uses 4 warps;
    # Indexer (128 elements) uses 1 warp for max SM occupancy.
    if d["is_attention"]:
        kernel = _fused_kv_compress_norm_rope_insert_sparse_attn
        num_warps = 4
    else:
        kernel = _fused_kv_compress_norm_rope_insert_indexer_attn
        num_warps = 1

    kernel[(num_tokens,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        d["token_to_req_indices"],
        d["positions"],
        d["slot_mapping"],
        d["block_table"],
        d["block_table"].stride(0),
        d["state_block_size"],
        d["rms_norm_weight"],
        d["rms_norm_eps"],
        d["cos_sin_cache"],
        d["cos_sin_cache"].stride(0),
        kv_cache,
        d["kv_slot_mapping"],
        d["kv_block_size"],
        HEAD_SIZE=d["head_dim"],
        TRITON_BLOCK_SIZE=triton.next_power_of_2(d["head_dim"]),
        STATE_WIDTH=d["state_width"],
        COMPRESS_RATIO=d["compress_ratio"],
        OVERLAP=d["overlap"],
        ROPE_HEAD_DIM=d["rope_head_dim"],
        FP8_MAX=448.0,
        QUANT_BLOCK=d["quant_block"],
        TOKEN_STRIDE=d["token_stride"],
        SCALE_DIM=d["scale_dim"],
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=num_warps,
    )


def benchmark_detailed(fn, warmup=50, iters=200):
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()

    torch.accelerator.synchronize()
    times = [starts[i].elapsed_time(ends[i]) for i in range(iters)]

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
    }


def make_bench_fn(head_dim, provider):
    """Return a function that triton.testing.do_bench can call."""

    def bench(num_tokens):
        d = make_test_data(
            num_tokens=num_tokens,
            head_dim=head_dim,
            compress_ratio=4,
            device="cuda",
        )
        fns = {
            "unfused": lambda: run_unfused(d),
            "triton": lambda: run_fused(d),
        }
        return fns[provider]

    return bench


def main():
    token_counts = [1, 4, 16, 64, 256, 1024, 2048, 4096, 8192, 16384]

    for path_name, head_dim in [
        ("DeepseekV4 Attention (head_dim=512)", 512),
        ("Indexer (head_dim=128)", 128),
    ]:
        print("=" * 70)
        print(f"{path_name}, compress_ratio=4")
        print("=" * 70)
        print(f"{'tokens':>8} | {'unfused':>9} {'triton':>9} | {'tri/unf':>8}")
        print("-" * 70)

        for num_tokens in token_counts:
            results = {}
            for provider in ["unfused", "triton"]:
                fn = make_bench_fn(head_dim, provider)(num_tokens)
                ms = triton.testing.do_bench(fn, warmup=100, rep=500)
                results[provider] = ms

            u, t = results["unfused"], results["triton"]
            print(f"{num_tokens:>8} | {u:>8.4f}  {t:>8.4f} | {u / t:>7.2f}x")
        print()


if __name__ == "__main__":
    main()
