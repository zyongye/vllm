# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark for mla_fp8_quantize_qkv CUDA kernel vs Triton reference.

Usage:
    python benchmarks/fused_kernels/bench_mla_fp8_quant.py
    python benchmarks/fused_kernels/bench_mla_fp8_quant.py --dtype fp16
    python benchmarks/fused_kernels/bench_mla_fp8_quant.py --tokens 1 8 64 512 2048
"""

import argparse

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement

import triton
import triton.language as tl
import vllm._custom_ops as ops

# ── Fixed DeepSeek V3 MLA shapes ─────────────────────────────────────────────
NUM_HEADS = 128
QK_NOPE = 128
QK_ROPE = 64
QK_DIM = QK_NOPE + QK_ROPE    # 192
V_DIM = 128
COMBINED_DIM = QK_NOPE + V_DIM  # 256

# ── Triton reference kernel ───────────────────────────────────────────────────
_BLOCK_T   = 16
_BLOCK_T_V = 32

@triton.jit
def _quant_fp8_qkv_fused(
    q_ptr,  qout_ptr,
    k_ptr,  kout_ptr,
    v_ptr,  vout_ptr,
    n_tokens_q, n_tokens_kv, n_heads,
    scale,
    BLOCK_T          : tl.constexpr,
    BLOCK_T_V        : tl.constexpr,
    V_SRC_TOK_STRIDE : tl.constexpr,
):
    QK    : tl.constexpr = 192
    QK_P2 : tl.constexpr = 256
    V     : tl.constexpr = 128

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    dims_qk  = tl.arange(0, QK_P2)
    dim_mask = (dims_qk < QK)[None, :]
    tok_q    = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_q   = (tok_q < n_tokens_q)[:, None] & dim_mask
    qk_offs_q = tok_q[:, None] * (n_heads * QK) + pid_h * QK + dims_qk[None, :]
    q = tl.load(q_ptr + qk_offs_q, mask=mask_q, other=0.0).to(tl.float32)
    tl.store(qout_ptr + qk_offs_q, (q * scale).to(tl.float8e4nv), mask=mask_q)

    tok_k     = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_k    = (tok_k < n_tokens_kv)[:, None] & dim_mask
    qk_offs_k = tok_k[:, None] * (n_heads * QK) + pid_h * QK + dims_qk[None, :]
    k = tl.load(k_ptr + qk_offs_k, mask=mask_k, other=0.0).to(tl.float32)
    tl.store(kout_ptr + qk_offs_k, (k * scale).to(tl.float8e4nv), mask=mask_k)

    dims_v   = tl.arange(0, V)
    tok_v    = pid_t * BLOCK_T_V + tl.arange(0, BLOCK_T_V)
    mask_v   = (tok_v < n_tokens_kv)[:, None]
    v_src    = tok_v[:, None] * V_SRC_TOK_STRIDE + pid_h * V + dims_v[None, :]
    v_dst    = tok_v[:, None] * (n_heads * V)    + pid_h * V + dims_v[None, :]
    v = tl.load(v_ptr + v_src, mask=mask_v, other=0.0).to(tl.float32)
    tl.store(vout_ptr + v_dst, (v * scale).to(tl.float8e4nv), mask=mask_v)


def fused_fp8_quantize(q, k, v, scale=1.0):
    n_kv = q.shape[0]
    grid = (triton.cdiv(n_kv, _BLOCK_T), NUM_HEADS)
    qout = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    kout = torch.empty_like(k, dtype=torch.float8_e4m3fn)
    vout = torch.empty(n_kv, NUM_HEADS, V_DIM, dtype=torch.float8_e4m3fn,
                       device=q.device)
    _quant_fp8_qkv_fused[grid](
        q, qout, k, kout, v, vout,
        n_kv, n_kv, NUM_HEADS, scale, _BLOCK_T, _BLOCK_T_V, v.stride(0),
    )
    return qout, kout, vout


def make_inputs(n_tokens: int, dtype: torch.dtype, device: str = "cuda"):
    """Construct q, k, v matching the mla_attention.py call path.

    v is a non-contiguous split view of kv_nope, exactly as produced by
    kv_b_proj output.split([qk_nope_head_dim, v_head_dim], dim=-1).
    """
    q = torch.randn(n_tokens, NUM_HEADS, QK_DIM, dtype=dtype, device=device)
    kv_nope = torch.randn(
        n_tokens, NUM_HEADS, COMBINED_DIM, dtype=dtype, device=device
    )
    k_nope, v = kv_nope.split([QK_NOPE, V_DIM], dim=-1)
    k_pe = torch.randn(n_tokens, NUM_HEADS, QK_ROPE, dtype=dtype, device=device)
    k = torch.cat([k_nope, k_pe], dim=-1)
    return q, k, v


def bytes_transferred(n_tokens: int, elem_bytes: int = 2) -> int:
    """Total bytes read + written by the kernel.

    Reads: q [N,H,192] + k [N,H,192] + v [N,H,128]  (bf16/fp16 = 2 bytes)
    Writes: q_fp8 [N,H,192] + k_fp8 [N,H,192] + v_fp8 [N,H,128]  (fp8 = 1 byte)
    """
    read = n_tokens * NUM_HEADS * (QK_DIM + QK_DIM + V_DIM) * elem_bytes
    write = n_tokens * NUM_HEADS * (QK_DIM + QK_DIM + V_DIM) * 1
    return read + write


def bench_cuda(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> TMeasurement:
    return TBenchmark.Timer(
        stmt="ops.mla_fp8_quantize_qkv(q, k, v, scale)",
        globals={"ops": ops, "q": q, "k": k, "v": v, "scale": scale},
        label="mla_fp8_quantize_qkv",
        sub_label=f"N={q.shape[0]} dtype={q.dtype}",
        description="cuda",
    ).blocked_autorange(min_run_time=1)


def bench_triton(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> TMeasurement:
    return TBenchmark.Timer(
        stmt="fused_fp8_quantize(q, k, v, scale)",
        globals={
            "fused_fp8_quantize": fused_fp8_quantize,
            "q": q, "k": k, "v": v, "scale": scale,
        },
        label="mla_fp8_quantize_qkv",
        sub_label=f"N={q.shape[0]} dtype={q.dtype}",
        description="triton",
    ).blocked_autorange(min_run_time=1)


def run(token_counts: list[int], dtype: torch.dtype, scale: float = 1.0):
    measurements: list[TMeasurement] = []

    print(f"\ndtype={dtype}  scale={scale}")
    print(f"{'N':>8}  {'impl':>8}  {'us':>9}  {'GB/s':>9}")
    print("-" * 44)

    for n in token_counts:
        q, k, v = make_inputs(n, dtype)

        # Warm-up (forces triton JIT compilation before timing).
        for _ in range(3):
            ops.mla_fp8_quantize_qkv(q, k, v, scale)
            fused_fp8_quantize(q, k, v, scale)
        torch.cuda.synchronize()

        m_cuda = bench_cuda(q, k, v, scale)
        m_triton = bench_triton(q, k, v, scale)

        bw = bytes_transferred(n, elem_bytes=q.element_size())
        for m, tag in [(m_cuda, "cuda"), (m_triton, "triton")]:
            us = m.median * 1e6
            gbs = bw / m.median / 1e9
            print(f"{n:>8}  {tag:>8}  {us:>9.2f}  {gbs:>9.1f}")

        measurements.extend([m_cuda, m_triton])

    print()
    TBenchmark.Compare(measurements).print()
    return measurements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Input tensor dtype (default: bf16)",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=[1, 8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        metavar="N",
        help="Token counts to benchmark",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Quantization scale (default: 1.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    run(args.tokens, dtype, args.scale)
