# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark fuse_mla_decode_rope_q_concat_kv_insert vs unfused ops.

Compares the fused kernel against:
  - concat_and_cache_mla_rope_fused (RoPE + kv insert)
  - concat_mla_q (q_nope + q_pe concat)

Reports latency and achieved memory bandwidth vs GPU peak.

Usage:
    python benchmarks/kernels/bench_fuse_mla_decode_rope_q_concat_kv_insert.py
"""

import argparse

import torch
from vllm.triton_utils import triton

from vllm import _custom_ops as ops

# DeepSeekV3 / TP=1 defaults
KV_LORA_RANK = 512
QK_ROPE_DIM = 64
NUM_HEADS = 128
MAX_POSITION = 4096
BLOCK_SIZE = 16

NUM_TOKENS = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def peak_membw_gbs() -> float:
    """Theoretical peak HBM bandwidth from device properties (GB/s)."""
    prop = torch.cuda.get_device_properties(0)
    return (prop.memory_clock_rate * 1e3 * prop.memory_bus_width / 8 * 2) / 1e9


def fused_bytes(
    B: int, N: int, L: int, R: int, elem_bytes: int, kv_elem_bytes: int
) -> int:
    """Theoretical HBM bytes for the fused kernel.

    Reads:  q_nope[B,N,L], q_pe[B,N,R], kv[B,L+R], cos_sin[B,R] (f32)
    Writes: q_pe[B,N,R] (in-place), q_out[B,N,L+R], kv_cache[B,L+R]
    """
    reads = (
        B * N * L * elem_bytes  # q_nope
        + B * N * R * elem_bytes  # q_pe
        + B * (L + R) * elem_bytes  # kv
        + B * R * 4
    )  # cos_sin_cache (float32)
    writes = (
        B * N * R * elem_bytes  # q_pe in-place
        + B * N * (L + R) * elem_bytes  # q_out
        + B * (L + R) * kv_elem_bytes
    )  # kv_cache
    return reads + writes


def make_inputs(B: int, N: int, dtype: torch.dtype, device: str = "cuda") -> dict:
    L, R = KV_LORA_RANK, QK_ROPE_DIM

    q_nope = (
        torch.randn(N, B, L, dtype=dtype, device=device).transpose(0, 1).contiguous()
    )
    q_pe = torch.randn(B, N, R, dtype=dtype, device=device)
    kv = torch.randn(B, L + R, dtype=dtype, device=device)
    positions = torch.randint(0, MAX_POSITION, (B,), device=device)
    cos_sin_f32 = torch.randn(MAX_POSITION, R, dtype=torch.float32, device=device)
    cos_sin_dtype = cos_sin_f32.to(dtype)  # for unfused ops (require dtype match)
    num_blocks = (B + BLOCK_SIZE - 1) // BLOCK_SIZE + 4
    kv_cache = torch.zeros(num_blocks, BLOCK_SIZE, L + R, dtype=dtype, device=device)
    slot_mapping = torch.arange(B, dtype=torch.long, device=device)
    kv_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    q_out = torch.empty(B, N, L + R, dtype=dtype, device=device)
    q_out_ref = torch.empty(B, N, L + R, dtype=dtype, device=device)

    return dict(
        q_nope=q_nope,
        q_pe=q_pe,
        kv=kv,
        positions=positions,
        cos_sin_f32=cos_sin_f32,
        cos_sin_dtype=cos_sin_dtype,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        kv_scale=kv_scale,
        q_out=q_out,
        q_out_ref=q_out_ref,
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=NUM_TOKENS,
        line_arg="provider",
        line_vals=["unfused", "fused"],
        line_names=[
            "Unfused (rope_fused + concat_mla_q)",
            "Fused (fuse_mla_decode_rope_q_concat_kv_insert)",
        ],
        styles=[("blue", "--"), ("green", "-")],
        ylabel="Latency (µs)",
        plot_name="fuse_mla_decode_rope_q_concat_kv_insert-bf16",
        args={},
    )
)
def bench_bf16(num_tokens: int, provider: str):
    dtype = torch.bfloat16
    d = make_inputs(num_tokens, NUM_HEADS, dtype)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "unfused":

        def fn():
            ops.concat_and_cache_mla_rope_fused(
                d["positions"],
                d["q_pe"],
                d["kv"][:, KV_LORA_RANK:].contiguous(),
                d["kv"][:, :KV_LORA_RANK].contiguous(),
                d["cos_sin_dtype"],
                True,
                d["slot_mapping"],
                d["kv_cache"],
                "auto",
                d["kv_scale"],
            )
            ops.concat_mla_q(d["q_nope"], d["q_pe"], d["q_out_ref"])
    else:

        def fn():
            ops.fuse_mla_decode_rope_q_concat_kv_insert(
                d["positions"],
                d["q_nope"],
                d["q_pe"],
                d["kv"],
                d["cos_sin_f32"],
                True,
                d["slot_mapping"],
                d["kv_cache"],
                "auto",
                d["kv_scale"],
                None,
                d["q_out"],
            )

    ms, min_ms, max_ms = triton.testing.do_bench(
        fn, warmup=100, rep=500, quantiles=quantiles
    )
    return ms * 1e3, max_ms * 1e3, min_ms * 1e3  # ms → µs


def run_bw_table(dtype: torch.dtype) -> None:
    """Print a bandwidth-utilization table for the fused kernel."""
    elem = torch.finfo(dtype).bits // 8
    peak_bw = peak_membw_gbs()
    device = torch.cuda.get_device_name(0)

    print(f"\n{'=' * 82}")
    print(f"  BW utilization — fused kernel  [{device}]")
    print(
        f"  dtype={dtype}, heads={NUM_HEADS}, kv_lora={KV_LORA_RANK}, "
        f"rope={QK_ROPE_DIM}, peak_bw={peak_bw:.0f} GB/s"
    )
    print(f"{'=' * 82}")
    print(f"{'tokens':>8} {'bytes_MB':>10} {'lat_us':>10} {'bw_GBs':>10} {'util%':>8}")
    print(f"{'-' * 52}")

    for B in NUM_TOKENS:
        d = make_inputs(B, NUM_HEADS, dtype)

        def fn():
            ops.fuse_mla_decode_rope_q_concat_kv_insert(
                d["positions"],
                d["q_nope"],
                d["q_pe"],
                d["kv"],
                d["cos_sin_f32"],
                True,
                d["slot_mapping"],
                d["kv_cache"],
                "auto",
                d["kv_scale"],
                None,
                d["q_out"],
            )

        ms = triton.testing.do_bench(fn, warmup=100, rep=500)
        lat_us = ms * 1e3
        total_bytes = fused_bytes(
            B, NUM_HEADS, KV_LORA_RANK, QK_ROPE_DIM, elem_bytes=elem, kv_elem_bytes=elem
        )
        bw_gbs = (total_bytes / 1e9) / (ms / 1e3)
        util = bw_gbs / peak_bw * 100

        print(
            f"{B:>8} {total_bytes / 1e6:>9.1f}MB {lat_us:>9.1f}µs "
            f"{bw_gbs:>9.1f} {util:>7.1f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path", default=None, help="Directory to save CSV results"
    )
    args = parser.parse_args()

    # Latency comparison table (no plots needed)
    bench_bf16.run(print_data=True, save_path=args.save_path, show_plots=False)

    # Memory bandwidth utilization
    run_bw_table(torch.bfloat16)
