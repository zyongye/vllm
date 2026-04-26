# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench: fast_topk_v2 vs persistent_topk for k in {512, 1024}.

Both ops select the top-k entries per row of a `[B, L]` float32 score
tensor. vLLM's `persistent_topk` is the existing path used by the indexer;
`fast_topk_v2` is the sm_90+ port from sglang that adds Hopper thread-block
clusters.

V4-Flash uses `index_topk = 512`; V4-Pro uses `index_topk = 1024`. We bench
both Ks at the realistic shape regimes (small-B, L up to 256K compressed).

Timing uses **CUDA graph replay** to amortize launch overhead (~3-5 µs on
Blackwell). We capture N invocations of the same kernel, replay the graph
many times, divide.

Run::

    .venv/bin/python benchmarks/kernels/benchmark_fast_topk_v2.py
"""

from __future__ import annotations

import argparse
import statistics
import sys

import torch

import vllm._C  # noqa: F401  ensures schemas are registered
from vllm.v1.attention.ops.deepseek_v4_ops.fast_topk import (
    fast_topk_v2_raw,
    plan_topk_v2,
    workspace_ints_per_batch,
)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024  # bytes; matches sparse_attn_indexer.py


def _capture_graph(callable_fn, *, calls_per_graph: int) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        callable_fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for _ in range(calls_per_graph):
                callable_fn()
    torch.cuda.current_stream().wait_stream(s)
    return g


def time_graph_us(graph: torch.cuda.CUDAGraph, *, calls_per_graph: int,
                  warmup: int = 5, replays: int = 30) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(replays):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / calls_per_graph)
    return statistics.median(samples)


def make_inputs(batch_size: int, seq_len: int, *, seed: int = 0):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(seed)
    L = (seq_len + 3) & ~3
    scores = torch.randn(batch_size, L, generator=g, dtype=torch.float32,
                         device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32,
                          device=device)
    return scores, seq_lens, L


def bench_persistent_topk(scores, seq_lens, k, *, calls_per_graph: int) -> float:
    B = scores.shape[0]
    output = scores.new_empty((B, k), dtype=torch.int32)
    workspace = scores.new_empty((RADIX_TOPK_WORKSPACE_SIZE,), dtype=torch.uint8)
    max_seq_len = scores.shape[1]

    def run():
        torch.ops._C.persistent_topk(
            scores, seq_lens, output, workspace, k, max_seq_len)

    graph = _capture_graph(run, calls_per_graph=calls_per_graph)
    return time_graph_us(graph, calls_per_graph=calls_per_graph)


def bench_fast_topk_v2(scores, seq_lens, k, *,
                       calls_per_graph: int) -> float:
    B = scores.shape[0]
    metadata = plan_topk_v2(seq_lens)
    workspace = scores.new_empty((B, workspace_ints_per_batch()),
                                 dtype=torch.int32)
    topk_indices = scores.new_empty((B, k), dtype=torch.int32)

    def run():
        fast_topk_v2_raw(scores, seq_lens, topk=k,
                         metadata=metadata, workspace=workspace,
                         topk_indices=topk_indices)

    graph = _capture_graph(run, calls_per_graph=calls_per_graph)
    return time_graph_us(graph, calls_per_graph=calls_per_graph)


def fmt(us: float) -> str:
    return f"{us:8.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[1, 4, 16, 32, 64, 128, 256])
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=[1024, 4096, 16384, 32768, 65536, 131072])
    parser.add_argument("--ks", type=int, nargs="+",
                        default=[512, 1024])
    parser.add_argument("--calls-per-graph", type=int, default=64)
    parser.add_argument("--replays", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark.", file=sys.stderr)
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"(SM {torch.cuda.get_device_capability(0)})")
    print(f"calls_per_graph={args.calls_per_graph}, replays={args.replays}")
    print("Per-call medians via CUDA graph replay (host launch overhead "
          "amortized).\n")

    for k in args.ks:
        print(f"=== k = {k} ===")
        print(f"{'B':>4} {'L':>7} | {'persistent_topk':>17} | "
              f"{'fast_topk_v2':>14} | {'speedup':>8} | {'path':<14}")
        print("-" * 80)
        for B in args.batch_sizes:
            for L in args.seq_lens:
                # Skip seq_lens beyond persistent_topk's k-dependent useful
                # range. Both kernels handle up to 256K with k=1024.
                try:
                    scores, seq_lens, _ = make_inputs(B, L, seed=B * L * k)
                    p_us = bench_persistent_topk(
                        scores, seq_lens, k,
                        calls_per_graph=args.calls_per_graph)
                    f_us = bench_fast_topk_v2(
                        scores, seq_lens, k,
                        calls_per_graph=args.calls_per_graph)
                    speedup = p_us / f_us if f_us > 0 else float("inf")

                    if L <= k:
                        path = "trivial"
                    elif L <= 4 * 4 * 1024:
                        path = "register-1p"
                    elif L <= 32768:
                        path = "register-2p"
                    elif B <= 15:
                        path = "cluster-fused"
                    else:
                        path = "cluster-2stg"

                    print(
                        f"{B:>4} {L:>7} | "
                        f"{fmt(p_us):>14} us | "
                        f"{fmt(f_us):>11} us | "
                        f"{speedup:>5.2f}x  | {path}"
                    )
                except RuntimeError as e:
                    print(f"{B:>4} {L:>7} | ERROR: {e}")
        print()


if __name__ == "__main__":
    main()
