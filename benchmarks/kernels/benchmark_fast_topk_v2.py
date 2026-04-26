# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench: fast_topk_v2 vs persistent_topk for k=512.

Both ops select the top-512 entries per row of a [B, L] float32 score tensor.
The vLLM `persistent_topk` is the existing path used by the indexer
(`sparse_attn_indexer.py`); `fast_topk_v2` is the new sm_90+ port from
sglang that adds Hopper thread-block clusters and a fused page-table gather.

For fairness:
- `persistent_topk` writes raw indices into a (B, k) tensor.
- `fast_topk_v2` writes page-table-resolved indices. To match the same
  output (raw indices) we use ``page_size=1`` with an identity page table —
  page_to_indices becomes a no-op, so the two kernels produce comparable
  results on the hot path.

Timing uses **CUDA graph replay** to amortize launch overhead. We capture
N invocations of the same kernel into a single graph, replay the graph
many times, and divide. This isolates kernel work from per-launch host
latency (~3–5 µs on Blackwell) which would otherwise dominate at small
shapes.

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
    plan_topk_v2,
    workspace_ints_per_batch,
)

K = 512
RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024  # bytes; matches sparse_attn_indexer.py


def _capture_graph(callable_fn, *, calls_per_graph: int) -> torch.cuda.CUDAGraph:
    """Capture a CUDA graph that invokes `callable_fn` `calls_per_graph` times.

    The kernel is run a few times outside capture (warmup + allocator priming)
    to avoid capturing one-shot setup work like cudaFuncSetAttribute.
    """
    # Outside-capture warmup: lets cudaFuncSetAttribute's static cache fire
    # and primes any cublas/cudnn lookups the kernel might trigger.
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
    """Median per-call latency (µs) over many graph replays."""
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
        # ms -> us, then per-call.
        samples.append(start.elapsed_time(end) * 1000.0 / calls_per_graph)
    return statistics.median(samples)


def make_inputs(batch_size: int, seq_len: int, *, seed: int = 0):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(seed)
    L = (seq_len + 3) & ~3  # TMA wants stride%4 == 0
    scores = torch.randn(batch_size, L, generator=g, dtype=torch.float32,
                         device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32,
                          device=device)
    # Identity page table with page_size=1 -> page_to_indices is identity.
    page_table = (
        torch.arange(L, dtype=torch.int32, device=device)
        .unsqueeze(0).expand(batch_size, -1).contiguous()
    )
    return scores, seq_lens, page_table, L


def bench_persistent_topk(scores, seq_lens, *, calls_per_graph: int) -> float:
    B = scores.shape[0]
    output = scores.new_empty((B, K), dtype=torch.int32)
    workspace = scores.new_empty((RADIX_TOPK_WORKSPACE_SIZE,),
                                 dtype=torch.uint8)
    max_seq_len = scores.shape[1]

    def run():
        torch.ops._C.persistent_topk(
            scores, seq_lens, output, workspace, K, max_seq_len)

    graph = _capture_graph(run, calls_per_graph=calls_per_graph)
    return time_graph_us(graph, calls_per_graph=calls_per_graph)


def bench_fast_topk_v2(scores, seq_lens, page_table, *,
                       calls_per_graph: int) -> tuple[float, float]:
    """Returns (kernel_only_us, with_plan_us). Both are per-call medians.

    - kernel_only: just torch.ops._C.fast_topk_v2 (plan was done once,
      metadata reused). This is the realistic decode-with-cudagraph case.
    - with_plan:   plan + main kernel captured together. Pessimistic for
      cudagraph (plan would normally be done once outside the graph and
      reused, since it depends only on seq_lens shape).
    """
    B = scores.shape[0]
    metadata = plan_topk_v2(seq_lens)
    workspace = scores.new_empty((B, workspace_ints_per_batch()),
                                 dtype=torch.int32)
    page_indices = scores.new_empty((B, K), dtype=torch.int32)

    def run_kernel_only():
        torch.ops._C.fast_topk_v2(
            scores, seq_lens, page_table, page_indices, 1, workspace, metadata)

    def run_with_plan():
        torch.ops._C.fast_topk_v2_plan(seq_lens, metadata, 0)
        torch.ops._C.fast_topk_v2(
            scores, seq_lens, page_table, page_indices, 1, workspace, metadata)

    g_kernel = _capture_graph(run_kernel_only, calls_per_graph=calls_per_graph)
    g_with_plan = _capture_graph(run_with_plan, calls_per_graph=calls_per_graph)
    return (
        time_graph_us(g_kernel, calls_per_graph=calls_per_graph),
        time_graph_us(g_with_plan, calls_per_graph=calls_per_graph),
    )


def fmt(us: float) -> str:
    return f"{us:8.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[1, 4, 16, 32, 64, 128, 256])
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=[1024, 4096, 16384, 32768, 65536, 131072])
    parser.add_argument("--calls-per-graph", type=int, default=64,
                        help="Invocations captured per graph (amortizes "
                             "graph-replay overhead, ~3-5 µs).")
    parser.add_argument("--replays", type=int, default=30,
                        help="Graph replays per measurement.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark.", file=sys.stderr)
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"(SM {torch.cuda.get_device_capability(0)})")
    print(f"k = {K}; calls_per_graph={args.calls_per_graph}, "
          f"replays={args.replays}")
    print("Per-call medians via CUDA graph replay (host launch overhead "
          "amortized).\n")

    print(f"{'B':>4} {'L':>7} | "
          f"{'persistent_topk':>17} | "
          f"{'fast_topk_v2':>14} | "
          f"{'fast_topk_v2+plan':>19} | "
          f"{'speedup':>9} | {'path':<14}")
    print("-" * 105)

    for B in args.batch_sizes:
        for L in args.seq_lens:
            try:
                scores, seq_lens, page_table, _ = make_inputs(B, L, seed=B * L)
                p_us = bench_persistent_topk(
                    scores, seq_lens, calls_per_graph=args.calls_per_graph)
                f_us, fp_us = bench_fast_topk_v2(
                    scores, seq_lens, page_table,
                    calls_per_graph=args.calls_per_graph)
                speedup = p_us / f_us if f_us > 0 else float("inf")

                if L <= 512:
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
                    f"{fmt(fp_us):>16} us | "
                    f"{speedup:>6.2f}x  | {path}"
                )
            except RuntimeError as e:
                print(f"{B:>4} {L:>7} | ERROR: {e}")


if __name__ == "__main__":
    main()
