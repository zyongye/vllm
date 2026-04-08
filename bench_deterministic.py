#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic scheduler stress test via HTTP endpoints.

Workflow per round:
  1. POST /v1/benchmark/add_requests  — inject token sequences per DP rank
  2. POST /v1/benchmark/step          — prefill all ranks in parallel
  3. POST /v1/benchmark/step (x N)    — decode steps, one token per request
  4. POST /resume                     — return to normal serving

Usage:
  python bench_deterministic.py [--url URL] [--dp DP_SIZE]
      [--batch-per-rank N] [--input-len L] [--decode-steps D]
"""

import argparse
import statistics
import time

import requests


def add_requests(
    url: str, dp_size: int, batch_per_rank: int, input_len: int, decode_steps: int
):
    """Add batch_per_rank requests to each DP rank. Returns {rank: [req_ids]}."""
    rank_req_ids: dict[int, list[str]] = {}
    for rank in range(dp_size):
        # Use distinct random tokens per request so prefix caching
        # doesn't make all but the first request trivially cheap.
        import random

        payload = {
            "requests": [
                {
                    "prompt_token_ids": random.choices(range(1, 32000), k=input_len),
                    "max_tokens": decode_steps,
                    "dp_rank": rank,
                }
                for _ in range(batch_per_rank)
            ]
        }
        resp = requests.post(
            f"{url}/v1/benchmark/add_requests", json=payload, timeout=30
        )
        resp.raise_for_status()
        rank_req_ids[rank] = resp.json()["request_ids"]
    return rank_req_ids


def run_step(url: str, per_rank_specs: dict[int, dict[str, int]]) -> dict[str, dict]:
    """POST /v1/benchmark/step with multi-rank spec. Returns rank -> result."""
    payload = {"ranks": {str(k): v for k, v in per_rank_specs.items()}}
    resp = requests.post(f"{url}/v1/benchmark/step", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def resume(url: str):
    resp = requests.post(f"{url}/resume", timeout=10)
    resp.raise_for_status()


def fmt_ms(ns: int) -> str:
    return f"{ns / 1e6:.2f} ms"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--dp", type=int, default=1, help="Number of DP ranks")
    parser.add_argument(
        "--batch-per-rank", type=int, default=8, help="Requests per DP rank"
    )
    parser.add_argument(
        "--input-len", type=int, default=512, help="Prompt token length"
    )
    parser.add_argument(
        "--decode-steps", type=int, default=32, help="Number of decode steps"
    )
    args = parser.parse_args()

    url = args.url
    dp = args.dp
    batch_per_rank = args.batch_per_rank
    input_len = args.input_len
    decode_steps = args.decode_steps
    total_reqs = dp * batch_per_rank

    print(f"Config: dp={dp}, batch_per_rank={batch_per_rank}, total_reqs={total_reqs}")
    print(f"        input_len={input_len}, decode_steps={decode_steps}")
    print()

    # 1. Add requests
    t0 = time.monotonic()
    rank_req_ids = add_requests(url, dp, batch_per_rank, input_len, decode_steps)
    add_ms = (time.monotonic() - t0) * 1000
    print(f"add_requests: {add_ms:.1f} ms  ({total_reqs} reqs across {dp} rank(s))")

    # 2. Prefill step
    prefill_specs: dict[int, dict[str, int]] = {
        rank: {req_id: input_len for req_id in req_ids}
        for rank, req_ids in rank_req_ids.items()
    }
    prefill_result = run_step(url, prefill_specs)

    prefill_ns_per_rank = [prefill_result[str(r)]["duration_ns"] for r in range(dp)]
    prefill_wall_ms = max(prefill_ns_per_rank) / 1e6
    prefill_tokens = sum(prefill_result[str(r)]["num_tokens"] for r in range(dp))
    print("\n=== Prefill ===")
    print(f"  Wall time (max rank): {prefill_wall_ms:.2f} ms")
    print(f"  Total tokens:         {prefill_tokens}")
    print(
        f"  Throughput:           {prefill_tokens / (prefill_wall_ms / 1000):.0f} tok/s"
    )
    for r in range(dp):
        d = prefill_result[str(r)]
        tok = d["num_tokens"]
        dur = fmt_ms(d["duration_ns"])
        print(f"  rank {r}: {dur}, {tok} tokens, executed={d['executed']}")

    # 3. Decode steps
    decode_specs: dict[int, dict[str, int]] = {
        rank: {req_id: 1 for req_id in req_ids}
        for rank, req_ids in rank_req_ids.items()
    }

    decode_wall_ms_list: list[float] = []
    decode_ns_per_rank_per_step: list[list[int]] = []

    print(f"\n=== Decode ({decode_steps} steps) ===")
    for step in range(decode_steps):
        result = run_step(url, decode_specs)
        wall_ns = max(result[str(r)]["duration_ns"] for r in range(dp))
        decode_wall_ms_list.append(wall_ns / 1e6)
        decode_ns_per_rank_per_step.append(
            [result[str(r)]["duration_ns"] for r in range(dp)]
        )

    total_decode_tokens = total_reqs * decode_steps
    total_decode_ms = sum(decode_wall_ms_list)
    mean_step_ms = statistics.mean(decode_wall_ms_list)
    median_step_ms = statistics.median(decode_wall_ms_list)
    p99_step_ms = sorted(decode_wall_ms_list)[int(len(decode_wall_ms_list) * 0.99)]

    print(f"  Steps:                {decode_steps}")
    print(f"  Total decode tokens:  {total_decode_tokens}")
    print(f"  Total wall time:      {total_decode_ms:.2f} ms")
    decode_tps = total_decode_tokens / (total_decode_ms / 1000)
    print(f"  Throughput:           {decode_tps:.0f} tok/s")
    print(f"  Step latency mean:    {mean_step_ms:.2f} ms")
    print(f"  Step latency median:  {median_step_ms:.2f} ms")
    print(f"  Step latency p99:     {p99_step_ms:.2f} ms")
    print(f"  TPOT (per request):   {mean_step_ms:.2f} ms")

    # 4. Resume
    resume(url)
    print("\nResumed normal serving.")

    # Summary
    print(f"""
========================================
 Deterministic Scheduler Benchmark
========================================
 DP size:          {dp}
 Batch/rank:       {batch_per_rank}  (total: {total_reqs} reqs)
 Input length:     {input_len} tokens
 Decode steps:     {decode_steps}
----------------------------------------
 Prefill wall:     {prefill_wall_ms:.2f} ms
 Prefill tok/s:    {prefill_tokens / (prefill_wall_ms / 1000):.0f}
 Decode tok/s:     {total_decode_tokens / (total_decode_ms / 1000):.0f}
 Step lat mean:    {mean_step_ms:.2f} ms
 Step lat median:  {median_step_ms:.2f} ms
 Step lat p99:     {p99_step_ms:.2f} ms
========================================
""")


if __name__ == "__main__":
    main()
