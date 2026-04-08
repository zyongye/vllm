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
      [--profile-prefill] [--profile-decode [STEP ...]]
      [--profile-prefix PREFIX]

Profiling examples (requires --profiler-config at server startup):
  # Profile only the prefill step:
  python bench_deterministic.py --profile-prefill

  # Profile decode steps 0 and 5:
  python bench_deterministic.py --profile-decode 0 5

  # Profile prefill + all decode steps with a custom prefix:
  python bench_deterministic.py --profile-prefill --profile-decode all \\
      --profile-prefix my_run
"""

import argparse
import statistics
import time

import requests


def add_requests(
    url: str, dp_size: int, batch_per_rank: int, input_len: int, decode_steps: int
):
    """Add batch_per_rank requests to each DP rank. Returns {rank: [req_ids]}."""
    import random

    rank_req_ids: dict[int, list[str]] = {}
    for rank in range(dp_size):
        # Use distinct random tokens per request so prefix caching
        # doesn't make all but the first request trivially cheap.
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


def run_step(
    url: str,
    per_rank_specs: dict[int, dict[str, int]],
    profile: bool = False,
    profile_prefix: str | None = None,
) -> dict[str, dict]:
    """POST /v1/benchmark/step with multi-rank spec. Returns rank -> result."""
    payload: dict = {"ranks": {str(k): v for k, v in per_rank_specs.items()}}
    if profile:
        payload["profile"] = True
        if profile_prefix is not None:
            payload["profile_prefix"] = profile_prefix
    resp = requests.post(f"{url}/v1/benchmark/step", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def resume(url: str):
    resp = requests.post(f"{url}/resume", timeout=10)
    resp.raise_for_status()


def fmt_ms(ns: int) -> str:
    return f"{ns / 1e6:.2f} ms"


def _parse_profile_decode(values: list[str] | None) -> set[int] | None:
    """Parse --profile-decode args into a set of step indices, or None=all."""
    if values is None:
        return set()  # flag not given → no decode profiling
    if not values or values == ["all"]:
        return None  # None sentinel = profile every decode step
    return {int(v) for v in values}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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

    # Profiling options (require --profiler-config at server startup)
    prof = parser.add_argument_group(
        "profiling",
        "Wrap specific steps with the GPU profiler "
        "(requires --profiler-config at server startup).",
    )
    prof.add_argument(
        "--profile-prefill",
        action="store_true",
        help="Profile the prefill step.",
    )
    prof.add_argument(
        "--profile-decode",
        nargs="*",
        metavar="STEP",
        help=(
            "Profile decode steps. With no args, profiles all steps. "
            "Pass step indices (0-based) to profile specific steps, "
            'or the literal string "all".'
        ),
    )
    prof.add_argument(
        "--profile-prefix",
        default=None,
        metavar="PREFIX",
        help=(
            "Trace-file name prefix. The phase and step index are appended "
            'automatically (e.g. "run1" → "run1_prefill", "run1_decode_00").'
        ),
    )

    args = parser.parse_args()

    url = args.url
    dp = args.dp
    batch_per_rank = args.batch_per_rank
    input_len = args.input_len
    decode_steps = args.decode_steps
    total_reqs = dp * batch_per_rank

    profile_prefill: bool = args.profile_prefill
    # None → all steps, set → specific indices, empty set → none
    profile_decode_steps: set[int] | None = _parse_profile_decode(args.profile_decode)
    profile_prefix: str | None = args.profile_prefix

    def decode_should_profile(step_idx: int) -> bool:
        if profile_decode_steps is None:
            return True  # all steps
        return step_idx in profile_decode_steps

    def make_prefix(phase: str) -> str | None:
        if profile_prefix:
            return f"{profile_prefix}_{phase}"
        return phase

    print(f"Config: dp={dp}, batch_per_rank={batch_per_rank}, total_reqs={total_reqs}")
    print(f"        input_len={input_len}, decode_steps={decode_steps}")
    if profile_prefill or profile_decode_steps != set():
        print(
            f"        profile: prefill={profile_prefill}, "
            f"decode={args.profile_decode}, prefix={profile_prefix!r}"
        )
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
    prefill_result = run_step(
        url,
        prefill_specs,
        profile=profile_prefill,
        profile_prefix=make_prefix("prefill") if profile_prefill else None,
    )

    prefill_ns_per_rank = [prefill_result[str(r)]["duration_ns"] for r in range(dp)]
    prefill_wall_ms = max(prefill_ns_per_rank) / 1e6
    prefill_tokens = sum(prefill_result[str(r)]["num_tokens"] for r in range(dp))
    profiled_tag = " [profiled]" if profile_prefill else ""
    print(f"\n=== Prefill{profiled_tag} ===")
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

    num_profiled_decode = sum(
        1 for i in range(decode_steps) if decode_should_profile(i)
    )
    profiled_tag = f" [{num_profiled_decode} profiled]" if num_profiled_decode else ""
    print(f"\n=== Decode ({decode_steps} steps{profiled_tag}) ===")
    for step in range(decode_steps):
        do_profile = decode_should_profile(step)
        prefix = make_prefix(f"decode_{step:04d}") if do_profile else None
        result = run_step(url, decode_specs, profile=do_profile, profile_prefix=prefix)
        wall_ns = max(result[str(r)]["duration_ns"] for r in range(dp))
        decode_wall_ms_list.append(wall_ns / 1e6)
        if do_profile:
            print(f"  step {step:4d}: {wall_ns / 1e6:.2f} ms [profiled]")

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
