# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP endpoints for deterministic benchmark scheduling.

POST /v1/benchmark/add_requests
    Add requests with explicit token IDs (bypasses tokenizer).
    Returns request IDs for use in /v1/benchmark/step.

POST /v1/benchmark/step
    Run exactly one deterministic forward pass.
    Supports single-rank and multi-rank DP execution.
    Engine stays paused after each step; use POST /resume to exit.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


def _engine(request: Request):
    return request.app.state.engine_client


@router.post("/v1/benchmark/add_requests")
async def add_requests(raw_request: Request):
    """Add requests using raw token IDs, bypassing the text tokenizer.

    Request body (JSON):
        {
            "requests": [
                {
                    "prompt_token_ids": [1, 2, 3, ...],
                    "max_tokens": 100,
                    "dp_rank": 0          // optional, for DP routing
                },
                ...
            ]
        }

    Response:
        {"request_ids": ["uuid-A", "uuid-B", ...]}

    The returned IDs are the internal scheduler request IDs. Use them
    in the "batch" or "ranks" fields of POST /v1/benchmark/step.
    """
    body = await raw_request.json()
    engine = _engine(raw_request)
    ids = await engine.add_benchmark_requests(body["requests"])
    return JSONResponse({"request_ids": ids})


@router.post("/v1/benchmark/step")
async def benchmark_step(raw_request: Request):
    """Run one deterministic forward pass, optionally across DP ranks.

    Single-rank body (shorthand for rank 0):
        {"batch": {"<req_id>": <num_tokens>, ...}}

    Multi-rank DP body (all ranks execute in parallel):
        {"ranks": {"0": {"<req_id>": <num_tokens>}, "1": {...}, ...}}

    Optional profiling fields (require --profiler-config at server startup):
        "profile": true          -- wrap the forward pass with the GPU profiler
        "profile_prefix": "str"  -- trace-file name prefix (default: none)

    Response (always keyed by rank string):
        {
            "0": {"duration_ns": int, "num_tokens": int, "executed": bool},
            "1": ...
        }

    The engine stays paused (PAUSED_ALL) after each step. Call
    POST /resume to return to normal greedy serving.
    """
    body = await raw_request.json()
    if "batch" in body:
        per_rank_specs: dict[int, dict[str, int]] = {0: body["batch"]}
    else:
        per_rank_specs = {int(k): v for k, v in body["ranks"].items()}

    profile: bool = body.get("profile", False)
    profile_prefix: str | None = body.get("profile_prefix", None)

    engine = _engine(raw_request)
    result = await engine.run_benchmark_step(
        per_rank_specs, profile=profile, profile_prefix=profile_prefix
    )
    return JSONResponse(result)


@router.post("/resume")
async def resume_serving(raw_request: Request):
    """Resume normal greedy serving after benchmark steps.

    Unpauses the engine so it returns to processing requests from the queue.
    Call this after all benchmark steps are complete.

    Response:
        {"status": "resumed"}
    """
    engine = _engine(raw_request)
    await engine.resume_generation()
    return JSONResponse({"status": "resumed"})
