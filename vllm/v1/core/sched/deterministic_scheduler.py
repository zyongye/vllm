# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic scheduler mixin for benchmarking.

Provides per-step control over which requests are batched and how many tokens
are processed per request. Implemented as a mixin so it works with both
Scheduler and AsyncScheduler.
"""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.sched.interface import SchedulerInterface

logger = init_logger(__name__)


class DeterministicMixin:
    """Mixin that adds explicit batch spec support to any Scheduler subclass.

    When a batch spec is set via set_batch_spec(), the next call to
    schedule() will process only the listed requests for exactly the
    specified number of tokens.

    When no spec is set, schedule() falls back to the wrapped scheduler's
    normal greedy behavior — zero behavior change for normal serving.

    Scope: _schedule_deterministic handles standard decode/prefill only.
    Spec-decode (EAGLE), encoder-decoder, and KV connectors raise
    AssertionError in deterministic mode.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_spec: dict[str, int] | None = None

    def set_batch_spec(self, spec: dict[str, int]) -> None:
        """Set the batch spec for the next schedule() call.

        Args:
            spec: mapping of request_id -> num_tokens_to_schedule.
                  Only listed requests will be scheduled in the next step.
        """
        self._batch_spec = spec

    def schedule(self) -> SchedulerOutput:
        if self._batch_spec is None:
            return super().schedule()  # type: ignore[misc]
        return self._schedule_deterministic()

    def _schedule_deterministic(self) -> SchedulerOutput:
        """Execute one deterministic scheduling step using the stored spec."""
        # getattr avoids mypy attr-defined errors on the mixin (attributes are
        # provided by the concrete Scheduler base class at runtime).
        if getattr(self, "use_eagle", False):  # type: ignore[attr-defined]
            raise AssertionError(
                "Deterministic scheduling does not support EAGLE spec-decode"
            )
        if getattr(self, "is_encoder_decoder", False):  # type: ignore[attr-defined]
            raise AssertionError(
                "Deterministic scheduling does not support encoder-decoder models"
            )
        if getattr(self, "connector", None) is not None:  # type: ignore[attr-defined]
            raise AssertionError(
                "Deterministic scheduling does not support KV connectors"
            )

        spec = self._batch_spec
        self._batch_spec = None

        # Carry forward finished_req_ids from the previous step, same as
        # Scheduler.schedule() does at the top of its loop.
        finished_req_ids = self.finished_req_ids.copy()  # type: ignore[attr-defined]
        self.finished_req_ids.clear()  # type: ignore[attr-defined]

        self.kv_cache_manager.new_step_starts()  # type: ignore[attr-defined]

        scheduled_new_reqs: list[NewRequestData] = []
        cached_reqs: list = []  # list of Request objects already running
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}

        assert spec is not None
        for req_id, num_tokens in spec.items():
            if req_id not in self.requests:  # type: ignore[attr-defined]
                raise ValueError(
                    f"Request '{req_id}' not found in scheduler. "
                    "Add it via /v1/benchmark/add_requests first."
                )
            request = self.requests[req_id]  # type: ignore[attr-defined]

            if request.status == RequestStatus.WAITING:
                # First time this request is scheduled: check prefix cache,
                # allocate KV blocks, move to running.
                kv_mgr = self.kv_cache_manager  # type: ignore[attr-defined]
                computed_blocks, num_computed = kv_mgr.get_computed_blocks(request)
                request.num_computed_tokens = num_computed
                # Record prefix-cached tokens (mirrors what Scheduler.schedule
                # does at line ~830; initialised to -1, set on first schedule).
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed

                new_blocks = self.kv_cache_manager.allocate_slots(  # type: ignore[attr-defined]
                    request,
                    num_tokens,
                    num_new_computed_tokens=num_computed,
                    new_computed_blocks=computed_blocks,
                )
                if new_blocks is None:
                    raise RuntimeError(
                        f"KV cache OOM scheduling request '{req_id}' "
                        f"for {num_tokens} tokens"
                    )

                self.waiting.remove_request(request)  # type: ignore[attr-defined]
                request.status = RequestStatus.RUNNING
                self.running.append(request)  # type: ignore[attr-defined]
                # v2 model runner needs prefill_token_ids; others ignore it.
                prefill_ids = (
                    request.all_token_ids  # type: ignore[attr-defined]
                    if self.use_v2_model_runner  # type: ignore[attr-defined]
                    else None
                )
                scheduled_new_reqs.append(
                    NewRequestData.from_request(
                        request, new_blocks.get_block_ids(), prefill_ids
                    )
                )

            elif request.status == RequestStatus.RUNNING:
                new_blocks = self.kv_cache_manager.allocate_slots(  # type: ignore[attr-defined]
                    request, num_tokens
                )
                if new_blocks is None:
                    raise RuntimeError(
                        f"KV cache OOM scheduling request '{req_id}' "
                        f"for {num_tokens} tokens"
                    )
                cached_reqs.append(request)

            else:
                raise ValueError(
                    f"Request '{req_id}' has status {request.status}; "
                    "only WAITING and RUNNING requests can be scheduled."
                )

            req_to_new_blocks[req_id] = new_blocks
            num_scheduled_tokens[req_id] = num_tokens

        # Build CachedRequestData for already-running requests.
        # Always include all_token_ids (safe; model runner uses it to refresh
        # its token cache when prev_step_scheduled_req_ids is stale).
        use_pp: bool = getattr(self, "use_pp", False)  # type: ignore[attr-defined]
        sched_cfg = getattr(self, "scheduler_config", None)  # type: ignore[attr-defined]
        async_sched: bool = sched_cfg is not None and sched_cfg.async_scheduling

        cached_req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        all_token_ids: dict[str, list[int]] = {}
        new_block_ids: list = []
        num_computed_tokens_list: list[int] = []
        num_output_tokens_list: list[int] = []

        for req in cached_reqs:
            req_id = req.request_id
            cached_req_ids.append(req_id)
            if use_pp and not async_sched:
                num_tokens = num_scheduled_tokens[req_id]
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            # Always send full token history for cached reqs so the model
            # runner can rebuild its state if needed (safe but slightly wasteful).
            all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens_list.append(req.num_computed_tokens)
            num_output_tokens_list.append(req.num_output_tokens)

        scheduled_cached_reqs = CachedRequestData(
            req_ids=cached_req_ids,
            resumed_req_ids=set(),
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens_list,
            num_output_tokens=num_output_tokens_list,
        )

        # Update prev_step_scheduled_req_ids so the next step can tell which
        # requests were already scheduled (mirrors what Scheduler.schedule does).
        self.prev_step_scheduled_req_ids.clear()  # type: ignore[attr-defined]
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())  # type: ignore[attr-defined]

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=scheduled_cached_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0]
            * len(
                self.kv_cache_config.kv_cache_groups  # type: ignore[attr-defined]
            ),
            finished_req_ids=finished_req_ids,
            free_encoder_mm_hashes=[],
        )

        # Call _update_after_schedule exactly as Scheduler.schedule() does.
        # This advances num_computed_tokens and (for AsyncScheduler) increments
        # num_output_placeholders so that update_from_output() is consistent.
        self._update_after_schedule(scheduler_output)  # type: ignore[attr-defined]
        return scheduler_output


def wrap_with_deterministic(
    base_cls: type["SchedulerInterface"],
) -> type["SchedulerInterface"]:
    """Return a new class that prepends DeterministicMixin to base_cls.

    If base_cls already has DeterministicMixin in its MRO, returns it as-is.
    """
    if issubclass(base_cls, DeterministicMixin):
        return base_cls
    return type(  # type: ignore[return-value]
        f"Deterministic{base_cls.__name__}",
        (DeterministicMixin, base_cls),
        {},
    )
