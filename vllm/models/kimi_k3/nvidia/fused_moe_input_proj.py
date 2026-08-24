# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE input projection for Kimi-K3 latent MoE.

The router gate, the routed-expert latent down projection and the shared-expert
gate/up projection all read the same hidden states with the same K, so they are
merged into one weight and, on small batches, one GEMM.
"""

import contextlib
from collections.abc import Iterator
from typing import Any

import torch
from torch.nn.parameter import Parameter

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.parameter import BasevLLMParameter

# Token-count cutoff for the single fused GEMM. At or below this many tokens the
# three projections are individually too narrow to fill the device, so one wide
# GEMM beats three launches; above it each GEMM already saturates it, and the
# router is split back out so the other two can stay in bf16. Set to 0 to run
# the two-GEMM form at every batch size.
FUSED_MOE_INPUT_PROJ_TOKEN_THRESHOLD = 1024


def _materialize(block: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Copy a column block of the fused output into a packed tensor.

    Column blocks of a row-major GEMM output are row-strided, and the
    downstream MXFP8 quantizer and ``situ_and_mul`` both assume packed rows, so
    each block is materialized in one elementwise kernel. Casting to bf16 here
    also reproduces the rounding the separate bf16 GEMMs used to apply.
    """
    out = torch.empty(block.shape, dtype=dtype, device=block.device)
    out.copy_(block)
    return out


class KimiK3FusedMoEInputProj(MergedColumnParallelLinear):
    """The three MoE input projections in one column-parallel weight.

    Local layout, ``[E + latent + 2 * shared_intermediate_per_partition,
    hidden]``:

    | rows                          | block                   | TP          |
    |-------------------------------|-------------------------|-------------|
    | ``[0, E)``                    | router                  | replicated  |
    | ``[E, E + latent)``           | latent down projection  | replicated  |
    | ``[E + latent, ...)``         | shared gate ‖ shared up | column      |

    Router first, so the two-GEMM path reads the contiguous suffix
    ``weight[E:]``; shared gate and up last and adjacent, so ``situ_and_mul``'s
    ``d = width // 2`` split needs no reordering.

    The replicated shards declare ``tp_size`` times their real width, so the
    merged shard offsets line up while each rank stores the full copy; loading
    them forces ``tp_rank = 0``. Same trick as
    ``_KimiGDNMergedColumnParallelLinear`` in ``kda.py``.
    """

    ROUTER_SHARD_ID = 0
    LATENT_SHARD_ID = 1
    SHARED_GATE_SHARD_ID = 2
    SHARED_UP_SHARD_ID = 3
    _REPLICATED_SHARD_IDS = frozenset({ROUTER_SHARD_ID, LATENT_SHARD_ID})

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        latent_size: int,
        shared_intermediate_size: int,
        prefix: str = "",
    ) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        super().__init__(
            input_size=hidden_size,
            output_sizes=[
                num_experts * tp_size,
                latent_size * tp_size,
                shared_intermediate_size,
                shared_intermediate_size,
            ],
            bias=False,
            quant_config=None,
            prefix=prefix,
        )
        self.num_experts = num_experts
        self.latent_size = latent_size

    @contextlib.contextmanager
    def _loading_shard(self, param: Any, loaded_shard_id: Any) -> Iterator[None]:
        """Force ``tp_rank = 0`` while a replicated shard is being loaded."""
        if loaded_shard_id not in self._REPLICATED_SHARD_IDS:
            yield
            return
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        self.tp_rank = 0
        if param_tp_rank is not None:
            param.tp_rank = 0
        try:
            yield
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        with self._loading_shard(param, loaded_shard_id):
            super().weight_loader(param, loaded_weight, loaded_shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        with self._loading_shard(param, loaded_shard_id):
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)

    def project(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the input projections.

        Args:
            hidden_states: ``[num_tokens, hidden_size]`` activations.

        Returns:
            ``(router_logits, latent, shared_gate_up)``. ``router_logits`` is
            fp32, the other two match ``hidden_states.dtype``; all three are
            packed row-major.
        """
        num_tokens = hidden_states.shape[0]
        weight = self.weight
        num_experts = self.num_experts
        latent_end = num_experts + self.latent_size

        if num_tokens <= FUSED_MOE_INPUT_PROJ_TOKEN_THRESHOLD:
            # One GEMM for all three. fp32 output is what makes that possible:
            # a single GEMM emits a single dtype, and the router needs fp32 for
            # stable expert selection. The other two blocks are rounded back to
            # the activation dtype in _materialize.
            fused = torch.mm(hidden_states, weight.t(), out_dtype=torch.float32)
            router_logits = fused[:, :num_experts].contiguous()
            latent = _materialize(fused[:, num_experts:latent_end], hidden_states.dtype)
            shared_gate_up = _materialize(fused[:, latent_end:], hidden_states.dtype)
            return router_logits, latent, shared_gate_up

        # Large batches: the GEMMs saturate the device on their own, so keep the
        # router separate (in fp32, as before) and fuse only the two bf16
        # projections, which live in the contiguous suffix of the weight.
        router_logits = torch.mm(
            hidden_states, weight[:num_experts].t(), out_dtype=torch.float32
        )
        fused = torch.mm(hidden_states, weight[num_experts:].t())
        latent = fused[:, : self.latent_size].contiguous()
        shared_gate_up = fused[:, self.latent_size :].contiguous()
        return router_logits, latent, shared_gate_up
