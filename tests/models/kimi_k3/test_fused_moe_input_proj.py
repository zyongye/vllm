# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused MoE input projection must equal the three separate projections.

A swapped shard order or a wrong offset still runs and still produces plausible
text, so these pin the layout and the arithmetic rather than the behaviour.
"""

import pytest
import torch
from torch.multiprocessing import spawn

from tests.utils import (
    ensure_current_vllm_config,
    init_test_distributed_environment,
    multi_gpu_test,
)
from vllm.models.kimi_k3.nvidia.fused_moe_input_proj import (
    FUSED_MOE_INPUT_PROJ_TOKEN_THRESHOLD,
    KimiK3FusedMoEInputProj,
)
from vllm.models.kimi_k3.nvidia.model import KimiMLP
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="the fused projection needs cuBLAS' bf16 x bf16 -> fp32 GEMM",
)

HIDDEN_SIZE = 1024
NUM_EXPERTS = 32
LATENT_SIZE = 512
SHARED_INTERMEDIATE_SIZE = 512
DTYPE = torch.bfloat16

# Straddles FUSED_MOE_INPUT_PROJ_TOKEN_THRESHOLD so both the one-GEMM and the
# two-GEMM branch are exercised.
TOKEN_COUNTS = (1, 8, 256, 1024, 1025, 2048)


def _checkpoint_weights(device: torch.device) -> dict[str, torch.Tensor]:
    """The four unsharded tensors a checkpoint would provide."""
    torch.manual_seed(0)
    shapes = {
        "router": (NUM_EXPERTS, HIDDEN_SIZE),
        "latent": (LATENT_SIZE, HIDDEN_SIZE),
        "shared_gate": (SHARED_INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "shared_up": (SHARED_INTERMEDIATE_SIZE, HIDDEN_SIZE),
    }
    return {
        name: torch.randn(shape, device=device, dtype=DTYPE) / HIDDEN_SIZE**0.5
        for name, shape in shapes.items()
    }


def _build_loaded_proj(
    weights: dict[str, torch.Tensor], device: torch.device
) -> KimiK3FusedMoEInputProj:
    """Build the projection and load it through the real weight loader."""
    with torch.device(device):
        proj = KimiK3FusedMoEInputProj(
            hidden_size=HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            latent_size=LATENT_SIZE,
            shared_intermediate_size=SHARED_INTERMEDIATE_SIZE,
            prefix="fused_moe_input_proj",
        )
    param = proj.weight
    for name, shard_id in (
        ("router", KimiK3FusedMoEInputProj.ROUTER_SHARD_ID),
        ("latent", KimiK3FusedMoEInputProj.LATENT_SHARD_ID),
        ("shared_gate", KimiK3FusedMoEInputProj.SHARED_GATE_SHARD_ID),
        ("shared_up", KimiK3FusedMoEInputProj.SHARED_UP_SHARD_ID),
    ):
        param.weight_loader(param, weights[name], shard_id)
    return proj


def _reference(
    weights: dict[str, torch.Tensor],
    hidden_states: torch.Tensor,
    shard: slice,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """What the three separate projections produce for this rank."""
    router_logits = torch.mm(
        hidden_states, weights["router"].t(), out_dtype=torch.float32
    )
    latent = torch.mm(hidden_states, weights["latent"].t())
    gate_up = torch.cat(
        (weights["shared_gate"][shard], weights["shared_up"][shard]), dim=0
    )
    return router_logits, latent, torch.mm(hidden_states, gate_up.t())


def _check_matches_separate_projections(
    device: torch.device, tp_size: int, rank: int
) -> None:
    weights = _checkpoint_weights(device)
    proj = _build_loaded_proj(weights, device)
    shard_size = SHARED_INTERMEDIATE_SIZE // tp_size
    shard = slice(rank * shard_size, (rank + 1) * shard_size)

    for num_tokens in TOKEN_COUNTS:
        hidden_states = torch.randn(num_tokens, HIDDEN_SIZE, device=device, dtype=DTYPE)
        router_logits, latent, gate_up = proj.project(hidden_states)
        want_router, want_latent, want_gate_up = _reference(
            weights, hidden_states, shard
        )

        # Every consumer downstream assumes packed rows, and the MoE stack
        # assumes the activation dtype.
        for name, actual, want in (
            ("router_logits", router_logits, want_router),
            ("latent", latent, want_latent),
            ("gate_up", gate_up, want_gate_up),
        ):
            assert actual.is_contiguous(), f"{name} is not packed at M={num_tokens}"
            assert actual.shape == want.shape, name
            assert actual.dtype == want.dtype, name

        # The router shares the reference's fp32 accumulate and output, so only
        # the GEMM's accumulation order can differ. The other two round that
        # accumulator to bf16, as the separate bf16 GEMMs did.
        torch.testing.assert_close(router_logits, want_router, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(latent, want_latent, atol=8e-3, rtol=8e-3)
        torch.testing.assert_close(gate_up, want_gate_up, atol=8e-3, rtol=8e-3)


def _check_shard_layout(device: torch.device, tp_size: int, rank: int) -> None:
    """The replicated shards must be whole and the shared shards rank-local.

    Two ranks that swapped the shared gate and up offsets still produce a
    plausible fused weight, so compare the blocks directly.
    """
    weights = _checkpoint_weights(device)
    proj = _build_loaded_proj(weights, device)
    shard_size = SHARED_INTERMEDIATE_SIZE // tp_size
    shard = slice(rank * shard_size, (rank + 1) * shard_size)

    expected = torch.cat(
        (
            weights["router"],
            weights["latent"],
            weights["shared_gate"][shard],
            weights["shared_up"][shard],
        ),
        dim=0,
    )
    assert proj.weight.shape == expected.shape
    torch.testing.assert_close(proj.weight.data, expected, atol=0, rtol=0)


def _check_shared_experts_consume_gate_up(
    device: torch.device, tp_size: int, rank: int
) -> None:
    """Handing the gate/up projection over must not change what KimiMLP computes."""
    with torch.device(device):
        mlp = KimiMLP(
            hidden_size=HIDDEN_SIZE,
            intermediate_size=SHARED_INTERMEDIATE_SIZE,
            hidden_act="situ",
            quant_config=None,
            reduce_results=False,
            prefix="shared_experts",
            activation_situ_beta=1.0,
        )
    torch.manual_seed(1)
    mlp.gate_up_proj.weight.data.normal_(std=HIDDEN_SIZE**-0.5)
    mlp.down_proj.weight.data.normal_(std=SHARED_INTERMEDIATE_SIZE**-0.5)

    hidden_states = torch.randn(8, HIDDEN_SIZE, device=device, dtype=DTYPE)
    expected = mlp(hidden_states)
    gate_up, _ = mlp.gate_up_proj(hidden_states)

    assert mlp.use_external_gate_up()
    assert mlp.gate_up_proj is None
    assert mlp.expects_gate_up_input

    # Same weights and same kernels from the activation on, so this is exact.
    torch.testing.assert_close(mlp(gate_up), expected, atol=0, rtol=0)


_CHECKS = {
    "matches_separate": _check_matches_separate_projections,
    "shard_layout": _check_shard_layout,
    "external_gate_up": _check_shared_experts_consume_gate_up,
}


def _worker(local_rank: int, world_size: int, port: str, check: str) -> None:
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    torch.set_default_dtype(DTYPE)
    with ensure_current_vllm_config():
        init_test_distributed_environment(
            world_size, 1, local_rank, port, local_rank=local_rank
        )
        _CHECKS[check](device, world_size, local_rank)


def _run_ranks(check: str, tp_size: int) -> None:
    spawn(
        _worker,
        args=(tp_size, str(get_open_port()), check),
        nprocs=tp_size,
        join=True,
    )


@requires_cuda
def test_fused_projection_matches_separate_projections() -> None:
    _run_ranks("matches_separate", 1)


@requires_cuda
def test_shared_experts_consume_gate_up() -> None:
    _run_ranks("external_gate_up", 1)


def test_token_threshold_straddled() -> None:
    """Both project() branches must be under test."""
    assert min(TOKEN_COUNTS) <= FUSED_MOE_INPUT_PROJ_TOKEN_THRESHOLD
    assert max(TOKEN_COUNTS) > FUSED_MOE_INPUT_PROJ_TOKEN_THRESHOLD


@requires_cuda
@multi_gpu_test(num_gpus=2)
def test_shard_layout_tp2() -> None:
    _run_ranks("shard_layout", 2)


@requires_cuda
@multi_gpu_test(num_gpus=2)
def test_fused_projection_matches_separate_projections_tp2() -> None:
    _run_ranks("matches_separate", 2)


@requires_cuda
@multi_gpu_test(num_gpus=4)
def test_shard_layout_tp4() -> None:
    _run_ranks("shard_layout", 4)


def _resolve(
    name: str,
    stacked_params_mapping: list[tuple[str, str, int]],
    params_dict: set[str],
) -> tuple[str, int] | None:
    """The stacked-shard resolution rule from KimiLinearModel.load_weights.

    First entry whose shard name is a substring wins, and an entry naming a
    parameter the model does not have is skipped rather than claiming the weight.
    """
    for param_name, weight_name, shard_id in stacked_params_mapping:
        if weight_name not in name:
            continue
        mapped = name.replace(weight_name, param_name)
        if mapped not in params_dict:
            continue
        return mapped, shard_id
    return None


# The two generic entries the fused projection has to coexist with.
_GENERIC = [(".gate_up_proj", ".gate_proj", 0), (".gate_up_proj", ".up_proj", 1)]
_MOE = "model.layers.3.block_sparse_moe"
_DENSE = "model.layers.0.mlp"


def test_fused_projection_claims_its_shards() -> None:
    """Router, latent and both shared shards must reach the fused parameter.

    ``.gate`` is a prefix of ``.gate_proj`` and ``.gate_up_proj``, and the
    shared experts' shards are also matched by the generic ``.gate_proj`` entry,
    so this pins that neither collision misroutes a weight.
    """
    proj = ".fused_moe_input_proj"
    mapping = [
        (f"{proj}.weight", ".gate.weight", KimiK3FusedMoEInputProj.ROUTER_SHARD_ID),
        (proj, ".routed_expert_down_proj", KimiK3FusedMoEInputProj.LATENT_SHARD_ID),
        (
            proj,
            ".shared_experts.gate_proj",
            KimiK3FusedMoEInputProj.SHARED_GATE_SHARD_ID,
        ),
        (proj, ".shared_experts.up_proj", KimiK3FusedMoEInputProj.SHARED_UP_SHARD_ID),
        *_GENERIC,
    ]
    params = {f"{_MOE}{proj}.weight", f"{_DENSE}.gate_up_proj.weight"}

    assert _resolve(f"{_MOE}.gate.weight", mapping, params) == (
        f"{_MOE}{proj}.weight",
        KimiK3FusedMoEInputProj.ROUTER_SHARD_ID,
    )
    assert _resolve(f"{_MOE}.routed_expert_down_proj.weight", mapping, params) == (
        f"{_MOE}{proj}.weight",
        KimiK3FusedMoEInputProj.LATENT_SHARD_ID,
    )
    assert _resolve(f"{_MOE}.shared_experts.gate_proj.weight", mapping, params) == (
        f"{_MOE}{proj}.weight",
        KimiK3FusedMoEInputProj.SHARED_GATE_SHARD_ID,
    )
    assert _resolve(f"{_MOE}.shared_experts.up_proj.weight", mapping, params) == (
        f"{_MOE}{proj}.weight",
        KimiK3FusedMoEInputProj.SHARED_UP_SHARD_ID,
    )

    # Dense layers keep their own merged projection, and the router bias is not
    # a stacked shard at all.
    assert _resolve(f"{_DENSE}.gate_proj.weight", mapping, params) == (
        f"{_DENSE}.gate_up_proj.weight",
        0,
    )
    assert _resolve(f"{_MOE}.gate.e_score_correction_bias", mapping, params) is None


def test_generic_entries_alone_leave_fused_shards_unclaimed() -> None:
    """Without the fused entries the shared shards must not be misrouted.

    They resolve to a ``gate_up_proj`` the fused build does not have, which is
    what lets the same mapping serve both builds.
    """
    params = {f"{_MOE}.fused_moe_input_proj.weight"}
    assert _resolve(f"{_MOE}.shared_experts.gate_proj.weight", _GENERIC, params) is None
