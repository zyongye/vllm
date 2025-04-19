# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, fields
from typing import Callable, Optional

import pytest
import torch

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.exp_fused_moe.routing import routing
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts, fused_experts_triton_exp)


def assert_close(ref,
                 tri,
                 maxtol=None,
                 rmstol=None,
                 description="--",
                 verbose=True):
    if tri.dtype.itemsize == 1:
        ref_as_type = ref.to(tri.dtype)
        if ref.dtype == tri.dtype:
            assert torch.all(ref_as_type == tri)
            return
        ref = ref_as_type

    if maxtol is None:
        maxtol = 2e-2
    if rmstol is None:
        rmstol = 4e-3
    """
    Compare reference values against obtained values.
    """

    # cast to float32:
    ref = ref.to(torch.float32).detach()
    tri = tri.to(torch.float32).detach()
    assert ref.shape == tri.shape, f"Tensors must have same size {ref.shape=} {tri.shape=}"

    # deal with infinite elements:
    inf_mask_ref = torch.isinf(ref)
    inf_mask_tri = torch.isinf(tri)
    assert torch.equal(inf_mask_ref,
                       inf_mask_tri), "Tensor must have same infinite elements"
    refn = torch.where(inf_mask_ref, 0, ref)
    trin = torch.where(inf_mask_tri, 0, tri)

    # normalise so that RMS calculation doesn't overflow:
    eps = 1.0e-30
    multiplier = 1.0 / (torch.max(torch.abs(refn)) + eps)
    refn *= multiplier
    trin *= multiplier

    ref_rms = torch.sqrt(torch.square(refn).mean()) + eps

    rel_err = torch.abs(refn - trin) / torch.maximum(ref_rms, torch.abs(refn))
    max_err = torch.max(rel_err).item()
    rms_err = torch.sqrt(torch.square(rel_err).mean()).item()

    if verbose:
        print("%s maximum relative error = %s (threshold = %s)" %
              (description, max_err, maxtol))
        print("%s RMS relative error = %s (threshold = %s)" %
              (description, rms_err, rmstol))

    if max_err > maxtol:
        bad_idxs = torch.nonzero(rel_err > maxtol)
        num_nonzero = bad_idxs.size(0)
        bad_idxs = bad_idxs[:1000]
        print("%d / %d mismatched elements (shape = %s) at coords %s" %
              (num_nonzero, rel_err.numel(), tuple(
                  rel_err.shape), bad_idxs.tolist()))

        bad_idxs = bad_idxs.unbind(-1)
        print("ref values: ", ref[*bad_idxs].cpu())
        print("tri values: ", tri[*bad_idxs].cpu())

    assert max_err <= maxtol
    assert rms_err <= rmstol


def forward_cuda_ref(x,
                     w1,
                     w2,
                     use_grouped_topk: bool,
                     top_k: int,
                     router_logits: torch.Tensor,
                     renormalize: bool,
                     topk_group: Optional[int] = None,
                     num_expert_group: Optional[int] = None,
                     global_num_experts: int = -1,
                     expert_map: Optional[torch.Tensor] = None,
                     custom_routing_function: Optional[Callable] = None,
                     scoring_func: str = "softmax",
                     e_score_correction_bias: Optional[torch.Tensor] = None,
                     apply_router_weight_on_input: bool = False,
                     activation: str = "silu") -> torch.Tensor:

    topk_weights, topk_ids = FusedMoE.select_experts(
        hidden_states=x,
        router_logits=router_logits,
        use_grouped_topk=use_grouped_topk,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias)

    return fused_experts(
        hidden_states=x,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=True,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map)


# dropin replacement for the forward_cuda
def forward_cuda_exp(x,
                     w1,
                     w2,
                     use_grouped_topk: bool,
                     top_k: int,
                     router_logits: torch.Tensor,
                     renormalize: bool,
                     topk_group: Optional[int] = None,
                     num_expert_group: Optional[int] = None,
                     global_num_experts: int = -1,
                     expert_map: Optional[torch.Tensor] = None,
                     custom_routing_function: Optional[Callable] = None,
                     scoring_func: str = "softmax",
                     e_score_correction_bias: Optional[torch.Tensor] = None,
                     apply_router_weight_on_input: bool = False,
                     activation: str = "silu"):
    # feature check
    assert renormalize == True, "renormalize can only be True in new triton MoE kernel, false not supported"
    assert use_grouped_topk == False, "use_grouped_topk is not supported in new triton MoE kernel"
    assert topk_group is None, "topk_group is not supported in new triton MoE kernel"
    assert num_expert_group is None, "num_expert_group is not supported in new triton MoE kernel"
    assert custom_routing_function is None, "custom_routing_function is not supported in new triton MoE kernel"
    assert scoring_func == "softmax", "scoring_func is not supported in new triton MoE kernel"
    assert e_score_correction_bias is None, "e_score_correction_bias is not supported in new triton MoE kernel"

    routing_data, gather_idx, scatter_idx = routing(router_logits, top_k)

    return fused_experts_triton_exp(
        hidden_states=x,
        w1=w1,
        w2=w2,
        routing_data=routing_data,
        gather_indx=gather_idx,
        scatter_indx=scatter_idx,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map)


@dataclass
class Case:
    num_token: int
    inter_size: int
    K: int
    num_expts_tot: int
    num_expts_act: int


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case)) for case in [
            Case(num_token=32,
                 inter_size=512,
                 K=32,
                 num_expts_tot=128,
                 num_expts_act=4),
            Case(num_token=16,
                 inter_size=512,
                 K=32,
                 num_expts_tot=128,
                 num_expts_act=4),
            Case(num_token=1024,
                 inter_size=2048,
                 K=32,
                 num_expts_tot=128,
                 num_expts_act=4),
        ]
    ],
)
def test_equiv(num_token, inter_size, K, num_expts_tot, num_expts_act):

    randbits = [torch.randperm(num_expts_tot) for _ in range(num_token)]
    x = [(-1)**i *
         ((16384 +
           ((i * 512) % 4096) + bits).to(torch.int16).view(torch.bfloat16))
         for i, bits in enumerate(randbits)]
    exp_data = torch.stack(x).to(device="cuda")
    # exp_data = torch.randn((num_token, num_expts_tot), dtype=torch.bfloat16, device="cuda")

    # create input tensor
    x = torch.randn((num_token, K), dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn((num_expts_tot, inter_size, K),
                     dtype=torch.bfloat16,
                     device="cuda")
    w2 = torch.randn((num_expts_tot, K, inter_size // 2),
                     dtype=torch.bfloat16,
                     device="cuda")

    exp_data_tri = exp_data.clone()
    x_tri = x.clone()
    w1_tri = w1.clone()
    w2_tri = w2.clone()

    out_triton = forward_cuda_exp(x_tri, w1_tri, w2_tri, False, num_expts_act,
                                  exp_data_tri, True)

    out_ref = forward_cuda_ref(x, w1, w2, False, num_expts_act, exp_data, True)
    assert_close(ref=out_ref, tri=out_triton)
