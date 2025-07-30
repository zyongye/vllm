
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from flashinfer import (get_reorder_rows_for_gated_act_gemm_row_indices,
                        block_scale_interleave,
                        fused_moe_trtllmgen,
                        DtypeFusedMoeTrtllmGen,
                        fp4_quantize,
                        mxfp8_quantize,
                        reorder_rows_for_gated_act_gemm,
                        shuffle_matrix_a,
                        shuffle_matrix_sf_a)

from vllm.model_executor.layers.fused_moe.triton_kernels_moe import (
    triton_kernel_moe_forward)

import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig
from triton_kernels.numerics import InFlexData
from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details.layout import (BlackwellMXScaleLayout,
                                                  HopperMXScaleLayout,
                                                  HopperMXValueLayout,
                                                  StridedLayout)
def rmse(y_pred, y_true):
    mse = F.mse_loss(y_pred, y_true)
    rmse = torch.sqrt(mse)
    return rmse

def max_error(y_pred, y_true):
    diff = torch.abs(y_pred - y_true)
    max_error = torch.max(diff)
    return max_error

# w13_bias: (torch.Size([32, 5888]), torch.float32), w2_bias: (torch.Size([32, 3072]), torch.float32)
# w13_weight: (torch.Size([32, 5888, 1536]), torch.uint8), w2_weight: (torch.Size([32, 3072, 1472]), torch.uint8)
# w13_weight_scale: (torch.Size([32, 5888, 96]), torch.uint8), w2_weight_scale: (torch.Size([32, 3072, 92]), torch.uint8)
# self.num_experts: 32, self.intermediate_size: 2944, self.hidden_size: 3072
# Model loading took 13.7153 GiB and 32.206199 seconds
# top_k: 4
# x_quant: (torch.Size([16384, 3072]), torch.float8_e4m3fn), x_scale: (torch.Size([1572864]), torch.float8_e4m3fn)
def compute_routing_renormalize(
    router_logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute routing weights and selected experts from router logits.

    Args:
        router_logits (torch.Tensor): Router logits of shape [batch_size, num_experts]
        top_k (int): Number of experts to route to per token

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - routing_weights: Expert weights of shape [batch_size, top_k]
            - selected_experts: Expert indices of shape [batch_size, top_k]
    """
    # routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(router_logits, top_k, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float)
    return routing_weights, selected_experts
def next_positive_power_of_2(x: int) -> int:
    if x < 1:
        return 1
    return 1 << (x - 1).bit_length()
def _get_tile_tokens_dim(x: torch.Tensor, top_k: int, num_experts: int):
    # Number of tokens in the input tensor.
    num_tokens = x.shape[0]
    # Factor to account for the imbalance of the experts.
    # factor equals to the max_real_num_tokens_per_expert / perfect_num_tokens_per_expert
    # 1.0 means perfect expert distribution.
    # > 1.0 means some experts have more tokens than the perfect distribution.
    # < 1.0 does not make sense.
    imbalance_factor = 1.3
    # Calculate the number of tokens per expert assuming perfect distribution.
    num_tokens_per_expert = (num_tokens * top_k) // num_experts
    # Apply the imbalance factor.
    num_tokens_per_expert = int(num_tokens_per_expert * imbalance_factor)
    # And pad the number to the next power of 2.
    tile_tokens_dim = next_positive_power_of_2(num_tokens_per_expert)
    # Cap to 8-64 tokens per CTA tile as it's the range supported by the kernel.
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)

    return tile_tokens_dim
def tg_moe(
        router_logits:torch.Tensor,
        x:torch.Tensor,
        w13:torch.Tensor,
        w2:torch.Tensor,
        num_experts:int,
        intermediate_size:int,
        hidden_size:int,
        top_k:int,
        bias13:torch.Tensor,
        bias2:torch.Tensor,
        gemm1_alpha:torch.Tensor,
        gemm1_beta:torch.Tensor,
    ):
    x_quant, x_scale = mxfp8_quantize(x, False)
    x_scale = x_scale.view(torch.float8_e4m3fn).view(-1)
    print(f'w13: {w13.shape, w13.dtype}, w2: {w2.shape, w2.dtype}')
    w13_quant, w13_scale = fp4_quantize(w13, global_scale=torch.ones(1, device=x.device), sf_vec_size=32, sf_use_ue8m0=True, is_sf_swizzled_layout=False)
    w2_quant, w2_scale = fp4_quantize(w2, global_scale=torch.ones(1, device=x.device), sf_vec_size=32, sf_use_ue8m0=True, is_sf_swizzled_layout=False)

    w13_scale = w13_scale.reshape(num_experts, w13.shape[1], w13.shape[2] // 32)
    w2_scale = w2_scale.reshape(num_experts, w2.shape[1], w2.shape[2] // 32)

    # Reorder rows of W1 and scales for fused gated activation
    gemm1_weights_mxfp4_interleaved = []
    gemm1_scales_mxfp4_interleaved = []
    bias13_interleaved = []
    for i in range(num_experts):
        gemm1_weights_mxfp4_interleaved.append(
            reorder_rows_for_gated_act_gemm(w13_quant[i].clone()))
        gemm1_scales_mxfp4_interleaved.append(
            reorder_rows_for_gated_act_gemm(w13_scale[i].clone()))
        bias13_interleaved.append(reorder_rows_for_gated_act_gemm(bias13[i].clone().reshape(-1, 1)))

    # Stack weights and scales for all experts
    gemm1_weights_mxfp4_interleaved = torch.stack(
        gemm1_weights_mxfp4_interleaved).reshape(num_experts,
                                            2 * intermediate_size,
                                            hidden_size // 2)
    gemm1_scales_mxfp4_interleaved = torch.stack(
        gemm1_scales_mxfp4_interleaved).reshape(num_experts,
                                            2 * intermediate_size,
                                            hidden_size // 32)
    bias13 = torch.stack(bias13_interleaved).reshape(num_experts, -1)

    # swap w1 and w3
    def swap_every_two_rows(x, axis=-1):
        shape = x.shape
        if axis < 0:
            axis = len(shape) + axis
        
        # Create a new shape with pairs swapped along specified axis
        new_shape = list(shape)
        new_shape[axis] = shape[axis] // 2
        new_shape.insert(axis + 1, 2)
        
        # Reshape to expose pairs, swap them, and reshape back
        x = x.reshape(*new_shape)
        x = x.flip(axis + 1)
        new_shape = list(shape)
        return x.reshape(*new_shape)
        
    gemm1_weights_mxfp4_interleaved = swap_every_two_rows(gemm1_weights_mxfp4_interleaved, -2)
    gemm1_scales_mxfp4_interleaved = swap_every_two_rows(gemm1_scales_mxfp4_interleaved, -2)
    bias13 = swap_every_two_rows(bias13, -1)
    # Shuffle weights and scaling factors for transposed mma output
    gemm1_weights_mxfp4_shuffled = []
    gemm1_scales_mxfp4_shuffled = []
    bias13_shuffled = []
    gemm2_weights_mxfp4_shuffled = []
    gemm2_scales_mxfp4_shuffled = []
    bias2_shuffled = []
    epilogue_tile_m = 128 # FIXME: this depends on the kernel internals
    for i in range(num_experts):
        gemm1_weights_mxfp4_shuffled.append(
            shuffle_matrix_a(
                gemm1_weights_mxfp4_interleaved[i].view(torch.uint8),
                epilogue_tile_m))
        gemm1_scales_mxfp4_shuffled.append(
            shuffle_matrix_sf_a(
                gemm1_scales_mxfp4_interleaved[i].view(torch.uint8),
                epilogue_tile_m))
        bias13_shuffled.append(
            shuffle_matrix_a(
                bias13[i].clone().reshape(-1, 1),
                epilogue_tile_m))
        gemm2_weights_mxfp4_shuffled.append(
            shuffle_matrix_a(w2_quant[i].view(torch.uint8).clone(),
                            epilogue_tile_m))
        gemm2_scales_mxfp4_shuffled.append(
            shuffle_matrix_sf_a(
                w2_scale[i].view(torch.uint8),
                epilogue_tile_m))
        bias2_shuffled.append(
            shuffle_matrix_a(
                bias2[i].clone().reshape(-1, 1),
                epilogue_tile_m))
    w13_weight = torch.stack(gemm1_weights_mxfp4_shuffled)
    w13_weight_scale = torch.stack(gemm1_scales_mxfp4_shuffled).view(
        torch.float8_e4m3fn).reshape(num_experts, 2 * intermediate_size,
                                    hidden_size // 32)
    
    bias13 = torch.stack(bias13_shuffled).reshape(
        num_experts, -1)
    bias2 = torch.stack(bias2_shuffled).reshape(
        num_experts, -1)

    w2_weight = torch.stack(gemm2_weights_mxfp4_shuffled)
    w2_weight_scale = torch.stack(gemm2_scales_mxfp4_shuffled).view(
        torch.float8_e4m3fn).reshape(num_experts, hidden_size,
                                    intermediate_size // 32)
    print(f'w13_quant: {w13_quant.shape, w13_quant.dtype}, w2_quant: {w2_quant.shape, w2_quant.dtype}, has nan {torch.isnan(w13_quant).any()}, {torch.isnan(w2_quant).any()}')
    print(f'w13_scale: {w13_scale.shape, w13_scale.dtype}, w2_scale: {w2_scale.shape, w2_scale.dtype}, has nan {torch.isnan(w13_scale).any()}, {torch.isnan(w2_scale).any()}')
    print(f'x_quant: {x_quant.shape, x_quant.dtype}, x_scale: {x_scale.shape, x_scale.dtype}, has nan {torch.isnan(x_quant).any()}, {torch.isnan(x_scale).any()}')

    trtllm_gen_output = fused_moe_trtllmgen(
        routing_logits=router_logits, # bfloat16
        routing_bias=None,
        hidden_states=x_quant,        # e4m3
        hidden_states_scale=x_scale,  # ue8m0
        gemm1_weights=w13_weight,     # uint8 (e2m1)
        gemm1_weights_scale=w13_weight_scale, # ue8m0
        gemm1_bias=bias13,              # fp32 per expert per channel
        gemm1_alpha=gemm1_alpha,             # fp32 per expert
        gemm1_beta=gemm1_beta,              # fp32 per expert
        gemm2_weights=w2_weight,      # uint8 (e2m1)
        gemm2_weights_scale=w2_weight_scale, # ue8m0
        gemm2_bias=bias2,              # fp32 per expert per channel
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size, # padded to 3072 (multiple of 256)
        hidden_size_output=hidden_size,      # padded to 3072 (multiple of 256)
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        tile_tokens_dim=_get_tile_tokens_dim(x, top_k, num_experts),
        routing_method_type=1,
        do_finalize=True,
        dtype_act=DtypeFusedMoeTrtllmGen.MxE4m3,
        dtype_weights=DtypeFusedMoeTrtllmGen.MxE2m1,
    )[0]
    torch.cuda.synchronize() # check if error
    return trtllm_gen_output


def swizzle_mxfp4(quant_tensor, scale):
    value_layout = StridedLayout
    scale_layout = StridedLayout
    scale_layout = BlackwellMXScaleLayout
    constraints = {
        "is_persistent": True,
        "epilogue_subtile": 1,
    }
    opt_flags.update_opt_flags_constraints(constraints)
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(wrap_torch_tensor(quant_tensor, dtype=FP4),
                                  value_layout)
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout)
    return quant_tensor, InFlexData(), scale


from flashinfer.test_trtllm_trtllm_gen_fused_moe import compute_moe_reference_with_routing
from flashinfer.fused_moe import RoutingMethodType

def test_tg_moe():
    torch.set_printoptions(threshold=10000000000, sci_mode=False, precision=3)
    num_experts = 32
    num_tokens = 32
    hidden_size = 3072
    intermediate_size = 3072
    topk = 4
    hidden_states = torch.randn(num_tokens, hidden_size, device="cuda:0", dtype=torch.bfloat16) * 0.1
    w13 = (torch.randn(num_experts, intermediate_size * 2, hidden_size, device="cuda:0", dtype=torch.bfloat16)) * 0.2
    w2 = (torch.randn(num_experts, hidden_size, intermediate_size, device="cuda:0", dtype=torch.bfloat16))
    bias13 = (torch.randn(num_experts, 2 * intermediate_size, device="cuda:0", dtype=torch.float32))
    bias2 = (torch.randn(num_experts, hidden_size, device="cuda:0", dtype=torch.float32))
    router_logits = torch.rand(num_tokens, num_experts, dtype=torch.float32).cuda()
    routing_weights, selected_experts = compute_routing_renormalize(router_logits, topk)


    w13_triton = w13.clone()
    w2_triton = w2.clone()
    
    gemm1_alpha = torch.tensor(
        [1.702] * num_experts,
        dtype=torch.float32).cuda()
    gemm1_beta = torch.tensor(
        [1.0] * num_experts,
        dtype=torch.float32).cuda()

    tg_result = tg_moe(
        router_logits.to(torch.bfloat16),
        hidden_states,
        w13,
        w2,
        num_experts,
        intermediate_size,
        hidden_size,
        topk,
        bias13,
        bias2,
        gemm1_alpha,
        gemm1_beta,
    )

    torch.cuda.synchronize()

    w13_ref = torch.empty_like(w13)
    w13_ref[:, :intermediate_size, :] = w13[:, intermediate_size:, :]
    w13_ref[:, intermediate_size:, :] = w13[:, :intermediate_size, :]
    bias13_ref = torch.empty_like(bias13)
    bias13_ref[:, :intermediate_size] = bias13[:, intermediate_size:]
    bias13_ref[:, intermediate_size:] = bias13[:, :intermediate_size]
    ref_result = compute_moe_reference_with_routing(
        num_tokens, # num_tokens,
        hidden_size, # hidden_size,
        intermediate_size, # intermediate_size,
        num_experts, # num_experts,
        router_logits.to(torch.bfloat16), # expert_logits,
        None, # routing_bias,
        hidden_states, # hidden_states,
        w13_ref, # gemm1_weights,
        w2, # gemm2_weights,
        bias13_ref, # gemm1_bias,
        gemm1_alpha, # gemm1_swiglu_alpha,
        gemm1_beta, # gemm1_swiglu_beta,
        bias2, # gemm2_bias,
        topk, # top_k,
        8, # padding,
        None, # n_groups,
        None, # top_k_groups,
        None, # routed_scaling,
        RoutingMethodType.Renormalize, # routing_method_type,
        DtypeFusedMoeTrtllmGen.MxE4m3, # dtype_act,
        DtypeFusedMoeTrtllmGen.MxE2m1, # dtype_weights,
    )[0].to(torch.bfloat16)

    ref_result_bf16 = compute_moe_reference_with_routing(
        num_tokens, # num_tokens,
        hidden_size, # hidden_size,
        intermediate_size, # intermediate_size,
        num_experts, # num_experts,
        router_logits.to(torch.bfloat16), # expert_logits,
        None, # routing_bias,
        hidden_states, # hidden_states,
        w13_ref, # gemm1_weights,
        w2, # gemm2_weights,
        bias13_ref, # gemm1_bias,
        gemm1_alpha, # gemm1_swiglu_alpha,
        gemm1_beta, # gemm1_swiglu_beta,
        bias2, # gemm2_bias,
        topk, # top_k,
        8, # padding,
        None, # n_groups,
        None, # top_k_groups,
        None, # routed_scaling,
        RoutingMethodType.Renormalize, # routing_method_type,
        DtypeFusedMoeTrtllmGen.Bfloat16, # dtype_act,
        DtypeFusedMoeTrtllmGen.MxE2m1, # dtype_weights,
    )[0].to(torch.bfloat16)

    w13_quant, w13_scale = fp4_quantize(w13_triton, global_scale=torch.ones(1, device=hidden_states.device), sf_vec_size=32, sf_use_ue8m0=True, is_sf_swizzled_layout=False)
    w2_quant, w2_scale = fp4_quantize(w2_triton, global_scale=torch.ones(1, device=hidden_states.device), sf_vec_size=32, sf_use_ue8m0=True, is_sf_swizzled_layout=False)
    w13_scale = w13_scale.reshape(num_experts, w13.shape[1], w13.shape[2] // 32)
    w2_scale = w2_scale.reshape(num_experts, w2.shape[1], w2.shape[2] // 32)

    w13_triton_weights_mxfp4_interleaved = []
    w13_triton_scales_mxfp4_interleaved = []
    bias13_triton_interleaved = []
    
    for i in range(num_experts):
        w13_triton_weights_mxfp4_interleaved.append(
            reorder_rows_for_gated_act_gemm(w13_quant[i].clone()))
        w13_triton_scales_mxfp4_interleaved.append(
            reorder_rows_for_gated_act_gemm(w13_scale[i].clone()))
        bias13_triton_interleaved.append(reorder_rows_for_gated_act_gemm(bias13[i].clone().reshape(-1, 1)))

    # Stack weights and scales for all experts
    w13_quant = torch.stack(
        w13_triton_weights_mxfp4_interleaved).reshape(num_experts,
                                            2 * intermediate_size,
                                            hidden_size // 2)
    w13_scale = torch.stack(
        w13_triton_scales_mxfp4_interleaved).reshape(num_experts,
                                            2 * intermediate_size,
                                            hidden_size // 32)
    bias13 = torch.stack(bias13_triton_interleaved).reshape(num_experts, -1)
    # w13_scale = torch.ones_like(w13_scale, dtype=torch.uint8) * 125

    w13_weight_triton_tensor, w13_flex, w13_scale = swizzle_mxfp4(
        w13_quant, w13_scale)
    w2_weight_triton_tensor, w2_flex, w2_scale = swizzle_mxfp4(w2_quant,
                                                    w2_scale)

    w13_precision_config = PrecisionConfig(
        weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex))
    w2_precision_config = PrecisionConfig(
        weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex))
    # print(f'w13_precision_config: {w13_precision_config}')
    # print(f'w2_precision_config: {w2_precision_config}')
    tirton_result = triton_kernel_moe_forward(
        hidden_states=hidden_states,
        w1=w13_weight_triton_tensor,
        w2=w2_weight_triton_tensor,
        gating_output=router_logits,
        topk=topk,
        renormalize=True,
        global_num_experts=num_experts,
        expert_map=None,
        w1_bias=bias13,
        w2_bias=bias2,
        w1_precision=w13_precision_config,
        w2_precision=w2_precision_config,
        apply_router_weight_on_input=False,
    )
    torch.cuda.synchronize()
    print('tg_result has nan ', torch.isnan(tg_result).any().item())
    print('tg_result', tg_result[:10, :10])
    print('tirton_result', tirton_result[:10, :10])
    print('ref_result', ref_result[:10, :10])
    print('ref_result_bf16', ref_result_bf16[:10, :10])
    for i in range(num_tokens):
        print(i, 'triton result RMSE:', rmse(tg_result[i], tirton_result[i]).item(), torch.allclose(tg_result[i], tirton_result[i], atol=1e-2),
              'ref result RMSE:', rmse(tg_result[i], ref_result[i]).item(), torch.allclose(tg_result[i], ref_result[i], atol=1e-2),
              'bf16 ref to triton RMSE:', rmse(ref_result_bf16[i], tirton_result[i]).item(), torch.allclose(ref_result_bf16[i], tirton_result[i], atol=1e-2))

if __name__ == "__main__":
    torch.manual_seed(0xDEADBEEF)
    # test_cuda_routing_data()
    test_tg_moe()
