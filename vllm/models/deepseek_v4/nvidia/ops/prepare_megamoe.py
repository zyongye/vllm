# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton input-staging kernel for DeepSeek V4 MegaMoE.

Quantizes hidden states with UE8M0 block-32 group scales and repacks the
routing top-k tensors into the int64/float32 layout that the DeepGEMM
MegaMoE kernels consume.

The activation format is inferred from the output ``x`` buffer dtype, matching
the symmetric-buffer contract in DeepGEMM's ``docs/mega_moe_mxfp4.md``:

- float8 buffer (``[num_tokens, hidden]``): per-group fp8 e4m3 (scale / 448).
- int8 buffer (``[num_tokens, hidden / 2]``): MXFP4, two E2M1 values per byte
  (low nibble = even col, high nibble = odd col, scale / 6).

The ``x_sf`` scale tensor is identical for both: int32, ``[num_tokens,
hidden / 128]``, four UE8M0 block-32 exponents packed per int32 (K-major).
"""

import torch

from vllm.models.deepseek_v4.common.ops.fused_indexer_q import _fp32x2_to_fp4x2
from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_megamoe_inputs_kernel(
    hidden_states,
    x_q,
    x_sf,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    USE_FP4: tl.constexpr,
) -> None:
    token_id = tl.program_id(0)
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < hidden_size
    hidden = tl.load(
        hidden_states + token_id * hidden_stride_m + k_offsets * hidden_stride_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(tl.abs(hidden), [num_groups, GROUP_K])
    amax = tl.max(hidden_groups, axis=1)
    amax = tl.maximum(amax, 1.0e-4)

    # MXFP4 E2M1 saturates at 6.0; fp8 e4m3 at 448.0.
    SCALE_DIVISOR: tl.constexpr = 6.0 if USE_FP4 else 448.0
    scale = amax / SCALE_DIVISOR
    scale_bits = scale.to(tl.uint32, bitcast=True)
    scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
        tl.uint32
    )
    scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
    rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)

    hidden_groups = tl.reshape(hidden, [num_groups, GROUP_K])
    scaled = hidden_groups * (1.0 / rounded_scale)[:, None]
    scaled = tl.reshape(scaled, [BLOCK_K])
    if USE_FP4:
        # Pack two E2M1 per byte: low nibble = even col, high nibble = odd col.
        x_lo, x_hi = tl.split(tl.reshape(scaled, [BLOCK_K // 2, 2]))
        packed = _fp32x2_to_fp4x2(x_lo, x_hi)
        half_offsets = k_block_id * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
        tl.store(
            x_q + token_id * x_stride_m + half_offsets * x_stride_k,
            packed.to(tl.int8, bitcast=True),
        )
    else:
        fp8 = scaled.to(tl.float8e4nv)
        tl.store(
            x_q + token_id * x_stride_m + k_offsets * x_stride_k,
            fp8,
            mask=k_mask,
        )

    scale_offsets = tl.arange(0, num_groups)
    packed_scale = tl.sum(scale_exp << (scale_offsets * 8), axis=0).to(tl.int32)
    tl.store(
        x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
        packed_scale,
    )

    if k_block_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = topk_offsets < top_k
        token_is_padding = False
        if is_padding is not None:
            token_is_padding = tl.load(is_padding + token_id * is_padding_stride_m)

        ids = tl.load(
            topk_ids + token_id * topk_ids_stride_m + topk_offsets * topk_ids_stride_k,
            mask=topk_mask,
            other=0,
        ).to(tl.int64)
        ids = tl.where(token_is_padding, -1, ids)
        tl.store(
            topk_idx_out
            + token_id * topk_idx_stride_m
            + topk_offsets * topk_idx_stride_k,
            ids,
            mask=topk_mask,
        )

        weights = tl.load(
            topk_weights
            + token_id * topk_weights_stride_m
            + topk_offsets * topk_weights_stride_k,
            mask=topk_mask,
            other=0.0,
        )
        weights = tl.where(token_is_padding, 0.0, weights)
        tl.store(
            topk_weights_out
            + token_id * topk_weights_out_stride_m
            + topk_offsets * topk_weights_out_stride_k,
            weights,
            mask=topk_mask,
        )


def prepare_megamoe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_q: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> None:
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % 128 != 0:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires hidden_size to be "
            "a multiple of 128."
        )
    top_k = topk_ids.shape[1]
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )

    # The activation format is inferred from the output buffer dtype: an int8
    # buffer carries packed MXFP4 (hidden / 2 wide), anything else is fp8.
    use_fp4 = x_q.dtype == torch.int8
    expected_width = hidden_size // 2 if use_fp4 else hidden_size
    if x_q.shape != (num_tokens, expected_width):
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging expected x buffer of shape "
            f"{(num_tokens, expected_width)} for "
            f"{'mxfp4' if use_fp4 else 'fp8'} activations, got "
            f"{tuple(x_q.shape)}."
        )

    block_k = 128
    grid = (num_tokens, triton.cdiv(hidden_size, block_k))
    block_topk = triton.next_power_of_2(top_k)
    padding_stride_m = is_padding.stride(0) if is_padding is not None else 0
    _prepare_megamoe_inputs_kernel[grid](
        hidden_states,
        x_q,
        x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        hidden_states.stride(0),
        hidden_states.stride(1),
        x_q.stride(0),
        x_q.stride(1),
        x_sf.stride(0),
        x_sf.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        padding_stride_m,
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        hidden_size,
        top_k,
        BLOCK_K=block_k,
        GROUP_K=32,
        BLOCK_TOPK=block_topk,
        USE_FP4=use_fp4,
        num_warps=4,
    )
