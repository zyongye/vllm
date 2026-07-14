# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone numerical check for the MXFP4 path of prepare_megamoe_inputs.

Compares the Triton input-staging kernel against a PyTorch reference that
mirrors DeepGEMM's per_token_cast_to_fp4(use_ue8m0=True, gran_k=32,
use_packed_ue8m0=True). Run with:
    .venv/bin/python benchmarks/test_prepare_megamoe_mxfp4.py
"""

import torch

from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs

E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)


def quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(6.0)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=ax.dtype
    )
    idx = torch.bucketize(ax, boundaries)
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def ref_per_token_cast_to_fp4(x: torch.Tensor, gran_k: int = 32):
    m, n = x.shape
    x_view = x.float().view(m, -1, gran_k)
    x_amax = x_view.abs().amax(dim=2).clamp_min(1e-4)
    sf = ceil_to_ue8m0(x_amax / 6.0)
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = quantize_to_fp4_e2m1(x_scaled).view(m, n)
    codes2 = codes.view(m, n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed.to(torch.int8), pack_ue8m0_to_int(sf)


def dequant_packed_fp4(packed: torch.Tensor, sf_int32: torch.Tensor, hidden: int):
    lut = torch.cat([E2M1_VALUES, -E2M1_VALUES]).to(packed.device)
    p = packed.to(torch.int32) & 0xFF
    lo = lut[p & 0xF]
    hi = lut[(p >> 4) & 0xF]
    vals = torch.stack([lo, hi], dim=-1).view(packed.shape[0], hidden)
    # unpack ue8m0: int32 holds 4 group exponents (group=32 cols)
    exps = sf_int32.view(torch.uint8).to(torch.int32).view(packed.shape[0], -1)
    scales = (exps.float() - 127.0).exp2()
    return vals.view(packed.shape[0], -1, 32) * scales.unsqueeze(2)


def main():
    torch.manual_seed(0)
    device = "cuda"
    top_k = 8
    for hidden in (2048, 4096, 7168):
        num_tokens = 64
        hs = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
        topk_ids = torch.randint(0, 256, (num_tokens, top_k), device=device)
        topk_w = torch.rand(num_tokens, top_k, dtype=torch.float32, device=device)

        x_fp4 = torch.empty(num_tokens, hidden // 2, dtype=torch.int8, device=device)
        x_sf = torch.empty(num_tokens, hidden // 128, dtype=torch.int32, device=device)
        idx_out = torch.empty(num_tokens, top_k, dtype=torch.int64, device=device)
        w_out = torch.empty(num_tokens, top_k, dtype=torch.float32, device=device)

        # MXFP4 path is selected by the int8 output buffer dtype.
        prepare_megamoe_inputs(
            hs,
            topk_w,
            topk_ids,
            x_fp4,
            x_sf,
            idx_out,
            w_out,
        )

        ref_packed, ref_sf = ref_per_token_cast_to_fp4(hs, gran_k=32)

        sf_match = torch.equal(x_sf, ref_sf)

        deq_k = dequant_packed_fp4(x_fp4, x_sf, hidden)
        deq_r = dequant_packed_fp4(ref_packed, ref_sf, hidden)
        denom = deq_r.abs().mean().clamp_min(1e-6)
        relerr = (deq_k - deq_r).abs().mean() / denom
        nib_match = (x_fp4 == ref_packed).float().mean().item()

        print(
            f"H={hidden}: x_sf exact={sf_match} "
            f"nibble_match={nib_match:.4f} dequant_relerr={relerr.item():.3e}"
        )
        assert sf_match, f"x_sf mismatch at H={hidden}"
        assert relerr < 5e-3, f"dequant relerr too high at H={hidden}: {relerr}"
    print("PASS")


if __name__ == "__main__":
    main()
