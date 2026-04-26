// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Shared types/utilities for the three DeepSeek V4 top-k strategies
// (Register / Streaming / Cluster). Ported from sglang's
// jit_kernel/include/sgl_kernel/deepseek_v4/topk/common.cuh.

#pragma once

#include "utils.cuh"

#include <cuda_fp16.h>
#include <cstdint>

namespace vllm::dsv4_topk {

inline constexpr uint32_t kMaxTopK = 1024;
inline constexpr uint32_t kBlockSize = 1024;
inline constexpr uint32_t kNumWarps = kBlockSize / kWarpThreads;
// 1 element per thread in the tie-breaking pass.
inline constexpr uint32_t kMaxTies = 1024;
inline constexpr uint32_t kRadixBins = 256;
static_assert(kMaxTopK <= kBlockSize && kMaxTies <= kBlockSize);

// Always vectorize global loads as float4.
using Vec4 = AlignedVector<float, 4>;

// page_to_indices: convert a flat compressed-token index into a (block * page_size + offset)
// page-table-resolved index. page_size must be a power of 2; page_bits = log2(page_size).
VLLM_DSV4_DEVICE int32_t page_to_indices(const int32_t* __restrict__ page_table,
                                         uint32_t i, uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

// Output-side description of the page-table fold-in: each strategy writes
// either (a) `transform(idx)` for entries already known to be in the top-k,
// or (b) `write(dst, src)` for entries whose final rank is determined later.
struct TransformParams {
  const int32_t* __restrict__ page_table;
  const int32_t* __restrict__ indices_in;
  int32_t* __restrict__ indices_out;
  uint32_t page_bits;

  VLLM_DSV4_DEVICE void transform(uint32_t idx) const {
    indices_out[idx] = page_to_indices(page_table, indices_in[idx], page_bits);
  }
  VLLM_DSV4_DEVICE void write(uint32_t dst, uint32_t src) const {
    indices_out[dst] = page_to_indices(page_table, src, page_bits);
  }
};

struct alignas(16) MatchBin {
  uint32_t bin;
  uint32_t above_count;
  uint32_t equal_count;
};

struct alignas(8) Tie {
  uint32_t idx;
  float score;
};

// Shared-memory layout for the final tie-breaking radix pass. Reused by both
// the streaming kernel (overlapping `score_buffer`) and the cluster kernel.
struct TieHandleSmem {
  alignas(128) uint32_t counter;
  alignas(128) MatchBin match;
  uint32_t histogram[kRadixBins];
  uint32_t warp_sum[kNumWarps];
};

// Order-preserving fp32 -> uint key, truncated to the top kBits. Used for the
// coarse histogram pass.
template <uint32_t kBits>
VLLM_DSV4_DEVICE uint32_t extract_coarse_bin(float x) {
  static_assert(0 < kBits && kBits < 15);
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return key >> (16 - kBits);
}

// Full 32-bit order-preserving key, used in tie-breaking.
VLLM_DSV4_DEVICE uint32_t extract_exact_bin(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

VLLM_DSV4_DEVICE uint32_t warp_inclusive_sum(uint32_t lane_id, uint32_t val) {
  static_assert(kWarpThreads == 32);
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane_id >= offset) val += n;
  }
  return val;
}

// Fast path when seq_len <= K: identity mapping, padded to K with -1.
VLLM_DSV4_DEVICE void trivial_transform(const TransformParams& params,
                                        uint32_t length, uint32_t K) {
  const auto tx = threadIdx.x;
  if (tx < length) {
    params.write(tx, tx);
  } else if (tx < K) {
    params.indices_out[tx] = -1;
  }
}

// Tie-break the threshold-bin candidates that didn't fit in the strict-above
// region. One block-wide radix pass over the full 32-bit key (fp32 bit
// pattern, with idx as a secondary key). Writes at most `K - num_above`
// entries via params.write(...).
VLLM_DSV4_DEVICE void tie_handle_transform(const Tie* __restrict__ ties,
                                           uint32_t num_ties, uint32_t num_above,
                                           uint32_t K, TransformParams params,
                                           void* _smem) {
  auto* smem = static_cast<TieHandleSmem*>(_smem);
  const auto tx = threadIdx.x;
  const auto lane_id = tx % kWarpThreads;
  const auto warp_id = tx / kWarpThreads;

  const bool has_elem = tx < num_ties;
  const auto tie = has_elem ? ties[tx] : Tie{0, 0.0f};
  const uint32_t key = extract_exact_bin(tie.score);
  const uint32_t idx = tie.idx;
  bool active = has_elem;
  uint32_t topk_remain = K - num_above;
  uint32_t write_pos = K;

  smem->counter = 0;
  __syncthreads();

  // 256 bins / 32 lanes = 8 warps span the histogram inter-warp prefix.
  constexpr uint32_t kRadixWarps = kRadixBins / kWarpThreads;

#pragma unroll
  for (int round = 0; round < 4; round++) {
    const uint32_t shift = 24 - round * 8;
    const uint32_t bin = (key >> shift) & 0xFFu;

    // 1. Histogram.
    if (tx < kRadixBins) smem->histogram[tx] = 0;
    __syncthreads();
    if (active) atomicAdd(&smem->histogram[bin], 1);
    __syncthreads();

    // 2. Two-pass prefix sum across the 256 bins.
    uint32_t hist_val = 0;
    uint32_t warp_inc = 0;
    if (tx < kRadixBins) {
      hist_val = smem->histogram[tx];
      warp_inc = warp_inclusive_sum(lane_id, hist_val);
      if (lane_id == kWarpThreads - 1) smem->warp_sum[warp_id] = warp_inc;
    }
    __syncthreads();
    if (tx < kRadixBins) {
      const auto tmp = (lane_id < kRadixWarps) ? smem->warp_sum[lane_id] : 0;
      const auto total = warp_reduce_sum(tmp);
      const auto inter = warp_reduce_sum(lane_id < warp_id ? tmp : 0);
      const auto prefix = inter + warp_inc;
      const auto above = total - prefix;
      // 3. Find threshold bin.
      if (above < topk_remain && above + hist_val >= topk_remain) {
        smem->match = {tx, above, topk_remain - above};
      }
    }
    __syncthreads();

    const auto thr = smem->match.bin;
    const auto n_above = smem->match.above_count;

    // 4. Scatter.
    if (active) {
      if (bin > thr) {
        write_pos = num_above + atomicAdd(&smem->counter, 1);
        active = false;
      } else if (bin < thr) {
        active = false;
      } else if (round == 3) {
        write_pos = K - atomicAdd(&smem->match.equal_count, -1u);
      }
      // bin == thr && round < 3: stay active for the next radix round.
    }

    topk_remain -= n_above;
    if (topk_remain == 0) break;
  }

  if (write_pos < K) params.write(write_pos, idx);
}

}  // namespace vllm::dsv4_topk
