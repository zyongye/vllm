// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Streaming top-k strategy for medium N. Uses a TMA-driven double-buffered
// histogram pass + scatter pass over chunks of `kSizePerStage` floats.
// Ported from
// jit_kernel/include/sgl_kernel/deepseek_v4/topk/streaming.cuh.

#pragma once

#include "common.cuh"
#include "ptx.cuh"
#include "utils.cuh"

#include <cfloat>
#include <cstdint>

namespace vllm::dsv4_topk {

template <uint32_t K>
struct StreamingTopK {
  static constexpr uint32_t kHistBits = 12;
  static constexpr uint32_t kHistBins = 1 << kHistBits;
  static constexpr uint32_t kElemPerStage = 8;
  static constexpr uint32_t kSizePerStage = kElemPerStage * kBlockSize;
  static constexpr uint32_t kNumStages = 2;  // double buffer

  static constexpr uint32_t kHistItems = kHistBins / kBlockSize;  // 4
  static_assert(kHistItems * kBlockSize == kHistBins);
  using HistVec = AlignedVector<uint32_t, kHistItems>;

  struct Smem {
    // [phase = 0 (histogram) | 1 (scatter)] x [buffer = 0 | 1]
    uint64_t barrier[2][kNumStages];
    alignas(128) uint32_t counter_gt;
    alignas(128) uint32_t counter_eq;
    alignas(128) MatchBin match;
    alignas(128) uint32_t warp_sum[kNumWarps];
    union {
      uint32_t histogram[kHistBins];
      HistVec histogram_vec[kBlockSize];
      Tie tie_buffer[kMaxTies];
    };
    union {
      float score_buffer[kNumStages][kSizePerStage];
      TieHandleSmem stage2;  // reused for the tie-handling phase
    };
  };

  // length must be 4-aligned (caller rounds up); TMA wants 16-byte alignment.
  template <bool kIsScatter>
  VLLM_DSV4_DEVICE static void issue_tma(const float* scores, uint32_t stage,
                                         uint32_t length, Smem* smem) {
    const auto buf_idx = stage % kNumStages;
    const auto offset = stage * kSizePerStage;
    const auto size = min(kSizePerStage, length - offset);
    const auto size_bytes = size * sizeof(float);
    const auto bar = &smem->barrier[kIsScatter][buf_idx];
    ptx::tma_load(smem->score_buffer[buf_idx], scores + offset, size_bytes,
                  bar);
    ptx::mbarrier_arrive_expect_tx(bar, size_bytes);
  }

  // Unified streaming pass. kIsScatter=false: build histogram (phase A).
  // kIsScatter=true: scatter using the threshold bin (phase C). Each barrier
  // is reused across iterations via the reuse-arrive pattern.
  template <bool kIsScatter>
  VLLM_DSV4_DEVICE static void stream_pass(const float* scores, uint32_t length,
                                           uint32_t thr_bin,
                                           int32_t* s_topk_indices,
                                           Smem* smem) {
    const auto tx = threadIdx.x;
    const auto num_iters = (length + kSizePerStage - 1) / kSizePerStage;
    const auto lane_id = tx % kWarpThreads;

    const auto length_aligned = (length + 3u) & ~3u;
    if (tx == 0) {
#pragma unroll
      for (uint32_t i = 0; i < kNumStages; i++) {
        if (i >= num_iters) break;
        issue_tma<kIsScatter>(scores, i, length_aligned, smem);
      }
    }

    for (uint32_t iter = 0; iter < num_iters; iter++) {
      const auto buf_idx = iter % kNumStages;
      const auto offset = iter * kSizePerStage;
      const auto this_size = min(kSizePerStage, length - offset);

      if (lane_id == 1) {
        const auto phase_bit = (iter / kNumStages) & 1;
        ptx::mbarrier_wait(&smem->barrier[kIsScatter][buf_idx], phase_bit);
      }
      __syncwarp();

#pragma unroll
      for (uint32_t i = 0; i < kElemPerStage; i++) {
        const auto local_idx = tx + i * kBlockSize;
        if (local_idx >= this_size) break;
        const auto score = smem->score_buffer[buf_idx][local_idx];
        const auto bin = extract_coarse_bin<kHistBits>(score);
        if constexpr (kIsScatter) {
          const auto global_idx = offset + local_idx;
          if (bin > thr_bin) {
            const auto pos = atomicAdd(&smem->counter_gt, 1);
            if (pos < K) s_topk_indices[pos] = global_idx;
          } else if (bin == thr_bin) {
            const auto pos = atomicAdd(&smem->counter_eq, 1);
            if (pos < kMaxTies) smem->tie_buffer[pos] = {global_idx, score};
          }
        } else {
          atomicAdd(&smem->histogram[bin], 1);
        }
      }

      __syncthreads();
      if (tx == 0) {
        if (const auto next_iter = iter + kNumStages; next_iter < num_iters) {
          issue_tma<kIsScatter>(scores, next_iter, length_aligned, smem);
        }
      }
    }
  }

  // Phase B: locate threshold bin via warp-level prefix scan.
  VLLM_DSV4_DEVICE static void find_threshold(uint32_t length, Smem* smem) {
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpThreads;
    const auto warp_id = tx / kWarpThreads;

    uint32_t orig[kHistItems];
    const auto hist_vec = smem->histogram_vec[tx];
    uint32_t local_sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < kHistItems; ++i) {
      orig[i] = hist_vec[i];
      local_sum += orig[i];
    }

    const auto warp_inc = warp_inclusive_sum(lane_id, local_sum);
    const auto warp_exc = warp_inc - local_sum;
    if (lane_id == kWarpThreads - 1) smem->warp_sum[warp_id] = warp_inc;
    __syncthreads();

    const auto tmp = smem->warp_sum[lane_id];
    uint32_t prefix_sum = warp_reduce_sum(lane_id < warp_id ? tmp : 0);
    prefix_sum += warp_exc;
#pragma unroll
    for (uint32_t i = 0; i < kHistItems; ++i) {
      prefix_sum += orig[i];
      const auto above = length - prefix_sum;
      if (above < K && above + orig[i] >= K) {
        smem->match = {
            .bin = tx * kHistItems + i,
            .above_count = above,
            .equal_count = orig[i],
        };
      }
    }
    __syncthreads();
  }

  VLLM_DSV4_DEVICE static void run(const float* scores, uint32_t length,
                                   int32_t* topk_indices, void* _smem) {
    const auto smem = static_cast<Smem*>(_smem);
    const auto tx = threadIdx.x;
    __builtin_assume(tx < kBlockSize);

    {
      HistVec zero;
      zero.fill(0);
      smem->histogram_vec[tx] = zero;
      if (tx < 2 * kNumStages) {
        const auto base_barrier = &smem->barrier[0][0];
        ptx::mbarrier_init(&base_barrier[tx], 1);
      }
      if (tx == 0) {
        smem->counter_gt = 0;
        smem->counter_eq = 0;
      }
      __syncthreads();
    }

    // Phase A: histogram.
    stream_pass<false>(scores, length, 0, nullptr, smem);

    // Phase B: threshold bin.
    find_threshold(length, smem);

    // Phase C: scatter.
    stream_pass<true>(scores, length, smem->match.bin, topk_indices, smem);
  }

  VLLM_DSV4_DEVICE static void transform(TransformParams params, void* _smem) {
    // Phase D: page-translate above entries, then refine ties.
    const auto smem = static_cast<Smem*>(_smem);
    const auto tx = threadIdx.x;
    const auto num_above = smem->match.above_count;
    if (tx < num_above) params.transform(tx);
    const auto num_equal = smem->counter_eq;
    if (num_above >= K || num_equal == 0) return;
    const auto clamped_ties = min(num_equal, kMaxTies);
    tie_handle_transform(smem->tie_buffer, clamped_ties, num_above, K, params,
                         &smem->stage2);
  }
};

}  // namespace vllm::dsv4_topk
