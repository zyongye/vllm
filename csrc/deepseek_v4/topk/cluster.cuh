// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Cluster top-k strategy for very large N. Uses Hopper thread-block clusters
// (cooperative_groups::this_cluster) to parallelize histogram + scatter across
// up to ``kClusterSize`` blocks per row. Each row is processed in two stages:
//   stage 1: per-block histogram, all-reduce across the cluster, threshold
//            scatter, and an epilogue that page-translates strictly-above
//            entries to global memory and stages ties into a per-row workspace.
//   stage 2: tie-break across the cluster's combined ties (run by cluster
//            rank 0 in the fused kernel, or as a separate launch otherwise).
// Ported from
// jit_kernel/include/sgl_kernel/deepseek_v4/topk/cluster.cuh.

#pragma once

#include "common.cuh"
#include "ptx.cuh"
#include "utils.cuh"

#include <cooperative_groups.h>
#include <cstdint>

namespace vllm::dsv4_topk {

template <uint32_t K>
struct ClusterTopK {
  static constexpr uint32_t kClusterSize = 8;
  static constexpr uint32_t kHistBits = 10;
  static constexpr uint32_t kHistBins = 1 << kHistBits;
  static constexpr uint32_t kElemPerStage = 8;
  static constexpr uint32_t kSizePerStage = kElemPerStage * kBlockSize;
  static constexpr uint32_t kNumStages = 4;
  static constexpr uint32_t kMaxLength = kClusterSize * kNumStages * kSizePerStage;
  static constexpr uint32_t kAboveBits = 11;

  struct Smem {
    uint64_t barrier[kNumStages];
    uint32_t local_above_equal[kClusterSize];
    uint32_t prefix_above_equal;
    alignas(128) uint32_t counter_gt;
    alignas(128) uint32_t counter_eq;
    alignas(128) MatchBin match;
    alignas(128) uint32_t warp_sum[kNumWarps];
    uint32_t histogram[kHistBins];
    alignas(128) float score_buffer[kNumStages][kSizePerStage];
    Tie tie_buffer[kMaxTies];
  };

  // Per-row metadata produced by the plan kernel and consumed by the fused /
  // stage-1 kernels. {batch_id, seq_len, has_next} arranged in an int4-sized
  // 16-byte struct so the planner can do contiguous int32x4 stores.
  struct alignas(16) Metadata {
    uint32_t batch_id;
    uint32_t seq_len;
    bool has_next;
  };

  // Per-row workspace storing {(num_above, num_ties)} + the gathered ties.
  struct WorkSpace {
    uint2 metadata;
    Tie ties[kMaxTies];
  };

  static constexpr uint32_t kWorkspaceInts = sizeof(WorkSpace) / sizeof(uint32_t);

  VLLM_DSV4_DEVICE static void stage1_init(void* _smem) {
    const auto tx = threadIdx.x;
    __builtin_assume(tx < kBlockSize);
    const auto smem = static_cast<Smem*>(_smem);
    if (tx < kHistBins) smem->histogram[tx] = 0;
    if (tx < kNumStages) ptx::mbarrier_init(&smem->barrier[tx], 1);
    __syncthreads();
  }

  VLLM_DSV4_DEVICE static void stage1_prologue(const float* scores,
                                               uint32_t length, void* _smem) {
    if (threadIdx.x == 0) {
      const auto smem = static_cast<Smem*>(_smem);
      const auto num_stages = (length + kSizePerStage - 1) / kSizePerStage;
      const auto length_aligned = (length + 3u) & ~3u;
#pragma unroll
      for (uint32_t stage = 0; stage < kNumStages; stage++) {
        if (stage >= num_stages) break;
        const auto offset = stage * kSizePerStage;
        const auto size = min(kSizePerStage, length_aligned - offset);
        const auto size_bytes = size * sizeof(float);
        const auto bar = &smem->barrier[stage];
        ptx::tma_load(smem->score_buffer[stage], scores + offset, size_bytes,
                      bar);
        ptx::mbarrier_arrive_expect_tx(bar, size_bytes);
      }
    }
  }

  VLLM_DSV4_DEVICE static void stage1(int32_t* indices, uint32_t length,
                                      void* _smem, bool reuse = false) {
    const auto smem = static_cast<Smem*>(_smem);
    const auto tx = threadIdx.x;
    __builtin_assume(tx < kBlockSize);
    const auto lane_id = tx % kWarpThreads;
    const auto warp_id = tx / kWarpThreads;

    // Local histogram.
#pragma unroll
    for (uint32_t stage = 0; stage < kNumStages; stage++) {
      const auto offset = stage * kSizePerStage;
      if (offset >= length) break;
      const auto size = min(kSizePerStage, length - offset);
      if (lane_id == 0) ptx::mbarrier_wait(&smem->barrier[stage], 0);
      __syncwarp();
#pragma unroll
      for (uint32_t i = 0; i < kElemPerStage; ++i) {
        const auto idx = tx + i * kBlockSize;
        if (idx >= size) break;
        const auto score = smem->score_buffer[stage][idx];
        const auto bin = extract_coarse_bin<kHistBits>(score);
        atomicAdd(&smem->histogram[bin], 1);
      }
    }

    static_assert(kHistBins <= kBlockSize);

    // Two-shot all-reduce across the cluster.
    {
      auto cluster = cooperative_groups::this_cluster();
      cluster.sync();
      const auto cluster_rank = blockIdx.y;
      const auto kLocalSize = kHistBins / kClusterSize;
      const auto offset = kLocalSize * cluster_rank;

      const auto src_tx = tx / kClusterSize;
      const auto src_rank = tx % kClusterSize;

      if (tx < kHistBins) {
        const auto addr = &smem->histogram[offset + src_tx];
        const auto src_addr = cluster.map_shared_rank(addr, src_rank);
        *src_addr = warp_reduce_sum<kClusterSize>(*src_addr);
      }
      cluster.sync();
    }

    // Each block now holds the full cluster histogram. Find the threshold.
    {
      const auto value = tx < kHistBins ? smem->histogram[tx] : 0;
      const auto warp_inc = warp_inclusive_sum(lane_id, value);
      if (lane_id == kWarpThreads - 1) {
        smem->warp_sum[warp_id] = warp_inc;
      }

      __syncthreads();
      const auto tmp = smem->warp_sum[lane_id];
      const auto total_length = warp_reduce_sum(tmp);
      uint32_t prefix_sum = warp_reduce_sum(lane_id < warp_id ? tmp : 0);
      prefix_sum += warp_inc;
      const auto above = total_length - prefix_sum;
      if (tx < kHistBins && above < K && above + value >= K) {
        smem->counter_gt = smem->counter_eq = 0;
        smem->match = {
            .bin = tx,
            .above_count = above,
            .equal_count = value,
        };
      }
      __syncthreads();
    }

    const auto thr_bin = smem->match.bin;

    // Scatter strictly-above entries to `indices`, stash ties in tie_buffer.
#pragma unroll
    for (uint32_t stage = 0; stage < kNumStages; stage++) {
      const auto offset = stage * kSizePerStage;
      if (offset >= length) break;
#pragma unroll
      for (uint32_t i = 0; i < kElemPerStage; ++i) {
        const auto buf_idx = tx + i * kBlockSize;
        const auto global_idx = offset + buf_idx;
        if (global_idx >= length) break;
        const auto score = smem->score_buffer[stage][buf_idx];
        const auto bin = extract_coarse_bin<kHistBits>(score);
        if (bin > thr_bin) {
          indices[atomicAdd(&smem->counter_gt, 1)] = global_idx;
        } else if (bin == thr_bin) {
          const auto pos = atomicAdd(&smem->counter_eq, 1);
          if (pos < kMaxTies) smem->tie_buffer[pos] = {global_idx, score};
        }
      }
    }
    if (reuse) {
      const auto num_stages = (length + kSizePerStage - 1) / kSizePerStage;
      if (tx < kHistBins) smem->histogram[tx] = 0;
      if (tx < num_stages) ptx::mbarrier_arrive(&smem->barrier[tx]);
    }
    __syncthreads();
  }

  VLLM_DSV4_DEVICE static void stage1_epilogue(TransformParams params,
                                               uint32_t offset, void* _ws,
                                               void* _smem) {
    auto cluster = cooperative_groups::this_cluster();
    const auto smem = static_cast<Smem*>(_smem);
    const auto tx = threadIdx.x;
    const auto local_above = smem->counter_gt;
    const auto local_equal = smem->counter_eq;
    const auto cluster_rank = blockIdx.y;

    constexpr uint32_t kAboveMask = (1 << kAboveBits) - 1;
    static_assert(kAboveMask >= K);

    static_assert(kMaxTies <= kBlockSize);
    const auto idx_above = tx < local_above ? params.indices_in[tx] : 0;
    const auto tie_value = tx < local_equal ? smem->tie_buffer[tx] : Tie{0, 0.0f};

    // Push counts to remote shared memory to reduce inter-block latency.
    if (tx < kClusterSize) {
      const auto value = (local_equal << kAboveBits) | local_above;
      const auto dst_addr = cluster.map_shared_rank(smem->local_above_equal, tx);
      dst_addr[cluster_rank] = value;
    }
    // After this final sync, every block can read only its own smem (peer
    // ranks may have already exited), so we don't touch remote smem again.
    cluster.sync();
    if (tx < kClusterSize) {
      const auto value = tx < cluster_rank ? smem->local_above_equal[tx] : 0;
      const auto kActiveMask = (1u << kClusterSize) - 1;
      smem->prefix_above_equal = warp_reduce_sum<kClusterSize>(value, kActiveMask);
    }
    __syncthreads();

    const auto prefix_packed = smem->prefix_above_equal;
    const auto prefix_above = prefix_packed & kAboveMask;
    const auto prefix_equal = prefix_packed >> kAboveBits;

    // Page-translate strictly-above entries.
    if (tx < local_above) {
      params.write(tx + prefix_above, idx_above + offset);
    }
    // Stage ties into the per-row workspace (regular global writes).
    const auto ws = static_cast<WorkSpace*>(_ws);
    if (tx < local_equal && tx + prefix_equal < kMaxTies) {
      ws->ties[tx + prefix_equal] = {tie_value.idx + offset, tie_value.score};
    }
    // Last cluster rank publishes the sums into ws->metadata.
    if (cluster_rank == kClusterSize - 1 && tx == 0) {
      const auto sum_above = prefix_above + local_above;
      const auto sum_equal = prefix_equal + local_equal;
      ws->metadata = make_uint2(sum_above, sum_equal);
    }
  }

  VLLM_DSV4_DEVICE static void transform(TransformParams params, const void* _ws,
                                         void* _smem) {
    const auto ws = static_cast<const WorkSpace*>(_ws);
    const auto meta = &ws->metadata;
    const auto num_above = meta->x;
    const auto num_equal = meta->y;
    if (num_above >= K || num_equal == 0) return;
    const auto clamped_ties = min(num_equal, kMaxTies);
    tie_handle_transform(ws->ties, clamped_ties, num_above, K, params, _smem);
  }
};

}  // namespace vllm::dsv4_topk
