// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// DeepSeek V4 indexer top-k (k = 512). Ported from sglang's
// jit_kernel/csrc/deepseek_v4/topk_v2.cuh.
//
// Combines three strategies (Register / Streaming / Cluster) dispatched per
// row by a separate plan kernel that decides a `cluster_threshold` from the
// observed seq_lens distribution. The host side picks one of three launch
// shapes:
//   1. all rows fit in the small (register) path -> single short kernel
//   2. small batch (<= kNumClusters) with some long rows -> fused cluster
//      kernel (stage 1 + tie-break in one launch)
//   3. larger batch -> persistent cluster stage 1 + non-cluster stage 2
//
// Architecture support: Hopper (sm_90a) and Blackwell datacenter (sm_100/
// sm_103). Requires thread-block clusters, TMA bulk async copy, mbarrier,
// and Programmatic Dependent Launch — sm_120 (consumer Blackwell) lacks
// clusters and is not supported. The heuristic constants in `topk_plan`
// were tuned on B200 (sglang upstream); they are functionally correct on
// H100/H200 too but may be suboptimal until retuned.

#include "topk/cluster.cuh"
#include "topk/common.cuh"
#include "topk/register.cuh"
#include "topk/streaming.cuh"
#include "topk/utils.cuh"

#include "core/registration.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>

namespace vllm::dsv4_topk {

using Large = ClusterTopK<512>;
using Medium = StreamingTopK<512>;
using Small = RegisterTopK<512>;

using Metadata = Large::Metadata;
constexpr uint32_t kNumClusters = 15;            // hardware-capped persistent count
constexpr uint32_t kClusterSize = Large::kClusterSize;
constexpr uint32_t kMax2PassLength = Small::kMax2PassLength;
constexpr uint32_t kMaxSupportedLength = Large::kMaxLength;

// Row 0 of the metadata tensor stores GlobalMetadata; rows [1..N+1) hold the
// per-item Metadata entries that the persistent stage-1 consumes.
struct alignas(16) GlobalMetadata {
  uint32_t cluster_threshold;
  uint32_t num_cluster_items;
  uint32_t reserved[2];
};
static_assert(sizeof(GlobalMetadata) == sizeof(Metadata),
              "metadata row 0 layout must match Metadata stride");

#define VLLM_SMALL_TOPK_KERNEL __global__ __launch_bounds__(kBlockSize, 2)
#define VLLM_LARGE_CLUSTER __cluster_dims__(1, kClusterSize, 1)
// Stage 1 is persistent + cluster -> high smem -> occupancy 1.
#define VLLM_LARGE_TOPK_STAGE_1 \
  __global__ __launch_bounds__(kBlockSize, 1) VLLM_LARGE_CLUSTER
// Stage 2 is non-cluster + small smem -> occupancy 2.
#define VLLM_LARGE_TOPK_STAGE_2 __global__ __launch_bounds__(kBlockSize, 2)
#define VLLM_FUSED_COMBINE_KERNEL \
  __global__ __launch_bounds__(kBlockSize, 1) VLLM_LARGE_CLUSTER
#define VLLM_PLAN_KERNEL __global__ __launch_bounds__(kBlockSize, 1)

struct TopKParams {
  const uint32_t* __restrict__ seq_lens;
  const float* __restrict__ scores;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int64_t score_stride;
  int64_t page_table_stride;
  uint8_t* __restrict__ workspace;
  const Metadata* __restrict__ metadata = nullptr;
  int64_t workspace_stride;  // bytes per batch
  uint32_t batch_size;
  uint32_t page_bits;

  VLLM_DSV4_DEVICE const float* get_scores(uint32_t batch_id) const {
    return scores + batch_id * score_stride;
  }
  VLLM_DSV4_DEVICE TransformParams get_transform(uint32_t batch_id,
                                                 int32_t* indices) const {
    return {
        .page_table = page_table + batch_id * page_table_stride,
        .indices_in = indices,
        .indices_out = page_indices + batch_id * 512,
        .page_bits = page_bits,
    };
  }
  VLLM_DSV4_DEVICE const GlobalMetadata& get_global_metadata() const {
    return *reinterpret_cast<const GlobalMetadata*>(metadata);
  }
  VLLM_DSV4_DEVICE const Metadata& get_item_metadata(uint32_t work_id) const {
    return metadata[1 + work_id];  // skip the GlobalMetadata row
  }
};

VLLM_DSV4_DEVICE uint2 partition_work(uint32_t length, uint32_t rank) {
  constexpr uint32_t kTMAAlign = 4;
  const auto total_units = (length + kTMAAlign - 1) / kTMAAlign;
  const auto base = total_units / kClusterSize;
  const auto extra = total_units % kClusterSize;
  const auto local_units = base + (rank < extra ? 1u : 0u);
  const auto offset_units = rank * base + min(rank, extra);
  const auto offset = offset_units * kTMAAlign;
  const auto finish = min(offset + local_units * kTMAAlign, length);
  return {offset, finish - offset};
}

// --------------------------------------------------------------------------
// Plan kernel: decides cluster_threshold from the observed seq_lens
// distribution and compacts items with seq_len > threshold into metadata[1..].
// --------------------------------------------------------------------------

VLLM_PLAN_KERNEL void topk_plan(const uint32_t* __restrict__ seq_lens,
                                Metadata* __restrict__ metadata,
                                uint32_t batch_size,
                                uint32_t static_cluster_threshold) {
  // (threshold, max_batch_size_for_that_threshold). Tuned on B200 by sglang.
  struct Pair {
    uint32_t threshold;
    uint32_t max_batch_size;
  };
  constexpr Pair kCandidates[] = {
      {32768, 30},   {40960, 45},   {49152, 45},   {65536, 60},
      {98304, 60},   {131072, 75},  {196608, 90},  {262144, 105},
  };
  constexpr uint32_t kNumCandidates =
      sizeof(kCandidates) / sizeof(kCandidates[0]);
  constexpr uint32_t kMinBatchSize = kCandidates[0].max_batch_size;
  static_assert(kCandidates[0].threshold == kMax2PassLength);
  static_assert(kCandidates[kNumCandidates - 1].threshold ==
                kMaxSupportedLength);

  __shared__ uint32_t s_count;
  __shared__ uint32_t s_counts[kNumCandidates];
  __shared__ uint32_t s_threshold;

  const auto tx = threadIdx.x;
  if (tx == 0) s_count = 0;
  if (tx < kNumCandidates) s_counts[tx] = 0;
  __syncthreads();

  if (static_cluster_threshold > 0) {
    if (tx == 0) s_threshold = static_cluster_threshold;
  } else if (batch_size <= kMinBatchSize) {
    if (tx == 0) s_threshold = kMax2PassLength;
  } else {
    for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
      const uint32_t sl = seq_lens[i];
      assert(sl <= kMaxSupportedLength);
      uint32_t count = 0;
#pragma unroll
      for (uint32_t j = 0; j < kNumCandidates; ++j) {
        count += (sl > kCandidates[j].threshold ? 1 : 0);
      }
      if (count > 0) {
        atomicAdd(&s_counts[count - 1], 1);
      }
    }
    __syncthreads();
    if (tx == 0) {
      uint32_t accum = 0;
      uint32_t chosen = kMaxSupportedLength;
#pragma unroll
      for (uint32_t i = 0; i < kNumCandidates; ++i) {
        const auto j = kNumCandidates - 1 - i;
        accum += s_counts[j];
        if (accum > kCandidates[j].max_batch_size) break;
        chosen = kCandidates[j].threshold;
      }
      s_threshold = chosen;
    }
  }
  __syncthreads();
  const auto cluster_threshold = max(s_threshold, kMax2PassLength);

  // Compact items with seq_len > cluster_threshold into metadata[1..N+1).
  for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
    const uint32_t sl = seq_lens[i];
    if (sl > cluster_threshold) {
      const auto pos = atomicAdd(&s_count, 1);
      metadata[1 + pos] = {i, sl, false};
    }
  }
  __syncthreads();
  const auto N = s_count;

  // has_next chain for the persistent consumer + sentinel slots.
  for (uint32_t i = tx; i < N; i += kBlockSize) {
    if (i + kNumClusters < N) metadata[1 + i].has_next = true;
  }
  if (tx < kNumClusters && tx >= N) metadata[1 + tx] = {0, 0, false};
  if (tx == 0) {
    auto* g = reinterpret_cast<GlobalMetadata*>(metadata);
    *g = {
        .cluster_threshold = cluster_threshold,
        .num_cluster_items = N,
        .reserved = {0, 0},
    };
  }
}

// --------------------------------------------------------------------------
// Short kernel: all rows fit in the register path (max_seq_len <=
// Small::kMax1PassLength).
// --------------------------------------------------------------------------

VLLM_SMALL_TOPK_KERNEL void topk_short_transform(
    const __grid_constant__ TopKParams params) {
  alignas(128) extern __shared__ uint8_t smem[];
  __shared__ int32_t s_topk_indices[512];
  const auto batch_id = blockIdx.x;
  const auto seq_len = params.seq_lens[batch_id];
  const auto transform = params.get_transform(batch_id, s_topk_indices);
  if (seq_len <= 512) {
    trivial_transform(transform, seq_len, 512);
  } else {
    Small::run(params.get_scores(batch_id), s_topk_indices, seq_len, smem,
               /*use_pdl=*/true);
    pdl_trigger_secondary<true>();
    Small::transform(transform);
  }
}

// --------------------------------------------------------------------------
// Persistent stage 1 (cluster). One CTA per cluster; the persistent block
// walks `metadata[1..N]` round-robin and runs Large::stage1 per item.
// --------------------------------------------------------------------------

VLLM_LARGE_TOPK_STAGE_1 void topk_combine_preprocess(
    const __grid_constant__ TopKParams params) {
  alignas(128) extern __shared__ uint8_t smem[];
  __shared__ int32_t s_topk_indices[512];
  uint32_t work_id = blockIdx.x;
  uint32_t batch_id = 0, seq_len = 0, length = 0, offset = 0;
  bool has_next = false;
  const auto cluster_rank = blockIdx.y;

  const auto prefetch_metadata = [&] {
    const auto m = params.get_item_metadata(work_id);
    batch_id = m.batch_id;
    seq_len = m.seq_len;
    has_next = m.has_next;
    work_id += kNumClusters;
  };
  const auto launch_prologue = [&] {
    const auto partition = partition_work(seq_len, cluster_rank);
    offset = partition.x;
    length = partition.y;
    Large::stage1_prologue(params.get_scores(batch_id) + offset, length, smem);
  };

  pdl_wait_primary<true>();
  pdl_trigger_secondary<true>();

  prefetch_metadata();
  if (seq_len == 0) return;
  Large::stage1_init(smem);
  launch_prologue();
  while (true) {
    const auto this_length = length;
    const auto this_offset = offset;
    const auto need_prefetch = has_next;
    const auto transform = params.get_transform(batch_id, s_topk_indices);
    const auto ws = params.workspace + batch_id * params.workspace_stride;
    if (need_prefetch) prefetch_metadata();
    Large::stage1(s_topk_indices, this_length, smem, /*reuse=*/true);
    if (need_prefetch) launch_prologue();
    Large::stage1_epilogue(transform, this_offset, ws, smem);
    if (!need_prefetch) break;
  }
}

// --------------------------------------------------------------------------
// Stage 2 (non-cluster). Per-row dispatch: trivial / Small / Medium / Large.
// --------------------------------------------------------------------------

VLLM_LARGE_TOPK_STAGE_2 void topk_combine_transform(
    const __grid_constant__ TopKParams params) {
  alignas(128) extern __shared__ uint8_t smem[];
  __shared__ int32_t s_topk_indices[512];
  const auto batch_id = blockIdx.x;
  const auto seq_len = params.seq_lens[batch_id];
  const auto cluster_threshold = params.get_global_metadata().cluster_threshold;
  const auto transform = params.get_transform(batch_id, s_topk_indices);
  if (seq_len <= 512) {
    trivial_transform(transform, seq_len, 512);
  } else if (seq_len <= kMax2PassLength) {
    if (seq_len <= Small::kMax1PassLength) {
      Small::run(params.get_scores(batch_id), s_topk_indices, seq_len, smem);
    } else {
      __syncwarp();
      Small::run<true>(params.get_scores(batch_id), s_topk_indices, seq_len,
                       smem);
    }
    Small::transform(transform);
  } else if (seq_len <= cluster_threshold) {
    Medium::run(params.get_scores(batch_id), seq_len, s_topk_indices, smem);
    Medium::transform(transform, smem);
  } else {
    const auto ws = params.workspace + batch_id * params.workspace_stride;
    pdl_wait_primary<true>();
    Large::transform(transform, ws, smem);
  }
}

// --------------------------------------------------------------------------
// Fused kernel for small batches. Both stage 1 and the tie-break run inside
// the same launch; cluster rank 0 finishes the row.
// --------------------------------------------------------------------------

VLLM_FUSED_COMBINE_KERNEL void topk_fused_transform(
    const __grid_constant__ TopKParams params) {
  alignas(128) extern __shared__ uint8_t smem[];
  __shared__ int32_t s_topk_indices[512];
  const auto batch_id = blockIdx.x;
  const auto cluster_rank = blockIdx.y;
  const auto seq_len = params.seq_lens[batch_id];
  const auto transform = params.get_transform(batch_id, s_topk_indices);
  if (seq_len <= 512) {
    if (cluster_rank != 0) return;
    trivial_transform(transform, seq_len, 512);
  } else if (seq_len <= Small::kMax1PassLength) {
    if (cluster_rank != 0) return;
    Small::run(params.get_scores(batch_id), s_topk_indices, seq_len, smem,
               /*use_pdl=*/true);
    Small::transform(transform);
  } else {
    const auto partition = partition_work(seq_len, cluster_rank);
    const auto offset = partition.x;
    const auto length = partition.y;
    const auto ws = params.workspace + batch_id * params.workspace_stride;
    Large::stage1_init(smem);
    pdl_wait_primary<true>();
    Large::stage1_prologue(params.get_scores(batch_id) + offset, length, smem);
    Large::stage1(s_topk_indices, length, smem);
    Large::stage1_epilogue(transform, offset, ws, smem);
    cooperative_groups::this_cluster().sync();
    if (cluster_rank != 0) return;
    Large::transform(transform, ws, smem);
  }
}

constexpr size_t kStage1SMEM = sizeof(Large::Smem) + 128;
constexpr size_t kStage2SMEM =
    (sizeof(Small::Smem) > sizeof(Medium::Smem) ? sizeof(Small::Smem)
                                                : sizeof(Medium::Smem)) +
    128;

// Per-(kernel, smem) memoization: each instantiation has its own static. This
// matters because cudaFuncSetAttribute is per-function and we want it to fire
// exactly once per kernel symbol.
template <auto* f, size_t kSmem>
void setup_kernel_smem_once() {
  [[maybe_unused]] static const auto result = [] {
    return cudaFuncSetAttribute(reinterpret_cast<const void*>(f),
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(kSmem));
  }();
  TORCH_CHECK(result == cudaSuccess,
              "fast_topk_v2: cudaFuncSetAttribute failed: ",
              cudaGetErrorString(result));
}

// --------------------------------------------------------------------------
// Host-side launchers
// --------------------------------------------------------------------------

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DTYPE(x, t) \
  TORCH_CHECK(x.scalar_type() == (t), #x " must be ", #t)
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

}  // namespace vllm::dsv4_topk

void fast_topk_v2_plan(const torch::Tensor& seq_lens, torch::Tensor& metadata,
                       int64_t static_cluster_threshold) {
  using namespace vllm::dsv4_topk;
  CHECK_CUDA(seq_lens);
  CHECK_CUDA(metadata);
  CHECK_DTYPE(seq_lens, torch::kInt32);
  CHECK_DTYPE(metadata, torch::kInt32);
  TORCH_CHECK(seq_lens.dim() == 1);
  TORCH_CHECK(metadata.dim() == 2 && metadata.size(1) == 4);
  TORCH_CHECK(metadata.size(0) == seq_lens.size(0) + 1,
              "metadata must be (batch_size + 1, 4)");
  CHECK_CONTIG(seq_lens);
  CHECK_CONTIG(metadata);

  const auto batch_size = static_cast<uint32_t>(seq_lens.size(0));
  if (batch_size <= kNumClusters) return;  // metadata unused in fused path

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(1);
  cfg.blockDim = dim3(kBlockSize);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cfg.numAttrs = 0;
  TORCH_CHECK(cudaLaunchKernelEx(
                  &cfg, &topk_plan,
                  reinterpret_cast<const uint32_t*>(seq_lens.data_ptr<int32_t>()),
                  reinterpret_cast<Metadata*>(metadata.data_ptr<int32_t>()),
                  batch_size,
                  static_cast<uint32_t>(static_cluster_threshold)) == cudaSuccess,
              "fast_topk_v2_plan launch failed: ",
              cudaGetErrorString(cudaGetLastError()));
}

void fast_topk_v2(const torch::Tensor& scores, const torch::Tensor& seq_lens,
                  const torch::Tensor& page_table, torch::Tensor& page_indices,
                  int64_t page_size, const torch::Tensor& workspace,
                  const torch::Tensor& metadata) {
  using namespace vllm::dsv4_topk;
  CHECK_CUDA(scores);
  CHECK_CUDA(seq_lens);
  CHECK_CUDA(page_table);
  CHECK_CUDA(page_indices);
  CHECK_CUDA(workspace);
  CHECK_CUDA(metadata);
  CHECK_DTYPE(scores, torch::kFloat32);
  CHECK_DTYPE(seq_lens, torch::kInt32);
  CHECK_DTYPE(page_table, torch::kInt32);
  CHECK_DTYPE(page_indices, torch::kInt32);
  CHECK_DTYPE(workspace, torch::kInt32);
  CHECK_DTYPE(metadata, torch::kInt32);

  TORCH_CHECK(scores.dim() == 2 && scores.stride(1) == 1,
              "scores must be 2D with last stride 1");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.is_contiguous());
  TORCH_CHECK(page_table.dim() == 2 && page_table.stride(1) == 1,
              "page_table must be 2D with last stride 1");
  TORCH_CHECK(page_indices.dim() == 2 && page_indices.is_contiguous() &&
                  page_indices.size(1) == 512,
              "page_indices must be (B, 512) contiguous");
  TORCH_CHECK(workspace.dim() == 2 && workspace.stride(1) == 1 &&
                  workspace.size(1) == Large::kWorkspaceInts,
              "workspace must be (B, kWorkspaceInts) with last stride 1");
  TORCH_CHECK(metadata.dim() == 2 && metadata.size(1) == 4 &&
                  metadata.is_contiguous(),
              "metadata must be (B + 1, 4) contiguous");

  const auto batch_size = static_cast<uint32_t>(scores.size(0));
  TORCH_CHECK(seq_lens.size(0) == batch_size);
  TORCH_CHECK(page_table.size(0) == batch_size);
  TORCH_CHECK(page_indices.size(0) == batch_size);
  TORCH_CHECK(workspace.size(0) == batch_size);
  TORCH_CHECK(metadata.size(0) == batch_size + 1);

  const auto max_seq_len = static_cast<uint32_t>(scores.size(1));
  TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0,
              "page_size must be a positive power of 2");
  TORCH_CHECK(scores.stride(0) % 4 == 0,
              "score stride must be a multiple of 4 (TMA 16-byte alignment)");

  // page_bits = log2(page_size). __builtin_ctzll is a host-side compiler
  // builtin available under C++17 (vLLM compiles host code with C++17).
  const auto page_bits = static_cast<uint32_t>(
      __builtin_ctzll(static_cast<unsigned long long>(page_size)));
  TopKParams params{
      .seq_lens =
          reinterpret_cast<const uint32_t*>(seq_lens.data_ptr<int32_t>()),
      .scores = scores.data_ptr<float>(),
      .page_table = page_table.data_ptr<int32_t>(),
      .page_indices = page_indices.data_ptr<int32_t>(),
      .score_stride = scores.stride(0),
      .page_table_stride = page_table.stride(0),
      .workspace = reinterpret_cast<uint8_t*>(workspace.data_ptr<int32_t>()),
      .metadata =
          reinterpret_cast<const Metadata*>(metadata.data_ptr<int32_t>()),
      .workspace_stride =
          workspace.stride(0) * static_cast<int64_t>(sizeof(int32_t)),
      .batch_size = batch_size,
      .page_bits = page_bits,
  };

  const auto stream = at::cuda::getCurrentCUDAStream().stream();

  // Helper: build a cudaLaunchConfig with optional PDL + cluster attributes.
  // The attribute storage must outlive cudaLaunchKernelEx (cfg.attrs points
  // into it), so we keep it as a local in each call site.
  auto make_cfg = [&](dim3 grid, dim3 block, size_t smem,
                      cudaLaunchAttribute* attrs, bool enable_cluster,
                      bool enable_pdl) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = static_cast<unsigned>(smem);
    cfg.stream = stream;
    int n = 0;
    if (enable_pdl) {
      attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attrs[n].val.programmaticStreamSerializationAllowed = 1;
      ++n;
    }
    if (enable_cluster) {
      attrs[n].id = cudaLaunchAttributeClusterDimension;
      attrs[n].val.clusterDim = {1, kClusterSize, 1};
      ++n;
    }
    cfg.numAttrs = n;
    cfg.attrs = n ? attrs : nullptr;
    return cfg;
  };

  auto check_launch = [](cudaError_t err) {
    TORCH_CHECK(err == cudaSuccess,
                "fast_topk_v2 launch failed: ", cudaGetErrorString(err));
  };

  if (max_seq_len <= Small::kMax1PassLength) {
    setup_kernel_smem_once<&topk_short_transform, kStage2SMEM>();
    cudaLaunchAttribute attrs[2];
    auto cfg = make_cfg(dim3(batch_size), dim3(kBlockSize), kStage2SMEM, attrs,
                        /*cluster=*/false, /*pdl=*/true);
    check_launch(cudaLaunchKernelEx(&cfg, topk_short_transform, params));
  } else if (batch_size <= kNumClusters) {
    constexpr size_t kFusedSMEM =
        kStage1SMEM > kStage2SMEM ? kStage1SMEM : kStage2SMEM;
    setup_kernel_smem_once<&topk_fused_transform, kFusedSMEM>();
    cudaLaunchAttribute attrs[2];
    auto cfg = make_cfg(dim3(batch_size, kClusterSize), dim3(kBlockSize),
                        kFusedSMEM, attrs, /*cluster=*/true, /*pdl=*/true);
    check_launch(cudaLaunchKernelEx(&cfg, topk_fused_transform, params));
  } else {
    const auto num_clusters = std::min<uint32_t>(batch_size, kNumClusters);
    setup_kernel_smem_once<&topk_combine_preprocess, kStage1SMEM>();
    cudaLaunchAttribute attrs1[2];
    auto cfg1 = make_cfg(dim3(num_clusters, kClusterSize), dim3(kBlockSize),
                         kStage1SMEM, attrs1, /*cluster=*/true, /*pdl=*/true);
    check_launch(cudaLaunchKernelEx(&cfg1, topk_combine_preprocess, params));

    setup_kernel_smem_once<&topk_combine_transform, kStage2SMEM>();
    cudaLaunchAttribute attrs2[2];
    auto cfg2 = make_cfg(dim3(batch_size), dim3(kBlockSize), kStage2SMEM,
                         attrs2, /*cluster=*/false, /*pdl=*/true);
    check_launch(cudaLaunchKernelEx(&cfg2, topk_combine_transform, params));
  }
}

int64_t fast_topk_v2_workspace_ints() {
  return static_cast<int64_t>(vllm::dsv4_topk::Large::kWorkspaceInts);
}

// Register impls here (instead of in torch_bindings.cpp) so they only exist
// when CMake compiles this source — i.e., when the target build has a
// compatible Hopper / Blackwell-datacenter arch. On other configs the schema
// remains defined but a call surfaces a clear "no impl" runtime error.
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("fast_topk_v2_plan", &fast_topk_v2_plan);
  m.impl("fast_topk_v2", &fast_topk_v2);
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CompositeExplicitAutograd, m) {
  m.impl("fast_topk_v2_workspace_ints", &fast_topk_v2_workspace_ints);
}
