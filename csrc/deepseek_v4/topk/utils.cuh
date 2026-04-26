// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Minimal device-side utilities used by the DeepSeek V4 indexer top-k port.
// Replaces sgl_kernel/{utils,warp,vec,type}.cuh — we only need the bits the
// top-k kernels actually touch.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace vllm::dsv4_topk {

#define VLLM_DSV4_DEVICE __forceinline__ __device__

inline constexpr uint32_t kWarpThreads = 32u;
inline constexpr uint32_t kFullMask = 0xffffffffu;

// Programmatic Dependent Launch (sm_90+). When enabled, the kernel waits for
// the predecessor on the same stream to advance past its dependents-launch
// trigger before doing anything memory-dependent. Used to overlap the
// fp8_paged_mqa_logits epilogue with the first stage of top-k.
template <bool kUsePDL>
VLLM_DSV4_DEVICE void pdl_wait_primary() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
}

template <bool kUsePDL>
VLLM_DSV4_DEVICE void pdl_trigger_secondary() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
}

// Warp-level XOR-shuffle reduce. kThreads must be a power of 2 and <= 32.
template <uint32_t kThreads = kWarpThreads, typename T>
VLLM_DSV4_DEVICE T warp_reduce_sum(T value, uint32_t active_mask = kFullMask) {
#pragma unroll
  for (auto offset = kThreads >> 1; offset > 0; offset >>= 1) {
    value = value + __shfl_xor_sync(active_mask, value, offset, 32);
  }
  return value;
}

// 128-bit-aligned vector of N elements of T (N must be a power of 2, total
// size <= 16 bytes). Used for vectorized loads/stores into shared memory.
template <typename T, std::size_t N>
struct alignas(sizeof(T) * N) AlignedVector {
  static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of two");
  static_assert(sizeof(T) * N <= 16,
                "AlignedVector exceeds the 128-bit CUDA vector limit");

  T data[N];

  VLLM_DSV4_DEVICE void load(const void* ptr, std::size_t offset = 0) {
    *reinterpret_cast<AlignedVector*>(this) =
        reinterpret_cast<const AlignedVector*>(ptr)[offset];
  }
  VLLM_DSV4_DEVICE void store(void* ptr, std::size_t offset = 0) const {
    reinterpret_cast<AlignedVector*>(ptr)[offset] = *this;
  }
  VLLM_DSV4_DEVICE void fill(T value) {
#pragma unroll
    for (std::size_t i = 0; i < N; ++i) data[i] = value;
  }
  VLLM_DSV4_DEVICE T& operator[](std::size_t i) { return data[i]; }
  VLLM_DSV4_DEVICE const T& operator[](std::size_t i) const { return data[i]; }
};

}  // namespace vllm::dsv4_topk
