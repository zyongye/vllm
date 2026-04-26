// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Thin wrappers around the CUDA PTX intrinsics used by the top-k pipeline.
// All of these require sm_90+. Ported from sglang's
// jit_kernel/include/sgl_kernel/deepseek_v4/topk/ptx.cuh.

#pragma once

#include "utils.cuh"

#include <cuda/ptx>
#include <cstdint>

namespace vllm::dsv4_topk::ptx {

VLLM_DSV4_DEVICE void mbarrier_init(uint64_t* addr, uint32_t arrives) {
  cuda::ptx::mbarrier_init(addr, arrives);
}

VLLM_DSV4_DEVICE void mbarrier_arrive(uint64_t* addr) {
  cuda::ptx::mbarrier_arrive(cuda::ptx::sem_relaxed, cuda::ptx::scope_cta,
                             cuda::ptx::space_shared, addr);
}

VLLM_DSV4_DEVICE void mbarrier_arrive_expect_tx(uint64_t* addr, uint32_t tx) {
  cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_relaxed,
                                       cuda::ptx::scope_cta,
                                       cuda::ptx::space_shared, addr, tx);
}

VLLM_DSV4_DEVICE void mbarrier_wait(uint64_t* addr, uint32_t phase) {
  while (!cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_relaxed,
                                              cuda::ptx::scope_cta, addr,
                                              phase))
    ;
}

VLLM_DSV4_DEVICE void tma_load(void* dst, const void* src, uint32_t num_bytes,
                               uint64_t* mbar) {
  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared, cuda::ptx::space_global,
                           dst, src, num_bytes, mbar);
}

// elect.sync: pick a single arbitrary thread out of an active mask. Used to
// fire a single TMA load per warp without the full ``if (tx == 0)`` cost.
VLLM_DSV4_DEVICE uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}

VLLM_DSV4_DEVICE bool elect_sync_cta(uint32_t tx) {
  const auto warp_id = tx / 32;
  const auto uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0);
  return (uniform_warp_id == 0 && elect_sync());
}

}  // namespace vllm::dsv4_topk::ptx
