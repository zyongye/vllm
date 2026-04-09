#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#include "quantization/w8a8/fp8/common.cuh"
#ifdef USE_ROCM
  #include "quantization/w8a8/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#endif

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
typedef __hip_bfloat16 __nv_bfloat16;
#endif

namespace vllm {

// NOTE Be EXTRA careful with raw_kv_scalar_t, for __half and __nv_bfloat16 it's
// using u16 as the backing type.
template <typename qk_t, bool IS_NEOX, typename raw_kv_scalar_t,
          typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_mla_rope_fused_kernel(
    const int64_t* __restrict__ positions,  // [num_tokens]
    qk_t* __restrict__ q_pe,        // [num_tokens, num_q_heads, rot_dim]
    qk_t* __restrict__ k_pe,        // [num_tokens, rot_dim]
    const qk_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const qk_t* __restrict__ rope_cos_sin_cache,  // [max_position, 2,
                                                  // rot_dim // 2]
    const int rot_dim, const int64_t q_pe_stride_token,
    const int64_t q_pe_stride_head, const int64_t k_pe_stride,
    const int64_t kv_c_stride, const int num_q_heads,
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank +
                                     // rot_dim)]
    const int64_t* __restrict__ kv_cache_slot_mapping,  // [num_tokens]
    const int block_stride, const int entry_stride, const int kv_lora_rank,
    const int block_size, const float* kv_cache_quant_scale) {
  // Each thread block is responsible for one token.
  const int64_t token_idx = blockIdx.x;
  const int64_t pos = positions[token_idx];

  const qk_t* cos_sin_ptr = rope_cos_sin_cache + pos * rot_dim;

  const int embed_dim = rot_dim / 2;

  // Q ROPE
  const int nq = num_q_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    int head_idx = i / embed_dim;
    int pair_idx = i % embed_dim;

    // NOTE: Would be nice to have interleaved sin/cos so we could just load
    // both at the same time.
    qk_t cos = VLLM_LDG(cos_sin_ptr + pair_idx);
    qk_t sin = VLLM_LDG(cos_sin_ptr + pair_idx + embed_dim);

    qk_t* q_pe_head_ptr =
        q_pe + token_idx * q_pe_stride_token + head_idx * q_pe_stride_head;

    int pair_idx_x, pair_idx_y;
    if constexpr (IS_NEOX) {
      // GPT-NeoX style rotary embedding.
      pair_idx_x = pair_idx;
      pair_idx_y = embed_dim + pair_idx;
    } else {
      // GPT-J style rotary embedding.
      pair_idx_x = pair_idx * 2;
      pair_idx_y = pair_idx * 2 + 1;
    }

    qk_t x_src = q_pe_head_ptr[pair_idx_x];
    qk_t y_src = q_pe_head_ptr[pair_idx_y];

    qk_t x_dst = x_src * cos - y_src * sin;
    qk_t y_dst = y_src * cos + x_src * sin;

    q_pe_head_ptr[pair_idx_x] = x_dst;
    q_pe_head_ptr[pair_idx_y] = y_dst;
  }

  const int64_t slot_idx = kv_cache_slot_mapping[token_idx];
  const int64_t block_idx = slot_idx / block_size;
  const int64_t entry_idx = slot_idx % block_size;

  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }

  // K with 1 HEAD
  for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
    int pair_idx = i;

    qk_t cos = VLLM_LDG(cos_sin_ptr + pair_idx);
    qk_t sin = VLLM_LDG(cos_sin_ptr + pair_idx + embed_dim);

    qk_t* k_pe_head_ptr = k_pe + token_idx * k_pe_stride;

    int pair_idx_x, pair_idx_y;
    if constexpr (IS_NEOX) {
      // GPT-NeoX style rotary embedding.
      pair_idx_x = pair_idx;
      pair_idx_y = embed_dim + pair_idx;
    } else {
      // GPT-J style rotary embedding.
      pair_idx_x = pair_idx * 2;
      pair_idx_y = pair_idx * 2 + 1;
    }

    qk_t x_src = k_pe_head_ptr[pair_idx_x];
    qk_t y_src = k_pe_head_ptr[pair_idx_y];

    qk_t x_dst = x_src * cos - y_src * sin;
    qk_t y_dst = y_src * cos + x_src * sin;

    k_pe_head_ptr[pair_idx_x] = x_dst;
    k_pe_head_ptr[pair_idx_y] = y_dst;

    // NOTE Why is this monster necessary?
    // When K is of type float16, the actual template replacement for
    // raw_kv_scalar_t with be u16. That's why it's used at the last moment
    // otherwise CUDA ALU would break.
    const raw_kv_scalar_t raw_x_value =
        *reinterpret_cast<const raw_kv_scalar_t*>(&x_dst);
    const raw_kv_scalar_t raw_y_value =
        *reinterpret_cast<const raw_kv_scalar_t*>(&y_dst);

    cache_t* kv_cache_ptr = kv_cache + block_idx * block_stride +
                            entry_idx * entry_stride + kv_lora_rank;

    // MLA Cache Store
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      kv_cache_ptr[pair_idx_x] = raw_x_value;
      kv_cache_ptr[pair_idx_y] = raw_y_value;
    } else {
      kv_cache_ptr[pair_idx_x] =
          fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(
              raw_x_value, *kv_cache_quant_scale);
      kv_cache_ptr[pair_idx_y] =
          fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(
              raw_y_value, *kv_cache_quant_scale);
    }
  }

  // NOPE
  for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
    const qk_t* src_ptr = kv_c + token_idx * kv_c_stride + i;
    const raw_kv_scalar_t src_value =
        *reinterpret_cast<const raw_kv_scalar_t*>(src_ptr);

    cache_t* kv_cache_ptr =
        kv_cache + block_idx * block_stride + entry_idx * entry_stride;

    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      kv_cache_ptr[i] = src_value;
    } else {
      kv_cache_ptr[i] = fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(
          src_value, *kv_cache_quant_scale);
    }
  }
}

}  // namespace vllm

#define CALL_CONCAT_AND_CACHE_MLA_ROPE_FUSED(RAW_KV_T, CACHE_T, KV_DTYPE)      \
  do {                                                                         \
    VLLM_DISPATCH_FLOATING_TYPES(q_pe.scalar_type(), "qk_scalar_type", [&] {   \
      using qk_t = scalar_t;                                                   \
      if (rope_is_neox) {                                                      \
        vllm::concat_and_cache_mla_rope_fused_kernel<qk_t, true, RAW_KV_T,     \
                                                     CACHE_T, KV_DTYPE>        \
            <<<grid, block, 0, stream>>>(                                      \
                positions.data_ptr<int64_t>(), q_pe.data_ptr<qk_t>(),          \
                k_pe.data_ptr<qk_t>(), kv_c.data_ptr<qk_t>(),                  \
                rope_cos_sin_cache.data_ptr<qk_t>(), rot_dim,                  \
                q_pe_stride_token, q_pe_stride_head, k_pe_stride, kv_c_stride, \
                num_q_heads, reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),  \
                kv_cache_slot_mapping.data_ptr<int64_t>(), block_stride,       \
                entry_stride, kv_lora_rank, block_size,                        \
                kv_cache_quant_scale.data_ptr<float>());                       \
      } else {                                                                 \
        vllm::concat_and_cache_mla_rope_fused_kernel<qk_t, false, RAW_KV_T,    \
                                                     CACHE_T, KV_DTYPE>        \
            <<<grid, block, 0, stream>>>(                                      \
                positions.data_ptr<int64_t>(), q_pe.data_ptr<qk_t>(),          \
                k_pe.data_ptr<qk_t>(), kv_c.data_ptr<qk_t>(),                  \
                rope_cos_sin_cache.data_ptr<qk_t>(), rot_dim,                  \
                q_pe_stride_token, q_pe_stride_head, k_pe_stride, kv_c_stride, \
                num_q_heads, reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),  \
                kv_cache_slot_mapping.data_ptr<int64_t>(), block_stride,       \
                entry_stride, kv_lora_rank, block_size,                        \
                kv_cache_quant_scale.data_ptr<float>());                       \
      }                                                                        \
    });                                                                        \
  } while (false)

// Executes RoPE on q_pe and k_pe, then writes k_pe and kv_c in the kv cache.
// q_pe and k_pe are modified in place.
// Replaces DeepseekScalingRotaryEmbedding.self.rotary_emb and
// concat_and_cache_mla.
void concat_and_cache_mla_rope_fused(
    torch::Tensor& positions,           // [num_tokens]
    torch::Tensor& q_pe,                // [num_tokens, num_q_heads, rot_dim]
    torch::Tensor& k_pe,                // [num_tokens, rot_dim]
    torch::Tensor& kv_c,                // [num_tokens, kv_lora_rank]
    torch::Tensor& rope_cos_sin_cache,  // [max_position, rot_dim]
    bool rope_is_neox,
    torch::Tensor&
        kv_cache_slot_mapping,  // [num_tokens] or [num_actual_tokens]
    torch::Tensor&
        kv_cache,  // [num_blocks, block_size, (kv_lora_rank + rot_dim)]
    const std::string& kv_cache_dtype, torch::Tensor& kv_cache_quant_scale) {
  const int64_t num_tokens = q_pe.size(0);

  const int num_q_heads = q_pe.size(1);
  const int rot_dim = q_pe.size(2);
  const int kv_lora_rank = kv_c.size(1);

  TORCH_CHECK(positions.size(0) >=
              num_tokens);  // CUDA Graphs might pad this for us
  TORCH_CHECK_EQ(positions.dim(), 1);
  TORCH_CHECK_EQ(positions.scalar_type(), c10::ScalarType::Long);

  TORCH_CHECK_EQ(q_pe.size(0), num_tokens);
  TORCH_CHECK_EQ(q_pe.size(1), num_q_heads);
  TORCH_CHECK_EQ(q_pe.size(2), rot_dim);
  TORCH_CHECK_EQ(q_pe.dim(), 3);

  TORCH_CHECK_EQ(k_pe.size(0), num_tokens);
  TORCH_CHECK_EQ(k_pe.size(1), rot_dim);
  TORCH_CHECK_EQ(k_pe.dim(), 2);
  TORCH_CHECK_EQ(k_pe.scalar_type(), q_pe.scalar_type());

  TORCH_CHECK_EQ(kv_c.size(0), num_tokens);
  TORCH_CHECK_EQ(kv_c.size(1), kv_lora_rank);
  TORCH_CHECK_EQ(kv_c.dim(), 2);
  TORCH_CHECK_EQ(kv_c.scalar_type(), q_pe.scalar_type());
  TORCH_CHECK_EQ(kv_c.dtype(), q_pe.dtype());

  TORCH_CHECK_EQ(rope_cos_sin_cache.size(1), rot_dim);
  TORCH_CHECK_EQ(rope_cos_sin_cache.scalar_type(), q_pe.scalar_type());

  TORCH_CHECK_EQ(kv_cache_slot_mapping.size(0), num_tokens);
  TORCH_CHECK_EQ(kv_cache_slot_mapping.scalar_type(), c10::ScalarType::Long);

  TORCH_CHECK_EQ(kv_cache.size(2), kv_lora_rank + rot_dim);
  TORCH_CHECK_EQ(kv_cache.dim(), 3);

  TORCH_CHECK_EQ(kv_cache_quant_scale.numel(), 1);
  TORCH_CHECK_EQ(kv_cache_quant_scale.scalar_type(), c10::ScalarType::Float);

  int64_t q_pe_stride_token = q_pe.stride(0);
  int64_t q_pe_stride_head = q_pe.stride(1);

  int64_t k_pe_stride = k_pe.stride(0);
  int64_t kv_c_stride = kv_c.stride(0);

  int block_size = kv_cache.size(1);

  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  int rope_block_size = std::min(num_q_heads * rot_dim / 2, 512);
  int mla_block_size = kv_lora_rank;
  int thread_block_size =
      std::min(std::max(rope_block_size, mla_block_size), 512);

  dim3 grid(num_tokens, 1, 1);
  dim3 block(thread_block_size, 1, 1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(positions));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype,
                             CALL_CONCAT_AND_CACHE_MLA_ROPE_FUSED);
}

// ---------------------------------------------------------------------------
// Fused MLA decode kernel: RoPE(q_pe) + copy q_nope → q_out (optional FP8),
//                          RoPE(k_pe from kv) + kv_c → kv_cache (optional FP8)
// ---------------------------------------------------------------------------

namespace vllm {

// q_pe is read-only: the in-place update is removed since the rotated values
// are only consumed via q_out and not needed afterward (saves 2
// stores/element).
template <typename qk_t, bool IS_NEOX, typename raw_kv_scalar_t,
          typename cache_t, Fp8KVCacheDataType kv_dt, bool DO_FP8_Q,
          int STATIC_KV_LORA_RANK = 0, int STATIC_ROT_DIM = 0,
          typename cos_sin_t = qk_t>
__global__ void fuse_mla_decode_rope_q_concat_kv_insert_kernel(
    const int64_t* __restrict__ positions,  // [num_tokens]
    const qk_t* __restrict__ q_nope,  // [num_tokens, num_q_heads, kv_lora_rank]
    const qk_t* __restrict__ q_pe,    // [num_tokens, num_q_heads, rot_dim]
                                      // (read-only)
    const qk_t* __restrict__ kv,      // [num_tokens, kv_lora_rank + rot_dim]
    const cos_sin_t* __restrict__ cos_sin_cache, const int rot_dim,
    const int kv_lora_rank, const int num_q_heads,
    const int64_t q_nope_stride_t, const int64_t q_nope_stride_h,
    const int64_t q_pe_stride_t, const int64_t q_pe_stride_h,
    const int64_t kv_stride_t, cache_t* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping, const int block_size,
    const int block_stride, const int entry_stride,
    const float* __restrict__ kv_scale,
    void* __restrict__ q_out,  // [num_tokens, num_q_heads,
                               // kv_lora_rank+rot_dim]
    const int64_t q_out_stride_t, const int64_t q_out_stride_h,
    const float* __restrict__ q_scale  // [1], only used when DO_FP8_Q
) {
  const int64_t token_idx = blockIdx.x;
  const int part_idx = blockIdx.y;
  const int64_t pos = positions[token_idx];

  constexpr bool kFixedShape = (STATIC_KV_LORA_RANK > 0 && STATIC_ROT_DIM > 0);
  const int rot_dim_val = kFixedShape ? STATIC_ROT_DIM : rot_dim;
  const int kv_lora_rank_val = kFixedShape ? STATIC_KV_LORA_RANK : kv_lora_rank;
  const int embed_dim = rot_dim_val / 2;

  using copy_vec_t = int4;
  constexpr int qk_vec_elts = sizeof(copy_vec_t) / sizeof(qk_t);
  constexpr int cache_vec_elts = sizeof(copy_vec_t) / sizeof(cache_t);

  const cos_sin_t* cos_sin_ptr = cos_sin_cache + pos * rot_dim_val;

  // Helper: write one element to q_out (BF16/FP16 or FP8).
  auto write_q_out = [&](int64_t offset, float val) {
    if constexpr (DO_FP8_Q) {
      reinterpret_cast<c10::Float8_e4m3fn*>(q_out)[offset] =
          vllm::scaled_fp8_conversion</*is_scale_inverted=*/false,
                                      c10::Float8_e4m3fn>(val, *q_scale);
    } else {
      reinterpret_cast<qk_t*>(q_out)[offset] = static_cast<qk_t>(val);
    }
  };

  // Wait for predecessor grid after all init/pointer setup so we overlap as
  // much work as possible with the previous kernel.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  // TRT-LLM-style partitioning:
  //   [0, num_q_heads)      : one block per Q head
  //   [num_q_heads]         : one block for K rope
  //   (num_q_heads, grid.y) : tiled blocks for kv_c cache insertion
  if (part_idx < num_q_heads) {
    const int head = part_idx;
    const qk_t* q_pe_head =
        q_pe + token_idx * q_pe_stride_t + (int64_t)head * q_pe_stride_h;
    const int64_t q_out_base =
        token_idx * q_out_stride_t + (int64_t)head * q_out_stride_h;

    // ---- Phase A: RoPE q_pe → q_out[.., L:] (q_pe read-only, no writeback)
    // ----
    for (int pair_idx = threadIdx.x; pair_idx < embed_dim;
         pair_idx += blockDim.x) {
      // Always do RoPE arithmetic in float32 to support float32 cos_sin_cache.
      const float cos_v = static_cast<float>(VLLM_LDG(cos_sin_ptr + pair_idx));
      const float sin_v =
          static_cast<float>(VLLM_LDG(cos_sin_ptr + pair_idx + embed_dim));

      int x_idx, y_idx;
      if constexpr (IS_NEOX) {
        x_idx = pair_idx;
        y_idx = embed_dim + pair_idx;
      } else {
        x_idx = pair_idx * 2;
        y_idx = pair_idx * 2 + 1;
      }

      const float x = static_cast<float>(q_pe_head[x_idx]);
      const float y = static_cast<float>(q_pe_head[y_idx]);
      const float xr = x * cos_v - y * sin_v;
      const float yr = y * cos_v + x * sin_v;
      // No in-place write back to q_pe.

      write_q_out(q_out_base + kv_lora_rank_val + x_idx, xr);
      write_q_out(q_out_base + kv_lora_rank_val + y_idx, yr);
    }

    // ---- Phase B: copy q_nope → q_out[.., :L] ----
    const qk_t* q_nope_src =
        q_nope + token_idx * q_nope_stride_t + (int64_t)head * q_nope_stride_h;
    qk_t* q_out_dst = reinterpret_cast<qk_t*>(q_out) + q_out_base;

    if constexpr (!DO_FP8_Q) {
      const bool can_vec_q =
          q_nope_stride_h == kv_lora_rank_val &&
          q_out_stride_h == kv_lora_rank_val + rot_dim_val &&
          kv_lora_rank_val % qk_vec_elts == 0 &&
          reinterpret_cast<uintptr_t>(q_nope_src) % alignof(copy_vec_t) == 0 &&
          reinterpret_cast<uintptr_t>(q_out_dst) % alignof(copy_vec_t) == 0;

      if (can_vec_q) {
        const int num_vec = kv_lora_rank_val / qk_vec_elts;
        const copy_vec_t* src_vec =
            reinterpret_cast<const copy_vec_t*>(q_nope_src);
        copy_vec_t* dst_vec = reinterpret_cast<copy_vec_t*>(q_out_dst);
        for (int vec_idx = threadIdx.x; vec_idx < num_vec;
             vec_idx += blockDim.x)
          dst_vec[vec_idx] = src_vec[vec_idx];
      } else {
        for (int l = threadIdx.x; l < kv_lora_rank_val; l += blockDim.x)
          q_out_dst[l] = q_nope_src[l];
      }
    } else {
      for (int l = threadIdx.x; l < kv_lora_rank_val; l += blockDim.x)
        write_q_out(q_out_base + l, static_cast<float>(q_nope_src[l]));
    }
    return;
  }

  // ---- KV cache insertion (skip padded tokens) ----
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int64_t entry_idx = slot_idx % block_size;
  cache_t* kv_cache_ptr =
      kv_cache + block_idx * block_stride + entry_idx * entry_stride;

  if (part_idx == num_q_heads) {
    // ---- Phase C: RoPE k_pe → kv_cache[L:] ----
    const qk_t* k_pe_src = kv + token_idx * kv_stride_t + kv_lora_rank_val;
    for (int pair_idx = threadIdx.x; pair_idx < embed_dim;
         pair_idx += blockDim.x) {
      const float cos_v = static_cast<float>(VLLM_LDG(cos_sin_ptr + pair_idx));
      const float sin_v =
          static_cast<float>(VLLM_LDG(cos_sin_ptr + pair_idx + embed_dim));

      int x_idx, y_idx;
      if constexpr (IS_NEOX) {
        x_idx = pair_idx;
        y_idx = embed_dim + pair_idx;
      } else {
        x_idx = pair_idx * 2;
        y_idx = pair_idx * 2 + 1;
      }

      const float x = static_cast<float>(k_pe_src[x_idx]);
      const float y = static_cast<float>(k_pe_src[y_idx]);
      const float xr_f = x * cos_v - y * sin_v;
      const float yr_f = y * cos_v + x * sin_v;
      const qk_t xr = static_cast<qk_t>(xr_f);
      const qk_t yr = static_cast<qk_t>(yr_f);
      const raw_kv_scalar_t raw_xr =
          *reinterpret_cast<const raw_kv_scalar_t*>(&xr);
      const raw_kv_scalar_t raw_yr =
          *reinterpret_cast<const raw_kv_scalar_t*>(&yr);

      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        kv_cache_ptr[kv_lora_rank_val + x_idx] = raw_xr;
        kv_cache_ptr[kv_lora_rank_val + y_idx] = raw_yr;
      } else {
        kv_cache_ptr[kv_lora_rank_val + x_idx] =
            fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(raw_xr,
                                                                 *kv_scale);
        kv_cache_ptr[kv_lora_rank_val + y_idx] =
            fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(raw_yr,
                                                                 *kv_scale);
      }
    }
    return;
  }

  // ---- Phase D: tiled kv_c copy → kv_cache[:L] ----
  constexpr int KV_COPY_TILE = 64;
  const int tile_idx = part_idx - num_q_heads - 1;
  const int tile_start = tile_idx * KV_COPY_TILE;
  const int tile_end = min(kv_lora_rank_val, tile_start + KV_COPY_TILE);
  if (tile_start >= kv_lora_rank_val) return;

  const qk_t* kv_c_src = kv + token_idx * kv_stride_t;
  if constexpr (kv_dt == Fp8KVCacheDataType::kAuto &&
                sizeof(cache_t) == sizeof(qk_t)) {
    const qk_t* tile_src = kv_c_src + tile_start;
    cache_t* tile_dst = kv_cache_ptr + tile_start;
    const int tile_len = tile_end - tile_start;
    const bool can_vec_kv =
        tile_start % cache_vec_elts == 0 && tile_len % cache_vec_elts == 0 &&
        reinterpret_cast<uintptr_t>(tile_src) % alignof(copy_vec_t) == 0 &&
        reinterpret_cast<uintptr_t>(tile_dst) % alignof(copy_vec_t) == 0;
    if (can_vec_kv) {
      const int num_vec = tile_len / cache_vec_elts;
      const copy_vec_t* src_vec = reinterpret_cast<const copy_vec_t*>(tile_src);
      copy_vec_t* dst_vec = reinterpret_cast<copy_vec_t*>(tile_dst);
      for (int v = threadIdx.x; v < num_vec; v += blockDim.x)
        dst_vec[v] = src_vec[v];
    } else {
      for (int i = tile_start + threadIdx.x; i < tile_end; i += blockDim.x)
        kv_cache_ptr[i] = *reinterpret_cast<const cache_t*>(&kv_c_src[i]);
    }
  } else {
    for (int i = tile_start + threadIdx.x; i < tile_end; i += blockDim.x) {
      const qk_t val = kv_c_src[i];
      const raw_kv_scalar_t raw_val =
          *reinterpret_cast<const raw_kv_scalar_t*>(&val);
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto)
        kv_cache_ptr[i] = raw_val;
      else
        kv_cache_ptr[i] = fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(
            raw_val, *kv_scale);
    }
  }
}

}  // namespace vllm

// Inner helper: launch the kernel for a fixed (qk_t, cos_sin_t) pair.
// Uses cudaLaunchKernelEx so PDL (programmatic stream serialization) can be
// enabled on SM90+ without a separate code path.
#define LAUNCH_FUSE_MLA_DECODE_ROPE_Q_CONCAT_KV_INSERT(                       \
    QK_T, COS_SIN_T, IS_NEOX, RAW_KV_T, CACHE_T, KV_DTYPE, DO_FP8_Q,          \
    STATIC_KV_LORA_RANK, STATIC_ROT_DIM, COS_SIN_PTR)                         \
  do {                                                                        \
    auto* _kernel = &vllm::fuse_mla_decode_rope_q_concat_kv_insert_kernel<    \
        QK_T, IS_NEOX, RAW_KV_T, CACHE_T, KV_DTYPE, DO_FP8_Q,                 \
        STATIC_KV_LORA_RANK, STATIC_ROT_DIM, COS_SIN_T>;                      \
    cudaLaunchConfig_t _cfg;                                                  \
    _cfg.gridDim = grid;                                                      \
    _cfg.blockDim = block;                                                    \
    _cfg.dynamicSmemBytes = 0;                                                \
    _cfg.stream = stream;                                                     \
    cudaLaunchAttribute _attr[1];                                             \
    _attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;         \
    _attr[0].val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0; \
    _cfg.numAttrs = 1;                                                        \
    _cfg.attrs = _attr;                                                       \
    cudaLaunchKernelEx(                                                       \
        &_cfg, _kernel, positions.data_ptr<int64_t>(),                        \
        q_nope.data_ptr<QK_T>(), q_pe.data_ptr<QK_T>(), kv.data_ptr<QK_T>(),  \
        COS_SIN_PTR, rot_dim, kv_lora_rank, num_q_heads, q_nope_stride_t,     \
        q_nope_stride_h, q_pe_stride_t, q_pe_stride_h, kv_stride_t,           \
        reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                      \
        slot_mapping.data_ptr<int64_t>(), block_size, block_stride,           \
        entry_stride, kv_scale.data_ptr<float>(), q_out.data_ptr(),           \
        q_out_stride_t, q_out_stride_h, q_scale_ptr);                         \
  } while (false)

#define LAUNCH_FUSE_MLA_DECODE_SPECIALIZED_OR_GENERIC(                        \
    QK_T, COS_SIN_T, IS_NEOX, RAW_KV_T, CACHE_T, KV_DTYPE, DO_FP8_Q,          \
    COS_SIN_PTR)                                                              \
  do {                                                                        \
    if (use_specialized_shape) {                                              \
      LAUNCH_FUSE_MLA_DECODE_ROPE_Q_CONCAT_KV_INSERT(                         \
          QK_T, COS_SIN_T, IS_NEOX, RAW_KV_T, CACHE_T, KV_DTYPE, DO_FP8_Q,    \
          512, 64, COS_SIN_PTR);                                              \
    } else {                                                                  \
      LAUNCH_FUSE_MLA_DECODE_ROPE_Q_CONCAT_KV_INSERT(                         \
          QK_T, COS_SIN_T, IS_NEOX, RAW_KV_T, CACHE_T, KV_DTYPE, DO_FP8_Q, 0, \
          0, COS_SIN_PTR);                                                    \
    }                                                                         \
  } while (false)

// Dispatch macro: expands KV dtype × qk_t × cos_sin_t × (IS_NEOX × DO_FP8_Q).
// cos_sin_cache may be float32 (flashinfer path) or the same dtype as q_pe.
#define CALL_FUSE_MLA_DECODE_ROPE_Q_CONCAT_KV_INSERT(RAW_KV_T, CACHE_T,      \
                                                     KV_DTYPE)               \
  do {                                                                       \
    VLLM_DISPATCH_FLOATING_TYPES(q_pe.scalar_type(), "qk_scalar_type", [&] { \
      using qk_t = scalar_t;                                                 \
      const bool cs_is_f32 =                                                 \
          (cos_sin_cache.scalar_type() == c10::ScalarType::Float);           \
      const bool use_specialized_shape =                                     \
          (kv_lora_rank == 512 && rot_dim == 64);                            \
      if (rope_is_neox) {                                                    \
        if (do_fp8_q) {                                                      \
          if (cs_is_f32)                                                     \
            LAUNCH_FUSE_MLA_DECODE_SPECIALIZED_OR_GENERIC(                   \
                qk_t, float, true, RAW_KV_T, CACHE_T, KV_DTYPE, true,        \
                cos_sin_cache.data_ptr<float>());                            \
          else                                                               \
            LAUNCH_FUSE_MLA_DECODE_SPECIALIZED_OR_GENERIC(                   \
                qk_t, qk_t, true, RAW_KV_T, CACHE_T, KV_DTYPE, true,         \
                cos_sin_cache.data_ptr<qk_t>());                             \
        } else {                                                             \
          if (cs_is_f32)                                                     \
            LAUNCH_FUSE_MLA_DECODE_SPECIALIZED_OR_GENERIC(                   \
                qk_t, float, true, RAW_KV_T, CACHE_T, KV_DTYPE, false,       \
                cos_sin_cache.data_ptr<float>());                            \
          else                                                               \
            LAUNCH_FUSE_MLA_DECODE_SPECIALIZED_OR_GENERIC(                   \
                qk_t, qk_t, true, RAW_KV_T, CACHE_T, KV_DTYPE, false,        \
                cos_sin_cache.data_ptr<qk_t>());                             \
        }                                                                    \
      } else {                                                               \
        if (do_fp8_q) {                                                      \
          if (cs_is_f32)                                                     \
            LAUNCH_FUSE_MLA_DECODE_SPECIALIZED_OR_GENERIC(                   \
                qk_t, float, false, RAW_KV_T, CACHE_T, KV_DTYPE, true,       \
                cos_sin_cache.data_ptr<float>());                            \
          else                                                               \
            LAUNCH_FUSE_MLA_DECODE_SPECIALIZED_OR_GENERIC(                   \
                qk_t, qk_t, false, RAW_KV_T, CACHE_T, KV_DTYPE, true,        \
                cos_sin_cache.data_ptr<qk_t>());                             \
        } else {                                                             \
          if (cs_is_f32)                                                     \
            LAUNCH_FUSE_MLA_DECODE_SPECIALIZED_OR_GENERIC(                   \
                qk_t, float, false, RAW_KV_T, CACHE_T, KV_DTYPE, false,      \
                cos_sin_cache.data_ptr<float>());                            \
          else                                                               \
            LAUNCH_FUSE_MLA_DECODE_SPECIALIZED_OR_GENERIC(                   \
                qk_t, qk_t, false, RAW_KV_T, CACHE_T, KV_DTYPE, false,       \
                cos_sin_cache.data_ptr<qk_t>());                             \
        }                                                                    \
      }                                                                      \
    });                                                                      \
  } while (false)

// Fused MLA decode forward:
//   1. RoPE(q_pe) + copy concat(q_nope, rope(q_pe)) → q_out  [q_pe read-only]
//   2. RoPE(k_pe from kv) + copy [kv_c | rope(k_pe)] → kv_cache
void fuse_mla_decode_rope_q_concat_kv_insert(
    torch::Tensor& positions,  // [num_tokens]
    torch::Tensor& q_nope,     // [num_tokens, num_q_heads, kv_lora_rank]
    torch::Tensor& q_pe,       // [num_tokens, num_q_heads, rot_dim]  (mutated)
    torch::Tensor& kv,         // [num_tokens, kv_lora_rank + rot_dim]
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool rope_is_neox,
    torch::Tensor& slot_mapping,  // [num_tokens]
    torch::Tensor&
        kv_cache,  // [num_blocks, block_size, kv_lora_rank+rot_dim] (mutated)
    const std::string& kv_cache_dtype,
    torch::Tensor& kv_scale,                      // [1] float32
    const c10::optional<torch::Tensor>& q_scale,  // [1] float32 or nullopt
    torch::Tensor&
        q_out  // [num_tokens, num_q_heads, kv_lora_rank+rot_dim] (mutated)
) {
  const int64_t num_tokens = q_pe.size(0);
  const int num_q_heads = q_pe.size(1);
  const int rot_dim = q_pe.size(2);
  const int kv_lora_rank = kv.size(1) - rot_dim;

  TORCH_CHECK(positions.size(0) >= num_tokens, "positions too short");
  TORCH_CHECK_EQ(positions.dim(), 1);
  TORCH_CHECK_EQ(positions.scalar_type(), c10::ScalarType::Long);

  TORCH_CHECK_EQ(q_nope.dim(), 3);
  TORCH_CHECK_EQ(q_nope.size(0), num_tokens);
  TORCH_CHECK_EQ(q_nope.size(1), num_q_heads);
  TORCH_CHECK_EQ(q_nope.size(2), kv_lora_rank);
  TORCH_CHECK_EQ(q_nope.scalar_type(), q_pe.scalar_type());

  TORCH_CHECK_EQ(q_pe.dim(), 3);
  TORCH_CHECK_EQ(q_pe.size(2), rot_dim);

  TORCH_CHECK_EQ(kv.dim(), 2);
  TORCH_CHECK_EQ(kv.size(0), num_tokens);
  TORCH_CHECK_EQ(kv.size(1), kv_lora_rank + rot_dim);
  TORCH_CHECK_EQ(kv.scalar_type(), q_pe.scalar_type());

  TORCH_CHECK_EQ(cos_sin_cache.size(1), rot_dim);
  TORCH_CHECK(cos_sin_cache.scalar_type() == c10::ScalarType::Float ||
                  cos_sin_cache.scalar_type() == q_pe.scalar_type(),
              "cos_sin_cache must be float32 or match q_pe dtype");

  TORCH_CHECK(slot_mapping.size(0) >= num_tokens, "slot_mapping too short");
  TORCH_CHECK_EQ(slot_mapping.scalar_type(), c10::ScalarType::Long);

  TORCH_CHECK_EQ(kv_cache.dim(), 3);
  TORCH_CHECK_EQ(kv_cache.size(2), kv_lora_rank + rot_dim);

  TORCH_CHECK_EQ(kv_scale.numel(), 1);
  TORCH_CHECK_EQ(kv_scale.scalar_type(), c10::ScalarType::Float);

  TORCH_CHECK_EQ(q_out.dim(), 3);
  TORCH_CHECK_EQ(q_out.size(0), num_tokens);
  TORCH_CHECK_EQ(q_out.size(1), num_q_heads);
  TORCH_CHECK_EQ(q_out.size(2), kv_lora_rank + rot_dim);

  const bool do_fp8_q = q_scale.has_value();
  const float* q_scale_ptr =
      do_fp8_q ? q_scale.value().data_ptr<float>() : nullptr;

  const int64_t q_nope_stride_t = q_nope.stride(0);
  const int64_t q_nope_stride_h = q_nope.stride(1);
  const int64_t q_pe_stride_t = q_pe.stride(0);
  const int64_t q_pe_stride_h = q_pe.stride(1);
  const int64_t kv_stride_t = kv.stride(0);
  const int64_t q_out_stride_t = q_out.stride(0);
  const int64_t q_out_stride_h = q_out.stride(1);

  const int block_size = kv_cache.size(1);
  const int block_stride = kv_cache.stride(0);
  const int entry_stride = kv_cache.stride(1);

  const bool use_specialized_shape = (kv_lora_rank == 512 && rot_dim == 64);
  const int num_kv_tiles = (kv_lora_rank + 63) / 64;
  const int thread_block_size = use_specialized_shape ? 128 : 256;
  dim3 grid(num_tokens, num_q_heads + 1 + num_kv_tiles, 1);
  dim3 block(thread_block_size, 1, 1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(positions));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Auto-enable PDL on SM90+.
  int sm_major = 0;
  cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor,
                         static_cast<int>(q_pe.get_device()));
  bool const enable_pdl = (sm_major >= 9);

  DISPATCH_BY_KV_CACHE_DTYPE(kv.dtype(), kv_cache_dtype,
                             CALL_FUSE_MLA_DECODE_ROPE_Q_CONCAT_KV_INSERT);
}
