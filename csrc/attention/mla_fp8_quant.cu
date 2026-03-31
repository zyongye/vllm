// CUDA kernel for fused FP8 quantization of Q, K, V tensors (MLA non-absorption
// mode). Replaces the Triton kernel fused_fp8_quantize in mla_attention.py.
// Fixed for DeepSeek V3 shapes: QK_NOPE=128, QK_ROPE=64, V=128, heads=128.

#include "dispatch_utils.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <cuda_fp8.h>
#include <mutex>

namespace vllm {

// ── 128-bit vector load type ──────────────────────────────────────────────────
template <typename T>
struct MlaVecType;

template <>
struct MlaVecType<c10::BFloat16> {
  using Type = uint4;
};

template <>
struct MlaVecType<c10::Half> {
  using Type = uint4;
};

// ── quantCopyN: N×128-bit loads (issued upfront for ILP) → conversion → stores
// Issuing N independent loads before any conversion allows the hardware to
// send all N requests to HBM simultaneously, hiding latency more effectively.
// Uses __nv_fp8x4_e4m3 whose constructor maps to cvt.rn.satfinite.e4m3x4.f32x4
// on SM90+/SM100 — 4× fewer conversion instructions than element-by-element.
template <typename T, int ELTS_PER_VEC, int N_VEC>
__device__ __forceinline__ void mlaQuantCopyN(__nv_fp8_e4m3* __restrict__ dst,
                                               const T* __restrict__ src,
                                               float scale) {
  static_assert(ELTS_PER_VEC == 8, "mlaQuantCopyN requires ELTS_PER_VEC == 8");
  using VecT = typename MlaVecType<T>::Type;

  // Issue all N_VEC loads upfront (all independent → hardware pipelines them).
  VecT loaded[N_VEC];
  #pragma unroll
  for (int v = 0; v < N_VEC; ++v)
    loaded[v] = __ldg(reinterpret_cast<const VecT*>(src + v * ELTS_PER_VEC));

  // Convert and store after all loads are issued.
  // float4{f0,f1,f2,f3} constructor: f0 in byte 0, f3 in byte 3 of __x.
  // Merged uint64: lo.__x in low 32 bits → elts[0] in dst byte 0.
  #pragma unroll
  for (int v = 0; v < N_VEC; ++v) {
    const T* elts = reinterpret_cast<const T*>(&loaded[v]);
    __nv_fp8x4_e4m3 lo(float4{
        static_cast<float>(elts[0]) * scale, static_cast<float>(elts[1]) * scale,
        static_cast<float>(elts[2]) * scale, static_cast<float>(elts[3]) * scale});
    __nv_fp8x4_e4m3 hi(float4{
        static_cast<float>(elts[4]) * scale, static_cast<float>(elts[5]) * scale,
        static_cast<float>(elts[6]) * scale, static_cast<float>(elts[7]) * scale});
    // Merge two uint32_t into one uint64_t → single st.global.b64.
    *reinterpret_cast<uint64_t*>(dst + v * ELTS_PER_VEC) =
        (static_cast<uint64_t>(hi.__x) << 32) | lo.__x;
  }
}

// ── Main fused QKV quantization kernel ───────────────────────────────────────
// Grid  : (ceil(total_kv_len / QK_TOKENS_PER_BLOCK), 1, num_heads)
// Block : BLOCK_SIZE threads
//
// Each thread handles N_VEC_PER_THREAD consecutive 128-bit vec-slots per token,
// issuing all N_VEC loads upfront before any conversion (better MLP hiding).
//
// ABSORPTION_MODE=false: quantizes Q (bounded by total_q_len),
//                        K (bounded by total_kv_len, contiguous),
//                        V (bounded by total_kv_len, non-contiguous src).
//
// V src layout : [n_tokens, num_heads, QK_NOPE_HEAD_DIM + V_HEAD_DIM]
//   i.e. stride(0) = num_heads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)
// V dst layout : [n_tokens, num_heads, V_HEAD_DIM]  (contiguous)
template <typename T, int BLOCK_SIZE, int QK_NOPE_HEAD_DIM, int QK_ROPE_HEAD_DIM,
          int V_HEAD_DIM, bool ABSORPTION_MODE, int N_VEC_PER_THREAD = 2>
__global__ void quantizeCopyInputToFp8Kernel(
    T const* q_buf, __nv_fp8_e4m3* quant_q_buf, T const* k_buf,
    __nv_fp8_e4m3* quant_k_buf, T const* v_buf, __nv_fp8_e4m3* quant_v_buf,
    int total_q_len, int total_kv_len, float const* quant_scale_qkv_ptr,
    float* bmm1_scale, float* bmm2_scale, float const* quant_scale_o,
    float const* dequant_scale_q, float const* dequant_scale_kv,
    float host_bmm1_scale) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  constexpr auto BYTES_PER_ELT = sizeof(T);
  constexpr auto BYTES_PER_LOAD = 16;
  constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
  constexpr auto QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;

  static_assert((QK_HEAD_DIM * BYTES_PER_ELT) % BYTES_PER_LOAD == 0,
                "QK head size needs to be multiple of 16 bytes.");
  static_assert((V_HEAD_DIM * BYTES_PER_ELT) % BYTES_PER_LOAD == 0,
                "V head size needs to be multiple of 16 bytes.");

  constexpr auto QK_VECS_PER_HEAD = QK_HEAD_DIM * BYTES_PER_ELT / BYTES_PER_LOAD;
  constexpr auto V_VECS_PER_HEAD = V_HEAD_DIM * BYTES_PER_ELT / BYTES_PER_LOAD;

  // Each thread covers N_VEC_PER_THREAD consecutive vec-slots.
  static_assert(QK_VECS_PER_HEAD % N_VEC_PER_THREAD == 0,
                "QK_VECS_PER_HEAD must be divisible by N_VEC_PER_THREAD.");
  static_assert(ABSORPTION_MODE || (V_VECS_PER_HEAD % N_VEC_PER_THREAD == 0),
                "V_VECS_PER_HEAD must be divisible by N_VEC_PER_THREAD.");

  constexpr auto QK_THREADS_PER_HEAD = QK_VECS_PER_HEAD / N_VEC_PER_THREAD;
  constexpr auto V_THREADS_PER_HEAD  = V_VECS_PER_HEAD  / N_VEC_PER_THREAD;

  static_assert(BLOCK_SIZE % QK_THREADS_PER_HEAD == 0,
                "Kernel block should be able to handle entire heads.");
  static_assert(ABSORPTION_MODE || (BLOCK_SIZE % V_THREADS_PER_HEAD == 0),
                "Kernel block should be able to handle entire V heads.");

  constexpr auto QK_TOKENS_PER_BLOCK = BLOCK_SIZE / QK_THREADS_PER_HEAD;
  constexpr auto V_TOKENS_PER_BLOCK  = BLOCK_SIZE / V_THREADS_PER_HEAD;

  size_t const head_idx = blockIdx.z;
  size_t const head_num = gridDim.z;

  // Compute bmm scales once, in a single thread.
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
      threadIdx.x == 0) {
    float dequant_scale_q_val = dequant_scale_q ? dequant_scale_q[0] : 1.f;
    float dequant_scale_kv_val = dequant_scale_kv ? dequant_scale_kv[0] : 1.f;
    float quant_scale_o_val = quant_scale_o ? quant_scale_o[0] : 1.f;
    if (bmm1_scale) {
      constexpr float kLog2e = 1.4426950408889634074f;
      float bmm1_scale_val =
          dequant_scale_q_val * dequant_scale_kv_val * host_bmm1_scale;
      bmm1_scale[0] = bmm1_scale_val;
      bmm1_scale[1] = bmm1_scale_val * kLog2e;
    }
    if (bmm2_scale) {
      bmm2_scale[0] = quant_scale_o_val * dequant_scale_kv_val;
    }
  }

  // Base dim index for this thread (start of N_VEC_PER_THREAD consecutive vecs).
  size_t const qk_base_dim =
      (threadIdx.x % QK_THREADS_PER_HEAD) * N_VEC_PER_THREAD * ELTS_PER_VEC;
  size_t const v_base_dim =
      (threadIdx.x % V_THREADS_PER_HEAD) * N_VEC_PER_THREAD * ELTS_PER_VEC;

  size_t const q_len_loop_end = size_t((total_q_len + QK_TOKENS_PER_BLOCK - 1) /
                                       QK_TOKENS_PER_BLOCK) *
                                QK_TOKENS_PER_BLOCK;
  size_t const k_len_loop_end = size_t((total_kv_len + QK_TOKENS_PER_BLOCK - 1) /
                                       QK_TOKENS_PER_BLOCK) *
                                QK_TOKENS_PER_BLOCK;
  size_t const v_len_loop_end =
      size_t((total_kv_len + V_TOKENS_PER_BLOCK - 1) / V_TOKENS_PER_BLOCK) *
      V_TOKENS_PER_BLOCK;

  float quant_scale_qkv_val =
      quant_scale_qkv_ptr ? quant_scale_qkv_ptr[0] : 1.f;

  // Quantize Q (both src and dst are contiguous).
  for (int q_token_idx =
           (threadIdx.x / QK_THREADS_PER_HEAD) + blockIdx.x * QK_TOKENS_PER_BLOCK;
       q_token_idx < q_len_loop_end;
       q_token_idx += QK_TOKENS_PER_BLOCK * gridDim.x) {
    if (q_token_idx < total_q_len) {
      auto const base_q = static_cast<size_t>(q_token_idx) * QK_HEAD_DIM *
                              head_num +
                          head_idx * QK_HEAD_DIM + qk_base_dim;
      mlaQuantCopyN<T, ELTS_PER_VEC, N_VEC_PER_THREAD>(
          quant_q_buf + base_q, &q_buf[base_q], quant_scale_qkv_val);
    }
  }

  // Quantize K and V only in non-absorption mode.
  if constexpr (!ABSORPTION_MODE) {
    // Quantize K (contiguous).
    for (int k_token_idx = (threadIdx.x / QK_THREADS_PER_HEAD) +
                           blockIdx.x * QK_TOKENS_PER_BLOCK;
         k_token_idx < k_len_loop_end;
         k_token_idx += QK_TOKENS_PER_BLOCK * gridDim.x) {
      if (k_token_idx < total_kv_len) {
        auto const base_k = static_cast<size_t>(k_token_idx) * QK_HEAD_DIM *
                                head_num +
                            head_idx * QK_HEAD_DIM + qk_base_dim;
        mlaQuantCopyN<T, ELTS_PER_VEC, N_VEC_PER_THREAD>(
            quant_k_buf + base_k, &k_buf[base_k], quant_scale_qkv_val);
      }
    }

    // Quantize V: src is non-contiguous.
    // v comes from kv_nope[:, :, QK_NOPE:]; its strides are:
    //   stride(0) = (QK_NOPE + V_HEAD_DIM) * num_heads  (token stride)
    //   stride(1) = (QK_NOPE + V_HEAD_DIM)              (head stride within token)
    //   stride(2) = 1
    // dst is contiguous: [n_tokens, num_heads, V_HEAD_DIM].
    constexpr size_t src_v_head_stride = QK_NOPE_HEAD_DIM + V_HEAD_DIM;
    size_t const src_v_token_stride = src_v_head_stride * head_num;
    for (int v_token_idx = (threadIdx.x / V_THREADS_PER_HEAD) +
                           blockIdx.x * V_TOKENS_PER_BLOCK;
         v_token_idx < v_len_loop_end;
         v_token_idx += V_TOKENS_PER_BLOCK * gridDim.x) {
      if (v_token_idx < total_kv_len) {
        auto const src_v = static_cast<size_t>(v_token_idx) *
                               src_v_token_stride +
                           head_idx * src_v_head_stride + v_base_dim;
        auto const dst_v = static_cast<size_t>(v_token_idx) * V_HEAD_DIM *
                               head_num +
                           head_idx * V_HEAD_DIM + v_base_dim;
        mlaQuantCopyN<T, ELTS_PER_VEC, N_VEC_PER_THREAD>(
            quant_v_buf + dst_v, &v_buf[src_v], quant_scale_qkv_val);
      }
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

}  // namespace vllm

// ── PDL helpers ──────────────────────────────────────────────────────────────
// Programmatic Dependent Launch (SM90+): allows the next kernel to start
// before this one finishes, overlapping tail of this kernel with prologue of
// the next. Controlled by VLLM_ENABLE_PDL=1 environment variable.
namespace {

inline int mlaSMVersion() {
  auto* props = at::cuda::getCurrentDeviceProperties();
  return props->major * 10 + props->minor;
}

inline bool mlaEnvEnablePDL() {
  static std::once_flag flag;
  static bool enablePDL = false;
  std::call_once(flag, [&]() {
    if (mlaSMVersion() >= 90) {
      const char* env = std::getenv("VLLM_ENABLE_PDL");
      enablePDL = (env != nullptr) && (env[0] == '1') && (env[1] == '\0');
    }
  });
  return enablePDL;
}

}  // namespace

// ── C++ wrapper (global scope, as required by torch_bindings.cpp) ─────────────
// Fixed DeepSeek V3 MLA non-absorption shapes:
//   QK_NOPE=128, QK_ROPE=64, V=128, BLOCK_SIZE=192, N_VEC_PER_THREAD=2
//
// BLOCK_SIZE=192 with N_VEC_PER_THREAD=2:
//   QK_THREADS_PER_HEAD = (128+64)*2/16 / 2 = 24/2 = 12; 192 % 12 == 0 ✓
//   V_THREADS_PER_HEAD  = 128*2/16 / 2 = 16/2 = 8;       192 % 8 == 0 ✓
//   QK_TOKENS_PER_BLOCK = 192/12 = 16 (same grid as N_VEC=1, BLOCK_SIZE=384)
//   Each thread issues 2 independent ld.global.v4.b32 before any conversion,
//   doubling memory-level parallelism vs single-load design.
void mla_fp8_quantize_qkv(torch::Tensor const& q,   // [n_q,  H, 192] bf16/fp16
                           torch::Tensor& q_out,     // [n_q,  H, 192] fp8
                           torch::Tensor const& k,   // [n_kv, H, 192] bf16/fp16
                           torch::Tensor& k_out,     // [n_kv, H, 192] fp8
                           torch::Tensor const& v,   // [n_kv, H, 128] non-cont ok
                           torch::Tensor& v_out,     // [n_kv, H, 128] fp8
                           double scale) {
  constexpr int QK_NOPE = 128;
  constexpr int QK_ROPE = 64;
  constexpr int V_DIM = 128;
  constexpr bool ABSORP = false;
  constexpr int BLOCK_SZ = 192;
  constexpr int N_VEC = 2;
  // QK_TOKENS_PER_BLOCK = 192 / (24/2) = 192/12 = 16
  constexpr int QK_TOKENS_PER_BLOCK = 16;

  int total_q = q.size(0);
  int total_kv = k.size(0);
  int num_heads = q.size(1);

  if (total_kv == 0) return;

  int grid_x = (total_kv + QK_TOKENS_PER_BLOCK - 1) / QK_TOKENS_PER_BLOCK;
  dim3 grid(grid_x, 1, num_heads);
  dim3 block(BLOCK_SZ);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  float scale_f = static_cast<float>(scale);

  // Use cudaLaunchKernelEx with PDL (Programmatic Dependent Launch) attribute.
  // When VLLM_ENABLE_PDL=1 and SM >= 90, the scheduler may overlap the tail of
  // this kernel with the prologue of the next, hiding tail-latency on wide grids.
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = mlaEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;

  VLLM_DISPATCH_HALF_TYPES(q.scalar_type(), "mla_fp8_quantize_qkv", [&] {
    cudaLaunchKernelEx(
        &config,
        vllm::quantizeCopyInputToFp8Kernel<scalar_t, BLOCK_SZ, QK_NOPE,
                                           QK_ROPE, V_DIM, ABSORP, N_VEC>,
        q.data_ptr<scalar_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(q_out.data_ptr()),
        k.data_ptr<scalar_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(k_out.data_ptr()),
        v.data_ptr<scalar_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(v_out.data_ptr()), total_q, total_kv,
        &scale_f,
        /*bmm1_scale=*/nullptr,
        /*bmm2_scale=*/nullptr,
        /*quant_scale_o=*/nullptr,
        /*dequant_scale_q=*/nullptr,
        /*dequant_scale_kv=*/nullptr,
        /*host_bmm1_scale=*/1.0f);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
