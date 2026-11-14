#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "nvfp4-mlp.cuh"
#include "nvfp4-quantize.cuh"

#if defined(GGML_CUDA_FP8_CUTLASS) && CUDART_VERSION >= 12080
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <algorithm>

namespace {

static bool use_cutlass_down() {
    static const bool enabled = [] { const char * value = std::getenv("GGML_CUDA_NVFP4_MLP_DOWN"); return value && std::atoi(value) != 0; }();
    return enabled;
}

static __host__ __device__ int64_t sf_offset(int row, int subblock, int blocks) {
    return int64_t(row / 128) * (blocks / 4) * 512 + int64_t(subblock / 4) * 512 +
            (row % 32) * 16 + ((row % 128) / 32) * 4 + subblock % 4;
}

#if defined(BLACKWELL_MMA_AVAILABLE)
static __device__ __forceinline__ void quantize_calibrated(
        const float vals[16], float inv_global, uint8_t & sf, uint32_t & q0, uint32_t & q1) {
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) { amax = fmaxf(amax, fabsf(vals[j])); }
    sf = ggml_cuda_fp32_to_ue4m3((amax/6.0f)*inv_global);
    const float local = 2.0f*ggml_cuda_ue4m3_to_fp32(sf);
    const float scale = local > 0.0f ? inv_global/local : 0.0f;
    const __nv_fp4x4_e2m1 a(make_float4(vals[0]*scale, vals[8]*scale, vals[1]*scale, vals[9]*scale));
    const __nv_fp4x4_e2m1 b(make_float4(vals[2]*scale, vals[10]*scale, vals[3]*scale, vals[11]*scale));
    const __nv_fp4x4_e2m1 c(make_float4(vals[4]*scale, vals[12]*scale, vals[5]*scale, vals[13]*scale));
    const __nv_fp4x4_e2m1 d(make_float4(vals[6]*scale, vals[14]*scale, vals[7]*scale, vals[15]*scale));
    q0 = uint32_t(a.__x) | (uint32_t(b.__x) << 16);
    q1 = uint32_t(c.__x) | (uint32_t(d.__x) << 16);
}

static __device__ void store_packed(block_fp4_mmq * output, uint32_t q0, uint32_t q1, uint8_t sf, int row, int sb, int rows,
        uint8_t * direct_values = nullptr, uint8_t * direct_scales = nullptr, int k = 0) {
    if (direct_values) {
        // Emit the consumer's packed values and swizzled scales directly. The native
        // MMQ block format is only needed when MMQ is the actual consumer.
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = j < 2 ? q0 : q1;
            const int shift = 16*(j%2);
            const uint32_t a = (q >> shift) & 255, b = (q >> (shift + 8)) & 255;
            lo |= ((a & 15) | ((b & 15) << 4)) << (8*j);
            hi |= ((a >> 4) | (b & 240)) << (8*j);
        }
        auto * dst = reinterpret_cast<uint32_t *>(direct_values + int64_t(row)*k/2 + sb*8);
        dst[0] = lo;
        dst[1] = hi;
        direct_scales[sf_offset(row, sb, k/16)] = sf;
        return;
    }
    auto * block = output + int64_t(sb/16)*rows + row;
    auto * values = reinterpret_cast<uint32_t *>(block->qs);
    values[2*(sb%16)] = q0;
    values[2*(sb%16) + 1] = q1;
    reinterpret_cast<uint8_t *>(block->d4)[sb%16] = sf;
}
#endif

template <typename T>
static __global__ void calibrated_pack(
        const T * input, const float * up_calibration, const float * gate_calibration,
        block_fp4_mmq * output, float * scales, int k, int rows, bool glu,
        uint8_t * direct_values = nullptr, uint8_t * direct_scales = nullptr) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t index = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int subblocks = k/16;
    if (index >= int64_t(rows)*subblocks) { return; }
    const int row = index/subblocks, sb = index%subblocks;
    const float global = gate_calibration ? fminf(*up_calibration, *gate_calibration) : *up_calibration;
    float values[16];
    union alignas(16) vector_block {
        T scalar[16];
        uint4 vector[sizeof(T)];
    } up, gate_values;
    const int64_t base = int64_t(row)*k*(glu ? 2 : 1) + sb*16;
    if (direct_values) {
#pragma unroll
        for (int j = 0; j < int(sizeof(T)); ++j) {
            up.vector[j] = reinterpret_cast<const uint4 *>(input + base)[j];
            if (glu) { gate_values.vector[j] = reinterpret_cast<const uint4 *>(input + base + k)[j]; }
        }
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float v = direct_values ? float(up.scalar[j]) : float(input[base + j]);
        const float gate = glu ? (direct_values ? float(gate_values.scalar[j]) : float(input[base + k + j])) : 0.0f;
        values[j] = glu ? v*(gate/(1.0f + expf(-gate))) : v;
    }
    uint32_t q0, q1;
    uint8_t sf;
    quantize_calibrated(values, 1.0f/global, sf, q0, q1);
    store_packed(output, q0, q1, sf, row, sb, rows, direct_values, direct_scales, k);
    if (sb == 0) { scales[row] = global; }
#else
    GGML_UNUSED_VARS(input, up_calibration, gate_calibration, output, scales, k, rows, glu);
    NO_DEVICE_CODE;
#endif
}

static __global__ void repack_weight(const block_nvfp4 * src, uint8_t * dst, uint8_t * scales, int k, int n) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= int64_t((n + 127) / 128 * 128) * (k / 16)) {
        return;
    }
    const int row = index / (k / 16), subblock = index % (k / 16);
    if (row >= n) {
        scales[sf_offset(row, subblock, k / 16)] = 0;
        return;
    }
    const auto & block = src[int64_t(row) * (k / 64) + subblock / 4];
    const int sub = subblock % 4;
    for (int j = 0; j < 8; ++j) {
        const int first = 2*j, second = first + 1;
        const uint8_t low = (block.qs[8*sub + first % 8] >> (first < 8 ? 0 : 4)) & 15;
        const uint8_t high = (block.qs[8*sub + second % 8] >> (second < 8 ? 0 : 4)) & 15;
        dst[index*8 + j] = low | (high << 4);
    }
    scales[sf_offset(row, subblock, k / 16)] = block.d[sub];
}

static __global__ void repack_activation(const block_fp4_mmq * src, uint8_t * dst, uint8_t * scales, int k, int rows) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= int64_t((rows + 127) / 128 * 128) * (k / 16)) {
        return;
    }
    const int row = index / (k / 16), subblock = index % (k / 16);
    if (row >= rows) {
        scales[sf_offset(row, subblock, k / 16)] = 0;
        return;
    }
    const auto & block = src[int64_t(subblock / 16) * rows + row];
    const int sub = subblock % 16;
    for (int j = 0; j < 8; ++j) {
        const int first = 2*j, second = first + 1;
        const uint8_t low = (uint8_t(block.qs[8*sub + first % 8]) >> (first < 8 ? 0 : 4)) & 15;
        const uint8_t high = (uint8_t(block.qs[8*sub + second % 8]) >> (second < 8 ? 0 : 4)) & 15;
        dst[index*8 + j] = low | (high << 4);
    }
    scales[sf_offset(row, subblock, k / 16)] = reinterpret_cast<const uint8_t *>(block.d4)[sub];
}

template <typename output_type>
struct nvfp4_mlp_consumer {
    using tile = cute::Shape<cute::_128, cute::_128, cute::_128>;
    using cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using element = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using token_scale = cutlass::epilogue::fusion::Sm90ColBroadcast<
            0, tile, float, float, cute::Stride<cute::_1, cute::_0, cute::_0>, 4, false>;
    using channel_scale = cutlass::epilogue::fusion::Sm90RowBroadcast<
            0, tile, float, float, cute::Stride<cute::_0, cute::_1, cute::_0>, 4, false>;
    using multiply = cutlass::epilogue::fusion::Sm90Compute<
            cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using scaled_acc = cutlass::epilogue::fusion::Sm90EVT<multiply, token_scale, cutlass::epilogue::fusion::Sm90AccFetch>;
    using epilogue_op = cutlass::epilogue::fusion::Sm90EVT<multiply, channel_scale, scaled_acc>;
    using epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, tile, cluster,
            cutlass::epilogue::collective::EpilogueTileAuto, float, float,
            void, cutlass::layout::RowMajor, 4, output_type, cutlass::layout::RowMajor, 8,
            cutlass::epilogue::collective::EpilogueScheduleAuto, epilogue_op>::CollectiveOp;
    using mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
            element, cutlass::layout::RowMajor, 32, element, cutlass::layout::ColumnMajor, 32,
            float, tile, cluster, cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename epilogue::SharedStorage)>,
            cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using kernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, mainloop, epilogue>;
    using gemm = cutlass::gemm::device::GemmUniversalAdapter<kernel>;
    using config = typename mainloop::Sm1xxBlkScaledConfig;

    static void run(ggml_backend_cuda_context & ctx, uint8_t * a, uint8_t * sa, const uint8_t * b, const uint8_t * sb,
            float * token_scales, float * channel_scales, output_type * output, int m, int n, int k) {
        const auto shape = cute::make_shape(m, n, k, 1);
        typename epilogue_op::Arguments scaling = { { channel_scales }, { { token_scales }, {}, {} }, {} };
        typename gemm::Arguments args {
            cutlass::gemm::GemmUniversalMode::kGemm, shape,
            { reinterpret_cast<const typename gemm::ElementA *>(a), { k, cute::_1{}, int64_t(0) },
              reinterpret_cast<const typename gemm::ElementB *>(b), { k, cute::_1{}, int64_t(0) },
              reinterpret_cast<cutlass::float_ue4m3_t *>(sa), config::tile_atom_to_shape_SFA(shape),
              reinterpret_cast<const cutlass::float_ue4m3_t *>(sb), config::tile_atom_to_shape_SFB(shape) },
            { scaling, nullptr, { n, cute::_1{}, int64_t(0) }, output, { n, cute::_1{}, int64_t(0) } }
        };
        args.hw_info.device_id = ctx.device;
        args.hw_info.sm_count = ggml_cuda_info().devices[ctx.device].nsm;
        args.scheduler.max_swizzle_size = 4;
        const auto status = gemm::can_implement(args);
        GGML_ASSERT(status == cutlass::Status::kSuccess);
        ggml_cuda_pool_alloc<uint8_t> workspace(ctx.pool(), std::max<size_t>(1, gemm::get_workspace_size(args)));
        gemm op;
        GGML_ASSERT(op.initialize(args, workspace.ptr, ctx.stream()) == cutlass::Status::kSuccess);
        GGML_ASSERT(op.run(ctx.stream()) == cutlass::Status::kSuccess);
    }
};

static __global__ void prepare_unit_scale(float * dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { dst[i] = 1.0f; }
}

static __global__ void prepare_channel_scale(float * dst, const float * up, const float * gate, int n, bool scalar_up, bool scalar_gate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = up[scalar_up ? 0 : i];
        dst[n + i] = gate[scalar_gate ? 0 : i];
    }
}

static __device__ float glu_value(const float * p, int row, int col, int n) {
    const float up = float(p[int64_t(row) * 2*n + col]);
    const float gate = float(p[int64_t(row) * 2*n + n + col]);
    return up * (gate / (1.0f + expf(-gate)));
}

static __global__ __launch_bounds__(1024) void norm_pack(
        const float * x, const float * residual_input, float * residual_output, const float * gamma,
        block_fp4_mmq * out, float * scales, int k, int rows, float eps, const float * up_calibration, const float * gate_calibration,
        uint8_t * direct_values = nullptr, uint8_t * direct_scales = nullptr) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    constexpr int threads = 1024;
    const int row = blockIdx.x;
    const int64_t base = int64_t(row)*k;
    float4 values[4];
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = threadIdx.x + j*threads;
        if (col < k/4) {
            values[j] = ((const float4 *) (x + base))[col];
            if (residual_input) {
                const float4 r = ((const float4 *) (residual_input + base))[col];
                values[j].x = __fadd_rn(values[j].x, r.x);
                values[j].y = __fadd_rn(values[j].y, r.y);
                values[j].z = __fadd_rn(values[j].z, r.z);
                values[j].w = __fadd_rn(values[j].w, r.w);
            }
            const float4 v = values[j];
            sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
    }
    __shared__ float scratch[WARP_SIZE];
    extern __shared__ float normalized[];
    sum = block_reduce<block_reduce_method::SUM, threads>(sum, scratch);
    const float inv_rms = rsqrtf(sum/k + eps);
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = threadIdx.x + j*threads;
        if (col < k/4) {
            const float4 v = values[j];
            if (residual_output) { ((float4 *) (residual_output + base))[col] = v; }
            const float4 g = ((const float4 *) gamma)[col];
            const float4 y = make_float4(__fmul_rn(inv_rms*v.x, g.x), __fmul_rn(inv_rms*v.y, g.y),
                    __fmul_rn(inv_rms*v.z, g.z), __fmul_rn(inv_rms*v.w, g.w));
            normalized[4*col + (4*col)/WARP_SIZE] = y.x;
            normalized[4*col + 1 + (4*col + 1)/WARP_SIZE] = y.y;
            normalized[4*col + 2 + (4*col + 2)/WARP_SIZE] = y.z;
            normalized[4*col + 3 + (4*col + 3)/WARP_SIZE] = y.w;
            amax = fmaxf(amax, fmaxf(fabsf(y.x), fabsf(y.y)));
            amax = fmaxf(amax, fmaxf(fabsf(y.z), fabsf(y.w)));
        }
    }
    __syncthreads();
    if (!up_calibration) { amax = block_reduce<block_reduce_method::MAX, threads>(amax, scratch); }
    const float row_scale = up_calibration ? fminf(*up_calibration, *gate_calibration) : amax/(6.0f*448.0f);
    if (threadIdx.x == 0) { scales[row] = row_scale; }
    const float inv_scale = row_scale > 0.0f ? 1.0f/row_scale : 0.0f;
    for (int sb = threadIdx.x; sb < k/QK_NVFP4_SUB; sb += threads) {
        float block_values[QK_NVFP4_SUB];
#pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB; ++j) { block_values[j] = normalized[QK_NVFP4_SUB*sb + j + (QK_NVFP4_SUB*sb + j)/WARP_SIZE]; }
        uint8_t scale;
        uint32_t q0, q1;
        if (up_calibration) { quantize_calibrated(block_values, inv_scale, scale, q0, q1); }
        else { quantize_nvfp4_subblock(block_values, inv_scale, scale, q0, q1); }
        store_packed(out, q0, q1, scale, row, sb, rows, direct_values, direct_scales, k);
    }
#else
    GGML_UNUSED_VARS(x, residual_input, residual_output, gamma, out, scales, k, rows, eps, up_calibration, gate_calibration);
    NO_DEVICE_CODE;
#endif
}

static __global__ void glu_pack(const float * p, block_fp4_mmq * out, float * scales, int n, int rows) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    constexpr int block_size = 128;
    const int row = blockIdx.x;
    __shared__ float scratch[block_size / WARP_SIZE];
    __shared__ float row_scale;
    float amax = 0.0f;
    for (int i = threadIdx.x; i < n; i += block_size) {
        amax = fmaxf(amax, fabsf(glu_value(p, row, i, n)));
    }
    amax = warp_reduce_max<WARP_SIZE>(amax);
    if (threadIdx.x % WARP_SIZE == 0) {
        scratch[threadIdx.x / WARP_SIZE] = amax;
    }
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
        amax = threadIdx.x < block_size / WARP_SIZE ? scratch[threadIdx.x] : 0.0f;
        amax = warp_reduce_max<WARP_SIZE>(amax);
        if (threadIdx.x == 0) {
            row_scale = amax / (6.0f * 448.0f);
            scales[row] = row_scale;
        }
    }
    __syncthreads();
    const float inv_scale = row_scale > 0.0f ? 1.0f / row_scale : 0.0f;
    for (int sb = threadIdx.x; sb < n / QK_NVFP4_SUB; sb += block_size) {
        float values[QK_NVFP4_SUB];
#pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB; ++j) {
            values[j] = glu_value(p, row, QK_NVFP4_SUB * sb + j, n);
        }
        uint8_t scale;
        uint32_t q0, q1;
        quantize_nvfp4_subblock(values, inv_scale, scale, q0, q1);
        auto * block = out + int64_t(sb / 16) * rows + row;
        auto * qs = reinterpret_cast<uint32_t *>(block->qs);
        qs[2 * (sb % 16)] = q0;
        qs[2 * (sb % 16) + 1] = q1;
        reinterpret_cast<uint8_t *>(block->d4)[sb % 16] = scale;
    }
#else
    GGML_UNUSED_VARS(p, out, scales, n, rows);
    NO_DEVICE_CODE;
#endif
}

static __global__ void glu_pack_shared(const float * p, block_fp4_mmq * out, float * scales, int n, int rows) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    constexpr int block_size = 512;
    const int row = blockIdx.x;
    extern __shared__ float values[];
    __shared__ float scratch[WARP_SIZE];
    __shared__ float row_scale;
    float amax = 0.0f;
    for (int sb = threadIdx.x; sb < n / QK_NVFP4_SUB; sb += block_size) {
#pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB; ++j) {
            const float value = glu_value(p, row, QK_NVFP4_SUB * sb + j, n);
            const int index = QK_NVFP4_SUB * sb + j;
            values[index + index / WARP_SIZE] = value;
            amax = fmaxf(amax, fabsf(value));
        }
    }
    amax = block_reduce<block_reduce_method::MAX, block_size>(amax, scratch);
    if (threadIdx.x == 0) {
        row_scale = amax / (6.0f * 448.0f);
        scales[row] = row_scale;
    }
    __syncthreads();
    const float inv_scale = row_scale > 0.0f ? 1.0f / row_scale : 0.0f;
    for (int sb = threadIdx.x; sb < n / QK_NVFP4_SUB; sb += block_size) {
        float block_values[QK_NVFP4_SUB];
#pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB; ++j) {
            const int index = QK_NVFP4_SUB * sb + j;
            block_values[j] = values[index + index / WARP_SIZE];
        }
        uint8_t scale;
        uint32_t q0, q1;
        quantize_nvfp4_subblock(block_values, inv_scale, scale, q0, q1);
        auto * block = out + int64_t(sb / 16) * rows + row;
        auto * qs = reinterpret_cast<uint32_t *>(block->qs);
        qs[2 * (sb % 16)] = q0;
        qs[2 * (sb % 16) + 1] = q1;
        reinterpret_cast<uint8_t *>(block->d4)[sb % 16] = scale;
    }
#else
    GGML_UNUSED_VARS(p, out, scales, n, rows);
    NO_DEVICE_CODE;
#endif
}

static void pack_glu(ggml_backend_cuda_context & ctx, const float * input, block_fp4_mmq * output, float * scales, int n, int rows) {
    const size_t shared_bytes = size_t(n + n / WARP_SIZE) * sizeof(float);
    if (shared_bytes + 256 <= ggml_cuda_info().devices[ctx.device].smpbo) {
        CUDA_SET_SHARED_MEMORY_LIMIT((glu_pack_shared), shared_bytes);
        glu_pack_shared<<<rows, 512, shared_bytes, ctx.stream()>>>(input, output, scales, n, rows);
    } else {
        glu_pack<<<rows, 128, 0, ctx.stream()>>>(input, output, scales, n, rows);
    }
}

} // namespace

bool ggml_cuda_nvfp4_mlp_available(int cc) {
    return blackwell_mma_available(cc) && cc >= GGML_CUDA_CC_BLACKWELL;
}

size_t ggml_cuda_nvfp4_mlp_weight_bytes(int64_t k, int64_t n) {
    return size_t(use_cutlass_down() ? 3*n : 2*n) * k / 2 + size_t(use_cutlass_down() ? 3*n : 2*n) * k / 16;
}

void ggml_cuda_nvfp4_mlp_prepare(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down, void * data, cudaStream_t stream) {
    const int k = up->ne[0], n = up->ne[1];
    auto * values = static_cast<uint8_t *>(data);
    auto * scales = values + size_t(2*n) * k / 2;
    const int64_t blocks = int64_t(n) * (k / QK_NVFP4_SUB);
    repack_weight<<<(blocks + 255) / 256, 256, 0, stream>>>(static_cast<const block_nvfp4 *>(up->data), values, scales, k, n);
    repack_weight<<<(blocks + 255) / 256, 256, 0, stream>>>(static_cast<const block_nvfp4 *>(gate->data), values + size_t(n)*k/2, scales + size_t(n)*k/16, k, n);
    if (use_cutlass_down()) {
        auto * down_values = scales + size_t(2*n)*k/16;
        auto * down_scales = down_values + size_t(n)*k/2;
        repack_weight<<<(blocks + 255)/256, 256, 0, stream>>>(static_cast<const block_nvfp4 *>(down->data), down_values, down_scales, n, k);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_nvfp4_mlp(ggml_backend_cuda_context & ctx, const ggml_tensor * input,
        const ggml_tensor * up_scale, const ggml_tensor * gate_scale,
        const ggml_tensor * down, ggml_tensor * output, const void * prepared, const ggml_tensor * norm, const ggml_tensor * gamma, ggml_tensor * residual, const ggml_tensor * up_calibration, const ggml_tensor * gate_calibration, const ggml_tensor * down_calibration) {
    const int k = input->ne[0], rows = input->ne[1], n = down->ne[0];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int J = ggml_cuda_mmq_get_J_best(GGML_TYPE_NVFP4, rows, ggml_cuda_mmq_needs_fallback(down->ne[1]),
            cc, ggml_cuda_info().devices[ctx.device].smpbo, GGML_PREC_Q4);
    if (!J) {
        return false;
    }
    const auto config = ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4, J, ggml_cuda_mmq_needs_fallback(k), cc, GGML_PREC_Q4);
    const int ntiles = ((k + config.I - 1)/config.I)*((rows + config.J - 1)/config.J);
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    const int waves = (ntiles + nsm - 1)/nsm;
    // Keep split-K's rounding on shapes where the native consumer partitions its reduction.
    const bool same_reduction = !config.stream_k || 100*ntiles/(nsm*waves) >= 90;
    static const bool direct_pack = [] { const char * value = std::getenv("GGML_CUDA_NVFP4_DIRECT_PACK"); return value && std::atoi(value) != 0; }();
    const bool direct_up = direct_pack && (norm || up_calibration);
    const bool direct_down = direct_pack && up_calibration && down_calibration && use_cutlass_down() && same_reduction;
    auto stream = ctx.stream();
    // Scratch stays in the backend pool, whose addresses remain valid during graph capture and replay.
    ggml_cuda_pool_alloc<block_fp4_mmq> packed(ctx.pool(), size_t(rows)*k/QK_FP4_MMQ + 512);
    const int padded_rows = (rows + 127)/128*128;
    ggml_cuda_pool_alloc<uint8_t> values(ctx.pool(), size_t(rows)*std::max(k, use_cutlass_down() ? n : k)/2);
    ggml_cuda_pool_alloc<uint8_t> scales(ctx.pool(), size_t(padded_rows)*std::max(k, use_cutlass_down() ? n : k)/16);
    ggml_cuda_pool_alloc<float> row_scales(ctx.pool(), rows);
    ggml_cuda_pool_alloc<float> channel_scales(ctx.pool(), std::max(2*n, k));
    static const bool bf16_enabled = [] { const char * value = getenv("GGML_CUDA_NVFP4_BF16_MLP"); return value && atoi(value) != 0; }();
    const bool bf16_upgate = bf16_enabled && up_calibration;
    ggml_cuda_pool_alloc<float> intermediate(ctx.pool());
    if (!bf16_upgate) { intermediate.alloc(size_t(rows)*2*n); }
    ggml_cuda_pool_alloc<cutlass::bfloat16_t> bf16_intermediate(ctx.pool());
    if (bf16_upgate) { bf16_intermediate.alloc(size_t(rows)*2*n); }
    ggml_cuda_pool_alloc<block_fp4_mmq> down_packed(ctx.pool(), size_t(rows)*n/QK_FP4_MMQ + 512);
    ggml_cuda_pool_alloc<float> down_scales(ctx.pool(), rows);
    if (direct_up && rows != padded_rows) {
        CUDA_CHECK(cudaMemsetAsync(scales.ptr, 0, size_t(padded_rows)*k/16, stream));
    }
    if (norm) {
        float eps;
        memcpy(&eps, norm->op_params, sizeof(eps));
        const auto * raw = residual ? residual->src[0] : norm->src[0];
        norm_pack<<<rows, 1024, size_t(k + k/WARP_SIZE)*sizeof(float), stream>>>(static_cast<const float *>(raw->data),
                residual ? static_cast<const float *>(residual->src[1]->data) : nullptr,
                residual ? static_cast<float *>(residual->data) : nullptr, static_cast<const float *>(gamma->data),
                packed.ptr, row_scales.ptr, k, rows, eps,
                up_calibration ? static_cast<const float *>(up_calibration->data) : nullptr,
                gate_calibration ? static_cast<const float *>(gate_calibration->data) : nullptr,
                direct_up ? values.ptr : nullptr, direct_up ? scales.ptr : nullptr);
    } else if (up_calibration) {
        calibrated_pack<<<(int64_t(rows)*(k/16)+255)/256, 256, 0, stream>>>(static_cast<const float *>(input->data),
                static_cast<const float *>(up_calibration->data), static_cast<const float *>(gate_calibration->data),
                packed.ptr, row_scales.ptr, k, rows, false, direct_up ? values.ptr : nullptr, direct_up ? scales.ptr : nullptr);
    } else {
        quantize_mmq_fp4_cuda(static_cast<const float *>(input->data), nullptr, packed.ptr, row_scales.ptr, GGML_TYPE_NVFP4,
                true, k, k, int64_t(k)*rows, int64_t(k)*rows, k, rows, 1, 1, stream);
    }
    if (!direct_up) {
        repack_activation<<<(int64_t(padded_rows)*(k/16)+255)/256, 256, 0, stream>>>(packed.ptr, values.ptr, scales.ptr, k, rows);
    }
    prepare_channel_scale<<<(n+255)/256, 256, 0, stream>>>(channel_scales.ptr,
            static_cast<const float *>(up_scale->data), static_cast<const float *>(gate_scale->data), n,
            ggml_nelements(up_scale) == 1, ggml_nelements(gate_scale) == 1);
    auto * weight_values = static_cast<const uint8_t *>(prepared);
    auto * weight_scales = weight_values + size_t(2*n)*k/2;
    if (bf16_upgate) {
        nvfp4_mlp_consumer<cutlass::bfloat16_t>::run(ctx, values.ptr, scales.ptr, weight_values,
                weight_scales, row_scales.ptr, channel_scales.ptr, bf16_intermediate.ptr, rows, 2*n, k);
    } else {
        nvfp4_mlp_consumer<float>::run(ctx, values.ptr, scales.ptr, weight_values,
                weight_scales, row_scales.ptr, channel_scales.ptr, intermediate.ptr, rows, 2*n, k);
    }
    if (direct_down && rows != padded_rows) {
        CUDA_CHECK(cudaMemsetAsync(scales.ptr, 0, size_t(padded_rows)*n/16, stream));
    }
    if (up_calibration && down_calibration) {
        if (bf16_upgate) {
            calibrated_pack<<<(int64_t(rows)*(n/16)+255)/256, 256, 0, stream>>>(bf16_intermediate.ptr,
                    static_cast<const float *>(down_calibration->data), nullptr, down_packed.ptr, down_scales.ptr, n, rows, true,
                    direct_down ? values.ptr : nullptr, direct_down ? scales.ptr : nullptr);
        } else {
            calibrated_pack<<<(int64_t(rows)*(n/16)+255)/256, 256, 0, stream>>>(intermediate.ptr,
                static_cast<const float *>(down_calibration->data), nullptr, down_packed.ptr, down_scales.ptr, n, rows, true,
                    direct_down ? values.ptr : nullptr, direct_down ? scales.ptr : nullptr);
        }
    } else {
        pack_glu(ctx, intermediate.ptr, down_packed.ptr, down_scales.ptr, n, rows);
    }
    if (use_cutlass_down() && same_reduction) {
        if (!direct_down) {
            repack_activation<<<(int64_t(padded_rows)*(n/16)+255)/256, 256, 0, stream>>>(down_packed.ptr, values.ptr, scales.ptr, n, rows);
        }
        prepare_unit_scale<<<(k+255)/256, 256, 0, stream>>>(channel_scales.ptr, k);
        const auto * down_values = weight_scales + size_t(2*n)*k/16;
        const auto * down_weight_scales = down_values + size_t(n)*k/2;
        nvfp4_mlp_consumer<float>::run(ctx, values.ptr, scales.ptr, down_values, down_weight_scales,
                down_scales.ptr, channel_scales.ptr, static_cast<float *>(output->data), rows, k, n);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    const int64_t packed_stride = int64_t(rows)*n*sizeof(block_fp4_mmq)/(QK_FP4_MMQ*sizeof(int));
    const mmq_args args = {
        static_cast<const char *>(down->data), GGML_TYPE_NVFP4, reinterpret_cast<const int *>(down_packed.ptr), nullptr, nullptr,
        static_cast<float *>(output->data), down_scales.ptr,
        n, k, rows, n/QK_NVFP4, rows, k,
        1, 1, int64_t(n)*k/QK_NVFP4, packed_stride, int64_t(k)*rows,
        1, 1, int64_t(n)*k/QK_NVFP4, packed_stride, int64_t(k)*rows,
        rows, J, {}
    };
    mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q4>(ctx, args, stream);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else
bool ggml_cuda_nvfp4_mlp_available(int) { return false; }
size_t ggml_cuda_nvfp4_mlp_weight_bytes(int64_t, int64_t) { return 0; }
void ggml_cuda_nvfp4_mlp_prepare(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, void *, cudaStream_t) {}
bool ggml_cuda_nvfp4_mlp(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, ggml_tensor *, const void *, const ggml_tensor *, const ggml_tensor *, ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *) { return false; }
#endif
