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

static __host__ __device__ int64_t sf_offset(int row, int subblock, int blocks) {
    return int64_t(row / 128) * (blocks / 4) * 512 + int64_t(subblock / 4) * 512 +
            (row % 32) * 16 + ((row % 128) / 32) * 4 + subblock % 4;
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

struct nvfp4_mlp_consumer {
    using output_type = float;
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
        const auto status = gemm::can_implement(args);
        GGML_ASSERT(status == cutlass::Status::kSuccess);
        ggml_cuda_pool_alloc<uint8_t> workspace(ctx.pool(), std::max<size_t>(1, gemm::get_workspace_size(args)));
        gemm op;
        GGML_ASSERT(op.initialize(args, workspace.ptr, ctx.stream()) == cutlass::Status::kSuccess);
        GGML_ASSERT(op.run(ctx.stream()) == cutlass::Status::kSuccess);
    }
};

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
    return size_t(2*n) * k / 2 + size_t(2*n) * k / 16;
}

void ggml_cuda_nvfp4_mlp_prepare(const ggml_tensor * up, const ggml_tensor * gate, void * data, cudaStream_t stream) {
    const int k = up->ne[0], n = up->ne[1];
    auto * values = static_cast<uint8_t *>(data);
    auto * scales = values + size_t(2*n) * k / 2;
    const int64_t blocks = int64_t(n) * (k / QK_NVFP4_SUB);
    repack_weight<<<(blocks + 255) / 256, 256, 0, stream>>>(static_cast<const block_nvfp4 *>(up->data), values, scales, k, n);
    repack_weight<<<(blocks + 255) / 256, 256, 0, stream>>>(static_cast<const block_nvfp4 *>(gate->data), values + size_t(n)*k/2, scales + size_t(n)*k/16, k, n);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_nvfp4_mlp(ggml_backend_cuda_context & ctx, const ggml_tensor * input,
        const ggml_tensor * up_scale, const ggml_tensor * gate_scale,
        const ggml_tensor * down, ggml_tensor * output, const void * prepared) {
    const int k = input->ne[0], rows = input->ne[1], n = down->ne[0];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int J = ggml_cuda_mmq_get_J_best(GGML_TYPE_NVFP4, rows, ggml_cuda_mmq_needs_fallback(down->ne[1]),
            cc, ggml_cuda_info().devices[ctx.device].smpbo, GGML_PREC_Q4);
    if (!J) {
        return false;
    }
    auto stream = ctx.stream();
    // Scratch stays in the backend pool, whose addresses remain valid during graph capture and replay.
    ggml_cuda_pool_alloc<block_fp4_mmq> packed(ctx.pool(), size_t(rows)*k/QK_FP4_MMQ + 512);
    ggml_cuda_pool_alloc<uint8_t> values(ctx.pool(), size_t(rows)*k/2);
    ggml_cuda_pool_alloc<uint8_t> scales(ctx.pool(), size_t(rows)*k/16);
    ggml_cuda_pool_alloc<float> row_scales(ctx.pool(), rows);
    ggml_cuda_pool_alloc<float> channel_scales(ctx.pool(), 2*n);
    ggml_cuda_pool_alloc<float> intermediate(ctx.pool(), size_t(rows)*2*n);
    ggml_cuda_pool_alloc<block_fp4_mmq> down_packed(ctx.pool(), size_t(rows)*n/QK_FP4_MMQ + 512);
    ggml_cuda_pool_alloc<float> down_scales(ctx.pool(), rows);
    quantize_mmq_fp4_cuda(static_cast<const float *>(input->data), nullptr, packed.ptr, row_scales.ptr, GGML_TYPE_NVFP4,
            true, k, k, int64_t(k)*rows, int64_t(k)*rows, k, rows, 1, 1, stream);
    repack_activation<<<(int64_t(rows)*(k/16)+255)/256, 256, 0, stream>>>(packed.ptr, values.ptr, scales.ptr, k, rows);
    prepare_channel_scale<<<(n+255)/256, 256, 0, stream>>>(channel_scales.ptr,
            static_cast<const float *>(up_scale->data), static_cast<const float *>(gate_scale->data), n,
            ggml_nelements(up_scale) == 1, ggml_nelements(gate_scale) == 1);
    auto * weight_values = static_cast<const uint8_t *>(prepared);
    auto * weight_scales = weight_values + size_t(2*n)*k/2;
    nvfp4_mlp_consumer::run(ctx, values.ptr, scales.ptr, weight_values,
            weight_scales, row_scales.ptr, channel_scales.ptr, intermediate.ptr, rows, 2*n, k);
    pack_glu(ctx, intermediate.ptr, down_packed.ptr, down_scales.ptr, n, rows);
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
void ggml_cuda_nvfp4_mlp_prepare(const ggml_tensor *, const ggml_tensor *, void *, cudaStream_t) {}
bool ggml_cuda_nvfp4_mlp(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, ggml_tensor *, const void *) { return false; }
#endif
