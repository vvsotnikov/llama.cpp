#include "fp8.cuh"
#include "unary.cuh"

#if defined(GGML_CUDA_FP8_CUTLASS)
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>

namespace {

// The GEMM reads token-major activations and column-major weights; GGUF already stores this weight layout.
using fp8_element = cutlass::float_e4m3_t;
using fp8_tile = cute::Shape<cute::_128, cute::_128, cute::_128>;
using fp8_cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
using fp8_token_scale = cutlass::epilogue::fusion::Sm90ColBroadcast<
        0, fp8_tile, float, float, cute::Stride<cute::_1, cute::_0, cute::_0>, 4, false>;
using fp8_weight_scale = cutlass::epilogue::fusion::Sm90RowBroadcast<
        0, fp8_tile, float, float, cute::Stride<cute::_0, cute::_1, cute::_0>, 4, false>;
using fp8_multiply = cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
// Apply activation scaling before channel scaling, matching the unfused FP32 rounding order.
using fp8_scaled_acc = cutlass::epilogue::fusion::Sm90EVT<
        fp8_multiply, fp8_token_scale, cutlass::epilogue::fusion::Sm90AccFetch>;
using fp8_epilogue_op = cutlass::epilogue::fusion::Sm90EVT<fp8_multiply, fp8_weight_scale, fp8_scaled_acc>;
using fp8_epilogue = cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, fp8_tile, fp8_cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float,
        void, cutlass::layout::RowMajor, 4, float, cutlass::layout::RowMajor, 4,
        cutlass::epilogue::collective::EpilogueScheduleAuto, fp8_epilogue_op>::CollectiveOp;
using fp8_mainloop = cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        fp8_element, cutlass::layout::RowMajor, 16,
        fp8_element, cutlass::layout::ColumnMajor, 16,
        float, fp8_tile, fp8_cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename fp8_epilogue::SharedStorage)>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using fp8_kernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>, fp8_mainloop, fp8_epilogue>;
using fp8_gemm = cutlass::gemm::device::GemmUniversalAdapter<fp8_kernel>;

static __global__ void fp8_prepare_global(
        const float * x, const float * gate, float * produced, float * row_scales, int k) {
    const int64_t base = int64_t(blockIdx.x) * k;
    float amax = 0.0f;
    for (int col = threadIdx.x; col < k; col += blockDim.x) {
        float v = x[base + col];
        if (gate) {
            const float g = gate[base + col];
            v *= g / (1.0f + expf(-g));
            produced[base + col] = v;
        }
        amax = fmaxf(amax, isfinite(v) ? fabsf(v) : 448.0f);
    }
    __shared__ float scratch[WARP_SIZE];
    amax = block_reduce<block_reduce_method::MAX, 256>(amax, scratch);
    if (threadIdx.x == 0) {
        row_scales[blockIdx.x] = amax;
    }
}

template <bool token_scale>
static __global__ void fp8_gated_normalize_global(
        const float * x, const float * gate, const float * weight, float * produced, float * row_amax,
        int heads, float eps, uint8_t * packed) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t base = int64_t(blockIdx.x)*heads*128;
    const float4 w = ((const float4 *) weight)[lane];
    float amax = 0.0f;
    float4 values[4];
    for (int head = warp; head < heads; head += 16) {
        const int64_t index = base/4 + head*32 + lane;
        const float4 v = ((const float4 *) x)[index];
        float sum = 0.0f;
        sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        sum = block_reduce<block_reduce_method::SUM, WARP_SIZE>(sum, (float *) nullptr);
        const float scale = rsqrtf(sum/128 + eps);
        const float4 g = ((const float4 *) gate)[index];
        float4 y;
        y.x = __fmul_rn(__fmul_rn(scale*v.x, w.x), ggml_cuda_op_silu_single(g.x));
        y.y = __fmul_rn(__fmul_rn(scale*v.y, w.y), ggml_cuda_op_silu_single(g.y));
        y.z = __fmul_rn(__fmul_rn(scale*v.z, w.z), ggml_cuda_op_silu_single(g.z));
        y.w = __fmul_rn(__fmul_rn(scale*v.w, w.w), ggml_cuda_op_silu_single(g.w));
        if constexpr (token_scale) { values[head/16] = y; }
        else { ((float4 *) produced)[index] = y; }
        amax = fmaxf(amax, isfinite(y.x) ? fabsf(y.x) : 448.0f);
        amax = fmaxf(amax, isfinite(y.y) ? fabsf(y.y) : 448.0f);
        amax = fmaxf(amax, isfinite(y.z) ? fabsf(y.z) : 448.0f);
        amax = fmaxf(amax, isfinite(y.w) ? fabsf(y.w) : 448.0f);
    }
    __shared__ float scratch[WARP_SIZE];
    amax = block_reduce<block_reduce_method::MAX, 512>(amax, scratch);
    const float scale = amax > 0.0f ? amax/448.0f : 1.0f;
    if (threadIdx.x == 0) { row_amax[blockIdx.x] = token_scale ? scale : amax; }
    if constexpr (token_scale) {
        for (int head = warp; head < heads; head += 16) {
            const float4 y = values[head/16];
            const __nv_fp8x4_e4m3 q(make_float4(y.x/scale, y.y/scale, y.z/scale, y.w/scale));
            ((uint32_t *) packed)[base/4 + head*32 + lane] = q.__x;
        }
    }
}

template <int vpt>
static __global__ void fp8_pack_tokens(
        const float * x, const float * gate, float * produced, uint8_t * packed, float * scales, int k) {
    const int64_t base = int64_t(blockIdx.x)*k;
    float4 values[vpt];
    float amax = 0.0f;
    __shared__ float scratch[WARP_SIZE];
#pragma unroll
    for (int j = 0; j < vpt; ++j) {
        const int col = threadIdx.x + j*256;
        if (col < k/4) {
            float4 v = ((const float4 *) (x + base))[col];
            if (gate) {
                const float4 g = ((const float4 *) (gate + base))[col];
                v.x *= g.x/(1.0f + expf(-g.x));
                v.y *= g.y/(1.0f + expf(-g.y));
                v.z *= g.z/(1.0f + expf(-g.z));
                v.w *= g.w/(1.0f + expf(-g.w));
                ((float4 *) (produced + base))[col] = v;
            }
            values[j] = v;
            amax = fmaxf(amax, isfinite(v.x) ? fabsf(v.x) : 448.0f);
            amax = fmaxf(amax, isfinite(v.y) ? fabsf(v.y) : 448.0f);
            amax = fmaxf(amax, isfinite(v.z) ? fabsf(v.z) : 448.0f);
            amax = fmaxf(amax, isfinite(v.w) ? fabsf(v.w) : 448.0f);
        }
    }
    amax = block_reduce<block_reduce_method::MAX, 256>(amax, scratch);
    const float scale = amax > 0.0f ? amax/448.0f : 1.0f;
    if (threadIdx.x == 0) {
        scales[blockIdx.x] = scale;
    }
#pragma unroll
    for (int j = 0; j < vpt; ++j) {
        const int col = threadIdx.x + j*256;
        if (col < k/4) {
            const float4 v = values[j];
            packed[base + 4*col]     = ggml_cuda_fp32_to_f8_e4m3(v.x/scale);
            packed[base + 4*col + 1] = ggml_cuda_fp32_to_f8_e4m3(v.y/scale);
            packed[base + 4*col + 2] = ggml_cuda_fp32_to_f8_e4m3(v.z/scale);
            packed[base + 4*col + 3] = ggml_cuda_fp32_to_f8_e4m3(v.w/scale);
        }
    }
}

template <int block_size, int vpt, bool token_scale>
static __global__ void fp8_normalize_global(
        const float * x, const float * weight, float * produced, float * row_scales,
        const float * residual, float * residual_out, const float * input_scale, bool scalar_scale, int k, float eps, uint8_t * packed) {
    const int64_t base = int64_t(blockIdx.x) * k;
    float4 values[vpt];
    float sum = 0.0f;
    __shared__ float scratch[WARP_SIZE];
    // Match the native vector RMS reduction and its rounding order before deriving the FP8 scale.
#pragma unroll
    for (int j = 0; j < vpt; ++j) {
        const int index = threadIdx.x + j * block_size;
        if (index < k / 4) {
            float4 v = ((const float4 *) (x + base))[index];
            if (input_scale) {
                const float4 scale = scalar_scale ? make_float4(*input_scale, *input_scale, *input_scale, *input_scale) :
                                                   ((const float4 *) input_scale)[index];
                v = make_float4(__fmul_rn(v.x, scale.x), __fmul_rn(v.y, scale.y), __fmul_rn(v.z, scale.z), __fmul_rn(v.w, scale.w));
            }
            if (residual) {
                const float4 r = ((const float4 *) (residual + base))[index];
                v = make_float4(__fadd_rn(v.x, r.x), __fadd_rn(v.y, r.y), __fadd_rn(v.z, r.z), __fadd_rn(v.w, r.w));
            }
            values[j] = v;
            sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
    }
    sum = block_reduce<block_reduce_method::SUM, block_size>(sum, scratch);
    const float inv_rms = rsqrtf(sum / k + eps);
    __syncthreads();
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < vpt; ++j) {
        const int index = threadIdx.x + j * block_size;
        if (index < k / 4) {
            const float4 v = values[j];
            if (residual_out) {
                ((float4 *) (residual_out + base))[index] = v;
            }
            const float4 w = ((const float4 *) weight)[index];
            float4 y = make_float4(inv_rms*v.x, inv_rms*v.y, inv_rms*v.z, inv_rms*v.w);
            y.x *= w.x;
            y.y *= w.y;
            y.z *= w.z;
            y.w *= w.w;
            ((float4 *) (produced + base))[index] = y;
            if constexpr (token_scale) {
                values[j] = y;
            }
            amax = fmaxf(amax, isfinite(y.x) ? fabsf(y.x) : 448.0f);
            amax = fmaxf(amax, isfinite(y.y) ? fabsf(y.y) : 448.0f);
            amax = fmaxf(amax, isfinite(y.z) ? fabsf(y.z) : 448.0f);
            amax = fmaxf(amax, isfinite(y.w) ? fabsf(y.w) : 448.0f);
        }
    }
    amax = block_reduce<block_reduce_method::MAX, block_size>(amax, scratch);
    const float scale = amax > 0.0f ? amax/448.0f : 1.0f;
    if (threadIdx.x == 0) {
        row_scales[blockIdx.x] = token_scale ? scale : amax;
    }
    if constexpr (token_scale) {
#pragma unroll
        for (int j = 0; j < vpt; ++j) {
            const int index = threadIdx.x + j*block_size;
            if (index < k/4) {
                const float4 y = values[j];
                packed[base + 4*index]     = ggml_cuda_fp32_to_f8_e4m3(y.x/scale);
                packed[base + 4*index + 1] = ggml_cuda_fp32_to_f8_e4m3(y.y/scale);
                packed[base + 4*index + 2] = ggml_cuda_fp32_to_f8_e4m3(y.z/scale);
                packed[base + 4*index + 3] = ggml_cuda_fp32_to_f8_e4m3(y.w/scale);
            }
        }
    }
}

static __global__ void fp8_scale_max(const float * scales, float * maximum, int n) {
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        v = fmaxf(v, scales[i]);
    }
    __shared__ float scratch[WARP_SIZE];
    v = block_reduce<block_reduce_method::MAX, 256>(v, scratch);
    if (threadIdx.x == 0) {
        *maximum = v > 0.0f ? v / 448.0f : 1.0f;
    }
}

static __global__ void fp8_pack_global(const float * x, uint8_t * y, float * scales, const float * maximum, int64_t ne, int rows) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const float scale = *maximum;
    if (i < ne) {
        y[i] = ggml_cuda_fp32_to_f8_e4m3(x[i] / scale);
    }
    if (i < rows) {
        scales[i] = scale;
    }
}

} // namespace
#endif

bool ggml_cuda_mul_mat_fp8_pipeline(
        ggml_backend_cuda_context & ctx, const ggml_tensor * input,
        const ggml_tensor * const * weights, const ggml_tensor * const * scales,
        ggml_tensor * const * outputs, int count, const ggml_tensor * producer,
        const ggml_tensor * residual, bool fuse_input_scale) {
#if defined(GGML_CUDA_FP8_CUTLASS)
    static const bool enabled = [] {
        const char * value = getenv("GGML_CUDA_FP8_PIPELINE");
        return value && strcmp(value, "0") != 0;
    }();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!enabled || !blackwell_mma_available(cc) || cc >= GGML_CUDA_CC_RUBIN || count < 1 || count > 2 ||
            !ggml_is_matrix(input) || !ggml_is_contiguous(input) || input->type != GGML_TYPE_F32 ||
            input->ne[0] <= 0 || input->ne[0] % 16 != 0 || input->ne[1] < 64 ||
            input->ne[0] > INT_MAX || input->ne[1] > INT_MAX) {
        return false;
    }
    const int k = input->ne[0];
    const int m = input->ne[1];
    static const bool token_scale = [] {
        const char * value = getenv("GGML_CUDA_FP8_TOKEN_SCALE");
        return value && strcmp(value, "0") != 0;
    }();
    const ggml_tensor * raw = input;
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * norm_weight = nullptr;
    const ggml_tensor * residual_input = nullptr;
    const ggml_tensor * input_scale = nullptr;
    float eps = 0.0f;
    bool head_norm = false;
    if (producer) {
        if (producer == input && producer->op == GGML_OP_RESHAPE) {
            const auto * gated = producer->src[0];
            if (!gated || gated->op != GGML_OP_MUL) {
                return false;
            }
            const auto * activation = gated->src[1];
            const auto * weighted = gated->src[0];
            if (!activation || activation->op != GGML_OP_UNARY || ggml_get_unary_op(activation) != GGML_UNARY_OP_SILU ||
                    !weighted || weighted->op != GGML_OP_MUL || !weighted->src[0] || weighted->src[0]->op != GGML_OP_RMS_NORM) {
                return false;
            }
            const auto * norm = weighted->src[0];
            raw = norm->src[0];
            gate = activation->src[0];
            norm_weight = weighted->src[1];
            if (!raw || !gate || !norm_weight || raw->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 ||
                    !ggml_is_contiguous(raw) || !ggml_is_contiguous(gate) || !ggml_are_same_shape(raw, gate) ||
                    raw->ne[0] != 128 || raw->ne[1] > 64 || k != 128*raw->ne[1] || ggml_nelements(raw) != ggml_nelements(input) ||
                    norm_weight->type != GGML_TYPE_F32 || !ggml_is_vector(norm_weight) || norm_weight->ne[0] != 128 ||
                    !ggml_is_contiguous(norm_weight) || !ggml_cuda_is_aligned(gate, 16)) {
                return false;
            }
            memcpy(&eps, norm->op_params, sizeof(eps));
            head_norm = true;
        } else if (producer == input && producer->op == GGML_OP_GLU &&
                ggml_get_glu_op(producer) == GGML_GLU_OP_SWIGLU && !ggml_get_op_params_i32(producer, 1)) {
            gate = producer->src[0];
            raw = producer->src[1];
            if (!gate || !raw || gate->type != GGML_TYPE_F32 || raw->type != GGML_TYPE_F32 ||
                    !ggml_are_same_shape(gate, raw) || !ggml_are_same_shape(raw, input) ||
                    !ggml_is_contiguous(gate) || !ggml_is_contiguous(raw)) {
                return false;
            }
        } else if (producer == input && producer->op == GGML_OP_MUL) {
            const ggml_tensor * norm = producer->src[0];
            norm_weight = producer->src[1];
            if (!norm || norm->op != GGML_OP_RMS_NORM || !norm_weight ||
                    norm_weight->type != GGML_TYPE_F32 || !ggml_is_vector(norm_weight) ||
                    norm_weight->ne[0] != k || !ggml_is_contiguous(norm_weight)) {
                return false;
            }
            raw = norm->src[0];
            if (!raw || raw->type != GGML_TYPE_F32 || !ggml_is_contiguous(raw) || !ggml_are_same_shape(raw, input)) {
                return false;
            }
            memcpy(&eps, norm->op_params, sizeof(eps));
        } else {
            return false;
        }
    }
    if (residual) {
        if (!norm_weight || raw != residual || residual->op != GGML_OP_ADD) {
            return false;
        }
        raw = residual->src[0];
        residual_input = residual->src[1];
        if (fuse_input_scale) {
            if (!raw || raw->op != GGML_OP_MUL) {
                return false;
            }
            input_scale = raw->src[1];
            raw = raw->src[0];
            if (!input_scale || input_scale->type != GGML_TYPE_F32 || !ggml_is_vector(input_scale) ||
                    !ggml_is_contiguous(input_scale) || (ggml_nelements(input_scale) != 1 && input_scale->ne[0] != k)) {
                return false;
            }
        }
        if (!raw || !residual_input || raw->type != GGML_TYPE_F32 || residual_input->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(raw) || !ggml_is_contiguous(residual_input) ||
                !ggml_are_same_shape(raw, input) || !ggml_are_same_shape(residual_input, input)) {
            return false;
        }
    }
    if (norm_weight && (k > 16384 || !ggml_cuda_is_aligned(raw, 16) ||
            !ggml_cuda_is_aligned(input, 16) || !ggml_cuda_is_aligned(norm_weight, 16) ||
            (residual_input && !ggml_cuda_is_aligned(residual_input, 16)) ||
            (input_scale && ggml_nelements(input_scale) > 1 && !ggml_cuda_is_aligned(input_scale, 16)))) {
        return false;
    }
    if (token_scale && (k > 32768 || !ggml_cuda_is_aligned(raw, 16) ||
            (gate && !ggml_cuda_is_aligned(gate, 16)))) {
        return false;
    }
    for (int i = 0; i < count; ++i) {
        const auto * w = weights[i];
        const auto * s = scales[i];
        const auto * d = outputs[i];
        if (!w || !s || !d || w->type != GGML_TYPE_F8_E4M3 || !ggml_is_matrix(w) || !ggml_is_contiguous(w) ||
                w->ne[0] != k || w->ne[1] <= 0 || w->ne[1] > INT_MAX || w->ne[1] % 16 != 0 ||
                s->type != GGML_TYPE_F32 || !ggml_is_vector(s) || !ggml_is_contiguous(s) || s->ne[0] != w->ne[1] ||
                d->type != GGML_TYPE_F32 || !ggml_is_matrix(d) || !ggml_is_contiguous(d) ||
                d->ne[0] != w->ne[1] || d->ne[1] != m ||
                !ggml_cuda_is_aligned(w, 16) || !ggml_cuda_is_aligned(s, 16) || !ggml_cuda_is_aligned(d, 16)) {
            return false;
        }
    }
    ggml_cuda_pool_alloc<uint8_t> packed(ctx.pool(), size_t(k) * m);
    ggml_cuda_pool_alloc<float> token_scales(ctx.pool(), m);
    fp8_gemm::Arguments args[2];
    size_t workspace_size = 0;
    for (int i = 0; i < count; ++i) {
        const int n = weights[i]->ne[1];
        fp8_epilogue_op::Arguments epilogue_args{
            { (const float *) scales[i]->data }, { { token_scales.ptr }, {}, {} }, {} };
        args[i] = fp8_gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1},
            { (const fp8_element *) packed.ptr, {k, cute::_1{}, int64_t(0)},
              (const fp8_element *) weights[i]->data, {k, cute::_1{}, int64_t(0)} },
            { epilogue_args, nullptr, {n, cute::_1{}, int64_t(0)},
              (float *) outputs[i]->data, {n, cute::_1{}, int64_t(0)} } };
        if (fp8_gemm::can_implement(args[i]) != cutlass::Status::kSuccess) {
            return false;
        }
        workspace_size = std::max(workspace_size, fp8_gemm::get_workspace_size(args[i]));
    }
    ggml_cuda_pool_alloc<uint8_t> workspace(ctx.pool(), std::max<size_t>(workspace_size, 1));
    if (head_norm) {
#define FP8_GATED(T) fp8_gated_normalize_global<T><<<m, 512, 0, ctx.stream()>>>( \
        (const float *) raw->data, (const float *) gate->data, (const float *) norm_weight->data, \
        (float *) input->data, token_scales.ptr, k/128, eps, packed.ptr)
        if (token_scale) { FP8_GATED(true); } else { FP8_GATED(false); }
#undef FP8_GATED
    } else if (norm_weight) {
#define FP8_NORM_IMPL(B, V, T) fp8_normalize_global<B, V, T><<<m, B, 0, ctx.stream()>>>( \
        (const float *) raw->data, (const float *) norm_weight->data, (float *) input->data, token_scales.ptr, \
        residual_input ? (const float *) residual_input->data : nullptr, residual ? (float *) residual->data : nullptr, \
        input_scale ? (const float *) input_scale->data : nullptr, input_scale && ggml_nelements(input_scale) == 1, k, eps, packed.ptr)
#define FP8_NORM(B, V) do { if (token_scale) { FP8_NORM_IMPL(B, V, true); } else { FP8_NORM_IMPL(B, V, false); } } while (0)
        const int n4 = k / 4;
        if      (n4 <=   32) { FP8_NORM(  32, 1); }
        else if (n4 <=  128) { FP8_NORM( 128, 1); }
        else if (n4 <=  256) { FP8_NORM( 256, 1); }
        else if (n4 <=  512) { FP8_NORM( 512, 1); }
        else if (n4 <= 1024) { FP8_NORM(1024, 1); }
        else                { FP8_NORM(1024, 4); }
#undef FP8_NORM
#undef FP8_NORM_IMPL
    } else if (token_scale) {
#define FP8_PACK(V) fp8_pack_tokens<V><<<m, 256, 0, ctx.stream()>>>( \
        (const float *) raw->data, gate ? (const float *) gate->data : nullptr, \
        gate ? (float *) input->data : nullptr, packed.ptr, token_scales.ptr, k)
        if      (k <=  4096) { FP8_PACK( 4); }
        else if (k <=  8192) { FP8_PACK( 8); }
        else if (k <= 16384) { FP8_PACK(16); }
        else if (k <= 20480) { FP8_PACK(20); }
        else                { FP8_PACK(32); }
#undef FP8_PACK
    } else {
        fp8_prepare_global<<<m, 256, 0, ctx.stream()>>>(
                (const float *) raw->data, gate ? (const float *) gate->data : nullptr,
                gate ? (float *) input->data : nullptr, token_scales.ptr, k);
    }
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_pool_alloc<float> maximum(ctx.pool());
    if (!token_scale) {
        maximum.alloc(1);
        fp8_scale_max<<<1, 256, 0, ctx.stream()>>>(token_scales.ptr, maximum.ptr, m);
        fp8_pack_global<<<(int64_t(k)*m + 255)/256, 256, 0, ctx.stream()>>>(
                (const float *) input->data, packed.ptr, token_scales.ptr, maximum.ptr, int64_t(k)*m, m);
        CUDA_CHECK(cudaGetLastError());
    }
    for (int i = 0; i < count; ++i) {
        fp8_gemm gemm;
        GGML_ASSERT(gemm.initialize(args[i], workspace.ptr, ctx.stream()) == cutlass::Status::kSuccess);
        GGML_ASSERT(gemm.run(ctx.stream()) == cutlass::Status::kSuccess);
    }
    return true;
#else
    GGML_UNUSED_VARS(ctx, input, weights, scales, outputs, count, producer, residual, fuse_input_scale);
    return false;
#endif
}
