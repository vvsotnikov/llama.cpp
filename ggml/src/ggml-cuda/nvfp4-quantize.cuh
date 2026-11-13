#pragma once

#include "common.cuh"

#if defined(BLACKWELL_MMA_AVAILABLE)
#if CUDART_VERSION >= 12080
static __device__ __forceinline__ float nvfp4_native_scale_error(
        const float vals[QK_NVFP4_SUB], const float inv_col_scale, const float inv_scale, const float scale) {
    const float scale_dequant = 2.0f * scale;
    float err = 0.0f;

#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; k += 4) {
        const float v0 = vals[k + 0] * inv_col_scale;
        const float v1 = vals[k + 1] * inv_col_scale;
        const float v2 = vals[k + 2] * inv_col_scale;
        const float v3 = vals[k + 3] * inv_col_scale;

        const __nv_fp4x4_e2m1 q(make_float4(v0 * inv_scale, v1 * inv_scale, v2 * inv_scale, v3 * inv_scale));
        const __nv_fp4x4_storage_t q_storage = q.__x;
        const __nv_fp4x2_storage_t q_lo = static_cast<__nv_fp4x2_storage_t>(q_storage);
        const __nv_fp4x2_storage_t q_hi = static_cast<__nv_fp4x2_storage_t>(q_storage >> 8U);

        const __half2_raw hraw2_lo = __nv_cvt_fp4x2_to_halfraw2(q_lo, __NV_E2M1);
        const __half2_raw hraw2_hi = __nv_cvt_fp4x2_to_halfraw2(q_hi, __NV_E2M1);
        const __half2 h2_lo = static_cast<__half2>(hraw2_lo);
        const __half2 h2_hi = static_cast<__half2>(hraw2_hi);
        const float2 dq_lo = __half22float2(h2_lo);
        const float2 dq_hi = __half22float2(h2_hi);

        const float err0 = fabsf(v0) - fabsf(dq_lo.x) * scale_dequant;
        const float err1 = fabsf(v1) - fabsf(dq_lo.y) * scale_dequant;
        const float err2 = fabsf(v2) - fabsf(dq_hi.x) * scale_dequant;
        const float err3 = fabsf(v3) - fabsf(dq_hi.y) * scale_dequant;

        err = fmaf(err0, err0, err);
        err = fmaf(err1, err1, err);
        err = fmaf(err2, err2, err);
        err = fmaf(err3, err3, err);
    }

    return err;
}
#endif

static __device__ __forceinline__ void quantize_nvfp4_subblock(
        const float vals[QK_NVFP4_SUB], float inv_col_scale, uint8_t & fp8_code, uint32_t & q0, uint32_t & q1) {
    q0 = 0;
    q1 = 0;

    float amax_sub = 0.0f;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; ++k) {
        amax_sub = fmaxf(amax_sub, fabsf(vals[k] * inv_col_scale));
    }

    static constexpr int test_offsets[5] = { 0, -1, 1, -2, 2 };
    const int first_fp8_code = (int) ggml_cuda_fp32_to_ue4m3(amax_sub / 6.0f);

    fp8_code = (uint8_t) first_fp8_code;
    float subblock_scale = ggml_cuda_ue4m3_to_fp32(fp8_code);
    float inv_scale_err = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
#if CUDART_VERSION >= 12080
    float best_err = nvfp4_native_scale_error(vals, inv_col_scale, inv_scale_err, subblock_scale);
#else
    float best_err = 0.0f;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; ++k) {
        const float v = vals[k] * inv_col_scale;
        const uint8_t q = ggml_cuda_float_to_fp4_e2m1(v, inv_scale_err);
        const float err_diff = fabsf(v) - fabsf(kvalues_fp4[q & 0x7]) * subblock_scale;
        best_err = fmaf(err_diff, err_diff, best_err);
    }
#endif // CUDART_VERSION >= 12080

#pragma unroll
    for (int i = 1; i < 5; ++i) {
        const int test_code = first_fp8_code + test_offsets[i];
        if (test_code < 0 || test_code > 0x7e) {
            continue;
        }

        const float test_scale = ggml_cuda_ue4m3_to_fp32((uint8_t) test_code);
        const float test_inv_scale = test_scale > 0.0f ? 0.5f / test_scale : 0.0f;
#if CUDART_VERSION >= 12080
        const float cur_err = nvfp4_native_scale_error(vals, inv_col_scale, test_inv_scale, test_scale);
#else
        float cur_err = 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) {
            const float v = vals[k] * inv_col_scale;
            const uint8_t q = ggml_cuda_float_to_fp4_e2m1(v, test_inv_scale);
            const float err_diff = fabsf(v) - fabsf(kvalues_fp4[q & 0x7]) * test_scale;
            cur_err = fmaf(err_diff, err_diff, cur_err);
        }
#endif // CUDART_VERSION >= 12080

        if (cur_err < best_err) {
            best_err = cur_err;
            fp8_code = (uint8_t) test_code;
            subblock_scale = test_scale;
        }
    }
#if CUDART_VERSION >= 12080
    const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
    const float s = inv_col_scale * inv_scale;

    __nv_fp4x4_e2m1 q0_lo(make_float4(vals[0] * s, vals[8]  * s, vals[1] * s, vals[9]  * s));
    __nv_fp4x4_e2m1 q0_hi(make_float4(vals[2] * s, vals[10] * s, vals[3] * s, vals[11] * s));
    __nv_fp4x4_e2m1 q1_lo(make_float4(vals[4] * s, vals[12] * s, vals[5] * s, vals[13] * s));
    __nv_fp4x4_e2m1 q1_hi(make_float4(vals[6] * s, vals[14] * s, vals[7] * s, vals[15] * s));

    const char2 q0_lo_c = *reinterpret_cast<char2 *>(&q0_lo);
    const char2 q0_hi_c = *reinterpret_cast<char2 *>(&q0_hi);
    const char2 q1_lo_c = *reinterpret_cast<char2 *>(&q1_lo);
    const char2 q1_hi_c = *reinterpret_cast<char2 *>(&q1_hi);

    q0 = uint32_t(uint8_t(q0_lo_c.x)) | (uint32_t(uint8_t(q0_lo_c.y)) <<  8) |
        (uint32_t(uint8_t(q0_hi_c.x)) << 16) | (uint32_t(uint8_t(q0_hi_c.y)) << 24);
    q1 = uint32_t(uint8_t(q1_lo_c.x)) | (uint32_t(uint8_t(q1_lo_c.y)) <<  8) |
        (uint32_t(uint8_t(q1_hi_c.x)) << 16) | (uint32_t(uint8_t(q1_hi_c.y)) << 24);
#else
    const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB / 4; ++k) {
        q0 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 0] * inv_col_scale, inv_scale)) << (8 * k);
        q0 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 8] * inv_col_scale, inv_scale)) << (8 * k + 4);
        q1 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 4] * inv_col_scale, inv_scale)) << (8 * k);
        q1 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 12] * inv_col_scale, inv_scale)) << (8 * k + 4);
    }
#endif // CUDART_VERSION >= 12080
}
#endif
