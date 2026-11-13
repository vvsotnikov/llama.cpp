#pragma once

#include "common.cuh"

bool ggml_cuda_mul_mat_fp8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * weight_scale = nullptr);

bool ggml_cuda_mul_mat_fp8_pipeline(
        ggml_backend_cuda_context & ctx, const ggml_tensor * input,
        const ggml_tensor * const * weights, const ggml_tensor * const * scales,
        ggml_tensor * const * outputs, int count, const ggml_tensor * producer = nullptr,
        const ggml_tensor * residual = nullptr, bool fuse_input_scale = false);
