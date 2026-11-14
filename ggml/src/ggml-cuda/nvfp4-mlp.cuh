#pragma once

#include "common.cuh"

bool ggml_cuda_nvfp4_mlp_available(int cc);
bool ggml_cuda_nvfp4_mlp_precision(const ggml_tensor * mm, int cc);
size_t ggml_cuda_nvfp4_mlp_weight_bytes(int64_t k, int64_t n);
void ggml_cuda_nvfp4_mlp_prepare(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down, void * data, cudaStream_t stream);
bool ggml_cuda_nvfp4_mlp(ggml_backend_cuda_context & ctx, const ggml_tensor * input,
        const ggml_tensor * up_scale, const ggml_tensor * gate_scale,
        const ggml_tensor * down, ggml_tensor * output, const void * prepared,
        const ggml_tensor * norm = nullptr, const ggml_tensor * gamma = nullptr, ggml_tensor * residual = nullptr, const ggml_tensor * up_calibration = nullptr,
        const ggml_tensor * gate_calibration = nullptr, const ggml_tensor * down_calibration = nullptr);
