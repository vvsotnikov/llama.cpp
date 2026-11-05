#pragma once
#include "common.cuh"
void ggml_cuda_iq3_fragment_prepare(const ggml_tensor * src, void * packed, cudaStream_t stream);
void ggml_cuda_iq3_fragment_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, ggml_tensor * dst, const void * packed);
