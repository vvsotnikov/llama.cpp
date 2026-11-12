#pragma once
#include "common.cuh"

struct ggml_cuda_iq3_prefill_block {
    int qs[32];
    float d[8];
};
static_assert(sizeof(ggml_cuda_iq3_prefill_block) == 160);

void ggml_cuda_iq3_prefill_prepare(const ggml_tensor * weight, void * data, cudaStream_t stream);
void ggml_cuda_iq3_prefill_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * weight,
        const ggml_tensor * input, ggml_tensor * dst, const void * data);
