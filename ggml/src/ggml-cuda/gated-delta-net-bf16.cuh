#pragma once
#include "common.cuh"

bool ggml_cuda_gdn_bf16_supported(int device, const ggml_tensor * dst);
void ggml_cuda_gdn_bf16_init(int device);
void ggml_cuda_gdn_bf16_run(ggml_backend_cuda_context & ctx,
                            const ggml_tensor *         dst,
                            const float *               initial,
                            float *                     state);
