#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

bool ggml_cuda_op_ssm_conv_qk_norm(ggml_backend_cuda_context & ctx,
                                   const ggml_tensor *         conv,
                                   ggml_tensor *               q,
                                   ggml_tensor *               k,
                                   ggml_tensor *               v,
                                   float                       eps,
                                   float                       scale);

bool ggml_cuda_op_ssm_conv_bf16_qk(ggml_backend_cuda_context & ctx,
                                   const ggml_tensor *         conv,
                                   ggml_tensor *               q,
                                   ggml_tensor *               k,
                                   ggml_tensor *               v,
                                   float                       eps,
                                   float                       scale);
