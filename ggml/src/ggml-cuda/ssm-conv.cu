#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 7:  launch_kernel(std::integral_constant<int, 7 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 7, 9, 15 right now.");
    }
}

template <int nc, bool silu>
static __global__ void ssm_conv_split_f32(
        const float * x, const float * history, const float * weight, const float * bias, float * out,
        int64_t x0, int64_t x1, int64_t x2, int64_t h0, int64_t h1, int64_t h2, int64_t w0, int64_t w1,
        int channels, int tokens) {
    const int ch = blockIdx.x*blockDim.x + threadIdx.x;
    if (ch >= channels) {
        return;
    }
    const int seq = blockIdx.y;
    const int begin = blockIdx.z*32;
    const int end = min(begin + 32, tokens);
    float values[nc], weights[nc];
#pragma unroll
    for (int j = 0; j < nc; ++j) {
        const int pos = begin + j - (nc - 1);
        values[j] = pos < 0 ? history[(pos + nc - 1)*h0 + ch*h1 + seq*h2] : x[pos*x0 + ch*x1 + seq*x2];
        weights[j] = weight[j*w0 + ch*w1];
    }
    for (int t = begin; t < end; ++t) {
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < nc; ++j) {
            sum += values[j]*weights[j];
        }
        sum += bias ? bias[ch] : 0.0f;
        out[(int64_t(seq)*tokens + t)*channels + ch] = silu ? ggml_cuda_op_silu_single(sum) : sum;
        if (t + 1 < end) {
#pragma unroll
            for (int j = 0; j < nc - 1; ++j) {
                values[j] = values[j + 1];
            }
            values[nc - 1] = x[(t + 1)*x0 + ch*x1 + seq*x2];
        }
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    if (dst->op == GGML_OP_SSM_CONV_SPLIT) {
        const ggml_tensor * h = dst->src[2];
        const dim3 blocks((nr + 127)/128, n_s, (n_t + 31)/32);
        auto launch = [&](auto NC, auto SILU) {
            ssm_conv_split_f32<decltype(NC)::value, decltype(SILU)::value><<<blocks, 128, 0, ctx.stream()>>>(
                    (const float *) src0->data, (const float *) h->data, (const float *) src1->data,
                    bias ? (const float *) bias->data : nullptr, (float *) out->data,
                    src0->nb[0]/4, src0->nb[1]/4, src0->nb[2]/4, h->nb[0]/4, h->nb[1]/4, h->nb[2]/4,
                    src1->nb[0]/4, src1->nb[1]/4, nr, n_t);
        };
        auto select = [&](auto NC) {
            if (fuse_silu) { launch(NC, std::true_type{}); }
            else           { launch(NC, std::false_type{}); }
        };
        switch (nc) {
            case 3:  select(std::integral_constant<int, 3>{}); break;
            case 4:  select(std::integral_constant<int, 4>{}); break;
            case 5:  select(std::integral_constant<int, 5>{}); break;
            case 7:  select(std::integral_constant<int, 7>{}); break;
            case 9:  select(std::integral_constant<int, 9>{}); break;
            case 15: select(std::integral_constant<int, 15>{}); break;
            default: GGML_ABORT("Unsupported split convolution kernel");
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

static __global__ void ssm_conv_qk_norm_f32(const float * x,
                                            const float * history,
                                            const float * weight,
                                            float *       q,
                                            float *       k,
                                            float *       v,
                                            int           channels,
                                            int           tokens,
                                            int           q_heads,
                                            float         eps,
                                            float         scale_out) {
    const int lane = threadIdx.x, head = blockIdx.x, seq = blockIdx.y;
    const int ch    = 128 * head + 4 * lane;
    const int begin = 4 * blockIdx.z;
    const int token = begin + threadIdx.y;
    if (token >= tokens) {
        return;
    }
    float4 y = make_float4(0, 0, 0, 0);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float4    value;
        const int pos = token + j - 3;
        if (pos < 0) {
            const auto * h = history + int64_t(seq) * channels * 3 + ch * 3 + pos + 3;
            value          = make_float4(h[0], h[3], h[6], h[9]);
        } else {
            value = *reinterpret_cast<const float4 *>(x + (int64_t(seq) * tokens + pos) * channels + ch);
        }
        const auto * w = weight + ch * 4 + j;
        y.x += value.x * w[0];
        y.y += value.y * w[4];
        y.z += value.z * w[8];
        y.w += value.w * w[12];
    }
    y = make_float4(ggml_cuda_op_silu_single(y.x + 0.0f), ggml_cuda_op_silu_single(y.y + 0.0f),
                    ggml_cuda_op_silu_single(y.z + 0.0f), ggml_cuda_op_silu_single(y.w + 0.0f));
    if (head < 2 * q_heads) {
        float sum = 0.0f;
        sum += y.x * y.x + y.y * y.y + y.z * y.z + y.w * y.w;
        sum              = warp_reduce_sum(sum);
        const float  inv = rsqrtf(sum / 128.0f + eps);
        const float4 out = make_float4(scale_out * (inv * y.x), scale_out * (inv * y.y), scale_out * (inv * y.z),
                                       scale_out * (inv * y.w));
        float *      dst = head < q_heads ? q : k;
        const int    h   = head < q_heads ? head : head - q_heads;
        *reinterpret_cast<float4 *>(dst + ((int64_t(seq) * tokens + token) * q_heads + h) * 128 + 4 * lane) = out;
    } else {
        *reinterpret_cast<float4 *>(v + (int64_t(seq) * tokens + token) * channels + (head - 2 * q_heads) * 128 +
                                    4 * lane) = y;
    }
}

bool ggml_cuda_op_ssm_conv_qk_norm(ggml_backend_cuda_context & ctx,
                                   const ggml_tensor *         conv,
                                   ggml_tensor *               q,
                                   ggml_tensor *               k,
                                   ggml_tensor *               v,
                                   float                       eps,
                                   float                       scale) {
    const auto *  x        = conv->src[0];
    const auto *  w        = conv->src[1];
    const auto *  history  = conv->src[2];
    const int64_t channels = conv->ne[0], tokens = conv->ne[1], seqs = conv->ne[2];
    if (!GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[ctx.device].cc) || channels <= 0 || channels > INT_MAX ||
        channels % 128 || tokens < 64 || tokens > 4 * 65535 || seqs < 1 || seqs > 65535 ||
        channels > INT64_MAX / tokens / seqs / 4 || conv->ne[3] != 1 || x->type != GGML_TYPE_F32 ||
        w->type != GGML_TYPE_F32 || history->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != channels ||
        !ggml_is_matrix(w) || !ggml_is_contiguous(w) || x->ne[0] != tokens || x->ne[1] != channels ||
        x->ne[2] != seqs || x->ne[3] != 1 || x->nb[0] != size_t(channels) * ggml_element_size(x) ||
        x->nb[1] != ggml_element_size(x) || x->nb[2] != size_t(tokens * channels) * ggml_element_size(x) ||
        !ggml_is_contiguous(history) || history->ne[0] != 3 || history->ne[1] != channels || history->ne[2] != seqs ||
        history->ne[3] != 1 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(q) || !ggml_is_contiguous(k) || !ggml_are_same_shape(q, k) || q->ne[0] != 128 ||
        q->ne[1] < 1 || q->ne[1] >= channels / 256 || q->ne[2] != tokens || q->ne[3] != seqs || v->ne[0] != 128 ||
        2 * q->ne[1] + v->ne[1] != channels / 128 || v->ne[2] != tokens || v->ne[3] != seqs || v->nb[0] != 4 ||
        v->nb[1] != 512 || v->nb[2] != size_t(channels) * 4 || v->nb[3] != size_t(channels * tokens) * 4) {
        return false;
    }
    const ggml_tensor * tensors[] = { x, w, history, q, k, v };
    for (const auto * t : tensors) {
        if (reinterpret_cast<uintptr_t>(t->data) % 16) {
            return false;
        }
    }
    ssm_conv_qk_norm_f32<<<dim3(channels / 128, seqs, (tokens + 3) / 4), dim3(32, 4), 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) history->data, (const float *) w->data, (float *) q->data,
        (float *) k->data, (float *) v->data, channels, tokens, q->ne[1], eps, scale);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static __device__ __forceinline__ float4 round_bf16_qk(float4 x) {
    return make_float4(__bfloat162float(__float2bfloat16_rn(x.x)), __bfloat162float(__float2bfloat16_rn(x.y)),
                       __bfloat162float(__float2bfloat16_rn(x.z)), __bfloat162float(__float2bfloat16_rn(x.w)));
}

static __device__ __forceinline__ void store_bf16_qk(nv_bfloat16 * dst, float4 x) {
    const auto a = __float22bfloat162_rn(make_float2(x.x, x.y));
    const auto b = __float22bfloat162_rn(make_float2(x.z, x.w));
    *reinterpret_cast<uint2 *>(dst) =
        make_uint2(*reinterpret_cast<const uint32_t *>(&a), *reinterpret_cast<const uint32_t *>(&b));
}

static __global__ void conv_bf16_qk(const float * x,
                                    const float * history,
                                    const float * weight,
                                    nv_bfloat16 * q,
                                    nv_bfloat16 * k,
                                    nv_bfloat16 * v,
                                    int           channels,
                                    int           tokens,
                                    int           q_heads,
                                    int           v_stride,
                                    float         eps,
                                    float         scale_out) {
    const int lane = threadIdx.x, head = 4 * blockIdx.x + threadIdx.y;
    if (128 * head >= channels) {
        return;
    }
    const int seq = blockIdx.y, ch = 128 * head + 4 * lane;
    const int begin = 32 * blockIdx.z, end = min(begin + 32, tokens);
    float4    values[4], weights[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int pos = begin + j - 3;
        if (pos < 0) {
            const auto * h = history + int64_t(seq) * channels * 3 + ch * 3 + pos + 3;
            values[j]      = make_float4(h[0], h[3], h[6], h[9]);
        } else {
            values[j] = *reinterpret_cast<const float4 *>(x + (int64_t(seq) * tokens + pos) * channels + ch);
        }
        const auto * w = weight + ch * 4 + j;
        weights[j]     = make_float4(w[0], w[4], w[8], w[12]);
    }
    for (int t = begin; t < end; ++t) {
        float4 y = make_float4(0, 0, 0, 0);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            y.x += values[j].x * weights[j].x;
            y.y += values[j].y * weights[j].y;
            y.z += values[j].z * weights[j].z;
            y.w += values[j].w * weights[j].w;
        }
        // Preserve the convolution's separate zero-bias addition and native SiLU.
        y = make_float4(ggml_cuda_op_silu_single(y.x + 0.0f), ggml_cuda_op_silu_single(y.y + 0.0f),
                        ggml_cuda_op_silu_single(y.z + 0.0f), ggml_cuda_op_silu_single(y.w + 0.0f));
        y = round_bf16_qk(y);
        if (head < 2 * q_heads) {
            float sum = 0.0f;
            sum += y.x * y.x + y.y * y.y + y.z * y.z + y.w * y.w;
            sum               = warp_reduce_sum(sum);
            const float   inv = rsqrtf(sum / 128.0f + eps);
            const float4  out = make_float4(scale_out * (inv * y.x), scale_out * (inv * y.y), scale_out * (inv * y.z),
                                            scale_out * (inv * y.w));
            nv_bfloat16 * dst = head < q_heads ? q : k;
            const int     h   = head < q_heads ? head : head - q_heads;
            store_bf16_qk(dst + ((int64_t(seq) * tokens + t) * q_heads + h) * 128 + 4 * lane, out);
        } else {
            store_bf16_qk(v + ((int64_t(seq) * tokens + t) * v_stride + (head - 2 * q_heads) * 128 + 4 * lane), y);
        }
        if (t + 1 < end) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                values[j] = values[j + 1];
            }
            values[3] = *reinterpret_cast<const float4 *>(x + (int64_t(seq) * tokens + t + 1) * channels + ch);
        }
    }
}

bool ggml_cuda_op_ssm_conv_bf16_qk(ggml_backend_cuda_context & ctx,
                                   const ggml_tensor *         conv,
                                   ggml_tensor *               q,
                                   ggml_tensor *               k,
                                   ggml_tensor *               v,
                                   float                       eps,
                                   float                       scale) {
    const auto *  x        = conv->src[0];
    const auto *  w        = conv->src[1];
    const auto *  history  = conv->src[2];
    const int64_t channels = conv->ne[0], tokens = conv->ne[1], seqs = conv->ne[2];
    if (!GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[ctx.device].cc) || channels <= 0 || channels > INT_MAX ||
        channels % 128 || tokens < 64 || tokens > 32 * 65535 || seqs < 1 || seqs > 65535 ||
        channels > INT64_MAX / tokens / seqs / 4 || conv->ne[3] != 1 || x->type != GGML_TYPE_F32 ||
        w->type != GGML_TYPE_F32 || history->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != channels ||
        !ggml_is_matrix(w) || !ggml_is_contiguous(w) || x->ne[0] != tokens || x->ne[1] != channels ||
        x->ne[2] != seqs || x->ne[3] != 1 || x->nb[0] != size_t(channels) * 4 || x->nb[1] != 4 ||
        x->nb[2] != size_t(tokens * channels) * 4 || !ggml_is_contiguous(history) || history->ne[0] != 3 ||
        history->ne[1] != channels || history->ne[2] != seqs || history->ne[3] != 1 || q->type != GGML_TYPE_BF16 ||
        k->type != GGML_TYPE_BF16 || v->type != GGML_TYPE_BF16 || !ggml_is_contiguous(q) || !ggml_is_contiguous(k) ||
        !ggml_are_same_shape(q, k) || q->ne[0] != 128 || q->ne[1] < 1 || q->ne[1] >= channels / 256 ||
        q->ne[2] != tokens || q->ne[3] != seqs || v->ne[0] != 128 || 2 * q->ne[1] + v->ne[1] != channels / 128 ||
        v->ne[2] != tokens || v->ne[3] != seqs || v->nb[0] != 2 || v->nb[1] != 256 ||
        v->nb[2] != size_t(channels) * 2 || v->nb[3] != size_t(channels * tokens) * 2) {
        return false;
    }
    const ggml_tensor * tensors[] = { x, w, history, q, k, v };
    for (const auto * t : tensors) {
        if (reinterpret_cast<uintptr_t>(t->data) % 16) {
            return false;
        }
    }
    conv_bf16_qk<<<dim3((channels / 128 + 3) / 4, seqs, (tokens + 31) / 32), dim3(32, 4), 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) history->data, (const float *) w->data, (nv_bfloat16 *) q->data,
        (nv_bfloat16 *) k->data, (nv_bfloat16 *) v->data, channels, tokens, q->ne[1], channels, eps, scale);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
