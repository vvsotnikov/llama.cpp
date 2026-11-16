#include "gated-delta-net-bf16.cuh"

#ifdef GGML_CUDA_GDN_BF16
extern "C" int ggml_gdn_bf16_arch();
extern "C" int ggml_gdn_bf16_init();
extern "C" int ggml_gdn_bf16_launch(void *,
                                    void *,
                                    void *,
                                    void *,
                                    void *,
                                    void *,
                                    void *,
                                    void *,
                                    void *,
                                    void *,
                                    int32_t,
                                    int32_t,
                                    int32_t,
                                    int64_t,
                                    int64_t,
                                    float,
                                    cudaStream_t);

static __global__ void gdn_bf16_alpha(const float * log_decay, float * decay, int32_t * seq, int tokens, int heads) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < int64_t(tokens) * heads) {
        decay[i] = expf(log_decay[i]);
    }
    if (i == 0) {
        seq[0] = 0;
        seq[1] = tokens;
    }
}

static __global__ void gdn_bf16_widen(const nv_bfloat16 * input, float * output, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = __bfloat162float(input[i]);
    }
}
#endif

bool ggml_cuda_gdn_bf16_supported(int device, const ggml_tensor * dst) {
#ifdef GGML_CUDA_GDN_BF16
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    if (ggml_cuda_info().devices[device].cc != ggml_gdn_bf16_arch() || q->type != GGML_TYPE_BF16 ||
        k->type != GGML_TYPE_BF16 || v->type != GGML_TYPE_BF16 || dst->type != GGML_TYPE_F32 ||
        dst->src[3]->type != GGML_TYPE_F32 || dst->src[4]->type != GGML_TYPE_F32 ||
        dst->src[5]->type != GGML_TYPE_F32 || q->ne[0] != 128 || k->ne[0] != 128 || v->ne[0] != 128 ||
        q->ne[1] != k->ne[1] || q->ne[1] < 1 || v->ne[1] < q->ne[1] || v->ne[1] % q->ne[1] || v->ne[1] > 128 ||
        v->ne[2] < 64 || v->ne[2] > 4096 || q->ne[2] != v->ne[2] || k->ne[2] != v->ne[2] || q->ne[3] != 1 ||
        k->ne[3] != 1 || v->ne[3] != 1 || dst->src[3]->ne[0] != 1 || ggml_get_op_params_i32(dst, 0) != 1 ||
        dst->src[8] != nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(q) || !ggml_is_contiguous(k) || !ggml_are_same_stride(q, k) || v->nb[0] != 2 ||
        v->nb[1] != 256 || v->nb[2] < 256 * v->ne[1] || v->nb[2] % 16 || !ggml_is_contiguous(dst->src[3]) ||
        !ggml_is_contiguous(dst->src[4]) || !ggml_is_contiguous(dst->src[5])) {
        return false;
    }
    return dst->src[6] == nullptr || v->ne[2] > dst->src[6]->ne[1] / 2;
#else
    GGML_UNUSED_VARS(device, dst);
    return false;
#endif
}

void ggml_cuda_gdn_bf16_init(int device) {
#ifdef GGML_CUDA_GDN_BF16
    if (ggml_cuda_info().devices[device].cc == ggml_gdn_bf16_arch()) {
        GGML_ASSERT(ggml_gdn_bf16_init() == 0);
    }
#else
    GGML_UNUSED(device);
#endif
}

void ggml_cuda_gdn_bf16_run(ggml_backend_cuda_context & ctx,
                            const ggml_tensor *         dst,
                            const float *               initial,
                            float *                     state) {
#ifdef GGML_CUDA_GDN_BF16
    GGML_ASSERT(ggml_cuda_gdn_bf16_supported(ctx.device, dst));
    const auto *  q      = dst->src[0];
    const auto *  k      = dst->src[1];
    const auto *  v      = dst->src[2];
    const int     tokens = v->ne[2], heads = v->ne[1];
    const int64_t values = int64_t(tokens) * heads * 128;
    for (const auto * t : { q, k, v }) {
        GGML_ASSERT((uintptr_t(t->data) & 15) == 0);
    }
    ggml_cuda_pool_alloc<nv_bfloat16> output(ctx.pool(), values);
    ggml_cuda_pool_alloc<float>       decay(ctx.pool(), int64_t(tokens) * heads);
    ggml_cuda_pool_alloc<int32_t>     seq(ctx.pool(), 2);
    ggml_cuda_pool_alloc<uint8_t>     workspace(ctx.pool(), ggml_cuda_info().devices[ctx.device].nsm * 128);
    GGML_ASSERT((uintptr_t(workspace.get()) & 127) == 0);
    auto stream = ctx.stream();
    gdn_bf16_alpha<<<(int64_t(tokens) * heads + 255) / 256, 256, 0, stream>>>((const float *) dst->src[3]->data,
                                                                              decay.get(), seq.get(), tokens, heads);
    GGML_ASSERT(ggml_gdn_bf16_launch(q->data, k->data, v->data, output.get(), decay.get(), dst->src[4]->data,
                                     (void *) initial, state, workspace.get(), seq.get(), tokens, q->ne[1], heads,
                                     v->nb[1] / 2, v->nb[2] / 2, 1.0f / sqrtf(128.0f), stream) == 0);
    gdn_bf16_widen<<<(values + 255) / 256, 256, 0, stream>>>(output.get(), (float *) dst->data, values);
    CUDA_CHECK(cudaGetLastError());
    static bool logged = false;
    if (!logged) {
        GGML_LOG_INFO("CUDA BF16 GDN: tokens=%d, heads=%d, F32 recurrent state\n", tokens, heads);
        logged = true;
    }
#else
    GGML_UNUSED_VARS(ctx, dst, initial, state);
    GGML_ABORT("BF16 GDN provider is not configured");
#endif
}
