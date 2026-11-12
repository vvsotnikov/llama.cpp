#include "prefill-iq3.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

static __global__ void iq3_prefill_pack(const block_iq3_s * src, ggml_cuda_iq3_prefill_block * dst, int blocks) {
#ifdef TURING_MMA_AVAILABLE
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int ib = index / 8, q = index % 8;
    if (ib >= blocks) {
        return;
    }
    const auto & b = src[ib];
    const int2 indices = make_int2(get_int_b2(b.qs, 2 * q), get_int_b2(b.qs, 2 * q + 1));
    const auto * qs = (const uint8_t *) &indices;
    const int high = b.qh[q], signs = get_int_b2(b.signs, q);
    const auto * sign = (const uint8_t *) &signs;
#pragma unroll
    for (int l = 0; l < QR3_S; ++l) {
        const int a = iq3s_grid[qs[2 * l] | ((high << (8 - 2 * l)) & 0x100)];
        const int b = iq3s_grid[qs[2 * l + 1] | ((high << (7 - 2 * l)) & 0x100)];
        const int sa = __vcmpne4(((sign[l] & 3) << 7) | ((sign[l] & 12) << 21), 0);
        const int sb = __vcmpne4(((sign[l] & 48) << 3) | ((sign[l] & 192) << 17), 0);
        const int va = __vsub4(a ^ sa, sa), vb = __vsub4(b ^ sb, sb);
        dst[ib].qs[4 * q + l] = ((__vadd4(va, 0x0f0f0f0f) >> 1) & 0x0f0f0f0f) |
            (((__vadd4(vb, 0x0f0f0f0f) >> 1) & 0x0f0f0f0f) << 4);
    }
    const int scale = 1 + 2 * ((b.scales[q / 2] >> (((2 * q) << 1) & 4)) & 15);
    dst[ib].d[q] = scale * __half2float(b.d);
#else
    GGML_UNUSED(src); GGML_UNUSED(dst); GGML_UNUSED(blocks);
    NO_DEVICE_CODE;
#endif
}

void ggml_cuda_iq3_prefill_prepare(const ggml_tensor * weight, void * data, cudaStream_t stream) {
    const int blocks = ggml_nelements(weight) / QK_K;
    iq3_prefill_pack<<<(blocks * 8 + 255) / 256, 256, 0, stream>>>(
            (const block_iq3_s *) weight->data, (ggml_cuda_iq3_prefill_block *) data, blocks);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_iq3_prefill_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * weight,
        const ggml_tensor * input, ggml_tensor * dst, const void * data) {
    const int k = weight->ne[0], n = weight->ne[1], m = input->ne[1];
    const auto & device = ggml_cuda_info().devices[ctx.device];
    const bool fallback = ggml_cuda_mmq_needs_fallback(n);
    int threads = 0;
    const int j = ggml_cuda_mmq_get_J_best(GGML_TYPE_IQ3_S, m, fallback,
            device.cc, device.smpbo, GGML_PREC_Q8, &threads);
    const int padded_k = GGML_PAD(k, MATRIX_ROW_PADDING);
    const size_t chunk = threads * sizeof(int);
    const size_t padding = GGML_PAD(j * sizeof(block_q8_1_mmq), chunk);
    ggml_cuda_pool_alloc<char> input_q8(ctx.pool(),
            size_t(m) * padded_k * sizeof(block_q8_1) / QK8_1 + padding);
    quantize_mmq_q8_1_cuda((const float *) input->data, nullptr, input_q8.get(), GGML_TYPE_IQ3_S,
            k, k, size_t(k) * m, size_t(k) * m, padded_k, m, 1, 1, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    const int64_t input_stride = int64_t(m) * padded_k * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const mmq_args args = {
        (const char *) data, GGML_TYPE_IQ3_S, (const int *) input_q8.get(), nullptr, nullptr, (float *) dst->data,
        nullptr, k, n, m, k / QK_K, m, n,
        1, 1, int64_t(k / QK_K) * n, input_stride, int64_t(n) * m,
        1, 1, int64_t(k / QK_K) * n, input_stride, int64_t(n) * m,
        m, j, {}};
    mul_mat_q_case<GGML_TYPE_IQ3_S, GGML_PREC_Q8, true>(ctx, args, ctx.stream());
}
