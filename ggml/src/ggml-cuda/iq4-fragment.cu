#include "iq4-fragment.cuh"
#include "mma.cuh"
#include "quantize.cuh"

using namespace ggml_cuda_mma;

__device__ __forceinline__ int iq4_fragment_lookup(uint32_t selector) {
    const uint32_t * table = (const uint32_t *) kvalues_iq4nl;
    uint32_t lo = __byte_perm(table[0], table[1], selector);
    uint32_t hi = __byte_perm(table[2], table[3], selector);
    return __byte_perm(lo, hi, 0x3210 | ((selector & 0x8888) >> 1));
}

static __global__ void iq4_fragment_mma(const uint32_t * packed, const int4 * y, float * out, int k, int n) {
    constexpr int W = 4, CG = 2, MT = 1;
    int lane = threadIdx.x, warp = threadIdx.y;
    int g = lane / 4, t = lane % 4, nb = k / 256;
    float total[MT][CG][4] = {};
    ggml_cuda_pdl_sync();
    for (int kb = warp; kb < nb; kb += W) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const uint32_t * record = packed + ((int64_t(blockIdx.x) * MT + m) * nb + kb) * 544;
            uint32_t m0 = record[512 + g], m1 = record[512 + g + 8];
            uint32_t sl0 = record[528 + g], sl1 = record[528 + g + 8];
            float d0 = __half2float(__ushort_as_half(m0 & 0xffff));
            float d1 = __half2float(__ushort_as_half(m1 & 0xffff));
            float partial[CG][4] = {};
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                uint32_t p0 = record[s * 64 + lane], p1 = record[s * 64 + 32 + lane];
                tile<16, 8, int> A;
                A.x[0] = iq4_fragment_lookup(p0); A.x[1] = iq4_fragment_lookup(p0 >> 16);
                A.x[2] = iq4_fragment_lookup(p1); A.x[3] = iq4_fragment_lookup(p1 >> 16);
                int ls0 = (((sl0 >> (4 * s)) & 15) | (((m0 >> (16 + 2 * s)) & 3) << 4)) - 32;
                int ls1 = (((sl1 >> (4 * s)) & 15) | (((m1 >> (16 + 2 * s)) & 3) << 4)) - 32;
#pragma unroll
                for (int c = 0; c < CG; ++c) {
                    const int4 * yp = y + (kb * CG + c) * MMQ_FEW_Q8_1_INT4;
                    const int4 q = yp[(s / 2) * 32 + lane];
                    const int4 sc = yp[4 * 32 + 4 * t + s / 2];
                    const float * scale = (const float *) &sc;
                    float dy0 = scale[2 * (s % 2)], dy1 = scale[2 * (s % 2) + 1];
                    tile<8, 8, int> B;
                    B.x[0] = s % 2 == 0 ? q.x : q.z;
                    B.x[1] = s % 2 == 0 ? q.y : q.w;
                    tile<16, 8, int> C;
                    mma(C, A, B);
                    partial[c][0] += dy0 * float(C.x[0] * ls0);
                    partial[c][1] += dy1 * float(C.x[1] * ls0);
                    partial[c][2] += dy0 * float(C.x[2] * ls1);
                    partial[c][3] += dy1 * float(C.x[3] * ls1);
                }
            }
#pragma unroll
            for (int c = 0; c < CG; ++c) {
                total[m][c][0] += d0 * partial[c][0]; total[m][c][1] += d0 * partial[c][1];
                total[m][c][2] += d1 * partial[c][2]; total[m][c][3] += d1 * partial[c][3];
            }
        }
    }
    ggml_cuda_pdl_lc();
    __shared__ float parts[W][MT][CG][4][32];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int c = 0; c < CG; ++c) {
#pragma unroll
            for (int i = 0; i < 4; ++i) parts[warp][m][c][i][lane] = total[m][c][i];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
#pragma unroll
            for (int c = 0; c < CG; ++c) {
#pragma unroll
                for (int w = 1; w < W; ++w) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) total[m][c][i] += parts[w][m][c][i][lane];
                }
                int row = (blockIdx.x * MT + m) * 16 + g;
                out[(8*c + 2*t)*n + row] = total[m][c][0]; out[(8*c + 2*t+1)*n + row] = total[m][c][1];
                out[(8*c + 2*t)*n + row+8] = total[m][c][2]; out[(8*c + 2*t+1)*n + row+8] = total[m][c][3];
            }
        }
    }
}


static __global__ void iq4_fragment_pack(const block_iq4_xs * src, uint32_t * dst, int nb, int64_t words) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= words) {
        return;
    }
    const int64_t record = i / 544;
    const int word = i % 544;
    const int kb = record % nb;
    const int row0 = (record / nb) * 16;
    uint32_t value;
    if (word >= 512) {
        const int row = row0 + (word - 512) % 16;
        const block_iq4_xs * bx = src + int64_t(row) * nb + kb;
        value = word < 528 ? *((const uint32_t *) bx) : *((const uint32_t *) bx->scales_l);
    } else {
        const int s = word / 64;
        const int half = (word % 64) / 32;
        const int lane = word % 32;
        const int g = lane / 4, t = lane % 4;
        value = 0;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_iq4_xs * bx = src + int64_t(row0 + g + 8 * r) * nb + kb;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const uint32_t q = (bx->qs[16 * s + 4 * t + e] >> (4 * half)) & 15;
                value |= q << (16 * r + 4 * e);
            }
        }
    }
    dst[i] = value;
}

void ggml_cuda_iq4_fragment_prepare(const ggml_tensor * src, void * packed, cudaStream_t stream) {
    const int64_t words = ggml_nbytes(src) / sizeof(uint32_t);
    iq4_fragment_pack<<<(words + 255) / 256, 256, 0, stream>>>((const block_iq4_xs *) src->data, (uint32_t *) packed, src->ne[0] / 256, words);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_iq4_fragment_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, ggml_tensor * dst, const void * packed) {
    const int k = src0->ne[0], n = src0->ne[1];
    GGML_ASSERT(src1->ne[1] == 16);
    ggml_cuda_pool_alloc<int4> q(ctx.pool(), int64_t(k / 256) * 2 * MMQ_FEW_Q8_1_INT4);
    quantize_mmq_few_q8_1_cuda((const float *) src1->data, q.get(), MMQ_Q8_1_DS_LAYOUT_D4,
        k, src1->nb[1] / sizeof(float), src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float), 16, 1, 1, ctx.stream());
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const ggml_cuda_kernel_launch_params params(dim3(n / 16), dim3(warp_size, 4), 0, ctx.stream());
    ggml_cuda_kernel_launch(iq4_fragment_mma, params, (const uint32_t *) packed, q.get(), (float *) dst->data, k, n);
}
