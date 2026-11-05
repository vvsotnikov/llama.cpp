#include "iq3-fragment.cuh"
#include "mma.cuh"
#include "quantize.cuh"
using namespace ggml_cuda_mma;
// Each record holds 16 rows by 256 coefficients in native IQ3_S byte capacity.
// Codes are already in warp/MMA register order. No shared-memory weight ring or ldmatrix is needed.
template <int W, int S = 1>
static __global__ void iq3_fragment_mma(const uint32_t * packed, const block_q8_1 * y, float * out, int k, int n, int stride_y, int stride_out) {
    int lane = threadIdx.x, warp = threadIdx.y;
    int g = lane/4, t = lane%4, nb = k/256;
    float total[4] = {};
    for (int kb = blockIdx.y*W + warp; kb < nb; kb += W*S) {
        const uint32_t * record = packed + (blockIdx.x*nb+kb)*440;
        uint32_t m0 = (record[416+g/2] >> (16*(g%2))) & 65535, m1 = (record[420+g/2] >> (16*(g%2))) & 65535;
        uint32_t sl0 = record[424+g], sl1 = record[432+g];
        float d0 = __half2float(__ushort_as_half(m0 & 0xffff));
        float d1 = __half2float(__ushort_as_half(m1 & 0xffff));
        float partial[4] = {};
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const uint32_t * tile_codes=record+s*52;
            uint32_t low=tile_codes[lane];
            int bit=20*lane, word=bit/32, shift=bit%32;
            uint32_t extra=__funnelshift_r(tile_codes[32+word],tile_codes[32+(word+1)%20],shift);
            tile<16, 8, int> A;
#pragma unroll
            for (int i=0;i<4;++i) {
                int index=((low>>(8*i))&255) | (((extra>>i)&1)<<8);
                int signs=(extra>>(4+4*i))&15;
                int mask=__vcmpne4(((signs&3)<<7)|((signs&12)<<21),0);
                int coeff=iq3s_grid[index]; A.x[i]=__vsub4(coeff^mask, mask);
            }
            const block_q8_1 & b = y[g*stride_y+kb*8+s];
            tile<8, 8, int> B;
            B.x[0] = ((const int *) b.qs)[t];
            B.x[1] = ((const int *) b.qs)[t+4];
            tile<16, 8, int> C;
            mma(C, A, B);
            int ls0 = 1 + 2*((sl0 >> (4*s)) & 15);
            int ls1 = 1 + 2*((sl1 >> (4*s)) & 15);
            float dy0 = __low2float(y[(2*t)*stride_y+kb*8+s].ds);
            float dy1 = __low2float(y[(2*t+1)*stride_y+kb*8+s].ds);
            partial[0] += dy0*float(C.x[0]*ls0);
            partial[1] += dy1*float(C.x[1]*ls0);
            partial[2] += dy0*float(C.x[2]*ls1);
            partial[3] += dy1*float(C.x[3]*ls1);
        }
        total[0] += d0*partial[0]; total[1] += d0*partial[1];
        total[2] += d1*partial[2]; total[3] += d1*partial[3];
    }
    __shared__ float parts[W][4][32];
#pragma unroll
    for (int i = 0; i < 4; ++i) parts[warp][i][lane] = total[i];
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int w = 1; w < W; ++w) {
#pragma unroll
            for (int i = 0; i < 4; ++i) total[i] += parts[w][i][lane];
        }
        out += blockIdx.y*stride_out*8;
        int row = blockIdx.x*16+g;
        out[(2*t)*stride_out+row] = total[0]; out[(2*t+1)*stride_out+row] = total[1];
        out[(2*t)*stride_out+row+8] = total[2]; out[(2*t+1)*stride_out+row+8] = total[3];
    }
}

static __global__ void iq3_fragment_reduce(const float * parts, float * out, int count) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= count) return;
    float value = parts[i];
#pragma unroll
    for (int s = 1; s < 4; ++s) value += parts[s*count+i];
    out[i] = value;
}

static __device__ __forceinline__ uint32_t iq3_fragment_extra(const block_iq3_s * src, int tile, int kb, int nb, int s, int lane) {
    int g=lane/4, t=lane%4; uint32_t extra=0;
#pragma unroll
    for (int i=0;i<4;++i) {
        const auto & b=src[(tile*16+g+8*(i%2))*nb+kb];
        int index=t+4*(i/2);
        extra|=uint32_t((b.qh[s]>>index)&1)<<i;
        extra|=uint32_t((b.signs[4*s+index/2]>>(4*(index%2)))&15)<<(4+4*i);
    }
    return extra;
}

static __global__ void iq3_fragment_pack(const block_iq3_s * src, uint32_t * dst, int nb, int n) {
    const size_t word = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const size_t total = size_t(n/16)*nb*440;
    if (word>=total)return;
    int record=word/440, w=word%440, tile=record/nb, kb=record%nb;
    uint32_t v=0;
    if (w<416) {
        int s=w/52, part=w%52;
        if (part<32) {
            int g=part/4, t=part%4;
#pragma unroll
            for (int i=0;i<4;++i) {
                const auto & b=src[(tile*16+g+8*(i%2))*nb+kb];
                v|=uint32_t(b.qs[8*s+t+4*(i/2)])<<(8*i);
            }
        } else {
            int bit=(part-32)*32, first=bit/20, last=(bit+31)/20;
            for (int lane=first;lane<=last;++lane) {
                uint32_t e=iq3_fragment_extra(src, tile, kb, nb, s, lane);
                int shift=20*lane-bit;
                v|=shift>=0?e<<shift:e>>(-shift);
            }
        }
    }else if (w<424) {
        int row=2*(w-416);
        v=uint32_t(__half_as_ushort(src[(tile*16+row)*nb+kb].d)) | (uint32_t(__half_as_ushort(src[(tile*16+row+1)*nb+kb].d))<<16);
    } else {
        memcpy(&v, src[(tile*16+w-424)*nb+kb].scales, 4);
    }
    dst[word]=v;
}

void ggml_cuda_iq3_fragment_prepare(const ggml_tensor * src, void * packed, cudaStream_t stream) {
    const size_t words=ggml_nbytes(src)/4;
    iq3_fragment_pack<<<(words+255)/256, 256, 0, stream>>>((const block_iq3_s *)src->data,(uint32_t *)packed, src->ne[0]/256, src->ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_iq3_fragment_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, ggml_tensor * dst, const void * packed) {
    const int64_t k=src0->ne[0],n=src0->ne[1],padded=GGML_PAD(k, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> q(ctx.pool(),8*padded*sizeof(block_q8_1)/QK8_1);
    quantize_row_q8_1_cuda((const float *)src1->data, nullptr, q.get(),src0->type, k,
        src1->nb[1]/sizeof(float),src1->nb[2]/sizeof(float),src1->nb[3]/sizeof(float),padded, 8, 1, 1, ctx.stream());
    if (k >= 8192) {
        ggml_cuda_pool_alloc<float> parts(ctx.pool(), n*8*4);
        iq3_fragment_mma<4, 4><<<dim3(n/16, 4),dim3(32, 4),0, ctx.stream()>>>((const uint32_t *)packed,
            (const block_q8_1 *)q.get(),parts.get(),k, n, padded/QK8_1, n);
        iq3_fragment_reduce<<<(n*8+255)/256, 256, 0, ctx.stream()>>>(parts.get(),(float *)dst->data, n*8);
    } else {
        iq3_fragment_mma<8><<<n/16, dim3(32, 8),0, ctx.stream()>>>((const uint32_t *)packed,
            (const block_q8_1 *)q.get(),(float *)dst->data, k, n, padded/QK8_1, dst->nb[1]/sizeof(float));
    }
    CUDA_CHECK(cudaGetLastError());
}
