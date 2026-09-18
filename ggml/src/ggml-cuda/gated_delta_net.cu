#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/cp-async.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       n_tokens_dst,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    // n_tokens_dst is the token count of the full dst tensor, n_tokens can be a tail of it
    attn_data += (sequence * n_tokens_dst * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// Chunked prefill path (non-KDA, S_v in {64, 128}, needs cp.async and 16 byte aligned q/k/v).
// Tokens are processed in chunks of GDN_CHUNK. Within a chunk the delta rule is solved in closed form
// (WY representation), the recurrent state is only carried across chunk boundaries.
// Math follows llm_build_delta_net_base::build_delta_net_chunking, state layout follows the recurrent kernel:
// state[col*S + d] = S[col][d], o = S q, S_t = a_t S_{t-1} + b_t (v_t - a_t S_{t-1} k_t) k_t^T.
// All smem row strides are multiples of 4 floats so that cp.async and float4 smem loads can be used.

#define GDN_CHUNK            64
#define GDN_CHUNK_THREADS   256
#define GDN_CHUNK_TILE_I      4 // token rows per thread in the tiled stages
#define GDN_CHUNK_STATE_COLS    64 // state columns per block in the state passing stage
#define GDN_CHUNK_STATE_THREADS 128
#define GDN_CHUNK_STATE_NCOL     4 // state columns per thread
#define GDN_CHUNK_STATE_PR      16 // rows per staged piece
#define GDN_CHUNK_STATE_SLOTS    6 // piece ring depth
#define GDN_CHUNK_CP         (GDN_CHUNK + 4) // padded stride of the 64x64 chunk matrices

struct gdn_chunk_params {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    int64_t sq1, sq2, sq3; // q/k strides (elements)
    int64_t sv1, sv2, sv3; // v strides
    int64_t sb1, sb2, sb3; // g/beta strides
    int     H;             // value heads
    int     H_k;           // q/k heads
    int     rq3;           // n_seqs / neq3
    int     n_full;        // tokens on the chunked path, multiple of GDN_CHUNK
    int     n_chunks;
    int     n_tokens;      // tokens of the full dst tensor
    float   scale;
    // workspace, all indexed by (seq*H + h)
    float * ws_g;  // [n_full]            cumulative gate within each chunk
    float * ws_w;  // [n_full][S]         W = T (beta exp(g) k)
    float * ws_u;  // [n_full][S]         U = T (beta v)
    float * ws_vn; // [n_full][S]         v_new = U - S0 W
    float * ws_h;  // [n_chunks][S][S]    state at chunk start
};

// cp_async_wait_group with a runtime count for the state kernel ring, n is clamped to the ring depth
static __device__ __forceinline__ void gdn_cp_async_wait_n(const int n) {
    switch (n) {
        case 0:  cp_async_wait_group<0>(); break;
        case 1:  cp_async_wait_group<1>(); break;
        case 2:  cp_async_wait_group<2>(); break;
        case 3:  cp_async_wait_group<3>(); break;
        default: cp_async_wait_group<4>(); break;
    }
}

static __device__ __forceinline__ void gdn_cp_async_16(float * dst, const float * src) {
    cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(dst), src);
}

// async copy of ROWS rows of S floats, dst row stride SP, src row stride stride_src (all multiples of 4)
template <int ROWS, int S, int SP, int THREADS = GDN_CHUNK_THREADS>
static __device__ __forceinline__ void gdn_cp_async_rows(float * dst, const float * src, const int64_t stride_src, const int tid) {
    static_assert((ROWS*S) % (4*THREADS) == 0, "bad copy split");
#pragma unroll
    for (int r = 0; r < ROWS*S/(4*THREADS); ++r) {
        const int idx = 4*(tid + r*THREADS);
        const int i   = idx / S;
        const int d   = idx % S;
        gdn_cp_async_16(dst + i*SP + d, src + i*stride_src + d);
    }
}

static __device__ __forceinline__ float gdn_dot4(const float4 a, const float4 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

template <int S>
static constexpr size_t gdn_chunk_smem_prepare() {
    // k, A, g, beta, beta*exp(g)
    return (GDN_CHUNK*(S + 4) + GDN_CHUNK*GDN_CHUNK_CP + 3*GDN_CHUNK) * sizeof(float);
}

#define GDN_CHUNK_OUT_KP 16 // k rows per piece in the output stage
#define GDN_CHUNK_OUT_TD 16 // state d tile width in the output stage

template <int S>
static constexpr __host__ __device__ int gdn_chunk_output_sbuf() {
    // two k pieces [KP][S+4] or two state tiles [S][TD+4]
    return 2*S*(GDN_CHUNK_OUT_TD + 4) > 2*GDN_CHUNK_OUT_KP*(S + 4) ? 2*S*(GDN_CHUNK_OUT_TD + 4) : 2*GDN_CHUNK_OUT_KP*(S + 4);
}

template <int S>
static constexpr size_t gdn_chunk_smem_output() {
    // q / v_new [GDN_CHUNK][S], k pieces / state tiles, packed lower triangular P, g
    return (GDN_CHUNK*S + gdn_chunk_output_sbuf<S>() + GDN_CHUNK*(GDN_CHUNK + 1)/2 + GDN_CHUNK) * sizeof(float);
}

template <int S>
static constexpr size_t gdn_chunk_smem_state() {
    // piece ring, U slice (double buffered), v_new slice, g, decay
    return (GDN_CHUNK_STATE_SLOTS*GDN_CHUNK_STATE_PR*S + 3*GDN_CHUNK*GDN_CHUNK_STATE_COLS + 2*GDN_CHUNK) * sizeof(float);
}

// Thread tiling shared by stage 1 and 3: thread (ti, tc) owns token rows ti*TI .. ti*TI+TI-1
// and interleaved columns tc + NT*n. Interleaving keeps the smem column reads conflict free.
#define GDN_CHUNK_NT  (GDN_CHUNK_THREADS / (GDN_CHUNK / GDN_CHUNK_TILE_I)) // 16 threads per row group
#define GDN_CHUNK_TJ  (GDN_CHUNK / GDN_CHUNK_NT)                            // 4 chunk columns per thread
static_assert(GDN_CHUNK % GDN_CHUNK_TILE_I == 0, "bad row tile");
static_assert(GDN_CHUNK_NT * GDN_CHUNK_TJ == GDN_CHUNK, "bad column tile");

// stage 1, one block per (head, seq, chunk): cumulative gate, A = tril(beta k k^T decay), solve (I + A) X = rhs for W and U.
// Each thread solves one column (W columns from k in smem, U columns from v in global memory) in registers.
template <int S>
__global__ void __launch_bounds__(GDN_CHUNK_THREADS, 2)
gdn_chunk_prepare(const gdn_chunk_params p) {
    constexpr int SP = S + 4;
    constexpr int CP = GDN_CHUNK_CP;
    constexpr int TI = GDN_CHUNK_TILE_I;
    constexpr int TJ = GDN_CHUNK_TJ;
    constexpr int NT = GDN_CHUNK_NT;
    static_assert(2*S <= GDN_CHUNK_THREADS, "one thread per rhs column");
    extern __shared__ __align__(16) float smem[];
    float * sk  = smem;                 // [GDN_CHUNK][SP] k
    float * sa  = sk + GDN_CHUNK*SP;    // [GDN_CHUNK][CP] A
    float * sg  = sa + GDN_CHUNK*CP;    // [GDN_CHUNK] cumulative gate
    float * sb  = sg + GDN_CHUNK;       // [GDN_CHUNK] beta
    float * sbg = sb + GDN_CHUNK;       // [GDN_CHUNK] beta exp(g)

    const int h   = blockIdx.x;
    const int seq = blockIdx.y;
    const int t0  = blockIdx.z * GDN_CHUNK;
    const int tid = threadIdx.x;
    const int ti  = tid / NT;
    const int tc  = tid % NT;
    const int i0  = ti*TI;

    ggml_cuda_pdl_sync();

    const int iq1 = h % p.H_k;
    const int iq3 = seq / p.rq3;
    const float * kb = p.k    + iq3*p.sq3 + (int64_t) t0*p.sq2 + iq1*p.sq1;
    const float * vb = p.v    + seq*p.sv3 + (int64_t) t0*p.sv2 + h*p.sv1;
    const float * gb = p.g    + seq*p.sb3 + (int64_t) t0*p.sb2 + h*p.sb1;
    const float * bb = p.beta + seq*p.sb3 + (int64_t) t0*p.sb2 + h*p.sb1;

    gdn_cp_async_rows<GDN_CHUNK, S, SP>(sk, kb, p.sq2, tid);
    cp_async_commit_group();

    // rhs column of this thread: threads [0, S) own W columns, threads [S, 2S) own U columns
    const bool active = tid < 2*S;
    const bool is_u   = tid >= S;
    const int  col    = is_u ? tid - S : tid;
    float x[GDN_CHUNK];
    if (is_u) {
#pragma unroll
        for (int i = 0; i < GDN_CHUNK; ++i) {
            x[i] = vb[i*p.sv2 + col];
        }
    }
    if (tid < GDN_CHUNK) {
        sg[tid] = gb[tid*p.sb2];
        sb[tid] = bb[tid*p.sb2];
    }
    cp_async_wait_group<0>();
    __syncthreads();

    if (tid == 0) {
        for (int i = 1; i < GDN_CHUNK; ++i) {
            sg[i] += sg[i - 1];
        }
    }
    __syncthreads();
    if (tid < GDN_CHUNK) {
        sbg[tid] = sb[tid]*expf(sg[tid]);
    }

    // A[i][j] = beta_i (k_i . k_j) exp(g_i - g_j) for j < i, else 0
    {
        float acc[TI][TJ] = {{0.0f}};
        for (int d4 = 0; d4 < S/4; ++d4) {
            float4 kr[TI];
#pragma unroll
            for (int r = 0; r < TI; ++r) {
                kr[r] = *(const float4 *) (sk + (i0 + r)*SP + 4*d4);
            }
#pragma unroll
            for (int c = 0; c < TJ; ++c) {
                const float4 kc = *(const float4 *) (sk + (tc + NT*c)*SP + 4*d4);
#pragma unroll
                for (int r = 0; r < TI; ++r) {
                    acc[r][c] += gdn_dot4(kr[r], kc);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < TI; ++r) {
            const int i = i0 + r;
#pragma unroll
            for (int c = 0; c < TJ; ++c) {
                const int j = tc + NT*c;
                sa[i*CP + j] = j < i ? sb[i]*acc[r][c]*expf(sg[i] - sg[j]) : 0.0f;
            }
        }
    }
    __syncthreads();

    if (active) {
        // rhs: W = beta_i exp(g_i) k_i[col], U = beta_i v_i[col]
        if (is_u) {
#pragma unroll
            for (int i = 0; i < GDN_CHUNK; ++i) {
                x[i] *= sb[i];
            }
        } else {
#pragma unroll
            for (int i = 0; i < GDN_CHUNK; ++i) {
                x[i] = sbg[i]*sk[i*SP + col];
            }
        }

        // forward substitution, (I + A) is unit lower triangular. Rows are processed in blocks of RB:
        // the part against earlier rows is RB independent chains, only the triangle inside the block is sequential.
        constexpr int RB = 16;
        static_assert(GDN_CHUNK % RB == 0 && RB % 4 == 0, "bad row block");
#pragma unroll
        for (int ib = 0; ib < GDN_CHUNK; ib += RB) {
            float acc[RB];
#pragma unroll
            for (int r = 0; r < RB; ++r) {
                acc[r] = x[ib + r];
            }
#pragma unroll
            for (int j = 0; j < ib; j += 4) {
                const float4 xj = make_float4(x[j], x[j + 1], x[j + 2], x[j + 3]);
#pragma unroll
                for (int r = 0; r < RB; ++r) {
                    acc[r] -= gdn_dot4(*(const float4 *) (sa + (ib + r)*CP + j), xj);
                }
            }
#pragma unroll
            for (int r = 1; r < RB; ++r) {
#pragma unroll
                for (int rr = 0; rr < r; ++rr) {
                    acc[r] -= sa[(ib + r)*CP + ib + rr] * acc[rr];
                }
            }
#pragma unroll
            for (int r = 0; r < RB; ++r) {
                x[ib + r] = acc[r];
            }
        }

        const int64_t base = ((int64_t) seq*p.H + h)*p.n_full + t0;
        float * out = (is_u ? p.ws_u : p.ws_w) + base*S + col;
#pragma unroll
        for (int i = 0; i < GDN_CHUNK; ++i) {
            out[i*S] = x[i];
        }
    }
    if (tid < GDN_CHUNK) {
        const int64_t base = ((int64_t) seq*p.H + h)*p.n_full + t0;
        p.ws_g[base + tid] = sg[tid];
    }
}

// stage 2, one block per (head, seq, column slice), sequential over chunks:
// v_new = U - S0 W, store S0, S1 = exp(g_last) S0 + v_new^T (k exp(g_last - g)).
// W and k are streamed through a ring of PR-row pieces so that the next pieces are always in flight.
// The kernel is bound by smem load bandwidth: each thread owns NCOL state columns and DP threads share a
// column, so every float4 load of W or k feeds 4*NCOL FMAs.
template <int S>
__global__ void __launch_bounds__(GDN_CHUNK_STATE_THREADS)
gdn_chunk_state(const gdn_chunk_params p, const float * __restrict__ state_in, float * __restrict__ state_out) {
    constexpr int THREADS = GDN_CHUNK_STATE_THREADS;
    constexpr int COLS    = GDN_CHUNK_STATE_COLS;   // state columns per block
    constexpr int NCOL    = GDN_CHUNK_STATE_NCOL;   // state columns per thread
    constexpr int DP      = THREADS*NCOL / COLS;    // threads per column group, each owns S/DP state rows
    constexpr int M4      = S / (4*DP);             // float4 groups per column per thread
    constexpr int PR      = GDN_CHUNK_STATE_PR;     // rows per staged piece
    constexpr int RG      = 8;                      // rows per reduction group
    constexpr int NP      = GDN_CHUNK / PR;         // pieces per matrix
    constexpr int PPC     = 2*NP;                   // pieces per chunk: W pieces, then k pieces
    constexpr int SLOTS   = GDN_CHUNK_STATE_SLOTS;  // ring depth
    static_assert(NCOL % 4 == 0 && (NCOL & (NCOL - 1)) == 0 && NCOL <= DP && (DP & (DP - 1)) == 0, "the transposed reduction needs power of two column counts");
    static_assert(DP <= WARP_SIZE && WARP_SIZE % DP == 0, "column groups must not straddle warps");
    static_assert(DP*COLS == THREADS*NCOL && S % COLS == 0, "bad column split");
    static_assert(S % (4*DP) == 0 && PR % RG == 0, "bad row split");
    static_assert(SLOTS >= 3 && SLOTS - 2 <= 4, "ring depth outside the wait helper range");
    // the first W piece of chunk c+1 is issued SLOTS-1 pieces ahead, while chunk c may still read g (at its first
    // piece) and U (during its NP W pieces); U is double buffered by chunk parity, g must not be issued early
    static_assert(SLOTS - 1 < PPC, "g of the next chunk would be issued before the current chunk read it");
    static_assert(SLOTS - 1 < PPC + NP, "U of chunk c+2 would be issued while chunk c still reads its buffer");
    static_assert((PR*S) % (4*THREADS) == 0 && (GDN_CHUNK*COLS) % (4*THREADS) == 0 && GDN_CHUNK % 4 == 0, "bad copy split");
    extern __shared__ __align__(16) float smem[];
    float * spiece = smem;                          // [SLOTS][PR][S]
    float * su     = spiece + SLOTS*PR*S;           // [2][GDN_CHUNK][COLS] U for the block's columns, by chunk parity
    float * svn    = su + 2*GDN_CHUNK*COLS;         // [GDN_CHUNK][COLS] v_new for the block's columns
    float * sg     = svn + GDN_CHUNK*COLS;          // [GDN_CHUNK] cumulative gate of the current chunk
    float * sdec   = sg + GDN_CHUNK;                // [GDN_CHUNK] exp(g_last - g_j)

    const int h   = blockIdx.x;
    const int seq = blockIdx.y;
    const int tid = threadIdx.x;
    const int cq  = tid / DP;          // column group: columns c0 + NCOL*cq .. +NCOL-1
    const int dp  = tid % DP;          // d partition: floats 4*DP*m + 4*dp .. +3
    const int c0  = blockIdx.z*COLS;
    const int cb  = c0 + NCOL*cq;      // first global column of this thread

    ggml_cuda_pdl_sync();

    const int iq1 = h % p.H_k;
    const int iq3 = seq / p.rq3;
    const int64_t sh = (int64_t) seq*p.H + h;
    const float * kb    = p.k + iq3*p.sq3 + iq1*p.sq1;
    const float * ws_w  = p.ws_w  + sh*p.n_full*S;
    const float * ws_u  = p.ws_u  + sh*p.n_full*S;
    const float * ws_g  = p.ws_g  + sh*p.n_full;
    float *       ws_vn = p.ws_vn + sh*p.n_full*S;
    float *       ws_h  = p.ws_h  + sh*p.n_chunks*S*S;

    const int n_pieces = p.n_chunks*PPC;

    // piece pc: chunk pc/PPC, W rows or k rows [l*PR, l*PR + PR). The first W piece also brings U and g.
    auto issue = [&](const int pc) {
        const int c  = pc / PPC;
        const int l  = pc % PPC;
        const int t0 = c*GDN_CHUNK;
        float * dstp = spiece + (pc % SLOTS)*PR*S;
        if (l < NP) {
            gdn_cp_async_rows<PR, S, S, THREADS>(dstp, ws_w + (int64_t) (t0 + l*PR)*S, S, tid);
            if (l == 0) {
                gdn_cp_async_rows<GDN_CHUNK, COLS, COLS, THREADS>(su + (c & 1)*GDN_CHUNK*COLS, ws_u + (int64_t) t0*S + c0, S, tid);
                if (tid < GDN_CHUNK/4) {
                    gdn_cp_async_16(sg + 4*tid, ws_g + t0 + 4*tid);
                }
            }
        } else {
            gdn_cp_async_rows<PR, S, S, THREADS>(dstp, kb + (int64_t) (t0 + (l - NP)*PR)*p.sq2, p.sq2, tid);
        }
        cp_async_commit_group();
    };

    // state registers: s[c][m] = S[cb + c][4*DP*m + 4*dp .. +3]
    float4 s[NCOL][M4];
#pragma unroll
    for (int c = 0; c < NCOL; ++c) {
        const float * sin = state_in + sh*S*S + (cb + c)*S + 4*dp;
#pragma unroll
        for (int m = 0; m < M4; ++m) {
            s[c][m] = *(const float4 *) (sin + 4*DP*m);
        }
    }

#pragma unroll
    for (int pc = 0; pc < SLOTS - 1; ++pc) {
        issue(pc);
    }

    float g_last = 1.0f;
    for (int pc = 0; pc < n_pieces; ++pc) {
        // wait for piece pc, the SLOTS-2 pieces issued after it may stay in flight
        gdn_cp_async_wait_n(min(SLOTS - 2, n_pieces - 1 - pc));
        __syncthreads();
        // the slot of piece pc-1 is free now
        if (pc + SLOTS - 1 < n_pieces) {
            issue(pc + SLOTS - 1);
        }

        const int c  = pc / PPC;
        const int l  = pc % PPC;
        const int t0 = c*GDN_CHUNK;
        const float * piece = spiece + (pc % SLOTS)*PR*S + 4*dp;

        if (l == 0) {
            g_last = expf(sg[GDN_CHUNK - 1]);
            if (tid < GDN_CHUNK) {
                sdec[tid] = expf(sg[GDN_CHUNK - 1] - sg[tid]);
            }
#pragma unroll
            for (int cc = 0; cc < NCOL; ++cc) {
                float * hc = ws_h + (int64_t) c*S*S + (cb + cc)*S + 4*dp;
#pragma unroll
                for (int m = 0; m < M4; ++m) {
                    *(float4 *) (hc + 4*DP*m) = s[cc][m];
                }
            }
        }

        if (l < NP) {
            // v_new[i][col] = U[i][col] - sum_d S[col][d] W[i][d]
            const int     i0  = l*PR;
            const float * suc = su + (c & 1)*GDN_CHUNK*COLS;
#pragma unroll
            for (int ig = 0; ig < PR; ig += RG) {
                float acc[RG][NCOL] = {{0.0f}};
#pragma unroll
                for (int i = 0; i < RG; ++i) {
#pragma unroll
                    for (int m = 0; m < M4; ++m) {
                        const float4 w = *(const float4 *) (piece + (ig + i)*S + 4*DP*m);
#pragma unroll
                        for (int cc = 0; cc < NCOL; ++cc) {
                            acc[i][cc] += gdn_dot4(s[cc][m], w);
                        }
                    }
                }
                // transposed reduction of the NCOL partial sums over the DP lanes of the column group: each of the
                // first log2(NCOL) rounds halves the values per lane, the lane keeps the half selected by its dp bit,
                // the remaining rounds are plain butterflies. Lane dp ends with the full sum of column dp / (DP/NCOL).
#pragma unroll
                for (int i = 0; i < RG; ++i) {
#pragma unroll
                    for (int half = NCOL/2, off = DP/2; half >= 1; half /= 2, off /= 2) {
                        const bool upper = dp & off;
#pragma unroll
                        for (int c = 0; c < half; ++c) {
                            const float send = upper ? acc[i][c] : acc[i][c + half];
                            const float keep = upper ? acc[i][c + half] : acc[i][c];
                            acc[i][c] = keep + __shfl_xor_sync(0xffffffff, send, off, 32);
                        }
                    }
#pragma unroll
                    for (int off = DP/(2*NCOL); off >= 1; off /= 2) {
                        acc[i][0] += __shfl_xor_sync(0xffffffff, acc[i][0], off, 32);
                    }
                }
                if (dp % (DP/NCOL) == 0) {
                    const int cc = dp / (DP/NCOL);
#pragma unroll
                    for (int i = 0; i < RG; ++i) {
                        const int   row = i0 + ig + i;
                        const float vn  = suc[row*COLS + NCOL*cq + cc] - acc[i][0];
                        svn[row*COLS + NCOL*cq + cc] = vn;
                        ws_vn[(int64_t) (t0 + row)*S + cb + cc] = vn;
                    }
                }
            }
        } else {
            if (l == NP) {
#pragma unroll
                for (int cc = 0; cc < NCOL; ++cc) {
#pragma unroll
                    for (int m = 0; m < M4; ++m) {
                        s[cc][m].x *= g_last; s[cc][m].y *= g_last; s[cc][m].z *= g_last; s[cc][m].w *= g_last;
                    }
                }
            }
            const int j0 = (l - NP)*PR;
#pragma unroll
            for (int j = 0; j < PR; ++j) {
                const float dec = sdec[j0 + j];
                float wc[NCOL];
#pragma unroll
                for (int c4 = 0; c4 < NCOL/4; ++c4) {
                    const float4 w = *(const float4 *) (svn + (j0 + j)*COLS + NCOL*cq + 4*c4);
                    wc[4*c4 + 0] = w.x*dec; wc[4*c4 + 1] = w.y*dec; wc[4*c4 + 2] = w.z*dec; wc[4*c4 + 3] = w.w*dec;
                }
#pragma unroll
                for (int m = 0; m < M4; ++m) {
                    const float4 k4 = *(const float4 *) (piece + j*S + 4*DP*m);
#pragma unroll
                    for (int cc = 0; cc < NCOL; ++cc) {
                        s[cc][m].x += wc[cc]*k4.x; s[cc][m].y += wc[cc]*k4.y; s[cc][m].z += wc[cc]*k4.z; s[cc][m].w += wc[cc]*k4.w;
                    }
                }
            }
        }
    }

#pragma unroll
    for (int cc = 0; cc < NCOL; ++cc) {
        float * sout = state_out + sh*S*S + (cb + cc)*S + 4*dp;
#pragma unroll
        for (int m = 0; m < M4; ++m) {
            *(float4 *) (sout + 4*DP*m) = s[cc][m];
        }
    }
}

// stage 3, one block per (head, seq, chunk): o_i = exp(g_i) S0 q_i + sum_{j<=i} (q_i . k_j) exp(g_i - g_j) v_new_j.
// smem is kept under 62 KB so that two blocks share an SM and hide each other's copy latency.
template <int S>
__global__ void __launch_bounds__(GDN_CHUNK_THREADS, 2)
gdn_chunk_output(const gdn_chunk_params p, float * __restrict__ dst) {
    constexpr int SP = S + 4;                 // padded stride of the k pieces
    constexpr int TI = GDN_CHUNK_TILE_I;
    constexpr int NT = GDN_CHUNK_NT;
    constexpr int NC = S / NT;                // output columns per thread
    constexpr int KP = GDN_CHUNK_OUT_KP;      // k rows per piece
    constexpr int NKP = GDN_CHUNK / KP;
    constexpr int TD = GDN_CHUNK_OUT_TD;      // d tile of the state staged in smem
    constexpr int TP = TD + 4;
    constexpr int NTILES = S / TD;
    static_assert(S % NT == 0, "bad output tile");
    static_assert(KP == NT, "piece c must hold the thread's column tc + NT*c");
    static_assert(S % TD == 0 && NTILES >= 2, "bad state tiles");
    static_assert((S*TD) % (4*GDN_CHUNK_THREADS) == 0 && (KP*S) % (4*GDN_CHUNK_THREADS) == 0, "bad copy split");
    extern __shared__ __align__(16) float smem[];
    float * sq   = smem;                                 // [GDN_CHUNK][S] q, then v_new
    float * sbuf = sq + GDN_CHUNK*S;                     // two k pieces [KP][SP], then two state tiles [S][TP]
    float * sp   = sbuf + gdn_chunk_output_sbuf<S>();    // packed lower triangular P, row i at i*(i+1)/2
    float * sg   = sp + GDN_CHUNK*(GDN_CHUNK + 1)/2;     // [GDN_CHUNK]

    const int h   = blockIdx.x;
    const int seq = blockIdx.y;
    const int c   = blockIdx.z;
    const int t0  = c*GDN_CHUNK;
    const int tid = threadIdx.x;
    const int ti  = tid / NT;
    const int tc  = tid % NT;
    const int i0  = ti*TI;

    ggml_cuda_pdl_sync();

    const int iq1 = h % p.H_k;
    const int iq3 = seq / p.rq3;
    const int64_t sh = (int64_t) seq*p.H + h;
    const float * qb    = p.q + iq3*p.sq3 + (int64_t) t0*p.sq2 + iq1*p.sq1;
    const float * kb    = p.k + iq3*p.sq3 + (int64_t) t0*p.sq2 + iq1*p.sq1;
    const float * ws_vn = p.ws_vn + (sh*p.n_full + t0)*S;
    const float * ws_h  = p.ws_h  + (sh*p.n_chunks + c)*S*S;
    const float * ws_g  = p.ws_g  + sh*p.n_full + t0;

    auto issue_k = [&](const int kp) {
        gdn_cp_async_rows<KP, S, SP>(sbuf + (kp & 1)*KP*SP, kb + (int64_t) kp*KP*p.sq2, p.sq2, tid);
        cp_async_commit_group();
    };
    auto issue_tile = [&](const int t) {
        float * dst_tile = sbuf + (t & 1)*S*TP;
#pragma unroll
        for (int r = 0; r < S*TD/(4*GDN_CHUNK_THREADS); ++r) {
            const int idx = 4*(tid + r*GDN_CHUNK_THREADS);
            const int col = idx / TD;
            const int dd  = idx % TD;
            gdn_cp_async_16(dst_tile + col*TP + dd, ws_h + col*S + t*TD + dd);
        }
        cp_async_commit_group();
    };

    gdn_cp_async_rows<GDN_CHUNK, S, S>(sq, qb, p.sq2, tid);
    cp_async_commit_group();
    issue_k(0);
    issue_k(1);
    if (tid < GDN_CHUNK) {
        sg[tid] = ws_g[tid];
    }

    // P[i][j] = (q_i . k_j) exp(g_i - g_j) for j <= i, k streamed in pieces, piece kp holds this thread's column tc + NT*kp
    for (int kp = 0; kp < NKP; ++kp) {
        if (kp + 1 < NKP) {
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();
        const float * piece = sbuf + (kp & 1)*KP*SP + tc*SP;
        float acc[TI] = {0.0f};
        for (int d4 = 0; d4 < S/4; ++d4) {
            const float4 kc = *(const float4 *) (piece + 4*d4);
#pragma unroll
            for (int r = 0; r < TI; ++r) {
                acc[r] += gdn_dot4(*(const float4 *) (sq + (i0 + r)*S + 4*d4), kc);
            }
        }
        const int j = tc + NT*kp;
#pragma unroll
        for (int r = 0; r < TI; ++r) {
            const int i = i0 + r;
            if (j <= i) {
                sp[i*(i + 1)/2 + j] = acc[r]*expf(sg[i] - sg[j]);
            }
        }
        __syncthreads();
        if (kp + 2 < NKP) {
            issue_k(kp + 2);
        }
    }

    // inter-chunk term S0 q_i, state staged in d tiles of TD columns, two tiles in flight
    issue_tile(0);
    issue_tile(1);
    float acc[TI][NC] = {{0.0f}};
    for (int t = 0; t < NTILES; ++t) {
        if (t + 1 < NTILES) {
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();
        const float * tile = sbuf + (t & 1)*S*TP;
#pragma unroll
        for (int dd4 = 0; dd4 < TD/4; ++dd4) {
            float4 qr[TI];
#pragma unroll
            for (int r = 0; r < TI; ++r) {
                qr[r] = *(const float4 *) (sq + (i0 + r)*S + t*TD + 4*dd4);
            }
#pragma unroll
            for (int n = 0; n < NC; ++n) {
                const float4 sv = *(const float4 *) (tile + (tc + NT*n)*TP + 4*dd4);
#pragma unroll
                for (int r = 0; r < TI; ++r) {
                    acc[r][n] += gdn_dot4(qr[r], sv);
                }
            }
        }
        __syncthreads();
        if (t + 2 < NTILES) {
            issue_tile(t + 2);
        }
    }
#pragma unroll
    for (int r = 0; r < TI; ++r) {
        const float g_i = expf(sg[i0 + r]);
#pragma unroll
        for (int n = 0; n < NC; ++n) {
            acc[r][n] *= g_i;
        }
    }

    // intra-chunk term P v_new, v_new replaces q
    gdn_cp_async_rows<GDN_CHUNK, S, S>(sq, ws_vn, S, tid);
    cp_async_commit_group();
    cp_async_wait_group<0>();
    __syncthreads();
    for (int j = 0; j < i0; ++j) {
        float pr[TI];
#pragma unroll
        for (int r = 0; r < TI; ++r) {
            const int i = i0 + r;
            pr[r] = sp[i*(i + 1)/2 + j];
        }
#pragma unroll
        for (int n = 0; n < NC; ++n) {
            const float vn = sq[j*S + tc + NT*n];
#pragma unroll
            for (int r = 0; r < TI; ++r) {
                acc[r][n] += pr[r]*vn;
            }
        }
    }
    // the diagonal block: row i0 + r only sees j <= i0 + r
#pragma unroll
    for (int jj = 0; jj < TI; ++jj) {
        const int j = i0 + jj;
        float pr[TI];
#pragma unroll
        for (int r = 0; r < TI; ++r) {
            const int i = i0 + r;
            pr[r] = jj <= r ? sp[i*(i + 1)/2 + j] : 0.0f;
        }
#pragma unroll
        for (int n = 0; n < NC; ++n) {
            const float vn = sq[j*S + tc + NT*n];
#pragma unroll
            for (int r = 0; r < TI; ++r) {
                acc[r][n] += pr[r]*vn;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < TI; ++r) {
        float * o = dst + ((int64_t) seq*p.n_tokens*p.H + (int64_t) (t0 + i0 + r)*p.H + h)*S;
#pragma unroll
        for (int n = 0; n < NC; ++n) {
            o[tc + NT*n] = acc[r][n] * p.scale;
        }
    }
}

template <int S>
static void launch_gated_delta_net_chunked(
        const gdn_chunk_params & p, const float * state_in, float * state_out, float * dst,
        int n_seqs, cudaStream_t stream) {
    // sm_120 allows at most 99 KB of dynamic smem per block, all three kernels stay below that
    constexpr int smem_prepare = (int) gdn_chunk_smem_prepare<S>();
    constexpr int smem_state   = (int) gdn_chunk_smem_state<S>();
    constexpr int smem_output  = (int) gdn_chunk_smem_output<S>();
    CUDA_SET_SHARED_MEMORY_LIMIT((gdn_chunk_prepare<S>), smem_prepare);
    CUDA_SET_SHARED_MEMORY_LIMIT((gdn_chunk_state<S>),   smem_state);
    CUDA_SET_SHARED_MEMORY_LIMIT((gdn_chunk_output<S>),  smem_output);

    const dim3 block_dims(GDN_CHUNK_THREADS, 1, 1);
    const dim3 block_dims_state(GDN_CHUNK_STATE_THREADS, 1, 1);
    const dim3 grid_chunks(p.H, n_seqs, p.n_chunks);
    const dim3 grid_state(p.H, n_seqs, S / GDN_CHUNK_STATE_COLS);

    ggml_cuda_kernel_launch(gdn_chunk_prepare<S>, ggml_cuda_kernel_launch_params(grid_chunks, block_dims,       smem_prepare, stream), p);
    ggml_cuda_kernel_launch(gdn_chunk_state<S>,   ggml_cuda_kernel_launch_params(grid_state,  block_dims_state, smem_state,   stream), p, state_in, state_out);
    ggml_cuda_kernel_launch(gdn_chunk_output<S>,  ggml_cuda_kernel_launch_params(grid_chunks, block_dims, smem_output,  stream), p, dst);
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs, int64_t n_tokens_dst,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, n_tokens_dst, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, n_tokens_dst, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, n_tokens_dst, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, n_tokens_dst, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // recurrent kernel over [t_begin, t_begin + n) starting from state_in, attention rows written at token t_begin
    auto launch_recurrent = [&](int64_t t_begin, int64_t n, const float * state_in) {
        const float * q_t = q_d + t_begin*sq2;
        const float * k_t = k_d + t_begin*sq2;
        const float * v_t = v_d + t_begin*sv2;
        const float * g_t = g_d + t_begin*sb2*(kda ? S_v : 1);
        const float * b_t = b_d + t_begin*sb2;
        float *       o_t = dst_d + t_begin*S_v*H;
        if (kda) {
            if (keep_rs) {
                launch_gated_delta_net<true, true>(q_t, k_t, v_t, g_t, b_t, state_in, o_t, state_d,
                    S_v, H, n, n_seqs, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
            } else {
                launch_gated_delta_net<true, false>(q_t, k_t, v_t, g_t, b_t, state_in, o_t, state_d,
                    S_v, H, n, n_seqs, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
            }
        } else {
            if (keep_rs) {
                launch_gated_delta_net<false, true>(q_t, k_t, v_t, g_t, b_t, state_in, o_t, state_d,
                    S_v, H, n, n_seqs, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
            } else {
                launch_gated_delta_net<false, false>(q_t, k_t, v_t, g_t, b_t, state_in, o_t, state_d,
                    S_v, H, n, n_seqs, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
            }
        }
    };

    // Chunked path for prefill. The last tokens stay on the recurrent kernel so that snapshot slots
    // (K > 1) are produced per token. With K == 1 the tail can be empty.
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    // the chunked kernels copy q, k and v with 16 byte cp.async
    const bool aligned = ((uintptr_t) q_d) % 16 == 0 && ((uintptr_t) k_d) % 16 == 0 && ((uintptr_t) v_d) % 16 == 0 &&
        sq1 % 4 == 0 && sq2 % 4 == 0 && sq3 % 4 == 0 && sv1 % 4 == 0 && sv2 % 4 == 0 && sv3 % 4 == 0;
    int64_t n_full = 0;
    if (!kda && (S_v == 64 || S_v == 128) && cp_async_available(cc) && aligned) {
        const int64_t avail = n_tokens - (keep_rs ? K : 0);
        n_full = avail >= GDN_CHUNK ? (avail / GDN_CHUNK) * GDN_CHUNK : 0;
    }

    if (n_full == 0) {
        launch_recurrent(0, n_tokens, s_d);
        return;
    }

    const int64_t n_chunks = n_full / GDN_CHUNK;
    const int64_t nsh      = n_seqs * H;

    ggml_cuda_pool_alloc<float> ws_g (ctx.pool(), nsh * n_full);
    ggml_cuda_pool_alloc<float> ws_w (ctx.pool(), nsh * n_full * S_v);
    ggml_cuda_pool_alloc<float> ws_u (ctx.pool(), nsh * n_full * S_v);
    ggml_cuda_pool_alloc<float> ws_vn(ctx.pool(), nsh * n_full * S_v);
    ggml_cuda_pool_alloc<float> ws_h (ctx.pool(), nsh * n_chunks * S_v * S_v);

    gdn_chunk_params p;
    p.q        = q_d;
    p.k        = k_d;
    p.v        = v_d;
    p.g        = g_d;
    p.beta     = b_d;
    p.sq1      = sq1; p.sq2 = sq2; p.sq3 = sq3;
    p.sv1      = sv1; p.sv2 = sv2; p.sv3 = sv3;
    p.sb1      = sb1; p.sb2 = sb2; p.sb3 = sb3;
    p.H        = (int) H;
    p.H_k      = (int) neqk1;
    p.rq3      = (int) rq3;
    p.n_full   = (int) n_full;
    p.n_chunks = (int) n_chunks;
    p.n_tokens = (int) n_tokens;
    p.scale    = scale;
    p.ws_g     = ws_g.get();
    p.ws_w     = ws_w.get();
    p.ws_u     = ws_u.get();
    p.ws_vn    = ws_vn.get();
    p.ws_h     = ws_h.get();

    // the chunked state lands in snapshot slot 0, the tail kernel continues from there in place
    switch (S_v) {
        case 64:
            launch_gated_delta_net_chunked<64> (p, s_d, state_d, dst_d, n_seqs, stream);
            break;
        case 128:
            launch_gated_delta_net_chunked<128>(p, s_d, state_d, dst_d, n_seqs, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }

    if (n_tokens > n_full) {
        launch_recurrent(n_full, n_tokens - n_full, state_d);
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
