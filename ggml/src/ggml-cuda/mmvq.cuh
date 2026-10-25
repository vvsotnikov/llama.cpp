#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

// Row range for the batched multi-row Q4_K decode tile (mmvq-batched): below
// the minimum the single-row-optimized MMVQ kernels win, above the maximum
// the tile's row staging (pad to 32 mma rows) no longer pays off.
#define MMVQ_BATCHED_MIN_ROWS 2
#define MMVQ_BATCHED_MAX_ROWS 32

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

// Returns true when the batched multi-row Q4_K decode GEMV tile should run
// instead of MMVQ/MMQ (DGX Spark only, multi-row decode batches, shapes the
// tile covers).
bool ggml_cuda_should_use_mmvq_batched(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

// Batched multi-row decode GEMV tile for Q4_K (DGX Spark only).
void ggml_cuda_mul_mat_vec_q_batched(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
