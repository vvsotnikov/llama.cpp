#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Experimental C1 CUDA Qwen35 tree verification. Row 0 is the pending token.
// Other parents must precede their child. Batch positions are temporary row IDs.
LLAMA_API bool llama_tree_supported(const struct llama_context * ctx);
LLAMA_API int llama_tree_decode(struct llama_context * ctx, struct llama_batch_ext * batch, const int32_t * parents, int32_t n);
// Keep a root-to-leaf path, including row 0, without recomputing target layers.
LLAMA_API bool llama_tree_commit(struct llama_context * ctx, const int32_t * rows, int32_t n);

#ifdef __cplusplus
}
#endif
