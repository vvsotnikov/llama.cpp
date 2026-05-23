#pragma once

// Persistent, per-layer GPU cache of MoE expert weights for speculative prefetch.
//
// Design summary (see project memory `explore-speculative-expert-prefetch.md`):
//   - For each ncmoe-managed MoE layer L (i.e. one whose `ffn_*_exps` tensors live on
//     the CPU backend per `--n-cpu-moe`), allocate a GPU-resident cache tensor that
//     shadows each of the layer's expert-weight matrices (ffn_up_exps, ffn_gate_exps,
//     ffn_down_exps). Cache is owned by the llama_context, persists across decode calls.
//   - Phase 2 (current): cache tensor is `[n_embd, n_ff, C]` with C ≪ n_expert (the
//     "actual VRAM-saver"). A per-layer F32 slot map `[1, n_expert]` translates the
//     router's expert ids into cache slot ids via ggml_get_rows + ggml_cast(I32);
//     mul_mat_id then reads from the smaller cache. The slot map is updated on the
//     host on every fill and uploaded to the GPU before each decode.
//
//   - Slot tracking (host-side): per-layer bitmap of which expert slabs in the cache
//     currently hold valid data. On a router pick that hits a valid slot → use as-is.
//     On a miss → fill from CPU on the copy stream (or compute on CPU as fallback).
//   - Speculative fill: a draft router predicts experts for layers N+1..N+k; their
//     slabs are asynchronously H2D-copied into the cache ahead of compute.
//
// This header is the public surface used by llama-context and the model graph
// builders. The implementation lives in llama-moe-expert-cache.cpp.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <vector>

struct llama_model;

struct llama_moe_expert_cache_layer {
    int32_t layer    = -1;  // layer index in the model
    int32_t n_expert = 0;   // total experts in this layer (e.g. 128)
    int32_t n_embd   = 0;
    int32_t n_ff     = 0;
    int32_t C        = 0;   // cache capacity (slots). Phase 1: == n_expert.

    // Per-layer cache tensors on GPU. Shapes match the source expert tensors. May be
    // nullptr if a particular weight is absent (e.g. fused gate_up not used here).
    struct ggml_tensor * cache_up    = nullptr;
    struct ggml_tensor * cache_gate  = nullptr;
    struct ggml_tensor * cache_down  = nullptr;

    // CPU-resident source tensors (the `blk.L.ffn_*_exps` that the model loaded onto
    // the CPU backend via the ncmoe override). Source of async H2D fills.
    const struct ggml_tensor * src_up   = nullptr;
    const struct ggml_tensor * src_gate = nullptr;
    const struct ggml_tensor * src_down = nullptr;

    // Phase 2 slot map. expert_to_slot[e] = s in [0,C) if expert e is in slot s,
    // else -1. slot_to_expert[s] = e if slot s holds expert e, else -1.
    // expert_to_slot has size n_expert; slot_to_expert has size C.
    std::vector<int32_t> expert_to_slot;
    std::vector<int32_t> slot_to_expert;

    // Host mirror of the GPU slot map. Stored as float so we can use the existing
    // ggml_get_rows op (which only supports floating sources) followed by ggml_cast
    // to i32 — no new ggml op required for the remap.
    // slot_map_host[e] = (float) expert_to_slot[e] when in cache; 0.0 otherwise
    // ("substitute-and-go" miss policy: a missing expert reads slot 0's data, which
    // produces garbage but doesn't OOB. With a perfect oracle this never happens.)
    std::vector<float> slot_map_host;

    // GPU mirror of slot_map_host (lives in the same cache backend buffer).
    // Shape [1, n_expert], type F32.
    struct ggml_tensor * slot_map_gpu = nullptr;

    // Per-slot validity (size C). 1 iff slot s currently holds a valid expert.
    std::vector<uint8_t> slot_valid;

    // LRU: per-slot last-use counter (size C); lowest = eviction victim on miss.
    std::vector<uint64_t> slot_lru;
    uint64_t              lru_counter = 0;
};

struct llama_moe_expert_cache {
    int32_t C = 0;  // configured slots/layer (0 = disabled). Phase 1: forced to n_expert
                    // when allocated, regardless of user-provided value.

    // GPU backend the cache lives on. Owned by the llama_context; not freed here.
    ggml_backend_t       backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;

    // [EXPERIMENTAL] dedicated CUDA copy stream + event for async fills. The copy
    // backend is a second ggml backend instance bound to the SAME device as `backend`,
    // so its stream runs concurrently with the compute stream. `fill_event` is recorded
    // on the copy stream after a batch of fills; the compute stream waits on it before
    // running the kernel that reads the filled slots. Both are owned by this struct.
    ggml_backend_t       copy_backend = nullptr;
    ggml_backend_event_t fill_event   = nullptr;
    bool                 async_fill   = false;  // false → fall back to sync (no copy stream)

    // ggml_context holding the cache tensor metadata; backend buffer holding their data.
    // Both owned by this struct via unique_ptr<...>.
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;

    // One entry per ncmoe-managed MoE layer (sorted by layer index).
    std::vector<llama_moe_expert_cache_layer> layers;

    // Oracle: per-layer per-decode-step list of expert ids the router selected.
    // Indexed as oracle_experts[layer_idx_in_oracle][step_idx]. layer_to_oracle maps
    // a model layer index to its slot in oracle_experts (or -1 if not in trace).
    bool                                                    oracle_loaded = false;
    std::vector<std::vector<std::vector<int32_t>>>          oracle_experts;
    std::vector<int32_t>                                    oracle_layer_to_idx; // size = n_layer
    int32_t                                                 oracle_max_step = 0;

    // Live (temporal) predictor: most-recently-observed router output per managed
    // layer. Updated by `record_router` (from the user's cb_eval) and consumed by
    // `predictor_prefill_from_observations` to drive fills for the next decode.
    // Empty inner vector means "no observation yet" (cold start).
    bool                                                    predictor_enabled = false;
    std::vector<std::vector<int32_t>>                       last_selected_experts; // [managed_idx]
    std::vector<int32_t>                                    layer_to_managed_idx;  // size n_layer, -1 if not managed

    // Stats (for diagnostics / bench reporting).
    uint64_t n_hits      = 0;
    uint64_t n_misses    = 0;
    uint64_t n_fills     = 0;
    uint64_t n_evictions = 0;
};

// Construct an empty disabled cache. Allocation happens later via `..._allocate`.
// Returns nullptr if `cache_size == 0`.
llama_moe_expert_cache * llama_moe_expert_cache_init(uint32_t cache_size);

// Allocate the per-layer cache tensors + backend buffer. Walks the model's layers,
// identifies those whose expert tensors are CPU-resident (the ncmoe-managed ones),
// picks the GPU backend, allocates cache tensors mirroring each `ffn_*_exps` tensor.
// No-op if cache is null or if no ncmoe layers are present. Logs a summary.
//
// `backends` is the list of context backends (first non-CPU is used as the cache's
// home for Phase 1; multi-GPU split is deferred).
void llama_moe_expert_cache_allocate(
        llama_moe_expert_cache * cache,
        const llama_model & model,
        const std::vector<ggml_backend_t> & backends);

// Lookup the cache tensor for layer L's `ffn_up_exps` (etc.). Returns nullptr if the
// layer isn't cache-managed. Used by the graph builders to swap CPU tensors for
// GPU cache tensors when emitting `mul_mat_id` calls.
struct ggml_tensor * llama_moe_expert_cache_get_up       (const llama_moe_expert_cache * cache, int layer);
struct ggml_tensor * llama_moe_expert_cache_get_gate     (const llama_moe_expert_cache * cache, int layer);
struct ggml_tensor * llama_moe_expert_cache_get_down     (const llama_moe_expert_cache * cache, int layer);
// Slot map tensor for layer L (F32 [1, n_expert]). The graph applies
// ggml_get_rows(slot_map, selected_experts) → ggml_cast(I32) to translate router
// expert ids into cache slot ids before mul_mat_id. Returns nullptr if not managed.
struct ggml_tensor * llama_moe_expert_cache_get_slot_map (const llama_moe_expert_cache * cache, int layer);

// Push any pending host-side slot map changes to the GPU. Called from the public
// llama_moe_oracle_prefill wrapper after a fill batch updates the host-side maps,
// so the upcoming compute graph sees the new mapping. Uses synchronous host→device
// copy on the compute backend (small tensor, ~512 bytes per layer).
void llama_moe_expert_cache_upload_slot_maps(llama_moe_expert_cache * cache);

// Synchronously copy expert E's weight slabs from the CPU source tensors into the
// corresponding slot of the GPU cache for layer L. Phase 1: slot == expert. The call
// blocks until the copy completes (good for warmup / correctness validation; the
// speculative-overlap version will move to an async copy stream in Task #16).
//
// Returns true on success, false if the layer is not cache-managed or the expert
// index is out of range.
bool llama_moe_expert_cache_fill_sync(
        llama_moe_expert_cache * cache,
        int layer,
        int expert);

// Oracle: load a MOE2-format trace and use it to predict which experts each layer
// will select at each decode step. Returns true on success. On success, the cache
// switches to oracle mode and the auto-warmup is skipped; per-step prefills happen
// via `llama_moe_expert_cache_prefill_step`.
//
// MOE2 format (see examples/moe-trace/moe-trace.cpp):
//   bytes 0..3 : "MOE2"
//   records: int32 hdr[6] = { tid, layer, call_idx, ne0, ne1, dtype } + payload bytes
//            tid 2 = ffn_moe_topk (i32) — the selected expert IDs
// Only tid==2, decode-step records (call_idx >= 1, ne1 == 1) are kept.
bool llama_moe_expert_cache_load_oracle(
        llama_moe_expert_cache * cache,
        const char * path);

// Returns true if an oracle has been loaded.
bool llama_moe_expert_cache_has_oracle(const llama_moe_expert_cache * cache);

// For step `call_idx` (1-indexed decode step matching the trace), look up the
// oracle's predicted experts for each managed layer and fill any that aren't
// already in the cache. Fills run async on the dedicated copy stream when
// available (initialized in `..._allocate`); otherwise synchronous.
//
// Records the cache's fill event after issuing; `..._wait_fills` queues the
// matching cross-stream wait on the compute backend's stream.
//
// `over_fetch` is currently ignored (Phase 1 cache holds all slots so there is no
// notion of over-fetch yet — the parameter is there for the Phase 2 interface).
void llama_moe_expert_cache_prefill_step(
        llama_moe_expert_cache * cache,
        int call_idx,
        int over_fetch = 0);

// Cross-stream sync: queue a stream-wait-event on `compute_backend`'s stream so
// that any subsequently-submitted compute work blocks until the most recent
// prefill batch has landed. Host-side this is non-blocking (just queues the
// wait). Called by the public llama_moe_oracle_prefill wrapper after the fills
// are issued.
void llama_moe_expert_cache_wait_fills(
        llama_moe_expert_cache * cache,
        ggml_backend_t compute_backend);

// Live (temporal-1) predictor mode. Call once at setup; replaces the oracle path.
void llama_moe_expert_cache_enable_predictor(llama_moe_expert_cache * cache);

// Record an observed router output for `layer` (called from cb_eval when an
// ffn_moe_topk tensor is read). The cache stashes the ids so the predictor can
// use them as the prediction for the next decode step.
void llama_moe_expert_cache_record_router(
        llama_moe_expert_cache * cache,
        int layer,
        const int32_t * ids,
        int n_ids);

// Issue fills for the experts the temporal predictor expects to be needed next
// (i.e. "the same experts the router picked last time, per layer"). Same async
// fill + event-record semantics as `prefill_step`. Idempotent — already-cached
// experts just get an LRU bump.
void llama_moe_expert_cache_predictor_prefill(llama_moe_expert_cache * cache);

// Fill EVERY expert slot of EVERY managed layer. Equivalent to "copy all the CPU
// expert tensors to GPU once". Lets us A/B-test the cache plumbing against a known-
// good baseline (output should match running with the experts on GPU directly).
void llama_moe_expert_cache_warmup_all(llama_moe_expert_cache * cache);

// Reset all slot validity (does not free the buffer). Useful between test scenarios.
void llama_moe_expert_cache_invalidate_all(llama_moe_expert_cache * cache);

void llama_moe_expert_cache_free(llama_moe_expert_cache * cache);
