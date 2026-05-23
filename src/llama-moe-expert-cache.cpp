#include "llama-moe-expert-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

llama_moe_expert_cache * llama_moe_expert_cache_init(uint32_t cache_size) {
    if (cache_size == 0) {
        return nullptr;
    }
    auto * cache = new llama_moe_expert_cache();
    cache->C = (int32_t) cache_size;
    LLAMA_LOG_INFO("%s: MoE expert cache requested, %u slots/layer (allocation deferred)\n",
                   __func__, cache_size);
    return cache;
}

void llama_moe_expert_cache_free(llama_moe_expert_cache * cache) {
    if (!cache) {
        return;
    }
    if (cache->n_hits + cache->n_misses > 0) {
        const uint64_t lookups = cache->n_hits + cache->n_misses;
        LLAMA_LOG_INFO("%s: stats — hits=%llu misses=%llu (%.1f%% hit) fills=%llu\n",
                       __func__,
                       (unsigned long long) cache->n_hits,
                       (unsigned long long) cache->n_misses,
                       lookups ? 100.0 * cache->n_hits / lookups : 0.0,
                       (unsigned long long) cache->n_fills);
    }
    if (cache->fill_event) {
        ggml_backend_event_free(cache->fill_event);
    }
    if (cache->copy_backend) {
        ggml_backend_free(cache->copy_backend);
    }
    // The buffer & context unique_ptrs free themselves.
    delete cache;
}

// Helper: is this tensor's data on the host (CPU) backend? Returns false for nullptr.
static bool tensor_on_host(const ggml_tensor * t) {
    if (!t || !t->buffer) {
        return false;
    }
    return ggml_backend_buffer_is_host(t->buffer);
}

// Helper: clone a tensor's shape onto another ggml_context (no data binding yet).
// Phase 1: cache tensor has the SAME shape as the source — n_expert slots.
static ggml_tensor * clone_shape(ggml_context * ctx, const ggml_tensor * src, const char * name) {
    ggml_tensor * dst = ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
    ggml_set_name(dst, name);
    return dst;
}

void llama_moe_expert_cache_allocate(
        llama_moe_expert_cache * cache,
        const llama_model & model,
        const std::vector<ggml_backend_t> & backends) {
    if (!cache) {
        return;
    }
    if (cache->buf) {
        LLAMA_LOG_WARN("%s: cache already allocated; skipping\n", __func__);
        return;
    }

    // Find a GPU backend to host the cache. Phase 1 assumes single-GPU.
    ggml_backend_t gpu_backend = nullptr;
    for (auto * b : backends) {
        const auto type = ggml_backend_dev_type(ggml_backend_get_device(b));
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            gpu_backend = b;
            break;
        }
    }
    if (!gpu_backend) {
        LLAMA_LOG_WARN("%s: no GPU backend found; MoE expert cache disabled\n", __func__);
        return;
    }
    cache->backend = gpu_backend;
    cache->buft    = ggml_backend_get_default_buffer_type(gpu_backend);

    // Walk the model layers and pick out the ones with CPU-resident expert tensors.
    const auto & hparams = model.hparams;
    const int    n_layer = (int) hparams.n_layer;

    std::vector<int> managed_layers;
    managed_layers.reserve(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        // Skip layers that don't have MoE experts.
        if (!layer.ffn_up_exps && !layer.ffn_gate_exps && !layer.ffn_down_exps) {
            continue;
        }
        // Managed = at least one of the expert tensors lives on the host.
        if (tensor_on_host(layer.ffn_up_exps) ||
            tensor_on_host(layer.ffn_gate_exps) ||
            tensor_on_host(layer.ffn_down_exps)) {
            managed_layers.push_back(il);
        }
    }
    if (managed_layers.empty()) {
        LLAMA_LOG_WARN("%s: no ncmoe-managed MoE layers found; MoE expert cache disabled\n", __func__);
        return;
    }

    // Phase 1: cache shape == source shape. The user-provided C is informational only.
    // Each managed layer contributes 3 cache tensors (up / gate / down).
    const size_t n_tensors_max = managed_layers.size() * 3;

    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * (n_tensors_max + 8);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    cache->ctx.reset(ggml_init(ip));
    if (!cache->ctx) {
        throw std::runtime_error("MoE expert cache: ggml_init failed");
    }

    cache->layers.reserve(managed_layers.size());
    for (int il : managed_layers) {
        const auto & layer = model.layers[il];
        llama_moe_expert_cache_layer L;
        L.layer    = il;
        L.n_expert = (int32_t) hparams.n_expert;

        // Phase 1: C = n_expert (full-size shadow). Phase 2 will shrink this.
        L.C = L.n_expert;

        auto clone_managed = [&](const ggml_tensor * src, const char * base) -> ggml_tensor * {
            if (!src || !tensor_on_host(src)) {
                return nullptr;
            }
            char name[128];
            snprintf(name, sizeof(name), "moe_cache_%s_%d", base, il);
            return clone_shape(cache->ctx.get(), src, name);
        };

        L.src_up   = layer.ffn_up_exps;
        L.src_gate = layer.ffn_gate_exps;
        L.src_down = layer.ffn_down_exps;

        L.cache_up   = clone_managed(layer.ffn_up_exps,   "up");
        L.cache_gate = clone_managed(layer.ffn_gate_exps, "gate");
        L.cache_down = clone_managed(layer.ffn_down_exps, "down");

        // Record dims from whichever expert tensor exists (they share n_embd / n_ff / n_expert).
        const ggml_tensor * any =
            L.cache_up   ? L.cache_up   :
            L.cache_gate ? L.cache_gate :
                           L.cache_down;
        if (!any) {
            // Defensive: managed_layers said this layer has at least one host tensor.
            continue;
        }
        L.n_embd = (int32_t) any->ne[0];
        L.n_ff   = (int32_t) any->ne[1];

        L.expert_to_slot.assign(L.n_expert, -1);
        L.slot_to_expert.assign(L.C, -1);
        L.slot_valid.assign(L.C, 0);
        L.slot_lru.assign(L.C, 0);

        cache->layers.push_back(std::move(L));
    }

    // Allocate one backend buffer covering every cache tensor.
    cache->buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(cache->ctx.get(), cache->buft));
    if (!cache->buf) {
        throw std::runtime_error("MoE expert cache: backend buffer allocation failed");
    }

    // Spin up a second backend instance on the SAME device for async fills. Its stream
    // runs concurrently with the compute backend's stream → H2D copies can overlap
    // with the compute kernel that reads them (when fills land before the kernel).
    auto * dev = ggml_backend_get_device(gpu_backend);
    cache->copy_backend = ggml_backend_dev_init(dev, nullptr);
    if (cache->copy_backend) {
        cache->fill_event = ggml_backend_event_new(dev);
        cache->async_fill = (cache->fill_event != nullptr);
        LLAMA_LOG_INFO("%s: async fill enabled (%s on the same device for the copy stream)\n",
                       __func__, ggml_backend_name(cache->copy_backend));
    } else {
        LLAMA_LOG_WARN("%s: failed to init copy backend; falling back to sync fills\n", __func__);
    }

    const size_t mb = ggml_backend_buffer_get_size(cache->buf.get()) / (1024 * 1024);
    LLAMA_LOG_INFO(
        "%s: MoE expert cache allocated on %s: %zu managed layers x %d slots/layer = %zu MiB total\n",
        __func__,
        ggml_backend_buft_name(cache->buft),
        cache->layers.size(),
        cache->layers.front().C,
        mb);
}

static const llama_moe_expert_cache_layer * find_layer(const llama_moe_expert_cache * cache, int layer) {
    if (!cache) {
        return nullptr;
    }
    for (const auto & L : cache->layers) {
        if (L.layer == layer) {
            return &L;
        }
    }
    return nullptr;
}

ggml_tensor * llama_moe_expert_cache_get_up(const llama_moe_expert_cache * cache, int layer) {
    const auto * L = find_layer(cache, layer);
    return L ? L->cache_up : nullptr;
}

ggml_tensor * llama_moe_expert_cache_get_gate(const llama_moe_expert_cache * cache, int layer) {
    const auto * L = find_layer(cache, layer);
    return L ? L->cache_gate : nullptr;
}

ggml_tensor * llama_moe_expert_cache_get_down(const llama_moe_expert_cache * cache, int layer) {
    const auto * L = find_layer(cache, layer);
    return L ? L->cache_down : nullptr;
}

// Mutable variant of find_layer for fill paths.
static llama_moe_expert_cache_layer * find_layer_mut(llama_moe_expert_cache * cache, int layer) {
    if (!cache) {
        return nullptr;
    }
    for (auto & L : cache->layers) {
        if (L.layer == layer) {
            return &L;
        }
    }
    return nullptr;
}

bool llama_moe_expert_cache_fill_sync(
        llama_moe_expert_cache * cache,
        int layer,
        int expert) {
    auto * L = find_layer_mut(cache, layer);
    if (!L) {
        return false;
    }
    if (expert < 0 || expert >= L->n_expert) {
        return false;
    }
    // Phase 1: slot == expert id (full-size shadow).
    const int slot = expert;

    const bool use_async = cache->async_fill && cache->copy_backend;

    auto copy_one = [&](ggml_tensor * cache_t, const ggml_tensor * src_t) {
        if (!cache_t || !src_t) {
            return;
        }
        // Each expert occupies cache_t->nb[2] bytes (= ne[0] * ne[1] * element_size,
        // rounded for quantized block alignment). Source layout matches because we
        // cloned the shape exactly.
        const size_t slab_bytes = cache_t->nb[2];
        const size_t src_off    = (size_t) expert * src_t->nb[2];
        const size_t dst_off    = (size_t) slot   * cache_t->nb[2];
        const char * src_data   = (const char *) src_t->data + src_off;
        if (use_async) {
            ggml_backend_tensor_set_async(cache->copy_backend, cache_t, src_data, dst_off, slab_bytes);
        } else {
            ggml_backend_tensor_set(cache_t, src_data, dst_off, slab_bytes);
        }
    };

    copy_one(L->cache_up,   L->src_up);
    copy_one(L->cache_gate, L->src_gate);
    copy_one(L->cache_down, L->src_down);

    L->slot_to_expert[slot]   = expert;
    L->expert_to_slot[expert] = slot;
    L->slot_valid[slot]       = 1;
    L->slot_lru[slot]         = ++L->lru_counter;
    cache->n_fills++;
    return true;
}

void llama_moe_expert_cache_warmup_all(llama_moe_expert_cache * cache) {
    if (!cache) {
        return;
    }
    size_t total = 0;
    for (auto & L : cache->layers) {
        for (int e = 0; e < L.n_expert; ++e) {
            if (llama_moe_expert_cache_fill_sync(cache, L.layer, e)) {
                ++total;
            }
        }
    }
    LLAMA_LOG_INFO("%s: warmed up %zu (layer, expert) cache slots\n", __func__, total);
}

void llama_moe_expert_cache_invalidate_all(llama_moe_expert_cache * cache) {
    if (!cache) {
        return;
    }
    for (auto & L : cache->layers) {
        std::fill(L.expert_to_slot.begin(), L.expert_to_slot.end(), -1);
        std::fill(L.slot_to_expert.begin(), L.slot_to_expert.end(), -1);
        std::fill(L.slot_valid.begin(),     L.slot_valid.end(),     0);
        std::fill(L.slot_lru.begin(),       L.slot_lru.end(),       0);
        L.lru_counter = 0;
    }
}

// MOE2 format:
//   bytes 0..3 : "MOE2"
//   records: int32 hdr[6] = { tid, layer, call_idx, ne0, ne1, dtype } + ne0*ne1*4 bytes
// tid 2 = ffn_moe_topk (i32), the selected expert IDs.
bool llama_moe_expert_cache_load_oracle(
        llama_moe_expert_cache * cache,
        const char * path) {
    if (!cache || !path) {
        return false;
    }
    FILE * f = fopen(path, "rb");
    if (!f) {
        LLAMA_LOG_ERROR("%s: failed to open '%s'\n", __func__, path);
        return false;
    }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "MOE2", 4) != 0) {
        LLAMA_LOG_ERROR("%s: bad magic in '%s' (expected MOE2)\n", __func__, path);
        fclose(f);
        return false;
    }

    cache->oracle_layer_to_idx.assign(cache->layers.size() ? 0 : 0, -1);
    // We need n_layer mapping; widen to the largest layer index we see, then we'll
    // rebuild from cache->layers below.
    std::vector<std::vector<std::vector<int32_t>>> per_layer; // [layer_idx][step]
    int32_t max_layer = -1;
    int32_t max_step  = 0;

    int32_t hdr[6];
    size_t n_topk_records = 0;
    while (fread(hdr, sizeof(int32_t), 6, f) == 6) {
        const int32_t tid    = hdr[0];
        const int32_t layer  = hdr[1];
        const int32_t call   = hdr[2];
        const int32_t ne0    = hdr[3];
        const int32_t ne1    = hdr[4];
        const int32_t dtype  = hdr[5];
        const size_t nbytes  = (size_t) ne0 * ne1 * 4;

        // Keep only the i32 topk records for decode steps (single-token, call >= 1).
        if (tid != 2 || dtype != 1 || call < 1 || ne1 != 1) {
            if (fseek(f, (long) nbytes, SEEK_CUR) != 0) {
                LLAMA_LOG_ERROR("%s: seek past record failed\n", __func__);
                fclose(f);
                return false;
            }
            continue;
        }

        if (layer < 0) {
            fseek(f, (long) nbytes, SEEK_CUR);
            continue;
        }
        if (layer > max_layer) {
            max_layer = layer;
            per_layer.resize(layer + 1);
        }
        if ((size_t) call >= per_layer[layer].size()) {
            per_layer[layer].resize(call + 1);
        }

        std::vector<int32_t> ids(ne0);
        if (fread(ids.data(), 4, ne0, f) != (size_t) ne0) {
            LLAMA_LOG_ERROR("%s: short read for topk record\n", __func__);
            fclose(f);
            return false;
        }
        per_layer[layer][call] = std::move(ids);
        if (call > max_step) {
            max_step = call;
        }
        ++n_topk_records;
    }
    fclose(f);

    if (n_topk_records == 0) {
        LLAMA_LOG_ERROR("%s: no decode-step topk records found in '%s'\n", __func__, path);
        return false;
    }

    // Rebuild per-managed-layer oracle table indexed by cache->layers order.
    cache->oracle_experts.clear();
    cache->oracle_experts.reserve(cache->layers.size());
    cache->oracle_layer_to_idx.assign((size_t) (max_layer + 1), -1);
    for (size_t i = 0; i < cache->layers.size(); ++i) {
        const int L = cache->layers[i].layer;
        cache->oracle_layer_to_idx[L] = (int32_t) i;
        if (L < (int) per_layer.size()) {
            cache->oracle_experts.push_back(std::move(per_layer[L]));
        } else {
            cache->oracle_experts.emplace_back();
        }
    }
    cache->oracle_max_step = max_step;
    cache->oracle_loaded   = true;

    LLAMA_LOG_INFO("%s: oracle loaded from %s: %zu topk records, max_step=%d, %zu managed layers covered\n",
                   __func__, path, n_topk_records, max_step, cache->layers.size());
    return true;
}

bool llama_moe_expert_cache_has_oracle(const llama_moe_expert_cache * cache) {
    return cache && cache->oracle_loaded;
}

void llama_moe_expert_cache_prefill_step(
        llama_moe_expert_cache * cache,
        int call_idx,
        int /*over_fetch*/) {
    if (!cache || !cache->oracle_loaded) {
        return;
    }
    if (call_idx < 1 || call_idx > cache->oracle_max_step) {
        return;
    }
    bool issued_any = false;
    for (size_t i = 0; i < cache->layers.size(); ++i) {
        auto & L = cache->layers[i];
        if ((int) i >= (int) cache->oracle_experts.size()) {
            continue;
        }
        const auto & steps = cache->oracle_experts[i];
        if ((size_t) call_idx >= steps.size()) {
            continue;
        }
        const auto & ids = steps[call_idx];
        for (int32_t e : ids) {
            if (e < 0 || e >= L.n_expert) {
                continue;
            }
            if (L.slot_valid[e]) {
                cache->n_hits++;
                L.slot_lru[e] = ++L.lru_counter;
                continue;
            }
            cache->n_misses++;
            if (llama_moe_expert_cache_fill_sync(cache, L.layer, e)) {
                issued_any = true;
            }
        }
    }
    // Stamp the copy-stream event so subsequent waits on the compute backend block
    // until these fills land. Always record, even if no fills were issued — that way
    // the event reflects "everything submitted up to now is done", which is the
    // contract `llama_moe_expert_cache_wait_fills` expects.
    if (cache->async_fill && cache->fill_event && cache->copy_backend) {
        ggml_backend_event_record(cache->fill_event, cache->copy_backend);
    }
    (void) issued_any;
}

void llama_moe_expert_cache_wait_fills(
        llama_moe_expert_cache * cache,
        ggml_backend_t compute_backend) {
    if (!cache || !cache->async_fill || !cache->fill_event || !compute_backend) {
        return;
    }
    // Cross-stream sync: make the compute stream block on the fill event. Async on
    // the host (queues a stream-wait-event), so the host can return immediately and
    // queue the compute graph on top of the wait.
    ggml_backend_event_wait(compute_backend, cache->fill_event);
}
