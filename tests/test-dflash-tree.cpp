#include "common.h"
#include "llama-cpp.h"
#include "llama-tree.h"
#include "sampling.h"
#include "../src/llama-ext.h"
#include "../src/llama-memory-hybrid.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static void compare(const float * a, const float * b, int32_t n, const char * label) {
    float maximum = 0.0f;
    double square = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        check(std::isfinite(a[i]) && std::isfinite(b[i]), "finite logits");
        const float d = std::fabs(a[i] - b[i]);
        maximum = std::max(maximum, d);
        square += (double) d * d;
    }
    const double rms = std::sqrt(square / n);
    printf("%s: max_abs=%g rms=%g\n", label, maximum, rms);
    check(maximum < 0.025f && rms < 0.003, label);
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 77;
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
    check(model != nullptr, "model load");
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = cp.n_ubatch = 128;
    cp.n_seq_max = 1;
    cp.n_rs_seq = 31;
    cp.n_outputs_max = cp.n_outputs_max_per_seq = 32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.kv_unified = true;
    if (std::getenv("TREE_TEST_KV_Q8")) {
        cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
    }
    llama_context_ptr tree(llama_init_from_model(model.get(), cp));
    llama_context_ptr reference(llama_init_from_model(model.get(), cp));
    check(tree && reference && llama_tree_supported(tree.get()), "CUDA tree support");
    const bool trace_layers = std::getenv("TREE_TEST_LAYER_TRACE") != nullptr;
    if (trace_layers) {
        for (uint32_t il = 0; il < (uint32_t) llama_model_n_layer(model.get()); ++il) {
            llama_set_embeddings_layer_inp(tree.get(), il, true);
            llama_set_embeddings_layer_inp(reference.get(), il, true);
        }
    }
    const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    common_params_sampling sampling;
    sampling.temp = 1.0f;
    sampling.top_k = 20;
    sampling.top_p = 0.95f;
    sampling.min_p = 0.0f;
    sampling.backend_sampling = false;
    common_sampler_ptr tree_sampler(common_sampler_init(model.get(), sampling));
    common_sampler_ptr serial_sampler(common_sampler_init(model.get(), sampling));
    const llama_tokens prefix = common_tokenize(tree.get(), "Review this C++ code carefully and explain each operation.\nint square(int value) { return value * value; }\n", true);
    const llama_tokens candidates = common_tokenize(tree.get(), "The function computes the square of its input. It returns an integer and has no side effects. We should check for overflow. Add tests for zero, negative numbers, large values, and the smallest representable integer. Explain what each case validates before changing the implementation.", false);
    check(candidates.size() >= 32, "candidate fixture length");
    std::vector<std::vector<int32_t>> shapes = {
        {-1, 0, 1, 1, 2, 3, 0, 6, 7, 7, 9, 0, 11, 12, 12, 14},
        {-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
    };
    shapes.push_back({-1, 0, 1, 1, 0, 4, 4, 0, 7, 7, 0, 10, 10, 0, 13, 13});
    for (int n : {16, 32}) {
        std::vector<int32_t> comb(n, -1);
        for (int i = 1; i < n; ++i) {
            comb[i] = i <= n / 2 ? i - 1 : n - i - 1;
        }
        shapes.push_back(comb);
    }
    auto prefill = [&](llama_context * ctx) {
        llama_memory_clear(llama_get_memory(ctx), true);
        common_batch batch(ctx);
        for (size_t i = 0; i < prefix.size(); ++i) {
            batch.add(prefix[i], i, 0, false);
        }
        check(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "prefix decode");
    };
    const llama_pos pos0 = prefix.size();
    int checks = 0;
    for (const auto & parents : shapes) {
        const int n = parents.size();
        prefill(tree.get());
        common_batch batch(tree.get());
        for (int i = 0; i < n; ++i) {
            batch.add(candidates[i], pos0 + i, 0, true);
        }
        auto invalid = parents;
        invalid[1] = 1;
        check(llama_tree_decode(tree.get(), batch.get(), invalid.data(), n) == -1, "reject cyclic parent");
        check(llama_tree_decode(tree.get(), batch.get(), parents.data(), n) == 0, "tree decode");
        std::vector<float> logits((size_t) vocab * n);
        for (int i = 0; i < n; ++i) {
            const float * row = llama_get_logits_ith(tree.get(), i);
            std::copy(row, row + vocab, logits.data() + (size_t) i * vocab);
        }
        for (int node = 0; node < n; ++node) {
            std::vector<int32_t> path;
            for (int i = node; i >= 0; i = parents[i]) {
                path.push_back(i);
            }
            std::reverse(path.begin(), path.end());
            prefill(reference.get());
            common_batch poisoned(reference.get());
            for (int i = 0; i < n; ++i) {
                const bool ancestor = std::find(path.begin(), path.end(), i) != path.end();
                poisoned.add(ancestor ? candidates[i] : (candidates[i] + 997) % vocab, pos0 + i, 0, true);
            }
            check(llama_tree_decode(reference.get(), poisoned.get(), parents.data(), n) == 0, "poisoned siblings decode");
            const float * isolated = llama_get_logits_ith(reference.get(), node);
            check(std::equal(isolated, isolated + vocab, logits.data() + (size_t) node * vocab), "sibling isolation: bitwise logits");
            const int32_t root_row = 0;
            check(llama_tree_commit(reference.get(), &root_row, 1), "clear poisoned tree");
            printf("node %d sibling isolation: bitwise pass\n", node);
            prefill(reference.get());
            {
                // Keep masked KV slots in place so attention reductions use the same order.
                auto * mem = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(reference.get()));
                auto & cells = const_cast<llama_kv_cells &>(mem->get_mem_attn()->get_cells(0));
                for (int i = 0; i <= node; ++i) {
                    if (std::find(path.begin(), path.end(), i) == path.end()) {
                        cells.pos_set(prefix.size() + i, pos0 + 64 + i);
                        cells.seq_add(prefix.size() + i, 0);
                    }
                }
            }
            common_batch serial(reference.get());
            for (int i = 0; i < n; ++i) {
                serial.add(candidates[i < (int) path.size() ? path[i] : i], pos0 + i, 0, true);
            }
            check(llama_process(reference.get(), LLAMA_PROCESS_TYPE_DECODE, serial.get()) == 0, "serial path");
            if (trace_layers && node == 3) {
                const int32_t width = llama_model_n_embd(model.get());
                for (uint32_t il = 0; il < (uint32_t) llama_model_n_layer(model.get()); ++il) {
                    const float * a = llama_get_embeddings_layer_inp(tree.get(), il) + (size_t) node * width;
                    const float * b = llama_get_embeddings_layer_inp(reference.get(), il) + (path.size() - 1) * width;
                    float maximum = 0.0f;
                    for (int32_t j = 0; j < width; ++j) {
                        maximum = std::max(maximum, std::fabs(a[j] - b[j]));
                    }
                    printf("layer %u branch max_abs=%g\n", il, maximum);
                }
            }
            compare(logits.data() + (size_t) node * vocab,
                    llama_get_logits_ith(reference.get(), path.size() - 1), vocab, "tree node vs serial ancestors");
            for (uint32_t seed = 42; seed < 47; ++seed) {
                check(common_sampler_sample_tree(tree_sampler.get(), tree.get(), node, seed, pos0 + path.size()) ==
                      common_sampler_sample_tree(serial_sampler.get(), reference.get(), path.size() - 1, seed, pos0 + path.size()),
                      "sampled token matches serial ancestors");
            }
            ++checks;
        }
        // Select a nonterminal row so that siblings and later physical rows must be discarded.
        std::vector<int32_t> kept;
        for (int i = 9; i >= 0; i = parents[i]) {
            kept.push_back(i);
        }
        std::reverse(kept.begin(), kept.end());
        check(llama_tree_commit(tree.get(), kept.data(), kept.size()), "tree path commit");
        check(llama_memory_seq_pos_max(llama_get_memory(tree.get()), 0) == pos0 + (int) kept.size() - 1, "committed position");
        const size_t state_size = llama_state_seq_get_size(tree.get(), 0);
        std::vector<uint8_t> state(state_size);
        check(llama_state_seq_get_data(tree.get(), state.data(), state.size(), 0) == state.size(), "save selected state");
        common_batch next(tree.get());
        next.add(candidates[n - 1], pos0 + kept.size(), 0, true);
        check(llama_process(tree.get(), LLAMA_PROCESS_TYPE_DECODE, next.get()) == 0, "decode after tree commit");
        std::vector<float> next_logits(llama_get_logits(tree.get()), llama_get_logits(tree.get()) + vocab);
        prefill(reference.get());
        auto * memory = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(reference.get()));
        auto & cells = const_cast<llama_kv_cells &>(memory->get_mem_attn()->get_cells(0));
        for (int i = 0; i <= kept.back(); ++i) {
            if (std::find(kept.begin(), kept.end(), i) == kept.end()) {
                cells.pos_set(prefix.size() + i, pos0 + 64 + i);
                cells.seq_add(prefix.size() + i, 0);
            }
        }
        common_batch serial(reference.get());
        for (int i = 0; i < n; ++i) {
            serial.add(candidates[i < (int) kept.size() ? kept[i] : i], pos0 + i, 0, true);
        }
        check(llama_process(reference.get(), LLAMA_PROCESS_TYPE_DECODE, serial.get()) == 0, "serial committed path");
        check(llama_memory_seq_rm(llama_get_memory(reference.get()), 0, pos0 + kept.size(), -1), "serial prefix rollback");
        const size_t reference_size = llama_state_seq_get_size(reference.get(), 0);
        std::vector<uint8_t> reference_state(reference_size);
        check(llama_state_seq_get_data(reference.get(), reference_state.data(), reference_state.size(), 0) == reference_size, "save serial state");
        if (state != reference_state) {
            size_t different = 0;
            size_t first = std::min(state.size(), reference_state.size());
            for (size_t i = 0; i < std::min(state.size(), reference_state.size()); ++i) {
                if (state[i] != reference_state[i]) {
                    first = std::min(first, i);
                    ++different;
                }
            }
            printf("state comparison: tree=%zu serial=%zu differing=%zu first=%zu\n", state.size(), reference_state.size(), different, first);
        }
        check(state == reference_state, "committed state matches serial state byte for byte");
        common_batch reference_next(reference.get());
        reference_next.add(candidates[n - 1], pos0 + kept.size(), 0, true);
        check(llama_process(reference.get(), LLAMA_PROCESS_TYPE_DECODE, reference_next.get()) == 0, "serial committed continuation");
        compare(next_logits.data(), llama_get_logits(reference.get()), vocab, "committed continuation");
        llama_memory_clear(llama_get_memory(tree.get()), true);
        llama_memory_clear(llama_get_memory(reference.get()), true);
        check(llama_state_seq_set_data(tree.get(), state.data(), state.size(), 0) == state.size(), "restore selected state");
        check(llama_state_seq_set_data(reference.get(), reference_state.data(), reference_state.size(), 0) == reference_state.size(), "restore serial state");
        check(llama_process(tree.get(), LLAMA_PROCESS_TYPE_DECODE, next.get()) == 0, "decode restored continuation");
        check(llama_process(reference.get(), LLAMA_PROCESS_TYPE_DECODE, reference_next.get()) == 0, "decode restored serial continuation");
        compare(llama_get_logits(tree.get()), llama_get_logits(reference.get()), vocab, "restored continuation");
        checks += 2;
    }
    printf("PASS: %d full-vocabulary comparisons\n", checks);
    return 0;
}
