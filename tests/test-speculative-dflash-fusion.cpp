#include "common.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "arg.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-ext.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <vector>

// Seed-exact replay test for the fused DFlash2 draft pass (the KV-injection decode and
// the noise decode running as one mixed token/embd pass).
//
// The test generates a tiny dummy DFlash2 model (selector + conv path, so the DFlash2
// lattice walk is exercised) and pins the property split of the fused pass:
//
//   1. with the fused pass enabled and disabled, the per-round accepted draft counts and
//      the generated token streams are identical (the fusion must not change what the
//      drafter proposes or what the verifier accepts);
//   2. the generated stream equals the target's own greedy chain - identical to a run
//      without speculation at all (verification preserves the target's output regardless
//      of the proposals);
//   3. the deferred injection stash is exercised on both paths: the prompt is prefilled
//      in one large batch (immediate injection) and every verify round is a small token
//      batch (deferred injection fused into the next draft pass).

namespace {

struct run_result {
    std::vector<llama_token> tokens;     // generated token stream
    std::vector<int>         n_accepted; // accepted draft tokens per round (excluding the target token)
};

#ifdef _WIN32
static void test_setenv(const char * name, const char * value) {
    _putenv_s(name, value);
}
static void test_unsetenv(const char * name) {
    _putenv_s(name, "");
}
#else
static void test_setenv(const char * name, const char * value) {
    setenv(name, value, 1);
}
static void test_unsetenv(const char * name) {
    unsetenv(name);
}
#endif

// deterministic pseudo-random fill for one tensor, keyed by tensor name
static void set_tensor_data(struct ggml_tensor * tensor, void * /*userdata*/) {
    std::mt19937 rng((uint32_t) std::hash<std::string>{}(tensor->name));
    std::normal_distribution<float> dist(0.0f, 0.1f);

    std::vector<float> tmp(ggml_nelements(tensor));
    for (auto & v : tmp) {
        v = dist(rng);
    }
    ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
}

// a minimal SPM tokenizer: n normal tokens, byte detokenization is never used
static void add_dummy_tokens(struct gguf_context * gguf, uint32_t n_vocab) {
    std::vector<std::string> tok_names(n_vocab);
    std::vector<const char *> toks(n_vocab);
    std::vector<float> scores(n_vocab, 0.0f);
    std::vector<int32_t> types(n_vocab, 1); // NORMAL
    for (uint32_t i = 0; i < n_vocab; ++i) {
        tok_names[i] = "tok" + std::to_string(i);
        toks[i] = tok_names[i].c_str();
    }
    gguf_set_arr_str(gguf, "tokenizer.ggml.tokens", toks.data(), toks.size());
    gguf_set_arr_data(gguf, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(gguf, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    // the DFlash drafter fills its noise blocks with the vocab's mask token
    gguf_set_val_u32(gguf, "tokenizer.ggml.mask_token_id", n_vocab - 1);
}

// test dimensions shared by the target and the drafter
static constexpr uint32_t k_n_vocab = 128;
static constexpr uint32_t k_n_embd  = 64;
static constexpr uint32_t k_n_layer = 2;
static constexpr uint32_t k_block_size = 4;
static constexpr uint32_t k_kernel     = 2;
static constexpr uint32_t k_group      = 2; // n_embd % group == 0
static constexpr uint32_t k_rank       = 2;
static constexpr uint32_t k_top_k      = 2; // n_embd >= top_k * (top_k + 1)

// a minimal dense llama target: 2 layers, hidden 64, 4 heads over 2 KV heads
static gguf_context_ptr make_llama_gguf() {
    gguf_context_ptr gguf(gguf_init_empty());

    const uint32_t n_ctx       = 256;
    const uint32_t n_embd_head = k_n_embd / 4;

    auto kv_name = [](enum llm_kv key) { return llama_arch_kv_name(LLM_ARCH_LLAMA, key); };
    auto set_kv = [&](enum llm_kv key, auto value) {
        const std::string name = kv_name(key);
        using T = decltype(value);
        if constexpr (std::is_same_v<T, uint32_t>) {
            gguf_set_val_u32(gguf.get(), name.c_str(), value);
        } else if constexpr (std::is_same_v<T, float>) {
            gguf_set_val_f32(gguf.get(), name.c_str(), value);
        } else if constexpr (std::is_same_v<T, const char *>) {
            gguf_set_val_str(gguf.get(), name.c_str(), value);
        } else {
            static_assert(!sizeof(T), "unsupported kv type");
        }
    };

    set_kv(LLM_KV_GENERAL_ARCHITECTURE, "llama");
    set_kv(LLM_KV_GENERAL_NAME,         "dflash-fusion-target");
    set_kv(LLM_KV_VOCAB_SIZE,           k_n_vocab);
    set_kv(LLM_KV_CONTEXT_LENGTH,       n_ctx);
    set_kv(LLM_KV_EMBEDDING_LENGTH,     k_n_embd);
    set_kv(LLM_KV_BLOCK_COUNT,          k_n_layer);
    set_kv(LLM_KV_FEED_FORWARD_LENGTH,  96u);
    set_kv(LLM_KV_ATTENTION_HEAD_COUNT, 4u);
    set_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, 2u);
    set_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
    set_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
    set_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1.0e-5f);
    set_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head);
    set_kv(LLM_KV_ROPE_FREQ_BASE,         10000.0f);

    set_kv(LLM_KV_TOKENIZER_MODEL, "llama");
    add_dummy_tokens(gguf.get(), k_n_vocab);

    return gguf;
}

// a minimal DFlash2 drafter over the llama target: block 4, selector top-k 2, the target's
// layers 0..1 as the extract layers (the encoder input width is n_extract * n_embd)
static gguf_context_ptr make_dflash_gguf() {
    gguf_context_ptr gguf(gguf_init_empty());

    const uint32_t n_ctx       = 256;
    const uint32_t n_vocab     = k_n_vocab;
    const uint32_t n_embd      = k_n_embd;
    const uint32_t n_head      = 4;
    const uint32_t n_head_kv   = 2;
    const uint32_t n_embd_head = n_embd / n_head;
    const uint32_t n_layer     = k_n_layer;
    const uint32_t n_ff        = 96;
    auto kv_name = [](enum llm_kv key) { return llama_arch_kv_name(LLM_ARCH_DFLASH, key); };
    auto set_kv = [&](enum llm_kv key, auto value) {
        const std::string name = kv_name(key);
        using T = decltype(value);
        if constexpr (std::is_same_v<T, uint32_t>) {
            gguf_set_val_u32(gguf.get(), name.c_str(), value);
        } else if constexpr (std::is_same_v<T, float>) {
            gguf_set_val_f32(gguf.get(), name.c_str(), value);
        } else if constexpr (std::is_same_v<T, const char *>) {
            gguf_set_val_str(gguf.get(), name.c_str(), value);
        } else {
            static_assert(!sizeof(T), "unsupported kv type");
        }
    };
    auto set_kv_u32 = [&](enum llm_kv key, uint32_t value) {
        const std::string name = kv_name(key);
        gguf_set_val_u32(gguf.get(), name.c_str(), value);
    };
    auto set_kv_arr_u32 = [&](enum llm_kv key, const std::vector<uint32_t> & v) {
        const std::string name = kv_name(key);
        gguf_set_arr_data(gguf.get(), name.c_str(), GGUF_TYPE_UINT32, v.data(), v.size());
    };
    auto set_kv_arr_i32 = [&](enum llm_kv key, const std::vector<int32_t> & v) {
        const std::string name = kv_name(key);
        gguf_set_arr_data(gguf.get(), name.c_str(), GGUF_TYPE_INT32, v.data(), v.size());
    };

    set_kv(LLM_KV_GENERAL_ARCHITECTURE, "dflash");
    set_kv(LLM_KV_GENERAL_NAME,         "dflash-fusion-test");
    set_kv(LLM_KV_TOKENIZER_MODEL, "llama");
    add_dummy_tokens(gguf.get(), n_vocab);
    set_kv(LLM_KV_VOCAB_SIZE,           n_vocab);
    set_kv(LLM_KV_CONTEXT_LENGTH,       n_ctx);
    set_kv(LLM_KV_EMBEDDING_LENGTH,     n_embd);
    set_kv(LLM_KV_BLOCK_COUNT,          n_layer);
    set_kv(LLM_KV_FEED_FORWARD_LENGTH,  n_ff);
    set_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
    set_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv);
    set_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
    set_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
    set_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1.0e-5f);
    set_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head);
    set_kv(LLM_KV_ROPE_FREQ_BASE,         10000.0f);

    // exercise the iSWA draft cache (base + sliding sub-cache) like the production drafter
    set_kv_u32(LLM_KV_ATTENTION_SLIDING_WINDOW,         32u);
    set_kv_arr_u32(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, { 0, 1 });

    set_kv(LLM_KV_DFLASH_BLOCK_SIZE,       k_block_size);
    set_kv(LLM_KV_DFLASH_CONV_KERNEL_SIZE, k_kernel);
    set_kv(LLM_KV_DFLASH_CONV_GROUP_SIZE,  k_group);
    set_kv(LLM_KV_DFLASH_SELECTOR_RANK,    k_rank);
    set_kv(LLM_KV_DFLASH_SELECTOR_TOP_K,   k_top_k);
    set_kv_arr_i32(LLM_KV_TARGET_LAYERS, { 0, (int32_t) n_layer - 1 });

    // the selector path is picked from the raw tensor name, so register its meta entry;
    // the model's own create_tensor call fixes the shape at load time, but the entry needs
    // a real shape here so the file round-trip can carry its (dummy) data
    ggml_tensor t;
    memset(&t, 0, sizeof(t));
    t.type = GGML_TYPE_F32;
    t.ne[0] = n_embd;
    t.ne[1] = k_rank;
    t.nb[0] = sizeof(float);
    t.nb[1] = t.ne[0]*sizeof(float);
    t.nb[2] = t.ne[0]*t.ne[1]*sizeof(float);
    t.nb[3] = t.nb[2];
    ggml_format_name(&t, "selector_hidden.weight");
    gguf_add_tensor(gguf.get(), &t);

    return gguf;
}

static std::string get_tmp_gguf_path() {
    static size_t id = 0;
    return (std::filesystem::temp_directory_path() / ("test-speculative-dflash-fusion-" + std::to_string(++id) + ".gguf")).string();
}

// deterministic prompt over valid non-special token ids
std::vector<llama_token> make_prompt(const llama_vocab * vocab, size_t n_tokens, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<llama_token> dist(0, llama_vocab_n_tokens(vocab) - 101);

    std::vector<llama_token> prompt;
    prompt.reserve(n_tokens);
    while (prompt.size() < n_tokens) {
        const llama_token id = dist(rng);
        if (id >= 0 && !llama_vocab_is_control(vocab, id) && !llama_vocab_is_eog(vocab, id)) {
            prompt.push_back(id);
        }
    }

    return prompt;
}

// replay a deterministic speculative generation, mirroring the speculative-simple example loop
run_result run_speculative(const std::string & model_path, const std::string & draft_path, const std::vector<llama_token> & prompt, bool fusion) {
    run_result result;
    common_params params;
    params.model.path = model_path;
    params.n_ctx      = 256;
    params.n_batch    = 64;
    params.n_ubatch   = 64;
    params.n_parallel = 1;
    params.n_predict  = 48;
    params.cpuparams.n_threads  = 4;
    params.cpuparams_batch.n_threads = 4;
    params.warmup = false; // the DFlash dummy has an encoder-mode graph that a token warmup cannot drive

    // greedy target sampling: the generated stream is the target's own chain
    params.sampling.temp     = 1.0f;
    params.sampling.top_k    = 1;
    params.sampling.seed     = 42;
    params.sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K };

    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
    params.speculative.draft.mparams.path = draft_path;
    params.speculative.draft.n_max = 3;

    test_setenv("LLAMA_SPEC_DFLASH_FUSION", fusion ? "1" : "0");

    auto llama_init_tgt = common_init_from_params(params);
    llama_model * model_tgt = llama_init_tgt->model();
    llama_context * ctx_tgt = llama_init_tgt->context();
    if (!model_tgt || !ctx_tgt) {
        LOG_ERR("%s: failed to initialize the target model and context\n", __func__);
        return result;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    common_params params_dft = common_base_params_to_speculative(params);
    common_speculative_init_result_ptr spec_init =
        common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
    llama_context * ctx_dft = spec_init->context();
    if (!ctx_dft) {
        LOG_ERR("%s: failed to create the draft context\n", __func__);
        return result;
    }

    common_params params_spec = params;
    params_spec.speculative.draft.ctx_tgt = ctx_tgt;
    params_spec.speculative.draft.ctx_dft = ctx_dft;

    common_speculative * spec = common_speculative_init(params_spec.speculative, 1);
    if (spec == nullptr) {
        LOG_ERR("%s: failed to initialize speculative decoding\n", __func__);
        return result;
    }

    const llama_seq_id seq_id = 0;

    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // process the prompt on the target and feed it to the speculative implementation(s)
    {
        common_batch batch_prompt(ctx_tgt);
        for (size_t i = 0; i + 1 < prompt.size(); ++i) {
            batch_prompt.add(prompt[i], (llama_pos) i, seq_id, false);
        }

        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get()) != 0) {
            LOG_ERR("%s: failed to decode the prompt\n", __func__);
            common_speculative_free(spec);
            return result;
        }

        if (!common_speculative_process(spec, batch_prompt)) {
            LOG_ERR("%s: failed to process the speculative prompt batch\n", __func__);
            common_speculative_free(spec);
            return result;
        }
    }

    llama_token id_last = prompt.back();

    llama_tokens prompt_tgt(prompt.begin(), prompt.end() - 1);

    common_speculative_begin(spec, seq_id, prompt_tgt);

    llama_memory_t mem_dft = llama_get_memory(ctx_dft);

    common_batch batch_tgt(ctx_tgt);
    llama_tokens draft;

    while ((int) result.tokens.size() < params.n_predict) {
        const llama_pos pos_round = (llama_pos) prompt_tgt.size(); // first position of this round

        common_speculative_get_draft_params(spec, seq_id) = {
            /* .drafting   = */ true,
            /* .n_max      = */ params.speculative.draft.n_max,
            /* .pos0       = */ pos_round,
            /* .id_last    = */ id_last,
            /* .prompt     = */ &prompt_tgt,
            /* .result     = */ &draft,
        };
        common_speculative_draft(spec);

        // drop the ephemeral noise-block cells before verification, so that process()
        // below re-feeds the verify rows exactly once
        GGML_ASSERT(llama_memory_seq_rm(mem_dft, seq_id, pos_round, -1));

        // evaluate [id_last, draft...] on the target
        batch_tgt.clear();
        batch_tgt.add(id_last, pos_round, seq_id, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            batch_tgt.add(draft[i], pos_round + 1 + (llama_pos) i, seq_id, true);
        }

        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get()) != 0) {
            LOG_ERR("%s: failed to decode the verify batch\n", __func__);
            common_speculative_free(spec);
            return result;
        }

        // feed the batch to the speculative implementation(s)
        if (!common_speculative_process(spec, batch_tgt)) {
            LOG_ERR("%s: failed to process the speculative verify batch\n", __func__);
            common_speculative_free(spec);
            return result;
        }

        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);

        GGML_ASSERT(ids.size() > 0);

        result.n_accepted.push_back((int) ids.size() - 1);

        bool eog = false;
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            result.tokens.push_back(id_last);

            if (llama_vocab_is_eog(vocab, id_last)) {
                eog = true;
                break;
            }
        }

        draft.clear();

        // drop any rejected draft tokens from both contexts
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, (llama_pos) prompt_tgt.size(), -1));
        GGML_ASSERT(llama_memory_seq_rm(mem_dft, seq_id, (llama_pos) prompt_tgt.size(), -1));

        if (eog) {
            break;
        }
    }

    common_speculative_free(spec);
    test_unsetenv("LLAMA_SPEC_DFLASH_FUSION");

    return result;
}

// plain greedy generation without speculation: the reference stream
run_result run_plain(const std::string & model_path, const std::vector<llama_token> & prompt) {
    run_result result;

    common_params params;
    params.model.path = model_path;
    params.n_ctx      = 256;
    params.n_batch    = 64;
    params.n_ubatch   = 64;
    params.n_parallel = 1;
    params.n_predict  = 48;
    params.cpuparams.n_threads  = 4;
    params.cpuparams_batch.n_threads = 4;
    params.warmup = false;
    params.sampling.temp     = 1.0f;
    params.sampling.top_k    = 1;
    params.sampling.seed     = 42;
    params.sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K };

    auto llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("%s: failed to initialize the model and context\n", __func__);
        return result;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_seq_id seq_id = 0;

    common_sampler_ptr smpl(common_sampler_init(model, params.sampling));

    {
        common_batch batch_prompt(ctx);
        for (size_t i = 0; i < prompt.size(); ++i) {
            batch_prompt.add(prompt[i], (llama_pos) i, seq_id, i + 1 == prompt.size());
        }

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get()) != 0) {
            LOG_ERR("%s: failed to decode the prompt\n", __func__);
            return result;
        }
    }

    for (int i = 0; i < params.n_predict; ++i) {
        const llama_token id = common_sampler_sample(smpl.get(), ctx, -1);
        common_sampler_accept(smpl.get(), id, true);

        result.tokens.push_back(id);

        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }

        common_batch batch(ctx);
        batch.add(id, (llama_pos) (prompt.size() + i), seq_id, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: failed to decode token %d\n", __func__, i);
            return result;
        }
    }

    return result;
}

bool streams_equal(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

// speculative rounds accept multiple tokens at once, so a speculative run can overshoot
// n_predict by up to one draft width; the streams must agree on the common prefix
bool stream_prefix_equal(const std::vector<llama_token> & spec, const std::vector<llama_token> & ref) {
    return spec.size() >= ref.size() && std::equal(ref.begin(), ref.end(), spec.begin());
}

bool accepts_equal(const std::vector<int> & a, const std::vector<int> & b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

} // namespace

int main(int argc, char ** argv) {
    (void) argc;

    // write_file_from_model builds the model in memory from metadata only, then writes the
    // GGUF file by hand: llama_model_save_to_file serializes the generic hyperparameters
    // only and would drop the DFlash-specific metadata the loader needs
    auto write_file_from_model = [&](gguf_context * gguf, const char * what, const std::string & path) {
        llama_model_ptr model(llama_model_init_from_user(gguf, set_tensor_data, nullptr, llama_model_default_params()));
        if (!model) {
            LOG_ERR("%s: failed to create the dummy %s model\n", __func__, what);
            return false;
        }
        gguf_context_ptr out(gguf_init_empty());
        gguf_set_kv(out.get(), gguf);
        std::set<std::string> added;
        for (const auto & [name, tensor] : model->tensors_by_name) {
            if (added.insert(name).second) {
                gguf_add_tensor(out.get(), tensor);
            }
        }
        gguf_write_to_file(out.get(), path.c_str(), false);
        return true;
    };

    const std::string tgt_path = get_tmp_gguf_path();
    const std::string dft_path = get_tmp_gguf_path();

    {
        gguf_context_ptr gguf = make_llama_gguf();
        if (!write_file_from_model(gguf.get(), "target", tgt_path)) {
            return 1;
        }
    }
    {
        gguf_context_ptr gguf = make_dflash_gguf();
        if (!write_file_from_model(gguf.get(), "drafter", dft_path)) {
            return 1;
        }
    }

    const auto prompt = [&]() {
        // load the vocab just for the prompt generation
        llama_model_ptr model(llama_model_load_from_file(tgt_path.c_str(), llama_model_default_params()));
        GGML_ASSERT(model);
        return make_prompt(llama_model_get_vocab(model.get()), 24, 7);
    }();

    printf("running the fused draft pass\n");
    run_result fused = run_speculative(tgt_path, dft_path, prompt, true);
    printf("running the unfused draft passes\n");
    run_result unfused = run_speculative(tgt_path, dft_path, prompt, false);
    printf("running the plain greedy reference\n");
    run_result plain = run_plain(tgt_path, prompt);

    int ret = 0;

    if (!accepts_equal(fused.n_accepted, unfused.n_accepted)) {
        LOG_ERR("%s: the fused and unfused runs accepted different draft counts\n", __func__);
        ret = 1;
    }
    if (!streams_equal(fused.tokens, unfused.tokens)) {
        LOG_ERR("%s: the fused and unfused runs generated different streams\n", __func__);
        ret = 1;
    }
    if (!stream_prefix_equal(fused.tokens, plain.tokens)) {
        LOG_ERR("%s: the speculative stream does not match the plain greedy reference\n", __func__);
        ret = 1;
    }
    if (fused.n_accepted.empty() || fused.tokens.size() < plain.tokens.size()) {
        LOG_ERR("%s: the fused run produced no drafts or fell short of the reference\n", __func__);
        ret = 1;
    }

    std::filesystem::remove(tgt_path);
    std::filesystem::remove(dft_path);

    if (ret == 0) {
        printf("%s: ok - fused == unfused == plain greedy (%zu tokens, %zu rounds)\n",
                __func__, fused.tokens.size(), fused.n_accepted.size());
    }
    return ret;
}
