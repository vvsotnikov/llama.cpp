#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "arg.h"
#include "log.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

// Seed-exact replay test for the bounded draft context (see
// common_params_speculative_draft::n_ctx, --spec-draft-ctx).
//
// The draft context is no longer sized at the full target context: it retains only the most
// recent `window` tokens per sequence (the oldest cells are dropped, positions stay absolute).
// The test pins the property split that the bounded window implies:
//
//   1. with the window inactive (the sequence never exceeds it), the drafter's KV content is
//      identical to the full-context draft, so the per-round accepted draft counts and the
//      generated token stream must be bit-identical to the legacy (--spec-draft-ctx 0) run;
//   2. with the window active (the sequence exceeds it), the drafter sees less context and its
//      proposals may differ, but the accepted token stream is still exactly the target's own
//      greedy chain - identical to the legacy run and to a run without speculation at all
//      (verification preserves the target's output regardless of the proposals);
//   3. the draft context allocation is decoupled from the target context: a windowed draft
//      context is created smaller than the target context, and the draft sequence's position
//      span stays bounded while the target sequence grows past the window.

namespace {

struct run_result {
    std::vector<llama_token> tokens;     // generated token stream
    std::vector<int>         n_accepted; // accepted draft tokens per round (excluding the target token)
    uint32_t n_ctx_dft = 0;              // created draft context size
    llama_pos span_dft  = 0;             // final position span of the draft sequence
};

common_params make_params(const common_params & base, int32_t draft_n_ctx) {
    common_params params = base;

    params.n_ctx        = 1024;
    params.n_batch      = 256;
    params.n_ubatch     = 256;
    params.n_parallel   = 1;
    params.n_predict    = 96;

    // greedy target sampling: the generated stream is the target's own chain
    params.sampling.temp     = 1.0f;
    params.sampling.top_k    = 1;
    params.sampling.seed     = 42;
    params.sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K };

    if (draft_n_ctx >= 0) {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
        params.speculative.draft.mparams.path = params.model.path; // the target drafts from itself
        params.speculative.draft.n_max = 8;
        params.speculative.draft.n_ctx = draft_n_ctx;
    } else {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_NONE };
    }

    return params;
}

// deterministic pseudo-random prompt over valid non-special token ids
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
run_result run_speculative(common_params params, const std::vector<llama_token> & prompt) {
    run_result result;

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

    result.n_ctx_dft = llama_n_ctx(ctx_dft);

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

    int n_past = (int) prompt.size() - 1;
    common_speculative_begin(spec, seq_id, prompt_tgt);

    llama_memory_t mem_dft = llama_get_memory(ctx_dft);

    common_batch batch_tgt(ctx_tgt);
    llama_tokens draft;

    while ((int) result.tokens.size() < params.n_predict) {
        const llama_pos pos_round = (llama_pos) prompt_tgt.size(); // first position of this round

        common_speculative_get_draft_params(spec, seq_id) = {
            /* .drafting   = */ true,
            /* .n_max      = */ 8,
            /* .pos0       = */ pos_round,
            /* .id_last    = */ id_last,
            /* .prompt     = */ &prompt_tgt,
            /* .result     = */ &draft,
        };
        common_speculative_draft(spec);

        // reset the draft context to the accepted prefix before verification, so that
        // process() below re-feeds the verify rows exactly once
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
        n_past = (int) prompt_tgt.size();
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1));
        GGML_ASSERT(llama_memory_seq_rm(mem_dft, seq_id, n_past, -1));

        if (eog) {
            break;
        }
    }

    const llama_pos pos_min = llama_memory_seq_pos_min(mem_dft, seq_id);
    const llama_pos pos_max = llama_memory_seq_pos_max(mem_dft, seq_id);
    result.span_dft = pos_min < 0 ? 0 : pos_max - pos_min + 1;

    common_speculative_free(spec);

    return result;
}

// plain greedy generation without speculation: the reference stream
run_result run_plain(common_params params, const std::vector<llama_token> & prompt) {
    run_result result;

    common_params & params_plain = params;
    params_plain.speculative = common_params_speculative{}; // no speculation

    auto llama_init = common_init_from_params(params_plain);
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("%s: failed to initialize the model and context\n", __func__);
        return result;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_seq_id seq_id = 0;

    common_sampler_ptr smpl(common_sampler_init(model, params_plain.sampling));

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

    for (int i = 0; i < params_plain.n_predict; ++i) {
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
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    LOG("%s: using model '%s'\n", __func__, params.model.path.c_str());

    llama_backend_init();

    // a prompt long enough that a 64-token window trims, but a 512-token window does not
    std::vector<llama_token> prompt;
    {
        auto probe = make_params(params, -1);
        auto llama_init = common_init_from_params(probe);
        if (!llama_init->model()) {
            LOG_ERR("%s: failed to load the model\n", __func__);
            llama_backend_free();
            return 1;
        }
        prompt = make_prompt(llama_model_get_vocab(llama_init->model()), 128, 1234);
    }

    int fails = 0;

    // legacy full-context draft (--spec-draft-ctx 0)
    const run_result legacy = run_speculative(make_params(params, 0), prompt);
    if (legacy.tokens.empty()) {
        LOG_ERR("%s: the legacy run failed\n", __func__);
        llama_backend_free();
        return 1;
    }

    // window larger than the whole sequence: the bounded context is active, the trim never fires
    const run_result big = run_speculative(make_params(params, 512), prompt);

    // window smaller than the sequence: the trim is exercised
    const run_result small = run_speculative(make_params(params, 64), prompt);

    // no-speculation greedy reference: the target's own chain
    const run_result nospec = run_plain(make_params(params, -1), prompt);

    // 1. the inactive-window run is bit-identical to the legacy run
    if (!streams_equal(legacy.tokens, big.tokens) || !accepts_equal(legacy.n_accepted, big.n_accepted)) {
        LOG_ERR("%s: the inactive-window run diverged from the legacy run (%zu vs %zu tokens, %zu vs %zu rounds)\n",
                __func__, legacy.tokens.size(), big.tokens.size(), legacy.n_accepted.size(), big.n_accepted.size());
        fails++;
    }

    // 2. the draft context allocation is decoupled from the target context
    if (big.n_ctx_dft >= 1024) {
        LOG_ERR("%s: the inactive-window draft context was not bounded (%u >= target 1024)\n", __func__, big.n_ctx_dft);
        fails++;
    }
    if (small.n_ctx_dft >= big.n_ctx_dft) {
        LOG_ERR("%s: the small-window draft context was not smaller (%u >= %u)\n", __func__, small.n_ctx_dft, big.n_ctx_dft);
        fails++;
    }

    // 3. every windowed run keeps the target's own greedy stream
    if (!stream_prefix_equal(big.tokens, nospec.tokens)) {
        LOG_ERR("%s: the inactive-window stream diverged from the no-speculation stream\n", __func__);
        fails++;
    }
    if (!stream_prefix_equal(small.tokens, nospec.tokens)) {
        LOG_ERR("%s: the small-window stream diverged from the no-speculation stream\n", __func__);
        fails++;
    }

    // 4. the draft sequence's position span stays bounded while the target grows past the window
    if (small.span_dft > 64 + 9) {
        LOG_ERR("%s: the small-window draft span is unbounded (%d > 73)\n", __func__, (int) small.span_dft);
        fails++;
    }

    LOG("%s: legacy %zu tokens (%zu rounds), inactive-window %zu tokens (draft ctx %u), "
        "small-window %zu tokens (draft ctx %u, draft span %d), no-spec %zu tokens\n",
        __func__, legacy.tokens.size(), legacy.n_accepted.size(), big.tokens.size(), big.n_ctx_dft,
        small.tokens.size(), small.n_ctx_dft, (int) small.span_dft, nospec.tokens.size());

    llama_backend_free();

    if (fails == 0) {
        LOG("%s: all checks passed\n", __func__);
        return 0;
    }

    LOG_ERR("%s: %d checks failed\n", __func__, fails);
    return 1;
}
