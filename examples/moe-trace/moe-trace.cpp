// moe-trace.cpp
// Step-1 instrumentation for the MoE speculative-prefetch PoC.
//
// Greedily generates from a Qwen3-MoE GGUF and dumps, per layer per llama_decode
// call, three tensors used by the offline cross-layer routing-predictability study:
//   tid 0  ffn_inp        [n_embd,        n_tokens]  f32  post-attention residual (a_L)
//   tid 1  ffn_moe_probs  [n_expert,      n_tokens]  f32  softmax router distribution
//   tid 2  ffn_moe_topk   [n_expert_used, n_tokens]  i32  true selected expert ids
//
// Output file format ("MOE2"):
//   bytes 0..3 : magic "MOE2"
//   then a stream of records:
//     int32 hdr[6] = { tid, layer, call_idx, ne0, ne1, dtype }   dtype 0=f32 1=i32
//     ne0*ne1 elements of 4 bytes each (row-major over [ne1, ne0])
//
// Built as a llama.cpp example: cmake --build <build> --target llama-moe-trace

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"  // ggml_backend_cpu_buffer_type for -ncmoe overrides

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

struct trace_state {
    FILE *  out      = nullptr;
    int32_t call_idx = 0;
    long    n_rec    = 0;
};

// llama.cpp's graph callback names per-layer tensors "<base>-<layer>"
static int parse_layer(const char * name) {
    const char * d = strrchr(name, '-');
    if (!d || !d[1]) {
        return -1;
    }
    for (const char * p = d + 1; *p; ++p) {
        if (*p < '0' || *p > '9') {
            return -1;
        }
    }
    return atoi(d + 1);
}

// ggml_backend_sched_eval_callback: invoked per graph node, twice (ask=true, then ask=false)
static bool moe_trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (trace_state *) user_data;

    int tid = -1;
    if      (strncmp(t->name, "ffn_inp-",       8)  == 0) tid = 0;
    else if (strncmp(t->name, "ffn_moe_probs-", 14) == 0) tid = 1;
    else if (strncmp(t->name, "ffn_moe_topk-",  13) == 0) tid = 2;

    if (ask) {
        return tid >= 0; // only request data for the tensors we record
    }
    if (tid < 0) {
        return true;
    }

    const int layer = parse_layer(t->name);
    if (layer < 0) {
        return true;
    }

    int32_t dtype;
    if      (t->type == GGML_TYPE_F32) dtype = 0;
    else if (t->type == GGML_TYPE_I32) dtype = 1;
    else                               return true; // unexpected type, skip

    if (!ggml_is_contiguous(t)) {
        return true;
    }

    const int32_t ne0    = (int32_t) t->ne[0];
    const int32_t ne1    = (int32_t) t->ne[1];
    const size_t  nbytes = (size_t) ne0 * ne1 * 4;

    std::vector<uint8_t> tmp;
    const void * data;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = t->data;
    } else {
        tmp.resize(nbytes);
        ggml_backend_tensor_get(t, tmp.data(), 0, nbytes);
        data = tmp.data();
    }

    const int32_t hdr[6] = { tid, layer, st->call_idx, ne0, ne1, dtype };
    fwrite(hdr,  sizeof(int32_t), 6, st->out);
    fwrite(data, 1, nbytes, st->out);
    st->n_rec++;
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string out_path     = "trace.bin";
    std::string prompt       = "Explain why the sky is blue.";
    int  n_predict           = 400;
    int  ngl                 = 0;   // CPU backend by default: no op fusion, host-readable tensors
    int  n_threads           = 24;
    int  n_cpu_moe           = 0;   // -ncmoe N: offload first N layers' MoE expert tensors to CPU
    int  moe_cache_size      = 0;   // -moecache C: per-layer GPU expert cache slots (0 = disabled)
    std::string oracle_path  = "";  // --oracle path: replay MOE2 trace to drive selective fills
    int  lookahead           = 0;   // --lookahead K: speculative prefill K steps ahead each iter
    bool wrap                = true; // wrap prompt in the Qwen chat template
    bool no_mmap             = false;
    bool no_trace            = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", what); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")          model_path = next("-m");
        else if (a == "-o")          out_path   = next("-o");
        else if (a == "-p")          prompt     = next("-p");
        else if (a == "-n")          n_predict  = atoi(next("-n"));
        else if (a == "-ngl")        ngl        = atoi(next("-ngl"));
        else if (a == "-t")          n_threads  = atoi(next("-t"));
        else if (a == "-ncmoe" || a == "--n-cpu-moe")    n_cpu_moe      = atoi(next("-ncmoe"));
        else if (a == "-moecache" || a == "--moe-cache-size") moe_cache_size = atoi(next("-moecache"));
        else if (a == "--oracle")    oracle_path = next("--oracle");
        else if (a == "--lookahead") lookahead  = atoi(next("--lookahead"));
        else if (a == "--raw")       wrap       = false;
        else if (a == "--no-mmap")   no_mmap    = true;
        else if (a == "--no-trace")  no_trace   = true;
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty()) {
        fprintf(stderr,
            "usage: %s -m model.gguf [-p prompt] [-o out.bin] [-n n_predict] [-ngl n] [-t threads] "
            "[-ncmoe N] [-moecache C] [--oracle trace.bin] [--raw] [--no-mmap] [--no-trace]\n",
            argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.use_mmap     = !no_mmap;

    // -ncmoe N: offload the first N layers' MoE expert tensors to CPU via per-layer
    // tensor_buft_overrides matching common's -ncmoe pattern. Storage must outlive load.
    std::vector<std::string> ncmoe_patterns;
    std::vector<llama_model_tensor_buft_override> overrides;
    if (n_cpu_moe > 0) {
        ncmoe_patterns.reserve(n_cpu_moe);
        for (int L = 0; L < n_cpu_moe; L++) {
            char buf[80];
            snprintf(buf, sizeof(buf), "blk\\.%d\\.ffn_(up|down|gate|gate_up)_(ch|)exps", L);
            ncmoe_patterns.emplace_back(buf);
        }
        overrides.reserve(ncmoe_patterns.size() + 1);
        for (const auto & p : ncmoe_patterns) {
            overrides.push_back({ p.c_str(), ggml_backend_cpu_buffer_type() });
        }
        overrides.push_back({ nullptr, nullptr });  // sentinel
        mp.tensor_buft_overrides = overrides.data();
    }

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const std::string text = wrap
        ? ("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n")
        : prompt;

    const int n_prompt = -llama_tokenize(vocab, text.c_str(), (int) text.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, text.c_str(), (int) text.size(), tokens.data(), n_prompt, true, true) < 0) {
        fprintf(stderr, "tokenize failed\n");
        return 1;
    }
    fprintf(stderr, "prompt tokens: %d\n", n_prompt);

    trace_state st;
    if (!no_trace) {
        st.out = fopen(out_path.c_str(), "wb");
        if (!st.out) { fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 1; }
        fwrite("MOE2", 1, 4, st.out);
    }

    llama_context_params cp     = llama_context_default_params();
    cp.n_ctx                    = n_prompt + n_predict + 16;
    cp.n_batch                  = n_prompt > 2048 ? n_prompt : 2048;
    cp.n_ubatch                 = cp.n_batch;
    cp.n_threads                = n_threads;
    cp.n_threads_batch          = n_threads;
    if (!no_trace) {
        cp.cb_eval              = moe_trace_cb;
        cp.cb_eval_user_data    = &st;
    }
    cp.moe_expert_cache_size    = (uint32_t) moe_cache_size;
    cp.no_perf                  = false;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    // [EXPERIMENTAL] MoE expert prefill setup.
    if (moe_cache_size > 0) {
        if (!oracle_path.empty()) {
            if (!llama_moe_oracle_load(ctx, oracle_path.c_str())) {
                fprintf(stderr, "failed to load oracle '%s'\n", oracle_path.c_str());
                return 1;
            }
            // Pre-issue fills for step 1 (+ lookahead). These run on the copy stream
            // while the host walks into the first decode call.
            llama_moe_oracle_prefill(ctx, 1);
            for (int k = 1; k <= lookahead; k++) {
                llama_moe_oracle_prefill(ctx, 1 + k);
            }
        } else {
            // No oracle: warm the entire cache to provide the Phase-1 ceiling
            // reference (output identical to running with experts on GPU directly).
            llama_moe_cache_warmup_all(ctx);
        }
    }

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // prefill
    st.call_idx = 0;
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_prompt))) {
        fprintf(stderr, "prefill decode failed\n");
        return 1;
    }

    const auto t_decode_start = std::chrono::steady_clock::now();
    int n_decode = 0;
    for (int call = 1; n_decode < n_predict; ++call) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) {
            fprintf(stderr, "\n[eog after %d generated tokens]\n", n_decode);
            break;
        }
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) { fwrite(buf, 1, n, stderr); }

        st.call_idx = call;
        if (!oracle_path.empty()) {
            // Queue the compute-stream wait FIRST so it captures the event state
            // produced by the previous iteration's prefills (i.e. fills for step
            // `call`). Then issue prefills for the NEXT step(s) — those run on the
            // copy stream concurrently with the decode we're about to submit.
            llama_moe_oracle_wait_fills(ctx);
            llama_moe_oracle_prefill(ctx, call + 1 + lookahead);
        }
        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) {
            fprintf(stderr, "decode failed at step %d\n", n_decode);
            break;
        }
        n_decode++;
    }
    const auto t_decode_end = std::chrono::steady_clock::now();
    const double dt_s = std::chrono::duration<double>(t_decode_end - t_decode_start).count();
    const double tok_per_s = n_decode > 0 ? n_decode / dt_s : 0.0;

    if (st.out) fclose(st.out);
    fprintf(stderr,
        "\ndone: %d generated tokens in %.2fs = %.2f tok/s%s\n",
        n_decode, dt_s, tok_per_s,
        no_trace ? "" : (std::string(" (trace -> ") + out_path + ", "
                          + std::to_string(st.n_rec) + " records)").c_str());
    fprintf(stderr,
        "config: ngl=%d ncmoe=%d moecache=%d mmap=%d trace=%d oracle=%s\n",
        ngl, n_cpu_moe, moe_cache_size, !no_mmap, !no_trace,
        oracle_path.empty() ? "(none)" : oracle_path.c_str());

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
