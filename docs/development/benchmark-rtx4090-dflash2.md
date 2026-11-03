# RTX 4090 DFlash2 performance

## Current cache-enabled profile

The fork now combines deferred recurrent-state commit with reuse of retired prompt-checkpoint storage. Both preserve prompt caching. The target, original Inco drafter, F16 KV, model-card sampling, thinking low, C1 and seven probabilistic proposals remain fixed as described below.

A direct three-way comparison uses previous published fork `82269c67feeee674ca4ff6e20497de5c483756c9`, recurrent replay `31308a853598aca0a582a62c52d38e7b0bfb3a17`, and replay plus two recycled checkpoint buffers:

| Dataset | Build | Prefill tok/s | Decode tok/s |
| --- | --- | ---: | ---: |
| HumanEval | Previous fork 82269 | 995.38 | 210.67 |
| HumanEval | Recurrent replay 31308 | 1001.90 | 216.43 |
| HumanEval | Replay + checkpoint reuse | 1220.85 | 215.75 |
| GSM8K | Previous fork 82269 | 499.17 | 212.32 |
| GSM8K | Recurrent replay 31308 | 507.23 | 217.84 |
| GSM8K | Replay + checkpoint reuse | 654.57 | 218.90 |

Relative to 82269, the combined profile improves prefill 22.65%/31.13% and decode 2.41%/3.10% on HumanEval/GSM8K. Checkpoint reuse alone improves prefill 21.85%/29.05%; its decode changes -0.32%/+0.48% do not establish a separate decode benefit. Recurrent allocation remains 213.28 MiB instead of 1197.00 MiB, as measured for the replay change.

This uses the unchanged `dflash==0.1.0` OpenAI benchmark: 16 prompts per dataset, 256-token output cap, seeds 42/43, and GPU target sampling. Builds run previous/replay/reuse at seed 42 and in reverse order at seed 43. All 192 scored requests complete in 289.86 seconds (4m50s), including server startup and excluded client warmups. All 64 three-way request groups have identical processed-prompt, generated, accepted and drafted token counts. Prefill pools processed tokens over prompt time; cached tokens are not credited as fresh prefill. Decode pools generated tokens minus the first over generation time, including drafting and verification. Two seeds are an engineering screen, not a confidence interval or an accuracy evaluation.

A diagnostic separates checkpoint sizing, vector resize and state export. Before recycling, each 156894368-byte target checkpoint spends 16.6-41.6 ms allocating and zero-initializing its byte vector, before roughly 17 ms exporting the state. The capture overwrites the serialized payload. Each short request creates two checkpoints. The server now retains at most two already-retired checkpoint allocations per slot and moves them into subsequent captures, avoiding repeated allocation/initialization when their sizes fit. It never recycles a live checkpoint or shares its writable storage with the prompt cache. Checkpoint positions, history limits, serialization and selection stay the same.

This can retain roughly 300 MiB of target checkpoint storage per idle slot for this model, plus draft state. First captures, newly retained history and growth beyond a recycled capacity still allocate. The measured prefill gain applies to these short independent questions with caching enabled, where checkpoints retire between questions; it is not a measured claim for every growing coding trajectory or long-context prefill.

API validation uses two live checkpoints and a 4096 MiB host prompt cache to exercise eviction and ownership. Cached append, tail edits, CPU/GPU sampling switches, probability fallback, streaming cancellation recovery and saved-slot replacement/restore match the reference output. The earlier replay validation below also covers the numerical recurrence and rollback bounds. No accuracy score or CUDA result beyond RTX 4090 is claimed.


## Earlier cache-enabled recurrent replay screen

The current fork adapts the deferred gated-delta-net commit from [Avalon commit 657b456a](https://github.com/freriadoc/Avalon/commit/657b456a7648cf9152616eafa629b5c7cd0f9e17) to dense Qwen35 on CUDA. Instead of keeping a full recurrent matrix for each speculative token, it retains one committed matrix and two compact input windows. The next pass replays the accepted prefix before processing its new window. Convolution history retains its native snapshots. Prompt caching and checkpoint restoration remain enabled.

Direct reference: previous published fork `82269c67feeee674ca4ff6e20497de5c483756c9`. Both variants use the exact target/drafter, F16 KV, model-card sampling, thinking low, GPU target sampling, C1, and seven probabilistic proposals described below. This round enables `--cache-prompt` for both builds.

| Dataset | Reference prefill tok/s | Replay prefill tok/s | Reference decode tok/s | Replay decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| HumanEval | 1025.59 | 1018.06 | 214.70 | 219.55 |
| GSM8K | 513.35 | 527.30 | 215.11 | 222.84 |

Decode improves 2.26%/3.59% in this screen. Prefill changes -0.73%/+2.72%; these runs do not establish a consistent prefill gain. Recurrent allocation falls from 1197.00 MiB to 213.28 MiB, saving 983.72 MiB, including the 24.28 MiB input stash. This is the recurrent buffer allocation, not total VRAM use.

The existing unmodified `dflash==0.1.0` OpenAI client uses 16 HumanEval and 16 GSM8K prompts, a 256-token output cap, and seeds 42/43. Reference/replay run in that order at seed 42 and reversed order at seed 43. All 128 scored requests complete in 197.32 seconds (3m17s), including server startup and excluded client warmups. The sample size is a quick engineering screen, not a full dataset evaluation or a confidence interval. Every one of 64 paired requests has identical processed-prompt, generated, accepted and drafted token counts.

Metrics retain their existing definitions: prefill pools server **processed** prompt tokens over prompt time; decode pools generated tokens minus the first over generation time, including drafting and verification. Independent benchmark questions can share prefixes. Reused tokens are not credited as newly processed prefill tokens. This screen does not measure whole coding-session completion time.

Validation includes all 84 CPU-versus-CUDA gated-delta-net cases (41 exercise replay), including pending-prefix replay before the existing MMA prefill; real-model accepted-prefix comparison against native snapshots for all eight prefix lengths; recurrent checkpoint restoration into fresh/dirty contexts; and rollback bounds. Cached API checks retain identical cross-build output for sampler replacement, CPU/GPU sampling switches, probability fallback, append, edited-tail restoration and streaming cancellation recovery. The 8K check reuses 8106 tokens for continuation and 7594 after a tail edit. These are numerical/API checks, not an accuracy evaluation.

Replay is enabled automatically only for all-CUDA dense Qwen35 contexts with one sequence and F32 recurrent state. Other backends and multi-sequence contexts retain snapshots. `LLAMA_GDN_REPLAY=0` selects the snapshot path for diagnostics. Replay checkpoints carry a distinct layout tag; snapshot-only contexts reject that format. CUDA qualification here is RTX 4090; other CUDA hardware and longer contexts remain unmeasured.

To reproduce the screen, build the reference and fork as below and start each with GPU target sampling and prompt caching enabled. Use the same DFlash commands below with `--num-prompts 16`, then restart at seed 43 in reversed build order. Keep the prefix checkpoints enabled. The cache-disabled multiplier in the historical section below does not describe this configuration.

Run the numerical regressions against the real target:

```sh
cmake --build build --target test-backend-ops test-llama-context -j4
build/bin/test-backend-ops test -b CUDA0 -o GATED_DELTA_NET
build/bin/test-llama-context -m "$TARGET_MODEL" -ngl 99 -fa on -ctk f16 -ctv f16 \
  --test '^(verify_prefix|rollback|rollback_bound)$' --seed 42
```

## Historical cache-disabled comparison

This fork avoids prompt-reuse checkpoint work for requests that disable prompt caching, retains the execution arena across backend-sampler changes when it fits, and avoids sampler backups when native speculative rollback makes them unnecessary. These are server/context changes; this round adds no CUDA kernels.

### Measured comparison

Reference: published fork `f8f699b5cdd7837e878a10a5b3a028604584733e`. Hardware: RTX 4090, 24 GB, 450 W limit, driver 610.88, WSL Ubuntu. Both builds use CUDA 12.6.85, Release, SM89. Target and original Inco drafter, model-card sampling, thinking low, C1 and seven probabilistic proposals are fixed below.

| Dataset | Build / target sampling | Server prefill tok/s | Native decode tok/s | Client output tok/s |
| --- | --- | ---: | ---: | ---: |
| HumanEval | Reference / CPU | 1019.93 | 201.13 | 172.11 |
| HumanEval | Fork / CPU | 2692.34 | 202.45 | 190.39 |
| HumanEval | Fork / GPU | 2698.66 | 204.31 | 192.05 |
| GSM8K | Reference / CPU | 628.26 | 209.52 | 179.99 |
| GSM8K | Fork / CPU | 2162.60 | 211.86 | 202.44 |
| GSM8K | Fork / GPU | 2164.35 | 213.61 | 204.02 |

The GPU variant improves server prefill 2.65x/3.45x, native decode 1.58%/1.95%, and client throughput 11.59%/13.35% for HumanEval/GSM8K. The prefill gain comes from avoiding state serialization and prompt-tail passes scheduled solely to create reuse checkpoints. Cache-enabled requests retain checkpoint creation.

The client is unmodified `dflash benchmark openai` from `dflash==0.1.0`. Each dataset uses 48 prompts in the client's fixed shuffled order, a 256-token output cap, and seeds 42/43. Three variants run in order at seed 42 and reversed order at seed 43. All 576 scored requests completed in 798.58 seconds (13 minutes 19 seconds), including server startup and excluded client warmups. These are bounded dataset subsets, not an accuracy evaluation.

Client output throughput divides generated tokens by the client's batch elapsed time, including prefill, generation and HTTP. The table averages the two client scores. Native decode pools `(output_tokens - 1)` over server generation time, including drafting and verification. Server prefill pools processed prompt tokens over server prompt time, including checkpoint work; the first token belongs to prompt time. Prompt caching is disabled throughout this comparison.

CPU/GPU variants have matching generated, accepted and drafted counts for every one of 192 paired scored requests. Removing checkpoint-only prompt splits changes prefill chunk boundaries, so generated text and acceptance can differ from the reference. Two seeds do not establish a confidence interval.

### Longer prompt check

A supplementary recorded 32,612-token prompt, 256-token output cap and two seeds gives:

| Build / target sampling | Server prefill tok/s | Native decode tok/s |
| --- | ---: | ---: |
| Reference / CPU | 2905.35 | 140.24 |
| Fork / CPU | 2952.80 | 146.76 |
| Fork / GPU | 2940.86 | 146.45 |

This is one prompt at two seeds, not a context-length sweep. The short-prompt multiplier must not be generalized to 32K or to bulk `llama-bench` throughput. This change removes request-processing work outside the main model computation.

### Implementation and validation

- Prompt-reuse checkpoints now require `cache_prompt=true`. Speculative correctness checkpoints use separate storage and retain their existing behavior. A cache-disabled diagnostic request under the previous policy serialized 159,970,116 bytes of state in 63.18 ms, in addition to the checkpoint-only prompt splits.
- Sampler binding changes invalidate graph reuse. The existing arena is retained if its graph-node capacity fits; a larger requirement triggers full reservation. Retired sampler bindings are erased before inspecting the new capacity.
- A sampler backup is made only when verification may restore a speculative checkpoint. The bounded recurrent-rollback path does not use that backup.

Four existing CUDA backend-sampler tests pass: sampler replacement, multi-output RNG transactions, sampling chains, and CPU suffix fallback. API checks cover changing candidate width, CPU/GPU switching, pre-sampling probabilities, streaming cancellation and restoring settings. Cache-policy checks verify zero cached tokens and no reuse checkpoints for cache-disabled requests, successful prompt reuse for cache-enabled requests, and response parity with the reference configured with `--ctx-checkpoints 0`. A test-only zero-snapshot context also completes checkpoint-based speculative rollback; that override is absent from measured builds.

## Reproduce

The command below enables prompt caching for the current profile. Historical cache-disabled measurements use `--no-cache-prompt` instead.

Build the desired revision:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_BUILD_TESTS=ON
cmake --build build --target llama-server llama-bench test-backend-ops test-backend-sampler -j4
```

Use these GGUFs, stored outside the checkout:

- Target: `unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502`, file `Qwen3.8-27B-UD-Q3_K_XL.gguf`, SHA256 `8c2a45ff85e7674ca185ec8eb6cdeab0e617ed9d8018caed0b64380eb2a67a5e`.
- Drafter: original `incoai/Qwen3.8-27B-DFlash2@015e795645c74b1a0eeef3b570031fb62e769bc5`, converted to Q4_K_M GGUF, SHA256 `8cb82324c55b62c8a522e483204fa7521ec6d4a4b72a5e0441efadef4583edaa`.

Set `TARGET_MODEL` and `DRAFT_MODEL` to their cache paths. The following command is the CPU target-sampling variant. Add `--backend-sampling` for the GPU variant above; sampling parameters and rejection policy stay the same. This flag controls target sampling, separately from the existing draft flag. Then start the server:

```sh
build/bin/llama-server -m "$TARGET_MODEL" -md "$DRAFT_MODEL" \
  --alias Qwen3.8-27B --host 127.0.0.1 --port 18406 \
  -np 1 -c 40960 -ngl 99 -ngld 99 --kv-unified --fit off \
  -ctk f16 -ctv f16 -fa on -b 2048 -ub 512 \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --spec-draft-sampling probabilistic --spec-draft-backend-sampling \
  --temp 1 --top-p .95 --top-k 20 --min-p 0 \
  --presence-penalty 0 --repeat-penalty 1 \
  --reasoning on --reasoning-effort low --seed 42 \
  --cache-prompt --cache-ram 0 --no-cache-idle-slots \
  --metrics --log-timestamps
```

Install `dflash==0.1.0` in a separate environment. Run the unchanged client:

```sh
for dataset in humaneval gsm8k; do
  dflash benchmark openai --base-url http://127.0.0.1:18406 \
    --model Qwen3.8-27B --dataset "$dataset" --num-prompts 48 \
    --max-new-tokens 256 --concurrency 1 --reasoning low \
    --temperature 1 --top-p .95 --top-k 20
done
```

Restart with `--seed 43` for the second seed. Keep separate binaries and logs for the baseline and fork; run only one GPU job at a time. For a quick screen, use `--num-prompts 16` instead of 48. On the measured machine this takes about a minute per variant including server startup.

For a separate target-only prefill component measurement, use unmodified `llama-bench`. This uses synthetic token IDs and does not load the DFlash2 drafter or measure its feature-capture overhead:

```sh
build/bin/llama-bench -m "$TARGET_MODEL" -p 512,8192,32768 -n 0 \
  -r 2 -b 2048 -ub 512 -ctk f16 -ctv f16 -fa on -ngl 99 -o json
```

The request-level prefill gain above is separate from this target-only component measurement.

The following 32 CUDA arithmetic cases passed against the CPU reference, including dispatch boundaries and odd output-row counts:

```sh
build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT \
  -p 'type_a=(iq3_s|iq4_xs),type_b=f32,m=33,|type_a=q8_0,type_b=f32,m=(47|48|64|65),n=(1|2|4|8|9),k=5120,'
```


For the context/sampler changes:

```sh
for test in test_set_sampler test_multi_output_dist_transaction test_multi_output_sampling_chain test_multi_output_cpu_suffix; do
  build/bin/test-backend-sampler --model "$TARGET_MODEL" --device gpu --test "$test"
done
```

## Earlier CUDA changes

The reference fork `f8f699b5c` already includes Ada small-matrix tuning (`feaf1d143`), recapture of warmed CUDA graphs (`ea258a72c`), and two-tap DFlash2 convolution fusion (`f8f699b5c`). Weight quantization and sampling policy are unchanged; FP32 reduction order can change. The original composed convolution remains the CPU reference and fallback, with 20 CUDA arithmetic cases:

```sh
build/bin/test-backend-ops test -b CUDA0 -o DYNAMIC_CONV_1D
```

An earlier separate comparison of that reference against unmodified upstream `259d91709d60945911bedb9bcfb2957c6e153ea7` measured native decode 180.68->197.82 tok/s on HumanEval and 188.79->205.65 on GSM8K. Those rates belong to that earlier run; the architecture comparison above directly remeasures `f8f699b5c` as its baseline.
