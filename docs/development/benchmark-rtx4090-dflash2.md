# RTX 4090 DFlash2 performance

This fork tunes the small CUDA matrix operations used during DFlash2 verification on Ada. It uses one warp for IQ3_S/IQ4_XS operations with five to eight token columns, and runs small Q8_0 projections as broadcast single-column batches. The latter changes tensor metadata for the kernel launch without copying data. It also recaptures warmed CUDA graphs after property changes and fuses the drafter's two-tap grouped dynamic convolutions. Weight quantization and sampling policy are unchanged. FP32 reduction order can change in the matrix kernels.

## Measured comparison

Baseline: unmodified upstream `259d91709d60945911bedb9bcfb2957c6e153ea7`. Hardware: RTX 4090, 24 GB, 450 W limit, driver 610.88, WSL Ubuntu. Both builds used CUDA 12.6.85, Release, SM89.

| Dataset | Baseline client tok/s | Fork client tok/s | Change | Baseline native decode tok/s | Fork native decode tok/s | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HumanEval | 155.22 | 169.24 | +9.04% | 180.68 | 197.82 | +9.49% |
| GSM8K | 162.65 | 176.77 | +8.68% | 188.79 | 205.65 | +8.93% |

The client is unmodified `dflash benchmark openai` from `dflash==0.1.0`. Each dataset used 48 prompts in the client's fixed shuffled order, a 256-token output cap, C1, and seeds 42 and 43. These counts are a bounded subset, not the complete datasets or an accuracy evaluation. One client warmup per dataset is excluded from scoring. Prompt caching was disabled.

Client throughput is output tokens divided by the client's measured batch elapsed time, including prompt processing, generation and HTTP. The table averages the two client scores. Native decode is the sum of `(output_tokens - 1)` divided by the sum of the server's generation-phase time. It includes drafting and verification but excludes prompt processing. The first token belongs to prompt processing in this server's timing convention.

The six-arm comparison ran upstream, the matrix-tuned recapture build, and that build plus dynamic-convolution fusion at seed 42, then reversed build order at seed 43. All 576 scored requests completed in 885.69 seconds (14 minutes 46 seconds). Convolution fusion preserved accepted and drafted token counts in every seed/dataset pair compared with the recapture build; the matrix changes can alter sampling trajectories. Two seeds do not establish a confidence interval or performance on all workloads.

The matrix tuning alone previously measured +4.94% and +4.10% native decode over upstream. The recapture and convolution sections below give their incremental comparisons. The Q8_0 small projection improved from 5.95 to 4.22 microseconds in isolation; its incremental request-level gain was inconsistent across seeds.

## Reproduce

Build the desired revision:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_BUILD_TESTS=ON
cmake --build build --target llama-server llama-bench test-backend-ops -j4
```

Use these GGUFs, stored outside the checkout:

- Target: `unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502`, file `Qwen3.8-27B-UD-Q3_K_XL.gguf`, SHA256 `8c2a45ff85e7674ca185ec8eb6cdeab0e617ed9d8018caed0b64380eb2a67a5e`.
- Drafter: original `incoai/Qwen3.8-27B-DFlash2@015e795645c74b1a0eeef3b570031fb62e769bc5`, converted to Q4_K_M GGUF, SHA256 `8cb82324c55b62c8a522e483204fa7521ec6d4a4b72a5e0441efadef4583edaa`.

Set `TARGET_MODEL` and `DRAFT_MODEL` to their cache paths, then start the server:

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
  --no-cache-prompt --cache-ram 0 --no-cache-idle-slots \
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

Restart with `--seed 43` for the second seed. Keep separate binaries and logs for the baseline and fork; run only one GPU job at a time. For a quick screen, use `--num-prompts 16` instead of 48. On the measured machine this took about 70 seconds including server startup.

For a separate target-only prefill component measurement, use unmodified `llama-bench`. This uses synthetic token IDs and does not load the DFlash2 drafter or measure its feature-capture overhead:

```sh
build/bin/llama-bench -m "$TARGET_MODEL" -p 512,8192,32768 -n 0 \
  -r 2 -b 2048 -ub 512 -ctk f16 -ctv f16 -fa on -ngl 99 -o json
```

The CUDA changes above target small-token verification. No prefill speedup is claimed from the short request benchmark.

The following 32 CUDA arithmetic cases passed against the CPU reference, including dispatch boundaries and odd output-row counts:

```sh
build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT \
  -p 'type_a=(iq3_s|iq4_xs),type_b=f32,m=33,|type_a=q8_0,type_b=f32,m=(47|48|64|65),n=(1|2|4|8|9),k=5120,'
```

## Recapture warmed CUDA graphs

After initial CUDA-graph warmup, a change of tensor properties now triggers capture directly. Previously it reset warmup and launched operations eagerly until the graph was stable again. Initial warmup is retained.

A separate comparison against `feaf1d143`, using the same client, 48 prompts per dataset, 256-token cap and seeds 42/43, completed in 590.96 seconds:

| Dataset | Base client tok/s | Recapture client tok/s | Base native decode tok/s | Recapture native decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| HumanEval | 160.72 | 166.04 | 188.30 | 193.94 |
| GSM8K | 167.17 | 173.81 | 194.91 | 201.56 |

Accepted and drafted token counts matched in every seed/dataset pair. The native decode gains were +2.99% and +3.41% over `feaf1d143`. Short-prompt native prefill improved +5.19% and +7.43%. A separate 32K prompt check completed at both seeds with unchanged draft counters; no long-prefill gain was established.

## Fuse two-tap grouped dynamic convolution

The drafter's convolution was composed from broadcasts, contiguous copies, padding, multiplies and an add. CUDA now recognizes the closed two-tap graph and calculates its output in one kernel without materializing broadcast weights. The primitive graph remains the CPU reference and fallback. The matcher rejects externally consumed intermediate tensors and producers inside the fusion span. If output storage overlaps an input, the kernel writes to temporary storage before copying the result back. Separate FP32 add/multiply rounding is retained.

Compared with the matrix-tuned recapture build in the six-arm comparison above:

| Dataset | Recapture native decode tok/s | With convolution fusion tok/s | Change |
| --- | ---: | ---: | ---: |
| HumanEval | 194.25 | 197.82 | +1.84% |
| GSM8K | 201.71 | 205.65 | +1.95% |

Twenty composed-graph arithmetic cases pass against the CPU reference. They cover both convolution sides, one/two/eight-token blocks, multiple sequences, odd channel counts, the actual 5120-channel shape, a live intermediate that must prevent fusion, and a computed input producer that must not be skipped:

```sh
build/bin/test-backend-ops test -b CUDA0 -o DYNAMIC_CONV_1D
```

The test utility's whole-graph performance mode now repeats all producers on every iteration. Prior timings that repeated only a final fusion-test operation must not be interpreted as whole-graph latency.
