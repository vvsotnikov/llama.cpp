# RTX Spark DFlash2 serving profile

This records a Windows ARM64 deployment of Qwen3.8-27B with a Q4_K_M DFlash2 draft on RTX Spark N1X. It combines the fork's existing CUDA paths with the measured model and host-thread configuration. No additional engine patch is included.

## Configuration and launch

Measured on November 16, 2026: RTX Spark N1X, SM121, 48 SMs, 64 GiB physical unified memory, 40 GiB Windows dedicated-GPU carveout, NVIDIA driver 621.14, CUDA 13.5.37, native ARM64 Release build with MSVC 19.44.35230, OpenMP disabled, AC power and the Windows Balanced power plan. CUDA-addressable memory and the Windows dedicated-GPU carveout are different quantities.

The measured fork revision is `5ce46333357f6f9bbc869079bc7cb6786645f740`, with no uncommitted engine changes. Build for SM121; an SM120-only build does not qualify the Spark-specific few-row path. The existing [Windows ARM64 CUDA toolchain](../../cmake/arm64-windows-msvc-cuda.cmake) can be used with `CMAKE_CUDA_ARCHITECTURES=121`. Build the `llama-server` target and make the CUDA runtime DLLs available to it.

| GGUF | SHA-256 |
| --- | --- |
| `Qwen3.8-27B-UD-Q3_K_XL.gguf` | `8c2a45ff85e7674ca185ec8eb6cdeab0e617ed9d8018caed0b64380eb2a67a5e` |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` |

From the repository root, in PowerShell:

```powershell
.\scripts\serve-dflash2-spark.ps1 `
    -Server .\build\bin\llama-server.exe `
    -Target C:\models\Qwen3.8-27B-UD-Q3_K_XL.gguf `
    -Draft C:\models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

The script runs the server in the foreground on `127.0.0.1:8080`, with an 8K context, one slot, 512-token batch and ubatch, F16 KV, unified KV, full GPU offload, Flash Attention, and a fixed 15-node DFlash2 tree. Fifteen nodes are a tree budget, not fifteen consecutive proposals. It uses GPU target sampling, `p_min=0`, one target/draft CPU thread, and zero polling. The launch flags do not replace DFlash2's selector with a backend sampler.

The script enables the existing tree-slot, async input/injection, DFlash phase-workspace, GLU activation-preparation, IQ4 fragment-layout and prepared-IQ3 prefill switches. They remain opt-in engine features. Prepared layouts add memory lazily; the short-question memory observation below is not a capacity guarantee for long prompts.

Sampling is temperature 1, top-k 20, top-p .95, min-p .05, repeat penalty 1 and presence penalty 0. Min-p .05 was inherited in the original measurements and is explicit in the launcher; this is not the model-card min-p-zero recipe. Requests used thinking enabled and reasoning effort `xhigh`. Avoid unrelated experimental environment overrides when reproducing the profile. The launcher restores its environment overrides on exit, including any pre-existing AV2 cost-table variable; it does not install or enable AV2.

## Observed deployment comparison

The control is the previous **private deployment**, based on `af6b5243d6871923dfecbe06814bd566d9dc383b`, with few-row MMQ, per-width graph workspaces and shared sampler-output synchronization. Its source-diff SHA-256 was `207cf306807b8661b78ad6897808c78b63c7b7fd0019eb6bffd4e1db024102b0`; that commit alone does not reproduce the private build. It used seven probabilistic proposals with CPU rejection sampling and no adaptive verification. It is not an upstream/master control.

Both deployments used the same target/draft GGUFs. The fork configuration used a 15-node tree and GPU target sampling. These are different deployment packages and produce different continuations; this comparison does not isolate the benefit of a kernel or prove distributional equivalence between all server settings.

| Metric | Previous private deployment | Fork profile |
| --- | ---: | ---: |
| Pooled decode, tok/s | 53.7473 | 60.3350 |
| GSM8K decode, tok/s | 58.6509 | 74.3798 |
| HumanEval decode, tok/s | 48.8743 | 49.1714 |
| Short-prompt prefill, tok/s | 271.4024 | 719.6046 |
| Client output, tok/s | 48.0501 | 56.4042 |
| Accepted draft tokens per target pass | 3.6955 | 4.6993 |
| Mean sampled GPU power, W | 35.56 | 34.28 |
| Maximum sampled GPU temperature, C | 81.25 | 78.22 |
| Mean sampled GPU SM clock, MHz | 1906.1 | 1892.6 |
| Peak sampled process GPU commitment, GiB | 16.04 | 19.80 |

Overall decode improved 12.26% on this subset. The gain is concentrated in GSM8K; HumanEval decode is essentially flat. This is twelve requests per configuration, not a production coding/agent evaluation. The exploratory prompt-cluster bootstrap interval was +2.15% to +28.50%, with only six prompt clusters. GPU power is the driver-reported GPU measurement, not whole-system power or energy per token. GPU commitment is a Windows process counter, not a physical-residency measurement.

## Measurement protocol

One server was resident at a time, in ABBA order: control seed 42, fork seed 42, fork seed 43, control seed 43. Each fresh server received one excluded 256-token warmup request, then the six measured questions with a 384-token output cap. Each measured request disabled prompt reuse, used streaming with usage reporting, and set `return_tokens=true`. Target prefill timings include the server's prompt/state/feature-injection work; they do not establish bulk-prefill or long-context throughput.

The prompt subset was the first four GSM8K test rows and first two HumanEval test rows, in source order:

- [GSM8K rows](https://datasets-server.huggingface.co/rows?dataset=openai%2Fgsm8k&config=main&split=test&offset=0&length=4).
- [HumanEval rows](https://datasets-server.huggingface.co/rows?dataset=openai%2Fopenai_humaneval&config=openai_humaneval&split=test&offset=0&length=2).

The prompt wrappers, expressed as Python string literals, were:

```python
gsm8k = row["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
humaneval = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n" + row["prompt"] + "\n```"
```

The client used these additional request fields:

```json
{
  "model": "spark-qwen",
  "max_tokens": 384,
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "seed": 42,
  "stream": true,
  "stream_options": {"include_usage": true},
  "cache_prompt": false,
  "return_tokens": true,
  "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "xhigh"}
}
```

The server's min-p .05 remained in effect. Change the seed to 43 for the reversed-order pair. Keep sampling settings, prompt formatting, model files and cache policy fixed between arms. Independent instances of the unchanged fork also produced different sampled continuations in an A/A check, so output identity was not used as a performance-retention criterion.

The [24 per-request measurements](rtx-spark-dflash2-requests.csv) allow recomputing the pooled figures. Decode is `1000 * sum(decode_timed_tokens) / sum(decode_ms)`; each request's timed-token count is its output count minus the first token, which belongs to prefill. Drafting and target verification are included in decode time. Prefill pools processed prompt tokens over prompt time, excluding reused tokens. Client output pools all generated tokens over client request wall time. The four CSV runs correspond to the ABBA order above; rows contain their seed and prompt identifier.

## Telemetry

Run the sidecar in another PowerShell session, supplying the actual server PID:

```powershell
.\scripts\log-gpu-telemetry.ps1 `
    -OutputCsv .\results\spark-telemetry.csv `
    -ServerProcessId 1234 -GpuIndex 0 -IntervalSeconds 3 `
    -StopFile .\results\telemetry.stop
```

Create the stop file to stop logging, or use `-DurationSeconds 60`. Choose a fresh output CSV and an absent stop file for each run. The logger samples NVIDIA power, temperature, utilization, clocks and memory, plus Windows process/system memory and AC status. `qpc_s` uses the Windows performance counter so client `time.perf_counter()` request intervals can select active samples. Process GPU counters sum all Windows adapter instances for that PID; `-GpuIndex` selects the NVIDIA device queries. Unsupported queries are retained in the `error` column; the logger does not stop the server or change clocks, power policy, memory allocation, or the driver.

The measurements used three-second telemetry with no recorded query errors. The [43 active-window samples](rtx-spark-dflash2-telemetry.csv) reproduce the table's sampled power, temperature, clock and commitment figures. They are selected using the request CSV's `qpc_start` and `qpc_end`, not server startup or warmup. Two simultaneous prepared-weight servers approached the Windows commit limit in a separate experiment. The deployment comparison above used one resident server and had ample commit headroom.

## Validation and exclusions

The existing tree target test passed 124 full-vocabulary comparisons with F16 KV and another 124 with Q8 KV, including sibling isolation, serial-ancestor sampling, committed-state byte equality, continuation and restore checks. Maximum absolute logit difference was zero in those comparisons. On Windows, two non-exported trivial accessors were linked into the test executable from their exact existing definitions; the test source and production DLLs were unchanged. This does not fix the test target's Windows DLL-linking limitation.

A four-chunk perplexity smoke with the same command reported 6.2090 +/- .50476 for the previous deployment and 6.1615 +/- .49874 for the fork. These short runs do not establish a model-quality improvement. API health, model listing and a completion check also passed after deployment.

No private AV2 controller, sampler output-view port, IQ3 gate/up/GLU fusion port, per-width scheduler port, IQ3_XXS extension, prefetch tuning or capture hook is part of this profile. The additional sampler port did not demonstrate a gain. The native/prepared IQ3 fusion port passed correctness checks but changed single-resident prefill by +1.23%, -1.61% and +0.42% at 512, 2048 and 8192 tokens respectively; it was removed rather than retained as an optimization. The fork's own pre-existing optional paths remain unchanged.
