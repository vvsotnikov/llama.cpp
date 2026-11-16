# Experimental CUDA prefill, decode and DFlash2 pipeline

This branch combines the fork's activation preparation, projection and attention
prefill paths with its IQ4 decode path, asynchronous DFlash2 injection, and
speculative tree execution. The runtime options below are required to enable
the experimental paths; checking out the branch alone does not enable them.

These options belong to this fork. They apply to dense Qwen3.5-family models
with mixed NVFP4 and per-channel FP8 weights, including the Qwen3.8-27B NVFP4
checkpoint converted by this fork. They do not describe the UD-Q3_K_XL path.

The configuration below uses native CUDA code, CUDA 13.4, and CUTLASS 4.5.
Set `CUDA_ARCH` for the device you build for. Build output and
model files should live outside the source checkout.

```sh
cmake -S . -B "$LLAMA_BUILD_DIR" \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DGGML_CUDA_CUTLASS_PATH="$CUTLASS_ROOT"
cmake --build "$LLAMA_BUILD_DIR" --target llama-server test-backend-ops -j 4
```

## Target checkpoint conversion

Set `TARGET_SNAPSHOT` to the cached safetensors snapshot of
`unsloth/Qwen3.8-27B-NVFP4`. The converter retains the checkpoint's FP8
projections and repacks its NVFP4 weights and scales. Use `auto` to preserve
this mixed layout:

```sh
export TARGET_GGUF="$HOME/.cache/huggingface/local-builds/unsloth--Qwen3.8-27B-NVFP4/Qwen3.8-27B-NVFP4-FP8.gguf"
mkdir -p "$(dirname "$TARGET_GGUF")"
python convert_hf_to_gguf.py "$TARGET_SNAPSHOT" \
    --outtype auto --outfile "$TARGET_GGUF"
```

Keep `DRAFTER_GGUF` separate: the serving example below expects the original
`incoai/Qwen3.8-27B-DFlash2` checkpoint converted to Q4_K_M.

## Precision and execution options

The options default to off. The reference path remains available by leaving
these options unset. The following group changes activation quantization:

```sh
export GGML_CUDA_FP8_TOKEN_SCALE=1
export GGML_CUDA_NVFP4_CALIBRATED=1
export GGML_CUDA_NVFP4_BF16_MLP=1
```

FP8 uses one activation scale per token instead of one scale for the entire
matrix. NVFP4 uses the checkpoint's input calibration scales instead of a
runtime scale search. Gate/up matrix outputs use BF16 before SwiGLU and FP4
packing. These options do not claim bit identity with the prior activation
policy. Recurrent state and the high/residual BF16 matrix products used to
approximate FP32 arithmetic retain their original precision.

The following group removes intermediate work or changes CUDA scheduling:

```sh
export GGML_CUDA_FP8_PIPELINE=1
export GGML_CUDA_NVFP4_MMQ_FUSION=1
export GGML_CUDA_NVFP4_MLP=1
export GGML_CUDA_SCALED_SWIGLU=1
export GGML_CUDA_SSM_CONV_SPLIT=1
export GGML_CUDA_SSM_CONV_QK_NORM=1
export GGML_CUDA_GDN_SCRATCH_REUSE=1
export GGML_CUDA_EARLY_QKVZ=1
export GGML_CUDA_BF16_SHARED_INPUT=1
export GGML_CUDA_FP8_BF16_NORM=1
export GGML_CUDA_FP8_QKV_PREPARE=1
export GGML_CUDA_FP8_GATED_NORM=1
export GGML_CUDA_NVFP4_MLP_IRREGULAR=1
export GGML_CUDA_NVFP4_MLP_NORM=1
export GGML_CUDA_NVFP4_MLP_DOWN=1
export GGML_CUDA_NVFP4_DIRECT_PACK=1
export GGML_CUDA_NORM_INPUT_REUSE=1
export GGML_CUDA_FA_PREFILL_GQA=2
export GGML_CUDA_FA_WIDE_PREFILL=1
```

Split convolution reads the current projection and retained convolution history
separately, avoiding concatenation of the complete prompt activation matrix.
Adjacent projections share input conversion. Normalization, residual addition,
and activation packing are fused where the graph and storage permit it.
Prepared matrix layouts are retained for later calls. Direct packing produces
the selected consumer's values and scale layout without an intermediate MMQ
layout conversion.

`GGML_CUDA_SSM_CONV_QK_NORM=1` combines four-tap split convolution, SiLU,
and query/key normalization for F32 inputs with 128-element heads. Four warps
process adjacent token positions independently. It applies to batches of at
least 64 tokens with a supported layout and no retained intermediate that the
fusion would discard. Unsupported layouts and overlapping buffers retain the
existing path.

`GGML_CUDA_GDN_SCRATCH_REUSE=1` uses 16 warps for the C16 whole-head recurrent
prefill kernel. Packed recurrent state is no longer needed after the input
projections, so its shared storage can hold the later Gram matrices and solved
updates. Query storage becomes delta storage only after Gram construction
finishes. This reduces shared-memory use and per-thread register pressure
while retaining F32 state and the existing high/residual arithmetic. Existing
shape, architecture, and resource guards remain in force; other schedules keep
the original kernel.

For selection diagnostics, set `GGML_CUDA_SSM_CONV_QK_NORM_TRACE=1` or
`GGML_CUDA_GDN_SCRATCH_REUSE_TRACE=1`. Disable tracing for timing runs.

The native NVFP4 MLP consumer uses a K256 tile for its F32 down projection.
Gate/up projections retain K128 tiles, including their BF16 output path.

`GGML_CUDA_FP8_BF16_NORM=1` lets a closed group of two FP8 and two BF16
projections share normalized input preparation. It requires per-token FP8
scaling and `GGML_CUDA_BF16_SHARED_INPUT=1`. The normalization kernel writes
the BF16 input alongside packed FP8 values; the F32 intermediate is omitted
only when no other graph consumer needs it. Explicit in-place normalization
retains the existing path.

`GGML_CUDA_FP8_QKV_PREPARE=1` groups the three FP8 attention projections
before query/key normalization for batches of at least 64 tokens. They reuse
one packed activation buffer. Dead normalization and SwiGLU outputs can also
be omitted when the graph and storage checks permit it. Other consumers and
retained outputs keep the original materialization.

The row-normalization fusion permits exact in-place row writes after the entire
row has been read. Matrix outputs may reuse consumed input storage after the
packing kernel completes. Partial aliases of retained row outputs are rejected.

`GGML_CUDA_FA_PREFILL_GQA=2` selects a measured attention grouping; it is a tuning
choice, not a claim that this grouping is best on every CUDA device.
`GGML_CUDA_FA_WIDE_PREFILL=1` doubles the query tile for 256-dimensional
attention with this grouping on NVIDIA Ampere or newer devices. It applies to
complete 64-query tiles; short batches and partial tiles retain the original
dispatch. The qualified configuration uses F16 KV and preserves the original
32-token softmax update interval.

## Serving with DFlash2

Use the target GGUF and the original Inco DFlash2 checkpoint converted to Q4_K_M.
The following is the C1 serving configuration used during development. Prompt
caching is enabled. Adjust the paths, context capacity, and port for the machine.

```sh
export LLAMA_DFLASH_TREE=1
export GGML_CUDA_TREE_SLOTS=1
export GGML_CUDA_ASYNC_INPUTS=1
export LLAMA_DFLASH_ASYNC_INJECT=1
export GGML_CUDA_GLU_MMQ_FUSION=1
export GGML_CUDA_IQ4_FRAGMENT=1

"$LLAMA_BUILD_DIR/bin/llama-server" \
    -m "$TARGET_GGUF" -md "$DRAFTER_GGUF" \
    -ngl 99 -ngld 99 -np 1 -c 40960 -b 2048 -ub 2048 \
    --fit off --kv-unified -ctk f16 -ctv f16 -fa on \
    --spec-type draft-dflash --spec-draft-n-max 15 \
    --spec-draft-sampling probabilistic --spec-draft-backend-sampling \
    --backend-sampling --temp 1 --top-p .95 --top-k 20 --min-p 0 \
    --presence-penalty 0 --repeat-penalty 1 \
    --reasoning on --reasoning-effort low \
    --cache-prompt --cache-ram 0 --no-cache-idle-slots \
    --host 127.0.0.1 --port 8080
```

The 15-node tree includes alternative continuations; it is not a 15-token
linear draft. A comparison with a seven-proposal chain must state that policy
difference. The device-checkpoint path uses the loaded model's CUDA device,
including integrated CUDA GPUs selected automatically.

## Optional BF16 recurrent prefill

This additional path is experimental and is disabled in the serving recipe
above. It rounds convolution outputs and normalized Q/K to BF16, while retaining
F32 gates, recurrent state and graph output. It changes the activation policy;
it does not claim bit identity with F32 recurrent prefill or a measured speedup
over the configuration above. Model quality and downstream timing must be
evaluated separately before selecting it.

The Linux/WSL provider uses FlashInfer's CuTe DSL GDN implementation with the
Q-head lookup adapted to ggml's modulo head layout. It is compiled ahead of time
for the selected SM120 or SM121 GPU. The builder uses an installed FlashInfer
package and the public cuDNN Frontend runtime shim; it does not download source
or install packages. Tested builder dependencies are PyTorch 2.14.0,
FlashInfer 0.7.2.post1, CUTLASS DSL 4.8.0 and CUDA 13.4. These are separate from
the C++ CUTLASS dependency used by the matrix kernels.

Set `CUDNN_FRONTEND_ROOT` to a checkout of
[NVIDIA/cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend), containing
`aot/sm120/src/runtime_shim.cpp`. Keep that checkout and `GDN_BUILD_DIR` outside
this source tree. Run the builder in the Python environment containing the
dependencies:

```sh
python scripts/build_cuda_gdn_bf16.py \
    --cuda-root "$CUDA_ROOT" --device 0 \
    --runtime-shim "$CUDNN_FRONTEND_ROOT/aot/sm120/src/runtime_shim.cpp" \
    --output-dir "$GDN_BUILD_DIR"

cmake -S . -B "$LLAMA_BUILD_DIR" \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DGGML_CUDA_CUTLASS_PATH="$CUTLASS_ROOT" \
    -DGGML_CUDA_GDN_BF16_LIBRARY="$GDN_BUILD_DIR/libggml-gdn-bf16.so"
cmake --build "$LLAMA_BUILD_DIR" --target llama-server test-backend-ops -j 4

export LD_LIBRARY_PATH="$GDN_BUILD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LLAMA_CUDA_BF16_GDN=1
export GGML_CUDA_SSM_BF16_QK=1
export GGML_CUDA_SSM_DEFER_HISTORY=1
```

The builder writes the adapted FlashInfer source, exported object, provider
source and a dependency/source-hash manifest beside the library. Preserve
FlashInfer's license header when redistributing those generated files. Rebuild
for the GPU on which the library will run; it is not a universal CUDA binary.
The normal build does not need this library. Without it, BF16 recurrent
operations use the CPU reference, so leave `LLAMA_CUDA_BF16_GDN` unset unless
the provider is configured and qualified on the intended GPU.

Selection is limited to single-sequence, non-tree prefill batches of 64 to
4096 tokens with 128-element heads and supported layouts. Short speculative
verification batches retain the F32 path. The convolution fusion preserves
the explicit BF16 rounding points and checks retained outputs and buffer
overlap. Deferring the convolution-history write keeps the current projection
live until Q/K/V have been written, allowing the fusion without removing its
alias checks. Set `GGML_CUDA_SSM_BF16_QK_TRACE=1` to check selection, then unset
it before timing.

## Validation

Run the relevant backend tests against the intended CUDA build, including
`SSM_CONV_SPLIT`, `SSM_CONV_SPLIT_QKV`, `SSM_CONV_QK_NORM`, `GATED_DELTA_NET`, `BF16_SHARED_INPUT`, `FP8_PIPELINE`,
`FP8_GATED_NORM`, `FP8_NORM_REUSE`, `FP8_BF16_NORM`, `FP8_QKV_PREPARE`, `NVFP4_MLP`, `NVFP4_MLP_NORM`, and
`NVFP4_NORM_REUSE`. Require nonzero test counts. For calibrated activation
quantization, compare packed values and scales against an independent native
quantizer as well as evaluating model likelihood and task quality. Some
`NVFP4_MLP_CALIBRATED` cases exceed the original F32 activation error tolerance;
do not report those as passes or relax that tolerance to conceal the difference. A comparison
to unquantized F32 activations alone does not establish native quantizer parity.

For performance, freeze the input token IDs, checkpoint, sampler, reasoning
mode, KV formats, output cap, concurrency, and cache policy. Report freshly
processed prompt tokens per prefill second and generated tokens per decode
second separately. Include drafting and verification in decode time. Report
actual cache hits and output lengths. Keep results from profilers separate from
uninstrumented request timing, and identify the exact binary and options.

For the optional provider, compare `GATED_DELTA_NET` against the CPU reference
and run `SSM_CONV_BF16_QK` with the fusion both enabled and disabled. The latter
includes short batches, retained intermediates and unsupported input layouts.
Also test speculative rejection followed by replay and cached continuation;
operator tests alone do not qualify recurrent history handling in the server.
