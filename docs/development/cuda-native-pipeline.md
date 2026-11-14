# Experimental CUDA prefill pipeline

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
export GGML_CUDA_EARLY_QKVZ=1
export GGML_CUDA_BF16_SHARED_INPUT=1
export GGML_CUDA_FP8_GATED_NORM=1
export GGML_CUDA_NVFP4_MLP_IRREGULAR=1
export GGML_CUDA_NVFP4_MLP_NORM=1
export GGML_CUDA_NVFP4_MLP_DOWN=1
export GGML_CUDA_NVFP4_DIRECT_PACK=1
export GGML_CUDA_NORM_INPUT_REUSE=1
export GGML_CUDA_FA_PREFILL_GQA=2
```

Split convolution reads the current projection and retained convolution history
separately, avoiding concatenation of the complete prompt activation matrix.
Adjacent projections share input conversion. Normalization, residual addition,
and activation packing are fused where the graph and storage permit it.
Prepared matrix layouts are retained for later calls. Direct packing produces
the selected consumer's values and scale layout without an intermediate MMQ
layout conversion.

The row-normalization fusion permits exact in-place row writes after the entire
row has been read. Matrix outputs may reuse consumed input storage after the
packing kernel completes. Partial aliases of retained row outputs are rejected.

`GGML_CUDA_FA_PREFILL_GQA=2` selects a measured attention grouping; it is a tuning
choice, not a claim that this grouping is best on every CUDA device.

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

## Validation

Run the relevant backend tests against the intended CUDA build, including
`SSM_CONV_SPLIT`, `SSM_CONV_SPLIT_QKV`, `BF16_SHARED_INPUT`, `FP8_PIPELINE`,
`FP8_GATED_NORM`, `FP8_NORM_REUSE`, `NVFP4_MLP`, `NVFP4_MLP_NORM`, and
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
