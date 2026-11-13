# CUDA FP8 producer pipeline

This optional path combines FP8 activation preparation with adjacent producers
and applies channel scales in the matrix multiplication epilogue. It uses
CUTLASS headers at build time and does not require a PyTorch or FlashInfer
runtime. The current consumer targets the Blackwell SM120 instruction family.
Other devices and unsupported graphs retain normal dispatch.

Provide a CUTLASS root containing `include/cutlass/cutlass.h`:

```sh
cmake -S . -B build -DGGML_CUDA=ON -DGGML_CUDA_CUTLASS_PATH=/path/to/cutlass
cmake --build build --config Release --target llama-server test-backend-ops -j
```

CUTLASS 4.5 and CUDA 13.4 were used for qualification. Select native CUDA
architectures appropriate to the deployment, including Blackwell instructions
when using this consumer. An ordinary build without the CUTLASS path preserves
the existing implementation.

Enable the path for a controlled comparison:

```sh
GGML_CUDA_FP8_PIPELINE=1 build/bin/llama-server ...
```

The environment variable is read once per process. It is disabled by default;
`GGML_CUDA_FP8_PIPELINE=0` explicitly selects the existing execution paths.
Keep all other model, cache, sampling and speculative decoding settings fixed.

Eligible operations have contiguous F8_E4M3 weight matrices, F32 inputs and
outputs, one F32 scale per output channel, aligned dimensions and at least 64
input tokens. One or two adjacent projections can share their packed input.
The packing scope belongs to that fused execution; it is not a persistent
activation cache.

Weighted RMS normalization, SwiGLU, and residual additions with an optional
input scale can contribute directly to activation-scale calculation. Normalized
inputs and residuals needed by other graph consumers are still written. Graph
consumer and storage checks reject unsafe fusions, and concurrent stream
regions retain the existing paths.

The implementation preserves whole-tensor activation scaling, F32 intermediate
precision, native vector RMS reduction geometry, and the multiplication order
of activation and weight scales. Small changes in rounding before later low-bit
operations can otherwise produce much larger full-model differences.

Run the focused tests with both paths:

```sh
GGML_CUDA_FP8_PIPELINE=0 build/bin/test-backend-ops test -b CUDA0 -o FP8_PIPELINE
GGML_CUDA_FP8_PIPELINE=1 build/bin/test-backend-ops test -b CUDA0 -o FP8_PIPELINE
```

Compare full-model numerical behavior and complete requests as well as isolated
operators. Report cold and cached prefill separately; include drafting and
verification in decode timing. Do not infer quality equivalence from a kernel
speed measurement.
