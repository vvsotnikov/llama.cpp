# Prepared IQ3_S weights for CUDA prefill

`GGML_CUDA_PREFILL_IQ3=1` enables an experimental prepared-weight path for
large, dense IQ3_S matrix multiplications on NVIDIA GPUs with integer MMA.
It is disabled by default.

The cached representation stores the original signed odd integer codes as
biased four-bit values, together with the float scales used by the native
MMQ loader. It moves codebook lookups and sign expansion out of repeated
matrix execution. Activation quantization, MMA, tiling and reduction retain
the existing MMQ arithmetic. The original GGUF weights remain available to
the normal small-batch decode path.

Preparation is lazy and adds a one-time cost to the first eligible graph.
The cache uses 160 bytes per 256 weights, in addition to the original GGUF
storage. It belongs to the weight buffer, is invalidated by weight updates,
and is freed with the buffer. Preparation occurs before CUDA graph capture;
readiness events order consumers on another stream. If storage cannot be
allocated with the retained headroom, execution uses the native path.

The current path requires contiguous, immutable weight matrices, F32 inputs
and outputs, K divisible by 256, and 256–8192 input rows. Views, trainable
weights, unsupported dimensions and smaller batches use the normal path.
It does not replace the existing prepared layouts for speculative verification.

Build with the normal CUDA configuration. Targeted backend checks:

```sh
GGML_CUDA_PREFILL_IQ3=1 build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT_IQ3_PREFILL -j 1
```

Measure full requests as well as individual operators: preparation can offset
steady-state savings on the first request. Report the extra storage and keep
the prompt-cache, sampling and speculative-decoding configuration fixed.
