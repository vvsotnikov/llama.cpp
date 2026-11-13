# Experimental CUDA NVFP4 feed-forward pipeline

This optional path combines a dense gate/up projection, SwiGLU, activation quantization and the down projection. It uses a prepared CUTLASS block-scaled gate/up weight layout and the existing native NVFP4 down-projection consumer. It is disabled by default.

The matrix epilogue applies the existing activation scale followed by the GGUF weight scale in F32 and stores F32 projection results. SwiGLU and native FP4 activation packing share a kernel. When a row fits in shared memory, the kernel retains its F32 SwiGLU values there while finding the row scale, then packs them without reading the projections again. The native F32 arithmetic, row scale and five-candidate FP4 subblock quantizer are retained. A two-pass implementation handles wider rows.

## Build and enable

Build with CUDA and the existing optional CUTLASS dependency:

```sh
cmake -S . -B build -DGGML_CUDA=ON -DGGML_CUDA_CUTLASS_PATH=/path/to/cutlass
cmake --build build --config Release -j --target llama-server test-backend-ops
```

Enable with `GGML_CUDA_NVFP4_MLP=1`. The path uses the Blackwell native FP4 instruction family; other devices keep the existing implementation. Small batches, unsupported tensor shapes, explicit activation-precision overrides and graphs with other consumers of a skipped intermediate also keep the existing implementation.

The current shape coverage requires two-dimensional contiguous F32 inputs, NVFP4 gate/up/down weights, token count 512-8192 divisible by128, input width512-32768 divisible by512, and intermediate width512-65536 divisible by512. Gate and up must share a CUDA weight buffer. Weight scales can be scalar or per output channel. Only SwiGLU without swapping or clipping is selected. These restrictions bound the tested prototype; they do not alter normal dispatch.

## Storage and lifetime

The prepared layout uses `2 * intermediate_width * input_width * (1/2 + 1/16)` extra bytes for gate/up values and subblock scales. The original weights remain available to decode and fallback paths. Preparation is performed outside CUDA graph capture and reused across requests. It can run during an existing small-batch model warmup, even though the execution path is selected only for large batches. Preparation completes before graph capture, so captured streams do not wait on uncaptured producer work. Prepared allocations belong to the weight buffer, are invalidated on writes and are released when the buffer is freed. If allocation fails or leaves less than2GiB free, the existing path is used. Startup preparation and extra resident storage must be included when assessing deployment costs.

## Validation

```sh
GGML_CUDA_NVFP4_MLP=0 build/bin/test-backend-ops test -b CUDA0 -o NVFP4_MLP
GGML_CUDA_NVFP4_MLP=1 build/bin/test-backend-ops test -b CUDA0 -o NVFP4_MLP
```

These cases cover scalar/channel scales, supported batches, boundary fallbacks, zero input and a retained SwiGLU intermediate. Also compare real-model logits and task quality, then measure the unchanged serving workload with the flag off/on. Operator checks and isolated matrix timings do not establish a serving speedup.
