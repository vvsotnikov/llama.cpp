# Experimental IQ3_S verification layout

`GGML_CUDA_IQ3_FRAGMENT=1` enables a CUDA Ada path for contiguous IQ3_S matrix products with eight input columns. The current dispatch requires at least 2048 input features and 5120 output features, with input features divisible by 256 and output features divisible by 16. Other products retain their existing dispatch. The feature is disabled by default.

The backend prepares groups of 16 weight rows into the register-fragment order consumed by integer MMA. Each group retains the original nine-bit codebook indices, sign bits, FP16 block scales and four-bit subgroup scales. There is no floating-point requantization and no new GGUF type. The packed representation has the same byte count as the source. Verification consumes the packed codes directly, avoiding the shared-memory native-block staging used by the generic few-row kernel.

The current experiment retains native weights for prefill and adds the prepared copy for verification. Extra GPU weight storage therefore equals the total byte count of eligible IQ3_S tensors. Preparation skips new entries when they would leave less than 1 GiB free. This is a provisional workspace reserve, not a model-capacity guarantee. Measure actual peak memory for the intended context and workload.

Prepared data belongs to the original CUDA buffer. Host API writes mark entries dirty; preparation refreshes the existing allocation before the next graph submission. Keeping addresses stable allows captured graphs to keep using those allocations. Cache keys include the source address and matrix geometry. Events order preparation across consuming streams. Graphs that write weight buffers use the native path and invalidate prepared entries. Readback, row gathering and copies continue to use canonical weights.

For large input dimensions, four independent K partitions produce partial sums that a second kernel reduces. Other eligible matrices use one block with eight warps per 16 output rows. Both paths retain Q8_1 activation quantization. Their FP32 reduction order differs from MMVQ, so bit-identical logits or sampled continuations are not guaranteed by exact weight preservation.

Build and run the focused numerical tests on the intended GPU:

```sh
GGML_CUDA_IQ3_FRAGMENT=0 ./build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT_IQ3_VERIFY
GGML_CUDA_IQ3_FRAGMENT=1 ./build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT_IQ3_VERIFY
```

The tests cover eight-row products, activation padding, output-row boundaries, weight views and adjacent seven-/nine-row fallback paths. End-to-end qualification must additionally cover the actual model, context capacity, cache reuse, sampling and quality. An operator timing is not an inference speedup.
