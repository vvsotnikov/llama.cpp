# Experimental CUDA host input staging

`GGML_CUDA_ASYNC_INPUTS=1` stages small contiguous host inputs before queuing their GPU copies. The switch is disabled by default.

The staging path copies the caller's bytes into owned pinned memory before returning. The caller can overwrite or release its input immediately. Copies run on the consuming CUDA stream, after earlier users of the destination and before the next graph. Each CUDA buffer using this path owns 64 slots of 64 KiB, totaling 4 MiB of pinned staging data, plus completion events. A slot can be reused after its copy event completes. A full ring waits for a copy to finish; buffer destruction drains pending copies before freeing the staging data. Larger or noncontiguous inputs and unavailable pinned memory retain the synchronous path.

The scheduler discovers this operation through the optional `ggml_backend_copy_input_async` registry extension. A successful call must capture host input bytes before returning and order the destination copy on the backend stream. Backends without this extension retain their existing input-copy behavior.

Copies preserve the original tensor bytes and layouts. The caller can overwrite or release the source immediately after a successful staged copy.

Build with the normal CUDA configuration and test ownership under queued GPU work, immediate host overwrite and release, ring wrap, unsupported layouts and pinned-memory fallback:

```sh
GGML_CUDA_ASYNC_INPUTS=1 ./build/bin/test-backend-tensor
GGML_CUDA_ASYNC_INPUTS=1 GGML_CUDA_NO_PINNED=1 ./build/bin/test-backend-tensor
```

Set the switch on the existing server command. Keep model, sampling, cache policy and workload identical for performance comparisons.
