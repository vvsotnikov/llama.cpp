# Experimental DFlash phase workspaces

`LLAMA_DFLASH_PHASE_GRAPHS=1` retains separate graph results and scheduler allocations for context injection and token drafting. It is disabled by default.

Both phases can have zero logits outputs. Without this switch they select the same graph-result slot, so alternating their different inputs rebuilds the graph and allocates it again. Separate workspaces retain each phase's graph and buffer addresses. Existing graph-parameter and input checks still decide whether a graph can be reused. CUDA graph capture remains a separate backend decision.

The switch applies to a DFlash context attached to a target context, with one sequence, graph reuse enabled, no pipeline parallelism, and no training optimizer. It does not change the target graph, numerical operations, draft width, sampling, or prompt-cache policy. Other configurations keep the existing path. CUDA DFlash2 is the qualified configuration; other DFlash variants and backends require validation before use.

The additional workspace is allocated lazily and released with the context or on scheduler reservation. Memory updates, graph reservation, and sampler changes invalidate retained graphs. Memory reporting includes both scheduler allocations. The extra compute memory depends on the model and batch shapes; allocation failure retains the existing error path, so leave this experimental switch off in configurations without spare memory.

Build normally and compare the same server command with the switch set to `0` and `1`. Keep all independent optimization switches equal between runs:

```sh
LLAMA_DFLASH_PHASE_GRAPHS=1 ./build/bin/llama-server \
    -m "$TARGET_GGUF" -md "$DRAFTER_GGUF" \
    --spec-type draft-dflash --spec-draft-n-max 7 \
    --spec-draft-sampling probabilistic --spec-draft-backend-sampling \
    --backend-sampling --kv-unified -ngl 99 -ngld 99 -np 1 -c 40960 \
    -b 2048 -ub 512 --fit off -fa on -ctk f16 -ctv f16 --cache-prompt \
    --temp 1 --top-p 0.95 --top-k 20 --min-p 0 \
    --presence-penalty 0 --repeat-penalty 1 --reasoning on --reasoning-effort low
```

Measure prefill and decode separately with the existing workload and sampling configuration. Include warmup and alternate control/candidate order. Check output tokens and draft counts: runtime kernel selection can change sampled continuations independently of this switch. Use fixed dispatch for a separate numerical comparison, and exercise cached appends, tail edits, sampler changes, cancellation, and slot save/restore. Report the memory cost and results on each tested GPU rather than inferring speed from the graph-reuse counter.
