param(
    [Parameter(Mandatory = $true)][string]$Server,
    [Parameter(Mandatory = $true)][string]$Target,
    [Parameter(Mandatory = $true)][string]$Draft,
    [ValidateRange(1, 65535)][int]$Port = 8080,
    [ValidateRange(512, 2147483647)][int]$Context = 8192
)
$ErrorActionPreference = 'Stop'
foreach ($path in @($Server, $Target, $Draft)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $path" }
}
$serverPath = (Resolve-Path -LiteralPath $Server).ProviderPath
$targetPath = (Resolve-Path -LiteralPath $Target).ProviderPath
$draftPath = (Resolve-Path -LiteralPath $Draft).ProviderPath
$environment = @{
    LLAMA_DFLASH_TREE = '1'
    GGML_CUDA_TREE_SLOTS = '1'
    GGML_CUDA_ASYNC_INPUTS = '1'
    LLAMA_DFLASH_ASYNC_INJECT = '1'
    LLAMA_DFLASH_PHASE_GRAPHS = '1'
    GGML_CUDA_GLU_MMQ_FUSION = '1'
    GGML_CUDA_IQ4_FRAGMENT = '1'
    GGML_CUDA_PREFILL_IQ3 = '1'
    LLAMA_DFLASH_AV_COST_US = $null
}
$previous = @{}
foreach ($name in $environment.Keys) {
    $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$arguments = @(
    '-m', $targetPath, '-md', $draftPath,
    '--alias', 'spark-qwen', '--host', '127.0.0.1', '--port', $Port,
    '--gpu-layers', 'all', '--spec-draft-ngl', 'all', '--fit', 'off',
    '--ctx-size', $Context, '--parallel', '1', '--batch-size', '512', '--ubatch-size', '512',
    '--flash-attn', 'on', '--load-mode', 'none', '--kv-unified', '-ctk', 'f16', '-ctv', 'f16',
    '--jinja', '--metrics', '--perf',
    '--spec-type', 'draft-dflash', '--spec-draft-n-max', '15',
    '--spec-draft-n-min', '0', '--spec-draft-p-min', '0', '--spec-draft-sampling', 'probabilistic',
    '--backend-sampling', '--spec-draft-backend-sampling',
    '--temp', '1', '--top-k', '20', '--top-p', '0.95', '--min-p', '0.05',
    '--repeat-penalty', '1', '--presence-penalty', '0',
    '-t', '1', '-tb', '1', '--spec-draft-threads', '1', '--spec-draft-threads-batch', '1',
    '--poll', '0', '--poll-batch', '0', '--spec-draft-poll', '0', '--spec-draft-poll-batch', '0',
    '--reasoning-effort', 'xhigh'
)
try {
    foreach ($name in $environment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $environment[$name], 'Process')
    }
    & $serverPath @arguments
    if ($LASTEXITCODE -ne 0) { throw "llama-server exited with code $LASTEXITCODE" }
} finally {
    foreach ($name in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process')
    }
}
