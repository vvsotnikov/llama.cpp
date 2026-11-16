param(
    [Parameter(Mandatory = $true)][string]$OutputCsv,
    [ValidateRange(0, 2147483647)][int]$ServerProcessId = 0,
    [ValidateRange(0, 2147483647)][int]$GpuIndex = 0,
    [ValidateRange(1, 60)][int]$IntervalSeconds = 3,
    [ValidateRange(0, 2147483647)][int]$DurationSeconds = 0,
    [string]$StopFile = ''
)
$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
$parent = Split-Path -Parent ([System.IO.Path]::GetFullPath($OutputCsv))
$null = New-Item -ItemType Directory -Force -Path $parent
if (Test-Path -LiteralPath $OutputCsv) { throw "Output already exists: $OutputCsv" }
$watch = [System.Diagnostics.Stopwatch]::StartNew()
$writer = [System.IO.StreamWriter]::new($OutputCsv, $false, [System.Text.UTF8Encoding]::new($false))
$writer.AutoFlush = $true
$first = $true
try {
    while (($DurationSeconds -le 0 -or $watch.Elapsed.TotalSeconds -lt $DurationSeconds) -and (-not $StopFile -or -not (Test-Path -LiteralPath $StopFile))) {
        $sampleStart = $watch.Elapsed.TotalSeconds
        $row = [ordered]@{
            utc = [DateTime]::UtcNow.ToString('o'); elapsed_s = [Math]::Round($sampleStart, 3)
            qpc_s = ([System.Diagnostics.Stopwatch]::GetTimestamp() / [double][System.Diagnostics.Stopwatch]::Frequency).ToString('F6', [System.Globalization.CultureInfo]::InvariantCulture)
            server_pid = $ServerProcessId; gpu_index = $GpuIndex; gpu_power_w = ''; gpu_temperature_c = ''
            gpu_util_percent = ''; gpu_memory_util_percent = ''; gpu_memory_used_mib = ''
            gpu_dedicated_total_mib = ''; gpu_sm_clock_mhz = ''; gpu_pstate = ''
            server_gpu_dedicated_bytes = ''; server_gpu_shared_bytes = ''; server_gpu_committed_bytes = ''
            server_working_set_bytes = ''; server_private_bytes = ''; server_cpu_seconds = ''
            system_available_bytes = ''; system_committed_bytes = ''; system_commit_limit_bytes = ''
            ac_online = ''; battery_discharge_rate_mw = ''; error = ''
        }
        try {
            $smi = @(& nvidia-smi --id=$GpuIndex --query-gpu=power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.sm,pstate --format=csv,noheader,nounits 2>$null)
            if ($LASTEXITCODE -ne 0 -or $smi.Count -ne 1) { throw 'nvidia-smi query failed' }
            $values = $smi[0] -split ',\s*'
            $keys = @('gpu_power_w','gpu_temperature_c','gpu_util_percent','gpu_memory_util_percent','gpu_memory_used_mib','gpu_dedicated_total_mib','gpu_sm_clock_mhz','gpu_pstate')
            for ($i = 0; $i -lt $keys.Count; $i++) { $row[$keys[$i]] = $values[$i].Trim() }
            if ($ServerProcessId -gt 0) {
                $process = Get-Process -Id $ServerProcessId -ErrorAction Stop
                $row.server_working_set_bytes = $process.WorkingSet64
                $row.server_private_bytes = $process.PrivateMemorySize64
                $row.server_cpu_seconds = $process.TotalProcessorTime.TotalSeconds
                $gpuMemory = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -ErrorAction Stop | Where-Object { $_.Name -like "pid_${ServerProcessId}_*" })
                if ($gpuMemory.Count -gt 0) {
                    $row.server_gpu_dedicated_bytes = ($gpuMemory | Measure-Object DedicatedUsage -Sum).Sum
                    $row.server_gpu_shared_bytes = ($gpuMemory | Measure-Object SharedUsage -Sum).Sum
                    $row.server_gpu_committed_bytes = ($gpuMemory | Measure-Object TotalCommitted -Sum).Sum
                }
            }
            $memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction Stop
            $row.system_available_bytes = $memory.AvailableBytes
            $row.system_committed_bytes = $memory.CommittedBytes
            $row.system_commit_limit_bytes = $memory.CommitLimit
            $battery = Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($battery) {
                $row.ac_online = $battery.PowerOnline
                if (-not $battery.PowerOnline) { $row.battery_discharge_rate_mw = $battery.DischargeRate }
            }
        } catch { $row.error = $_.Exception.Message }
        $csv = @([pscustomobject]$row | ConvertTo-Csv -NoTypeInformation)
        if ($first) { $writer.WriteLine($csv[0]); $first = $false }
        $writer.WriteLine($csv[1])
        $remaining = $IntervalSeconds - ($watch.Elapsed.TotalSeconds - $sampleStart)
        if ($remaining -gt 0) { Start-Sleep -Milliseconds ([int]($remaining * 1000)) }
    }
} finally { $writer.Dispose() }
