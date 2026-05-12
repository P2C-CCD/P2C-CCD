param(
  [string]$Python = "python",
  [string]$RunName = "native_dense_group_walltime_run_id",
  [string]$OutputDir = "src\benchmark",
  [string]$Device = "cuda",
  [int]$BatchSize = 65536,
  [int]$WarmupPasses = 1
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $RepoRoot
$env:PYTHONPATH = Join-Path $RepoRoot "src\python"

& $Python -m p2cccd.bench.native_dense_group_benchmark `
  --output-dir $OutputDir `
  --run-name $RunName `
  --device $Device `
  --batch-size $BatchSize `
  --warmup-passes $WarmupPasses
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
