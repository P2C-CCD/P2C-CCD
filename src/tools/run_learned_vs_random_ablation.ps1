param(
  [string]$Python = "python",
  [string]$RunName = "learned_vs_random_ablation_run_id",
  [int]$RandomSeedCount = 30,
  [int]$GroupSize = 512,
  [int]$PositivesPerGroup = 4,
  [int]$MaxGroups = 512,
  [int]$BatchSize = 65536,
  [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$PythonPath = Join-Path $RepoRoot "src\python"
$env:PYTHONPATH = $PythonPath

& $Python -m p2cccd.bench.learned_vs_random_ablation `
  --run-name $RunName `
  --random-seed-count $RandomSeedCount `
  --group-size $GroupSize `
  --positives-per-group $PositivesPerGroup `
  --max-groups $MaxGroups `
  --batch-size $BatchSize `
  --device $Device
