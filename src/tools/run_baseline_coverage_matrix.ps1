param(
    [string]$Root = ".",
    [string]$OutputDir = "src\benchmark",
    [string]$RunName = "baseline_matrix_run_id",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath $Root).Path
$env:PYTHONPATH = Join-Path $repoRoot "src\python"

& $PythonExe -m p2cccd.bench.baseline_coverage_matrix `
    --root $repoRoot `
    --output-dir (Join-Path $repoRoot $OutputDir) `
    --run-name $RunName
