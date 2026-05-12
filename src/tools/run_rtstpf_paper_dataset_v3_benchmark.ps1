param(
    [string]$Python = "python",
    [string]$Checkpoint = "src\outputs\stpf_training\rtstpf_paper_dataset_v3_paper_full_run_id\model_state.pt",
    [string]$ShardRoot = "src\datasets\training\rtstpf_paper_dataset_v3\shards\rtstpf_paper_dataset_v3_paper_full_run_id",
    [string]$TiManifest = "src\datasets\manifests\tight_inclusion_nyu_large_manifest_run_id.json",
    [string]$TiShardRoot = "src\datasets\training\tight_inclusion_nyu\shards\tight_inclusion_nyu_large_run_id",
    [int]$BatchSize = 32768,
    [int]$MaxTiQueries = 0,
    [int]$CalibrationMaxTiQueries = 0,
    [int]$MaxSchedulingRows = 0
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root

$env:PYTHONPATH = (Join-Path $Root "src\python")

$DenseOutputPrefix = "src\benchmark\rtstpf_paper_dataset_v3_complete_benchmark_run_id"
$SchedulingOutputPrefix = "src\benchmark\rtstpf_paper_dataset_v3_group_scheduling_run_id"
$TiRunName = "rtstpf_paper_dataset_v3_ti_sota_walltime_run_id"

Write-Host "[1/3] Dense full-scene exact-work / wall-time benchmark"
& $Python -m p2cccd.bench.rtstpf_checkpoint_complete_benchmark `
    --checkpoint $Checkpoint `
    --paper-full-shard-root $ShardRoot `
    --output-prefix $DenseOutputPrefix `
    --device cuda `
    --batch-size $BatchSize

Write-Host "[2/3] Group-level RTSTPFExact scheduling benchmark"
$schedArgs = @(
    "-m", "p2cccd.bench.rtstpf_scheduling_shard_benchmark",
    "--checkpoint", $Checkpoint,
    "--shard-root", $ShardRoot,
    "--split", "heldout_test",
    "--output-prefix", $SchedulingOutputPrefix,
    "--device", "cuda",
    "--batch-size", "$BatchSize"
)
if ($MaxSchedulingRows -gt 0) {
    $schedArgs += @("--max-rows", "$MaxSchedulingRows")
}
& $Python @schedArgs

Write-Host "[3/3] Primitive-level Tight-Inclusion SOTA wall-time benchmark"
$tiArgs = @(
    "-m", "p2cccd.bench.tight_inclusion_rtstpf_benchmark",
    "--manifest", $TiManifest,
    "--checkpoint", $Checkpoint,
    "--split", "heldout_test",
    "--calibration-split", "validation",
    "--output-dir", "src\benchmark",
    "--run-name", $TiRunName,
    "--shard-root", $TiShardRoot,
    "--device", "cuda",
    "--batch-size", "$BatchSize"
)
if ($MaxTiQueries -gt 0) {
    $tiArgs += @("--max-queries", "$MaxTiQueries")
}
if ($CalibrationMaxTiQueries -gt 0) {
    $tiArgs += @("--calibration-max-queries", "$CalibrationMaxTiQueries")
}
& $Python @tiArgs

Write-Host "DONE"
