param(
    [string]$Python = "python",
    [string]$Preset = "paper_full",
    [int]$ChunkRows = 1000000,
    [int]$PositiveOversample = 4,
    [string]$ModelPreset = "medium_mlp",
    [string]$Device = "cuda",
    [int]$Epochs = 8,
    [int]$BatchSize = 32768,
    [double]$LearningRate = 8.0e-4
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root

$env:PYTHONPATH = (Join-Path $Root "src\python")

$DatasetRoot = "src\baseline\datasets\continuous-collision-detection"
$Manifest = "src\datasets\manifests\rtstpf_paper_dataset_v2_manifest_run_id.json"
$DesignReport = "src\benchmark\rtstpf_paper_dataset_v2_design_run_id.md"
$ShardDir = "src\datasets\training\rtstpf_paper_dataset_v2\shards\rtstpf_paper_dataset_v2_$Preset`_run_id"
$ShardReport = "src\benchmark\rtstpf_paper_dataset_v2_$Preset`_run_id.md"
$RunName = "rtstpf_paper_dataset_v2_$Preset`_run_id"
$ReportName = "rtstpf_paper_dataset_v2_$Preset`_training_run_id"

Write-Host "[1/3] Build manifest"
& $Python -m p2cccd.bench.rtstpf_paper_dataset_v2 manifest `
    --root $DatasetRoot `
    --output $Manifest `
    --report $DesignReport

Write-Host "[2/3] Build chunked shards: $Preset"
& $Python -m p2cccd.bench.rtstpf_paper_dataset_v2 shards `
    --manifest $Manifest `
    --output-dir $ShardDir `
    --preset $Preset `
    --chunk-rows $ChunkRows `
    --positive-oversample $PositiveOversample `
    --report $ShardReport

Write-Host "[3/3] Streaming train STPF"
@"
from pathlib import Path
from p2cccd.bench.tight_inclusion_stpf_training import run_tight_inclusion_stpf_training

result = run_tight_inclusion_stpf_training(
    Path(r"$ShardDir"),
    run_name=r"$RunName",
    report_name=r"$ReportName",
    model_preset=r"$ModelPreset",
    device=r"$Device",
    epochs=$Epochs,
    batch_size=$BatchSize,
    learning_rate=$LearningRate,
    train_eval_max_rows=2000000,
    validation_eval_max_rows=2000000,
)
print(result["report_path"])
print(result["model_state_path"])
"@ | & $Python -

Write-Host "DONE"
