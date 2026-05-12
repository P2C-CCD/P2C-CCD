# Benchmark Suites

Benchmark suites provide reproducible runners for four benchmark categories:

- `correctness`: small false-negative and candidate-recall regression runs.
- `performance`: CPU smoke performance runs for CI-friendly baseline timing.
- `performance_optix`: optional OptiX RT Core smoke runs for `RTExact` and `RTSTPFExact`; requires `p2cccd_cpp` built with `P2CCCD_ENABLE_OPTIX=ON`.
- `ablation`: patch granularity, slab count, proxy family, and STPF-head ablations.
- `ood_stress`: OOD and hard-contact scenes for monotonic safety and fallback behavior.

Suite configs live in:

```text
src/configs/benchmark_suites/
  correctness.json
  correctness_external.json
  performance.json
  ablation.json
  ood_stress.json
  rt_style_reproduction.json
  learned_style_comparison.json
  curobo_downstream.json
```

## CLI

Run from the repository root with the `cudadev` environment:

```powershell
conda activate cudadev
$env:PYTHONPATH = "src/python"
python -m p2cccd.bench.run_suite --config src/configs/benchmark_suites/correctness.json
```

Override the output root:

```powershell
$env:PYTHONPATH = "src/python"
python -m p2cccd.bench.run_suite `
  --config src/configs/benchmark_suites/ablation.json `
  --output-root src/outputs/benchmark_suites
```

Each run writes:

```text
<output_root>/<suite_name>/<run_id>/
  benchmark.csv
  benchmark.jsonl
  run_meta.json
```

The output uses `BenchmarkRunMeta` plus `BenchmarkRowV2`, so every row has dataset, scene, method, config hash, seed, latency percentiles, exact-call family counters, RT timing split, candidate/proposal health metrics, NoQueueDecouple microbenchmark fields, and VRAM fields.

For the method-by-method audit map, see `src/docs/benchmarks_and_baselines.md`.

## Python API

```python
from p2cccd.bench import run_benchmark_suite_from_config_path

result = run_benchmark_suite_from_config_path(
    "src/configs/benchmark_suites/correctness.json",
    output_root="src/outputs/benchmark_suites",
)

print(result.meta_run_id)
print(len(result.rows))
```

## Supported Methods

The suite runner currently supports:

- `PureExactCPU`
- `BVHExact`
- `SortBroadPhaseExact`
- `RTExact`
- `RTSTPFExact`
- `RTDCDStyle`
- `RTCCDStyle`
- `NeuralSVCDStyle`
- `CabiNetStyle`
- `CuRoboDownstream`
- `NoProposal`
- `IntervalOnly`
- `RankingOnly`
- `PatchGranularityAblation`
- `SlabProxyAblation`
- `NoQueueDecouple`
- `ReferenceAvailability`

Suite configs support the internal generated analytic dataset plus first-layer external query batches:

- `internal_generated`: deterministic analytic samples used by CI and smoke suites.
- `scalable_ccd_sample`: local Sample-Scalable-CCD-Data query batches.
- `rigid_ipc`: local Rigid-IPC fixture-derived body-pair query batches.
- `tight_inclusion_reference`: reference-stack availability checks for Tight Inclusion.

The performance, ablation, and OOD/stress configs contain named proxy cases for ABC/Fusion360, MoveIt, Thingi10K, Google Scanned Objects, and PartNet-Mobility. ABC and Thingi10K now also have dedicated dataset-specific Python runners for full training and held-out benchmark execution; the generic suite configs still keep the smaller deterministic smoke cases for CI-friendly regression.

## Safety Contract

Correctness-oriented rows should keep:

- `candidate_recall = 1.0`
- `fn_count = 0`

The suite runner does not hide failures. Rows are exported even when a case has false negatives, so regression dashboards can catch the exact failing method/config.
