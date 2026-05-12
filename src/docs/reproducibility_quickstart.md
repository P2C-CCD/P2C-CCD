# Reproducibility Quickstart

This file is the first stop for a clean, CPU-friendly repository check. It does
not rerun large GPU training or benchmark sweeps.

## Environment

Python commands for smoke gates must use the project environment:

```powershell
conda activate cudadev
python --version
```

Expected in the current workspace: Python 3.12 in the `cudadev` environment.
This is the CPU smoke / CI environment. It is not a full CUDA/OptiX/TensorRT
lock file; see `gpu_optix_environment.md` for GPU benchmark prerequisites.

## Minimal Python Gates

Run from the repository root:

```powershell
conda activate cudadev
python -m pytest src\tests\python\test_contracts.py src\tests\python\test_correctness_and_performance_gates.py src\tests\python\test_quality_gate_inventory.py -q
```

For a shorter smoke check:

```powershell
conda activate cudadev
python -m pytest src\tests\python\test_contracts.py src\tests\python\test_quality_gate_inventory.py -q
```

Latest local smoke result:

```text
17 passed
```

## Paper Case Bundle Check

The public release bundles the paper-level case figures and curated
evidence under release-local paths. Verify that all listed files, directories,
and rerun entry points are present:

```powershell
conda activate cudadev
python scripts\verify_release_cases.py
```

The case map is documented in:

```text
src\docs\paper_case_reproduction.md
```

## C++ CPU Gates

If `src\build` exists:

```powershell
ctest --test-dir src\build -C Release --output-on-failure
```

Latest local C++ result:

```text
100% tests passed, 0 tests failed out of 15
```

To configure from scratch:

```powershell
cmake -S src -B src\build -DP2CCCD_BUILD_TESTS=ON
cmake --build src\build --config Release
ctest --test-dir src\build -C Release --output-on-failure
```

## Large Evidence Reports

Do not rerun these unless intentionally spending benchmark time. Use the existing
reports as the current evidence state:

```text
src\benchmark\all_dataset_strict_five_path_full_replay_run_id\all_dataset_strict_five_path_full_replay_run_id.md
src\benchmark\ti_full_query_every_candidate_walltime_run_id\ti_full_query_every_candidate_walltime_run_id.md
```

Hash manifests for paper-critical evidence and ignored model/data artifacts:

```text
artifacts\evidence_manifest.md
src\docs\model_artifacts_manifest.md
```

`artifacts\evidence_manifest.*` is a redacted public placeholder. The release
local case manifest and bundled evidence live entirely inside this repository.

Verify the model/data artifact manifest bundled with this repository:

```powershell
conda activate cudadev
python src\tools\verify_artifact_manifest.py src\docs\model_artifacts_manifest.json
```

## Claim Safety

Before writing paper text, check:

```text
artifacts\claim_safety_check.md
```

Key boundary:

- Dense/high-cost exact-work reductions are supported.
- Full TI/NYU primitive every-candidate wall-time is a native Tight-Inclusion
  baseline, not a sparse-query RTSTPF speedup claim.
- The neural network schedules exact work; it does not certify the final result.
