# P2CCCD

`P2CCCD` is the main implementation directory for **Proposal-to-Certificate
Continuous Collision Detection**.

The current prototype is no longer just a skeleton. It implements and evaluates
a proposal-to-certificate CCD pipeline:

```text
conservative candidate generation
  -> learned STPF group scheduling
  -> exact local certificate / conservative fallback
```

The learned component is a scheduler. It does not output the final
collision/no-collision truth. RT/OptiX paths generate or traverse conservative
candidates. Final correctness comes from exact certificates or conservative
fallback.

## Layout

```text
P2CCCD/
  cpp/                 C++ hot-path data structures, candidate generation,
                       certificate backends, scheduling, and bindings.
  python/p2cccd/       Python orchestration, dataset adapters, STPF helpers,
                       benchmark runners, and visualization scripts.
  configs/             Runtime configs and benchmark-suite definitions.
  docs/                Reproducibility, architecture, adapter, and benchmark docs.
  tests/               C++ and Python quality gates.
  benchmark/           Curated benchmark reports bundled with this public
                       repository snapshot.
  datasets/            Dataset manifests and release notes. Large local dataset
                       roots are intentionally not bundled.
  baseline/            Helper scripts and path conventions for external CCD
                       baselines that are not bundled in this snapshot.
```

## Current Evidence Boundary

Supported claims:

- Dense/high-cost candidate groups can reduce exact certificate work with
  `FN=0` under the evaluated certificate/fallback protocol.
- All currently adapter-ready candidate-row datasets have a strict five-path
  replay report.
- The full Tight-Inclusion / NYU primitive every-candidate native baseline has
  been run as a correctness and SOTA-reference baseline.

Explicit non-claims:

- Do not claim `RTSTPFExact` is faster on every sparse primitive CCD query.
- Do not claim the neural network certifies collision or separation.
- Do not claim RT cores perform exact CCD certificates.
- Do not treat selected-real or consolidation matrices as exhaustive full-run
  source-pair coverage.

See also:

- `../../README.md`
- `../../artifacts/claim_safety_check.md`
- `docs/reproducibility_quickstart.md`
- `docs/paper_case_reproduction.md`
- `docs/model_artifacts_manifest.md`

## Quick Start

Use the CPU smoke / CI environment before running Python commands:

```powershell
conda activate cudadev
```

Install the local Python package in editable mode:

```powershell
cd src
python -m pip install -e .
```

Run the CI-friendly Python subset from the repository root:

```powershell
conda activate cudadev
python -m pytest src\tests\python\test_contracts.py src\tests\python\test_correctness_and_performance_gates.py src\tests\python\test_quality_gate_inventory.py -q
```

Configure and build the default CPU C++ targets:

```powershell
cmake -S src -B src\build -DP2CCCD_BUILD_TESTS=ON
cmake --build src\build --config Release
ctest --test-dir src\build -C Release --output-on-failure
```

Optional CUDA/OptiX builds are gated by CMake options:

```powershell
cmake -S src -B src\build_cuda -DP2CCCD_ENABLE_CUDA=ON
cmake -S src -B src\build_optix -DP2CCCD_ENABLE_OPTIX=ON
```

GPU/OptiX/TensorRT experiments require local vendor SDK paths and ignored
model/data artifacts. See `docs/gpu_optix_environment.md` and
`docs/model_artifacts_manifest.md`.

This public release intentionally omits heavy local dataset roots, external
baseline repositories, build products, and generated training outputs. The
expected mount points for those assets are documented in `datasets/README.md`,
`baseline/README.md`, and `docs/third_party_manifest.md`.

## Key Reproducibility Entry Points

- Test documentation: `tests/README.md`
- Machine-readable quality gates: `tests/quality_gates.json`
- Current architecture notes: `docs/architecture.md`
- Reproducibility quickstart: `docs/reproducibility_quickstart.md`
- Paper case map: `docs/paper_case_reproduction.md`
- Benchmark-suite configs: `configs/benchmark_suites/`
- Dataset and license manifests: `datasets/manifests/`

Recent full-run evidence:

- `benchmark/all_dataset_strict_five_path_full_replay_run_id/all_dataset_strict_five_path_full_replay_run_id.md`
- `benchmark/ti_full_query_every_candidate_walltime_run_id/ti_full_query_every_candidate_walltime_run_id.md`

## Evidence Artifacts

Paper-facing evidence manifests included with this repository are under:

```text
../../artifacts/
```

This release snapshot is scoped to the public code/data repository rather than
the full paper workspace. The evidence and claim-boundary files here are the
entry points used to verify bundled reports.
