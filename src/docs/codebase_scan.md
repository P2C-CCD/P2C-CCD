# Codebase Scan

This scan records the current `src` implementation layout after the TODO list was reordered by stable task ID.

## Source Layout

- `cpp/`: C++ runtime contracts, geometry core, RT candidate generation including optional OptiX RT Core device hit emission, exact certificate engine, optional CUDA exact kernels, and pybind11 bindings.
- `python/p2cccd/`: Python contracts, validators, serialization, config loading, data generation, proposal/STPF helpers, external CCD adapter contracts, benchmark runners, and CUDA-aware binding wrappers.
- `python/p2cccd/bench/`: runnable Python baselines, ablations, style comparisons, suite runner, and V2 benchmark export.
- `python/p2cccd/data/`: programmatic motion samplers, analytic swept-sphere oracle, dataset shards, metrics, and optional Warp-aware helpers.
- `python/p2cccd/datasets/ccd/`: external CCD adapter contracts and first-layer source adapters.
- `tests/cpp/`: CTest-backed C++ contract, geometry, RT candidate, certificate, CUDA-stub/cross-check, and proposal queue tests.
- `tests/python/`: pytest coverage for contracts, data generation, external adapters, baselines, benchmark export, suite runner, STPF, and training helpers.
- `tools/`: standalone visualization/debug tools for early pipeline stages.

## Current Verification State

- TODO entries are contiguous from `1` through `167`, with no duplicate IDs.
- As of the run_id audit, all 167 TODO entries are marked `done`; `pending` and `blocked` counts are both zero.
- TODO sections are now ordered by stable numeric task ID.
- TODO 106 is implemented as a host-batch CUDA-aware Python binding boundary with raw device-pointer ABI disabled.
- `Benchmarks And Baselines` has a dedicated audit map in `docs/benchmarks_and_baselines.md`.
- Python baseline modules have matching tests and public `p2cccd.bench` exports.
- Generated `__pycache__` directories are present after test runs but ignored by `.gitignore`.
- Root-level historical planning files may still mention older TODO counts, but they now carry current-state notes and should not be used as implementation status sources.

## Known Boundaries

- Real OptiX device-side `RawCandidateHit` emission is implemented behind `P2CCCD_ENABLE_OPTIX`; CPU-compatible runners remain the default for CI.
- `BVHExact` has an Embree/Coal-compatible abstraction name, but not a real Embree or Coal backend yet.
- `NeuralSVCDStyle`, `CabiNetStyle`, `RTDCDStyle`, `RTCCDStyle`, and `CuRoboDownstream` are style/downstream comparisons, not official third-party integrations.
- Python bindings expose candidate generation, CPU exact execution, audit replay, and CUDA exact host-batch entrypoints.
- External dataset suites beyond first-layer adapter smoke coverage still require real local full-dataset runs before paper-scale reporting.

## Scan Commands

```powershell
conda activate cudadev
python -m pytest src\tests\python -q
ctest --test-dir src\build -C Release --output-on-failure
rg -n "TODO|FIXME|HACK|NotImplemented|pass$|placeholder|stub|blocked" src --glob "!build/**" --glob "!**/__pycache__/**"
```
