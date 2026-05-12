# Milestone Execution Checklist

## Milestone 1: Contracts And Geometry

- Runtime contracts validate on C++ and Python.
- `default_runtime.json` loads and rejects invalid epsilon/runtime limits.
- Mesh, patch, slab, swept AABB, and capsule proxy tests pass.
- Visualization output exists for geometry/proxy debugging.

Verification:

```powershell
conda activate cudadev
$env:PYTHONPATH = "src/python"
python -m pytest src/tests/python/test_contracts.py src/tests/python/test_geometry_core.py -q
```

## Milestone 2: Candidate Recall

- CPU reference candidate generation runs.
- Candidate compaction preserves query/candidate identities.
- Candidate recall against CPU oracle is 1.0.
- Slab refinement does not reduce recall.

Verification:

```powershell
conda activate cudadev
$env:PYTHONPATH = "src/python"
python -m pytest src/tests/python/test_candidate_generation.py src/tests/python/test_rt_exact_baseline.py -q
```

## Milestone 3: Exact Certificates

- Point-triangle and edge-edge CPU exact oracles pass regression cases.
- Exact queue processing preserves every work item.
- Collision, separation, and undecided branches emit certificates or refinement requests.
- Audit log rows are emitted and replayable.

Verification:

```powershell
conda activate cudadev
$env:PYTHONPATH = "src/python"
python -m pytest src/tests/python/test_certificate_engine.py -q
```

## Milestone 4: End-To-End Benchmarks

- `RTExact` passes final FN = 0.
- `RTSTPFExact` passes no-drop monotonic scheduling.
- Benchmark suites export `BenchmarkRunMeta`, CSV, JSONL, and `run_meta.json`.
- `BenchmarkRowV2` includes latency percentiles, family-wise exact calls, candidate/proposal health metrics, and profiling fields.

Verification:

```powershell
conda activate cudadev
$env:PYTHONPATH = "src/python"
python -m pytest src/tests/python/test_near_term_execution_order.py src/tests/python/test_correctness_and_performance_gates.py -q
python -m p2cccd.bench.run_suite --config src/configs/benchmark_suites/correctness.json
```
