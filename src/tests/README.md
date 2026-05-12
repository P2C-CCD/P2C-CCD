# Tests

Run Python tests from the repository root after activating the required environment:

```powershell
conda activate cudadev
python -m pytest src\tests\python -q
```

Run C++ tests from the configured build directory:

```powershell
ctest --test-dir src\build -C Release --output-on-failure
```

Quality gate groups:

- unit: proxy bounds, contract validators, schema checks
- regression: CPU oracle replay, audit replay contracts, and CUDA-aware binding ABI smoke tests
- correctness: candidate recall, proposal monotonicity, CPU exact certificates, end-to-end final FN=0
- perf: RT / proposal / exact timing breakdown export smoke tests
- external-export: external CCD/CAD license gates, BenchmarkRowV2 export validators, and seed reproducibility smoke tests
- ci-minimal: `configs/benchmark_suites/ci_minimal_cpu.json` runs `PureExactCPU`, `BVHExact`, and CPU-reference `RTExact` without OptiX or CUDA runtime paths

The machine-readable quality gate inventory is `tests/quality_gates.json`. It maps TODO 107-116 to concrete pytest files, CTest names, benchmark suite configs, and accelerator-runtime requirements.

CI-friendly minimal Python subset:

```powershell
conda activate cudadev
python -m pytest src\tests\python\test_contracts.py src\tests\python\test_correctness_and_performance_gates.py src\tests\python\test_quality_gate_inventory.py -q
```
