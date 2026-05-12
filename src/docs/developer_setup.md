# Developer Setup

This note is the repository-local setup checklist for `src`.

## Python

Use the CUDA-aware conda environment requested by the project:

```powershell
conda activate cudadev
$env:PYTHONPATH = "src/python"
python -m pytest src/tests/python -q
```

Required Python-side smoke imports:

```powershell
conda activate cudadev
$env:PYTHONPATH = "src/python"
python -c "import p2cccd; import torch; import onnxruntime as ort; print(ort.get_available_providers())"
```

## CMake

The primary build tree is `src/build`.

```powershell
cmake -S src -B src/build -DCMAKE_BUILD_TYPE=Release
cmake --build src/build --config Release
ctest --test-dir src/build -C Release --output-on-failure
```

CPU-only builds should keep CUDA and OptiX optional. CUDA/OptiX failures must not break the CI-friendly CPU suite.

CUDA-aware Python binding smoke check:

```powershell
conda activate cudadev
$env:PYTHONPATH = "src/python"
python -c "from p2cccd.cuda_bindings import get_cuda_binding_status; print(get_cuda_binding_status())"
```

For a CUDA-enabled pybind build, configure both Python and CUDA:

```powershell
cmake -S src -B src/build_cuda_py -DP2CCCD_ENABLE_PYTHON=ON -DP2CCCD_ENABLE_CUDA=ON
cmake --build src/build_cuda_py --config Release --target p2cccd_cpp
```

The Python CUDA binding boundary is host-batch only. Raw device-pointer ABI is intentionally disabled until ownership and lifetime contracts are certified.

For the real OptiX RT candidate backend:

```powershell
cmake -S src -B src/build_optix -DP2CCCD_ENABLE_OPTIX=ON -DP2CCCD_ENABLE_PYTHON=ON
cmake --build src/build_optix --config Release
$env:PYTHONPATH = "src/python"
$env:CUDA_PATH = "<path-to-your-cuda-toolkit>"
python -c "from p2cccd.bench.suite_runner import run_benchmark_suite_from_config_path; r=run_benchmark_suite_from_config_path('src/configs/benchmark_suites/performance_optix.json', run_id='manual_optix_smoke'); print([(c.case_name, c.final_fn_zero) for c in r.case_results])"
```

## Environment Variables

- `PYTHONPATH=src/python`: required when running package modules from the repository root.
- `CUDA_PATH` or `CUDA_HOME`: used only for environment reporting and optional CUDA builds.
- `OPTIX_ROOT` or project-specific OptiX CMake variables: required only when the optional `optix_rt` candidate backend is enabled.
- `OPTIX_VERSION`: optional metadata field for `BenchmarkRunMeta`.

## Local Reference Roots

External reference stacks are expected under:

```text
src/baseline/
  Tight-Inclusion/
  CCD-Wrapper/
  Scalable-CCD/
  Sample-Scalable-CCD-Data/
  Exact-Root-Parity-CCD/
  rigid-ipc/
```

Reference adapters are license-gated. Missing or incomplete license/source metadata should fail before data ingestion.

## Minimal Health Check

Run these before starting a larger benchmark:

```powershell
conda activate cudadev
$env:PYTHONPATH = "src/python"
python -m pytest src/tests/python/test_correctness_and_performance_gates.py -q
ctest --test-dir src/build -C Release --output-on-failure
```
