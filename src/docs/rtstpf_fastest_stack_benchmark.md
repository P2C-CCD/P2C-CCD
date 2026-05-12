## RTSTPF Fastest-Stack Benchmark

This note records the current fastest executable `RTSTPFExact` stack for external CCD batches and the benchmark used to validate it.

### Target stack

The runtime target is:

- `optix_rt` for RT candidate generation
- C++ proposal feature extraction and C++ proposal scheduling through `p2cccd_cpp`
- ONNX Runtime for learned STPF inference
- TensorRT Execution Provider preferred, `CUDAExecutionProvider` fallback
- CUDA exact for external exact batches when `enable_cuda_exact=True` and the batch is large enough for the CUDA path

### Current code paths

- `RTSTPFExactConfig` in [rt_stpf_exact.py](../python/p2cccd/bench/rt_stpf_exact.py)
  - `inference_backend`: `torch` or `ort`
  - `enable_cuda_exact`: explicit external CUDA exact gate
  - `ort_model_path`, `ort_prefer_tensorrt`, `ort_allow_cuda_fallback`, `ort_allow_cpu_fallback`, `ort_opset_version`, `ort_warmup_passes`
- ORT export/inference helper in [ort_inference.py](../python/p2cccd/proposal/ort_inference.py)
- CUDA exact gate in [rt_exact.py](../python/p2cccd/bench/rt_exact.py)
- real benchmark runner in [rtstpf_fastest_stack_benchmark.py](../python/p2cccd/bench/rtstpf_fastest_stack_benchmark.py)

### Benchmark dataset

The reference benchmark uses:

- source: `Sample-Scalable-CCD-Data`
- scene: `armadillo-rollers`
- family: `vf`
- step: `326`
- query count: `4096`

This is a real external CCD batch, not an internal synthetic workload.

### Runtime state on this machine

The current machine now has a usable TensorRT runtime under:

- `$env:P2CCCD_TENSORRT_ROOT`

`ORT` runtime initialization in [ort_inference.py](../python/p2cccd/proposal/ort_inference.py) registers:

- TensorRT `bin`
- `CONDA_PREFIX\\Library\\bin`
- `torch\\lib`

With that runtime visible, `RTSTPFExact-ORT` can run with:

- `TensorrtExecutionProvider`

### Benchmark timing boundary

TensorRT EP has a meaningful cold-start cost from engine creation and cache materialization. The current benchmark path explicitly runs ORT warmup passes before the timed proposal section:

- `RTSTPFExactConfig.ort_warmup_passes`

This keeps the first engine build out of `proposal_ms` and makes the exported benchmark closer to steady-state latency.
