# RTSTPFExact Baseline

`RTSTPFExact` is the first end-to-end P2CCCD pipeline that inserts the
spacetime proposal field between conservative RT candidates and exact local
certificates.

## Pipeline

```text
external/internal query batch
  -> conservative RT-compatible candidate generation
  -> ProposalFeatureRow extraction
  -> STPFModel / ORT batched inference
  -> monotonic proposal-aware ExactWorkItem scheduling
  -> PureExactCPU or CUDA exact certificate execution
  -> BenchmarkRow + audit/certificate coverage
```

The Python workbench entry points are:

```python
from p2cccd.bench import (
    RTSTPFExactConfig,
    run_rt_stpf_exact_on_external_batch,
    run_rt_stpf_exact_on_generated_dataset,
    run_rt_stpf_exact_on_internal_samples,
)
```

## Correctness Contract

The proposal stage is not allowed to reduce correctness:

- Every compact `CandidateRecord` must produce exactly one `ProposalFeatureRow`.
- Every compact `CandidateRecord` must produce exactly one `ExactWorkItem`.
- The scheduled exact interval remains the conservative full interval `[0, 1]`.
- The scheduled feature-family mask is `base_mask | predicted_mask`, never a
  narrower replacement.
- Missing, invalid, OOD, or high-uncertainty proposal outputs route to
  `ProposalSource.FALLBACK`.
- `validate_rt_exact_coverage` rejects dropped candidates, duplicate parent
  candidates, duplicate work items, and missing certificates.

This means an untrained STPF can change priority order and optionally add
families, but it cannot drop uncertified candidates or shrink the certified
time interval.

## Current Scope

This is a correctness-stable Python baseline with an optional OptiX RT Core
candidate backend:

- RT candidate generation uses the deterministic CPU AABB backend by default for
  CI-friendly runs.
- With `p2cccd_cpp` built using `P2CCCD_ENABLE_OPTIX=ON`,
  `RTSTPFExactConfig(rt_backend_name="optix_rt")` uses the real OptiX candidate
  emitter while preserving the same `CandidateRecord` contract.
- External exact certification now supports explicit CUDA routing through
  `RTSTPFExactConfig(enable_cuda_exact=True)`. Internal/generated analytic
  proxy datasets still use the existing swept-sphere oracle path.
- `RTSTPFExact` now only supports learned proposal paths. `use_dummy_policy=True`
  is rejected at config validation time.
- When `p2cccd_cpp` is present, proposal feature extraction plus learned
  exact-queue scheduling are executed through the C++ runtime fast path.
- For the learned ORT route, `RTSTPFExact` now exports runtime proposal rows as
  compiled NumPy arrays and schedules exact work directly from arrays through
  pybind. Python stays on the hot path only for ORT/TensorRT inference itself;
  debug/result artifact materialization runs after timing.
- Learned inference now supports two backends:
  - `inference_backend="torch"` for the existing PyTorch path
  - `inference_backend="ort"` for ONNX Runtime, with TensorRT EP preferred and
    CUDA / CPU fallback
- Supported learned model presets now include:
  - `micro_mlp` (`hidden_dim=32`, `num_layers=1`)
  - `tiny_mlp` (`hidden_dim=64`, `num_layers=1`)
  - `lightweight_mlp` (`hidden_dim=128`, `num_layers=2`)
  - `medium_mlp` (`hidden_dim=256`, `num_layers=4`)
  - `high_capacity_mlp` (`hidden_dim=512`, `num_layers=6`)
- The current architecture sweep over `Thingi10K held-out` and
  `ABC official large held-out` keeps `lightweight_mlp` as the recommended
  default learned preset because it minimizes mean end-to-end latency across
  those two real benchmark families while preserving `FN = 0` and
  `Recall = 1.0`.
- Runtime execution can now be selected through `execution_profile`:
  - `manual`
    - honor the learned `inference_backend` exactly as provided
  - `fastest_learned`
    - force the learned fast path: `optix_rt + ORT(TensorRT EP preferred) + CUDA exact`
  - `auto_fastest`
    - currently resolves to the learned ORT fast path
- Learned STPF benchmark timing now excludes one-time model construction and
  checkpoint loading, but still includes runtime proposal feature extraction,
  inference, and exact-work scheduling.
- Learned ORT benchmark timing now also excludes Python-side
  `ProposalFeatureRow` / `ProposalPrediction` reconstruction used only for
  result packaging.
- For sparse candidate batches, `RTSTPFExact` automatically falls back from
  CUDA inference to CPU inference when the feature-row count is below
  `cpu_inference_row_threshold` (default `2048`). This avoids paying GPU launch
  and initialization overhead on tiny proposal batches.
- `RTSTPFExactResult.resolved_execution_profile_name` records which execution
  profile actually ran, so benchmark reports can distinguish requested policy
  from the final fast path.
- Benchmark timing is exported in the existing `BenchmarkRow` fields:
  `rt_ms`, `proposal_ms`, `exact_ms`, `total_ms`, and `fallback_ratio`.

## Validation

The regression test `test_rt_stpf_exact_baseline.py` covers:

- learned end-to-end candidate-to-certificate coverage,
- real `STPFModel` inference batching on the Scalable CCD sample adapter,
- ORT-backed STPF inference on the Scalable CCD sample adapter,
- OOD fallback without candidate drop,
- generated internal dataset execution,
- missing-model validation,
- backend-name validation.
