# Benchmark Suites And Profiling

This note ties TODO 153-160 to runnable files.

## Suite Configs

- Correctness: `configs/benchmark_suites/correctness.json` and `correctness_external.json`.
- Performance: `configs/benchmark_suites/performance.json`.
- Ablation: `configs/benchmark_suites/ablation.json`.
- OOD/stress: `configs/benchmark_suites/ood_stress.json`.

The external correctness suite supports Scalable CCD sample and Rigid-IPC query batches when the local baseline data is present. Tight Inclusion is represented as a reference-availability gate because it is a conservative implementation reference rather than a query dataset.

## BenchmarkRowV2 Metrics

Rows include:

- family-wise exact calls: point-triangle, edge-edge, conservative, unknown, total;
- candidate/proposal health: candidate inflation ratio, undecided-to-resolved ratio, fallback ratio, exact queue occupancy;
- latency: min, p50, p90, p95, p99, max;
- RT split: build, update, trace;
- NoQueueDecouple fields: candidate buffer bandwidth, proposal enqueue/dequeue time, total tail latency.

## Profiling Hooks

Use `BenchmarkProfiler` for CPU-side staged timing:

```python
from p2cccd.bench import BenchmarkProfiler

profiler = BenchmarkProfiler()
with profiler.stage("candidate_generation"):
    ...
with profiler.stage("exact_certificates"):
    ...
summary = profiler.summary()
```

GPU timeline capture remains external to the Python runner. Use Nsight Systems around the same CLI:

```powershell
nsys profile -o p2cccd_suite `
  python -m p2cccd.bench.run_suite --config src/configs/benchmark_suites/performance.json
```

Nsight Compute should be attached only to OptiX/CUDA device kernels such as the `optix_rt` candidate emitter and CUDA exact batches; CPU smoke suites should not require it.

## NoQueueDecouple Outputs

`NoQueueDecoupleCaseResult` reports:

- `candidates_per_sec`;
- `approx_bandwidth_mb_s`;
- `trace_ms`;
- `proposal_enqueue_dequeue_ms`;
- `total_tail_latency_ms`.

The suite runner forwards these to `BenchmarkRowV2` so CSV/JSONL exports can be used directly for queue-decoupling tables.
