# Benchmark Export V2

Benchmark export now has two layers:

- `BenchmarkRunMeta`: one JSON object per run, written to `run_meta.json`.
- `BenchmarkRowV2`: one row per method/scene/config result, written to both CSV and JSONL.

The old `BenchmarkRow` remains valid for internal runners. Use `benchmark_row_v2_from_legacy(...)` to adapt existing baseline outputs into the V2 export format.

## Required Files

`export_benchmark_run(run_dir, meta, rows)` writes:

```text
run_dir/
  benchmark.csv
  benchmark.jsonl
  run_meta.json
```

`benchmark.csv` is convenient for tables. `benchmark.jsonl` is the stable machine-readable row stream. `run_meta.json` stores run-level metadata that should not be repeated in every row.

## Run Metadata

`BenchmarkRunMeta` records:

- dataset, scene, method, seed, run id, and config hash;
- canonical JSON config payload;
- git commit, host, platform, Python version;
- GPU name, driver version, CUDA version, OptiX version;
- VRAM total/free in MB;
- output file names and notes.

Environment values are auto-detected from `nvidia-smi`, `nvcc`, `CUDA_PATH`, and `OPTIX_VERSION` when available. Tests and scripted experiments can pass an explicit environment dictionary for reproducibility.

## Row V2 Metrics

`BenchmarkRowV2` extends the legacy row with:

- RT build/update/trace split: `rt_build_ms`, `rt_update_ms`, `rt_trace_ms`;
- latency percentiles: min, p50, p90, p95, p99, max;
- family-wise exact calls: point-triangle, edge-edge, conservative, unknown;
- `exact_calls_total` consistency check;
- candidate/proposal health metrics: `candidate_inflation_ratio`, `undecided_to_resolved_ratio`, and `exact_queue_occupancy`;
- NoQueueDecouple/profiling metrics: `candidate_buffer_bandwidth_mb_s`, `proposal_enqueue_dequeue_ms`, and `total_tail_latency_ms`;
- `vram_peak_mb`.

The V2 validator enforces monotonic latency percentiles, non-negative profiling metrics, and requires the family-wise exact-call fields to sum to `exact_calls_total`.

## Minimal Usage

```python
from p2cccd.bench import (
    benchmark_row_v2_from_legacy,
    create_benchmark_run_meta,
    export_benchmark_run,
)

meta = create_benchmark_run_meta(
    dataset_name="Sample-Scalable-CCD-Data",
    scene_name="cloth-funnel:227ee",
    method_name="RTSTPFExact",
    config={"slabs": 4, "proposal": "dummy"},
    seed=13,
)

row_v2 = benchmark_row_v2_from_legacy(
    legacy_row,
    meta,
    latency_samples_ms=query_latencies,
    family_exact_calls={"edge_edge": 128, "point_triangle": 96},
    candidate_inflation_ratio=1.4,
    exact_queue_occupancy=0.92,
    rt_build_ms=rt_timing.build_ms,
    rt_update_ms=rt_timing.update_ms,
    rt_trace_ms=rt_timing.trace_ms,
    vram_peak_mb=peak_vram,
)

export_benchmark_run("outputs/runs/run001", meta, [row_v2])
```

## Config Hash

`benchmark_config_hash(config)` uses canonical JSON with sorted keys and SHA-256. The same semantic config dictionary gives the same hash independent of insertion order.

## Current Scope

This is a Python benchmark export format. It does not replace the C++ runtime buffer contracts. Existing runners can continue returning `BenchmarkRow` until they are upgraded to produce per-query latencies, exact family counters, and real GPU memory measurements directly.
