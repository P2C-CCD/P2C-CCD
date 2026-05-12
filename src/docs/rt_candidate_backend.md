# RT Candidate Backend

This note documents the current `P2CCCD` RT candidate layer after the CPU
reference backend and the first real OptiX RT Core candidate emitter.

## Current Backends

### CPU Reference

`CpuReferenceCandidateGenerator` is the correctness/debug backend. It consumes a
validated `ProxyScene`, runs the same AABB/time-slab overlap logic used by the
raw candidate buffer, compacts duplicate raw hits into stable
`CandidateRecord` rows, and returns a `CandidateGenerationResult`.

`ProxyScene` currently requires a global non-overlapping slab partition:
all primitives with the same `slab_id` must share the same `[t0, t1]`, and
different slab ids must not overlap in time. This is intentional because
`CandidateRecord` stores one `slab_id` rather than an arbitrary time interval.
Non-aligned per-object slab grids are rejected instead of silently dropping
time-overlapping candidates.

The result contains:

- `raw_buffer`: un-compacted `RawCandidateHit` records.
- `candidates`: compact candidate records for exact/certificate stages.
- `timing`: `build_ms`, `update_ms`, `trace_ms`, `compact_ms`, `stats_ms`,
  and `total_ms`.
- `density`: STPF-oriented candidate density statistics.

For the CPU backend, `build_ms` currently measures scene validation and backend
setup. `update_ms` remains zero in the reference path because there is no
separate dynamic acceleration-structure update stage.

### OptiX RT Core Backend

`OptixCandidateTracer` is an optional real RT backend. The default build does
not require CUDA or OptiX, but an OptiX build emits `RawCandidateHit` records on
the device and then reuses the same compaction path as the CPU backend.
Configure with:

```powershell
cmake -S src -B src\build_optix -DP2CCCD_ENABLE_OPTIX=ON -DP2CCCD_ENABLE_PYTHON=ON
cmake --build src\build_optix --config Release --target p2cccd_cpp
```

The CMake gate follows the minimal pattern extracted from `Code/RTTest`:

- Enabling OptiX also enables CUDA.
- `find_package(CUDAToolkit REQUIRED)` provides `CUDA::cuda_driver` and
  `CUDA::cudart`.
- `P2CCCD_OPTIX_ROOT` should point to the local OptiX SDK root when OptiX is
  enabled.
- `P2CCCD_OPTIX_INCLUDE_DIR` must contain `optix.h`.
- `p2cccd_core` receives `P2CCCD_HAS_OPTIX=1`, the OptiX include directory,
  CUDA driver/runtime links, and `P2CCCD_OPTIX_CANDIDATE_PTX_PATH`.
- CMake compiles `cpp/rt_candidate/optix_candidate_program.cu` to PTX.

Runtime behavior:

- conservative proxy AABBs are encoded as OptiX custom primitives;
- first-time GAS builds are reported in `build_ms`;
- steady-state in-place GAS updates are reported in `update_ms` when the proxy
  primitive count matches the cached traversable;
- the raygen program launches one conservative x-axis query per proxy;
- GAS AABBs are inflated in transverse axes by the maximum query half extent so
  AABB overlap recall remains conservative;
- the intersection program performs the final exact proxy-AABB overlap, slab,
  object, and ordering filters;
- valid overlaps are written directly as 16-byte-aligned `RawCandidateHit`
  records using an atomic append;
- host compaction produces stable `CandidateRecord` rows.
- OptiX context/module/program groups/pipeline/SBT/stream are cached per
  process after the first successful initialization instead of being rebuilt on
  every candidate-generation call;
- reusable device buffers for proxies, AABBs, hit buffers, launch params, and
  GAS scratch/output storage are grown geometrically and reused across calls to
  reduce `cudaMalloc/cudaFree` churn on repeated scenes or benchmark cases.
- the Python `CppOptixBroadPhaseBackend` now batches many same-family
  `same_query_only` proxy groups into one packed OptiX scene instead of
  launching one tiny OptiX scene per query; batched groups are separated by
  unique slab ids, non-overlapping slab time intervals, and x-axis packing so
  they can be mapped back to original query ids without cross-query candidates.
- external CCD query batches in `rt_exact.py` can now bypass the Python
  `ExternalCCDQuery -> BroadPhasePrimitive` object-construction loop and call a
  C++ `generate_candidates_for_external_batch(...)` fast path through pybind;
  this keeps the external query encoding, batched proxy-scene construction,
  candidate tracing, and candidate remapping in C++ and materially reduces
  external `rt_build_ms`.

The backend name is `optix_rt`. If OptiX is unavailable and
`allow_cpu_fallback=true`, the tracer returns `optix_cpu_fallback` and preserves
the no-false-negative CPU reference behavior.

Current boundary:

- the current OptiX backend is conservative and correctness-oriented and still
  uses a single-GAS proxy scene rather than a full TLAS/BLAS hierarchy;
- repeated scenes with unchanged proxy counts now use the OptiX
  `OPTIX_BUILD_OPERATION_UPDATE` path instead of always rebuilding GAS from
  scratch;
- scenes that change primitive count or outgrow the cached buffers still fall
  back to a full rebuild, which is reflected in `build_ms`;
- the first call on a larger scene may still pay one-time device-buffer growth
  cost before later steady-state calls reuse the same allocations;
- Python `RTExact` / `RTSTPFExact` can select it through
  `backend_name="optix_rt"` or `rt_backend_name="optix_rt"` when the OptiX
  pybind module is discoverable;
- CPU-only CI remains valid because the default backend is unchanged.

Smoke command:

```powershell
conda activate cudadev
$env:PYTHONPATH = "src/python"
$env:CUDA_PATH = "<path-to-your-cuda-toolkit>"
python -c "from p2cccd.bench.suite_runner import run_benchmark_suite_from_config_path; r=run_benchmark_suite_from_config_path('src/configs/benchmark_suites/performance_optix.json', run_id='manual_optix_smoke'); print([(c.case_name, c.final_fn_zero) for c in r.case_results])"
```

## Candidate Density Export

`WriteCandidateDensityCsv` and `WriteCandidateDensityJsonl` export stable
STPF-training rows. The schema version is stored per row and the CSV header is
returned by `CandidateDensityCsvHeader()`.

The exported fields are:

```text
schema_version,query_id,proxy_count,object_count,slab_count,
cross_object_same_slab_pair_count,raw_hit_count,compact_candidate_count,
raw_hits_per_proxy,candidates_per_proxy,candidates_per_slab,
aabb_overlap_ratio,avg_rt_hits_per_candidate,build_ms,update_ms,
trace_ms,compact_ms,stats_ms,total_ms,backend_name
```

These rows are intended for Phase 2 STPF feature/dataset code and for early
performance smoke tests.
