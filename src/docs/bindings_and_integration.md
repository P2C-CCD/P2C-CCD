# Bindings And Integration

This note covers TODO 101-106.

The `p2cccd_cpp` module is the CPU-stable pybind11 bridge between the C++ core and the Python workbench. It intentionally exposes low-level contracts and CPU execution APIs first. High-level dataset adapters can build on these bindings without changing the C++ ABI.

## Exposed Contracts

The module exposes the stable runtime records from `cpp/common/runtime_contracts.h`:

- `CandidateRecord`
- `ExactWorkItem`
- `CertificateResult`
- `AuditLogRow`
- `BenchmarkRow`

It also exposes matching enums for proxy type, proposal source, certificate status, refinement mode, and audit stage.

Each core record has a `validate_*` function that calls the C++ validator and raises `ValueError` on contract violations. This keeps Python-side regression replay aligned with the C++ schema.

## Candidate API

Candidate generation is exposed at the proxy-scene level:

- `build_proxy_scene(input)`
- `validate_proxy_scene(scene)`
- `generate_raw_candidates_cpu(scene, query_id=0)`
- `compact_raw_candidates(scene, raw_buffer)`
- `generate_candidates_for_proxy_scene(scene, query_id=0, backend_name="cpu_reference", allow_optix_cpu_fallback=False)`

`query_id=0` means "use `scene.query_id`". The OptiX backend is exposed through `backend_name="optix"` when `p2cccd_cpp` is built with `P2CCCD_ENABLE_OPTIX=ON`; that path emits `RawCandidateHit` records on device and returns compact `CandidateRecord` rows with backend name `optix_rt`.

## Exact Certificate API

The CPU exact oracle is exposed through:

- `evaluate_point_triangle_interval(primitive, interval_t0, interval_t1, config)`
- `evaluate_edge_edge_interval(primitive, interval_t0, interval_t1, config)`
- `evaluate_certificate_query_cpu(query)`
- `process_exact_work_queue_cpu(work_queue, config)`
- `validate_exact_work_queue_coverage(work_queue, result)`
- `generate_conservative_refinement_work_items(parent, certificate, config)`

The queue processor emits `ExactWorkQueueResult.certificates` and `ExactWorkQueueResult.audit_log`, so regression tests can verify no work item disappears without certificate coverage.

## CUDA-Aware Exact API

TODO 106 is implemented as a host-owned CUDA-aware binding boundary, not a raw device-pointer ABI. The exposed functions are:

- `is_cuda_exact_built()`
- `cuda_binding_status()`
- `evaluate_point_triangle_batch_cuda(primitives, interval_t0, interval_t1, config)`
- `evaluate_edge_edge_batch_cuda(primitives, interval_t0, interval_t1, config)`
- `cross_check_cpu_cuda_exact(point_triangles, edge_edges, interval_t0, interval_t1, config, eps_cert)`

The Python convenience module `p2cccd.cuda_bindings` wraps those functions and reports:

- whether `p2cccd_cpp` is importable;
- whether the host batch exact API exists;
- whether the CUDA exact backend was actually built;
- whether raw device-pointer ABI is enabled.

The current safety policy is explicit: `CUDA_DEVICE_POINTER_ABI_ENABLED = False`. Python callers pass host-owned primitive batches and receive host-owned `PrimitiveIntervalResult` rows. If `P2CCCD_ENABLE_CUDA=OFF`, the same entrypoints are present through the C++ stub and fail fast with a clear "CUDA exact backend was not built" error instead of silently falling back inside the binding.

## Audit Replay

Audit rows are exposed as first-class Python objects via `AuditLogRow`.

For replay-oriented regression tests, the binding provides:

- `validate_audit_log_rows(rows)`
- `audit_log_rows_for_query(rows, query_id)`

These helpers are intentionally minimal. They validate and filter C++ audit rows without imposing a Python replay policy.

## Build

Configure Python bindings with:

```powershell
conda activate cudadev
cmake -S src -B src\build -DP2CCCD_ENABLE_PYTHON=ON
cmake --build src\build --config Release --target p2cccd_cpp
```

If an installed `pybind11` package is unavailable, CMake falls back to the vendored `src/lib/pybind11` source tree.

To build the optional CUDA-aware exact backend:

```powershell
conda activate cudadev
cmake -S src -B src\build_cuda_py -DP2CCCD_ENABLE_PYTHON=ON -DP2CCCD_ENABLE_CUDA=ON
cmake --build src\build_cuda_py --config Release --target p2cccd_cpp
```

CPU-only builds still expose the CUDA-aware status API and stub entrypoints, so CI can validate the Python ABI without requiring CUDA runtime availability.
