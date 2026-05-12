# Python Wrappers And Benchmark Runner Utilities

This note covers TODO 95-97.

## Candidate Generation Wrapper

`p2cccd.candidate_generation` provides a stable Python entry point for candidate generation:

- `generate_candidates_for_internal_samples`
- `generate_candidates_for_generated_dataset`
- `generate_candidates_for_external_batch`

The wrapper detects whether a high-level dataset-oriented `p2cccd_cpp` candidate-generation entry point exists. The current pybind module exposes lower-level proxy-scene APIs, so these dataset wrappers still use the Python CPU broad-phase fallback and record `used_cpp_backend=False` plus a fallback reason. If a caller sets `prefer_cpp_backend=True` and `allow_python_fallback=False`, the wrapper fails fast.

## Certificate Engine Wrapper

`p2cccd.certificate_engine` provides execution wrappers for exact certificate processing:

- `execute_certificate_engine_for_internal_samples`
- `execute_certificate_engine_for_generated_dataset`
- `execute_certificate_engine_for_external_batch`

These wrappers consume candidate-generation results, build a no-STPF exact work queue, run the existing CPU exact fallback, validate one-certificate-per-work-item coverage, and return certificates plus audit rows. As with candidate generation, C++ binding availability is detected but not assumed.

## Benchmark Runner Utilities

Additional `python/p2cccd/bench/` modules support benchmark operation:

- `io.py`: read/write complete `BenchmarkRunMeta` plus `BenchmarkRowV2` runs and discover run directories.
- `suite_configs.py`: discover bundled suite configs and load them by name.
- `profiling.py`: lightweight stage profiler for benchmark runner instrumentation.
- `paper_tables.py`: convert `BenchmarkRowV2` rows to compact paper-table rows and Markdown tables.

These utilities complement the existing suite runner and export path. They do not replace the production benchmark contract; they make it easier to reuse the current benchmark rows from scripts, tests, and paper-table generation.
