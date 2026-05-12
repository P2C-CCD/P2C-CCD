# Benchmarks And Baselines Audit

This document is the repository-level audit map for TODO tasks 79-93. It separates correctness baselines, ablation baselines, style reproductions, and downstream robotics comparisons so benchmark numbers are not misinterpreted.

## Coverage Matrix

| TODO | Method or scope | Module | Test | Primary role | Current boundary |
| --- | --- | --- | --- | --- | --- |
| 79 | `PureExactCPU` | `python/p2cccd/bench/pure_exact_cpu.py` | `tests/python/test_pure_exact_cpu_baseline.py` | correctness reference | Uses internal swept-sphere analytic oracle and external adapter labels until C++ exact replay is exposed to Python. |
| 80 | `BVHExact` | `python/p2cccd/bench/bvh_exact.py` | `tests/python/test_bvh_exact_baseline.py` | CPU broad-phase plus exact baseline | Uses deterministic CPU AABB broad phase with Embree/Coal-compatible naming; not an Embree/Coal backend yet. |
| 81 | `RTExact` | `python/p2cccd/bench/rt_exact.py` | `tests/python/test_rt_exact_baseline.py` | RT-candidate plus exact baseline without STPF | Uses conservative candidate generation and timing split; `optix_rt` can route through the C++ OptiX `RawCandidateHit` device emitter. |
| 82 | `RTSTPFExact` | `python/p2cccd/bench/rt_stpf_exact.py` | `tests/python/test_rt_stpf_exact_baseline.py` | full RT plus proposal plus exact pipeline | Learned-only proposal inference; correctness relies on monotonic scheduling and exact coverage. |
| 83 | `NoProposal` | `python/p2cccd/bench/no_proposal.py` | `tests/python/test_no_proposal_ablation.py` | safety-preserving proposal ablation | Routes all candidates to fallback exact work; intended to measure proposal overhead and scheduling benefit. |
| 84 | `SortBroadPhaseExact` | `python/p2cccd/bench/sort_broad_phase_exact.py` | `tests/python/test_sort_broad_phase_exact.py` | sort/sweep broad-phase comparison | CPU implementation with GPU-compatible counters; not a GPU radix-sort implementation yet. |
| 85 | `IntervalOnly`, `RankingOnly` | `python/p2cccd/bench/stpf_head_ablations.py` | `tests/python/test_stpf_head_ablations.py` | STPF head isolation | Keeps monotonic no-drop scheduling; isolates interval and ranking heads from full STPF scheduling. |
| 86 | `NoQueueDecouple` | `python/p2cccd/bench/no_queue_decouple.py` | `tests/python/test_no_queue_decouple_microbenchmark.py` | queue-decoupling microbenchmark | Synthetic candidate stream microbenchmark; does not represent end-to-end CCD correctness. |
| 87 | `PatchGranularityAblation` | `python/p2cccd/bench/patch_granularity_ablation.py` | `tests/python/test_patch_granularity_ablation.py` | patch granularity workbench | CPU analytic workbench; final tuning must be repeated on benchmark scenes and OptiX backend. |
| 88 | `SlabProxyAblation` | `python/p2cccd/bench/slab_proxy_ablation.py` | `tests/python/test_slab_proxy_ablation.py` | slab count and proxy family workbench | Conservative Python workbench; real performance ranking depends on production RT and exact backends. |
| 89 | `BenchmarkRunMeta`, `BenchmarkRowV2` | `python/p2cccd/bench/summary.py` | `tests/python/test_benchmark_export_v2.py` | benchmark export contract | CSV, JSONL, and `run_meta.json` are implemented for current Python runners. |
| 90 | suite runner and bundled configs | `python/p2cccd/bench/suite_runner.py` | `tests/python/test_benchmark_suite_runner.py` | reproducible benchmark execution | Supports internal generated datasets; external CCD suite kinds remain future adapter work. |
| 91 | `RTDCDStyle`, `RTCCDStyle` | `python/p2cccd/bench/rt_style_reproduction.py` | `tests/python/test_rt_style_reproduction.py` | RT-DCD/RT-CCD style comparison | Style reproduction only; not an official reproduction of any third-party implementation. |
| 92 | `NeuralSVCDStyle`, `CabiNetStyle` | `python/p2cccd/bench/learned_style_comparison.py` | `tests/python/test_learned_style_comparison.py` | learned-surrogate style comparison | Deterministic surrogate baselines; not trained or official NeuralSVCD/CabiNet integrations. |
| 93 | `CuRoboDownstream` | `python/p2cccd/bench/curobo_downstream.py` | `tests/python/test_curobo_downstream.py` | downstream robot link-pair comparison | Discrete trajectory checker; not certified continuous CCD and not official cuRobo integration. |

## Safety Interpretation

Correctness baselines are `PureExactCPU`, `BVHExact`, `RTExact`, and `RTSTPFExact`. Their benchmark rows should be interpreted with `fn_count`, `candidate_recall`, and `final_fn_zero`.

Ablation and style baselines are not allowed to hide false negatives. A method may be intentionally non-certified, but its benchmark rows must still report `fn_count` and `candidate_recall` against the oracle.

Style names are intentionally suffixed with `Style` or `Downstream` when the implementation is not an official third-party integration. This keeps paper-facing comparisons honest.

## Runner Entry Points

From `src`:

```powershell
conda activate cudadev
$env:PYTHONPATH = "python"
python -m p2cccd.bench.run_suite --config configs\benchmark_suites\correctness.json
python -m p2cccd.bench.run_suite --config configs\benchmark_suites\ablation.json
python -m p2cccd.bench.run_suite --config configs\benchmark_suites\rt_style_reproduction.json
python -m p2cccd.bench.run_suite --config configs\benchmark_suites\learned_style_comparison.json
python -m p2cccd.bench.run_suite --config configs\benchmark_suites\curobo_downstream.json
```

## Known Upgrade Targets

- Replace analytic Python stand-ins with pybind-exposed C++ exact replay where appropriate.
- Scale the new `optix_rt` device hit emission beyond smoke scenes and add TLAS/BLAS update-oriented performance runs.
- Add Embree or Coal-backed broad-phase adapters behind the existing `BVHExact` abstraction.
- Add official third-party integrations only when licenses, source manifests, and reproducibility metadata are present.
