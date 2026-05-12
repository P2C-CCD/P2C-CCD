# Thingi10K Paper Benchmark

This note documents the held-out Thingi10K benchmark used to compare the paper path against baseline variants.

## Scope

Implemented module:

```text
src/python/p2cccd/bench/thingi10k_paper_benchmark.py
```

Benchmark dataset export:

```text
src/datasets/benchmark/ood_stress/thingi10k/thingi10k_paper_benchmark_run_id
```

Benchmark report artifacts:

```text
src/benchmark/thingi10k_paper_benchmark_run_id.md
src/benchmark/thingi10k_paper_benchmark_run_id.json
```

## Method Set

The benchmark executes:

- `PureExactCPU`
- `BVHExact`
- `RTExact`
- `NoProposal`
- `RTSTPFExact-Random`
- `RTSTPFExact-Trained`

It also runs a high-density hard-case workload over the same held-out eval slice.

## Reproduction

```powershell
conda activate cudadev
$env:PYTHONPATH = "src/python"
@'
from pathlib import Path
from p2cccd.bench.thingi10k_paper_benchmark import (
    run_thingi10k_paper_benchmark,
    write_thingi10k_paper_benchmark_report,
    write_thingi10k_paper_benchmark_summary_json,
)

result = run_thingi10k_paper_benchmark()
write_thingi10k_paper_benchmark_report(Path("src/benchmark") / "thingi10k_paper_benchmark_run_id.md", result)
write_thingi10k_paper_benchmark_summary_json(Path("src/benchmark") / "thingi10k_paper_benchmark_run_id.json", result)
'@ | python -
```

## Executed Result

Executed on run_id with the default held-out eval slice:

- asset count: `96`
- pair count: `128`
- query count: `512`
- trained `RTSTPFExact` final `FN = 0`
- trained `RTSTPFExact` candidate recall: `1.0`
- trained `RTSTPFExact total_ms = 46.5899`
- `NoProposal total_ms = 33.3487`
- `RTSTPFExact-Random total_ms = 54.6670`

Current sparse-batch benchmark behavior:

- `RTSTPFExact` now excludes one-time model load from `proposal_ms`.
- Sparse held-out Thingi10K batches automatically use CPU inference instead of
  CUDA inference when the proposal row count is tiny.
- This optimization reduced trained sparse-batch `proposal_ms` from roughly
  `17.34 ms` to `12.36 ms`, and reduced the total trained benchmark time from
  roughly `50.26 ms` to `46.59 ms`.

Hard-case dense result:

- avg candidates/query: `432`
- trained exact-work reduction vs `NoProposal`: `99.9093%`
- trained exact-work reduction vs random STPF: `67.3234%` on exact-work units (`22979.8337 -> 7508.8561`)
- trained `fn_count = 0`

Interpretation:

- On the sparse held-out paper benchmark, Thingi10K currently behaves like the CAD held-out path: correctness is stable, but wall time is still dominated by proposal overhead relative to `NoProposal`.
- On the dense hard-case workload, the trained model materially reduces exact work and clearly outperforms random STPF.
