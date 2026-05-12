# Thingi10K Training

This note documents the official Thingi10K training path added for the P2CCCD OOD and dirty-mesh proposal workbench.

## Scope

Implemented modules:

```text
src/python/p2cccd/datasets/objects/thingi10k_training.py
src/python/p2cccd/bench/thingi10k_training.py
```

Outputs:

```text
src/datasets/thingi10k
src/datasets/training/ood_train/thingi10k/shards/thingi10k_training_run_id
src/outputs/stpf_training/thingi10k_training_run_id
src/benchmark/thingi10k_training_run_id.md
src/benchmark/thingi10k_training_run_id.json
```

## Data Path

1. Materialize an official Thingi10K subset from the `thingi10k` Python package into local OBJ files with JSON sidecars.
2. Rank assets by dirty/OOD metadata and split them into held-out train/eval slices.
3. Generate deterministic proxy motion pairs and analytic oracle labels.
4. Export `base_train`, `base_eval`, `dense_train`, and `dense_eval` shards.
5. Train STPF on mixed base+dense rows.
6. Evaluate `NoProposal`, random STPF, and trained STPF on the dense eval workload.

## Default Run

Run from the repository root:

```powershell
conda activate cudadev
$env:PYTHONPATH = "src/python"
@'
from pathlib import Path
from p2cccd.bench.thingi10k_training import (
    run_thingi10k_training_experiment,
    write_thingi10k_training_report,
    write_thingi10k_training_summary_json,
)

result = run_thingi10k_training_experiment()
write_thingi10k_training_report(Path("src/benchmark") / "thingi10k_training_run_id.md", result)
write_thingi10k_training_summary_json(Path("src/benchmark") / "thingi10k_training_run_id.json", result)
'@ | python -
```

## Executed Result

The executed full run on run_id produced:

- asset count: `96`
- train pair count: `320`
- eval pair count: `128`
- base train queries: `1280`
- base eval queries: `512`
- dense eval rows: `221184`
- dense avg candidates/query: `432`
- trained exact-work reduction vs `NoProposal`: `99.9093%`
- trained exact-work reduction vs random STPF: `70.0961%`
- trained `fn_count = 0`

The report and JSON summary in `src/benchmark` are the canonical artifacts for this run.

