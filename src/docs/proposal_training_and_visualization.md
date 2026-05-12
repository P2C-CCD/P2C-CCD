# Proposal Training And Visualization

This note covers TODO 99-100 and TODO 161.

## Proposal Training Modules

The proposal package now has two training layers:

- `proposal/training.py`: core STPF model training, evaluation, NPZ row loading, and loss composition.
- `proposal/training_runner.py`: run-level orchestration for rows or NPZ shards, output directories, history CSV/JSONL, model state export, and summary JSON.

Typical usage:

```python
from p2cccd.proposal import STPFTrainingRunConfig, run_stpf_training_from_npz_shards

run = run_stpf_training_from_npz_shards(
    ["src/outputs/datasets/train_rows.npz"],
    STPFTrainingRunConfig(output_dir="src/outputs/stpf_training", run_name="smoke"),
)
print(run.final_train_loss)
```

The runner remains deterministic when the same row order, seed, config, and PyTorch backend are used.

## Training-Time Metrics

Each `STPFEpochMetrics` row records:

- `interval_top1_recall`;
- `family_top2_recall`;
- `estimated_exact_work_reduction`;
- `mean_predicted_cost`;
- `mean_target_cost`;
- loss, epoch, split, and row count.

`training_runner.py` writes these metrics to `history.csv` and `history.jsonl` when enabled, and the final train/validation values are mirrored in `summary.json`. This is the training-time metric stream used before paper-scale benchmark evaluation.

## Visualization Helpers

The `p2cccd.viz` package provides dependency-free helpers for:

- candidate density by slab,
- exact work source/family summaries,
- certificate trace status summaries,
- small SVG bar charts,
- standalone HTML pipeline debug reports.

Typical usage:

```python
from p2cccd.viz import write_pipeline_debug_html

write_pipeline_debug_html(
    "src/outputs/pipeline_debug.html",
    candidates=result.candidates,
    work_items=result.work_items,
    certificates=result.certificates,
)
```

These helpers are for debugging and paper-figure prototyping. Formal benchmark tables still come from `BenchmarkRowV2` and the benchmark runner/export modules.
