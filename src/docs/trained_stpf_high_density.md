# Trained STPF High-Density Workbench

This note documents the high-candidate-density STPF workbench added to support
real STPF training and an exact-work reduction experiment.

## Scope

This path is intentionally a workbench, not a replacement for the main
`RTExact` / `RTSTPFExact` external CCD benchmark suites.

It is used to answer a narrower question:

- can a truly trained STPF checkpoint reduce exact work on a workload with
  many conservative candidates per query?

The workbench keeps the existing P2CCCD proposal model and training runner, but
builds a denser synthetic candidate field per analytic motion sample.

## What It Generates

For each generated motion sample, the workbench expands one query into:

- `slab_count` temporal slabs;
- `patches_per_object^2` patch-pair variants;
- one `CandidateRecord` and one `ProposalFeatureRow` per slab/patch variant.

The current default configuration used in the benchmark report is:

- `slab_count = 8`
- `patches_per_object = 4`

That yields:

- `128` candidates per query.

## Labels

Each generated candidate row stores:

- interval targets from the analytic contact/reference time;
- family targets from the sample family;
- priority targets that strongly favor slab/patch candidates aligned with the
  contact interval;
- normalized cost targets;
- uncertainty targets that distinguish likely representatives from obvious
  misses.

The row schema stays compatible with the existing `ProposalFeatureRow`.

## Training

Training reuses the standard Python STPF pipeline:

- `p2cccd.proposal.train_stpf_model`
- `p2cccd.proposal.run_stpf_training`

The workbench writes a normal checkpoint:

- `model_state.pt`
- `history.csv`
- `history.jsonl`
- `summary.json`

under `src/outputs/stpf_training/...`.

## Benchmark Semantics

The benchmark compares three methods:

- `NoProposal`
- `RTSTPFExact-Random`
- `RTSTPFExact-Trained`

The key output metric is:

- `exact_work_units`

rather than raw exact wall time.

This is deliberate. The workbench is meant to isolate whether the trained STPF
reduces exact workload under high candidate density, before claiming any final
runtime speedup on the main external CCD benchmark.

## Interpretation

This benchmark should be read as:

- proof that a trained STPF checkpoint can reduce exact work on a dense
  conservative-candidate workload;
- a bridge between the current correctness-first RT pipeline and a future
  proposal-sensitive exact runtime.

It should not be overstated as:

- the final OptiX + learned STPF paper benchmark;
- the definitive runtime comparison against external CCD baselines.

## Reproduction Sketch

```python
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset
from p2cccd.bench.trained_stpf_high_density import (
    HighDensitySTPFConfig,
    run_trained_stpf_high_density_experiment,
    write_trained_stpf_high_density_report,
)

train_dataset = generate_exact_oracle_dataset(
    DatasetGenerationConfig(mesh_count_per_split=32, robot_link_count=0, seed=401, include_robot_links=False)
)
eval_dataset = generate_exact_oracle_dataset(
    DatasetGenerationConfig(mesh_count_per_split=20, robot_link_count=0, seed=402, include_robot_links=False)
)
result = run_trained_stpf_high_density_experiment(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    config=HighDensitySTPFConfig(slab_count=8, patches_per_object=4, representative_attempt_limit=3),
    training_output_dir="src/outputs/stpf_training",
    run_name="trained_stpf_high_density_20260421_final",
    training_device="cuda",
    benchmark_device="cuda",
    epochs=25,
    batch_size=1024,
    learning_rate=8.0e-4,
    seed=4242,
)
write_trained_stpf_high_density_report(
    "src/benchmark/trained_stpf_high_density_report_run_id.md",
    result,
)
```
