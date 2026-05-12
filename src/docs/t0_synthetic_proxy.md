# T0 Synthetic Proxy

descriptionrecord `T0 synthetic_proxy` descriptionanddescription. Objectiveisdescription, description layerdescriptionfixedunderdescription, descriptionandafterdescription external benchmark descriptionsplitdescription.

## descriptionuse

`T0 synthetic_proxy` descriptionanddescription, descriptionasdescription external correctness
benchmark.

descriptionsplitdescription:

1. `base synthetic_proxy`
   - descriptionconnectfrom `python/p2cccd/data/`  analytic swept-sphere proxy description.
   - each query description `ProposalFeatureRow`.
2. `dense synthetic_proxy`
   - based onsame batch base sample, description `slab x patch x patch` augmentationdescriptiontohigh candidate density workload.
   - used fordescriptionanddescription STPF inhigh candidate-density underisdescriptionreduction exact work.

## Code entry point

- description/description runner:
  [t0_synthetic_proxy.py](../python/p2cccd/bench/t0_synthetic_proxy.py)
- dense workload augmentation:
  [trained_stpf_high_density.py](../python/p2cccd/bench/trained_stpf_high_density.py)
- analytic base dataset:
  [dataset.py](../python/p2cccd/data/dataset.py)
- NPZ shard:
  [shards.py](../python/p2cccd/data/shards.py)

## Output

defaultOutputsplitdescription:

1. training shard
   - `src/datasets/training/synthetic_proxy/shards/<run_name>/`
   - contains `base_train.npz`, `base_eval.npz`, `dense_train.npz`, `dense_eval.npz`
2. model artifacts
   - `src/outputs/stpf_training/<run_name>/`
   - contains `history.csv`, `history.jsonl`, `model_state.pt`, `summary.json`
3. benchmark report
   - `src/benchmark/<run_name>_report.md`
   - `src/benchmark/<run_name>_summary.json`

## defaultdescriptionProtocol

description `t0_synthetic_proxy_20260422_main` descriptionuse:

- train mesh count per split: `384`
- train robot link count: `192`
- eval mesh count per split: `128`
- eval robot link count: `64`
- slab count: `8`
- patches per object: `4`
- dense avg candidates/query: `128`
- train device: `cuda`
- benchmark device: `cuda`

descriptiondefaultwritedescription:

- report: `src/benchmark/t0_synthetic_proxy_run_id.md`
- summary: `src/benchmark/t0_synthetic_proxy_run_id.json`

description public release descriptionleveldescription; descriptionunderdescriptionafterdescriptionnewgenerate.

## description

```python
from p2cccd.bench.t0_synthetic_proxy import (
    T0SyntheticProxyExperimentConfig,
    run_t0_synthetic_proxy_experiment,
    write_t0_synthetic_proxy_report,
    write_t0_synthetic_proxy_summary_json,
)
from p2cccd.bench.trained_stpf_high_density import HighDensitySTPFConfig
from p2cccd.proposal.training import STPFTrainingConfig

cfg = T0SyntheticProxyExperimentConfig(
    train_mesh_count_per_split=384,
    train_robot_link_count=192,
    eval_mesh_count_per_split=128,
    eval_robot_link_count=64,
    seed=42,
    high_density=HighDensitySTPFConfig(slab_count=8, patches_per_object=4),
    training=STPFTrainingConfig(
        epochs=8,
        batch_size=2048,
        learning_rate=1.0e-3,
        seed=42,
        device="cuda",
        validation_fraction=0.0,
    ),
    run_name="t0_synthetic_proxy_20260422_main",
    benchmark_device="cuda",
)

result = run_t0_synthetic_proxy_experiment(cfg)
write_t0_synthetic_proxy_report("src/benchmark/t0_synthetic_proxy_run_id.md", result)
write_t0_synthetic_proxy_summary_json("src/benchmark/t0_synthetic_proxy_run_id.json", result)
```

## description

this benchmarkdescriptionMetricsis notdescription wall time, instead:

- `interval_top1_recall`
- `family_top2_recall`
- `trained_exact_work_reduction_vs_no_proposal`
- `trained_exact_work_reduction_vs_random`
- `fn_count`

descriptionconnect: `T0 synthetic_proxy` Objectiveisdescription STPF indescription synthetic proxy
distributionondescriptiontodescription proposal description, rather thanindescriptionreplacedescription external CCD benchmark.
