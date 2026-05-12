# NeuralSVCD And CabiNet Style Comparisons

This module adds learned-method style comparisons for supplementary experiments. It does not claim source-level reproduction of NeuralSVCD or CabiNet.

## NeuralSVCDStyle

`NeuralSVCDStyle` behaves like a neural continuous collision surrogate:

- samples multiple times along the motion interval;
- converts the closest sampled signed margin into a collision score;
- uses uncertainty fallback to route ambiguous/OOD queries into exact replay;
- reports missed positives as `fn_count`.

The default config uses `time_sample_count = 9` and conservative fallback. It is intended to approximate a learned continuous surrogate while preserving the safety checks used by P2CCCD.

## CabiNetStyle

`CabiNetStyle` behaves like a pose-level neural collision predictor:

- samples object poses, by default only start/end poses;
- predicts a collision score from learned-proxy overlap margin;
- optionally routes high-uncertainty queries to exact replay;
- can miss continuous tunneling when only endpoint poses are considered.

This matches the role noted in the research plan: CabiNet-style comparisons are useful learned-collision baselines, but should not be presented as certified continuous CCD.

## Runner

```python
from p2cccd.bench import (
    CabiNetStyleConfig,
    NeuralSVCDStyleConfig,
    run_cabinet_style_on_generated_dataset,
    run_neural_svcd_style_on_generated_dataset,
)
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset

dataset = generate_exact_oracle_dataset(DatasetGenerationConfig(mesh_count_per_split=4))
neural = run_neural_svcd_style_on_generated_dataset(dataset, NeuralSVCDStyleConfig())
cabinet = run_cabinet_style_on_generated_dataset(dataset, CabiNetStyleConfig())
```

The suite config is:

```text
src/configs/benchmark_suites/learned_style_comparison.json
```

## Metrics

Both styles return the same benchmark contract as other baselines:

- `candidate_recall`
- `fn_count`
- `fp_count`
- `avg_candidates`
- `avg_exact_evals`
- `proposal_ms`
- `exact_ms`

Additional `LearnedStyleStats` fields record surrogate positives, uncertainty fallback count, OOD fallback count, omitted queries, average collision score, and average uncertainty.

## Limitations

These comparisons use deterministic analytic surrogates instead of trained neural weights. They are suitable for pipeline integration, table schema validation, and safety-ablation behavior. A paper-quality comparison should later replace the deterministic surrogate with the actual released model or a trained in-repo model and keep the same `BenchmarkRowV2` export path.
