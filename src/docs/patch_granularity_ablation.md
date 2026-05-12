# Patch Granularity Ablation

`PatchGranularityAblation` evaluates how different patch counts per object affect the conservative candidate stage before exact certification.

The current Python workbench path uses the internal analytic swept-sphere samples because the C++ patch builder is not exposed through pybind yet. Each original query is expanded into multiple synthetic subpatch proxies per object, then routed through the same CPU AABB broad-phase abstraction used by `BVHExact`. Exact labels still come from the analytic oracle, so the ablation reports whether a granularity option preserved `candidate_recall = 1.0` and `fn_count = 0`.

The runner intentionally requires `same_query_only = True` because the current analytic oracle is query-local. Scene-level cross-query patch pairing should be added only after the C++ patch builder and exact replay are exposed to Python.

## Runner

```python
from p2cccd.bench import (
    PatchGranularityAblationConfig,
    PatchGranularityAblationOption,
    run_patch_granularity_ablation_on_generated_dataset,
)
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset

dataset = generate_exact_oracle_dataset(DatasetGenerationConfig(mesh_count_per_split=8))
result = run_patch_granularity_ablation_on_generated_dataset(
    dataset,
    PatchGranularityAblationConfig(
        options=(
            PatchGranularityAblationOption("coarse", 1),
            PatchGranularityAblationOption("medium", 2),
            PatchGranularityAblationOption("fine", 4),
        )
    ),
)
```

The selected row is the feasible option with the lowest weighted score:

```text
score = candidate_weight * compact_candidate_count
      + raw_hit_weight * raw_hit_count
      + proxy_weight * proxy_count
      + radius_weight * avg_patch_radius
```

## Safety Rule

An option is feasible only if:

- all positive oracle queries are covered by at least one candidate;
- `candidate_recall >= min_candidate_recall`;
- `fn_count == 0`;
- `fp_count == 0`.

The default options keep `radius_scale = 1.0`, so subpatch proxies remain conservative for the analytic samples. Smaller `radius_scale` values are allowed for ablation, but unsafe rows are marked infeasible and will not be selected.

## Export

Use `patch_granularity_ablation_rows_to_csv(result)` for stable CSV rows. Important columns:

- `patches_per_object`: number of generated subpatch proxies per object.
- `proxy_count`: total generated proxy primitives.
- `raw_hit_count`: raw broad-phase pair count before compacting by query id.
- `compact_candidate_count`: unique query ids forwarded to exact certification.
- `candidate_recall`, `fn_count`, `fp_count`: correctness gates.
- `score`: weighted selection objective.

## Limitations

This is a workbench-level ablation, not the final RT Core patch builder. The production path should later replace synthetic subpatch expansion with the C++ BVH/rigid-part patch builder and run the same counters on OptiX-generated candidates.
