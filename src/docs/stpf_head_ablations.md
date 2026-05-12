# STPF Head Ablations

`IntervalOnly` and `RankingOnly` isolate which STPF heads contribute to the
safe exact-work schedule.

## Shared Pipeline

```text
external/internal query batch
  -> conservative RT-compatible candidate generation
  -> ProposalFeatureRow extraction
  -> STPFModel inference
  -> head-specific monotonic scheduling
  -> PureExactCPU certificate execution
  -> BenchmarkRow + coverage validation
```

Python entry points:

```python
from p2cccd.bench import (
    IntervalOnlyConfig,
    RankingOnlyConfig,
    run_interval_only_on_external_batch,
    run_interval_only_on_generated_dataset,
    run_interval_only_on_internal_samples,
    run_ranking_only_on_external_batch,
    run_ranking_only_on_generated_dataset,
    run_ranking_only_on_internal_samples,
)
```

## IntervalOnly

`IntervalOnly` keeps only the interval head signal:

- priority is `max(interval_scores)`,
- feature-family mask is the conservative base query family only,
- family ranking scores are ignored,
- exact interval remains the full conservative `[0, 1]`.

The interval head therefore affects queue ordering, but it cannot narrow the
certified interval or drop any candidate.

## RankingOnly

`RankingOnly` keeps only the family-ranking signal:

- priority is `max(family_scores)`,
- feature-family mask is `base_mask | predicted_family_mask`,
- interval scores are ignored,
- exact interval remains the full conservative `[0, 1]`.

The family-ranking head can add candidate feature families for exact replay,
but it cannot replace or shrink the conservative base mask.

## Safety Contract

Both ablations preserve the same no-false-negative contract as `RTSTPFExact`:

- every compact `CandidateRecord` produces one `ProposalFeatureRow`,
- every compact `CandidateRecord` produces one `ExactWorkItem`,
- no scheduled interval is narrower than `[0, 1]`,
- missing, invalid, OOD, or high-uncertainty proposal outputs route to
  `ProposalSource.FALLBACK`,
- `validate_rt_exact_coverage` rejects dropped work and missing certificates.

This makes the two ablations suitable for paper tables that compare full STPF
against individual head contributions without changing correctness semantics.

## Current Scope

- Python workbench implementation.
- Uses the learned/default `STPFModel` path; dummy proposal paths are retired
  from `RTSTPFExact` and are not part of the current head-ablation contract.
- Uses the same CPU reference RT candidate path as `RTSTPFExact`.
- Uses `PureExactCPU` for certificate execution.
- Does not yet use learned interval narrowing because certified narrowing
  requires a separate conservative interval proof.

## Validation

`test_stpf_head_ablations.py` covers:

- direct scheduling semantics for interval-only and ranking-only modes,
- default-model external CCD execution,
- real `STPFModel` inference on the Scalable CCD sample adapter,
- OOD fallback without candidate drop,
- internal generated dataset execution,
- missing-model validation.
