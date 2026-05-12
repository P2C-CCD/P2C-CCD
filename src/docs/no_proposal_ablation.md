# NoProposal Ablation

`NoProposal` is the safety-preserving ablation for measuring what STPF
scheduling contributes beyond conservative candidate generation and exact
certification.

## Pipeline

```text
external/internal query batch
  -> conservative RT-compatible candidate generation
  -> no proposal feature extraction
  -> no proposal inference
  -> fallback ExactWorkItem scheduling
  -> PureExactCPU certificate execution
  -> BenchmarkRow + audit/certificate coverage
```

Python entry points:

```python
from p2cccd.bench import (
    NoProposalConfig,
    run_no_proposal_on_external_batch,
    run_no_proposal_on_generated_dataset,
    run_no_proposal_on_internal_samples,
)
```

## Ablation Contract

`NoProposal` intentionally differs from `RTExact`:

- `RTExact` is the raw no-STPF correctness runner and uses
  `ProposalSource.RAW`.
- `NoProposal` is the proposal-ablation fallback path and uses
  `ProposalSource.FALLBACK` for every compact candidate.
- `NoProposal` exports `fallback_ratio = 1.0` when at least one candidate is
  scheduled, while `RTExact` exports `fallback_ratio = 0.0`.
- `NoProposal` keeps `proposal_ms = 0.0`, because no proposal feature rows,
  model inference, or proposal outputs are generated.

The correctness contract is the same as the conservative fallback branch in
`RTSTPFExact`:

- every compact `CandidateRecord` maps to exactly one `ExactWorkItem`,
- every `ExactWorkItem` covers the full interval `[0, 1]`,
- feature-family masks come from the conservative query family,
- `validate_rt_exact_coverage` rejects dropped candidates, duplicate work
  items, and missing certificates.

This makes `NoProposal` the direct comparison point for quantifying STPF
queue ordering, proposal overhead, and exact-work reduction without risking
false negatives.

## Current Scope

Like the other Python baselines, this is a correctness workbench path:

- RT candidate generation uses the deterministic CPU AABB broad phase unless a
  future OptiX-compatible backend is injected.
- Exact certificate execution uses `PureExactCPU`.
- External inputs are adapter-level CCD query batches.
- Full benchmark metadata and percentile export remain part of the later
  `BenchmarkRowV2` work.

## Validation

`test_no_proposal_ablation.py` covers:

- fallback routing for every compact candidate,
- semantic difference from dummy `RTSTPFExact` scheduling,
- Scalable CCD sample execution,
- internal generated dataset execution,
- config validation for backend and work-item ids.
