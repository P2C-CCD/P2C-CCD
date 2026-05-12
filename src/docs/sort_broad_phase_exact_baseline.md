# SortBroadPhaseExact Baseline

`SortBroadPhaseExact` is the explicit sorting-based broad-phase baseline for
answering the review question about GPU broad-phase comparison.

## Pipeline

```text
external/internal query batch
  -> swept feature AABB primitives
  -> sort by AABB lower endpoint
  -> sweep active intervals
  -> full AABB overlap filtering
  -> PureExactCPU narrow phase
  -> BenchmarkRow + sort broad-phase counters
```

Python entry points:

```python
from p2cccd.bench import (
    SortBroadPhaseConfig,
    run_sort_broad_phase_exact_on_external_batch,
    run_sort_broad_phase_exact_on_generated_dataset,
    run_sort_broad_phase_exact_on_internal_samples,
    sort_sweep_broad_phase_pairs,
)
```

## Why This Is Separate From BVHExact

`BVHExact` is the CPU broad-phase abstraction slot intended for Embree/Coal
replacement. Its current default is also a deterministic CPU AABB sweep, but
the method is framed as a CPU BVH/broad-phase baseline.

`SortBroadPhaseExact` is framed around the data-parallel GPU comparison:

- primitives are converted to sortable lower-endpoint records,
- the broad phase is decomposed into sort and sweep stages,
- counters expose endpoint count, active interval tests, AABB overlap tests,
  pair count, `sort_ms`, `sweep_ms`, and total broad-phase time,
- `backend_name="gpu_sort_sweep_compatible"` is reserved for a later CUDA
  radix-sort/sweep implementation with the same output contract.

Sort broad-phase timing is exported through the `BenchmarkRowV2` timing fields
when the suite runner writes CSV/JSONL results. `proposal_ms` remains zero
because this baseline does not use STPF.

## Correctness Policy

The sorting broad phase is conservative for the current swept-AABB primitives:

- a query is sent to exact if the two feature swept AABBs overlap,
- a query is culled only when the swept AABBs are separated,
- active pairs are deterministic and sorted by query and primitive ids,
- narrow phase is still `PureExactCPU`, with the same conservative
  undecided-as-collision default as other correctness baselines.

This baseline does not use STPF and does not perform exact queue scheduling.
It measures broad-phase candidate generation plus exact replay.

## Current Scope

- CPU implementation of the GPU-compatible sort/sweep algorithm.
- Supports internal generated samples and external CCD adapter batches.
- Supports sweep axis selection through `SortBroadPhaseConfig.axis`.
- CPU implementation of the GPU-compatible algorithm is the correctness
  baseline. A CUDA/Thrust/CUB radix-sort implementation is treated as a separate
  performance optimization, not as part of the current correctness contract.

## Validation

`test_sort_broad_phase_exact.py` covers:

- low-level sort/sweep pair filtering and exported counters,
- synthetic external CCD correctness,
- Scalable CCD sample comparison against `BVHExact` pair counts,
- internal generated dataset execution,
- GPU-compatible backend naming,
- config validation for backend and axis.
