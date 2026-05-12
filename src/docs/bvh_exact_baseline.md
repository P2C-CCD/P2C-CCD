# BVHExact Baseline

`BVHExact` is the first broad-phase plus exact-narrow-phase correctness baseline for P2CCCD.

Runtime module:

```text
src/python/p2cccd/bench/bvh_exact.py
```

## Pipeline

```text
internal samples / external CCD query batch
  -> swept feature AABBs
  -> BroadPhaseBackend.find_pairs()
  -> PureExactCPU on surviving pairs
  -> BenchmarkRow
```

The current broad phase is `CpuAabbBroadPhaseBackend`, a deterministic CPU sort-and-sweep over swept AABBs. It is intentionally small and conservative so the baseline can run before a real Embree or Coal implementation is wired in.

## Backend Contract

The broad-phase contract is:

```text
Sequence[BroadPhasePrimitive] -> tuple[BroadPhasePair, BroadPhaseStats]
```

`BroadPhasePrimitive` stores:

- `primitive_id`
- `query_id`
- role `a` or `b`
- swept `Aabb`
- witness family name
- source metadata

`BroadPhasePair` stores the surviving pair ids and query id. `BroadPhaseStats` reports primitive count, pair count, backend name, and elapsed broad-phase time.

`BVHExactConfig.backend_name` currently accepts:

- `cpu_aabb_sort_sweep`
- `embree_compatible`
- `coal_compatible`

The last two names are compatibility labels for later backend replacement. They do not link Embree or Coal yet.

## Supported Inputs

- Internal generated `MotionDiscPairSample` / `GeneratedDataset` samples.
- External `DatasetQueryBatch` instances from CCD adapters.
- Scalable CCD sample batches for `vf` and `ee` through `ScalableCCDSampleAdapter`.

External query feature mapping:

- `vf`: point AABB vs triangle swept AABB
- `ee`: first edge swept AABB vs second edge swept AABB

Internal generated samples use swept sphere-proxy AABBs built from endpoint centers and radius inflation.

## Correctness Policy

Broad-phase culling is conservative: a query can be skipped only if its two swept feature AABBs do not overlap. Surviving pairs are evaluated by `PureExactCPU`.

For correctness-gate use, keep `PureExactCPUConfig.conservative_undecided_as_collision=True`. This preserves `final FN = 0` behavior at the cost of possible false positives.

## Benchmark Fields

`BVHExactResult.benchmark` contains:

- `query_count`: total input queries
- `avg_candidates`: broad-phase pair count divided by total queries
- `avg_exact_evals`: exact narrow-phase work divided by total queries
- `exact_ms`: PureExactCPU time on surviving pairs
- `total_ms`: broad-phase time plus exact time
- `candidate_recall`: `1.0` when no labeled collision is culled

`BVHExactResult.final_fn_zero` is the convenience correctness gate used by tests.

## Limitations

- The implementation is Python-side and intended for correctness/workbench runs, not final performance.
- It does not build an Embree scene or call Coal yet.
- It operates over adapter-level query pairs, not a full mesh-scene BVH.
- Actual Embree-backed CPU broad phase remains a follow-up implementation item; the adapter plan is documented in `src/docs/reference_stack_integration.md`.
