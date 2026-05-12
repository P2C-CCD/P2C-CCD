# PureExactCPU Baseline

`PureExactCPU` is the first correctness baseline that can run before Python bindings expose the C++ exact certificate engine.

Runtime module:

```text
src/python/p2cccd/bench/pure_exact_cpu.py
```

## Supported Inputs

- Internal programmatic samples from `GeneratedDataset` / `MotionDiscPairSample`.
- External CCD primitive query batches from `DatasetQueryBatch`.
- Scalable CCD sample `vf` and `ee` query batches through `ScalableCCDSampleAdapter`.

## External Primitive Semantics

External `vf` queries are mapped to point-triangle CCD:

```text
v_t0, f0_t0, f1_t0, f2_t0, v_t1, f0_t1, f1_t1, f2_t1
```

External `ee` queries are mapped to edge-edge CCD:

```text
ea0_t0, ea1_t0, eb0_t0, eb1_t0, ea0_t1, ea1_t1, eb0_t1, eb1_t1
```

The baseline uses recursive time-interval subdivision with endpoint/midpoint distance checks and a conservative midpoint-motion lower bound.

## False-Negative Policy

For external correctness gating, `PureExactCPUConfig.conservative_undecided_as_collision` defaults to `True`.

This means:

- `collision` predicts collision,
- `separation` predicts separation,
- `undecided` predicts collision.

The policy can increase false positives, but it protects the first correctness gate from false negatives while the fully bound C++ exact replay path is still pending.

## Current Output

The runner returns:

- per-query `PureExactQueryResult`,
- aggregate `BenchmarkRow`,
- source/scene/batch identifiers,
- `final_fn_zero` convenience property.

This is enough for TODO 79 and for the next runner stage:

```text
DatasetQueryBatch -> PureExactCPU / RTExact -> BenchmarkRowV2
```

## Limitations

- The Python implementation is a correctness/workbench baseline, not a final high-performance exact engine.
- TOI is conservative and may be an upper bound from sampled/subdivided intervals.
- External full-dataset and external C++ baseline build validation are separate tasks.
