# Adapter Contracts

This note defines the first external benchmark adapter contract for the correctness / CCD layer described in `P2CCCD_revise_run_id.md` section 2.1.

## Scope

The first layer is for correctness evidence, not STPF training scale:

- import external CCD query batches,
- preserve source scene/query identity,
- keep ground-truth collision labels when available,
- map each query family to P2CCCD witness families,
- make later `PureExactCPU`, `RTExact`, and `RTSTPFExact` benchmark runners consume the same representation.

## Runtime Contract

Python contract module:

```text
src/python/p2cccd/datasets/ccd/contracts.py
```

Core records:

- `DatasetScene`: one external scene or sequence.
- `DatasetQueryBatch`: one homogeneous query batch from a source scene and query family.
- `ExternalCCDQuery`: one vertex-face or edge-edge CCD primitive query with four vertices at `t0` and four vertices at `t1`.
- `SourceLicense`: minimal source/license metadata gate for external data.

Schema version:

```text
CCD_ADAPTER_SCHEMA_VERSION = 1
```

## Query Family Mapping

| External family | P2CCCD family | Meaning |
|---|---|---|
| `vf` | `point_triangle` | vertex-face query |
| `ee` | `edge_edge` | edge-edge query |

For Scalable CCD sample data, each query CSV stores 8 vertex rows:

```text
vf: v_t0, f0_t0, f1_t0, f2_t0, v_t1, f0_t1, f1_t1, f2_t1
ee: ea0_t0, ea1_t0, eb0_t0, eb1_t0, ea0_t1, ea1_t1, eb0_t1, eb1_t1
```

The adapter converts rational coordinate pairs to `float` for the initial runner contract. If exact rational replay is needed later, add a parallel rational payload instead of changing the existing float fields.

## Safety Rules

- An adapter must never silently drop source queries.
- A loaded `DatasetQueryBatch` must validate through `validate_query_batch`.
- Ground-truth labels may be `None` only when the external source does not provide labels.
- Source license metadata must exist before a dataset adapter is used in a benchmark suite.
- Missing P1 sources must appear as unavailable instead of being ignored; locally present P1 sources such as `rigid-ipc` may remain adapter-pending.

## Next Integration Point

The next runner stage should implement:

```text
DatasetQueryBatch -> PureExactCPU / RTExact input -> BenchmarkRowV2
```

That runner must keep `source_name`, `scene_name`, `batch_id`, `query_id`, and `source_query_index` in the output row so false negatives can be traced back to the original external query.
