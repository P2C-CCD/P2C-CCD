# Complete Example Query Scale Design

## Scope

This note defines practical query-count scaling targets for the four complete examples:

1. `T0 synthetic_proxy`
2. `trained_stpf_high_density`
3. `ABC CAD`
4. `Thingi10K`

The goal is to cover a common benchmark ladder:

- `1,000` queries
- `4,000` queries
- `10,000` queries
- `100,000` queries

Throughout this note, `queries` means **top-level pair-level CCD queries**, not broad-phase candidate pairs and not primitive-level exact checks.

## Query Formulas

### T0 synthetic_proxy

`generate_exact_oracle_dataset(DatasetGenerationConfig(..., include_robot_links=True))`

Query count:

```text
query_count = 5 * mesh_count_per_split + robot_link_count
```

Reason:

- the generated mesh samples span the five default mesh splits
- robot-link validation contributes `robot_link_count` additional queries

### trained_stpf_high_density

The complete-example five-method pipeline uses the query-level eval dataset, not the candidate-expanded dense rows.

With:

```text
include_robot_links = False
```

query count is:

```text
query_count = 5 * mesh_count_per_split
```

If the dense workload uses the default `128 candidates/query`, then:

```text
candidate_count = 128 * query_count
```

### ABC CAD

`build_abc_paper_benchmark_dataset(...)`

Each selected CAD pair is expanded into four motion variants, so:

```text
query_count = 4 * pair_limit
```

To make `pair_limit` feasible, the held-out asset slice must satisfy:

```text
C(asset_count, 2) >= pair_limit
```

### Thingi10K

`build_thingi10k_paper_benchmark_dataset(...)`

Each eval pair is expanded into four motion variants, so:

```text
query_count = 4 * eval_pair_limit
```

The held-out eval split must satisfy:

```text
C(eval_asset_count, 2) >= eval_pair_limit
```

where:

```text
eval_asset_count ≈ asset_limit - round(asset_limit * train_fraction)
```

## Recommended Query Ladder

### 1. T0 synthetic_proxy

| Target queries | `mesh_count_per_split` | `robot_link_count` | Result |
| --- | ---: | ---: | ---: |
| `1,000` | `180` | `100` | `5*180 + 100 = 1,000` |
| `4,000` | `720` | `400` | `5*720 + 400 = 4,000` |
| `10,000` | `1,800` | `1,000` | `5*1,800 + 1,000 = 10,000` |
| `100,000` | `18,000` | `10,000` | `5*18,000 + 10,000 = 100,000` |

Notes:

- This scale is analytically generated and is the easiest path to `100k` top-level queries.
- This is the cleanest place to stress-test throughput before moving to real-object datasets.

### 2. trained_stpf_high_density

| Target queries | `mesh_count_per_split` | Query count | Candidate count at `128/query` |
| --- | ---: | ---: | ---: |
| `1,000` | `200` | `1,000` | `128,000` |
| `4,000` | `800` | `4,000` | `512,000` |
| `10,000` | `2,000` | `10,000` | `1,280,000` |
| `100,000` | `20,000` | `100,000` | `12,800,000` |

Notes:

- This line is suitable for learned-STPF stress tests because candidate inflation is explicit and controllable.
- The `100k` query point is valid, but the candidate-expanded workload is already very large; it is better treated as a focused STPF benchmark rather than a routine five-method sweep.

### 3. ABC CAD

Recommended benchmark scale uses the official root, not the demo subset.

| Target queries | `pair_limit` | Minimum held-out assets | Recommended `benchmark_asset_count` | Result |
| --- | ---: | ---: | ---: | ---: |
| `1,000` | `250` | `23` | `48` | `4*250 = 1,000` |
| `4,000` | `1,000` | `46` | `64` | `4*1,000 = 4,000` |
| `10,000` | `2,500` | `72` | `80` | `4*2,500 = 10,000` |
| `100,000` | `25,000` | `225` | `225` for benchmark-only, `>=320` source assets for strict held-out split | `4*25,000 = 100,000` |

Notes:

- The current local official root has `256` meshes. That is enough for `1k`, `4k`, and `10k`.
- `100k` queries requires `25,000` held-out pairs, so the benchmark needs about `225` held-out assets.
- If strict train/benchmark separation must be preserved from the same official pool, expand the official root to at least `320` assets so that one can keep a meaningful training slice and still reserve `225` held-out benchmark assets.

### 4. Thingi10K

| Target queries | `eval_pair_limit` | Minimum eval assets | Recommended `asset_limit` | Recommended `train_fraction` | Result |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1,000` | `250` | `23` | `96` | `0.75` | `4*250 = 1,000` |
| `4,000` | `1,000` | `46` | `96` | `0.50` | `4*1,000 = 4,000` |
| `10,000` | `2,500` | `72` | `128` | `0.40` | `4*2,500 = 10,000` |
| `100,000` | `25,000` | `225` | `300` | `0.25` | `4*25,000 = 100,000` |

Notes:

- Current `asset_limit=96` is enough for `1k` and `4k`.
- `10k` queries needs a somewhat larger official subset.
- `100k` queries is realistic only after expanding the official subset to about `300` assets and relaxing the split so the eval pool reaches about `225` assets.

## Practical Benchmark Recommendations

### Recommended standard ladder

For routine benchmark reporting:

- `1,000`
- `4,000`
- `10,000`

These scales are practical across all four examples with moderate dataset preparation.

### Recommended extreme ladder

For throughput stress tests:

- `100,000` for `T0 synthetic_proxy`
- `100,000` for `trained_stpf_high_density` only in dedicated STPF runs
- `100,000` for `ABC CAD` only after expanding the official root
- `100,000` for `Thingi10K` only after expanding the official subset

### What should not be mixed

- The query-level five-method benchmark should not be confused with candidate-expanded dense rows.
- A `100k` query benchmark on synthetic query-level samples is not equivalent to `100k` primitive-level exact checks.
- For real-object datasets, `100k` top-level queries usually implies much larger candidate and exact-work volume underneath.

## Bottom Line

If a single consistent design is needed now:

- `T0 synthetic_proxy`: `1k / 4k / 10k / 100k`
- `trained_stpf_high_density`: `1k / 4k / 10k / 100k`
- `ABC CAD`: `1k / 4k / 10k`, and `100k` after expanding the official root
- `Thingi10K`: `1k / 4k / 10k`, and `100k` after expanding the official subset

This is the cleanest query-scale ladder that is technically consistent with the current repository and dataset builders.
