# Slab Count And Proxy Family Ablation

`SlabProxyAblation` evaluates two candidate-generation knobs before STPF scheduling:

- `slab_count`: how many uniform time slabs each query motion interval is split into.
- `proxy_family`: which proxy type combination is assigned to object A and object B.

The Python workbench path expands each analytic swept-sphere query into one proxy pair per slab, then runs the same CPU AABB broad-phase abstraction used by `BVHExact`. This keeps the runner deterministic and usable before the C++ patch builder and OptiX device programs are exposed through pybind.

## Runner

```python
from p2cccd.bench import (
    SlabProxyAblationConfig,
    run_slab_proxy_ablation_on_generated_dataset,
    slab_count_ablation_options,
)
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset

dataset = generate_exact_oracle_dataset(DatasetGenerationConfig(mesh_count_per_split=8))
result = run_slab_proxy_ablation_on_generated_dataset(
    dataset,
    SlabProxyAblationConfig(options=slab_count_ablation_options((1, 2, 4, 8))),
)
```

Proxy-family options can be created separately:

```python
from p2cccd.bench import proxy_family_ablation_options

config = SlabProxyAblationConfig(options=proxy_family_ablation_options(slab_count=4))
```

## Correctness Gate

Each row is feasible only if:

- every positive oracle query has at least one active candidate in the slab containing the contact interval;
- `candidate_recall >= min_candidate_recall`;
- `fn_count == 0`;
- `fp_count == 0`.

The runner intentionally requires `same_query_only = True` because the internal analytic oracle is query-local. Scene-level cross-query slab pairing belongs in the future C++/OptiX replay path.

## Metrics

Important CSV columns:

- `slab_count`: number of uniform slabs per query.
- `proxy_family`: `aabb+aabb`, `capsule+capsule`, or mixed combinations.
- `proxy_count`: generated proxy primitives.
- `raw_hit_count`: raw broad-phase overlaps.
- `compact_candidate_count`: unique `(query_id, slab_id)` candidates sent to exact replay.
- `candidate_recall`, `fn_count`, `fp_count`: safety checks.
- `avg_proxy_volume`: proxy-shape volume estimate for proxy-family comparisons.
- `proxy_cost_units`: simple deterministic proxy-cost model; capsule proxies are charged more than swept AABBs.
- `score`: weighted selection objective.

The default score is candidate-density focused:

```text
score = candidate_weight * compact_candidate_count
      + raw_hit_weight * raw_hit_count
      + proxy_weight * proxy_count
      + proxy_cost_weight * proxy_cost_units
      + slab_weight * slab_count
```

## Limitations

The current proxy-family comparison records family and cost differences, but broad-phase overlap is still performed with conservative AABBs. The production evaluation should replace this layer with C++ proxy construction and OptiX traversal so capsule and swept-AABB families are benchmarked through the real RT backend.
