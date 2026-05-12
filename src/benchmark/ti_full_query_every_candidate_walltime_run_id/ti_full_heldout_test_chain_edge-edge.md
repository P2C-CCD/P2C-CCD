# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `heldout_test`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chain` | `edge-edge` | `4312500` | `4312500` | `0%` | `557` | `4311940` | `3` | `0` | `1` | `1872.24` | `80894.7` | `53310` | `18.7582` | `16.9` | `18.7` | `29.7` |
