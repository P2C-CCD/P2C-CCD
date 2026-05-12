# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `heldout_test`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `golf-ball` | `vertex-face` | `1287500` | `1287500` | `0%` | `3417` | `1284083` | `0` | `0` | `1` | `591.42` | `28836.8` | `44647.8` | `22.3975` | `20.7` | `21.4` | `39.2` |
