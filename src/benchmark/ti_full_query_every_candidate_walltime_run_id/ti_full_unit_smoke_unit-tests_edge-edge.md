# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `unit_smoke`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unit-tests` | `edge-edge` | `74` | `74` | `0%` | `36` | `38` | `0` | `0` | `1` | `1247.23` | `1248.79` | `59.2572` | `16875.6` | `28` | `3950.46` | `212742` |
