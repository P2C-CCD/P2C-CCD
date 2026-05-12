# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `validation`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `erleben-cube-cliff-edges` | `vertex-face` | `125` | `125` | `0%` | `15` | `65` | `45` | `0` | `1` | `1292.66` | `1296.78` | `96.393` | `10374.2` | `24.9` | `52.62` | `218591` |
