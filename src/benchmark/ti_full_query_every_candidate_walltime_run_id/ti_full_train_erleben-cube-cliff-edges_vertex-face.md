# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `train`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `erleben-cube-cliff-edges` | `vertex-face` | `637` | `637` | `0%` | `18` | `592` | `27` | `0` | `1` | `1258.82` | `1271.92` | `500.818` | `1996.73` | `17.3` | `23.02` | `218.368` |
