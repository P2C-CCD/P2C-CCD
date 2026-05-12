# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `validation`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `erleben-wedges` | `edge-edge` | `177` | `177` | `0%` | `11` | `166` | `0` | `0` | `1` | `6.6829` | `10.45` | `16937.8` | `59.0395` | `18.8` | `34.4` | `1170.02` |
