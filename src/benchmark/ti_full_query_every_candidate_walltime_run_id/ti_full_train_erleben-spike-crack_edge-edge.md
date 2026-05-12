# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `train`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `erleben-spike-crack` | `edge-edge` | `645` | `645` | `0%` | `8` | `637` | `0` | `0` | `1` | `2.318` | `15.0677` | `42806.8` | `23.3608` | `18.3` | `26.26` | `66.564` |
