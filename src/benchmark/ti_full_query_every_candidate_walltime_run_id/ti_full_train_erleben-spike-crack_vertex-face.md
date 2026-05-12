# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `train`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `erleben-spike-crack` | `vertex-face` | `500` | `500` | `0%` | `10` | `490` | `0` | `0` | `1` | `23.2444` | `33.146` | `15084.8` | `66.292` | `17.5` | `62.1` | `1043.22` |
