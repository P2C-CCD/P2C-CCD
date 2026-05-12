# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `validation`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chain` | `vertex-face` | `537500` | `537500` | `0%` | `10` | `537490` | `0` | `0` | `1` | `252.997` | `10050.2` | `53481.5` | `18.6981` | `17` | `18.8` | `28.6` |
