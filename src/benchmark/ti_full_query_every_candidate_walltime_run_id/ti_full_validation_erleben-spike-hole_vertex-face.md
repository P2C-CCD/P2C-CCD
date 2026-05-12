# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `validation`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `erleben-spike-hole` | `vertex-face` | `1000` | `1000` | `0%` | `0` | `1000` | `0` | `0` | `0` | `0.683` | `20.1405` | `49651.2` | `20.1405` | `17.4` | `19.1` | `23.9` |
