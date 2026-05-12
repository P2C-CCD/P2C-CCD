# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `validation`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chain` | `edge-edge` | `2162500` | `2162500` | `0%` | `270` | `2162230` | `0` | `0` | `1` | `921.373` | `40216.4` | `53771.6` | `18.5972` | `16.9` | `18.6` | `28.3` |
