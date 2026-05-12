# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `train`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `erleben-cube-internal-edges` | `edge-edge` | `2125` | `2125` | `0%` | `40` | `2079` | `6` | `0` | `1` | `2.6469` | `44.233` | `48041.1` | `20.8155` | `16.9` | `19.9` | `41.88` |
