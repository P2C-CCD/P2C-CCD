# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `heldout_test`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `erleben-spike-wedge` | `edge-edge` | `375` | `375` | `0%` | `19` | `354` | `2` | `0` | `1` | `75.8629` | `84.8388` | `4420.15` | `226.237` | `20.5` | `314.22` | `4437.13` |
