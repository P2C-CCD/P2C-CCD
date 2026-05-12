# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `train`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `erleben-sliding-spike` | `edge-edge` | `257` | `257` | `0%` | `0` | `257` | `0` | `0` | `0` | `0.3059` | `5.7906` | `44382.3` | `22.5315` | `18.3` | `22.44` | `32.644` |
