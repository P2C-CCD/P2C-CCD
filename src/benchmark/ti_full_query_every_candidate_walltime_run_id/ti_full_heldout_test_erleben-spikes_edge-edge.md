# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `heldout_test`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `erleben-spikes` | `edge-edge` | `250` | `250` | `0%` | `4` | `246` | `0` | `0` | `1` | `9.4711` | `14.8971` | `16781.8` | `59.5884` | `20.1` | `48.69` | `1233.64` |
