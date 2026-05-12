# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `heldout_test`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mat-twist` | `edge-edge` | `812500` | `812500` | `0%` | `174` | `812326` | `0` | `0` | `1` | `868.862` | `16803` | `48354.3` | `20.6807` | `18.8` | `20.6` | `34.3` |
