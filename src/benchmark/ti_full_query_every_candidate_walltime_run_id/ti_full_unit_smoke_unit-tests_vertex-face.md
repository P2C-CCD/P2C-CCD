# Tight-Inclusion Full-Query C++ Benchmark

- Method: `TightInclusion`
- Split: `unit_smoke`
- Parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`

| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unit-tests` | `vertex-face` | `616` | `616` | `0%` | `302` | `314` | `0` | `0` | `1` | `1472.29` | `1480.83` | `415.982` | `2403.95` | `15.85` | `59.1` | `153625` |
