# RT-DCD And RT-CCD Style Reproductions

This module provides paper-style reproduction baselines for comparison, not a claim of full source-level reproduction of any specific external implementation.

## RTDCDStyle

`RTDCDStyle` samples each motion interval at a fixed number of discrete times, builds instantaneous proxy AABBs, runs RT-like broad-phase overlap, and then replays the analytic exact oracle only for active candidates.

This intentionally behaves like a discrete collision detector:

- endpoint-only sampling can miss tunneling contacts;
- increasing `sample_count` improves coverage but increases proxy and candidate work;
- reported `candidate_recall` and `fn_count` expose missed continuous contacts.

Use this baseline to show why a CCD system cannot rely on DCD-style time sampling alone.

## RTCCDStyle

`RTCCDStyle` splits each motion interval into uniform time slabs, builds conservative swept proxy AABBs per slab, runs RT-like broad-phase overlap, and replays the exact oracle on candidate slabs.

This is closer to the current P2CCCD RT candidate path:

- time slabs preserve continuous contact coverage when proxies are conservative;
- `slab_count` controls the granularity/cost tradeoff;
- proxy family is recorded through `ProxyType`, while the Python workbench still uses AABB overlap underneath.

## Runner

```python
from p2cccd.bench import (
    RTCCDStyleConfig,
    RTDCDStyleConfig,
    run_rt_ccd_style_on_generated_dataset,
    run_rt_dcd_style_on_generated_dataset,
)
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset

dataset = generate_exact_oracle_dataset(DatasetGenerationConfig(mesh_count_per_split=4))
dcd = run_rt_dcd_style_on_generated_dataset(dataset, RTDCDStyleConfig(sample_count=2))
ccd = run_rt_ccd_style_on_generated_dataset(dataset, RTCCDStyleConfig(slab_count=4))
```

The suite config is:

```text
src/configs/benchmark_suites/rt_style_reproduction.json
```

## Limitations

The Python implementation is a faithful style reproduction for benchmark structure:

- RT-DCD means discrete time sampling plus RT-like broad phase;
- RT-CCD means uniform swept time slabs plus RT-like broad phase.

It is not a GPU OptiX device-program reproduction yet. Once the real OptiX candidate emitter exists, this baseline should be rerun with the same exported `BenchmarkRowV2` schema.
