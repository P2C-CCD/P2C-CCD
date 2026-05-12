# cuRobo-Style Downstream Comparison

Task 93 adds a cuRobo-style downstream comparison for robot link-pair motion validation. This is a deterministic style baseline, not an official cuRobo integration.

The runner samples a link-pair trajectory at a fixed number of poses and checks each pose pair with inflated link spheres. It compares that discrete decision against the internal analytic continuous swept-sphere oracle. Sparse trajectories can therefore produce false negatives, which is the intended signal for this downstream comparison.

## Contract

- Input: generated `MotionDiscPairSample` rows, normally filtered to `robot_link_validation`.
- Method name: `CuRoboDownstream`.
- Output: standard `BenchmarkRow` and `BenchmarkRowV2` rows through the suite runner.
- Exact calls: zero, because this baseline represents a downstream discrete checker rather than certified CCD.
- Safety status: not certified. Use `fn_count`, `candidate_recall`, and `final_fn_zero` to quantify misses.

## Configuration

```python
CuRoboDownstreamConfig(
    trajectory_step_count=16,
    link_sphere_radius_scale=1.0,
    collision_activation_distance=0.0,
    robot_link_only=True,
)
```

- `trajectory_step_count`: number of sampled poses in `[0, 1]`. Must be at least 2.
- `link_sphere_radius_scale`: radius inflation multiplier for the link proxy spheres.
- `collision_activation_distance`: non-negative safety buffer around the sampled collision threshold.
- `robot_link_only`: when true, mesh-pair samples are filtered out.

## Suite Runner

From `src`:

```powershell
conda activate cudadev
$env:PYTHONPATH = "python"
python -m p2cccd.bench.run_suite --config configs\benchmark_suites\curobo_downstream.json
```

The bundled smoke suite compares endpoint-only sampling, denser trajectory sampling, and denser sampling with a small activation distance. Export uses the same `BenchmarkRunMeta` plus `BenchmarkRowV2` path as the other benchmark suites.

## Interpretation

`CuRoboDownstream` is useful for downstream robotics validation because it gives a direct baseline for discrete trajectory checking. It should not be used as the correctness reference for P2CCCD. The correctness contract still belongs to `PureExactCPU`, `BVHExact`, `RTExact`, and eventually the certified RT/STPF/exact pipeline.
