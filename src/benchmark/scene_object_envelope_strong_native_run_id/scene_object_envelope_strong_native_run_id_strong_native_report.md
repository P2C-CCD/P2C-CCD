# Strong native scene/object envelope benchmark

This benchmark is a native full-scene/object-envelope audit. It exports candidate-level labels from the same Tight-Inclusion exact kernel, trains a frozen 32-feature tiny ranker on training scenes, and replays held-out adjacent scene frames with exact fallback. The learned model only orders proposals; final correctness remains exact/certified by fallback.

## Protocol

- Train scenes: `armadillo-rollers,cloth-ball,cloth-funnel,n-body-simulation`
- Eval scenes: `puffer-ball,rod-twist`
- Frozen checkpoint: `src/outputs/stpf_training/scene_object_envelope_strong_native_run_id/model_state.pt`
- Native row CSV: `src/benchmark/scene_object_envelope_strong_native_run_id/scene_object_envelope_strong_native_run_id_rows.csv`
- Comparison CSV: `src/benchmark/scene_object_envelope_strong_native_run_id/scene_object_envelope_strong_native_run_id_strong_native_comparison.csv`

## Aggregate comparison

| Method | Exact calls | Reduction vs dense | Positive proposal hits | Positive groups | FN | Native exact ms | Total wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `AllExact+TI` | 252,295,029 | 0.000% | 0 | 22 | 0 | 102122.040 | 102122.040 |
| `FrozenLearnedAnyHit+TI` | 242,049 | 99.904% | 0 | 22 | 0 | 307.086 | 288228.171 |
| `ProximityHeuristicAnyHit+TI` | 181,591 | 99.928% | 10 | 22 | 0 | 241.641 | 19821.757 |
| `MotionHeuristicAnyHit+TI` | 238,223 | 99.906% | 2 | 22 | 0 | 294.546 | 19994.613 |
| `RandomAnyHit+TI` | 241,389 | 99.904% | 1 | 22 | 0 | 306.211 | 2220.992 |
| `FairFrontierLearnedAnyHit+TI` | 233,221 | 99.908% | 7 | 22 | 0 | 257.653 | 1292.968 |
| `FairFrontierProximityAnyHit+TI` | 232,242 | 99.908% | 7 | 22 | 0 | 296.947 | 1315.845 |
| `FairFrontierMotionAnyHit+TI` | 232,478 | 99.908% | 7 | 22 | 0 | 284.521 | 1301.092 |
| `FairFrontierRandomAnyHit+TI` | 236,895 | 99.906% | 2 | 22 | 0 | 300.026 | 1316.577 |

## Main reading

- Learned exact calls vs dense: `242,049` / `252,295,029`.
- Learned exact calls vs proximity heuristic: `242,049` vs `181,591`.
- Learned FN: `0`.

## Figures

- `src/benchmark/scene_object_envelope_strong_native_run_id/scene_object_envelope_strong_native_run_id_exact_call_comparison.png`
- `src/benchmark/scene_object_envelope_strong_native_run_id/scene_object_envelope_strong_native_run_id_exact_call_comparison.pdf`
