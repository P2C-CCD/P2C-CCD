# Native scene/object motion-trap envelope benchmark

This benchmark is a native object-envelope stress test for P2C-CCD scheduling.  It is generated from full-scene adjacent PLY frames, but the measured detection work is native CCD replay, not full simulation solver time.

## Purpose

The Scalable-CCD sample scene/object envelope audit is a real full-scene compatibility and fallback test, but simple proximity and motion orderings remain competitive there.  The motion-trap benchmark isolates a harder ordering regime: disconnected rigid sphere objects swap positions between adjacent frames, so a swept object-object envelope contains many endpoint-ambiguous VF/EE primitive candidates.  The final collision decision still comes only from the Tight-Inclusion exact backend and conservative fallback.

## Data

- Dataset root: `src/datasets/native_scene_object_envelope_motion_trap_run_id`
- Format: binary little-endian PLY adjacent full-scene frames.
- Train scene: `motion-trap-train`, 12 objects, 1,368 vertices, 2,688 faces, 6 object-pair traps.
- Held-out scene: `motion-trap-heldout`, 16 objects, 1,824 vertices, 3,584 faces, 8 object-pair traps.
- Protocol: object-object swept AABB envelopes; self-object pairs are excluded by the benchmark runner.
- Feature export: `57,809` sampled candidate-feature rows for `motion-trap-train` with positives retained and negatives subsampled by deterministic stride.

## Reproduction

Run from the repository root with the project conda environment:

```powershell
conda activate cudadev

python src/tools/generate_scene_object_envelope_motion_trap_dataset.py `
  --train-pairs 6 `
  --heldout-pairs 8 `
  --lat 8 `
  --lon 16

python src/tools/run_scene_object_envelope_strong_native_benchmark.py `
  --run-name scene_object_envelope_motion_trap_native_small_run_id `
  --source-root src/datasets/native_scene_object_envelope_motion_trap_run_id `
  --train-scenes motion-trap-train `
  --eval-scenes motion-trap-heldout `
  --epochs 30 `
  --feature-negative-stride 16 `
  --proposal-top-k 32 `
  --optimized-frontier-k 128 `
  --optimized-scan-limit-per-group 4096 `
  --exclude-self-object-pairs `
  --force-build
```

## Results

Primary output directory:

```text
src/benchmark/scene_object_envelope_motion_trap_native_small_run_id
```

Held-out aggregate:

| Method | Exact calls | Positive groups | FN | Native exact backend ms | Total wall ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `AllExact+TI` | 110,464 | 16 | 0 | 870,055.860 | 870,055.860 |
| `LearnedResidualAnyHit+TI` | 16 | 16 | 0 | 755.489 | 875.153 |
| `ProximityHeuristicAnyHit+TI` | 25 | 16 | 0 | 0.851 | 9.614 |
| `MotionHeuristicAnyHit+TI` | 23 | 16 | 0 | 0.377 | 8.795 |
| `RandomAnyHit+TI` | 23 | 16 | 0 | 1.616 | 2.455 |

The supported claim is exact-call scheduling pressure under a native scene/object envelope protocol: learned residual ordering uses one exact call per positive group and fewer exact calls than the fixed proximity, motion, and random orderings.  This result is not a fixed-heuristic wall-time dominance claim, because candidate hardness differs by ordering and the simple fixed rules hit easier Tight-Inclusion instances in this generated stress scene.

## Paper figure provenance

The paper-facing rendered output is bundled as
`assets/figures/results/result_motion_trap_native_envelope.png`. The
release-local evidence paths are indexed by `artifacts/release_case_manifest.json`.
