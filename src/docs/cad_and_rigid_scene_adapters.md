# CAD And Rigid Scene Adapters

This note covers TODO 143-150.

## Rigid-IPC

Runtime adapter:

```text
src/python/p2cccd/datasets/ccd/rigid_ipc_adapter.py
```

The adapter treats `src/baseline/rigid-ipc` as a source repository and does not vendor it into the P2CCCD build. It provides:

- fixture and mesh discovery,
- source license metadata,
- `DatasetScene` indexing for Rigid-IPC JSON fixtures,
- rigid body metadata extraction from `rigid_body_problem.rigid_bodies`,
- conservative body radius estimates from referenced meshes or inline vertices,
- proxy body-pair `DatasetQueryBatch` generation for correctness stress triage.

The generated body-pair batches have `ground_truth_collides=None`. They are not official Rigid-IPC CCD query labels; they are traceable proxy inputs that preserve fixture/body identity until a full trajectory replay path is implemented.

Visualization helper:

```text
src/python/p2cccd/viz/rigid_ipc_views.py
```

The Rigid-IPC debug view renders a 2D projected scene overview with body radius proxies, moving/static body status, velocity arrows, and generated body-pair query edges. A generated example is:

```text
src/outputs/rigid_ipc_chain_3_links_overview.html
```

## ABC Dataset

Runtime adapter:

```text
src/python/p2cccd/datasets/cad/abc_adapter.py
```

Default local root:

```text
src/datasets/abc
```

The adapter ingests supported mesh files (`obj`, `off`, ASCII `stl`, ASCII `ply`), computes dependency-free mesh statistics, loads optional patch sidecars such as `*.patch.json` and `*.patches.json`, and ranks deterministic industrial hard-negative mesh pairs by scale/topology similarity.

Training / shard pipeline:

```text
src/python/p2cccd/datasets/cad/abc_training.py
src/python/p2cccd/bench/abc_training.py
```

The current pipeline supports two modes:

- user-provided real `ABC` mesh root under `src/datasets/abc`, or
- an automatically bootstrapped local ABC-compatible demo subset when the official
  multi-GB object chunks are unavailable locally.

It generates deterministic CAD mesh-pair proxy motions, exports base and dense
training shards, and runs STPF training plus high-density eval benchmark.

## Fusion 360 Gallery

Runtime adapter:

```text
src/python/p2cccd/datasets/cad/fusion360_adapter.py
```

Default local root:

```text
src/datasets/fusion360
```

The adapter indexes sequence directories, loads part meshes plus optional `assembly.json`, generates deterministic part pairs, and emits simple linear assembly-approach motion samples. These samples are intended for downstream performance and proposal-pipeline workbench generation, not as certified ground truth.

## Better STEP

Runtime adapter:

```text
src/python/p2cccd/datasets/cad/better_step_adapter.py
```

Default local root:

```text
src/datasets/better_step
```

The adapter is a CAD-native preprocessing bridge. It discovers `.step` and `.stp` files, extracts lightweight STEP header/product/schema/unit metadata, loads optional JSON sidecars, and emits `StepPreprocessRecord` rows that can later drive a meshing backend before `npz` or parquet shard generation. It intentionally does not pretend to produce triangle meshes by itself.

## YCB And Google Scanned Objects

Runtime adapters:

```text
src/python/p2cccd/datasets/objects/ycb_gso_adapter.py
```

Default local roots:

```text
src/datasets/ycb
src/datasets/google_scanned_objects
```

These adapters index manipulation-object or scanned-object meshes and generate deterministic robot-validation object-pair motion samples. They are for downstream OOD/fallback and robot-scene validation workbench generation.

## MoveIt Resources

Runtime adapter:

```text
src/python/p2cccd/datasets/robot/moveit_adapter.py
```

Default local root:

```text
src/datasets/moveit_resources
```

The adapter indexes lightweight MoveIt-style planning scene files, nearby mesh assets, robot names, and link lists, then emits deterministic `RobotMotionQuery` rows for Panda/UR/Fanuc-style planning scene smoke tests.

## Thingi10K

Runtime adapter:

```text
src/python/p2cccd/datasets/objects/thingi10k_adapter.py
```

Default local root:

```text
src/datasets/thingi10k
```

The adapter ranks dirty/OOD meshes by filename and metadata hints such as `dirty`, `non_manifold`, and `self_intersections`, then emits stress samples carrying proxy-inflation and fallback metadata.

Official subset materialization plus proxy-training pipeline:

```text
src/python/p2cccd/datasets/objects/thingi10k_training.py
src/python/p2cccd/bench/thingi10k_training.py
src/python/p2cccd/bench/thingi10k_paper_benchmark.py
```

The current Thingi10K path now supports:

- local materialization of an official `thingi10k` package subset with sidecar metadata,
- deterministic dirty/OOD proxy motion generation with held-out train/eval separation,
- STPF training shard export under `datasets/training/ood_train/thingi10k`,
- held-out paper benchmark export under `datasets/benchmark/ood_stress/thingi10k`,
- high-density hard-case evaluation for learned-STPF exact-work reduction.

## PartNet And PartNet-Mobility

Runtime adapter:

```text
src/python/p2cccd/datasets/objects/partnet_adapter.py
```

Default local roots:

```text
src/datasets/partnet
src/datasets/partnet_mobility
```

PartNet ingestion records part-aware mesh metadata. PartNet-Mobility ingestion discovers articulated scenes, mobility sidecars, joint counts, and part-pair motion samples for extension-scene validation.

## ShapeNet And Objaverse-XL

Runtime adapter:

```text
src/python/p2cccd/datasets/objects/shapenet_objaverse_adapter.py
```

Default local roots:

```text
src/datasets/shapenet
src/datasets/objaverse_xl
```

These adapters are intentionally subset-oriented. They index local mesh subsets, preserve category/object metadata, and emit large-scale OOD object-pair samples with no-false-negative fallback semantics. They should be used after the P0/P1 adapters are stable and after source-license metadata is present locally.

## Reproducibility Rules

- Adapters expose license metadata and should not be used for publishable benchmark artifacts without an available local license or upstream terms record.
- Generated rows must retain source-relative paths, scene or sequence names, and pair/sample ids.
- CAD adapters are ingestion and sampling infrastructure. Final SIGGRAPH/TOG benchmark claims still need external dataset version pins, exact query replay, and benchmark-suite configs.
