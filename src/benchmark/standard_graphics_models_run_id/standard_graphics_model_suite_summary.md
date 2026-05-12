# Standard Graphics Model Collision Suite (standard_graphics_models_run_id)

This suite covers Stanford Bunny, Stanford Dragon, Armadillo, Fandisk, Spot the Cow, Suzanne, Utah Teapot, and Cornell Box with a rendered real-mesh rigid-body contact case plus STPF training/inference benchmark rows.

## Model Coverage

| Model | Role | Local source | Case intent |
| --- | --- | --- | --- |
| Stanford Bunny | dynamic rigid falling mesh | `Code/datasets/standard_graphics_models/common_3d_test_models/common-3d-test-models/data/stanford-bunny.obj` | Canonical sanity-check scan with curved local surface detail. |
| Stanford Dragon | dynamic rigid falling mesh | `Code/datasets/standard_graphics_models/common_3d_test_models/common-3d-test-models/data/xyzrgb_dragon.obj` | High-frequency scanned model with many local triangle features. |
| Armadillo | dynamic rigid falling mesh | `Code/datasets/standard_graphics_models/common_3d_test_models/common-3d-test-models/data/armadillo.obj` | Articulated silhouette and concave body regions for contact scheduling. |
| Fandisk | dynamic rigid falling mesh | `Code/datasets/standard_graphics_models/common_3d_test_models/common-3d-test-models/data/fandisk.obj` | Sharp CAD features and non-smooth geometry. |
| Spot the Cow | dynamic rigid falling mesh | `Code/datasets/standard_graphics_models/common_3d_test_models/common-3d-test-models/data/spot.obj` | Clean manifold model with simple topology for topology-correctness checks. |
| Suzanne | dynamic rigid falling mesh | `Code/datasets/standard_graphics_models/common_3d_test_models/common-3d-test-models/data/suzanne.obj` | Low-polygon visual test with mixed sharp and smooth regions. |
| Utah Teapot | dynamic rigid falling mesh | `Code/datasets/standard_graphics_models/common_3d_test_models/common-3d-test-models/data/teapot.obj` | Classic curved-surface graphics model used here as a curved rigid collider. |
| Cornell Box | static scene reference | `Code/datasets/standard_graphics_models/mcguire_archive/CornellBox/CornellBox-Original.obj` | Classic scene container/background reference. |

## Rendered Physics Output

- Case directory: `src/MyDemo/standard_graphics_models_run_id/classic_models_cornell_room_drop`
- MP4: `src\MyDemo\standard_graphics_models_run_id\classic_models_cornell_room_drop\global.mp4`
- Physics: semi-implicit Euler rigid-body drop, unilateral dense triangle floor, restitution, and Coulomb friction.
- Note: Cornell Box is rendered as a static graphics-scene reference; active conservative CCD contacts are dynamic meshes versus the dense floor.

## Training And Inference

- Shards: `src/datasets/training/standard_graphics_models_run_id/shards`
- Train rows: `16384`
- Validation rows: `8192`
- Device: `cuda`
- Final validation interval top-1 recall: `0.9324951171875`
- Final validation family top-2 recall: `0.25`
- Calibrated zero-FN threshold: `0.4731125012040138`

## Benchmark Cases

| Case | Dataset/model | Dense exact calls | RTSTPF exact calls | Reduction | Recall | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `bunny_frictional_floor_drop` | Stanford Bunny | 589824000 | 1024 | 576000.00x | 1.000 | 0 |
| `dragon_frictional_floor_drop` | Stanford Dragon | 589824000 | 903 | 653182.72x | 1.000 | 0 |
| `armadillo_frictional_floor_drop` | Armadillo | 589824000 | 1024 | 576000.00x | 1.000 | 0 |
| `fandisk_frictional_floor_drop` | Fandisk | 318160896 | 1024 | 310704.00x | 1.000 | 0 |
| `spot_frictional_floor_drop` | Spot the Cow | 143917056 | 1024 | 140544.00x | 1.000 | 0 |
| `suzanne_frictional_floor_drop` | Suzanne | 23789568 | 1024 | 23232.00x | 1.000 | 0 |
| `teapot_frictional_floor_drop` | Utah Teapot | 155320320 | 1024 | 151680.00x | 1.000 | 0 |
| `cornell_box_multimodel_scene_drop` | Cornell Box + seven dynamic standard graphics models | 2411544576 | 1024 | 2355024.00x | 1.000 | 0 |

## Interpretation

The strongest advantage appears when high face-count scan/CAD models share a dense support surface: dense exact CCD scales with all object-ground triangle pairs, while the trained proposal keeps exact checks concentrated around the physically generated contact windows. The calibrated threshold is reported with zero false negatives on the validation split.
