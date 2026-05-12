# Datasets Manifest

This manifest currently covers the section 2.1 correctness / CCD layer.

| Source | Local path | Source URL | Version status | Adapter |
|---|---|---|---|---|
| Tight Inclusion | `src/baseline/Tight-Inclusion` | https://github.com/Continuous-Collision-Detection/Tight-Inclusion | local clone | `TightInclusionAdapter` |
| CCD-Wrapper | `src/baseline/CCD-Wrapper` | https://github.com/Continuous-Collision-Detection/CCD-Wrapper | local clone | `CCDWrapperAdapter` |
| Scalable CCD | `src/baseline/Scalable-CCD` | https://github.com/Continuous-Collision-Detection/Scalable-CCD | local clone | source registry |
| Sample Scalable CCD Data | `src/baseline/Sample-Scalable-CCD-Data` | https://github.com/Continuous-Collision-Detection/Sample-Scalable-CCD-Data | local clone | `ScalableCCDSampleAdapter` |
| Exact Root Parity CCD | `src/baseline/Exact-Root-Parity-CCD` | https://github.com/Continuous-Collision-Detection/Exact-Root-Parity-CCD | local clone | `RootParityAdapter` |
| Rigid IPC scenes | `src/baseline/rigid-ipc` | https://ipc-sim.github.io/rigid-ipc/ | local clone with fixtures/meshes | `RigidIPCSceneAdapter` |
| ABC Dataset | `src/datasets/abc` | https://deep-geometry.github.io/abc-dataset/ | user-provided dataset root or local ABC-compatible demo subset | `ABCDatasetAdapter` |
| Fusion 360 Gallery | `src/datasets/fusion360` | https://github.com/AutodeskAILab/Fusion360GalleryDataset | user-provided dataset root | `Fusion360GalleryAdapter` |
| Thingi10K | `src/datasets/thingi10k` | https://github.com/Thingi10K/Thingi10K | local official subset materialized from the `thingi10k` package cache | `Thingi10KAdapter` / `thingi10k_training` |

For reproducible runs, record each clone's commit hash before publishing benchmark numbers.
