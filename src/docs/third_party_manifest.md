# Third-Party Manifest

This file separates external reference stacks from P2CCCD runtime dependencies.

## First-Layer Correctness / CCD Stack

| Stack | Local path | Role | Runtime dependency |
|---|---|---|---|
| Tight Inclusion | `src/baseline/Tight-Inclusion` | conservative CCD reference implementation | no |
| CCD-Wrapper | `src/baseline/CCD-Wrapper` | benchmark harness and method organization reference | no |
| Scalable CCD | `src/baseline/Scalable-CCD` | scalable broad/narrow CCD reference | no |
| Sample Scalable CCD Data | `src/baseline/Sample-Scalable-CCD-Data` | external correctness query data | no |
| Exact Root Parity CCD | `src/baseline/Exact-Root-Parity-CCD` | exact narrow-phase P1 reference | no |
| Rigid IPC scenes | `src/baseline/rigid-ipc` | rigid-body correctness scenes | no, source adapter only |
| ABC Dataset | `src/datasets/abc` | CAD mesh-pair and hard-negative ingestion | no, user-provided data root |
| Fusion 360 Gallery | `src/datasets/fusion360` | CAD sequence and assembly-motion ingestion | no, user-provided data root |

## Policy

- Keep external repositories under `src/baseline/`.
- Do not add these repositories to the main CMake build until a specific baseline runner requires it.
- Prefer adapter contracts over source-level coupling.
- Keep license files discoverable before a benchmark source is enabled.
- If an external source is unavailable, report it explicitly in the source registry.

## Current P2CCCD Adapter Entry Points

```text
p2cccd.datasets.ccd.discover_first_layer_sources
p2cccd.datasets.ccd.ScalableCCDSampleAdapter
p2cccd.datasets.ccd.TightInclusionAdapter
p2cccd.datasets.ccd.CCDWrapperAdapter
p2cccd.datasets.ccd.RootParityAdapter
p2cccd.datasets.ccd.RigidIPCSceneAdapter
p2cccd.datasets.cad.ABCDatasetAdapter
p2cccd.datasets.cad.Fusion360GalleryAdapter
```
