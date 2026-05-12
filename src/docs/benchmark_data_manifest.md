# Benchmark Data Manifest

This manifest records the external data/repository layer currently expected under:

```text
src/baseline/
```

The external repositories remain reference inputs. They should not be blindly vendored into the main P2CCCD build.

## First-Layer Correctness Sources

| Source | Local directory | Priority | Current role | Adapter status |
|---|---|---:|---|---|
| Tight Inclusion | `baseline/Tight-Inclusion` | P0 | conservative CCD reference source | source adapter implemented |
| CCD-Wrapper | `baseline/CCD-Wrapper` | P0 | benchmark harness reference | harness adapter implemented |
| Scalable CCD | `baseline/Scalable-CCD` | P0 | scalable CCD source/reference | registry detection implemented |
| Sample Scalable CCD Data | `baseline/Sample-Scalable-CCD-Data` | P0 | runnable external query data | query adapter implemented |
| Exact Root Parity CCD | `baseline/Exact-Root-Parity-CCD` | P1 | exact narrow-phase reference | source adapter implemented |
| Rigid IPC scenes | `baseline/rigid-ipc` | P1 | rigid-body correctness scenes | scene adapter implemented |
| ABC Dataset | `datasets/abc` | P1 | CAD-derived mesh pairs and industrial hard negatives | CAD ingestion adapter implemented; data root is user-provided |
| Fusion 360 Gallery | `datasets/fusion360` | P1 | human-designed CAD sequences and assembly-style motion | CAD sequence adapter implemented; data root is user-provided |

## Runnable Query Source

The first runnable external query adapter is:

```text
p2cccd.datasets.ccd.ScalableCCDSampleAdapter
```

It reads:

- `queries/*.csv` rational-coordinate query files,
- `mma_bool/*_mma_bool.json` Mathematica boolean collision labels,
- `boxes/*.json` broad-phase box-pair metadata,
- `frames/*` mesh frame paths when file names are directly mappable.

The adapter currently supports both Scalable CCD query families:

- `vf` as point-triangle,
- `ee` as edge-edge.

Additional P1 ingestion adapters:

- `p2cccd.datasets.ccd.RigidIPCSceneAdapter` indexes Rigid-IPC fixtures and meshes, extracts rigid body metadata, and emits proxy body-pair `DatasetQueryBatch` rows with unknown labels for correctness stress triage.
- `p2cccd.datasets.cad.ABCDatasetAdapter` ingests CAD mesh assets, sidecar patch metadata, mesh statistics, and deterministic industrial hard-negative mesh pairs.
- `p2cccd.datasets.cad.Fusion360GalleryAdapter` ingests human-designed CAD sequences, assembly metadata, part pairs, and deterministic linear assembly-approach motion samples.

## Reproducibility Checklist

Every external benchmark run should record:

- git commit of P2CCCD,
- source repository directory name,
- source repository commit if available,
- adapter schema version,
- scene name,
- batch id,
- query count,
- known label count,
- collision label count,
- config hash,
- random seed if sampling is involved,
- GPU/driver/CUDA/OptiX versions when the runner uses GPU paths,
- output CSV/JSONL path,
- failure case query ids for false negative or certificate mismatch triage.
