# Licensing Manifest

This file is a gate for external correctness benchmark usage. P2CCCD adapters should not run a source in benchmark mode unless license/source metadata is available.

| Source | Local license path | Current availability | Usage note |
|---|---|---:|---|
| Tight Inclusion | `src/baseline/Tight-Inclusion/LICENSE` | available | reference implementation and comparison source |
| CCD-Wrapper | `src/baseline/CCD-Wrapper/LICENSE` | available | benchmark harness reference |
| Scalable CCD | `src/baseline/Scalable-CCD/LICENSE` | available | source/reference stack |
| Sample Scalable CCD Data | `src/baseline/Sample-Scalable-CCD-Data/LICENSE` | available | first runnable external query data |
| Exact Root Parity CCD | `src/baseline/Exact-Root-Parity-CCD/LICENSE` | available | exact narrow-phase reference |
| Rigid IPC scenes | `src/baseline/rigid-ipc/LICENSE` | available | fixture/mesh terms must be preserved in derived correctness cases |
| ABC Dataset | `src/datasets/abc/LICENSE` | user-provided | required before running ABC-derived benchmark suites or redistributing shards |
| Fusion 360 Gallery | `src/datasets/fusion360/LICENSE` | user-provided | required before running Fusion360-derived sequence or assembly-motion suites |
| Thingi10K | `src/datasets/thingi10k/LICENSE` | available | local official subset preserves source metadata and per-model license strings; do not redistribute without checking upstream terms |

Before distributing benchmark artifacts, verify that any copied data, generated shards, and figures comply with the upstream source terms.
