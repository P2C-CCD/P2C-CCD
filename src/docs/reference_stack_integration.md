# Reference Stack Integration

This note records the local reference/dependency status for the current P2CCCD plan.

## Local Dependency Map

| stack | local path | current role | integration boundary |
| --- | --- | --- | --- |
| Tight Inclusion | `src/baseline/Tight-Inclusion` and `src/lib/Tight-Inclusion` | conservative CCD reference and correctness oracle comparison | Python adapter discovers entry points; no vendored runtime call inside P2CCCD yet |
| CCD-Wrapper | `src/baseline/CCD-Wrapper` | benchmark harness and method organization reference | adapter mirrors method naming; not linked into core runtime |
| Scalable CCD | `src/baseline/Scalable-CCD` | broad/narrow benchmark reference | sample data adapter is wired first; full dataset remains external input |
| Sample Scalable CCD Data | `src/baseline/Sample-Scalable-CCD-Data` | analytic ground-truth query batches | runnable through `correctness_external.json` when local data is present |
| Exact Root Parity CCD | `src/baseline/Exact-Root-Parity-CCD` | exact narrow-phase reference | reference-only adapter; not a main runtime path |
| Rigid-IPC | `src/baseline/rigid-ipc` | complex rigid-body correctness scenes | fixture/mesh discovery and body-pair query batch generation are implemented |
| IPC Toolkit | `src/lib/ipc-toolkit` | robust geometry and IPC reference | reference dependency; exact certificate engine remains project-local |
| Embree | `src/lib/embree` | future CPU broad-phase backend for `BVHExact` | adapter plan below; current backend is deterministic CPU AABB sweep |
| Coal | not currently found under `src/lib` or optional external SDK roots | modern robotics collision stack target | preferred over new FCL/hpp-fcl planning references; install before real robot broad-phase integration |
| Pinocchio | not currently found under `src/lib` or optional external SDK roots | robot kinematics source for downstream validation | future adapter dependency |
| Tesseract | not currently found under `src/lib` or optional external SDK roots | industrial planning scene source | future adapter dependency |
| MoveIt benchmark resources | not currently found under `src/lib` or optional external SDK roots | Panda/UR/Fanuc-style planning scenes | future dataset adapter dependency |
| OptiX Toolkit / OptiX Apps | `$env:P2CCCD_OPTIX_ROOT` and `src/lib/OptiX_Apps` | reusable OptiX helper utilities and examples | keep behind optional RT candidate backend; do not change current candidate contracts |
| TensorRT / ONNX Runtime GPU | `$env:P2CCCD_TENSORRT_ROOT`, Python ONNX Runtime GPU | future proposal inference acceleration | Python STPF path remains CPU/GPU PyTorch first; export to ONNX/TensorRT later |
| RTCollisionDetection | not currently found under local dependency roots | RT-style baseline reference | do not block current RTExact/STPF baseline |

## Coal Policy

For robotics collision-stack planning, use Coal as the modern target instead of adding new FCL or hpp-fcl integration work. Existing docs may mention Embree/Coal-compatible naming because the current `BVHExact` abstraction can be backed by either a generic ray/box accelerator or a robotics collision stack later. New robot-scene code should name the target as Coal unless a legacy comparison explicitly requires FCL or hpp-fcl.

## Embree-Backed BVHExact Adapter Plan

Current state:

- `BVHExact` uses a deterministic CPU swept-AABB broad phase.
- The config names `embree_compatible` and `coal_compatible` are compatibility labels only.
- No Embree scene is built by the runner yet.

Minimal Embree adapter design:

- Keep `BVHExactConfig.backend_name` stable.
- Add a C++ broad-phase backend that converts each swept proxy AABB to an Embree user geometry or bounding box primitive.
- Keep narrow phase unchanged: Embree only emits candidate pairs, then `PureExactCPU` / certificate engine resolves them.
- Add a CPU cross-check: Embree candidate set must be a superset of the deterministic CPU AABB sweep.
- Export broad-phase counters and timing through existing `BenchmarkRowV2` RT/profiling fields.

Acceptance gates:

- candidate recall against CPU oracle remains 1.0;
- final FN remains 0;
- Embree candidate pairs are deterministic under fixed input order or explicitly sorted before downstream processing;
- CPU-only tests still pass without Embree installed.

## OptiX Toolkit Review Note

The local `OptiX_Apps` tree is useful for:

- context/device setup patterns;
- shader binding table layout examples;
- acceleration structure build/update examples;
- pipeline compile options and module logging;
- launch parameter organization.

Do not import an example app structure directly into P2CCCD. The P2CCCD RT candidate contract is already defined around proxy scene build, raw hit emission, compaction, and `CandidateRecord`. Any OptiX helper reuse must sit behind the optional backend gate and preserve:

- `RawCandidateHit` layout;
- `CandidateRecord` schema;
- candidate recall = 1.0 correctness gate;
- RT build/update/trace timing split;
- CPU fallback path for CI.
