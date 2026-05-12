# ABC Real Mesh-Mesh Exact CCD Benchmark

## Purpose

this benchmark benchmark userealdescriptionreplacedescription `abc_paper_benchmark.py` in
`swept-sphere proxy oracle`. descriptionused fordescription:

- descriptionreal CAD mesh A / B,
- description `[t0, t1]` within translation-only descriptiontrajectory,
- usecurrent `CertificateEngine`  `point-triangle + edge-edge` exact primitive
  Path, isdescriptionconstructdescriptionreal `mesh-mesh exact CCD` description benchmark.

## currentdescription

### C++ description

- `cpp/geometry/mesh_io.h`
- `cpp/geometry/mesh_io.cpp`
- `cpp/certificate/mesh_exact_query.h`
- `cpp/certificate/mesh_exact_query.cpp`
- `cpp/bindings/py_module.cpp`

addeddescription:

- `load_triangle_mesh(path)`
  - support `OBJ`
  - support `ASCII STL`
  - support `binary STL`
- `center_mesh_at_aabb_center(mesh)`
  - description mesh with AABB center translated to the local origin
- `build_mesh_exact_certificate_query(...)`
  - fromdescriptionreal mesh anddescriptiontrajectoryconstruct `ExactCertificateQuery`
  - descriptionwhengenerate:
    - `point_triangle_primitives`
    - `edge_edge_primitives`
  - defaultdescriptionuse conservative swept-AABB pruning

### Python runner

- `python/p2cccd/bench/abc_mesh_exact_benchmark.py`

description:

1. from `ABC demo root` or `official minimal root` select held-out mesh
2. generate CAD pair
3. descriptionusedescriptionhas CAD motion sampler  translation-only query trajectory
4. call C++ pybind:
   - `load_triangle_mesh`
   - `center_mesh_at_aabb_center`
   - `build_mesh_exact_certificate_query`
   - `evaluate_certificate_query_cpu`
5. Output benchmark reportand dataset manifest

## Query description

this benchmark exact benchmark anddescription proxy benchmark description:

- descriptionPath:
  - CAD mesh descriptionused forgenerate proxy radius / stats
  - oracle is `evaluate_swept_sphere_oracle`
- newPath:
  - CAD mesh descriptionrealloaddescription triangle mesh
  - exact query description cross-mesh `point-triangle` and `edge-edge` primitive description
  - descriptionfrom `CertificateEngine`

## descriptionModel

currentdescriptionis:

- **translation-only rigid motion**

descriptionperformdescription:

1. mesh descriptionby AABB center descriptionin
2. then overlays benchmark query :
   - `translation_a_t0 -> translation_a_t1`
   - `translation_b_t0 -> translation_b_t1`

descriptionisreal `mesh-mesh exact CCD benchmark`,
butdescriptionis notdescription benchmark.

## Output

datasetOutput:

- `src/datasets/benchmark/cad_motion_bench/<run_name>/dataset_manifest.json`
- `src/datasets/benchmark/cad_motion_bench/<run_name>/queries.jsonl`

benchmark Output:

- `src/benchmark/<run_name>.md`
- `src/benchmark/<run_name>.json`

## currentdescription

- `src/benchmark/abc_mesh_exact_benchmark_demo_run_id.md`
- `src/benchmark/abc_mesh_exact_benchmark_official_run_id.md`

## statisticsProtocol notes

reportdescriptionhasdescriptionstatistics:

- `exact collision certificates`
  - `CertificateEngine` descriptionconnectOutput `collision`
- `conservative-positive queries`
  - `collision`
  - descriptiononby conservative policy description `undecided`

thereforedescription.

## currentdescription

1. descriptionsupport translation-only, description rigid rotation.
2. description benchmark Pathis exact baseline, is not RT/STPF descriptiontodescription.
3. conservative swept-AABB pruning descriptionis primitive builder description, is notcomplete broad phase.
4. description exact certificate enter `undecided` when, current benchmark descriptionby conservative-positive description.

## description

```powershell
conda activate cudadev
python -c "import sys; sys.path.insert(0, r'src\python'); from p2cccd.bench.abc_mesh_exact_benchmark import ABCMeshExactBenchmarkConfig, run_abc_mesh_exact_benchmark; result = run_abc_mesh_exact_benchmark(ABCMeshExactBenchmarkConfig(run_name='abc_mesh_exact_benchmark_manual')); print(result.artifacts.report_path)"
```

official root description:

```powershell
conda activate cudadev
python -c "import sys; from pathlib import Path; sys.path.insert(0, r'src\python'); from p2cccd.bench.abc_mesh_exact_benchmark import ABCMeshExactBenchmarkConfig, run_abc_mesh_exact_benchmark; root = Path(r'src\datasets\abc_official'); result = run_abc_mesh_exact_benchmark(ABCMeshExactBenchmarkConfig(root=root, use_official_root=True, run_name='abc_mesh_exact_benchmark_official_manual')); print(result.artifacts.report_path)"
```
