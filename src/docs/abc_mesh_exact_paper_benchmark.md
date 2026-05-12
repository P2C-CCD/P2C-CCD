# ABC Real Mesh-Mesh Exact CCD Benchmark: Paper-Path Comparison

`python/p2cccd/bench/abc_mesh_exact_paper_benchmark.py` runs a paper-path style
comparison on the real `mesh-mesh exact CCD` benchmark introduced in
`abc_mesh_exact_benchmark.py`.

## Compared methods

- `PureMeshExactCPU`
- `BVHExact`
- `RTExact`
- `NoProposal`
- `RTSTPFExact-Random`
- `RTSTPFExact-Trained` when an ABC proxy-CAD checkpoint is available

## Current scope

- Uses the same real `OBJ/STL` mesh-motion queries as the mesh exact benchmark.
- Ground truth is cached from the real exact benchmark path and reused by all
  compared methods.
- `BVHExact / RTExact / NoProposal / RTSTPFExact` currently use query-level
  whole-mesh swept-AABB candidate generation before entering the exact
  mesh-mesh certificate stage.
- `FN` and `Recall` are computed only over `exact collision certificates`.
  `undecided` queries are excluded from the denominator.

## Why this benchmark is useful

It lets the repository compare the paper pipeline shape against other paths on
real mesh exact queries without falling back to swept-sphere proxy labels.

It does **not** yet represent the final full-mesh patch-level RT candidate
pipeline. Candidate generation is still query-level on whole-mesh swept AABBs.

## Reproduction

Demo subset:

```powershell
conda activate cudadev
python -c "import sys; from pathlib import Path; sys.path.insert(0, r'src\python'); from p2cccd.bench.abc_mesh_exact_benchmark import ABCMeshExactBenchmarkConfig; from p2cccd.bench.abc_mesh_exact_paper_benchmark import ABCMeshExactPaperBenchmarkConfig, run_abc_mesh_exact_paper_benchmark; cfg = ABCMeshExactPaperBenchmarkConfig(exact_benchmark=ABCMeshExactBenchmarkConfig(root=Path(r'src\datasets\abc'), allow_demo_bootstrap=True, benchmark_asset_offset=24, benchmark_asset_count=24, pair_limit=40, max_faces_per_mesh=256, benchmark_output_dir='src/benchmark', benchmark_dataset_dir='src/datasets/benchmark/cad_motion_bench', run_name='abc_mesh_exact_demo_ground_truth_run_id'), rt_backend_name='optix_compatible', run_name='abc_mesh_exact_paper_comparison_demo_run_id'); print(run_abc_mesh_exact_paper_benchmark(cfg).artifacts.report_path)"
```

Official minimal root:

```powershell
conda activate cudadev
python -c "import sys; from pathlib import Path; sys.path.insert(0, r'src\python'); from p2cccd.bench.abc_mesh_exact_benchmark import ABCMeshExactBenchmarkConfig; from p2cccd.bench.abc_mesh_exact_paper_benchmark import ABCMeshExactPaperBenchmarkConfig, run_abc_mesh_exact_paper_benchmark; cfg = ABCMeshExactPaperBenchmarkConfig(exact_benchmark=ABCMeshExactBenchmarkConfig(root=Path(r'src\datasets\abc_official'), use_official_root=True, allow_official_download=False, benchmark_asset_offset=24, benchmark_asset_count=24, pair_limit=40, max_faces_per_mesh=256, benchmark_output_dir='src/benchmark', benchmark_dataset_dir='src/datasets/benchmark/cad_motion_bench', run_name='abc_mesh_exact_official_ground_truth_run_id'), rt_backend_name='optix_compatible', run_name='abc_mesh_exact_paper_comparison_official_run_id'); print(run_abc_mesh_exact_paper_benchmark(cfg).artifacts.report_path)"
```
