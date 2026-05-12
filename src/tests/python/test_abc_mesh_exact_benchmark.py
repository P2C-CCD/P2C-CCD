from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench.abc_mesh_exact_benchmark import (
    ABCMeshExactBenchmarkConfig,
    build_abc_mesh_exact_benchmark_dataset,
    run_abc_mesh_exact_benchmark,
)
from p2cccd.bench.bvh_exact import _try_load_p2cccd_cpp


def _require_cpp() -> None:
    cpp = _try_load_p2cccd_cpp()
    if cpp is None or not hasattr(cpp, "build_mesh_exact_certificate_query"):
        pytest.skip("mesh exact benchmark bindings are not built")


def test_build_abc_mesh_exact_benchmark_dataset_writes_manifest(tmp_path: Path) -> None:
    _require_cpp()
    config = ABCMeshExactBenchmarkConfig(
        root=tmp_path / "abc",
        allow_demo_bootstrap=True,
        benchmark_asset_offset=0,
        benchmark_asset_count=6,
        pair_limit=1,
        max_faces_per_mesh=32,
        benchmark_output_dir=str(tmp_path / "benchmark"),
        benchmark_dataset_dir=str(tmp_path / "datasets"),
        run_name="unit_mesh_exact_dataset",
    )
    dataset = build_abc_mesh_exact_benchmark_dataset(config)
    assert dataset.manifest_path.exists()
    assert dataset.queries_jsonl_path.exists()
    assert len(dataset.queries) == 4
    assert len(dataset.pair_ids) == 1


def test_run_abc_mesh_exact_benchmark_on_demo_subset(tmp_path: Path) -> None:
    _require_cpp()
    config = ABCMeshExactBenchmarkConfig(
        root=tmp_path / "abc",
        allow_demo_bootstrap=True,
        benchmark_asset_offset=0,
        benchmark_asset_count=6,
        pair_limit=1,
        max_faces_per_mesh=32,
        benchmark_output_dir=str(tmp_path / "benchmark"),
        benchmark_dataset_dir=str(tmp_path / "datasets"),
        run_name="unit_mesh_exact_run",
    )
    result = run_abc_mesh_exact_benchmark(config)
    assert result.benchmark.query_count == 4
    assert result.benchmark.candidate_recall == pytest.approx(1.0)
    assert result.artifacts.report_path.exists()
    assert result.artifacts.summary_json_path.exists()
    assert any(query.point_triangle_total_pairs > 0 for query in result.query_results)
    assert any(query.edge_edge_total_pairs > 0 for query in result.query_results)
