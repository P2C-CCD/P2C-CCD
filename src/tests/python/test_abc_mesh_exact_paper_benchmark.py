from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench.abc_mesh_exact_paper_benchmark import (  # noqa: E402
    ABCMeshExactPaperBenchmarkConfig,
    run_abc_mesh_exact_paper_benchmark,
)
from p2cccd.bench.abc_mesh_exact_benchmark import ABCMeshExactBenchmarkConfig  # noqa: E402
from p2cccd.bench.bvh_exact import _try_load_p2cccd_cpp  # noqa: E402


def _require_cpp() -> None:
    cpp = _try_load_p2cccd_cpp()
    if cpp is None or not hasattr(cpp, "build_mesh_exact_certificate_query"):
        pytest.skip("mesh exact benchmark bindings are not built")


def test_run_abc_mesh_exact_paper_benchmark_demo_smoke(tmp_path: Path) -> None:
    _require_cpp()
    config = ABCMeshExactPaperBenchmarkConfig(
        exact_benchmark=ABCMeshExactBenchmarkConfig(
            root=tmp_path / "abc",
            allow_demo_bootstrap=True,
            benchmark_asset_offset=0,
            benchmark_asset_count=6,
            pair_limit=1,
            max_faces_per_mesh=32,
            benchmark_output_dir=str(tmp_path / "benchmark"),
            benchmark_dataset_dir=str(tmp_path / "datasets"),
            run_name="unit_mesh_exact_paper_ground_truth",
        ),
        benchmark_output_dir=str(tmp_path / "benchmark"),
        run_name="unit_mesh_exact_paper_benchmark",
        include_random_stpf=True,
        include_trained_stpf=False,
        rt_backend_name="cpu_reference_rt",
        model_device="cpu",
    )
    result = run_abc_mesh_exact_paper_benchmark(config)
    assert result.pure_mesh_exact.benchmark.query_count == 4
    assert result.bvh_exact.benchmark.fn_count == 0
    assert result.rt_exact.benchmark.candidate_recall == pytest.approx(1.0)
    assert result.no_proposal.benchmark.fn_count == 0
    assert result.rtstpf_random is not None
    assert result.artifacts.report_path.exists()
    assert result.artifacts.summary_json_path.exists()
