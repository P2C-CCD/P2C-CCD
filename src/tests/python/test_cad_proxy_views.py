from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench.abc_paper_benchmark import (  # noqa: E402
    ABCPaperBenchmarkConfig,
    build_abc_paper_benchmark_dataset,
)
from p2cccd.bench.bvh_exact import _try_load_p2cccd_cpp  # noqa: E402
from p2cccd.viz import write_abc_cad_collision_animation_html  # noqa: E402


def _require_cpp() -> None:
    cpp = _try_load_p2cccd_cpp()
    if cpp is None or not hasattr(cpp, "load_triangle_mesh"):
        pytest.skip("CAD proxy visualization bindings are not built")


def test_write_abc_cad_collision_animation_html(tmp_path: Path) -> None:
    _require_cpp()
    dataset = build_abc_paper_benchmark_dataset(
        ABCPaperBenchmarkConfig(
            root=tmp_path / "abc",
            allow_demo_bootstrap=True,
            benchmark_asset_offset=0,
            benchmark_asset_count=6,
            pair_limit=1,
            benchmark_output_dir=str(tmp_path / "benchmark"),
            benchmark_dataset_dir=str(tmp_path / "datasets"),
            run_name="unit_abc_cad_proxy_animation",
        )
    )
    output = tmp_path / "benchmark" / "cad_proxy_animation.html"
    write_abc_cad_collision_animation_html(
        dataset.dataset_manifest_path,
        output,
        pair_id=dataset.pair_ids[0],
        split="near_contact_hard_negatives",
    )
    html = output.read_text(encoding="utf-8")
    assert output.exists()
    assert "BounceReplay" in html
    assert "contact interval" in html
