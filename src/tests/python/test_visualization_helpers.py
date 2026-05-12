from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.candidate_generation import generate_candidates_for_generated_dataset  # noqa: E402
from p2cccd.certificate_engine import execute_certificate_engine_for_generated_dataset  # noqa: E402
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset  # noqa: E402
from p2cccd.viz import (  # noqa: E402
    candidate_density_by_slab,
    render_candidate_density_svg,
    render_certificate_trace_svg,
    render_exact_work_svg,
    summarize_certificate_trace,
    summarize_exact_work,
    write_pipeline_debug_html,
)


def _pipeline_result():
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=1, robot_link_count=1, seed=515)
    )
    candidates = generate_candidates_for_generated_dataset(dataset)
    exact = execute_certificate_engine_for_generated_dataset(dataset, candidates)
    return candidates, exact


def test_visualization_summaries_cover_candidate_exact_and_certificate_data() -> None:
    candidates, exact = _pipeline_result()

    density = candidate_density_by_slab(candidates.candidates)
    work_summary = summarize_exact_work(exact.work_items)
    certificate_summary = summarize_certificate_trace(exact.certificates)

    assert sum(entry.candidate_count for entry in density) == len(candidates.candidates)
    assert work_summary.work_item_count == len(exact.work_items)
    assert certificate_summary.certificate_count == len(exact.certificates)
    assert certificate_summary.collision_count + certificate_summary.separation_count >= 1


def test_visualization_helpers_render_svg_and_html(tmp_path: Path) -> None:
    candidates, exact = _pipeline_result()

    candidate_svg = render_candidate_density_svg(candidates.candidates)
    work_svg = render_exact_work_svg(exact.work_items)
    certificate_svg = render_certificate_trace_svg(exact.certificates)
    html_path = write_pipeline_debug_html(
        tmp_path / "pipeline.html",
        candidates=candidates.candidates,
        work_items=exact.work_items,
        certificates=exact.certificates,
    )

    assert "<svg" in candidate_svg and "Candidate Density" in candidate_svg
    assert "<svg" in work_svg and "Exact Work" in work_svg
    assert "<svg" in certificate_svg and "Certificate Trace" in certificate_svg
    assert html_path.exists()
    assert "P2CCCD Pipeline Debug View" in html_path.read_text(encoding="utf-8")
