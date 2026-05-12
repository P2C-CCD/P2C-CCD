from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd import BenchmarkRow  # noqa: E402
from p2cccd.bench import (  # noqa: E402
    BenchmarkProfiler,
    benchmark_row_v2_from_legacy,
    bundled_suite_config_dir,
    create_benchmark_run_meta,
    discover_benchmark_run_dirs,
    discover_bundled_suite_configs,
    format_paper_table_markdown,
    load_bundled_suite_config,
    paper_table_rows_from_benchmark_rows,
    read_benchmark_run,
    write_benchmark_run,
)


def _environment() -> dict[str, object]:
    return {
        "git_commit": "abc123",
        "host_name": "runner-host",
        "platform": "runner-platform",
        "python_version": "3.12.0",
        "gpu_name": "Runner GPU",
        "driver_version": "555.00",
        "cuda_version": "12.6",
        "optix_version": "8.0",
        "vram_total_mb": 24000,
        "vram_free_mb": 12000,
    }


def _legacy_row() -> BenchmarkRow:
    return BenchmarkRow(
        query_count=8,
        fn_count=0,
        fp_count=1,
        candidate_recall=1.0,
        avg_candidates=1.5,
        avg_exact_evals=2.0,
        avg_subdivision_depth=0.5,
        fallback_ratio=0.0,
        rt_ms=1.0,
        proposal_ms=0.25,
        exact_ms=2.0,
        total_ms=3.25,
        qps=2461.5,
    )


def test_benchmark_result_io_round_trips_v2_run(tmp_path: Path) -> None:
    meta = create_benchmark_run_meta(
        dataset_name="unit_dataset",
        scene_name="unit_scene",
        method_name="RTExact",
        config={"mode": "unit"},
        seed=5,
        run_id="runner_io_unit",
        environment=_environment(),
    )
    row = benchmark_row_v2_from_legacy(_legacy_row(), meta)

    paths = write_benchmark_run(tmp_path / "run", meta, (row,))
    loaded = read_benchmark_run(paths.run_dir)

    assert loaded.meta.run_id == "runner_io_unit"
    assert loaded.rows == (row,)
    assert discover_benchmark_run_dirs(tmp_path) == (paths.run_dir,)


def test_bundled_suite_config_discovery_and_loading() -> None:
    infos = discover_bundled_suite_configs()
    names = {info.name for info in infos}

    assert bundled_suite_config_dir().exists()
    assert "correctness.json" in names
    assert "ablation.json" in names
    assert all(info.case_count > 0 for info in infos)
    assert load_bundled_suite_config("correctness").suite_name


def test_benchmark_profiler_records_stage_summaries() -> None:
    profiler = BenchmarkProfiler()
    profiler.record("rt_trace", 1.5, {"backend": "unit"})
    profiler.record("rt_trace", 0.5)
    profiler.record("exact", 2.0)

    summary = profiler.summary()

    assert summary.event_count == 3
    assert summary.stage_ms["rt_trace"] == 2.0
    assert summary.stage_ms["exact"] == 2.0
    assert summary.total_ms == 4.0
    assert profiler.latency_samples_ms() == (1.5, 0.5, 2.0)


def test_paper_table_helpers_format_benchmark_rows() -> None:
    meta = create_benchmark_run_meta(
        dataset_name="unit_dataset",
        scene_name="unit_scene",
        method_name="RTExact",
        config={"mode": "unit"},
        seed=5,
        run_id="paper_table_unit",
        environment=_environment(),
    )
    row = benchmark_row_v2_from_legacy(_legacy_row(), meta, latency_samples_ms=(1.0, 2.0, 4.0))

    paper_rows = paper_table_rows_from_benchmark_rows((row,))
    markdown = format_paper_table_markdown(paper_rows, include_scene=True)

    assert paper_rows[0].method_name == "RTExact"
    assert paper_rows[0].latency_p95_ms > 0.0
    assert "| method | dataset | scene |" in markdown
    assert "RTExact" in markdown
