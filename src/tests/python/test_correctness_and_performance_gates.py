from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd import BenchmarkRowV2, read_jsonl, validate_benchmark_row_v2  # noqa: E402
from p2cccd.bench import (  # noqa: E402
    BenchmarkSuiteCaseConfig,
    BenchmarkSuiteConfig,
    BenchmarkSuiteDatasetConfig,
    load_benchmark_suite_config,
    run_benchmark_suite,
    run_benchmark_suite_from_config_path,
    validate_benchmark_suite_config,
)


CONFIG_ROOT = PROJECT_ROOT / "configs" / "benchmark_suites"


def _cpu_only_environment() -> dict[str, object]:
    return {
        "git_commit": "ci-test",
        "host_name": "ci-minimal",
        "platform": "cpu-only",
        "python_version": "3.12.0",
        "gpu_name": "not_required",
        "driver_version": "not_required",
        "cuda_version": "not_required",
        "optix_version": "not_required",
        "vram_total_mb": 0,
        "vram_free_mb": 0,
    }


def test_correctness_suite_final_fn_zero_gate(tmp_path: Path) -> None:
    result = run_benchmark_suite_from_config_path(
        CONFIG_ROOT / "correctness.json",
        output_root=tmp_path,
        environment=_cpu_only_environment(),
        run_id="correctness_fn_zero_gate",
    )

    assert {case.method for case in result.case_results} == {"PureExactCPU", "BVHExact", "RTExact"}
    assert all(case.final_fn_zero for case in result.case_results)
    assert all(row.fn_count == 0 for row in result.rows)
    assert all(row.candidate_recall == 1.0 for row in result.rows)

    assert result.export_paths is not None
    exported_rows = list(read_jsonl(result.export_paths.jsonl_path, BenchmarkRowV2))
    assert exported_rows == list(result.rows)
    assert all(validate_benchmark_row_v2(row) is row for row in exported_rows)


def test_performance_smoke_exports_timing_breakdown(tmp_path: Path) -> None:
    suite = BenchmarkSuiteConfig(
        schema_version=1,
        suite_name="performance_timing_smoke",
        suite_type="performance",
        seed=41,
        output_root=str(tmp_path),
        notes="unit-level performance timing export smoke",
        cases=(
            BenchmarkSuiteCaseConfig(
                name="rt_exact_timing",
                method="RTExact",
                dataset=BenchmarkSuiteDatasetConfig(
                    mesh_count_per_split=1,
                    robot_link_count=0,
                    include_robot_links=False,
                ),
                config={"backend_name": "cpu_reference_rt"},
            ),
            BenchmarkSuiteCaseConfig(
                name="queue_decouple_timing",
                method="NoQueueDecouple",
                config={
                    "candidate_count": 128,
                    "repeat_count": 2,
                    "warmup_count": 0,
                    "batch_size": 64,
                    "seed": 41,
                },
            ),
        ),
    )

    result = run_benchmark_suite(
        suite,
        environment=_cpu_only_environment(),
        run_id="performance_timing_smoke",
    )

    assert result.export_paths is not None
    exported_rows = list(read_jsonl(result.export_paths.jsonl_path, BenchmarkRowV2))
    assert exported_rows == list(result.rows)

    rt_row = next(row for row in exported_rows if row.method_name == "RTExact")
    assert rt_row.rt_build_ms >= 0.0
    assert rt_row.rt_update_ms >= 0.0
    assert rt_row.rt_trace_ms >= 0.0
    assert rt_row.rt_ms == pytest.approx(rt_row.rt_build_ms + rt_row.rt_update_ms + rt_row.rt_trace_ms)
    assert rt_row.latency_p50_ms >= 0.0
    assert rt_row.latency_p95_ms >= rt_row.latency_p50_ms
    assert rt_row.total_ms >= rt_row.rt_ms
    assert rt_row.candidate_inflation_ratio >= 0.0
    assert rt_row.exact_queue_occupancy >= 0.0

    no_queue_rows = [row for row in exported_rows if row.method_name.startswith("NoQueueDecouple:")]
    assert no_queue_rows
    assert all(row.rt_trace_ms >= 0.0 for row in no_queue_rows)
    assert all(row.total_ms >= row.rt_ms for row in no_queue_rows)
    assert all(row.candidate_buffer_bandwidth_mb_s >= 0.0 for row in no_queue_rows)
    assert all(row.total_tail_latency_ms >= 0.0 for row in no_queue_rows)
    queue_row = next(row for row in no_queue_rows if row.method_name.endswith("queue_decoupled_batch_proposal"))
    assert queue_row.proposal_enqueue_dequeue_ms > 0.0

    with result.export_paths.csv_path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    for field_name in (
        "rt_build_ms",
        "rt_update_ms",
        "rt_trace_ms",
        "rt_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "exact_calls_total",
        "candidate_inflation_ratio",
        "undecided_to_resolved_ratio",
        "exact_queue_occupancy",
        "candidate_buffer_bandwidth_mb_s",
        "proposal_enqueue_dequeue_ms",
        "total_tail_latency_ms",
    ):
        assert field_name in header


def test_ci_minimal_cpu_suite_runs_without_accelerator_paths(tmp_path: Path) -> None:
    config_path = CONFIG_ROOT / "ci_minimal_cpu.json"
    config = load_benchmark_suite_config(config_path)

    assert validate_benchmark_suite_config(config) is config
    assert config.suite_name == "ci_minimal_cpu"
    assert config.suite_type == "correctness"
    assert {case.method for case in config.cases} == {"PureExactCPU", "BVHExact", "RTExact"}
    assert all(case.dataset.mesh_count_per_split == 1 for case in config.cases)
    assert all(case.dataset.robot_link_count == 0 for case in config.cases)
    assert all(not case.dataset.include_robot_links for case in config.cases)

    case_payload = json.dumps([case.config for case in config.cases], sort_keys=True).lower()
    assert "cuda" not in case_payload
    assert "optix" not in case_payload

    result = run_benchmark_suite(
        config,
        output_root=tmp_path,
        environment=_cpu_only_environment(),
        run_id="ci_minimal_cpu",
    )

    assert result.export_paths is not None
    assert all(case.final_fn_zero for case in result.case_results)
    assert all(row.fn_count == 0 for row in result.rows)
    assert all(row.candidate_recall == 1.0 for row in result.rows)
    assert result.export_paths.csv_path.exists()
    assert result.export_paths.jsonl_path.exists()
    assert result.export_paths.run_meta_path.exists()
