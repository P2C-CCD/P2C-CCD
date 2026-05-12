from __future__ import annotations

import csv
from dataclasses import replace
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd import (  # noqa: E402
    BenchmarkRow,
    BenchmarkRowV2,
    BenchmarkRunMeta,
    ValidationError,
    from_json,
    read_jsonl,
    schema_field_names,
    validate_benchmark_row_v2,
    validate_benchmark_run_meta,
)
from p2cccd.bench import (  # noqa: E402
    benchmark_config_hash,
    benchmark_row_v2_from_legacy,
    canonical_config_json,
    create_benchmark_run_meta,
    export_benchmark_run,
    latency_percentiles,
)


def _environment() -> dict[str, object]:
    return {
        "git_commit": "abc123",
        "host_name": "unit-host",
        "platform": "unit-platform",
        "python_version": "3.12.0",
        "gpu_name": "Unit GPU",
        "driver_version": "555.00",
        "cuda_version": "12.6",
        "optix_version": "8.0",
        "vram_total_mb": 24000,
        "vram_free_mb": 12000,
    }


def _legacy_row() -> BenchmarkRow:
    return BenchmarkRow(
        query_count=4,
        fn_count=0,
        fp_count=1,
        candidate_recall=1.0,
        avg_candidates=2.5,
        avg_exact_evals=2.5,
        avg_subdivision_depth=1.25,
        fallback_ratio=0.25,
        rt_ms=0.9,
        proposal_ms=0.4,
        exact_ms=1.7,
        total_ms=3.0,
        qps=1333.3,
    )


def _meta() -> BenchmarkRunMeta:
    return create_benchmark_run_meta(
        dataset_name="unit_dataset",
        scene_name="unit_scene",
        method_name="RTSTPFExact",
        config={"b": 2, "a": 1},
        seed=42,
        run_id="run_unit_001",
        environment=_environment(),
        notes="unit",
    )


def test_config_hash_and_latency_percentiles_are_stable() -> None:
    assert canonical_config_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert benchmark_config_hash({"a": 1, "b": 2}) == benchmark_config_hash({"b": 2, "a": 1})

    percentiles = latency_percentiles([1.0, 2.0, 4.0, 8.0], fallback_total_ms=0.0, fallback_query_count=0)

    assert percentiles["min"] == 1.0
    assert percentiles["p50"] == 3.0
    assert percentiles["p90"] > percentiles["p50"]
    assert percentiles["max"] == 8.0


def test_benchmark_row_v2_from_legacy_carries_required_metadata() -> None:
    meta = _meta()
    row_v2 = benchmark_row_v2_from_legacy(
        _legacy_row(),
        meta,
        latency_samples_ms=[1.0, 2.0, 4.0, 8.0],
        family_exact_calls={"point_triangle": 4, "edge_edge": 2},
        candidate_inflation_ratio=1.5,
        undecided_to_resolved_ratio=0.25,
        exact_queue_occupancy=0.75,
        rt_build_ms=0.2,
        rt_update_ms=0.3,
        rt_trace_ms=0.4,
        candidate_buffer_bandwidth_mb_s=123.0,
        proposal_enqueue_dequeue_ms=0.11,
        total_tail_latency_ms=8.0,
        vram_peak_mb=8000,
    )

    assert row_v2.run_id == "run_unit_001"
    assert row_v2.dataset_name == "unit_dataset"
    assert row_v2.method_name == "RTSTPFExact"
    assert row_v2.config_hash == meta.config_hash
    assert row_v2.seed == 42
    assert row_v2.rt_build_ms == 0.2
    assert row_v2.rt_update_ms == 0.3
    assert row_v2.rt_trace_ms == 0.4
    assert row_v2.rt_ms == pytest.approx(0.9)
    assert row_v2.family_point_triangle_exact_calls == 4
    assert row_v2.family_edge_edge_exact_calls == 2
    assert row_v2.family_unknown_exact_calls == 4
    assert row_v2.exact_calls_total == 10
    assert row_v2.candidate_inflation_ratio == 1.5
    assert row_v2.undecided_to_resolved_ratio == 0.25
    assert row_v2.exact_queue_occupancy == 0.75
    assert row_v2.candidate_buffer_bandwidth_mb_s == 123.0
    assert row_v2.proposal_enqueue_dequeue_ms == 0.11
    assert row_v2.total_tail_latency_ms == 8.0
    assert row_v2.vram_peak_mb == 8000
    assert validate_benchmark_row_v2(row_v2) is row_v2


def test_benchmark_v2_export_writes_csv_jsonl_and_run_meta(tmp_path: Path) -> None:
    meta = _meta()
    row_v2 = benchmark_row_v2_from_legacy(
        _legacy_row(),
        meta,
        latency_samples_ms=[1.0, 2.0, 4.0, 8.0],
        family_exact_calls={"conservative": 10},
        vram_peak_mb=7000,
    )

    paths = export_benchmark_run(tmp_path / "run", meta, [row_v2])

    assert paths.csv_path.exists()
    assert paths.jsonl_path.exists()
    assert paths.run_meta_path.exists()
    assert paths.csv_path.read_text(encoding="utf-8").splitlines()[0] == ",".join(
        schema_field_names(BenchmarkRowV2)
    )
    assert list(read_jsonl(paths.jsonl_path, BenchmarkRowV2)) == [row_v2]

    loaded_meta = from_json(BenchmarkRunMeta, paths.run_meta_path.read_text(encoding="utf-8"))
    assert loaded_meta.row_count == 1
    assert loaded_meta.gpu_name == "Unit GPU"
    assert loaded_meta.cuda_version == "12.6"
    assert loaded_meta.output_csv == "benchmark.csv"


def test_benchmark_v2_validators_reject_invalid_rows() -> None:
    meta = _meta()
    row_v2 = benchmark_row_v2_from_legacy(_legacy_row(), meta)

    with pytest.raises(ValidationError):
        validate_benchmark_run_meta(replace(meta, run_id=""))

    with pytest.raises(ValidationError):
        validate_benchmark_row_v2(replace(row_v2, latency_p90_ms=0.1))

    with pytest.raises(ValidationError):
        validate_benchmark_row_v2(replace(row_v2, family_unknown_exact_calls=999))

    with pytest.raises(ValidationError):
        validate_benchmark_row_v2(replace(row_v2, candidate_inflation_ratio=-1.0))

    with pytest.raises(ValidationError):
        validate_benchmark_row_v2(replace(row_v2, candidate_buffer_bandwidth_mb_s=-1.0))

    with pytest.raises(ValueError):
        export_benchmark_run(Path("unused"), meta, [replace(row_v2, run_id="different")])


def test_benchmark_row_v2_validator_covers_meta_csv_jsonl_and_run_meta(tmp_path: Path) -> None:
    meta = _meta()
    row_a = benchmark_row_v2_from_legacy(
        _legacy_row(),
        meta,
        latency_samples_ms=[0.75, 1.25, 2.0],
        family_exact_calls={"point_triangle": 4, "edge_edge": 3, "conservative": 2, "unknown": 1},
        rt_build_ms=0.1,
        rt_update_ms=0.2,
        rt_trace_ms=0.6,
    )
    row_b = validate_benchmark_row_v2(
        replace(
            row_a,
            scene_name="unit_scene_b",
            query_count=2,
            fn_count=0,
            fp_count=0,
            candidate_recall=1.0,
            avg_candidates=1.5,
            avg_exact_evals=1.0,
            exact_calls_total=2,
            family_point_triangle_exact_calls=1,
            family_edge_edge_exact_calls=1,
            family_conservative_exact_calls=0,
            family_unknown_exact_calls=0,
        )
    )

    paths = export_benchmark_run(tmp_path / "validator_run", meta, [row_a, row_b])

    loaded_rows = list(read_jsonl(paths.jsonl_path, BenchmarkRowV2))
    loaded_meta = from_json(BenchmarkRunMeta, paths.run_meta_path.read_text(encoding="utf-8"))
    with paths.csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert loaded_rows == [row_a, row_b]
    assert validate_benchmark_run_meta(loaded_meta) is loaded_meta
    assert loaded_meta.row_count == 2
    assert loaded_meta.output_csv == "benchmark.csv"
    assert loaded_meta.output_jsonl == "benchmark.jsonl"
    assert loaded_meta.output_run_meta_json == "run_meta.json"
    assert len(csv_rows) == 2
    assert set(csv_rows[0]) == set(schema_field_names(BenchmarkRowV2))
    assert int(csv_rows[0]["query_count"]) == row_a.query_count
    assert float(csv_rows[0]["rt_ms"]) == pytest.approx(row_a.rt_ms)
    assert int(csv_rows[1]["exact_calls_total"]) == row_b.exact_calls_total

    bad_jsonl = tmp_path / "bad_benchmark.jsonl"
    bad_jsonl.write_text('{"schema_version":1,"run_id":"missing_required_fields"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        list(read_jsonl(bad_jsonl, BenchmarkRowV2))
