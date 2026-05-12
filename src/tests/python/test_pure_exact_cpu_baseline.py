from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench import (  # noqa: E402
    PureExactCPUConfig,
    evaluate_external_ccd_query,
    run_pure_exact_cpu_on_external_batch,
    run_pure_exact_cpu_on_generated_dataset,
)
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset  # noqa: E402
from p2cccd.datasets.ccd import (  # noqa: E402
    CCD_ADAPTER_SCHEMA_VERSION,
    CCDQueryFamily,
    ExternalCCDQuery,
    ScalableCCDSampleAdapter,
)


BASELINE_ROOT = PROJECT_ROOT / "baseline"


def _vf_query(*, z0: float, z1: float, label: bool) -> ExternalCCDQuery:
    return ExternalCCDQuery(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt",
        query_id=1,
        source_query_index=0,
        family=CCDQueryFamily.VERTEX_FACE,
        vertices_t0=((0.25, 0.25, z0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        vertices_t1=((0.25, 0.25, z1), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ground_truth_collides=label,
    )


def _ee_query(*, z0: float, z1: float, label: bool) -> ExternalCCDQuery:
    return ExternalCCDQuery(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="edge_edge",
        batch_id="ee",
        query_id=2,
        source_query_index=0,
        family=CCDQueryFamily.EDGE_EDGE,
        vertices_t0=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, z0), (0.0, 1.0, z0)),
        vertices_t1=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, z1), (0.0, 1.0, z1)),
        ground_truth_collides=label,
    )


def test_pure_exact_cpu_detects_synthetic_point_triangle_and_edge_edge_collisions() -> None:
    config = PureExactCPUConfig(eps_time=1.0e-6, eps_space=1.0e-8, max_subdivision_depth=28)
    pt_collision = evaluate_external_ccd_query(_vf_query(z0=1.0, z1=-1.0, label=True), config)
    ee_collision = evaluate_external_ccd_query(_ee_query(z0=1.0, z1=-1.0, label=True), config)
    pt_separation = evaluate_external_ccd_query(_vf_query(z0=2.0, z1=2.0, label=False), config)

    assert pt_collision.predicted_collision
    assert pt_collision.toi_upper <= 0.5
    assert ee_collision.predicted_collision
    assert ee_collision.toi_upper <= 0.5
    assert not pt_separation.predicted_collision
    assert pt_separation.status == "separation"


def test_pure_exact_cpu_runs_on_internal_generated_dataset_without_fn_or_fp() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=2, seed=101)
    )

    result = run_pure_exact_cpu_on_generated_dataset(dataset)

    assert result.benchmark.query_count == len(dataset.rows)
    assert result.benchmark.fn_count == 0
    assert result.benchmark.fp_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.final_fn_zero


def test_pure_exact_cpu_runs_on_external_scalable_sample_batch_with_no_false_negatives() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch("cloth-funnel", family="vf", step=227, limit=8)
    config = PureExactCPUConfig(eps_time=1.0e-5, eps_space=1.0e-8, max_subdivision_depth=24)

    result = run_pure_exact_cpu_on_external_batch(batch, config)

    assert result.source_name == "Sample-Scalable-CCD-Data"
    assert result.batch_id == "cloth-funnel:227vf"
    assert result.benchmark.query_count == 8
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.final_fn_zero
    assert len(result.query_results) == 8
    assert all(query.exact_evals > 0 for query in result.query_results)


def test_pure_exact_cpu_rejects_invalid_config() -> None:
    try:
        evaluate_external_ccd_query(_vf_query(z0=1.0, z1=-1.0, label=True), PureExactCPUConfig(eps_time=0.0))
    except ValueError as exc:
        assert "eps_time" in str(exc)
    else:
        raise AssertionError("expected invalid eps_time error")
