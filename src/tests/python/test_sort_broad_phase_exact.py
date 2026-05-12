from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench import (  # noqa: E402
    Aabb,
    BVHExactConfig,
    BroadPhasePrimitive,
    PureExactCPUConfig,
    SortBroadPhaseConfig,
    run_bvh_exact_on_external_batch,
    run_sort_broad_phase_exact_on_external_batch,
    run_sort_broad_phase_exact_on_generated_dataset,
    sort_sweep_broad_phase_pairs,
)
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset  # noqa: E402
from p2cccd.datasets.ccd import (  # noqa: E402
    CCD_ADAPTER_SCHEMA_VERSION,
    CCDQueryFamily,
    DatasetQueryBatch,
    ExternalCCDQuery,
    ScalableCCDSampleAdapter,
)


BASELINE_ROOT = PROJECT_ROOT / "baseline"


def _vf_query(
    *,
    query_id: int,
    z0: float,
    z1: float,
    label: bool,
    batch_id: str = "pt",
) -> ExternalCCDQuery:
    return ExternalCCDQuery(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id=batch_id,
        query_id=query_id,
        source_query_index=query_id,
        family=CCDQueryFamily.VERTEX_FACE,
        vertices_t0=((0.25, 0.25, z0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        vertices_t1=((0.25, 0.25, z1), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ground_truth_collides=label,
    )


def test_sort_sweep_broad_phase_filters_pairs_and_exports_counters() -> None:
    primitives = (
        BroadPhasePrimitive(
            primitive_id=1,
            query_id=1,
            role="a",
            aabb=Aabb(min=(0.0, 0.0, 0.0), max=(1.0, 1.0, 1.0)),
            family="unit",
        ),
        BroadPhasePrimitive(
            primitive_id=2,
            query_id=1,
            role="b",
            aabb=Aabb(min=(0.5, 0.5, 0.5), max=(1.5, 1.5, 1.5)),
            family="unit",
        ),
        BroadPhasePrimitive(
            primitive_id=3,
            query_id=2,
            role="a",
            aabb=Aabb(min=(10.0, 0.0, 0.0), max=(11.0, 1.0, 1.0)),
            family="unit",
        ),
        BroadPhasePrimitive(
            primitive_id=4,
            query_id=2,
            role="b",
            aabb=Aabb(min=(12.0, 0.0, 0.0), max=(13.0, 1.0, 1.0)),
            family="unit",
        ),
    )

    pairs, stats = sort_sweep_broad_phase_pairs(primitives)

    assert [(pair.query_id, pair.primitive_a_id, pair.primitive_b_id) for pair in pairs] == [(1, 1, 2)]
    assert stats.primitive_count == 4
    assert stats.endpoint_count == 8
    assert stats.pair_count == 1
    assert stats.active_interval_tests == 1
    assert stats.aabb_overlap_tests == 1
    assert stats.backend_name == "cpu_sort_sweep"
    assert stats.axis == 0


def test_sort_broad_phase_exact_runs_on_synthetic_external_batch() -> None:
    batch = DatasetQueryBatch(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt",
        family=CCDQueryFamily.VERTEX_FACE,
        queries=(
            _vf_query(query_id=0, z0=1.0, z1=-1.0, label=True),
            _vf_query(query_id=1, z0=2.0, z1=2.0, label=False),
        ),
    )
    config = SortBroadPhaseConfig(
        exact=PureExactCPUConfig(eps_time=1.0e-6, eps_space=1.0e-8, max_subdivision_depth=28)
    )

    result = run_sort_broad_phase_exact_on_external_batch(batch, config)

    assert result.benchmark.query_count == 2
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.benchmark.avg_candidates == 0.5
    assert result.benchmark.rt_ms == result.sort_stats.total_ms
    assert result.benchmark.proposal_ms == 0.0
    assert result.sort_stats.primitive_count == 4
    assert result.sort_stats.pair_count == 1
    assert result.final_fn_zero


def test_sort_broad_phase_exact_matches_bvh_pair_count_on_scalable_sample() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch("cloth-funnel", family="ee", step=227, limit=8)
    exact = PureExactCPUConfig(eps_time=1.0e-5, eps_space=1.0e-8, max_subdivision_depth=24)

    sort_result = run_sort_broad_phase_exact_on_external_batch(
        batch,
        SortBroadPhaseConfig(exact=exact),
    )
    bvh_result = run_bvh_exact_on_external_batch(batch, BVHExactConfig(exact=exact))

    assert sort_result.source_name == "Sample-Scalable-CCD-Data"
    assert sort_result.batch_id == "cloth-funnel:227ee"
    assert sort_result.benchmark.query_count == 8
    assert sort_result.benchmark.fn_count == 0
    assert sort_result.benchmark.candidate_recall == 1.0
    assert sort_result.sort_stats.pair_count == bvh_result.broad_phase_stats.pair_count
    assert sort_result.sort_stats.endpoint_count == 16 * 2
    assert sort_result.final_fn_zero


def test_sort_broad_phase_exact_runs_on_internal_generated_dataset() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=2, seed=606)
    )

    result = run_sort_broad_phase_exact_on_generated_dataset(dataset)

    assert result.benchmark.query_count == len(dataset.rows)
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.benchmark.avg_candidates <= 1.0
    assert result.sort_stats.pair_count <= len(dataset.rows)
    assert result.final_fn_zero


def test_sort_broad_phase_exact_accepts_gpu_compatible_backend_name() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch("cloth-funnel", family="vf", step=227, limit=2)

    result = run_sort_broad_phase_exact_on_external_batch(
        batch,
        SortBroadPhaseConfig(backend_name="gpu_sort_sweep_compatible", axis=1),
    )

    assert result.sort_stats.backend_name == "gpu_sort_sweep_compatible"
    assert result.sort_stats.axis == 1
    assert result.benchmark.query_count == 2


def test_sort_broad_phase_exact_rejects_invalid_config() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch("cloth-funnel", family="vf", step=227, limit=1)

    try:
        run_sort_broad_phase_exact_on_external_batch(
            batch,
            SortBroadPhaseConfig(backend_name="unknown"),
        )
    except ValueError as exc:
        assert "backend_name" in str(exc)
    else:
        raise AssertionError("expected backend validation error")

    try:
        run_sort_broad_phase_exact_on_external_batch(batch, SortBroadPhaseConfig(axis=3))
    except ValueError as exc:
        assert "axis" in str(exc)
    else:
        raise AssertionError("expected axis validation error")
