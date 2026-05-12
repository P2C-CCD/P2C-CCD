from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench import (  # noqa: E402
    FEATURE_FAMILY_EDGE_EDGE,
    FEATURE_FAMILY_POINT_TRIANGLE,
    NoProposalConfig,
    PureExactCPUConfig,
    RTSTPFExactConfig,
    run_no_proposal_on_external_batch,
    run_no_proposal_on_generated_dataset,
    run_rt_stpf_exact_on_external_batch,
)
from p2cccd.contracts import ProposalSource  # noqa: E402
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


def test_no_proposal_routes_all_candidates_to_fallback_exact_queue() -> None:
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
    config = NoProposalConfig(
        exact=PureExactCPUConfig(eps_time=1.0e-6, eps_space=1.0e-8, max_subdivision_depth=28)
    )

    result = run_no_proposal_on_external_batch(batch, config)

    assert result.benchmark.query_count == 2
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.benchmark.proposal_ms == 0.0
    assert result.benchmark.fallback_ratio == 1.0
    assert result.queue_conserved
    assert result.final_fn_zero
    assert result.no_proposal_stats.raw_candidate_count == 1
    assert result.no_proposal_stats.work_item_count == 1
    assert result.no_proposal_stats.fallback_count == 1
    assert result.no_proposal_stats.monotonic_safe
    assert result.work_items[0].source == ProposalSource.FALLBACK
    assert result.work_items[0].interval_t0 == 0.0
    assert result.work_items[0].interval_t1 == 1.0
    assert result.work_items[0].feature_family_mask & FEATURE_FAMILY_POINT_TRIANGLE


def test_no_proposal_is_distinct_from_dummy_stpf_scheduling() -> None:
    batch = DatasetQueryBatch(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt_compare",
        family=CCDQueryFamily.VERTEX_FACE,
        queries=(_vf_query(query_id=2, z0=1.0, z1=-1.0, label=True, batch_id="pt_compare"),),
    )
    exact = PureExactCPUConfig(eps_time=1.0e-6, eps_space=1.0e-8, max_subdivision_depth=28)

    no_proposal = run_no_proposal_on_external_batch(batch, NoProposalConfig(exact=exact))
    stpf = run_rt_stpf_exact_on_external_batch(batch, RTSTPFExactConfig(exact=exact))

    assert len(no_proposal.candidates) == len(stpf.candidates) == 1
    assert no_proposal.benchmark.fn_count == stpf.benchmark.fn_count == 0
    assert no_proposal.benchmark.fallback_ratio == 1.0
    assert stpf.benchmark.fallback_ratio == 0.0
    assert no_proposal.benchmark.proposal_ms == 0.0
    assert stpf.benchmark.proposal_ms >= 0.0
    assert no_proposal.work_items[0].source == ProposalSource.FALLBACK
    assert stpf.work_items[0].source == ProposalSource.REFINED


def test_no_proposal_runs_on_external_scalable_sample_batch() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch("cloth-funnel", family="ee", step=227, limit=8)
    config = NoProposalConfig(
        exact=PureExactCPUConfig(eps_time=1.0e-5, eps_space=1.0e-8, max_subdivision_depth=24)
    )

    result = run_no_proposal_on_external_batch(batch, config)

    assert result.source_name == "Sample-Scalable-CCD-Data"
    assert result.benchmark.query_count == 8
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.benchmark.fallback_ratio == 1.0
    assert result.queue_conserved
    assert all(item.source == ProposalSource.FALLBACK for item in result.work_items)
    assert all(item.feature_family_mask == FEATURE_FAMILY_EDGE_EDGE for item in result.work_items)


def test_no_proposal_runs_on_internal_generated_dataset_without_false_negatives() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=2, seed=505)
    )

    result = run_no_proposal_on_generated_dataset(dataset)

    assert result.benchmark.query_count == len(dataset.rows)
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.benchmark.proposal_ms == 0.0
    assert result.candidate_stats.compact_candidate_count <= len(dataset.rows)
    assert result.no_proposal_stats.fallback_count == len(result.candidates)
    assert result.queue_conserved
    assert result.final_fn_zero


def test_no_proposal_rejects_invalid_config() -> None:
    batch = DatasetQueryBatch(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt_invalid",
        family=CCDQueryFamily.VERTEX_FACE,
        queries=(_vf_query(query_id=3, z0=1.0, z1=-1.0, label=True, batch_id="pt_invalid"),),
    )

    try:
        run_no_proposal_on_external_batch(batch, NoProposalConfig(rt_backend_name="unknown"))
    except ValueError as exc:
        assert "rt_backend_name" in str(exc)
    else:
        raise AssertionError("expected backend validation error")

    try:
        run_no_proposal_on_external_batch(batch, NoProposalConfig(first_work_item_id=0))
    except ValueError as exc:
        assert "first_work_item_id" in str(exc)
    else:
        raise AssertionError("expected first_work_item_id validation error")
