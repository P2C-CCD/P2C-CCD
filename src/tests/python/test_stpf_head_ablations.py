from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench import (  # noqa: E402
    FEATURE_FAMILY_EDGE_EDGE,
    FEATURE_FAMILY_POINT_TRIANGLE,
    IntervalOnlyConfig,
    PureExactCPUConfig,
    RankingOnlyConfig,
    run_interval_only_on_external_batch,
    run_interval_only_on_generated_dataset,
    run_ranking_only_on_external_batch,
    run_ranking_only_on_generated_dataset,
    schedule_exact_work_items_interval_only,
    schedule_exact_work_items_ranking_only,
)
from p2cccd.contracts import CandidateRecord, ProposalSource, ProxyType  # noqa: E402
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset  # noqa: E402
from p2cccd.datasets.ccd import (  # noqa: E402
    CCD_ADAPTER_SCHEMA_VERSION,
    CCDQueryFamily,
    DatasetQueryBatch,
    ExternalCCDQuery,
    ScalableCCDSampleAdapter,
)
from p2cccd.proposal import (  # noqa: E402
    PROPOSAL_FAMILY_COUNT,
    PROPOSAL_FEATURE_DIM,
    PROPOSAL_INTERVAL_BIN_COUNT,
    ProposalFeatureRow,
    ProposalPrediction,
    STPFConfig,
    STPFModel,
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


def _candidate(candidate_id: int = 101) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        query_id=7,
        slab_id=0,
        object_a_id=1,
        patch_a_id=1,
        object_b_id=2,
        patch_b_id=2,
        proxy_type_a=ProxyType.SWEPT_AABB,
        proxy_type_b=ProxyType.SWEPT_AABB,
        rt_hit_count=1,
        motion_bound=[0.0, 0.0, 0.0, 0.0],
        flags=1,
    )


def _row(candidate_id: int = 101) -> ProposalFeatureRow:
    return ProposalFeatureRow(
        query_id=7,
        candidate_id=candidate_id,
        features=[0.0] * PROPOSAL_FEATURE_DIM,
        interval_targets=[1.0] + [0.0] * (PROPOSAL_INTERVAL_BIN_COUNT - 1),
        family_targets=[1.0] + [0.0] * (PROPOSAL_FAMILY_COUNT - 1),
        priority_target=1.0,
        cost_target=1.0,
        uncertainty_target=0.0,
        target_mask=31,
    )


def _prediction(candidate_id: int = 101) -> ProposalPrediction:
    return ProposalPrediction(
        candidate_id=candidate_id,
        interval_scores=[0.20, 0.70] + [0.0] * (PROPOSAL_INTERVAL_BIN_COUNT - 2),
        family_scores=[0.10, 0.90] + [0.0] * (PROPOSAL_FAMILY_COUNT - 2),
        priority_score=0.99,
        cost_score=2.0,
        uncertainty_score=0.0,
        source="unit",
    )


def test_interval_only_uses_interval_scores_without_family_expansion() -> None:
    candidates = (_candidate(),)
    rows = (_row(),)
    predictions = (_prediction(),)

    work_items, stats = schedule_exact_work_items_interval_only(
        candidates,
        rows,
        predictions,
        family_by_runtime_query_id={7: FEATURE_FAMILY_POINT_TRIANGLE},
        config=IntervalOnlyConfig(),
    )

    assert stats.fallback_count == 0
    assert work_items[0].source == ProposalSource.REFINED
    assert work_items[0].priority_score == 0.70
    assert work_items[0].feature_family_mask == FEATURE_FAMILY_POINT_TRIANGLE
    assert work_items[0].interval_t0 == 0.0
    assert work_items[0].interval_t1 == 1.0


def test_ranking_only_uses_family_scores_without_interval_priority() -> None:
    candidates = (_candidate(),)
    rows = (_row(),)
    predictions = (_prediction(),)

    work_items, stats = schedule_exact_work_items_ranking_only(
        candidates,
        rows,
        predictions,
        family_by_runtime_query_id={7: FEATURE_FAMILY_POINT_TRIANGLE},
        config=RankingOnlyConfig(),
    )

    assert stats.fallback_count == 0
    assert work_items[0].source == ProposalSource.REFINED
    assert work_items[0].priority_score == 0.90
    assert work_items[0].feature_family_mask == (
        FEATURE_FAMILY_POINT_TRIANGLE | FEATURE_FAMILY_EDGE_EDGE
    )
    assert work_items[0].interval_t0 == 0.0
    assert work_items[0].interval_t1 == 1.0


def test_interval_and_ranking_only_run_default_model_on_external_batch() -> None:
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
    exact = PureExactCPUConfig(eps_time=1.0e-6, eps_space=1.0e-8, max_subdivision_depth=28)

    interval_result = run_interval_only_on_external_batch(batch, IntervalOnlyConfig(exact=exact))
    ranking_result = run_ranking_only_on_external_batch(batch, RankingOnlyConfig(exact=exact))

    for result in (interval_result, ranking_result):
        assert result.benchmark.query_count == 2
        assert result.benchmark.fn_count == 0
        assert result.benchmark.candidate_recall == 1.0
        assert result.benchmark.fallback_ratio == 0.0
        assert result.queue_conserved
        assert result.final_fn_zero
        assert len(result.proposal_predictions) == len(result.candidates) == 1
        assert result.work_items[0].source == ProposalSource.REFINED
        assert result.work_items[0].interval_t0 == 0.0
        assert result.work_items[0].interval_t1 == 1.0

    assert interval_result.ablation_mode == "interval_only"
    assert ranking_result.ablation_mode == "ranking_only"
    assert interval_result.work_items[0].feature_family_mask == FEATURE_FAMILY_POINT_TRIANGLE
    assert ranking_result.work_items[0].feature_family_mask & FEATURE_FAMILY_POINT_TRIANGLE


def test_interval_and_ranking_only_run_real_stpf_model_on_scalable_sample() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch("cloth-funnel", family="ee", step=227, limit=8)
    model = STPFModel(STPFConfig(feature_dim=PROPOSAL_FEATURE_DIM, hidden_dim=16, num_layers=1))
    exact = PureExactCPUConfig(eps_time=1.0e-5, eps_space=1.0e-8, max_subdivision_depth=24)

    interval_result = run_interval_only_on_external_batch(
        batch,
        IntervalOnlyConfig(use_dummy_policy=False, proposal_batch_size=3, exact=exact),
        model=model,
        device="cpu",
    )
    ranking_result = run_ranking_only_on_external_batch(
        batch,
        RankingOnlyConfig(use_dummy_policy=False, proposal_batch_size=3, exact=exact),
        model=model,
        device="cpu",
    )

    for result in (interval_result, ranking_result):
        assert result.source_name == "Sample-Scalable-CCD-Data"
        assert result.benchmark.query_count == 8
        assert result.benchmark.fn_count == 0
        assert result.benchmark.candidate_recall == 1.0
        assert result.queue_conserved
        assert result.final_fn_zero
        assert all(prediction.source == "stpf" for prediction in result.proposal_predictions)
        assert all(item.feature_family_mask & FEATURE_FAMILY_EDGE_EDGE for item in result.work_items)


def test_head_ablations_route_ood_to_fallback_without_dropping_candidates() -> None:
    batch = DatasetQueryBatch(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt_ood",
        family=CCDQueryFamily.VERTEX_FACE,
        queries=(_vf_query(query_id=2, z0=1.0, z1=-1.0, label=True, batch_id="pt_ood"),),
    )
    exact = PureExactCPUConfig(eps_time=1.0e-6, eps_space=1.0e-8, max_subdivision_depth=28)

    interval_result = run_interval_only_on_external_batch(
        batch,
        IntervalOnlyConfig(exact=exact, ood_abs_feature_threshold=0.1),
    )
    ranking_result = run_ranking_only_on_external_batch(
        batch,
        RankingOnlyConfig(exact=exact, ood_abs_feature_threshold=0.1),
    )

    for result in (interval_result, ranking_result):
        assert result.benchmark.fn_count == 0
        assert result.benchmark.fallback_ratio == 1.0
        assert result.queue_conserved
        assert result.schedule_stats.fallback_count == len(result.candidates)
        assert result.schedule_stats.ood_fallback_count == len(result.candidates)
        assert all(item.source == ProposalSource.FALLBACK for item in result.work_items)


def test_head_ablations_run_on_internal_generated_dataset() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=2, seed=707)
    )

    interval_result = run_interval_only_on_generated_dataset(dataset)
    ranking_result = run_ranking_only_on_generated_dataset(dataset)

    for result in (interval_result, ranking_result):
        assert result.benchmark.query_count == len(dataset.rows)
        assert result.benchmark.fn_count == 0
        assert result.benchmark.candidate_recall == 1.0
        assert result.candidate_stats.compact_candidate_count <= len(dataset.rows)
        assert result.queue_conserved
        assert result.final_fn_zero


def test_head_ablations_reject_missing_model_when_dummy_policy_disabled() -> None:
    batch = DatasetQueryBatch(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt_model",
        family=CCDQueryFamily.VERTEX_FACE,
        queries=(_vf_query(query_id=3, z0=1.0, z1=-1.0, label=True, batch_id="pt_model"),),
    )

    try:
        run_interval_only_on_external_batch(
            batch, IntervalOnlyConfig(use_dummy_policy=False, allow_default_model=False)
        )
    except ValueError as exc:
        assert "model" in str(exc)
    else:
        raise AssertionError("expected missing model validation error")

    try:
        run_ranking_only_on_external_batch(
            batch, RankingOnlyConfig(use_dummy_policy=False, allow_default_model=False)
        )
    except ValueError as exc:
        assert "model" in str(exc)
    else:
        raise AssertionError("expected missing model validation error")
