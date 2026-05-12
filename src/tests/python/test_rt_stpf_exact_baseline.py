from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench import (  # noqa: E402
    FEATURE_FAMILY_EDGE_EDGE,
    FEATURE_FAMILY_POINT_TRIANGLE,
    PureExactCPUConfig,
    RTSTPFExactConfig,
    run_rt_stpf_exact_on_external_batch,
    run_rt_stpf_exact_on_generated_dataset,
)
from p2cccd.bench.bvh_exact import _try_load_p2cccd_cpp  # noqa: E402
from p2cccd.bench.rt_exact import RTExactConfig, _family_mask_for_external, _make_external_candidates  # noqa: E402
from p2cccd.bench.rt_stpf_exact import (  # noqa: E402
    proposal_feature_rows_from_rt_candidates,
    schedule_exact_work_items_with_stpf,
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
from p2cccd.proposal import PROPOSAL_FEATURE_DIM, STPFConfig, STPFModel  # noqa: E402
from p2cccd.proposal.inference import ProposalPrediction  # noqa: E402


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


def test_rt_stpf_exact_learned_default_keeps_candidate_coverage() -> None:
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
    config = RTSTPFExactConfig(
        exact=PureExactCPUConfig(eps_time=1.0e-6, eps_space=1.0e-8, max_subdivision_depth=28)
    )

    result = run_rt_stpf_exact_on_external_batch(batch, config)

    assert result.benchmark.query_count == 2
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.benchmark.proposal_ms >= 0.0
    assert result.benchmark.fallback_ratio == 0.0
    assert len(result.candidates) == 1
    assert result.queue_conserved
    assert result.final_fn_zero
    assert result.schedule_stats.raw_candidate_count == 1
    assert result.schedule_stats.feature_row_count == 1
    assert result.schedule_stats.proposal_output_count == 1
    assert result.schedule_stats.work_item_count == 1
    assert result.schedule_stats.monotonic_safe
    assert result.work_items[0].source == ProposalSource.REFINED
    assert result.work_items[0].interval_t0 == 0.0
    assert result.work_items[0].interval_t1 == 1.0
    assert result.work_items[0].feature_family_mask & FEATURE_FAMILY_POINT_TRIANGLE
    assert result.inference_backend_name == "torch"
    assert result.inference_provider_name in {"cpu", "cuda:0", "cuda"}


def test_rt_stpf_exact_auto_fastest_resolves_to_learned_on_sparse_batch() -> None:
    batch = DatasetQueryBatch(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt_auto_fastest",
        family=CCDQueryFamily.VERTEX_FACE,
        queries=(
            _vf_query(query_id=100, z0=1.0, z1=-1.0, label=True, batch_id="pt_auto_fastest"),
            _vf_query(query_id=101, z0=2.0, z1=2.0, label=False, batch_id="pt_auto_fastest"),
        ),
    )
    result = run_rt_stpf_exact_on_external_batch(
        batch,
        RTSTPFExactConfig(
            execution_profile="auto_fastest",
            exact=PureExactCPUConfig(eps_time=1.0e-6, eps_space=1.0e-8, max_subdivision_depth=28),
        ),
    )

    assert result.final_fn_zero
    assert result.queue_conserved
    assert result.inference_backend_name == "ort"
    assert "ExecutionProvider" in result.inference_provider_name
    assert result.resolved_execution_profile_name.startswith("auto_fastest:learned_ort@")


def test_rt_stpf_exact_runs_real_stpf_model_on_external_scalable_sample() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch("cloth-funnel", family="ee", step=227, limit=8)
    model = STPFModel(STPFConfig(feature_dim=PROPOSAL_FEATURE_DIM, hidden_dim=16, num_layers=1))
    config = RTSTPFExactConfig(
        proposal_batch_size=3,
        exact=PureExactCPUConfig(eps_time=1.0e-5, eps_space=1.0e-8, max_subdivision_depth=24),
    )

    result = run_rt_stpf_exact_on_external_batch(batch, config, model=model, device="cpu")

    assert result.source_name == "Sample-Scalable-CCD-Data"
    assert result.benchmark.query_count == 8
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.queue_conserved
    assert result.final_fn_zero
    assert len(result.feature_rows) == len(result.candidates)
    assert len(result.proposal_predictions) == len(result.candidates)
    assert all(prediction.source == "stpf" for prediction in result.proposal_predictions)
    assert all(item.feature_family_mask & FEATURE_FAMILY_EDGE_EDGE for item in result.work_items)


def test_rt_stpf_exact_runs_ort_backend_on_external_scalable_sample(tmp_path: Path) -> None:
    try:
        import onnxruntime as ort  # noqa: F401
    except ImportError:
        return
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch("cloth-funnel", family="ee", step=227, limit=8)
    model = STPFModel(STPFConfig(feature_dim=PROPOSAL_FEATURE_DIM, hidden_dim=16, num_layers=1))
    onnx_path = tmp_path / "stpf_test.onnx"
    config = RTSTPFExactConfig(
        inference_backend="ort",
        ort_model_path=str(onnx_path),
        proposal_batch_size=4,
        model_device="cpu",
        exact=PureExactCPUConfig(eps_time=1.0e-5, eps_space=1.0e-8, max_subdivision_depth=24),
    )

    result = run_rt_stpf_exact_on_external_batch(batch, config, model=model, device="cpu")

    assert onnx_path.exists()
    assert result.inference_backend_name == "ort"
    assert "ExecutionProvider" in result.inference_provider_name
    assert result.exact_backend_name in {"pure_exact_cpu", "cuda_exact"}
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0

def test_rt_stpf_exact_cpp_schedule_respects_uncertainty_fallback() -> None:
    if _try_load_p2cccd_cpp() is None:
        return
    batch = DatasetQueryBatch(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt_cpp_schedule",
        family=CCDQueryFamily.VERTEX_FACE,
        queries=(_vf_query(query_id=11, z0=1.0, z1=-1.0, label=True, batch_id="pt_cpp_schedule"),),
    )
    rt_config = RTExactConfig(
        exact=PureExactCPUConfig(eps_time=1.0e-6, eps_space=1.0e-8, max_subdivision_depth=28)
    )
    candidates, candidate_stats, runtime_ids = _make_external_candidates(batch, rt_config)
    family_by_runtime_query_id = {
        runtime_ids[query.query_id]: _family_mask_for_external(query)
        for query in batch.queries
    }
    rows = proposal_feature_rows_from_rt_candidates(
        candidates,
        family_by_runtime_query_id=family_by_runtime_query_id,
        candidate_stats=candidate_stats,
    )
    predictions = (
        ProposalPrediction(
            candidate_id=rows[0].candidate_id,
            interval_scores=list(rows[0].interval_targets),
            family_scores=list(rows[0].family_targets),
            priority_score=1.0,
            cost_score=1.0,
            uncertainty_score=1.0,
            source="stpf",
        ),
    )

    work_items, stats = schedule_exact_work_items_with_stpf(
        candidates,
        rows,
        predictions,
        family_by_runtime_query_id=family_by_runtime_query_id,
        config=RTSTPFExactConfig(),
    )

    assert stats.fallback_count == len(candidates)
    assert stats.high_uncertainty_fallback_count == len(candidates)
    assert all(item.source == ProposalSource.FALLBACK for item in work_items)


def test_rt_stpf_exact_ood_routes_to_fallback_without_dropping_candidates() -> None:
    batch = DatasetQueryBatch(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt_ood",
        family=CCDQueryFamily.VERTEX_FACE,
        queries=(_vf_query(query_id=2, z0=1.0, z1=-1.0, label=True, batch_id="pt_ood"),),
    )
    config = RTSTPFExactConfig(
        ood_abs_feature_threshold=0.1,
        exact=PureExactCPUConfig(eps_time=1.0e-6, eps_space=1.0e-8, max_subdivision_depth=28),
    )

    result = run_rt_stpf_exact_on_external_batch(batch, config)

    assert result.benchmark.fn_count == 0
    assert result.queue_conserved
    assert result.schedule_stats.fallback_count == len(result.candidates)
    assert result.schedule_stats.ood_fallback_count == len(result.candidates)
    assert result.schedule_stats.invalid_proposal_fallback_count == 0
    assert result.benchmark.fallback_ratio == 1.0
    assert all(item.source == ProposalSource.FALLBACK for item in result.work_items)
    assert all(item.feature_family_mask & FEATURE_FAMILY_POINT_TRIANGLE for item in result.work_items)


def test_rt_stpf_exact_runs_on_internal_generated_dataset() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=2, seed=404)
    )

    result = run_rt_stpf_exact_on_generated_dataset(dataset)

    assert result.benchmark.query_count == len(dataset.rows)
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.candidate_stats.compact_candidate_count <= len(dataset.rows)
    assert result.queue_conserved
    assert result.final_fn_zero


def test_rt_stpf_exact_rejects_missing_model_when_dummy_policy_disabled() -> None:
    batch = DatasetQueryBatch(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt_model",
        family=CCDQueryFamily.VERTEX_FACE,
        queries=(_vf_query(query_id=3, z0=1.0, z1=-1.0, label=True, batch_id="pt_model"),),
    )

    try:
        run_rt_stpf_exact_on_external_batch(
            batch,
            RTSTPFExactConfig(allow_default_model=False),
        )
    except ValueError as exc:
        assert "model" in str(exc)
    else:
        raise AssertionError("expected missing STPF model validation error")


def test_rt_stpf_exact_rejects_dummy_policy_path() -> None:
    batch = DatasetQueryBatch(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt_dummy_reject",
        family=CCDQueryFamily.VERTEX_FACE,
        queries=(_vf_query(query_id=30, z0=1.0, z1=-1.0, label=True, batch_id="pt_dummy_reject"),),
    )

    try:
        run_rt_stpf_exact_on_external_batch(
            batch,
            RTSTPFExactConfig(use_dummy_policy=True),
        )
    except ValueError as exc:
        assert "dummy" in str(exc)
    else:
        raise AssertionError("expected dummy-path validation error")


def test_rt_stpf_exact_rejects_fastest_dummy_profile() -> None:
    batch = DatasetQueryBatch(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt_fastest_dummy_reject",
        family=CCDQueryFamily.VERTEX_FACE,
        queries=(_vf_query(query_id=31, z0=1.0, z1=-1.0, label=True, batch_id="pt_fastest_dummy_reject"),),
    )

    try:
        run_rt_stpf_exact_on_external_batch(
            batch,
            RTSTPFExactConfig(execution_profile="fastest_dummy"),
        )
    except ValueError as exc:
        assert "execution_profile" in str(exc)
    else:
        raise AssertionError("expected fastest_dummy validation error")


def test_rt_stpf_exact_rejects_unknown_backend_name() -> None:
    batch = DatasetQueryBatch(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt_backend",
        family=CCDQueryFamily.VERTEX_FACE,
        queries=(_vf_query(query_id=4, z0=1.0, z1=-1.0, label=True, batch_id="pt_backend"),),
    )

    try:
        run_rt_stpf_exact_on_external_batch(
            batch,
            RTSTPFExactConfig(rt_backend_name="unknown"),
        )
    except ValueError as exc:
        assert "rt_backend_name" in str(exc)
    else:
        raise AssertionError("expected backend validation error")
