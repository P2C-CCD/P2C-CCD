from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

import p2cccd.bench.rt_exact as rt_exact_module  # noqa: E402
from p2cccd.bench import (  # noqa: E402
    FEATURE_FAMILY_EDGE_EDGE,
    FEATURE_FAMILY_POINT_TRIANGLE,
    PureExactCPUConfig,
    RTExactConfig,
    run_rt_exact_on_external_batch,
    run_rt_exact_on_generated_dataset,
    validate_rt_exact_coverage,
)
from p2cccd.contracts import AuditStage, CertificateStatus, ProposalSource  # noqa: E402
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset  # noqa: E402
from p2cccd.datasets.ccd import (  # noqa: E402
    CCD_ADAPTER_SCHEMA_VERSION,
    CCDQueryFamily,
    DatasetQueryBatch,
    ExternalCCDQuery,
    ScalableCCDSampleAdapter,
)


BASELINE_ROOT = PROJECT_ROOT / "baseline"


def _vf_query(*, query_id: int, z0: float, z1: float, label: bool) -> ExternalCCDQuery:
    return ExternalCCDQuery(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt",
        query_id=query_id,
        source_query_index=query_id,
        family=CCDQueryFamily.VERTEX_FACE,
        vertices_t0=((0.25, 0.25, z0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        vertices_t1=((0.25, 0.25, z1), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ground_truth_collides=label,
    )


def test_rt_exact_runs_conservative_candidates_to_certificates_without_stpf() -> None:
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
    config = RTExactConfig(
        exact=PureExactCPUConfig(eps_time=1.0e-6, eps_space=1.0e-8, max_subdivision_depth=28)
    )

    result = run_rt_exact_on_external_batch(batch, config)

    assert result.benchmark.query_count == 2
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.benchmark.proposal_ms == 0.0
    assert result.benchmark.avg_candidates == 0.5
    assert len(result.candidates) == 1
    assert len(result.work_items) == 1
    assert len(result.certificates) == 1
    assert result.certificates[0].status == CertificateStatus.COLLISION
    assert result.work_items[0].source == ProposalSource.RAW
    assert result.work_items[0].feature_family_mask == FEATURE_FAMILY_POINT_TRIANGLE
    assert result.queue_conserved
    assert result.final_fn_zero
    assert all(row.stage != AuditStage.PROPOSAL for row in result.audit_log)


def test_rt_exact_external_candidates_use_cpp_fast_path_when_available(monkeypatch) -> None:
    batch = DatasetQueryBatch(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt",
        family=CCDQueryFamily.VERTEX_FACE,
        queries=(
            _vf_query(query_id=5, z0=1.0, z1=-1.0, label=True),
            _vf_query(query_id=6, z0=2.0, z1=2.0, label=False),
        ),
    )

    class _FakeTiming:
        def __init__(self) -> None:
            self.build_ms = 4.0
            self.update_ms = 0.5
            self.trace_ms = 1.5
            self.compact_ms = 0.25
            self.total_ms = 6.25

    class _FakeRuntimeQueryId:
        def __init__(self, source_query_id: int, runtime_query_id: int) -> None:
            self.source_query_id = source_query_id
            self.runtime_query_id = runtime_query_id

    class _FakeResult:
        def __init__(self) -> None:
            self.backend_name = "optix_rt"
            self.timing = _FakeTiming()
            self.primitive_count = 4
            self.raw_hit_count = 1
            self.compact_candidate_count = 1
            self.candidate_recall = 1.0
            self.candidates = (
                rt_exact_module.CandidateRecord(
                    candidate_id=5_000_016,
                    query_id=5,
                    slab_id=0,
                    object_a_id=1,
                    patch_a_id=0,
                    object_b_id=2,
                    patch_b_id=6,
                    proxy_type_a=rt_exact_module.ProxyType.SWEPT_AABB,
                    proxy_type_b=rt_exact_module.ProxyType.SWEPT_AABB,
                    rt_hit_count=1,
                    motion_bound=[0.0, 0.0, 0.0, 0.0],
                    flags=rt_exact_module.RAW_CANDIDATE_VALID
                    | rt_exact_module.RAW_CANDIDATE_AABB_OVERLAP,
                ),
            )
            self.runtime_query_ids = (
                _FakeRuntimeQueryId(5, 5),
                _FakeRuntimeQueryId(6, 6),
            )

    class _FakeCpp:
        ExternalBatchCandidateResult = _FakeResult
        RuntimeQueryIdMapping = _FakeRuntimeQueryId

        @staticmethod
        def generate_candidates_for_external_batch(
            incoming_batch,
            *,
            backend_name: str,
            allow_optix_cpu_fallback: bool,
        ):
            assert incoming_batch is batch
            assert backend_name == "optix_rt"
            assert allow_optix_cpu_fallback is False
            return _FakeResult()

    monkeypatch.setattr(rt_exact_module, "_try_load_p2cccd_cpp", lambda: _FakeCpp())
    monkeypatch.setattr(
        rt_exact_module,
        "external_query_to_broad_phase_primitives",
        lambda query: (_ for _ in ()).throw(AssertionError("python primitive path should not run")),
    )

    candidates, stats, runtime_ids = rt_exact_module._make_external_candidates(
        batch,
        RTExactConfig(backend_name="optix_rt"),
    )

    assert len(candidates) == 1
    assert candidates[0].query_id == 5
    assert stats.backend_name == "optix_rt"
    assert stats.primitive_count == 4
    assert stats.raw_hit_count == 1
    assert stats.compact_candidate_count == 1
    assert stats.timing.build_ms == 4.0
    assert stats.timing.update_ms == 0.5
    assert stats.timing.trace_ms == 1.5
    assert stats.timing.compact_ms == 0.25
    assert stats.timing.total_ms == 6.25
    assert runtime_ids == {5: 5, 6: 6}


def test_rt_exact_runs_on_external_scalable_sample_batch() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch("cloth-funnel", family="ee", step=227, limit=8)
    config = RTExactConfig(
        exact=PureExactCPUConfig(eps_time=1.0e-5, eps_space=1.0e-8, max_subdivision_depth=24)
    )

    result = run_rt_exact_on_external_batch(batch, config)

    assert result.source_name == "Sample-Scalable-CCD-Data"
    assert result.batch_id == "cloth-funnel:227ee"
    assert result.benchmark.query_count == 8
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.candidate_stats.backend_name == "cpu_reference_rt"
    assert result.candidate_stats.raw_hit_count == len(result.candidates)
    assert len(result.candidates) == len(result.work_items) == len(result.certificates)
    assert all(item.feature_family_mask == FEATURE_FAMILY_EDGE_EDGE for item in result.work_items)
    assert result.final_fn_zero


def test_rt_exact_runs_on_internal_generated_dataset_without_false_negatives() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=2, seed=303)
    )

    result = run_rt_exact_on_generated_dataset(dataset)

    assert result.benchmark.query_count == len(dataset.rows)
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.candidate_stats.compact_candidate_count <= len(dataset.rows)
    assert result.queue_conserved
    assert result.final_fn_zero


def test_rt_exact_coverage_guard_rejects_disappearing_certificate() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch("cloth-funnel", family="vf", step=227, limit=2)
    result = run_rt_exact_on_external_batch(batch)

    try:
        validate_rt_exact_coverage(result.candidates, result.work_items, result.certificates[:-1])
    except ValueError as exc:
        assert "certificate" in str(exc)
    else:
        raise AssertionError("expected missing certificate coverage error")


def test_rt_exact_coverage_guard_rejects_duplicate_parent_candidate() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch("cloth-funnel", family="vf", step=227, limit=4)
    result = run_rt_exact_on_external_batch(batch)
    if len(result.work_items) < 2:
        return
    duplicated = list(result.work_items)
    duplicated[1] = replace(duplicated[1], parent_candidate_id=duplicated[0].parent_candidate_id)

    try:
        validate_rt_exact_coverage(result.candidates, duplicated, result.certificates)
    except ValueError as exc:
        assert "duplicates a parent candidate" in str(exc)
    else:
        raise AssertionError("expected duplicate parent candidate error")


def test_rt_exact_rejects_unknown_backend_name() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch("cloth-funnel", family="vf", step=227, limit=1)

    try:
        run_rt_exact_on_external_batch(batch, RTExactConfig(backend_name="unknown"))
    except ValueError as exc:
        assert "backend_name" in str(exc)
    else:
        raise AssertionError("expected backend validation error")
