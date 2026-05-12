from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from p2cccd.contracts import (
    AuditLogRow,
    BenchmarkRow,
    CandidateRecord,
    CertificateResult,
    ExactWorkItem,
    ProposalSource,
)
from p2cccd.data.dataset import GeneratedDataset
from p2cccd.data.oracle import evaluate_swept_sphere_oracle
from p2cccd.data.samplers import MotionDiscPairSample
from p2cccd.datasets.ccd import DatasetQueryBatch
from p2cccd.validators import (
    validate_benchmark_row,
    validate_candidate_record,
    validate_exact_work_item,
)

from .bvh_exact import BroadPhaseBackend
from .pure_exact_cpu import PureExactCPUConfig, PureExactQueryResult
from .rt_exact import (
    FEATURE_FAMILY_CONSERVATIVE,
    RTExactConfig,
    RtCandidateStats,
    _family_mask_for_external,
    _make_external_candidates,
    _make_internal_candidates,
    _process_external_exact_work_queue,
    _process_internal_exact_work_queue,
    validate_rt_exact_coverage,
)


@dataclass(frozen=True, slots=True)
class NoProposalConfig:
    exact: PureExactCPUConfig = field(default_factory=PureExactCPUConfig)
    rt_backend_name: str = "cpu_reference_rt"
    enable_cuda_exact: bool = True
    same_query_only: bool = True
    preserve_candidate_order: bool = True
    first_work_item_id: int = 1
    first_event_id: int = 1
    first_timestamp_us: int = 1


@dataclass(frozen=True, slots=True)
class NoProposalStats:
    raw_candidate_count: int
    work_item_count: int
    fallback_count: int
    reordered_count: int
    monotonic_safe: bool


@dataclass(frozen=True, slots=True)
class NoProposalResult:
    benchmark: BenchmarkRow
    query_results: tuple[PureExactQueryResult, ...]
    candidates: tuple[CandidateRecord, ...]
    work_items: tuple[ExactWorkItem, ...]
    certificates: tuple[CertificateResult, ...]
    audit_log: tuple[AuditLogRow, ...]
    candidate_stats: RtCandidateStats
    no_proposal_stats: NoProposalStats
    exact_backend_name: str
    source_name: str
    scene_name: str
    batch_id: str

    @property
    def final_fn_zero(self) -> bool:
        return self.benchmark.fn_count == 0

    @property
    def queue_conserved(self) -> bool:
        return len(self.candidates) == len(self.work_items) == len(self.certificates)


def _validate_config(config: NoProposalConfig) -> NoProposalConfig:
    if config.rt_backend_name not in {"cpu_reference_rt", "optix_compatible", "optix_rt"}:
        raise ValueError("NoProposalConfig.rt_backend_name is unsupported")
    if config.first_work_item_id <= 0:
        raise ValueError("NoProposalConfig.first_work_item_id must be positive")
    if config.first_event_id <= 0:
        raise ValueError("NoProposalConfig.first_event_id must be positive")
    if config.first_timestamp_us <= 0:
        raise ValueError("NoProposalConfig.first_timestamp_us must be positive")
    return config


def _rt_config_from_no_proposal(config: NoProposalConfig) -> RTExactConfig:
    return RTExactConfig(
        exact=config.exact,
        backend_name=config.rt_backend_name,
        enable_cuda_exact=config.enable_cuda_exact,
        same_query_only=config.same_query_only,
        first_work_item_id=config.first_work_item_id,
        first_event_id=config.first_event_id,
        first_timestamp_us=config.first_timestamp_us,
    )


def schedule_exact_work_items_no_proposal(
    candidates: Sequence[CandidateRecord],
    *,
    family_by_runtime_query_id: dict[int, int],
    config: NoProposalConfig,
) -> tuple[tuple[ExactWorkItem, ...], NoProposalStats]:
    cfg = _validate_config(config)
    scheduled: list[ExactWorkItem] = []
    for candidate in candidates:
        validate_candidate_record(candidate)
        scheduled.append(
            ExactWorkItem(
                work_item_id=1,
                parent_candidate_id=candidate.candidate_id,
                query_id=candidate.query_id,
                slab_id=candidate.slab_id,
                patch_a_id=candidate.patch_a_id,
                patch_b_id=candidate.patch_b_id,
                interval_t0=0.0,
                interval_t1=1.0,
                feature_family_mask=family_by_runtime_query_id.get(
                    candidate.query_id,
                    FEATURE_FAMILY_CONSERVATIVE,
                ),
                topk_feature_ids_offset=0,
                depth=0,
                priority_score=float(candidate.rt_hit_count),
                source=ProposalSource.FALLBACK,
            )
        )

    if not cfg.preserve_candidate_order:
        scheduled.sort(key=lambda item: item.priority_score, reverse=True)
    for offset, item in enumerate(scheduled):
        scheduled[offset] = validate_exact_work_item(
            ExactWorkItem(
                work_item_id=cfg.first_work_item_id + offset,
                parent_candidate_id=item.parent_candidate_id,
                query_id=item.query_id,
                slab_id=item.slab_id,
                patch_a_id=item.patch_a_id,
                patch_b_id=item.patch_b_id,
                interval_t0=item.interval_t0,
                interval_t1=item.interval_t1,
                feature_family_mask=item.feature_family_mask,
                topk_feature_ids_offset=item.topk_feature_ids_offset,
                depth=item.depth,
                priority_score=item.priority_score,
                source=item.source,
            )
        )

    original_parent_order = [candidate.candidate_id for candidate in candidates]
    scheduled_parent_order = [item.parent_candidate_id for item in scheduled]
    reordered_count = sum(
        1
        for lhs, rhs in zip(original_parent_order, scheduled_parent_order)
        if lhs != rhs
    )
    stats = NoProposalStats(
        raw_candidate_count=len(candidates),
        work_item_count=len(scheduled),
        fallback_count=len(scheduled),
        reordered_count=reordered_count,
        monotonic_safe=True,
    )
    return tuple(scheduled), stats


def _make_benchmark_row(
    query_results: Sequence[PureExactQueryResult],
    *,
    candidate_stats: RtCandidateStats,
    no_proposal_stats: NoProposalStats,
    exact_elapsed_ms: float,
) -> BenchmarkRow:
    if not query_results:
        raise ValueError("NoProposal requires at least one query")
    known = [result for result in query_results if result.ground_truth_collision is not None]
    fn_count = sum(
        1
        for result in known
        if result.ground_truth_collision is True and not result.predicted_collision
    )
    fp_count = sum(
        1
        for result in known
        if result.ground_truth_collision is False and result.predicted_collision
    )
    total_exact_evals = sum(result.exact_evals for result in query_results)
    total_depth = sum(result.max_depth for result in query_results)
    total_ms = candidate_stats.timing.total_ms + exact_elapsed_ms
    row = BenchmarkRow(
        query_count=len(query_results),
        fn_count=fn_count,
        fp_count=fp_count,
        candidate_recall=candidate_stats.candidate_recall,
        avg_candidates=candidate_stats.compact_candidate_count / len(query_results),
        avg_exact_evals=total_exact_evals / len(query_results),
        avg_subdivision_depth=total_depth / len(query_results),
        fallback_ratio=no_proposal_stats.fallback_count
        / max(1, no_proposal_stats.raw_candidate_count),
        rt_ms=candidate_stats.timing.total_ms,
        proposal_ms=0.0,
        exact_ms=exact_elapsed_ms,
        total_ms=total_ms,
        qps=0.0 if total_ms <= 0.0 else 1000.0 * len(query_results) / total_ms,
    )
    return validate_benchmark_row(row)


def run_no_proposal_on_external_batch(
    batch: DatasetQueryBatch,
    config: NoProposalConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> NoProposalResult:
    if not batch.queries:
        raise ValueError("NoProposal external batch requires at least one query")
    cfg = _validate_config(config or NoProposalConfig())
    rt_config = _rt_config_from_no_proposal(cfg)
    candidates, candidate_stats, runtime_ids = _make_external_candidates(
        batch,
        rt_config,
        backend=backend,
    )
    family_by_runtime_query_id = {
        runtime_ids[query.query_id]: _family_mask_for_external(query)
        for query in batch.queries
    }
    work_items, no_proposal_stats = schedule_exact_work_items_no_proposal(
        candidates,
        family_by_runtime_query_id=family_by_runtime_query_id,
        config=cfg,
    )

    query_results, certificates, audit_log, exact_elapsed_ms, exact_backend_name = _process_external_exact_work_queue(
        batch,
        candidates,
        work_items,
        runtime_ids,
        rt_config,
    )
    validate_rt_exact_coverage(candidates, work_items, certificates)
    return NoProposalResult(
        benchmark=_make_benchmark_row(
            query_results,
            candidate_stats=candidate_stats,
            no_proposal_stats=no_proposal_stats,
            exact_elapsed_ms=exact_elapsed_ms,
        ),
        query_results=query_results,
        candidates=candidates,
        work_items=work_items,
        certificates=certificates,
        audit_log=audit_log,
        candidate_stats=candidate_stats,
        no_proposal_stats=no_proposal_stats,
        exact_backend_name=exact_backend_name,
        source_name=batch.source_name,
        scene_name=batch.scene_name,
        batch_id=batch.batch_id,
    )


def run_no_proposal_on_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    config: NoProposalConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> NoProposalResult:
    if not samples:
        raise ValueError("NoProposal internal sample run requires at least one sample")
    cfg = _validate_config(config or NoProposalConfig())
    rt_config = _rt_config_from_no_proposal(cfg)
    oracle_traces_by_query_id = {
        sample.query_id: evaluate_swept_sphere_oracle(sample) for sample in samples
    }
    candidates, candidate_stats, runtime_ids = _make_internal_candidates(
        samples,
        rt_config,
        backend=backend,
        oracle_traces_by_query_id=oracle_traces_by_query_id,
    )
    family_by_runtime_query_id = {
        runtime_ids[sample.query_id]: FEATURE_FAMILY_CONSERVATIVE
        for sample in samples
    }
    work_items, no_proposal_stats = schedule_exact_work_items_no_proposal(
        candidates,
        family_by_runtime_query_id=family_by_runtime_query_id,
        config=cfg,
    )

    query_results, certificates, audit_log, exact_elapsed_ms, exact_backend_name = _process_internal_exact_work_queue(
        samples,
        candidates,
        work_items,
        runtime_ids,
        rt_config,
        oracle_traces_by_query_id=oracle_traces_by_query_id,
    )
    validate_rt_exact_coverage(candidates, work_items, certificates)
    return NoProposalResult(
        benchmark=_make_benchmark_row(
            query_results,
            candidate_stats=candidate_stats,
            no_proposal_stats=no_proposal_stats,
            exact_elapsed_ms=exact_elapsed_ms,
        ),
        query_results=query_results,
        candidates=candidates,
        work_items=work_items,
        certificates=certificates,
        audit_log=audit_log,
        candidate_stats=candidate_stats,
        no_proposal_stats=no_proposal_stats,
        exact_backend_name=exact_backend_name,
        source_name="internal_analytic_oracle",
        scene_name="programmatic_motion_disc_pairs",
        batch_id="internal_samples",
    )


def run_no_proposal_on_generated_dataset(
    dataset: GeneratedDataset,
    config: NoProposalConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> NoProposalResult:
    return run_no_proposal_on_internal_samples(dataset.samples, config, backend=backend)
