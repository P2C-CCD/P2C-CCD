from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Literal, Sequence

from p2cccd.contracts import (
    AuditLogRow,
    BenchmarkRow,
    CandidateRecord,
    CertificateResult,
    ExactWorkItem,
    ProposalSource,
)
from p2cccd.data.dataset import GeneratedDataset
from p2cccd.data.samplers import MotionDiscPairSample
from p2cccd.datasets.ccd import DatasetQueryBatch
from p2cccd.proposal.features import ProposalFeatureRow
from p2cccd.proposal.inference import (
    DEFAULT_OOD_ABS_FEATURE_THRESHOLD,
    ProposalPrediction,
    is_ood_feature_row,
)
from p2cccd.proposal.stpf_model import STPFModelPreset
from p2cccd.validators import validate_candidate_record, validate_exact_work_item

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
from .rt_stpf_exact import (
    RTSTPFExactConfig,
    STPFScheduleStats,
    _make_benchmark_row,
    _predicted_family_mask,
    _run_stpf_predictions,
    _valid_prediction,
    _validate_config as _validate_rt_stpf_config,
    proposal_feature_rows_from_rt_candidates,
)


AblationMode = Literal["interval_only", "ranking_only"]


@dataclass(frozen=True, slots=True)
class STPFHeadAblationConfig:
    exact: PureExactCPUConfig = field(default_factory=PureExactCPUConfig)
    rt_backend_name: str = "cpu_reference_rt"
    same_query_only: bool = True
    use_dummy_policy: bool = False
    allow_default_model: bool = True
    model_preset: STPFModelPreset | str = STPFModelPreset.LIGHTWEIGHT_MLP
    model_seed: int = 13
    model_checkpoint_path: str | None = None
    model_device: str | None = None
    proposal_batch_size: int = 1024
    family_score_threshold: float = 0.5
    uncertainty_fallback_threshold: float = 0.95
    ood_abs_feature_threshold: float = DEFAULT_OOD_ABS_FEATURE_THRESHOLD
    preserve_candidate_order: bool = False
    first_work_item_id: int = 1
    first_event_id: int = 1
    first_timestamp_us: int = 1


@dataclass(frozen=True, slots=True)
class IntervalOnlyConfig(STPFHeadAblationConfig):
    """Use only the interval head for proposal scheduling."""


@dataclass(frozen=True, slots=True)
class RankingOnlyConfig(STPFHeadAblationConfig):
    """Use only the family-ranking head for proposal scheduling."""


@dataclass(frozen=True, slots=True)
class STPFHeadAblationResult:
    benchmark: BenchmarkRow
    query_results: tuple[PureExactQueryResult, ...]
    candidates: tuple[CandidateRecord, ...]
    feature_rows: tuple[ProposalFeatureRow, ...]
    proposal_predictions: tuple[ProposalPrediction, ...]
    work_items: tuple[ExactWorkItem, ...]
    certificates: tuple[CertificateResult, ...]
    audit_log: tuple[AuditLogRow, ...]
    candidate_stats: RtCandidateStats
    schedule_stats: STPFScheduleStats
    ablation_mode: AblationMode
    source_name: str
    scene_name: str
    batch_id: str

    @property
    def final_fn_zero(self) -> bool:
        return self.benchmark.fn_count == 0

    @property
    def queue_conserved(self) -> bool:
        return (
            len(self.candidates)
            == len(self.feature_rows)
            == len(self.proposal_predictions)
            == len(self.work_items)
            == len(self.certificates)
        )


def _to_rt_stpf_config(config: STPFHeadAblationConfig) -> RTSTPFExactConfig:
    return RTSTPFExactConfig(
        exact=config.exact,
        rt_backend_name=config.rt_backend_name,
        same_query_only=config.same_query_only,
        use_dummy_policy=config.use_dummy_policy,
        allow_default_model=config.allow_default_model,
        model_preset=config.model_preset,
        model_seed=config.model_seed,
        model_checkpoint_path=config.model_checkpoint_path,
        model_device=config.model_device,
        proposal_batch_size=config.proposal_batch_size,
        family_score_threshold=config.family_score_threshold,
        uncertainty_fallback_threshold=config.uncertainty_fallback_threshold,
        ood_abs_feature_threshold=config.ood_abs_feature_threshold,
        preserve_candidate_order=config.preserve_candidate_order,
        first_work_item_id=config.first_work_item_id,
        first_event_id=config.first_event_id,
        first_timestamp_us=config.first_timestamp_us,
    )


def _to_rt_exact_config(config: STPFHeadAblationConfig) -> RTExactConfig:
    return RTExactConfig(
        exact=config.exact,
        backend_name=config.rt_backend_name,
        same_query_only=config.same_query_only,
        first_work_item_id=config.first_work_item_id,
        first_event_id=config.first_event_id,
        first_timestamp_us=config.first_timestamp_us,
    )


def _validate_config(config: STPFHeadAblationConfig) -> RTSTPFExactConfig:
    return _validate_rt_stpf_config(_to_rt_stpf_config(config))


def _max_finite(values: Sequence[float], *, fallback: float = 0.0) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return fallback
    return max(finite_values)


def _fallback_flags(
    row: ProposalFeatureRow | None,
    prediction: ProposalPrediction | None,
    config: RTSTPFExactConfig,
) -> tuple[bool, bool, bool, bool, bool]:
    missing = prediction is None
    invalid = prediction is not None and not _valid_prediction(prediction)
    ood = row is None or is_ood_feature_row(
        row,
        abs_feature_threshold=config.ood_abs_feature_threshold,
    )
    high_uncertainty = bool(
        prediction is not None
        and math.isfinite(float(prediction.uncertainty_score))
        and prediction.uncertainty_score >= config.uncertainty_fallback_threshold
    )
    return missing or invalid or ood or high_uncertainty, missing, invalid, ood, high_uncertainty


def _schedule_with_head_ablation(
    candidates: Sequence[CandidateRecord],
    rows: Sequence[ProposalFeatureRow],
    predictions: Sequence[ProposalPrediction],
    *,
    family_by_runtime_query_id: dict[int, int],
    config: RTSTPFExactConfig,
    mode: AblationMode,
) -> tuple[tuple[ExactWorkItem, ...], STPFScheduleStats]:
    rows_by_candidate = {row.candidate_id: row for row in rows}
    predictions_by_candidate = {prediction.candidate_id: prediction for prediction in predictions}
    scheduled: list[ExactWorkItem] = []
    fallback_count = 0
    missing_count = 0
    invalid_count = 0
    ood_count = 0
    uncertainty_count = 0

    for candidate in candidates:
        validate_candidate_record(candidate)
        row = rows_by_candidate.get(candidate.candidate_id)
        prediction = predictions_by_candidate.get(candidate.candidate_id)
        fallback, missing, invalid, ood, high_uncertainty = _fallback_flags(row, prediction, config)
        fallback_count += 1 if fallback else 0
        missing_count += 1 if missing else 0
        invalid_count += 1 if invalid else 0
        ood_count += 1 if ood else 0
        uncertainty_count += 1 if high_uncertainty else 0

        base_mask = family_by_runtime_query_id.get(candidate.query_id, FEATURE_FAMILY_CONSERVATIVE)
        feature_mask = base_mask
        priority = float(candidate.rt_hit_count)
        source = ProposalSource.FALLBACK

        if not fallback and prediction is not None:
            source = ProposalSource.REFINED
            if mode == "interval_only":
                priority = _max_finite(prediction.interval_scores)
            elif mode == "ranking_only":
                feature_mask = base_mask | _predicted_family_mask(
                    prediction,
                    config.family_score_threshold,
                )
                priority = _max_finite(prediction.family_scores)
            else:
                raise ValueError(f"unsupported STPF head ablation mode: {mode}")

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
                feature_family_mask=feature_mask,
                topk_feature_ids_offset=0,
                depth=0,
                priority_score=priority,
                source=source,
            )
        )

    if not config.preserve_candidate_order:
        scheduled.sort(key=lambda item: item.priority_score, reverse=True)
    for offset, item in enumerate(scheduled):
        scheduled[offset] = validate_exact_work_item(
            ExactWorkItem(
                work_item_id=config.first_work_item_id + offset,
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
    stats = STPFScheduleStats(
        raw_candidate_count=len(candidates),
        feature_row_count=len(rows),
        proposal_output_count=len(predictions),
        work_item_count=len(scheduled),
        fallback_count=fallback_count,
        missing_proposal_fallback_count=missing_count,
        invalid_proposal_fallback_count=invalid_count,
        ood_fallback_count=ood_count,
        high_uncertainty_fallback_count=uncertainty_count,
        reordered_count=reordered_count,
        monotonic_safe=True,
    )
    return tuple(scheduled), stats


def schedule_exact_work_items_interval_only(
    candidates: Sequence[CandidateRecord],
    rows: Sequence[ProposalFeatureRow],
    predictions: Sequence[ProposalPrediction],
    *,
    family_by_runtime_query_id: dict[int, int],
    config: IntervalOnlyConfig | STPFHeadAblationConfig,
) -> tuple[tuple[ExactWorkItem, ...], STPFScheduleStats]:
    return _schedule_with_head_ablation(
        candidates,
        rows,
        predictions,
        family_by_runtime_query_id=family_by_runtime_query_id,
        config=_validate_config(config),
        mode="interval_only",
    )


def schedule_exact_work_items_ranking_only(
    candidates: Sequence[CandidateRecord],
    rows: Sequence[ProposalFeatureRow],
    predictions: Sequence[ProposalPrediction],
    *,
    family_by_runtime_query_id: dict[int, int],
    config: RankingOnlyConfig | STPFHeadAblationConfig,
) -> tuple[tuple[ExactWorkItem, ...], STPFScheduleStats]:
    return _schedule_with_head_ablation(
        candidates,
        rows,
        predictions,
        family_by_runtime_query_id=family_by_runtime_query_id,
        config=_validate_config(config),
        mode="ranking_only",
    )


def _run_external(
    batch: DatasetQueryBatch,
    config: STPFHeadAblationConfig,
    *,
    mode: AblationMode,
    backend: BroadPhaseBackend | None,
    model,
    device: str | None,
) -> STPFHeadAblationResult:
    if not batch.queries:
        raise ValueError(f"{mode} external batch requires at least one query")
    stpf_config = _validate_config(config)
    rt_config = _to_rt_exact_config(config)
    candidates, candidate_stats, runtime_ids = _make_external_candidates(
        batch,
        rt_config,
        backend=backend,
    )
    family_by_runtime_query_id = {
        runtime_ids[query.query_id]: _family_mask_for_external(query)
        for query in batch.queries
    }

    proposal_start = time.perf_counter()
    feature_rows = proposal_feature_rows_from_rt_candidates(
        candidates,
        family_by_runtime_query_id=family_by_runtime_query_id,
        candidate_stats=candidate_stats,
    )
    proposal_predictions = _run_stpf_predictions(
        feature_rows,
        stpf_config,
        runtime=model,
        device=device,
    )
    work_items, schedule_stats = _schedule_with_head_ablation(
        candidates,
        feature_rows,
        proposal_predictions,
        family_by_runtime_query_id=family_by_runtime_query_id,
        config=stpf_config,
        mode=mode,
    )
    proposal_elapsed_ms = (time.perf_counter() - proposal_start) * 1000.0

    query_results, certificates, audit_log, exact_elapsed_ms, _ = _process_external_exact_work_queue(
        batch,
        candidates,
        work_items,
        runtime_ids,
        rt_config,
    )
    validate_rt_exact_coverage(candidates, work_items, certificates)
    return STPFHeadAblationResult(
        benchmark=_make_benchmark_row(
            query_results,
            candidate_stats=candidate_stats,
            schedule_stats=schedule_stats,
            proposal_elapsed_ms=proposal_elapsed_ms,
            exact_elapsed_ms=exact_elapsed_ms,
        ),
        query_results=query_results,
        candidates=candidates,
        feature_rows=feature_rows,
        proposal_predictions=proposal_predictions,
        work_items=work_items,
        certificates=certificates,
        audit_log=audit_log,
        candidate_stats=candidate_stats,
        schedule_stats=schedule_stats,
        ablation_mode=mode,
        source_name=batch.source_name,
        scene_name=batch.scene_name,
        batch_id=batch.batch_id,
    )


def _run_internal(
    samples: Sequence[MotionDiscPairSample],
    config: STPFHeadAblationConfig,
    *,
    mode: AblationMode,
    backend: BroadPhaseBackend | None,
    model,
    device: str | None,
) -> STPFHeadAblationResult:
    if not samples:
        raise ValueError(f"{mode} internal sample run requires at least one sample")
    stpf_config = _validate_config(config)
    rt_config = _to_rt_exact_config(config)
    candidates, candidate_stats, runtime_ids = _make_internal_candidates(
        samples,
        rt_config,
        backend=backend,
    )
    family_by_runtime_query_id = {
        runtime_ids[sample.query_id]: FEATURE_FAMILY_CONSERVATIVE
        for sample in samples
    }

    proposal_start = time.perf_counter()
    feature_rows = proposal_feature_rows_from_rt_candidates(
        candidates,
        family_by_runtime_query_id=family_by_runtime_query_id,
        candidate_stats=candidate_stats,
    )
    proposal_predictions = _run_stpf_predictions(
        feature_rows,
        stpf_config,
        runtime=model,
        device=device,
    )
    work_items, schedule_stats = _schedule_with_head_ablation(
        candidates,
        feature_rows,
        proposal_predictions,
        family_by_runtime_query_id=family_by_runtime_query_id,
        config=stpf_config,
        mode=mode,
    )
    proposal_elapsed_ms = (time.perf_counter() - proposal_start) * 1000.0

    query_results, certificates, audit_log, exact_elapsed_ms, _ = _process_internal_exact_work_queue(
        samples,
        candidates,
        work_items,
        runtime_ids,
        rt_config,
    )
    validate_rt_exact_coverage(candidates, work_items, certificates)
    return STPFHeadAblationResult(
        benchmark=_make_benchmark_row(
            query_results,
            candidate_stats=candidate_stats,
            schedule_stats=schedule_stats,
            proposal_elapsed_ms=proposal_elapsed_ms,
            exact_elapsed_ms=exact_elapsed_ms,
        ),
        query_results=query_results,
        candidates=candidates,
        feature_rows=feature_rows,
        proposal_predictions=proposal_predictions,
        work_items=work_items,
        certificates=certificates,
        audit_log=audit_log,
        candidate_stats=candidate_stats,
        schedule_stats=schedule_stats,
        ablation_mode=mode,
        source_name="internal_analytic_oracle",
        scene_name="programmatic_motion_disc_pairs",
        batch_id="internal_samples",
    )


def run_interval_only_on_external_batch(
    batch: DatasetQueryBatch,
    config: IntervalOnlyConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
    model=None,
    device: str | None = None,
) -> STPFHeadAblationResult:
    return _run_external(
        batch,
        config or IntervalOnlyConfig(),
        mode="interval_only",
        backend=backend,
        model=model,
        device=device,
    )


def run_interval_only_on_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    config: IntervalOnlyConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
    model=None,
    device: str | None = None,
) -> STPFHeadAblationResult:
    return _run_internal(
        samples,
        config or IntervalOnlyConfig(),
        mode="interval_only",
        backend=backend,
        model=model,
        device=device,
    )


def run_interval_only_on_generated_dataset(
    dataset: GeneratedDataset,
    config: IntervalOnlyConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
    model=None,
    device: str | None = None,
) -> STPFHeadAblationResult:
    return run_interval_only_on_internal_samples(
        dataset.samples,
        config,
        backend=backend,
        model=model,
        device=device,
    )


def run_ranking_only_on_external_batch(
    batch: DatasetQueryBatch,
    config: RankingOnlyConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
    model=None,
    device: str | None = None,
) -> STPFHeadAblationResult:
    return _run_external(
        batch,
        config or RankingOnlyConfig(),
        mode="ranking_only",
        backend=backend,
        model=model,
        device=device,
    )


def run_ranking_only_on_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    config: RankingOnlyConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
    model=None,
    device: str | None = None,
) -> STPFHeadAblationResult:
    return _run_internal(
        samples,
        config or RankingOnlyConfig(),
        mode="ranking_only",
        backend=backend,
        model=model,
        device=device,
    )


def run_ranking_only_on_generated_dataset(
    dataset: GeneratedDataset,
    config: RankingOnlyConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
    model=None,
    device: str | None = None,
) -> STPFHeadAblationResult:
    return run_ranking_only_on_internal_samples(
        dataset.samples,
        config,
        backend=backend,
        model=model,
        device=device,
    )
