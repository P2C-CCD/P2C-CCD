from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

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
from p2cccd.proposal.features import (
    PROPOSAL_FAMILY_COUNT,
    PROPOSAL_FEATURE_DIM,
    PROPOSAL_INTERVAL_BIN_COUNT,
    TARGET_COST,
    TARGET_FAMILY,
    TARGET_INTERVAL,
    TARGET_PRIORITY,
    TARGET_UNCERTAINTY,
    ProposalFeatureRow,
    validate_proposal_feature_row,
)
from p2cccd.proposal.inference import (
    DEFAULT_OOD_ABS_FEATURE_THRESHOLD,
    ProposalPrediction,
    batched_stpf_inference,
    is_ood_feature_row,
    validate_proposal_prediction,
)
from p2cccd.proposal.ort_inference import (
    DEFAULT_ORT_OPSET_VERSION,
    ORTInferenceSession,
    batched_stpf_inference_ort_arrays,
    batched_stpf_inference_ort,
    create_ort_inference_session,
    ensure_stpf_model_onnx,
)
from p2cccd.proposal.stpf_model import (
    STPFModelPreset,
    build_stpf_model,
    build_stpf_model_from_checkpoint_payload,
)
from p2cccd.validators import (
    validate_benchmark_row,
    validate_candidate_record,
    validate_exact_work_item,
)

from .bvh_exact import BroadPhaseBackend
from .bvh_exact import _try_load_p2cccd_cpp
from .pure_exact_cpu import PureExactCPUConfig, PureExactQueryResult
from .rt_exact import (
    FEATURE_FAMILY_CONSERVATIVE,
    FEATURE_FAMILY_EDGE_EDGE,
    FEATURE_FAMILY_POINT_TRIANGLE,
    RTExactConfig,
    RtCandidateTiming,
    RtCandidateStats,
    _candidate_record_from_cpp,
    _family_mask_for_external,
    _make_external_candidates,
    _make_internal_candidates,
    _process_external_exact_work_queue,
    _process_internal_exact_work_queue,
    validate_rt_exact_coverage,
)

_WARMED_ORT_RUNTIME_IDS: set[int] = set()


@dataclass(frozen=True, slots=True)
class STPFScheduleStats:
    raw_candidate_count: int
    feature_row_count: int
    proposal_output_count: int
    work_item_count: int
    fallback_count: int
    missing_proposal_fallback_count: int
    invalid_proposal_fallback_count: int
    ood_fallback_count: int
    high_uncertainty_fallback_count: int
    reordered_count: int
    monotonic_safe: bool


@dataclass(frozen=True, slots=True)
class RTSTPFExactConfig:
    exact: PureExactCPUConfig = field(default_factory=PureExactCPUConfig)
    rt_backend_name: str = "cpu_reference_rt"
    enable_cuda_exact: bool = True
    same_query_only: bool = True
    execution_profile: str = "manual"
    use_dummy_policy: bool = False
    allow_default_model: bool = True
    model_preset: STPFModelPreset | str = STPFModelPreset.LIGHTWEIGHT_MLP
    model_seed: int = 13
    model_checkpoint_path: str | None = None
    model_device: str | None = None
    inference_backend: str = "torch"
    ort_model_path: str | None = None
    ort_prefer_tensorrt: bool = True
    ort_allow_cuda_fallback: bool = True
    ort_allow_cpu_fallback: bool = True
    ort_opset_version: int = DEFAULT_ORT_OPSET_VERSION
    ort_warmup_passes: int = 1
    cpu_inference_row_threshold: int = 2048
    proposal_batch_size: int = 1024
    family_score_threshold: float = 0.5
    uncertainty_fallback_threshold: float = 0.95
    ood_abs_feature_threshold: float = DEFAULT_OOD_ABS_FEATURE_THRESHOLD
    auto_fastest_dummy_candidate_ratio_threshold: float = 1.25
    preserve_candidate_order: bool = False
    first_work_item_id: int = 1
    first_event_id: int = 1
    first_timestamp_us: int = 1


@dataclass(frozen=True, slots=True)
class RTSTPFExactResult:
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
    resolved_execution_profile_name: str
    inference_backend_name: str
    inference_provider_name: str
    exact_backend_name: str
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
            == len(self.work_items)
            == len(self.certificates)
        )


def _validate_config(config: RTSTPFExactConfig) -> RTSTPFExactConfig:
    if config.rt_backend_name not in {"cpu_reference_rt", "optix_compatible", "optix_rt"}:
        raise ValueError("RTSTPFExactConfig.rt_backend_name is unsupported")
    if config.execution_profile not in {"manual", "fastest_learned", "auto_fastest"}:
        raise ValueError("RTSTPFExactConfig.execution_profile is unsupported")
    if config.inference_backend not in {"torch", "ort"}:
        raise ValueError("RTSTPFExactConfig.inference_backend is unsupported")
    if config.use_dummy_policy:
        raise ValueError("RTSTPFExact no longer supports dummy proposal paths; use a learned STPF path instead")
    if config.proposal_batch_size <= 0:
        raise ValueError("RTSTPFExactConfig.proposal_batch_size must be positive")
    if config.ort_opset_version <= 0:
        raise ValueError("RTSTPFExactConfig.ort_opset_version must be positive")
    if config.ort_warmup_passes < 0:
        raise ValueError("RTSTPFExactConfig.ort_warmup_passes must be non-negative")
    if config.cpu_inference_row_threshold < 0:
        raise ValueError("RTSTPFExactConfig.cpu_inference_row_threshold must be non-negative")
    if config.model_seed < 0:
        raise ValueError("RTSTPFExactConfig.model_seed must be non-negative")
    if not math.isfinite(config.family_score_threshold):
        raise ValueError("RTSTPFExactConfig.family_score_threshold must be finite")
    if not math.isfinite(config.uncertainty_fallback_threshold) or config.uncertainty_fallback_threshold < 0.0:
        raise ValueError("RTSTPFExactConfig.uncertainty_fallback_threshold must be non-negative")
    if not math.isfinite(config.ood_abs_feature_threshold) or config.ood_abs_feature_threshold <= 0.0:
        raise ValueError("RTSTPFExactConfig.ood_abs_feature_threshold must be positive")
    if (
        not math.isfinite(config.auto_fastest_dummy_candidate_ratio_threshold)
        or config.auto_fastest_dummy_candidate_ratio_threshold <= 0.0
    ):
        raise ValueError("RTSTPFExactConfig.auto_fastest_dummy_candidate_ratio_threshold must be positive")
    if config.first_work_item_id <= 0:
        raise ValueError("RTSTPFExactConfig.first_work_item_id must be positive")
    if config.first_event_id <= 0:
        raise ValueError("RTSTPFExactConfig.first_event_id must be positive")
    if config.first_timestamp_us <= 0:
        raise ValueError("RTSTPFExactConfig.first_timestamp_us must be positive")
    return config


def _candidate_ratio(candidate_stats: RtCandidateStats, *, query_count: int) -> float:
    return candidate_stats.compact_candidate_count / max(1, query_count)


def _resolve_execution_profile(
    config: RTSTPFExactConfig,
    *,
    query_count: int,
    candidate_stats: RtCandidateStats,
) -> tuple[RTSTPFExactConfig, str]:
    profile = config.execution_profile
    if profile == "manual":
        return config, "manual"
    if profile == "fastest_learned":
        return (
            replace(
                config,
                use_dummy_policy=False,
                inference_backend="ort",
                ort_prefer_tensorrt=True,
                enable_cuda_exact=True,
            ),
            "fastest_learned",
        )
    ratio = _candidate_ratio(candidate_stats, query_count=query_count)
    return (
        replace(
            config,
            use_dummy_policy=False,
            inference_backend="ort",
            ort_prefer_tensorrt=True,
            enable_cuda_exact=True,
        ),
        f"auto_fastest:learned_ort@{ratio:.3f}",
    )


def _clamp_feature(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return max(-1.0e6, min(1.0e6, number))


@lru_cache(maxsize=1)
def _load_cpp_proposal_module() -> Any | None:
    cpp = _try_load_p2cccd_cpp()
    required = (
        "ProposalFeatureRow",
        "ProposalOutput",
        "ProposalSchedulingConfig",
        "ProposalScheduleStats",
        "ProposalRuntimeScheduleResult",
        "build_runtime_proposal_feature_rows",
        "run_dummy_runtime_proposal_schedule",
        "schedule_runtime_exact_work_items",
    )
    if cpp is None or any(not hasattr(cpp, name) for name in required):
        return None
    return cpp


def _proposal_feature_row_from_cpp(row: Any) -> ProposalFeatureRow:
    return validate_proposal_feature_row(
        ProposalFeatureRow(
            schema_version=int(row.schema_version),
            query_id=int(row.query_id),
            candidate_id=int(row.candidate_id),
            slab_id=int(row.slab_id),
            object_a_id=int(row.object_a_id),
            patch_a_id=int(row.patch_a_id),
            object_b_id=int(row.object_b_id),
            patch_b_id=int(row.patch_b_id),
            features=[float(value) for value in row.features],
            interval_targets=[float(value) for value in row.interval_targets],
            family_targets=[float(value) for value in row.family_targets],
            priority_target=float(row.priority_target),
            cost_target=float(row.cost_target),
            uncertainty_target=float(row.uncertainty_target),
            target_mask=int(row.target_mask),
        )
    )


def _proposal_prediction_from_cpp_output(output: Any) -> ProposalPrediction:
    uncertainty = float(output.uncertainty_score)
    return validate_proposal_prediction(
        ProposalPrediction(
            candidate_id=int(output.candidate_id),
            interval_scores=[float(value) for value in output.interval_scores],
            family_scores=[float(value) for value in output.family_scores],
            priority_score=float(output.priority_score),
            cost_score=float(output.cost_score),
            uncertainty_score=uncertainty,
            source="dummy_ood_fallback" if uncertainty >= 1.0 else "dummy",
        )
    )


def _cpp_array_fast_path_available(cpp: Any | None) -> bool:
    return cpp is not None and hasattr(cpp, "build_runtime_proposal_feature_arrays") and hasattr(
        cpp, "schedule_runtime_exact_work_items_from_arrays"
    )


def _cpp_direct_array_schedule_available(cpp: Any | None) -> bool:
    return cpp is not None and hasattr(cpp, "schedule_runtime_exact_work_items_from_proposal_arrays")


def _exact_work_item_from_cpp(item: Any) -> ExactWorkItem:
    return validate_exact_work_item(
        ExactWorkItem(
            work_item_id=int(item.work_item_id),
            parent_candidate_id=int(item.parent_candidate_id),
            query_id=int(item.query_id),
            slab_id=int(item.slab_id),
            patch_a_id=int(item.patch_a_id),
            patch_b_id=int(item.patch_b_id),
            interval_t0=float(item.interval_t0),
            interval_t1=float(item.interval_t1),
            feature_family_mask=int(item.feature_family_mask),
            topk_feature_ids_offset=int(item.topk_feature_ids_offset),
            depth=int(item.depth),
            priority_score=float(item.priority_score),
            source=ProposalSource(int(item.source)),
        )
    )


def _to_cpp_proposal_scheduling_config(cpp: Any, config: RTSTPFExactConfig) -> Any:
    cpp_config = cpp.ProposalSchedulingConfig()
    cpp_config.first_work_item_id = int(config.first_work_item_id)
    cpp_config.conservative_feature_family_mask = int(FEATURE_FAMILY_CONSERVATIVE)
    cpp_config.fallback_interval_t0 = 0.0
    cpp_config.fallback_interval_t1 = 1.0
    cpp_config.family_score_threshold = float(config.family_score_threshold)
    cpp_config.uncertainty_fallback_threshold = float(config.uncertainty_fallback_threshold)
    cpp_config.ood_abs_feature_threshold = float(config.ood_abs_feature_threshold)
    cpp_config.preserve_candidate_order = bool(config.preserve_candidate_order)
    return cpp_config


def _schedule_stats_from_cpp(
    stats: Any,
    *,
    feature_row_count: int,
) -> STPFScheduleStats:
    return STPFScheduleStats(
        raw_candidate_count=int(stats.raw_candidate_count),
        feature_row_count=int(feature_row_count),
        proposal_output_count=int(stats.proposal_output_count),
        work_item_count=int(stats.work_item_count),
        fallback_count=int(stats.fallback_count),
        missing_proposal_fallback_count=int(stats.missing_proposal_fallback_count),
        invalid_proposal_fallback_count=int(stats.invalid_proposal_fallback_count),
        ood_fallback_count=int(stats.ood_fallback_count),
        high_uncertainty_fallback_count=int(stats.high_uncertainty_fallback_count),
        reordered_count=int(stats.reordered_count),
        monotonic_safe=bool(stats.monotonic_safe),
    )


def _rt_candidate_stats_from_cpp_external_result(cpp_result: Any, requested_backend_name: str) -> RtCandidateStats:
    backend_name = str(cpp_result.backend_name)
    if backend_name == "cpu_reference" and requested_backend_name == "cpu_reference_rt":
        backend_name = "cpu_reference_rt"
    return RtCandidateStats(
        backend_name=backend_name,
        primitive_count=int(cpp_result.primitive_count),
        raw_hit_count=int(cpp_result.raw_hit_count),
        compact_candidate_count=int(cpp_result.compact_candidate_count),
        candidate_recall=float(cpp_result.candidate_recall),
        timing=RtCandidateTiming(
            build_ms=float(cpp_result.timing.build_ms),
            update_ms=float(cpp_result.timing.update_ms),
            trace_ms=float(cpp_result.timing.trace_ms),
            compact_ms=float(cpp_result.timing.compact_ms),
            total_ms=float(cpp_result.timing.total_ms),
        ),
    )


def _run_cpp_dummy_runtime_schedule(
    candidates: Sequence[CandidateRecord],
    *,
    family_by_runtime_query_id: dict[int, int],
    candidate_stats: RtCandidateStats,
    config: RTSTPFExactConfig,
    materialize_artifacts: bool,
) -> Any | None:
    cpp = _load_cpp_proposal_module()
    if cpp is None:
        return None
    return cpp.run_dummy_runtime_proposal_schedule(
        candidates,
        int(candidate_stats.primitive_count),
        int(candidate_stats.raw_hit_count),
        int(candidate_stats.compact_candidate_count),
        {int(query_id): int(mask) for query_id, mask in family_by_runtime_query_id.items()},
        _to_cpp_proposal_scheduling_config(cpp, config),
        bool(materialize_artifacts),
    )


def _family_targets(mask: int) -> list[float]:
    targets = [0.0] * PROPOSAL_FAMILY_COUNT
    if mask & FEATURE_FAMILY_POINT_TRIANGLE:
        targets[0] = 1.0
    if mask & FEATURE_FAMILY_EDGE_EDGE:
        targets[1] = 1.0
    return targets


def proposal_feature_rows_from_rt_candidates(
    candidates: Sequence[CandidateRecord],
    *,
    family_by_runtime_query_id: dict[int, int],
    candidate_stats: RtCandidateStats,
) -> tuple[ProposalFeatureRow, ...]:
    cpp = _load_cpp_proposal_module()
    if cpp is not None:
        cpp_rows = cpp.build_runtime_proposal_feature_rows(
            candidates,
            int(candidate_stats.primitive_count),
            int(candidate_stats.raw_hit_count),
            int(candidate_stats.compact_candidate_count),
            {int(query_id): int(mask) for query_id, mask in family_by_runtime_query_id.items()},
        )
        return tuple(_proposal_feature_row_from_cpp(row) for row in cpp_rows)

    rows: list[ProposalFeatureRow] = []
    candidates_per_proxy = candidate_stats.compact_candidate_count / max(1, candidate_stats.primitive_count)
    aabb_overlap_ratio = candidate_stats.raw_hit_count / max(1, candidate_stats.primitive_count // 2)
    avg_hits = candidate_stats.raw_hit_count / max(1, candidate_stats.compact_candidate_count)
    for candidate in candidates:
        validate_candidate_record(candidate)
        features = [0.0] * PROPOSAL_FEATURE_DIM
        features[0] = 0.0
        features[1] = 1.0
        features[2] = 1.0
        features[3] = _clamp_feature(candidate.rt_hit_count)
        features[4] = _clamp_feature(int(candidate.proxy_type_a))
        features[5] = _clamp_feature(int(candidate.proxy_type_b))
        for index, value in enumerate(candidate.motion_bound[:4]):
            features[6 + index] = _clamp_feature(value)
        features[19] = _clamp_feature(min(1.0, aabb_overlap_ratio))
        features[29] = _clamp_feature(candidates_per_proxy)
        features[30] = _clamp_feature(aabb_overlap_ratio)
        features[31] = _clamp_feature(avg_hits)

        interval_targets = [0.0] * PROPOSAL_INTERVAL_BIN_COUNT
        interval_targets[0] = 1.0
        base_family_mask = family_by_runtime_query_id.get(
            candidate.query_id,
            FEATURE_FAMILY_CONSERVATIVE,
        )
        row = ProposalFeatureRow(
            query_id=candidate.query_id,
            candidate_id=candidate.candidate_id,
            slab_id=candidate.slab_id,
            object_a_id=candidate.object_a_id,
            patch_a_id=candidate.patch_a_id,
            object_b_id=candidate.object_b_id,
            patch_b_id=candidate.patch_b_id,
            features=features,
            interval_targets=interval_targets,
            family_targets=_family_targets(base_family_mask),
            priority_target=min(1.0, 0.5 * candidate.rt_hit_count + 0.5 * candidates_per_proxy),
            cost_target=max(1.0, 1.0 + float(candidate.rt_hit_count)),
            uncertainty_target=0.25 if base_family_mask == FEATURE_FAMILY_CONSERVATIVE else 0.0,
            target_mask=TARGET_INTERVAL
            | TARGET_FAMILY
            | TARGET_PRIORITY
            | TARGET_COST
            | TARGET_UNCERTAINTY,
        )
        rows.append(validate_proposal_feature_row(row))
    return tuple(rows)


def _build_cpp_proposal_feature_arrays(
    candidates: Sequence[CandidateRecord],
    *,
    family_by_runtime_query_id: dict[int, int],
    candidate_stats: RtCandidateStats,
) -> dict[str, Any] | None:
    cpp = _load_cpp_proposal_module()
    if not _cpp_array_fast_path_available(cpp):
        return None
    return cpp.build_runtime_proposal_feature_arrays(
        candidates,
        int(candidate_stats.primitive_count),
        int(candidate_stats.raw_hit_count),
        int(candidate_stats.compact_candidate_count),
        {int(query_id): int(mask) for query_id, mask in family_by_runtime_query_id.items()},
    )


def _materialize_proposal_feature_rows_from_arrays(
    feature_arrays: dict[str, Any],
) -> tuple[ProposalFeatureRow, ...]:
    schema_version = int(feature_arrays["schema_version"])
    query_ids = np.asarray(feature_arrays["query_id"]).reshape(-1)
    candidate_ids = np.asarray(feature_arrays["candidate_id"]).reshape(-1)
    slab_ids = np.asarray(feature_arrays["slab_id"]).reshape(-1)
    object_a_ids = np.asarray(feature_arrays["object_a_id"]).reshape(-1)
    patch_a_ids = np.asarray(feature_arrays["patch_a_id"]).reshape(-1)
    object_b_ids = np.asarray(feature_arrays["object_b_id"]).reshape(-1)
    patch_b_ids = np.asarray(feature_arrays["patch_b_id"]).reshape(-1)
    features = np.asarray(feature_arrays["features"], dtype=np.float32)
    interval_targets = np.asarray(feature_arrays["interval_targets"], dtype=np.float32)
    family_targets = np.asarray(feature_arrays["family_targets"], dtype=np.float32)
    priority_target = np.asarray(feature_arrays["priority_target"], dtype=np.float32).reshape(-1)
    cost_target = np.asarray(feature_arrays["cost_target"], dtype=np.float32).reshape(-1)
    uncertainty_target = np.asarray(feature_arrays["uncertainty_target"], dtype=np.float32).reshape(-1)
    target_mask = np.asarray(feature_arrays["target_mask"]).reshape(-1)

    rows: list[ProposalFeatureRow] = []
    for index in range(int(candidate_ids.shape[0])):
        rows.append(
            validate_proposal_feature_row(
                ProposalFeatureRow(
                    schema_version=schema_version,
                    query_id=int(query_ids[index]),
                    candidate_id=int(candidate_ids[index]),
                    slab_id=int(slab_ids[index]),
                    object_a_id=int(object_a_ids[index]),
                    patch_a_id=int(patch_a_ids[index]),
                    object_b_id=int(object_b_ids[index]),
                    patch_b_id=int(patch_b_ids[index]),
                    features=[float(value) for value in features[index].tolist()],
                    interval_targets=[float(value) for value in interval_targets[index].tolist()],
                    family_targets=[float(value) for value in family_targets[index].tolist()],
                    priority_target=float(priority_target[index]),
                    cost_target=float(cost_target[index]),
                    uncertainty_target=float(uncertainty_target[index]),
                    target_mask=int(target_mask[index]),
                )
            )
        )
    return tuple(rows)


def _materialize_proposal_predictions_from_arrays(
    feature_arrays: dict[str, Any],
    prediction_arrays: dict[str, Any],
    *,
    provider_name: str,
) -> tuple[ProposalPrediction, ...]:
    candidate_ids = np.asarray(feature_arrays["candidate_id"]).reshape(-1)
    interval_scores = np.asarray(prediction_arrays["interval_scores"], dtype=np.float32)
    family_scores = np.asarray(prediction_arrays["family_scores"], dtype=np.float32)
    priority_scores = np.asarray(prediction_arrays["priority_score"], dtype=np.float32).reshape(-1)
    cost_scores = np.asarray(prediction_arrays["cost_score"], dtype=np.float32).reshape(-1)
    uncertainty_scores = np.asarray(prediction_arrays["uncertainty_score"], dtype=np.float32).reshape(-1)
    ood_mask = np.asarray(
        prediction_arrays.get("ood_mask", np.zeros(candidate_ids.shape[0], dtype=np.bool_)),
        dtype=np.bool_,
    ).reshape(-1)
    source_name = f"stpf_ort:{provider_name}"

    predictions: list[ProposalPrediction] = []
    for index in range(int(candidate_ids.shape[0])):
        predictions.append(
            validate_proposal_prediction(
                ProposalPrediction(
                    candidate_id=int(candidate_ids[index]),
                    interval_scores=[float(value) for value in interval_scores[index].tolist()],
                    family_scores=[float(value) for value in family_scores[index].tolist()],
                    priority_score=float(priority_scores[index]),
                    cost_score=float(cost_scores[index]),
                    uncertainty_score=float(uncertainty_scores[index]),
                    source=f"{source_name}:ood_fallback" if bool(ood_mask[index]) else source_name,
                )
            )
        )
    return tuple(predictions)


def _materialize_exact_work_items(work_items: Sequence[Any]) -> tuple[ExactWorkItem, ...]:
    converted: list[ExactWorkItem] = []
    for item in work_items:
        if isinstance(item, ExactWorkItem):
            converted.append(validate_exact_work_item(item))
        else:
            converted.append(_exact_work_item_from_cpp(item))
    return tuple(converted)


def _run_cpp_ort_runtime_schedule(
    candidates: Sequence[CandidateRecord],
    *,
    feature_arrays: dict[str, Any],
    family_by_runtime_query_id: dict[int, int],
    runtime: ORTInferenceSession,
    config: RTSTPFExactConfig,
) -> tuple[dict[str, Any], tuple[Any, ...], STPFScheduleStats]:
    cpp = _load_cpp_proposal_module()
    if not _cpp_array_fast_path_available(cpp):
        raise RuntimeError("compiled ORT proposal fast path is unavailable")
    prediction_arrays = batched_stpf_inference_ort_arrays(
        runtime,
        feature_arrays,
        batch_size=config.proposal_batch_size,
        ood_abs_feature_threshold=config.ood_abs_feature_threshold,
    )
    family_masks = {int(query_id): int(mask) for query_id, mask in family_by_runtime_query_id.items()}
    if _cpp_direct_array_schedule_available(cpp):
        cpp_work_items, cpp_stats = cpp.schedule_runtime_exact_work_items_from_proposal_arrays(
            feature_arrays,
            prediction_arrays,
            family_masks,
            _to_cpp_proposal_scheduling_config(cpp, config),
        )
    else:
        cpp_work_items, cpp_stats = cpp.schedule_runtime_exact_work_items_from_arrays(
            candidates,
            feature_arrays,
            prediction_arrays,
            family_masks,
            _to_cpp_proposal_scheduling_config(cpp, config),
        )
    row_count = int(np.asarray(feature_arrays["candidate_id"]).shape[0])
    return (
        prediction_arrays,
        tuple(cpp_work_items),
        _schedule_stats_from_cpp(cpp_stats, feature_row_count=row_count),
    )


def _normalized_dummy_family_scores(base_family_mask: int) -> list[float]:
    scores = [0.0] * PROPOSAL_FAMILY_COUNT
    active = []
    if base_family_mask & FEATURE_FAMILY_POINT_TRIANGLE:
        active.append(0)
    if base_family_mask & FEATURE_FAMILY_EDGE_EDGE:
        active.append(1)
    if not active:
        active = [0, 1]
    weight = 1.0 / float(len(active))
    for index in active:
        scores[index] = weight
    return scores


def _run_dummy_stpf_fast_path(
    candidates: Sequence[CandidateRecord],
    *,
    family_by_runtime_query_id: dict[int, int],
    candidate_stats: RtCandidateStats,
    config: RTSTPFExactConfig,
    materialize_artifacts: bool = True,
) -> tuple[tuple[ProposalFeatureRow, ...], tuple[ProposalPrediction, ...], tuple[ExactWorkItem, ...], STPFScheduleStats]:
    cpp = _load_cpp_proposal_module()
    if cpp is not None:
        cpp_result = cpp.run_dummy_runtime_proposal_schedule(
            list(candidates),
            int(candidate_stats.primitive_count),
            int(candidate_stats.raw_hit_count),
            int(candidate_stats.compact_candidate_count),
            {int(query_id): int(mask) for query_id, mask in family_by_runtime_query_id.items()},
            _to_cpp_proposal_scheduling_config(cpp, config),
            bool(materialize_artifacts),
        )
        feature_rows = tuple(_proposal_feature_row_from_cpp(row) for row in cpp_result.feature_rows)
        proposal_predictions = tuple(
            _proposal_prediction_from_cpp_output(output) for output in cpp_result.proposal_outputs
        )
        work_items = tuple(_exact_work_item_from_cpp(item) for item in cpp_result.work_queue)
        schedule_stats = _schedule_stats_from_cpp(
            cpp_result.stats,
            feature_row_count=len(candidates),
        )
        return feature_rows, proposal_predictions, work_items, schedule_stats

    candidates_per_proxy = candidate_stats.compact_candidate_count / max(1, candidate_stats.primitive_count)
    aabb_overlap_ratio = candidate_stats.raw_hit_count / max(1, candidate_stats.primitive_count // 2)
    avg_hits = candidate_stats.raw_hit_count / max(1, candidate_stats.compact_candidate_count)

    rows: list[ProposalFeatureRow] = []
    predictions: list[ProposalPrediction] = []
    scheduled: list[ExactWorkItem] = []
    fallback_count = 0
    ood_count = 0

    for candidate in candidates:
        base_family_mask = family_by_runtime_query_id.get(
            candidate.query_id,
            FEATURE_FAMILY_CONSERVATIVE,
        )
        interval_targets = [1.0] + [0.0] * (PROPOSAL_INTERVAL_BIN_COUNT - 1) if materialize_artifacts else []
        family_targets = _family_targets(base_family_mask) if materialize_artifacts else []
        priority_target = min(1.0, 0.5 * candidate.rt_hit_count + 0.5 * candidates_per_proxy)
        cost_target = max(1.0, 1.0 + float(candidate.rt_hit_count))
        uncertainty_target = 0.25 if base_family_mask == FEATURE_FAMILY_CONSERVATIVE else 0.0

        feature_candidates = [
            1.0,
            float(_clamp_feature(candidate.rt_hit_count)),
            float(_clamp_feature(int(candidate.proxy_type_a))),
            float(_clamp_feature(int(candidate.proxy_type_b))),
            float(_clamp_feature(min(1.0, aabb_overlap_ratio))),
            float(_clamp_feature(candidates_per_proxy)),
            float(_clamp_feature(aabb_overlap_ratio)),
            float(_clamp_feature(avg_hits)),
        ]
        feature_candidates.extend(float(_clamp_feature(value)) for value in candidate.motion_bound[:4])
        ood = any(
            (not math.isfinite(value)) or abs(value) > config.ood_abs_feature_threshold
            for value in feature_candidates
        )
        if ood:
            fallback_count += 1
            ood_count += 1

        family_scores = _normalized_dummy_family_scores(base_family_mask)
        if materialize_artifacts:
            features = [0.0] * PROPOSAL_FEATURE_DIM
            features[0] = 0.0
            features[1] = 1.0
            features[2] = 1.0
            features[3] = feature_candidates[1]
            features[4] = feature_candidates[2]
            features[5] = feature_candidates[3]
            for index in range(4):
                features[6 + index] = feature_candidates[8 + index]
            features[19] = feature_candidates[4]
            features[29] = feature_candidates[5]
            features[30] = feature_candidates[6]
            features[31] = feature_candidates[7]
            rows.append(
                ProposalFeatureRow(
                    query_id=candidate.query_id,
                    candidate_id=candidate.candidate_id,
                    slab_id=candidate.slab_id,
                    object_a_id=candidate.object_a_id,
                    patch_a_id=candidate.patch_a_id,
                    object_b_id=candidate.object_b_id,
                    patch_b_id=candidate.patch_b_id,
                    features=features,
                    interval_targets=interval_targets,
                    family_targets=family_targets,
                    priority_target=priority_target,
                    cost_target=cost_target,
                    uncertainty_target=uncertainty_target,
                    target_mask=TARGET_INTERVAL
                    | TARGET_FAMILY
                    | TARGET_PRIORITY
                    | TARGET_COST
                    | TARGET_UNCERTAINTY,
                )
            )
            predictions.append(
                ProposalPrediction(
                    candidate_id=candidate.candidate_id,
                    interval_scores=list(interval_targets),
                    family_scores=family_scores,
                    priority_score=max(0.0, float(priority_target)),
                    cost_score=max(0.0, float(cost_target)),
                    uncertainty_score=1.0 if ood else max(0.0, float(uncertainty_target)),
                    source="dummy_ood_fallback" if ood else "dummy",
                )
            )

        if ood:
            feature_mask = base_family_mask
            priority = float(candidate.rt_hit_count)
            source = ProposalSource.FALLBACK
        else:
            feature_mask = base_family_mask
            if family_scores[0] >= config.family_score_threshold:
                feature_mask |= FEATURE_FAMILY_POINT_TRIANGLE
            if family_scores[1] >= config.family_score_threshold:
                feature_mask |= FEATURE_FAMILY_EDGE_EDGE
            priority = max(0.0, float(priority_target))
            source = ProposalSource.REFINED
        scheduled.append(
            ExactWorkItem(
                work_item_id=0,
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
        item.work_item_id = config.first_work_item_id + offset

    original_parent_order = [candidate.candidate_id for candidate in candidates]
    scheduled_parent_order = [item.parent_candidate_id for item in scheduled]
    reordered_count = sum(
        1
        for lhs, rhs in zip(original_parent_order, scheduled_parent_order)
        if lhs != rhs
    )
    stats = STPFScheduleStats(
        raw_candidate_count=len(candidates),
        feature_row_count=len(candidates),
        proposal_output_count=len(candidates),
        work_item_count=len(scheduled),
        fallback_count=fallback_count,
        missing_proposal_fallback_count=0,
        invalid_proposal_fallback_count=0,
        ood_fallback_count=ood_count,
        high_uncertainty_fallback_count=0,
        reordered_count=reordered_count,
        monotonic_safe=True,
    )
    return tuple(rows), tuple(predictions), tuple(scheduled), stats


def _predicted_family_mask(prediction: ProposalPrediction, threshold: float) -> int:
    mask = 0
    if prediction.family_scores[0] >= threshold:
        mask |= FEATURE_FAMILY_POINT_TRIANGLE
    if prediction.family_scores[1] >= threshold:
        mask |= FEATURE_FAMILY_EDGE_EDGE
    return mask


def _valid_prediction(prediction: ProposalPrediction | None) -> bool:
    if prediction is None:
        return False
    try:
        validate_proposal_prediction(prediction)
    except ValueError:
        return False
    return True


def _select_runtime_model_device(
    config: RTSTPFExactConfig,
    *,
    row_count: int,
    requested_device: str | None = None,
) -> str:
    resolved_device = requested_device or config.model_device or "cpu"
    if (
        config.cpu_inference_row_threshold > 0
        and row_count <= config.cpu_inference_row_threshold
        and str(resolved_device).lower().startswith("cuda")
    ):
        return "cpu"
    return resolved_device


def _runtime_device_for_backend(
    config: RTSTPFExactConfig,
    *,
    row_count: int,
    requested_device: str | None = None,
) -> str:
    if config.inference_backend == "ort":
        resolved_device = requested_device or config.model_device or "cuda"
        if (
            config.cpu_inference_row_threshold > 0
            and row_count <= config.cpu_inference_row_threshold
            and str(resolved_device).lower().startswith("cuda")
        ):
            return "cpu"
        return resolved_device
    return _select_runtime_model_device(
        config,
        row_count=row_count,
        requested_device=requested_device,
    )


def _model_tag(config: RTSTPFExactConfig) -> str:
    preset = str(config.model_preset)
    checkpoint_stem = (
        Path(config.model_checkpoint_path).stem
        if config.model_checkpoint_path is not None
        else f"default_seed{config.model_seed}"
    )
    return f"{checkpoint_stem}_{preset}"


def _resolve_stpf_model(
    config: RTSTPFExactConfig,
    *,
    model=None,
    device: str | None = None,
):
    import torch

    resolved_device = device or config.model_device or "cpu"
    if model is not None:
        model.to(resolved_device)
        model.eval()
        return model

    if config.model_checkpoint_path is None and not config.allow_default_model:
        raise ValueError("RTSTPFExact requires a model when allow_default_model is False")

    torch.manual_seed(config.model_seed)
    resolved_model = build_stpf_model(config.model_preset)
    checkpoint_path = None if config.model_checkpoint_path is None else Path(config.model_checkpoint_path)
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"RTSTPFExact model checkpoint not found: {checkpoint_path}")
        checkpoint_payload = torch.load(checkpoint_path, map_location=resolved_device)
        resolved_model, state_dict = build_stpf_model_from_checkpoint_payload(
            checkpoint_payload,
            fallback_preset=config.model_preset,
        )
        resolved_model.load_state_dict(state_dict)
    resolved_model.to(resolved_device)
    resolved_model.eval()
    return resolved_model


def _resolve_ort_runtime(
    config: RTSTPFExactConfig,
    *,
    model=None,
    device: str | None = None,
) -> ORTInferenceSession:
    resolved_model = _resolve_stpf_model(
        config,
        model=model,
        device="cpu",
    )
    onnx_path = ensure_stpf_model_onnx(
        resolved_model,
        output_path=config.ort_model_path,
        checkpoint_path=config.model_checkpoint_path,
        model_tag=_model_tag(config),
        opset_version=config.ort_opset_version,
    )
    return create_ort_inference_session(
        onnx_path,
        requested_device=device,
        prefer_tensorrt=config.ort_prefer_tensorrt,
        allow_cuda_fallback=config.ort_allow_cuda_fallback,
        allow_cpu_fallback=config.ort_allow_cpu_fallback,
    )


def _resolve_stpf_runtime(
    config: RTSTPFExactConfig,
    *,
    model=None,
    device: str | None = None,
):
    if config.inference_backend == "ort":
        return _resolve_ort_runtime(
            config,
            model=model,
            device=device,
        )
    return _resolve_stpf_model(
        config,
        model=model,
        device=device,
    )


def _run_stpf_predictions(
    rows: Sequence[ProposalFeatureRow],
    config: RTSTPFExactConfig,
    *,
    runtime=None,
    device: str | None = None,
) -> tuple[ProposalPrediction, ...]:
    if not rows:
        return ()
    if config.inference_backend == "ort":
        if runtime is None:
            runtime = _resolve_ort_runtime(
                config,
                model=None,
                device=device,
            )
        return tuple(
            batched_stpf_inference_ort(
                runtime,
                rows,
                batch_size=config.proposal_batch_size,
                ood_abs_feature_threshold=config.ood_abs_feature_threshold,
            )
        )
    resolved_model = runtime if runtime is not None else _resolve_stpf_model(
        config,
        model=None,
        device=device,
    )
    return tuple(
        batched_stpf_inference(
            resolved_model,
            rows,
            batch_size=config.proposal_batch_size,
            device=device or config.model_device,
            ood_abs_feature_threshold=config.ood_abs_feature_threshold,
        )
    )


def _warmup_stpf_runtime(
    rows: Sequence[ProposalFeatureRow],
    config: RTSTPFExactConfig,
    *,
    runtime=None,
    device: str | None = None,
) -> None:
    if (
        not rows
        or config.inference_backend != "ort"
        or config.ort_warmup_passes <= 0
        or runtime is None
    ):
        return
    runtime_id = id(runtime)
    if runtime_id in _WARMED_ORT_RUNTIME_IDS:
        return
    for _ in range(config.ort_warmup_passes):
        _run_stpf_predictions(
            rows,
            config,
            runtime=runtime,
            device=device,
        )
    _WARMED_ORT_RUNTIME_IDS.add(runtime_id)


def _warmup_stpf_runtime_arrays(
    feature_arrays: dict[str, Any],
    config: RTSTPFExactConfig,
    *,
    runtime=None,
) -> None:
    if (
        not feature_arrays
        or config.inference_backend != "ort"
        or config.ort_warmup_passes <= 0
        or runtime is None
    ):
        return
    runtime_id = id(runtime)
    if runtime_id in _WARMED_ORT_RUNTIME_IDS:
        return
    for _ in range(config.ort_warmup_passes):
        batched_stpf_inference_ort_arrays(
            runtime,
            feature_arrays,
            batch_size=config.proposal_batch_size,
            ood_abs_feature_threshold=config.ood_abs_feature_threshold,
        )
    _WARMED_ORT_RUNTIME_IDS.add(runtime_id)


def _runtime_backend_name(config: RTSTPFExactConfig) -> str:
    return config.inference_backend


def _runtime_provider_name(runtime) -> str:
    if runtime is None:
        return "none"
    if isinstance(runtime, ORTInferenceSession):
        return runtime.provider_name
    return str(next(runtime.parameters()).device)


def schedule_exact_work_items_with_stpf(
    candidates: Sequence[CandidateRecord],
    rows: Sequence[ProposalFeatureRow],
    predictions: Sequence[ProposalPrediction],
    *,
    family_by_runtime_query_id: dict[int, int],
    config: RTSTPFExactConfig,
) -> tuple[tuple[ExactWorkItem, ...], STPFScheduleStats]:
    cpp = _load_cpp_proposal_module()
    if cpp is not None:
        cpp_work_items, cpp_stats = cpp.schedule_runtime_exact_work_items(
            candidates,
            rows,
            predictions,
            {int(query_id): int(mask) for query_id, mask in family_by_runtime_query_id.items()},
            _to_cpp_proposal_scheduling_config(cpp, config),
        )
        return (
            tuple(_exact_work_item_from_cpp(item) for item in cpp_work_items),
            _schedule_stats_from_cpp(cpp_stats, feature_row_count=len(rows)),
        )

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
        fallback = missing or invalid or ood or high_uncertainty
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
            feature_mask = base_mask | _predicted_family_mask(prediction, config.family_score_threshold)
            priority = float(prediction.priority_score)
            source = ProposalSource.REFINED

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


def _rt_config_from_stpf(config: RTSTPFExactConfig) -> RTExactConfig:
    return RTExactConfig(
        exact=config.exact,
        backend_name=config.rt_backend_name,
        enable_cuda_exact=config.enable_cuda_exact,
        same_query_only=config.same_query_only,
        first_work_item_id=config.first_work_item_id,
        first_event_id=config.first_event_id,
        first_timestamp_us=config.first_timestamp_us,
    )


def _make_benchmark_row(
    query_results: Sequence[PureExactQueryResult],
    *,
    candidate_stats: RtCandidateStats,
    schedule_stats: STPFScheduleStats,
    proposal_elapsed_ms: float,
    exact_elapsed_ms: float,
) -> BenchmarkRow:
    if not query_results:
        raise ValueError("RTSTPFExact requires at least one query")
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
    total_ms = candidate_stats.timing.total_ms + proposal_elapsed_ms + exact_elapsed_ms
    row = BenchmarkRow(
        query_count=len(query_results),
        fn_count=fn_count,
        fp_count=fp_count,
        candidate_recall=candidate_stats.candidate_recall,
        avg_candidates=candidate_stats.compact_candidate_count / len(query_results),
        avg_exact_evals=total_exact_evals / len(query_results),
        avg_subdivision_depth=total_depth / len(query_results),
        fallback_ratio=schedule_stats.fallback_count / max(1, schedule_stats.raw_candidate_count),
        rt_ms=candidate_stats.timing.total_ms,
        proposal_ms=proposal_elapsed_ms,
        exact_ms=exact_elapsed_ms,
        total_ms=total_ms,
        qps=0.0 if total_ms <= 0.0 else 1000.0 * len(query_results) / total_ms,
    )
    return validate_benchmark_row(row)


def run_rt_stpf_exact_on_external_batch(
    batch: DatasetQueryBatch,
    config: RTSTPFExactConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
    model=None,
    device: str | None = None,
) -> RTSTPFExactResult:
    if not batch.queries:
        raise ValueError("RTSTPFExact external batch requires at least one query")
    requested_cfg = _validate_config(config or RTSTPFExactConfig())
    rt_config = _rt_config_from_stpf(requested_cfg)
    candidates, candidate_stats, runtime_ids = _make_external_candidates(
        batch,
        rt_config,
        backend=backend,
    )
    cfg, resolved_execution_profile_name = _resolve_execution_profile(
        requested_cfg,
        query_count=len(batch.queries),
        candidate_stats=candidate_stats,
    )
    family_by_runtime_query_id = {
        runtime_ids[query.query_id]: _family_mask_for_external(query)
        for query in batch.queries
    }

    resolved_runtime = None
    inference_backend_name = _runtime_backend_name(cfg)
    inference_provider_name = "none"
    feature_rows: tuple[ProposalFeatureRow, ...] = ()
    proposal_predictions: tuple[ProposalPrediction, ...] = ()
    feature_arrays: dict[str, Any] | None = None
    prediction_arrays: dict[str, Any] | None = None
    compiled_ort_fast_path = cfg.inference_backend == "ort" and _cpp_array_fast_path_available(
        _load_cpp_proposal_module()
    )
    feature_rows_start = time.perf_counter()
    if compiled_ort_fast_path:
        feature_arrays = _build_cpp_proposal_feature_arrays(
            candidates,
            family_by_runtime_query_id=family_by_runtime_query_id,
            candidate_stats=candidate_stats,
        )
        if feature_arrays is None:
            compiled_ort_fast_path = False
    if compiled_ort_fast_path:
        row_count = int(np.asarray(feature_arrays["candidate_id"]).shape[0])
        feature_rows_elapsed_ms = (time.perf_counter() - feature_rows_start) * 1000.0
        inference_device = _runtime_device_for_backend(
            cfg,
            row_count=row_count,
            requested_device=device,
        )
        if row_count > 0:
            resolved_runtime = _resolve_stpf_runtime(
                cfg,
                model=model,
                device=inference_device,
            )
            inference_provider_name = _runtime_provider_name(resolved_runtime)
            _warmup_stpf_runtime_arrays(
                feature_arrays,
                cfg,
                runtime=resolved_runtime,
            )
        proposal_start = time.perf_counter()
        if row_count > 0:
            prediction_arrays, work_items, schedule_stats = _run_cpp_ort_runtime_schedule(
                candidates,
                feature_arrays=feature_arrays,
                family_by_runtime_query_id=family_by_runtime_query_id,
                runtime=resolved_runtime,
                config=cfg,
            )
        else:
            prediction_arrays = {
                "interval_scores": np.zeros((0, PROPOSAL_INTERVAL_BIN_COUNT), dtype=np.float32),
                "family_scores": np.zeros((0, PROPOSAL_FAMILY_COUNT), dtype=np.float32),
                "priority_score": np.zeros((0,), dtype=np.float32),
                "cost_score": np.zeros((0,), dtype=np.float32),
                "uncertainty_score": np.zeros((0,), dtype=np.float32),
                "ood_mask": np.zeros((0,), dtype=np.bool_),
            }
            work_items = ()
            schedule_stats = STPFScheduleStats(
                raw_candidate_count=0,
                feature_row_count=0,
                proposal_output_count=0,
                work_item_count=0,
                fallback_count=0,
                missing_proposal_fallback_count=0,
                invalid_proposal_fallback_count=0,
                ood_fallback_count=0,
                high_uncertainty_fallback_count=0,
                reordered_count=0,
                monotonic_safe=True,
            )
        proposal_elapsed_ms = feature_rows_elapsed_ms + (time.perf_counter() - proposal_start) * 1000.0
    else:
        feature_rows = proposal_feature_rows_from_rt_candidates(
            candidates,
            family_by_runtime_query_id=family_by_runtime_query_id,
            candidate_stats=candidate_stats,
        )
        feature_rows_elapsed_ms = (time.perf_counter() - feature_rows_start) * 1000.0
        inference_device = _runtime_device_for_backend(
            cfg,
            row_count=len(feature_rows),
            requested_device=device,
        )
        if feature_rows:
            resolved_runtime = _resolve_stpf_runtime(
                cfg,
                model=model,
                device=inference_device,
            )
            inference_provider_name = _runtime_provider_name(resolved_runtime)
            _warmup_stpf_runtime(
                feature_rows,
                cfg,
                runtime=resolved_runtime,
                device=inference_device,
            )
        proposal_start = time.perf_counter()
        proposal_predictions = _run_stpf_predictions(
            feature_rows,
            cfg,
            runtime=resolved_runtime,
            device=inference_device,
        )
        work_items, schedule_stats = schedule_exact_work_items_with_stpf(
            candidates,
            feature_rows,
            proposal_predictions,
            family_by_runtime_query_id=family_by_runtime_query_id,
            config=cfg,
        )
        proposal_elapsed_ms = feature_rows_elapsed_ms + (time.perf_counter() - proposal_start) * 1000.0

    query_results, certificates, audit_log, exact_elapsed_ms, exact_backend_name = _process_external_exact_work_queue(
        batch,
        candidates,
        work_items,
        runtime_ids,
        rt_config,
    )
    materialized_work_items = _materialize_exact_work_items(work_items)
    validate_rt_exact_coverage(candidates, materialized_work_items, certificates)
    if compiled_ort_fast_path and feature_arrays is not None and prediction_arrays is not None:
        feature_rows = _materialize_proposal_feature_rows_from_arrays(feature_arrays)
        proposal_predictions = _materialize_proposal_predictions_from_arrays(
            feature_arrays,
            prediction_arrays,
            provider_name=inference_provider_name,
        )
    return RTSTPFExactResult(
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
        work_items=materialized_work_items,
        certificates=certificates,
        audit_log=audit_log,
        candidate_stats=candidate_stats,
        schedule_stats=schedule_stats,
        resolved_execution_profile_name=resolved_execution_profile_name,
        inference_backend_name=inference_backend_name,
        inference_provider_name=inference_provider_name,
        exact_backend_name=exact_backend_name,
        source_name=batch.source_name,
        scene_name=batch.scene_name,
        batch_id=batch.batch_id,
    )


def run_rt_stpf_exact_on_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    config: RTSTPFExactConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
    model=None,
    device: str | None = None,
) -> RTSTPFExactResult:
    if not samples:
        raise ValueError("RTSTPFExact internal sample run requires at least one sample")
    requested_cfg = _validate_config(config or RTSTPFExactConfig())
    rt_config = _rt_config_from_stpf(requested_cfg)
    oracle_traces_by_query_id = {
        sample.query_id: evaluate_swept_sphere_oracle(sample) for sample in samples
    }
    candidates, candidate_stats, runtime_ids = _make_internal_candidates(
        samples,
        rt_config,
        backend=backend,
        oracle_traces_by_query_id=oracle_traces_by_query_id,
    )
    cfg, resolved_execution_profile_name = _resolve_execution_profile(
        requested_cfg,
        query_count=len(samples),
        candidate_stats=candidate_stats,
    )
    family_by_runtime_query_id = {
        runtime_ids[sample.query_id]: FEATURE_FAMILY_CONSERVATIVE
        for sample in samples
    }

    resolved_runtime = None
    inference_backend_name = _runtime_backend_name(cfg)
    inference_provider_name = "none"
    inference_device = device
    feature_rows: tuple[ProposalFeatureRow, ...] = ()
    proposal_predictions: tuple[ProposalPrediction, ...] = ()
    feature_arrays: dict[str, Any] | None = None
    prediction_arrays: dict[str, Any] | None = None
    compiled_ort_fast_path = cfg.inference_backend == "ort" and _cpp_array_fast_path_available(
        _load_cpp_proposal_module()
    )
    feature_rows_start = time.perf_counter()
    if compiled_ort_fast_path:
        feature_arrays = _build_cpp_proposal_feature_arrays(
            candidates,
            family_by_runtime_query_id=family_by_runtime_query_id,
            candidate_stats=candidate_stats,
        )
        if feature_arrays is None:
            compiled_ort_fast_path = False
    if compiled_ort_fast_path:
        row_count = int(np.asarray(feature_arrays["candidate_id"]).shape[0])
        feature_rows_elapsed_ms = (time.perf_counter() - feature_rows_start) * 1000.0
        inference_device = _runtime_device_for_backend(
            cfg,
            row_count=row_count,
            requested_device=device,
        )
        if row_count > 0:
            resolved_runtime = _resolve_stpf_runtime(
                cfg,
                model=model,
                device=inference_device,
            )
            inference_provider_name = _runtime_provider_name(resolved_runtime)
            _warmup_stpf_runtime_arrays(
                feature_arrays,
                cfg,
                runtime=resolved_runtime,
            )
        proposal_start = time.perf_counter()
        if row_count > 0:
            prediction_arrays, work_items, schedule_stats = _run_cpp_ort_runtime_schedule(
                candidates,
                feature_arrays=feature_arrays,
                family_by_runtime_query_id=family_by_runtime_query_id,
                runtime=resolved_runtime,
                config=cfg,
            )
        else:
            prediction_arrays = {
                "interval_scores": np.zeros((0, PROPOSAL_INTERVAL_BIN_COUNT), dtype=np.float32),
                "family_scores": np.zeros((0, PROPOSAL_FAMILY_COUNT), dtype=np.float32),
                "priority_score": np.zeros((0,), dtype=np.float32),
                "cost_score": np.zeros((0,), dtype=np.float32),
                "uncertainty_score": np.zeros((0,), dtype=np.float32),
                "ood_mask": np.zeros((0,), dtype=np.bool_),
            }
            work_items = ()
            schedule_stats = STPFScheduleStats(
                raw_candidate_count=0,
                feature_row_count=0,
                proposal_output_count=0,
                work_item_count=0,
                fallback_count=0,
                missing_proposal_fallback_count=0,
                invalid_proposal_fallback_count=0,
                ood_fallback_count=0,
                high_uncertainty_fallback_count=0,
                reordered_count=0,
                monotonic_safe=True,
            )
        proposal_elapsed_ms = feature_rows_elapsed_ms + (time.perf_counter() - proposal_start) * 1000.0
    else:
        feature_rows = proposal_feature_rows_from_rt_candidates(
            candidates,
            family_by_runtime_query_id=family_by_runtime_query_id,
            candidate_stats=candidate_stats,
        )
        feature_rows_elapsed_ms = (time.perf_counter() - feature_rows_start) * 1000.0
        inference_device = _runtime_device_for_backend(
            cfg,
            row_count=len(feature_rows),
            requested_device=device,
        )
        if feature_rows:
            resolved_runtime = _resolve_stpf_runtime(
                cfg,
                model=model,
                device=inference_device,
            )
            inference_provider_name = _runtime_provider_name(resolved_runtime)
            _warmup_stpf_runtime(
                feature_rows,
                cfg,
                runtime=resolved_runtime,
                device=inference_device,
            )
        proposal_start = time.perf_counter()
        proposal_predictions = _run_stpf_predictions(
            feature_rows,
            cfg,
            runtime=resolved_runtime,
            device=inference_device,
        )
        work_items, schedule_stats = schedule_exact_work_items_with_stpf(
            candidates,
            feature_rows,
            proposal_predictions,
            family_by_runtime_query_id=family_by_runtime_query_id,
            config=cfg,
        )
        proposal_elapsed_ms = feature_rows_elapsed_ms + (time.perf_counter() - proposal_start) * 1000.0

    query_results, certificates, audit_log, exact_elapsed_ms, exact_backend_name = _process_internal_exact_work_queue(
        samples,
        candidates,
        work_items,
        runtime_ids,
        rt_config,
        oracle_traces_by_query_id=oracle_traces_by_query_id,
    )
    materialized_work_items = _materialize_exact_work_items(work_items)
    validate_rt_exact_coverage(candidates, materialized_work_items, certificates)
    if compiled_ort_fast_path and feature_arrays is not None and prediction_arrays is not None:
        feature_rows = _materialize_proposal_feature_rows_from_arrays(feature_arrays)
        proposal_predictions = _materialize_proposal_predictions_from_arrays(
            feature_arrays,
            prediction_arrays,
            provider_name=inference_provider_name,
        )
    return RTSTPFExactResult(
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
        work_items=materialized_work_items,
        certificates=certificates,
        audit_log=audit_log,
        candidate_stats=candidate_stats,
        schedule_stats=schedule_stats,
        resolved_execution_profile_name=resolved_execution_profile_name,
        inference_backend_name=inference_backend_name,
        inference_provider_name=inference_provider_name,
        exact_backend_name=exact_backend_name,
        source_name="internal_analytic_oracle",
        scene_name="programmatic_motion_disc_pairs",
        batch_id="internal_samples",
    )


def run_rt_stpf_exact_on_generated_dataset(
    dataset: GeneratedDataset,
    config: RTSTPFExactConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
    model=None,
    device: str | None = None,
) -> RTSTPFExactResult:
    return run_rt_stpf_exact_on_internal_samples(
        dataset.samples,
        config,
        backend=backend,
        model=model,
        device=device,
    )
