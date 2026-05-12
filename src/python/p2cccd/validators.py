from __future__ import annotations

import math
from dataclasses import is_dataclass
from enum import IntEnum
from typing import Any

from .contracts import (
    CONTRACT_SCHEMA_VERSION,
    AuditLogRow,
    AuditStage,
    BenchmarkRow,
    BenchmarkRowV2,
    BenchmarkRunMeta,
    BENCHMARK_EXPORT_SCHEMA_VERSION,
    CandidateRecord,
    CertificateRefinementMode,
    CertificateResult,
    CertificateStatus,
    CONTRACT_TYPES,
    ExactWorkItem,
    INT64_MAX,
    INT64_MIN,
    MAX_FAMILY_SCORES,
    MAX_INTERVAL_SCORES,
    MOTION_BOUND_SIZE,
    ProposalOutput,
    ProposalSource,
    ProxyType,
    UINT16_MAX,
    UINT32_MAX,
    UINT64_MAX,
    UINT8_MAX,
    schema_field_names,
)


class ValidationError(ValueError):
    """Raised when a runtime contract violates the stable schema."""


def _raise(message: str) -> None:
    raise ValidationError(message)


def _require_int(
    name: str,
    value: Any,
    *,
    minimum: int = 0,
    maximum: int | None = None,
    nonzero: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _raise(f"{name} must be an integer")
    if value < minimum:
        _raise(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        _raise(f"{name} must be <= {maximum}")
    if nonzero and value == 0:
        _raise(f"{name} is required")
    return value


def _require_signed_int(name: str, value: Any, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _raise(f"{name} must be an integer")
    if value < minimum:
        _raise(f"{name} must be >= {minimum}")
    if value > maximum:
        _raise(f"{name} must be <= {maximum}")
    return value


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _raise(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        _raise(f"{name} must be finite")
    return number


def _non_negative(name: str, value: Any) -> float:
    number = _finite(name, value)
    if number < 0.0:
        _raise(f"{name} must be non-negative")
    return number


def _positive(name: str, value: Any) -> float:
    number = _finite(name, value)
    if number <= 0.0:
        _raise(f"{name} must be positive")
    return number


def _string(name: str, value: Any, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        _raise(f"{name} must be a string")
    if nonempty and not value:
        _raise(f"{name} is required")
    return value


def _ratio(name: str, value: Any) -> float:
    number = _finite(name, value)
    if number < 0.0 or number > 1.0:
        _raise(f"{name} must be in [0, 1]")
    return number


def _enum(name: str, value: Any, enum_type: type[IntEnum], *, allow_unknown: bool = True) -> IntEnum:
    if isinstance(value, bool):
        _raise(f"{name} must be an integer enum value, not bool")
    try:
        enum_value = enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} has invalid enum value {value!r}") from exc
    if not allow_unknown and enum_value.value == 0:
        _raise(f"{name} must be a concrete enum value")
    return enum_value


def _fixed_numeric_list(
    name: str,
    values: Any,
    length: int,
    *,
    non_negative: bool = False,
) -> None:
    if not isinstance(values, list):
        _raise(f"{name} must be a list")
    if len(values) != length:
        _raise(f"{name} must have length {length}")
    checker = _non_negative if non_negative else _finite
    for index, value in enumerate(values):
        checker(f"{name}[{index}]", value)


def _unit_interval(t0: Any, t1: Any) -> None:
    start = _finite("interval_t0", t0)
    end = _finite("interval_t1", t1)
    if start < 0.0 or end > 1.0 or start > end:
        _raise("interval must satisfy 0 <= interval_t0 <= interval_t1 <= 1")


def validate_dict_schema(data: dict[str, Any], contract_type: str | type | object) -> None:
    if not isinstance(data, dict):
        _raise("contract payload must be a dict")
    try:
        required = set(schema_field_names(contract_type))
    except KeyError as exc:
        raise ValidationError(f"unknown contract type {contract_type!r}") from exc
    keys = set(data.keys())
    missing = sorted(required - keys)
    extra = sorted(keys - required)
    if missing:
        _raise(f"missing required field(s): {', '.join(missing)}")
    if extra:
        _raise(f"unknown field(s): {', '.join(extra)}")


def validate_candidate_record(record: CandidateRecord) -> CandidateRecord:
    _require_int("CandidateRecord.schema_version", record.schema_version, minimum=1, maximum=UINT32_MAX)
    if record.schema_version != CONTRACT_SCHEMA_VERSION:
        _raise("CandidateRecord.schema_version is unsupported")
    _require_int("CandidateRecord.candidate_id", record.candidate_id, maximum=UINT64_MAX, nonzero=True)
    _require_int("CandidateRecord.query_id", record.query_id, maximum=UINT64_MAX, nonzero=True)
    _require_int("CandidateRecord.slab_id", record.slab_id, maximum=UINT32_MAX)
    _require_int("CandidateRecord.object_a_id", record.object_a_id, maximum=UINT32_MAX)
    _require_int("CandidateRecord.patch_a_id", record.patch_a_id, maximum=UINT32_MAX)
    _require_int("CandidateRecord.object_b_id", record.object_b_id, maximum=UINT32_MAX)
    _require_int("CandidateRecord.patch_b_id", record.patch_b_id, maximum=UINT32_MAX)
    _enum("CandidateRecord.proxy_type_a", record.proxy_type_a, ProxyType, allow_unknown=False)
    _enum("CandidateRecord.proxy_type_b", record.proxy_type_b, ProxyType, allow_unknown=False)
    _require_int("CandidateRecord.rt_hit_count", record.rt_hit_count, maximum=UINT32_MAX, nonzero=True)
    _fixed_numeric_list(
        "CandidateRecord.motion_bound",
        record.motion_bound,
        MOTION_BOUND_SIZE,
        non_negative=True,
    )
    _require_int("CandidateRecord.proxy_features_offset", record.proxy_features_offset, maximum=UINT32_MAX)
    _require_int("CandidateRecord.flags", record.flags, maximum=UINT32_MAX)
    return record


def validate_proposal_output(output: ProposalOutput) -> ProposalOutput:
    _require_int("ProposalOutput.candidate_id", output.candidate_id, maximum=UINT64_MAX, nonzero=True)
    _fixed_numeric_list("ProposalOutput.interval_scores", output.interval_scores, MAX_INTERVAL_SCORES)
    _fixed_numeric_list("ProposalOutput.family_scores", output.family_scores, MAX_FAMILY_SCORES)
    _finite("ProposalOutput.priority_score", output.priority_score)
    _non_negative("ProposalOutput.cost_score", output.cost_score)
    _non_negative("ProposalOutput.uncertainty_score", output.uncertainty_score)
    return output


def validate_exact_work_item(item: ExactWorkItem) -> ExactWorkItem:
    _require_int("ExactWorkItem.work_item_id", item.work_item_id, maximum=UINT64_MAX, nonzero=True)
    _require_int(
        "ExactWorkItem.parent_candidate_id",
        item.parent_candidate_id,
        maximum=UINT64_MAX,
        nonzero=True,
    )
    _require_int("ExactWorkItem.query_id", item.query_id, maximum=UINT64_MAX, nonzero=True)
    _require_int("ExactWorkItem.slab_id", item.slab_id, maximum=UINT32_MAX)
    _require_int("ExactWorkItem.patch_a_id", item.patch_a_id, maximum=UINT32_MAX)
    _require_int("ExactWorkItem.patch_b_id", item.patch_b_id, maximum=UINT32_MAX)
    _unit_interval(item.interval_t0, item.interval_t1)
    _require_int(
        "ExactWorkItem.feature_family_mask",
        item.feature_family_mask,
        maximum=UINT32_MAX,
        nonzero=True,
    )
    _require_int("ExactWorkItem.topk_feature_ids_offset", item.topk_feature_ids_offset, maximum=UINT32_MAX)
    _require_int("ExactWorkItem.depth", item.depth, maximum=UINT16_MAX)
    _finite("ExactWorkItem.priority_score", item.priority_score)
    _enum("ExactWorkItem.source", item.source, ProposalSource)
    return item


def validate_certificate_result(result: CertificateResult) -> CertificateResult:
    _require_int("CertificateResult.work_item_id", result.work_item_id, maximum=UINT64_MAX, nonzero=True)
    _require_int("CertificateResult.query_id", result.query_id, maximum=UINT64_MAX, nonzero=True)
    status = _enum("CertificateResult.status", result.status, CertificateStatus)
    _unit_interval(result.interval_t0, result.interval_t1)
    toi_upper = _finite("CertificateResult.toi_upper", result.toi_upper)
    safe_margin_lb = _finite("CertificateResult.safe_margin_lb", result.safe_margin_lb)
    _require_int("CertificateResult.witness_family", result.witness_family, maximum=UINT8_MAX)
    _require_signed_int("CertificateResult.witness_id_a", result.witness_id_a, minimum=INT64_MIN, maximum=INT64_MAX)
    _require_signed_int("CertificateResult.witness_id_b", result.witness_id_b, minimum=INT64_MIN, maximum=INT64_MAX)
    _require_int("CertificateResult.covered_feature_mask", result.covered_feature_mask, maximum=UINT32_MAX)
    _positive("CertificateResult.eps_time", result.eps_time)
    _positive("CertificateResult.eps_space", result.eps_space)
    _require_int("CertificateResult.reason_code", result.reason_code, maximum=UINT16_MAX)
    next_refinement_mode = _enum(
        "CertificateResult.next_refinement_mode",
        result.next_refinement_mode,
        CertificateRefinementMode,
    )

    if status == CertificateStatus.COLLISION:
        if toi_upper < result.interval_t0 or toi_upper > result.interval_t1:
            _raise("CertificateResult.toi_upper must lie inside the certified interval")
        if result.witness_id_a < 0 or result.witness_id_b < 0:
            _raise("CertificateResult collision witnesses are required")
        if next_refinement_mode != CertificateRefinementMode.NONE:
            _raise("CertificateResult collision cannot request refinement")
    if status == CertificateStatus.SEPARATION:
        if safe_margin_lb < 0.0:
            _raise("CertificateResult.safe_margin_lb must be non-negative")
        if result.covered_feature_mask == 0:
            _raise("CertificateResult.covered_feature_mask is required for separation")
        if next_refinement_mode != CertificateRefinementMode.NONE:
            _raise("CertificateResult separation cannot request refinement")
    if status == CertificateStatus.UNDECIDED and result.reason_code == 0:
        _raise("CertificateResult.reason_code is required for undecided results")
    if status == CertificateStatus.UNDECIDED and next_refinement_mode == CertificateRefinementMode.NONE:
        _raise("CertificateResult.next_refinement_mode is required for undecided results")
    return result


def validate_audit_log_row(row: AuditLogRow) -> AuditLogRow:
    _require_int("AuditLogRow.event_id", row.event_id, maximum=UINT64_MAX, nonzero=True)
    _require_int("AuditLogRow.query_id", row.query_id, maximum=UINT64_MAX, nonzero=True)
    _require_int("AuditLogRow.candidate_id", row.candidate_id, maximum=UINT64_MAX)
    _require_int("AuditLogRow.work_item_id", row.work_item_id, maximum=UINT64_MAX)
    _enum("AuditLogRow.stage", row.stage, AuditStage)
    _require_int("AuditLogRow.action", row.action, maximum=UINT16_MAX)
    _require_int("AuditLogRow.depth", row.depth, maximum=UINT16_MAX)
    _unit_interval(row.interval_t0, row.interval_t1)
    _require_int("AuditLogRow.timestamp_us", row.timestamp_us, maximum=UINT64_MAX, nonzero=True)
    _finite("AuditLogRow.aux_value0", row.aux_value0)
    _finite("AuditLogRow.aux_value1", row.aux_value1)
    return row


def validate_benchmark_row(row: BenchmarkRow) -> BenchmarkRow:
    _require_int("BenchmarkRow.query_count", row.query_count, maximum=UINT64_MAX, nonzero=True)
    _require_int("BenchmarkRow.fn_count", row.fn_count, maximum=UINT64_MAX)
    _require_int("BenchmarkRow.fp_count", row.fp_count, maximum=UINT64_MAX)
    if row.fn_count > row.query_count:
        _raise("BenchmarkRow.fn_count cannot exceed query_count")
    _ratio("BenchmarkRow.candidate_recall", row.candidate_recall)
    _non_negative("BenchmarkRow.avg_candidates", row.avg_candidates)
    _non_negative("BenchmarkRow.avg_exact_evals", row.avg_exact_evals)
    _non_negative("BenchmarkRow.avg_subdivision_depth", row.avg_subdivision_depth)
    _ratio("BenchmarkRow.fallback_ratio", row.fallback_ratio)
    _non_negative("BenchmarkRow.rt_ms", row.rt_ms)
    _non_negative("BenchmarkRow.proposal_ms", row.proposal_ms)
    _non_negative("BenchmarkRow.exact_ms", row.exact_ms)
    _non_negative("BenchmarkRow.total_ms", row.total_ms)
    _non_negative("BenchmarkRow.qps", row.qps)
    return row


def validate_benchmark_run_meta(meta: BenchmarkRunMeta) -> BenchmarkRunMeta:
    _require_int("BenchmarkRunMeta.schema_version", meta.schema_version, minimum=1, maximum=UINT32_MAX)
    if meta.schema_version != BENCHMARK_EXPORT_SCHEMA_VERSION:
        _raise("BenchmarkRunMeta.schema_version is unsupported")
    _string("BenchmarkRunMeta.run_id", meta.run_id, nonempty=True)
    _string("BenchmarkRunMeta.created_utc", meta.created_utc, nonempty=True)
    _string("BenchmarkRunMeta.dataset_name", meta.dataset_name, nonempty=True)
    _string("BenchmarkRunMeta.scene_name", meta.scene_name, nonempty=True)
    _string("BenchmarkRunMeta.method_name", meta.method_name, nonempty=True)
    _string("BenchmarkRunMeta.config_hash", meta.config_hash, nonempty=True)
    _string("BenchmarkRunMeta.config_json", meta.config_json, nonempty=True)
    _require_signed_int("BenchmarkRunMeta.seed", meta.seed, minimum=INT64_MIN, maximum=INT64_MAX)
    _require_int("BenchmarkRunMeta.row_count", meta.row_count, maximum=UINT64_MAX)
    _string("BenchmarkRunMeta.git_commit", meta.git_commit)
    _string("BenchmarkRunMeta.host_name", meta.host_name)
    _string("BenchmarkRunMeta.platform", meta.platform)
    _string("BenchmarkRunMeta.python_version", meta.python_version)
    _string("BenchmarkRunMeta.gpu_name", meta.gpu_name, nonempty=True)
    _string("BenchmarkRunMeta.driver_version", meta.driver_version, nonempty=True)
    _string("BenchmarkRunMeta.cuda_version", meta.cuda_version, nonempty=True)
    _string("BenchmarkRunMeta.optix_version", meta.optix_version, nonempty=True)
    _require_int("BenchmarkRunMeta.vram_total_mb", meta.vram_total_mb, maximum=UINT64_MAX)
    _require_int("BenchmarkRunMeta.vram_free_mb", meta.vram_free_mb, maximum=UINT64_MAX)
    _string("BenchmarkRunMeta.output_csv", meta.output_csv, nonempty=True)
    _string("BenchmarkRunMeta.output_jsonl", meta.output_jsonl, nonempty=True)
    _string("BenchmarkRunMeta.output_run_meta_json", meta.output_run_meta_json, nonempty=True)
    _string("BenchmarkRunMeta.notes", meta.notes)
    return meta


def validate_benchmark_row_v2(row: BenchmarkRowV2) -> BenchmarkRowV2:
    _require_int("BenchmarkRowV2.schema_version", row.schema_version, minimum=1, maximum=UINT32_MAX)
    if row.schema_version != BENCHMARK_EXPORT_SCHEMA_VERSION:
        _raise("BenchmarkRowV2.schema_version is unsupported")
    _string("BenchmarkRowV2.run_id", row.run_id, nonempty=True)
    _string("BenchmarkRowV2.dataset_name", row.dataset_name, nonempty=True)
    _string("BenchmarkRowV2.scene_name", row.scene_name, nonempty=True)
    _string("BenchmarkRowV2.method_name", row.method_name, nonempty=True)
    _string("BenchmarkRowV2.config_hash", row.config_hash, nonempty=True)
    _require_signed_int("BenchmarkRowV2.seed", row.seed, minimum=INT64_MIN, maximum=INT64_MAX)
    _require_int("BenchmarkRowV2.query_count", row.query_count, maximum=UINT64_MAX, nonzero=True)
    _require_int("BenchmarkRowV2.fn_count", row.fn_count, maximum=UINT64_MAX)
    _require_int("BenchmarkRowV2.fp_count", row.fp_count, maximum=UINT64_MAX)
    if row.fn_count > row.query_count:
        _raise("BenchmarkRowV2.fn_count cannot exceed query_count")
    _ratio("BenchmarkRowV2.candidate_recall", row.candidate_recall)
    _non_negative("BenchmarkRowV2.avg_candidates", row.avg_candidates)
    _non_negative("BenchmarkRowV2.avg_exact_evals", row.avg_exact_evals)
    _non_negative("BenchmarkRowV2.avg_subdivision_depth", row.avg_subdivision_depth)
    _ratio("BenchmarkRowV2.fallback_ratio", row.fallback_ratio)
    _non_negative("BenchmarkRowV2.candidate_inflation_ratio", row.candidate_inflation_ratio)
    _non_negative("BenchmarkRowV2.undecided_to_resolved_ratio", row.undecided_to_resolved_ratio)
    _non_negative("BenchmarkRowV2.exact_queue_occupancy", row.exact_queue_occupancy)
    _non_negative("BenchmarkRowV2.rt_build_ms", row.rt_build_ms)
    _non_negative("BenchmarkRowV2.rt_update_ms", row.rt_update_ms)
    _non_negative("BenchmarkRowV2.rt_trace_ms", row.rt_trace_ms)
    _non_negative("BenchmarkRowV2.rt_ms", row.rt_ms)
    _non_negative("BenchmarkRowV2.proposal_ms", row.proposal_ms)
    _non_negative("BenchmarkRowV2.exact_ms", row.exact_ms)
    _non_negative("BenchmarkRowV2.total_ms", row.total_ms)
    latency_values = (
        _non_negative("BenchmarkRowV2.latency_min_ms", row.latency_min_ms),
        _non_negative("BenchmarkRowV2.latency_p50_ms", row.latency_p50_ms),
        _non_negative("BenchmarkRowV2.latency_p90_ms", row.latency_p90_ms),
        _non_negative("BenchmarkRowV2.latency_p95_ms", row.latency_p95_ms),
        _non_negative("BenchmarkRowV2.latency_p99_ms", row.latency_p99_ms),
        _non_negative("BenchmarkRowV2.latency_max_ms", row.latency_max_ms),
    )
    if tuple(sorted(latency_values)) != latency_values:
        _raise("BenchmarkRowV2 latency percentiles must be monotonic")
    _non_negative("BenchmarkRowV2.qps", row.qps)
    point_triangle = _require_int(
        "BenchmarkRowV2.family_point_triangle_exact_calls",
        row.family_point_triangle_exact_calls,
        maximum=UINT64_MAX,
    )
    edge_edge = _require_int(
        "BenchmarkRowV2.family_edge_edge_exact_calls",
        row.family_edge_edge_exact_calls,
        maximum=UINT64_MAX,
    )
    conservative = _require_int(
        "BenchmarkRowV2.family_conservative_exact_calls",
        row.family_conservative_exact_calls,
        maximum=UINT64_MAX,
    )
    unknown = _require_int(
        "BenchmarkRowV2.family_unknown_exact_calls",
        row.family_unknown_exact_calls,
        maximum=UINT64_MAX,
    )
    exact_total = _require_int("BenchmarkRowV2.exact_calls_total", row.exact_calls_total, maximum=UINT64_MAX)
    if point_triangle + edge_edge + conservative + unknown != exact_total:
        _raise("BenchmarkRowV2 exact_calls_total must equal family-wise exact calls")
    _non_negative("BenchmarkRowV2.candidate_buffer_bandwidth_mb_s", row.candidate_buffer_bandwidth_mb_s)
    _non_negative("BenchmarkRowV2.proposal_enqueue_dequeue_ms", row.proposal_enqueue_dequeue_ms)
    _non_negative("BenchmarkRowV2.total_tail_latency_ms", row.total_tail_latency_ms)
    _require_int("BenchmarkRowV2.vram_peak_mb", row.vram_peak_mb, maximum=UINT64_MAX)
    return row


def validate_contract(contract: Any) -> Any:
    if not is_dataclass(contract):
        _raise("contract must be a dataclass instance")
    validators = {
        CandidateRecord: validate_candidate_record,
        ProposalOutput: validate_proposal_output,
        ExactWorkItem: validate_exact_work_item,
        CertificateResult: validate_certificate_result,
        AuditLogRow: validate_audit_log_row,
        BenchmarkRow: validate_benchmark_row,
        BenchmarkRunMeta: validate_benchmark_run_meta,
        BenchmarkRowV2: validate_benchmark_row_v2,
    }
    validator = validators.get(type(contract))
    if validator is None:
        _raise(f"unsupported contract type {type(contract).__name__}")
    return validator(contract)


def contract_type_from_name(name: str) -> type:
    try:
        return CONTRACT_TYPES[name]
    except KeyError as exc:
        raise ValidationError(f"unknown contract type {name!r}") from exc
