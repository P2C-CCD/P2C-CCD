from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List


UINT8_MAX = 2**8 - 1
UINT16_MAX = 2**16 - 1
UINT32_MAX = 2**32 - 1
UINT64_MAX = 2**64 - 1
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1

CONTRACT_SCHEMA_VERSION = 1
BENCHMARK_EXPORT_SCHEMA_VERSION = 3
MAX_INTERVAL_SCORES = 8
MAX_FAMILY_SCORES = 8
MOTION_BOUND_SIZE = 4


class ProxyType(IntEnum):
    UNKNOWN = 0
    SWEPT_AABB = 1
    CAPSULE = 2


class ProposalSource(IntEnum):
    RAW = 0
    REFINED = 1
    FALLBACK = 2


class CertificateStatus(IntEnum):
    COLLISION = 0
    SEPARATION = 1
    UNDECIDED = 2


class CertificateRefinementMode(IntEnum):
    NONE = 0
    BISECT_INTERVAL = 1
    REQUEST_GEOMETRY = 2
    ESCALATE_PRECISION = 3


class AuditStage(IntEnum):
    RT = 0
    PROPOSAL = 1
    EXACT = 2
    REFINE = 3
    CERTIFY = 4


@dataclass(slots=True)
class CandidateRecord:
    schema_version: int = CONTRACT_SCHEMA_VERSION
    candidate_id: int = 0
    query_id: int = 0
    slab_id: int = 0
    object_a_id: int = 0
    patch_a_id: int = 0
    object_b_id: int = 0
    patch_b_id: int = 0
    proxy_type_a: ProxyType = ProxyType.UNKNOWN
    proxy_type_b: ProxyType = ProxyType.UNKNOWN
    rt_hit_count: int = 0
    motion_bound: List[float] = field(default_factory=lambda: [0.0] * MOTION_BOUND_SIZE)
    proxy_features_offset: int = 0
    flags: int = 0


@dataclass(slots=True)
class ProposalOutput:
    candidate_id: int = 0
    interval_scores: List[float] = field(default_factory=lambda: [0.0] * MAX_INTERVAL_SCORES)
    family_scores: List[float] = field(default_factory=lambda: [0.0] * MAX_FAMILY_SCORES)
    priority_score: float = 0.0
    cost_score: float = 0.0
    uncertainty_score: float = 0.0


@dataclass(slots=True)
class ExactWorkItem:
    work_item_id: int = 0
    parent_candidate_id: int = 0
    query_id: int = 0
    slab_id: int = 0
    patch_a_id: int = 0
    patch_b_id: int = 0
    interval_t0: float = 0.0
    interval_t1: float = 1.0
    feature_family_mask: int = 0
    topk_feature_ids_offset: int = 0
    depth: int = 0
    priority_score: float = 0.0
    source: ProposalSource = ProposalSource.RAW


@dataclass(slots=True)
class CertificateResult:
    work_item_id: int = 0
    query_id: int = 0
    status: CertificateStatus = CertificateStatus.UNDECIDED
    interval_t0: float = 0.0
    interval_t1: float = 1.0
    toi_upper: float = 1.0
    safe_margin_lb: float = 0.0
    witness_family: int = 0
    witness_id_a: int = -1
    witness_id_b: int = -1
    covered_feature_mask: int = 0
    eps_time: float = 1.0e-4
    eps_space: float = 1.0e-6
    reason_code: int = 0
    next_refinement_mode: CertificateRefinementMode = CertificateRefinementMode.NONE


@dataclass(slots=True)
class AuditLogRow:
    event_id: int = 0
    query_id: int = 0
    candidate_id: int = 0
    work_item_id: int = 0
    stage: AuditStage = AuditStage.RT
    action: int = 0
    depth: int = 0
    interval_t0: float = 0.0
    interval_t1: float = 1.0
    timestamp_us: int = 0
    aux_value0: float = 0.0
    aux_value1: float = 0.0


@dataclass(slots=True)
class BenchmarkRow:
    query_count: int = 0
    fn_count: int = 0
    fp_count: int = 0
    candidate_recall: float = 0.0
    avg_candidates: float = 0.0
    avg_exact_evals: float = 0.0
    avg_subdivision_depth: float = 0.0
    fallback_ratio: float = 0.0
    rt_ms: float = 0.0
    proposal_ms: float = 0.0
    exact_ms: float = 0.0
    total_ms: float = 0.0
    qps: float = 0.0


@dataclass(slots=True)
class BenchmarkRunMeta:
    schema_version: int = BENCHMARK_EXPORT_SCHEMA_VERSION
    run_id: str = ""
    created_utc: str = ""
    dataset_name: str = ""
    scene_name: str = ""
    method_name: str = ""
    config_hash: str = ""
    config_json: str = ""
    seed: int = 0
    row_count: int = 0
    git_commit: str = ""
    host_name: str = ""
    platform: str = ""
    python_version: str = ""
    gpu_name: str = "unknown"
    driver_version: str = "unknown"
    cuda_version: str = "unknown"
    optix_version: str = "unknown"
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    output_csv: str = "benchmark.csv"
    output_jsonl: str = "benchmark.jsonl"
    output_run_meta_json: str = "run_meta.json"
    notes: str = ""


@dataclass(slots=True)
class BenchmarkRowV2:
    schema_version: int = BENCHMARK_EXPORT_SCHEMA_VERSION
    run_id: str = ""
    dataset_name: str = ""
    scene_name: str = ""
    method_name: str = ""
    config_hash: str = ""
    seed: int = 0
    query_count: int = 0
    fn_count: int = 0
    fp_count: int = 0
    candidate_recall: float = 0.0
    avg_candidates: float = 0.0
    avg_exact_evals: float = 0.0
    avg_subdivision_depth: float = 0.0
    fallback_ratio: float = 0.0
    candidate_inflation_ratio: float = 0.0
    undecided_to_resolved_ratio: float = 0.0
    exact_queue_occupancy: float = 0.0
    rt_build_ms: float = 0.0
    rt_update_ms: float = 0.0
    rt_trace_ms: float = 0.0
    rt_ms: float = 0.0
    proposal_ms: float = 0.0
    exact_ms: float = 0.0
    total_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_max_ms: float = 0.0
    qps: float = 0.0
    family_point_triangle_exact_calls: int = 0
    family_edge_edge_exact_calls: int = 0
    family_conservative_exact_calls: int = 0
    family_unknown_exact_calls: int = 0
    exact_calls_total: int = 0
    candidate_buffer_bandwidth_mb_s: float = 0.0
    proposal_enqueue_dequeue_ms: float = 0.0
    total_tail_latency_ms: float = 0.0
    vram_peak_mb: int = 0


CONTRACT_SCHEMAS: dict[str, tuple[tuple[str, str], ...]] = {
    "CandidateRecord": (
        ("schema_version", "uint32"),
        ("candidate_id", "uint64"),
        ("query_id", "uint64"),
        ("slab_id", "uint32"),
        ("object_a_id", "uint32"),
        ("patch_a_id", "uint32"),
        ("object_b_id", "uint32"),
        ("patch_b_id", "uint32"),
        ("proxy_type_a", "ProxyType:uint8"),
        ("proxy_type_b", "ProxyType:uint8"),
        ("rt_hit_count", "uint32"),
        ("motion_bound", "float32[4]"),
        ("proxy_features_offset", "uint32"),
        ("flags", "uint32"),
    ),
    "ProposalOutput": (
        ("candidate_id", "uint64"),
        ("interval_scores", "float32[8]"),
        ("family_scores", "float32[8]"),
        ("priority_score", "float32"),
        ("cost_score", "float32"),
        ("uncertainty_score", "float32"),
    ),
    "ExactWorkItem": (
        ("work_item_id", "uint64"),
        ("parent_candidate_id", "uint64"),
        ("query_id", "uint64"),
        ("slab_id", "uint32"),
        ("patch_a_id", "uint32"),
        ("patch_b_id", "uint32"),
        ("interval_t0", "float64"),
        ("interval_t1", "float64"),
        ("feature_family_mask", "uint32"),
        ("topk_feature_ids_offset", "uint32"),
        ("depth", "uint16"),
        ("priority_score", "float32"),
        ("source", "ProposalSource:uint8"),
    ),
    "CertificateResult": (
        ("work_item_id", "uint64"),
        ("query_id", "uint64"),
        ("status", "CertificateStatus:uint8"),
        ("interval_t0", "float64"),
        ("interval_t1", "float64"),
        ("toi_upper", "float64"),
        ("safe_margin_lb", "float64"),
        ("witness_family", "uint8"),
        ("witness_id_a", "int64"),
        ("witness_id_b", "int64"),
        ("covered_feature_mask", "uint32"),
        ("eps_time", "float64"),
        ("eps_space", "float64"),
        ("reason_code", "uint16"),
        ("next_refinement_mode", "CertificateRefinementMode:uint8"),
    ),
    "AuditLogRow": (
        ("event_id", "uint64"),
        ("query_id", "uint64"),
        ("candidate_id", "uint64"),
        ("work_item_id", "uint64"),
        ("stage", "AuditStage:uint8"),
        ("action", "uint16"),
        ("depth", "uint16"),
        ("interval_t0", "float64"),
        ("interval_t1", "float64"),
        ("timestamp_us", "uint64"),
        ("aux_value0", "float64"),
        ("aux_value1", "float64"),
    ),
    "BenchmarkRow": (
        ("query_count", "uint64"),
        ("fn_count", "uint64"),
        ("fp_count", "uint64"),
        ("candidate_recall", "float64"),
        ("avg_candidates", "float64"),
        ("avg_exact_evals", "float64"),
        ("avg_subdivision_depth", "float64"),
        ("fallback_ratio", "float64"),
        ("rt_ms", "float64"),
        ("proposal_ms", "float64"),
        ("exact_ms", "float64"),
        ("total_ms", "float64"),
        ("qps", "float64"),
    ),
    "BenchmarkRunMeta": (
        ("schema_version", "uint32"),
        ("run_id", "string"),
        ("created_utc", "string"),
        ("dataset_name", "string"),
        ("scene_name", "string"),
        ("method_name", "string"),
        ("config_hash", "string"),
        ("config_json", "string"),
        ("seed", "int64"),
        ("row_count", "uint64"),
        ("git_commit", "string"),
        ("host_name", "string"),
        ("platform", "string"),
        ("python_version", "string"),
        ("gpu_name", "string"),
        ("driver_version", "string"),
        ("cuda_version", "string"),
        ("optix_version", "string"),
        ("vram_total_mb", "uint64"),
        ("vram_free_mb", "uint64"),
        ("output_csv", "string"),
        ("output_jsonl", "string"),
        ("output_run_meta_json", "string"),
        ("notes", "string"),
    ),
    "BenchmarkRowV2": (
        ("schema_version", "uint32"),
        ("run_id", "string"),
        ("dataset_name", "string"),
        ("scene_name", "string"),
        ("method_name", "string"),
        ("config_hash", "string"),
        ("seed", "int64"),
        ("query_count", "uint64"),
        ("fn_count", "uint64"),
        ("fp_count", "uint64"),
        ("candidate_recall", "float64"),
        ("avg_candidates", "float64"),
        ("avg_exact_evals", "float64"),
        ("avg_subdivision_depth", "float64"),
        ("fallback_ratio", "float64"),
        ("candidate_inflation_ratio", "float64"),
        ("undecided_to_resolved_ratio", "float64"),
        ("exact_queue_occupancy", "float64"),
        ("rt_build_ms", "float64"),
        ("rt_update_ms", "float64"),
        ("rt_trace_ms", "float64"),
        ("rt_ms", "float64"),
        ("proposal_ms", "float64"),
        ("exact_ms", "float64"),
        ("total_ms", "float64"),
        ("latency_min_ms", "float64"),
        ("latency_p50_ms", "float64"),
        ("latency_p90_ms", "float64"),
        ("latency_p95_ms", "float64"),
        ("latency_p99_ms", "float64"),
        ("latency_max_ms", "float64"),
        ("qps", "float64"),
        ("family_point_triangle_exact_calls", "uint64"),
        ("family_edge_edge_exact_calls", "uint64"),
        ("family_conservative_exact_calls", "uint64"),
        ("family_unknown_exact_calls", "uint64"),
        ("exact_calls_total", "uint64"),
        ("candidate_buffer_bandwidth_mb_s", "float64"),
        ("proposal_enqueue_dequeue_ms", "float64"),
        ("total_tail_latency_ms", "float64"),
        ("vram_peak_mb", "uint64"),
    ),
}


CONTRACT_TYPES = {
    "CandidateRecord": CandidateRecord,
    "ProposalOutput": ProposalOutput,
    "ExactWorkItem": ExactWorkItem,
    "CertificateResult": CertificateResult,
    "AuditLogRow": AuditLogRow,
    "BenchmarkRow": BenchmarkRow,
    "BenchmarkRunMeta": BenchmarkRunMeta,
    "BenchmarkRowV2": BenchmarkRowV2,
}


def contract_name(contract_type: str | type | object) -> str:
    if isinstance(contract_type, str):
        return contract_type
    if isinstance(contract_type, type):
        return contract_type.__name__
    return type(contract_type).__name__


def schema_field_names(contract_type: str | type | object) -> tuple[str, ...]:
    name = contract_name(contract_type)
    return tuple(field_name for field_name, _ in CONTRACT_SCHEMAS[name])
