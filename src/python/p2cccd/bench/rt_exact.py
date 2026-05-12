from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Sequence

from p2cccd.contracts import (
    AuditLogRow,
    AuditStage,
    BenchmarkRow,
    CandidateRecord,
    CertificateRefinementMode,
    CertificateResult,
    CertificateStatus,
    ExactWorkItem,
    ProposalSource,
    ProxyType,
)
from p2cccd.data.dataset import GeneratedDataset
from p2cccd.data.oracle import ExactOracleTrace, evaluate_swept_sphere_oracle
from p2cccd.data.samplers import MotionDiscPairSample
from p2cccd.datasets.ccd import CCDQueryFamily, DatasetQueryBatch, ExternalCCDQuery
from p2cccd.validators import (
    validate_audit_log_row,
    validate_benchmark_row,
    validate_candidate_record,
    validate_certificate_result,
    validate_exact_work_item,
)

from .bvh_exact import (
    BroadPhaseBackend,
    BroadPhasePair,
    BroadPhasePrimitive,
    BroadPhaseStats,
    CppOptixBroadPhaseBackend,
    CpuAabbBroadPhaseBackend,
    _try_load_p2cccd_cpp,
    external_query_to_broad_phase_primitives,
    internal_sample_to_broad_phase_primitives,
)
from .pure_exact_cpu import (
    PureExactCPUConfig,
    PureExactQueryResult,
    evaluate_external_ccd_query,
)


FEATURE_FAMILY_POINT_TRIANGLE = 1 << 0
FEATURE_FAMILY_EDGE_EDGE = 1 << 1
FEATURE_FAMILY_CONSERVATIVE = FEATURE_FAMILY_POINT_TRIANGLE | FEATURE_FAMILY_EDGE_EDGE

RAW_CANDIDATE_VALID = 1 << 0
RAW_CANDIDATE_AABB_OVERLAP = 1 << 1

RT_AUDIT_CANDIDATE = 1
EXACT_AUDIT_DEQUEUED = 1
EXACT_AUDIT_COLLISION = 2
EXACT_AUDIT_SEPARATION = 3
EXACT_AUDIT_UNDECIDED = 4

CERTIFICATE_REASON_NONE = 0
CERTIFICATE_REASON_MAX_SUBDIVISION = 2


@dataclass(frozen=True, slots=True)
class RtCandidateTiming:
    build_ms: float = 0.0
    update_ms: float = 0.0
    trace_ms: float = 0.0
    compact_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class RtCandidateStats:
    backend_name: str
    primitive_count: int
    raw_hit_count: int
    compact_candidate_count: int
    candidate_recall: float
    timing: RtCandidateTiming


@dataclass(frozen=True, slots=True)
class RTExactConfig:
    exact: PureExactCPUConfig = field(default_factory=PureExactCPUConfig)
    backend_name: str = "cpu_reference_rt"
    enable_cuda_exact: bool = True
    same_query_only: bool = True
    first_work_item_id: int = 1
    first_event_id: int = 1
    first_timestamp_us: int = 1


@dataclass(frozen=True, slots=True)
class RTExactResult:
    benchmark: BenchmarkRow
    query_results: tuple[PureExactQueryResult, ...]
    candidates: tuple[CandidateRecord, ...]
    work_items: tuple[ExactWorkItem, ...]
    certificates: tuple[CertificateResult, ...]
    audit_log: tuple[AuditLogRow, ...]
    candidate_stats: RtCandidateStats
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


def _validate_config(config: RTExactConfig) -> RTExactConfig:
    if config.backend_name not in {"cpu_reference_rt", "optix_compatible", "optix_rt"}:
        raise ValueError("RTExactConfig.backend_name is unsupported")
    if config.first_work_item_id <= 0:
        raise ValueError("RTExactConfig.first_work_item_id must be positive")
    if config.first_event_id <= 0:
        raise ValueError("RTExactConfig.first_event_id must be positive")
    if config.first_timestamp_us <= 0:
        raise ValueError("RTExactConfig.first_timestamp_us must be positive")
    return config


def _default_rt_broad_phase(config: RTExactConfig) -> BroadPhaseBackend:
    if config.backend_name in {"optix_compatible", "optix_rt"}:
        return CppOptixBroadPhaseBackend(
            name="optix_rt",
            allow_cpu_fallback=config.backend_name == "optix_compatible",
        )
    return CpuAabbBroadPhaseBackend(name=config.backend_name)


def _runtime_query_ids(query_ids: Sequence[int]) -> dict[int, int]:
    runtime_by_source: dict[int, int] = {}
    used: set[int] = set()
    for index, query_id in enumerate(query_ids):
        runtime_id = query_id if query_id > 0 else index + 1
        while runtime_id in used:
            runtime_id += 1
        runtime_by_source[query_id] = runtime_id
        used.add(runtime_id)
    return runtime_by_source


def _family_mask_for_external(query: ExternalCCDQuery) -> int:
    if query.family is CCDQueryFamily.VERTEX_FACE:
        return FEATURE_FAMILY_POINT_TRIANGLE
    if query.family is CCDQueryFamily.EDGE_EDGE:
        return FEATURE_FAMILY_EDGE_EDGE
    raise ValueError(f"unsupported external CCD query family: {query.family}")


def _witness_family_for_result(result: PureExactQueryResult, feature_family_mask: int) -> int:
    if result.family == "point_triangle":
        return FEATURE_FAMILY_POINT_TRIANGLE
    if result.family == "edge_edge":
        return FEATURE_FAMILY_EDGE_EDGE
    return feature_family_mask


def _oracle_trace_by_query_id(samples: Sequence[MotionDiscPairSample]) -> dict[int, ExactOracleTrace]:
    return {sample.query_id: evaluate_swept_sphere_oracle(sample) for sample in samples}


def _load_cpp_cuda_exact_module() -> Any | None:
    cpp = _try_load_p2cccd_cpp()
    required = (
        "LinearVertexTrajectory",
        "PointTriangleIntervalPrimitive",
        "EdgeEdgeIntervalPrimitive",
        "CertificateEngineConfig",
        "evaluate_point_triangle_batch_cuda",
        "evaluate_edge_edge_batch_cuda",
        "is_cuda_exact_built",
    )
    if cpp is None or any(not hasattr(cpp, name) for name in required):
        return None
    return cpp


def _load_cpp_external_candidate_module() -> Any | None:
    cpp = _try_load_p2cccd_cpp()
    required = (
        "generate_candidates_for_external_batch",
        "ExternalBatchCandidateResult",
        "RuntimeQueryIdMapping",
    )
    if cpp is None or any(not hasattr(cpp, name) for name in required):
        return None
    return cpp


def _to_cpp_certificate_engine_config(cpp: Any, config: PureExactCPUConfig) -> Any:
    cpp_config = cpp.CertificateEngineConfig()
    cpp_config.eps_time = float(config.eps_time)
    cpp_config.eps_space = float(config.eps_space)
    cpp_config.max_subdivision_depth = int(config.max_subdivision_depth)
    return cpp_config


def _to_cpp_linear_trajectory(
    cpp: Any,
    *,
    feature_id: int,
    position_t0: tuple[float, float, float],
    position_t1: tuple[float, float, float],
) -> Any:
    trajectory = cpp.LinearVertexTrajectory()
    trajectory.feature_id = int(feature_id)
    trajectory.position_t0 = [float(value) for value in position_t0]
    trajectory.position_t1 = [float(value) for value in position_t1]
    return trajectory


def _make_cpp_external_primitive(cpp: Any, query: ExternalCCDQuery) -> Any:
    if query.family is CCDQueryFamily.VERTEX_FACE:
        primitive = cpp.PointTriangleIntervalPrimitive()
        primitive.point_id = 0
        primitive.triangle_id = 1
        primitive.point = _to_cpp_linear_trajectory(
            cpp,
            feature_id=0,
            position_t0=query.vertices_t0[0],
            position_t1=query.vertices_t1[0],
        )
        primitive.triangle_v0 = _to_cpp_linear_trajectory(
            cpp,
            feature_id=1,
            position_t0=query.vertices_t0[1],
            position_t1=query.vertices_t1[1],
        )
        primitive.triangle_v1 = _to_cpp_linear_trajectory(
            cpp,
            feature_id=2,
            position_t0=query.vertices_t0[2],
            position_t1=query.vertices_t1[2],
        )
        primitive.triangle_v2 = _to_cpp_linear_trajectory(
            cpp,
            feature_id=3,
            position_t0=query.vertices_t0[3],
            position_t1=query.vertices_t1[3],
        )
        return primitive

    primitive = cpp.EdgeEdgeIntervalPrimitive()
    primitive.edge_a_id = 0
    primitive.edge_b_id = 1
    primitive.edge_a0 = _to_cpp_linear_trajectory(
        cpp,
        feature_id=0,
        position_t0=query.vertices_t0[0],
        position_t1=query.vertices_t1[0],
    )
    primitive.edge_a1 = _to_cpp_linear_trajectory(
        cpp,
        feature_id=1,
        position_t0=query.vertices_t0[1],
        position_t1=query.vertices_t1[1],
    )
    primitive.edge_b0 = _to_cpp_linear_trajectory(
        cpp,
        feature_id=2,
        position_t0=query.vertices_t0[2],
        position_t1=query.vertices_t1[2],
    )
    primitive.edge_b1 = _to_cpp_linear_trajectory(
        cpp,
        feature_id=3,
        position_t0=query.vertices_t0[3],
        position_t1=query.vertices_t1[3],
    )
    return primitive


def _cpp_refinement_mode_to_contract(raw_mode: int) -> CertificateRefinementMode:
    if raw_mode == 1:
        return CertificateRefinementMode.BISECT_INTERVAL
    if raw_mode == 2:
        return CertificateRefinementMode.REQUEST_GEOMETRY
    return CertificateRefinementMode.NONE


def _status_string_from_cpp(cpp: Any, raw_status: int) -> str:
    if raw_status == int(cpp.CertificateStatus.COLLISION):
        return "collision"
    if raw_status == int(cpp.CertificateStatus.SEPARATION):
        return "separation"
    return "undecided"


def _certificate_from_cpp_primitive_result(
    cpp: Any,
    primitive_result: Any,
    work_item: ExactWorkItem,
    config: PureExactCPUConfig,
) -> CertificateResult:
    raw_status = int(primitive_result.status)
    if raw_status == int(cpp.CertificateStatus.COLLISION):
        status = CertificateStatus.COLLISION
    elif raw_status == int(cpp.CertificateStatus.SEPARATION):
        status = CertificateStatus.SEPARATION
    else:
        status = CertificateStatus.UNDECIDED
    certificate = CertificateResult(
        work_item_id=work_item.work_item_id,
        query_id=work_item.query_id,
        status=status,
        interval_t0=float(primitive_result.interval_t0),
        interval_t1=float(primitive_result.interval_t1),
        toi_upper=float(primitive_result.toi_upper),
        safe_margin_lb=max(0.0, float(primitive_result.safe_margin_lb)),
        witness_family=int(primitive_result.witness_family),
        witness_id_a=int(primitive_result.witness_id_a),
        witness_id_b=int(primitive_result.witness_id_b),
        covered_feature_mask=int(primitive_result.covered_feature_mask),
        eps_time=config.eps_time,
        eps_space=config.eps_space,
        reason_code=int(primitive_result.reason_code),
        next_refinement_mode=_cpp_refinement_mode_to_contract(int(primitive_result.next_refinement_mode)),
    )
    return validate_certificate_result(certificate)


def _query_result_from_cpp_primitive_result(
    cpp: Any,
    query: ExternalCCDQuery,
    primitive_result: Any,
    config: PureExactCPUConfig,
) -> PureExactQueryResult:
    status = _status_string_from_cpp(cpp, int(primitive_result.status))
    predicted_collision = status == "collision" or (
        status == "undecided" and config.conservative_undecided_as_collision
    )
    return PureExactQueryResult(
        query_id=query.query_id,
        family=query.family.p2cccd_witness_family,
        predicted_collision=predicted_collision,
        ground_truth_collision=query.ground_truth_collides,
        status=status,
        toi_upper=float(primitive_result.toi_upper),
        safe_margin_lb=max(0.0, float(primitive_result.safe_margin_lb)),
        exact_evals=1,
        max_depth=config.max_subdivision_depth if status == "undecided" else 0,
    )


def _process_external_exact_work_queue_cuda(
    batch: DatasetQueryBatch,
    candidates: Sequence[CandidateRecord],
    work_items: Sequence[ExactWorkItem],
    runtime_ids: dict[int, int],
    config: RTExactConfig,
) -> tuple[tuple[PureExactQueryResult, ...], tuple[CertificateResult, ...], tuple[AuditLogRow, ...], float, str] | None:
    if not config.enable_cuda_exact or len(work_items) < 32:
        return None
    cpp = _load_cpp_cuda_exact_module()
    if cpp is None or not bool(cpp.is_cuda_exact_built()):
        return None

    query_by_runtime_id = {runtime_ids[query.query_id]: query for query in batch.queries}
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    pt_entries: list[tuple[ExternalCCDQuery, CandidateRecord, ExactWorkItem, Any]] = []
    ee_entries: list[tuple[ExternalCCDQuery, CandidateRecord, ExactWorkItem, Any]] = []

    try:
        for item in work_items:
            query = query_by_runtime_id.get(item.query_id)
            if query is None:
                raise ValueError("exact work item references an unknown external query")
            parent = candidate_by_id[item.parent_candidate_id]
            primitive = _make_cpp_external_primitive(cpp, query)
            if query.family is CCDQueryFamily.VERTEX_FACE:
                pt_entries.append((query, parent, item, primitive))
            else:
                ee_entries.append((query, parent, item, primitive))

        cpp_config = _to_cpp_certificate_engine_config(cpp, config.exact)
        start = time.perf_counter()
        pt_results = tuple(
            cpp.evaluate_point_triangle_batch_cuda(
                [entry[3] for entry in pt_entries],
                0.0,
                1.0,
                cpp_config,
            )
        ) if pt_entries else ()
        ee_results = tuple(
            cpp.evaluate_edge_edge_batch_cuda(
                [entry[3] for entry in ee_entries],
                0.0,
                1.0,
                cpp_config,
            )
        ) if ee_entries else ()
        exact_elapsed_ms = (time.perf_counter() - start) * 1000.0
    except Exception:
        return None

    results_by_work_item: dict[int, tuple[PureExactQueryResult, CertificateResult]] = {}
    for entry, primitive_result in zip(pt_entries, pt_results):
        query, _, item, _ = entry
        query_result = _query_result_from_cpp_primitive_result(cpp, query, primitive_result, config.exact)
        certificate = _certificate_from_cpp_primitive_result(cpp, primitive_result, item, config.exact)
        results_by_work_item[item.work_item_id] = (query_result, certificate)
    for entry, primitive_result in zip(ee_entries, ee_results):
        query, _, item, _ = entry
        query_result = _query_result_from_cpp_primitive_result(cpp, query, primitive_result, config.exact)
        certificate = _certificate_from_cpp_primitive_result(cpp, primitive_result, item, config.exact)
        results_by_work_item[item.work_item_id] = (query_result, certificate)

    audit_log, event_id, timestamp_us = _append_rt_audit(
        candidates,
        first_event_id=config.first_event_id,
        first_timestamp_us=config.first_timestamp_us,
    )
    certificates: list[CertificateResult] = []
    exact_by_runtime_id: dict[int, PureExactQueryResult] = {}
    for item in work_items:
        parent = candidate_by_id[item.parent_candidate_id]
        audit_log.append(
            _audit_row(
                event_id=event_id,
                query_id=item.query_id,
                candidate_id=parent.candidate_id,
                work_item_id=item.work_item_id,
                stage=AuditStage.EXACT,
                action=EXACT_AUDIT_DEQUEUED,
                depth=item.depth,
                timestamp_us=timestamp_us,
            )
        )
        event_id += 1
        timestamp_us += 1
        query_result, certificate = results_by_work_item[item.work_item_id]
        exact_by_runtime_id[item.query_id] = query_result
        certificates.append(certificate)
        event_id, timestamp_us = _append_exact_audit(
            audit_log,
            event_id=event_id,
            timestamp_us=timestamp_us,
            candidate=parent,
            item=item,
            certificate=certificate,
        )

    query_results: list[PureExactQueryResult] = []
    for query in batch.queries:
        runtime_id = runtime_ids[query.query_id]
        if runtime_id in exact_by_runtime_id:
            query_results.append(exact_by_runtime_id[runtime_id])
        else:
            query_results.append(
                PureExactQueryResult(
                    query_id=query.query_id,
                    family=query.family.p2cccd_witness_family,
                    predicted_collision=False,
                    ground_truth_collision=query.ground_truth_collides,
                    status="rt_candidate_separation",
                    toi_upper=1.0,
                    safe_margin_lb=0.0,
                    exact_evals=0,
                    max_depth=0,
                )
            )
    return tuple(query_results), tuple(certificates), tuple(audit_log), exact_elapsed_ms, "cuda_exact"


def _make_candidate_id(runtime_query_id: int, ordinal: int) -> int:
    return runtime_query_id * 1_000_003 + ordinal + 1


def _make_timing(
    *,
    build_ms: float,
    broad_phase_stats: BroadPhaseStats,
    compact_ms: float,
) -> RtCandidateTiming:
    rt_build_ms = float(build_ms) + float(getattr(broad_phase_stats, "build_ms", 0.0))
    rt_update_ms = float(getattr(broad_phase_stats, "update_ms", 0.0))
    # BenchmarkRowV2 currently has no explicit compact/stats split, so these
    # backend-side candidate-stage costs stay in the trace bucket.
    rt_trace_ms = (
        float(getattr(broad_phase_stats, "trace_ms", broad_phase_stats.elapsed_ms))
        + float(getattr(broad_phase_stats, "compact_ms", 0.0))
        + float(getattr(broad_phase_stats, "stats_ms", 0.0))
    )
    total_ms = rt_build_ms + rt_update_ms + rt_trace_ms + compact_ms
    return RtCandidateTiming(
        build_ms=rt_build_ms,
        update_ms=rt_update_ms,
        trace_ms=rt_trace_ms,
        compact_ms=compact_ms,
        total_ms=total_ms,
    )


def _candidate_recall(
    known_collisions: Sequence[bool | None],
    active_indices: set[int],
) -> float:
    collision_indices = {
        index for index, collides in enumerate(known_collisions) if collides is True
    }
    if not collision_indices:
        return 1.0
    return len(collision_indices & active_indices) / len(collision_indices)


def _external_candidate_from_pair(
    query: ExternalCCDQuery,
    runtime_query_id: int,
    pair: BroadPhasePair,
    *,
    ordinal: int,
) -> CandidateRecord:
    object_a_id, object_b_id = query.box_pair if query.box_pair is not None else (1, 2)
    candidate = CandidateRecord(
        candidate_id=_make_candidate_id(runtime_query_id, ordinal),
        query_id=runtime_query_id,
        slab_id=0,
        object_a_id=int(object_a_id),
        patch_a_id=0 if query.family is CCDQueryFamily.VERTEX_FACE else 1,
        object_b_id=int(object_b_id),
        patch_b_id=max(1, int(query.source_query_index) + 1),
        proxy_type_a=ProxyType.SWEPT_AABB,
        proxy_type_b=ProxyType.SWEPT_AABB,
        rt_hit_count=1,
        motion_bound=[0.0, 0.0, 0.0, 0.0],
        flags=RAW_CANDIDATE_VALID | RAW_CANDIDATE_AABB_OVERLAP,
    )
    if pair.query_id != query.query_id:
        raise ValueError("RT candidate pair query_id does not match external query")
    return validate_candidate_record(candidate)


def _candidate_record_from_cpp(candidate: Any) -> CandidateRecord:
    converted = CandidateRecord(
        schema_version=int(candidate.schema_version),
        candidate_id=int(candidate.candidate_id),
        query_id=int(candidate.query_id),
        slab_id=int(candidate.slab_id),
        object_a_id=int(candidate.object_a_id),
        patch_a_id=int(candidate.patch_a_id),
        object_b_id=int(candidate.object_b_id),
        patch_b_id=int(candidate.patch_b_id),
        proxy_type_a=ProxyType(int(candidate.proxy_type_a)),
        proxy_type_b=ProxyType(int(candidate.proxy_type_b)),
        rt_hit_count=int(candidate.rt_hit_count),
        motion_bound=[float(value) for value in candidate.motion_bound],
        proxy_features_offset=int(candidate.proxy_features_offset),
        flags=int(candidate.flags),
    )
    return validate_candidate_record(converted)


def _internal_candidate_from_pair(
    sample: MotionDiscPairSample,
    runtime_query_id: int,
    pair: BroadPhasePair,
    *,
    ordinal: int,
) -> CandidateRecord:
    candidate = CandidateRecord(
        candidate_id=_make_candidate_id(runtime_query_id, ordinal),
        query_id=runtime_query_id,
        slab_id=sample.slab_id,
        object_a_id=sample.object_a_id,
        patch_a_id=sample.patch_a_id,
        object_b_id=sample.object_b_id,
        patch_b_id=sample.patch_b_id,
        proxy_type_a=sample.proxy_type_a,
        proxy_type_b=sample.proxy_type_b,
        rt_hit_count=max(1, pair.primitive_b_id - pair.primitive_a_id),
        motion_bound=[0.0, 0.0, 0.0, 0.0],
        flags=RAW_CANDIDATE_VALID | RAW_CANDIDATE_AABB_OVERLAP,
    )
    if pair.query_id != sample.query_id:
        raise ValueError("RT candidate pair query_id does not match internal sample")
    return validate_candidate_record(candidate)


def _make_external_candidates(
    batch: DatasetQueryBatch,
    config: RTExactConfig,
    *,
    backend: BroadPhaseBackend | None = None,
) -> tuple[tuple[CandidateRecord, ...], RtCandidateStats, dict[int, int]]:
    if backend is None and config.same_query_only:
        cpp = _load_cpp_external_candidate_module()
        if cpp is not None:
            cpp_result = cpp.generate_candidates_for_external_batch(
                batch,
                backend_name=config.backend_name,
                allow_optix_cpu_fallback=config.backend_name == "optix_compatible",
            )
            runtime_ids = {
                int(entry.source_query_id): int(entry.runtime_query_id)
                for entry in cpp_result.runtime_query_ids
            }
            candidates = tuple(_candidate_record_from_cpp(candidate) for candidate in cpp_result.candidates)
            backend_name = str(cpp_result.backend_name)
            if backend_name == "cpu_reference" and config.backend_name == "cpu_reference_rt":
                backend_name = "cpu_reference_rt"
            stats = RtCandidateStats(
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
            return candidates, stats, runtime_ids

    build_start = time.perf_counter()
    primitives: list[BroadPhasePrimitive] = []
    for query in batch.queries:
        primitives.extend(external_query_to_broad_phase_primitives(query))
    build_ms = (time.perf_counter() - build_start) * 1000.0

    broad_phase = backend or _default_rt_broad_phase(config)
    pairs, broad_phase_stats = broad_phase.find_pairs(
        primitives,
        same_query_only=config.same_query_only,
    )
    query_by_id = {query.query_id: query for query in batch.queries}
    runtime_ids = _runtime_query_ids([query.query_id for query in batch.queries])
    index_by_query_id = {query.query_id: index for index, query in enumerate(batch.queries)}

    compact_start = time.perf_counter()
    candidates: list[CandidateRecord] = []
    for ordinal, pair in enumerate(pairs):
        query = query_by_id.get(pair.query_id)
        if query is None:
            raise ValueError("RT candidate pair references an unknown external query")
        candidates.append(
            _external_candidate_from_pair(
                query,
                runtime_ids[query.query_id],
                pair,
                ordinal=ordinal,
            )
        )
    compact_ms = (time.perf_counter() - compact_start) * 1000.0
    active_indices = {
        index_by_query_id[pair.query_id]
        for pair in pairs
        if pair.query_id in index_by_query_id
    }
    stats = RtCandidateStats(
        backend_name=broad_phase_stats.backend_name,
        primitive_count=len(primitives),
        raw_hit_count=len(pairs),
        compact_candidate_count=len(candidates),
        candidate_recall=_candidate_recall(
            [query.ground_truth_collides for query in batch.queries],
            active_indices,
        ),
        timing=_make_timing(
            build_ms=build_ms,
            broad_phase_stats=broad_phase_stats,
            compact_ms=compact_ms,
        ),
    )
    return tuple(candidates), stats, runtime_ids


def _make_internal_candidates(
    samples: Sequence[MotionDiscPairSample],
    config: RTExactConfig,
    *,
    backend: BroadPhaseBackend | None = None,
    oracle_traces_by_query_id: dict[int, ExactOracleTrace] | None = None,
) -> tuple[tuple[CandidateRecord, ...], RtCandidateStats, dict[int, int]]:
    build_start = time.perf_counter()
    primitives: list[BroadPhasePrimitive] = []
    for sample in samples:
        primitives.extend(internal_sample_to_broad_phase_primitives(sample))
    build_ms = (time.perf_counter() - build_start) * 1000.0

    broad_phase = backend or _default_rt_broad_phase(config)
    pairs, broad_phase_stats = broad_phase.find_pairs(
        primitives,
        same_query_only=config.same_query_only,
    )
    sample_by_query_id = {sample.query_id: sample for sample in samples}
    runtime_ids = _runtime_query_ids([sample.query_id for sample in samples])
    index_by_query_id = {sample.query_id: index for index, sample in enumerate(samples)}

    compact_start = time.perf_counter()
    candidates: list[CandidateRecord] = []
    for ordinal, pair in enumerate(pairs):
        sample = sample_by_query_id.get(pair.query_id)
        if sample is None:
            raise ValueError("RT candidate pair references an unknown internal sample")
        candidates.append(
            _internal_candidate_from_pair(
                sample,
                runtime_ids[sample.query_id],
                pair,
                ordinal=ordinal,
            )
        )
    compact_ms = (time.perf_counter() - compact_start) * 1000.0
    trace_by_query_id = oracle_traces_by_query_id or _oracle_trace_by_query_id(samples)
    known_collisions = [trace_by_query_id[sample.query_id].collided for sample in samples]
    active_indices = {
        index_by_query_id[pair.query_id]
        for pair in pairs
        if pair.query_id in index_by_query_id
    }
    stats = RtCandidateStats(
        backend_name=broad_phase_stats.backend_name,
        primitive_count=len(primitives),
        raw_hit_count=len(pairs),
        compact_candidate_count=len(candidates),
        candidate_recall=_candidate_recall(known_collisions, active_indices),
        timing=_make_timing(
            build_ms=build_ms,
            broad_phase_stats=broad_phase_stats,
            compact_ms=compact_ms,
        ),
    )
    return tuple(candidates), stats, runtime_ids


def schedule_exact_work_items_without_stpf(
    candidates: Sequence[CandidateRecord],
    *,
    family_by_runtime_query_id: dict[int, int],
    first_work_item_id: int = 1,
) -> tuple[ExactWorkItem, ...]:
    if first_work_item_id <= 0:
        raise ValueError("first_work_item_id must be positive")
    work_items: list[ExactWorkItem] = []
    for offset, candidate in enumerate(candidates):
        validate_candidate_record(candidate)
        item = ExactWorkItem(
            work_item_id=first_work_item_id + offset,
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
            source=ProposalSource.RAW,
        )
        work_items.append(validate_exact_work_item(item))
    return tuple(work_items)


def _certificate_from_result(
    result: PureExactQueryResult,
    work_item: ExactWorkItem,
    config: PureExactCPUConfig,
) -> CertificateResult:
    if result.status == "collision":
        certificate = CertificateResult(
            work_item_id=work_item.work_item_id,
            query_id=work_item.query_id,
            status=CertificateStatus.COLLISION,
            interval_t0=work_item.interval_t0,
            interval_t1=work_item.interval_t1,
            toi_upper=min(work_item.interval_t1, max(work_item.interval_t0, result.toi_upper)),
            safe_margin_lb=0.0,
            witness_family=_witness_family_for_result(result, work_item.feature_family_mask),
            witness_id_a=int(work_item.patch_a_id),
            witness_id_b=int(work_item.patch_b_id),
            covered_feature_mask=0,
            eps_time=config.eps_time,
            eps_space=config.eps_space,
            reason_code=CERTIFICATE_REASON_NONE,
            next_refinement_mode=CertificateRefinementMode.NONE,
        )
    elif result.status == "separation":
        certificate = CertificateResult(
            work_item_id=work_item.work_item_id,
            query_id=work_item.query_id,
            status=CertificateStatus.SEPARATION,
            interval_t0=work_item.interval_t0,
            interval_t1=work_item.interval_t1,
            toi_upper=work_item.interval_t1,
            safe_margin_lb=max(0.0, result.safe_margin_lb),
            witness_family=0,
            witness_id_a=-1,
            witness_id_b=-1,
            covered_feature_mask=work_item.feature_family_mask,
            eps_time=config.eps_time,
            eps_space=config.eps_space,
            reason_code=CERTIFICATE_REASON_NONE,
            next_refinement_mode=CertificateRefinementMode.NONE,
        )
    else:
        certificate = CertificateResult(
            work_item_id=work_item.work_item_id,
            query_id=work_item.query_id,
            status=CertificateStatus.UNDECIDED,
            interval_t0=work_item.interval_t0,
            interval_t1=work_item.interval_t1,
            toi_upper=work_item.interval_t1,
            safe_margin_lb=0.0,
            witness_family=0,
            witness_id_a=-1,
            witness_id_b=-1,
            covered_feature_mask=0,
            eps_time=config.eps_time,
            eps_space=config.eps_space,
            reason_code=CERTIFICATE_REASON_MAX_SUBDIVISION,
            next_refinement_mode=CertificateRefinementMode.BISECT_INTERVAL,
        )
    return validate_certificate_result(certificate)


def _audit_row(
    *,
    event_id: int,
    query_id: int,
    candidate_id: int,
    work_item_id: int,
    stage: AuditStage,
    action: int,
    depth: int,
    timestamp_us: int,
    aux_value0: float = 0.0,
    aux_value1: float = 0.0,
) -> AuditLogRow:
    return validate_audit_log_row(
        AuditLogRow(
            event_id=event_id,
            query_id=query_id,
            candidate_id=candidate_id,
            work_item_id=work_item_id,
            stage=stage,
            action=action,
            depth=depth,
            interval_t0=0.0,
            interval_t1=1.0,
            timestamp_us=timestamp_us,
            aux_value0=aux_value0,
            aux_value1=aux_value1,
        )
    )


def _audit_action_for_certificate(certificate: CertificateResult) -> int:
    if certificate.status is CertificateStatus.COLLISION:
        return EXACT_AUDIT_COLLISION
    if certificate.status is CertificateStatus.SEPARATION:
        return EXACT_AUDIT_SEPARATION
    return EXACT_AUDIT_UNDECIDED


def _append_rt_audit(
    candidates: Sequence[CandidateRecord],
    *,
    first_event_id: int,
    first_timestamp_us: int,
) -> tuple[list[AuditLogRow], int, int]:
    audit_log: list[AuditLogRow] = []
    event_id = first_event_id
    timestamp_us = first_timestamp_us
    for candidate in candidates:
        audit_log.append(
            _audit_row(
                event_id=event_id,
                query_id=candidate.query_id,
                candidate_id=candidate.candidate_id,
                work_item_id=0,
                stage=AuditStage.RT,
                action=RT_AUDIT_CANDIDATE,
                depth=0,
                timestamp_us=timestamp_us,
                aux_value0=float(candidate.rt_hit_count),
            )
        )
        event_id += 1
        timestamp_us += 1
    return audit_log, event_id, timestamp_us


def _append_exact_audit(
    audit_log: list[AuditLogRow],
    *,
    event_id: int,
    timestamp_us: int,
    candidate: CandidateRecord,
    item: ExactWorkItem,
    certificate: CertificateResult,
) -> tuple[int, int]:
    audit_log.append(
        _audit_row(
            event_id=event_id,
            query_id=item.query_id,
            candidate_id=candidate.candidate_id,
            work_item_id=item.work_item_id,
            stage=AuditStage.EXACT,
            action=_audit_action_for_certificate(certificate),
            depth=item.depth,
            timestamp_us=timestamp_us,
            aux_value0=certificate.safe_margin_lb
            if certificate.status is CertificateStatus.SEPARATION
            else certificate.toi_upper,
            aux_value1=float(certificate.next_refinement_mode),
        )
    )
    return event_id + 1, timestamp_us + 1


def _process_external_exact_work_queue(
    batch: DatasetQueryBatch,
    candidates: Sequence[CandidateRecord],
    work_items: Sequence[ExactWorkItem],
    runtime_ids: dict[int, int],
    config: RTExactConfig,
) -> tuple[tuple[PureExactQueryResult, ...], tuple[CertificateResult, ...], tuple[AuditLogRow, ...], float, str]:
    accelerated = _process_external_exact_work_queue_cuda(
        batch,
        candidates,
        work_items,
        runtime_ids,
        config,
    )
    if accelerated is not None:
        return accelerated

    query_by_runtime_id = {runtime_ids[query.query_id]: query for query in batch.queries}
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    audit_log, event_id, timestamp_us = _append_rt_audit(
        candidates,
        first_event_id=config.first_event_id,
        first_timestamp_us=config.first_timestamp_us,
    )
    certificates: list[CertificateResult] = []
    exact_by_runtime_id: dict[int, PureExactQueryResult] = {}

    start = time.perf_counter()
    for item in work_items:
        query = query_by_runtime_id.get(item.query_id)
        if query is None:
            raise ValueError("exact work item references an unknown external query")
        parent = candidate_by_id[item.parent_candidate_id]
        audit_log.append(
            _audit_row(
                event_id=event_id,
                query_id=item.query_id,
                candidate_id=parent.candidate_id,
                work_item_id=item.work_item_id,
                stage=AuditStage.EXACT,
                action=EXACT_AUDIT_DEQUEUED,
                depth=item.depth,
                timestamp_us=timestamp_us,
            )
        )
        event_id += 1
        timestamp_us += 1

        exact_result = evaluate_external_ccd_query(query, config.exact)
        exact_by_runtime_id[item.query_id] = exact_result
        certificate = _certificate_from_result(exact_result, item, config.exact)
        certificates.append(certificate)
        event_id, timestamp_us = _append_exact_audit(
            audit_log,
            event_id=event_id,
            timestamp_us=timestamp_us,
            candidate=parent,
            item=item,
            certificate=certificate,
        )
    exact_elapsed_ms = (time.perf_counter() - start) * 1000.0

    query_results: list[PureExactQueryResult] = []
    for query in batch.queries:
        runtime_id = runtime_ids[query.query_id]
        if runtime_id in exact_by_runtime_id:
            query_results.append(exact_by_runtime_id[runtime_id])
        else:
            query_results.append(
                PureExactQueryResult(
                    query_id=query.query_id,
                    family=query.family.p2cccd_witness_family,
                    predicted_collision=False,
                    ground_truth_collision=query.ground_truth_collides,
                    status="rt_candidate_separation",
                    toi_upper=1.0,
                    safe_margin_lb=0.0,
                    exact_evals=0,
                    max_depth=0,
                )
            )
    return tuple(query_results), tuple(certificates), tuple(audit_log), exact_elapsed_ms, "pure_exact_cpu"


def _process_internal_exact_work_queue(
    samples: Sequence[MotionDiscPairSample],
    candidates: Sequence[CandidateRecord],
    work_items: Sequence[ExactWorkItem],
    runtime_ids: dict[int, int],
    config: RTExactConfig,
    *,
    oracle_traces_by_query_id: dict[int, ExactOracleTrace] | None = None,
) -> tuple[tuple[PureExactQueryResult, ...], tuple[CertificateResult, ...], tuple[AuditLogRow, ...], float, str]:
    sample_by_runtime_id = {runtime_ids[sample.query_id]: sample for sample in samples}
    trace_by_query_id = oracle_traces_by_query_id or _oracle_trace_by_query_id(samples)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    audit_log, event_id, timestamp_us = _append_rt_audit(
        candidates,
        first_event_id=config.first_event_id,
        first_timestamp_us=config.first_timestamp_us,
    )
    certificates: list[CertificateResult] = []
    exact_by_runtime_id: dict[int, PureExactQueryResult] = {}

    start = time.perf_counter()
    for item in work_items:
        sample = sample_by_runtime_id.get(item.query_id)
        if sample is None:
            raise ValueError("exact work item references an unknown internal sample")
        parent = candidate_by_id[item.parent_candidate_id]
        audit_log.append(
            _audit_row(
                event_id=event_id,
                query_id=item.query_id,
                candidate_id=parent.candidate_id,
                work_item_id=item.work_item_id,
                stage=AuditStage.EXACT,
                action=EXACT_AUDIT_DEQUEUED,
                depth=item.depth,
                timestamp_us=timestamp_us,
            )
        )
        event_id += 1
        timestamp_us += 1

        trace = trace_by_query_id[sample.query_id]
        exact_result = PureExactQueryResult(
            query_id=sample.query_id,
            family="swept_sphere_proxy",
            predicted_collision=trace.collided,
            ground_truth_collision=trace.collided,
            status="collision" if trace.collided else "separation",
            toi_upper=trace.toi,
            safe_margin_lb=max(0.0, trace.safe_margin),
            exact_evals=max(1, int(math.ceil(trace.exact_cost))),
            max_depth=0,
        )
        exact_by_runtime_id[item.query_id] = exact_result
        certificate = _certificate_from_result(exact_result, item, config.exact)
        certificates.append(certificate)
        event_id, timestamp_us = _append_exact_audit(
            audit_log,
            event_id=event_id,
            timestamp_us=timestamp_us,
            candidate=parent,
            item=item,
            certificate=certificate,
        )
    exact_elapsed_ms = (time.perf_counter() - start) * 1000.0

    query_results: list[PureExactQueryResult] = []
    for sample in samples:
        runtime_id = runtime_ids[sample.query_id]
        if runtime_id in exact_by_runtime_id:
            query_results.append(exact_by_runtime_id[runtime_id])
        else:
            trace = trace_by_query_id[sample.query_id]
            query_results.append(
                PureExactQueryResult(
                    query_id=sample.query_id,
                    family="swept_sphere_proxy",
                    predicted_collision=False,
                    ground_truth_collision=trace.collided,
                    status="rt_candidate_separation",
                    toi_upper=1.0,
                    safe_margin_lb=max(0.0, trace.safe_margin),
                    exact_evals=0,
                    max_depth=0,
                )
            )
    return tuple(query_results), tuple(certificates), tuple(audit_log), exact_elapsed_ms, "swept_sphere_oracle_cpu"


def validate_rt_exact_coverage(
    candidates: Sequence[CandidateRecord],
    work_items: Sequence[ExactWorkItem],
    certificates: Sequence[CertificateResult],
) -> None:
    if len(candidates) != len(work_items):
        raise ValueError("RTExact must schedule exactly one exact work item per candidate")
    if len(work_items) != len(certificates):
        raise ValueError("RTExact must emit exactly one certificate per exact work item")

    candidate_ids: set[int] = set()
    for candidate in candidates:
        validate_candidate_record(candidate)
        if candidate.candidate_id in candidate_ids:
            raise ValueError("RTExact candidate queue contains duplicate candidate_id")
        candidate_ids.add(candidate.candidate_id)

    parent_candidate_ids: set[int] = set()
    work_item_ids: set[int] = set()
    for item in work_items:
        validate_exact_work_item(item)
        if item.parent_candidate_id not in candidate_ids:
            raise ValueError("RTExact work item references an unknown candidate")
        if item.parent_candidate_id in parent_candidate_ids:
            raise ValueError("RTExact work queue duplicates a parent candidate")
        parent_candidate_ids.add(item.parent_candidate_id)
        if item.work_item_id in work_item_ids:
            raise ValueError("RTExact work queue contains duplicate work_item_id")
        work_item_ids.add(item.work_item_id)

    certificate_work_item_ids: set[int] = set()
    for certificate in certificates:
        validate_certificate_result(certificate)
        if certificate.work_item_id not in work_item_ids:
            raise ValueError("RTExact certificate references an unknown work item")
        if certificate.work_item_id in certificate_work_item_ids:
            raise ValueError("RTExact certificates duplicate a work item")
        certificate_work_item_ids.add(certificate.work_item_id)

    if certificate_work_item_ids != work_item_ids:
        raise ValueError("RTExact work item disappeared without certificate coverage")


def _make_benchmark_row(
    query_results: Sequence[PureExactQueryResult],
    *,
    candidate_stats: RtCandidateStats,
    exact_elapsed_ms: float,
) -> BenchmarkRow:
    if not query_results:
        raise ValueError("RTExact requires at least one query")
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
        fallback_ratio=0.0,
        rt_ms=candidate_stats.timing.total_ms,
        proposal_ms=0.0,
        exact_ms=exact_elapsed_ms,
        total_ms=total_ms,
        qps=0.0 if total_ms <= 0.0 else 1000.0 * len(query_results) / total_ms,
    )
    return validate_benchmark_row(row)


def run_rt_exact_on_external_batch(
    batch: DatasetQueryBatch,
    config: RTExactConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> RTExactResult:
    if not batch.queries:
        raise ValueError("RTExact external batch requires at least one query")
    cfg = _validate_config(config or RTExactConfig())
    candidates, candidate_stats, runtime_ids = _make_external_candidates(
        batch,
        cfg,
        backend=backend,
    )
    family_by_runtime_query_id = {
        runtime_ids[query.query_id]: _family_mask_for_external(query)
        for query in batch.queries
    }
    work_items = schedule_exact_work_items_without_stpf(
        candidates,
        family_by_runtime_query_id=family_by_runtime_query_id,
        first_work_item_id=cfg.first_work_item_id,
    )
    query_results, certificates, audit_log, exact_elapsed_ms, exact_backend_name = _process_external_exact_work_queue(
        batch,
        candidates,
        work_items,
        runtime_ids,
        cfg,
    )
    validate_rt_exact_coverage(candidates, work_items, certificates)
    return RTExactResult(
        benchmark=_make_benchmark_row(
            query_results,
            candidate_stats=candidate_stats,
            exact_elapsed_ms=exact_elapsed_ms,
        ),
        query_results=query_results,
        candidates=candidates,
        work_items=work_items,
        certificates=certificates,
        audit_log=audit_log,
        candidate_stats=candidate_stats,
        exact_backend_name=exact_backend_name,
        source_name=batch.source_name,
        scene_name=batch.scene_name,
        batch_id=batch.batch_id,
    )


def run_rt_exact_on_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    config: RTExactConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> RTExactResult:
    if not samples:
        raise ValueError("RTExact internal sample run requires at least one sample")
    cfg = _validate_config(config or RTExactConfig())
    oracle_traces_by_query_id = _oracle_trace_by_query_id(samples)
    candidates, candidate_stats, runtime_ids = _make_internal_candidates(
        samples,
        cfg,
        backend=backend,
        oracle_traces_by_query_id=oracle_traces_by_query_id,
    )
    family_by_runtime_query_id = {
        runtime_ids[sample.query_id]: FEATURE_FAMILY_CONSERVATIVE
        for sample in samples
    }
    work_items = schedule_exact_work_items_without_stpf(
        candidates,
        family_by_runtime_query_id=family_by_runtime_query_id,
        first_work_item_id=cfg.first_work_item_id,
    )
    query_results, certificates, audit_log, exact_elapsed_ms, exact_backend_name = _process_internal_exact_work_queue(
        samples,
        candidates,
        work_items,
        runtime_ids,
        cfg,
        oracle_traces_by_query_id=oracle_traces_by_query_id,
    )
    validate_rt_exact_coverage(candidates, work_items, certificates)
    return RTExactResult(
        benchmark=_make_benchmark_row(
            query_results,
            candidate_stats=candidate_stats,
            exact_elapsed_ms=exact_elapsed_ms,
        ),
        query_results=query_results,
        candidates=candidates,
        work_items=work_items,
        certificates=certificates,
        audit_log=audit_log,
        candidate_stats=candidate_stats,
        exact_backend_name=exact_backend_name,
        source_name="internal_analytic_oracle",
        scene_name="programmatic_motion_disc_pairs",
        batch_id="internal_samples",
    )


def run_rt_exact_on_generated_dataset(
    dataset: GeneratedDataset,
    config: RTExactConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> RTExactResult:
    return run_rt_exact_on_internal_samples(dataset.samples, config, backend=backend)
