from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

from p2cccd.contracts import (
    BenchmarkRow,
    CandidateRecord,
    CertificateRefinementMode,
    CertificateResult,
    CertificateStatus,
    ExactWorkItem,
    ProxyType,
)
from p2cccd.validators import (
    validate_benchmark_row,
    validate_candidate_record,
    validate_certificate_result,
)

from .abc_mesh_exact_benchmark import (
    ABCMeshExactBenchmarkConfig,
    ABCMeshExactBenchmarkDataset,
    ABCMeshExactBenchmarkResult,
    ABCMeshExactMotionQuery,
    ABCMeshExactQueryResult,
    run_abc_mesh_exact_benchmark,
)
from .bvh_exact import Aabb, BroadPhaseBackend, BroadPhasePrimitive, CppOptixBroadPhaseBackend, CpuAabbBroadPhaseBackend
from .no_proposal import (
    NoProposalConfig,
    NoProposalStats,
    _make_benchmark_row as _make_no_proposal_benchmark_row,
    schedule_exact_work_items_no_proposal,
)
from .pure_exact_cpu import PureExactCPUConfig, PureExactQueryResult
from .rt_exact import (
    FEATURE_FAMILY_CONSERVATIVE,
    RAW_CANDIDATE_AABB_OVERLAP,
    RAW_CANDIDATE_VALID,
    RTExactConfig,
    RtCandidateStats,
    RtCandidateTiming,
    _make_benchmark_row as _make_rt_benchmark_row,
    schedule_exact_work_items_without_stpf,
)
from .rt_stpf_exact import (
    RTSTPFExactConfig,
    STPFScheduleStats,
    _make_benchmark_row as _make_stpf_benchmark_row,
    _run_stpf_predictions,
    proposal_feature_rows_from_rt_candidates,
    schedule_exact_work_items_with_stpf,
)


def _default_trained_checkpoint() -> str:
    return str(
        Path("src/outputs/stpf_training")
        / "abc_training_20260422_demo_main"
        / "model_state.pt"
    )


def _default_model_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass(frozen=True, slots=True)
class ABCMeshExactPaperBenchmarkConfig:
    exact_benchmark: ABCMeshExactBenchmarkConfig = field(
        default_factory=lambda: ABCMeshExactBenchmarkConfig(
            run_name="abc_mesh_exact_paper_benchmark_ground_truth_run_id"
        )
    )
    rt_backend_name: str = "optix_compatible"
    include_random_stpf: bool = True
    include_trained_stpf: bool = True
    trained_checkpoint_path: str = field(default_factory=_default_trained_checkpoint)
    model_device: str = field(default_factory=_default_model_device)
    benchmark_output_dir: str = "src/benchmark"
    run_name: str = "abc_mesh_exact_paper_benchmark_run_id"


@dataclass(frozen=True, slots=True)
class _MethodSummary:
    method: str
    query_count: int
    fn_count: int
    fp_count: int
    candidate_recall: float
    avg_candidates: float
    avg_exact_evals: float
    avg_subdivision_depth: float
    fallback_ratio: float
    rt_ms: float
    proposal_ms: float
    exact_ms: float
    total_ms: float
    qps: float
    rt_build_ms: float
    rt_update_ms: float
    rt_trace_ms: float
    candidate_count: int
    work_item_count: int
    certificate_count: int


@dataclass(frozen=True, slots=True)
class ABCMeshExactPaperMethodResult:
    method_name: str
    benchmark: BenchmarkRow
    query_results: tuple[PureExactQueryResult, ...]
    candidates: tuple[CandidateRecord, ...]
    work_items: tuple[ExactWorkItem, ...]
    certificates: tuple[CertificateResult, ...]
    candidate_stats: RtCandidateStats | None
    proposal_ms: float = 0.0
    schedule_stats: STPFScheduleStats | NoProposalStats | None = None


@dataclass(frozen=True, slots=True)
class ABCMeshExactPaperBenchmarkArtifacts:
    report_path: Path
    summary_json_path: Path


@dataclass(frozen=True, slots=True)
class ABCMeshExactPaperBenchmarkResult:
    config: ABCMeshExactPaperBenchmarkConfig
    ground_truth: ABCMeshExactBenchmarkResult
    pure_mesh_exact: ABCMeshExactPaperMethodResult
    bvh_exact: ABCMeshExactPaperMethodResult
    rt_exact: ABCMeshExactPaperMethodResult
    no_proposal: ABCMeshExactPaperMethodResult
    rtstpf_random: ABCMeshExactPaperMethodResult | None
    rtstpf_trained: ABCMeshExactPaperMethodResult | None
    artifacts: ABCMeshExactPaperBenchmarkArtifacts


def _cpp_mesh_module() -> Any:
    from .bvh_exact import _try_load_p2cccd_cpp

    cpp = _try_load_p2cccd_cpp()
    required = ("load_triangle_mesh", "validate_triangle_mesh", "center_mesh_at_aabb_center")
    if cpp is None or any(not hasattr(cpp, name) for name in required):
        raise RuntimeError("p2cccd_cpp mesh bindings are unavailable")
    return cpp


def _status_to_enum(status: str) -> CertificateStatus:
    if status == "collision":
        return CertificateStatus.COLLISION
    if status == "separation":
        return CertificateStatus.SEPARATION
    return CertificateStatus.UNDECIDED


def _certificate_from_ground_truth(
    query_result: ABCMeshExactQueryResult,
    work_item: ExactWorkItem,
    exact: PureExactCPUConfig,
) -> CertificateResult:
    status = _status_to_enum(query_result.status)
    return validate_certificate_result(
        CertificateResult(
            work_item_id=work_item.work_item_id,
            query_id=work_item.query_id,
            status=status,
            interval_t0=0.0,
            interval_t1=1.0,
            toi_upper=float(query_result.toi_upper),
            safe_margin_lb=float(query_result.safe_margin_lb),
            witness_family=int(query_result.witness_family),
            witness_id_a=int(query_result.witness_id_a),
            witness_id_b=int(query_result.witness_id_b),
            covered_feature_mask=FEATURE_FAMILY_CONSERVATIVE,
            eps_time=exact.eps_time,
            eps_space=exact.eps_space,
            reason_code=0 if status is not CertificateStatus.UNDECIDED else 2,
            next_refinement_mode=(
                CertificateRefinementMode.NONE
                if status is not CertificateStatus.UNDECIDED
                else CertificateRefinementMode.BISECT_INTERVAL
            ),
        )
    )


def _pure_result_from_ground_truth(
    query: ABCMeshExactMotionQuery,
    query_result: ABCMeshExactQueryResult,
    *,
    predicted_collision: bool,
    exact: PureExactCPUConfig,
) -> PureExactQueryResult:
    exact_evals = int(query_result.point_triangle_kept_pairs + query_result.edge_edge_kept_pairs)
    max_depth = exact.max_subdivision_depth if query_result.status == "undecided" else 0
    if query_result.status == "collision":
        ground_truth_collision: bool | None = True
    elif query_result.status == "separation":
        ground_truth_collision = False
    else:
        ground_truth_collision = None
    return PureExactQueryResult(
        query_id=query.query_id,
        family="mesh_mesh_exact",
        predicted_collision=bool(predicted_collision),
        ground_truth_collision=ground_truth_collision,
        status=query_result.status if predicted_collision else "rt_candidate_separation",
        toi_upper=float(query_result.toi_upper) if predicted_collision else 1.0,
        safe_margin_lb=float(query_result.safe_margin_lb) if predicted_collision else 0.0,
        exact_evals=exact_evals if predicted_collision else 0,
        max_depth=max_depth if predicted_collision else 0,
    )


def _summary_from_method(result: ABCMeshExactPaperMethodResult) -> _MethodSummary:
    candidate_stats = result.candidate_stats
    rt_build_ms = 0.0 if candidate_stats is None else candidate_stats.timing.build_ms
    rt_update_ms = 0.0 if candidate_stats is None else candidate_stats.timing.update_ms
    rt_trace_ms = 0.0 if candidate_stats is None else candidate_stats.timing.trace_ms
    return _MethodSummary(
        method=result.method_name,
        query_count=result.benchmark.query_count,
        fn_count=result.benchmark.fn_count,
        fp_count=result.benchmark.fp_count,
        candidate_recall=result.benchmark.candidate_recall,
        avg_candidates=result.benchmark.avg_candidates,
        avg_exact_evals=result.benchmark.avg_exact_evals,
        avg_subdivision_depth=result.benchmark.avg_subdivision_depth,
        fallback_ratio=result.benchmark.fallback_ratio,
        rt_ms=result.benchmark.rt_ms,
        proposal_ms=result.benchmark.proposal_ms,
        exact_ms=result.benchmark.exact_ms,
        total_ms=result.benchmark.total_ms,
        qps=result.benchmark.qps,
        rt_build_ms=rt_build_ms,
        rt_update_ms=rt_update_ms,
        rt_trace_ms=rt_trace_ms,
        candidate_count=len(result.candidates),
        work_item_count=len(result.work_items),
        certificate_count=len(result.certificates),
    )


def _mesh_cache_entry(cpp: Any, mesh_cache: dict[str, dict[str, object]], path: str) -> dict[str, object]:
    cached = mesh_cache.get(path)
    if cached is not None:
        return cached
    mesh = cpp.load_triangle_mesh(path)
    cpp.validate_triangle_mesh(mesh)
    centered_mesh, _center = cpp.center_mesh_at_aabb_center(mesh)
    vertices = tuple(tuple(float(value) for value in vertex) for vertex in centered_mesh.vertices_ref)
    if not vertices:
        raise ValueError(f"mesh {path} is empty after centering")
    mins = tuple(min(vertex[axis] for vertex in vertices) for axis in range(3))
    maxs = tuple(max(vertex[axis] for vertex in vertices) for axis in range(3))
    extents = tuple(maxs[axis] - mins[axis] for axis in range(3))
    diagonal = math.sqrt(sum(component * component for component in extents))
    cached = {
        "mesh": centered_mesh,
        "aabb": Aabb(min=mins, max=maxs),
        "diagonal": diagonal,
    }
    mesh_cache[path] = cached
    return cached


def _swept_translation_aabb(base_aabb: Aabb, t0: tuple[float, float, float], t1: tuple[float, float, float]) -> Aabb:
    return Aabb(
        min=tuple(
            min(base_aabb.min[axis] + t0[axis], base_aabb.min[axis] + t1[axis])
            for axis in range(3)
        ),
        max=tuple(
            max(base_aabb.max[axis] + t0[axis], base_aabb.max[axis] + t1[axis])
            for axis in range(3)
        ),
    )


def _translation_length(t0: tuple[float, float, float], t1: tuple[float, float, float]) -> float:
    return math.sqrt(sum((t1[axis] - t0[axis]) * (t1[axis] - t0[axis]) for axis in range(3)))


def _make_query_primitives(
    dataset: ABCMeshExactBenchmarkDataset,
    mesh_cache: dict[str, dict[str, object]],
) -> tuple[tuple[BroadPhasePrimitive, ...], dict[int, tuple[float, float, float, float]]]:
    cpp = _cpp_mesh_module()
    primitives: list[BroadPhasePrimitive] = []
    motion_bounds: dict[int, tuple[float, float, float, float]] = {}
    for index, query in enumerate(dataset.queries):
        mesh_a = _mesh_cache_entry(cpp, mesh_cache, query.mesh_a_path)
        mesh_b = _mesh_cache_entry(cpp, mesh_cache, query.mesh_b_path)
        swept_a = _swept_translation_aabb(mesh_a["aabb"], query.translation_a_t0, query.translation_a_t1)
        swept_b = _swept_translation_aabb(mesh_b["aabb"], query.translation_b_t0, query.translation_b_t1)
        primitive_base = index * 2
        primitives.append(
            BroadPhasePrimitive(
                primitive_id=primitive_base + 1,
                query_id=query.query_id,
                role="a",
                aabb=swept_a,
                family="mesh_mesh_exact",
            )
        )
        primitives.append(
            BroadPhasePrimitive(
                primitive_id=primitive_base + 2,
                query_id=query.query_id,
                role="b",
                aabb=swept_b,
                family="mesh_mesh_exact",
            )
        )
        motion_bounds[query.query_id] = (
            _translation_length(query.translation_a_t0, query.translation_a_t1),
            _translation_length(query.translation_b_t0, query.translation_b_t1),
            float(mesh_a["diagonal"]),
            float(mesh_b["diagonal"]),
        )
    return tuple(primitives), motion_bounds


def _make_candidate_stats(
    *,
    build_ms: float,
    backend_name: str,
    primitive_count: int,
    raw_hit_count: int,
    compact_candidate_count: int,
    broad_elapsed_ms: float,
    broad_build_ms: float = 0.0,
    broad_update_ms: float = 0.0,
    broad_trace_ms: float = 0.0,
    compact_ms: float = 0.0,
    candidate_recall: float,
) -> RtCandidateStats:
    timing = RtCandidateTiming(
        build_ms=float(build_ms) + float(broad_build_ms),
        update_ms=float(broad_update_ms),
        trace_ms=float(broad_trace_ms if broad_trace_ms > 0.0 else broad_elapsed_ms),
        compact_ms=float(compact_ms),
        total_ms=float(build_ms)
        + float(broad_build_ms)
        + float(broad_update_ms)
        + float(broad_trace_ms if broad_trace_ms > 0.0 else broad_elapsed_ms)
        + float(compact_ms),
    )
    return RtCandidateStats(
        backend_name=backend_name,
        primitive_count=int(primitive_count),
        raw_hit_count=int(raw_hit_count),
        compact_candidate_count=int(compact_candidate_count),
        candidate_recall=float(candidate_recall),
        timing=timing,
    )


def _candidate_recall(
    dataset: ABCMeshExactBenchmarkDataset,
    exact_rows_by_query_id: dict[int, ABCMeshExactQueryResult],
    active_query_ids: set[int],
) -> float:
    positives = {
        query.query_id
        for query in dataset.queries
        if exact_rows_by_query_id[query.query_id].status == "collision"
    }
    if not positives:
        return 1.0
    return len(positives & active_query_ids) / float(len(positives))


def _make_mesh_candidates(
    dataset: ABCMeshExactBenchmarkDataset,
    exact_rows_by_query_id: dict[int, ABCMeshExactQueryResult],
    *,
    backend: BroadPhaseBackend,
    mesh_cache: dict[str, dict[str, object]],
) -> tuple[tuple[CandidateRecord, ...], RtCandidateStats]:
    build_start = time.perf_counter()
    primitives, motion_bounds = _make_query_primitives(dataset, mesh_cache)
    build_ms = (time.perf_counter() - build_start) * 1000.0
    pairs, broad_stats = backend.find_pairs(primitives, same_query_only=True)
    query_by_id = {query.query_id: query for query in dataset.queries}

    compact_start = time.perf_counter()
    candidates: list[CandidateRecord] = []
    for ordinal, pair in enumerate(pairs):
        query = query_by_id[pair.query_id]
        bounds = motion_bounds[query.query_id]
        candidates.append(
            validate_candidate_record(
                CandidateRecord(
                    candidate_id=(int(query.query_id) * 1_000_003) + ordinal + 1,
                    query_id=int(query.query_id),
                    slab_id=int(query.slab_id),
                    object_a_id=1,
                    patch_a_id=int(query.patch_a_id),
                    object_b_id=2,
                    patch_b_id=int(query.patch_b_id),
                    proxy_type_a=ProxyType.SWEPT_AABB,
                    proxy_type_b=ProxyType.SWEPT_AABB,
                    rt_hit_count=1,
                    motion_bound=[float(value) for value in bounds],
                    proxy_features_offset=0,
                    flags=RAW_CANDIDATE_VALID | RAW_CANDIDATE_AABB_OVERLAP,
                )
            )
        )
    compact_ms = (time.perf_counter() - compact_start) * 1000.0
    active_query_ids = {pair.query_id for pair in pairs}
    stats = _make_candidate_stats(
        build_ms=build_ms,
        backend_name=broad_stats.backend_name,
        primitive_count=len(primitives),
        raw_hit_count=len(pairs),
        compact_candidate_count=len(candidates),
        broad_elapsed_ms=float(broad_stats.elapsed_ms),
        broad_build_ms=float(getattr(broad_stats, "build_ms", 0.0)),
        broad_update_ms=float(getattr(broad_stats, "update_ms", 0.0)),
        broad_trace_ms=float(getattr(broad_stats, "trace_ms", 0.0)),
        compact_ms=compact_ms,
        candidate_recall=_candidate_recall(dataset, exact_rows_by_query_id, active_query_ids),
    )
    return tuple(candidates), stats


def _evaluate_exact_work_queue_from_ground_truth(
    dataset: ABCMeshExactBenchmarkDataset,
    exact_rows_by_query_id: dict[int, ABCMeshExactQueryResult],
    candidates: Sequence[CandidateRecord],
    work_items: Sequence[ExactWorkItem],
    exact: PureExactCPUConfig,
) -> tuple[tuple[PureExactQueryResult, ...], tuple[CertificateResult, ...], float]:
    query_by_id = {query.query_id: query for query in dataset.queries}
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    exact_by_query_id: dict[int, PureExactQueryResult] = {}
    certificates: list[CertificateResult] = []
    exact_elapsed_ms = 0.0
    for item in work_items:
        _ = candidate_by_id[item.parent_candidate_id]
        exact_row = exact_rows_by_query_id[item.query_id]
        exact_elapsed_ms += float(exact_row.total_ms)
        exact_by_query_id[item.query_id] = _pure_result_from_ground_truth(
            query_by_id[item.query_id],
            exact_row,
            predicted_collision=True,
            exact=exact,
        )
        certificates.append(_certificate_from_ground_truth(exact_row, item, exact))

    query_results: list[PureExactQueryResult] = []
    for query in dataset.queries:
        exact_row = exact_rows_by_query_id[query.query_id]
        if query.query_id in exact_by_query_id:
            query_results.append(exact_by_query_id[query.query_id])
        else:
            query_results.append(
                _pure_result_from_ground_truth(
                    query,
                    exact_row,
                    predicted_collision=False,
                    exact=exact,
                )
            )
    return tuple(query_results), tuple(certificates), exact_elapsed_ms


def _make_pure_mesh_exact_method(ground_truth: ABCMeshExactBenchmarkResult) -> ABCMeshExactPaperMethodResult:
    total_exact_ms = sum(float(row.total_ms) for row in ground_truth.query_results)
    query_results = tuple(
        PureExactQueryResult(
            query_id=row.query_id,
            family="mesh_mesh_exact",
            predicted_collision=bool(row.predicted_collision),
            ground_truth_collision=(
                True if row.status == "collision" else False if row.status == "separation" else None
            ),
            status=row.status,
            toi_upper=float(row.toi_upper),
            safe_margin_lb=float(row.safe_margin_lb),
            exact_evals=int(row.point_triangle_kept_pairs + row.edge_edge_kept_pairs),
            max_depth=ground_truth.config.exact.max_subdivision_depth if row.status == "undecided" else 0,
        )
        for row in ground_truth.query_results
    )
    benchmark = validate_benchmark_row(
        BenchmarkRow(
            query_count=ground_truth.benchmark.query_count,
            fn_count=0,
            fp_count=0,
            candidate_recall=1.0,
            avg_candidates=0.0,
            avg_exact_evals=sum(result.exact_evals for result in query_results) / max(1, len(query_results)),
            avg_subdivision_depth=sum(result.max_depth for result in query_results) / max(1, len(query_results)),
            fallback_ratio=0.0,
            rt_ms=0.0,
            proposal_ms=0.0,
            exact_ms=total_exact_ms,
            total_ms=total_exact_ms,
            qps=0.0 if total_exact_ms <= 0.0 else 1000.0 * len(query_results) / total_exact_ms,
        )
    )
    return ABCMeshExactPaperMethodResult(
        method_name="PureMeshExactCPU",
        benchmark=benchmark,
        query_results=query_results,
        candidates=(),
        work_items=(),
        certificates=(),
        candidate_stats=None,
    )


def _run_rt_like_method(
    method_name: str,
    dataset: ABCMeshExactBenchmarkDataset,
    exact_rows_by_query_id: dict[int, ABCMeshExactQueryResult],
    *,
    backend: BroadPhaseBackend,
    config: RTExactConfig,
    mesh_cache: dict[str, dict[str, object]],
) -> ABCMeshExactPaperMethodResult:
    candidates, candidate_stats = _make_mesh_candidates(
        dataset,
        exact_rows_by_query_id,
        backend=backend,
        mesh_cache=mesh_cache,
    )
    family_by_query_id = {query.query_id: FEATURE_FAMILY_CONSERVATIVE for query in dataset.queries}
    work_items = schedule_exact_work_items_without_stpf(
        candidates,
        family_by_runtime_query_id=family_by_query_id,
        first_work_item_id=config.first_work_item_id,
    )
    query_results, certificates, exact_elapsed_ms = _evaluate_exact_work_queue_from_ground_truth(
        dataset,
        exact_rows_by_query_id,
        candidates,
        work_items,
        config.exact,
    )
    benchmark = _make_rt_benchmark_row(
        query_results,
        candidate_stats=candidate_stats,
        exact_elapsed_ms=exact_elapsed_ms,
    )
    return ABCMeshExactPaperMethodResult(
        method_name=method_name,
        benchmark=benchmark,
        query_results=query_results,
        candidates=candidates,
        work_items=work_items,
        certificates=certificates,
        candidate_stats=candidate_stats,
    )


def _run_no_proposal_method(
    dataset: ABCMeshExactBenchmarkDataset,
    exact_rows_by_query_id: dict[int, ABCMeshExactQueryResult],
    *,
    backend: BroadPhaseBackend,
    config: NoProposalConfig,
    mesh_cache: dict[str, dict[str, object]],
) -> ABCMeshExactPaperMethodResult:
    candidates, candidate_stats = _make_mesh_candidates(
        dataset,
        exact_rows_by_query_id,
        backend=backend,
        mesh_cache=mesh_cache,
    )
    family_by_query_id = {query.query_id: FEATURE_FAMILY_CONSERVATIVE for query in dataset.queries}
    work_items, no_proposal_stats = schedule_exact_work_items_no_proposal(
        candidates,
        family_by_runtime_query_id=family_by_query_id,
        config=config,
    )
    query_results, certificates, exact_elapsed_ms = _evaluate_exact_work_queue_from_ground_truth(
        dataset,
        exact_rows_by_query_id,
        candidates,
        work_items,
        config.exact,
    )
    benchmark = _make_no_proposal_benchmark_row(
        query_results,
        candidate_stats=candidate_stats,
        no_proposal_stats=no_proposal_stats,
        exact_elapsed_ms=exact_elapsed_ms,
    )
    return ABCMeshExactPaperMethodResult(
        method_name="NoProposal",
        benchmark=benchmark,
        query_results=query_results,
        candidates=candidates,
        work_items=work_items,
        certificates=certificates,
        candidate_stats=candidate_stats,
        schedule_stats=no_proposal_stats,
    )


def _run_stpf_method(
    method_name: str,
    dataset: ABCMeshExactBenchmarkDataset,
    exact_rows_by_query_id: dict[int, ABCMeshExactQueryResult],
    *,
    backend: BroadPhaseBackend,
    config: RTSTPFExactConfig,
    mesh_cache: dict[str, dict[str, object]],
) -> ABCMeshExactPaperMethodResult:
    candidates, candidate_stats = _make_mesh_candidates(
        dataset,
        exact_rows_by_query_id,
        backend=backend,
        mesh_cache=mesh_cache,
    )
    family_by_query_id = {query.query_id: FEATURE_FAMILY_CONSERVATIVE for query in dataset.queries}
    proposal_start = time.perf_counter()
    feature_rows = proposal_feature_rows_from_rt_candidates(
        candidates,
        family_by_runtime_query_id=family_by_query_id,
        candidate_stats=candidate_stats,
    )
    predictions = _run_stpf_predictions(
        feature_rows,
        config,
        runtime=None,
        device=config.model_device,
    )
    work_items, schedule_stats = schedule_exact_work_items_with_stpf(
        candidates,
        feature_rows,
        predictions,
        family_by_runtime_query_id=family_by_query_id,
        config=config,
    )
    proposal_ms = (time.perf_counter() - proposal_start) * 1000.0
    query_results, certificates, exact_elapsed_ms = _evaluate_exact_work_queue_from_ground_truth(
        dataset,
        exact_rows_by_query_id,
        candidates,
        work_items,
        config.exact,
    )
    benchmark = _make_stpf_benchmark_row(
        query_results,
        candidate_stats=candidate_stats,
        schedule_stats=schedule_stats,
        proposal_elapsed_ms=proposal_ms,
        exact_elapsed_ms=exact_elapsed_ms,
    )
    return ABCMeshExactPaperMethodResult(
        method_name=method_name,
        benchmark=benchmark,
        query_results=query_results,
        candidates=candidates,
        work_items=work_items,
        certificates=certificates,
        candidate_stats=candidate_stats,
        proposal_ms=proposal_ms,
        schedule_stats=schedule_stats,
    )


def _report_markdown(result: ABCMeshExactPaperBenchmarkResult) -> str:
    summaries = [
        _summary_from_method(result.pure_mesh_exact),
        _summary_from_method(result.bvh_exact),
        _summary_from_method(result.rt_exact),
        _summary_from_method(result.no_proposal),
    ]
    if result.rtstpf_random is not None:
        summaries.append(_summary_from_method(result.rtstpf_random))
    if result.rtstpf_trained is not None:
        summaries.append(_summary_from_method(result.rtstpf_trained))

    lines = [
        "# ABC Real Mesh-Mesh Exact CCD Benchmark: paper-path comparison",
        "",
        "## Summary",
        f"- Run name: `{result.config.run_name}`",
        f"- Ground-truth run: `{result.ground_truth.config.run_name}`",
        f"- Source root: `{result.ground_truth.dataset.source_root}`",
        f"- Query count: `{len(result.ground_truth.dataset.queries)}`",
        f"- Pair count: `{len(result.ground_truth.dataset.pair_ids)}`",
        f"- Used demo subset: `{result.ground_truth.dataset.used_demo_subset}`",
        f"- RT backend request: `{result.config.rt_backend_name}`",
        "",
        "## method comparison",
        "| Method | FN | Recall | Candidates | Work Items | rt ms | proposal ms | exact ms | total ms | qps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.method} | {summary.fn_count} | {summary.candidate_recall:.4f} | "
            f"{summary.candidate_count} | {summary.work_item_count} | {summary.rt_ms:.4f} | "
            f"{summary.proposal_ms:.4f} | {summary.exact_ms:.4f} | {summary.total_ms:.4f} | {summary.qps:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Ground Truth Protocol",
            f"- Ground-truth conservative-positive queries: `{sum(1 for row in result.ground_truth.query_results if row.predicted_collision)}`",
            f"- Ground-truth exact collision certificates: `{sum(1 for row in result.ground_truth.query_results if row.status == 'collision')}`",
            f"- Ground-truth undecided certificates: `{sum(1 for row in result.ground_truth.query_results if row.status == 'undecided')}`",
            "",
        "## Notes",
        "- thisdescriptionbased onreal mesh-mesh exact benchmark, same batch query  ground truth description `PureMeshExactCPU` Pathdescriptionafterdescriptionuse. ",
        "- descriptionreportdescription `FN` and `Recall` description `exact collision certificates` statistics; `undecided` descriptionentersplitdescription, descriptionasdescription. ",
        "- `BVHExact / RTExact / NoProposal / RTSTPFExact` currentcompareis query-level whole-mesh swept-AABB candidate generation, descriptionenterreal mesh exact certificate. ",
        "- descriptionis not full patch-level RT candidate path; thereforein candidate inflation descriptionsceneunder, proposal descriptionoverhead. ",
        "- `RTSTPFExact-Trained` ifdescriptionuse, defaultdescriptionusedescriptionhas ABC proxy-CAD checkpoint, descriptionas cross-domain transfer, rather than same-distribution mesh-exact description. ",
        ]
    )
    return "\n".join(lines) + "\n"


def run_abc_mesh_exact_paper_benchmark(
    config: ABCMeshExactPaperBenchmarkConfig | None = None,
) -> ABCMeshExactPaperBenchmarkResult:
    cfg = config or ABCMeshExactPaperBenchmarkConfig()
    ground_truth = run_abc_mesh_exact_benchmark(cfg.exact_benchmark)
    exact_rows_by_query_id = {row.query_id: row for row in ground_truth.query_results}
    mesh_cache: dict[str, dict[str, object]] = {}

    pure_mesh_exact = _make_pure_mesh_exact_method(ground_truth)
    bvh_exact = _run_rt_like_method(
        "BVHExact",
        ground_truth.dataset,
        exact_rows_by_query_id,
        backend=CpuAabbBroadPhaseBackend(name="mesh_cpu_aabb_bvh"),
        config=RTExactConfig(exact=cfg.exact_benchmark.exact, backend_name="cpu_reference_rt"),
        mesh_cache=mesh_cache,
    )
    if cfg.rt_backend_name in {"optix_rt", "optix_compatible"}:
        rt_backend: BroadPhaseBackend = CppOptixBroadPhaseBackend(
            name="optix_rt",
            allow_cpu_fallback=cfg.rt_backend_name == "optix_compatible",
        )
    else:
        rt_backend = CpuAabbBroadPhaseBackend(name=cfg.rt_backend_name)
    rt_exact = _run_rt_like_method(
        "RTExact",
        ground_truth.dataset,
        exact_rows_by_query_id,
        backend=rt_backend,
        config=RTExactConfig(exact=cfg.exact_benchmark.exact, backend_name=cfg.rt_backend_name),
        mesh_cache=mesh_cache,
    )
    no_proposal = _run_no_proposal_method(
        ground_truth.dataset,
        exact_rows_by_query_id,
        backend=rt_backend,
        config=NoProposalConfig(exact=cfg.exact_benchmark.exact, rt_backend_name=cfg.rt_backend_name),
        mesh_cache=mesh_cache,
    )

    rtstpf_random = None
    if cfg.include_random_stpf:
        rtstpf_random = _run_stpf_method(
            "RTSTPFExact-Random",
            ground_truth.dataset,
            exact_rows_by_query_id,
            backend=rt_backend,
            config=RTSTPFExactConfig(
                exact=cfg.exact_benchmark.exact,
                rt_backend_name=cfg.rt_backend_name,
                use_dummy_policy=False,
                allow_default_model=True,
                model_checkpoint_path=None,
                model_device=cfg.model_device,
            ),
            mesh_cache=mesh_cache,
        )

    rtstpf_trained = None
    checkpoint_path = Path(cfg.trained_checkpoint_path)
    if cfg.include_trained_stpf and checkpoint_path.exists():
        rtstpf_trained = _run_stpf_method(
            "RTSTPFExact-Trained",
            ground_truth.dataset,
            exact_rows_by_query_id,
            backend=rt_backend,
            config=RTSTPFExactConfig(
                exact=cfg.exact_benchmark.exact,
                rt_backend_name=cfg.rt_backend_name,
                use_dummy_policy=False,
                allow_default_model=False,
                model_checkpoint_path=str(checkpoint_path),
                model_device=cfg.model_device,
            ),
            mesh_cache=mesh_cache,
        )

    artifacts = ABCMeshExactPaperBenchmarkArtifacts(
        report_path=Path(cfg.benchmark_output_dir) / f"{cfg.run_name}.md",
        summary_json_path=Path(cfg.benchmark_output_dir) / f"{cfg.run_name}.json",
    )
    result = ABCMeshExactPaperBenchmarkResult(
        config=cfg,
        ground_truth=ground_truth,
        pure_mesh_exact=pure_mesh_exact,
        bvh_exact=bvh_exact,
        rt_exact=rt_exact,
        no_proposal=no_proposal,
        rtstpf_random=rtstpf_random,
        rtstpf_trained=rtstpf_trained,
        artifacts=artifacts,
    )

    summaries = [
        asdict(_summary_from_method(pure_mesh_exact)),
        asdict(_summary_from_method(bvh_exact)),
        asdict(_summary_from_method(rt_exact)),
        asdict(_summary_from_method(no_proposal)),
    ]
    if rtstpf_random is not None:
        summaries.append(asdict(_summary_from_method(rtstpf_random)))
    if rtstpf_trained is not None:
        summaries.append(asdict(_summary_from_method(rtstpf_trained)))

    artifacts.report_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.report_path.write_text(_report_markdown(result), encoding="utf-8")
    artifacts.summary_json_path.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "ground_truth_report": str(ground_truth.artifacts.report_path),
                "dataset": {
                    "source_root": str(ground_truth.dataset.source_root),
                    "query_count": len(ground_truth.dataset.queries),
                    "pair_count": len(ground_truth.dataset.pair_ids),
                    "manifest_path": str(ground_truth.dataset.manifest_path),
                    "queries_jsonl_path": str(ground_truth.dataset.queries_jsonl_path),
                },
                "method_summaries": summaries,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "ABCMeshExactPaperBenchmarkArtifacts",
    "ABCMeshExactPaperBenchmarkConfig",
    "ABCMeshExactPaperBenchmarkResult",
    "ABCMeshExactPaperMethodResult",
    "run_abc_mesh_exact_paper_benchmark",
]
