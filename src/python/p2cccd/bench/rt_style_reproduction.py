from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

from p2cccd.contracts import BenchmarkRow, ProxyType
from p2cccd.data.dataset import GeneratedDataset
from p2cccd.data.oracle import ExactOracleTrace, evaluate_swept_sphere_oracle
from p2cccd.data.samplers import MotionDiscPairSample
from p2cccd.validators import validate_benchmark_row

from .bvh_exact import (
    Aabb,
    BroadPhaseBackend,
    BroadPhasePrimitive,
    CpuAabbBroadPhaseBackend,
)
from .pure_exact_cpu import PureExactQueryResult
from .rt_exact import RtCandidateStats, RtCandidateTiming


Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class RTDCDStyleConfig:
    sample_count: int = 2
    radius_scale: float = 1.0
    backend_name: str = "cpu_rt_dcd_style"
    same_query_only: bool = True


@dataclass(frozen=True, slots=True)
class RTCCDStyleConfig:
    slab_count: int = 4
    radius_scale: float = 1.0
    proxy_type_a: ProxyType = ProxyType.SWEPT_AABB
    proxy_type_b: ProxyType = ProxyType.SWEPT_AABB
    backend_name: str = "cpu_rt_ccd_style"
    same_query_only: bool = True


@dataclass(frozen=True, slots=True)
class RTStyleReproductionResult:
    benchmark: BenchmarkRow
    query_results: tuple[PureExactQueryResult, ...]
    candidate_stats: RtCandidateStats
    style_name: str
    source_name: str
    scene_name: str
    batch_id: str

    @property
    def final_fn_zero(self) -> bool:
        return self.benchmark.fn_count == 0


def validate_rt_dcd_style_config(config: RTDCDStyleConfig) -> RTDCDStyleConfig:
    if config.sample_count < 2:
        raise ValueError("RTDCDStyleConfig.sample_count must be at least 2")
    if not math.isfinite(config.radius_scale) or config.radius_scale <= 0.0:
        raise ValueError("RTDCDStyleConfig.radius_scale must be finite and positive")
    if not config.same_query_only:
        raise ValueError("RTDCDStyle currently requires same_query_only=True")
    return config


def validate_rt_ccd_style_config(config: RTCCDStyleConfig) -> RTCCDStyleConfig:
    if config.slab_count <= 0:
        raise ValueError("RTCCDStyleConfig.slab_count must be positive")
    if not math.isfinite(config.radius_scale) or config.radius_scale <= 0.0:
        raise ValueError("RTCCDStyleConfig.radius_scale must be finite and positive")
    if config.proxy_type_a is ProxyType.UNKNOWN or config.proxy_type_b is ProxyType.UNKNOWN:
        raise ValueError("RTCCDStyleConfig proxy types must not be UNKNOWN")
    if not config.same_query_only:
        raise ValueError("RTCCDStyle currently requires same_query_only=True")
    return config


def _lerp(lhs: Vec3, rhs: Vec3, t: float) -> Vec3:
    return (
        lhs[0] + (rhs[0] - lhs[0]) * t,
        lhs[1] + (rhs[1] - lhs[1]) * t,
        lhs[2] + (rhs[2] - lhs[2]) * t,
    )


def _aabb_from_points(points: Sequence[Vec3], *, inflation: float) -> Aabb:
    if not points:
        raise ValueError("cannot build RT-style AABB from an empty point set")
    if inflation < 0.0:
        raise ValueError("RT-style AABB inflation must be non-negative")
    mins = tuple(min(point[axis] for point in points) - inflation for axis in range(3))
    maxs = tuple(max(point[axis] for point in points) + inflation for axis in range(3))
    return Aabb(min=mins, max=maxs)  # type: ignore[arg-type]


def _sample_times(sample_count: int) -> tuple[float, ...]:
    if sample_count == 1:
        return (0.0,)
    return tuple(index / float(sample_count - 1) for index in range(sample_count))


def _dcd_primitives_for_sample(
    sample: MotionDiscPairSample,
    config: RTDCDStyleConfig,
) -> tuple[BroadPhasePrimitive, ...]:
    primitives: list[BroadPhasePrimitive] = []
    for sample_index, sample_time in enumerate(_sample_times(config.sample_count)):
        center_a = _lerp(sample.center_a_t0, sample.center_a_t1, sample_time)
        center_b = _lerp(sample.center_b_t0, sample.center_b_t1, sample_time)
        for role, center, radius, object_id, patch_id, role_offset in (
            ("a", center_a, sample.radius_a, sample.object_a_id, sample.patch_a_id, 0),
            ("b", center_b, sample.radius_b, sample.object_b_id, sample.patch_b_id, 100000),
        ):
            primitive_id = sample.query_id * 1_000_000 + sample_index * 1000 + role_offset
            primitives.append(
                BroadPhasePrimitive(
                    primitive_id=primitive_id,
                    query_id=sample.query_id,
                    role=role,
                    aabb=_aabb_from_points((center,), inflation=radius * config.radius_scale),
                    family=f"rt_dcd_sample_{sample_index}",
                    metadata={
                        "sample_id": sample.sample_id,
                        "time_sample_index": sample_index,
                        "time": sample_time,
                        "object_id": object_id,
                        "patch_id": patch_id,
                        "style": "RTDCDStyle",
                    },
                )
            )
    return tuple(primitives)


def _slab_interval(slab_id: int, slab_count: int) -> tuple[float, float]:
    return slab_id / float(slab_count), (slab_id + 1) / float(slab_count)


def _ccd_primitives_for_sample(
    sample: MotionDiscPairSample,
    config: RTCCDStyleConfig,
) -> tuple[BroadPhasePrimitive, ...]:
    primitives: list[BroadPhasePrimitive] = []
    for slab_id in range(config.slab_count):
        slab_t0, slab_t1 = _slab_interval(slab_id, config.slab_count)
        for role, full_t0, full_t1, radius, object_id, patch_id, proxy_type, role_offset in (
            (
                "a",
                sample.center_a_t0,
                sample.center_a_t1,
                sample.radius_a,
                sample.object_a_id,
                sample.patch_a_id,
                config.proxy_type_a,
                0,
            ),
            (
                "b",
                sample.center_b_t0,
                sample.center_b_t1,
                sample.radius_b,
                sample.object_b_id,
                sample.patch_b_id,
                config.proxy_type_b,
                100000,
            ),
        ):
            center_t0 = _lerp(full_t0, full_t1, slab_t0)
            center_t1 = _lerp(full_t0, full_t1, slab_t1)
            primitive_id = sample.query_id * 1_000_000 + slab_id * 1000 + role_offset
            primitives.append(
                BroadPhasePrimitive(
                    primitive_id=primitive_id,
                    query_id=sample.query_id,
                    role=role,
                    aabb=_aabb_from_points((center_t0, center_t1), inflation=radius * config.radius_scale),
                    family=f"rt_ccd_slab_{slab_id}",
                    metadata={
                        "sample_id": sample.sample_id,
                        "slab_id": slab_id,
                        "slab_t0": slab_t0,
                        "slab_t1": slab_t1,
                        "object_id": object_id,
                        "patch_id": patch_id,
                        "proxy_type": proxy_type.name,
                        "style": "RTCCDStyle",
                    },
                )
            )
    return tuple(primitives)


def _positive_slab_ids(trace: ExactOracleTrace, slab_count: int) -> set[int]:
    if not trace.collided:
        return set()
    t0 = max(0.0, min(1.0, trace.contact_interval_t0))
    t1 = max(0.0, min(1.0, trace.contact_interval_t1))
    if t1 < t0:
        t0, t1 = t1, t0
    first = min(slab_count - 1, max(0, int(t0 * slab_count)))
    last_time = max(0.0, min(1.0 - 1.0e-12, t1))
    last = min(slab_count - 1, max(0, int(last_time * slab_count)))
    return set(range(first, last + 1))


def _make_query_result(
    sample: MotionDiscPairSample,
    trace: ExactOracleTrace,
    *,
    active: bool,
    status_prefix: str,
) -> PureExactQueryResult:
    predicted_collision = bool(active and trace.collided)
    exact_evals = max(1, int(math.ceil(trace.exact_cost))) if active else 0
    if predicted_collision:
        status = "collision"
    elif active:
        status = "separation"
    else:
        status = f"{status_prefix}_no_candidate"
    return PureExactQueryResult(
        query_id=sample.query_id,
        family="swept_sphere_proxy",
        predicted_collision=predicted_collision,
        ground_truth_collision=trace.collided,
        status=status,
        toi_upper=trace.toi if predicted_collision else 1.0,
        safe_margin_lb=max(0.0, trace.safe_margin),
        exact_evals=exact_evals,
        max_depth=0,
    )


def _make_benchmark_row(
    query_results: Sequence[PureExactQueryResult],
    *,
    candidate_count: int,
    rt_ms: float,
    exact_ms: float,
) -> BenchmarkRow:
    query_count = len(query_results)
    if query_count <= 0:
        raise ValueError("RT-style reproduction requires at least one query")
    fn_count = sum(
        1
        for result in query_results
        if result.ground_truth_collision is True and not result.predicted_collision
    )
    fp_count = sum(
        1
        for result in query_results
        if result.ground_truth_collision is False and result.predicted_collision
    )
    positive_count = sum(1 for result in query_results if result.ground_truth_collision is True)
    covered_positive_count = positive_count - fn_count
    total_exact_evals = sum(result.exact_evals for result in query_results)
    total_ms = rt_ms + exact_ms
    row = BenchmarkRow(
        query_count=query_count,
        fn_count=fn_count,
        fp_count=fp_count,
        candidate_recall=1.0 if positive_count == 0 else covered_positive_count / float(positive_count),
        avg_candidates=candidate_count / float(query_count),
        avg_exact_evals=total_exact_evals / float(query_count),
        avg_subdivision_depth=0.0,
        fallback_ratio=0.0,
        rt_ms=rt_ms,
        proposal_ms=0.0,
        exact_ms=exact_ms,
        total_ms=total_ms,
        qps=0.0 if total_ms <= 0.0 else 1000.0 * query_count / total_ms,
    )
    return validate_benchmark_row(row)


def _make_candidate_stats(
    *,
    backend_name: str,
    primitive_count: int,
    raw_hit_count: int,
    compact_candidate_count: int,
    candidate_recall: float,
    build_ms: float,
    trace_ms: float,
    compact_ms: float,
) -> RtCandidateStats:
    timing = RtCandidateTiming(
        build_ms=build_ms,
        update_ms=0.0,
        trace_ms=trace_ms,
        compact_ms=compact_ms,
        total_ms=build_ms + trace_ms + compact_ms,
    )
    return RtCandidateStats(
        backend_name=backend_name,
        primitive_count=primitive_count,
        raw_hit_count=raw_hit_count,
        compact_candidate_count=compact_candidate_count,
        candidate_recall=candidate_recall,
        timing=timing,
    )


def run_rt_dcd_style_on_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    config: RTDCDStyleConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> RTStyleReproductionResult:
    if not samples:
        raise ValueError("RTDCDStyle requires at least one sample")
    cfg = validate_rt_dcd_style_config(config or RTDCDStyleConfig())
    broad_phase = backend or CpuAabbBroadPhaseBackend(name=cfg.backend_name)

    build_start = time.perf_counter()
    primitives: list[BroadPhasePrimitive] = []
    for sample in samples:
        primitives.extend(_dcd_primitives_for_sample(sample, cfg))
    build_ms = (time.perf_counter() - build_start) * 1000.0

    pairs, stats = broad_phase.find_pairs(primitives, same_query_only=cfg.same_query_only)
    primitive_by_id = {primitive.primitive_id: primitive for primitive in primitives}
    compact_start = time.perf_counter()
    candidate_keys = {
        (
            pair.query_id,
            int(primitive_by_id[pair.primitive_a_id].metadata["time_sample_index"]),
        )
        for pair in pairs
    }
    active_query_ids = {query_id for query_id, _ in candidate_keys}
    compact_ms = (time.perf_counter() - compact_start) * 1000.0

    exact_start = time.perf_counter()
    traces = {sample.query_id: evaluate_swept_sphere_oracle(sample) for sample in samples}
    query_results = tuple(
        _make_query_result(
            sample,
            traces[sample.query_id],
            active=sample.query_id in active_query_ids,
            status_prefix="rt_dcd",
        )
        for sample in samples
    )
    exact_ms = (time.perf_counter() - exact_start) * 1000.0
    benchmark = _make_benchmark_row(
        query_results,
        candidate_count=len(candidate_keys),
        rt_ms=build_ms + stats.elapsed_ms + compact_ms,
        exact_ms=exact_ms,
    )
    return RTStyleReproductionResult(
        benchmark=benchmark,
        query_results=query_results,
        candidate_stats=_make_candidate_stats(
            backend_name=cfg.backend_name,
            primitive_count=len(primitives),
            raw_hit_count=len(pairs),
            compact_candidate_count=len(candidate_keys),
            candidate_recall=benchmark.candidate_recall,
            build_ms=build_ms,
            trace_ms=stats.elapsed_ms,
            compact_ms=compact_ms,
        ),
        style_name="RTDCDStyle",
        source_name="internal_analytic_oracle",
        scene_name="rt_dcd_style_discrete_samples",
        batch_id=f"samples_{cfg.sample_count}",
    )


def run_rt_dcd_style_on_generated_dataset(
    dataset: GeneratedDataset,
    config: RTDCDStyleConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> RTStyleReproductionResult:
    return run_rt_dcd_style_on_internal_samples(dataset.samples, config, backend=backend)


def run_rt_ccd_style_on_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    config: RTCCDStyleConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> RTStyleReproductionResult:
    if not samples:
        raise ValueError("RTCCDStyle requires at least one sample")
    cfg = validate_rt_ccd_style_config(config or RTCCDStyleConfig())
    broad_phase = backend or CpuAabbBroadPhaseBackend(name=cfg.backend_name)

    build_start = time.perf_counter()
    primitives: list[BroadPhasePrimitive] = []
    for sample in samples:
        primitives.extend(_ccd_primitives_for_sample(sample, cfg))
    build_ms = (time.perf_counter() - build_start) * 1000.0

    pairs, stats = broad_phase.find_pairs(primitives, same_query_only=cfg.same_query_only)
    primitive_by_id = {primitive.primitive_id: primitive for primitive in primitives}
    compact_start = time.perf_counter()
    candidate_keys = {
        (
            pair.query_id,
            int(primitive_by_id[pair.primitive_a_id].metadata["slab_id"]),
        )
        for pair in pairs
    }
    active_slabs_by_query: dict[int, set[int]] = {}
    for query_id, slab_id in candidate_keys:
        active_slabs_by_query.setdefault(query_id, set()).add(slab_id)
    compact_ms = (time.perf_counter() - compact_start) * 1000.0

    exact_start = time.perf_counter()
    traces = {sample.query_id: evaluate_swept_sphere_oracle(sample) for sample in samples}
    query_results: list[PureExactQueryResult] = []
    for sample in samples:
        trace = traces[sample.query_id]
        contact_slabs = _positive_slab_ids(trace, cfg.slab_count)
        active = bool(active_slabs_by_query.get(sample.query_id, set()) & contact_slabs) if trace.collided else (
            sample.query_id in active_slabs_by_query
        )
        query_results.append(
            _make_query_result(
                sample,
                trace,
                active=active,
                status_prefix="rt_ccd",
            )
        )
    exact_ms = (time.perf_counter() - exact_start) * 1000.0
    query_results_tuple = tuple(query_results)
    benchmark = _make_benchmark_row(
        query_results_tuple,
        candidate_count=len(candidate_keys),
        rt_ms=build_ms + stats.elapsed_ms + compact_ms,
        exact_ms=exact_ms,
    )
    return RTStyleReproductionResult(
        benchmark=benchmark,
        query_results=query_results_tuple,
        candidate_stats=_make_candidate_stats(
            backend_name=cfg.backend_name,
            primitive_count=len(primitives),
            raw_hit_count=len(pairs),
            compact_candidate_count=len(candidate_keys),
            candidate_recall=benchmark.candidate_recall,
            build_ms=build_ms,
            trace_ms=stats.elapsed_ms,
            compact_ms=compact_ms,
        ),
        style_name="RTCCDStyle",
        source_name="internal_analytic_oracle",
        scene_name="rt_ccd_style_uniform_slabs",
        batch_id=f"slabs_{cfg.slab_count}",
    )


def run_rt_ccd_style_on_generated_dataset(
    dataset: GeneratedDataset,
    config: RTCCDStyleConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> RTStyleReproductionResult:
    return run_rt_ccd_style_on_internal_samples(dataset.samples, config, backend=backend)
