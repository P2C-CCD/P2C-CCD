from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Sequence

from p2cccd.contracts import BenchmarkRow
from p2cccd.data.dataset import GeneratedDataset
from p2cccd.data.oracle import evaluate_swept_sphere_oracle
from p2cccd.data.samplers import MotionDiscPairSample
from p2cccd.datasets.ccd import CCDQueryFamily, DatasetQueryBatch, ExternalCCDQuery
from p2cccd.validators import validate_benchmark_row


Vec3 = tuple[float, float, float]
DistanceFn = Callable[[float], float]
MotionRadiusFn = Callable[[float, float], float]


@dataclass(frozen=True, slots=True)
class PureExactCPUConfig:
    eps_time: float = 1.0e-5
    eps_space: float = 1.0e-8
    max_subdivision_depth: int = 32
    conservative_undecided_as_collision: bool = True


@dataclass(frozen=True, slots=True)
class PureExactQueryResult:
    query_id: int
    family: str
    predicted_collision: bool
    ground_truth_collision: bool | None
    status: str
    toi_upper: float
    safe_margin_lb: float
    exact_evals: int
    max_depth: int


@dataclass(frozen=True, slots=True)
class PureExactCPUResult:
    benchmark: BenchmarkRow
    query_results: tuple[PureExactQueryResult, ...]
    source_name: str
    scene_name: str
    batch_id: str

    @property
    def final_fn_zero(self) -> bool:
        return self.benchmark.fn_count == 0


def _validate_config(config: PureExactCPUConfig) -> PureExactCPUConfig:
    if not math.isfinite(config.eps_time) or config.eps_time <= 0.0:
        raise ValueError("PureExactCPUConfig.eps_time must be finite and positive")
    if not math.isfinite(config.eps_space) or config.eps_space <= 0.0:
        raise ValueError("PureExactCPUConfig.eps_space must be finite and positive")
    if config.max_subdivision_depth < 0:
        raise ValueError("PureExactCPUConfig.max_subdivision_depth must be non-negative")
    return config


def _add(lhs: Vec3, rhs: Vec3) -> Vec3:
    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _sub(lhs: Vec3, rhs: Vec3) -> Vec3:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def _scale(value: Vec3, scale: float) -> Vec3:
    return (value[0] * scale, value[1] * scale, value[2] * scale)


def _dot(lhs: Vec3, rhs: Vec3) -> float:
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]


def _cross(lhs: Vec3, rhs: Vec3) -> Vec3:
    return (
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    )


def _norm(value: Vec3) -> float:
    return math.sqrt(max(0.0, _dot(value, value)))


def _lerp(lhs: Vec3, rhs: Vec3, t: float) -> Vec3:
    return _add(_scale(lhs, 1.0 - t), _scale(rhs, t))


def _distance_point_segment(point: Vec3, a: Vec3, b: Vec3) -> float:
    ab = _sub(b, a)
    ab_squared = _dot(ab, ab)
    if ab_squared <= 1.0e-14:
        return _norm(_sub(point, a))
    t = min(1.0, max(0.0, _dot(_sub(point, a), ab) / ab_squared))
    return _norm(_sub(point, _add(a, _scale(ab, t))))


def _distance_segment_segment(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> float:
    d1 = _sub(q1, p1)
    d2 = _sub(q2, p2)
    r = _sub(p1, p2)
    a = _dot(d1, d1)
    e = _dot(d2, d2)
    f = _dot(d2, r)
    s = 0.0
    t = 0.0
    if a <= 1.0e-14 and e <= 1.0e-14:
        return _norm(_sub(p1, p2))
    if a <= 1.0e-14:
        t = min(1.0, max(0.0, f / e))
    else:
        c = _dot(d1, r)
        if e <= 1.0e-14:
            s = min(1.0, max(0.0, -c / a))
        else:
            b = _dot(d1, d2)
            denom = a * e - b * b
            if abs(denom) > 1.0e-14:
                s = min(1.0, max(0.0, (b * f - c * e) / denom))
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = min(1.0, max(0.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = min(1.0, max(0.0, (b - c) / a))
    closest_a = _add(p1, _scale(d1, s))
    closest_b = _add(p2, _scale(d2, t))
    return _norm(_sub(closest_a, closest_b))


def _distance_point_triangle(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> float:
    ab = _sub(b, a)
    ac = _sub(c, a)
    normal = _cross(ab, ac)
    if _dot(normal, normal) <= 1.0e-14:
        return min(
            _distance_point_segment(point, a, b),
            _distance_point_segment(point, b, c),
            _distance_point_segment(point, c, a),
        )

    ap = _sub(point, a)
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return _norm(_sub(point, a))

    bp = _sub(point, b)
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return _norm(_sub(point, b))

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return _norm(_sub(point, _add(a, _scale(ab, v))))

    cp = _sub(point, c)
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return _norm(_sub(point, c))

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return _norm(_sub(point, _add(a, _scale(ac, w))))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return _norm(_sub(point, _add(b, _scale(_sub(c, b), w))))

    return abs(_dot(ap, normal) / max(_norm(normal), 1.0e-14))


def _trajectory_position(vertices_t0: Sequence[Vec3], vertices_t1: Sequence[Vec3], index: int, t: float) -> Vec3:
    return _lerp(vertices_t0[index], vertices_t1[index], t)


def _max_vertex_displacement(vertices_t0: Sequence[Vec3], vertices_t1: Sequence[Vec3], indices: Sequence[int]) -> float:
    return max((_norm(_sub(vertices_t1[index], vertices_t0[index])) for index in indices), default=0.0)


def _evaluate_recursive(
    distance_fn: DistanceFn,
    motion_radius_fn: MotionRadiusFn,
    config: PureExactCPUConfig,
    *,
    t0: float,
    t1: float,
    depth: int = 0,
) -> tuple[str, float, float, int, int]:
    mid = 0.5 * (t0 + t1)
    samples = ((t0, distance_fn(t0)), (mid, distance_fn(mid)), (t1, distance_fn(t1)))
    exact_evals = 3
    min_sample_t, min_sample_distance = min(samples, key=lambda item: item[1])
    if min_sample_distance <= config.eps_space:
        return "collision", min_sample_t, 0.0, exact_evals, depth

    mid_distance = samples[1][1]
    lower_bound = mid_distance - motion_radius_fn(t0, t1)
    if lower_bound > config.eps_space:
        return "separation", t1, max(0.0, lower_bound - config.eps_space), exact_evals, depth

    if depth >= config.max_subdivision_depth or t1 - t0 <= config.eps_time:
        return "undecided", t1, 0.0, exact_evals, depth

    left = _evaluate_recursive(
        distance_fn,
        motion_radius_fn,
        config,
        t0=t0,
        t1=mid,
        depth=depth + 1,
    )
    exact_evals += left[3]
    if left[0] == "collision":
        return left[0], left[1], left[2], exact_evals, max(depth, left[4])

    right = _evaluate_recursive(
        distance_fn,
        motion_radius_fn,
        config,
        t0=mid,
        t1=t1,
        depth=depth + 1,
    )
    exact_evals += right[3]
    if right[0] == "collision":
        return right[0], right[1], right[2], exact_evals, max(left[4], right[4])
    if left[0] == "separation" and right[0] == "separation":
        return "separation", t1, min(left[2], right[2]), exact_evals, max(left[4], right[4])
    return "undecided", t1, 0.0, exact_evals, max(left[4], right[4])


def evaluate_external_ccd_query(
    query: ExternalCCDQuery,
    config: PureExactCPUConfig | None = None,
) -> PureExactQueryResult:
    cfg = _validate_config(config or PureExactCPUConfig())
    vertices_t0 = query.vertices_t0
    vertices_t1 = query.vertices_t1

    if query.family is CCDQueryFamily.VERTEX_FACE:
        total_displacement = _max_vertex_displacement(vertices_t0, vertices_t1, (0,)) + _max_vertex_displacement(
            vertices_t0, vertices_t1, (1, 2, 3)
        )

        def distance_fn(t: float) -> float:
            return _distance_point_triangle(
                _trajectory_position(vertices_t0, vertices_t1, 0, t),
                _trajectory_position(vertices_t0, vertices_t1, 1, t),
                _trajectory_position(vertices_t0, vertices_t1, 2, t),
                _trajectory_position(vertices_t0, vertices_t1, 3, t),
            )

        def motion_radius_fn(t0: float, t1: float) -> float:
            return 0.5 * max(0.0, t1 - t0) * total_displacement
    elif query.family is CCDQueryFamily.EDGE_EDGE:
        total_displacement = _max_vertex_displacement(vertices_t0, vertices_t1, (0, 1)) + _max_vertex_displacement(
            vertices_t0, vertices_t1, (2, 3)
        )

        def distance_fn(t: float) -> float:
            return _distance_segment_segment(
                _trajectory_position(vertices_t0, vertices_t1, 0, t),
                _trajectory_position(vertices_t0, vertices_t1, 1, t),
                _trajectory_position(vertices_t0, vertices_t1, 2, t),
                _trajectory_position(vertices_t0, vertices_t1, 3, t),
            )

        def motion_radius_fn(t0: float, t1: float) -> float:
            return 0.5 * max(0.0, t1 - t0) * total_displacement
    else:
        raise ValueError(f"unsupported external CCD query family: {query.family}")

    status, toi_upper, safe_margin_lb, exact_evals, max_depth = _evaluate_recursive(
        distance_fn,
        motion_radius_fn,
        cfg,
        t0=0.0,
        t1=1.0,
    )
    predicted = status == "collision" or (
        status == "undecided" and cfg.conservative_undecided_as_collision
    )
    return PureExactQueryResult(
        query_id=query.query_id,
        family=query.family.p2cccd_witness_family,
        predicted_collision=predicted,
        ground_truth_collision=query.ground_truth_collides,
        status=status,
        toi_upper=toi_upper,
        safe_margin_lb=safe_margin_lb,
        exact_evals=exact_evals,
        max_depth=max_depth,
    )


def _result_to_benchmark_row(
    query_results: Sequence[PureExactQueryResult],
    *,
    elapsed_ms: float,
) -> BenchmarkRow:
    if not query_results:
        raise ValueError("PureExactCPU requires at least one query result")
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
    avg_exact_evals = sum(result.exact_evals for result in query_results) / len(query_results)
    avg_depth = sum(result.max_depth for result in query_results) / len(query_results)
    qps = 0.0 if elapsed_ms <= 0.0 else 1000.0 * len(query_results) / elapsed_ms
    row = BenchmarkRow(
        query_count=len(query_results),
        fn_count=fn_count,
        fp_count=fp_count,
        candidate_recall=1.0 if fn_count == 0 else 1.0 - (fn_count / max(1, len(known))),
        avg_candidates=1.0,
        avg_exact_evals=avg_exact_evals,
        avg_subdivision_depth=avg_depth,
        fallback_ratio=0.0,
        rt_ms=0.0,
        proposal_ms=0.0,
        exact_ms=elapsed_ms,
        total_ms=elapsed_ms,
        qps=qps,
    )
    return validate_benchmark_row(row)


def run_pure_exact_cpu_on_external_batch(
    batch: DatasetQueryBatch,
    config: PureExactCPUConfig | None = None,
) -> PureExactCPUResult:
    cfg = _validate_config(config or PureExactCPUConfig())
    start = time.perf_counter()
    query_results = tuple(evaluate_external_ccd_query(query, cfg) for query in batch.queries)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PureExactCPUResult(
        benchmark=_result_to_benchmark_row(query_results, elapsed_ms=elapsed_ms),
        query_results=query_results,
        source_name=batch.source_name,
        scene_name=batch.scene_name,
        batch_id=batch.batch_id,
    )


def run_pure_exact_cpu_on_internal_samples(
    samples: Sequence[MotionDiscPairSample],
) -> PureExactCPUResult:
    if not samples:
        raise ValueError("PureExactCPU internal sample run requires at least one sample")
    start = time.perf_counter()
    query_results: list[PureExactQueryResult] = []
    for sample in samples:
        trace = evaluate_swept_sphere_oracle(sample)
        query_results.append(
            PureExactQueryResult(
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
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PureExactCPUResult(
        benchmark=_result_to_benchmark_row(tuple(query_results), elapsed_ms=elapsed_ms),
        query_results=tuple(query_results),
        source_name="internal_analytic_oracle",
        scene_name="programmatic_motion_disc_pairs",
        batch_id="internal_samples",
    )


def run_pure_exact_cpu_on_generated_dataset(dataset: GeneratedDataset) -> PureExactCPUResult:
    return run_pure_exact_cpu_on_internal_samples(dataset.samples)
