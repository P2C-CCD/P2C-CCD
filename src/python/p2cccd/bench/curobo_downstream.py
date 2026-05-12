from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

from p2cccd.contracts import BenchmarkRow
from p2cccd.data.dataset import GeneratedDataset
from p2cccd.data.oracle import evaluate_swept_sphere_oracle
from p2cccd.data.samplers import MotionDiscPairSample, PairFamily
from p2cccd.validators import validate_benchmark_row

from .pure_exact_cpu import PureExactQueryResult


Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class CuRoboDownstreamConfig:
    trajectory_step_count: int = 16
    link_sphere_radius_scale: float = 1.0
    collision_activation_distance: float = 0.0
    robot_link_only: bool = True


@dataclass(frozen=True, slots=True)
class CuRoboDownstreamStats:
    style_name: str
    query_count: int
    robot_link_query_count: int
    trajectory_step_count: int
    pose_pair_check_count: int
    discrete_collision_count: int
    avg_min_sampled_margin: float
    min_sampled_margin: float
    checker_ms: float


@dataclass(frozen=True, slots=True)
class CuRoboDownstreamResult:
    benchmark: BenchmarkRow
    query_results: tuple[PureExactQueryResult, ...]
    stats: CuRoboDownstreamStats
    source_name: str
    scene_name: str
    batch_id: str

    @property
    def final_fn_zero(self) -> bool:
        return self.benchmark.fn_count == 0


def validate_curobo_downstream_config(config: CuRoboDownstreamConfig) -> CuRoboDownstreamConfig:
    if config.trajectory_step_count < 2:
        raise ValueError("CuRoboDownstreamConfig.trajectory_step_count must be at least 2")
    if not math.isfinite(config.link_sphere_radius_scale) or config.link_sphere_radius_scale <= 0.0:
        raise ValueError("CuRoboDownstreamConfig.link_sphere_radius_scale must be finite and positive")
    if not math.isfinite(config.collision_activation_distance) or config.collision_activation_distance < 0.0:
        raise ValueError("CuRoboDownstreamConfig.collision_activation_distance must be finite and non-negative")
    return config


def _lerp(lhs: Vec3, rhs: Vec3, t: float) -> Vec3:
    return (
        lhs[0] + (rhs[0] - lhs[0]) * t,
        lhs[1] + (rhs[1] - lhs[1]) * t,
        lhs[2] + (rhs[2] - lhs[2]) * t,
    )


def _sub(lhs: Vec3, rhs: Vec3) -> Vec3:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def _norm(value: Vec3) -> float:
    return math.sqrt(max(0.0, value[0] * value[0] + value[1] * value[1] + value[2] * value[2]))


def _sample_times(step_count: int) -> tuple[float, ...]:
    return tuple(index / float(step_count - 1) for index in range(step_count))


def _robot_link_samples(
    samples: Sequence[MotionDiscPairSample],
    *,
    robot_link_only: bool,
) -> tuple[MotionDiscPairSample, ...]:
    if not robot_link_only:
        return tuple(samples)
    return tuple(sample for sample in samples if sample.family is PairFamily.ROBOT_LINK_PAIR)


def _sampled_margin(sample: MotionDiscPairSample, config: CuRoboDownstreamConfig) -> float:
    radius_sum = (sample.radius_a + sample.radius_b) * config.link_sphere_radius_scale
    min_distance = min(
        _norm(
            _sub(
                _lerp(sample.center_a_t0, sample.center_a_t1, t),
                _lerp(sample.center_b_t0, sample.center_b_t1, t),
            )
        )
        for t in _sample_times(config.trajectory_step_count)
    )
    return min_distance - radius_sum


def _query_result_for_sample(
    sample: MotionDiscPairSample,
    config: CuRoboDownstreamConfig,
) -> tuple[PureExactQueryResult, float]:
    trace = evaluate_swept_sphere_oracle(sample)
    margin = _sampled_margin(sample, config)
    predicted_collision = margin <= config.collision_activation_distance
    family_name = "robot_link_pair" if sample.family is PairFamily.ROBOT_LINK_PAIR else "mesh_pair"
    if predicted_collision:
        status = "curobo_discrete_collision"
    else:
        status = "curobo_discrete_clear"
    return (
        PureExactQueryResult(
            query_id=sample.query_id,
            family=family_name,
            predicted_collision=predicted_collision,
            ground_truth_collision=trace.collided,
            status=status,
            toi_upper=trace.toi if predicted_collision and trace.collided else 1.0,
            safe_margin_lb=max(0.0, margin),
            exact_evals=0,
            max_depth=0,
        ),
        margin,
    )


def _make_benchmark_row(
    query_results: Sequence[PureExactQueryResult],
    *,
    discrete_collision_count: int,
    checker_ms: float,
) -> BenchmarkRow:
    query_count = len(query_results)
    if query_count <= 0:
        raise ValueError("CuRoboDownstream requires at least one robot link query")
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
    row = BenchmarkRow(
        query_count=query_count,
        fn_count=fn_count,
        fp_count=fp_count,
        candidate_recall=1.0 if positive_count == 0 else covered_positive_count / float(positive_count),
        avg_candidates=discrete_collision_count / float(query_count),
        avg_exact_evals=0.0,
        avg_subdivision_depth=0.0,
        fallback_ratio=0.0,
        rt_ms=0.0,
        proposal_ms=checker_ms,
        exact_ms=0.0,
        total_ms=checker_ms,
        qps=0.0 if checker_ms <= 0.0 else 1000.0 * query_count / checker_ms,
    )
    return validate_benchmark_row(row)


def run_curobo_downstream_on_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    config: CuRoboDownstreamConfig | None = None,
) -> CuRoboDownstreamResult:
    cfg = validate_curobo_downstream_config(config or CuRoboDownstreamConfig())
    robot_samples = _robot_link_samples(samples, robot_link_only=cfg.robot_link_only)
    if not robot_samples:
        raise ValueError("CuRoboDownstream requires at least one robot link pair sample")

    checker_start = time.perf_counter()
    evaluated = tuple(_query_result_for_sample(sample, cfg) for sample in robot_samples)
    checker_ms = (time.perf_counter() - checker_start) * 1000.0
    query_results = tuple(result for result, _ in evaluated)
    margins = tuple(margin for _, margin in evaluated)
    discrete_collision_count = sum(1 for result in query_results if result.predicted_collision)
    stats = CuRoboDownstreamStats(
        style_name="CuRoboDownstream",
        query_count=len(query_results),
        robot_link_query_count=sum(1 for sample in robot_samples if sample.family is PairFamily.ROBOT_LINK_PAIR),
        trajectory_step_count=cfg.trajectory_step_count,
        pose_pair_check_count=len(robot_samples) * cfg.trajectory_step_count,
        discrete_collision_count=discrete_collision_count,
        avg_min_sampled_margin=sum(margins) / max(1, len(margins)),
        min_sampled_margin=min(margins),
        checker_ms=checker_ms,
    )
    return CuRoboDownstreamResult(
        benchmark=_make_benchmark_row(
            query_results,
            discrete_collision_count=discrete_collision_count,
            checker_ms=checker_ms,
        ),
        query_results=query_results,
        stats=stats,
        source_name="internal_analytic_oracle",
        scene_name="curobo_downstream_robot_link_motion",
        batch_id=f"trajectory_steps_{cfg.trajectory_step_count}",
    )


def run_curobo_downstream_on_generated_dataset(
    dataset: GeneratedDataset,
    config: CuRoboDownstreamConfig | None = None,
) -> CuRoboDownstreamResult:
    return run_curobo_downstream_on_internal_samples(dataset.samples, config)
