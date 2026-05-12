from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

from p2cccd.contracts import BenchmarkRow
from p2cccd.data.dataset import GeneratedDataset
from p2cccd.data.oracle import evaluate_swept_sphere_oracle
from p2cccd.data.samplers import MotionDiscPairSample
from p2cccd.validators import validate_benchmark_row

from .pure_exact_cpu import PureExactQueryResult


Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class NeuralSVCDStyleConfig:
    time_sample_count: int = 9
    collision_threshold: float = 0.5
    uncertainty_fallback_threshold: float = 0.35
    conservative_fallback: bool = True
    fallback_on_ood: bool = True
    radius_scale: float = 1.0
    temperature_scale: float = 0.25
    uncertainty_scale: float = 0.6


@dataclass(frozen=True, slots=True)
class CabiNetStyleConfig:
    pose_sample_count: int = 2
    collision_threshold: float = 0.5
    uncertainty_fallback_threshold: float = 0.35
    conservative_fallback: bool = False
    fallback_on_ood: bool = False
    learned_proxy_radius_scale: float = 1.0
    temperature_scale: float = 0.25
    uncertainty_scale: float = 0.6


@dataclass(frozen=True, slots=True)
class LearnedStyleStats:
    style_name: str
    query_count: int
    surrogate_positive_count: int
    uncertainty_fallback_count: int
    ood_fallback_count: int
    exact_candidate_count: int
    omitted_query_count: int
    avg_collision_score: float
    avg_uncertainty: float
    surrogate_ms: float
    exact_ms: float


@dataclass(frozen=True, slots=True)
class LearnedStyleComparisonResult:
    benchmark: BenchmarkRow
    query_results: tuple[PureExactQueryResult, ...]
    stats: LearnedStyleStats
    style_name: str
    source_name: str
    scene_name: str
    batch_id: str

    @property
    def final_fn_zero(self) -> bool:
        return self.benchmark.fn_count == 0


@dataclass(frozen=True, slots=True)
class _SurrogateDecision:
    query_id: int
    collision_score: float
    uncertainty: float
    surrogate_positive: bool
    uncertainty_fallback: bool
    ood_fallback: bool
    exact_candidate: bool


def validate_neural_svcd_style_config(config: NeuralSVCDStyleConfig) -> NeuralSVCDStyleConfig:
    if config.time_sample_count < 2:
        raise ValueError("NeuralSVCDStyleConfig.time_sample_count must be at least 2")
    _ratio("NeuralSVCDStyleConfig.collision_threshold", config.collision_threshold)
    _ratio("NeuralSVCDStyleConfig.uncertainty_fallback_threshold", config.uncertainty_fallback_threshold)
    _positive("NeuralSVCDStyleConfig.radius_scale", config.radius_scale)
    _positive("NeuralSVCDStyleConfig.temperature_scale", config.temperature_scale)
    _positive("NeuralSVCDStyleConfig.uncertainty_scale", config.uncertainty_scale)
    return config


def validate_cabinet_style_config(config: CabiNetStyleConfig) -> CabiNetStyleConfig:
    if config.pose_sample_count < 2:
        raise ValueError("CabiNetStyleConfig.pose_sample_count must be at least 2")
    _ratio("CabiNetStyleConfig.collision_threshold", config.collision_threshold)
    _ratio("CabiNetStyleConfig.uncertainty_fallback_threshold", config.uncertainty_fallback_threshold)
    _positive("CabiNetStyleConfig.learned_proxy_radius_scale", config.learned_proxy_radius_scale)
    _positive("CabiNetStyleConfig.temperature_scale", config.temperature_scale)
    _positive("CabiNetStyleConfig.uncertainty_scale", config.uncertainty_scale)
    return config


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _ratio(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


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


def _sample_times(count: int) -> tuple[float, ...]:
    return tuple(index / float(count - 1) for index in range(count))


def _sampled_margin(
    sample: MotionDiscPairSample,
    *,
    sample_count: int,
    radius_scale: float,
) -> float:
    radius_sum = (sample.radius_a + sample.radius_b) * radius_scale
    min_distance = min(
        _norm(
            _sub(
                _lerp(sample.center_a_t0, sample.center_a_t1, t),
                _lerp(sample.center_b_t0, sample.center_b_t1, t),
            )
        )
        for t in _sample_times(sample_count)
    )
    return min_distance - radius_sum


def _score_from_margin(
    margin: float,
    *,
    radius_sum: float,
    temperature_scale: float,
    uncertainty_scale: float,
) -> tuple[float, float]:
    temperature = max(1.0e-9, radius_sum * temperature_scale)
    uncertainty_width = max(1.0e-9, radius_sum * uncertainty_scale)
    clamped = max(-60.0, min(60.0, margin / temperature))
    collision_score = 1.0 / (1.0 + math.exp(clamped))
    uncertainty = math.exp(-abs(margin) / uncertainty_width)
    return float(collision_score), float(uncertainty)


def _make_decision(
    sample: MotionDiscPairSample,
    *,
    sample_count: int,
    radius_scale: float,
    collision_threshold: float,
    uncertainty_fallback_threshold: float,
    conservative_fallback: bool,
    fallback_on_ood: bool,
    temperature_scale: float,
    uncertainty_scale: float,
) -> _SurrogateDecision:
    margin = _sampled_margin(sample, sample_count=sample_count, radius_scale=radius_scale)
    radius_sum = (sample.radius_a + sample.radius_b) * radius_scale
    score, uncertainty = _score_from_margin(
        margin,
        radius_sum=radius_sum,
        temperature_scale=temperature_scale,
        uncertainty_scale=uncertainty_scale,
    )
    surrogate_positive = score >= collision_threshold
    uncertainty_fallback = conservative_fallback and uncertainty >= uncertainty_fallback_threshold
    ood_fallback = conservative_fallback and fallback_on_ood and sample.ood
    exact_candidate = surrogate_positive or uncertainty_fallback or ood_fallback
    return _SurrogateDecision(
        query_id=sample.query_id,
        collision_score=score,
        uncertainty=uncertainty,
        surrogate_positive=surrogate_positive,
        uncertainty_fallback=uncertainty_fallback,
        ood_fallback=ood_fallback,
        exact_candidate=exact_candidate,
    )


def _query_result_from_decision(
    sample: MotionDiscPairSample,
    decision: _SurrogateDecision,
    *,
    status_prefix: str,
) -> PureExactQueryResult:
    trace = evaluate_swept_sphere_oracle(sample)
    predicted_collision = bool(decision.exact_candidate and trace.collided)
    exact_evals = max(1, int(math.ceil(trace.exact_cost))) if decision.exact_candidate else 0
    if predicted_collision:
        status = "collision"
    elif decision.exact_candidate:
        status = "separation"
    else:
        status = f"{status_prefix}_surrogate_omitted"
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
    exact_candidate_count: int,
    surrogate_ms: float,
    exact_ms: float,
) -> BenchmarkRow:
    query_count = len(query_results)
    if query_count <= 0:
        raise ValueError("learned-style comparison requires at least one query")
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
    covered_positive = positive_count - fn_count
    total_exact_evals = sum(result.exact_evals for result in query_results)
    total_ms = surrogate_ms + exact_ms
    row = BenchmarkRow(
        query_count=query_count,
        fn_count=fn_count,
        fp_count=fp_count,
        candidate_recall=1.0 if positive_count == 0 else covered_positive / float(positive_count),
        avg_candidates=exact_candidate_count / float(query_count),
        avg_exact_evals=total_exact_evals / float(query_count),
        avg_subdivision_depth=0.0,
        fallback_ratio=0.0,
        rt_ms=0.0,
        proposal_ms=surrogate_ms,
        exact_ms=exact_ms,
        total_ms=total_ms,
        qps=0.0 if total_ms <= 0.0 else 1000.0 * query_count / total_ms,
    )
    return validate_benchmark_row(row)


def _make_stats(
    *,
    style_name: str,
    decisions: Sequence[_SurrogateDecision],
    surrogate_ms: float,
    exact_ms: float,
) -> LearnedStyleStats:
    query_count = len(decisions)
    exact_candidate_count = sum(1 for decision in decisions if decision.exact_candidate)
    return LearnedStyleStats(
        style_name=style_name,
        query_count=query_count,
        surrogate_positive_count=sum(1 for decision in decisions if decision.surrogate_positive),
        uncertainty_fallback_count=sum(1 for decision in decisions if decision.uncertainty_fallback),
        ood_fallback_count=sum(1 for decision in decisions if decision.ood_fallback),
        exact_candidate_count=exact_candidate_count,
        omitted_query_count=query_count - exact_candidate_count,
        avg_collision_score=sum(decision.collision_score for decision in decisions) / max(1, query_count),
        avg_uncertainty=sum(decision.uncertainty for decision in decisions) / max(1, query_count),
        surrogate_ms=surrogate_ms,
        exact_ms=exact_ms,
    )


def run_neural_svcd_style_on_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    config: NeuralSVCDStyleConfig | None = None,
) -> LearnedStyleComparisonResult:
    if not samples:
        raise ValueError("NeuralSVCDStyle requires at least one sample")
    cfg = validate_neural_svcd_style_config(config or NeuralSVCDStyleConfig())
    surrogate_start = time.perf_counter()
    decisions = tuple(
        _make_decision(
            sample,
            sample_count=cfg.time_sample_count,
            radius_scale=cfg.radius_scale,
            collision_threshold=cfg.collision_threshold,
            uncertainty_fallback_threshold=cfg.uncertainty_fallback_threshold,
            conservative_fallback=cfg.conservative_fallback,
            fallback_on_ood=cfg.fallback_on_ood,
            temperature_scale=cfg.temperature_scale,
            uncertainty_scale=cfg.uncertainty_scale,
        )
        for sample in samples
    )
    surrogate_ms = (time.perf_counter() - surrogate_start) * 1000.0
    exact_start = time.perf_counter()
    query_results = tuple(
        _query_result_from_decision(sample, decision, status_prefix="neural_svcd")
        for sample, decision in zip(samples, decisions)
    )
    exact_ms = (time.perf_counter() - exact_start) * 1000.0
    stats = _make_stats(
        style_name="NeuralSVCDStyle",
        decisions=decisions,
        surrogate_ms=surrogate_ms,
        exact_ms=exact_ms,
    )
    return LearnedStyleComparisonResult(
        benchmark=_make_benchmark_row(
            query_results,
            exact_candidate_count=stats.exact_candidate_count,
            surrogate_ms=surrogate_ms,
            exact_ms=exact_ms,
        ),
        query_results=query_results,
        stats=stats,
        style_name="NeuralSVCDStyle",
        source_name="internal_analytic_oracle",
        scene_name="neural_svcd_style_surrogate",
        batch_id=f"time_samples_{cfg.time_sample_count}",
    )


def run_neural_svcd_style_on_generated_dataset(
    dataset: GeneratedDataset,
    config: NeuralSVCDStyleConfig | None = None,
) -> LearnedStyleComparisonResult:
    return run_neural_svcd_style_on_internal_samples(dataset.samples, config)


def run_cabinet_style_on_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    config: CabiNetStyleConfig | None = None,
) -> LearnedStyleComparisonResult:
    if not samples:
        raise ValueError("CabiNetStyle requires at least one sample")
    cfg = validate_cabinet_style_config(config or CabiNetStyleConfig())
    surrogate_start = time.perf_counter()
    decisions = tuple(
        _make_decision(
            sample,
            sample_count=cfg.pose_sample_count,
            radius_scale=cfg.learned_proxy_radius_scale,
            collision_threshold=cfg.collision_threshold,
            uncertainty_fallback_threshold=cfg.uncertainty_fallback_threshold,
            conservative_fallback=cfg.conservative_fallback,
            fallback_on_ood=cfg.fallback_on_ood,
            temperature_scale=cfg.temperature_scale,
            uncertainty_scale=cfg.uncertainty_scale,
        )
        for sample in samples
    )
    surrogate_ms = (time.perf_counter() - surrogate_start) * 1000.0
    exact_start = time.perf_counter()
    query_results = tuple(
        _query_result_from_decision(sample, decision, status_prefix="cabinet")
        for sample, decision in zip(samples, decisions)
    )
    exact_ms = (time.perf_counter() - exact_start) * 1000.0
    stats = _make_stats(
        style_name="CabiNetStyle",
        decisions=decisions,
        surrogate_ms=surrogate_ms,
        exact_ms=exact_ms,
    )
    return LearnedStyleComparisonResult(
        benchmark=_make_benchmark_row(
            query_results,
            exact_candidate_count=stats.exact_candidate_count,
            surrogate_ms=surrogate_ms,
            exact_ms=exact_ms,
        ),
        query_results=query_results,
        stats=stats,
        style_name="CabiNetStyle",
        source_name="internal_analytic_oracle",
        scene_name="cabinet_style_pose_surrogate",
        batch_id=f"pose_samples_{cfg.pose_sample_count}",
    )


def run_cabinet_style_on_generated_dataset(
    dataset: GeneratedDataset,
    config: CabiNetStyleConfig | None = None,
) -> LearnedStyleComparisonResult:
    return run_cabinet_style_on_internal_samples(dataset.samples, config)
