from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Sequence

from p2cccd.contracts import BenchmarkRow
from p2cccd.data.dataset import GeneratedDataset
from p2cccd.data.oracle import evaluate_swept_sphere_oracle
from p2cccd.data.samplers import MotionDiscPairSample
from p2cccd.datasets.ccd import DatasetQueryBatch, ExternalCCDQuery
from p2cccd.validators import validate_benchmark_row

from .bvh_exact import (
    BroadPhasePair,
    BroadPhasePrimitive,
    external_query_to_broad_phase_primitives,
    internal_sample_to_broad_phase_primitives,
)
from .pure_exact_cpu import (
    PureExactCPUConfig,
    PureExactQueryResult,
    evaluate_external_ccd_query,
)


@dataclass(frozen=True, slots=True)
class SortBroadPhaseConfig:
    exact: PureExactCPUConfig = field(default_factory=PureExactCPUConfig)
    backend_name: str = "cpu_sort_sweep"
    axis: int = 0
    same_query_only: bool = True


@dataclass(frozen=True, slots=True)
class SortBroadPhaseStats:
    primitive_count: int
    endpoint_count: int
    pair_count: int
    active_interval_tests: int
    aabb_overlap_tests: int
    sort_ms: float
    sweep_ms: float
    total_ms: float
    backend_name: str
    axis: int


@dataclass(frozen=True, slots=True)
class SortBroadPhaseExactResult:
    benchmark: BenchmarkRow
    query_results: tuple[PureExactQueryResult, ...]
    sort_stats: SortBroadPhaseStats
    source_name: str
    scene_name: str
    batch_id: str

    @property
    def final_fn_zero(self) -> bool:
        return self.benchmark.fn_count == 0


def _validate_config(config: SortBroadPhaseConfig) -> SortBroadPhaseConfig:
    if config.backend_name not in {"cpu_sort_sweep", "gpu_sort_sweep_compatible"}:
        raise ValueError("SortBroadPhaseConfig.backend_name is unsupported")
    if config.axis not in {0, 1, 2}:
        raise ValueError("SortBroadPhaseConfig.axis must be 0, 1, or 2")
    return config


def sort_sweep_broad_phase_pairs(
    primitives: Sequence[BroadPhasePrimitive],
    *,
    config: SortBroadPhaseConfig | None = None,
) -> tuple[tuple[BroadPhasePair, ...], SortBroadPhaseStats]:
    cfg = _validate_config(config or SortBroadPhaseConfig())

    sort_start = time.perf_counter()
    sorted_primitives = sorted(
        primitives,
        key=lambda primitive: (
            primitive.aabb.min[cfg.axis],
            primitive.aabb.max[cfg.axis],
            primitive.query_id,
            primitive.primitive_id,
        ),
    )
    sort_ms = (time.perf_counter() - sort_start) * 1000.0

    sweep_start = time.perf_counter()
    active: list[BroadPhasePrimitive] = []
    pair_keys: set[tuple[int, int, int, str]] = set()
    pairs: list[BroadPhasePair] = []
    active_interval_tests = 0
    aabb_overlap_tests = 0

    for rhs in sorted_primitives:
        active = [
            primitive
            for primitive in active
            if primitive.aabb.max[cfg.axis] >= rhs.aabb.min[cfg.axis]
        ]
        for lhs in active:
            active_interval_tests += 1
            if lhs.role == rhs.role:
                continue
            if cfg.same_query_only and lhs.query_id != rhs.query_id:
                continue
            if lhs.family != rhs.family:
                continue

            aabb_overlap_tests += 1
            if not lhs.aabb.overlaps(rhs.aabb):
                continue

            first, second = (lhs, rhs) if lhs.role <= rhs.role else (rhs, lhs)
            query_id = lhs.query_id if lhs.query_id == rhs.query_id else min(lhs.query_id, rhs.query_id)
            key = (query_id, first.primitive_id, second.primitive_id, lhs.family)
            if key in pair_keys:
                continue
            pair_keys.add(key)
            pairs.append(
                BroadPhasePair(
                    query_id=query_id,
                    primitive_a_id=first.primitive_id,
                    primitive_b_id=second.primitive_id,
                    family=lhs.family,
                )
            )
        active.append(rhs)

    pairs.sort(key=lambda pair: (pair.query_id, pair.primitive_a_id, pair.primitive_b_id))
    sweep_ms = (time.perf_counter() - sweep_start) * 1000.0
    stats = SortBroadPhaseStats(
        primitive_count=len(primitives),
        endpoint_count=2 * len(primitives),
        pair_count=len(pairs),
        active_interval_tests=active_interval_tests,
        aabb_overlap_tests=aabb_overlap_tests,
        sort_ms=sort_ms,
        sweep_ms=sweep_ms,
        total_ms=sort_ms + sweep_ms,
        backend_name=cfg.backend_name,
        axis=cfg.axis,
    )
    return tuple(pairs), stats


def _separation_result_for_culled_external_query(query: ExternalCCDQuery) -> PureExactQueryResult:
    return PureExactQueryResult(
        query_id=query.query_id,
        family=query.family.p2cccd_witness_family,
        predicted_collision=False,
        ground_truth_collision=query.ground_truth_collides,
        status="sort_broad_phase_separation",
        toi_upper=1.0,
        safe_margin_lb=0.0,
        exact_evals=0,
        max_depth=0,
    )


def _separation_result_for_culled_internal_sample(sample: MotionDiscPairSample) -> PureExactQueryResult:
    trace = evaluate_swept_sphere_oracle(sample)
    return PureExactQueryResult(
        query_id=sample.query_id,
        family="swept_sphere_proxy",
        predicted_collision=False,
        ground_truth_collision=trace.collided,
        status="sort_broad_phase_separation",
        toi_upper=1.0,
        safe_margin_lb=max(0.0, trace.safe_margin),
        exact_evals=0,
        max_depth=0,
    )


def _make_benchmark_row(
    query_results: Sequence[PureExactQueryResult],
    *,
    sort_stats: SortBroadPhaseStats,
    exact_elapsed_ms: float,
    total_query_count: int,
) -> BenchmarkRow:
    if total_query_count <= 0:
        raise ValueError("SortBroadPhaseExact requires at least one query")
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
    total_ms = sort_stats.total_ms + exact_elapsed_ms
    row = BenchmarkRow(
        query_count=total_query_count,
        fn_count=fn_count,
        fp_count=fp_count,
        candidate_recall=1.0 if fn_count == 0 else 1.0 - (fn_count / max(1, len(known))),
        avg_candidates=sort_stats.pair_count / total_query_count,
        avg_exact_evals=total_exact_evals / total_query_count,
        avg_subdivision_depth=total_depth / total_query_count,
        fallback_ratio=0.0,
        rt_ms=sort_stats.total_ms,
        proposal_ms=0.0,
        exact_ms=exact_elapsed_ms,
        total_ms=total_ms,
        qps=0.0 if total_ms <= 0.0 else 1000.0 * total_query_count / total_ms,
    )
    return validate_benchmark_row(row)


def run_sort_broad_phase_exact_on_external_batch(
    batch: DatasetQueryBatch,
    config: SortBroadPhaseConfig | None = None,
) -> SortBroadPhaseExactResult:
    if not batch.queries:
        raise ValueError("SortBroadPhaseExact external batch requires at least one query")
    cfg = _validate_config(config or SortBroadPhaseConfig())
    primitives: list[BroadPhasePrimitive] = []
    query_by_id = {query.query_id: query for query in batch.queries}
    for query in batch.queries:
        primitives.extend(external_query_to_broad_phase_primitives(query))

    pairs, stats = sort_sweep_broad_phase_pairs(primitives, config=cfg)
    active_query_ids = {pair.query_id for pair in pairs}
    if active_query_ids - set(query_by_id):
        raise ValueError("sort broad phase produced a pair for an unknown external query")

    start = time.perf_counter()
    results: list[PureExactQueryResult] = []
    for query in batch.queries:
        if query.query_id in active_query_ids:
            results.append(evaluate_external_ccd_query(query, cfg.exact))
        else:
            results.append(_separation_result_for_culled_external_query(query))
    exact_elapsed_ms = (time.perf_counter() - start) * 1000.0

    return SortBroadPhaseExactResult(
        benchmark=_make_benchmark_row(
            results,
            sort_stats=stats,
            exact_elapsed_ms=exact_elapsed_ms,
            total_query_count=len(batch.queries),
        ),
        query_results=tuple(results),
        sort_stats=stats,
        source_name=batch.source_name,
        scene_name=batch.scene_name,
        batch_id=batch.batch_id,
    )


def run_sort_broad_phase_exact_on_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    config: SortBroadPhaseConfig | None = None,
) -> SortBroadPhaseExactResult:
    if not samples:
        raise ValueError("SortBroadPhaseExact internal sample run requires at least one sample")
    cfg = _validate_config(config or SortBroadPhaseConfig())
    primitives: list[BroadPhasePrimitive] = []
    sample_by_query_id = {sample.query_id: sample for sample in samples}
    for sample in samples:
        primitives.extend(internal_sample_to_broad_phase_primitives(sample))

    pairs, stats = sort_sweep_broad_phase_pairs(primitives, config=cfg)
    active_query_ids = {pair.query_id for pair in pairs}
    if active_query_ids - set(sample_by_query_id):
        raise ValueError("sort broad phase produced a pair for an unknown internal sample")

    start = time.perf_counter()
    results: list[PureExactQueryResult] = []
    for sample in samples:
        if sample.query_id in active_query_ids:
            trace = evaluate_swept_sphere_oracle(sample)
            results.append(
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
        else:
            results.append(_separation_result_for_culled_internal_sample(sample))
    exact_elapsed_ms = (time.perf_counter() - start) * 1000.0

    return SortBroadPhaseExactResult(
        benchmark=_make_benchmark_row(
            results,
            sort_stats=stats,
            exact_elapsed_ms=exact_elapsed_ms,
            total_query_count=len(samples),
        ),
        query_results=tuple(results),
        sort_stats=stats,
        source_name="internal_analytic_oracle",
        scene_name="programmatic_motion_disc_pairs",
        batch_id="internal_samples",
    )


def run_sort_broad_phase_exact_on_generated_dataset(
    dataset: GeneratedDataset,
    config: SortBroadPhaseConfig | None = None,
) -> SortBroadPhaseExactResult:
    return run_sort_broad_phase_exact_on_internal_samples(dataset.samples, config)
