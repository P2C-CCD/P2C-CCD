from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import math
import time
from typing import Sequence

from p2cccd.data.dataset import GeneratedDataset
from p2cccd.data.oracle import ExactOracleTrace, evaluate_swept_sphere_oracle
from p2cccd.data.samplers import MotionDiscPairSample

from .bvh_exact import (
    Aabb,
    BroadPhaseBackend,
    BroadPhasePrimitive,
    CpuAabbBroadPhaseBackend,
)


Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PatchGranularityAblationOption:
    name: str
    patches_per_object: int
    radius_scale: float = 1.0
    offset_scale: float = 0.0


@dataclass(frozen=True, slots=True)
class PatchGranularityAblationConfig:
    options: tuple[PatchGranularityAblationOption, ...] = (
        PatchGranularityAblationOption("coarse_1x", 1, 1.0, 0.0),
        PatchGranularityAblationOption("medium_2x", 2, 1.0, 0.0),
        PatchGranularityAblationOption("fine_4x", 4, 1.0, 0.0),
    )
    backend_name: str = "cpu_aabb_sort_sweep"
    same_query_only: bool = True
    min_candidate_recall: float = 1.0
    candidate_weight: float = 1.0
    raw_hit_weight: float = 0.05
    proxy_weight: float = 0.01
    radius_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class PatchGranularityAblationRow:
    option_index: int
    option_name: str
    selected: bool
    feasible: bool
    query_count: int
    patches_per_object: int
    radius_scale: float
    offset_scale: float
    patch_count: int
    proxy_count: int
    raw_hit_count: int
    compact_candidate_count: int
    positive_query_count: int
    covered_positive_count: int
    candidate_recall: float
    avg_patch_radius: float
    avg_candidates_per_query: float
    candidates_per_proxy: float
    score: float
    broad_phase_ms: float
    exact_ms: float
    total_ms: float
    fn_count: int
    fp_count: int


@dataclass(frozen=True, slots=True)
class PatchGranularityAblationResult:
    rows: tuple[PatchGranularityAblationRow, ...]
    best_index: int
    source_name: str
    scene_name: str
    batch_id: str
    query_count: int

    @property
    def selected_row(self) -> PatchGranularityAblationRow | None:
        for row in self.rows:
            if row.selected:
                return row
        return None


def validate_patch_granularity_ablation_config(
    config: PatchGranularityAblationConfig,
) -> PatchGranularityAblationConfig:
    if not config.options:
        raise ValueError("PatchGranularityAblationConfig.options must not be empty")
    names: set[str] = set()
    for option in config.options:
        if not option.name:
            raise ValueError("patch granularity option name must not be empty")
        if option.name in names:
            raise ValueError(f"duplicate patch granularity option name: {option.name}")
        names.add(option.name)
        if option.patches_per_object <= 0:
            raise ValueError("patches_per_object must be positive")
        if not math.isfinite(option.radius_scale) or option.radius_scale <= 0.0:
            raise ValueError("radius_scale must be finite and positive")
        if not math.isfinite(option.offset_scale) or option.offset_scale < 0.0:
            raise ValueError("offset_scale must be finite and non-negative")
    if not 0.0 <= config.min_candidate_recall <= 1.0:
        raise ValueError("min_candidate_recall must be in [0, 1]")
    if not config.same_query_only:
        raise ValueError("patch granularity ablation currently requires same_query_only=True")
    for name, value in (
        ("candidate_weight", config.candidate_weight),
        ("raw_hit_weight", config.raw_hit_weight),
        ("proxy_weight", config.proxy_weight),
        ("radius_weight", config.radius_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    return config


def _add(lhs: Vec3, rhs: Vec3) -> Vec3:
    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _mul(value: Vec3, scale: float) -> Vec3:
    return (value[0] * scale, value[1] * scale, value[2] * scale)


def _aabb_from_points(points: Sequence[Vec3], *, inflation: float) -> Aabb:
    if not points:
        raise ValueError("cannot build patch AABB from an empty point set")
    if inflation < 0.0:
        raise ValueError("patch AABB inflation must be non-negative")
    mins = tuple(min(point[axis] for point in points) - inflation for axis in range(3))
    maxs = tuple(max(point[axis] for point in points) + inflation for axis in range(3))
    return Aabb(min=mins, max=maxs)  # type: ignore[arg-type]


def _patch_offset(index: int, count: int, radius: float, offset_scale: float) -> Vec3:
    if count == 1 or offset_scale == 0.0:
        return (0.0, 0.0, 0.0)
    centered_index = float(index) - 0.5 * float(count - 1)
    offset = centered_index * radius * offset_scale
    axis = index % 3
    if axis == 0:
        return (0.0, offset, 0.0)
    if axis == 1:
        return (0.0, 0.0, offset)
    return (0.5 * offset, 0.5 * offset, 0.0)


def _patch_primitives_for_sample(
    sample: MotionDiscPairSample,
    option: PatchGranularityAblationOption,
) -> tuple[BroadPhasePrimitive, ...]:
    primitives: list[BroadPhasePrimitive] = []
    roles = (
        (
            "a",
            sample.center_a_t0,
            sample.center_a_t1,
            sample.radius_a,
            sample.object_a_id,
            sample.patch_a_id,
            0,
        ),
        (
            "b",
            sample.center_b_t0,
            sample.center_b_t1,
            sample.radius_b,
            sample.object_b_id,
            sample.patch_b_id,
            100000,
        ),
    )
    for role, center_t0, center_t1, radius, object_id, base_patch_id, role_offset in roles:
        patch_radius = radius * option.radius_scale
        for patch_index in range(option.patches_per_object):
            offset = _patch_offset(patch_index, option.patches_per_object, radius, option.offset_scale)
            shifted_t0 = _add(center_t0, offset)
            shifted_t1 = _add(center_t1, offset)
            patch_id = base_patch_id * 1000 + patch_index
            primitive_id = sample.query_id * 1000000 + role_offset + patch_index
            primitives.append(
                BroadPhasePrimitive(
                    primitive_id=primitive_id,
                    query_id=sample.query_id,
                    role=role,
                    aabb=_aabb_from_points((shifted_t0, shifted_t1), inflation=patch_radius),
                    family="swept_sphere_proxy",
                    metadata={
                        "sample_id": sample.sample_id,
                        "object_id": object_id,
                        "base_patch_id": base_patch_id,
                        "patch_id": patch_id,
                        "patch_index": patch_index,
                        "patch_radius": patch_radius,
                        "granularity": option.name,
                    },
                )
            )
    return tuple(primitives)


def _precompute_traces(samples: Sequence[MotionDiscPairSample]) -> dict[int, ExactOracleTrace]:
    return {sample.query_id: evaluate_swept_sphere_oracle(sample) for sample in samples}


def _score_row(
    *,
    config: PatchGranularityAblationConfig,
    compact_candidate_count: int,
    raw_hit_count: int,
    proxy_count: int,
    avg_patch_radius: float,
) -> float:
    return (
        config.candidate_weight * float(compact_candidate_count)
        + config.raw_hit_weight * float(raw_hit_count)
        + config.proxy_weight * float(proxy_count)
        + config.radius_weight * float(avg_patch_radius)
    )


def _evaluate_option(
    samples: Sequence[MotionDiscPairSample],
    traces_by_query_id: dict[int, ExactOracleTrace],
    option: PatchGranularityAblationOption,
    option_index: int,
    config: PatchGranularityAblationConfig,
    backend: BroadPhaseBackend,
) -> PatchGranularityAblationRow:
    primitive_start = time.perf_counter()
    primitives: list[BroadPhasePrimitive] = []
    patch_radii: list[float] = []
    for sample in samples:
        sample_primitives = _patch_primitives_for_sample(sample, option)
        primitives.extend(sample_primitives)
        patch_radii.extend(float(primitive.metadata["patch_radius"]) for primitive in sample_primitives)
    primitive_ms = (time.perf_counter() - primitive_start) * 1000.0

    pairs, stats = backend.find_pairs(primitives, same_query_only=config.same_query_only)
    active_query_ids = {pair.query_id for pair in pairs}
    positive_query_ids = {
        query_id for query_id, trace in traces_by_query_id.items() if trace.collided
    }
    covered_positive_count = len(positive_query_ids & active_query_ids)
    candidate_recall = (
        1.0
        if not positive_query_ids
        else covered_positive_count / float(len(positive_query_ids))
    )

    exact_start = time.perf_counter()
    fn_count = 0
    fp_count = 0
    for sample in samples:
        trace = traces_by_query_id[sample.query_id]
        predicted_collision = trace.collided if sample.query_id in active_query_ids else False
        if trace.collided and not predicted_collision:
            fn_count += 1
        if not trace.collided and predicted_collision:
            fp_count += 1
    exact_ms = (time.perf_counter() - exact_start) * 1000.0

    compact_candidate_count = len(active_query_ids)
    raw_hit_count = len(pairs)
    proxy_count = len(primitives)
    avg_patch_radius = sum(patch_radii) / max(1, len(patch_radii))
    feasible = (
        fn_count == 0
        and fp_count == 0
        and candidate_recall + 1.0e-12 >= config.min_candidate_recall
    )
    score = _score_row(
        config=config,
        compact_candidate_count=compact_candidate_count,
        raw_hit_count=raw_hit_count,
        proxy_count=proxy_count,
        avg_patch_radius=avg_patch_radius,
    )
    broad_phase_ms = stats.elapsed_ms + primitive_ms
    total_ms = broad_phase_ms + exact_ms
    return PatchGranularityAblationRow(
        option_index=option_index,
        option_name=option.name,
        selected=False,
        feasible=feasible,
        query_count=len(samples),
        patches_per_object=option.patches_per_object,
        radius_scale=option.radius_scale,
        offset_scale=option.offset_scale,
        patch_count=proxy_count,
        proxy_count=proxy_count,
        raw_hit_count=raw_hit_count,
        compact_candidate_count=compact_candidate_count,
        positive_query_count=len(positive_query_ids),
        covered_positive_count=covered_positive_count,
        candidate_recall=float(candidate_recall),
        avg_patch_radius=float(avg_patch_radius),
        avg_candidates_per_query=compact_candidate_count / max(1, len(samples)),
        candidates_per_proxy=raw_hit_count / max(1, proxy_count),
        score=float(score),
        broad_phase_ms=float(broad_phase_ms),
        exact_ms=float(exact_ms),
        total_ms=float(total_ms),
        fn_count=fn_count,
        fp_count=fp_count,
    )


def _select_best_row(rows: Sequence[PatchGranularityAblationRow]) -> int:
    feasible_rows = [row for row in rows if row.feasible]
    if not feasible_rows:
        return -1
    best = min(
        feasible_rows,
        key=lambda row: (
            row.score,
            row.compact_candidate_count,
            row.raw_hit_count,
            row.proxy_count,
            row.option_index,
        ),
    )
    return best.option_index


def _mark_selected(
    rows: Sequence[PatchGranularityAblationRow],
    best_index: int,
) -> tuple[PatchGranularityAblationRow, ...]:
    return tuple(
        PatchGranularityAblationRow(
            option_index=row.option_index,
            option_name=row.option_name,
            selected=row.option_index == best_index,
            feasible=row.feasible,
            query_count=row.query_count,
            patches_per_object=row.patches_per_object,
            radius_scale=row.radius_scale,
            offset_scale=row.offset_scale,
            patch_count=row.patch_count,
            proxy_count=row.proxy_count,
            raw_hit_count=row.raw_hit_count,
            compact_candidate_count=row.compact_candidate_count,
            positive_query_count=row.positive_query_count,
            covered_positive_count=row.covered_positive_count,
            candidate_recall=row.candidate_recall,
            avg_patch_radius=row.avg_patch_radius,
            avg_candidates_per_query=row.avg_candidates_per_query,
            candidates_per_proxy=row.candidates_per_proxy,
            score=row.score,
            broad_phase_ms=row.broad_phase_ms,
            exact_ms=row.exact_ms,
            total_ms=row.total_ms,
            fn_count=row.fn_count,
            fp_count=row.fp_count,
        )
        for row in rows
    )


def run_patch_granularity_ablation_on_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    config: PatchGranularityAblationConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
    source_name: str = "internal_analytic_oracle",
    scene_name: str = "programmatic_motion_disc_pairs",
    batch_id: str = "internal_samples",
) -> PatchGranularityAblationResult:
    if not samples:
        raise ValueError("patch granularity ablation requires at least one sample")
    cfg = validate_patch_granularity_ablation_config(config or PatchGranularityAblationConfig())
    broad_phase = backend or CpuAabbBroadPhaseBackend(name=cfg.backend_name)
    traces_by_query_id = _precompute_traces(samples)
    rows = [
        _evaluate_option(samples, traces_by_query_id, option, index, cfg, broad_phase)
        for index, option in enumerate(cfg.options)
    ]
    best_index = _select_best_row(rows)
    return PatchGranularityAblationResult(
        rows=_mark_selected(rows, best_index),
        best_index=best_index,
        source_name=source_name,
        scene_name=scene_name,
        batch_id=batch_id,
        query_count=len(samples),
    )


def run_patch_granularity_ablation_on_generated_dataset(
    dataset: GeneratedDataset,
    config: PatchGranularityAblationConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> PatchGranularityAblationResult:
    return run_patch_granularity_ablation_on_internal_samples(
        dataset.samples,
        config,
        backend=backend,
        source_name="internal_analytic_oracle",
        scene_name="generated_exact_oracle_dataset",
        batch_id="generated_dataset",
    )


def patch_granularity_ablation_csv_header() -> tuple[str, ...]:
    return (
        "source_name",
        "scene_name",
        "batch_id",
        "best_index",
        "option_index",
        "option_name",
        "selected",
        "feasible",
        "query_count",
        "patches_per_object",
        "radius_scale",
        "offset_scale",
        "patch_count",
        "proxy_count",
        "raw_hit_count",
        "compact_candidate_count",
        "positive_query_count",
        "covered_positive_count",
        "candidate_recall",
        "avg_patch_radius",
        "avg_candidates_per_query",
        "candidates_per_proxy",
        "score",
        "broad_phase_ms",
        "exact_ms",
        "total_ms",
        "fn_count",
        "fp_count",
    )


def patch_granularity_ablation_rows_to_csv(result: PatchGranularityAblationResult) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=patch_granularity_ablation_csv_header(), lineterminator="\n")
    writer.writeheader()
    for row in result.rows:
        writer.writerow(
            {
                "source_name": result.source_name,
                "scene_name": result.scene_name,
                "batch_id": result.batch_id,
                "best_index": result.best_index,
                "option_index": row.option_index,
                "option_name": row.option_name,
                "selected": int(row.selected),
                "feasible": int(row.feasible),
                "query_count": row.query_count,
                "patches_per_object": row.patches_per_object,
                "radius_scale": row.radius_scale,
                "offset_scale": row.offset_scale,
                "patch_count": row.patch_count,
                "proxy_count": row.proxy_count,
                "raw_hit_count": row.raw_hit_count,
                "compact_candidate_count": row.compact_candidate_count,
                "positive_query_count": row.positive_query_count,
                "covered_positive_count": row.covered_positive_count,
                "candidate_recall": row.candidate_recall,
                "avg_patch_radius": row.avg_patch_radius,
                "avg_candidates_per_query": row.avg_candidates_per_query,
                "candidates_per_proxy": row.candidates_per_proxy,
                "score": row.score,
                "broad_phase_ms": row.broad_phase_ms,
                "exact_ms": row.exact_ms,
                "total_ms": row.total_ms,
                "fn_count": row.fn_count,
                "fp_count": row.fp_count,
            }
        )
    return output.getvalue()
