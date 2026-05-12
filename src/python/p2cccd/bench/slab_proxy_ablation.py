from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import math
import time
from typing import Iterable, Sequence

from p2cccd.contracts import ProxyType
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
class SlabProxyAblationOption:
    name: str
    slab_count: int
    proxy_type_a: ProxyType = ProxyType.SWEPT_AABB
    proxy_type_b: ProxyType = ProxyType.SWEPT_AABB
    radius_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class SlabProxyAblationConfig:
    options: tuple[SlabProxyAblationOption, ...] = (
        SlabProxyAblationOption("slab1_aabb", 1, ProxyType.SWEPT_AABB, ProxyType.SWEPT_AABB),
        SlabProxyAblationOption("slab2_aabb", 2, ProxyType.SWEPT_AABB, ProxyType.SWEPT_AABB),
        SlabProxyAblationOption("slab4_aabb", 4, ProxyType.SWEPT_AABB, ProxyType.SWEPT_AABB),
        SlabProxyAblationOption("slab4_capsule", 4, ProxyType.CAPSULE, ProxyType.CAPSULE),
    )
    backend_name: str = "cpu_aabb_sort_sweep"
    same_query_only: bool = True
    min_candidate_recall: float = 1.0
    candidate_weight: float = 1.0
    raw_hit_weight: float = 0.05
    proxy_weight: float = 0.01
    proxy_cost_weight: float = 0.0
    slab_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class SlabProxyAblationRow:
    option_index: int
    option_name: str
    selected: bool
    feasible: bool
    query_count: int
    slab_count: int
    proxy_family: str
    proxy_type_a: ProxyType
    proxy_type_b: ProxyType
    radius_scale: float
    proxy_count: int
    raw_hit_count: int
    compact_candidate_count: int
    active_query_count: int
    positive_query_count: int
    covered_positive_count: int
    candidate_recall: float
    avg_candidates_per_query: float
    candidates_per_slab: float
    raw_hits_per_proxy: float
    avg_proxy_volume: float
    proxy_cost_units: float
    score: float
    broad_phase_ms: float
    exact_ms: float
    total_ms: float
    fn_count: int
    fp_count: int


@dataclass(frozen=True, slots=True)
class SlabProxyAblationResult:
    rows: tuple[SlabProxyAblationRow, ...]
    best_index: int
    source_name: str
    scene_name: str
    batch_id: str
    query_count: int

    @property
    def selected_row(self) -> SlabProxyAblationRow | None:
        for row in self.rows:
            if row.selected:
                return row
        return None


def slab_count_ablation_options(
    slab_counts: Iterable[int] = (1, 2, 4, 8),
    *,
    proxy_type: ProxyType = ProxyType.SWEPT_AABB,
) -> tuple[SlabProxyAblationOption, ...]:
    return tuple(
        SlabProxyAblationOption(
            name=f"slab{int(slab_count)}_{_proxy_type_token(proxy_type)}",
            slab_count=int(slab_count),
            proxy_type_a=proxy_type,
            proxy_type_b=proxy_type,
        )
        for slab_count in slab_counts
    )


def proxy_family_ablation_options(
    *,
    slab_count: int = 1,
) -> tuple[SlabProxyAblationOption, ...]:
    return (
        SlabProxyAblationOption(
            name=f"slab{slab_count}_aabb_aabb",
            slab_count=slab_count,
            proxy_type_a=ProxyType.SWEPT_AABB,
            proxy_type_b=ProxyType.SWEPT_AABB,
        ),
        SlabProxyAblationOption(
            name=f"slab{slab_count}_capsule_capsule",
            slab_count=slab_count,
            proxy_type_a=ProxyType.CAPSULE,
            proxy_type_b=ProxyType.CAPSULE,
        ),
        SlabProxyAblationOption(
            name=f"slab{slab_count}_aabb_capsule",
            slab_count=slab_count,
            proxy_type_a=ProxyType.SWEPT_AABB,
            proxy_type_b=ProxyType.CAPSULE,
        ),
    )


def validate_slab_proxy_ablation_config(config: SlabProxyAblationConfig) -> SlabProxyAblationConfig:
    if not config.options:
        raise ValueError("SlabProxyAblationConfig.options must not be empty")
    names: set[str] = set()
    for option in config.options:
        if not option.name:
            raise ValueError("slab/proxy ablation option name must not be empty")
        if option.name in names:
            raise ValueError(f"duplicate slab/proxy ablation option name: {option.name}")
        names.add(option.name)
        if option.slab_count <= 0:
            raise ValueError("slab_count must be positive")
        if option.proxy_type_a is ProxyType.UNKNOWN or option.proxy_type_b is ProxyType.UNKNOWN:
            raise ValueError("proxy_type_a/proxy_type_b must not be UNKNOWN")
        if not isinstance(option.proxy_type_a, ProxyType) or not isinstance(option.proxy_type_b, ProxyType):
            raise ValueError("proxy_type_a/proxy_type_b must be ProxyType values")
        if not math.isfinite(option.radius_scale) or option.radius_scale <= 0.0:
            raise ValueError("radius_scale must be finite and positive")
    if not 0.0 <= config.min_candidate_recall <= 1.0:
        raise ValueError("min_candidate_recall must be in [0, 1]")
    if not config.same_query_only:
        raise ValueError("slab/proxy ablation currently requires same_query_only=True")
    for name, value in (
        ("candidate_weight", config.candidate_weight),
        ("raw_hit_weight", config.raw_hit_weight),
        ("proxy_weight", config.proxy_weight),
        ("proxy_cost_weight", config.proxy_cost_weight),
        ("slab_weight", config.slab_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    return config


def _proxy_type_token(proxy_type: ProxyType) -> str:
    if proxy_type is ProxyType.SWEPT_AABB:
        return "aabb"
    if proxy_type is ProxyType.CAPSULE:
        return "capsule"
    return proxy_type.name.lower()


def _proxy_family(option: SlabProxyAblationOption) -> str:
    return f"{_proxy_type_token(option.proxy_type_a)}+{_proxy_type_token(option.proxy_type_b)}"


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


def _aabb_from_points(points: Sequence[Vec3], *, inflation: float) -> Aabb:
    if not points:
        raise ValueError("cannot build slab proxy AABB from an empty point set")
    if inflation < 0.0:
        raise ValueError("slab proxy AABB inflation must be non-negative")
    mins = tuple(min(point[axis] for point in points) - inflation for axis in range(3))
    maxs = tuple(max(point[axis] for point in points) + inflation for axis in range(3))
    return Aabb(min=mins, max=maxs)  # type: ignore[arg-type]


def _aabb_volume(aabb: Aabb) -> float:
    return max(0.0, aabb.max[0] - aabb.min[0]) * max(0.0, aabb.max[1] - aabb.min[1]) * max(
        0.0, aabb.max[2] - aabb.min[2]
    )


def _proxy_volume(proxy_type: ProxyType, aabb: Aabb, center_t0: Vec3, center_t1: Vec3, radius: float) -> float:
    if proxy_type is ProxyType.CAPSULE:
        length = _norm(_sub(center_t1, center_t0))
        return math.pi * radius * radius * length + 4.0 / 3.0 * math.pi * radius * radius * radius
    return _aabb_volume(aabb)


def _proxy_cost(proxy_type: ProxyType, center_t0: Vec3, center_t1: Vec3, radius: float) -> float:
    if proxy_type is ProxyType.CAPSULE:
        slenderness = _norm(_sub(center_t1, center_t0)) / max(radius, 1.0e-9)
        return 1.25 + 0.05 * min(20.0, slenderness)
    return 1.0


def _slab_interval(slab_id: int, slab_count: int) -> tuple[float, float]:
    t0 = float(slab_id) / float(slab_count)
    t1 = float(slab_id + 1) / float(slab_count)
    return t0, t1


def _slab_ids_for_positive_trace(trace: ExactOracleTrace, slab_count: int) -> tuple[int, ...]:
    if not trace.collided:
        return ()
    t0 = min(1.0, max(0.0, trace.contact_interval_t0))
    t1 = min(1.0, max(0.0, trace.contact_interval_t1))
    if t1 < t0:
        t0, t1 = t1, t0
    first = min(slab_count - 1, max(0, int(t0 * slab_count)))
    last_time = max(0.0, min(1.0 - 1.0e-12, t1))
    last = min(slab_count - 1, max(0, int(last_time * slab_count)))
    return tuple(range(first, last + 1))


def _slab_proxy_primitives_for_sample(
    sample: MotionDiscPairSample,
    option: SlabProxyAblationOption,
) -> tuple[tuple[BroadPhasePrimitive, ...], float, float]:
    primitives: list[BroadPhasePrimitive] = []
    total_volume = 0.0
    total_proxy_cost = 0.0
    roles = (
        (
            "a",
            sample.center_a_t0,
            sample.center_a_t1,
            sample.radius_a,
            sample.object_a_id,
            sample.patch_a_id,
            option.proxy_type_a,
            0,
        ),
        (
            "b",
            sample.center_b_t0,
            sample.center_b_t1,
            sample.radius_b,
            sample.object_b_id,
            sample.patch_b_id,
            option.proxy_type_b,
            100000,
        ),
    )
    for slab_id in range(option.slab_count):
        slab_t0, slab_t1 = _slab_interval(slab_id, option.slab_count)
        for role, full_t0, full_t1, radius, object_id, patch_id, proxy_type, role_offset in roles:
            center_t0 = _lerp(full_t0, full_t1, slab_t0)
            center_t1 = _lerp(full_t0, full_t1, slab_t1)
            proxy_radius = radius * option.radius_scale
            aabb = _aabb_from_points((center_t0, center_t1), inflation=proxy_radius)
            primitive_id = sample.query_id * 1000000 + slab_id * 1000 + role_offset
            total_volume += _proxy_volume(proxy_type, aabb, center_t0, center_t1, proxy_radius)
            total_proxy_cost += _proxy_cost(proxy_type, center_t0, center_t1, proxy_radius)
            primitives.append(
                BroadPhasePrimitive(
                    primitive_id=primitive_id,
                    query_id=sample.query_id,
                    role=role,
                    aabb=aabb,
                    family=f"slab_proxy_{slab_id}",
                    metadata={
                        "sample_id": sample.sample_id,
                        "object_id": object_id,
                        "patch_id": patch_id,
                        "slab_id": slab_id,
                        "slab_t0": slab_t0,
                        "slab_t1": slab_t1,
                        "proxy_type": proxy_type.name,
                        "proxy_radius": proxy_radius,
                        "ablation": option.name,
                    },
                )
            )
    return tuple(primitives), total_volume, total_proxy_cost


def _precompute_traces(samples: Sequence[MotionDiscPairSample]) -> dict[int, ExactOracleTrace]:
    return {sample.query_id: evaluate_swept_sphere_oracle(sample) for sample in samples}


def _score_row(
    *,
    config: SlabProxyAblationConfig,
    compact_candidate_count: int,
    raw_hit_count: int,
    proxy_count: int,
    proxy_cost_units: float,
    slab_count: int,
) -> float:
    return (
        config.candidate_weight * float(compact_candidate_count)
        + config.raw_hit_weight * float(raw_hit_count)
        + config.proxy_weight * float(proxy_count)
        + config.proxy_cost_weight * float(proxy_cost_units)
        + config.slab_weight * float(slab_count)
    )


def _evaluate_option(
    samples: Sequence[MotionDiscPairSample],
    traces_by_query_id: dict[int, ExactOracleTrace],
    option: SlabProxyAblationOption,
    option_index: int,
    config: SlabProxyAblationConfig,
    backend: BroadPhaseBackend,
) -> SlabProxyAblationRow:
    primitive_start = time.perf_counter()
    primitives: list[BroadPhasePrimitive] = []
    total_volume = 0.0
    proxy_cost_units = 0.0
    for sample in samples:
        sample_primitives, sample_volume, sample_proxy_cost = _slab_proxy_primitives_for_sample(sample, option)
        primitives.extend(sample_primitives)
        total_volume += sample_volume
        proxy_cost_units += sample_proxy_cost
    primitive_ms = (time.perf_counter() - primitive_start) * 1000.0

    primitive_by_id = {primitive.primitive_id: primitive for primitive in primitives}
    pairs, stats = backend.find_pairs(primitives, same_query_only=config.same_query_only)
    candidate_keys = {
        (
            pair.query_id,
            int(primitive_by_id[pair.primitive_a_id].metadata["slab_id"]),
        )
        for pair in pairs
    }
    active_query_ids = {query_id for query_id, _ in candidate_keys}
    active_slabs_by_query: dict[int, set[int]] = {}
    for query_id, slab_id in candidate_keys:
        active_slabs_by_query.setdefault(query_id, set()).add(slab_id)

    positive_slab_ids = {
        query_id: set(_slab_ids_for_positive_trace(trace, option.slab_count))
        for query_id, trace in traces_by_query_id.items()
        if trace.collided
    }
    covered_positive_count = sum(
        1
        for query_id, slab_ids in positive_slab_ids.items()
        if active_slabs_by_query.get(query_id, set()) & slab_ids
    )
    positive_query_count = len(positive_slab_ids)
    candidate_recall = 1.0 if positive_query_count == 0 else covered_positive_count / float(positive_query_count)

    exact_start = time.perf_counter()
    fn_count = 0
    fp_count = 0
    for sample in samples:
        trace = traces_by_query_id[sample.query_id]
        predicted_collision = False
        if trace.collided:
            predicted_collision = bool(
                active_slabs_by_query.get(sample.query_id, set())
                & set(_slab_ids_for_positive_trace(trace, option.slab_count))
            )
        if trace.collided and not predicted_collision:
            fn_count += 1
        if not trace.collided and predicted_collision:
            fp_count += 1
    exact_ms = (time.perf_counter() - exact_start) * 1000.0

    proxy_count = len(primitives)
    raw_hit_count = len(pairs)
    compact_candidate_count = len(candidate_keys)
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
        proxy_cost_units=proxy_cost_units,
        slab_count=option.slab_count,
    )
    broad_phase_ms = stats.elapsed_ms + primitive_ms
    total_ms = broad_phase_ms + exact_ms
    return SlabProxyAblationRow(
        option_index=option_index,
        option_name=option.name,
        selected=False,
        feasible=feasible,
        query_count=len(samples),
        slab_count=option.slab_count,
        proxy_family=_proxy_family(option),
        proxy_type_a=option.proxy_type_a,
        proxy_type_b=option.proxy_type_b,
        radius_scale=option.radius_scale,
        proxy_count=proxy_count,
        raw_hit_count=raw_hit_count,
        compact_candidate_count=compact_candidate_count,
        active_query_count=len(active_query_ids),
        positive_query_count=positive_query_count,
        covered_positive_count=covered_positive_count,
        candidate_recall=float(candidate_recall),
        avg_candidates_per_query=compact_candidate_count / max(1, len(samples)),
        candidates_per_slab=compact_candidate_count / max(1, option.slab_count),
        raw_hits_per_proxy=raw_hit_count / max(1, proxy_count),
        avg_proxy_volume=total_volume / max(1, proxy_count),
        proxy_cost_units=float(proxy_cost_units),
        score=float(score),
        broad_phase_ms=float(broad_phase_ms),
        exact_ms=float(exact_ms),
        total_ms=float(total_ms),
        fn_count=fn_count,
        fp_count=fp_count,
    )


def _select_best_row(rows: Sequence[SlabProxyAblationRow]) -> int:
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
    rows: Sequence[SlabProxyAblationRow],
    best_index: int,
) -> tuple[SlabProxyAblationRow, ...]:
    return tuple(
        SlabProxyAblationRow(
            option_index=row.option_index,
            option_name=row.option_name,
            selected=row.option_index == best_index,
            feasible=row.feasible,
            query_count=row.query_count,
            slab_count=row.slab_count,
            proxy_family=row.proxy_family,
            proxy_type_a=row.proxy_type_a,
            proxy_type_b=row.proxy_type_b,
            radius_scale=row.radius_scale,
            proxy_count=row.proxy_count,
            raw_hit_count=row.raw_hit_count,
            compact_candidate_count=row.compact_candidate_count,
            active_query_count=row.active_query_count,
            positive_query_count=row.positive_query_count,
            covered_positive_count=row.covered_positive_count,
            candidate_recall=row.candidate_recall,
            avg_candidates_per_query=row.avg_candidates_per_query,
            candidates_per_slab=row.candidates_per_slab,
            raw_hits_per_proxy=row.raw_hits_per_proxy,
            avg_proxy_volume=row.avg_proxy_volume,
            proxy_cost_units=row.proxy_cost_units,
            score=row.score,
            broad_phase_ms=row.broad_phase_ms,
            exact_ms=row.exact_ms,
            total_ms=row.total_ms,
            fn_count=row.fn_count,
            fp_count=row.fp_count,
        )
        for row in rows
    )


def run_slab_proxy_ablation_on_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    config: SlabProxyAblationConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
    source_name: str = "internal_analytic_oracle",
    scene_name: str = "programmatic_motion_disc_pairs",
    batch_id: str = "internal_samples",
) -> SlabProxyAblationResult:
    if not samples:
        raise ValueError("slab/proxy ablation requires at least one sample")
    cfg = validate_slab_proxy_ablation_config(config or SlabProxyAblationConfig())
    broad_phase = backend or CpuAabbBroadPhaseBackend(name=cfg.backend_name)
    traces_by_query_id = _precompute_traces(samples)
    rows = [
        _evaluate_option(samples, traces_by_query_id, option, index, cfg, broad_phase)
        for index, option in enumerate(cfg.options)
    ]
    best_index = _select_best_row(rows)
    return SlabProxyAblationResult(
        rows=_mark_selected(rows, best_index),
        best_index=best_index,
        source_name=source_name,
        scene_name=scene_name,
        batch_id=batch_id,
        query_count=len(samples),
    )


def run_slab_proxy_ablation_on_generated_dataset(
    dataset: GeneratedDataset,
    config: SlabProxyAblationConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> SlabProxyAblationResult:
    return run_slab_proxy_ablation_on_internal_samples(
        dataset.samples,
        config,
        backend=backend,
        source_name="internal_analytic_oracle",
        scene_name="generated_exact_oracle_dataset",
        batch_id="generated_dataset",
    )


def slab_proxy_ablation_csv_header() -> tuple[str, ...]:
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
        "slab_count",
        "proxy_family",
        "proxy_type_a",
        "proxy_type_b",
        "radius_scale",
        "proxy_count",
        "raw_hit_count",
        "compact_candidate_count",
        "active_query_count",
        "positive_query_count",
        "covered_positive_count",
        "candidate_recall",
        "avg_candidates_per_query",
        "candidates_per_slab",
        "raw_hits_per_proxy",
        "avg_proxy_volume",
        "proxy_cost_units",
        "score",
        "broad_phase_ms",
        "exact_ms",
        "total_ms",
        "fn_count",
        "fp_count",
    )


def slab_proxy_ablation_rows_to_csv(result: SlabProxyAblationResult) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=slab_proxy_ablation_csv_header(), lineterminator="\n")
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
                "slab_count": row.slab_count,
                "proxy_family": row.proxy_family,
                "proxy_type_a": row.proxy_type_a.name,
                "proxy_type_b": row.proxy_type_b.name,
                "radius_scale": row.radius_scale,
                "proxy_count": row.proxy_count,
                "raw_hit_count": row.raw_hit_count,
                "compact_candidate_count": row.compact_candidate_count,
                "active_query_count": row.active_query_count,
                "positive_query_count": row.positive_query_count,
                "covered_positive_count": row.covered_positive_count,
                "candidate_recall": row.candidate_recall,
                "avg_candidates_per_query": row.avg_candidates_per_query,
                "candidates_per_slab": row.candidates_per_slab,
                "raw_hits_per_proxy": row.raw_hits_per_proxy,
                "avg_proxy_volume": row.avg_proxy_volume,
                "proxy_cost_units": row.proxy_cost_units,
                "score": row.score,
                "broad_phase_ms": row.broad_phase_ms,
                "exact_ms": row.exact_ms,
                "total_ms": row.total_ms,
                "fn_count": row.fn_count,
                "fp_count": row.fp_count,
            }
        )
    return output.getvalue()
