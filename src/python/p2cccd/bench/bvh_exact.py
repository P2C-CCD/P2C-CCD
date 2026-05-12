from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
import os
from pathlib import Path
import sys
import time
from typing import Protocol, Sequence

from p2cccd.contracts import BenchmarkRow
from p2cccd.data.dataset import GeneratedDataset
from p2cccd.data.oracle import evaluate_swept_sphere_oracle
from p2cccd.data.samplers import MotionDiscPairSample
from p2cccd.datasets.ccd import CCDQueryFamily, DatasetQueryBatch, ExternalCCDQuery
from p2cccd.validators import validate_benchmark_row

from .pure_exact_cpu import (
    PureExactCPUConfig,
    PureExactQueryResult,
    evaluate_external_ccd_query,
)


Vec3 = tuple[float, float, float]
_OPTIX_BATCH_SCENE_QUERY_ID = 1
_OPTIX_BATCH_SLAB_STRIDE = 2.0
_OPTIX_BATCH_X_MARGIN = 1.0e-2


@dataclass(frozen=True, slots=True)
class Aabb:
    min: Vec3
    max: Vec3

    def overlaps(self, other: "Aabb") -> bool:
        return all(self.min[axis] <= other.max[axis] and other.min[axis] <= self.max[axis] for axis in range(3))


@dataclass(frozen=True, slots=True)
class BroadPhasePrimitive:
    primitive_id: int
    query_id: int
    role: str
    aabb: Aabb
    family: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BroadPhasePair:
    query_id: int
    primitive_a_id: int
    primitive_b_id: int
    family: str


@dataclass(frozen=True, slots=True)
class BroadPhaseStats:
    primitive_count: int
    pair_count: int
    backend_name: str
    elapsed_ms: float
    build_ms: float = 0.0
    update_ms: float = 0.0
    trace_ms: float = 0.0
    compact_ms: float = 0.0
    stats_ms: float = 0.0
    total_ms: float = 0.0


class BroadPhaseBackend(Protocol):
    name: str

    def find_pairs(
        self,
        primitives: Sequence[BroadPhasePrimitive],
        *,
        same_query_only: bool = True,
    ) -> tuple[tuple[BroadPhasePair, ...], BroadPhaseStats]:
        ...


@dataclass(frozen=True, slots=True)
class CpuAabbBroadPhaseBackend:
    """Deterministic CPU broad phase with an Embree/Coal-replaceable interface."""

    name: str = "cpu_aabb_sort_sweep"

    def find_pairs(
        self,
        primitives: Sequence[BroadPhasePrimitive],
        *,
        same_query_only: bool = True,
    ) -> tuple[tuple[BroadPhasePair, ...], BroadPhaseStats]:
        start = time.perf_counter()
        sorted_primitives = sorted(
            primitives,
            key=lambda primitive: (
                primitive.aabb.min[0],
                primitive.aabb.max[0],
                primitive.query_id,
                primitive.primitive_id,
            ),
        )
        pairs: list[BroadPhasePair] = []
        for index, lhs in enumerate(sorted_primitives):
            for rhs in sorted_primitives[index + 1 :]:
                if rhs.aabb.min[0] > lhs.aabb.max[0]:
                    break
                if lhs.role == rhs.role:
                    continue
                if same_query_only and lhs.query_id != rhs.query_id:
                    continue
                if lhs.family != rhs.family:
                    continue
                if lhs.aabb.overlaps(rhs.aabb):
                    first, second = (lhs, rhs) if lhs.role <= rhs.role else (rhs, lhs)
                    pairs.append(
                        BroadPhasePair(
                            query_id=lhs.query_id,
                            primitive_a_id=first.primitive_id,
                            primitive_b_id=second.primitive_id,
                            family=lhs.family,
                        )
                    )
        pairs.sort(key=lambda pair: (pair.query_id, pair.primitive_a_id, pair.primitive_b_id))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return (
            tuple(pairs),
            BroadPhaseStats(
                primitive_count=len(primitives),
                pair_count=len(pairs),
                backend_name=self.name,
                elapsed_ms=elapsed_ms,
                trace_ms=elapsed_ms,
                total_ms=elapsed_ms,
            ),
        )


def _translated_aabb_x(aabb: Aabb, x_offset: float) -> Aabb:
    return Aabb(
        min=(aabb.min[0] + x_offset, aabb.min[1], aabb.min[2]),
        max=(aabb.max[0] + x_offset, aabb.max[1], aabb.max[2]),
    )


def _batched_optix_scene_family_specs(
    primitives: Sequence[BroadPhasePrimitive],
) -> tuple[tuple[str, tuple[tuple[int, tuple[BroadPhasePrimitive, ...]], ...]], ...]:
    grouped: dict[str, dict[int, list[BroadPhasePrimitive]]] = {}
    for primitive in primitives:
        grouped.setdefault(primitive.family, {}).setdefault(primitive.query_id, []).append(primitive)
    specs: list[tuple[str, tuple[tuple[int, tuple[BroadPhasePrimitive, ...]], ...]]] = []
    for family, per_query in sorted(grouped.items()):
        queries: list[tuple[int, tuple[BroadPhasePrimitive, ...]]] = []
        for query_id, group in sorted(per_query.items()):
            queries.append(
                (
                    int(query_id),
                    tuple(sorted(group, key=lambda primitive: (primitive.primitive_id, primitive.role))),
                )
            )
        specs.append((family, tuple(queries)))
    return tuple(specs)


def _try_load_p2cccd_cpp():
    project_root = Path(__file__).resolve().parents[3]
    build_dirs = (
        project_root / "build_optix" / "cpp" / "Release",
        project_root / "build_optix" / "cpp" / "Debug",
        project_root / "build" / "cpp" / "Release",
        project_root / "build" / "cpp" / "Debug",
    )
    for build_dir in reversed(build_dirs):
        if build_dir.exists():
            if str(build_dir) not in sys.path:
                sys.path.insert(0, str(build_dir))
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(build_dir))
                except OSError:
                    pass

    cuda_root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_root and hasattr(os, "add_dll_directory"):
        cuda_bin = Path(cuda_root) / "bin"
        if cuda_bin.exists():
            try:
                os.add_dll_directory(str(cuda_bin))
            except OSError:
                pass

    try:
        return importlib.import_module("p2cccd_cpp")
    except ImportError:
        return None


@dataclass(frozen=True, slots=True)
class CppOptixBroadPhaseBackend:
    """OptiX-backed broad phase through the C++ proxy-scene pybind API."""

    name: str = "optix_rt"
    allow_cpu_fallback: bool = True

    def find_pairs(
        self,
        primitives: Sequence[BroadPhasePrimitive],
        *,
        same_query_only: bool = True,
    ) -> tuple[tuple[BroadPhasePair, ...], BroadPhaseStats]:
        if not same_query_only:
            if not self.allow_cpu_fallback:
                raise RuntimeError("CppOptixBroadPhaseBackend currently requires same_query_only=True")
            return CpuAabbBroadPhaseBackend(name="optix_cpu_fallback").find_pairs(
                primitives,
                same_query_only=same_query_only,
            )

        cpp = _try_load_p2cccd_cpp()
        required = ("ProxyScene", "ProxyPrimitive", "Aabb", "PatchMotionBound", "generate_candidates_for_proxy_scene")
        if cpp is None or any(not hasattr(cpp, name) for name in required):
            if not self.allow_cpu_fallback:
                raise RuntimeError("p2cccd_cpp OptiX candidate binding is unavailable")
            return CpuAabbBroadPhaseBackend(name="optix_cpu_fallback").find_pairs(
                primitives,
                same_query_only=same_query_only,
            )

        start = time.perf_counter()
        pairs: list[BroadPhasePair] = []
        used_backend = self.name
        total_build_ms = 0.0
        total_update_ms = 0.0
        total_trace_ms = 0.0
        total_compact_ms = 0.0
        total_stats_ms = 0.0
        total_candidate_ms = 0.0
        for family, grouped_queries in _batched_optix_scene_family_specs(primitives):
            scene = cpp.ProxyScene()
            scene.query_id = _OPTIX_BATCH_SCENE_QUERY_ID
            proxy_primitives = []
            primitive_query_by_id: dict[int, int] = {}
            x_cursor = 0.0
            for slab_id, (query_id, group) in enumerate(grouped_queries):
                group_min_x = min(primitive.aabb.min[0] for primitive in group)
                group_max_x = max(primitive.aabb.max[0] for primitive in group)
                x_offset = x_cursor - group_min_x
                interval_t0 = float(slab_id) * _OPTIX_BATCH_SLAB_STRIDE
                interval_t1 = interval_t0 + 1.0
                for primitive in group:
                    proxy = cpp.ProxyPrimitive()
                    proxy.proxy_id = len(proxy_primitives)
                    proxy.object_id = 1 if primitive.role == "a" else 2
                    proxy.patch_id = int(primitive.primitive_id)
                    proxy.slab_id = slab_id
                    proxy.motion_segment_id = 0
                    proxy.proxy_type = cpp.ProxyType.SWEPT_AABB
                    proxy.t0 = interval_t0
                    proxy.t1 = interval_t1
                    translated_aabb = _translated_aabb_x(primitive.aabb, x_offset)
                    bounds = cpp.Aabb()
                    bounds.min = [float(value) for value in translated_aabb.min]
                    bounds.max = [float(value) for value in translated_aabb.max]
                    proxy.bounds = bounds
                    motion_bound = cpp.PatchMotionBound()
                    motion_bound.patch_id = int(primitive.primitive_id)
                    motion_bound.t0 = interval_t0
                    motion_bound.t1 = interval_t1
                    motion_bound.translation_bound = 0.0
                    motion_bound.rotation_angle = 0.0
                    motion_bound.center_rotation_bound = 0.0
                    motion_bound.surface_rotation_bound = 0.0
                    motion_bound.radial_motion_bound = 0.0
                    motion_bound.conservative_radius = max(
                        0.0,
                        max(
                            translated_aabb.max[axis] - translated_aabb.min[axis]
                            for axis in range(3)
                        ) * 0.5,
                    )
                    proxy.motion_bound = motion_bound
                    proxy_primitives.append(proxy)
                    primitive_query_by_id[int(primitive.primitive_id)] = int(query_id)
                x_cursor += (group_max_x - group_min_x) + _OPTIX_BATCH_X_MARGIN
            scene.primitives = proxy_primitives
            try:
                result = cpp.generate_candidates_for_proxy_scene(
                    scene,
                    backend_name="optix",
                    allow_optix_cpu_fallback=self.allow_cpu_fallback,
                )
            except Exception:
                if not self.allow_cpu_fallback:
                    raise
                return CpuAabbBroadPhaseBackend(name="optix_cpu_fallback").find_pairs(
                    primitives,
                    same_query_only=same_query_only,
                )
            used_backend = str(result.backend_name)
            total_build_ms += float(getattr(result.timing, "build_ms", 0.0))
            total_update_ms += float(getattr(result.timing, "update_ms", 0.0))
            total_trace_ms += float(getattr(result.timing, "trace_ms", 0.0))
            total_compact_ms += float(getattr(result.timing, "compact_ms", 0.0))
            total_stats_ms += float(getattr(result.timing, "stats_ms", 0.0))
            total_candidate_ms += float(getattr(result.timing, "total_ms", 0.0))
            for candidate in result.candidates:
                candidate_query_id = primitive_query_by_id.get(int(candidate.patch_a_id))
                other_query_id = primitive_query_by_id.get(int(candidate.patch_b_id))
                if candidate_query_id is None or other_query_id is None:
                    raise RuntimeError("OptiX candidate references an unknown primitive_id")
                if candidate_query_id != other_query_id:
                    raise RuntimeError("OptiX batched traversal produced a cross-query candidate")
                pairs.append(
                    BroadPhasePair(
                        query_id=candidate_query_id,
                        primitive_a_id=int(candidate.patch_a_id),
                        primitive_b_id=int(candidate.patch_b_id),
                        family=family,
                    )
                )

        pairs.sort(key=lambda pair: (pair.query_id, pair.primitive_a_id, pair.primitive_b_id))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return (
            tuple(pairs),
            BroadPhaseStats(
                primitive_count=len(primitives),
                pair_count=len(pairs),
                backend_name=used_backend,
                elapsed_ms=elapsed_ms,
                build_ms=total_build_ms,
                update_ms=total_update_ms,
                trace_ms=total_trace_ms,
                compact_ms=total_compact_ms,
                stats_ms=total_stats_ms,
                total_ms=total_candidate_ms if total_candidate_ms > 0.0 else elapsed_ms,
            ),
        )


@dataclass(frozen=True, slots=True)
class BVHExactConfig:
    exact: PureExactCPUConfig = field(default_factory=PureExactCPUConfig)
    backend_name: str = "cpu_aabb_sort_sweep"
    same_query_only: bool = True


@dataclass(frozen=True, slots=True)
class BVHExactResult:
    benchmark: BenchmarkRow
    query_results: tuple[PureExactQueryResult, ...]
    broad_phase_stats: BroadPhaseStats
    source_name: str
    scene_name: str
    batch_id: str

    @property
    def final_fn_zero(self) -> bool:
        return self.benchmark.fn_count == 0


def _validate_config(config: BVHExactConfig) -> BVHExactConfig:
    if config.backend_name not in {"cpu_aabb_sort_sweep", "embree_compatible", "coal_compatible"}:
        raise ValueError("BVHExactConfig.backend_name is unsupported")
    return config


def _aabb_from_points(points: Sequence[Vec3], *, inflation: float = 0.0) -> Aabb:
    if not points:
        raise ValueError("cannot build AABB from an empty point set")
    if inflation < 0.0:
        raise ValueError("AABB inflation must be non-negative")
    mins = tuple(min(point[axis] for point in points) - inflation for axis in range(3))
    maxs = tuple(max(point[axis] for point in points) + inflation for axis in range(3))
    return Aabb(min=mins, max=maxs)  # type: ignore[arg-type]


def _external_feature_indices(family: CCDQueryFamily) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    if family is CCDQueryFamily.VERTEX_FACE:
        return (0,), (1, 2, 3), family.p2cccd_witness_family
    if family is CCDQueryFamily.EDGE_EDGE:
        return (0, 1), (2, 3), family.p2cccd_witness_family
    raise ValueError(f"unsupported external CCD query family: {family}")


def external_query_to_broad_phase_primitives(query: ExternalCCDQuery) -> tuple[BroadPhasePrimitive, BroadPhasePrimitive]:
    lhs_indices, rhs_indices, family = _external_feature_indices(query.family)
    lhs_points = [query.vertices_t0[index] for index in lhs_indices] + [
        query.vertices_t1[index] for index in lhs_indices
    ]
    rhs_points = [query.vertices_t0[index] for index in rhs_indices] + [
        query.vertices_t1[index] for index in rhs_indices
    ]
    base_id = query.query_id * 2
    return (
        BroadPhasePrimitive(
            primitive_id=base_id,
            query_id=query.query_id,
            role="a",
            aabb=_aabb_from_points(lhs_points),
            family=family,
            metadata={"source_query_index": query.source_query_index},
        ),
        BroadPhasePrimitive(
            primitive_id=base_id + 1,
            query_id=query.query_id,
            role="b",
            aabb=_aabb_from_points(rhs_points),
            family=family,
            metadata={"source_query_index": query.source_query_index},
        ),
    )


def internal_sample_to_broad_phase_primitives(sample: MotionDiscPairSample) -> tuple[BroadPhasePrimitive, BroadPhasePrimitive]:
    def sphere_aabb(center_t0: Vec3, center_t1: Vec3, radius: float) -> Aabb:
        return _aabb_from_points((center_t0, center_t1), inflation=radius)

    base_id = sample.query_id * 2
    return (
        BroadPhasePrimitive(
            primitive_id=base_id,
            query_id=sample.query_id,
            role="a",
            aabb=sphere_aabb(sample.center_a_t0, sample.center_a_t1, sample.radius_a),
            family="swept_sphere_proxy",
            metadata={"sample_id": sample.sample_id},
        ),
        BroadPhasePrimitive(
            primitive_id=base_id + 1,
            query_id=sample.query_id,
            role="b",
            aabb=sphere_aabb(sample.center_b_t0, sample.center_b_t1, sample.radius_b),
            family="swept_sphere_proxy",
            metadata={"sample_id": sample.sample_id},
        ),
    )


def _make_benchmark_row(
    query_results: Sequence[PureExactQueryResult],
    *,
    broad_phase_stats: BroadPhaseStats,
    exact_elapsed_ms: float,
    total_query_count: int,
) -> BenchmarkRow:
    if total_query_count <= 0:
        raise ValueError("BVHExact requires at least one query")
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
    total_ms = broad_phase_stats.elapsed_ms + exact_elapsed_ms
    row = BenchmarkRow(
        query_count=total_query_count,
        fn_count=fn_count,
        fp_count=fp_count,
        candidate_recall=1.0 if fn_count == 0 else 1.0 - (fn_count / max(1, len(known))),
        avg_candidates=broad_phase_stats.pair_count / total_query_count,
        avg_exact_evals=total_exact_evals / total_query_count,
        avg_subdivision_depth=total_depth / total_query_count,
        fallback_ratio=0.0,
        rt_ms=0.0,
        proposal_ms=0.0,
        exact_ms=exact_elapsed_ms,
        total_ms=total_ms,
        qps=0.0 if total_ms <= 0.0 else 1000.0 * total_query_count / total_ms,
    )
    return validate_benchmark_row(row)


def _separation_result_for_culled_external_query(query: ExternalCCDQuery) -> PureExactQueryResult:
    return PureExactQueryResult(
        query_id=query.query_id,
        family=query.family.p2cccd_witness_family,
        predicted_collision=False,
        ground_truth_collision=query.ground_truth_collides,
        status="broad_phase_separation",
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
        status="broad_phase_separation",
        toi_upper=1.0,
        safe_margin_lb=max(0.0, trace.safe_margin),
        exact_evals=0,
        max_depth=0,
    )


def run_bvh_exact_on_external_batch(
    batch: DatasetQueryBatch,
    config: BVHExactConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> BVHExactResult:
    cfg = _validate_config(config or BVHExactConfig())
    broad_phase = backend or CpuAabbBroadPhaseBackend(name=cfg.backend_name)
    primitives: list[BroadPhasePrimitive] = []
    query_by_id = {query.query_id: query for query in batch.queries}
    for query in batch.queries:
        primitives.extend(external_query_to_broad_phase_primitives(query))
    pairs, stats = broad_phase.find_pairs(primitives, same_query_only=cfg.same_query_only)
    active_query_ids = {pair.query_id for pair in pairs}

    start = time.perf_counter()
    results: list[PureExactQueryResult] = []
    for query in batch.queries:
        if query.query_id in active_query_ids:
            results.append(evaluate_external_ccd_query(query, cfg.exact))
        else:
            results.append(_separation_result_for_culled_external_query(query))
    exact_elapsed_ms = (time.perf_counter() - start) * 1000.0
    if active_query_ids - set(query_by_id):
        raise ValueError("broad phase produced a pair for an unknown query")
    return BVHExactResult(
        benchmark=_make_benchmark_row(
            results,
            broad_phase_stats=stats,
            exact_elapsed_ms=exact_elapsed_ms,
            total_query_count=len(batch.queries),
        ),
        query_results=tuple(results),
        broad_phase_stats=stats,
        source_name=batch.source_name,
        scene_name=batch.scene_name,
        batch_id=batch.batch_id,
    )


def run_bvh_exact_on_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    config: BVHExactConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> BVHExactResult:
    if not samples:
        raise ValueError("BVHExact internal sample run requires at least one sample")
    cfg = _validate_config(config or BVHExactConfig())
    broad_phase = backend or CpuAabbBroadPhaseBackend(name=cfg.backend_name)
    primitives: list[BroadPhasePrimitive] = []
    sample_by_query_id = {sample.query_id: sample for sample in samples}
    for sample in samples:
        primitives.extend(internal_sample_to_broad_phase_primitives(sample))
    pairs, stats = broad_phase.find_pairs(primitives, same_query_only=cfg.same_query_only)
    active_query_ids = {pair.query_id for pair in pairs}

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
    if active_query_ids - set(sample_by_query_id):
        raise ValueError("broad phase produced a pair for an unknown internal sample")
    return BVHExactResult(
        benchmark=_make_benchmark_row(
            results,
            broad_phase_stats=stats,
            exact_elapsed_ms=exact_elapsed_ms,
            total_query_count=len(samples),
        ),
        query_results=tuple(results),
        broad_phase_stats=stats,
        source_name="internal_analytic_oracle",
        scene_name="programmatic_motion_disc_pairs",
        batch_id="internal_samples",
    )


def run_bvh_exact_on_generated_dataset(
    dataset: GeneratedDataset,
    config: BVHExactConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> BVHExactResult:
    return run_bvh_exact_on_internal_samples(dataset.samples, config, backend=backend)
