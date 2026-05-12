from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

import p2cccd.bench.bvh_exact as bvh_exact_module  # noqa: E402
from p2cccd.bench import (  # noqa: E402
    Aabb,
    BVHExactConfig,
    BroadPhasePrimitive,
    CpuAabbBroadPhaseBackend,
    PureExactCPUConfig,
    external_query_to_broad_phase_primitives,
    run_bvh_exact_on_external_batch,
    run_bvh_exact_on_generated_dataset,
)
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset  # noqa: E402
from p2cccd.datasets.ccd import (  # noqa: E402
    CCD_ADAPTER_SCHEMA_VERSION,
    CCDQueryFamily,
    ExternalCCDQuery,
    ScalableCCDSampleAdapter,
)


BASELINE_ROOT = PROJECT_ROOT / "baseline"


def _vf_query(*, z0: float, z1: float, label: bool) -> ExternalCCDQuery:
    return ExternalCCDQuery(
        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
        source_name="unit",
        scene_name="point_triangle",
        batch_id="pt",
        query_id=10,
        source_query_index=0,
        family=CCDQueryFamily.VERTEX_FACE,
        vertices_t0=((0.25, 0.25, z0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        vertices_t1=((0.25, 0.25, z1), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ground_truth_collides=label,
    )


def test_cpu_aabb_broad_phase_filters_non_overlapping_pairs() -> None:
    backend = CpuAabbBroadPhaseBackend()
    primitives = (
        BroadPhasePrimitive(
            primitive_id=1,
            query_id=1,
            role="a",
            aabb=Aabb(min=(0.0, 0.0, 0.0), max=(1.0, 1.0, 1.0)),
            family="unit",
        ),
        BroadPhasePrimitive(
            primitive_id=2,
            query_id=1,
            role="b",
            aabb=Aabb(min=(0.5, 0.5, 0.5), max=(1.5, 1.5, 1.5)),
            family="unit",
        ),
        BroadPhasePrimitive(
            primitive_id=3,
            query_id=2,
            role="a",
            aabb=Aabb(min=(10.0, 0.0, 0.0), max=(11.0, 1.0, 1.0)),
            family="unit",
        ),
        BroadPhasePrimitive(
            primitive_id=4,
            query_id=2,
            role="b",
            aabb=Aabb(min=(12.0, 0.0, 0.0), max=(13.0, 1.0, 1.0)),
            family="unit",
        ),
    )

    pairs, stats = backend.find_pairs(primitives)

    assert [(pair.query_id, pair.primitive_a_id, pair.primitive_b_id) for pair in pairs] == [(1, 1, 2)]
    assert stats.primitive_count == 4
    assert stats.pair_count == 1
    assert stats.backend_name == "cpu_aabb_sort_sweep"


def test_cpp_optix_backend_batches_same_family_queries_into_one_scene(monkeypatch) -> None:
    captured_scenes = []

    class _FakeAabb:
        def __init__(self) -> None:
            self.min = [0.0, 0.0, 0.0]
            self.max = [0.0, 0.0, 0.0]

    class _FakePatchMotionBound:
        def __init__(self) -> None:
            self.patch_id = 0
            self.t0 = 0.0
            self.t1 = 1.0
            self.translation_bound = 0.0
            self.rotation_angle = 0.0
            self.center_rotation_bound = 0.0
            self.surface_rotation_bound = 0.0
            self.radial_motion_bound = 0.0
            self.conservative_radius = 0.0

    class _FakeProxyPrimitive:
        def __init__(self) -> None:
            self.proxy_id = 0
            self.object_id = 0
            self.patch_id = 0
            self.slab_id = 0
            self.motion_segment_id = 0
            self.proxy_type = 0
            self.t0 = 0.0
            self.t1 = 1.0
            self.bounds = _FakeAabb()
            self.motion_bound = _FakePatchMotionBound()

    class _FakeProxyScene:
        def __init__(self) -> None:
            self.query_id = 0
            self.primitives = []

    class _FakeCandidate:
        def __init__(self, patch_a_id: int, patch_b_id: int) -> None:
            self.patch_a_id = patch_a_id
            self.patch_b_id = patch_b_id

    class _FakeTiming:
        def __init__(self) -> None:
            self.build_ms = 1.0
            self.update_ms = 0.5
            self.trace_ms = 2.0
            self.compact_ms = 0.25
            self.stats_ms = 0.125
            self.total_ms = 3.875

    class _FakeResult:
        def __init__(self, candidates) -> None:
            self.backend_name = "optix_rt"
            self.timing = _FakeTiming()
            self.candidates = candidates

    class _FakeProxyType:
        SWEPT_AABB = 1

    class _FakeCpp:
        ProxyScene = _FakeProxyScene
        ProxyPrimitive = _FakeProxyPrimitive
        Aabb = _FakeAabb
        PatchMotionBound = _FakePatchMotionBound
        ProxyType = _FakeProxyType

        @staticmethod
        def generate_candidates_for_proxy_scene(scene, backend_name="optix", allow_optix_cpu_fallback=True):
            captured_scenes.append(scene)
            candidates = []
            grouped = {}
            for primitive in scene.primitives:
                grouped.setdefault(primitive.slab_id, []).append(primitive)
            for slab_id, group in sorted(grouped.items()):
                assert len(group) == 2
                group = sorted(group, key=lambda primitive: primitive.patch_id)
                candidates.append(_FakeCandidate(group[0].patch_id, group[1].patch_id))
            return _FakeResult(candidates)

    monkeypatch.setattr(bvh_exact_module, "_try_load_p2cccd_cpp", lambda: _FakeCpp())

    primitives = (
        BroadPhasePrimitive(
            primitive_id=10,
            query_id=5,
            role="a",
            aabb=Aabb(min=(0.0, 0.0, 0.0), max=(1.0, 1.0, 1.0)),
            family="unit",
        ),
        BroadPhasePrimitive(
            primitive_id=11,
            query_id=5,
            role="b",
            aabb=Aabb(min=(0.25, 0.0, 0.0), max=(1.25, 1.0, 1.0)),
            family="unit",
        ),
        BroadPhasePrimitive(
            primitive_id=20,
            query_id=6,
            role="a",
            aabb=Aabb(min=(0.0, 0.0, 0.0), max=(2.0, 1.0, 1.0)),
            family="unit",
        ),
        BroadPhasePrimitive(
            primitive_id=21,
            query_id=6,
            role="b",
            aabb=Aabb(min=(0.5, 0.0, 0.0), max=(2.5, 1.0, 1.0)),
            family="unit",
        ),
    )

    pairs, stats = bvh_exact_module.CppOptixBroadPhaseBackend().find_pairs(primitives, same_query_only=True)

    assert [(pair.query_id, pair.primitive_a_id, pair.primitive_b_id) for pair in pairs] == [
        (5, 10, 11),
        (6, 20, 21),
    ]
    assert len(captured_scenes) == 1
    assert captured_scenes[0].query_id == 1
    assert [primitive.slab_id for primitive in captured_scenes[0].primitives] == [0, 0, 1, 1]
    assert captured_scenes[0].primitives[0].t0 == 0.0
    assert captured_scenes[0].primitives[0].t1 == 1.0
    assert captured_scenes[0].primitives[2].t0 == 2.0
    assert captured_scenes[0].primitives[2].t1 == 3.0
    assert captured_scenes[0].primitives[2].bounds.min[0] > captured_scenes[0].primitives[1].bounds.max[0]
    assert stats.backend_name == "optix_rt"
    assert stats.build_ms == 1.0
    assert stats.update_ms == 0.5
    assert stats.trace_ms == 2.0


def test_external_query_to_broad_phase_primitives_uses_feature_swept_aabbs() -> None:
    lhs, rhs = external_query_to_broad_phase_primitives(_vf_query(z0=1.0, z1=-1.0, label=True))

    assert lhs.role == "a"
    assert rhs.role == "b"
    assert lhs.family == "point_triangle"
    assert rhs.family == "point_triangle"
    assert lhs.aabb.min[2] == -1.0
    assert lhs.aabb.max[2] == 1.0
    assert rhs.aabb.min[2] == 0.0
    assert rhs.aabb.max[2] == 0.0
    assert lhs.aabb.overlaps(rhs.aabb)


def test_bvh_exact_runs_on_internal_generated_dataset_without_false_negatives() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=2, seed=202)
    )

    result = run_bvh_exact_on_generated_dataset(dataset)

    assert result.benchmark.query_count == len(dataset.rows)
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.benchmark.avg_candidates <= 1.0
    assert result.broad_phase_stats.pair_count <= len(dataset.rows)
    assert result.final_fn_zero


def test_bvh_exact_runs_on_external_scalable_sample_batch() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch("cloth-funnel", family="ee", step=227, limit=8)
    config = BVHExactConfig(
        exact=PureExactCPUConfig(eps_time=1.0e-5, eps_space=1.0e-8, max_subdivision_depth=24)
    )

    result = run_bvh_exact_on_external_batch(batch, config)

    assert result.source_name == "Sample-Scalable-CCD-Data"
    assert result.batch_id == "cloth-funnel:227ee"
    assert result.benchmark.query_count == 8
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.benchmark.avg_candidates <= 1.0
    assert result.broad_phase_stats.primitive_count == 16
    assert result.final_fn_zero


def test_bvh_exact_accepts_embree_compatible_backend_name_for_later_swap() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch("cloth-funnel", family="vf", step=227, limit=2)
    result = run_bvh_exact_on_external_batch(
        batch,
        BVHExactConfig(backend_name="embree_compatible"),
    )

    assert result.broad_phase_stats.backend_name == "embree_compatible"
    assert result.benchmark.query_count == 2


def test_bvh_exact_rejects_unknown_backend_name() -> None:
    try:
        run_bvh_exact_on_external_batch(
            ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data").load_query_batch(
                "cloth-funnel",
                family="vf",
                step=227,
                limit=1,
            ),
            BVHExactConfig(backend_name="unknown"),
        )
    except ValueError as exc:
        assert "backend_name" in str(exc)
    else:
        raise AssertionError("expected backend validation error")
