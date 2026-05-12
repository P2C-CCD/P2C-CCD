from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

for candidate in (
    PROJECT_ROOT / "build" / "cpp" / "Release",
    PROJECT_ROOT / "build" / "cpp" / "Debug",
    PROJECT_ROOT / "build" / "cpp",
):
    if candidate.exists():
        sys.path.insert(0, str(candidate))


def _cpp():
    try:
        import p2cccd_cpp  # type: ignore[import-not-found]
    except ImportError as exc:
        pytest.skip(f"p2cccd_cpp is not built: {exc}")
    return p2cccd_cpp


def _pose(cpp, x: float, y: float = 0.0, z: float = 0.0):
    pose = cpp.PoseSample()
    pose.translation = [x, y, z]
    pose.rotation_xyzw = [0.0, 0.0, 0.0, 1.0]
    return pose


def _static_motion(cpp, x: float):
    motion = cpp.MotionSegment()
    motion.t0 = 0.0
    motion.t1 = 1.0
    motion.pose_t0 = _pose(cpp, x)
    motion.pose_t1 = _pose(cpp, x)
    return motion


def _patch(cpp, patch_id: int, x: float):
    patch = cpp.Patch()
    patch.patch_id = patch_id
    patch.triangle_ids = [patch_id]
    patch.triangle_count = 1
    patch.area = 1.0
    patch.local_center = [x, 0.0, 0.0]
    patch.radius = 0.2
    return patch


def _overlapping_proxy_scene(cpp):
    object_a = cpp.ProxyObjectBuildInput()
    object_a.object_id = 10
    object_a.proxy_type = cpp.ProxyType.SWEPT_AABB
    object_a.patches = [_patch(cpp, 1, 0.0)]
    object_a.motion_segments = [_static_motion(cpp, 0.0)]
    object_a.slabs_per_motion_segment = 2
    object_a.eps_proxy = 0.05

    object_b = cpp.ProxyObjectBuildInput()
    object_b.object_id = 20
    object_b.proxy_type = cpp.ProxyType.CAPSULE
    object_b.patches = [_patch(cpp, 2, 0.25)]
    object_b.motion_segments = [_static_motion(cpp, 0.0)]
    object_b.slabs_per_motion_segment = 2
    object_b.eps_proxy = 0.05

    scene_input = cpp.ProxySceneBuildInput()
    scene_input.query_id = 42
    scene_input.objects = [object_a, object_b]
    return cpp.build_proxy_scene(scene_input)


def _vertex(cpp, feature_id: int, p0: tuple[float, float, float], p1: tuple[float, float, float]):
    vertex = cpp.LinearVertexTrajectory()
    vertex.feature_id = feature_id
    vertex.position_t0 = list(p0)
    vertex.position_t1 = list(p1)
    return vertex


def _moving_point_static_triangle(cpp):
    primitive = cpp.PointTriangleIntervalPrimitive()
    primitive.point_id = 10
    primitive.triangle_id = 20
    primitive.point = _vertex(cpp, 10, (0.25, 0.25, 1.0), (0.25, 0.25, -3.0))
    primitive.triangle_v0 = _vertex(cpp, 101, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    primitive.triangle_v1 = _vertex(cpp, 102, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    primitive.triangle_v2 = _vertex(cpp, 103, (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    return primitive


def _work_item(cpp):
    item = cpp.ExactWorkItem()
    item.work_item_id = 9001
    item.parent_candidate_id = 77
    item.query_id = 42
    item.patch_a_id = 1
    item.patch_b_id = 2
    item.interval_t0 = 0.0
    item.interval_t1 = 1.0
    item.feature_family_mask = cpp.FEATURE_FAMILY_POINT_TRIANGLE
    item.priority_score = 1.0
    item.source = cpp.ProposalSource.RAW
    return item


def _certificate_config(cpp):
    config = cpp.CertificateEngineConfig()
    config.eps_time = 1.0e-5
    config.eps_space = 1.0e-6
    config.max_subdivision_depth = 32
    return config


def test_cpp_contract_bindings_validate_core_records() -> None:
    cpp = _cpp()

    candidate = cpp.CandidateRecord()
    candidate.candidate_id = 1
    candidate.query_id = 42
    candidate.proxy_type_a = cpp.ProxyType.SWEPT_AABB
    candidate.proxy_type_b = cpp.ProxyType.CAPSULE
    candidate.rt_hit_count = 1
    candidate.motion_bound = [0.0, 0.0, 0.0, 0.1]
    assert cpp.validate_candidate_record(candidate)

    work_item = _work_item(cpp)
    assert cpp.validate_exact_work_item(work_item)

    benchmark = cpp.BenchmarkRow()
    benchmark.query_count = 2
    benchmark.candidate_recall = 1.0
    benchmark.fallback_ratio = 0.0
    assert cpp.validate_benchmark_row(benchmark)


def test_cpp_candidate_generation_api_builds_proxy_scene_and_candidates() -> None:
    cpp = _cpp()

    scene = _overlapping_proxy_scene(cpp)
    assert cpp.validate_proxy_scene(scene)
    assert scene.query_id == 42
    assert len(scene.primitives) == 4

    raw_buffer = cpp.generate_raw_candidates_cpu(scene)
    assert len(raw_buffer.hits) == 2

    compact = cpp.compact_raw_candidates(scene, raw_buffer)
    assert len(compact) == 2
    assert all(candidate.query_id == 42 for candidate in compact)

    result = cpp.generate_candidates_for_proxy_scene(scene)
    assert result.backend_name == "cpu_reference"
    assert len(result.candidates) == 2
    assert result.density.query_id == 42
    assert result.density.raw_hit_count == 2
    assert result.timing.total_ms >= 0.0


def test_cpp_exact_oracle_and_audit_replay_api() -> None:
    cpp = _cpp()

    query = cpp.ExactCertificateQuery()
    query.work_item = _work_item(cpp)
    query.config = _certificate_config(cpp)
    query.point_triangle_primitives = [_moving_point_static_triangle(cpp)]

    primitive_result = cpp.evaluate_point_triangle_interval(
        query.point_triangle_primitives[0],
        0.0,
        1.0,
        query.config,
    )
    assert primitive_result.status == cpp.CertificateStatus.COLLISION
    assert primitive_result.witness_id_a == 10
    assert primitive_result.witness_id_b == 20

    certificate = cpp.evaluate_certificate_query_cpu(query)
    assert certificate.status == cpp.CertificateStatus.COLLISION
    assert cpp.validate_certificate_result(certificate)

    queue_config = cpp.ExactWorkQueueConfig()
    queue_config.first_event_id = 500
    queue_config.first_timestamp_us = 800
    result = cpp.process_exact_work_queue_cpu([query], queue_config)
    assert result.processed_count == 1
    assert len(result.certificates) == 1
    assert len(result.audit_log) == 2
    assert cpp.validate_exact_work_queue_coverage([query], result)
    assert cpp.validate_audit_log_rows(result.audit_log)

    replay_rows = cpp.audit_log_rows_for_query(result.audit_log, 42)
    assert len(replay_rows) == 2
    assert replay_rows[0].action == cpp.EXACT_AUDIT_DEQUEUED


def test_cpp_cuda_aware_binding_entrypoints_are_safe_without_device_pointer_abi() -> None:
    cpp = _cpp()
    if not hasattr(cpp, "is_cuda_exact_built"):
        pytest.skip("p2cccd_cpp was built before CUDA-aware entrypoints were added")

    assert isinstance(cpp.is_cuda_exact_built(), bool)
    status = dict(cpp.cuda_binding_status())
    assert status["host_batch_exact_api"] is True
    assert status["device_pointer_abi"] is False
    assert cpp.CUDA_DEVICE_POINTER_ABI_ENABLED is False

    if not cpp.is_cuda_exact_built():
        with pytest.raises(ValueError, match="CUDA exact backend was not built"):
            cpp.evaluate_point_triangle_batch_cuda([], 0.0, 1.0, _certificate_config(cpp))
