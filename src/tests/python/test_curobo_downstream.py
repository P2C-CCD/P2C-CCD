from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

import pytest  # noqa: E402

from p2cccd.bench import (  # noqa: E402
    CuRoboDownstreamConfig,
    run_benchmark_suite_from_config_path,
    run_curobo_downstream_on_generated_dataset,
    run_curobo_downstream_on_internal_samples,
    validate_curobo_downstream_config,
)
from p2cccd.contracts import ProxyType  # noqa: E402
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset  # noqa: E402
from p2cccd.data.samplers import MotionDiscPairSample, PairFamily  # noqa: E402


CONFIG_ROOT = PROJECT_ROOT / "configs" / "benchmark_suites"


def _robot_link_tunneling_sample() -> MotionDiscPairSample:
    return MotionDiscPairSample(
        sample_id=1,
        query_id=1,
        candidate_id=1,
        split="robot_link_validation",
        family=PairFamily.ROBOT_LINK_PAIR,
        object_a_id=110,
        patch_a_id=1001,
        object_b_id=120,
        patch_b_id=1101,
        slab_id=0,
        center_a_t0=(0.0, 0.0, 0.0),
        center_a_t1=(0.0, 0.0, 0.0),
        center_b_t0=(-1.0, 0.0, 0.0),
        center_b_t1=(1.0, 0.0, 0.0),
        radius_a=0.1,
        radius_b=0.1,
        proxy_type_a=ProxyType.CAPSULE,
        proxy_type_b=ProxyType.CAPSULE,
        hardness=1.0,
    )


def test_curobo_downstream_endpoint_sampling_can_miss_continuous_tunneling() -> None:
    result = run_curobo_downstream_on_internal_samples(
        (_robot_link_tunneling_sample(),),
        CuRoboDownstreamConfig(trajectory_step_count=2),
    )

    assert result.benchmark.query_count == 1
    assert result.benchmark.fn_count == 1
    assert result.benchmark.candidate_recall == 0.0
    assert result.stats.discrete_collision_count == 0
    assert result.stats.pose_pair_check_count == 2
    assert not result.final_fn_zero


def test_curobo_downstream_dense_sampling_covers_midpoint_tunneling() -> None:
    result = run_curobo_downstream_on_internal_samples(
        (_robot_link_tunneling_sample(),),
        CuRoboDownstreamConfig(trajectory_step_count=3),
    )

    assert result.benchmark.query_count == 1
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.stats.discrete_collision_count == 1
    assert result.stats.pose_pair_check_count == 3
    assert result.final_fn_zero


def test_curobo_downstream_runs_on_generated_robot_link_dataset() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=0, robot_link_count=4, seed=313)
    )

    result = run_curobo_downstream_on_generated_dataset(
        dataset,
        CuRoboDownstreamConfig(trajectory_step_count=8),
    )

    assert result.benchmark.query_count == 4
    assert result.stats.query_count == 4
    assert result.stats.robot_link_query_count == 4
    assert result.stats.pose_pair_check_count == 32
    assert result.stats.checker_ms >= 0.0


def test_curobo_downstream_rejects_invalid_config_and_empty_robot_dataset() -> None:
    with pytest.raises(ValueError):
        validate_curobo_downstream_config(CuRoboDownstreamConfig(trajectory_step_count=1))
    with pytest.raises(ValueError):
        validate_curobo_downstream_config(CuRoboDownstreamConfig(link_sphere_radius_scale=0.0))
    with pytest.raises(ValueError):
        validate_curobo_downstream_config(CuRoboDownstreamConfig(collision_activation_distance=-0.1))

    mesh_only_dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=1, robot_link_count=0, include_robot_links=False)
    )
    with pytest.raises(ValueError):
        run_curobo_downstream_on_generated_dataset(mesh_only_dataset, CuRoboDownstreamConfig())


def test_curobo_downstream_suite_config_runs_without_export() -> None:
    result = run_benchmark_suite_from_config_path(
        CONFIG_ROOT / "curobo_downstream.json",
        export=False,
        run_id="curobo_downstream_suite_unit",
        environment={
            "git_commit": "abc123",
            "host_name": "unit-host",
            "platform": "unit-platform",
            "python_version": "3.12.0",
            "gpu_name": "Unit GPU",
            "driver_version": "555.00",
            "cuda_version": "12.6",
            "optix_version": "8.0",
            "vram_total_mb": 24000,
            "vram_free_mb": 12000,
        },
    )

    assert len(result.rows) == 3
    assert {row.method_name for row in result.rows} == {"CuRoboDownstream"}
    assert result.case_results[0].method == "CuRoboDownstream"
    assert result.export_paths is None
