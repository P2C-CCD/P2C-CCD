from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

import pytest  # noqa: E402

from p2cccd.bench import (  # noqa: E402
    CabiNetStyleConfig,
    NeuralSVCDStyleConfig,
    run_benchmark_suite_from_config_path,
    run_cabinet_style_on_generated_dataset,
    run_cabinet_style_on_internal_samples,
    run_neural_svcd_style_on_generated_dataset,
    run_neural_svcd_style_on_internal_samples,
    validate_cabinet_style_config,
    validate_neural_svcd_style_config,
)
from p2cccd.contracts import ProxyType  # noqa: E402
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset  # noqa: E402
from p2cccd.data.samplers import MotionDiscPairSample, PairFamily  # noqa: E402


CONFIG_ROOT = PROJECT_ROOT / "configs" / "benchmark_suites"


def _tunneling_sample() -> MotionDiscPairSample:
    return MotionDiscPairSample(
        sample_id=1,
        query_id=1,
        candidate_id=1,
        split="unit",
        family=PairFamily.MESH_PAIR,
        object_a_id=10,
        patch_a_id=1,
        object_b_id=20,
        patch_b_id=2,
        slab_id=0,
        center_a_t0=(0.0, 0.0, 0.0),
        center_a_t1=(0.0, 0.0, 0.0),
        center_b_t0=(-1.0, 0.0, 0.0),
        center_b_t1=(1.0, 0.0, 0.0),
        radius_a=0.1,
        radius_b=0.1,
        proxy_type_a=ProxyType.SWEPT_AABB,
        proxy_type_b=ProxyType.SWEPT_AABB,
        hardness=1.0,
    )


def test_neural_svcd_style_time_surrogate_covers_tunneling() -> None:
    result = run_neural_svcd_style_on_internal_samples(
        (_tunneling_sample(),),
        NeuralSVCDStyleConfig(time_sample_count=9),
    )

    assert result.style_name == "NeuralSVCDStyle"
    assert result.benchmark.fn_count == 0
    assert result.benchmark.candidate_recall == 1.0
    assert result.stats.exact_candidate_count == 1
    assert result.final_fn_zero


def test_cabinet_style_endpoint_pose_surrogate_can_miss_tunneling() -> None:
    result = run_cabinet_style_on_internal_samples(
        (_tunneling_sample(),),
        CabiNetStyleConfig(pose_sample_count=2, conservative_fallback=False),
    )

    assert result.style_name == "CabiNetStyle"
    assert result.benchmark.fn_count == 1
    assert result.benchmark.candidate_recall == 0.0
    assert result.stats.exact_candidate_count == 0
    assert not result.final_fn_zero


def test_learned_style_comparisons_run_on_generated_dataset() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=1, seed=1919)
    )

    neural = run_neural_svcd_style_on_generated_dataset(dataset, NeuralSVCDStyleConfig())
    cabinet = run_cabinet_style_on_generated_dataset(dataset, CabiNetStyleConfig())

    assert neural.benchmark.query_count == len(dataset.samples)
    assert cabinet.benchmark.query_count == len(dataset.samples)
    assert neural.stats.query_count == len(dataset.samples)
    assert cabinet.stats.query_count == len(dataset.samples)
    assert neural.stats.avg_collision_score >= 0.0
    assert cabinet.stats.avg_uncertainty >= 0.0


def test_learned_style_config_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        validate_neural_svcd_style_config(NeuralSVCDStyleConfig(time_sample_count=1))
    with pytest.raises(ValueError):
        validate_neural_svcd_style_config(NeuralSVCDStyleConfig(collision_threshold=2.0))
    with pytest.raises(ValueError):
        validate_neural_svcd_style_config(NeuralSVCDStyleConfig(radius_scale=0.0))
    with pytest.raises(ValueError):
        validate_cabinet_style_config(CabiNetStyleConfig(pose_sample_count=1))
    with pytest.raises(ValueError):
        validate_cabinet_style_config(CabiNetStyleConfig(learned_proxy_radius_scale=0.0))


def test_learned_style_comparison_suite_config_runs_without_export() -> None:
    result = run_benchmark_suite_from_config_path(
        CONFIG_ROOT / "learned_style_comparison.json",
        export=False,
        run_id="learned_style_suite_unit",
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
    assert {row.method_name for row in result.rows} == {"NeuralSVCDStyle", "CabiNetStyle"}
    assert result.export_paths is None
