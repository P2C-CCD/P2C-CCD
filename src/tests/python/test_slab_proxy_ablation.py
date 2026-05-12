from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench import (  # noqa: E402
    SlabProxyAblationConfig,
    SlabProxyAblationOption,
    proxy_family_ablation_options,
    run_slab_proxy_ablation_on_generated_dataset,
    run_slab_proxy_ablation_on_internal_samples,
    slab_count_ablation_options,
    slab_proxy_ablation_csv_header,
    slab_proxy_ablation_rows_to_csv,
    validate_slab_proxy_ablation_config,
)
from p2cccd.contracts import ProxyType  # noqa: E402
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset  # noqa: E402
from p2cccd.data.samplers import MotionDiscPairSample, PairFamily  # noqa: E402


def test_slab_proxy_ablation_runs_slab_count_options_and_preserves_recall() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=2, seed=1111)
    )
    config = SlabProxyAblationConfig(options=slab_count_ablation_options((1, 2, 4)))

    result = run_slab_proxy_ablation_on_generated_dataset(dataset, config)

    assert result.query_count == len(dataset.samples)
    assert [row.slab_count for row in result.rows] == [1, 2, 4]
    assert len(result.rows) == 3
    assert result.best_index >= 0
    assert result.selected_row is not None
    assert result.selected_row.feasible
    assert result.selected_row.fn_count == 0
    assert result.selected_row.fp_count == 0
    assert all(row.candidate_recall == 1.0 for row in result.rows)
    assert [row.proxy_count for row in result.rows] == [
        2 * len(dataset.samples),
        4 * len(dataset.samples),
        8 * len(dataset.samples),
    ]
    assert all(row.compact_candidate_count <= row.query_count * row.slab_count for row in result.rows)


def test_slab_proxy_ablation_reports_proxy_family_and_proxy_cost() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=1, robot_link_count=1, seed=1212)
    )
    config = SlabProxyAblationConfig(
        options=proxy_family_ablation_options(slab_count=2),
        candidate_weight=0.0,
        raw_hit_weight=0.0,
        proxy_weight=0.0,
        proxy_cost_weight=1.0,
    )

    result = run_slab_proxy_ablation_on_generated_dataset(dataset, config)
    families = [row.proxy_family for row in result.rows]

    assert families == ["aabb+aabb", "capsule+capsule", "aabb+capsule"]
    assert result.rows[0].proxy_type_a is ProxyType.SWEPT_AABB
    assert result.rows[1].proxy_type_a is ProxyType.CAPSULE
    assert result.rows[1].proxy_cost_units > result.rows[0].proxy_cost_units
    assert result.selected_row is result.rows[0]


def test_slab_proxy_ablation_marks_nonconservative_slab_option_infeasible() -> None:
    sample = MotionDiscPairSample(
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
        center_b_t0=(0.9, 0.0, 0.0),
        center_b_t1=(0.9, 0.0, 0.0),
        radius_a=0.5,
        radius_b=0.5,
        proxy_type_a=ProxyType.SWEPT_AABB,
        proxy_type_b=ProxyType.SWEPT_AABB,
        hardness=0.8,
    )
    config = SlabProxyAblationConfig(
        options=(
            SlabProxyAblationOption("safe", 2, radius_scale=1.0),
            SlabProxyAblationOption("unsafe_tiny", 2, radius_scale=1.0e-9),
        )
    )

    result = run_slab_proxy_ablation_on_internal_samples((sample,), config)
    unsafe = result.rows[1]

    assert result.rows[0].feasible
    assert not unsafe.feasible
    assert unsafe.candidate_recall < 1.0
    assert unsafe.fn_count > 0
    assert result.selected_row is result.rows[0]


def test_slab_proxy_ablation_csv_export_is_stable() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=1, robot_link_count=0, seed=1313)
    )
    result = run_slab_proxy_ablation_on_generated_dataset(dataset)

    csv_text = slab_proxy_ablation_rows_to_csv(result)

    assert csv_text.splitlines()[0] == ",".join(slab_proxy_ablation_csv_header())
    assert "slab1_aabb" in csv_text
    assert "slab4_capsule" in csv_text
    assert len(csv_text.splitlines()) == len(result.rows) + 1


def test_slab_proxy_ablation_config_validation_rejects_bad_options() -> None:
    try:
        validate_slab_proxy_ablation_config(SlabProxyAblationConfig(options=()))
    except ValueError as exc:
        assert "options" in str(exc)
    else:
        raise AssertionError("expected empty-options validation error")

    try:
        validate_slab_proxy_ablation_config(
            SlabProxyAblationConfig(options=(SlabProxyAblationOption("bad", 0),))
        )
    except ValueError as exc:
        assert "slab_count" in str(exc)
    else:
        raise AssertionError("expected slab_count validation error")

    try:
        validate_slab_proxy_ablation_config(
            SlabProxyAblationConfig(
                options=(
                    SlabProxyAblationOption(
                        "bad",
                        1,
                        proxy_type_a=ProxyType.UNKNOWN,
                        proxy_type_b=ProxyType.SWEPT_AABB,
                    ),
                )
            )
        )
    except ValueError as exc:
        assert "UNKNOWN" in str(exc)
    else:
        raise AssertionError("expected proxy type validation error")

    try:
        validate_slab_proxy_ablation_config(SlabProxyAblationConfig(same_query_only=False))
    except ValueError as exc:
        assert "same_query_only" in str(exc)
    else:
        raise AssertionError("expected same_query_only validation error")
