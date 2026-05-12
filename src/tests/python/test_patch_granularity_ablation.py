from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench import (  # noqa: E402
    PatchGranularityAblationConfig,
    PatchGranularityAblationOption,
    patch_granularity_ablation_csv_header,
    patch_granularity_ablation_rows_to_csv,
    run_patch_granularity_ablation_on_generated_dataset,
    run_patch_granularity_ablation_on_internal_samples,
    validate_patch_granularity_ablation_config,
)
from p2cccd.contracts import ProxyType  # noqa: E402
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset  # noqa: E402
from p2cccd.data.samplers import MotionDiscPairSample, PairFamily  # noqa: E402


def test_patch_granularity_ablation_runs_all_options_and_selects_safe_row() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=2, seed=707)
    )
    config = PatchGranularityAblationConfig(
        options=(
            PatchGranularityAblationOption("coarse", 1),
            PatchGranularityAblationOption("medium", 2),
            PatchGranularityAblationOption("fine", 4),
        )
    )

    result = run_patch_granularity_ablation_on_generated_dataset(dataset, config)

    assert result.query_count == len(dataset.samples)
    assert len(result.rows) == 3
    assert result.best_index >= 0
    assert result.selected_row is not None
    assert result.selected_row.selected
    assert result.selected_row.feasible
    assert result.selected_row.fn_count == 0
    assert result.selected_row.fp_count == 0
    assert result.selected_row.candidate_recall == 1.0
    assert sum(1 for row in result.rows if row.selected) == 1


def test_patch_granularity_ablation_proxy_count_tracks_granularity() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=1, robot_link_count=0, seed=808)
    )
    config = PatchGranularityAblationConfig(
        options=(
            PatchGranularityAblationOption("p1", 1),
            PatchGranularityAblationOption("p2", 2),
            PatchGranularityAblationOption("p4", 4),
        )
    )

    result = run_patch_granularity_ablation_on_generated_dataset(dataset, config)
    proxy_counts = [row.proxy_count for row in result.rows]
    raw_hits = [row.raw_hit_count for row in result.rows]

    assert proxy_counts == sorted(proxy_counts)
    assert proxy_counts == [2 * len(dataset.samples), 4 * len(dataset.samples), 8 * len(dataset.samples)]
    assert raw_hits == sorted(raw_hits)
    assert all(row.patch_count == row.proxy_count for row in result.rows)


def test_patch_granularity_ablation_marks_nonconservative_option_infeasible() -> None:
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
    config = PatchGranularityAblationConfig(
        options=(
            PatchGranularityAblationOption("safe", 1, radius_scale=1.0),
            PatchGranularityAblationOption("unsafe_tiny", 1, radius_scale=1.0e-9),
        )
    )

    result = run_patch_granularity_ablation_on_internal_samples((sample,), config)
    unsafe = result.rows[1]

    assert result.rows[0].feasible
    assert not unsafe.feasible
    assert unsafe.candidate_recall < 1.0
    assert unsafe.fn_count > 0
    assert result.selected_row is result.rows[0]


def test_patch_granularity_ablation_csv_export_is_stable() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=1, robot_link_count=0, seed=1001)
    )
    result = run_patch_granularity_ablation_on_generated_dataset(dataset)

    csv_text = patch_granularity_ablation_rows_to_csv(result)

    assert csv_text.splitlines()[0] == ",".join(patch_granularity_ablation_csv_header())
    assert "coarse_1x" in csv_text
    assert "fine_4x" in csv_text
    assert len(csv_text.splitlines()) == len(result.rows) + 1


def test_patch_granularity_ablation_config_validation_rejects_bad_options() -> None:
    try:
        validate_patch_granularity_ablation_config(PatchGranularityAblationConfig(options=()))
    except ValueError as exc:
        assert "options" in str(exc)
    else:
        raise AssertionError("expected empty-options validation error")

    try:
        validate_patch_granularity_ablation_config(
            PatchGranularityAblationConfig(
                options=(PatchGranularityAblationOption("bad", 0),)
            )
        )
    except ValueError as exc:
        assert "patches_per_object" in str(exc)
    else:
        raise AssertionError("expected patches_per_object validation error")

    try:
        validate_patch_granularity_ablation_config(
            PatchGranularityAblationConfig(
                options=(PatchGranularityAblationOption("bad", 1, radius_scale=0.0),)
            )
        )
    except ValueError as exc:
        assert "radius_scale" in str(exc)
    else:
        raise AssertionError("expected radius_scale validation error")

    try:
        validate_patch_granularity_ablation_config(PatchGranularityAblationConfig(same_query_only=False))
    except ValueError as exc:
        assert "same_query_only" in str(exc)
    else:
        raise AssertionError("expected same_query_only validation error")
