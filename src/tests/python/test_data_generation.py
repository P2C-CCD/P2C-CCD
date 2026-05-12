from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

import numpy as np

from p2cccd.data import (  # noqa: E402
    DEFAULT_SPLITS,
    SPLIT_ROBOT_LINK,
    DatasetGenerationConfig,
    compute_label_metrics,
    dataset_to_npz_arrays,
    default_metadata,
    detect_warp,
    estimated_exact_work_reduction,
    evaluate_swept_sphere_oracle_with_optional_warp,
    evaluate_swept_sphere_oracle,
    family_topk_recall,
    generate_exact_oracle_dataset,
    generate_mesh_pair_motion_samples,
    generate_robot_link_pair_motion_samples,
    interval_top1_recall,
    read_npz_shard,
    samples_to_warp_arrays,
    write_npz_shard,
)
from p2cccd.proposal.features import (  # noqa: E402
    PROPOSAL_FAMILY_COUNT,
    PROPOSAL_FEATURE_DIM,
    PROPOSAL_INTERVAL_BIN_COUNT,
    validate_proposal_feature_row,
)


def test_mesh_and_robot_samplers_are_deterministic_and_cover_splits() -> None:
    mesh_samples_a = generate_mesh_pair_motion_samples(count_per_split=2, seed=11)
    mesh_samples_b = generate_mesh_pair_motion_samples(count_per_split=2, seed=11)
    robot_samples = generate_robot_link_pair_motion_samples(count=3, seed=7)

    assert mesh_samples_a == mesh_samples_b
    assert len(mesh_samples_a) == 2 * len(DEFAULT_SPLITS)
    assert {sample.split for sample in mesh_samples_a} == set(DEFAULT_SPLITS)
    assert len(robot_samples) == 3
    assert {sample.split for sample in robot_samples} == {SPLIT_ROBOT_LINK}


def test_exact_oracle_dataset_generates_valid_feature_rows_and_labels() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=2, seed=19)
    )

    assert len(dataset.rows) == 2 * len(DEFAULT_SPLITS) + 2
    assert len(dataset.rows) == len(dataset.samples) == len(dataset.traces)
    assert SPLIT_ROBOT_LINK in dataset.split_names
    assert any(trace.collided for trace in dataset.traces)
    assert any(not trace.collided for trace in dataset.traces)

    for row in dataset.rows:
        assert validate_proposal_feature_row(row) is row
        assert len(row.features) == PROPOSAL_FEATURE_DIM
        assert sum(row.interval_targets) == 1.0
        assert len(row.family_targets) == PROPOSAL_FAMILY_COUNT
        assert row.cost_target >= 1.0
        assert 0.0 <= row.priority_target <= 1.0

    arrays = dataset_to_npz_arrays(dataset)
    assert arrays["features"].shape == (len(dataset.rows), PROPOSAL_FEATURE_DIM)
    assert arrays["interval_targets"].shape == (len(dataset.rows), PROPOSAL_INTERVAL_BIN_COUNT)
    assert arrays["family_targets"].shape == (len(dataset.rows), PROPOSAL_FAMILY_COUNT)
    assert arrays["oracle_trace"].shape[0] == len(dataset.rows)
    assert arrays["split_ids"].dtype == np.int32


def test_oracle_trace_has_conservative_contact_semantics() -> None:
    samples = generate_mesh_pair_motion_samples(
        count_per_split=1,
        seed=3,
        splits=("multiple_contact_intervals",),
    )
    trace = evaluate_swept_sphere_oracle(samples[0])

    assert trace.collided
    assert 0.0 <= trace.toi <= 1.0
    assert trace.contact_interval_t0 <= trace.contact_interval_t1
    assert trace.safe_margin <= 0.0


def test_npz_shard_roundtrip_has_stable_schema(tmp_path: Path) -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=1, robot_link_count=1, seed=23)
    )
    path = tmp_path / "dataset_shard.npz"
    metadata = default_metadata(dataset, seed=23)

    write_npz_shard(path, dataset, metadata=metadata)
    loaded = read_npz_shard(path)

    assert loaded["metadata"]["schema_version"] == 1
    assert loaded["metadata"]["row_count"] == len(dataset.rows)
    assert loaded["metadata"]["split_names"] == list(dataset.split_names)
    assert loaded["arrays"]["features"].shape[1] == PROPOSAL_FEATURE_DIM


def test_data_metrics_match_perfect_labels() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=1, robot_link_count=0, seed=5)
    )
    label_metrics = compute_label_metrics(dataset.rows)
    interval_scores = [row.interval_targets for row in dataset.rows]
    family_scores = [row.family_targets for row in dataset.rows]

    assert label_metrics.row_count == len(dataset.rows)
    assert 0.0 <= label_metrics.positive_ratio <= 1.0
    assert interval_top1_recall(interval_scores, dataset.rows) == 1.0
    assert family_topk_recall(family_scores, dataset.rows, k=2) == 1.0
    assert estimated_exact_work_reduction(baseline_work=10.0, proposed_work=7.5) == 0.25


def test_warp_helpers_prepare_arrays_and_keep_cpu_reference_fallback() -> None:
    samples = generate_mesh_pair_motion_samples(count_per_split=1, seed=17)
    arrays = samples_to_warp_arrays(samples)
    result = evaluate_swept_sphere_oracle_with_optional_warp(samples, prefer_warp=True)
    availability = detect_warp()

    assert arrays["center_a_t0"].shape == (len(samples), 3)
    assert arrays["radius_sum"].shape == (len(samples),)
    assert result.warp_available == availability
    assert len(result.traces) == len(samples)
    assert result.backend in {"cpu_reference", "cpu_reference_with_warp_ready_arrays"}


def test_warp_helpers_can_require_warp_explicitly() -> None:
    samples = generate_mesh_pair_motion_samples(count_per_split=1, seed=18)
    if detect_warp().installed:
        result = evaluate_swept_sphere_oracle_with_optional_warp(samples, require_warp=True)
        assert len(result.traces) == len(samples)
    else:
        try:
            evaluate_swept_sphere_oracle_with_optional_warp(samples, require_warp=True)
        except RuntimeError as exc:
            assert "warp" in str(exc).lower()
        else:
            raise AssertionError("expected missing Warp runtime error")
