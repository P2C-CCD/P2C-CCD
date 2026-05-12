from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

import torch
import numpy as np

from p2cccd.data import (  # noqa: E402
    DatasetGenerationConfig,
    default_metadata,
    generate_exact_oracle_dataset,
    write_npz_shard,
)
from p2cccd.proposal import (  # noqa: E402
    STPFConfig,
    STPFModel,
    STPFModelPreset,
    STPFTrainingConfig,
    evaluate_stpf_model,
    rows_from_npz_shard,
    run_stpf_training,
    STPFTrainingRunConfig,
    train_stpf_model,
    validate_training_config,
)


def test_train_stpf_model_runs_multitask_loop_and_reports_metrics() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=2, seed=31)
    )
    config = STPFTrainingConfig(
        epochs=2,
        batch_size=4,
        learning_rate=2.0e-3,
        seed=31,
        validation_fraction=0.25,
        model_config=STPFConfig(hidden_dim=24, num_layers=2),
    )

    result = train_stpf_model(dataset.rows, config)

    assert result.train_row_count > 0
    assert result.validation_row_count > 0
    assert len(result.history) == 4
    assert all(torch.isfinite(torch.tensor(metric.loss)) for metric in result.history)
    assert all(0.0 <= metric.interval_top1_recall <= 1.0 for metric in result.history)
    assert all(0.0 <= metric.family_top2_recall <= 1.0 for metric in result.history)


def test_evaluate_stpf_model_restores_training_state() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=1, robot_link_count=0, seed=7)
    )
    model = STPFModel(STPFConfig(hidden_dim=16, num_layers=1))
    model.train()
    config = STPFTrainingConfig(batch_size=2, model_config=model.config)

    metrics = evaluate_stpf_model(model, dataset.rows, config, epoch=0, split="smoke")

    assert model.training
    assert metrics.row_count == len(dataset.rows)
    assert metrics.mean_target_cost > 0.0


def test_training_rows_can_be_loaded_from_npz_shard(tmp_path: Path) -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=1, robot_link_count=1, seed=41)
    )
    shard_path = tmp_path / "train_rows.npz"
    write_npz_shard(shard_path, dataset, metadata=default_metadata(dataset, seed=41))

    rows = rows_from_npz_shard(shard_path)

    assert len(rows) == len(dataset.rows)
    assert rows[0].candidate_id == dataset.rows[0].candidate_id
    assert np.allclose(rows[0].features, dataset.rows[0].features, rtol=1.0e-6, atol=1.0e-6)


def test_training_config_validation_rejects_invalid_values() -> None:
    try:
        validate_training_config(STPFTrainingConfig(epochs=0))
    except ValueError as exc:
        assert "epochs" in str(exc)
    else:
        raise AssertionError("expected epochs validation error")


def test_run_stpf_training_saves_model_metadata_in_checkpoint(tmp_path: Path) -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=1, robot_link_count=0, seed=53)
    )
    run = run_stpf_training(
        dataset.rows,
        STPFTrainingRunConfig(
            training=STPFTrainingConfig(
                epochs=1,
                batch_size=4,
                seed=53,
                validation_fraction=0.0,
                model_preset=STPFModelPreset.MICRO_MLP,
            ),
            output_dir=str(tmp_path),
            run_name="metadata_checkpoint",
        ),
    )

    payload = torch.load(run.artifacts.model_state_path, map_location="cpu")

    assert payload["model_preset"] == "micro_mlp"
    assert payload["model_config"]["hidden_dim"] == 32
    assert payload["model_config"]["num_layers"] == 1
