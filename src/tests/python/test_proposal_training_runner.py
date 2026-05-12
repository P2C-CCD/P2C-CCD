from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.data import (  # noqa: E402
    DatasetGenerationConfig,
    default_metadata,
    generate_exact_oracle_dataset,
    write_npz_shard,
)
from p2cccd.proposal import (  # noqa: E402
    STPFConfig,
    STPFTrainingConfig,
    STPFTrainingRunConfig,
    load_training_rows_from_npz_shards,
    run_stpf_training,
    run_stpf_training_from_npz_shards,
    summarize_training_result,
)


def _training_config() -> STPFTrainingConfig:
    return STPFTrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=2.0e-3,
        validation_fraction=0.25,
        seed=123,
        model_config=STPFConfig(hidden_dim=16, num_layers=1),
    )


def test_run_stpf_training_writes_history_model_and_summary(tmp_path: Path) -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=1, robot_link_count=1, seed=123)
    )
    run_config = STPFTrainingRunConfig(
        training=_training_config(),
        output_dir=str(tmp_path),
        run_name="unit_train",
    )

    run = run_stpf_training(dataset.rows, run_config)

    assert run.artifacts.history_csv is not None and run.artifacts.history_csv.exists()
    assert run.artifacts.history_jsonl is not None and run.artifacts.history_jsonl.exists()
    assert run.artifacts.model_state_path is not None and run.artifacts.model_state_path.exists()
    assert run.artifacts.summary_json.exists()
    assert run.final_train_loss >= 0.0
    assert run.result.train_row_count > 0

    summary = json.loads(run.artifacts.summary_json.read_text(encoding="utf-8"))
    assert summary["history_count"] == len(run.result.history)
    assert summarize_training_result(run.result)["train_row_count"] == run.result.train_row_count


def test_run_stpf_training_from_npz_shards_loads_rows_and_can_skip_model_save(tmp_path: Path) -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=1, robot_link_count=1, seed=124)
    )
    shard_path = tmp_path / "rows.npz"
    write_npz_shard(shard_path, dataset, metadata=default_metadata(dataset, seed=124))

    rows = load_training_rows_from_npz_shards((shard_path,))
    run = run_stpf_training_from_npz_shards(
        (shard_path,),
        STPFTrainingRunConfig(
            training=_training_config(),
            output_dir=str(tmp_path),
            run_name="unit_npz_train",
            save_model_state=False,
        ),
    )

    assert len(rows) == len(dataset.rows)
    assert run.artifacts.model_state_path is None
    assert run.artifacts.history_csv is not None and run.artifacts.history_csv.exists()
    assert run.result.train_row_count + run.result.validation_row_count == len(dataset.rows)
