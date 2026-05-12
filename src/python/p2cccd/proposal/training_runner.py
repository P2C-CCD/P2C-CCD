from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from .features import ProposalFeatureRow
from .training import (
    STPFEpochMetrics,
    STPFTrainingConfig,
    STPFTrainingResult,
    rows_from_npz_shard,
    train_stpf_model,
)
from .stpf_model import stpf_config_to_dict


@dataclass(frozen=True, slots=True)
class STPFTrainingRunConfig:
    training: STPFTrainingConfig = STPFTrainingConfig()
    output_dir: str = "src/outputs/stpf_training"
    run_name: str = "stpf_training_run"
    save_model_state: bool = True
    save_history_csv: bool = True
    save_history_jsonl: bool = True


@dataclass(frozen=True, slots=True)
class STPFTrainingArtifacts:
    output_dir: Path
    history_csv: Path | None
    history_jsonl: Path | None
    model_state_path: Path | None
    summary_json: Path


@dataclass(frozen=True, slots=True)
class STPFTrainingRunResult:
    result: STPFTrainingResult
    artifacts: STPFTrainingArtifacts

    @property
    def final_train_loss(self) -> float:
        train_metrics = [metric for metric in self.result.history if metric.split == "train"]
        return train_metrics[-1].loss if train_metrics else 0.0

    @property
    def final_validation_loss(self) -> float:
        validation_metrics = [metric for metric in self.result.history if metric.split == "validation"]
        return validation_metrics[-1].loss if validation_metrics else 0.0


def load_training_rows_from_npz_shards(paths: Sequence[str | Path]) -> list[ProposalFeatureRow]:
    rows: list[ProposalFeatureRow] = []
    for path in paths:
        rows.extend(rows_from_npz_shard(path))
    if not rows:
        raise ValueError("no ProposalFeatureRow rows loaded from NPZ shards")
    return rows


def training_history_to_dicts(history: Sequence[STPFEpochMetrics]) -> list[dict[str, object]]:
    return [asdict(metric) for metric in history]


def write_training_history_csv(path: str | Path, history: Sequence[STPFEpochMetrics]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = training_history_to_dicts(history)
    if not rows:
        raise ValueError("cannot write empty STPF training history")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_training_history_jsonl(path: str | Path, history: Sequence[STPFEpochMetrics]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in training_history_to_dicts(history):
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    return output_path


def summarize_training_result(result: STPFTrainingResult) -> dict[str, object]:
    train_metrics = [metric for metric in result.history if metric.split == "train"]
    validation_metrics = [metric for metric in result.history if metric.split == "validation"]
    final_train = train_metrics[-1] if train_metrics else None
    final_validation = validation_metrics[-1] if validation_metrics else None
    return {
        "train_row_count": result.train_row_count,
        "validation_row_count": result.validation_row_count,
        "history_count": len(result.history),
        "final_train_loss": None if final_train is None else final_train.loss,
        "final_validation_loss": None if final_validation is None else final_validation.loss,
        "final_train_interval_top1_recall": None if final_train is None else final_train.interval_top1_recall,
        "final_validation_interval_top1_recall": None
        if final_validation is None
        else final_validation.interval_top1_recall,
        "model_preset": str(result.config.model_preset),
        "epochs": result.config.epochs,
        "batch_size": result.config.batch_size,
        "seed": result.config.seed,
    }


def run_stpf_training(
    rows: Sequence[ProposalFeatureRow],
    run_config: STPFTrainingRunConfig | None = None,
    *,
    validation_rows: Sequence[ProposalFeatureRow] | None = None,
) -> STPFTrainingRunResult:
    cfg = run_config or STPFTrainingRunConfig()
    output_dir = Path(cfg.output_dir) / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    result = train_stpf_model(rows, cfg.training, validation_rows=validation_rows)
    history_csv = write_training_history_csv(output_dir / "history.csv", result.history) if cfg.save_history_csv else None
    history_jsonl = (
        write_training_history_jsonl(output_dir / "history.jsonl", result.history)
        if cfg.save_history_jsonl
        else None
    )
    model_state_path = output_dir / "model_state.pt" if cfg.save_model_state else None
    if model_state_path is not None:
        torch.save(
            {
                "state_dict": result.model.state_dict(),
                "model_config": stpf_config_to_dict(result.model.config),
                "model_preset": str(result.config.model_preset),
                "epochs": int(result.config.epochs),
                "batch_size": int(result.config.batch_size),
                "seed": int(result.config.seed),
            },
            model_state_path,
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summarize_training_result(result), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    return STPFTrainingRunResult(
        result=result,
        artifacts=STPFTrainingArtifacts(
            output_dir=output_dir,
            history_csv=history_csv,
            history_jsonl=history_jsonl,
            model_state_path=model_state_path,
            summary_json=summary_path,
        ),
    )


def run_stpf_training_from_npz_shards(
    shard_paths: Sequence[str | Path],
    run_config: STPFTrainingRunConfig | None = None,
) -> STPFTrainingRunResult:
    return run_stpf_training(load_training_rows_from_npz_shards(shard_paths), run_config)
