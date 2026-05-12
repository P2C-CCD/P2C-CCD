from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Sequence

import torch

from .features import (
    ProposalFeatureRow,
    rows_to_feature_tensor,
    rows_to_target_tensors,
    validate_proposal_feature_row,
)
from .stpf_model import (
    STPFConfig,
    STPFModel,
    STPFModelPreset,
    build_stpf_model,
    stpf_cost_aware_loss,
    stpf_multitask_loss,
)


@dataclass(frozen=True, slots=True)
class STPFTrainingConfig:
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    seed: int = 13
    device: str = "cpu"
    validation_fraction: float = 0.2
    shuffle: bool = True
    grad_clip_norm: float = 1.0
    cost_aware_weight: float = 0.25
    model_preset: STPFModelPreset | str = STPFModelPreset.LIGHTWEIGHT_MLP
    model_config: STPFConfig | None = None


@dataclass(frozen=True, slots=True)
class STPFEpochMetrics:
    epoch: int
    split: str
    row_count: int
    loss: float
    interval_top1_recall: float
    family_top2_recall: float
    estimated_exact_work_reduction: float
    mean_predicted_cost: float
    mean_target_cost: float


@dataclass(slots=True)
class STPFTrainingResult:
    model: STPFModel
    config: STPFTrainingConfig
    train_row_count: int
    validation_row_count: int
    history: list[STPFEpochMetrics]


def validate_training_config(config: STPFTrainingConfig) -> STPFTrainingConfig:
    if config.epochs <= 0:
        raise ValueError("STPFTrainingConfig.epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("STPFTrainingConfig.batch_size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("STPFTrainingConfig.learning_rate must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("STPFTrainingConfig.weight_decay must be non-negative")
    if not 0.0 <= config.validation_fraction < 1.0:
        raise ValueError("STPFTrainingConfig.validation_fraction must be in [0, 1)")
    if config.grad_clip_norm < 0.0:
        raise ValueError("STPFTrainingConfig.grad_clip_norm must be non-negative")
    if config.cost_aware_weight < 0.0:
        raise ValueError("STPFTrainingConfig.cost_aware_weight must be non-negative")
    return config


def rows_from_npz_shard(path: str | Path) -> list[ProposalFeatureRow]:
    from p2cccd.data.shards import read_npz_shard

    shard = read_npz_shard(path)
    arrays = shard["arrays"]
    ids = arrays["ids"]
    features = arrays["features"]
    interval_targets = arrays["interval_targets"]
    family_targets = arrays["family_targets"]
    scalar_targets = arrays["scalar_targets"]
    rows: list[ProposalFeatureRow] = []
    for index in range(features.shape[0]):
        row = ProposalFeatureRow(
            schema_version=int(ids[index, 0]),
            query_id=int(ids[index, 1]),
            candidate_id=int(ids[index, 2]),
            slab_id=int(ids[index, 3]),
            object_a_id=int(ids[index, 4]),
            patch_a_id=int(ids[index, 5]),
            object_b_id=int(ids[index, 6]),
            patch_b_id=int(ids[index, 7]),
            features=[float(value) for value in features[index].tolist()],
            interval_targets=[float(value) for value in interval_targets[index].tolist()],
            family_targets=[float(value) for value in family_targets[index].tolist()],
            priority_target=float(scalar_targets[index, 0]),
            cost_target=float(scalar_targets[index, 1]),
            uncertainty_target=float(scalar_targets[index, 2]),
            target_mask=int(ids[index, 8]),
        )
        rows.append(validate_proposal_feature_row(row))
    return rows


def stpf_training_loss(
    output,
    targets,
    *,
    cost_aware_weight: float,
) -> torch.Tensor:
    loss = stpf_multitask_loss(output, targets)
    if cost_aware_weight > 0.0:
        loss = loss + cost_aware_weight * stpf_cost_aware_loss(output, targets)
    return loss


def _split_rows(
    rows: Sequence[ProposalFeatureRow],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[ProposalFeatureRow], list[ProposalFeatureRow]]:
    indexed_rows = list(rows)
    if not indexed_rows:
        raise ValueError("training rows must not be empty")
    if validation_fraction == 0.0 or len(indexed_rows) == 1:
        return indexed_rows, []
    rng = random.Random(seed)
    rng.shuffle(indexed_rows)
    validation_count = max(1, int(round(len(indexed_rows) * validation_fraction)))
    validation_count = min(validation_count, len(indexed_rows) - 1)
    return indexed_rows[validation_count:], indexed_rows[:validation_count]


def _iter_batches(
    rows: Sequence[ProposalFeatureRow],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[list[ProposalFeatureRow]]:
    indexed_rows = list(rows)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indexed_rows)
    return [indexed_rows[index : index + batch_size] for index in range(0, len(indexed_rows), batch_size)]


def evaluate_stpf_model(
    model: STPFModel,
    rows: Sequence[ProposalFeatureRow],
    config: STPFTrainingConfig,
    *,
    epoch: int,
    split: str,
) -> STPFEpochMetrics:
    validate_training_config(config)
    for row in rows:
        validate_proposal_feature_row(row)
    if not rows:
        return STPFEpochMetrics(
            epoch=epoch,
            split=split,
            row_count=0,
            loss=0.0,
            interval_top1_recall=0.0,
            family_top2_recall=0.0,
            estimated_exact_work_reduction=0.0,
            mean_predicted_cost=0.0,
            mean_target_cost=0.0,
        )

    was_training = model.training
    model.eval()
    interval_scores: list[list[float]] = []
    family_scores: list[list[float]] = []
    predicted_costs: list[float] = []
    risk_adjusted_work = 0.0
    weighted_loss = 0.0
    with torch.no_grad():
        for batch in _iter_batches(rows, batch_size=config.batch_size, shuffle=False, seed=config.seed):
            features = rows_to_feature_tensor(batch, device=config.device)
            targets = rows_to_target_tensors(batch, device=config.device)
            output = model(features)
            loss = stpf_training_loss(
                output,
                targets,
                cost_aware_weight=config.cost_aware_weight,
            )
            weighted_loss += float(loss.detach().cpu()) * len(batch)
            interval_scores.extend(torch.softmax(output.interval_logits, dim=-1).cpu().tolist())
            family_scores.extend(torch.sigmoid(output.family_logits).cpu().tolist())
            predicted_costs.extend(output.cost_score.cpu().tolist())
            batch_work = output.cost_score * (1.0 - 0.5 * output.priority_score) * (
                1.0 + 0.25 * output.uncertainty_score
            )
            risk_adjusted_work += float(torch.clamp(batch_work, min=0.0).sum().cpu())
    if was_training:
        model.train()

    from p2cccd.data.metrics import (
        estimated_exact_work_reduction,
        family_topk_recall,
        interval_top1_recall,
    )

    baseline_work = sum(max(0.0, row.cost_target) for row in rows)
    return STPFEpochMetrics(
        epoch=epoch,
        split=split,
        row_count=len(rows),
        loss=weighted_loss / len(rows),
        interval_top1_recall=interval_top1_recall(interval_scores, rows),
        family_top2_recall=family_topk_recall(family_scores, rows, k=2),
        estimated_exact_work_reduction=estimated_exact_work_reduction(
            baseline_work=baseline_work,
            proposed_work=max(0.0, risk_adjusted_work),
        ),
        mean_predicted_cost=sum(predicted_costs) / len(predicted_costs),
        mean_target_cost=baseline_work / len(rows),
    )


def train_stpf_model(
    rows: Sequence[ProposalFeatureRow],
    config: STPFTrainingConfig | None = None,
    *,
    validation_rows: Sequence[ProposalFeatureRow] | None = None,
    model: STPFModel | None = None,
) -> STPFTrainingResult:
    cfg = validate_training_config(config or STPFTrainingConfig())
    for row in rows:
        validate_proposal_feature_row(row)
    if validation_rows is not None:
        for row in validation_rows:
            validate_proposal_feature_row(row)

    train_rows, split_validation_rows = _split_rows(
        rows,
        validation_fraction=0.0 if validation_rows is not None else cfg.validation_fraction,
        seed=cfg.seed,
    )
    eval_rows = list(validation_rows) if validation_rows is not None else split_validation_rows
    if not train_rows:
        raise ValueError("at least one training row is required")

    torch.manual_seed(cfg.seed)
    if model is None:
        if cfg.model_config is not None:
            model = STPFModel(cfg.model_config)
        else:
            model = build_stpf_model(cfg.model_preset)
    model.to(cfg.device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history: list[STPFEpochMetrics] = []
    for epoch in range(1, cfg.epochs + 1):
        for batch in _iter_batches(
            train_rows,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            seed=cfg.seed + epoch,
        ):
            features = rows_to_feature_tensor(batch, device=cfg.device)
            targets = rows_to_target_tensors(batch, device=cfg.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(features)
            loss = stpf_training_loss(output, targets, cost_aware_weight=cfg.cost_aware_weight)
            loss.backward()
            if cfg.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        history.append(evaluate_stpf_model(model, train_rows, cfg, epoch=epoch, split="train"))
        if eval_rows:
            history.append(evaluate_stpf_model(model, eval_rows, cfg, epoch=epoch, split="validation"))

    return STPFTrainingResult(
        model=model,
        config=cfg,
        train_row_count=len(train_rows),
        validation_row_count=len(eval_rows),
        history=history,
    )
