from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from .features import (
    PROPOSAL_FAMILY_COUNT,
    PROPOSAL_INTERVAL_BIN_COUNT,
    ProposalFeatureRow,
    rows_to_feature_tensor,
    validate_proposal_feature_row,
)


DEFAULT_OOD_ABS_FEATURE_THRESHOLD = 1.0e5


@dataclass(slots=True)
class ProposalPrediction:
    candidate_id: int = 0
    interval_scores: list[float] = field(
        default_factory=lambda: [0.0] * PROPOSAL_INTERVAL_BIN_COUNT
    )
    family_scores: list[float] = field(default_factory=lambda: [0.0] * PROPOSAL_FAMILY_COUNT)
    priority_score: float = 0.0
    cost_score: float = 0.0
    uncertainty_score: float = 0.0
    source: str = "dummy"


def _finite_non_negative(value: float) -> bool:
    return math.isfinite(float(value)) and float(value) >= 0.0


def _normalized_scores(values: Sequence[float], size: int, *, fallback_indices: tuple[int, ...]) -> list[float]:
    if len(values) != size:
        raise ValueError(f"score vector must have length {size}")
    cleaned = [float(value) if math.isfinite(float(value)) and float(value) >= 0.0 else 0.0 for value in values]
    total = sum(cleaned)
    if total > 0.0:
        return [value / total for value in cleaned]
    fallback = [0.0] * size
    for index in fallback_indices:
        if 0 <= index < size:
            fallback[index] = 1.0 / max(1, len(fallback_indices))
    return fallback


def validate_proposal_prediction(prediction: ProposalPrediction) -> ProposalPrediction:
    if prediction.candidate_id == 0:
        raise ValueError("ProposalPrediction.candidate_id is required")
    if len(prediction.interval_scores) != PROPOSAL_INTERVAL_BIN_COUNT:
        raise ValueError(
            f"ProposalPrediction.interval_scores must have length {PROPOSAL_INTERVAL_BIN_COUNT}"
        )
    if len(prediction.family_scores) != PROPOSAL_FAMILY_COUNT:
        raise ValueError(
            f"ProposalPrediction.family_scores must have length {PROPOSAL_FAMILY_COUNT}"
        )
    for field_name, values in (
        ("interval_scores", prediction.interval_scores),
        ("family_scores", prediction.family_scores),
    ):
        for value in values:
            if not _finite_non_negative(value):
                raise ValueError(f"ProposalPrediction.{field_name} must be finite and non-negative")
    for field_name in ("priority_score", "cost_score", "uncertainty_score"):
        if not _finite_non_negative(getattr(prediction, field_name)):
            raise ValueError(f"ProposalPrediction.{field_name} must be finite and non-negative")
    return prediction


def is_ood_feature_row(
    row: ProposalFeatureRow,
    *,
    abs_feature_threshold: float = DEFAULT_OOD_ABS_FEATURE_THRESHOLD,
) -> bool:
    if abs_feature_threshold <= 0.0 or not math.isfinite(abs_feature_threshold):
        raise ValueError("abs_feature_threshold must be finite and positive")
    for value in row.features:
        feature = float(value)
        if not math.isfinite(feature) or abs(feature) > abs_feature_threshold:
            return True
    return False


def dummy_proposal_policy(
    rows: Sequence[ProposalFeatureRow],
    *,
    ood_abs_feature_threshold: float = DEFAULT_OOD_ABS_FEATURE_THRESHOLD,
) -> list[ProposalPrediction]:
    predictions: list[ProposalPrediction] = []
    for row in rows:
        validate_proposal_feature_row(row)
        ood = is_ood_feature_row(row, abs_feature_threshold=ood_abs_feature_threshold)
        prediction = ProposalPrediction(
            candidate_id=row.candidate_id,
            interval_scores=_normalized_scores(
                row.interval_targets, PROPOSAL_INTERVAL_BIN_COUNT, fallback_indices=(0,)
            ),
            family_scores=_normalized_scores(
                row.family_targets, PROPOSAL_FAMILY_COUNT, fallback_indices=(0, 1)
            ),
            priority_score=max(0.0, float(row.priority_target)),
            cost_score=max(0.0, float(row.cost_target)),
            uncertainty_score=1.0 if ood else max(0.0, float(row.uncertainty_target)),
            source="dummy_ood_fallback" if ood else "dummy",
        )
        validate_proposal_prediction(prediction)
        predictions.append(prediction)
    return predictions


def batched_stpf_inference(
    model,
    rows: Sequence[ProposalFeatureRow],
    *,
    batch_size: int = 1024,
    device: str | None = None,
    ood_abs_feature_threshold: float | None = DEFAULT_OOD_ABS_FEATURE_THRESHOLD,
) -> list[ProposalPrediction]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if ood_abs_feature_threshold is not None and (
        ood_abs_feature_threshold <= 0.0 or not math.isfinite(ood_abs_feature_threshold)
    ):
        raise ValueError("ood_abs_feature_threshold must be finite and positive")
    if not rows:
        return []

    import torch

    if device is None:
        try:
            device = str(next(model.parameters()).device)
        except StopIteration:
            device = "cpu"

    predictions: list[ProposalPrediction | None] = [None] * len(rows)
    inference_rows: list[ProposalFeatureRow] = []
    inference_indices: list[int] = []
    for index, row in enumerate(rows):
        validate_proposal_feature_row(row)
        if ood_abs_feature_threshold is not None and is_ood_feature_row(
            row, abs_feature_threshold=ood_abs_feature_threshold
        ):
            predictions[index] = dummy_proposal_policy(
                [row], ood_abs_feature_threshold=ood_abs_feature_threshold
            )[0]
        else:
            inference_rows.append(row)
            inference_indices.append(index)

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(inference_rows), batch_size):
                batch_rows = inference_rows[start : start + batch_size]
                batch_indices = inference_indices[start : start + batch_size]
                features = rows_to_feature_tensor(batch_rows, device=device)
                output = model(features)
                interval_scores = torch.softmax(output.interval_logits, dim=-1).detach().cpu().tolist()
                family_scores = torch.sigmoid(output.family_logits).detach().cpu().tolist()
                priority = output.priority_score.detach().cpu().tolist()
                cost = output.cost_score.detach().cpu().tolist()
                uncertainty = output.uncertainty_score.detach().cpu().tolist()
                for index, row in enumerate(batch_rows):
                    prediction = ProposalPrediction(
                        candidate_id=row.candidate_id,
                        interval_scores=[float(value) for value in interval_scores[index]],
                        family_scores=[float(value) for value in family_scores[index]],
                        priority_score=float(priority[index]),
                        cost_score=float(cost[index]),
                        uncertainty_score=float(uncertainty[index]),
                        source="stpf",
                    )
                    validate_proposal_prediction(prediction)
                    predictions[batch_indices[index]] = prediction
    finally:
        if was_training:
            model.train()

    if any(prediction is None for prediction in predictions):
        raise RuntimeError("proposal inference failed to produce one prediction per feature row")
    return [prediction for prediction in predictions if prediction is not None]
