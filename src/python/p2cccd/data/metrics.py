from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from p2cccd.proposal.features import ProposalFeatureRow, validate_proposal_feature_row


@dataclass(frozen=True, slots=True)
class LabelMetrics:
    row_count: int
    positive_count: int
    positive_ratio: float
    mean_priority_target: float
    mean_cost_target: float
    mean_uncertainty_target: float


def compute_label_metrics(rows: Sequence[ProposalFeatureRow]) -> LabelMetrics:
    for row in rows:
        validate_proposal_feature_row(row)
    row_count = len(rows)
    if row_count == 0:
        return LabelMetrics(
            row_count=0,
            positive_count=0,
            positive_ratio=0.0,
            mean_priority_target=0.0,
            mean_cost_target=0.0,
            mean_uncertainty_target=0.0,
        )
    positive_count = sum(1 for row in rows if row.priority_target >= 0.5)
    return LabelMetrics(
        row_count=row_count,
        positive_count=positive_count,
        positive_ratio=positive_count / row_count,
        mean_priority_target=sum(row.priority_target for row in rows) / row_count,
        mean_cost_target=sum(row.cost_target for row in rows) / row_count,
        mean_uncertainty_target=sum(row.uncertainty_target for row in rows) / row_count,
    )


def interval_top1_recall(interval_scores: Sequence[Sequence[float]], rows: Sequence[ProposalFeatureRow]) -> float:
    if len(interval_scores) != len(rows):
        raise ValueError("interval_scores and rows must have matching lengths")
    if not rows:
        return 0.0
    hits = 0
    for scores, row in zip(interval_scores, rows):
        target_index = max(range(len(row.interval_targets)), key=lambda i: row.interval_targets[i])
        predicted_index = max(range(len(scores)), key=lambda i: scores[i])
        hits += int(target_index == predicted_index)
    return hits / len(rows)


def family_topk_recall(
    family_scores: Sequence[Sequence[float]],
    rows: Sequence[ProposalFeatureRow],
    *,
    k: int = 2,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(family_scores) != len(rows):
        raise ValueError("family_scores and rows must have matching lengths")
    if not rows:
        return 0.0
    hits = 0
    for scores, row in zip(family_scores, rows):
        active_targets = {index for index, value in enumerate(row.family_targets) if value > 0.0}
        predicted = {
            index
            for index, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:k]
        }
        hits += int(active_targets.issubset(predicted))
    return hits / len(rows)


def estimated_exact_work_reduction(*, baseline_work: float, proposed_work: float) -> float:
    if baseline_work < 0.0 or proposed_work < 0.0:
        raise ValueError("work values must be non-negative")
    if baseline_work == 0.0:
        return 0.0
    return (baseline_work - proposed_work) / baseline_work
