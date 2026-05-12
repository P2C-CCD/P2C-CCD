from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


PROPOSAL_FEATURE_DIM = 32
PROPOSAL_INTERVAL_BIN_COUNT = 8
PROPOSAL_FAMILY_COUNT = 8

TARGET_INTERVAL = 1 << 0
TARGET_FAMILY = 1 << 1
TARGET_PRIORITY = 1 << 2
TARGET_COST = 1 << 3
TARGET_UNCERTAINTY = 1 << 4


@dataclass(slots=True)
class ProposalFeatureRow:
    schema_version: int = 1
    query_id: int = 0
    candidate_id: int = 0
    slab_id: int = 0
    object_a_id: int = 0
    patch_a_id: int = 0
    object_b_id: int = 0
    patch_b_id: int = 0
    features: list[float] = field(default_factory=lambda: [0.0] * PROPOSAL_FEATURE_DIM)
    interval_targets: list[float] = field(
        default_factory=lambda: [0.0] * PROPOSAL_INTERVAL_BIN_COUNT
    )
    family_targets: list[float] = field(default_factory=lambda: [0.0] * PROPOSAL_FAMILY_COUNT)
    priority_target: float = 0.0
    cost_target: float = 0.0
    uncertainty_target: float = 0.0
    target_mask: int = 0


def validate_proposal_feature_row(row: ProposalFeatureRow) -> ProposalFeatureRow:
    if row.schema_version != 1:
        raise ValueError("ProposalFeatureRow.schema_version is unsupported")
    if row.query_id == 0:
        raise ValueError("ProposalFeatureRow.query_id is required")
    if row.candidate_id == 0:
        raise ValueError("ProposalFeatureRow.candidate_id is required")
    if len(row.features) != PROPOSAL_FEATURE_DIM:
        raise ValueError(f"ProposalFeatureRow.features must have length {PROPOSAL_FEATURE_DIM}")
    if len(row.interval_targets) != PROPOSAL_INTERVAL_BIN_COUNT:
        raise ValueError(
            f"ProposalFeatureRow.interval_targets must have length {PROPOSAL_INTERVAL_BIN_COUNT}"
        )
    if len(row.family_targets) != PROPOSAL_FAMILY_COUNT:
        raise ValueError(
            f"ProposalFeatureRow.family_targets must have length {PROPOSAL_FAMILY_COUNT}"
        )
    for field_name, values in (
        ("features", row.features),
        ("interval_targets", row.interval_targets),
        ("family_targets", row.family_targets),
    ):
        for value in values:
            if not isinstance(value, (int, float)):
                raise ValueError(f"ProposalFeatureRow.{field_name} values must be numeric")
    return row


def rows_to_feature_tensor(rows: Sequence[ProposalFeatureRow], *, device: str | None = None):
    import torch

    for row in rows:
        validate_proposal_feature_row(row)
    return torch.tensor([row.features for row in rows], dtype=torch.float32, device=device)


def rows_to_target_tensors(rows: Sequence[ProposalFeatureRow], *, device: str | None = None):
    import torch

    for row in rows:
        validate_proposal_feature_row(row)
    return {
        "interval_targets": torch.tensor(
            [row.interval_targets for row in rows], dtype=torch.float32, device=device
        ),
        "family_targets": torch.tensor(
            [row.family_targets for row in rows], dtype=torch.float32, device=device
        ),
        "priority_target": torch.tensor(
            [row.priority_target for row in rows], dtype=torch.float32, device=device
        ),
        "cost_target": torch.tensor(
            [row.cost_target for row in rows], dtype=torch.float32, device=device
        ),
        "uncertainty_target": torch.tensor(
            [row.uncertainty_target for row in rows], dtype=torch.float32, device=device
        ),
        "target_mask": torch.tensor([row.target_mask for row in rows], dtype=torch.int64, device=device),
    }


def make_feature_rows_from_dicts(rows: Iterable[dict]) -> list[ProposalFeatureRow]:
    built: list[ProposalFeatureRow] = []
    for row in rows:
        built_row = ProposalFeatureRow(**row)
        validate_proposal_feature_row(built_row)
        built.append(built_row)
    return built
