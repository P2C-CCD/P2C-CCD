from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .features import (
    PROPOSAL_FAMILY_COUNT,
    PROPOSAL_FEATURE_DIM,
    PROPOSAL_INTERVAL_BIN_COUNT,
)


class STPFModelPreset(StrEnum):
    MICRO_MLP = "micro_mlp"
    TINY_MLP = "tiny_mlp"
    LIGHTWEIGHT_MLP = "lightweight_mlp"
    MEDIUM_MLP = "medium_mlp"
    HIGH_CAPACITY_MLP = "high_capacity_mlp"


@dataclass(frozen=True, slots=True)
class STPFConfig:
    feature_dim: int = PROPOSAL_FEATURE_DIM
    hidden_dim: int = 128
    num_layers: int = 2
    interval_bins: int = PROPOSAL_INTERVAL_BIN_COUNT
    family_count: int = PROPOSAL_FAMILY_COUNT
    dropout: float = 0.0


@dataclass(slots=True)
class STPFOutput:
    interval_logits: torch.Tensor
    family_logits: torch.Tensor
    priority_score: torch.Tensor
    cost_score: torch.Tensor
    uncertainty_score: torch.Tensor


class STPFModel(nn.Module):
    """STPF MLP with interval, family, priority, cost, and uncertainty heads."""

    def __init__(self, config: STPFConfig | None = None):
        super().__init__()
        self.config = config or STPFConfig()
        if self.config.feature_dim <= 0:
            raise ValueError("STPFConfig.feature_dim must be positive")
        if self.config.hidden_dim <= 0:
            raise ValueError("STPFConfig.hidden_dim must be positive")
        if self.config.num_layers <= 0:
            raise ValueError("STPFConfig.num_layers must be positive")
        if self.config.interval_bins <= 0:
            raise ValueError("STPFConfig.interval_bins must be positive")
        if self.config.family_count <= 0:
            raise ValueError("STPFConfig.family_count must be positive")
        if not 0.0 <= self.config.dropout < 1.0:
            raise ValueError("STPFConfig.dropout must be in [0, 1)")

        layers: list[nn.Module] = []
        input_dim = self.config.feature_dim
        for _ in range(self.config.num_layers):
            layers.append(nn.Linear(input_dim, self.config.hidden_dim))
            layers.append(nn.GELU())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            input_dim = self.config.hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.interval_head = nn.Linear(self.config.hidden_dim, self.config.interval_bins)
        self.family_head = nn.Linear(self.config.hidden_dim, self.config.family_count)
        self.priority_head = nn.Linear(self.config.hidden_dim, 1)
        self.cost_head = nn.Linear(self.config.hidden_dim, 1)
        self.uncertainty_head = nn.Linear(self.config.hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> STPFOutput:
        if features.ndim != 2:
            raise ValueError("STPFModel expects a [batch, feature_dim] tensor")
        if features.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"STPFModel expected feature_dim={self.config.feature_dim}, got {features.shape[-1]}"
            )
        hidden = self.trunk(features)
        return STPFOutput(
            interval_logits=self.interval_head(hidden),
            family_logits=self.family_head(hidden),
            priority_score=torch.sigmoid(self.priority_head(hidden)).squeeze(-1),
            cost_score=F.softplus(self.cost_head(hidden)).squeeze(-1),
            uncertainty_score=torch.sigmoid(self.uncertainty_head(hidden)).squeeze(-1),
        )


def stpf_config_for_preset(
    preset: STPFModelPreset | str = STPFModelPreset.LIGHTWEIGHT_MLP,
    **overrides: int | float,
) -> STPFConfig:
    preset_value = STPFModelPreset(str(preset))
    if preset_value is STPFModelPreset.MICRO_MLP:
        values: dict[str, int | float] = {
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout": 0.0,
        }
    elif preset_value is STPFModelPreset.TINY_MLP:
        values = {
            "hidden_dim": 64,
            "num_layers": 1,
            "dropout": 0.0,
        }
    elif preset_value is STPFModelPreset.LIGHTWEIGHT_MLP:
        values: dict[str, int | float] = {
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.0,
        }
    elif preset_value is STPFModelPreset.MEDIUM_MLP:
        values = {
            "hidden_dim": 256,
            "num_layers": 4,
            "dropout": 0.05,
        }
    elif preset_value is STPFModelPreset.HIGH_CAPACITY_MLP:
        values = {
            "hidden_dim": 512,
            "num_layers": 6,
            "dropout": 0.10,
        }
    else:
        raise ValueError(f"unsupported STPF model preset: {preset}")

    allowed = set(STPFConfig.__dataclass_fields__.keys())
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"unknown STPFConfig override fields: {unknown}")
    values.update(overrides)
    return STPFConfig(**values)  # type: ignore[arg-type]


def build_stpf_model(
    preset: STPFModelPreset | str = STPFModelPreset.LIGHTWEIGHT_MLP,
    **config_overrides: int | float,
) -> STPFModel:
    return STPFModel(stpf_config_for_preset(preset, **config_overrides))


def stpf_config_to_dict(config: STPFConfig) -> dict[str, int | float]:
    return asdict(config)


def stpf_config_from_dict(payload: Mapping[str, Any]) -> STPFConfig:
    allowed = set(STPFConfig.__dataclass_fields__.keys())
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown STPFConfig fields in payload: {unknown}")
    values: dict[str, int | float] = {}
    for key in allowed:
        if key in payload:
            values[key] = payload[key]
    return STPFConfig(**values)  # type: ignore[arg-type]


def build_stpf_model_from_checkpoint_payload(
    payload: Any,
    *,
    fallback_preset: STPFModelPreset | str = STPFModelPreset.LIGHTWEIGHT_MLP,
) -> tuple[STPFModel, Mapping[str, Any]]:
    if isinstance(payload, Mapping) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint state_dict must be a mapping")
        if "model_config" in payload:
            model_config = stpf_config_from_dict(payload["model_config"])
            return STPFModel(model_config), state_dict
        if "model_preset" in payload:
            return build_stpf_model(str(payload["model_preset"])), state_dict
        return build_stpf_model(fallback_preset), state_dict
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping or state_dict-like object")
    return build_stpf_model(fallback_preset), payload


def stpf_multitask_loss(
    output: STPFOutput,
    targets: Mapping[str, torch.Tensor],
    *,
    interval_weight: float = 1.0,
    family_weight: float = 1.0,
    priority_weight: float = 1.0,
    cost_weight: float = 1.0,
    uncertainty_weight: float = 1.0,
) -> torch.Tensor:
    interval_targets = targets["interval_targets"].to(output.interval_logits.device)
    family_targets = targets["family_targets"].to(output.family_logits.device)
    priority_target = targets["priority_target"].to(output.priority_score.device)
    cost_target = targets["cost_target"].to(output.cost_score.device)
    uncertainty_target = targets["uncertainty_target"].to(output.uncertainty_score.device)

    interval_index = interval_targets.argmax(dim=-1)
    interval_loss = F.cross_entropy(output.interval_logits, interval_index)
    family_loss = F.binary_cross_entropy_with_logits(output.family_logits, family_targets)
    priority_loss = F.mse_loss(output.priority_score, priority_target)
    cost_loss = F.mse_loss(output.cost_score, cost_target)
    uncertainty_loss = F.mse_loss(output.uncertainty_score, uncertainty_target)
    return (
        interval_weight * interval_loss
        + family_weight * family_loss
        + priority_weight * priority_loss
        + cost_weight * cost_loss
        + uncertainty_weight * uncertainty_loss
    )


def stpf_cost_aware_loss(
    output: STPFOutput,
    targets: Mapping[str, torch.Tensor],
    *,
    work_reduction_weight: float = 1.0,
    uncertainty_work_weight: float = 0.25,
) -> torch.Tensor:
    if work_reduction_weight < 0.0:
        raise ValueError("work_reduction_weight must be non-negative")
    if uncertainty_work_weight < 0.0:
        raise ValueError("uncertainty_work_weight must be non-negative")

    device = output.cost_score.device
    priority_target = targets["priority_target"].to(device)
    cost_target = targets["cost_target"].to(device)
    uncertainty_target = targets["uncertainty_target"].to(device)

    normalized_target_cost = torch.log1p(torch.clamp(cost_target, min=0.0))
    predicted_reduction = output.priority_score * torch.log1p(torch.clamp(output.cost_score, min=0.0))
    target_reduction = priority_target * normalized_target_cost
    reduction_loss = F.smooth_l1_loss(predicted_reduction, target_reduction)

    predicted_risk_adjusted_work = output.cost_score * (
        1.0 + uncertainty_work_weight * output.uncertainty_score
    )
    target_risk_adjusted_work = cost_target * (1.0 + uncertainty_work_weight * uncertainty_target)
    work_loss = F.smooth_l1_loss(predicted_risk_adjusted_work, target_risk_adjusted_work)
    return work_reduction_weight * (reduction_loss + work_loss)
