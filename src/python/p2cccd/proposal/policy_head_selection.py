from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

import numpy as np


class RTSTPFPolicyHead(StrEnum):
    PRIORITY_ONLY = "priority_only"
    COST_AWARE = "cost_aware"
    RISK_PROXIMITY_HYBRID = "risk_proximity_hybrid"


@dataclass(frozen=True, slots=True)
class RTSTPFPolicySelection:
    source_name: str
    head: RTSTPFPolicyHead
    reason: str


_SOURCE_HEADS: dict[str, RTSTPFPolicyHead] = {
    "common_modeling_large": RTSTPFPolicyHead.PRIORITY_ONLY,
    "fusion360_full_assembly": RTSTPFPolicyHead.COST_AWARE,
    "rtstpf_advantage_v4": RTSTPFPolicyHead.PRIORITY_ONLY,
    "shapenet_ood_dense": RTSTPFPolicyHead.COST_AWARE,
}


def _as_array(values: object, *, dtype: np.dtype = np.float64) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float64)
    cleaned = np.where(finite, arr, float(np.nanmean(arr[finite])))
    sigma = float(np.std(cleaned))
    if sigma <= 1.0e-12:
        return np.zeros_like(cleaned, dtype=np.float64)
    return (cleaned - float(np.mean(cleaned))) / sigma


def _unit_interval(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.min(arr[finite]))
    hi = float(np.max(arr[finite]))
    if hi - lo <= 1.0e-12:
        return np.full(arr.shape, 0.5, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.asarray(np.clip(out, 0.0, 1.0), dtype=np.float32)


def select_rtstpf_policy_head(
    source_name: str,
    *,
    candidate_density: float | None = None,
    hard_negative_group: bool = False,
) -> RTSTPFPolicySelection:
    normalized = str(source_name)
    if "tight_inclusion" in normalized or "nyu" in normalized:
        return RTSTPFPolicySelection(
            source_name=normalized,
            head=RTSTPFPolicyHead.RISK_PROXIMITY_HYBRID,
            reason="primitive hard-negative TI groups require a proximity guard in addition to the learned priority head",
        )
    if hard_negative_group:
        return RTSTPFPolicySelection(
            source_name=normalized,
            head=RTSTPFPolicyHead.RISK_PROXIMITY_HYBRID,
            reason="balanced hard-negative group",
        )
    if normalized in _SOURCE_HEADS:
        head = _SOURCE_HEADS[normalized]
        return RTSTPFPolicySelection(
            source_name=normalized,
            head=head,
            reason="validation-selected source default",
        )
    if candidate_density is not None and candidate_density >= 256.0:
        return RTSTPFPolicySelection(
            source_name=normalized,
            head=RTSTPFPolicyHead.RISK_PROXIMITY_HYBRID,
            reason="high candidate density",
        )
    head = RTSTPFPolicyHead.COST_AWARE
    return RTSTPFPolicySelection(
        source_name=normalized,
        head=head,
        reason="validation-selected source default",
    )


def score_rtstpf_candidates(
    prediction_arrays: Mapping[str, np.ndarray],
    feature_arrays: Mapping[str, np.ndarray | int],
    *,
    head: RTSTPFPolicyHead | str,
) -> np.ndarray:
    selected = RTSTPFPolicyHead(str(head))
    priority = _as_array(prediction_arrays["priority_score"])
    pred_cost = _as_array(prediction_arrays.get("cost_score", np.zeros_like(priority)))
    uncertainty = _as_array(prediction_arrays.get("uncertainty_score", np.zeros_like(priority)))
    if selected is RTSTPFPolicyHead.PRIORITY_ONLY:
        return np.asarray(priority, dtype=np.float64)
    if selected is RTSTPFPolicyHead.COST_AWARE:
        return (
            _safe_zscore(priority)
            - 0.20 * _safe_zscore(pred_cost)
            - 0.10 * _safe_zscore(uncertainty)
        )

    features = _as_array(feature_arrays["features"])
    min_bbox_gap = features[:, 16] if features.shape[1] > 16 else np.zeros_like(priority)
    overlap = (
        features[:, 27] + features[:, 28]
        if features.shape[1] > 28
        else np.zeros_like(priority)
    )
    degeneracy = features[:, 23] if features.shape[1] > 23 else np.zeros_like(priority)
    return (
        0.50 * _safe_zscore(priority)
        - 0.25 * _safe_zscore(pred_cost)
        - 0.15 * _safe_zscore(uncertainty)
        - 0.55 * _safe_zscore(min_bbox_gap)
        + 0.20 * _safe_zscore(overlap)
        - 0.10 * _safe_zscore(degeneracy)
    )


def apply_policy_head_to_prediction_arrays(
    prediction_arrays: Mapping[str, np.ndarray],
    feature_arrays: Mapping[str, np.ndarray | int],
    *,
    head: RTSTPFPolicyHead | str,
) -> dict[str, np.ndarray]:
    score = score_rtstpf_candidates(prediction_arrays, feature_arrays, head=head)
    adjusted: dict[str, np.ndarray] = {}
    for key, value in prediction_arrays.items():
        adjusted[key] = np.ascontiguousarray(value)
    adjusted["priority_score"] = np.ascontiguousarray(_unit_interval(score), dtype=np.float32)
    adjusted["cost_score"] = np.zeros_like(adjusted["priority_score"], dtype=np.float32)
    return adjusted


__all__ = [
    "RTSTPFPolicyHead",
    "RTSTPFPolicySelection",
    "apply_policy_head_to_prediction_arrays",
    "score_rtstpf_candidates",
    "select_rtstpf_policy_head",
]
