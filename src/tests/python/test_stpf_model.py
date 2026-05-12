from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

import torch

from p2cccd.proposal import (  # noqa: E402
    PROPOSAL_FAMILY_COUNT,
    PROPOSAL_FEATURE_DIM,
    PROPOSAL_INTERVAL_BIN_COUNT,
    ProposalFeatureRow,
    STPFConfig,
    STPFModel,
    STPFModelPreset,
    build_stpf_model_from_checkpoint_payload,
    build_stpf_model,
    rows_to_feature_tensor,
    rows_to_target_tensors,
    stpf_config_for_preset,
    stpf_cost_aware_loss,
    stpf_multitask_loss,
    validate_proposal_feature_row,
)


def _row(candidate_id: int) -> ProposalFeatureRow:
    row = ProposalFeatureRow(
        query_id=1,
        candidate_id=candidate_id,
        features=[float(candidate_id)] * PROPOSAL_FEATURE_DIM,
        priority_target=0.5,
        cost_target=1.25,
        uncertainty_target=0.2,
        target_mask=31,
    )
    row.interval_targets[2] = 1.0
    row.family_targets[0] = 1.0
    row.family_targets[1] = 1.0
    return row


def test_proposal_feature_rows_convert_to_tensors() -> None:
    rows = [_row(1), _row(2)]
    for row in rows:
        assert validate_proposal_feature_row(row) is row

    features = rows_to_feature_tensor(rows)
    targets = rows_to_target_tensors(rows)
    assert features.shape == (2, PROPOSAL_FEATURE_DIM)
    assert targets["interval_targets"].shape == (2, PROPOSAL_INTERVAL_BIN_COUNT)
    assert targets["family_targets"].shape == (2, PROPOSAL_FAMILY_COUNT)
    assert targets["target_mask"].tolist() == [31, 31]


def test_stpf_model_forward_and_backward() -> None:
    config = STPFConfig(feature_dim=PROPOSAL_FEATURE_DIM, hidden_dim=32, num_layers=2)
    model = STPFModel(config)
    rows = [_row(1), _row(2), _row(3), _row(4)]
    features = rows_to_feature_tensor(rows)
    targets = rows_to_target_tensors(rows)

    output = model(features)
    assert output.interval_logits.shape == (4, PROPOSAL_INTERVAL_BIN_COUNT)
    assert output.family_logits.shape == (4, PROPOSAL_FAMILY_COUNT)
    assert output.priority_score.shape == (4,)
    assert output.cost_score.shape == (4,)
    assert output.uncertainty_score.shape == (4,)
    assert torch.all(output.cost_score >= 0.0)

    loss = stpf_multitask_loss(output, targets)
    assert torch.isfinite(loss)
    cost_loss = stpf_cost_aware_loss(output, targets)
    assert torch.isfinite(cost_loss)
    loss.backward()
    assert model.interval_head.weight.grad is not None
    assert model.family_head.weight.grad is not None
    assert model.priority_head.weight.grad is not None
    assert model.cost_head.weight.grad is not None
    assert model.uncertainty_head.weight.grad is not None


def test_stpf_model_rejects_wrong_feature_dim() -> None:
    model = STPFModel(STPFConfig(feature_dim=PROPOSAL_FEATURE_DIM, hidden_dim=16))
    bad_features = torch.zeros(2, PROPOSAL_FEATURE_DIM + 1)
    try:
        model(bad_features)
    except ValueError as exc:
        assert "feature_dim" in str(exc)
    else:
        raise AssertionError("expected feature_dim validation error")


def test_stpf_model_presets_include_higher_capacity_variants() -> None:
    micro = stpf_config_for_preset(STPFModelPreset.MICRO_MLP)
    tiny = stpf_config_for_preset(STPFModelPreset.TINY_MLP)
    light = stpf_config_for_preset(STPFModelPreset.LIGHTWEIGHT_MLP)
    medium = stpf_config_for_preset("medium_mlp")
    high = stpf_config_for_preset("high_capacity_mlp", hidden_dim=64, num_layers=3)

    assert micro.hidden_dim < tiny.hidden_dim < light.hidden_dim < medium.hidden_dim
    assert micro.num_layers == 1
    assert tiny.num_layers == 1
    assert light.hidden_dim < medium.hidden_dim
    assert high.hidden_dim == 64
    assert high.num_layers == 3

    model = build_stpf_model("high_capacity_mlp", hidden_dim=32, num_layers=2, dropout=0.0)
    output = model(torch.zeros(2, PROPOSAL_FEATURE_DIM))
    assert output.interval_logits.shape == (2, PROPOSAL_INTERVAL_BIN_COUNT)


def test_checkpoint_payload_can_restore_non_default_model_structure() -> None:
    model = build_stpf_model(STPFModelPreset.MICRO_MLP)
    payload = {
        "state_dict": model.state_dict(),
        "model_config": {
            "feature_dim": PROPOSAL_FEATURE_DIM,
            "hidden_dim": 32,
            "num_layers": 1,
            "interval_bins": PROPOSAL_INTERVAL_BIN_COUNT,
            "family_count": PROPOSAL_FAMILY_COUNT,
            "dropout": 0.0,
        },
        "model_preset": "micro_mlp",
    }

    restored_model, state_dict = build_stpf_model_from_checkpoint_payload(payload)

    assert restored_model.config.hidden_dim == 32
    assert restored_model.config.num_layers == 1
    assert "trunk.0.weight" in state_dict


def test_cost_aware_loss_rejects_negative_weights() -> None:
    model = STPFModel(STPFConfig(feature_dim=PROPOSAL_FEATURE_DIM, hidden_dim=16))
    rows = [_row(1), _row(2)]
    output = model(rows_to_feature_tensor(rows))
    targets = rows_to_target_tensors(rows)

    try:
        stpf_cost_aware_loss(output, targets, work_reduction_weight=-1.0)
    except ValueError as exc:
        assert "work_reduction_weight" in str(exc)
    else:
        raise AssertionError("expected negative weight validation error")
