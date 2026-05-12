from __future__ import annotations

import math
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
    batched_stpf_inference,
    dummy_proposal_policy,
    is_ood_feature_row,
    validate_proposal_prediction,
)


def _row(candidate_id: int) -> ProposalFeatureRow:
    row = ProposalFeatureRow(
        query_id=77,
        candidate_id=candidate_id,
        features=[0.01 * float(candidate_id)] * PROPOSAL_FEATURE_DIM,
        priority_target=float(candidate_id),
        cost_target=1.0 + float(candidate_id),
        uncertainty_target=0.1,
        target_mask=31,
    )
    row.interval_targets[candidate_id % PROPOSAL_INTERVAL_BIN_COUNT] = 1.0
    row.family_targets[0] = 1.0
    row.family_targets[1] = 1.0
    return row


def test_dummy_policy_preserves_candidate_ids_and_outputs_valid_scores() -> None:
    rows = [_row(1), _row(2), _row(3)]
    predictions = dummy_proposal_policy(rows)

    assert [prediction.candidate_id for prediction in predictions] == [1, 2, 3]
    for prediction in predictions:
        assert validate_proposal_prediction(prediction) is prediction
        assert len(prediction.interval_scores) == PROPOSAL_INTERVAL_BIN_COUNT
        assert len(prediction.family_scores) == PROPOSAL_FAMILY_COUNT
        assert math.isclose(sum(prediction.interval_scores), 1.0)
        assert math.isclose(sum(prediction.family_scores), 1.0)
        assert prediction.source == "dummy"


def test_dummy_policy_marks_ood_rows_as_high_uncertainty_fallback() -> None:
    row = _row(4)
    row.features[0] = 20.0

    assert is_ood_feature_row(row, abs_feature_threshold=10.0)
    prediction = dummy_proposal_policy([row], ood_abs_feature_threshold=10.0)[0]
    assert prediction.candidate_id == row.candidate_id
    assert prediction.uncertainty_score == 1.0
    assert prediction.source == "dummy_ood_fallback"


def test_batched_stpf_inference_preserves_order_and_shapes() -> None:
    rows = [_row(1), _row(2), _row(3), _row(4), _row(5)]
    model = STPFModel(STPFConfig(feature_dim=PROPOSAL_FEATURE_DIM, hidden_dim=16, num_layers=2))
    model.train()

    predictions = batched_stpf_inference(model, rows, batch_size=2, device="cpu")

    assert model.training
    assert [prediction.candidate_id for prediction in predictions] == [1, 2, 3, 4, 5]
    for prediction in predictions:
        assert validate_proposal_prediction(prediction) is prediction
        assert len(prediction.interval_scores) == PROPOSAL_INTERVAL_BIN_COUNT
        assert len(prediction.family_scores) == PROPOSAL_FAMILY_COUNT
        assert math.isclose(sum(prediction.interval_scores), 1.0, rel_tol=1.0e-5)
        assert prediction.source == "stpf"


def test_batched_stpf_inference_routes_ood_rows_to_fallback() -> None:
    rows = [_row(1), _row(2), _row(3)]
    rows[1].features[0] = float("inf")
    model = STPFModel(STPFConfig(feature_dim=PROPOSAL_FEATURE_DIM, hidden_dim=16, num_layers=1))

    predictions = batched_stpf_inference(
        model,
        rows,
        batch_size=2,
        device="cpu",
        ood_abs_feature_threshold=10.0,
    )

    assert [prediction.candidate_id for prediction in predictions] == [1, 2, 3]
    assert predictions[0].source == "stpf"
    assert predictions[1].source == "dummy_ood_fallback"
    assert predictions[1].uncertainty_score == 1.0
    assert predictions[2].source == "stpf"


def test_batched_stpf_inference_restores_training_state_on_model_error() -> None:
    class RaisingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(PROPOSAL_FEATURE_DIM, 1)

        def forward(self, features):
            raise RuntimeError("forced proposal model failure")

    model = RaisingModel()
    model.train()

    try:
        batched_stpf_inference(model, [_row(1)], batch_size=1, device="cpu")
    except RuntimeError as exc:
        assert "forced proposal model failure" in str(exc)
    else:
        raise AssertionError("expected model failure")
    assert model.training


def test_batched_stpf_inference_all_ood_rows_bypass_model() -> None:
    class RaisingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(PROPOSAL_FEATURE_DIM, 1)

        def forward(self, features):
            raise RuntimeError("OOD rows should not call model")

    row = _row(5)
    row.features[0] = float("nan")
    model = RaisingModel()
    predictions = batched_stpf_inference(
        model,
        [row],
        batch_size=1,
        device="cpu",
        ood_abs_feature_threshold=10.0,
    )

    assert len(predictions) == 1
    assert predictions[0].candidate_id == row.candidate_id
    assert predictions[0].source == "dummy_ood_fallback"


def test_batched_stpf_inference_rejects_invalid_batch_size() -> None:
    model = STPFModel(STPFConfig(feature_dim=PROPOSAL_FEATURE_DIM, hidden_dim=16, num_layers=1))
    try:
        batched_stpf_inference(model, [_row(1)], batch_size=0, device="cpu")
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("expected batch_size validation error")
