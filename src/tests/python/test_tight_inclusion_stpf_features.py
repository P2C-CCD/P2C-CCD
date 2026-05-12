from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.datasets.tight_inclusion_queries import read_tight_inclusion_query  # noqa: E402
from p2cccd.datasets.tight_inclusion_stpf_features import (  # noqa: E402
    tight_inclusion_query_to_proposal_row,
)
from p2cccd.proposal.features import (  # noqa: E402
    PROPOSAL_FAMILY_COUNT,
    PROPOSAL_FEATURE_DIM,
    PROPOSAL_INTERVAL_BIN_COUNT,
    TARGET_COST,
    TARGET_FAMILY,
    TARGET_INTERVAL,
    TARGET_PRIORITY,
    TARGET_UNCERTAINTY,
)


DATASET_ROOT = PROJECT_ROOT / "baseline" / "datasets" / "continuous-collision-detection"

pytestmark = pytest.mark.skipif(
    not DATASET_ROOT.exists(),
    reason="Tight-Inclusion / NYU full-query dataset is not available locally",
)


def test_tight_inclusion_query_to_proposal_row_uses_fixed_stpf_schema() -> None:
    query = read_tight_inclusion_query(
        DATASET_ROOT / "golf-ball" / "vertex-face" / "vertex-face-0003.csv",
        9956,
        dataset_root=DATASET_ROOT,
    )

    row = tight_inclusion_query_to_proposal_row(query)

    assert row.query_id != 0
    assert row.candidate_id == query.query_index + 1
    assert len(row.features) == PROPOSAL_FEATURE_DIM
    assert len(row.interval_targets) == PROPOSAL_INTERVAL_BIN_COUNT
    assert len(row.family_targets) == PROPOSAL_FAMILY_COUNT
    assert row.family_targets[0] == 1.0
    assert sum(row.interval_targets) == 1.0
    assert np.isfinite(np.asarray(row.features, dtype=np.float64)).all()
    assert 0.0 <= row.priority_target <= 1.0
    assert row.cost_target > 0.0
    assert 0.0 <= row.uncertainty_target <= 1.0
    assert row.target_mask == (
        TARGET_INTERVAL | TARGET_FAMILY | TARGET_PRIORITY | TARGET_COST | TARGET_UNCERTAINTY
    )


def test_edge_edge_query_sets_edge_family() -> None:
    query = read_tight_inclusion_query(
        DATASET_ROOT / "unit-tests" / "edge-edge" / "edge-edge-0001.csv",
        0,
        dataset_root=DATASET_ROOT,
    )

    row = tight_inclusion_query_to_proposal_row(query)

    assert row.features[0] == 0.0
    assert row.features[1] == 1.0
    assert row.family_targets[1] == 1.0
