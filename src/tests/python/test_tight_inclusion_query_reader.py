from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.datasets.tight_inclusion_queries import (  # noqa: E402
    TIGHT_INCLUSION_QUERY_ROWS,
    build_file_level_split,
    discover_tight_inclusion_csv_files,
    inspect_tight_inclusion_csv,
    iter_dataset_queries,
    iter_tight_inclusion_queries,
    parse_rational_vertex_row,
    read_tight_inclusion_query,
)


DATASET_ROOT = PROJECT_ROOT / "baseline" / "datasets" / "continuous-collision-detection"


pytestmark = pytest.mark.skipif(
    not DATASET_ROOT.exists(),
    reason="Tight-Inclusion / NYU full-query dataset is not available locally",
)


def test_parse_rational_vertex_row_preserves_truth_and_coordinates() -> None:
    vertex, numerators, denominators, truth = parse_rational_vertex_row("1,2,-3,4,5,10,1")

    assert truth is True
    assert np.allclose(vertex, np.asarray([0.5, -0.75, 0.5]))
    assert numerators.tolist() == [1, -3, 5]
    assert denominators.tolist() == [2, 4, 10]


def test_unit_tests_all_queries_are_stream_readable() -> None:
    csv_files = discover_tight_inclusion_csv_files(DATASET_ROOT, cases=("unit-tests",), inspect=True)

    assert len(csv_files) == 8
    assert {item.kind for item in csv_files} == {"vertex-face", "edge-edge"}
    assert all(item.line_count % TIGHT_INCLUSION_QUERY_ROWS == 0 for item in csv_files)
    assert sum(item.query_count for item in csv_files) > 0

    query_count = 0
    positive_count = 0
    for query in iter_dataset_queries(DATASET_ROOT, cases=("unit-tests",)):
        assert query.case_name == "unit-tests"
        assert query.kind in {"vertex-face", "edge-edge"}
        assert query.vertices_t0_t1.shape == (8, 3)
        assert query.vertices_t0.shape == (4, 3)
        assert query.vertices_t1.shape == (4, 3)
        assert query.numerators.shape == (8, 3)
        assert query.denominators.shape == (8, 3)
        assert np.isfinite(query.vertices_t0_t1).all()
        assert query.rational_magnitude_features.shape == (4,)
        query_count += 1
        positive_count += int(query.ground_truth)

    assert query_count == sum(item.query_count for item in csv_files)
    assert positive_count >= 0


def test_golf_ball_specific_query_can_be_random_accessed() -> None:
    csv_path = DATASET_ROOT / "golf-ball" / "vertex-face" / "vertex-face-0003.csv"
    info = inspect_tight_inclusion_csv(csv_path, dataset_root=DATASET_ROOT)

    assert info.case_name == "golf-ball"
    assert info.kind == "vertex-face"
    assert info.query_count > 9956

    query = read_tight_inclusion_query(csv_path, 9956, dataset_root=DATASET_ROOT)

    assert query.case_name == "golf-ball"
    assert query.kind == "vertex-face"
    assert query.query_index == 9956
    assert query.vertices_t0_t1.shape == (8, 3)
    assert isinstance(query.ground_truth, bool)
    assert np.isfinite(query.vertices_t0_t1).all()


def test_streaming_limit_does_not_read_entire_large_csv() -> None:
    csv_path = DATASET_ROOT / "golf-ball" / "vertex-face" / "vertex-face-0003.csv"
    queries = list(iter_tight_inclusion_queries(csv_path, dataset_root=DATASET_ROOT, start_query_index=9956, limit=2))

    assert [query.query_index for query in queries] == [9956, 9957]
    assert all(query.case_name == "golf-ball" for query in queries)


def test_file_level_split_is_disjoint_and_keeps_unit_smoke() -> None:
    files = discover_tight_inclusion_csv_files(
        DATASET_ROOT,
        cases=("unit-tests", "erleben-spike-hole"),
        inspect=False,
    )
    split = build_file_level_split(files, seed=fixed_seed)

    train = {item.csv_path for item in split.train}
    validation = {item.csv_path for item in split.validation}
    heldout = {item.csv_path for item in split.heldout_test}

    assert train
    assert heldout
    assert train.isdisjoint(validation)
    assert train.isdisjoint(heldout)
    assert validation.isdisjoint(heldout)
    assert {item.case_name for item in split.unit_smoke} == {"unit-tests"}
    assert len(split.full_stress) == len(files)
