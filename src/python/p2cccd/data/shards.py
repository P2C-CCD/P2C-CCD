from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from p2cccd.proposal.features import (
    PROPOSAL_FAMILY_COUNT,
    PROPOSAL_FEATURE_DIM,
    PROPOSAL_INTERVAL_BIN_COUNT,
    ProposalFeatureRow,
    validate_proposal_feature_row,
)

from .dataset import DATASET_SHARD_SCHEMA_VERSION, GeneratedDataset, split_ids_for_samples
from .oracle import ExactOracleTrace
from .samplers import MotionDiscPairSample


def _ids_array(rows: list[ProposalFeatureRow]) -> np.ndarray:
    return np.asarray(
        [
            [
                row.schema_version,
                row.query_id,
                row.candidate_id,
                row.slab_id,
                row.object_a_id,
                row.patch_a_id,
                row.object_b_id,
                row.patch_b_id,
                row.target_mask,
            ]
            for row in rows
        ],
        dtype=np.uint64,
    )


def _trace_array(traces: list[ExactOracleTrace]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(trace.collided),
                trace.toi,
                trace.closest_time,
                trace.min_distance,
                trace.safe_margin,
                trace.exact_cost,
                trace.contact_interval_t0,
                trace.contact_interval_t1,
            ]
            for trace in traces
        ],
        dtype=np.float64,
    )


def _sample_array(samples: list[MotionDiscPairSample]) -> np.ndarray:
    return np.asarray(
        [
            [
                sample.sample_id,
                int(sample.family),
                int(sample.proxy_type_a),
                int(sample.proxy_type_b),
                sample.radius_a,
                sample.radius_b,
                sample.hardness,
                float(sample.ood),
            ]
            for sample in samples
        ],
        dtype=np.float64,
    )


def dataset_to_npz_arrays(dataset: GeneratedDataset) -> dict[str, np.ndarray]:
    rows = dataset.rows
    for row in rows:
        validate_proposal_feature_row(row)
    if len(dataset.samples) != len(rows) or len(dataset.traces) != len(rows):
        raise ValueError("dataset rows, samples, and traces must have matching lengths")

    return {
        "ids": _ids_array(rows),
        "features": np.asarray([row.features for row in rows], dtype=np.float32).reshape(
            len(rows), PROPOSAL_FEATURE_DIM
        ),
        "interval_targets": np.asarray(
            [row.interval_targets for row in rows], dtype=np.float32
        ).reshape(len(rows), PROPOSAL_INTERVAL_BIN_COUNT),
        "family_targets": np.asarray([row.family_targets for row in rows], dtype=np.float32).reshape(
            len(rows), PROPOSAL_FAMILY_COUNT
        ),
        "scalar_targets": np.asarray(
            [
                [row.priority_target, row.cost_target, row.uncertainty_target]
                for row in rows
            ],
            dtype=np.float32,
        ).reshape(len(rows), 3),
        "split_ids": np.asarray(
            split_ids_for_samples(dataset.samples, dataset.split_names), dtype=np.int32
        ),
        "oracle_trace": _trace_array(dataset.traces),
        "sample_metadata": _sample_array(dataset.samples),
    }


def default_metadata(dataset: GeneratedDataset, *, seed: int, source: str = "python_analytic_oracle") -> dict[str, Any]:
    return {
        "schema_version": DATASET_SHARD_SCHEMA_VERSION,
        "row_count": len(dataset.rows),
        "source": source,
        "seed": seed,
        "split_names": list(dataset.split_names),
        "feature_dim": PROPOSAL_FEATURE_DIM,
        "interval_bins": PROPOSAL_INTERVAL_BIN_COUNT,
        "family_count": PROPOSAL_FAMILY_COUNT,
        "oracle": "analytic_swept_sphere_proxy",
    }


def write_npz_shard(
    path: str | Path,
    dataset: GeneratedDataset,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dataset_to_npz_arrays(dataset)
    meta = dict(metadata or {})
    meta.setdefault("schema_version", DATASET_SHARD_SCHEMA_VERSION)
    meta.setdefault("row_count", len(dataset.rows))
    meta.setdefault("split_names", list(dataset.split_names))
    arrays["metadata_json"] = np.asarray(json.dumps(meta, sort_keys=True), dtype=np.str_)
    np.savez_compressed(output_path, **arrays)


def read_npz_shard(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"].item()))
    if metadata.get("schema_version") != DATASET_SHARD_SCHEMA_VERSION:
        raise ValueError("unsupported dataset shard schema_version")
    if int(metadata.get("row_count", -1)) != int(arrays["features"].shape[0]):
        raise ValueError("dataset shard metadata row_count does not match features")
    return {"metadata": metadata, "arrays": arrays}
