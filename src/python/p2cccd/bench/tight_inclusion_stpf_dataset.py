from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from p2cccd.data.shards import read_npz_shard
from p2cccd.data.dataset import DATASET_SHARD_SCHEMA_VERSION
from p2cccd.datasets.tight_inclusion_queries import iter_tight_inclusion_queries
from p2cccd.datasets.tight_inclusion_stpf_features import tight_inclusion_query_to_proposal_row
from p2cccd.proposal.features import (
    PROPOSAL_FAMILY_COUNT,
    PROPOSAL_FEATURE_DIM,
    PROPOSAL_INTERVAL_BIN_COUNT,
    ProposalFeatureRow,
    validate_proposal_feature_row,
)


@dataclass(frozen=True, slots=True)
class STPFShardRecord:
    row: ProposalFeatureRow
    case_name: str
    kind: str
    ground_truth: bool
    csv_path: str
    query_index: int


PRESET_ROW_LIMITS: dict[str, dict[str, int | None]] = {
    "smoke": {"train": 50_000, "validation": 10_000, "heldout_test": 10_000, "unit_smoke": None},
    "medium": {"train": 2_000_000, "validation": 400_000, "heldout_test": 400_000, "unit_smoke": None},
    "large": {"train": 10_000_000, "validation": 2_000_000, "heldout_test": 2_000_000, "unit_smoke": None},
    "full": {"train": None, "validation": None, "heldout_test": None, "unit_smoke": None},
}


def _records_to_npz(path: Path, records: Sequence[STPFShardRecord], *, metadata: dict[str, object]) -> None:
    record_list = list(records)
    row_list = [validate_proposal_feature_row(record.row) for record in record_list]
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(
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
            for row in row_list
        ],
        dtype=np.uint64,
    )
    arrays = {
        "ids": ids.reshape(len(row_list), 9),
        "features": np.asarray([row.features for row in row_list], dtype=np.float32).reshape(
            len(row_list), PROPOSAL_FEATURE_DIM
        ),
        "interval_targets": np.asarray([row.interval_targets for row in row_list], dtype=np.float32).reshape(
            len(row_list), PROPOSAL_INTERVAL_BIN_COUNT
        ),
        "family_targets": np.asarray([row.family_targets for row in row_list], dtype=np.float32).reshape(
            len(row_list), PROPOSAL_FAMILY_COUNT
        ),
        "scalar_targets": np.asarray(
            [[row.priority_target, row.cost_target, row.uncertainty_target] for row in row_list],
            dtype=np.float32,
        ).reshape(len(row_list), 3),
        "ground_truth": np.asarray([record.ground_truth for record in record_list], dtype=np.bool_),
        "case_names": np.asarray([record.case_name for record in record_list], dtype=np.str_),
        "kind_names": np.asarray([record.kind for record in record_list], dtype=np.str_),
        "csv_paths": np.asarray([record.csv_path for record in record_list], dtype=np.str_),
        "source_query_indices": np.asarray([record.query_index for record in record_list], dtype=np.uint64),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    **metadata,
                    "schema_version": DATASET_SHARD_SCHEMA_VERSION,
                    "row_count": len(row_list),
                    "feature_dim": PROPOSAL_FEATURE_DIM,
                    "interval_bins": PROPOSAL_INTERVAL_BIN_COUNT,
                    "family_count": PROPOSAL_FAMILY_COUNT,
                },
                sort_keys=True,
            ),
            dtype=np.str_,
        ),
    }
    np.savez_compressed(path, **arrays)


def _existing_chunk_row_count(path: Path) -> int:
    shard = read_npz_shard(path)
    return int(shard["metadata"]["row_count"])


def _split_rows(manifest: dict[str, object], split: str) -> list[dict[str, object]]:
    return [row for row in manifest["files"] if row["split"] == split]  # type: ignore[index]


def _row_matches_case_kind_quota(
    file_row: dict[str, object],
    emitted_by_case_kind: dict[tuple[str, str], int],
    *,
    max_rows_per_case_kind: int | None,
) -> bool:
    if max_rows_per_case_kind is None:
        return True
    key = (str(file_row["case"]), str(file_row["kind"]))
    return emitted_by_case_kind.get(key, 0) < max_rows_per_case_kind


def _iter_manifest_split_rows(
    manifest: dict[str, object],
    *,
    split: str,
    row_limit: int | None,
    max_rows_per_case_kind: int | None,
    positive_oversample: int,
) -> Iterable[STPFShardRecord]:
    if positive_oversample <= 0:
        raise ValueError("positive_oversample must be positive")
    root = Path(str(manifest["dataset_root"]))
    emitted = 0
    emitted_by_case_kind: dict[tuple[str, str], int] = {}
    for file_row in _split_rows(manifest, split):
        key = (str(file_row["case"]), str(file_row["kind"]))
        if not _row_matches_case_kind_quota(
            file_row,
            emitted_by_case_kind,
            max_rows_per_case_kind=max_rows_per_case_kind,
        ):
            continue
        csv_path = root / str(file_row["csv_path"])
        for query in iter_tight_inclusion_queries(csv_path, dataset_root=root):
            row = tight_inclusion_query_to_proposal_row(query)
            repeat_count = positive_oversample if row.priority_target >= 0.999 else 1
            for _ in range(repeat_count):
                yield STPFShardRecord(
                    row=row,
                    case_name=query.case_name,
                    kind=query.kind,
                    ground_truth=query.ground_truth,
                    csv_path=str(file_row["csv_path"]),
                    query_index=query.query_index,
                )
                emitted += 1
                emitted_by_case_kind[key] = emitted_by_case_kind.get(key, 0) + 1
                if row_limit is not None and emitted >= row_limit:
                    return
                if (
                    max_rows_per_case_kind is not None
                    and emitted_by_case_kind[key] >= max_rows_per_case_kind
                ):
                    break
            if (
                max_rows_per_case_kind is not None
                and emitted_by_case_kind.get(key, 0) >= max_rows_per_case_kind
            ):
                break


def _write_split_chunks(
    manifest: dict[str, object],
    *,
    split: str,
    output_dir: Path,
    row_limit: int | None,
    chunk_rows: int,
    resume: bool,
    max_rows_per_case_kind: int | None,
    positive_oversample: int,
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    buffer: list[STPFShardRecord] = []
    chunk_index = 0
    for row in _iter_manifest_split_rows(
        manifest,
        split=split,
        row_limit=row_limit,
        max_rows_per_case_kind=max_rows_per_case_kind,
        positive_oversample=positive_oversample,
    ):
        buffer.append(row)
        if len(buffer) >= chunk_rows:
            path = output_dir / split / f"chunk_{chunk_index:06d}.npz"
            existed_before = path.exists()
            if resume and existed_before:
                row_count = _existing_chunk_row_count(path)
            else:
                _records_to_npz(
                    path,
                    buffer,
                    metadata={"source": "tight_inclusion_nyu", "split": split, "chunk_index": chunk_index},
                )
                row_count = len(buffer)
            chunks.append({"split": split, "path": path.as_posix(), "row_count": row_count, "resumed": bool(resume and existed_before)})
            buffer = []
            chunk_index += 1
    if buffer:
        path = output_dir / split / f"chunk_{chunk_index:06d}.npz"
        existed_before = path.exists()
        if resume and existed_before:
            row_count = _existing_chunk_row_count(path)
        else:
            _records_to_npz(
                path,
                buffer,
                metadata={"source": "tight_inclusion_nyu", "split": split, "chunk_index": chunk_index},
            )
            row_count = len(buffer)
        chunks.append({"split": split, "path": path.as_posix(), "row_count": row_count, "resumed": bool(resume and existed_before)})
    return chunks


def build_tight_inclusion_stpf_shards(
    manifest_path: Path,
    *,
    output_dir: Path,
    preset: str = "smoke",
    splits: Sequence[str] = ("train", "validation", "heldout_test", "unit_smoke"),
    chunk_rows: int = 500_000,
    resume: bool = True,
    max_rows_per_case_kind: int | None = None,
    positive_oversample: int = 1,
) -> dict[str, object]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if preset not in PRESET_ROW_LIMITS:
        raise ValueError(f"unsupported preset {preset!r}; expected one of {sorted(PRESET_ROW_LIMITS)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    row_limits = PRESET_ROW_LIMITS[preset]
    chunks: list[dict[str, object]] = []
    for split in splits:
        chunks.extend(
            _write_split_chunks(
                manifest,
                split=split,
                output_dir=output_dir,
                row_limit=row_limits.get(split),
                chunk_rows=chunk_rows,
                resume=resume,
                max_rows_per_case_kind=max_rows_per_case_kind,
                positive_oversample=positive_oversample,
            )
        )
    summary = {
        "source_manifest": Path(manifest_path).as_posix(),
        "preset": preset,
        "chunk_rows": int(chunk_rows),
        "resume": bool(resume),
        "max_rows_per_case_kind": max_rows_per_case_kind,
        "positive_oversample": int(positive_oversample),
        "chunks": chunks,
        "rows_by_split": {
            split: sum(int(chunk["row_count"]) for chunk in chunks if chunk["split"] == split)
            for split in splits
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESET_ROW_LIMITS), default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    parser.add_argument("--split", dest="splits", action="append", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-rows-per-case-kind", type=int, default=None)
    parser.add_argument("--positive-oversample", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_tight_inclusion_stpf_shards(
        args.manifest,
        output_dir=args.output,
        preset=args.preset,
        splits=tuple(args.splits) if args.splits else ("train", "validation", "heldout_test", "unit_smoke"),
        chunk_rows=args.chunk_rows,
        resume=not args.no_resume,
        max_rows_per_case_kind=args.max_rows_per_case_kind,
        positive_oversample=args.positive_oversample,
    )


if __name__ == "__main__":
    main()
