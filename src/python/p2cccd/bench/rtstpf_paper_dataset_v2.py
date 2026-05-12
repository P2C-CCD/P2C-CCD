from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from p2cccd.data.dataset import DATASET_SHARD_SCHEMA_VERSION
from p2cccd.datasets.ccd100_full_scene_queries import (
    FullSceneCCDQueryFile,
    discover_full_scene_query_files,
    iter_full_scene_queries,
)
from p2cccd.datasets.tight_inclusion_queries import (
    TIGHT_INCLUSION_QUERY_ROWS,
    TightInclusionPrimitiveQuery,
    inspect_tight_inclusion_csv,
    iter_tight_inclusion_queries,
    parse_query_lines,
)
from p2cccd.datasets.tight_inclusion_stpf_features import tight_inclusion_query_to_proposal_row
from p2cccd.proposal.features import (
    PROPOSAL_FAMILY_COUNT,
    PROPOSAL_FEATURE_DIM,
    PROPOSAL_INTERVAL_BIN_COUNT,
    ProposalFeatureRow,
    validate_proposal_feature_row,
)


MANIFEST_SCHEMA_VERSION = 1
SOURCE_ORIGINAL = "original_full_query"
SOURCE_ROUNDED = "rounded_ground_truth"
SOURCE_FULL_SCENE = "full_scene_toi"


PRESET_ROW_LIMITS: dict[str, dict[str, int | None]] = {
    "smoke": {"train": 100_000, "validation": 20_000, "heldout_test": 20_000},
    "large": {"train": 20_000_000, "validation": 2_000_000, "heldout_test": 2_000_000},
    "paper_full": {"train": 80_000_000, "validation": 8_000_000, "heldout_test": 8_000_000},
    "full_all": {"train": None, "validation": None, "heldout_test": None},
}


@dataclass(frozen=True, slots=True)
class PaperDatasetV2Record:
    row: ProposalFeatureRow
    source_type: str
    case_name: str
    kind: str
    ground_truth: bool
    csv_path: str
    query_index: int


def _stable_unit_interval(text: str) -> float:
    value = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)
    return float(value) / float(0xFFFFFFFFFFFF)


def _split_for_file(source_type: str, case_name: str, kind: str, csv_path: str) -> str:
    if case_name == "unit-tests":
        return "unit_smoke"
    value = _stable_unit_interval(f"{source_type}/{case_name}/{kind}/{csv_path}")
    if value < 0.70:
        return "train"
    if value < 0.80:
        return "validation"
    return "heldout_test"


def _kind_from_original_path(path: Path) -> str:
    parts = set(path.parts)
    if "vertex-face" in parts:
        return "vertex-face"
    if "edge-edge" in parts:
        return "edge-edge"
    raise ValueError(f"cannot infer kind from {path}")


def _discover_tight_inclusion_tree(
    root: Path,
    *,
    source_type: str,
    source_root: Path,
    manifest_root: Path,
    inspect_counts: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for csv_path in sorted(source_root.glob("*/*/*.csv")):
        if csv_path.name.startswith("._"):
            continue
        parent = csv_path.parent.name
        if parent not in {"vertex-face", "edge-edge"}:
            continue
        case_name = csv_path.parent.parent.name
        kind = _kind_from_original_path(csv_path)
        query_count = None
        positive_count = None
        if inspect_counts:
            audit = inspect_tight_inclusion_csv(csv_path, dataset_root=source_root)
            query_count = int(audit.query_count)
        rel = csv_path.resolve().relative_to(manifest_root.resolve()).as_posix()
        rows.append(
            {
                "source_type": source_type,
                "case": case_name,
                "kind": kind,
                "csv_path": rel,
                "label_path": None,
                "bytes": int(csv_path.stat().st_size),
                "query_count": query_count,
                "positive_count": positive_count,
                "negative_count": None,
                "split": _split_for_file(source_type, case_name, kind, rel),
            }
        )
    return rows


def _discover_full_scene_tree(root: Path, *, inspect_counts: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in discover_full_scene_query_files(root):
        rel = item.csv_path.resolve().relative_to(root.resolve()).as_posix()
        label_rel = item.mma_bool_path.resolve().relative_to(root.resolve()).as_posix()
        rows.append(
            {
                "source_type": SOURCE_FULL_SCENE,
                "case": item.scene_name,
                "kind": item.kind,
                "csv_path": rel,
                "label_path": label_rel,
                "step_id": item.step_id,
                "box_path": None if item.box_path is None else item.box_path.resolve().relative_to(root.resolve()).as_posix(),
                "bytes": int(item.byte_size),
                "query_count": int(item.query_count) if inspect_counts else int(item.query_count),
                "positive_count": int(item.positive_count) if inspect_counts else int(item.positive_count),
                "negative_count": int(item.query_count - item.positive_count) if inspect_counts else int(item.query_count - item.positive_count),
                "split": _split_for_file(SOURCE_FULL_SCENE, item.scene_name, item.kind, rel),
            }
        )
    return rows


def build_rtstpf_paper_dataset_v2_manifest(
    dataset_root: Path,
    *,
    inspect_counts: bool = False,
    seed: int = 424242,
) -> dict[str, object]:
    root = Path(dataset_root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    rows: list[dict[str, object]] = []
    rows.extend(
        _discover_tight_inclusion_tree(
            root,
            source_type=SOURCE_ORIGINAL,
            source_root=root,
            manifest_root=root,
            inspect_counts=inspect_counts,
        )
    )
    rounded_root = root / "rounded_ground_truth"
    if rounded_root.exists():
        rows.extend(
            _discover_tight_inclusion_tree(
                root,
                source_type=SOURCE_ROUNDED,
                source_root=rounded_root,
                manifest_root=root,
                inspect_counts=inspect_counts,
            )
        )
    rows.extend(_discover_full_scene_tree(root, inspect_counts=inspect_counts))
    rows.sort(key=lambda row: (str(row["source_type"]), str(row["case"]), str(row["kind"]), str(row["csv_path"])))
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": root.as_posix(),
        "seed": int(seed),
        "split_policy": {
            "train": 0.70,
            "validation": 0.10,
            "heldout_test": 0.20,
            "unit_smoke_case": "unit-tests",
            "split_granularity": "source_type/case/kind/csv_file",
        },
        "sources": [SOURCE_ORIGINAL, SOURCE_ROUNDED, SOURCE_FULL_SCENE],
        "summary": _summarize_rows(rows),
        "files": rows,
    }


def _summarize_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_source: dict[str, dict[str, int]] = {}
    by_split: dict[str, dict[str, int]] = {}
    by_case: dict[str, dict[str, int]] = {}
    total_bytes = 0
    total_known_queries = 0
    total_known_positive = 0
    for row in rows:
        source = str(row["source_type"])
        split = str(row["split"])
        case = str(row["case"])
        query_count = int(row["query_count"] or 0)
        positive_count = int(row["positive_count"] or 0)
        total_bytes += int(row["bytes"])
        total_known_queries += query_count
        total_known_positive += positive_count
        for table, key in ((by_source, source), (by_split, split), (by_case, case)):
            table.setdefault(key, {"files": 0, "bytes": 0, "known_queries": 0, "known_positive": 0})
            table[key]["files"] += 1
            table[key]["bytes"] += int(row["bytes"])
            table[key]["known_queries"] += query_count
            table[key]["known_positive"] += positive_count
    return {
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "known_query_count": total_known_queries,
        "known_positive_count": total_known_positive,
        "known_positive_ratio": total_known_positive / total_known_queries if total_known_queries else None,
        "by_source": by_source,
        "by_split": by_split,
        "by_case": by_case,
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _full_scene_query_file_from_manifest(root: Path, row: dict[str, object]) -> FullSceneCCDQueryFile:
    csv_path = root / str(row["csv_path"])
    label_path = root / str(row["label_path"])
    box_path = None if row.get("box_path") is None else root / str(row["box_path"])
    return FullSceneCCDQueryFile(
        scene_name=str(row["case"]),
        kind=str(row["kind"]),  # type: ignore[arg-type]
        step_id=int(row.get("step_id", 0)),
        csv_path=csv_path,
        mma_bool_path=label_path,
        box_path=box_path if box_path is not None and box_path.exists() else None,
        query_count=int(row["query_count"] or 0),
        positive_count=int(row["positive_count"] or 0),
        byte_size=int(row["bytes"]),
    )


def _iter_queries_for_manifest_file(
    root: Path,
    row: dict[str, object],
) -> Iterator[TightInclusionPrimitiveQuery]:
    source_type = str(row["source_type"])
    csv_path = root / str(row["csv_path"])
    if source_type == SOURCE_FULL_SCENE:
        yield from iter_full_scene_queries(_full_scene_query_file_from_manifest(root, row), dataset_root=root)
        return
    dataset_root = root / "rounded_ground_truth" if source_type == SOURCE_ROUNDED else root
    yield from _iter_tight_inclusion_queries_tolerant(
        csv_path,
        case_name=str(row["case"]),
        kind=str(row["kind"]),
        dataset_root=dataset_root,
    )


def _normalize_truth_line(line: str) -> str:
    columns = line.strip().split(",")
    if len(columns) != 7:
        return line
    # Rounded Root-Parity files can encode non-colliding/ambiguous blocks as -1.
    # STPF is only a scheduler; treat every non-1 label as negative so the exact
    # fallback remains responsible for final correctness.
    columns[-1] = "1" if columns[-1].strip() == "1" else "0"
    return ",".join(columns)


def _iter_tight_inclusion_queries_tolerant(
    csv_path: Path,
    *,
    case_name: str,
    kind: str,
    dataset_root: Path,
) -> Iterator[TightInclusionPrimitiveQuery]:
    path = Path(csv_path)
    query_lines: list[str] = []
    query_index = 0
    first_line = 1
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            query_lines.append(_normalize_truth_line(stripped))
            if len(query_lines) == TIGHT_INCLUSION_QUERY_ROWS:
                yield parse_query_lines(
                    query_lines,
                    case_name=case_name,
                    kind=kind,  # type: ignore[arg-type]
                    csv_path=path,
                    query_index=query_index,
                    first_line_number=first_line,
                )
                query_index += 1
                query_lines = []
                first_line = line_number + 1
    if query_lines:
        raise ValueError(f"{path} has trailing partial query with {len(query_lines)} rows")


def _records_to_npz(path: Path, records: Sequence[PaperDatasetV2Record], *, metadata: dict[str, object]) -> None:
    record_list = list(records)
    rows = [validate_proposal_feature_row(record.row) for record in record_list]
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "ids": np.asarray(
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
        ).reshape(len(rows), 9),
        "features": np.asarray([row.features for row in rows], dtype=np.float32).reshape(len(rows), PROPOSAL_FEATURE_DIM),
        "interval_targets": np.asarray([row.interval_targets for row in rows], dtype=np.float32).reshape(
            len(rows), PROPOSAL_INTERVAL_BIN_COUNT
        ),
        "family_targets": np.asarray([row.family_targets for row in rows], dtype=np.float32).reshape(
            len(rows), PROPOSAL_FAMILY_COUNT
        ),
        "scalar_targets": np.asarray(
            [[row.priority_target, row.cost_target, row.uncertainty_target] for row in rows],
            dtype=np.float32,
        ).reshape(len(rows), 3),
        "ground_truth": np.asarray([record.ground_truth for record in record_list], dtype=np.bool_),
        "source_types": np.asarray([record.source_type for record in record_list], dtype=np.str_),
        "case_names": np.asarray([record.case_name for record in record_list], dtype=np.str_),
        "kind_names": np.asarray([record.kind for record in record_list], dtype=np.str_),
        "csv_paths": np.asarray([record.csv_path for record in record_list], dtype=np.str_),
        "source_query_indices": np.asarray([record.query_index for record in record_list], dtype=np.uint64),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    **metadata,
                    "schema_version": DATASET_SHARD_SCHEMA_VERSION,
                    "row_count": len(rows),
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


def _source_case_kind_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (str(row["source_type"]), str(row["case"]), str(row["kind"]))


def _manifest_rows_for_split(manifest: dict[str, object], split: str) -> list[dict[str, object]]:
    return [row for row in manifest["files"] if row["split"] == split]  # type: ignore[index]


def _iter_records_balanced(
    manifest: dict[str, object],
    *,
    split: str,
    row_limit: int | None,
    positive_oversample: int,
) -> Iterator[PaperDatasetV2Record]:
    root = Path(str(manifest["dataset_root"]))
    file_rows = _manifest_rows_for_split(manifest, split)
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in file_rows:
        groups.setdefault(_source_case_kind_key(row), []).append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: str(row["csv_path"]))
    group_count = max(1, len(groups))
    per_group_limit = None if row_limit is None else int(math.ceil(float(row_limit) / float(group_count)))
    emitted_total = 0
    for key in sorted(groups):
        emitted_group = 0
        for file_row in groups[key]:
            for query in _iter_queries_for_manifest_file(root, file_row):
                row = tight_inclusion_query_to_proposal_row(query)
                repeat_count = positive_oversample if query.ground_truth else 1
                for _ in range(repeat_count):
                    yield PaperDatasetV2Record(
                        row=row,
                        source_type=str(file_row["source_type"]),
                        case_name=query.case_name,
                        kind=query.kind,
                        ground_truth=query.ground_truth,
                        csv_path=str(file_row["csv_path"]),
                        query_index=query.query_index,
                    )
                    emitted_total += 1
                    emitted_group += 1
                    if row_limit is not None and emitted_total >= row_limit:
                        return
                    if per_group_limit is not None and emitted_group >= per_group_limit:
                        break
                if per_group_limit is not None and emitted_group >= per_group_limit:
                    break
            if per_group_limit is not None and emitted_group >= per_group_limit:
                break


def build_rtstpf_paper_dataset_v2_shards(
    manifest_path: Path,
    *,
    output_dir: Path,
    preset: str = "smoke",
    splits: Sequence[str] = ("train", "validation", "heldout_test"),
    chunk_rows: int = 500_000,
    positive_oversample: int = 4,
    resume: bool = True,
) -> dict[str, object]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if preset not in PRESET_ROW_LIMITS:
        raise ValueError(f"unsupported preset {preset!r}; expected {sorted(PRESET_ROW_LIMITS)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, object]] = []
    row_limits = PRESET_ROW_LIMITS[preset]
    for split in splits:
        buffer: list[PaperDatasetV2Record] = []
        chunk_index = 0
        for record in _iter_records_balanced(
            manifest,
            split=split,
            row_limit=row_limits.get(split),
            positive_oversample=positive_oversample,
        ):
            buffer.append(record)
            if len(buffer) >= chunk_rows:
                path = output_dir / split / f"chunk_{chunk_index:06d}.npz"
                existed = path.exists()
                if not (resume and existed):
                    _records_to_npz(
                        path,
                        buffer,
                        metadata={
                            "source": "rtstpf_paper_dataset_v2",
                            "split": split,
                            "preset": preset,
                            "chunk_index": chunk_index,
                        },
                    )
                chunks.append({"split": split, "path": path.as_posix(), "row_count": len(buffer), "resumed": bool(resume and existed)})
                buffer = []
                chunk_index += 1
        if buffer:
            path = output_dir / split / f"chunk_{chunk_index:06d}.npz"
            existed = path.exists()
            if not (resume and existed):
                _records_to_npz(
                    path,
                    buffer,
                    metadata={
                        "source": "rtstpf_paper_dataset_v2",
                        "split": split,
                        "preset": preset,
                        "chunk_index": chunk_index,
                    },
                )
            chunks.append({"split": split, "path": path.as_posix(), "row_count": len(buffer), "resumed": bool(resume and existed)})
    shard_manifest = {
        "schema_version": DATASET_SHARD_SCHEMA_VERSION,
        "dataset": "rtstpf_paper_dataset_v2",
        "preset": preset,
        "manifest_path": Path(manifest_path).as_posix(),
        "output_dir": output_dir.as_posix(),
        "chunk_rows": int(chunk_rows),
        "positive_oversample": int(positive_oversample),
        "chunks": chunks,
        "row_counts_by_split": {
            split: sum(int(chunk["row_count"]) for chunk in chunks if chunk["split"] == split)
            for split in splits
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(shard_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return shard_manifest


def write_design_report(path: Path, manifest: dict[str, object], *, shard_plan: dict[str, object] | None = None) -> Path:
    summary = manifest["summary"]
    lines = [
        "# RTSTPFExact Paper Dataset v2 design report",
        "",
        "## Objective",
        "",
        "- coverage `continuous-collision-detection` Objectivedescriptionunder 100GB+ description. ",
        "- description original full-query, rounded ground truth, full-scene TOI description STPF proposal training rows. ",
        "- descriptionuse chunked NPZ + streaming trainer, supportdescriptionwhendescriptionsplitdescription, descriptionloaddescription rows. ",
        "",
        "## data source",
        "",
        f"- Dataset root: `{manifest['dataset_root']}`",
        f"- File count: `{summary['file_count']}`",
        f"- Bytes indexed: `{summary['total_bytes']}`",
        f"- Known queries: `{summary['known_query_count']}`",
        f"- Known positives: `{summary['known_positive_count']}`",
        f"- Known positive ratio: `{summary['known_positive_ratio']}`",
        "",
        "| Source | Files | Bytes | Known queries | Known positives |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source, row in sorted(summary["by_source"].items()):
        lines.append(
            f"| `{source}` | `{row['files']}` | `{row['bytes']}` | `{row['known_queries']}` | `{row['known_positive']}` |"
        )
    lines.extend(["", "## Split", "", "| Split | Files | Bytes | Known queries | Known positives |", "| --- | ---: | ---: | ---: | ---: |"])
    for split, row in sorted(summary["by_split"].items()):
        lines.append(
            f"| `{split}` | `{row['files']}` | `{row['bytes']}` | `{row['known_queries']}` | `{row['known_positive']}` |"
        )
    lines.extend(
        [
            "",
            "## description",
            "",
            "| Preset | Train rows | Validation rows | Heldout rows | usedescription |",
            "| --- | ---: | ---: | ---: | --- |",
            "| `smoke` | `100000` | `20000` | `20000` | reader/shard/training smoke |",
            "| `large` | `20000000` | `2000000` | `2000000` | description |",
            "| `paper_full` | `80000000` | `8000000` | `8000000` | description, descriptionsplitcoverage 100GB+ description |",
            "| `full_all` | all | all | all | description streaming, descriptionwhendescription |",
            "",
            "## description",
            "",
            "- split granularity is `source_type/case/kind/csv_file`, avoid the same CSV description query leakage. ",
            "- shard writer by `source_type/case/kind` perform group-balanced Output, avoid row limit description case description. ",
            "- positive query default oversample, guaranteedescription primitive/full-scene description STPF descriptionlevel. ",
            "- `RTSTPFExact` description proposal/scheduling; description collision Conclusiondescription exact certificate / fallback description. ",
            "- full-scene TOI descriptionuse `queries/*.csv + mma_bool/*.json`, rounded/original descriptionusedescription truth CSV. ",
        ]
    )
    if shard_plan is not None:
        lines.extend(
            [
                "",
                "## descriptiongenerate Shard",
                "",
                f"- Output dir: `{shard_plan['output_dir']}`",
                f"- Preset: `{shard_plan['preset']}`",
                f"- Positive oversample: `{shard_plan['positive_oversample']}`",
                "",
                "| Split | Rows |",
                "| --- | ---: |",
            ]
        )
        for split, count in sorted(shard_plan["row_counts_by_split"].items()):
            lines.append(f"| `{split}` | `{count}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--root", type=Path, required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--report", type=Path, required=True)
    manifest_parser.add_argument("--inspect-counts", action="store_true")
    shard_parser = subparsers.add_parser("shards")
    shard_parser.add_argument("--manifest", type=Path, required=True)
    shard_parser.add_argument("--output-dir", type=Path, required=True)
    shard_parser.add_argument("--preset", default="smoke", choices=sorted(PRESET_ROW_LIMITS))
    shard_parser.add_argument("--chunk-rows", type=int, default=500_000)
    shard_parser.add_argument("--positive-oversample", type=int, default=4)
    shard_parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "manifest":
        manifest = build_rtstpf_paper_dataset_v2_manifest(args.root, inspect_counts=args.inspect_counts)
        write_manifest(args.output, manifest)
        write_design_report(args.report, manifest)
        return
    if args.cmd == "shards":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        shard_plan = build_rtstpf_paper_dataset_v2_shards(
            args.manifest,
            output_dir=args.output_dir,
            preset=args.preset,
            chunk_rows=args.chunk_rows,
            positive_oversample=args.positive_oversample,
        )
        write_design_report(args.report, manifest, shard_plan=shard_plan)
        return
    raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
