from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from p2cccd.datasets.ccd100_full_scene_queries import (
    FullSceneCCDQueryFile,
    discover_full_scene_query_files,
    iter_full_scene_queries,
)
from p2cccd.datasets.tight_inclusion_queries import (
    TightInclusionCSVFile,
    discover_tight_inclusion_csv_files,
    iter_tight_inclusion_queries,
)
from p2cccd.datasets.tight_inclusion_stpf_features import tight_inclusion_query_to_proposal_row

from .tight_inclusion_stpf_dataset import STPFShardRecord, _existing_chunk_row_count, _records_to_npz


DATASET_NAME = "rtstpfexact_ccd100_full_run_id"


@dataclass(frozen=True, slots=True)
class CCD100SourceBudget:
    original: int | None
    rounded: int | None
    full_scene: int | None


@dataclass(frozen=True, slots=True)
class CCD100Preset:
    train: CCD100SourceBudget
    validation: CCD100SourceBudget
    heldout_test: CCD100SourceBudget


PRESETS: dict[str, CCD100Preset] = {
    "smoke": CCD100Preset(
        train=CCD100SourceBudget(original=100_000, rounded=100_000, full_scene=300_000),
        validation=CCD100SourceBudget(original=20_000, rounded=20_000, full_scene=60_000),
        heldout_test=CCD100SourceBudget(original=20_000, rounded=20_000, full_scene=60_000),
    ),
    "balanced_large": CCD100Preset(
        train=CCD100SourceBudget(original=8_000_000, rounded=6_000_000, full_scene=6_000_000),
        validation=CCD100SourceBudget(original=1_500_000, rounded=1_000_000, full_scene=1_500_000),
        heldout_test=CCD100SourceBudget(original=1_500_000, rounded=1_000_000, full_scene=1_500_000),
    ),
    "paper_full": CCD100Preset(
        train=CCD100SourceBudget(original=45_000_000, rounded=35_000_000, full_scene=10_000_000),
        validation=CCD100SourceBudget(original=5_000_000, rounded=4_000_000, full_scene=2_000_000),
        heldout_test=CCD100SourceBudget(original=8_000_000, rounded=6_000_000, full_scene=2_000_000),
    ),
    "full_streaming": CCD100Preset(
        train=CCD100SourceBudget(original=None, rounded=None, full_scene=None),
        validation=CCD100SourceBudget(original=None, rounded=None, full_scene=None),
        heldout_test=CCD100SourceBudget(original=None, rounded=None, full_scene=None),
    ),
}


def _stable_unit(token: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{token}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64 - 1)


def _split_for_token(token: str, seed: int) -> str:
    value = _stable_unit(token, seed)
    if value < 0.70:
        return "train"
    if value < 0.80:
        return "validation"
    return "heldout_test"


def _csv_token(path: Path, root: Path, source: str) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = path.as_posix()
    return f"{source}:{rel}"


def _positive_repeat(priority_target: float, positive_oversample: int) -> int:
    if positive_oversample <= 0:
        raise ValueError("positive_oversample must be positive")
    return positive_oversample if priority_target >= 0.999 else 1


def _iter_original_records(
    manifest_path: Path,
    *,
    split: str,
    row_limit: int | None,
    positive_oversample: int,
) -> Iterator[STPFShardRecord]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    root = Path(str(manifest["dataset_root"]))
    emitted = 0
    for file_row in manifest["files"]:
        if file_row["split"] != split:
            continue
        csv_path = root / str(file_row["csv_path"])
        for query in iter_tight_inclusion_queries(csv_path, dataset_root=root):
            row = tight_inclusion_query_to_proposal_row(query)
            for _ in range(_positive_repeat(row.priority_target, positive_oversample)):
                yield STPFShardRecord(
                    row=row,
                    case_name=f"original/{query.case_name}",
                    kind=query.kind,
                    ground_truth=query.ground_truth,
                    csv_path=str(file_row["csv_path"]),
                    query_index=query.query_index,
                )
                emitted += 1
                if row_limit is not None and emitted >= row_limit:
                    return


def _iter_rounded_records(
    rounded_root: Path,
    *,
    split: str,
    row_limit: int | None,
    positive_oversample: int,
    seed: int,
) -> Iterator[STPFShardRecord]:
    root = Path(rounded_root)
    csv_files = discover_tight_inclusion_csv_files(root, inspect=False)
    emitted = 0
    for csv_file in csv_files:
        token = _csv_token(csv_file.csv_path, root, "rounded")
        if _split_for_token(token, seed) != split:
            continue
        for query in iter_tight_inclusion_queries(csv_file.csv_path, dataset_root=root):
            row = tight_inclusion_query_to_proposal_row(query)
            for _ in range(_positive_repeat(row.priority_target, positive_oversample)):
                rel = csv_file.csv_path.resolve().relative_to(root.resolve()).as_posix()
                yield STPFShardRecord(
                    row=row,
                    case_name=f"rounded/{query.case_name}",
                    kind=query.kind,
                    ground_truth=query.ground_truth,
                    csv_path=rel,
                    query_index=query.query_index,
                )
                emitted += 1
                if row_limit is not None and emitted >= row_limit:
                    return


def _iter_full_scene_records(
    dataset_root: Path,
    *,
    split: str,
    row_limit: int | None,
    positive_oversample: int,
    seed: int,
) -> Iterator[STPFShardRecord]:
    root = Path(dataset_root)
    query_files = discover_full_scene_query_files(root)
    emitted = 0
    for query_file in query_files:
        token = _csv_token(query_file.csv_path, root, "full_scene")
        if _split_for_token(token, seed) != split:
            continue
        for query in iter_full_scene_queries(query_file, dataset_root=root):
            row = tight_inclusion_query_to_proposal_row(query)
            for _ in range(_positive_repeat(row.priority_target, positive_oversample)):
                rel = query_file.csv_path.resolve().relative_to(root.resolve()).as_posix()
                yield STPFShardRecord(
                    row=row,
                    case_name=f"full_scene/{query.case_name}",
                    kind=query.kind,
                    ground_truth=query.ground_truth,
                    csv_path=rel,
                    query_index=query.query_index,
                )
                emitted += 1
                if row_limit is not None and emitted >= row_limit:
                    return


def _chain_iterables(iterables: Sequence[Iterable[STPFShardRecord]]) -> Iterator[STPFShardRecord]:
    for iterable in iterables:
        yield from iterable


def _budget_for_split(preset: CCD100Preset, split: str) -> CCD100SourceBudget:
    if split == "train":
        return preset.train
    if split == "validation":
        return preset.validation
    if split == "heldout_test":
        return preset.heldout_test
    raise ValueError(f"unsupported split for CCD100 preset: {split}")


def _write_split_chunks(
    records: Iterable[STPFShardRecord],
    *,
    output_dir: Path,
    split: str,
    chunk_rows: int,
    resume: bool,
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    buffer: list[STPFShardRecord] = []
    chunk_index = 0
    for record in records:
        buffer.append(record)
        if len(buffer) >= chunk_rows:
            path = output_dir / split / f"chunk_{chunk_index:06d}.npz"
            existed_before = path.exists()
            if resume and existed_before:
                row_count = _existing_chunk_row_count(path)
            else:
                _records_to_npz(
                    path,
                    buffer,
                    metadata={
                        "source": DATASET_NAME,
                        "split": split,
                        "chunk_index": chunk_index,
                    },
                )
                row_count = len(buffer)
            chunks.append(
                {"split": split, "path": path.as_posix(), "row_count": row_count, "resumed": bool(resume and existed_before)}
            )
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
                metadata={"source": DATASET_NAME, "split": split, "chunk_index": chunk_index},
            )
            row_count = len(buffer)
        chunks.append(
            {"split": split, "path": path.as_posix(), "row_count": row_count, "resumed": bool(resume and existed_before)}
        )
    return chunks


def _summarize_full_scene(dataset_root: Path) -> dict[str, object]:
    files = discover_full_scene_query_files(dataset_root)
    by_scene: dict[str, dict[str, int]] = {}
    for item in files:
        row = by_scene.setdefault(item.scene_name, {"files": 0, "queries": 0, "positives": 0, "bytes": 0})
        row["files"] += 1
        row["queries"] += item.query_count
        row["positives"] += item.positive_count
        row["bytes"] += item.byte_size
    return {
        "file_count": len(files),
        "query_count": sum(item.query_count for item in files),
        "positive_count": sum(item.positive_count for item in files),
        "bytes": sum(item.byte_size for item in files),
        "by_scene": by_scene,
    }


def _summarize_rounded(rounded_root: Path) -> dict[str, object]:
    files = discover_tight_inclusion_csv_files(rounded_root, inspect=False)
    by_case: dict[str, dict[str, int]] = {}
    for item in files:
        row = by_case.setdefault(item.case_name, {"files": 0, "bytes": 0})
        row["files"] += 1
        row["bytes"] += item.byte_size
    return {
        "file_count": len(files),
        "bytes": sum(item.byte_size for item in files),
        "by_case": by_case,
        "query_count": None,
        "positive_count": None,
    }


def build_rtstpfexact_ccd100_shards(
    dataset_root: Path,
    original_manifest_path: Path,
    *,
    output_dir: Path,
    preset: str = "smoke",
    splits: Sequence[str] = ("train", "validation", "heldout_test"),
    chunk_rows: int = 1_000_000,
    resume: bool = True,
    positive_oversample: int = 8,
    seed: int = 424242,
) -> dict[str, object]:
    if preset not in PRESETS:
        raise ValueError(f"unsupported CCD100 preset {preset!r}; expected one of {sorted(PRESETS)}")
    root = Path(dataset_root)
    rounded_root = root / "rounded_ground_truth"
    if not rounded_root.exists():
        raise FileNotFoundError(rounded_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    preset_config = PRESETS[preset]
    chunks: list[dict[str, object]] = []
    for split in splits:
        budget = _budget_for_split(preset_config, split)
        records = _chain_iterables(
            [
                _iter_original_records(
                    original_manifest_path,
                    split=split,
                    row_limit=budget.original,
                    positive_oversample=positive_oversample,
                ),
                _iter_rounded_records(
                    rounded_root,
                    split=split,
                    row_limit=budget.rounded,
                    positive_oversample=positive_oversample,
                    seed=seed,
                ),
                _iter_full_scene_records(
                    root,
                    split=split,
                    row_limit=budget.full_scene,
                    positive_oversample=positive_oversample,
                    seed=seed,
                ),
            ]
        )
        chunks.extend(
            _write_split_chunks(
                records,
                output_dir=output_dir,
                split=split,
                chunk_rows=chunk_rows,
                resume=resume,
            )
        )
    original_manifest = json.loads(Path(original_manifest_path).read_text(encoding="utf-8"))
    summary = {
        "dataset_name": DATASET_NAME,
        "dataset_root": root.as_posix(),
        "original_manifest_path": Path(original_manifest_path).as_posix(),
        "preset": preset,
        "seed": seed,
        "chunk_rows": int(chunk_rows),
        "positive_oversample": int(positive_oversample),
        "resume": bool(resume),
        "splits": list(splits),
        "source_summary": {
            "original": original_manifest.get("summary", {}),
            "rounded": _summarize_rounded(rounded_root),
            "full_scene": _summarize_full_scene(root),
        },
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--split", dest="splits", action="append", default=None)
    parser.add_argument("--positive-oversample", type=int, default=8)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_rtstpfexact_ccd100_shards(
        args.root,
        args.original_manifest,
        output_dir=args.output,
        preset=args.preset,
        splits=tuple(args.splits) if args.splits else ("train", "validation", "heldout_test"),
        chunk_rows=args.chunk_rows,
        resume=not args.no_resume,
        positive_oversample=args.positive_oversample,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
