from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

from p2cccd.datasets.tight_inclusion_queries import (
    TIGHT_INCLUSION_QUERY_ROWS,
    TightInclusionCSVFile,
    build_file_level_split,
    discover_tight_inclusion_csv_files,
    infer_case_name,
    infer_query_kind,
)


MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TightInclusionCSVLabelAudit:
    csv_file: TightInclusionCSVFile
    positive_count: int | None
    negative_count: int | None
    sha256: str | None = None


def _parse_cases(values: Iterable[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    cases: list[str] = []
    for value in values:
        for item in str(value).split(","):
            stripped = item.strip()
            if stripped:
                cases.append(stripped)
    return tuple(cases) if cases else None


def audit_tight_inclusion_csv_labels(
    csv_path: Path,
    *,
    dataset_root: Path,
    count_labels: bool,
    compute_sha256: bool = False,
) -> TightInclusionCSVLabelAudit:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    line_count = 0
    query_count = 0
    positive_count = 0
    current_truth: int | None = None
    current_rows = 0
    digest = hashlib.sha256() if compute_sha256 else None
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if digest is not None:
                digest.update(raw_line)
            line_count += 1
            if count_labels:
                stripped = raw_line.rstrip(b"\r\n")
                comma_count = stripped.count(b",")
                if comma_count != 6:
                    raise ValueError(f"{path}:{line_number} expected 7 CSV columns, got {comma_count + 1}")
                truth_byte = stripped[-1:]
                if truth_byte == b"0":
                    truth = 0
                elif truth_byte == b"1":
                    truth = 1
                else:
                    raise ValueError(f"{path}:{line_number} truth must be 0 or 1, got {truth_byte!r}")
                if current_rows == 0:
                    current_truth = truth
                elif truth != current_truth:
                    raise ValueError(f"{path}:{line_number} truth label changes inside one 8-row query")
            current_rows += 1
            if current_rows == TIGHT_INCLUSION_QUERY_ROWS:
                query_count += 1
                if count_labels and current_truth == 1:
                    positive_count += 1
                current_rows = 0
                current_truth = None
    if current_rows != 0:
        raise ValueError(f"{path} has {current_rows} trailing rows, not a complete 8-row query")
    if line_count % TIGHT_INCLUSION_QUERY_ROWS != 0:
        raise ValueError(f"{path} line_count={line_count} is not divisible by 8")
    csv_file = TightInclusionCSVFile(
        case_name=infer_case_name(path, dataset_root),
        kind=infer_query_kind(path),
        csv_path=path,
        line_count=line_count,
        query_count=query_count,
        byte_size=path.stat().st_size,
    )
    return TightInclusionCSVLabelAudit(
        csv_file=csv_file,
        positive_count=positive_count if count_labels else None,
        negative_count=(query_count - positive_count) if count_labels else None,
        sha256=None if digest is None else digest.hexdigest(),
    )


def _split_by_path(split) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name, items in (
        ("train", split.train),
        ("validation", split.validation),
        ("heldout_test", split.heldout_test),
        ("unit_smoke", split.unit_smoke),
    ):
        for item in items:
            mapping[str(item.csv_path.resolve())] = name
    return mapping


def build_tight_inclusion_manifest(
    dataset_root: Path,
    *,
    cases: Iterable[str] | None = None,
    kinds: Iterable[str] = ("vertex-face", "edge-edge"),
    count_labels: bool = True,
    compute_sha256: bool = False,
    seed: int = 424242,
) -> dict[str, object]:
    root = Path(dataset_root).resolve()
    csv_files = discover_tight_inclusion_csv_files(root, cases=cases, kinds=kinds, inspect=False)
    audits = [
        audit_tight_inclusion_csv_labels(
            item.csv_path,
            dataset_root=root,
            count_labels=count_labels,
            compute_sha256=compute_sha256,
        )
        for item in csv_files
    ]
    split = build_file_level_split([audit.csv_file for audit in audits], seed=seed)
    split_mapping = _split_by_path(split)
    rows: list[dict[str, object]] = []
    for audit in audits:
        csv_file = audit.csv_file
        relative_path = csv_file.csv_path.resolve().relative_to(root).as_posix()
        rows.append(
            {
                "case": csv_file.case_name,
                "kind": csv_file.kind,
                "csv_path": relative_path,
                "bytes": csv_file.byte_size,
                "line_count": csv_file.line_count,
                "query_count": csv_file.query_count,
                "positive_count": audit.positive_count,
                "negative_count": audit.negative_count,
                "sha256": audit.sha256,
                "split": split_mapping[str(csv_file.csv_path.resolve())],
            }
        )
    summary = _summarize_manifest_rows(rows)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": root.as_posix(),
        "seed": int(seed),
        "sha256_enabled": bool(compute_sha256),
        "split_policy": {
            "train": 0.70,
            "validation": 0.10,
            "heldout_test": 0.20,
            "unit_smoke_case": "unit-tests",
            "split_granularity": "case/kind/csv_file",
        },
        "summary": summary,
        "files": rows,
    }


def _summarize_manifest_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    total_files = len(rows)
    total_bytes = sum(int(row["bytes"]) for row in rows)
    total_queries = sum(int(row["query_count"]) for row in rows)
    total_positive = sum(int(row["positive_count"] or 0) for row in rows)
    counted_labels = all(row["positive_count"] is not None for row in rows)
    by_case_kind: dict[str, dict[str, int]] = {}
    by_split: dict[str, int] = {}
    for row in rows:
        case_key = f"{row['case']}/{row['kind']}"
        by_case_kind.setdefault(case_key, {"files": 0, "queries": 0})
        by_case_kind[case_key]["files"] += 1
        by_case_kind[case_key]["queries"] += int(row["query_count"])
        by_split[str(row["split"])] = by_split.get(str(row["split"]), 0) + int(row["query_count"])
    return {
        "file_count": total_files,
        "total_bytes": total_bytes,
        "total_queries": total_queries,
        "labels_counted": counted_labels,
        "positive_count": total_positive if counted_labels else None,
        "negative_count": (total_queries - total_positive) if counted_labels else None,
        "positive_ratio": (total_positive / total_queries) if counted_labels and total_queries else None,
        "queries_by_case_kind": by_case_kind,
        "queries_by_split": by_split,
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_manifest_report(path: Path, manifest: dict[str, object]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = manifest["summary"]
    lines = [
        "# Tight-Inclusion / NYU full-query dataset audit",
        "",
        f"- Dataset root: `{manifest['dataset_root']}`",
        f"- CSV files: `{summary['file_count']}`",
        f"- Total bytes: `{summary['total_bytes']}`",
        f"- Total primitive CCD queries: `{summary['total_queries']}`",
        f"- Labels counted: `{summary['labels_counted']}`",
        f"- Positive count: `{summary['positive_count']}`",
        f"- Negative count: `{summary['negative_count']}`",
        f"- Positive ratio: `{summary['positive_ratio']}`",
        "",
        "## Split Query Counts",
        "",
        "| Split | Queries |",
        "| --- | ---: |",
    ]
    for split, count in sorted(summary["queries_by_split"].items()):
        lines.append(f"| `{split}` | `{count}` |")
    lines.extend(["", "## Case / Kind", "", "| Case / Kind | Files | Queries |", "| --- | ---: | ---: |"])
    for key, values in sorted(summary["queries_by_case_kind"].items()):
        lines.append(f"| `{key}` | `{values['files']}` | `{values['queries']}` |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--case", "--cases", dest="cases", action="append")
    parser.add_argument("--kind", "--kinds", dest="kinds", action="append", default=None)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--skip-label-count", action="store_true")
    parser.add_argument("--sha256", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_tight_inclusion_manifest(
        args.root,
        cases=_parse_cases(args.cases),
        kinds=_parse_cases(args.kinds) or ("vertex-face", "edge-edge"),
        count_labels=not args.skip_label_count,
        compute_sha256=args.sha256,
        seed=args.seed,
    )
    write_manifest(args.output, manifest)
    write_manifest_report(args.report, manifest)


if __name__ == "__main__":
    main()
