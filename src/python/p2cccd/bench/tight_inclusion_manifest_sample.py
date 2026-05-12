from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


PRESET_BUDGETS: dict[str, dict[str, int | None]] = {
    "medium": {"train": 250_000, "validation": 50_000, "heldout_test": 50_000, "unit_smoke": None},
    "large": {"train": 1_250_000, "validation": 250_000, "heldout_test": 250_000, "unit_smoke": None},
    "full": {"train": None, "validation": None, "heldout_test": None, "unit_smoke": None},
}


def _group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["split"]), str(row["case"]), str(row["kind"])


def _select_rows_for_group(rows: list[dict[str, Any]], budget: int | None) -> list[dict[str, Any]]:
    if budget is None:
        return list(rows)
    selected: list[dict[str, Any]] = []
    selected_paths: set[str] = set()
    query_count = 0

    def add(row: dict[str, Any]) -> None:
        nonlocal query_count
        path = str(row["csv_path"])
        if path in selected_paths:
            return
        selected.append(row)
        selected_paths.add(path)
        query_count += int(row["query_count"])

    positive_rows = sorted(
        [row for row in rows if int(row.get("positive_count", 0)) > 0],
        key=lambda row: (-int(row.get("positive_count", 0)), str(row["csv_path"])),
    )
    non_positive_rows = sorted(
        [row for row in rows if int(row.get("positive_count", 0)) <= 0],
        key=lambda row: str(row["csv_path"]),
    )
    for row in positive_rows:
        if query_count >= budget and selected:
            break
        add(row)
    for row in non_positive_rows:
        if query_count >= budget and selected:
            break
        add(row)
    return sorted(selected, key=lambda row: str(row["csv_path"]))


def _summary(files: list[dict[str, Any]]) -> dict[str, Any]:
    queries_by_split: dict[str, int] = defaultdict(int)
    queries_by_case_kind: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "queries": 0})
    total_bytes = 0
    total_queries = 0
    positive_count = 0
    negative_count = 0
    for row in files:
        split = str(row["split"])
        case_kind = f"{row['case']}/{row['kind']}"
        query_count = int(row["query_count"])
        row_positive = int(row.get("positive_count", 0))
        row_negative = int(row.get("negative_count", query_count - row_positive))
        queries_by_split[split] += query_count
        queries_by_case_kind[case_kind]["files"] += 1
        queries_by_case_kind[case_kind]["queries"] += query_count
        total_bytes += int(row.get("bytes", 0))
        total_queries += query_count
        positive_count += row_positive
        negative_count += row_negative
    return {
        "file_count": len(files),
        "labels_counted": True,
        "negative_count": negative_count,
        "positive_count": positive_count,
        "positive_ratio": positive_count / max(1, total_queries),
        "queries_by_case_kind": dict(sorted(queries_by_case_kind.items())),
        "queries_by_split": dict(sorted(queries_by_split.items())),
        "total_bytes": total_bytes,
        "total_queries": total_queries,
    }


def build_tight_inclusion_sample_manifest(
    source_manifest: Path,
    *,
    output: Path,
    report: Path | None = None,
    preset: str = "medium",
) -> dict[str, Any]:
    source = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    if preset not in PRESET_BUDGETS:
        raise ValueError(f"unsupported preset {preset!r}; expected one of {sorted(PRESET_BUDGETS)}")
    budgets = PRESET_BUDGETS[preset]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source["files"]:
        grouped[_group_key(row)].append(row)
    selected: list[dict[str, Any]] = []
    for (split, _case, _kind), rows in sorted(grouped.items()):
        selected.extend(_select_rows_for_group(rows, budgets.get(split)))
    selected = sorted(selected, key=lambda row: (str(row["split"]), str(row["case"]), str(row["kind"]), str(row["csv_path"])))
    sampled = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": source["dataset_root"],
        "files": selected,
        "sample_policy": {
            "source_manifest": Path(source_manifest).as_posix(),
            "preset": preset,
            "per_case_kind_query_budget": budgets,
            "selection": "file-level, split/case/kind stratified, positive files first, no query-level leakage",
        },
        "schema_version": source.get("schema_version", 1),
        "seed": source.get("seed"),
        "sha256_enabled": source.get("sha256_enabled", False),
        "split_policy": source.get("split_policy"),
        "summary": _summary(selected),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sampled, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        _write_report(report, sampled)
    return sampled


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    lines = [
        "# Tight-Inclusion / NYU Sampled Manifest Audit",
        "",
        f"- Source: `{manifest['sample_policy']['source_manifest']}`",
        f"- Preset: `{manifest['sample_policy']['preset']}`",
        f"- Files: `{summary['file_count']}`",
        f"- Total queries: `{summary['total_queries']}`",
        f"- Positive: `{summary['positive_count']}`",
        f"- Negative: `{summary['negative_count']}`",
        f"- Positive ratio: `{summary['positive_ratio']}`",
        "",
        "## Queries By Split",
        "",
        "| Split | Queries |",
        "| --- | ---: |",
    ]
    for split, count in summary["queries_by_split"].items():
        lines.append(f"| `{split}` | `{count}` |")
    lines.extend(["", "## Queries By Case/Kind", "", "| Case/Kind | Files | Queries |", "| --- | ---: | ---: |"])
    for case_kind, row in summary["queries_by_case_kind"].items():
        lines.append(f"| `{case_kind}` | `{row['files']}` | `{row['queries']}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--preset", choices=sorted(PRESET_BUDGETS), default="medium")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_tight_inclusion_sample_manifest(args.source, output=args.output, report=args.report, preset=args.preset)


if __name__ == "__main__":
    main()
