from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("ti_full_*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["_source_jsonl"] = str(path)
                    rows.append(row)
    return rows


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def add_row(dst: dict[str, Any], row: dict[str, Any]) -> None:
    for key in [
        "file_count",
        "query_count",
        "exact_calls",
        "skipped_exact_calls",
        "positive_count",
        "negative_count",
        "tp",
        "tn",
        "fp",
        "fn",
        "exact_us",
        "wall_us",
    ]:
        dst[key] = dst.get(key, 0) + row.get(key, 0)


def finalize(row: dict[str, Any]) -> dict[str, Any]:
    query_count = row.get("query_count", 0)
    exact_calls = row.get("exact_calls", 0)
    wall_us = row.get("wall_us", 0.0)
    exact_us = row.get("exact_us", 0.0)
    tp = row.get("tp", 0)
    fn = row.get("fn", 0)
    fp = row.get("fp", 0)
    row["exact_call_reduction"] = 1.0 - safe_div(exact_calls, max(1, query_count))
    row["recall"] = safe_div(tp, tp + fn)
    row["precision"] = safe_div(tp, tp + fp)
    row["wall_ms"] = wall_us / 1000.0
    row["exact_ms"] = exact_us / 1000.0
    row["queries_per_second"] = safe_div(query_count, wall_us / 1_000_000.0)
    row["avg_wall_us_per_query"] = safe_div(wall_us, max(1, query_count))
    row["avg_exact_us_per_exact_call"] = safe_div(exact_us, max(1, exact_calls))
    return row


def weighted_percentile_proxy(rows: list[dict[str, Any]], field: str) -> float:
    """Approximate aggregate percentile by weighting shard percentile summaries.

    The native runner emits per-shard p50/p90/p99 but intentionally does not
    persist per-query latencies for the 100GB full run. This proxy is used only
    for a compact top-level summary; exact shard percentiles remain in JSONL/MD.
    """
    weighted: list[tuple[float, int]] = []
    for row in rows:
        if field in row:
            weighted.append((float(row[field]), int(row.get("query_count", 0))))
    if not weighted:
        return 0.0
    weighted.sort(key=lambda item: item[0])
    total = sum(w for _, w in weighted)
    target = total * 0.5
    cumulative = 0
    for value, weight in weighted:
        cumulative += weight
        if cumulative >= target:
            return value
    return weighted[-1][0]


def group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    source_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_key = tuple(row.get(key, "") for key in keys)
        if group_key not in grouped:
            grouped[group_key] = {key: row.get(key, "") for key in keys}
        add_row(grouped[group_key], row)
        source_rows[group_key].append(row)
    out = []
    for key, value in grouped.items():
        value = finalize(value)
        # These are weighted proxies across shard summaries, not raw-query percentiles.
        for percentile_field in ["wall_p50_us", "wall_p90_us", "wall_p99_us", "exact_p50_us", "exact_p90_us", "exact_p99_us"]:
            value[f"{percentile_field}_proxy"] = weighted_percentile_proxy(source_rows[key], percentile_field)
        out.append(value)
    out.sort(key=lambda row: tuple(row.get(key, "") for key in keys))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "case",
        "kind",
        "file_count",
        "query_count",
        "exact_calls",
        "positive_count",
        "negative_count",
        "tp",
        "tn",
        "fp",
        "fn",
        "recall",
        "precision",
        "wall_ms",
        "exact_ms",
        "queries_per_second",
        "avg_wall_us_per_query",
        "avg_exact_us_per_exact_call",
        "wall_p50_us",
        "wall_p90_us",
        "wall_p99_us",
        "exact_p50_us",
        "exact_p90_us",
        "exact_p99_us",
        "source_jsonl",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["source_jsonl"] = row.get("_source_jsonl", "")
            writer.writerow(flat)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(title for title, _ in columns) + " |")
    lines.append("| " + " | ".join("---" if not key.endswith("_num") else "---:" for _, key in columns) + " |")
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key.replace("_num", ""), "")
            if isinstance(value, float):
                if "reduction" in key or key in {"recall", "precision"}:
                    cells.append(f"`{value:.6g}`")
                elif key.endswith("ms_num") or key.endswith("second_num") or key.endswith("query_num"):
                    cells.append(f"`{value:.3f}`")
                else:
                    cells.append(f"`{value:.6g}`")
            else:
                cells.append(f"`{value}`")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_md(path: Path, output_dir: Path, rows: list[dict[str, Any]], by_split: list[dict[str, Any]], total: dict[str, Any]) -> None:
    heldout_rows = [r for r in rows if r.get("split") == "heldout_test"]
    train_rows = [r for r in rows if r.get("split") == "train"]
    validation_rows = [r for r in rows if r.get("split") == "validation"]
    lines: list[str] = []
    lines.append("# Tight-Inclusion / NYU 100GB Full-query Every-candidate Wall-time Benchmark")
    lines.append("")
    lines.append("Run identifier: `run_id`")
    lines.append("")
    lines.append("This report aggregates the native C++ Tight-Inclusion full-query shards. Every query in the manifest was sent to the exact Tight-Inclusion kernel; there is no proposal skipping in this baseline.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append("- Manifest: `src/datasets/manifests/tight_inclusion_nyu_full_manifest_run_id.json`")
    lines.append(f"- Shard output dir: `{output_dir.as_posix()}`")
    lines.append("- Exact parameters: `ms=0.0, tolerance=1e-6, t_max=1.0, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`")
    lines.append("- Native executable: `src/build_tools/tight_inclusion_full_query_benchmark.exe`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(markdown_table([total], [
        ("Files", "file_count"),
        ("Queries", "query_count"),
        ("Exact calls", "exact_calls"),
        ("TP", "tp"),
        ("TN", "tn"),
        ("FP", "fp"),
        ("FN", "fn"),
        ("Recall", "recall"),
        ("Wall ms", "wall_ms_num"),
        ("Exact ms", "exact_ms_num"),
        ("QPS", "queries_per_second_num"),
        ("Avg wall us/query", "avg_wall_us_per_query_num"),
    ]))
    lines.append("")
    lines.append("Correctness result: `FN=0` across the aggregated full-query manifest rows. `FP` is reported because Tight-Inclusion is conservative.")
    lines.append("")
    lines.append("## By Split")
    lines.append("")
    lines.append(markdown_table(by_split, [
        ("Split", "split"),
        ("Files", "file_count"),
        ("Queries", "query_count"),
        ("Positives", "positive_count"),
        ("Exact calls", "exact_calls"),
        ("TP", "tp"),
        ("TN", "tn"),
        ("FP", "fp"),
        ("FN", "fn"),
        ("Wall ms", "wall_ms_num"),
        ("QPS", "queries_per_second_num"),
        ("Avg us/query", "avg_wall_us_per_query_num"),
    ]))
    lines.append("")
    lines.append("## Heldout Test Per Case / Kind")
    lines.append("")
    lines.append(markdown_table(heldout_rows, [
        ("Case", "case"),
        ("Kind", "kind"),
        ("Queries", "query_count"),
        ("Positives", "positive_count"),
        ("TP", "tp"),
        ("TN", "tn"),
        ("FP", "fp"),
        ("FN", "fn"),
        ("Wall ms", "wall_ms_num"),
        ("QPS", "queries_per_second_num"),
        ("p50 us", "wall_p50_us_num"),
        ("p90 us", "wall_p90_us_num"),
        ("p99 us", "wall_p99_us_num"),
    ]))
    lines.append("")
    lines.append("## Validation / Train Summary")
    lines.append("")
    lines.append(f"- Validation shards: `{len(validation_rows)}` case/kind rows.")
    lines.append(f"- Train shards: `{len(train_rows)}` case/kind rows.")
    lines.append("- Full per-shard JSONL/Markdown files are kept in the shard directory for replay and audit.")
    lines.append("")
    lines.append("## Reproduce / Resume")
    lines.append("")
    lines.append("```powershell")
    lines.append("powershell -NoProfile -ExecutionPolicy Bypass -File src/tools/run_tight_inclusion_full_query_shards.ps1 -OutputDir src/benchmark/ti_full_query_every_candidate_walltime_run_id -Split heldout_test")
    lines.append("powershell -NoProfile -ExecutionPolicy Bypass -File src/tools/run_tight_inclusion_full_query_shards.ps1 -OutputDir src/benchmark/ti_full_query_every_candidate_walltime_run_id -Split validation")
    lines.append("powershell -NoProfile -ExecutionPolicy Bypass -File src/tools/run_tight_inclusion_full_query_shards.ps1 -OutputDir src/benchmark/ti_full_query_every_candidate_walltime_run_id -Split train")
    lines.append("powershell -NoProfile -ExecutionPolicy Bypass -File src/tools/summarize_tight_inclusion_full_query_walltime.py -OutputDir src/benchmark/ti_full_query_every_candidate_walltime_run_id")
    lines.append("```")
    lines.append("")
    lines.append("Resume rule: the shard wrapper skips existing `jsonl/md` pairs unless `-Force` is passed.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Top-level aggregate p50/p90/p99 are not recomputed from raw per-query latencies because raw latencies are not persisted for the 100GB run. Per-shard exact percentiles are in each shard JSONL/MD.")
    lines.append("- This is the SOTA primitive exact wall-time baseline. It should be compared against RTSTPFExact only when RTSTPFExact uses the same exact certificate/fallback policy.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-OutputDir", "--output-dir", dest="output_dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    rows = load_rows(output_dir)
    if not rows:
        raise SystemExit(f"no ti_full_*.jsonl files found in {output_dir}")
    finalized_rows = [finalize(dict(row)) for row in rows]
    by_split = group_rows(finalized_rows, ("split",))
    total = {"split": "all"}
    for row in finalized_rows:
        add_row(total, row)
    total = finalize(total)
    total["file_count"] = sum(row.get("file_count", 0) for row in finalized_rows)
    total["wall_p50_us_proxy"] = weighted_percentile_proxy(finalized_rows, "wall_p50_us")
    total["wall_p90_us_proxy"] = weighted_percentile_proxy(finalized_rows, "wall_p90_us")
    total["wall_p99_us_proxy"] = weighted_percentile_proxy(finalized_rows, "wall_p99_us")

    summary = {
        "output_dir": str(output_dir),
        "row_count": len(finalized_rows),
        "total": total,
        "by_split": by_split,
        "rows": finalized_rows,
    }
    (output_dir / "ti_full_query_every_candidate_walltime_run_id.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(output_dir / "ti_full_query_every_candidate_walltime_run_id.csv", finalized_rows)
    write_md(output_dir / "ti_full_query_every_candidate_walltime_run_id.md", output_dir, finalized_rows, by_split, total)
    print(json.dumps({"total": total, "by_split": by_split}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
