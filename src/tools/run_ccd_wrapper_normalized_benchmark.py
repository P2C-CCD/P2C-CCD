#!/usr/bin/env python3
"""Run a normalized CCD-Wrapper correctness/timing table.

The upstream CCD-Wrapper benchmark prints human-readable text only.  This
runner keeps the upstream executable unchanged, runs each enabled method on the
same query split, and emits CSV/JSON/Markdown summaries under benchmark/.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CCD_WRAPPER = ROOT / "baseline" / "CCD-Wrapper"
DEFAULT_QUERIES = ROOT / "baseline" / "Sample-Queries"
DEFAULT_OUT = ROOT / "benchmark" / "ccd_wrapper_normalized_sample_queries_run_id"


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METHOD_RE = re.compile(r"Benchmarking\s+([A-Za-z0-9_+\-]+)")
TOTAL_RE = re.compile(r"total # of queries:\s*(\d+)")
POS_RE = re.compile(r"total positives:\s*(\d+)")
FP_RE = re.compile(r"# of false positives:\s*(\d+)")
FN_RE = re.compile(r"# of false negatives:\s*(\d+)")
TIME_RE = re.compile(r"average time:\s*([-+]?nan|[0-9.eE+\-]+)", re.IGNORECASE)


METHOD_RUNS = [
    {
        "method": "FloatingPointRootParity",
        "exe": CCD_WRAPPER / "build-p-fprp-ci" / "ccd_benchmark.exe",
    },
    {
        "method": "BSC",
        "exe": CCD_WRAPPER / "build-p-normalized-public-fixed_seed" / "ccd_benchmark.exe",
    },
    {
        "method": "UnivariateIntervalRootFinder",
        "exe": CCD_WRAPPER / "build-p-normalized-public-fixed_seed" / "ccd_benchmark.exe",
    },
    {
        "method": "MultivariateIntervalRootFinder",
        "exe": CCD_WRAPPER / "build-p-normalized-public-fixed_seed" / "ccd_benchmark.exe",
    },
    {
        "method": "TightInclusion",
        "exe": CCD_WRAPPER / "build-p-normalized-ti-noavx-fixed_seed" / "ccd_benchmark.exe",
    },
]


def prepend_windows_toolchain_paths(env: dict[str, str]) -> dict[str, str]:
    prefixes: list[str] = []
    for key in ("P2CCCD_TOOLCHAIN_BIN", "P2CCCD_MINGW_BIN"):
        value = os.environ.get(key)
        if value:
            prefixes.append(value)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_bin = Path(conda_prefix) / "Library" / "bin"
        if conda_bin.exists():
            prefixes.append(str(conda_bin))
    if prefixes:
        env["PATH"] = ";".join(prefixes + [env.get("PATH", "")])
    return env


def clean_text(text: str) -> str:
    text = ANSI_RE.sub("", text)
    # Upstream prints progress using carriage returns.  Treat them as lines so
    # the metrics blocks remain parseable in captured logs.
    return text.replace("\r", "\n")


def parse_output(text: str, method_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    clean = clean_text(text)
    dataset = None
    primitive = None
    current_method = method_name
    block: dict[str, object] = {}

    def flush() -> None:
        nonlocal block
        if "total_queries" not in block:
            block = {}
            return
        total = int(block["total_queries"])
        positives = int(block.get("positives", 0))
        fp = int(block.get("false_positives", 0))
        fn = int(block.get("false_negatives", 0))
        tn = max(0, total - positives - fp)
        tp = max(0, positives - fn)
        row = {
            "method": current_method,
            "dataset_split": dataset or "unknown",
            "primitive": primitive or "unknown",
            "total_queries": total,
            "positives": positives,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "recall": (tp / positives) if positives else 1.0,
            "precision": (tp / (tp + fp)) if (tp + fp) else 1.0,
            "fp_rate": (fp / max(1, total - positives)),
            "avg_us_per_query": float(block.get("avg_us_per_query", 0.0)),
        }
        rows.append(row)
        block = {}

    for raw_line in clean.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = METHOD_RE.search(line)
        if m:
            current_method = m.group(1)
            continue
        if line.startswith("Running handcrafted dataset"):
            flush()
            dataset = "handcrafted"
            primitive = None
            continue
        if line.startswith("Running simulation dataset"):
            flush()
            dataset = "simulation"
            primitive = None
            continue
        if line.startswith("Vertex-Face"):
            flush()
            primitive = "vertex-face"
            continue
        if line.startswith("Edge-Edge"):
            flush()
            primitive = "edge-edge"
            continue
        m = TOTAL_RE.search(line)
        if m:
            block["total_queries"] = int(m.group(1))
            continue
        m = POS_RE.search(line)
        if m:
            block["positives"] = int(m.group(1))
            continue
        m = FP_RE.search(line)
        if m:
            block["false_positives"] = int(m.group(1))
            continue
        m = FN_RE.search(line)
        if m:
            block["false_negatives"] = int(m.group(1))
            continue
        m = TIME_RE.search(line)
        if m:
            value = float(m.group(1))
            block["avg_us_per_query"] = 0.0 if math.isnan(value) else value
            flush()
            continue
    flush()
    return rows


def command_for(method: str, exe: Path, queries: Path, split: str) -> list[str]:
    cmd = [str(exe), "--data", str(queries), "-m", method]
    if split == "handcrafted":
        cmd.append("--no-simulation")
    elif split == "simulation":
        cmd.append("--no-handcrafted")
    elif split != "all":
        raise ValueError(f"unsupported split: {split}")
    return cmd


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "method",
        "dataset_split",
        "primitive",
        "total_queries",
        "positives",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "recall",
        "precision",
        "fp_rate",
        "avg_us_per_query",
        "exit_code",
        "timed_out",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_method: dict[str, dict[str, object]] = {}
    for row in rows:
        key = str(row["method"])
        out = by_method.setdefault(
            key,
            {
                "method": key,
                "total_queries": 0,
                "positives": 0,
                "true_positives": 0,
                "true_negatives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "weighted_time_us": 0.0,
            },
        )
        n = int(row["total_queries"])
        out["total_queries"] = int(out["total_queries"]) + n
        out["positives"] = int(out["positives"]) + int(row["positives"])
        out["true_positives"] = int(out["true_positives"]) + int(row["true_positives"])
        out["true_negatives"] = int(out["true_negatives"]) + int(row["true_negatives"])
        out["false_positives"] = int(out["false_positives"]) + int(row["false_positives"])
        out["false_negatives"] = int(out["false_negatives"]) + int(row["false_negatives"])
        out["weighted_time_us"] = float(out["weighted_time_us"]) + float(row["avg_us_per_query"]) * n
    result = []
    for out in by_method.values():
        total = int(out["total_queries"])
        pos = int(out["positives"])
        tp = int(out["true_positives"])
        fp = int(out["false_positives"])
        negatives = max(1, total - pos)
        out["recall"] = (tp / pos) if pos else 1.0
        out["precision"] = (tp / (tp + fp)) if (tp + fp) else 1.0
        out["fp_rate"] = fp / negatives
        out["avg_us_per_query"] = float(out["weighted_time_us"]) / max(1, total)
        del out["weighted_time_us"]
        result.append(out)
    return sorted(result, key=lambda r: str(r["method"]))


def write_markdown(path: Path, rows: list[dict[str, object]], summary: list[dict[str, object]], meta: dict[str, object]) -> None:
    lines = [
        "# CCD-Wrapper normalized benchmark",
        "",
        f"Generated: `{meta['generated_at']}`",
        "",
        "Scope: upstream CCD-Wrapper methods evaluated on the same query split.",
        "This is a wrapper breadth/correctness table, not a P2C-CCD speedup claim.",
        "",
        "## Configuration",
        "",
        f"- Query root: `{meta['queries']}`",
        f"- Split: `{meta['split']}`",
        f"- Public-method executable: `{meta['public_exe']}`",
        f"- TightInclusion executable: `{meta['tight_inclusion_exe']}`",
        f"- FPRP executable: `{meta['fprp_exe']}`",
        "",
        "## Method Summary",
        "",
        "| Method | Queries | Positives | FP | FN | Recall | Precision | Avg us/query |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {method} | {total_queries} | {positives} | {false_positives} | "
            "{false_negatives} | {recall:.6f} | {precision:.6f} | {avg_us_per_query:.3f} |".format(**row)
        )
    lines += [
        "",
        "## Primitive-Level Rows",
        "",
        "| Method | Split | Primitive | Queries | Positives | FP | FN | Recall | Avg us/query |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {dataset_split} | {primitive} | {total_queries} | {positives} | "
            "{false_positives} | {false_negatives} | {recall:.6f} | {avg_us_per_query:.3f} |".format(**row)
        )
    lines += [
        "",
        "## Run Status",
        "",
        "| Method | Exit code | Timed out | Parsed rows | Log |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for run in meta.get("runs", []):
        lines.append(
            f"| {run['method']} | {run['exit_code']} | {run['timed_out']} | "
            f"{run['rows']} | `{Path(str(run['log'])).name}` |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `FloatingPointRootParity` is run from the existing FPRP-only build because its upstream source is private/SSH-fetched in this checkout.",
        "- `BSC`, `UnivariateIntervalRootFinder`, and `MultivariateIntervalRootFinder` are run from the local public-method build.",
        "- `TightInclusion` is run from a TI-only no-AVX build because the upstream AVX/native-flags build crashes under the current Windows MinGW/GCC15 toolchain.",
        "- All rows are parsed from unmodified upstream `ccd_benchmark` stdout; raw logs are stored in `raw_logs/`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", choices=["handcrafted", "simulation", "all"], default="handcrafted")
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=[item["method"] for item in METHOD_RUNS],
        help="subset of methods to run",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out / "raw_logs"
    raw_dir.mkdir(exist_ok=True)

    env = prepend_windows_toolchain_paths(os.environ.copy())

    all_rows: list[dict[str, object]] = []
    run_meta = []
    requested = set(args.methods)
    for item in METHOD_RUNS:
        method = item["method"]
        if method not in requested:
            continue
        exe = Path(item["exe"])
        if not exe.exists():
            raise FileNotFoundError(f"missing executable for {method}: {exe}")
        cmd = command_for(method, exe, args.queries, args.split)
        started = dt.datetime.now().isoformat(timespec="seconds")
        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(CCD_WRAPPER),
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_sec,
                check=False,
            )
            return_code = proc.returncode
            stdout = proc.stdout or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            return_code = -999
            stdout = (exc.stdout or "")
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            stdout += f"\nTIMEOUT after {args.timeout_sec} seconds\n"
        log = clean_text(stdout)
        log_path = raw_dir / f"{method}.log"
        log_path.write_text(log, encoding="utf-8", errors="replace")
        rows = parse_output(log, method)
        for row in rows:
            row["exit_code"] = return_code
            row["timed_out"] = timed_out
        all_rows.extend(rows)
        run_meta.append(
            {
                "method": method,
                "exe": str(exe),
                "exit_code": return_code,
                "timed_out": timed_out,
                "started_at": started,
                "rows": len(rows),
                "log": str(log_path),
                "command": cmd,
            }
        )
        if return_code != 0:
            print(f"warning: {method} exited with {return_code}", file=sys.stderr)

    summary = aggregate(all_rows)
    meta = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "queries": str(args.queries),
        "split": args.split,
        "public_exe": str(CCD_WRAPPER / "build-p-normalized-public-fixed_seed" / "ccd_benchmark.exe"),
        "tight_inclusion_exe": str(CCD_WRAPPER / "build-p-normalized-ti-noavx-fixed_seed" / "ccd_benchmark.exe"),
        "fprp_exe": str(CCD_WRAPPER / "build-p-fprp-ci" / "ccd_benchmark.exe"),
        "runs": run_meta,
    }

    write_csv(args.out / "ccd_wrapper_normalized_rows.csv", all_rows)
    write_csv(args.out / "ccd_wrapper_normalized_summary.csv", summary)
    (args.out / "ccd_wrapper_normalized_metrics.json").write_text(
        json.dumps({"meta": meta, "rows": all_rows, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.out / "ccd_wrapper_normalized_benchmark.md", all_rows, summary, meta)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
