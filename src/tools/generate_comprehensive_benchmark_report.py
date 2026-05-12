from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
P2CCCD_ROOT = ROOT / "src"
BENCHMARK_ROOT = P2CCCD_ROOT / "benchmark"
MYDEMO_ROOT = P2CCCD_ROOT / "MyDemo"
REPORT_NAME = "comprehensive_case_benchmark_report_run_id"
OUTPUT_DIR = BENCHMARK_ROOT / REPORT_NAME
MAX_PARSE_CSV_BYTES = 25 * 1024 * 1024


@dataclass
class BenchmarkFile:
    benchmark_group: str
    path: str
    kind: str
    case: str
    dataset: str
    row_count: int | None
    candidate_rows: int | None
    positive_rows: int | None
    method_count: int
    best_call_reduction: float | None
    best_work_reduction: float | None
    min_fn: int | None
    summary: str


@dataclass
class MethodRow:
    source_file: str
    case: str
    dataset: str
    split: str
    method: str
    candidates: float | None
    positives: float | None
    exact_calls: float | None
    call_reduction: float | None
    work_reduction: float | None
    fn: int | None
    note: str


@dataclass
class VisualArtifact:
    case_dir: str
    mp4_count: int
    png_count: int
    json_count: int
    total_bytes: int
    verified_mp4_count: int
    min_frames: int | None
    max_frames: int | None
    min_fps: float | None
    max_fps: float | None
    representative: str


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None
    except Exception:
        return None


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def get_first(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def normalize_method_row(source: Path, row: dict[str, Any], default_case: str = "", default_dataset: str = "") -> MethodRow:
    case = str(get_first(row, ("case", "scenario", "run_name")) or default_case or source.parent.name)
    dataset = str(get_first(row, ("dataset", "dataset_model", "family", "source")) or default_dataset or "")
    method = str(get_first(row, ("method", "name", "algorithm")) or "")
    split = str(get_first(row, ("split", "role")) or "")
    return MethodRow(
        source_file=rel(source),
        case=case,
        dataset=dataset,
        split=split,
        method=method,
        candidates=as_float(get_first(row, ("candidates", "candidate_rows", "candidate_count", "candidate_density", "row_count"))),
        positives=as_float(get_first(row, ("positives", "positive_rows", "truth_positives", "positive_count"))),
        exact_calls=as_float(get_first(row, ("exact_calls", "rtstpf_exact_calls", "exact_call_count"))),
        call_reduction=as_float(get_first(row, ("call_reduction", "exact_call_reduction", "reduction", "exact_reduction"))),
        work_reduction=as_float(get_first(row, ("work_reduction", "exact_work_reduction"))),
        fn=as_int(get_first(row, ("fn", "false_negative", "false_negatives"))),
        note=str(get_first(row, ("description", "note", "timing_scope")) or ""),
    )


def extract_method_rows(path: Path, data: Any) -> list[MethodRow]:
    if not isinstance(data, dict):
        return []
    rows: list[MethodRow] = []
    default_case = str(get_first(data, ("case", "case_name", "run_name", "source_case", "title")) or path.parent.name)
    default_dataset = str(get_first(data, ("dataset", "source", "tag")) or "")

    for key in ("rows", "benchmark_rows"):
        value = data.get(key)
        if isinstance(value, list):
            rows.extend(normalize_method_row(path, row, default_case, default_dataset) for row in value if isinstance(row, dict))

    methods = data.get("methods")
    if isinstance(methods, dict):
        for method, payload in methods.items():
            if isinstance(payload, dict):
                row = dict(payload)
                row.setdefault("method", method)
                rows.append(normalize_method_row(path, row, default_case, default_dataset))
    benchmark = data.get("benchmark")
    if isinstance(benchmark, dict):
        for method, payload in benchmark.items():
            if isinstance(payload, dict):
                row = dict(payload)
                row.setdefault("method", method)
                rows.append(normalize_method_row(path, row, default_case, default_dataset))
    metrics = data.get("benchmark_metrics")
    if isinstance(metrics, dict):
        row = dict(metrics)
        row.setdefault("method", "RTSTPF/P2C")
        rows.append(normalize_method_row(path, row, default_case, default_dataset))
    return rows


def summarize_benchmark_file(path: Path, data: Any, rows: list[MethodRow]) -> BenchmarkFile:
    if path.is_relative_to(BENCHMARK_ROOT):
        parts = path.relative_to(BENCHMARK_ROOT).parts
        group = parts[0] if len(parts) > 1 else "_benchmark_root"
    else:
        group = path.parent.name
    case = path.stem
    dataset = ""
    row_count = candidate_rows = positive_rows = None
    summary = ""
    if isinstance(data, dict):
        case = str(get_first(data, ("case", "case_name", "run_name", "source_case", "title")) or case)
        dataset = str(get_first(data, ("dataset", "source", "tag")) or "")
        row_count = as_int(get_first(data, ("row_count", "rows", "candidate_count")))
        candidate_rows = as_int(get_first(data, ("candidate_rows", "candidate_count", "query_count")))
        positive_rows = as_int(get_first(data, ("positive_rows", "positive_count", "truth_positives")))
        desc = get_first(data, ("description", "scope", "scenario"))
        summary = str(desc)[:260] if desc is not None else ""
    if rows:
        reductions = [row.call_reduction for row in rows if row.call_reduction is not None]
        work_reductions = [row.work_reduction for row in rows if row.work_reduction is not None]
        fns = [row.fn for row in rows if row.fn is not None]
    else:
        reductions = []
        work_reductions = []
        fns = []
    return BenchmarkFile(
        benchmark_group=group,
        path=rel(path),
        kind="json",
        case=case,
        dataset=dataset,
        row_count=row_count,
        candidate_rows=candidate_rows,
        positive_rows=positive_rows,
        method_count=len(rows),
        best_call_reduction=max(reductions) if reductions else None,
        best_work_reduction=max(work_reductions) if work_reductions else None,
        min_fn=min(fns) if fns else None,
        summary=summary,
    )


def read_csv_method_rows(path: Path) -> list[MethodRow]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [normalize_method_row(path, row) for row in reader]
    except Exception:
        return []


def summarize_csv_file(path: Path, rows: list[MethodRow]) -> BenchmarkFile:
    reductions = [row.call_reduction for row in rows if row.call_reduction is not None]
    work_reductions = [row.work_reduction for row in rows if row.work_reduction is not None]
    fns = [row.fn for row in rows if row.fn is not None]
    size_mb = path.stat().st_size / (1024.0 * 1024.0)
    if rows:
        summary = ""
    elif path.stat().st_size > MAX_PARSE_CSV_BYTES:
        summary = f"Large CSV ({size_mb:.2f} MiB); recorded as raw benchmark artifact without expanding candidate-level rows."
    else:
        summary = "CSV did not expose normalized method-level fields."
    return BenchmarkFile(
        benchmark_group=path.relative_to(BENCHMARK_ROOT).parts[0]
        if len(path.relative_to(BENCHMARK_ROOT).parts) > 1
        else "_benchmark_root",
        path=rel(path),
        kind="csv",
        case=path.stem,
        dataset="",
        row_count=None,
        candidate_rows=None,
        positive_rows=None,
        method_count=len(rows),
        best_call_reduction=max(reductions) if reductions else None,
        best_work_reduction=max(work_reductions) if work_reductions else None,
        min_fn=min(fns) if fns else None,
        summary=summary,
    )


def validate_mp4(path: Path) -> tuple[bool, int | None, float | None]:
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        ok = bool(cap.isOpened())
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else None
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if ok else None
        cap.release()
        return ok, frames, fps
    except Exception:
        return False, None, None


def scan_visual_artifacts() -> list[VisualArtifact]:
    artifacts: list[VisualArtifact] = []
    if not MYDEMO_ROOT.exists():
        return artifacts
    skip_dirs = {
        "_frames",
        "frames",
        "physical_scene_frames",
        "__pycache__",
    }

    def collect(case_dir: Path, suffixes: tuple[str, ...]) -> list[Path]:
        found: list[Path] = []
        stack = [case_dir]
        while stack:
            current = stack.pop()
            for child in current.iterdir():
                if child.is_dir():
                    if child.name in skip_dirs or child.name.endswith("_frames"):
                        continue
                    stack.append(child)
                elif child.suffix.lower() in suffixes:
                    found.append(child)
        return found

    for case_dir in sorted((path for path in MYDEMO_ROOT.iterdir() if path.is_dir()), key=lambda p: p.name):
        if case_dir.name.startswith("_"):
            continue
        mp4s = collect(case_dir, (".mp4",))
        pngs = collect(case_dir, (".png",))
        jsons = collect(case_dir, (".json",))
        total_bytes = sum(path.stat().st_size for path in [*mp4s, *pngs, *jsons] if path.exists())
        verified = 0
        frames: list[int] = []
        fps_values: list[float] = []
        for mp4 in mp4s:
            ok, frame_count, fps = validate_mp4(mp4)
            if ok:
                verified += 1
            if frame_count is not None:
                frames.append(frame_count)
            if fps is not None and math.isfinite(fps):
                fps_values.append(fps)
        representative = ""
        if mp4s:
            representative = rel(sorted(mp4s, key=lambda p: p.stat().st_size, reverse=True)[0])
        elif pngs:
            representative = rel(sorted(pngs, key=lambda p: p.stat().st_size, reverse=True)[0])
        artifacts.append(
            VisualArtifact(
                case_dir=rel(case_dir),
                mp4_count=len(mp4s),
                png_count=len(pngs),
                json_count=len(jsons),
                total_bytes=total_bytes,
                verified_mp4_count=verified,
                min_frames=min(frames) if frames else None,
                max_frames=max(frames) if frames else None,
                min_fps=min(fps_values) if fps_values else None,
                max_fps=max(fps_values) if fps_values else None,
                representative=representative,
            )
        )
    return artifacts


def scan_benchmarks() -> tuple[list[BenchmarkFile], list[MethodRow]]:
    benchmark_files: list[BenchmarkFile] = []
    method_rows: list[MethodRow] = []
    for path in sorted(BENCHMARK_ROOT.rglob("*")):
        if OUTPUT_DIR in path.parents:
            continue
        if path.suffix.lower() == ".json":
            data = load_json(path)
            rows = extract_method_rows(path, data)
            method_rows.extend(rows)
            benchmark_files.append(summarize_benchmark_file(path, data, rows))
        elif path.suffix.lower() == ".csv":
            rows = [] if path.stat().st_size > MAX_PARSE_CSV_BYTES else read_csv_method_rows(path)
            method_rows.extend(rows)
            benchmark_files.append(summarize_csv_file(path, rows))
    return benchmark_files, method_rows


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.3f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(benchmark_files: list[BenchmarkFile], method_rows: list[MethodRow], visual_artifacts: list[VisualArtifact]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_dicts = [asdict(row) for row in benchmark_files]
    method_dicts = [asdict(row) for row in method_rows]
    visual_dicts = [asdict(row) for row in visual_artifacts]
    write_csv(OUTPUT_DIR / f"{REPORT_NAME}_benchmark_files.csv", benchmark_dicts)
    write_csv(OUTPUT_DIR / f"{REPORT_NAME}_method_rows.csv", method_dicts)
    write_csv(OUTPUT_DIR / f"{REPORT_NAME}_visual_artifacts.csv", visual_dicts)
    summary = {
        "run_name": REPORT_NAME,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_file_count": len(benchmark_files),
        "method_row_count": len(method_rows),
        "visual_case_count": len(visual_artifacts),
        "mp4_count": sum(row.mp4_count for row in visual_artifacts),
        "verified_mp4_count": sum(row.verified_mp4_count for row in visual_artifacts),
        "outputs": {
            "markdown": rel(OUTPUT_DIR / f"{REPORT_NAME}.md"),
            "benchmark_files_csv": rel(OUTPUT_DIR / f"{REPORT_NAME}_benchmark_files.csv"),
            "method_rows_csv": rel(OUTPUT_DIR / f"{REPORT_NAME}_method_rows.csv"),
            "visual_artifacts_csv": rel(OUTPUT_DIR / f"{REPORT_NAME}_visual_artifacts.csv"),
        },
        "benchmark_files": benchmark_dicts,
        "visual_artifacts": visual_dicts,
    }
    (OUTPUT_DIR / f"{REPORT_NAME}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    covered_groups = sorted({row.benchmark_group for row in benchmark_files})
    visual_case_names = {Path(row.case_dir).name for row in visual_artifacts}
    benchmark_group_names = {row.benchmark_group for row in benchmark_files}
    missing_benchmark = sorted(name for name in visual_case_names if name not in benchmark_group_names and not name.startswith("_"))

    strong_rows = [
        row
        for row in method_rows
        if row.call_reduction is not None and row.fn in (None, 0) and row.method and row.call_reduction > 0.5
    ]
    strong_rows.sort(key=lambda row: (row.call_reduction or 0.0), reverse=True)
    strong_rows = strong_rows[:30]

    lines: list[str] = [
        "# Comprehensive P2CCCD Benchmark And Case Report",
        "",
        f"- Generated UTC: `{summary['generated_utc']}`",
        f"- Benchmark files scanned: `{len(benchmark_files)}`",
        f"- Normalized method rows: `{len(method_rows)}`",
        f"- Visual case directories: `{len(visual_artifacts)}`",
        f"- MP4 artifacts: `{summary['mp4_count']}`; verified openable: `{summary['verified_mp4_count']}`",
        "",
        "## Scope",
        "",
        "- This report is a repository-wide aggregation of existing benchmark artifacts and newly added visualization cases.",
        "- It does not rerun expensive training unless a source benchmark script has already produced machine-readable results.",
        "- Rows with `FN=0` retain the original report's certificate/no-false-negative semantics; rows without FN are treated as visual or metadata evidence, not correctness proof.",
        "",
        "## Benchmark Groups",
        "",
        *markdown_table(["Group"], [[group] for group in covered_groups]),
        "",
        "## High-Reduction Rows",
        "",
        *markdown_table(
            ["Case", "Dataset", "Split", "Method", "Candidates", "Exact calls", "Call reduction", "Work reduction", "FN", "Source"],
            [
                [
                    row.case[:48],
                    row.dataset[:36],
                    row.split,
                    row.method[:32],
                    "n/a" if row.candidates is None else f"{row.candidates:.0f}",
                    "n/a" if row.exact_calls is None else f"{row.exact_calls:.3g}",
                    pct(row.call_reduction),
                    pct(row.work_reduction),
                    "n/a" if row.fn is None else str(row.fn),
                    row.source_file,
                ]
                for row in strong_rows
            ],
        ),
        "",
        "## Newly Added / Recent Case Coverage",
        "",
    ]
    recent_keywords = ("run_id", "run_id", "run_id", "scalable_ccd_scene", "standard_graphics", "many_object", "soft_body")
    recent_files = [row for row in benchmark_files if any(key in row.path or key in row.case for key in recent_keywords)]
    recent_files.sort(key=lambda row: row.path)
    lines.extend(
        markdown_table(
            ["Benchmark", "Case", "Methods", "Candidates", "Positives", "Best call red.", "Best work red.", "Min FN"],
            [
                [
                    row.benchmark_group[:48],
                    row.case[:48],
                    str(row.method_count),
                    "n/a" if row.candidate_rows is None else str(row.candidate_rows),
                    "n/a" if row.positive_rows is None else str(row.positive_rows),
                    pct(row.best_call_reduction),
                    pct(row.best_work_reduction),
                    "n/a" if row.min_fn is None else str(row.min_fn),
                ]
                for row in recent_files[:80]
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Visual Artifact Validation",
            "",
            *markdown_table(
                ["Case dir", "MP4", "Verified", "Frames", "FPS", "Representative"],
                [
                    [
                        Path(row.case_dir).name[:48],
                        str(row.mp4_count),
                        str(row.verified_mp4_count),
                        f"{row.min_frames}-{row.max_frames}" if row.min_frames is not None else "n/a",
                        f"{row.min_fps:.1f}-{row.max_fps:.1f}" if row.min_fps is not None and row.max_fps is not None else "n/a",
                        row.representative,
                    ]
                    for row in visual_artifacts
                ],
            ),
            "",
            "## Gaps / Follow-up",
            "",
        ]
    )
    if missing_benchmark:
        lines.extend(f"- Visual case `{name}` has no same-name benchmark group; verify whether its metrics are merged under another benchmark directory." for name in missing_benchmark[:60])
    else:
        lines.append("- No visual-only case directory was detected by same-name matching.")
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```powershell",
            "conda activate cudadev",
            "python src\\tools\\generate_comprehensive_benchmark_report.py",
            "```",
            "",
        ]
    )
    (OUTPUT_DIR / f"{REPORT_NAME}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    benchmark_files, method_rows = scan_benchmarks()
    visual_artifacts = scan_visual_artifacts()
    write_report(benchmark_files, method_rows, visual_artifacts)
    print(
        json.dumps(
            {
                "run_name": REPORT_NAME,
                "output_dir": rel(OUTPUT_DIR),
                "benchmark_files": len(benchmark_files),
                "method_rows": len(method_rows),
                "visual_cases": len(visual_artifacts),
                "mp4_count": sum(row.mp4_count for row in visual_artifacts),
                "verified_mp4_count": sum(row.verified_mp4_count for row in visual_artifacts),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
