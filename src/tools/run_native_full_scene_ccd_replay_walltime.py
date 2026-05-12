#!/usr/bin/env python3
"""Run native full-scene CCD replay / detection wall-time.

This benchmark replays Scalable-CCD full-scene timestep query files through a
native C++ Tight-Inclusion runner.  It measures CCD detection time, not
simulation/contact-solver step time.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / "src"
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from p2cccd.proposal.policy_head_selection import (  # noqa: E402
    RTSTPFPolicyHead,
    score_rtstpf_candidates,
)
RUN_NAME = "native_full_scene_ccd_replay_walltime_run_id"
SOURCE_MANIFEST = (
    ROOT
    / "datasets"
    / "training"
    / "scalable_ccd_scene_groups"
    / "shards"
    / "scalable_ccd_sample_scene_candidate_groups_run_id"
    / "manifest.json"
)
SCALABLE_TRAINING_CHECKPOINT = (
    ROOT
    / "outputs"
    / "stpf_training"
    / "scalable_ccd_scene_supplementary_training_run_id"
    / "model_state.pt"
)
OUT_DIR = ROOT / "benchmark" / RUN_NAME
BUILD_EXE = ROOT / "build_tools" / "full_scene_ccd_replay_benchmark.exe"
CPP_SOURCE = ROOT / "tools" / "full_scene_ccd_replay_benchmark.cpp"
TI_SRC = ROOT / "baseline" / "Tight-Inclusion" / "src"
TI_LIB = ROOT / "baseline" / "Tight-Inclusion" / "build-release" / "libtight_inclusion.a"
SPDLOG_LIB = ROOT / "baseline" / "Tight-Inclusion" / "build-release" / "_deps" / "spdlog-build" / "libspdlog.a"

PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


SPLITS = {
    "armadillo-rollers": "train",
    "cloth-ball": "train",
    "n-body-simulation": "train",
    "puffer-ball": "train",
    "cloth-funnel": "validation",
    "rod-twist": "heldout_test",
}


def resolve_eigen_src() -> Path:
    candidates: list[Path] = []
    if value := os.environ.get("P2CCCD_EIGEN_SRC"):
        candidates.append(Path(value))
    if value := os.environ.get("EIGEN3_INCLUDE_DIR"):
        candidates.append(Path(value))
    cpm_root = Path.home() / ".cache" / "CPM" / "eigen"
    if cpm_root.exists():
        candidates.extend(sorted((path for path in cpm_root.iterdir() if path.is_dir()), reverse=True))
    for candidate in candidates:
        if (candidate / "Eigen" / "Core").exists():
            return candidate
        if (candidate / "eigen3" / "Eigen" / "Core").exists():
            return candidate / "eigen3"
    return candidates[0] if candidates else Path("eigen3")


EIGEN_SRC = resolve_eigen_src()


def resolve_gpp() -> Path:
    if value := os.environ.get("P2CCCD_GPP"):
        return Path(value)
    if value := shutil.which("g++"):
        return Path(value)
    if value := shutil.which("clang++"):
        return Path(value)
    return Path("g++.exe")


GPP = resolve_gpp()


def prepend_windows_toolchain_paths(env: dict[str, str]) -> dict[str, str]:
    prefixes: list[str] = []
    for key in ("P2CCCD_TOOLCHAIN_BIN", "P2CCCD_MINGW_BIN"):
        value = os.environ.get(key)
        if value:
            prefixes.append(value)
    if GPP.is_absolute() and GPP.parent.exists():
        prefixes.append(str(GPP.parent))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_bin = Path(conda_prefix) / "Library" / "bin"
        if conda_bin.exists():
            prefixes.append(str(conda_bin))
    if prefixes:
        env["PATH"] = ";".join(prefixes + [env.get("PATH", "")])
    return env


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def as_p(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return WORKSPACE / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_manifest(source_manifest: Path, out_dir: Path) -> Path:
    payload = load_json(source_manifest)
    files = []
    for group in payload["groups"]:
        scene = str(group["scene"])
        files.append(
            {
                "scene": scene,
                "kind": str(group["kind"]),
                "split": SPLITS.get(scene, "scene_eval"),
                "timestep": int(group["timestep"]),
                "csv_path": as_p(group["source_query_csv"]).as_posix(),
                "bool_json": as_p(group["source_mma_bool_json"]).as_posix(),
                "frame0": as_p(group.get("frame0", "")).as_posix() if group.get("frame0") else "",
                "frame1": as_p(group.get("frame1", "")).as_posix() if group.get("frame1") else "",
                "query_count": int(group["query_count"]),
                "positive_count": int(group["positive_count"]),
                "npz": as_p(group["npz"]).as_posix(),
            }
        )
    manifest = {
        "schema_version": 1,
        "run_name": RUN_NAME,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "scope": "native_full_scene_ccd_replay_detection_walltime",
        "source_manifest": rel(source_manifest),
        "source_dataset": "Sample-Scalable-CCD-Data full-scene timestep query files",
        "timing_contract": {
            "included": [
                "native C++ query order construction or score sort",
                "native Tight-Inclusion exact CCD calls",
                "scene-step any-hit stopping for *AnyHit methods",
            ],
            "reported_separately": ["query CSV and label JSON load/parse time"],
            "excluded": [
                "simulation dynamics",
                "contact response and solver step",
                "rendering",
                "offline STPF model inference used to precompute schedule scores",
            ],
        },
        "files": files,
    }
    path = out_dir / f"{RUN_NAME}_manifest.json"
    write_json(path, manifest)
    return path


def load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"].item()))
    return arrays, metadata


def trained_scores(checkpoint: Path, arrays: dict[str, np.ndarray], device: str) -> np.ndarray:
    import torch
    from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model_from_checkpoint_payload

    payload = torch.load(checkpoint, map_location=device)
    model, state_dict = build_stpf_model_from_checkpoint_payload(payload, fallback_preset=STPFModelPreset.TINY_MLP)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    features = np.asarray(arrays["features"], dtype=np.float32)
    priority_scores = np.zeros(features.shape[0], dtype=np.float32)
    cost_scores = np.zeros(features.shape[0], dtype=np.float32)
    uncertainty_scores = np.zeros(features.shape[0], dtype=np.float32)
    batch_size = 65536
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
            out = model(batch)
            stop = start + int(batch.shape[0])
            priority_scores[start:stop] = out.priority_score.detach().cpu().numpy()
            cost_scores[start:stop] = out.cost_score.detach().cpu().numpy()
            uncertainty_scores[start:stop] = out.uncertainty_score.detach().cpu().numpy()
    return score_rtstpf_candidates(
        {
            "priority_score": priority_scores,
            "cost_score": cost_scores,
            "uncertainty_score": uncertainty_scores,
        },
        {"features": features},
        head=RTSTPFPolicyHead.COST_AWARE,
    )


def write_schedule(path: Path, manifest_path: Path, method: str, device: str) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(fixed_seed)
    rows = 0
    groups = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["csv_path", "query_index", "score"])
        for file in manifest["files"]:
            arrays, _ = load_npz(Path(file["npz"]))
            labels = np.asarray(arrays["ground_truth"], dtype=np.bool_)
            costs = np.asarray(arrays["costs"], dtype=np.float64)
            if method == "TrainedSTPFAnyHit":
                if not SCALABLE_TRAINING_CHECKPOINT.exists():
                    raise FileNotFoundError(f"missing STPF checkpoint: {SCALABLE_TRAINING_CHECKPOINT}")
                scores = trained_scores(SCALABLE_TRAINING_CHECKPOINT, arrays, device)
            elif method == "HeuristicProximityAnyHit":
                scores = np.asarray(arrays["features"][:, 22], dtype=np.float64)
            elif method == "RandomAnyHit":
                scores = rng.random(labels.shape[0])
            elif method == "OracleAnyHit":
                scores = labels.astype(np.float64) + 1.0e-6 / np.maximum(costs, 1.0e-9)
            else:
                raise ValueError(f"no schedule writer for {method}")
            csv_path = str(file["csv_path"])
            for query_index, score in enumerate(scores):
                writer.writerow([csv_path, query_index, f"{float(score):.12g}"])
            rows += int(scores.shape[0])
            groups += 1
    return {"path": rel(path), "method": method, "rows": rows, "groups": groups}


def build_native_runner(force: bool = False) -> None:
    if not CPP_SOURCE.exists():
        raise FileNotFoundError(CPP_SOURCE)
    if not TI_LIB.exists():
        raise FileNotFoundError(TI_LIB)
    if not SPDLOG_LIB.exists():
        raise FileNotFoundError(SPDLOG_LIB)
    if not EIGEN_SRC.exists():
        raise FileNotFoundError(EIGEN_SRC)
    if BUILD_EXE.exists() and not force and BUILD_EXE.stat().st_mtime >= CPP_SOURCE.stat().st_mtime:
        return
    BUILD_EXE.parent.mkdir(parents=True, exist_ok=True)
    if GPP.is_absolute() and not GPP.exists():
        raise FileNotFoundError(GPP)
    if not GPP.is_absolute() and shutil.which(str(GPP)) is None:
        raise FileNotFoundError(GPP)
    cmd = [
        str(GPP),
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        f"-I{TI_SRC}",
        f"-I{EIGEN_SRC}",
        str(CPP_SOURCE),
        str(TI_LIB),
        str(SPDLOG_LIB),
        "-o",
        str(BUILD_EXE),
    ]
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to build native runner\nSTDOUT:\n"
            + completed.stdout
            + "\nSTDERR:\n"
            + completed.stderr
        )


def run_native(method: str, manifest: Path, schedule: Path | None, out_dir: Path, args: argparse.Namespace) -> Path:
    output = out_dir / "raw_jsonl" / f"{method}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BUILD_EXE),
        "--manifest",
        str(manifest),
        "--output-jsonl",
        str(output),
        "--method",
        method,
        "--ms",
        str(args.ms),
        "--tolerance",
        str(args.tolerance),
        "--t-max",
        str(args.t_max),
        "--max-itr",
        str(args.max_itr),
    ]
    if schedule is not None:
        cmd.extend(["--schedule", str(schedule)])
    if args.max_files:
        cmd.extend(["--max-files", str(args.max_files)])
    if args.max_queries_per_file:
        cmd.extend(["--max-queries-per-file", str(args.max_queries_per_file)])
    env = prepend_windows_toolchain_paths(os.environ.copy())
    completed = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, capture_output=True, check=False)
    (out_dir / "raw_logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_logs" / f"{method}.stdout.log").write_text(completed.stdout, encoding="utf-8", errors="replace")
    (out_dir / "raw_logs" / f"{method}.stderr.log").write_text(completed.stderr, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(f"{method} failed with exit code {completed.returncode}: {completed.stderr}")
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, dict[str, Any]] = {}
    for row in rows:
        method = str(row["method"])
        out = by_method.setdefault(
            method,
            {
                "method": method,
                "scene_step_kind_groups": 0,
                "candidates": 0,
                "positives": 0,
                "exact_calls": 0,
                "skipped_candidates": 0,
                "scene_tp": 0,
                "scene_tn": 0,
                "scene_fp": 0,
                "scene_fn": 0,
                "exact_false_positive_count": 0,
                "exact_false_negative_count": 0,
                "load_ms": 0.0,
                "schedule_ms": 0.0,
                "exact_ms": 0.0,
                "detection_wall_ms": 0.0,
                "total_wall_ms": 0.0,
            },
        )
        out["scene_step_kind_groups"] += 1
        out["candidates"] += int(row["query_count"])
        out["positives"] += int(row["positive_count"])
        out["exact_calls"] += int(row["exact_calls"])
        out["skipped_candidates"] += int(row["skipped_candidates"])
        out["scene_tp"] += int(row["scene_tp"])
        out["scene_tn"] += int(row["scene_tn"])
        out["scene_fp"] += int(row["scene_fp"])
        out["scene_fn"] += int(row["scene_fn"])
        out["exact_false_positive_count"] += int(row["exact_false_positive_count"])
        out["exact_false_negative_count"] += int(row["exact_false_negative_count"])
        out["load_ms"] += float(row["load_us"]) / 1000.0
        out["schedule_ms"] += float(row["schedule_us"]) / 1000.0
        out["exact_ms"] += float(row["exact_us"]) / 1000.0
        out["detection_wall_ms"] += float(row["detection_wall_us"]) / 1000.0
        out["total_wall_ms"] += float(row["total_wall_us"]) / 1000.0
    summary = []
    for out in by_method.values():
        candidates = max(1, int(out["candidates"]))
        scene_pos = int(out["scene_tp"]) + int(out["scene_fn"])
        out["call_reduction"] = 1.0 - int(out["exact_calls"]) / candidates
        out["scene_recall"] = int(out["scene_tp"]) / max(1, scene_pos)
        out["avg_exact_calls_per_group"] = int(out["exact_calls"]) / max(1, int(out["scene_step_kind_groups"]))
        out["avg_detection_ms_per_group"] = float(out["detection_wall_ms"]) / max(
            1, int(out["scene_step_kind_groups"])
        )
        summary.append(out)
    return sorted(summary, key=lambda item: str(item["method"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(out_dir: Path, summary: list[dict[str, Any]], rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    lines = [
        f"# Native Full-scene CCD Replay Wall-time ({RUN_NAME})",
        "",
        "Scope: native full-scene trajectory replay / CCD detection wall-time.",
        "This is not native full-scene simulation wall-time because no dynamics or contact solver step is timed.",
        "",
        "## Timing Contract",
        "",
        "- Dataset: Scalable-CCD full-scene timestep query files converted into scene-step-kind replay groups.",
        "- Exact backend: native C++ Tight-Inclusion `vertexFaceCCD` / `edgeEdgeCCD`.",
        "- Any-hit methods stop after the first certified scene-step hit; `NoProposalAllExactEnumeration` enumerates every candidate.",
        "- `detection_wall_ms` includes native schedule ordering/sort plus native exact CCD calls; query file load/parse is reported separately.",
        "- STPF schedule scores are precomputed from the existing checkpoint; model inference time is excluded from this detection-wall table.",
        "- Dataset caveat: all 12 scene-step-kind groups contain positives and are highly positive-dense, so any-hit methods often stop after one exact call; this table validates native replay/detection timing, not learned schedule superiority over natural order.",
        "- Exact FP/FN columns compare native Tight-Inclusion outcomes against the Scalable-CCD `mma_bool` labels on executed candidates; scene FN is the any-hit detection failure count.",
        "",
        "## Summary",
        "",
        "| Method | Groups | Candidates | Exact calls | Call reduction | Scene FN | Exact FP | Exact FN | Scene recall | Exact ms | Detection wall ms | Total wall ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {method} | {scene_step_kind_groups} | {candidates} | {exact_calls} | "
            "{call_reduction:.6f} | {scene_fn} | {exact_false_positive_count} | "
            "{exact_false_negative_count} | {scene_recall:.6f} | {exact_ms:.3f} | "
            "{detection_wall_ms:.3f} | {total_wall_ms:.3f} |".format(**row)
        )
    lines += [
        "",
        "## Per-scene Rows",
        "",
        "| Method | Scene | Kind | Timestep | Candidates | Positives | Exact calls | Scene FN | Detection wall ms | Load ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {scene} | {kind} | {timestep} | {query_count} | {positive_count} | "
            "{exact_calls} | {scene_fn} | {detection_wall_ms:.3f} | {load_ms:.3f} |".format(
                **{
                    **row,
                    "detection_wall_ms": float(row["detection_wall_us"]) / 1000.0,
                    "load_ms": float(row["load_us"]) / 1000.0,
                }
            )
        )
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Manifest: `{metadata['manifest']}`",
        f"- Native runner: `{metadata['native_runner']}`",
        f"- Source manifest: `{metadata['source_manifest']}`",
        f"- Raw JSONL dir: `{metadata['raw_jsonl_dir']}`",
        "",
        "## Claim Boundary",
        "",
        "Safe wording: native full-scene CCD replay / any-hit detection wall-time.",
        "Unsafe wording: native full-scene simulation wall-time or end-to-end simulator speedup.",
    ]
    (out_dir / f"{RUN_NAME}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--skip-all-exact", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-queries-per-file", type=int, default=0)
    parser.add_argument("--ms", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--max-itr", type=int, default=1000000)
    args = parser.parse_args()

    started = dt.datetime.now().isoformat(timespec="seconds")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_native_runner(force=args.force_build)
    manifest = build_manifest(SOURCE_MANIFEST, OUT_DIR)
    schedules = {}
    for method in ("TrainedSTPFAnyHit", "HeuristicProximityAnyHit", "RandomAnyHit", "OracleAnyHit"):
        schedules[method] = write_schedule(OUT_DIR / "schedules" / f"{method}.csv", manifest, method, args.device)

    method_specs: list[tuple[str, Path | None]] = [
        ("NaturalAnyHit", None),
        ("RandomAnyHit", OUT_DIR / "schedules" / "RandomAnyHit.csv"),
        ("HeuristicProximityAnyHit", OUT_DIR / "schedules" / "HeuristicProximityAnyHit.csv"),
        ("TrainedSTPFAnyHit", OUT_DIR / "schedules" / "TrainedSTPFAnyHit.csv"),
        ("OracleAnyHit", OUT_DIR / "schedules" / "OracleAnyHit.csv"),
    ]
    if not args.skip_all_exact:
        method_specs.insert(0, ("NoProposalAllExactEnumeration", None))

    all_rows: list[dict[str, Any]] = []
    run_outputs = []
    for method, schedule in method_specs:
        jsonl = run_native(method, manifest, schedule, OUT_DIR, args)
        run_outputs.append({"method": method, "jsonl": rel(jsonl), "schedule": rel(schedule) if schedule else ""})
        all_rows.extend(read_jsonl(jsonl))

    summary = aggregate(all_rows)
    write_csv(OUT_DIR / f"{RUN_NAME}_rows.csv", all_rows)
    write_csv(OUT_DIR / f"{RUN_NAME}_summary.csv", summary)
    metadata = {
        "run_name": RUN_NAME,
        "started_at": started,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "manifest": rel(manifest),
        "source_manifest": rel(SOURCE_MANIFEST),
        "native_runner": rel(BUILD_EXE),
        "cpp_source": rel(CPP_SOURCE),
        "raw_jsonl_dir": rel(OUT_DIR / "raw_jsonl"),
        "schedules": schedules,
        "runs": run_outputs,
        "timing_scope": "native full-scene CCD replay / detection wall-time; no solver/contact response",
    }
    write_json(OUT_DIR / f"{RUN_NAME}.json", {"metadata": metadata, "summary": summary, "rows": all_rows})
    write_report(OUT_DIR, summary, all_rows, metadata)
    print(OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
