#!/usr/bin/env python3
"""Build and run native TI wall-time for real scene/object conservative envelopes.

The benchmark loads adjacent full-scene mesh frames, partitions connected
components as objects, constructs a swept object-pair AABB envelope and swept
primitive VF/EE candidates, then times the native Tight-Inclusion exact backend.
It is intentionally separate from Scalable-CCD query-file replay: query CSVs are
not used as the evaluated candidate set.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / "src"
RUN_NAME = "scene_object_envelope_native_ti_walltime_run_id"
SOURCE_ROOT = ROOT / "baseline" / "Sample-Scalable-CCD-Data"
OUT_DIR = ROOT / "benchmark" / RUN_NAME
CPP_SOURCE = ROOT / "tools" / "scene_object_envelope_native_ti_walltime.cpp"
BUILD_EXE = ROOT / "build_tools" / "scene_object_envelope_native_ti_walltime.exe"
DEFAULT_STPF_CHECKPOINT = ROOT / "outputs" / "stpf_training" / "native_ti_heldout_dense_group_run_id" / "group_heldout" / "model_state.pt"
TI_SRC = ROOT / "baseline" / "Tight-Inclusion" / "src"
TI_LIB = ROOT / "baseline" / "Tight-Inclusion" / "build-release" / "libtight_inclusion.a"
SPDLOG_LIB = ROOT / "baseline" / "Tight-Inclusion" / "build-release" / "_deps" / "spdlog-build" / "libspdlog.a"

SCENE_ORDER = (
    "armadillo-rollers",
    "cloth-ball",
    "cloth-funnel",
    "n-body-simulation",
    "puffer-ball",
    "rod-twist",
)


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


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()


def frame_number(path: Path) -> int | None:
    if path.name.startswith("._") or path.name == ".DS_Store":
        return None
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


def discover_frame_pairs(source_root: Path, scenes: list[str]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for scene in scenes:
        frames_dir = source_root / scene / "frames"
        if not frames_dir.exists():
            raise FileNotFoundError(frames_dir)
        numbered = [(number, path) for path in frames_dir.glob("*.ply") if (number := frame_number(path)) is not None]
        numbered.sort(key=lambda item: (item[0], item[1].name))
        if len(numbered) < 2:
            raise RuntimeError(f"{frames_dir} has fewer than two PLY frames")
        pairs.append(
            {
                "scene": scene,
                "timestep0": numbered[0][0],
                "timestep1": numbered[1][0],
                "frame0": numbered[0][1],
                "frame1": numbered[1][1],
            }
        )
    return pairs


def build_native_runner(force: bool) -> None:
    for path in (CPP_SOURCE, TI_LIB, SPDLOG_LIB, EIGEN_SRC):
        if not path.exists():
            raise FileNotFoundError(path)
    if GPP.is_absolute() and not GPP.exists():
        raise FileNotFoundError(GPP)
    if not GPP.is_absolute() and shutil.which(str(GPP)) is None:
        raise FileNotFoundError(GPP)
    if BUILD_EXE.exists() and not force and BUILD_EXE.stat().st_mtime >= CPP_SOURCE.stat().st_mtime:
        return
    BUILD_EXE.parent.mkdir(parents=True, exist_ok=True)
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
            "failed to build native scene/object envelope runner\nSTDOUT:\n"
            + completed.stdout
            + "\nSTDERR:\n"
            + completed.stderr
        )


def export_tiny_stpf_weights(checkpoint: Path, output: Path) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    mean = payload["feature_mean"].reshape(-1).astype("float64")
    std = payload["feature_std"].reshape(-1).astype("float64")
    tensors = {
        "w0": state["net.0.weight"].detach().cpu().numpy().astype("float64").reshape(-1),
        "b0": state["net.0.bias"].detach().cpu().numpy().astype("float64").reshape(-1),
        "w1": state["net.2.weight"].detach().cpu().numpy().astype("float64").reshape(-1),
        "b1": state["net.2.bias"].detach().cpu().numpy().astype("float64").reshape(-1),
        "w2": state["net.4.weight"].detach().cpu().numpy().astype("float64").reshape(-1),
        "b2": state["net.4.bias"].detach().cpu().numpy().astype("float64").reshape(-1),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("version 1\n")
        handle.write("input_dim 32\n")
        handle.write("hidden0 64\n")
        handle.write("hidden1 32\n")
        handle.write("mean " + " ".join(f"{v:.17g}" for v in mean) + "\n")
        handle.write("std " + " ".join(f"{v:.17g}" for v in std) + "\n")
        for key in ("w0", "b0", "w1", "b1", "w2", "b2"):
            handle.write(key + " " + " ".join(f"{v:.17g}" for v in tensors[key]) + "\n")


def run_scene(pair: dict[str, Any], args: argparse.Namespace, out_dir: Path) -> Path:
    output = out_dir / "raw_jsonl" / f"{safe_name(pair['scene'])}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BUILD_EXE),
        "--scene",
        str(pair["scene"]),
        "--frame0",
        str(pair["frame0"]),
        "--frame1",
        str(pair["frame1"]),
        "--output-jsonl",
        str(output),
        "--thickness",
        str(args.thickness),
        "--ms",
        str(args.ms),
        "--tolerance",
        str(args.tolerance),
        "--t-max",
        str(args.t_max),
        "--max-itr",
        str(args.max_itr),
        "--max-vf-candidates",
        str(args.max_vf_candidates),
        "--max-ee-candidates",
        str(args.max_ee_candidates),
        "--stpf-weights",
        str(args.stpf_weights),
        "--proposal-top-k",
        str(args.proposal_top_k),
        "--optimized-frontier-k",
        str(args.optimized_frontier_k),
        "--optimized-scan-limit-per-group",
        str(args.optimized_scan_limit_per_group),
        "--optimized-random-gate-object-count",
        str(args.optimized_random_gate_object_count),
    ]
    if args.feature_jsonl_dir:
        feature_path = out_dir / args.feature_jsonl_dir / f"{safe_name(pair['scene'])}.jsonl"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(
            [
                "--feature-jsonl",
                str(feature_path),
                "--feature-negative-stride",
                str(args.feature_negative_stride),
            ]
        )
    if args.feature_export_only:
        cmd.append("--feature-export-only")
    if args.exclude_self_object_pairs:
        cmd.append("--exclude-self-object-pairs")
    env = prepend_windows_toolchain_paths(os.environ.copy())
    completed = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, capture_output=True, check=False)
    log_dir = out_dir / "raw_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{safe_name(pair['scene'])}.stdout.log").write_text(completed.stdout, encoding="utf-8", errors="replace")
    (log_dir / f"{safe_name(pair['scene'])}.stderr.log").write_text(completed.stderr, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(f"{pair['scene']} failed with exit code {completed.returncode}: {completed.stderr}")
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"{pair['scene']} produced an empty raw JSONL file: {output}")
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty raw JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"raw JSONL file contains no rows: {path}")
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['method']}::{row['kind']}"
        out = grouped.setdefault(
            key,
            {
                "method": row["method"],
                "kind": row["kind"],
                "scene_kind_groups": 0,
                "candidates": 0,
                "exact_calls": 0,
                "proposal_exact_calls": 0,
                "fallback_exact_calls": 0,
                "positive_exact_calls": 0,
                "negative_exact_calls": 0,
                "positive_proposal_exact_calls": 0,
                "positive_fallback_exact_calls": 0,
                "negative_proposal_exact_calls": 0,
                "negative_fallback_exact_calls": 0,
                "positive_proposal_hits": 0,
                "positive_fallback_hits": 0,
                "positive_first_hit_rank_sum": 0,
                "positive_first_hit_groups": 0,
                "native_hits": 0,
                "detected_groups": 0,
                "capped_groups": 0,
                "group_count": 0,
                "positive_groups": 0,
                "negative_groups": 0,
                "fallback_groups": 0,
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "envelope_ms": 0.0,
                "ordering_ms": 0.0,
                "native_exact_backend_ms": 0.0,
                "total_wall_ms": 0.0,
            },
        )
        out["scene_kind_groups"] += 1
        out["candidates"] += int(row["candidate_count"])
        out["exact_calls"] += int(row["exact_calls"])
        for key in (
            "proposal_exact_calls",
            "fallback_exact_calls",
            "positive_exact_calls",
            "negative_exact_calls",
            "positive_proposal_exact_calls",
            "positive_fallback_exact_calls",
            "negative_proposal_exact_calls",
            "negative_fallback_exact_calls",
            "positive_proposal_hits",
            "positive_fallback_hits",
            "positive_first_hit_rank_sum",
            "positive_first_hit_groups",
        ):
            out[key] += int(row.get(key, 0))
        out["native_hits"] += int(row["native_hit_count"])
        out["detected_groups"] += int(bool(row["detected_hit"]))
        out["capped_groups"] += int(bool(row["candidate_capped"]))
        out["group_count"] += int(row.get("group_count", 0))
        out["positive_groups"] += int(row.get("positive_groups", 0))
        out["negative_groups"] += int(row.get("negative_groups", 0))
        out["fallback_groups"] += int(row.get("fallback_groups", 0))
        out["tp"] += int(row.get("tp", 0))
        out["tn"] += int(row.get("tn", 0))
        out["fp"] += int(row.get("fp", 0))
        out["fn"] += int(row.get("fn", 0))
        out["envelope_ms"] += float(row["envelope_ms"])
        out["ordering_ms"] += float(row.get("ordering_ms", 0.0))
        out["native_exact_backend_ms"] += float(row["native_exact_backend_ms"])
        out["total_wall_ms"] += float(row["total_wall_ms"])
    summary = []
    for out in grouped.values():
        out["call_reduction_vs_envelope_all_exact"] = (
            1.0 - int(out["exact_calls"]) / max(1, int(out["candidates"]))
            if str(out["method"]).endswith("AnyHit+TI")
            else 0.0
        )
        out["exact_calls_per_second"] = (
            1000.0 * int(out["exact_calls"]) / max(1.0e-12, float(out["native_exact_backend_ms"]))
        )
        out["scheduler_backend_ms"] = float(out["ordering_ms"]) + float(out["native_exact_backend_ms"])
        out["fallback_rate"] = int(out["fallback_groups"]) / max(1, int(out["group_count"]))
        out["positive_proposal_hit_rate"] = int(out["positive_proposal_hits"]) / max(1, int(out["positive_groups"]))
        out["positive_fallback_hit_rate"] = int(out["positive_fallback_hits"]) / max(1, int(out["positive_groups"]))
        out["positive_first_hit_mean_exact_calls"] = int(out["positive_first_hit_rank_sum"]) / max(
            1, int(out["positive_first_hit_groups"])
        )
        summary.append(out)
    return sorted(summary, key=lambda row: (str(row["kind"]), str(row["method"])))


def build_main_scheduler_table(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, dict[str, Any]] = {}

    def add_row(target: str, row: dict[str, Any]) -> None:
        out = by_method.setdefault(
            target,
            {
                "method": target,
                "candidates": 0,
                "exact_calls": 0,
                "proposal_exact_calls": 0,
                "fallback_exact_calls": 0,
                "positive_exact_calls": 0,
                "negative_exact_calls": 0,
                "positive_proposal_exact_calls": 0,
                "positive_fallback_exact_calls": 0,
                "negative_proposal_exact_calls": 0,
                "negative_fallback_exact_calls": 0,
                "positive_proposal_hits": 0,
                "positive_fallback_hits": 0,
                "positive_first_hit_rank_sum": 0,
                "positive_first_hit_groups": 0,
                "native_exact_backend_ms": 0.0,
                "ordering_ms": 0.0,
                "scheduler_backend_ms": 0.0,
                "group_count": 0,
                "positive_groups": 0,
                "negative_groups": 0,
                "fallback_groups": 0,
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
            },
        )
        for key in (
            "candidates",
            "exact_calls",
            "proposal_exact_calls",
            "fallback_exact_calls",
            "positive_exact_calls",
            "negative_exact_calls",
            "positive_proposal_exact_calls",
            "positive_fallback_exact_calls",
            "negative_proposal_exact_calls",
            "negative_fallback_exact_calls",
            "positive_proposal_hits",
            "positive_fallback_hits",
            "positive_first_hit_rank_sum",
            "positive_first_hit_groups",
            "group_count",
            "positive_groups",
            "negative_groups",
            "fallback_groups",
            "tp",
            "tn",
            "fp",
            "fn",
        ):
            out[key] += int(row.get(key, 0))
        for key in ("native_exact_backend_ms", "ordering_ms", "scheduler_backend_ms"):
            out[key] += float(row.get(key, 0.0))

    wanted = {
        "EnvelopeAllExact+TI": "AllExact+TI",
        "FairFrontierLearnedAnyHit+TI": "FairFrontierLearnedAnyHit+TI",
        "FairFrontierLearnedResidualAnyHit+TI": "FairFrontierLearnedResidualAnyHit+TI",
        "FairFrontierRandomAnyHit+TI": "FairFrontierRandomAnyHit+TI",
        "FairFrontierProximityAnyHit+TI": "FairFrontierProximityAnyHit+TI",
        "FairFrontierMotionAnyHit+TI": "FairFrontierMotionAnyHit+TI",
        "OptimizedFrozenLearnedAnyHit+TI": "OptimizedFrozenLearnedAnyHit+TI",
        "FrozenLearnedAnyHit+TI": "FrozenLearnedAnyHit+TI",
        "LearnedResidualAnyHit+TI": "LearnedResidualAnyHit+TI",
        "RandomAnyHit+TI": "RandomAnyHit+TI",
        "ProximityHeuristicAnyHit+TI": "ProximityHeuristicAnyHit+TI",
        "MotionHeuristicAnyHit+TI": "MotionHeuristicAnyHit+TI",
    }
    for row in summary:
        if row["method"] in wanted:
            add_row(wanted[row["method"]], row)

    oracle_sources = []
    for kind in sorted({row["kind"] for row in summary}):
        heuristics = [
            row
            for row in summary
            if row["kind"] == kind and row["method"] in {"ProximityHeuristicAnyHit+TI", "MotionHeuristicAnyHit+TI"}
        ]
        if heuristics:
            oracle_sources.append(min(heuristics, key=lambda row: (int(row["exact_calls"]), float(row["native_exact_backend_ms"]))))
    for row in oracle_sources:
        add_row("BestFixedHeuristicOracle+TI", row)

    fair_oracle_sources = []
    for kind in sorted({row["kind"] for row in summary}):
        heuristics = [
            row
            for row in summary
            if row["kind"] == kind and row["method"] in {"FairFrontierProximityAnyHit+TI", "FairFrontierMotionAnyHit+TI"}
        ]
        if heuristics:
            fair_oracle_sources.append(min(heuristics, key=lambda row: (int(row["exact_calls"]), float(row["scheduler_backend_ms"]))))
    for row in fair_oracle_sources:
        add_row("FairFrontierBestHeuristicOracle+TI", row)

    ordered = [
        "AllExact+TI",
        "FairFrontierLearnedAnyHit+TI",
        "FairFrontierLearnedResidualAnyHit+TI",
        "FairFrontierRandomAnyHit+TI",
        "FairFrontierProximityAnyHit+TI",
        "FairFrontierMotionAnyHit+TI",
        "FairFrontierBestHeuristicOracle+TI",
        "OptimizedFrozenLearnedAnyHit+TI",
        "FrozenLearnedAnyHit+TI",
        "LearnedResidualAnyHit+TI",
        "RandomAnyHit+TI",
        "ProximityHeuristicAnyHit+TI",
        "MotionHeuristicAnyHit+TI",
        "BestFixedHeuristicOracle+TI",
    ]
    rows = []
    all_exact_calls = max(1, int(by_method.get("AllExact+TI", {}).get("exact_calls", 1)))
    for method in ordered:
        if method not in by_method:
            continue
        row = by_method[method]
        row["call_reduction_vs_all_exact"] = 1.0 - int(row["exact_calls"]) / all_exact_calls
        row["fallback_rate"] = int(row["fallback_groups"]) / max(1, int(row["group_count"]))
        row["coverage_rate"] = 1.0
        row["positive_proposal_hit_rate"] = int(row["positive_proposal_hits"]) / max(1, int(row["positive_groups"]))
        row["positive_fallback_hit_rate"] = int(row["positive_fallback_hits"]) / max(1, int(row["positive_groups"]))
        row["positive_first_hit_mean_exact_calls"] = int(row["positive_first_hit_rank_sum"]) / max(
            1, int(row["positive_first_hit_groups"])
        )
        rows.append(row)
    return rows


def build_fair_frontier_ranking_table(main_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = [
        "FairFrontierLearnedAnyHit+TI",
        "FairFrontierLearnedResidualAnyHit+TI",
        "FairFrontierRandomAnyHit+TI",
        "FairFrontierProximityAnyHit+TI",
        "FairFrontierMotionAnyHit+TI",
        "FairFrontierBestHeuristicOracle+TI",
    ]
    return [row for row in main_table if row["method"] in methods]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    out_dir: Path,
    summary: list[dict[str, Any]],
    main_table: list[dict[str, Any]],
    fair_frontier_table: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    all_exact_rows = [row for row in summary if row["method"] == "EnvelopeAllExact+TI"]
    all_candidates = sum(int(row["candidates"]) for row in all_exact_rows)
    all_exact_calls = sum(int(row["exact_calls"]) for row in all_exact_rows)
    all_exact_ms = sum(float(row["native_exact_backend_ms"]) for row in all_exact_rows)
    all_envelope_ms = sum(float(row["envelope_ms"]) for row in all_exact_rows)
    lines = [
        f"# Scene/Object Conservative Envelope Native TI Wall-time ({metadata['run_name']})",
        "",
        "## Scope",
        "",
        "- Input is real adjacent full-scene mesh frames, not synthetic dense groups and not pre-expanded query CSV replay.",
        "- The native C++ runner partitions connected components as objects, builds swept object AABB envelopes, then builds swept VF/EE primitive AABB candidates inside those object envelopes.",
        "- Exact backend is native Tight-Inclusion `vertexFaceCCD` / `edgeEdgeCCD`.",
        "- `native_exact_backend_ms` times only native exact CCD calls. `envelope_ms` reports conservative envelope/candidate construction separately.",
        "- `EnvelopeAllExact+TI` enumerates every generated envelope candidate. Scheduled rows rank candidates within each scene/object-envelope/kind group, test a bounded top-K proposal set, and conservatively fall back to full exact replay when no hit is certified.",
        "- `FairFrontier*+TI` rows use the same fast native frontier, the same frontier-K, the same proposal top-K, and the same partial-sort/fallback replay. Only the ranking score inside the shared frontier changes.",
        "- `OptimizedFrozenLearnedAnyHit+TI` uses a fast native geometric frontier and then applies the frozen STPF model only inside that frontier before exact replay; high object-count scenes use a fixed low-overhead gate instead of full learned scoring. Its scheduler backend time is `ordering_ms + native_exact_backend_ms`.",
        "- `BestFixedHeuristicOracle+TI` is a retrospective non-deployable oracle that picks the better fixed heuristic per primitive family after evaluation.",
        "- This is CCD detection wall-time over a conservative scene/object envelope; it is not full simulation/contact-solver wall-time and is not Scalable-CCD kernel time.",
        "",
        "## Overall",
        "",
        f"- Scenes: `{len(metadata['scene_pairs'])}`",
        f"- All-exact envelope candidates: `{all_candidates}`",
        f"- All-exact native exact calls: `{all_exact_calls}`",
        f"- All-exact native exact backend wall-time: `{all_exact_ms:.3f} ms`",
        f"- Conservative envelope construction wall-time: `{all_envelope_ms:.3f} ms`",
        f"- Proposal top-K before fallback: `{metadata['parameters']['proposal_top_k']}`",
        f"- Optimized learned frontier-K: `{metadata['parameters']['optimized_frontier_k']}`",
        f"- Optimized learned scan limit per group: `{metadata['parameters']['optimized_scan_limit_per_group']}`",
        f"- Optimized learned random gate object count: `{metadata['parameters']['optimized_random_gate_object_count']}`",
        f"- Frozen STPF checkpoint: `{metadata['stpf_checkpoint']}`",
        "",
        "## Main Scheduler Table",
        "",
        "| Method | Groups | Candidates | Exact calls | Call reduction | Native backend ms | Ordering ms | Scheduler backend ms | TP/TN/FP/FN | Coverage / fallback |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in main_table:
        lines.append(
            "| {method} | {group_count} | {candidates} | {exact_calls} | {call_reduction_vs_all_exact:.6f} | "
            "{native_exact_backend_ms:.3f} | {ordering_ms:.3f} | {scheduler_backend_ms:.3f} | {tp}/{tn}/{fp}/{fn} | "
            "{coverage_rate:.3f} / {fallback_rate:.3f} |".format(**row)
        )
    lines += [
        "",
        "## Fair Frontier Ranking Diagnostics",
        "",
        "These rows isolate the learned-vs-heuristic ranking question under the same native frontier, the same top-K proposal budget, and the same partial-sort/order path. `Positive proposal hits` counts positive groups certified inside the bounded proposal stage before conservative fallback.",
        "",
        "| Method | Positive groups | Positive proposal hits | Proposal hit rate | Positive exact calls | Positive proposal calls | Positive fallback calls | Mean exact calls to first positive | Total exact calls | Scheduler backend ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in fair_frontier_table:
        lines.append(
            "| {method} | {positive_groups} | {positive_proposal_hits} | {positive_proposal_hit_rate:.6f} | "
            "{positive_exact_calls} | {positive_proposal_exact_calls} | {positive_fallback_exact_calls} | "
            "{positive_first_hit_mean_exact_calls:.3f} | {exact_calls} | {scheduler_backend_ms:.3f} |".format(**row)
        )
    lines += [
        "",
        "## Summary",
        "",
        "| Method | Kind | Groups | Candidates | Exact calls | Positive exact calls | Positive proposal hits | Native hits | TP/TN/FP/FN | Fallback rate | Call reduction | Envelope ms | Ordering ms | Native exact backend ms | Scheduler backend ms | Total wall ms | Calls/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {method} | {kind} | {group_count} | {candidates} | {exact_calls} | "
            "{positive_exact_calls} | {positive_proposal_hits} | {native_hits} | "
            "{tp}/{tn}/{fp}/{fn} | {fallback_rate:.6f} | {call_reduction_vs_envelope_all_exact:.6f} | "
            "{envelope_ms:.3f} | {ordering_ms:.3f} | {native_exact_backend_ms:.3f} | "
            "{scheduler_backend_ms:.3f} | {total_wall_ms:.3f} | {exact_calls_per_second:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Per-scene Rows",
            "",
            "| Scene | Kind | Method | Objects | Object envelopes | Groups | Candidates | Exact calls | TP/TN/FP/FN | Fallback rate | Capped | Envelope ms | Ordering ms | Native exact backend ms | Total wall ms |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {scene} | {kind} | {method} | {objects} | {object_envelopes} | {group_count} | "
            "{candidate_count} | {exact_calls} | {tp}/{tn}/{fp}/{fn} | {fallback_rate:.6f} | {candidate_capped} | "
            "{envelope_ms:.3f} | {ordering_ms:.3f} | {native_exact_backend_ms:.3f} | {total_wall_ms:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Native runner: `{metadata['native_runner']}`",
            f"- C++ source: `{metadata['cpp_source']}`",
            f"- Raw JSONL dir: `{metadata['raw_jsonl_dir']}`",
            f"- Summary CSV: `{metadata['summary_csv']}`",
            f"- Main scheduler CSV: `{metadata['main_scheduler_csv']}`",
            f"- Fair frontier ranking CSV: `{metadata['fair_frontier_csv']}`",
            f"- Row CSV: `{metadata['row_csv']}`",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "conda activate cudadev",
            f"python src/tools/run_scene_object_envelope_native_ti_walltime.py --run-name {metadata['run_name']} --proposal-top-k {metadata['parameters']['proposal_top_k']} --optimized-frontier-k {metadata['parameters']['optimized_frontier_k']} --optimized-scan-limit-per-group {metadata['parameters']['optimized_scan_limit_per_group']} --optimized-random-gate-object-count {metadata['parameters']['optimized_random_gate_object_count']}",
            "```",
            "",
        ]
    )
    (out_dir / f"{metadata['run_name']}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--scenes", default=",".join(SCENE_ORDER))
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--thickness", type=float, default=0.0)
    parser.add_argument("--ms", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--max-itr", type=int, default=1000000)
    parser.add_argument("--max-vf-candidates", type=int, default=0)
    parser.add_argument("--max-ee-candidates", type=int, default=0)
    parser.add_argument("--proposal-top-k", type=int, default=4096)
    parser.add_argument("--optimized-frontier-k", type=int, default=1024)
    parser.add_argument("--optimized-scan-limit-per-group", type=int, default=1048576)
    parser.add_argument("--optimized-random-gate-object-count", type=int, default=128)
    parser.add_argument("--feature-jsonl-dir", type=Path, default=None)
    parser.add_argument("--feature-negative-stride", type=int, default=0)
    parser.add_argument("--feature-export-only", action="store_true")
    parser.add_argument("--exclude-self-object-pairs", action="store_true")
    parser.add_argument("--stpf-checkpoint", type=Path, default=DEFAULT_STPF_CHECKPOINT)
    parser.add_argument("--stpf-weights", type=Path, default=None)
    args = parser.parse_args()

    out_dir = ROOT / "benchmark" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stpf_weights is None:
        args.stpf_weights = out_dir / "frozen_tiny_stpf_weights.txt"
    if args.proposal_top_k < 0:
        raise ValueError("--proposal-top-k must be non-negative")
    if args.optimized_frontier_k <= 0:
        raise ValueError("--optimized-frontier-k must be positive")
    if args.optimized_scan_limit_per_group < 0:
        raise ValueError("--optimized-scan-limit-per-group must be non-negative")
    if args.optimized_random_gate_object_count < 0:
        raise ValueError("--optimized-random-gate-object-count must be non-negative")
    if args.feature_negative_stride < 0:
        raise ValueError("--feature-negative-stride must be non-negative")
    export_tiny_stpf_weights(args.stpf_checkpoint, args.stpf_weights)
    build_native_runner(force=args.force_build)
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    pairs = discover_frame_pairs(args.source_root, scenes)

    started = dt.datetime.now().isoformat(timespec="seconds")
    all_rows: list[dict[str, Any]] = []
    run_outputs = []
    for pair in pairs:
        jsonl = run_scene(pair, args, out_dir)
        run_outputs.append({"scene": pair["scene"], "jsonl": rel(jsonl), "frame0": rel(pair["frame0"]), "frame1": rel(pair["frame1"])})
        all_rows.extend(read_jsonl(jsonl))

    summary = aggregate(all_rows)
    main_table = build_main_scheduler_table(summary)
    fair_frontier_table = build_fair_frontier_ranking_table(main_table)
    row_csv = out_dir / f"{args.run_name}_rows.csv"
    summary_csv = out_dir / f"{args.run_name}_summary.csv"
    main_scheduler_csv = out_dir / f"{args.run_name}_main_scheduler_table.csv"
    fair_frontier_csv = out_dir / f"{args.run_name}_fair_frontier_ranking_table.csv"
    write_csv(row_csv, all_rows)
    write_csv(summary_csv, summary)
    write_csv(main_scheduler_csv, main_table)
    write_csv(fair_frontier_csv, fair_frontier_table)
    metadata = {
        "run_name": args.run_name,
        "started_at": started,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_root": rel(args.source_root),
        "stpf_checkpoint": rel(args.stpf_checkpoint),
        "stpf_weights": rel(args.stpf_weights),
        "native_runner": rel(BUILD_EXE),
        "cpp_source": rel(CPP_SOURCE),
        "raw_jsonl_dir": rel(out_dir / "raw_jsonl"),
        "row_csv": rel(row_csv),
        "summary_csv": rel(summary_csv),
        "main_scheduler_csv": rel(main_scheduler_csv),
        "fair_frontier_csv": rel(fair_frontier_csv),
        "scene_pairs": run_outputs,
        "parameters": {
            "thickness": args.thickness,
            "ms": args.ms,
            "tolerance": args.tolerance,
            "t_max": args.t_max,
            "max_itr": args.max_itr,
            "max_vf_candidates": args.max_vf_candidates,
            "max_ee_candidates": args.max_ee_candidates,
            "proposal_top_k": args.proposal_top_k,
            "optimized_frontier_k": args.optimized_frontier_k,
            "optimized_scan_limit_per_group": args.optimized_scan_limit_per_group,
            "optimized_random_gate_object_count": args.optimized_random_gate_object_count,
            "feature_jsonl_dir": None if args.feature_jsonl_dir is None else args.feature_jsonl_dir.as_posix(),
            "feature_negative_stride": args.feature_negative_stride,
            "feature_export_only": bool(args.feature_export_only),
            "exclude_self_object_pairs": bool(args.exclude_self_object_pairs),
        },
    }
    (out_dir / f"{args.run_name}.json").write_text(
        json.dumps({"metadata": metadata, "main_table": main_table, "summary": summary, "rows": all_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(out_dir, summary, main_table, fair_frontier_table, all_rows, metadata)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
