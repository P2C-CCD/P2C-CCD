#!/usr/bin/env python3
"""Train and replay a stronger native scene/object-envelope CCD benchmark.

This driver keeps the evaluated candidate set native: adjacent full-scene PLY
frames are converted to swept object-pair envelopes and VF/EE primitive
candidates, Tight-Inclusion supplies exact labels, and the learned model only
orders proposals before the correctness-preserving exact fallback.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from torch import nn


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / "src"
WRAPPER = ROOT / "tools" / "run_scene_object_envelope_native_ti_walltime.py"
BENCH_ROOT = ROOT / "benchmark"
OUTPUT_ROOT = ROOT / "outputs" / "stpf_training"
RUN_NAME = "scene_object_envelope_strong_native_run_id"
DEFAULT_TRAIN_SCENES = "armadillo-rollers,cloth-ball,cloth-funnel,n-body-simulation"
DEFAULT_EVAL_SCENES = "puffer-ball,rod-twist"


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except Exception:
        return path.as_posix()


class TinyEnvelopeRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def run_command(cmd: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
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
    completed = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True, check=False)
    log_path.write_text(
        "COMMAND:\n"
        + " ".join(cmd)
        + "\n\nSTDOUT:\n"
        + completed.stdout
        + "\n\nSTDERR:\n"
        + completed.stderr,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}; see {log_path}")


def call_native_wrapper(args: argparse.Namespace, *, run_name: str, scenes: str, export_only: bool, checkpoint: Path | None) -> Path:
    source_root = args.source_root if args.source_root.is_absolute() else WORKSPACE / args.source_root
    cmd = [
        sys.executable,
        str(WRAPPER),
        "--run-name",
        run_name,
        "--scenes",
        scenes,
        "--source-root",
        str(source_root),
        "--proposal-top-k",
        str(args.proposal_top_k),
        "--optimized-frontier-k",
        str(args.optimized_frontier_k),
        "--optimized-scan-limit-per-group",
        str(args.optimized_scan_limit_per_group),
        "--optimized-random-gate-object-count",
        "0",
    ]
    if args.force_build:
        cmd.append("--force-build")
    if args.exclude_self_object_pairs:
        cmd.append("--exclude-self-object-pairs")
    if checkpoint is not None:
        cmd.extend(["--stpf-checkpoint", str(checkpoint)])
    if export_only:
        cmd.extend(
            [
                "--feature-jsonl-dir",
                "raw_features",
                "--feature-negative-stride",
                str(args.feature_negative_stride),
                "--feature-export-only",
            ]
        )
    run_dir = BENCH_ROOT / run_name
    run_command(cmd, cwd=WORKSPACE, log_path=run_dir / "driver_logs" / ("export.log" if export_only else "eval.log"))
    return run_dir


def load_feature_rows(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    features: list[list[float]] = []
    labels: list[int] = []
    per_scene: dict[str, dict[str, int]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                scene = str(row["scene"])
                stats = per_scene.setdefault(scene, {"rows": 0, "positives": 0, "negatives": 0})
                label = int(row["label"])
                features.append([float(value) for value in row["features"]])
                labels.append(label)
                stats["rows"] += 1
                if label:
                    stats["positives"] += 1
                else:
                    stats["negatives"] += 1
    if not features:
        raise RuntimeError("no feature rows were exported")
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    metadata = {
        "feature_paths": [rel(path) for path in paths],
        "row_count": int(x.shape[0]),
        "positive_count": int(y.sum()),
        "negative_count": int(x.shape[0] - int(y.sum())),
        "per_scene": per_scene,
    }
    return x, y, metadata


def train_ranker(args: argparse.Namespace, export_dir: Path, run_name: str) -> tuple[Path, dict[str, Any]]:
    feature_paths = sorted((export_dir / "raw_features").glob("*.jsonl"))
    x, y, metadata = load_feature_rows(feature_paths)
    mean = x.mean(axis=0).astype(np.float64)
    std = np.maximum(x.std(axis=0), 1.0e-6).astype(np.float64)
    x_norm = ((x - mean.astype(np.float32)) / std.astype(np.float32)).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(x_norm.shape[0])
    validation_size = max(1, int(0.1 * indices.size))
    val_idx = indices[:validation_size]
    train_idx = indices[validation_size:]

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = TinyEnvelopeRanker().to(device)
    positives = max(1.0, float(y[train_idx].sum()))
    negatives = max(1.0, float(train_idx.size - y[train_idx].sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)

    x_tensor = torch.from_numpy(x_norm)
    y_tensor = torch.from_numpy(y)
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_order = rng.permutation(train_idx)
        losses: list[float] = []
        for start in range(0, epoch_order.size, args.batch_size):
            batch_idx = epoch_order[start : start + args.batch_size]
            xb = x_tensor[batch_idx].to(device, non_blocking=True)
            yb = y_tensor[batch_idx].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate_ranker(model, x_tensor, y_tensor, train_idx[: min(train_idx.size, 200000)], val_idx, device, args.batch_size)
        metrics.update({"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0})
        history.append(metrics)

    out_dir = OUTPUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "model_state.pt"
    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "training_protocol": "native_scene_object_envelope_exact_label_bce",
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "train_scenes": args.train_scenes,
            "eval_scenes": args.eval_scenes,
        },
        checkpoint,
    )
    summary = {
        "checkpoint": rel(checkpoint),
        "device": str(device),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "feature_export": metadata,
        "history": history,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_training_report(out_dir / "training_summary.md", summary)
    return checkpoint, summary


@torch.no_grad()
def evaluate_ranker(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    train_probe_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    def split_metrics(indices: np.ndarray) -> dict[str, float]:
        model.eval()
        scores: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for start in range(0, indices.size, batch_size):
            batch_idx = indices[start : start + batch_size]
            scores.append(model(x[batch_idx].to(device)).detach().cpu().numpy())
            labels.append(y[batch_idx].detach().cpu().numpy())
        score = np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32)
        label = np.concatenate(labels) if labels else np.zeros((0,), dtype=np.float32)
        if score.size == 0:
            return {"top1_positive_rate": 0.0, "top01_positive_rate": 0.0, "mean_positive_score": 0.0, "mean_negative_score": 0.0}
        order = np.argsort(-score)
        top1_count = max(1, int(0.01 * order.size))
        top01_count = max(1, int(0.001 * order.size))
        positives = label >= 0.5
        negatives = ~positives
        return {
            "top1_positive_rate": float(label[order[:top1_count]].mean()),
            "top01_positive_rate": float(label[order[:top01_count]].mean()),
            "mean_positive_score": float(score[positives].mean()) if positives.any() else 0.0,
            "mean_negative_score": float(score[negatives].mean()) if negatives.any() else 0.0,
        }

    out = {}
    for prefix, indices in (("train_probe", train_probe_idx), ("validation", val_idx)):
        for key, value in split_metrics(indices).items():
            out[f"{prefix}_{key}"] = value
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_comparison(eval_dir: Path, run_name: str, checkpoint: Path, training_summary: dict[str, Any]) -> dict[str, Any]:
    main_csv = eval_dir / f"{run_name}_main_scheduler_table.csv"
    rows_csv = eval_dir / f"{run_name}_rows.csv"
    rows = read_csv(rows_csv)
    main_rows = read_csv(main_csv)
    by_method = {row["method"]: row for row in main_rows}

    all_exact_method = "EnvelopeAllExact+TI" if "EnvelopeAllExact+TI" in by_method else "AllExact+TI"
    methods = [
        all_exact_method,
        "FrozenLearnedAnyHit+TI",
        "LearnedResidualAnyHit+TI",
        "ProximityHeuristicAnyHit+TI",
        "MotionHeuristicAnyHit+TI",
        "RandomAnyHit+TI",
        "FairFrontierLearnedAnyHit+TI",
        "FairFrontierLearnedResidualAnyHit+TI",
        "FairFrontierProximityAnyHit+TI",
        "FairFrontierMotionAnyHit+TI",
        "FairFrontierRandomAnyHit+TI",
    ]
    dense_calls = float(by_method[all_exact_method]["exact_calls"])
    table: list[dict[str, Any]] = []
    for method in methods:
        if method not in by_method:
            continue
        row = by_method[method]
        exact_calls = float(row["exact_calls"])
        table.append(
            {
                "method": method,
                "exact_calls": int(exact_calls),
                "exact_call_reduction_vs_dense": 1.0 - exact_calls / max(1.0, dense_calls),
                "positive_proposal_hits": int(float(row.get("positive_proposal_hits", 0))),
                "positive_groups": int(float(row.get("positive_groups", 0))),
                "fn": int(float(row.get("fn", 0))),
                "native_exact_backend_ms": float(row["native_exact_backend_ms"]),
                "total_wall_ms": float(row.get("total_wall_ms") or row.get("scheduler_backend_ms") or row["native_exact_backend_ms"]),
            }
        )
    write_csv(eval_dir / f"{run_name}_strong_native_comparison.csv", table)

    make_plots(eval_dir, run_name, table)
    summary = {
        "run_name": run_name,
        "eval_dir": rel(eval_dir),
        "checkpoint": rel(checkpoint),
        "train_scenes": training_summary.get("train_scenes"),
        "eval_scenes": training_summary.get("eval_scenes"),
        "comparison_table": rel(eval_dir / f"{run_name}_strong_native_comparison.csv"),
        "plot_png": rel(eval_dir / f"{run_name}_exact_call_comparison.png"),
        "plot_pdf": rel(eval_dir / f"{run_name}_exact_call_comparison.pdf"),
        "rows": table,
        "native_rows_csv": rel(rows_csv),
        "native_main_csv": rel(main_csv),
        "scene_rows": rows,
    }
    (eval_dir / f"{run_name}_strong_native_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_strong_report(eval_dir / f"{run_name}_strong_native_report.md", summary, training_summary)
    return summary


def make_plots(eval_dir: Path, run_name: str, table: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (eval_dir / f"{run_name}_plot_error.txt").write_text(repr(exc), encoding="utf-8")
        return
    plot_rows = [row for row in table if row["method"] != "EnvelopeAllExact+TI"]
    labels = [row["method"].replace("+TI", "").replace("AnyHit", "") for row in plot_rows]
    calls = np.asarray([row["exact_calls"] for row in plot_rows], dtype=np.float64)
    dense = next(row["exact_calls"] for row in table if row["method"] in {"EnvelopeAllExact+TI", "AllExact+TI"})
    plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=220)
    colors = ["#1f77b4" if "Learned" in label else "#6b7280" for label in labels]
    ax.bar(np.arange(len(labels)), calls / 1.0e6, color=colors, width=0.68)
    ax.axhline(float(dense) / 1.0e6, color="#b91c1c", linewidth=1.4, linestyle="--", label="All exact")
    if dense / max(1.0, float(np.min(calls))) > 100.0:
        ax.set_yscale("log")
    ax.set_ylabel("Native exact calls (M)")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(eval_dir / f"{run_name}_exact_call_comparison.png")
    fig.savefig(eval_dir / f"{run_name}_exact_call_comparison.pdf")
    plt.close(fig)


def write_training_report(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["history"]
    last = rows[-1] if rows else {}
    lines = [
        "# Native scene/object envelope STPF training",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Feature rows: `{summary['feature_export']['row_count']}`",
        f"- Positives: `{summary['feature_export']['positive_count']}`",
        f"- Negatives: `{summary['feature_export']['negative_count']}`",
        f"- Final validation top-0.1% positive rate: `{last.get('validation_top01_positive_rate', 0.0):.6f}`",
        f"- Final validation top-1% positive rate: `{last.get('validation_top1_positive_rate', 0.0):.6f}`",
        "",
        "## Per-scene exported labels",
        "",
        "| Scene | Rows | Positives | Negatives |",
        "|---|---:|---:|---:|",
    ]
    for scene, stats in sorted(summary["feature_export"]["per_scene"].items()):
        lines.append(f"| `{scene}` | {stats['rows']} | {stats['positives']} | {stats['negatives']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_strong_report(path: Path, summary: dict[str, Any], training_summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    learned = next((row for row in rows if row["method"] == "LearnedResidualAnyHit+TI"), None)
    learned_label = "LearnedResidualAnyHit+TI"
    if learned is None:
        learned = next((row for row in rows if row["method"] == "FrozenLearnedAnyHit+TI"), None)
        learned_label = "FrozenLearnedAnyHit+TI"
    prox = next((row for row in rows if row["method"] == "ProximityHeuristicAnyHit+TI"), None)
    dense = next((row for row in rows if row["method"] in {"EnvelopeAllExact+TI", "AllExact+TI"}), None)
    lines = [
        "# Strong native scene/object envelope benchmark",
        "",
        "This benchmark is a native full-scene/object-envelope audit. It exports candidate-level labels from the same Tight-Inclusion exact kernel, trains a frozen 32-feature tiny ranker on training scenes, and replays held-out adjacent scene frames with exact fallback. The learned model only orders proposals; final correctness remains exact/certified by fallback.",
        "",
        "## Protocol",
        "",
        f"- Train scenes: `{training_summary.get('train_scenes')}`",
        f"- Eval scenes: `{training_summary.get('eval_scenes')}`",
        f"- Frozen checkpoint: `{summary['checkpoint']}`",
        f"- Native row CSV: `{summary['native_rows_csv']}`",
        f"- Comparison CSV: `{summary['comparison_table']}`",
        "",
        "## Aggregate comparison",
        "",
        "| Method | Exact calls | Reduction vs dense | Positive proposal hits | Positive groups | FN | Native exact ms | Total wall ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['method']}` | {row['exact_calls']:,} | {100.0 * row['exact_call_reduction_vs_dense']:.3f}% | "
            f"{row['positive_proposal_hits']} | {row['positive_groups']} | {row['fn']} | "
            f"{row['native_exact_backend_ms']:.3f} | {row['total_wall_ms']:.3f} |"
        )
    if learned and prox and dense:
        lines.extend(
            [
                "",
                "## Main reading",
                "",
                f"- `{learned_label}` exact calls vs dense: `{learned['exact_calls']:,}` / `{dense['exact_calls']:,}`.",
                f"- `{learned_label}` exact calls vs proximity heuristic: `{learned['exact_calls']:,}` vs `{prox['exact_calls']:,}`.",
                f"- `{learned_label}` FN: `{learned['fn']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- `{summary['plot_png']}`",
            f"- `{summary['plot_pdf']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--source-root", type=Path, default=ROOT / "baseline" / "Sample-Scalable-CCD-Data")
    parser.add_argument("--train-scenes", default=DEFAULT_TRAIN_SCENES)
    parser.add_argument("--eval-scenes", default=DEFAULT_EVAL_SCENES)
    parser.add_argument("--feature-negative-stride", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=fixed_seed)
    parser.add_argument("--proposal-top-k", type=int, default=32)
    parser.add_argument("--optimized-frontier-k", type=int, default=128)
    parser.add_argument("--optimized-scan-limit-per-group", type=int, default=4096)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--exclude-self-object-pairs", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    started = dt.datetime.now().isoformat(timespec="seconds")
    export_name = f"{args.run_name}_feature_export"
    export_dir = BENCH_ROOT / export_name
    if not args.skip_export:
        export_dir = call_native_wrapper(args, run_name=export_name, scenes=args.train_scenes, export_only=True, checkpoint=args.checkpoint)
    checkpoint = args.checkpoint
    training_summary: dict[str, Any] = {"train_scenes": args.train_scenes, "eval_scenes": args.eval_scenes}
    if not args.skip_train:
        checkpoint, training_summary = train_ranker(args, export_dir, args.run_name)
        training_summary["train_scenes"] = args.train_scenes
        training_summary["eval_scenes"] = args.eval_scenes
    if checkpoint is None:
        raise ValueError("a checkpoint is required when --skip-train is used")
    eval_dir = call_native_wrapper(args, run_name=args.run_name, scenes=args.eval_scenes, export_only=False, checkpoint=checkpoint)
    summary = build_comparison(eval_dir, args.run_name, checkpoint, training_summary)
    summary["started_at"] = started
    summary["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    print(eval_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
