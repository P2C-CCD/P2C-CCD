from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from native_ti_four_source_dense_group_benchmark import (
    Candidate,
    P_DRIVE_ROOT,
    ROOT,
    TI_RUNNER,
    TinySTPF,
    SourceSpec,
    _write_json,
    discover_sources,
    feature_from_rows,
    frame_from_triangle,
    load_mesh_basis,
    write_query_rows,
)


RUN_NAME = "native_ti_heldout_dense_group_run_id"
BENCH_DIR = ROOT / "src" / "benchmark" / RUN_NAME
OUTPUT_DIR = ROOT / "src" / "outputs" / "stpf_training" / RUN_NAME
DATASET_ROOT = BENCH_DIR / "ti_csv_dataset"


def configure_run_name(run_name: str) -> None:
    global RUN_NAME, BENCH_DIR, OUTPUT_DIR, DATASET_ROOT
    RUN_NAME = run_name
    BENCH_DIR = ROOT / "src" / "benchmark" / RUN_NAME
    OUTPUT_DIR = ROOT / "src" / "outputs" / "stpf_training" / RUN_NAME
    DATASET_ROOT = BENCH_DIR / "ti_csv_dataset"


def vf_query_heldout(tri: np.ndarray, positive: bool, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    center, u, v, n = frame_from_triangle(tri)
    radius = float(np.mean(np.linalg.norm(tri - center, axis=1)))
    radius = max(radius, 0.02)
    local_tri = np.stack(
        [
            center + u * radius * 0.95 + v * radius * 0.05,
            center - u * radius * 0.45 + v * radius * 0.85,
            center - u * radius * 0.55 - v * radius * 0.75,
        ]
    )
    lateral = (rng.random() - 0.5) * 0.08 * radius
    if positive:
        p0 = center + n * (0.24 * radius) + u * lateral
        p1 = center - n * (0.24 * radius) + u * lateral
    else:
        # Hard negative: close in space but moving parallel to the face, with
        # clearance far above TI tolerance. This avoids the trivial far-away
        # negative split used by the easy mixed benchmark.
        clearance = (0.085 + 0.035 * rng.random()) * radius
        p0 = center + n * clearance + u * lateral - v * (0.18 * radius)
        p1 = center + n * clearance + u * (lateral + 0.12 * radius) + v * (0.18 * radius)
    rows = np.vstack([p0, local_tri[0], local_tri[1], local_tri[2], p1, local_tri[0], local_tri[1], local_tri[2]])
    return rows, local_tri, radius


def ee_query_heldout(edge: np.ndarray, tri: np.ndarray, positive: bool, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    center, u, v, n = frame_from_triangle(tri)
    length = float(np.linalg.norm(edge[1] - edge[0]))
    length = max(length, 0.025)
    half = 0.55 * length
    jitter = (rng.random(3) - 0.5) * 0.02 * length
    a0 = center - u * half + jitter
    a1 = center + u * half + jitter
    if positive:
        b0_t0 = center - v * half + n * (0.20 * length)
        b1_t0 = center + v * half + n * (0.20 * length)
        b0_t1 = center - v * half - n * (0.20 * length)
        b1_t1 = center + v * half - n * (0.20 * length)
    else:
        clearance = (0.11 + 0.05 * rng.random()) * length
        drift = (rng.random() - 0.5) * 0.10 * length
        b0_t0 = center - v * half + n * clearance + u * drift
        b1_t0 = center + v * half + n * clearance + u * drift
        b0_t1 = b0_t0 + u * (0.08 * length)
        b1_t1 = b1_t0 + u * (0.08 * length)
    rows = np.vstack([a0, a1, b0_t0, b1_t0, a0, a1, b0_t1, b1_t1])
    return rows, length


def build_source_csv_and_candidates(
    spec: SourceSpec,
    start_group_id: int,
    group_count: int,
    group_size: int,
    seed: int,
    negative_group_ratio: float,
) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    basis = load_mesh_basis(spec.path, seed)
    vf_rel = f"{spec.name}/vertex-face/vertex-face-0000.csv"
    ee_rel = f"{spec.name}/edge-edge/edge-edge-0000.csv"
    vf_path = DATASET_ROOT / vf_rel
    ee_path = DATASET_ROOT / ee_rel
    vf_path.parent.mkdir(parents=True, exist_ok=True)
    ee_path.parent.mkdir(parents=True, exist_ok=True)
    triangles = basis["triangles"]
    edges = basis["edges"]
    candidates: list[Candidate] = []
    vf_index = 0
    ee_index = 0
    with vf_path.open("w", encoding="utf-8", newline="") as vf, ee_path.open("w", encoding="utf-8", newline="") as ee:
        for local_group in range(group_count):
            group_id = start_group_id + local_group
            group_has_positive = bool(rng.random() >= negative_group_ratio)
            positive_slot = int(rng.integers(0, group_size)) if group_has_positive else -1
            for slot in range(group_size):
                positive = slot == positive_slot
                kind = "vertex-face" if (slot + local_group) % 2 == 0 else "edge-edge"
                tri = triangles[int(rng.integers(0, len(triangles)))]
                if kind == "vertex-face":
                    rows, _, _ = vf_query_heldout(tri, positive, rng)
                    query_index = vf_index
                    vf_index += 1
                    write_query_rows(vf, rows, int(positive))
                    rel = vf_rel
                else:
                    edge = edges[int(rng.integers(0, len(edges)))]
                    rows, _ = ee_query_heldout(edge, tri, positive, rng)
                    query_index = ee_index
                    ee_index += 1
                    write_query_rows(ee, rows, int(positive))
                    rel = ee_rel
                candidates.append(
                    Candidate(
                        group_id=group_id,
                        source=spec.name,
                        kind=kind,
                        csv_path=rel,
                        query_index=query_index,
                        truth=int(positive),
                        feature=feature_from_rows(kind, rows),
                    )
                )
    return candidates


def train_frozen_checkpoint(
    split_name: str,
    train_candidates: list[Candidate],
    score_candidates: list[Candidate],
    *,
    seed: int,
    device: str,
    epochs: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_features = np.stack([c.feature for c in train_candidates]).astype(np.float32)
    train_labels = np.asarray([c.truth for c in train_candidates], dtype=np.float32)
    score_features = np.stack([c.feature for c in score_candidates]).astype(np.float32)
    order = rng.permutation(len(train_candidates))
    val_n = max(1, int(0.2 * len(order)))
    val_idx = order[:val_n]
    fit_idx = order[val_n:]
    mean = train_features[fit_idx].mean(axis=0, keepdims=True)
    std = train_features[fit_idx].std(axis=0, keepdims=True) + 1.0e-6
    x_fit = torch.from_numpy((train_features[fit_idx] - mean) / std).to(device)
    y_fit = torch.from_numpy(train_labels[fit_idx]).to(device)
    x_val = torch.from_numpy((train_features[val_idx] - mean) / std).to(device)
    y_val = torch.from_numpy(train_labels[val_idx]).to(device)
    x_score = torch.from_numpy((score_features - mean) / std).to(device)
    model = TinySTPF(train_features.shape[1]).to(device)
    pos_weight = torch.tensor(
        [(len(y_fit) - float(y_fit.sum())) / max(float(y_fit.sum()), 1.0)],
        device=device,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    batch = min(16384, len(fit_idx))
    losses: list[float] = []
    for _ in range(epochs):
        perm = torch.randperm(len(fit_idx), device=device)
        epoch_loss = 0.0
        for start in range(0, len(fit_idx), batch):
            idx = perm[start : start + batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_fit[idx]), y_fit[idx])
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu()) * len(idx)
        losses.append(epoch_loss / max(len(fit_idx), 1))
    model.eval()
    with torch.no_grad():
        val_scores = torch.sigmoid(model(x_val)).detach().cpu().numpy()
        score_values = torch.sigmoid(model(x_score)).detach().cpu().numpy()
    for candidate, score in zip(score_candidates, score_values):
        candidate.learned_score = float(score)
    val_pred = val_scores >= 0.5
    val_truth = y_val.detach().cpu().numpy() > 0.5
    recall = float(np.sum(val_pred & val_truth) / max(np.sum(val_truth), 1))
    precision = float(np.sum(val_pred & val_truth) / max(np.sum(val_pred), 1))
    split_dir = OUTPUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = split_dir / "model_state.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_mean": mean.astype(np.float32),
            "feature_std": std.astype(np.float32),
            "run_name": RUN_NAME,
            "split_name": split_name,
        },
        checkpoint,
    )
    return {
        "split": split_name,
        "device": device,
        "epochs": epochs,
        "train_rows": int(len(fit_idx)),
        "validation_rows": int(len(val_idx)),
        "train_positive_ratio": float(train_labels.mean()),
        "validation_recall_at_0_5": recall,
        "validation_precision_at_0_5": precision,
        "losses": losses,
        "checkpoint": checkpoint.as_posix(),
    }


def assign_random_scores(candidates: list[Candidate], seed: int) -> None:
    rng = np.random.default_rng(seed)
    for candidate, score in zip(candidates, rng.random(len(candidates))):
        candidate.random_score = float(score)


def heuristic_scores(candidates: list[Candidate]) -> dict[str, dict[int, float]]:
    features = np.stack([c.feature for c in candidates]).astype(np.float64)
    energy = np.linalg.norm(features[:, :16], axis=1)
    scores = {
        "SingleHeuristicProximity": features[:, 25],
        "HeuristicSmallGap": -features[:, 11],
        "HeuristicMotionHigh": features[:, 8],
        "HeuristicFeatureEnergy": energy,
        "HeuristicExtentLow": -features[:, 5],
    }
    return {
        name: {id(candidate): float(score) for candidate, score in zip(candidates, values)}
        for name, values in scores.items()
    }


def write_schedule(path: Path, candidates: list[Candidate], score_by_candidate_id: dict[int, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group_id", "case", "kind", "csv_path", "query_index", "score"])
        for c in candidates:
            writer.writerow(
                [
                    c.group_id,
                    c.source,
                    c.kind,
                    c.csv_path,
                    c.query_index,
                    f"{score_by_candidate[id(c)]:.12g}",
                ]
            )


def cxx_path(path: Path) -> str:
    try:
        return str(P_DRIVE_ROOT / path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def run_three_method(label: str, learned_schedule: Path, random_schedule: Path) -> dict[str, Any]:
    output_json = BENCH_DIR / f"{label}_three_method.json"
    output_md = BENCH_DIR / f"{label}_three_method.md"
    output_csv = BENCH_DIR / f"{label}_three_method.csv"
    cmd = [
        cxx_path(TI_RUNNER),
        "--dataset-root",
        cxx_path(DATASET_ROOT),
        "--learned-schedule",
        cxx_path(learned_schedule),
        "--random-schedule",
        cxx_path(random_schedule),
        "--output-json",
        cxx_path(output_json),
        "--output-md",
        cxx_path(output_md),
        "--output-csv",
        cxx_path(output_csv),
    ]
    begin = time.perf_counter()
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    elapsed_ms = (time.perf_counter() - begin) * 1000.0
    if completed.returncode != 0:
        raise RuntimeError(f"TI runner failed for {label}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    payload["runner_elapsed_ms"] = elapsed_ms
    payload["stdout_tail"] = completed.stdout[-2000:]
    payload["stderr_tail"] = completed.stderr[-2000:]
    return payload


def method_by_name(payload: dict[str, Any], method: str) -> dict[str, Any]:
    for row in payload["methods"]:
        if row["method"] == method:
            return row
    raise KeyError(method)


def renamed(row: dict[str, Any], method: str) -> dict[str, Any]:
    out = dict(row)
    out["method"] = method
    return out


def evaluate_split(
    split_name: str,
    train_candidates: list[Candidate],
    test_candidates: list[Candidate],
    *,
    seed: int,
    device: str,
    epochs: int,
) -> dict[str, Any]:
    training = train_frozen_checkpoint(
        split_name,
        train_candidates,
        test_candidates,
        seed=seed,
        device=device,
        epochs=epochs,
    )
    assign_random_scores(test_candidates, seed + 7919)
    split_dir = BENCH_DIR / "schedules" / split_name
    learned_scores = {id(c): c.learned_score for c in test_candidates}
    random_scores = {id(c): c.random_score for c in test_candidates}
    heuristic = heuristic_scores(test_candidates)
    learned_schedule = split_dir / "learned.csv"
    random_schedule = split_dir / "random.csv"
    write_schedule(learned_schedule, test_candidates, learned_scores)
    write_schedule(random_schedule, test_candidates, random_scores)
    learned_payload = run_three_method(f"{split_name}_learned_random", learned_schedule, random_schedule)

    rows: list[dict[str, Any]] = [
        renamed(method_by_name(learned_payload, "NoProposal+TI"), "AllExact+TI"),
        renamed(method_by_name(learned_payload, "Random+TI"), "Random+TI"),
        renamed(method_by_name(learned_payload, "RTSTPFExact+TI"), "FrozenLearned+TI"),
    ]

    single_name = "SingleHeuristicProximity"
    single_schedule = split_dir / f"{single_name}.csv"
    write_schedule(single_schedule, test_candidates, heuristic[single_name])
    single_payload = run_three_method(f"{split_name}_{single_name}", single_schedule, random_schedule)
    rows.append(renamed(method_by_name(single_payload, "RTSTPFExact+TI"), f"{single_name}+TI"))

    heuristic_rows: list[dict[str, Any]] = []
    for name, scores in heuristic.items():
        path = split_dir / f"{name}.csv"
        write_schedule(path, test_candidates, scores)
        payload = run_three_method(f"{split_name}_{name}", path, random_schedule)
        heuristic_rows.append(renamed(method_by_name(payload, "RTSTPFExact+TI"), f"{name}+TI"))
    best = min(heuristic_rows, key=lambda r: (int(r["exact_calls"]), float(r["wall_ms"])))
    best = renamed(best, f"BestFixedHeuristicOracle+TI ({best['method'].replace('+TI', '')})")
    rows.append(best)

    groups = sorted({c.group_id for c in test_candidates})
    positive_groups = sorted({c.group_id for c in test_candidates if c.truth})
    return {
        "split": split_name,
        "train_candidate_count": len(train_candidates),
        "test_candidate_count": len(test_candidates),
        "test_group_count": len(groups),
        "test_positive_group_count": len(positive_groups),
        "test_negative_group_count": len(groups) - len(positive_groups),
        "training": training,
        "schedules": {
            "learned": learned_schedule.as_posix(),
            "random": random_schedule.as_posix(),
            "single_heuristic": single_schedule.as_posix(),
        },
        "methods": rows,
        "all_heuristics": heuristic_rows,
        "raw_payloads": {
            "learned_random": learned_payload,
        },
    }


def format_method_row(row: dict[str, Any]) -> str:
    reduction = 100.0 * float(row["exact_call_reduction"])
    return (
        f"| {row['method']} | {int(row['group_count'])} | {int(row['candidate_count'])} | "
        f"{int(row['positive_group_count'])} | {int(row['exact_calls'])} | {reduction:.3f}% | "
        f"{int(row['tp'])}/{int(row['tn'])}/{int(row['fp'])}/{int(row['fn'])} | "
        f"{float(row['first_positive_rank_mean']):.3f} | {float(row['exact_ms']):.3f} | {float(row['wall_ms']):.3f} |"
    )


def write_report(sources: list[SourceSpec], splits: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines: list[str] = [
        f"# Frozen-checkpoint Held-out Native TI Dense-group Benchmark ({RUN_NAME})",
        "",
        "## Scope",
        "",
        "- Exact payload: native Tight-Inclusion `vertexFaceCCD` / `edgeEdgeCCD`.",
        "- STPF is trained once per split, checkpointed, then evaluated only on held-out groups/source.",
        "- Test rows report all-exact, random, frozen learned, one predeclared fixed heuristic, and a retrospective best fixed heuristic oracle.",
        "- Negative candidates are near-miss hard negatives with clearance above TI tolerance; final TP/TN/FP/FN is still measured by native TI.",
        "- The best heuristic oracle is diagnostic only; it is selected after test evaluation and is not a deployable baseline.",
        "",
        "## Sources",
        "",
        "| Source | Dataset | Mesh | Bytes |",
        "| --- | --- | --- | ---: |",
    ]
    for source in sources:
        lines.append(f"| `{source.name}` | {source.dataset} | `{source.path.as_posix()}` | {source.path.stat().st_size} |")
    lines += [
        "",
        "## Generation Parameters",
        "",
        f"- Groups per source: `{args.groups_per_source}`",
        f"- Group size: `{args.group_size}`",
        f"- Negative group ratio: `{args.negative_group_ratio}`",
        f"- Epochs: `{args.epochs}`",
        f"- Seed: `{args.seed}`",
        "",
    ]
    for split in splits:
        lines += [
            f"## Split: `{split['split']}`",
            "",
            f"- Frozen checkpoint: `{split['training']['checkpoint']}`",
            f"- Train candidates: `{split['train_candidate_count']}`",
            f"- Test groups/candidates: `{split['test_group_count']}` / `{split['test_candidate_count']}`",
            f"- Test positive/negative groups: `{split['test_positive_group_count']}` / `{split['test_negative_group_count']}`",
            f"- Validation recall@0.5 / precision@0.5: `{split['training']['validation_recall_at_0_5']:.6f}` / `{split['training']['validation_precision_at_0_5']:.6f}`",
            "",
            "| Method | Groups | Candidates | Positive groups | Exact calls | Call reduction | TP/TN/FP/FN | First positive rank | Exact ms | Wall ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
        for row in split["methods"]:
            lines.append(format_method_row(row))
        lines += [
            "",
            "All fixed heuristic candidates:",
            "",
            "| Heuristic | Exact calls | Call reduction | TP/TN/FP/FN | First positive rank | Wall ms |",
            "| --- | ---: | ---: | --- | ---: | ---: |",
        ]
        for row in split["all_heuristics"]:
            lines.append(
                f"| {row['method']} | {int(row['exact_calls'])} | {100.0 * float(row['exact_call_reduction']):.3f}% | "
                f"{int(row['tp'])}/{int(row['tn'])}/{int(row['fp'])}/{int(row['fn'])} | "
                f"{float(row['first_positive_rank_mean']):.3f} | {float(row['wall_ms']):.3f} |"
            )
        lines.append("")
    lines += [
        "## Reproduction",
        "",
        "```powershell",
        "conda activate cudadev",
        (
            "python src/tools/native_ti_heldout_dense_group_benchmark.py "
            f"--run-name {RUN_NAME} --groups-per-source {args.groups_per_source} "
            f"--group-size {args.group_size} --negative-group-ratio {args.negative_group_ratio} "
            f"--epochs {args.epochs} --seed {args.seed} --heldout-source {args.heldout_source}"
        ),
        "```",
        "",
    ]
    (BENCH_DIR / f"{RUN_NAME}.md").write_text("\n".join(lines), encoding="utf-8")


def source_local_group(c: Candidate, groups_per_source: int) -> int:
    return int(c.group_id % groups_per_source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-checkpoint held-out native TI dense-group benchmark.")
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--groups-per-source", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--negative-group-ratio", type=float, default=0.5)
    parser.add_argument("--group-train-fraction", type=float, default=0.5)
    parser.add_argument("--heldout-source", default="shapenetcore")
    parser.add_argument("--seed", type=int, default=fixed_seed)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    configure_run_name(args.run_name)
    if not TI_RUNNER.exists():
        raise FileNotFoundError(TI_RUNNER)
    if not (0.0 <= args.negative_group_ratio < 1.0):
        raise ValueError("--negative-group-ratio must be in [0, 1)")
    if not (0.0 < args.group_train_fraction < 1.0):
        raise ValueError("--group-train-fraction must be in (0, 1)")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    sources = discover_sources()
    if args.heldout_source not in {s.name for s in sources}:
        raise ValueError(f"--heldout-source must be one of {[s.name for s in sources]}")

    all_candidates: list[Candidate] = []
    for index, source in enumerate(sources):
        all_candidates.extend(
            build_source_csv_and_candidates(
                source,
                start_group_id=index * args.groups_per_source,
                group_count=args.groups_per_source,
                group_size=args.group_size,
                seed=args.seed + index * 101,
                negative_group_ratio=args.negative_group_ratio,
            )
        )

    group_train_count = int(math.floor(args.groups_per_source * args.group_train_fraction))
    group_train = [c for c in all_candidates if source_local_group(c, args.groups_per_source) < group_train_count]
    group_test = [c for c in all_candidates if source_local_group(c, args.groups_per_source) >= group_train_count]
    source_train = [c for c in all_candidates if c.source != args.heldout_source]
    source_test = [c for c in all_candidates if c.source == args.heldout_source]

    splits = [
        evaluate_split(
            "group_heldout",
            group_train,
            group_test,
            seed=args.seed + 17,
            device=args.device,
            epochs=args.epochs,
        ),
        evaluate_split(
            f"source_heldout_{args.heldout_source}",
            source_train,
            source_test,
            seed=args.seed + 29,
            device=args.device,
            epochs=args.epochs,
        ),
    ]
    manifest = {
        "run_name": RUN_NAME,
        "dataset_root": DATASET_ROOT,
        "sources": [asdict(s) for s in sources],
        "parameters": vars(args),
        "candidate_count": len(all_candidates),
        "positive_count": int(sum(c.truth for c in all_candidates)),
        "group_count": len({c.group_id for c in all_candidates}),
        "positive_group_count": len({c.group_id for c in all_candidates if c.truth}),
        "negative_group_count": len({c.group_id for c in all_candidates}) - len({c.group_id for c in all_candidates if c.truth}),
        "splits": splits,
    }
    _write_json(BENCH_DIR / f"{RUN_NAME}.json", manifest)
    write_report(sources, splits, args)


if __name__ == "__main__":
    main()
