from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from p2cccd.proposal.policy_head_selection import (
    RTSTPFPolicyHead,
    score_rtstpf_candidates,
    select_rtstpf_policy_head,
)
from p2cccd.proposal.stpf_model import (
    STPFModelPreset,
    build_stpf_model_from_checkpoint_payload,
)


RUN_NAME = "tight_inclusion_dense_group_real_exact_run_id"


@dataclass(frozen=True, slots=True)
class DenseGroupScheduleStats:
    schedule_csv: str
    group_count: int
    group_size: int
    positive_per_group: int
    candidate_count: int
    source_shard: str
    checkpoint: str
    score_mode: str
    mean_label_rank: float
    p90_label_rank: float
    label_exact_call_reduction: float


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    sigma = float(np.std(arr))
    if sigma <= 1.0e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - float(np.mean(arr))) / sigma


def _load_model(checkpoint: Path, *, device: str):
    payload = torch.load(checkpoint, map_location=device)
    model, state_dict = build_stpf_model_from_checkpoint_payload(
        payload,
        fallback_preset=STPFModelPreset.MEDIUM_MLP,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _predict_scores(
    *,
    checkpoint: Path,
    features: np.ndarray,
    feature_arrays: dict[str, np.ndarray],
    source_name: str,
    device: str,
) -> tuple[np.ndarray, str, str]:
    model = _load_model(checkpoint, device=device)
    tensor = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)
    with torch.no_grad():
        output = model(tensor)
    prediction_arrays = {
        "priority_score": output.priority_score.detach().cpu().numpy(),
        "cost_score": output.cost_score.detach().cpu().numpy(),
        "uncertainty_score": output.uncertainty_score.detach().cpu().numpy(),
    }
    selection = select_rtstpf_policy_head(
        source_name,
        candidate_density=float(features.shape[0]),
        hard_negative_group=True,
    )
    scores = score_rtstpf_candidates(
        prediction_arrays,
        feature_arrays,
        head=selection.head,
    )
    return np.asarray(scores, dtype=np.float64), str(selection.head), selection.reason


def _make_groups(
    *,
    shard: Path,
    checkpoint: Path,
    group_count: int,
    group_size: int,
    positives_per_group: int,
    seed: int,
    source_name: str,
    device: str,
    random_schedule: bool,
) -> tuple[list[dict[str, Any]], DenseGroupScheduleStats]:
    if positives_per_group <= 0 or positives_per_group >= group_size:
        raise ValueError("positives_per_group must be in [1, group_size)")
    arrays = np.load(shard, allow_pickle=True)
    truth = np.asarray(arrays["ground_truth"], dtype=np.bool_)
    positive_indices = np.flatnonzero(truth)
    negative_indices = np.flatnonzero(~truth)
    if positive_indices.size < group_count * positives_per_group:
        raise ValueError("not enough positive primitive queries in shard")
    negatives_per_group = int(group_size) - int(positives_per_group)
    if negative_indices.size < group_count * negatives_per_group:
        raise ValueError("not enough negative primitive queries in shard")

    rng = np.random.default_rng(seed)
    target_cost = np.asarray(arrays["scalar_targets"][:, 1], dtype=np.float64)
    hard_negatives = negative_indices[np.argsort(target_cost[negative_indices])[::-1]]
    rng.shuffle(positive_indices)

    selected: list[int] = []
    group_ids: list[int] = []
    for group_id in range(1, group_count + 1):
        pos_start = (group_id - 1) * positives_per_group
        neg_start = (group_id - 1) * negatives_per_group
        rows = np.concatenate(
            [
                positive_indices[pos_start : pos_start + positives_per_group],
                hard_negatives[neg_start : neg_start + negatives_per_group],
            ]
        )
        rng.shuffle(rows)
        selected.extend(int(row) for row in rows)
        group_ids.extend([group_id] * rows.shape[0])

    selected_array = np.asarray(selected, dtype=np.int64)
    features = np.asarray(arrays["features"][selected_array], dtype=np.float32)
    feature_arrays = {"features": features}
    if random_schedule:
        scores = rng.random(selected_array.shape[0], dtype=np.float64)
        score_mode = "random_uniform"
        reason = "random baseline schedule"
    else:
        scores, score_mode, reason = _predict_scores(
            checkpoint=checkpoint,
            features=features,
            feature_arrays=feature_arrays,
            source_name=source_name,
            device=device,
        )

    case_names = np.asarray(arrays["case_names"]).astype(str)
    kind_names = np.asarray(arrays["kind_names"]).astype(str)
    csv_paths = np.asarray(arrays["csv_paths"]).astype(str)
    query_indices = np.asarray(arrays["source_query_indices"], dtype=np.uint64)

    rows_out: list[dict[str, Any]] = []
    for local_index, row_index in enumerate(selected_array):
        rows_out.append(
            {
                "group_id": int(group_ids[local_index]),
                "case": str(case_names[row_index]),
                "kind": str(kind_names[row_index]),
                "csv_path": str(csv_paths[row_index]).replace("\\", "/"),
                "query_index": int(query_indices[row_index]),
                "score": float(scores[local_index]),
                "truth": bool(truth[row_index]),
            }
        )

    ranks: list[int] = []
    for group_id in range(1, group_count + 1):
        group_rows = [row for row in rows_out if row["group_id"] == group_id]
        group_rows.sort(key=lambda item: float(item["score"]), reverse=True)
        truth_order = [bool(row["truth"]) for row in group_rows]
        ranks.append(truth_order.index(True) + 1)
    mean_rank = float(np.mean(ranks)) if ranks else 0.0
    p90_rank = float(np.quantile(np.asarray(ranks, dtype=np.float64), 0.90)) if ranks else 0.0
    calls = int(sum(ranks))
    reduction = 1.0 - calls / float(max(1, group_count * group_size))
    stats = DenseGroupScheduleStats(
        schedule_csv="",
        group_count=int(group_count),
        group_size=int(group_size),
        positive_per_group=int(positives_per_group),
        candidate_count=int(group_count * group_size),
        source_shard=shard.as_posix(),
        checkpoint=checkpoint.as_posix(),
        score_mode=score_mode,
        mean_label_rank=mean_rank,
        p90_label_rank=p90_rank,
        label_exact_call_reduction=float(reduction),
    )
    # Keep reason in the score mode string for the human report without adding
    # unstable extra columns to the C++ schedule CSV.
    object.__setattr__(stats, "score_mode", f"{score_mode}; {reason}")
    return rows_out, stats


def _write_schedule(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group_id", "case", "kind", "csv_path", "query_index", "score"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def _run_native_ti(
    *,
    executable: Path,
    dataset_root: Path,
    schedule_csv: Path,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    command = [
        str(executable),
        "--dataset-root",
        str(dataset_root),
        "--schedule",
        str(schedule_csv),
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError("native TI dense group benchmark failed:\n" + completed.stdout)
    return json.loads(output_json.read_text(encoding="utf-8"))


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    learned = payload["native_results"]["learned"]
    random_result = payload["native_results"]["random"]
    learned_stats = payload["schedule_stats"]["learned"]
    random_stats = payload["schedule_stats"]["random"]
    lines = [
        "# Tight-Inclusion real exact-payload dense-group early-stop benchmark",
        "",
        "## Protocol",
        "",
        "- Inputis NYU/Tight-Inclusion primitive CCD heldout shard, by group construct balanced hard-negative candidate group. ",
        "- schedule description learned RTSTPF policy head selection description; comparisondescriptionas random uniform schedule. ",
        "- exact payload descriptionasreal `ticcd::vertexFaceCCD` / `ticcd::edgeEdgeCCD`, is not proxy oracle. ",
        "- final correctness is group-level conservative early-stop: if group descriptionindescription collision primitive, description exact descriptiontoNo.description positive afterdescription; descriptioncomplete group. ",
        "",
        "## description",
        "",
        "| Method | Groups | Candidates | Exact calls | Exact-call reduction | TP | TN | FP | FN | Recall | First-positive rank | Exact ms | Wall ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| RTSTPFExact+TI learned | `{learned['group_count']}` | `{learned['candidate_count']}` | `{learned['learned_exact_calls']}` | `{learned['exact_call_reduction']:.4%}` | `{learned['tp']}` | `{learned['tn']}` | `{learned['fp']}` | `{learned['fn']}` | `{learned['recall']:.6f}` | `{learned['first_positive_rank_mean']:.3f}` | `{learned['exact_ms']:.3f}` | `{learned['wall_ms']:.3f}` |",
        f"| Random+TI | `{random_result['group_count']}` | `{random_result['candidate_count']}` | `{random_result['learned_exact_calls']}` | `{random_result['exact_call_reduction']:.4%}` | `{random_result['tp']}` | `{random_result['tn']}` | `{random_result['fp']}` | `{random_result['fn']}` | `{random_result['recall']:.6f}` | `{random_result['first_positive_rank_mean']:.3f}` | `{random_result['exact_ms']:.3f}` | `{random_result['wall_ms']:.3f}` |",
        "",
        "## Schedule description",
        "",
        "| Method | Score mode | Label mean rank | Label p90 rank | Label call reduction | Schedule CSV |",
        "| --- | --- | ---: | ---: | ---: | --- |",
        f"| Learned | `{learned_stats['score_mode']}` | `{learned_stats['mean_label_rank']:.3f}` | `{learned_stats['p90_label_rank']:.3f}` | `{learned_stats['label_exact_call_reduction']:.4%}` | `{learned_stats['schedule_csv']}` |",
        f"| Random | `{random_stats['score_mode']}` | `{random_stats['mean_label_rank']:.3f}` | `{random_stats['p90_label_rank']:.3f}` | `{random_stats['label_exact_call_reduction']:.4%}` | `{random_stats['schedule_csv']}` |",
        "",
        "## Conclusion",
        "",
        "- this benchmark description P0-220 description: native dense group  exact payload descriptionfrom proxy-oracle descriptiontoreal Tight-Inclusion primitive exact. ",
        "- `FN=0` isthrough conservative group early-stop guarantee: STPF only determines exact description, description candidate. ",
        "- descriptionis selected heldout hard-negative group, is notdescription 100GB description; descriptionwithdescriptionusesame executable and schedule CSV description. ",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(
    *,
    shard: Path,
    checkpoint: Path,
    dataset_root: Path,
    executable: Path,
    output_dir: Path,
    run_name: str,
    group_count: int,
    group_size: int,
    positives_per_group: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    learned_rows, learned_stats = _make_groups(
        shard=shard,
        checkpoint=checkpoint,
        group_count=group_count,
        group_size=group_size,
        positives_per_group=positives_per_group,
        seed=seed,
        source_name="tight_inclusion_nyu_hard_negative",
        device=device,
        random_schedule=False,
    )
    random_rows, random_stats = _make_groups(
        shard=shard,
        checkpoint=checkpoint,
        group_count=group_count,
        group_size=group_size,
        positives_per_group=positives_per_group,
        seed=seed,
        source_name="tight_inclusion_nyu_hard_negative",
        device=device,
        random_schedule=True,
    )
    learned_schedule = output_dir / f"{run_name}_learned_schedule.csv"
    random_schedule = output_dir / f"{run_name}_random_schedule.csv"
    _write_schedule(learned_schedule, learned_rows)
    _write_schedule(random_schedule, random_rows)
    object.__setattr__(learned_stats, "schedule_csv", learned_schedule.as_posix())
    object.__setattr__(random_stats, "schedule_csv", random_schedule.as_posix())

    native_results = {
        "learned": _run_native_ti(
            executable=executable,
            dataset_root=dataset_root,
            schedule_csv=learned_schedule,
            output_json=output_dir / f"{run_name}_learned_native_ti.json",
            output_md=output_dir / f"{run_name}_learned_native_ti.md",
        ),
        "random": _run_native_ti(
            executable=executable,
            dataset_root=dataset_root,
            schedule_csv=random_schedule,
            output_json=output_dir / f"{run_name}_random_native_ti.json",
            output_md=output_dir / f"{run_name}_random_native_ti.md",
        ),
    }
    payload: dict[str, Any] = {
        "run_name": run_name,
        "shard": shard.as_posix(),
        "checkpoint": checkpoint.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "executable": executable.as_posix(),
        "schedule_stats": {
            "learned": asdict(learned_stats),
            "random": asdict(random_stats),
        },
        "native_results": native_results,
    }
    json_path = output_dir / f"{run_name}.json"
    md_path = output_dir / f"{run_name}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(md_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard",
        type=Path,
        default=Path("src/datasets/training/tight_inclusion_nyu/shards/tight_inclusion_nyu_large_run_id/heldout_test/chunk_000000.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("src/outputs/stpf_training/tight_inclusion_nyu_large_run_id/model_state.pt"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("src/baseline/datasets/continuous-collision-detection"),
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("src/build_tools/tight_inclusion_dense_group_early_stop_benchmark.exe"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("src/benchmark"))
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--group-count", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--positives-per-group", type=int, default=1)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_suite(
        shard=args.shard,
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        executable=args.executable,
        output_dir=args.output_dir,
        run_name=args.run_name,
        group_count=args.group_count,
        group_size=args.group_size,
        positives_per_group=args.positives_per_group,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
