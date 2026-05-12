from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import median
from typing import Sequence

import numpy as np
import torch

from p2cccd.data.shards import read_npz_shard
from p2cccd.proposal.stpf_model import build_stpf_model_from_checkpoint_payload


@dataclass(slots=True)
class SchedulingGroup:
    case_name: str
    kind_name: str
    query_id: int
    scores: list[float]
    truths: list[bool]
    costs: list[float]


@dataclass(frozen=True, slots=True)
class SchedulingMetrics:
    score_mode: str
    group_count: int
    positive_group_count: int
    candidate_count: int
    positive_candidate_count: int
    no_proposal_exact_calls: int
    rtstpf_exact_calls: int
    oracle_best_exact_calls: int
    no_proposal_exact_work: float
    rtstpf_exact_work: float
    oracle_best_exact_work: float
    exact_call_reduction: float
    exact_work_reduction: float
    oracle_exact_call_reduction: float
    oracle_exact_work_reduction: float
    fn_count: int
    first_positive_rank_mean: float
    first_positive_rank_p50: float
    first_positive_rank_p90: float
    first_positive_rank_p99: float


def _chunks_for_split(shard_manifest: dict[str, object], split: str) -> list[Path]:
    return [
        Path(str(chunk["path"]))
        for chunk in shard_manifest["chunks"]  # type: ignore[index]
        if str(chunk["split"]) == split
    ]


def _score_from_output(output, mode: str) -> np.ndarray:
    priority = output.priority_score.detach().cpu().numpy().astype(np.float64)
    uncertainty = output.uncertainty_score.detach().cpu().numpy().astype(np.float64)
    cost = output.cost_score.detach().cpu().numpy().astype(np.float64)
    if mode == "priority":
        return priority
    if mode == "priority_uncertainty":
        return priority + 0.5 * uncertainty
    if mode == "priority_cost":
        return priority * np.log1p(np.clip(cost, 0.0, None))
    if mode == "priority_uncertainty_cost":
        return (priority + 0.5 * uncertainty) * np.log1p(np.clip(cost, 0.0, None))
    raise ValueError(f"unsupported score mode {mode!r}")


def _load_model(checkpoint: Path, device: str):
    payload = torch.load(checkpoint, map_location=device)
    model, state_dict = build_stpf_model_from_checkpoint_payload(payload, fallback_preset="medium_mlp")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _collect_groups(
    *,
    checkpoint: Path,
    shard_paths: Sequence[Path],
    device: str,
    batch_size: int,
    score_mode: str,
    max_rows: int | None,
) -> dict[tuple[str, str, int], SchedulingGroup]:
    model = _load_model(checkpoint, device)
    groups: dict[tuple[str, str, int], SchedulingGroup] = {}
    row_count = 0
    with torch.no_grad():
        for path in shard_paths:
            arrays = read_npz_shard(path)["arrays"]
            features = np.asarray(arrays["features"], dtype=np.float32)
            ids = np.asarray(arrays["ids"], dtype=np.uint64)
            truths = np.asarray(arrays["ground_truth"], dtype=np.bool_)
            costs = np.asarray(arrays["scalar_targets"], dtype=np.float64)[:, 1]
            cases = np.asarray(arrays["case_names"]).astype(str)
            kinds = np.asarray(arrays["kind_names"]).astype(str)
            shard_rows = int(features.shape[0])
            for start in range(0, shard_rows, batch_size):
                if max_rows is not None and row_count >= max_rows:
                    break
                end = min(shard_rows, start + batch_size)
                if max_rows is not None:
                    end = min(end, start + (max_rows - row_count))
                batch_features = torch.as_tensor(features[start:end], dtype=torch.float32, device=device)
                output = model(batch_features)
                scores = _score_from_output(output, score_mode)
                for local, score in enumerate(scores, start=start):
                    query_id = int(ids[local, 1])
                    key = (str(cases[local]), str(kinds[local]), query_id)
                    group = groups.get(key)
                    if group is None:
                        group = SchedulingGroup(
                            case_name=str(cases[local]),
                            kind_name=str(kinds[local]),
                            query_id=query_id,
                            scores=[],
                            truths=[],
                            costs=[],
                        )
                        groups[key] = group
                    group.scores.append(float(score))
                    group.truths.append(bool(truths[local]))
                    group.costs.append(float(max(0.0, costs[local])))
                row_count += end - start
            if max_rows is not None and row_count >= max_rows:
                break
    return groups


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _evaluate_groups(groups: dict[tuple[str, str, int], SchedulingGroup], *, score_mode: str) -> tuple[SchedulingMetrics, dict[str, dict[str, float]]]:
    baseline_calls = 0
    scheduled_calls = 0
    oracle_calls = 0
    baseline_work = 0.0
    scheduled_work = 0.0
    oracle_work = 0.0
    positive_groups = 0
    positive_candidates = 0
    ranks: list[float] = []
    per_case: dict[str, dict[str, float]] = {}
    for group in groups.values():
        count = len(group.truths)
        truth_indices = [index for index, truth in enumerate(group.truths) if truth]
        has_positive = bool(truth_indices)
        positive_groups += int(has_positive)
        positive_candidates += len(truth_indices)
        baseline_calls += count
        group_baseline_work = float(sum(group.costs))
        baseline_work += group_baseline_work
        order = sorted(range(count), key=lambda index: group.scores[index], reverse=True)
        if has_positive:
            first_rank = next(rank for rank, index in enumerate(order, start=1) if group.truths[index])
            selected = order[:first_rank]
            oracle_index = max(truth_indices, key=lambda index: group.costs[index])
            oracle_group_calls = 1
            oracle_group_work = float(group.costs[oracle_index])
            ranks.append(float(first_rank))
        else:
            first_rank = count
            selected = order
            oracle_group_calls = count
            oracle_group_work = group_baseline_work
        group_calls = len(selected)
        group_work = float(sum(group.costs[index] for index in selected))
        scheduled_calls += group_calls
        scheduled_work += group_work
        oracle_calls += oracle_group_calls
        oracle_work += oracle_group_work
        row = per_case.setdefault(
            group.case_name,
            {
                "groups": 0.0,
                "positive_groups": 0.0,
                "candidates": 0.0,
                "positive_candidates": 0.0,
                "baseline_calls": 0.0,
                "scheduled_calls": 0.0,
                "oracle_calls": 0.0,
                "baseline_work": 0.0,
                "scheduled_work": 0.0,
                "oracle_work": 0.0,
                "rank_sum": 0.0,
            },
        )
        row["groups"] += 1.0
        row["positive_groups"] += float(has_positive)
        row["candidates"] += float(count)
        row["positive_candidates"] += float(len(truth_indices))
        row["baseline_calls"] += float(count)
        row["scheduled_calls"] += float(group_calls)
        row["oracle_calls"] += float(oracle_group_calls)
        row["baseline_work"] += group_baseline_work
        row["scheduled_work"] += group_work
        row["oracle_work"] += oracle_group_work
        row["rank_sum"] += float(first_rank) if has_positive else 0.0
    for row in per_case.values():
        row["exact_call_reduction"] = 1.0 - row["scheduled_calls"] / max(1.0, row["baseline_calls"])
        row["exact_work_reduction"] = 1.0 - row["scheduled_work"] / max(1.0e-12, row["baseline_work"])
        row["oracle_exact_call_reduction"] = 1.0 - row["oracle_calls"] / max(1.0, row["baseline_calls"])
        row["oracle_exact_work_reduction"] = 1.0 - row["oracle_work"] / max(1.0e-12, row["baseline_work"])
        row["first_positive_rank_mean"] = row["rank_sum"] / max(1.0, row["positive_groups"])
    metrics = SchedulingMetrics(
        score_mode=score_mode,
        group_count=len(groups),
        positive_group_count=positive_groups,
        candidate_count=baseline_calls,
        positive_candidate_count=positive_candidates,
        no_proposal_exact_calls=baseline_calls,
        rtstpf_exact_calls=scheduled_calls,
        oracle_best_exact_calls=oracle_calls,
        no_proposal_exact_work=baseline_work,
        rtstpf_exact_work=scheduled_work,
        oracle_best_exact_work=oracle_work,
        exact_call_reduction=1.0 - scheduled_calls / max(1, baseline_calls),
        exact_work_reduction=1.0 - scheduled_work / max(1.0e-12, baseline_work),
        oracle_exact_call_reduction=1.0 - oracle_calls / max(1, baseline_calls),
        oracle_exact_work_reduction=1.0 - oracle_work / max(1.0e-12, baseline_work),
        fn_count=0,
        first_positive_rank_mean=float(sum(ranks) / max(1, len(ranks))),
        first_positive_rank_p50=float(median(ranks)) if ranks else 0.0,
        first_positive_rank_p90=_quantile(ranks, 0.90),
        first_positive_rank_p99=_quantile(ranks, 0.99),
    )
    return metrics, per_case


def run_scheduling_shard_benchmark(
    *,
    checkpoint: Path,
    shard_root: Path,
    split: str = "heldout_test",
    score_modes: Sequence[str] = ("priority", "priority_uncertainty", "priority_cost", "priority_uncertainty_cost"),
    device: str = "cuda",
    batch_size: int = 32768,
    max_rows: int | None = None,
) -> dict[str, object]:
    shard_manifest = json.loads((Path(shard_root) / "manifest.json").read_text(encoding="utf-8"))
    shard_paths = _chunks_for_split(shard_manifest, split)
    results: list[dict[str, object]] = []
    for mode in score_modes:
        groups = _collect_groups(
            checkpoint=checkpoint,
            shard_paths=shard_paths,
            device=device,
            batch_size=batch_size,
            score_mode=mode,
            max_rows=max_rows,
        )
        metrics, per_case = _evaluate_groups(groups, score_mode=mode)
        results.append({"metrics": asdict(metrics), "per_case": per_case})
    best = max(results, key=lambda item: float(item["metrics"]["exact_work_reduction"])) if results else None
    return {
        "checkpoint": Path(checkpoint).as_posix(),
        "shard_root": Path(shard_root).as_posix(),
        "split": split,
        "max_rows": max_rows,
        "results": results,
        "best_score_mode": None if best is None else best["metrics"]["score_mode"],
    }


def write_scheduling_report(path: Path, payload: dict[str, object]) -> Path:
    lines = [
        "# RTSTPFExact Group Scheduling Benchmark",
        "",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Shard root: `{payload['shard_root']}`",
        f"- Split: `{payload['split']}`",
        f"- Max rows: `{payload['max_rows']}`",
        f"- Best score mode: `{payload['best_score_mode']}`",
        "",
        "## Score Mode Summary",
        "",
        "| Score mode | Groups | Positive groups | Candidates | Exact calls | Call reduction | Work reduction | Oracle call reduction | FN | Rank mean | Rank p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["results"]:  # type: ignore[index]
        metrics = item["metrics"]
        lines.append(
            f"| `{metrics['score_mode']}` | `{metrics['group_count']}` | `{metrics['positive_group_count']}` | "
            f"`{metrics['candidate_count']}` | `{metrics['rtstpf_exact_calls']}` | "
            f"`{metrics['exact_call_reduction']}` | `{metrics['exact_work_reduction']}` | "
            f"`{metrics['oracle_exact_call_reduction']}` | `{metrics['fn_count']}` | "
            f"`{metrics['first_positive_rank_mean']}` | `{metrics['first_positive_rank_p90']}` |"
        )
    lines.extend(["", "## Best Per-Case", ""])
    best_mode = payload.get("best_score_mode")
    best_item = next((item for item in payload["results"] if item["metrics"]["score_mode"] == best_mode), None)  # type: ignore[index]
    if best_item is not None:
        lines.extend(
            [
                "| Case | Groups | Positive groups | Candidates | Exact calls | Call reduction | Work reduction | Rank mean |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for case, row in sorted(best_item["per_case"].items()):  # type: ignore[index]
            lines.append(
                f"| `{case}` | `{int(row['groups'])}` | `{int(row['positive_groups'])}` | "
                f"`{int(row['candidates'])}` | `{int(row['scheduled_calls'])}` | "
                f"`{row['exact_call_reduction']}` | `{row['exact_work_reduction']}` | "
                f"`{row['first_positive_rank_mean']}` |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--split", default="heldout_test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_scheduling_shard_benchmark(
        checkpoint=args.checkpoint,
        shard_root=args.shard_root,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        max_rows=None if args.max_rows <= 0 else int(args.max_rows),
    )
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_scheduling_report(md_path, payload)
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
