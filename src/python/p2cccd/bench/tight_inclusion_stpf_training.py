from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch

from p2cccd.data.shards import read_npz_shard
from p2cccd.proposal.ort_inference import ensure_stpf_model_onnx
from p2cccd.proposal.stpf_model import (
    STPFModel,
    STPFModelPreset,
    build_stpf_model,
    stpf_config_to_dict,
)
from p2cccd.proposal.training import STPFTrainingConfig, stpf_training_loss, validate_training_config


@dataclass(frozen=True, slots=True)
class StreamingEpochMetrics:
    epoch: int
    split: str
    row_count: int
    positive_count: int
    positive_ratio: float
    loss: float
    interval_top1_recall: float
    family_top2_recall: float
    estimated_exact_work_reduction: float
    mean_predicted_cost: float
    mean_target_cost: float


def _chunks_for_split(shard_manifest: dict[str, object], split: str) -> list[Path]:
    return [
        Path(str(chunk["path"]))
        for chunk in shard_manifest["chunks"]  # type: ignore[index]
        if chunk["split"] == split
    ]


def _row_count_for_paths(paths: Sequence[Path]) -> int:
    total = 0
    for path in paths:
        total += int(read_npz_shard(path)["metadata"]["row_count"])
    return total


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    return read_npz_shard(path)["arrays"]


def _targets_from_arrays(arrays: dict[str, np.ndarray], indices: np.ndarray, *, device: str) -> dict[str, torch.Tensor]:
    scalar_targets = np.asarray(arrays["scalar_targets"][indices], dtype=np.float32)
    return {
        "interval_targets": torch.as_tensor(
            np.asarray(arrays["interval_targets"][indices], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        ),
        "family_targets": torch.as_tensor(
            np.asarray(arrays["family_targets"][indices], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        ),
        "priority_target": torch.as_tensor(scalar_targets[:, 0], dtype=torch.float32, device=device),
        "cost_target": torch.as_tensor(scalar_targets[:, 1], dtype=torch.float32, device=device),
        "uncertainty_target": torch.as_tensor(scalar_targets[:, 2], dtype=torch.float32, device=device),
        "target_mask": torch.as_tensor(
            np.asarray(arrays["ids"][indices, 8], dtype=np.int64),
            dtype=torch.int64,
            device=device,
        ),
    }


def _iter_index_batches(row_count: int, *, batch_size: int, shuffle: bool, seed: int) -> list[np.ndarray]:
    if row_count <= 0:
        return []
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(row_count)
    else:
        indices = np.arange(row_count, dtype=np.int64)
    return [indices[start : start + batch_size] for start in range(0, row_count, batch_size)]


def _truth_array(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "ground_truth" in arrays:
        return np.asarray(arrays["ground_truth"], dtype=np.bool_)
    return np.asarray(arrays["scalar_targets"][:, 0] >= 0.999, dtype=np.bool_)


def _kind_names(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "kind_names" in arrays:
        return np.asarray(arrays["kind_names"]).astype(str)
    target_index = np.argmax(np.asarray(arrays["family_targets"], dtype=np.float32), axis=1)
    return np.where(target_index == 0, "vertex-face", "edge-edge")


def _case_names(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "case_names" in arrays:
        return np.asarray(arrays["case_names"]).astype(str)
    return np.asarray(["unknown"] * int(arrays["features"].shape[0]), dtype=np.str_)


def _csv_paths(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "csv_paths" in arrays:
        return np.asarray(arrays["csv_paths"]).astype(str)
    return np.asarray(["unknown"] * int(arrays["features"].shape[0]), dtype=np.str_)


def _source_query_indices(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "source_query_indices" in arrays:
        return np.asarray(arrays["source_query_indices"], dtype=np.uint64)
    return np.asarray(arrays["ids"][:, 2], dtype=np.uint64)


def _family_top2_hits(family_scores: np.ndarray, family_targets: np.ndarray) -> int:
    if family_scores.shape[0] == 0:
        return 0
    predicted = np.argpartition(family_scores, kth=-2, axis=1)[:, -2:]
    targets = np.argmax(family_targets, axis=1)
    return int(sum(int(target in predicted[index]) for index, target in enumerate(targets)))


def _score(priority: np.ndarray, uncertainty: np.ndarray, *, uncertainty_weight: float) -> np.ndarray:
    return np.asarray(priority, dtype=np.float64) + float(uncertainty_weight) * np.asarray(uncertainty, dtype=np.float64)


def _curve_template(thresholds: Sequence[float]) -> dict[float, dict[str, float]]:
    return {
        float(threshold): {
            "threshold": float(threshold),
            "tp": 0.0,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "exact_calls": 0.0,
            "selected_cost": 0.0,
            "baseline_cost": 0.0,
        }
        for threshold in thresholds
    }


def _update_threshold_curve(
    curve: dict[float, dict[str, float]],
    *,
    scores: np.ndarray,
    truth: np.ndarray,
    costs: np.ndarray,
) -> None:
    truth = np.asarray(truth, dtype=np.bool_)
    costs = np.asarray(costs, dtype=np.float64)
    for threshold, row in curve.items():
        selected = scores >= threshold
        result = truth & selected
        row["tp"] += float(np.count_nonzero(result & truth))
        row["fp"] += float(np.count_nonzero(result & ~truth))
        row["fn"] += float(np.count_nonzero((~result) & truth))
        row["tn"] += float(np.count_nonzero((~result) & ~truth))
        row["exact_calls"] += float(np.count_nonzero(selected))
        row["selected_cost"] += float(costs[selected].sum()) if selected.any() else 0.0
        row["baseline_cost"] += float(costs.sum())


def _finalize_threshold_curve(curve: dict[float, dict[str, float]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for threshold, row in sorted(curve.items()):
        tp = row["tp"]
        fn = row["fn"]
        fp = row["fp"]
        exact_calls = row["exact_calls"]
        baseline_cost = row["baseline_cost"]
        selected_cost = row["selected_cost"]
        rows.append(
            {
                **row,
                "threshold": threshold,
                "recall": tp / max(1.0, tp + fn),
                "precision": tp / max(1.0, tp + fp),
                "exact_call_reduction": 1.0 - exact_calls / max(1.0, tp + fn + fp + row["tn"]),
                "exact_work_reduction": 1.0 - selected_cost / max(1.0e-12, baseline_cost),
            }
        )
    return rows


def evaluate_streaming_shards(
    model: STPFModel,
    shard_paths: Sequence[Path],
    config: STPFTrainingConfig,
    *,
    epoch: int,
    split: str,
    uncertainty_weight: float,
    thresholds: Sequence[float] | None = None,
    group_threshold: float | None = None,
    max_rows: int | None = None,
    fn_risk_top_k: int = 20,
) -> tuple[StreamingEpochMetrics, dict[str, Any]]:
    cfg = validate_training_config(config)
    was_training = model.training
    model.eval()
    curve = _curve_template(thresholds or [])
    per_case: dict[str, dict[str, int]] = {}
    per_kind: dict[str, dict[str, int]] = {}
    positive_risks: list[dict[str, object]] = []
    row_count = 0
    positive_count = 0
    weighted_loss = 0.0
    interval_hits = 0
    family_hits = 0
    predicted_cost_sum = 0.0
    target_cost_sum = 0.0
    risk_adjusted_work = 0.0
    min_positive_score: float | None = None
    with torch.no_grad():
        for path in shard_paths:
            arrays = _load_arrays(path)
            shard_rows = int(arrays["features"].shape[0])
            for indices in _iter_index_batches(shard_rows, batch_size=cfg.batch_size, shuffle=False, seed=cfg.seed):
                if max_rows is not None and row_count >= max_rows:
                    break
                if max_rows is not None and row_count + int(indices.shape[0]) > max_rows:
                    indices = indices[: max_rows - row_count]
                features = torch.as_tensor(
                    np.asarray(arrays["features"][indices], dtype=np.float32),
                    dtype=torch.float32,
                    device=cfg.device,
                )
                targets = _targets_from_arrays(arrays, indices, device=cfg.device)
                output = model(features)
                loss = stpf_training_loss(output, targets, cost_aware_weight=cfg.cost_aware_weight)
                batch_size = int(indices.shape[0])
                weighted_loss += float(loss.detach().cpu()) * batch_size
                interval_scores = torch.softmax(output.interval_logits, dim=-1).detach().cpu().numpy()
                family_scores = torch.sigmoid(output.family_logits).detach().cpu().numpy()
                interval_targets = np.asarray(arrays["interval_targets"][indices], dtype=np.float32)
                family_targets = np.asarray(arrays["family_targets"][indices], dtype=np.float32)
                interval_hits += int(np.count_nonzero(np.argmax(interval_scores, axis=1) == np.argmax(interval_targets, axis=1)))
                family_hits += _family_top2_hits(family_scores, family_targets)
                priority = output.priority_score.detach().cpu().numpy()
                cost = output.cost_score.detach().cpu().numpy()
                uncertainty = output.uncertainty_score.detach().cpu().numpy()
                target_cost = np.asarray(arrays["scalar_targets"][indices, 1], dtype=np.float64)
                truth = _truth_array(arrays)[indices]
                scores = _score(priority, uncertainty, uncertainty_weight=uncertainty_weight)
                if np.any(truth):
                    positive_scores = scores[truth]
                    local_min = float(np.min(positive_scores))
                    min_positive_score = local_min if min_positive_score is None else min(min_positive_score, local_min)
                if curve:
                    _update_threshold_curve(curve, scores=scores, truth=truth, costs=target_cost)
                case_names = _case_names(arrays)[indices]
                kind_names = _kind_names(arrays)[indices]
                group_selected = None if group_threshold is None else scores >= float(group_threshold)
                source_csvs = _csv_paths(arrays)[indices]
                source_queries = _source_query_indices(arrays)[indices]
                for local_index, (case, kind, is_positive, score_value, source_csv, source_query) in enumerate(
                    zip(case_names, kind_names, truth, scores, source_csvs, source_queries)
                ):
                    case_row = per_case.setdefault(
                        str(case),
                        {"rows": 0, "positive_count": 0, "tp": 0, "fn": 0, "exact_calls": 0},
                    )
                    kind_row = per_kind.setdefault(
                        str(kind),
                        {"rows": 0, "positive_count": 0, "tp": 0, "fn": 0, "exact_calls": 0},
                    )
                    case_row["rows"] += 1
                    kind_row["rows"] += 1
                    selected = bool(group_selected is not None and group_selected[local_index])
                    if selected:
                        case_row["exact_calls"] += 1
                        kind_row["exact_calls"] += 1
                    if bool(is_positive):
                        case_row["positive_count"] += 1
                        kind_row["positive_count"] += 1
                        if selected:
                            case_row["tp"] += 1
                            kind_row["tp"] += 1
                        elif group_selected is not None:
                            case_row["fn"] += 1
                            kind_row["fn"] += 1
                        positive_risks.append(
                            {
                                "score": float(score_value),
                                "case": str(case),
                                "kind": str(kind),
                                "csv_path": str(source_csv),
                                "query_index": int(source_query),
                            }
                        )
                batch_work = output.cost_score * (1.0 - 0.5 * output.priority_score) * (
                    1.0 + 0.25 * output.uncertainty_score
                )
                risk_adjusted_work += float(torch.clamp(batch_work, min=0.0).sum().cpu())
                predicted_cost_sum += float(np.asarray(cost, dtype=np.float64).sum())
                target_cost_sum += float(target_cost.sum())
                positive_count += int(np.count_nonzero(truth))
                row_count += batch_size
            if max_rows is not None and row_count >= max_rows:
                break
    if was_training:
        model.train()
    deduped_risks: dict[tuple[str, int, str], dict[str, object]] = {}
    for item in positive_risks:
        key = (str(item["csv_path"]), int(item["query_index"]), str(item["kind"]))
        previous = deduped_risks.get(key)
        if previous is None or float(item["score"]) < float(previous["score"]):
            deduped_risks[key] = item
    positive_risks = sorted(deduped_risks.values(), key=lambda item: float(item["score"]))
    for grouped in (per_case, per_kind):
        for row in grouped.values():
            row["recall"] = row["tp"] / max(1, row["positive_count"])
            row["exact_call_rate"] = row["exact_calls"] / max(1, row["rows"])
    metrics = StreamingEpochMetrics(
        epoch=epoch,
        split=split,
        row_count=row_count,
        positive_count=positive_count,
        positive_ratio=positive_count / max(1, row_count),
        loss=weighted_loss / max(1, row_count),
        interval_top1_recall=interval_hits / max(1, row_count),
        family_top2_recall=family_hits / max(1, row_count),
        estimated_exact_work_reduction=1.0 - risk_adjusted_work / max(1.0e-12, target_cost_sum),
        mean_predicted_cost=predicted_cost_sum / max(1, row_count),
        mean_target_cost=target_cost_sum / max(1, row_count),
    )
    details = {
        "min_positive_score": min_positive_score,
        "threshold_curve": _finalize_threshold_curve(curve),
        "per_case": per_case,
        "per_kind": per_kind,
        "fn_risk_top_k": positive_risks[:fn_risk_top_k],
    }
    return metrics, details


def train_stpf_model_from_npz_stream(
    train_paths: Sequence[Path],
    validation_paths: Sequence[Path],
    config: STPFTrainingConfig,
    *,
    uncertainty_weight: float,
    train_eval_max_rows: int | None = None,
    validation_eval_max_rows: int | None = None,
) -> tuple[STPFModel, list[StreamingEpochMetrics], dict[str, Any]]:
    cfg = validate_training_config(config)
    torch.manual_seed(cfg.seed)
    model = build_stpf_model(cfg.model_preset) if cfg.model_config is None else STPFModel(cfg.model_config)
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    history: list[StreamingEpochMetrics] = []
    rng = random.Random(cfg.seed)
    for epoch in range(1, cfg.epochs + 1):
        chunk_order = list(train_paths)
        if cfg.shuffle:
            rng.shuffle(chunk_order)
        model.train()
        for shard_index, path in enumerate(chunk_order):
            arrays = _load_arrays(path)
            shard_rows = int(arrays["features"].shape[0])
            for indices in _iter_index_batches(
                shard_rows,
                batch_size=cfg.batch_size,
                shuffle=cfg.shuffle,
                seed=cfg.seed + epoch * 1009 + shard_index,
            ):
                features = torch.as_tensor(
                    np.asarray(arrays["features"][indices], dtype=np.float32),
                    dtype=torch.float32,
                    device=cfg.device,
                )
                targets = _targets_from_arrays(arrays, indices, device=cfg.device)
                optimizer.zero_grad(set_to_none=True)
                output = model(features)
                loss = stpf_training_loss(output, targets, cost_aware_weight=cfg.cost_aware_weight)
                loss.backward()
                if cfg.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
        train_metrics, _ = evaluate_streaming_shards(
            model,
            train_paths,
            cfg,
            epoch=epoch,
            split="train",
            uncertainty_weight=uncertainty_weight,
            max_rows=train_eval_max_rows,
        )
        history.append(train_metrics)
        if validation_paths:
            validation_metrics, _ = evaluate_streaming_shards(
                model,
                validation_paths,
                cfg,
                epoch=epoch,
                split="validation",
                uncertainty_weight=uncertainty_weight,
                max_rows=validation_eval_max_rows,
            )
            history.append(validation_metrics)
    calibration_probe, calibration_details = evaluate_streaming_shards(
        model,
        validation_paths or train_paths,
        cfg,
        epoch=cfg.epochs,
        split="calibration",
        uncertainty_weight=uncertainty_weight,
        max_rows=validation_eval_max_rows,
    )
    min_positive_score = calibration_details["min_positive_score"]
    calibrated_threshold = -1.0e30 if min_positive_score is None else float(min_positive_score)
    thresholds = sorted(set([0.0, 0.25, 0.5, 0.75, 0.9, calibrated_threshold]))
    _, final_details = evaluate_streaming_shards(
        model,
        validation_paths or train_paths,
        cfg,
        epoch=cfg.epochs,
        split="calibration",
        uncertainty_weight=uncertainty_weight,
        thresholds=thresholds,
        group_threshold=calibrated_threshold,
        max_rows=validation_eval_max_rows,
    )
    final_details["calibrated_threshold"] = calibrated_threshold
    final_details["calibration_probe"] = asdict(calibration_probe)
    return model, history, final_details


def _write_history_csv(path: Path, history: Sequence[StreamingEpochMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in history]
    if not rows:
        raise ValueError("cannot write empty training history")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_history_jsonl(path: Path, history: Sequence[StreamingEpochMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in history:
            handle.write(json.dumps(asdict(row), sort_keys=True, separators=(",", ":")) + "\n")


def _latest_metric(history: Sequence[StreamingEpochMetrics], split: str) -> dict[str, Any] | None:
    rows = [item for item in history if item.split == split]
    return None if not rows else asdict(rows[-1])


def run_tight_inclusion_stpf_training(
    shards_dir: Path,
    *,
    run_name: str = "tight_inclusion_nyu_full_run_id",
    report_name: str = "tight_inclusion_stpf_training_run_id",
    output_dir: Path = Path("src/outputs/stpf_training"),
    report_dir: Path = Path("src/benchmark"),
    model_preset: str = "medium_mlp",
    device: str = "cuda",
    epochs: int = 6,
    batch_size: int = 32768,
    learning_rate: float = 8.0e-4,
    train_split: str = "train",
    validation_split: str = "validation",
    train_eval_max_rows: int | None = None,
    validation_eval_max_rows: int | None = None,
    uncertainty_weight: float = 0.25,
) -> dict[str, object]:
    manifest_path = Path(shards_dir) / "manifest.json"
    shard_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_paths = _chunks_for_split(shard_manifest, train_split) or _chunks_for_split(shard_manifest, "unit_smoke")
    validation_paths = (
        _chunks_for_split(shard_manifest, validation_split)
        or _chunks_for_split(shard_manifest, "heldout_test")
        or _chunks_for_split(shard_manifest, "unit_smoke")
    )
    config = STPFTrainingConfig(
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        seed=424242,
        device=device,
        validation_fraction=0.0,
        model_preset=STPFModelPreset(str(model_preset)),
    )
    run_output_dir = Path(output_dir) / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    model, history, calibration = train_stpf_model_from_npz_stream(
        train_paths,
        validation_paths,
        config,
        uncertainty_weight=uncertainty_weight,
        train_eval_max_rows=train_eval_max_rows,
        validation_eval_max_rows=validation_eval_max_rows,
    )
    history_csv = run_output_dir / "history.csv"
    history_jsonl = run_output_dir / "history.jsonl"
    _write_history_csv(history_csv, history)
    _write_history_jsonl(history_jsonl, history)
    model_state_path = run_output_dir / "model_state.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": stpf_config_to_dict(model.config),
            "model_preset": str(config.model_preset),
            "epochs": int(config.epochs),
            "batch_size": int(config.batch_size),
            "seed": int(config.seed),
            "training_runner": "tight_inclusion_npz_stream",
        },
        model_state_path,
    )
    onnx_path: str | None = None
    onnx_error: str | None = None
    try:
        onnx_path = str(
            ensure_stpf_model_onnx(
                model,
                checkpoint_path=model_state_path,
                output_path=run_output_dir / "model.onnx",
                model_tag=run_name,
            )
        )
    except Exception as exc:  # pragma: no cover - environment dependent.
        onnx_error = repr(exc)
    train_row_count = _row_count_for_paths(train_paths)
    validation_row_count = _row_count_for_paths(validation_paths)
    summary = {
        "run_name": run_name,
        "report_name": report_name,
        "shards_dir": Path(shards_dir).as_posix(),
        "train_shards": [path.as_posix() for path in train_paths],
        "validation_shards": [path.as_posix() for path in validation_paths],
        "train_split": train_split,
        "validation_split": validation_split,
        "train_row_count": train_row_count,
        "validation_row_count": validation_row_count,
        "train_eval_max_rows": train_eval_max_rows,
        "validation_eval_max_rows": validation_eval_max_rows,
        "model_preset": str(config.model_preset),
        "device": config.device,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "optimizer": "AdamW",
        "loss": "multitask + cost-aware",
        "final_train": _latest_metric(history, "train"),
        "final_validation": _latest_metric(history, "validation"),
        "history": [asdict(item) for item in history],
        "calibration": calibration,
        "history_csv": history_csv.as_posix(),
        "history_jsonl": history_jsonl.as_posix(),
        "model_state_path": model_state_path.as_posix(),
        "onnx_path": onnx_path,
        "onnx_error": onnx_error,
    }
    summary_path = run_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{report_name}.json"
    md_path = report_dir / f"{report_name}.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(md_path, summary)
    return {**summary, "summary_json": json_path.as_posix(), "report_path": md_path.as_posix()}


def _fmt_optional(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _write_report(path: Path, summary: dict[str, object]) -> None:
    final_train = summary.get("final_train") or {}
    final_validation = summary.get("final_validation") or {}
    calibration = summary.get("calibration") or {}
    lines = [
        "# Tight-Inclusion / NYU Learned STPF Streaming training report",
        "",
        f"- Run name: `{summary['run_name']}`",
        f"- Shards: `{summary['shards_dir']}`",
        f"- Model: `{summary['model_preset']}`",
        f"- Device: `{summary['device']}`",
        f"- Batch size: `{summary['batch_size']}`",
        f"- Epochs: `{summary['epochs']}`",
        f"- Optimizer: `{summary['optimizer']}`",
        f"- Loss: `{summary['loss']}`",
        f"- Model state: `{summary['model_state_path']}`",
        f"- ONNX: `{summary['onnx_path']}`",
        f"- ONNX export error: `{summary['onnx_error']}`",
        f"- Train split: `{summary['train_split']}`",
        f"- Validation split: `{summary['validation_split']}`",
        "",
        "## Dataset",
        "",
        "| Split | Rows | Eval cap | Positive ratio | Interval top1 | Family top2 | Estimated exact-work reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| `train` | `{summary['train_row_count']}` | `{_fmt_optional(summary['train_eval_max_rows'])}` | "
            f"`{final_train.get('positive_ratio', 'n/a')}` | `{final_train.get('interval_top1_recall', 'n/a')}` | "
            f"`{final_train.get('family_top2_recall', 'n/a')}` | `{final_train.get('estimated_exact_work_reduction', 'n/a')}` |"
        ),
        (
            f"| `validation` | `{summary['validation_row_count']}` | `{_fmt_optional(summary['validation_eval_max_rows'])}` | "
            f"`{final_validation.get('positive_ratio', 'n/a')}` | `{final_validation.get('interval_top1_recall', 'n/a')}` | "
            f"`{final_validation.get('family_top2_recall', 'n/a')}` | `{final_validation.get('estimated_exact_work_reduction', 'n/a')}` |"
        ),
        "",
        "## Threshold Calibration",
        "",
        f"- Calibrated zero-FN threshold: `{calibration.get('calibrated_threshold')}`",
        "",
        "| Threshold | TP | TN | FP | FN | Recall | Exact-call reduction | Exact-work reduction |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in calibration.get("threshold_curve", []):
        lines.append(
            f"| `{row['threshold']}` | `{int(row['tp'])}` | `{int(row['tn'])}` | `{int(row['fp'])}` | "
            f"`{int(row['fn'])}` | `{row['recall']}` | `{row['exact_call_reduction']}` | `{row['exact_work_reduction']}` |"
        )
    lines.extend(["", "## Per-Kind Recall", "", "| Kind | Rows | Positives | TP | FN | Recall | Exact-call rate |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for kind, row in sorted(calibration.get("per_kind", {}).items()):
        lines.append(
            f"| `{kind}` | `{row['rows']}` | `{row['positive_count']}` | `{row['tp']}` | `{row['fn']}` | "
            f"`{row['recall']}` | `{row['exact_call_rate']}` |"
        )
    lines.extend(["", "## Per-Case Recall", "", "| Case | Rows | Positives | TP | FN | Recall | Exact-call rate |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for case, row in sorted(calibration.get("per_case", {}).items()):
        lines.append(
            f"| `{case}` | `{row['rows']}` | `{row['positive_count']}` | `{row['tp']}` | `{row['fn']}` | "
            f"`{row['recall']}` | `{row['exact_call_rate']}` |"
        )
    lines.extend(["", "## FN Risk Top-K", "", "| Score | Case | Kind | CSV | Query |", "| ---: | --- | --- | --- | ---: |"])
    for row in calibration.get("fn_risk_top_k", []):
        lines.append(
            f"| `{row['score']}` | `{row['case']}` | `{row['kind']}` | `{row['csv_path']}` | `{row['query_index']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--run-name", default="tight_inclusion_nyu_full_run_id")
    parser.add_argument("--report-name", default="tight_inclusion_stpf_training_run_id")
    parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/stpf_training"))
    parser.add_argument("--report-dir", type=Path, default=Path("src/benchmark"))
    parser.add_argument("--model-preset", default="medium_mlp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--train-eval-max-rows", type=int, default=None)
    parser.add_argument("--validation-eval-max-rows", type=int, default=None)
    parser.add_argument("--uncertainty-weight", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_tight_inclusion_stpf_training(
        args.shards,
        run_name=args.run_name,
        report_name=args.report_name,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        model_preset=args.model_preset,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_split=args.train_split,
        validation_split=args.validation_split,
        train_eval_max_rows=args.train_eval_max_rows,
        validation_eval_max_rows=args.validation_eval_max_rows,
        uncertainty_weight=args.uncertainty_weight,
    )


if __name__ == "__main__":
    main()
