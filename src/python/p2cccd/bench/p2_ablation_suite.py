from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import mean
import time
from typing import Any, Mapping, Sequence

import numpy as np

from p2cccd.contracts import ProxyType
from p2cccd.data.samplers import MotionDiscPairSample, PairFamily
from p2cccd.proposal.features import ProposalFeatureRow
from p2cccd.proposal.inference import batched_stpf_inference
from p2cccd.proposal.ort_inference import (
    batched_stpf_inference_ort_arrays,
    create_ort_inference_session,
    ensure_stpf_model_onnx,
)
from p2cccd.proposal.policy_head_selection import (
    score_rtstpf_candidates,
    select_rtstpf_policy_head,
)
from p2cccd.proposal.stpf_model import STPFModelPreset
from p2cccd.proposal.training import STPFTrainingConfig
from p2cccd.proposal.training_runner import STPFTrainingRunConfig, run_stpf_training

from .bvh_exact import CppOptixBroadPhaseBackend, _try_load_p2cccd_cpp
from .common_modeling_ort_walltime_benchmark import _feature_arrays_from_npz
from .learned_vs_random_ablation import (
    RankChallengeSpec,
    _evaluate_scores,
    _feature_subset,
    _make_rank_challenge_indices,
    _random_summary,
)
from .native_dense_group_benchmark import DEFAULT_CASES, NativeDenseGroupCaseSpec, _load_model, run_native_dense_group_case
from .patch_granularity_ablation import (
    PatchGranularityAblationConfig,
    PatchGranularityAblationOption,
    PatchGranularityAblationRow,
    run_patch_granularity_ablation_on_internal_samples,
)
from .slab_proxy_ablation import (
    SlabProxyAblationConfig,
    SlabProxyAblationRow,
    proxy_family_ablation_options,
    run_slab_proxy_ablation_on_internal_samples,
    slab_count_ablation_options,
)


RUN_DATE = "run_id"
RUN_NAME = f"p2_ablation_suite_{RUN_DATE}"


@dataclass(frozen=True, slots=True)
class TrainingSourceSpec:
    name: str
    train_shard: Path
    eval_shard: Path
    semantic_source: str


TRAINING_SOURCES: tuple[TrainingSourceSpec, ...] = (
    TrainingSourceSpec(
        name="common_modeling_large",
        train_shard=Path(
            "src/datasets/training/common_modeling_high_density/shards/"
            "common_modeling_high_density_scenarios_large_run_id/dense_train.npz"
        ),
        eval_shard=Path(
            "src/datasets/training/common_modeling_high_density/shards/"
            "common_modeling_high_density_scenarios_large_run_id/dense_eval.npz"
        ),
        semantic_source="common graphics / modeling dense contact",
    ),
    TrainingSourceSpec(
        name="fusion360_full_assembly",
        train_shard=Path(
            "src/datasets/training/fusion360_full/shards/"
            "fusion360_full_large_training_run_id/dense_train.npz"
        ),
        eval_shard=Path(
            "src/datasets/training/fusion360_full/shards/"
            "fusion360_full_large_training_run_id/dense_eval.npz"
        ),
        semantic_source="Fusion360-like CAD assembly",
    ),
    TrainingSourceSpec(
        name="rtstpf_advantage_v4",
        train_shard=Path(
            "src/datasets/training/rtstpf_advantage_cases_v4/shards/"
            "rtstpf_advantage_cases_v4_large_training_run_id/dense_train.npz"
        ),
        eval_shard=Path(
            "src/datasets/training/rtstpf_advantage_cases_v4/shards/"
            "rtstpf_advantage_cases_v4_large_training_run_id/dense_eval.npz"
        ),
        semantic_source="hand-built dense advantage / hard-negative scenes",
    ),
    TrainingSourceSpec(
        name="shapenet_ood_dense",
        train_shard=Path(
            "src/datasets/training/shapenet_ood_dense_cases/shards/"
            "shapenet_ood_dense_cases_run_id/dense_train.npz"
        ),
        eval_shard=Path(
            "src/datasets/training/shapenet_ood_dense_cases/shards/"
            "shapenet_ood_dense_cases_run_id/dense_eval.npz"
        ),
        semantic_source="ShapeNet-like OOD dense mesh cases",
    ),
)


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_markdown_table(lines: list[str], rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = [_format_cell(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "N/A"
        if abs(value) >= 1000.0:
            return f"{value:.1f}"
        return f"{value:.4f}"
    return str(value)


def _mean(values: Sequence[float]) -> float:
    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    return float(mean(cleaned)) if cleaned else 0.0


def _rows_from_npz_sample(
    path: Path,
    *,
    limit: int,
    seed: int,
    drop_hard_negatives: bool = False,
) -> list[ProposalFeatureRow]:
    rng = np.random.default_rng(int(seed))
    with np.load(path, allow_pickle=False) as chunk:
        ids = np.asarray(chunk["ids"])
        features = np.asarray(chunk["features"], dtype=np.float32)
        interval_targets = np.asarray(chunk["interval_targets"], dtype=np.float32)
        family_targets = np.asarray(chunk["family_targets"], dtype=np.float32)
        scalar_targets = np.asarray(chunk["scalar_targets"], dtype=np.float32)
        row_count = int(features.shape[0])
        if row_count <= 0:
            raise ValueError(f"empty shard: {path}")
        eligible = np.arange(row_count, dtype=np.int64)
        if drop_hard_negatives:
            priority = scalar_targets[:, 0]
            cost = scalar_targets[:, 1]
            uncertainty = scalar_targets[:, 2]
            proximity_columns = [index for index in (14, 15, 16, 23, 24, 25) if index < features.shape[1]]
            if proximity_columns:
                proximity = np.min(np.abs(features[:, proximity_columns]), axis=1)
            else:
                proximity = np.full((row_count,), 1.0, dtype=np.float32)
            hard_negative = (
                (priority < 0.5)
                & ((cost >= 0.35) | (uncertainty >= 0.12) | (proximity <= 1.0e-3))
            )
            eligible = eligible[~hard_negative]
            if eligible.size == 0:
                raise ValueError(f"hard-negative filter removed every row in {path}")
        take = min(int(limit), int(eligible.size))
        selected = np.sort(rng.choice(eligible, size=take, replace=False))
        rows: list[ProposalFeatureRow] = []
        for index in selected.tolist():
            rows.append(
                ProposalFeatureRow(
                    schema_version=int(ids[index, 0]),
                    query_id=int(ids[index, 1]),
                    candidate_id=int(ids[index, 2]),
                    slab_id=int(ids[index, 3]),
                    object_a_id=int(ids[index, 4]),
                    patch_a_id=int(ids[index, 5]),
                    object_b_id=int(ids[index, 6]),
                    patch_b_id=int(ids[index, 7]),
                    features=[float(value) for value in features[index].tolist()],
                    interval_targets=[float(value) for value in interval_targets[index].tolist()],
                    family_targets=[float(value) for value in family_targets[index].tolist()],
                    priority_target=float(scalar_targets[index, 0]),
                    cost_target=float(scalar_targets[index, 1]),
                    uncertainty_target=float(scalar_targets[index, 2]),
                    target_mask=int(ids[index, 8]),
                )
            )
        return rows


def _torch_inference_arrays(
    model: Any,
    feature_arrays: Mapping[str, np.ndarray | int],
    *,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    import torch

    features = np.asarray(feature_arrays["features"], dtype=np.float32)
    interval_scores = np.empty((features.shape[0], 8), dtype=np.float32)
    family_scores = np.empty((features.shape[0], 8), dtype=np.float32)
    priority_scores = np.empty((features.shape[0],), dtype=np.float32)
    cost_scores = np.empty((features.shape[0],), dtype=np.float32)
    uncertainty_scores = np.empty((features.shape[0],), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), int(batch_size)):
            end = min(int(features.shape[0]), start + int(batch_size))
            batch = torch.as_tensor(features[start:end], dtype=torch.float32, device=device)
            output = model(batch)
            interval_scores[start:end, :] = torch.softmax(output.interval_logits, dim=-1).detach().cpu().numpy()
            family_scores[start:end, :] = torch.sigmoid(output.family_logits).detach().cpu().numpy()
            priority_scores[start:end] = output.priority_score.detach().cpu().numpy().reshape(-1)
            cost_scores[start:end] = output.cost_score.detach().cpu().numpy().reshape(-1)
            uncertainty_scores[start:end] = output.uncertainty_score.detach().cpu().numpy().reshape(-1)
    return {
        "interval_scores": np.ascontiguousarray(interval_scores),
        "family_scores": np.ascontiguousarray(family_scores),
        "priority_score": np.ascontiguousarray(priority_scores),
        "cost_score": np.ascontiguousarray(cost_scores),
        "uncertainty_score": np.ascontiguousarray(uncertainty_scores),
        "ood_mask": np.zeros((features.shape[0],), dtype=np.bool_),
    }


def _evaluate_trained_variant_on_source(
    *,
    variant_name: str,
    model: Any,
    eval_source: TrainingSourceSpec,
    batch_size: int,
    eval_spec: RankChallengeSpec,
    random_seed_count: int,
    inference_device: str,
) -> dict[str, Any]:
    feature_arrays = _feature_arrays_from_npz(eval_source.eval_shard)
    indices, remixed_group_ids = _make_rank_challenge_indices(feature_arrays, eval_spec)
    selected_features = _feature_subset(feature_arrays, indices)
    selected_features["query_id"] = np.ascontiguousarray(remixed_group_ids, dtype=np.uint64)
    trace = np.asarray(selected_features["oracle_trace"], dtype=np.float64)
    positive = np.ascontiguousarray(trace[:, 0] > 0.5)
    full_cost = np.ascontiguousarray(np.maximum(trace[:, 5], 1.0e-12), dtype=np.float64)
    started = time.perf_counter()
    predictions = _torch_inference_arrays(
        model,
        selected_features,
        batch_size=batch_size,
        device=inference_device,
    )
    inference_ms = (time.perf_counter() - started) * 1000.0
    density = float(positive.shape[0]) / float(eval_spec.max_groups)
    selected_head = select_rtstpf_policy_head(
        eval_source.name,
        candidate_density=density,
        hard_negative_group=False,
    )
    scores = score_rtstpf_candidates(predictions, selected_features, head=selected_head.head)
    metrics = _evaluate_scores(
        method=variant_name,
        scores=np.ascontiguousarray(scores, dtype=np.float64),
        positive=positive,
        full_cost=full_cost,
        group_size=int(eval_spec.group_size),
        positives_per_group=int(eval_spec.positives_per_group),
    )
    random_metrics = []
    for seed_offset in range(int(random_seed_count)):
        rng = np.random.default_rng(int(eval_spec.random_seed) + 9973 * (seed_offset + 1))
        random_metrics.append(
            _evaluate_scores(
                method="RandomUniform",
                scores=rng.random(positive.shape[0], dtype=np.float64),
                positive=positive,
                full_cost=full_cost,
                group_size=int(eval_spec.group_size),
                positives_per_group=int(eval_spec.positives_per_group),
            )
        )
    random_summary = _random_summary(random_metrics)
    random_work = float(random_summary["scheduled_exact_work_mean"])
    win_rate = float(
        np.mean(
            [
                1.0 if float(metrics.scheduled_exact_work) < float(item.scheduled_exact_work) else 0.0
                for item in random_metrics
            ]
        )
    )
    out = {
        **asdict(metrics),
        "variant": variant_name,
        "eval_source": eval_source.name,
        "semantic_source": eval_source.semantic_source,
        "policy_head": str(selected_head.head),
        "policy_reason": selected_head.reason,
        "inference_ms": inference_ms,
        "random_scheduled_exact_work_mean": random_work,
        "random_scheduled_exact_calls_mean": float(random_summary["scheduled_exact_calls_mean"]),
        "speedup_vs_random_work": random_work / max(1.0e-12, float(metrics.scheduled_exact_work)),
        "win_rate_vs_random": win_rate,
    }
    return out


def run_training_source_ablation(
    *,
    output_dir: Path,
    train_epochs: int,
    train_base_per_source: int,
    train_batch_size: int,
    eval_batch_size: int,
    eval_max_groups: int,
    random_seed_count: int,
    training_device: str,
    inference_device: str,
) -> dict[str, Any]:
    eval_spec = RankChallengeSpec(max_groups=int(eval_max_groups), random_seed=424242)
    validation_rows: list[ProposalFeatureRow] = []
    for source_index, source in enumerate(TRAINING_SOURCES):
        validation_rows.extend(
            _rows_from_npz_sample(
                source.eval_shard,
                limit=max(128, int(train_base_per_source) // 4),
                seed=4500 + source_index,
            )
        )

    variants: list[dict[str, Any]] = [
        {
            "name": "full_mixed_25pct",
            "sources": TRAINING_SOURCES,
            "rows_per_source": max(64, int(round(train_base_per_source * 0.25))),
            "drop_hard_negatives": False,
            "train_fraction": 0.25,
        },
        {
            "name": "full_mixed_50pct",
            "sources": TRAINING_SOURCES,
            "rows_per_source": max(64, int(round(train_base_per_source * 0.50))),
            "drop_hard_negatives": False,
            "train_fraction": 0.50,
        },
        {
            "name": "full_mixed_100pct",
            "sources": TRAINING_SOURCES,
            "rows_per_source": int(train_base_per_source),
            "drop_hard_negatives": False,
            "train_fraction": 1.00,
        },
        {
            "name": "no_hard_negative_100pct",
            "sources": TRAINING_SOURCES,
            "rows_per_source": int(train_base_per_source),
            "drop_hard_negatives": True,
            "train_fraction": 1.00,
        },
    ]
    for left_out in TRAINING_SOURCES:
        variants.append(
            {
                "name": f"leave_out_{left_out.name}",
                "sources": tuple(source for source in TRAINING_SOURCES if source.name != left_out.name),
                "left_out": left_out.name,
                "rows_per_source": int(math.ceil(train_base_per_source * len(TRAINING_SOURCES) / 3.0)),
                "drop_hard_negatives": False,
                "train_fraction": 1.00,
            }
        )

    rows: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []
    training_root = Path("src/outputs/stpf_training") / f"p2_training_source_ablation_{RUN_DATE}"
    for variant_index, variant in enumerate(variants):
        train_rows: list[ProposalFeatureRow] = []
        source_names: list[str] = []
        for source_index, source in enumerate(variant["sources"]):
            source_names.append(source.name)
            train_rows.extend(
                _rows_from_npz_sample(
                    source.train_shard,
                    limit=int(variant["rows_per_source"]),
                    seed=9700 + 101 * variant_index + source_index,
                    drop_hard_negatives=bool(variant["drop_hard_negatives"]),
                )
            )
        run = run_stpf_training(
            train_rows,
            STPFTrainingRunConfig(
                training=STPFTrainingConfig(
                    epochs=int(train_epochs),
                    batch_size=int(train_batch_size),
                    learning_rate=1.0e-3,
                    weight_decay=1.0e-4,
                    seed=424242 + variant_index,
                    device=str(training_device),
                    validation_fraction=0.0,
                    shuffle=True,
                    grad_clip_norm=1.0,
                    cost_aware_weight=0.25,
                    model_preset=STPFModelPreset.LIGHTWEIGHT_MLP,
                ),
                output_dir=str(training_root),
                run_name=str(variant["name"]),
            ),
            validation_rows=validation_rows,
        )
        model = run.result.model
        model.to(inference_device)
        model.eval()
        per_eval: list[dict[str, Any]] = []
        for eval_source in TRAINING_SOURCES:
            eval_row = _evaluate_trained_variant_on_source(
                variant_name=str(variant["name"]),
                model=model,
                eval_source=eval_source,
                batch_size=int(eval_batch_size),
                eval_spec=eval_spec,
                random_seed_count=int(random_seed_count),
                inference_device=inference_device,
            )
            eval_row.update(
                {
                    "train_sources": ";".join(source_names),
                    "left_out": variant.get("left_out", ""),
                    "drop_hard_negatives": bool(variant["drop_hard_negatives"]),
                    "train_fraction": float(variant["train_fraction"]),
                    "train_row_count": int(run.result.train_row_count),
                    "validation_row_count": int(run.result.validation_row_count),
                    "final_train_loss": float(run.final_train_loss),
                    "final_validation_loss": float(run.final_validation_loss),
                    "model_state": run.artifacts.model_state_path,
                }
            )
            rows.append(eval_row)
            per_eval.append(eval_row)
        variant_summaries.append(
            {
                "variant": variant["name"],
                "train_sources": ";".join(source_names),
                "left_out": variant.get("left_out", ""),
                "drop_hard_negatives": bool(variant["drop_hard_negatives"]),
                "train_row_count": int(run.result.train_row_count),
                "final_train_loss": float(run.final_train_loss),
                "final_validation_loss": float(run.final_validation_loss),
                "mean_exact_work_reduction": _mean([float(row["exact_work_reduction"]) for row in per_eval]),
                "mean_speedup_vs_random_work": _mean([float(row["speedup_vs_random_work"]) for row in per_eval]),
                "mean_first_positive_rank": _mean([float(row["first_positive_rank_mean"]) for row in per_eval]),
                "mean_win_rate_vs_random": _mean([float(row["win_rate_vs_random"]) for row in per_eval]),
                "max_fn": max(int(row["fn"]) for row in per_eval),
            }
        )

    payload = {
        "run_name": f"p2_training_source_ablation_{RUN_DATE}",
        "date": RUN_DATE,
        "scope": "P2-1 sampled retraining over four current dense sources; evaluation uses balanced hard-negative rank challenges.",
        "training_device": training_device,
        "inference_device": inference_device,
        "train_epochs": int(train_epochs),
        "train_base_per_source": int(train_base_per_source),
        "eval_rank_challenge": asdict(eval_spec),
        "random_seed_count": int(random_seed_count),
        "sources": [asdict(source) for source in TRAINING_SOURCES],
        "variant_summaries": variant_summaries,
        "rows": rows,
    }
    _write_json(output_dir / "p2_training_source_ablation_run_id.json", payload)
    _write_csv(output_dir / "p2_training_source_ablation_run_id.csv", rows)
    _write_training_source_report(output_dir / "p2_training_source_ablation_run_id.md", payload)
    return payload


def _mass_from_radius(radius: float) -> float:
    return 4.0 / 3.0 * math.pi * float(radius) ** 3


def _sample(
    *,
    sample_id: int,
    query_id: int,
    split: str,
    family: PairFamily,
    center_a_t0: tuple[float, float, float],
    center_a_t1: tuple[float, float, float],
    center_b_t0: tuple[float, float, float],
    center_b_t1: tuple[float, float, float],
    radius_a: float,
    radius_b: float,
    proxy_type_a: ProxyType,
    proxy_type_b: ProxyType,
    hardness: float,
    object_a_id: int,
    object_b_id: int,
) -> MotionDiscPairSample:
    return MotionDiscPairSample(
        sample_id=sample_id,
        query_id=query_id,
        candidate_id=sample_id + 900000,
        split=split,
        family=family,
        object_a_id=object_a_id,
        patch_a_id=1 + (sample_id % 17),
        object_b_id=object_b_id,
        patch_b_id=101 + (sample_id % 19),
        slab_id=sample_id % 8,
        center_a_t0=center_a_t0,
        center_a_t1=center_a_t1,
        center_b_t0=center_b_t0,
        center_b_t1=center_b_t1,
        radius_a=float(radius_a),
        radius_b=float(radius_b),
        proxy_type_a=proxy_type_a,
        proxy_type_b=proxy_type_b,
        hardness=float(hardness),
        ood=family is PairFamily.ROBOT_LINK_PAIR,
        mass_a=_mass_from_radius(radius_a),
        mass_b=_mass_from_radius(radius_b),
        restitution=0.45 if split.startswith("car_wall") else 0.70,
    )


def _synthetic_patch_slab_scenes() -> dict[str, tuple[MotionDiscPairSample, ...]]:
    rng = np.random.default_rng(424242)
    scenes: dict[str, list[MotionDiscPairSample]] = {
        "car_wall_local_refinement": [],
        "standard_graphics_dense_contact": [],
        "real_mesh_contact_proxy": [],
    }
    sample_id = 1
    for index in range(96):
        y = float(rng.normal(0.0, 0.22))
        z = float(rng.normal(0.0, 0.08))
        radius_a = float(rng.uniform(0.20, 0.42))
        radius_b = float(rng.uniform(0.18, 0.34))
        combined = radius_a + radius_b
        near = index % 4 != 0
        end_gap = combined - float(rng.uniform(0.02, 0.20)) if near else combined + float(rng.uniform(0.06, 0.28))
        scenes["car_wall_local_refinement"].append(
            _sample(
                sample_id=sample_id,
                query_id=10 + index,
                split="car_wall_positive" if near else "car_wall_near_miss",
                family=PairFamily.MESH_PAIR,
                center_a_t0=(-4.0, y, z),
                center_a_t1=(-end_gap, y + float(rng.normal(0.0, 0.04)), z),
                center_b_t0=(0.0, y + float(rng.normal(0.0, 0.05)), z),
                center_b_t1=(0.0, y + float(rng.normal(0.0, 0.05)), z),
                radius_a=radius_a,
                radius_b=radius_b,
                proxy_type_a=ProxyType.CAPSULE,
                proxy_type_b=ProxyType.SWEPT_AABB,
                hardness=0.85 if near else 0.45,
                object_a_id=10,
                object_b_id=20,
            )
        )
        sample_id += 1
    for index in range(120):
        radius_a = float(rng.uniform(0.12, 0.55))
        radius_b = float(rng.uniform(0.12, 0.45))
        combined = radius_a + radius_b
        x = float(rng.normal(0.0, 0.6))
        y = float(rng.normal(0.0, 0.4))
        positive = index % 3 != 0
        end_height = combined - float(rng.uniform(0.01, 0.18)) if positive else combined + float(rng.uniform(0.03, 0.35))
        scenes["standard_graphics_dense_contact"].append(
            _sample(
                sample_id=sample_id,
                query_id=200 + index,
                split="graphics_drop_positive" if positive else "graphics_drop_near_miss",
                family=PairFamily.MESH_PAIR,
                center_a_t0=(x, y, 3.0 + float(rng.uniform(0.0, 1.0))),
                center_a_t1=(x + float(rng.normal(0.0, 0.12)), y + float(rng.normal(0.0, 0.12)), end_height),
                center_b_t0=(x, y, 0.0),
                center_b_t1=(x, y, 0.0),
                radius_a=radius_a,
                radius_b=radius_b,
                proxy_type_a=ProxyType.SWEPT_AABB,
                proxy_type_b=ProxyType.SWEPT_AABB,
                hardness=0.75 if positive else 0.40,
                object_a_id=30 + (index % 4),
                object_b_id=40,
            )
        )
        sample_id += 1
    for index in range(128):
        radius_a = float(rng.uniform(0.18, 0.85))
        radius_b = float(rng.uniform(0.15, 0.70))
        combined = radius_a + radius_b
        positive = index % 5 != 0
        lateral = float(rng.normal(0.0, 0.18))
        end_gap = combined - float(rng.uniform(0.015, 0.22)) if positive else combined + float(rng.uniform(0.04, 0.32))
        scenes["real_mesh_contact_proxy"].append(
            _sample(
                sample_id=sample_id,
                query_id=400 + index,
                split="real_mesh_dense_positive" if positive else "real_mesh_dense_near_miss",
                family=PairFamily.ROBOT_LINK_PAIR if index % 4 == 0 else PairFamily.MESH_PAIR,
                center_a_t0=(-2.5, lateral, float(rng.normal(0.0, 0.3))),
                center_a_t1=(0.25, lateral + float(rng.normal(0.0, 0.12)), float(rng.normal(0.0, 0.3))),
                center_b_t0=(end_gap, 0.0, 0.0),
                center_b_t1=(end_gap + float(rng.normal(0.0, 0.04)), 0.0, 0.0),
                radius_a=radius_a,
                radius_b=radius_b,
                proxy_type_a=ProxyType.CAPSULE,
                proxy_type_b=ProxyType.CAPSULE if index % 4 == 0 else ProxyType.SWEPT_AABB,
                hardness=0.90 if positive else 0.50,
                object_a_id=50 + (index % 9),
                object_b_id=70 + (index % 7),
            )
        )
        sample_id += 1
    return {key: tuple(value) for key, value in scenes.items()}


def _patch_row_to_dict(scene_name: str, row: PatchGranularityAblationRow) -> dict[str, Any]:
    return {
        "mode": "patch_granularity",
        "scene": scene_name,
        **asdict(row),
        "exact_work_proxy": float(row.raw_hit_count),
    }


def _slab_row_to_dict(scene_name: str, row: SlabProxyAblationRow, *, mode: str) -> dict[str, Any]:
    out = {
        "mode": mode,
        "scene": scene_name,
        **asdict(row),
        "exact_work_proxy": float(row.raw_hit_count),
    }
    out["proxy_type_a"] = str(row.proxy_type_a.name)
    out["proxy_type_b"] = str(row.proxy_type_b.name)
    return out


def run_patch_slab_proxy_ablation(*, output_dir: Path) -> dict[str, Any]:
    backend = CppOptixBroadPhaseBackend(name="optix_rt", allow_cpu_fallback=True)
    scenes = _synthetic_patch_slab_scenes()
    patch_config = PatchGranularityAblationConfig(
        options=(
            PatchGranularityAblationOption("patches1_conservative", 1, radius_scale=1.00, offset_scale=0.0),
            PatchGranularityAblationOption("patches2_local", 2, radius_scale=1.00, offset_scale=0.18),
            PatchGranularityAblationOption("patches4_local", 4, radius_scale=1.00, offset_scale=0.24),
            PatchGranularityAblationOption("patches8_fine", 8, radius_scale=1.00, offset_scale=0.30),
        ),
        backend_name=backend.name,
        min_candidate_recall=1.0,
    )
    slab_config = SlabProxyAblationConfig(
        options=slab_count_ablation_options((1, 2, 4, 8), proxy_type=ProxyType.SWEPT_AABB),
        backend_name=backend.name,
        min_candidate_recall=1.0,
    )
    proxy_config = SlabProxyAblationConfig(
        options=proxy_family_ablation_options(slab_count=4),
        backend_name=backend.name,
        min_candidate_recall=1.0,
    )

    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for scene_name, samples in scenes.items():
        patch_result = run_patch_granularity_ablation_on_internal_samples(
            samples,
            patch_config,
            backend=backend,
            source_name="p2_proxy_scene_synthetic",
            scene_name=scene_name,
            batch_id=f"{scene_name}_patch",
        )
        for row in patch_result.rows:
            row_dict = _patch_row_to_dict(scene_name, row)
            rows.append(row_dict)
            if row.selected:
                selected_rows.append(row_dict)
        slab_result = run_slab_proxy_ablation_on_internal_samples(
            samples,
            slab_config,
            backend=backend,
            source_name="p2_proxy_scene_synthetic",
            scene_name=scene_name,
            batch_id=f"{scene_name}_slab",
        )
        for row in slab_result.rows:
            row_dict = _slab_row_to_dict(scene_name, row, mode="slab_count")
            rows.append(row_dict)
            if row.selected:
                selected_rows.append(row_dict)
        proxy_result = run_slab_proxy_ablation_on_internal_samples(
            samples,
            proxy_config,
            backend=backend,
            source_name="p2_proxy_scene_synthetic",
            scene_name=scene_name,
            batch_id=f"{scene_name}_proxy",
        )
        for row in proxy_result.rows:
            row_dict = _slab_row_to_dict(scene_name, row, mode="proxy_family")
            rows.append(row_dict)
            if row.selected:
                selected_rows.append(row_dict)

    payload = {
        "run_name": f"p2_patch_slab_proxy_native_{RUN_DATE}",
        "date": RUN_DATE,
        "scope": (
            "P2-2 production proxy-path ablation. Candidate generation uses the C++/OptiX "
            "proxy-scene wrapper when available and CPU fallback only if the binding cannot run; "
            "correctness is checked against the analytic swept-sphere oracle."
        ),
        "backend_requested": backend.name,
        "scene_count": len(scenes),
        "rows": rows,
        "selected_rows": selected_rows,
        "max_selected_fn": max(int(row.get("fn_count", 0)) for row in selected_rows) if selected_rows else 0,
        "min_selected_recall": min(float(row.get("candidate_recall", 0.0)) for row in selected_rows)
        if selected_rows
        else 0.0,
    }
    _write_json(output_dir / "p2_patch_slab_proxy_native_run_id.json", payload)
    _write_csv(output_dir / "p2_patch_slab_proxy_native_run_id.csv", rows)
    _write_patch_slab_report(output_dir / "p2_patch_slab_proxy_native_run_id.md", payload)
    return payload


def _sample_rows_for_backend_reference(path: Path, *, limit: int, seed: int) -> list[ProposalFeatureRow]:
    return _rows_from_npz_sample(path, limit=limit, seed=seed, drop_hard_negatives=False)


def _backend_inference_case(
    *,
    spec: NativeDenseGroupCaseSpec,
    model: Any,
    onnx_path: Path,
    feature_arrays: Mapping[str, np.ndarray | int],
    backend_name: str,
    batch_size: int,
    device: str,
    prefer_tensorrt: bool,
) -> dict[str, Any]:
    runtime = create_ort_inference_session(
        onnx_path,
        requested_device=device,
        prefer_tensorrt=prefer_tensorrt,
        allow_cuda_fallback=True,
        allow_cpu_fallback=True,
    )
    started = time.perf_counter()
    batched_stpf_inference_ort_arrays(
        runtime,
        feature_arrays,
        batch_size=batch_size,
        ood_abs_feature_threshold=None,
    )
    inference_ms = (time.perf_counter() - started) * 1000.0
    row_count = int(np.asarray(feature_arrays["features"]).shape[0])
    return {
        "case": spec.name,
        "backend": backend_name,
        "scope": "full_dense_eval",
        "requested_device": device,
        "prefer_tensorrt": bool(prefer_tensorrt),
        "provider_actual": runtime.provider_name,
        "provider_order": ";".join(runtime.provider_order),
        "row_count": row_count,
        "proposal_ms": inference_ms,
        "schedule_ms": 0.0,
        "total_detection_ms": inference_ms,
        "rows_per_second": 1000.0 * row_count / max(1.0e-12, inference_ms),
        "fn": "",
        "exact_calls": "",
        "notes": "proposal-only timing",
    }


def run_backend_stack_ablation(
    *,
    output_dir: Path,
    device: str,
    batch_size: int,
    python_reference_limit: int,
) -> dict[str, Any]:
    import torch

    cpp = _try_load_p2cccd_cpp()
    rows: list[dict[str, Any]] = []
    for case_index, spec in enumerate(DEFAULT_CASES[:2]):
        model = _load_model(spec.checkpoint, device=device)
        actual_torch_device = str(next(model.parameters()).device)
        onnx_path = ensure_stpf_model_onnx(
            model,
            checkpoint_path=spec.checkpoint,
            output_path=spec.checkpoint.with_suffix(".onnx"),
            model_tag=spec.checkpoint.parent.name,
        )
        feature_arrays = _feature_arrays_from_npz(spec.dense_shard)
        row_count = int(np.asarray(feature_arrays["features"]).shape[0])

        sample_rows = _sample_rows_for_backend_reference(
            spec.dense_shard,
            limit=int(python_reference_limit),
            seed=33000 + case_index,
        )
        started = time.perf_counter()
        batched_stpf_inference(
            model,
            sample_rows,
            batch_size=min(int(batch_size), int(python_reference_limit)),
            device=actual_torch_device,
            ood_abs_feature_threshold=None,
        )
        python_ms = (time.perf_counter() - started) * 1000.0
        rows.append(
            {
                "case": spec.name,
                "backend": "python_rows_torch_reference",
                "scope": "sampled_reference",
                "requested_device": device,
                "provider_actual": f"torch_{actual_torch_device}",
                "row_count": int(len(sample_rows)),
                "proposal_ms": python_ms,
                "schedule_ms": 0.0,
                "total_detection_ms": python_ms,
                "rows_per_second": 1000.0 * len(sample_rows) / max(1.0e-12, python_ms),
                "fn": "",
                "exact_calls": "",
                "notes": "Python row objects + Torch reference; sampled to avoid conflating object materialization with full dense timing.",
            }
        )

        model.to("cpu")
        started = time.perf_counter()
        _torch_inference_arrays(model, feature_arrays, batch_size=int(batch_size), device="cpu")
        torch_ms = (time.perf_counter() - started) * 1000.0
        rows.append(
            {
                "case": spec.name,
                "backend": "torch_array_cpu",
                "scope": "full_dense_eval",
                "requested_device": "cpu",
                "provider_actual": f"torch_cpu_cuda_available={torch.cuda.is_available()}",
                "row_count": row_count,
                "proposal_ms": torch_ms,
                "schedule_ms": 0.0,
                "total_detection_ms": torch_ms,
                "rows_per_second": 1000.0 * row_count / max(1.0e-12, torch_ms),
                "fn": "",
                "exact_calls": "",
                "notes": "array inference only",
            }
        )

        rows.append(
            _backend_inference_case(
                spec=spec,
                model=model,
                onnx_path=onnx_path,
                feature_arrays=feature_arrays,
                backend_name="ort_cpu",
                batch_size=int(batch_size),
                device="cpu",
                prefer_tensorrt=False,
            )
        )
        rows.append(
            _backend_inference_case(
                spec=spec,
                model=model,
                onnx_path=onnx_path,
                feature_arrays=feature_arrays,
                backend_name="ort_cuda",
                batch_size=int(batch_size),
                device=device,
                prefer_tensorrt=False,
            )
        )
        rows.append(
            _backend_inference_case(
                spec=spec,
                model=model,
                onnx_path=onnx_path,
                feature_arrays=feature_arrays,
                backend_name="ort_tensorrt_preferred",
                batch_size=int(batch_size),
                device=device,
                prefer_tensorrt=True,
            )
        )

        native = run_native_dense_group_case(
            spec,
            device=device,
            batch_size=int(batch_size),
            warmup_passes=0,
        )
        rows.append(
            {
                "case": spec.name,
                "backend": "ort_cuda_plus_cpp_scheduler",
                "scope": "full_dense_eval",
                "requested_device": device,
                "prefer_tensorrt": True,
                "provider_actual": native.get("ort_provider", ""),
                "provider_order": ";".join(native.get("provider_order", [])),
                "row_count": int(native["row_count"]),
                "proposal_ms": float(native["ort_inference_ms"]),
                "schedule_ms": float(native["cxx_call_ms"]),
                "total_detection_ms": float(native["e2e_rtstpf_ms"]),
                "rows_per_second": float(native["proposal_rows_per_second"]),
                "fn": int(native.get("fn", 0)),
                "exact_calls": int(native.get("learned_exact_calls", 0)),
                "no_proposal_exact_calls": int(native.get("no_proposal_exact_calls", 0)),
                "exact_work_reduction": float(native.get("exact_work_reduction", 0.0)),
                "notes": "full ORT proposal + C++ dense-group exact early-stop replay",
            }
        )

    payload = {
        "run_name": f"p2_backend_stack_ablation_{RUN_DATE}",
        "date": RUN_DATE,
        "scope": "P2-3 backend stack timing on common_modeling_large and fusion360_full_assembly dense eval shards.",
        "cpp_binding_available": cpp is not None,
        "device_requested": device,
        "batch_size": int(batch_size),
        "python_reference_limit": int(python_reference_limit),
        "rows": rows,
    }
    _write_json(output_dir / "p2_backend_stack_ablation_run_id.json", payload)
    _write_csv(output_dir / "p2_backend_stack_ablation_run_id.csv", rows)
    _write_backend_report(output_dir / "p2_backend_stack_ablation_run_id.md", payload)
    return payload


def _write_training_source_report(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# P2-1 Training-source Ablation",
        "",
        "## Scope",
        "",
        str(payload["scope"]),
        "",
        "The run retrains lightweight STPF variants from sampled rows because this workstation exposes a CPU-only PyTorch build; ORT/C++ paths are still evaluated in the other P2 parts.",
        "",
        "## Variant Summary",
        "",
    ]
    _write_markdown_table(
        lines,
        payload["variant_summaries"],
        (
            "variant",
            "train_row_count",
            "left_out",
            "drop_hard_negatives",
            "mean_exact_work_reduction",
            "mean_speedup_vs_random_work",
            "mean_first_positive_rank",
            "mean_win_rate_vs_random",
            "max_fn",
        ),
    )
    lines.extend(["", "## Per-source Rows", ""])
    _write_markdown_table(
        lines,
        payload["rows"],
        (
            "variant",
            "eval_source",
            "scheduled_exact_calls",
            "scheduled_exact_work",
            "exact_work_reduction",
            "first_positive_rank_mean",
            "speedup_vs_random_work",
            "win_rate_vs_random",
            "fn",
        ),
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_patch_slab_report(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# P2-2 Patch / Slab / Proxy Ablation",
        "",
        "## Scope",
        "",
        str(payload["scope"]),
        "",
        "## Selected Safe Rows",
        "",
    ]
    _write_markdown_table(
        lines,
        payload["selected_rows"],
        (
            "mode",
            "scene",
            "option_name",
            "candidate_recall",
            "compact_candidate_count",
            "raw_hit_count",
            "proxy_count",
            "total_ms",
            "fn_count",
            "fp_count",
        ),
    )
    lines.extend(["", "## All Rows", ""])
    _write_markdown_table(
        lines,
        payload["rows"],
        (
            "mode",
            "scene",
            "option_name",
            "selected",
            "feasible",
            "candidate_recall",
            "compact_candidate_count",
            "raw_hit_count",
            "proxy_count",
            "total_ms",
            "fn_count",
        ),
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_backend_report(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# P2-3 Backend Stack Ablation",
        "",
        "## Scope",
        "",
        str(payload["scope"]),
        "",
        "## Results",
        "",
    ]
    _write_markdown_table(
        lines,
        payload["rows"],
        (
            "case",
            "backend",
            "scope",
            "provider_actual",
            "row_count",
            "proposal_ms",
            "schedule_ms",
            "total_detection_ms",
            "rows_per_second",
            "fn",
            "exact_calls",
        ),
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_suite_report(path: Path, payload: Mapping[str, Any]) -> None:
    p21 = payload["p2_1"]
    p22 = payload["p2_2"]
    p23 = payload["p2_3"]
    lines = [
        "# P2 Ablation Suite",
        "",
        f"Run identifier: {payload['date']}",
        "",
        "## Artifacts",
        "",
        "- P2-1: `p2_training_source_ablation_run_id.*`",
        "- P2-2: `p2_patch_slab_proxy_native_run_id.*`",
        "- P2-3: `p2_backend_stack_ablation_run_id.*`",
        "",
        "## Guardrails",
        "",
        "- P2-1 is sampled retraining with CPU PyTorch on this workstation.",
        "- P2-2 uses the production C++/OptiX proxy-candidate wrapper where available, with analytic swept-sphere oracle certificates.",
        "- P2-3 reports actual ORT provider names; TensorRT preference is not counted as TensorRT unless the active provider is `TensorrtExecutionProvider`.",
        "",
        "## Headline Checks",
        "",
        f"- P2-1 variants: {len(p21['variant_summaries'])}; per-source rows: {len(p21['rows'])}; max FN: {max(int(row['fn']) for row in p21['rows']) if p21['rows'] else 0}.",
        f"- P2-2 selected safe rows: {len(p22['selected_rows'])}; min selected recall: {float(p22['min_selected_recall']):.4f}; max selected FN: {int(p22['max_selected_fn'])}.",
        f"- P2-3 rows: {len(p23['rows'])}; C++ binding available: {p23['cpp_binding_available']}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_p2_ablation_suite(
    *,
    output_dir: Path,
    device: str,
    training_device: str,
    train_epochs: int,
    train_base_per_source: int,
    train_batch_size: int,
    eval_batch_size: int,
    eval_max_groups: int,
    random_seed_count: int,
    backend_batch_size: int,
    python_reference_limit: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    p2_1 = run_training_source_ablation(
        output_dir=output_dir,
        train_epochs=int(train_epochs),
        train_base_per_source=int(train_base_per_source),
        train_batch_size=int(train_batch_size),
        eval_batch_size=int(eval_batch_size),
        eval_max_groups=int(eval_max_groups),
        random_seed_count=int(random_seed_count),
        training_device=training_device,
        inference_device=training_device,
    )
    p2_2 = run_patch_slab_proxy_ablation(output_dir=output_dir)
    p2_3 = run_backend_stack_ablation(
        output_dir=output_dir,
        device=device,
        batch_size=int(backend_batch_size),
        python_reference_limit=int(python_reference_limit),
    )
    payload = {
        "run_name": RUN_NAME,
        "date": RUN_DATE,
        "output_dir": output_dir,
        "p2_1": p2_1,
        "p2_2": p2_2,
        "p2_3": p2_3,
    }
    _write_json(output_dir / f"{RUN_NAME}.json", payload)
    _write_suite_report(output_dir / f"{RUN_NAME}.md", payload)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the P2 TOG publication-style ablation suite.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/benchmark") / RUN_NAME,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--training-device", default="cpu")
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--train-base-per-source", type=int, default=2048)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--eval-max-groups", type=int, default=512)
    parser.add_argument("--random-seed-count", type=int, default=30)
    parser.add_argument("--backend-batch-size", type=int, default=65536)
    parser.add_argument("--python-reference-limit", type=int, default=65536)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_p2_ablation_suite(
        output_dir=Path(args.output_dir),
        device=str(args.device),
        training_device=str(args.training_device),
        train_epochs=int(args.train_epochs),
        train_base_per_source=int(args.train_base_per_source),
        train_batch_size=int(args.train_batch_size),
        eval_batch_size=int(args.eval_batch_size),
        eval_max_groups=int(args.eval_max_groups),
        random_seed_count=int(args.random_seed_count),
        backend_batch_size=int(args.backend_batch_size),
        python_reference_limit=int(args.python_reference_limit),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
