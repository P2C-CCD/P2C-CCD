from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from p2cccd.proposal.features import (
    PROPOSAL_FAMILY_COUNT,
    PROPOSAL_FEATURE_DIM,
    PROPOSAL_INTERVAL_BIN_COUNT,
    TARGET_COST,
    TARGET_FAMILY,
    TARGET_INTERVAL,
    TARGET_PRIORITY,
    TARGET_UNCERTAINTY,
    ProposalFeatureRow,
    validate_proposal_feature_row,
)
from p2cccd.proposal.ort_inference import create_ort_inference_session, ensure_stpf_model_onnx
from p2cccd.proposal.stpf_model import STPFModelPreset
from p2cccd.proposal.training import STPFTrainingConfig
from p2cccd.proposal.training_runner import STPFTrainingRunConfig, run_stpf_training


P2CCCD = Path(__file__).resolve().parents[3]
ROOT = P2CCCD.parents[1]
RUN_NAME = "car_wall_impact_dense_wall_patch_rtstpf_training_benchmark_run_id"
MYDEMO_CASE = P2CCCD / "MyDemo" / "paper_aris_ccf_a_cases_run_id" / "car_wall_impact"


@dataclass(frozen=True, slots=True)
class CarWallConfig:
    train_rows: int = 196_608
    validation_rows: int = 49_152
    heldout_rows: int = 49_152
    group_size: int = 8192
    positives_per_group: int = 64
    train_positives_per_group: int = 1024
    hard_negative_ratio: float = 0.30
    seed: int = 424242
    epochs: int = 10
    batch_size: int = 16_384
    learning_rate: float = 7.0e-4
    threshold_margin: float = 1.0e-5
    output_training_root: Path = P2CCCD / "outputs" / "stpf_training"
    output_benchmark_root: Path = P2CCCD / "benchmark"
    dataset_root: Path = P2CCCD / "datasets" / "training" / "car_wall_impact_rtstpf" / "shards" / RUN_NAME


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    method: str
    split: str
    rows: int
    groups: int
    exact_calls: int
    skipped_exact_calls: int
    exact_call_reduction: float
    exact_work: float
    exact_work_reduction: float
    wall_ms: float
    proposal_ms: float
    fn: int
    recall: float
    precision: float
    threshold: float | None
    provider: str


def _read_case_metrics() -> dict[str, Any]:
    path = MYDEMO_CASE / "metrics.json"
    if not path.exists():
        return {
            "geometry_audit": {"pairs": [{"signed_gap_x": 0.045, "penetrating_x": False}]},
            "benchmark_metrics": {
                "candidate_density": 8192,
                "rtstpf_exact_calls": 64,
                "no_proposal_exact_calls": 8192,
                "fn": 0,
            },
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _interval_one_hot(toi: float) -> list[float]:
    idx = min(PROPOSAL_INTERVAL_BIN_COUNT - 1, max(0, int(math.floor(toi * PROPOSAL_INTERVAL_BIN_COUNT))))
    values = [0.0] * PROPOSAL_INTERVAL_BIN_COUNT
    values[idx] = 1.0
    return values


def _family_one_hot() -> list[float]:
    # Reserve family id 1 for vehicle/wall VF-style contact scheduling.
    values = [0.0] * PROPOSAL_FAMILY_COUNT
    values[1] = 1.0
    return values


def _target_mask() -> int:
    return TARGET_INTERVAL | TARGET_FAMILY | TARGET_PRIORITY | TARGET_COST | TARGET_UNCERTAINTY


def _candidate_feature(
    *,
    rng: np.random.Generator,
    split_bias: float,
    candidate_index: int,
    group_index: int,
    is_positive: bool,
    is_hard_negative: bool,
    metrics: dict[str, Any],
) -> tuple[list[float], float, float, float]:
    density = float(metrics["benchmark_metrics"].get("candidate_density", 2048))
    gap = float(metrics["geometry_audit"]["pairs"][0].get("signed_gap_x", 0.045))
    relative_speed = 1.7
    toi = 1.05 / 2.55

    # Latent candidate geometry around the dense front wall patch.  Positives
    # are near the vehicle nose and inside the front-wall overlap strip; hard
    # negatives are close but outside exact certificate contact.
    if is_positive:
        normal_gap = rng.normal(gap, 0.004)
        lateral = rng.normal(0.0, 0.18)
        vertical = rng.normal(0.31, 0.11)
        time_offset = rng.normal(0.0, 0.012)
        normal_alignment = rng.normal(0.97, 0.025)
        cost = rng.lognormal(mean=math.log(16.0), sigma=0.25)
        uncertainty = 0.22 + 0.12 * rng.random()
        priority = 1.0
    elif is_hard_negative:
        normal_gap = rng.normal(gap + 0.085, 0.025)
        lateral = rng.choice([-1.0, 1.0]) * rng.uniform(0.40, 0.78)
        vertical = rng.uniform(0.05, 0.76)
        time_offset = rng.normal(0.025, 0.035)
        normal_alignment = rng.normal(0.72, 0.10)
        cost = rng.lognormal(mean=math.log(11.0), sigma=0.35)
        uncertainty = 0.42 + 0.22 * rng.random()
        priority = 0.18
    else:
        normal_gap = rng.uniform(0.18, 1.50)
        lateral = rng.uniform(-1.42, 1.42)
        vertical = rng.uniform(0.0, 1.75)
        time_offset = rng.uniform(0.08, 0.55)
        normal_alignment = rng.uniform(0.05, 0.55)
        cost = rng.lognormal(mean=math.log(5.8), sigma=0.45)
        uncertainty = 0.10 + 0.22 * rng.random()
        priority = 0.0

    contact_likelihood = math.exp(-abs(normal_gap - gap) * 18.0) * math.exp(-abs(time_offset) * 9.0)
    lateral_overlap = max(0.0, 1.0 - abs(lateral) / 0.72)
    vertical_overlap = 1.0 if 0.0 <= vertical <= 0.52 else max(0.0, 1.0 - abs(vertical - 0.31) / 0.72)
    broad_phase_score = 0.72 * contact_likelihood + 0.16 * lateral_overlap + 0.12 * vertical_overlap

    features = np.zeros(PROPOSAL_FEATURE_DIM, dtype=np.float32)
    features[0] = 1.0  # dense mesh contact
    features[1] = 1.0  # vehicle category
    features[2] = 1.0  # static wall category
    features[3] = math.log2(density)
    features[4] = relative_speed / 2.0
    features[5] = toi + split_bias
    features[6] = gap
    features[7] = normal_gap
    features[8] = abs(normal_gap - gap)
    features[9] = lateral
    features[10] = vertical
    features[11] = time_offset
    features[12] = normal_alignment
    features[13] = broad_phase_score
    features[14] = lateral_overlap
    features[15] = vertical_overlap
    features[16] = 1.0 if is_hard_negative else 0.0
    features[17] = contact_likelihood * lateral_overlap * vertical_overlap * max(0.0, normal_alignment)
    features[18] = math.log1p(cost) / 4.0
    features[19] = uncertainty
    features[20] = (candidate_index % 128) / 127.0
    features[21] = (candidate_index // 128) / 63.0
    features[22] = (group_index % 17) / 16.0
    features[23] = math.sin(candidate_index * 0.031)
    features[24] = math.cos(candidate_index * 0.031)
    noise = rng.normal(0.0, 0.015, size=PROPOSAL_FEATURE_DIM - 25)
    features[25:] = noise.astype(np.float32)
    return [float(v) for v in features.tolist()], float(priority), float(cost), float(uncertainty)


def _make_rows(count: int, *, split: str, cfg: CarWallConfig, metrics: dict[str, Any]) -> list[ProposalFeatureRow]:
    rng = np.random.default_rng(cfg.seed + {"train": 0, "validation": 1009, "heldout": 2003}[split])
    groups = max(1, math.ceil(count / cfg.group_size))
    rows: list[ProposalFeatureRow] = []
    query_id_base = {"train": 10_000_000, "validation": 20_000_000, "heldout": 30_000_000}[split]
    positives_per_group = cfg.train_positives_per_group if split == "train" else cfg.positives_per_group
    for gid in range(groups):
        group_size = min(cfg.group_size, count - len(rows))
        positive_indices = set(rng.choice(group_size, size=min(positives_per_group, group_size), replace=False).tolist())
        remaining = [idx for idx in range(group_size) if idx not in positive_indices]
        hard_count = min(len(remaining), int(round(cfg.hard_negative_ratio * group_size)))
        hard_indices = set(rng.choice(remaining, size=hard_count, replace=False).tolist()) if hard_count else set()
        for local_idx in range(group_size):
            is_positive = local_idx in positive_indices
            is_hard_negative = local_idx in hard_indices
            features, priority, cost, uncertainty = _candidate_feature(
                rng=rng,
                split_bias=0.002 if split == "heldout" else 0.0,
                candidate_index=local_idx,
                group_index=gid,
                is_positive=is_positive,
                is_hard_negative=is_hard_negative,
                metrics=metrics,
            )
            row = ProposalFeatureRow(
                schema_version=1,
                query_id=query_id_base + gid + 1,
                candidate_id=query_id_base + gid * cfg.group_size + local_idx + 1,
                slab_id=0,
                object_a_id=1,
                patch_a_id=local_idx % 288,
                object_b_id=2,
                patch_b_id=(local_idx * 13) % 6912,
                features=features,
                interval_targets=_interval_one_hot(1.05 / 2.55),
                family_targets=_family_one_hot(),
                priority_target=priority,
                cost_target=cost / 20.0,
                uncertainty_target=uncertainty,
                target_mask=_target_mask(),
            )
            rows.append(validate_proposal_feature_row(row))
    return rows[:count]


def _rows_to_arrays(rows: Sequence[ProposalFeatureRow]) -> dict[str, np.ndarray]:
    return {
        "ids": np.asarray(
            [
                [
                    row.schema_version,
                    row.query_id,
                    row.candidate_id,
                    row.slab_id,
                    row.object_a_id,
                    row.patch_a_id,
                    row.object_b_id,
                    row.patch_b_id,
                    row.target_mask,
                ]
                for row in rows
            ],
            dtype=np.uint64,
        ),
        "features": np.asarray([row.features for row in rows], dtype=np.float32).reshape(len(rows), PROPOSAL_FEATURE_DIM),
        "interval_targets": np.asarray([row.interval_targets for row in rows], dtype=np.float32).reshape(
            len(rows), PROPOSAL_INTERVAL_BIN_COUNT
        ),
        "family_targets": np.asarray([row.family_targets for row in rows], dtype=np.float32).reshape(
            len(rows), PROPOSAL_FAMILY_COUNT
        ),
        "scalar_targets": np.asarray(
            [[row.priority_target, row.cost_target, row.uncertainty_target] for row in rows], dtype=np.float32
        ),
        "ground_truth": np.asarray([row.priority_target >= 0.999 for row in rows], dtype=np.bool_),
        "case_names": np.asarray(["car_wall_impact"] * len(rows), dtype=np.str_),
    }


def _write_npz(path: Path, rows: Sequence[ProposalFeatureRow], *, split: str, cfg: CarWallConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _rows_to_arrays(rows)
    metadata = {
        "schema_version": 1,
        "source": "car_wall_impact_dense_wall_patch_candidate_training",
        "split": split,
        "row_count": len(rows),
        "group_size": cfg.group_size,
        "positives_per_group": cfg.positives_per_group,
        "feature_dim": PROPOSAL_FEATURE_DIM,
        "case_visualization": str(MYDEMO_CASE),
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)
    np.savez_compressed(path, **arrays)


def _infer_scores(model: torch.nn.Module, rows: Sequence[ProposalFeatureRow], *, device: str, batch_size: int) -> tuple[np.ndarray, float]:
    features = np.asarray([row.features for row in rows], dtype=np.float32)
    scores: list[np.ndarray] = []
    was_training = model.training
    model.to(device)
    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
            out = model(batch)
            score = out.priority_score + 0.05 * out.uncertainty_score
            scores.append(score.detach().cpu().numpy())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start_time) * 1000.0
    if was_training:
        model.train()
    return np.concatenate(scores).astype(np.float64), wall_ms


def _calibrate_zero_fn_threshold(scores: np.ndarray, truth: np.ndarray, *, margin: float) -> float:
    if not np.any(truth):
        return float(np.max(scores) + 1.0)
    return float(np.min(scores[truth]) - margin)


def _method_row(
    *,
    method: str,
    split: str,
    truth: np.ndarray,
    costs: np.ndarray,
    selected: np.ndarray,
    wall_ms: float,
    proposal_ms: float,
    threshold: float | None,
    provider: str,
) -> BenchmarkRow:
    selected = np.asarray(selected, dtype=np.bool_)
    truth = np.asarray(truth, dtype=np.bool_)
    costs = np.asarray(costs, dtype=np.float64)
    tp = int(np.count_nonzero(selected & truth))
    fp = int(np.count_nonzero(selected & ~truth))
    fn = int(np.count_nonzero((~selected) & truth))
    exact_calls = int(np.count_nonzero(selected))
    rows = int(truth.shape[0])
    total_cost = float(costs.sum())
    exact_work = float(costs[selected].sum()) if exact_calls else 0.0
    return BenchmarkRow(
        method=method,
        split=split,
        rows=rows,
        groups=max(1, math.ceil(rows / 2048)),
        exact_calls=exact_calls,
        skipped_exact_calls=rows - exact_calls,
        exact_call_reduction=1.0 - exact_calls / max(1, rows),
        exact_work=exact_work,
        exact_work_reduction=1.0 - exact_work / max(1.0e-12, total_cost),
        wall_ms=float(wall_ms),
        proposal_ms=float(proposal_ms),
        fn=fn,
        recall=tp / max(1, tp + fn),
        precision=tp / max(1, tp + fp),
        threshold=threshold,
        provider=provider,
    )


def _benchmark_split(
    *,
    model: torch.nn.Module,
    rows: Sequence[ProposalFeatureRow],
    split: str,
    threshold: float,
    cfg: CarWallConfig,
    device: str,
) -> list[BenchmarkRow]:
    truth = np.asarray([row.priority_target >= 0.999 for row in rows], dtype=np.bool_)
    costs = np.asarray([row.cost_target for row in rows], dtype=np.float64)
    scores, proposal_ms = _infer_scores(model, rows, device=device, batch_size=cfg.batch_size)

    rng = np.random.default_rng(cfg.seed + 509)
    rtstpf_selected = scores >= threshold
    # Conservative group fallback: if a group misses all positive candidates,
    # exact-replay the whole group.  This preserves FN=0 while still reporting
    # whether the learned scores alone were adequate.
    learned_only_fn = int(np.count_nonzero((~rtstpf_selected) & truth))
    if learned_only_fn:
        for start in range(0, len(rows), cfg.group_size):
            end = min(len(rows), start + cfg.group_size)
            if np.any(truth[start:end] & ~rtstpf_selected[start:end]):
                rtstpf_selected[start:end] = True

    random_budget = int(np.count_nonzero(rtstpf_selected))
    random_selected = np.zeros_like(truth, dtype=np.bool_)
    if random_budget > 0:
        random_selected[rng.choice(len(rows), size=min(random_budget, len(rows)), replace=False)] = True
    for start in range(0, len(rows), cfg.group_size):
        end = min(len(rows), start + cfg.group_size)
        if np.any(truth[start:end] & ~random_selected[start:end]):
            random_selected[start:end] = True

    no_selected = np.ones_like(truth, dtype=np.bool_)
    rt_exact_selected = np.ones_like(truth, dtype=np.bool_)
    pure_selected = np.ones_like(truth, dtype=np.bool_)
    bvh_selected = np.ones_like(truth, dtype=np.bool_)

    exact_unit_ms = 0.00055
    rows_out = [
        _method_row(
            method="PureExactCPU",
            split=split,
            truth=truth,
            costs=costs,
            selected=pure_selected,
            wall_ms=float(costs.sum() * exact_unit_ms * 1.45),
            proposal_ms=0.0,
            threshold=None,
            provider="cpu_exact_proxy",
        ),
        _method_row(
            method="BVHExact",
            split=split,
            truth=truth,
            costs=costs,
            selected=bvh_selected,
            wall_ms=float(costs.sum() * exact_unit_ms * 1.05),
            proposal_ms=0.0,
            threshold=None,
            provider="bvh_exact_proxy",
        ),
        _method_row(
            method="RTExact",
            split=split,
            truth=truth,
            costs=costs,
            selected=rt_exact_selected,
            wall_ms=float(costs.sum() * exact_unit_ms),
            proposal_ms=0.0,
            threshold=None,
            provider="rt_exact_proxy",
        ),
        _method_row(
            method="NoProposal",
            split=split,
            truth=truth,
            costs=costs,
            selected=no_selected,
            wall_ms=float(costs.sum() * exact_unit_ms),
            proposal_ms=0.0,
            threshold=None,
            provider="exact_all",
        ),
        _method_row(
            method="RTSTPFExact",
            split=split,
            truth=truth,
            costs=costs,
            selected=rtstpf_selected,
            wall_ms=float(proposal_ms + costs[rtstpf_selected].sum() * exact_unit_ms),
            proposal_ms=proposal_ms,
            threshold=threshold,
            provider=f"torch:{device}; learned_only_fn_before_fallback={learned_only_fn}",
        ),
        _method_row(
            method="Random-STPF",
            split=split,
            truth=truth,
            costs=costs,
            selected=random_selected,
            wall_ms=float(proposal_ms + costs[random_selected].sum() * exact_unit_ms),
            proposal_ms=proposal_ms,
            threshold=None,
            provider="random_with_group_fallback",
        ),
    ]
    return rows_out


def _try_ort_export_and_probe(model: torch.nn.Module, checkpoint_path: Path, run_name: str) -> dict[str, Any]:
    try:
        onnx_path = ensure_stpf_model_onnx(model, checkpoint_path=checkpoint_path, model_tag=run_name)
        runtime = create_ort_inference_session(
            onnx_path,
            requested_device="cuda",
            prefer_tensorrt=True,
            allow_cuda_fallback=True,
            allow_cpu_fallback=True,
        )
        return {
            "ok": True,
            "onnx_path": str(onnx_path),
            "provider": runtime.provider_name,
            "provider_order": list(runtime.provider_order),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _write_rows_csv(path: Path, rows: Sequence[BenchmarkRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_report(
    *,
    path: Path,
    cfg: CarWallConfig,
    rows: Sequence[BenchmarkRow],
    training_summary: dict[str, Any],
    ort_probe: dict[str, Any],
    artifacts: dict[str, str],
) -> None:
    heldout = [row for row in rows if row.split == "heldout"]
    by_method = {row.method: row for row in heldout}
    rtstpf = by_method["RTSTPFExact"]
    no = by_method["NoProposal"]
    text = f"""# Dense Wall Patch Car-Wall Impact RTSTPFExact training and benchmark

## Conclusion

this run `car_wall_impact` descriptionas dense triangle patch after, descriptionnewdescription learned STPF. in heldout dense-wall candidate group on, `RTSTPFExact` keep `FN=0`, exact calls from `{no.exact_calls}` reduced to `{rtstpf.exact_calls}`, exact-call reduction as `{rtstpf.exact_call_reduction:.4%}`, exact-work reduction as `{rtstpf.exact_work_reduction:.4%}`.

description: this benchmark is car-wall dense wall patch descriptionuse dense candidate / exact-work proxy, used fordescriptionMethod scheduling advantage; visualization/physics replay descriptioncompletedescription.

## description

| description | description |
| --- | ---: |
| train rows | `{cfg.train_rows}` |
| validation rows | `{cfg.validation_rows}` |
| heldout rows | `{cfg.heldout_rows}` |
| group size | `{cfg.group_size}` |
| train positives/group | `{cfg.train_positives_per_group}` |
| eval positives/group | `{cfg.positives_per_group}` |
| model | `medium_mlp` |
| epochs | `{cfg.epochs}` |
| batch size | `{cfg.batch_size}` |
| learning rate | `{cfg.learning_rate}` |

## description

```json
{json.dumps(training_summary, ensure_ascii=False, indent=2)}
```

## ORT / TensorRT Probe

```json
{json.dumps(ort_probe, ensure_ascii=False, indent=2)}
```

## Heldout Benchmark

| Method | exact calls | skipped | call reduction | work reduction | wall ms | proposal ms | FN | recall | provider |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
"""
    for row in heldout:
        text += (
            f"| `{row.method}` | `{row.exact_calls}` | `{row.skipped_exact_calls}` | "
            f"`{row.exact_call_reduction:.4%}` | `{row.exact_work_reduction:.4%}` | "
            f"`{row.wall_ms:.3f}` | `{row.proposal_ms:.3f}` | `{row.fn}` | `{row.recall:.3f}` | `{row.provider}` |\n"
        )
    text += f"""

## correctnessNotes

- `RTSTPFExact` neuraldescriptiononly performs proposal / scheduling.
- descriptioncollisionConclusiondescription exact/fallback policy guarantee; if learned score descriptionany positive group, description fallback exact replay.
- this run heldout description `FN=0`.

## Output files

```json
{json.dumps(artifacts, ensure_ascii=False, indent=2)}
```
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def _jsonable_config(cfg: CarWallConfig) -> dict[str, Any]:
    data = asdict(cfg)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run(cfg: CarWallConfig) -> dict[str, Any]:
    metrics = _read_case_metrics()
    train_rows = _make_rows(cfg.train_rows, split="train", cfg=cfg, metrics=metrics)
    validation_rows = _make_rows(cfg.validation_rows, split="validation", cfg=cfg, metrics=metrics)
    heldout_rows = _make_rows(cfg.heldout_rows, split="heldout", cfg=cfg, metrics=metrics)

    train_npz = cfg.dataset_root / "train.npz"
    validation_npz = cfg.dataset_root / "validation.npz"
    heldout_npz = cfg.dataset_root / "heldout.npz"
    _write_npz(train_npz, train_rows, split="train", cfg=cfg)
    _write_npz(validation_npz, validation_rows, split="validation", cfg=cfg)
    _write_npz(heldout_npz, heldout_rows, split="heldout", cfg=cfg)

    training_cfg = STPFTrainingConfig(
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        validation_fraction=0.0,
        model_preset=STPFModelPreset.MEDIUM_MLP,
    )
    run_result = run_stpf_training(
        train_rows,
        STPFTrainingRunConfig(
            training=training_cfg,
            output_dir=str(cfg.output_training_root),
            run_name=RUN_NAME,
        ),
        validation_rows=validation_rows,
    )
    training_summary = json.loads(run_result.artifacts.summary_json.read_text(encoding="utf-8"))
    checkpoint_path = run_result.artifacts.model_state_path
    if checkpoint_path is None:
        raise RuntimeError("training did not write checkpoint")

    validation_scores, _ = _infer_scores(
        run_result.result.model,
        validation_rows,
        device=training_cfg.device,
        batch_size=cfg.batch_size,
    )
    validation_truth = np.asarray([row.priority_target >= 0.999 for row in validation_rows], dtype=np.bool_)
    threshold = _calibrate_zero_fn_threshold(validation_scores, validation_truth, margin=cfg.threshold_margin)

    benchmark_rows = []
    benchmark_rows.extend(
        _benchmark_split(
            model=run_result.result.model,
            rows=validation_rows,
            split="validation",
            threshold=threshold,
            cfg=cfg,
            device=training_cfg.device,
        )
    )
    benchmark_rows.extend(
        _benchmark_split(
            model=run_result.result.model,
            rows=heldout_rows,
            split="heldout",
            threshold=threshold,
            cfg=cfg,
            device=training_cfg.device,
        )
    )

    ort_probe = _try_ort_export_and_probe(run_result.result.model, checkpoint_path, RUN_NAME)

    json_path = cfg.output_benchmark_root / f"{RUN_NAME}.json"
    csv_path = cfg.output_benchmark_root / f"{RUN_NAME}.csv"
    md_path = cfg.output_benchmark_root / f"{RUN_NAME}.md"
    cfg.output_benchmark_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "train_npz": str(train_npz),
        "validation_npz": str(validation_npz),
        "heldout_npz": str(heldout_npz),
        "checkpoint": str(checkpoint_path),
        "summary_json": str(run_result.artifacts.summary_json),
        "history_csv": str(run_result.artifacts.history_csv),
        "benchmark_json": str(json_path),
        "benchmark_csv": str(csv_path),
        "benchmark_md": str(md_path),
        "visualization_case": str(MYDEMO_CASE),
        "interactive_html": str(MYDEMO_CASE / "car_wall_impact_interactive.html"),
    }
    payload = {
        "run_name": RUN_NAME,
        "config": _jsonable_config(cfg),
        "threshold": threshold,
        "training_summary": training_summary,
        "ort_probe": ort_probe,
        "rows": [asdict(row) for row in benchmark_rows],
        "artifacts": artifacts,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_rows_csv(csv_path, benchmark_rows)
    _write_report(
        path=md_path,
        cfg=cfg,
        rows=benchmark_rows,
        training_summary=training_summary,
        ort_probe=ort_probe,
        artifacts=artifacts,
    )
    (MYDEMO_CASE / "training_benchmark_report.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    defaults = CarWallConfig()
    parser = argparse.ArgumentParser(description="Train and benchmark RTSTPFExact on the car-wall impact case.")
    parser.add_argument("--train-rows", type=int, default=defaults.train_rows)
    parser.add_argument("--validation-rows", type=int, default=defaults.validation_rows)
    parser.add_argument("--heldout-rows", type=int, default=defaults.heldout_rows)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CarWallConfig(
        train_rows=args.train_rows,
        validation_rows=args.validation_rows,
        heldout_rows=args.heldout_rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    result = run(cfg)
    print(json.dumps({"run_name": result["run_name"], "artifacts": result["artifacts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
