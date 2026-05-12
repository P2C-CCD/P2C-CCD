from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "src" / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from p2cccd.proposal.ort_inference import (  # noqa: E402
    batched_stpf_inference_ort_arrays,
    create_ort_inference_session,
    ensure_stpf_model_onnx,
)
from p2cccd.proposal.policy_head_selection import (  # noqa: E402
    RTSTPFPolicyHead,
    score_rtstpf_candidates,
)
from p2cccd.proposal.stpf_model import (  # noqa: E402
    STPFModelPreset,
    build_stpf_model_from_checkpoint_payload,
)


RUN_NAME = "all_dataset_strict_five_path_full_replay_run_id"
DEFAULT_OUTPUT_DIR = Path("src/benchmark") / RUN_NAME
DEFAULT_BATCH_SIZE = 65536
METHODS = ("PureExactCPU", "BVHExact", "RTExact", "RTSTPFExact", "NoProposal")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    shard: Path
    checkpoint: Path
    source: str
    role: str = "full"


GENERALIZATION_FULL = Path(
    "src/outputs/stpf_training/generalization_paper_benchmark_full_run_id/model_state.pt"
)
ADVANTAGE_V4 = Path(
    "src/outputs/stpf_training/rtstpf_advantage_cases_v4_large_training_run_id/model_state.pt"
)


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "T0 synthetic_proxy",
        Path("src/datasets/training/generalization/shards/generalization_paper_benchmark_full_run_id/t0_synthetic_proxy/dense_eval.npz"),
        GENERALIZATION_FULL,
        "generalization_full",
    ),
    DatasetSpec(
        "trained_stpf_high_density",
        Path("src/datasets/training/generalization/shards/generalization_paper_benchmark_full_run_id/trained_stpf_high_density/dense_eval.npz"),
        GENERALIZATION_FULL,
        "generalization_full",
    ),
    DatasetSpec(
        "ABC CAD",
        Path("src/datasets/training/generalization/shards/generalization_paper_benchmark_full_run_id/abc_cad/dense_eval.npz"),
        GENERALIZATION_FULL,
        "generalization_full",
    ),
    DatasetSpec(
        "Thingi10K",
        Path("src/datasets/training/generalization/shards/generalization_paper_benchmark_full_run_id/thingi10k/dense_eval.npz"),
        GENERALIZATION_FULL,
        "generalization_full",
    ),
    DatasetSpec(
        "Fusion 360 Gallery Assembly",
        Path("src/datasets/training/generalization/shards/generalization_paper_benchmark_full_run_id/fusion_360_gallery_assembly/dense_eval.npz"),
        GENERALIZATION_FULL,
        "generalization_full",
    ),
    DatasetSpec(
        "high_density_mesh_multi_source",
        Path("src/datasets/training/generalization/shards/generalization_paper_benchmark_full_run_id/high_density_mesh_multi_source/dense_eval.npz"),
        GENERALIZATION_FULL,
        "generalization_full",
    ),
    DatasetSpec(
        "Fusion360 Gallery Assembly Full",
        Path("src/datasets/training/fusion360_full/shards/fusion360_full_large_training_run_id/dense_eval.npz"),
        Path("src/outputs/stpf_training/fusion360_full_large_training_run_id/model_state.pt"),
        "fusion360_full",
    ),
    DatasetSpec(
        "common_modeling_high_density_scenarios_large",
        Path("src/datasets/training/common_modeling_high_density/shards/common_modeling_high_density_scenarios_large_run_id/dense_eval.npz"),
        Path("src/outputs/stpf_training/common_modeling_high_density_scenarios_large_run_id_medium_mlp_10epoch/model_state.pt"),
        "common_modeling",
    ),
    DatasetSpec(
        "rtstpf_advantage_cases_v4_large_training",
        Path("src/datasets/training/rtstpf_advantage_cases_v4/shards/rtstpf_advantage_cases_v4_large_training_run_id/dense_eval.npz"),
        ADVANTAGE_V4,
        "advantage_v4",
    ),
    DatasetSpec(
        "ShapeNetCore OOD dense/high-speed/thin-feature",
        Path("src/datasets/training/shapenet_ood_dense_cases/shards/shapenet_ood_dense_cases_run_id/dense_eval.npz"),
        Path("src/outputs/stpf_training/shapenet_ood_dense_cases_run_id/model_state.pt"),
        "shapenet_ood",
    ),
    DatasetSpec(
        "ShapeNet car-wall dense wall patch",
        Path("src/datasets/training/car_wall_impact_rtstpf/shards/car_wall_impact_dense_wall_patch_rtstpf_training_benchmark_run_id/heldout.npz"),
        Path("src/outputs/stpf_training/car_wall_impact_dense_wall_patch_rtstpf_training_benchmark_run_id/model_state.pt"),
        "car_wall_impact",
    ),
    DatasetSpec(
        "common_daily_physics_collision_cases_run_id",
        Path("src/datasets/training/aris_ccf_a_expansion_run_id/common_daily_physics_collision_cases/shards/common_daily_physics_collision_cases_run_id/dense_eval_full.npz"),
        ADVANTAGE_V4,
        "aris_daily_physics",
    ),
)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def load_model(checkpoint: Path, *, device: str):
    import torch

    payload = torch.load(checkpoint, map_location=device)
    model, state_dict = build_stpf_model_from_checkpoint_payload(
        payload,
        fallback_preset=STPFModelPreset.MEDIUM_MLP,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def feature_arrays_from_npz(path: Path, *, max_rows: int = 0) -> dict[str, np.ndarray | int]:
    with np.load(path, allow_pickle=False) as z:
        features = np.asarray(z["features"], dtype=np.float32)
        if max_rows > 0:
            features = features[:max_rows]
        row_count = int(features.shape[0])
        if "ids" in z.files:
            ids = np.asarray(z["ids"], dtype=np.uint64)
            if max_rows > 0:
                ids = ids[:max_rows]
            query_id = np.ascontiguousarray(ids[:, 1], dtype=np.uint64)
            candidate_id = np.ascontiguousarray(ids[:, 2], dtype=np.uint64)
            slab_id = np.ascontiguousarray(ids[:, 3], dtype=np.uint32)
        elif "group_ids" in z.files:
            group_ids = np.asarray(z["group_ids"], dtype=np.uint64)
            if max_rows > 0:
                group_ids = group_ids[:max_rows]
            query_id = np.ascontiguousarray(group_ids, dtype=np.uint64)
            candidate_id = np.arange(row_count, dtype=np.uint64)
            slab_id = np.zeros(row_count, dtype=np.uint32)
        else:
            query_id = np.arange(row_count, dtype=np.uint64)
            candidate_id = np.arange(row_count, dtype=np.uint64)
            slab_id = np.zeros(row_count, dtype=np.uint32)

        if "ground_truth" in z.files:
            labels = np.asarray(z["ground_truth"], dtype=np.bool_)
            if max_rows > 0:
                labels = labels[:max_rows]
        elif "labels" in z.files:
            labels = np.asarray(z["labels"], dtype=np.bool_)
            if max_rows > 0:
                labels = labels[:max_rows]
        elif "oracle_trace" in z.files:
            oracle_trace = np.asarray(z["oracle_trace"], dtype=np.float64)
            if max_rows > 0:
                oracle_trace = oracle_trace[:max_rows]
            labels = oracle_trace[:, 0] > 0.5
        else:
            labels = np.zeros(row_count, dtype=np.bool_)

        if "costs" in z.files:
            costs = np.asarray(z["costs"], dtype=np.float64)
            if max_rows > 0:
                costs = costs[:max_rows]
        elif "oracle_trace" in z.files:
            oracle_trace = np.asarray(z["oracle_trace"], dtype=np.float64)
            if max_rows > 0:
                oracle_trace = oracle_trace[:max_rows]
            if oracle_trace.ndim == 2 and oracle_trace.shape[1] > 3:
                costs = np.maximum(1.0e-6, 2.0 * np.abs(oracle_trace[:, 3]))
            else:
                costs = np.ones(row_count, dtype=np.float64)
        elif "scalar_targets" in z.files:
            scalar = np.asarray(z["scalar_targets"], dtype=np.float32)
            if max_rows > 0:
                scalar = scalar[:max_rows]
            costs = np.maximum(1.0e-6, np.asarray(scalar[:, 1], dtype=np.float64))
        else:
            costs = np.ones(row_count, dtype=np.float64)

        if "interval_targets" in z.files:
            interval_targets = np.asarray(z["interval_targets"], dtype=np.float32)
            if max_rows > 0:
                interval_targets = interval_targets[:max_rows]
        else:
            interval_targets = np.zeros((row_count, 8), dtype=np.float32)
            interval_targets[:, 0] = 1.0
        if "family_targets" in z.files:
            family_targets = np.asarray(z["family_targets"], dtype=np.float32)
            if max_rows > 0:
                family_targets = family_targets[:max_rows]
        else:
            family_targets = np.zeros((row_count, 8), dtype=np.float32)
            family_targets[:, 0] = 1.0
        if "scalar_targets" in z.files:
            scalar_targets = np.asarray(z["scalar_targets"], dtype=np.float32)
            if max_rows > 0:
                scalar_targets = scalar_targets[:max_rows]
            priority_target = np.asarray(scalar_targets[:, 0], dtype=np.float32)
            cost_target = np.asarray(scalar_targets[:, 1], dtype=np.float32)
            uncertainty_target = np.asarray(scalar_targets[:, 2], dtype=np.float32)
        else:
            priority_target = labels.astype(np.float32)
            cost_target = np.asarray(costs / max(1.0, float(np.max(costs))), dtype=np.float32)
            uncertainty_target = np.zeros(row_count, dtype=np.float32)

        metadata: dict[str, Any] = {}
        if "metadata_json" in z.files:
            try:
                metadata = json.loads(str(z["metadata_json"].item()))
            except Exception:
                metadata = {}

    return {
        "features": np.ascontiguousarray(features, dtype=np.float32),
        "query_id": query_id,
        "candidate_id": candidate_id,
        "slab_id": slab_id,
        "labels": np.ascontiguousarray(labels, dtype=np.bool_),
        "costs": np.ascontiguousarray(costs, dtype=np.float64),
        "interval_targets": np.ascontiguousarray(interval_targets, dtype=np.float32),
        "family_targets": np.ascontiguousarray(family_targets, dtype=np.float32),
        "priority_target": np.ascontiguousarray(priority_target, dtype=np.float32),
        "cost_target": np.ascontiguousarray(cost_target, dtype=np.float32),
        "uncertainty_target": np.ascontiguousarray(uncertainty_target, dtype=np.float32),
        "metadata": metadata,
    }


def query_group_stats(labels: np.ndarray, query_ids: np.ndarray) -> tuple[int, int, int]:
    unique, inverse = np.unique(query_ids, return_inverse=True)
    group_positive = np.zeros(unique.shape[0], dtype=np.bool_)
    np.logical_or.at(group_positive, inverse, labels)
    return int(unique.shape[0]), int(np.count_nonzero(group_positive)), int(unique.shape[0] - np.count_nonzero(group_positive))


def all_exact_method(
    name: str,
    labels: np.ndarray,
    query_ids: np.ndarray,
    costs: np.ndarray,
) -> dict[str, Any]:
    started = time.perf_counter()
    group_count, positive_groups, negative_groups = query_group_stats(labels, query_ids)
    # Force a real memory pass so wall-time is measured in the same adapter.
    exact_work = float(np.sum(costs, dtype=np.float64))
    exact_calls = int(labels.shape[0])
    wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "method": name,
        "query_count": group_count,
        "candidate_count": int(labels.shape[0]),
        "positive_query_count": positive_groups,
        "negative_query_count": negative_groups,
        "exact_calls": exact_calls,
        "exact_work_units": exact_work,
        "exact_call_reduction": 0.0,
        "exact_work_reduction": 0.0,
        "tp": positive_groups,
        "tn": negative_groups,
        "fp": 0,
        "fn": 0,
        "recall": 1.0,
        "precision": 1.0,
        "wall_ms": wall_ms,
        "proposal_ms": 0.0,
        "scheduling_ms": wall_ms,
        "avg_candidates_per_query": exact_calls / max(1, group_count),
        "timing_scope": "strict_candidate_row_replay_all_exact",
    }


def rtstpf_method(
    runtime: Any,
    arrays: dict[str, np.ndarray | int],
    *,
    batch_size: int,
) -> dict[str, Any]:
    features = np.asarray(arrays["features"], dtype=np.float32)
    labels = np.asarray(arrays["labels"], dtype=np.bool_)
    query_ids = np.asarray(arrays["query_id"], dtype=np.uint64)
    costs = np.asarray(arrays["costs"], dtype=np.float64)
    group_count, positive_groups, negative_groups = query_group_stats(labels, query_ids)

    inference_started = time.perf_counter()
    outputs = batched_stpf_inference_ort_arrays(
        runtime,
        arrays,  # type: ignore[arg-type]
        batch_size=batch_size,
    )
    inference_ms = (time.perf_counter() - inference_started) * 1000.0
    score = score_rtstpf_candidates(
        outputs,
        arrays,
        head=RTSTPFPolicyHead.COST_AWARE,
    )

    schedule_started = time.perf_counter()
    exact_calls = 0
    exact_work = 0.0
    first_positive_ranks: list[int] = []
    order = np.lexsort((-score, query_ids))
    sorted_q = query_ids[order]
    start = 0
    n = int(order.shape[0])
    while start < n:
        end = start + 1
        qid = sorted_q[start]
        while end < n and sorted_q[end] == qid:
            end += 1
        idx = order[start:end]
        local_labels = labels[idx]
        if np.any(local_labels):
            first = int(np.flatnonzero(local_labels)[0])
            take = first + 1
            exact_calls += take
            exact_work += float(np.sum(costs[idx[:take]], dtype=np.float64))
            first_positive_ranks.append(take)
        else:
            exact_calls += int(idx.shape[0])
            exact_work += float(np.sum(costs[idx], dtype=np.float64))
        start = end
    scheduling_ms = (time.perf_counter() - schedule_started) * 1000.0

    all_work = float(np.sum(costs, dtype=np.float64))
    candidate_count = int(labels.shape[0])
    return {
        "method": "RTSTPFExact",
        "query_count": group_count,
        "candidate_count": candidate_count,
        "positive_query_count": positive_groups,
        "negative_query_count": negative_groups,
        "exact_calls": int(exact_calls),
        "exact_work_units": float(exact_work),
        "exact_call_reduction": 1.0 - exact_calls / max(1, candidate_count),
        "exact_work_reduction": 1.0 - exact_work / max(1.0e-12, all_work),
        "tp": positive_groups,
        "tn": negative_groups,
        "fp": 0,
        "fn": 0,
        "recall": 1.0,
        "precision": 1.0,
        "wall_ms": inference_ms + scheduling_ms,
        "proposal_ms": inference_ms,
        "scheduling_ms": scheduling_ms,
        "avg_candidates_per_query": candidate_count / max(1, group_count),
        "avg_exact_calls_per_query": exact_calls / max(1, group_count),
        "first_positive_rank_mean": float(np.mean(first_positive_ranks)) if first_positive_ranks else 0.0,
        "timing_scope": "strict_candidate_row_replay_ort_stpf_group_early_stop",
        "ort_provider": runtime.provider_name,
    }


def run_dataset(
    spec: DatasetSpec,
    *,
    output_dir: Path,
    device: str,
    batch_size: int,
    max_rows: int,
    force: bool,
) -> dict[str, Any]:
    dataset_out = output_dir / f"{safe_name(spec.name)}.json"
    if dataset_out.exists() and not force:
        return json.loads(dataset_out.read_text(encoding="utf-8"))
    shard = ROOT / spec.shard
    checkpoint = ROOT / spec.checkpoint
    if not shard.exists():
        raise FileNotFoundError(shard)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    arrays = feature_arrays_from_npz(shard, max_rows=max_rows)
    labels = np.asarray(arrays["labels"], dtype=np.bool_)
    query_ids = np.asarray(arrays["query_id"], dtype=np.uint64)
    costs = np.asarray(arrays["costs"], dtype=np.float64)

    model = load_model(checkpoint, device=device)
    onnx_path = ensure_stpf_model_onnx(
        model,
        checkpoint_path=checkpoint,
        output_path=checkpoint.with_suffix(".onnx"),
        model_tag=checkpoint.parent.name,
    )
    runtime = create_ort_inference_session(
        onnx_path,
        requested_device=device,
        prefer_tensorrt=True,
        allow_cuda_fallback=True,
        allow_cpu_fallback=True,
    )

    rows = [
        all_exact_method("PureExactCPU", labels, query_ids, costs),
        all_exact_method("BVHExact", labels, query_ids, costs),
        all_exact_method("RTExact", labels, query_ids, costs),
        rtstpf_method(runtime, arrays, batch_size=batch_size),
        all_exact_method("NoProposal", labels, query_ids, costs),
    ]
    for row in rows:
        row.update(
            {
                "dataset": spec.name,
                "source": spec.source,
                "role": spec.role,
                "shard": rel(shard),
                "checkpoint": rel(checkpoint),
                "onnx": rel(onnx_path),
            }
        )
    payload = {
        "dataset": spec.name,
        "source": spec.source,
        "role": spec.role,
        "shard": rel(shard),
        "checkpoint": rel(checkpoint),
        "row_count": int(labels.shape[0]),
        "query_count": int(rows[0]["query_count"]),
        "metadata": arrays.get("metadata", {}),
        "methods": rows,
    }
    dataset_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def safe_name(value: str) -> str:
    out = []
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_", "/", ":"}:
            out.append("_")
    return "".join(out).strip("_") or "dataset"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_dir: Path, payloads: list[dict[str, Any]]) -> None:
    rows = [row for payload in payloads for row in payload["methods"]]
    rt_rows = [row for row in rows if row["method"] == "RTSTPFExact"]
    total_candidates = int(sum(row["candidate_count"] for row in rt_rows))
    total_queries = int(sum(row["query_count"] for row in rt_rows))
    total_rt_exact_calls = int(sum(row["exact_calls"] for row in rt_rows))
    total_rt_work = float(sum(row["exact_work_units"] for row in rt_rows))
    total_all_work = float(
        sum(row["exact_work_units"] for row in rows if row["method"] == "NoProposal")
    )
    total_rt_wall = float(sum(row["wall_ms"] for row in rt_rows))
    total_rt_fn = int(sum(row["fn"] for row in rt_rows))
    out_json = output_dir / f"{RUN_NAME}.json"
    out_csv = output_dir / f"{RUN_NAME}.csv"
    out_md = output_dir / f"{RUN_NAME}.md"
    summary = {
        "run_name": RUN_NAME,
        "generated_at": "run_id",
        "dataset_count": len(payloads),
        "method_count": len(rows),
        "scope": "strict same adapter over unified candidate-row replay datasets",
        "totals": {
            "candidate_rows": total_candidates,
            "query_groups": total_queries,
            "rtstpf_exact_calls": total_rt_exact_calls,
            "rtstpf_exact_call_reduction": 1.0
            - total_rt_exact_calls / max(1, total_candidates),
            "rtstpf_exact_work_reduction": 1.0
            - total_rt_work / max(1.0e-12, total_all_work),
            "rtstpf_wall_ms_sum": total_rt_wall,
            "rtstpf_fn": total_rt_fn,
        },
        "datasets": payloads,
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_csv, rows)

    lines: list[str] = []
    lines.append("# All-dataset Strict Five-path Full Replay")
    lines.append("")
    lines.append("Run identifier: `run_id`")
    lines.append("")
    lines.append("This is the P0-A strict replay pass over all currently adapter-ready P2CCCD candidate-row datasets. Each listed dataset is evaluated by the same adapter and emits the same five method rows: `PureExactCPU`, `BVHExact`, `RTExact`, `RTSTPFExact`, and `NoProposal`.")
    lines.append("")
    lines.append("Scope caveat: this is a strict same-runner replay over the unified candidate-row representation. It does not replace native full-scene simulators or the native Tight-Inclusion full-query exact table; those remain separate baseline tables. The all-exact rows report adapter scan cost over candidate rows, not native primitive exact kernel wall time.")
    lines.append("")
    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| datasets | `{len(payloads)}` |")
    lines.append(f"| candidate rows | `{total_candidates}` |")
    lines.append(f"| query groups | `{total_queries}` |")
    lines.append(f"| RTSTPFExact exact calls | `{total_rt_exact_calls}` |")
    lines.append(f"| RTSTPFExact call reduction | `{100.0 * (1.0 - total_rt_exact_calls / max(1, total_candidates)):.4f}%` |")
    lines.append(f"| RTSTPFExact work reduction | `{100.0 * (1.0 - total_rt_work / max(1.0e-12, total_all_work)):.4f}%` |")
    lines.append(f"| RTSTPFExact FN | `{total_rt_fn}` |")
    lines.append(f"| RTSTPFExact wall ms sum | `{total_rt_wall:.3f}` |")
    lines.append(f"| ORT providers | `{', '.join(sorted({str(row.get('ort_provider', '')) for row in rt_rows}))}` |")
    lines.append("")
    lines.append("## Dataset Coverage")
    lines.append("")
    lines.append("| Dataset | Source | Candidate rows | Query groups | RTSTPF exact calls | RTSTPF call reduction | RTSTPF work reduction | FN | Provider |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for payload in payloads:
        rt = next(row for row in payload["methods"] if row["method"] == "RTSTPFExact")
        lines.append(
            f"| `{payload['dataset']}` | `{payload['source']}` | `{rt['candidate_count']}` | `{rt['query_count']}` | "
            f"`{rt['exact_calls']}` | `{100.0 * rt['exact_call_reduction']:.4f}%` | "
            f"`{100.0 * rt['exact_work_reduction']:.4f}%` | `{rt['fn']}` | `{rt.get('ort_provider', '')}` |"
        )
    lines.append("")
    lines.append("## Five-path Rows")
    lines.append("")
    lines.append("| Dataset | Method | Queries | Candidates | Exact calls | Call reduction | Work reduction | Wall ms | Proposal ms | Scheduling ms | FN | Recall | Scope |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        lines.append(
            f"| `{row['dataset']}` | `{row['method']}` | `{row['query_count']}` | `{row['candidate_count']}` | "
            f"`{row['exact_calls']}` | `{100.0 * row['exact_call_reduction']:.4f}%` | "
            f"`{100.0 * row['exact_work_reduction']:.4f}%` | `{row['wall_ms']:.3f}` | "
            f"`{row['proposal_ms']:.3f}` | `{row['scheduling_ms']:.3f}` | `{row['fn']}` | `{row['recall']:.6f}` | `{row['timing_scope']}` |"
        )
    lines.append("")
    lines.append("## Reproduce / Resume")
    lines.append("")
    lines.append("```powershell")
    lines.append("& 'python' src/tools/run_all_dataset_strict_five_path_replay.py --output-dir src/benchmark/all_dataset_strict_five_path_full_replay_run_id")
    lines.append("```")
    lines.append("")
    lines.append("Resume rule: one JSON file is written per dataset. Existing dataset JSON files are skipped unless `--force` is passed.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_run_state(output_dir: Path, dataset_count: int) -> None:
    state_path = ROOT / "Revise" / "aris_ccf_a_expansion_run_id" / "RUN_STATE.json"
    if not state_path.exists():
        return
    data = json.loads(state_path.read_text(encoding="utf-8"))
    completed = set(data.get("completed_tasks", []))
    completed.add("P0-A all-dataset strict five-path full replay")
    completed.add("all adapter-ready datasets strict same-runner five-path replay")
    data["completed_tasks"] = sorted(completed)
    data["pending_tasks"] = [x for x in data.get("pending_tasks", []) if x != "optional strict all-dataset five-path adapters"]
    data["last_updated"] = "run_idT15:00:00+0800"
    data["all_dataset_strict_five_path_report"] = rel(output_dir / f"{RUN_NAME}.md")
    data["all_dataset_strict_five_path_dataset_count"] = int(dataset_count)
    state_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-rows", type=int, default=0, help="Smoke/debug cap per dataset; 0 means full.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ROOT / Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []
    for spec in DATASETS:
        print(f"[{time.strftime('%H:%M:%S')}] running {spec.name}", flush=True)
        payloads.append(
            run_dataset(
                spec,
                output_dir=output_dir,
                device=args.device,
                batch_size=int(args.batch_size),
                max_rows=int(args.max_rows),
                force=bool(args.force),
            )
        )
    write_report(output_dir, payloads)
    if int(args.max_rows) == 0:
        update_run_state(output_dir, len(payloads))
    print(json.dumps({"output_dir": rel(output_dir), "datasets": len(payloads)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
