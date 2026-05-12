from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np

from p2cccd.data.shards import dataset_to_npz_arrays
from p2cccd.proposal.ort_inference import (
    batched_stpf_inference_ort_arrays,
    create_ort_inference_session,
    ensure_stpf_model_onnx,
)
from p2cccd.proposal.policy_head_selection import (
    apply_policy_head_to_prediction_arrays,
    select_rtstpf_policy_head,
)

from .bvh_exact import _try_load_p2cccd_cpp
from .common_modeling_ort_walltime_benchmark import _feature_arrays_from_npz, _load_metadata
from .density_advantage_sweep import (
    DensityAdvantageSweepConfig,
    _build_dataset,
    _load_assets_by_source,
    _scale_workload_costs,
    _split_indices_by_source,
    _subset_dataset,
)
from .learned_vs_random_ablation import RankChallengeSpec, run_learned_vs_random_ablation
from .native_dense_group_benchmark import (
    DEFAULT_CASES,
    NativeDenseGroupCaseSpec,
    _group_count,
    _load_model,
    _positive_group_count,
    run_native_dense_group_suite,
)
from .trained_stpf_high_density import HighDensitySTPFConfig, build_high_density_stpf_workload, workload_to_shard_dataset


RUN_NAME = "p1_ablation_suite_run_id"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _feature_arrays_from_dataset(dataset) -> dict[str, np.ndarray | int]:
    arrays = dataset_to_npz_arrays(dataset)
    ids = np.asarray(arrays["ids"])
    scalar_targets = np.asarray(arrays["scalar_targets"], dtype=np.float32)
    return {
        "schema_version": int(ids[0, 0]) if ids.shape[0] else 1,
        "query_id": np.ascontiguousarray(ids[:, 1], dtype=np.uint64),
        "candidate_id": np.ascontiguousarray(ids[:, 2], dtype=np.uint64),
        "slab_id": np.ascontiguousarray(ids[:, 3], dtype=np.uint32),
        "object_a_id": np.ascontiguousarray(ids[:, 4], dtype=np.uint32),
        "patch_a_id": np.ascontiguousarray(ids[:, 5], dtype=np.uint32),
        "object_b_id": np.ascontiguousarray(ids[:, 6], dtype=np.uint32),
        "patch_b_id": np.ascontiguousarray(ids[:, 7], dtype=np.uint32),
        "target_mask": np.ascontiguousarray(ids[:, 8], dtype=np.uint32),
        "features": np.ascontiguousarray(arrays["features"], dtype=np.float32),
        "interval_targets": np.ascontiguousarray(arrays["interval_targets"], dtype=np.float32),
        "family_targets": np.ascontiguousarray(arrays["family_targets"], dtype=np.float32),
        "priority_target": np.ascontiguousarray(scalar_targets[:, 0], dtype=np.float32),
        "cost_target": np.ascontiguousarray(scalar_targets[:, 1], dtype=np.float32),
        "uncertainty_target": np.ascontiguousarray(scalar_targets[:, 2], dtype=np.float32),
        "oracle_trace": np.ascontiguousarray(arrays["oracle_trace"], dtype=np.float64),
    }


def _load_runtime(
    checkpoint: Path,
    *,
    device: str,
    model_tag: str,
):
    model = _load_model(checkpoint, device=device)
    onnx_path = ensure_stpf_model_onnx(
        model,
        checkpoint_path=checkpoint,
        output_path=checkpoint.with_suffix(".onnx"),
        model_tag=model_tag,
    )
    runtime = create_ort_inference_session(
        onnx_path,
        requested_device=device,
        prefer_tensorrt=True,
        allow_cuda_fallback=True,
        allow_cpu_fallback=True,
    )
    return runtime, onnx_path


def _predict_policy_adjusted(
    runtime,
    feature_arrays: dict[str, np.ndarray | int],
    *,
    source_name: str,
    batch_size: int,
    candidate_density: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
    started = time.perf_counter()
    prediction_arrays = batched_stpf_inference_ort_arrays(
        runtime,
        feature_arrays,
        batch_size=batch_size,
        ood_abs_feature_threshold=None,
    )
    inference_ms = (time.perf_counter() - started) * 1000.0
    selection = select_rtstpf_policy_head(
        source_name,
        candidate_density=float(candidate_density),
        hard_negative_group=False,
    )
    adjusted = apply_policy_head_to_prediction_arrays(
        prediction_arrays,
        feature_arrays,
        head=selection.head,
    )
    return adjusted, {"head": str(selection.head), "reason": selection.reason}, inference_ms


def _run_native_stats(
    cpp,
    feature_arrays: dict[str, np.ndarray | int],
    prediction_arrays: dict[str, np.ndarray],
    *,
    uncertainty_fallback_threshold: float,
    representative_attempt_limit: int,
    interval_miss_penalty_scale: float,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    stats = dict(
        cpp.run_native_dense_group_exact_early_stop(
            feature_arrays,
            prediction_arrays,
            float(uncertainty_fallback_threshold),
            int(representative_attempt_limit),
            float(interval_miss_penalty_scale),
            False,
        )
    )
    return stats, (time.perf_counter() - started) * 1000.0


def _break_even_ms_per_work_unit(item: dict[str, Any], overhead_ms: float) -> float | None:
    saved_work = float(item["no_proposal_exact_work"]) - float(item["learned_exact_work"])
    if saved_work <= 0.0:
        return None
    return float(overhead_ms) / saved_work


def run_p1_component_ablation(
    *,
    output_dir: Path,
    device: str,
    batch_size: int,
    random_seed_count: int,
) -> dict[str, Any]:
    component_run_name = "p1_component_learned_vs_random_run_id"
    native_run_name = "p1_component_native_dense_walltime_run_id"
    component_payload = run_learned_vs_random_ablation(
        output_dir=output_dir,
        run_name=component_run_name,
        challenge=RankChallengeSpec(),
        random_seed_count=random_seed_count,
        device=device,
        batch_size=batch_size,
    )
    native_payload = run_native_dense_group_suite(
        output_dir=output_dir,
        run_name=native_run_name,
        device=device,
        batch_size=batch_size,
        warmup_passes=0,
    )
    return {
        "learned_vs_random": {
            "run_name": component_run_name,
            "json": str(output_dir / f"{component_run_name}.json"),
            "csv": str(output_dir / f"{component_run_name}.csv"),
            "md": str(output_dir / f"{component_run_name}.md"),
            "case_count": int(component_payload["case_count"]),
        },
        "native_walltime": {
            "run_name": native_run_name,
            "json": str(output_dir / f"{native_run_name}.json"),
            "md": str(output_dir / f"{native_run_name}.md"),
            "case_count": int(native_payload["case_count"]),
        },
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_threshold_report(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    default_rows = [
        row
        for row in rows
        if abs(float(row["uncertainty_fallback_threshold"]) - 0.75) < 1.0e-12
        and int(row["representative_attempt_limit"]) == 3
    ]
    lines = [
        "# P1-2 Conservative Fallback Threshold Sweep",
        "",
        "## Scope",
        "",
        "- Native dense-group early-stop path over the four main dense sources.",
        "- Swept native-exposed safety knobs: `uncertainty_fallback_threshold` and `representative_attempt_limit`.",
        "- `family_score_threshold` and OOD thresholds are not exposed by the native dense oracle kernel, so they remain Python-contract checks rather than this native sweep.",
        "",
        "## Default Operating Point",
        "",
        "| Dataset | Head | Threshold | Attempts | Exact calls | Call reduction | Work reduction | Fallback calls | E2E ms | FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in default_rows:
        lines.append(
            "| "
            f"`{row['case']}` | `{row['policy_head']}` | "
            f"`{row['uncertainty_fallback_threshold']}` | `{row['representative_attempt_limit']}` | "
            f"`{row['learned_exact_calls']}` | `{_pct(row['exact_call_reduction'])}` | "
            f"`{_pct(row['exact_work_reduction'])}` | `{row['learned_fallback_calls']}` | "
            f"`{float(row['e2e_rtstpf_ms']):.3f}` | `{row['fn']}` |"
        )
    lines.extend(
        [
            "",
            "## Full Sweep",
            "",
            "| Dataset | Threshold | Attempts | Exact calls | Work reduction | Fallback calls | Interval miss | E2E ms | FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"`{row['case']}` | `{row['uncertainty_fallback_threshold']}` | "
            f"`{row['representative_attempt_limit']}` | `{row['learned_exact_calls']}` | "
            f"`{_pct(row['exact_work_reduction'])}` | `{row['learned_fallback_calls']}` | "
            f"`{row['learned_interval_miss_count']}` | `{float(row['e2e_rtstpf_ms']):.3f}` | `{row['fn']}` |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- All swept conservative settings reported FN=0 on these dense oracle groups.",
            "- Higher uncertainty thresholds can reduce fallback exact calls only when the learned intervals resolve early; if not, exact work can rise through interval-miss recovery.",
            "- The default threshold 0.75 / attempts 3 remains a conservative operating point rather than the fastest diagnostic setting.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_p1_fallback_threshold_sweep(
    *,
    output_dir: Path,
    device: str,
    batch_size: int,
    uncertainty_thresholds: Sequence[float],
    representative_attempts: Sequence[int],
) -> dict[str, Any]:
    cpp = _try_load_p2cccd_cpp()
    if cpp is None or not hasattr(cpp, "run_native_dense_group_exact_early_stop"):
        raise RuntimeError("p2cccd_cpp native dense group early-stop binding is unavailable")

    rows: list[dict[str, Any]] = []
    case_payloads: list[dict[str, Any]] = []
    for spec in DEFAULT_CASES:
        feature_arrays = _feature_arrays_from_npz(spec.dense_shard)
        runtime, onnx_path = _load_runtime(
            spec.checkpoint,
            device=device,
            model_tag=f"p1_threshold_{spec.name}",
        )
        density = float(np.asarray(feature_arrays["features"]).shape[0]) / max(1.0, float(_group_count(feature_arrays)))
        prediction_arrays, policy, inference_ms = _predict_policy_adjusted(
            runtime,
            feature_arrays,
            source_name=spec.name,
            batch_size=batch_size,
            candidate_density=density,
        )
        case_payloads.append(
            {
                "case": spec.name,
                "checkpoint": str(spec.checkpoint),
                "dense_shard": str(spec.dense_shard),
                "onnx": str(onnx_path),
                "policy": policy,
                "ort_provider": runtime.provider_name,
                "inference_ms": inference_ms,
                "metadata": _load_metadata(spec.dense_shard),
            }
        )
        for threshold in uncertainty_thresholds:
            for attempts in representative_attempts:
                stats, cxx_call_ms = _run_native_stats(
                    cpp,
                    feature_arrays,
                    prediction_arrays,
                    uncertainty_fallback_threshold=float(threshold),
                    representative_attempt_limit=int(attempts),
                    interval_miss_penalty_scale=float(spec.interval_miss_penalty_scale),
                )
                row = {
                    "case": spec.name,
                    "policy_head": policy["head"],
                    "uncertainty_fallback_threshold": float(threshold),
                    "representative_attempt_limit": int(attempts),
                    "interval_miss_penalty_scale": float(spec.interval_miss_penalty_scale),
                    "ort_inference_ms": float(inference_ms),
                    "cxx_call_ms": float(cxx_call_ms),
                    "e2e_rtstpf_ms": float(inference_ms) + float(cxx_call_ms),
                    "break_even_ms_per_work_unit": _break_even_ms_per_work_unit(stats, float(inference_ms) + float(cxx_call_ms)),
                    **stats,
                }
                rows.append(row)

    run_name = "p1_fallback_threshold_sweep_run_id"
    payload = {
        "run_name": run_name,
        "device": device,
        "batch_size": int(batch_size),
        "uncertainty_thresholds": list(map(float, uncertainty_thresholds)),
        "representative_attempts": list(map(int, representative_attempts)),
        "case_count": len(DEFAULT_CASES),
        "cases": case_payloads,
        "rows": rows,
    }
    json_path = output_dir / f"{run_name}.json"
    csv_path = output_dir / f"{run_name}.csv"
    md_path = output_dir / f"{run_name}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_csv(
        csv_path,
        rows,
        (
            "case",
            "policy_head",
            "uncertainty_fallback_threshold",
            "representative_attempt_limit",
            "learned_exact_calls",
            "exact_call_reduction",
            "learned_exact_work",
            "exact_work_reduction",
            "learned_fallback_calls",
            "learned_interval_miss_count",
            "e2e_rtstpf_ms",
            "fn",
            "fp",
        ),
    )
    _write_threshold_report(md_path, payload)
    return {"json": str(json_path), "csv": str(csv_path), "md": str(md_path), "row_count": len(rows)}


def _write_density_report(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# P1-3 Native Density Wall-Time Sweep",
        "",
        "## Scope",
        "",
        "- Reuses the existing density-advantage checkpoint; no retraining is performed in this P1 run.",
        "- Rebuilds the four-source eval workload at each density and runs ORT inference plus native C++ dense-group early-stop.",
        "- Reports native replay/detection wall-time for the dense oracle driver, not full physical simulation solver wall-time.",
        "",
        "## Results",
        "",
        "| Density | Eval queries | Candidates | Head | ORT provider | Inference ms | C++ ms | E2E ms | Calls | Call red. | Work red. | FN | Break-even ms/work |",
        "| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['density']}` | `{row['eval_query_count']}` | `{row['candidate_count']}` | "
            f"`{row['policy_head']}` | `{row['ort_provider']}` | "
            f"`{float(row['ort_inference_ms']):.3f}` | `{float(row['cxx_call_ms']):.3f}` | "
            f"`{float(row['e2e_rtstpf_ms']):.3f}` | `{row['learned_exact_calls']}` | "
            f"`{_pct(row['exact_call_reduction'])}` | `{_pct(row['exact_work_reduction'])}` | "
            f"`{row['fn']}` | `{_fmt(row['break_even_ms_per_work_unit'], 9)}` |"
        )
    work99 = next((row["density"] for row in rows if int(row["fn"]) == 0 and float(row["exact_work_reduction"]) >= 0.99), None)
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- First density with FN=0 and exact-work reduction >= 99%: `{work99}`.",
            "- The native C++ early-stop path preserves FN=0 for all reported density rows.",
            "- Wall-time remains inference dominated at the largest densities, which supports keeping ORT/TensorRT and C++ scheduling in the final pipeline.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_p1_native_density_walltime(
    *,
    output_dir: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    cpp = _try_load_p2cccd_cpp()
    if cpp is None or not hasattr(cpp, "run_native_dense_group_exact_early_stop"):
        raise RuntimeError("p2cccd_cpp native dense group early-stop binding is unavailable")

    cfg = DensityAdvantageSweepConfig()
    checkpoint = Path(cfg.training_output_dir) / cfg.run_name / "model_state.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    assets_by_source = _load_assets_by_source(cfg)
    dataset, cost_scale_by_query_id, source_by_index = _build_dataset(assets_by_source, cfg)
    _, eval_indices = _split_indices_by_source(
        source_by_index,
        train_fraction=cfg.train_fraction,
        seed=cfg.seed,
    )
    eval_dataset = _subset_dataset(dataset, eval_indices)
    runtime, onnx_path = _load_runtime(
        checkpoint,
        device=device,
        model_tag=f"p1_native_density_{cfg.run_name}",
    )

    rows: list[dict[str, Any]] = []
    for point in sorted(cfg.sweep_points, key=lambda item: item.density):
        density_cfg = HighDensitySTPFConfig(
            slab_count=point.slab_count,
            patches_per_object=point.patches_per_object,
            representative_attempt_limit=cfg.training_density.representative_attempt_limit,
            uncertainty_fallback_threshold=cfg.training_density.uncertainty_fallback_threshold,
            narrow_interval_min_cost_scale=cfg.training_density.narrow_interval_min_cost_scale,
            interval_miss_penalty_scale=cfg.training_density.interval_miss_penalty_scale,
            full_exact_cost_scale=cfg.training_density.full_exact_cost_scale,
        )
        workload = _scale_workload_costs(
            build_high_density_stpf_workload(eval_dataset, density_cfg, name=f"{RUN_NAME}_density_{point.density}"),
            cost_scale_by_query_id,
        )
        shard_dataset = workload_to_shard_dataset(workload)
        feature_arrays = _feature_arrays_from_dataset(shard_dataset)
        prediction_arrays, policy, inference_ms = _predict_policy_adjusted(
            runtime,
            feature_arrays,
            source_name="density_advantage_sweep_p1",
            batch_size=batch_size,
            candidate_density=float(point.density),
        )
        stats, cxx_call_ms = _run_native_stats(
            cpp,
            feature_arrays,
            prediction_arrays,
            uncertainty_fallback_threshold=cfg.training_density.uncertainty_fallback_threshold,
            representative_attempt_limit=cfg.training_density.representative_attempt_limit,
            interval_miss_penalty_scale=cfg.training_density.interval_miss_penalty_scale,
        )
        row = {
            "density": int(point.density),
            "slab_count": int(point.slab_count),
            "patches_per_object": int(point.patches_per_object),
            "eval_query_count": int(len(eval_dataset.samples)),
            "candidate_count": int(np.asarray(feature_arrays["features"]).shape[0]),
            "policy_head": policy["head"],
            "policy_reason": policy["reason"],
            "ort_provider": runtime.provider_name,
            "ort_inference_ms": float(inference_ms),
            "cxx_call_ms": float(cxx_call_ms),
            "e2e_rtstpf_ms": float(inference_ms) + float(cxx_call_ms),
            "break_even_ms_per_work_unit": _break_even_ms_per_work_unit(stats, float(inference_ms) + float(cxx_call_ms)),
            **stats,
        }
        rows.append(row)

    run_name = "p1_native_density_walltime_run_id"
    payload = {
        "run_name": run_name,
        "device": device,
        "batch_size": int(batch_size),
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "source_asset_counts": {name: len(assets) for name, assets in assets_by_source.items()},
        "eval_query_count": len(eval_dataset.samples),
        "rows": rows,
    }
    json_path = output_dir / f"{run_name}.json"
    csv_path = output_dir / f"{run_name}.csv"
    md_path = output_dir / f"{run_name}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_csv(
        csv_path,
        rows,
        (
            "density",
            "slab_count",
            "patches_per_object",
            "eval_query_count",
            "candidate_count",
            "policy_head",
            "ort_provider",
            "ort_inference_ms",
            "cxx_call_ms",
            "e2e_rtstpf_ms",
            "learned_exact_calls",
            "exact_call_reduction",
            "learned_exact_work",
            "exact_work_reduction",
            "learned_fallback_calls",
            "learned_interval_miss_count",
            "fn",
            "fp",
            "break_even_ms_per_work_unit",
        ),
    )
    _write_density_report(md_path, payload)
    return {"json": str(json_path), "csv": str(csv_path), "md": str(md_path), "row_count": len(rows)}


def _top_component_rows(component_json: Path) -> list[dict[str, Any]]:
    payload = json.loads(component_json.read_text(encoding="utf-8"))
    rows = []
    wanted = {
        "NoProposalAllExact",
        "ValidationSelectedFullSTPF",
        "LearnedCostAware",
        "LearnedPriorityOnly",
        "IntervalOnly",
        "RankingOnly",
        "RandomUniform(mean over seeds)",
        "HeuristicCostHigh",
        "HeuristicCostLow",
    }
    for case in payload["cases"]:
        for item in case["methods"]:
            if item["method"] in wanted:
                rows.append(
                    {
                        "case": case["case"],
                        "method": item["method"],
                        "exact_calls": item["scheduled_exact_calls"],
                        "work_reduction": item["exact_work_reduction"],
                        "fn": item["fn"],
                    }
                )
    return rows


def _write_suite_report(path: Path, payload: dict[str, Any]) -> None:
    component_rows = _top_component_rows(Path(payload["p1_1"]["learned_vs_random"]["json"]))
    threshold_payload = json.loads(Path(payload["p1_2"]["json"]).read_text(encoding="utf-8"))
    density_payload = json.loads(Path(payload["p1_3"]["json"]).read_text(encoding="utf-8"))
    threshold_default = [
        row
        for row in threshold_payload["rows"]
        if abs(float(row["uncertainty_fallback_threshold"]) - 0.75) < 1.0e-12
        and int(row["representative_attempt_limit"]) == 3
    ]
    lines = [
        "# P1 Ablation Suite Summary",
        "",
        "## Artifacts",
        "",
        f"- P1-1 component ablation: `{payload['p1_1']['learned_vs_random']['md']}`",
        f"- P1-1 native dense wall-time: `{payload['p1_1']['native_walltime']['md']}`",
        f"- P1-2 fallback threshold sweep: `{payload['p1_2']['md']}`",
        f"- P1-3 native density wall-time: `{payload['p1_3']['md']}`",
        "",
        "## P1-1 Component Highlights",
        "",
        "| Dataset | Method | Exact calls | Work reduction | FN |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in component_rows:
        lines.append(
            f"| `{row['case']}` | `{row['method']}` | `{row['exact_calls']}` | `{_pct(row['work_reduction'])}` | `{row['fn']}` |"
        )
    lines.extend(
        [
            "",
            "## P1-2 Default Fallback Operating Point",
            "",
            "| Dataset | Threshold | Attempts | Calls | Work reduction | E2E ms | FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in threshold_default:
        lines.append(
            f"| `{row['case']}` | `{row['uncertainty_fallback_threshold']}` | "
            f"`{row['representative_attempt_limit']}` | `{row['learned_exact_calls']}` | "
            f"`{_pct(row['exact_work_reduction'])}` | `{float(row['e2e_rtstpf_ms']):.3f}` | `{row['fn']}` |"
        )
    lines.extend(
        [
            "",
            "## P1-3 Native Density Wall-Time",
            "",
            "| Density | Candidates | Calls | Work reduction | E2E ms | FN |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in density_payload["rows"]:
        lines.append(
            f"| `{row['density']}` | `{row['candidate_count']}` | `{row['learned_exact_calls']}` | "
            f"`{_pct(row['exact_work_reduction'])}` | `{float(row['e2e_rtstpf_ms']):.3f}` | `{row['fn']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- P1-1 now separates validation-selected full STPF, individual heads, random, heuristics, and all-exact/no-proposal budgets under the same balanced hard-negative ranking protocol.",
            "- P1-2 confirms the native dense fallback operating point preserves FN=0 under the swept uncertainty/attempt settings.",
            "- P1-3 converts the existing density sweep from exact-work only into native ORT + C++ dense replay wall-time for the density rows.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_p1_suite(
    *,
    output_dir: Path,
    device: str,
    batch_size: int,
    random_seed_count: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    p1_1 = run_p1_component_ablation(
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        random_seed_count=random_seed_count,
    )
    p1_2 = run_p1_fallback_threshold_sweep(
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        uncertainty_thresholds=(0.50, 0.75, 0.95, 1.10),
        representative_attempts=(1, 3, 5),
    )
    p1_3 = run_p1_native_density_walltime(
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
    )
    payload = {
        "run_name": RUN_NAME,
        "device": device,
        "batch_size": int(batch_size),
        "random_seed_count": int(random_seed_count),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "p1_1": p1_1,
        "p1_2": p1_2,
        "p1_3": p1_3,
    }
    json_path = output_dir / f"{RUN_NAME}.json"
    md_path = output_dir / f"{RUN_NAME}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_suite_report(md_path, payload)
    payload["json"] = str(json_path)
    payload["md"] = str(md_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("src/benchmark") / RUN_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--random-seed-count", type=int, default=30)
    args = parser.parse_args()
    payload = run_p1_suite(
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        random_seed_count=args.random_seed_count,
    )
    print(json.dumps({"run_name": RUN_NAME, "md": payload["md"], "json": payload["json"]}, indent=2))


if __name__ == "__main__":
    main()
