from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from p2cccd.proposal.ort_inference import (
    batched_stpf_inference_ort_arrays,
    create_ort_inference_session,
    ensure_stpf_model_onnx,
)
from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model_from_checkpoint_payload

from .bvh_exact import _try_load_p2cccd_cpp
from .rt_exact import FEATURE_FAMILY_CONSERVATIVE
from .rt_stpf_exact import (
    RTSTPFExactConfig,
    _schedule_stats_from_cpp,
    _to_cpp_proposal_scheduling_config,
)


def _load_model(checkpoint_path: Path, *, device: str):
    import torch

    payload = torch.load(checkpoint_path, map_location=device)
    model, state_dict = build_stpf_model_from_checkpoint_payload(
        payload,
        fallback_preset=STPFModelPreset.MEDIUM_MLP,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _feature_arrays_from_npz(path: Path) -> dict[str, np.ndarray | int]:
    with np.load(path, allow_pickle=False) as chunk:
        ids = np.asarray(chunk["ids"])
        scalar_targets = np.asarray(chunk["scalar_targets"], dtype=np.float32)
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
            "features": np.ascontiguousarray(chunk["features"], dtype=np.float32),
            "interval_targets": np.ascontiguousarray(chunk["interval_targets"], dtype=np.float32),
            "family_targets": np.ascontiguousarray(chunk["family_targets"], dtype=np.float32),
            "priority_target": np.ascontiguousarray(scalar_targets[:, 0], dtype=np.float32),
            "cost_target": np.ascontiguousarray(scalar_targets[:, 1], dtype=np.float32),
            "uncertainty_target": np.ascontiguousarray(scalar_targets[:, 2], dtype=np.float32),
            "oracle_trace": np.ascontiguousarray(chunk["oracle_trace"], dtype=np.float64),
        }


def _positive_count(feature_arrays: dict[str, np.ndarray | int]) -> int:
    trace = np.asarray(feature_arrays["oracle_trace"], dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] < 1:
        return 0
    query_ids = np.asarray(feature_arrays["query_id"], dtype=np.uint64)
    positive_query_ids = np.unique(query_ids[trace[:, 0] > 0.5])
    return int(positive_query_ids.shape[0])


def _load_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as chunk:
        if "metadata_json" not in chunk.files:
            return {}
        return json.loads(str(chunk["metadata_json"].item()))


def _schedule_with_cpp(
    cpp: Any,
    feature_arrays: dict[str, np.ndarray | int],
    prediction_arrays: dict[str, np.ndarray],
    *,
    uncertainty_fallback_threshold: float,
) -> tuple[int, Any, float]:
    query_ids = np.unique(np.asarray(feature_arrays["query_id"], dtype=np.uint64))
    family_masks = {int(query_id): int(FEATURE_FAMILY_CONSERVATIVE) for query_id in query_ids.tolist()}
    cfg = RTSTPFExactConfig(
        inference_backend="ort",
        model_preset=STPFModelPreset.MEDIUM_MLP,
        uncertainty_fallback_threshold=float(uncertainty_fallback_threshold),
        ood_abs_feature_threshold=1.0e12,
        family_score_threshold=0.5,
        proposal_batch_size=65536,
    )
    cpp_cfg = _to_cpp_proposal_scheduling_config(cpp, cfg)
    started = time.perf_counter()
    work_items, cpp_stats = cpp.schedule_runtime_exact_work_items_from_proposal_arrays(
        feature_arrays,
        prediction_arrays,
        family_masks,
        cpp_cfg,
    )
    schedule_ms = (time.perf_counter() - started) * 1000.0
    stats = _schedule_stats_from_cpp(cpp_stats, feature_row_count=int(np.asarray(feature_arrays["query_id"]).shape[0]))
    return int(len(work_items)), stats, schedule_ms


def run_common_modeling_ort_walltime_benchmark(
    *,
    checkpoint_path: str | Path,
    dense_shard_path: str | Path,
    report_path: str | Path,
    json_path: str | Path | None = None,
    device: str = "cuda",
    batch_size: int = 65536,
    uncertainty_fallback_threshold: float = 0.75,
    warmup_passes: int = 2,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    shard = Path(dense_shard_path)
    report = Path(report_path)
    json_output = Path(json_path) if json_path is not None else report.with_suffix(".json")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not shard.exists():
        raise FileNotFoundError(shard)

    cpp = _try_load_p2cccd_cpp()
    if cpp is None or not hasattr(cpp, "schedule_runtime_exact_work_items_from_proposal_arrays"):
        raise RuntimeError("compiled p2cccd_cpp array scheduling binding is unavailable")

    model = _load_model(checkpoint, device=device)
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
    feature_arrays = _feature_arrays_from_npz(shard)
    row_count = int(np.asarray(feature_arrays["features"]).shape[0])
    query_count = int(np.unique(np.asarray(feature_arrays["query_id"], dtype=np.uint64)).shape[0])

    warmup_ms: list[float] = []
    prediction_arrays: dict[str, np.ndarray] | None = None
    for _ in range(max(0, int(warmup_passes))):
        started = time.perf_counter()
        prediction_arrays = batched_stpf_inference_ort_arrays(
            runtime,
            feature_arrays,
            batch_size=batch_size,
            ood_abs_feature_threshold=None,
        )
        warmup_ms.append((time.perf_counter() - started) * 1000.0)

    started = time.perf_counter()
    prediction_arrays = batched_stpf_inference_ort_arrays(
        runtime,
        feature_arrays,
        batch_size=batch_size,
        ood_abs_feature_threshold=None,
    )
    inference_ms = (time.perf_counter() - started) * 1000.0
    work_item_count, schedule_stats, schedule_ms = _schedule_with_cpp(
        cpp,
        feature_arrays,
        prediction_arrays,
        uncertainty_fallback_threshold=uncertainty_fallback_threshold,
    )
    total_ms = inference_ms + schedule_ms
    result = {
        "checkpoint_path": checkpoint.as_posix(),
        "onnx_path": onnx_path.as_posix(),
        "dense_shard_path": shard.as_posix(),
        "metadata": _load_metadata(shard),
        "device": device,
        "batch_size": int(batch_size),
        "ort_provider": runtime.provider_name,
        "provider_order": list(runtime.provider_order),
        "row_count": row_count,
        "query_count": query_count,
        "positive_query_count": _positive_count(feature_arrays),
        "warmup_ms": warmup_ms,
        "ort_inference_ms": inference_ms,
        "cpp_schedule_ms": schedule_ms,
        "proposal_total_ms": total_ms,
        "candidate_rows_per_second": 1000.0 * row_count / max(1.0e-9, inference_ms),
        "proposal_rows_per_second": 1000.0 * row_count / max(1.0e-9, total_ms),
        "work_item_count": work_item_count,
        "schedule_stats": asdict(schedule_stats),
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(report, result)
    return result


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Common Modeling RTSTPFExact ORT/TensorRT + C++ Scheduling Wall-Time Benchmark",
        "",
        "## Protocol",
        "",
        f"- checkpoint: `{result['checkpoint_path']}`",
        f"- onnx: `{result['onnx_path']}`",
        f"- dense shard: `{result['dense_shard_path']}`",
        f"- ORT provider: `{result['ort_provider']}`",
        f"- provider order: `{', '.join(result['provider_order'])}`",
        "- proposal descriptionuse ORT; provider description TensorRT EP, automatically on failure CUDA/CPU fallback. ",
        "- scheduling descriptionuse `p2cccd_cpp.schedule_runtime_exact_work_items_from_proposal_arrays`, avoid Python per-row selection loop. ",
        "- descriptionreport proposal/scheduling wall time; descriptioncollision correctness description exact certificate/fallback layerguarantee. ",
        "",
        "## description",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| candidate rows | `{result['row_count']}` |",
        f"| scene/query groups | `{result['query_count']}` |",
        f"| positive query groups | `{result['positive_query_count']}` |",
        f"| ORT inference ms | `{result['ort_inference_ms']:.3f}` |",
        f"| C++ scheduling ms | `{result['cpp_schedule_ms']:.3f}` |",
        f"| proposal total ms | `{result['proposal_total_ms']:.3f}` |",
        f"| candidate rows/s, inference only | `{result['candidate_rows_per_second']:.1f}` |",
        f"| candidate rows/s, inference+schedule | `{result['proposal_rows_per_second']:.1f}` |",
        f"| work items emitted | `{result['work_item_count']}` |",
        "",
        "## C++ Schedule Stats",
        "",
        "| Field | Value |",
        "| --- | ---: |",
    ]
    for key, value in result["schedule_stats"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dense-shard", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--uncertainty-fallback-threshold", type=float, default=0.75)
    parser.add_argument("--warmup-passes", type=int, default=2)
    args = parser.parse_args()
    result = run_common_modeling_ort_walltime_benchmark(
        checkpoint_path=args.checkpoint,
        dense_shard_path=args.dense_shard,
        report_path=args.report,
        json_path=args.json,
        device=args.device,
        batch_size=args.batch_size,
        uncertainty_fallback_threshold=args.uncertainty_fallback_threshold,
        warmup_passes=args.warmup_passes,
    )
    print(json.dumps({"report": args.report.as_posix(), "provider": result["ort_provider"]}, indent=2))


if __name__ == "__main__":
    main()
