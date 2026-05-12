from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from p2cccd.proposal.ort_inference import (
    batched_stpf_inference_ort_arrays,
    create_ort_inference_session,
    ensure_stpf_model_onnx,
)
from p2cccd.proposal.policy_head_selection import (
    apply_policy_head_to_prediction_arrays,
    select_rtstpf_policy_head,
)
from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model_from_checkpoint_payload

from .bvh_exact import _try_load_p2cccd_cpp
from .common_modeling_ort_walltime_benchmark import _feature_arrays_from_npz, _load_metadata


RUN_NAME = "native_dense_group_walltime_run_id"


@dataclass(frozen=True, slots=True)
class NativeDenseGroupCaseSpec:
    name: str
    checkpoint: Path
    dense_shard: Path
    uncertainty_fallback_threshold: float = 0.75
    representative_attempt_limit: int = 3
    interval_miss_penalty_scale: float = 0.22


def _repo_path(path: str) -> Path:
    return Path(path)


DEFAULT_CASES: tuple[NativeDenseGroupCaseSpec, ...] = (
    NativeDenseGroupCaseSpec(
        name="common_modeling_large",
        checkpoint=_repo_path(
            "src/outputs/stpf_training/"
            "common_modeling_high_density_scenarios_large_run_id_medium_mlp_10epoch/"
            "model_state.pt"
        ),
        dense_shard=_repo_path(
            "src/datasets/training/common_modeling_high_density/shards/"
            "common_modeling_high_density_scenarios_large_run_id/dense_eval.npz"
        ),
    ),
    NativeDenseGroupCaseSpec(
        name="fusion360_full_assembly",
        checkpoint=_repo_path(
            "src/outputs/stpf_training/fusion360_full_large_training_run_id/model_state.pt"
        ),
        dense_shard=_repo_path(
            "src/datasets/training/fusion360_full/shards/"
            "fusion360_full_large_training_run_id/dense_eval.npz"
        ),
    ),
    NativeDenseGroupCaseSpec(
        name="rtstpf_advantage_v4",
        checkpoint=_repo_path(
            "src/outputs/stpf_training/rtstpf_advantage_cases_v4_large_training_run_id/model_state.pt"
        ),
        dense_shard=_repo_path(
            "src/datasets/training/rtstpf_advantage_cases_v4/shards/"
            "rtstpf_advantage_cases_v4_large_training_run_id/dense_eval.npz"
        ),
    ),
    NativeDenseGroupCaseSpec(
        name="shapenet_ood_dense",
        checkpoint=_repo_path(
            "src/outputs/stpf_training/shapenet_ood_dense_cases_run_id/model_state.pt"
        ),
        dense_shard=_repo_path(
            "src/datasets/training/shapenet_ood_dense_cases/shards/"
            "shapenet_ood_dense_cases_run_id/dense_eval.npz"
        ),
    ),
)


def _load_model(checkpoint_path: Path, *, device: str):
    import torch

    torch_device = device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        torch_device = "cpu"
    payload = torch.load(checkpoint_path, map_location=torch_device)
    model, state_dict = build_stpf_model_from_checkpoint_payload(
        payload,
        fallback_preset=STPFModelPreset.MEDIUM_MLP,
    )
    model.load_state_dict(state_dict)
    model.to(torch_device)
    model.eval()
    return model


def _group_count(feature_arrays: dict[str, np.ndarray | int]) -> int:
    return int(np.unique(np.asarray(feature_arrays["query_id"], dtype=np.uint64)).shape[0])


def _positive_group_count(feature_arrays: dict[str, np.ndarray | int]) -> int:
    trace = np.asarray(feature_arrays["oracle_trace"], dtype=np.float64)
    query_ids = np.asarray(feature_arrays["query_id"], dtype=np.uint64)
    return int(np.unique(query_ids[trace[:, 0] > 0.5]).shape[0])


def _break_even_ms_per_work_unit(result: dict[str, Any]) -> float | None:
    saved_work = float(result["no_proposal_exact_work"]) - float(result["learned_exact_work"])
    if saved_work <= 0.0:
        return None
    # This is the real exact-kernel cost needed for saved work to amortize proposal overhead.
    overhead_ms = float(result["ort_inference_ms"]) + float(result["native_total_ms"])
    return overhead_ms / saved_work


def run_native_dense_group_case(
    spec: NativeDenseGroupCaseSpec,
    *,
    device: str = "cuda",
    batch_size: int = 65536,
    warmup_passes: int = 1,
) -> dict[str, Any]:
    if not spec.checkpoint.exists():
        raise FileNotFoundError(spec.checkpoint)
    if not spec.dense_shard.exists():
        raise FileNotFoundError(spec.dense_shard)

    cpp = _try_load_p2cccd_cpp()
    if cpp is None or not hasattr(cpp, "run_native_dense_group_exact_early_stop"):
        raise RuntimeError("p2cccd_cpp native dense group early-stop binding is unavailable")

    model = _load_model(spec.checkpoint, device=device)
    onnx_path = ensure_stpf_model_onnx(
        model,
        checkpoint_path=spec.checkpoint,
        output_path=spec.checkpoint.with_suffix(".onnx"),
        model_tag=spec.checkpoint.parent.name,
    )
    runtime = create_ort_inference_session(
        onnx_path,
        requested_device=device,
        prefer_tensorrt=True,
        allow_cuda_fallback=True,
        allow_cpu_fallback=True,
    )
    feature_arrays = _feature_arrays_from_npz(spec.dense_shard)
    row_count = int(np.asarray(feature_arrays["features"]).shape[0])

    warmup_ms: list[float] = []
    for _ in range(max(0, int(warmup_passes))):
        started = time.perf_counter()
        batched_stpf_inference_ort_arrays(
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
    density = row_count / max(1, _group_count(feature_arrays))
    policy_selection = select_rtstpf_policy_head(
        spec.name,
        candidate_density=float(density),
        hard_negative_group=False,
    )
    prediction_arrays = apply_policy_head_to_prediction_arrays(
        prediction_arrays,
        feature_arrays,
        head=policy_selection.head,
    )

    cxx_started = time.perf_counter()
    native_stats = dict(
        cpp.run_native_dense_group_exact_early_stop(
            feature_arrays,
            prediction_arrays,
            float(spec.uncertainty_fallback_threshold),
            int(spec.representative_attempt_limit),
            float(spec.interval_miss_penalty_scale),
            False,
        )
    )
    cxx_call_ms = (time.perf_counter() - cxx_started) * 1000.0

    result: dict[str, Any] = {
        "case": spec.name,
        "checkpoint": spec.checkpoint.as_posix(),
        "onnx": onnx_path.as_posix(),
        "dense_shard": spec.dense_shard.as_posix(),
        "metadata": _load_metadata(spec.dense_shard),
        "device": device,
        "batch_size": int(batch_size),
        "ort_provider": runtime.provider_name,
        "provider_order": list(runtime.provider_order),
        "policy_head": str(policy_selection.head),
        "policy_reason": policy_selection.reason,
        "warmup_ms": warmup_ms,
        "ort_inference_ms": inference_ms,
        "cxx_call_ms": cxx_call_ms,
        "row_count": row_count,
        "group_count_from_python": _group_count(feature_arrays),
        "positive_group_count_from_python": _positive_group_count(feature_arrays),
        **native_stats,
    }
    result["e2e_rtstpf_ms"] = float(result["ort_inference_ms"]) + float(result["cxx_call_ms"])
    result["proposal_rows_per_second"] = 1000.0 * row_count / max(1.0e-9, float(result["e2e_rtstpf_ms"]))
    result["break_even_ms_per_work_unit"] = _break_even_ms_per_work_unit(result)
    return result


def run_native_dense_group_suite(
    *,
    cases: Sequence[NativeDenseGroupCaseSpec] = DEFAULT_CASES,
    output_dir: str | Path = "src/benchmark",
    run_name: str = RUN_NAME,
    device: str = "cuda",
    batch_size: int = 65536,
    warmup_passes: int = 1,
) -> dict[str, Any]:
    results = [
        run_native_dense_group_case(
            spec,
            device=device,
            batch_size=batch_size,
            warmup_passes=warmup_passes,
        )
        for spec in cases
    ]
    payload: dict[str, Any] = {
        "run_name": run_name,
        "device": device,
        "batch_size": int(batch_size),
        "case_count": len(results),
        "cases": results,
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run_name}.json"
    md_path = out_dir / f"{run_name}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(md_path, payload)
    return payload


def _format_optional(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.9f}"


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Native dense group scheduling and exact early-stop wall-time table",
        "",
        "## Protocol",
        "",
        "- Python is responsible only fordescription, load shard, call ORT andsummarizereport; candidatedescription, group early-stop, fallback statisticsin `p2cccd_cpp.run_native_dense_group_exact_early_stop` description. ",
        "- Proposal inference descriptionuse ORT, defaultdescription `TensorrtExecutionProvider`, automatically on failure CUDA/CPU fallback. ",
        "- Exact layercurrentdescriptionuse dense shard within analytic proxy exact oracle / exact-cost trace as native early-stop driver; description TI/CUDA primitive exact hot path descriptioninsameconnectdescriptionunderdescription exact payload. ",
        "- STPF only determines candidate evaluation order / fallback policy, does not directly output final collision truth. ",
        "- default learned policy descriptionuse validation/source-aware head selection: descriptiondata sourcein `priority_only`, `cost_aware`, `risk_proximity_hybrid` descriptionfixedselect, avoid hard-negative group ondescription cost-aware head description random description. ",
        "",
        "## description",
        "",
        "| Dataset | Policy head | Groups | Candidates | Positive groups | ORT provider | Inference ms | C++ native ms | E2E RTSTPF ms | Exact calls | Call reduction | Work reduction | FN | Break-even ms/work |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["cases"]:
        lines.append(
            "| "
            f"`{item['case']}` | "
            f"`{item.get('policy_head', 'n/a')}` | "
            f"`{item['group_count']}` | "
            f"`{item['candidate_count']}` | "
            f"`{item['positive_group_count']}` | "
            f"`{item['ort_provider']}` | "
            f"`{item['ort_inference_ms']:.3f}` | "
            f"`{item['cxx_call_ms']:.3f}` | "
            f"`{item['e2e_rtstpf_ms']:.3f}` | "
            f"`{item['learned_exact_calls']}` | "
            f"`{item['exact_call_reduction']:.4%}` | "
            f"`{item['exact_work_reduction']:.4%}` | "
            f"`{item['fn']}` | "
            f"`{_format_optional(item['break_even_ms_per_work_unit'])}` |"
        )
    lines.extend(
        [
            "",
            "## splitdescriptionMetrics",
            "",
            "| Dataset | Parse ms | Schedule ms | Early-stop exact ms | Native total ms | NoProposal calls | NoProposal work | Learned work | Fallback calls | Interval miss | TP/TN/FP/FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in payload["cases"]:
        lines.append(
            "| "
            f"`{item['case']}` | "
            f"`{item['parse_ms']:.3f}` | "
            f"`{item['schedule_ms']:.3f}` | "
            f"`{item['exact_ms']:.3f}` | "
            f"`{item['native_total_ms']:.3f}` | "
            f"`{item['no_proposal_exact_calls']}` | "
            f"`{item['no_proposal_exact_work']:.4f}` | "
            f"`{item['learned_exact_work']:.4f}` | "
            f"`{item['learned_fallback_calls']}` | "
            f"`{item['learned_interval_miss_count']}` | "
            f"`{item['tp']}/{item['tn']}/{item['fp']}/{item['fn']}` |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- thisdescriptionreport native dense-group hot path: selection Filewritedescriptionand Python per-candidate scheduling descriptionfromdescriptionPathdescription. ",
            "- current exact driver descriptionis proxy oracle/cost trace, thereforedescriptionas native scheduling + early-stop realdescription; if used as final SOTA primitive wall-time description, description exact payload description Tight-Inclusion or CUDA primitive exact kernel. ",
            "- `Break-even ms/work` descriptionreal exact kernel each work unit descriptionwhen, learned early-stop description exact work descriptionwithdescription ORT + C++ scheduling overhead. ",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("src/benchmark"))
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument(
        "--case",
        action="append",
        choices=[spec.name for spec in DEFAULT_CASES],
        help="Run only selected default case; can be repeated.",
    )
    args = parser.parse_args()
    selected_cases = DEFAULT_CASES
    if args.case:
        allowed = set(args.case)
        selected_cases = tuple(spec for spec in DEFAULT_CASES if spec.name in allowed)
    payload = run_native_dense_group_suite(
        cases=selected_cases,
        output_dir=args.output_dir,
        run_name=args.run_name,
        device=args.device,
        batch_size=args.batch_size,
        warmup_passes=args.warmup_passes,
    )
    print(json.dumps({"run_name": payload["run_name"], "case_count": payload["case_count"]}, indent=2))


if __name__ == "__main__":
    main()
