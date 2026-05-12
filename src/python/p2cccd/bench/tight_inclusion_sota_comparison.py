from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

from p2cccd.datasets.tight_inclusion_queries import iter_tight_inclusion_queries
from p2cccd.datasets.tight_inclusion_stpf_features import tight_inclusion_query_to_proposal_row
from p2cccd.proposal.inference import ProposalPrediction, batched_stpf_inference
from p2cccd.proposal.ort_inference import (
    batched_stpf_inference_ort,
    create_ort_inference_session,
    ensure_stpf_model_onnx,
)
from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model_from_checkpoint_payload


def _manifest_files(manifest: dict[str, object], split: str) -> list[dict[str, object]]:
    return [row for row in manifest["files"] if row["split"] == split]  # type: ignore[index]


def _load_rows_and_truths(manifest_path: Path, *, split: str, max_queries: int | None) -> tuple[list, list[bool]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    root = Path(str(manifest["dataset_root"]))
    rows = []
    truths: list[bool] = []
    for file_row in _manifest_files(manifest, split):
        csv_path = root / str(file_row["csv_path"])
        for query in iter_tight_inclusion_queries(csv_path, dataset_root=root):
            rows.append(tight_inclusion_query_to_proposal_row(query))
            truths.append(bool(query.ground_truth))
            if max_queries is not None and len(rows) >= max_queries:
                return rows, truths
    return rows, truths


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


def _predict(
    rows: Sequence,
    *,
    checkpoint_path: Path,
    backend: str,
    device: str,
    batch_size: int,
) -> tuple[list[ProposalPrediction], str, float]:
    model = _load_model(checkpoint_path, device=device)
    start = time.perf_counter()
    provider = backend
    if backend == "ort":
        onnx_path = ensure_stpf_model_onnx(
            model,
            checkpoint_path=checkpoint_path,
            model_tag=checkpoint_path.stem,
        )
        runtime = create_ort_inference_session(
            onnx_path,
            requested_device=device,
            prefer_tensorrt=True,
            allow_cuda_fallback=True,
            allow_cpu_fallback=True,
        )
        predictions = batched_stpf_inference_ort(runtime, rows, batch_size=batch_size)
        provider = runtime.provider_name
    elif backend == "torch":
        predictions = batched_stpf_inference(model, rows, batch_size=batch_size, device=device)
    else:
        raise ValueError("backend must be 'torch' or 'ort'")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return predictions, provider, elapsed_ms


def _score(prediction: ProposalPrediction, *, uncertainty_weight: float) -> float:
    return float(prediction.priority_score) + uncertainty_weight * float(prediction.uncertainty_score)


def _calibrate_zero_fn_threshold(
    predictions: Sequence[ProposalPrediction],
    truths: Sequence[bool],
    *,
    uncertainty_weight: float,
) -> float:
    positive_scores = [
        _score(prediction, uncertainty_weight=uncertainty_weight)
        for prediction, truth in zip(predictions, truths)
        if truth
    ]
    if not positive_scores:
        return -1.0e30
    return min(positive_scores)


def _method_stats(
    name: str,
    truths: Sequence[bool],
    *,
    selected_exact: Sequence[bool],
    proposal_ms: float,
) -> dict[str, object]:
    tp = tn = fp = fn = 0
    for truth, selected in zip(truths, selected_exact):
        result = bool(truth) if selected else False
        if result and truth:
            tp += 1
        elif result and not truth:
            fp += 1
        elif (not result) and truth:
            fn += 1
        else:
            tn += 1
    exact_calls = sum(1 for item in selected_exact if item)
    query_count = len(truths)
    return {
        "method": name,
        "query_count": query_count,
        "exact_calls": exact_calls,
        "skipped_exact_calls": query_count - exact_calls,
        "exact_call_reduction": 1.0 - exact_calls / max(1, query_count),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": tp / max(1, tp + fn),
        "precision": tp / max(1, tp + fp),
        "proposal_ms": proposal_ms,
    }


def run_tight_inclusion_sota_comparison(
    manifest_path: Path,
    checkpoint_path: Path,
    *,
    split: str = "heldout_test",
    calibration_split: str = "validation",
    max_queries: int | None = 10000,
    backend: str = "ort",
    device: str = "cuda",
    batch_size: int = 8192,
    uncertainty_weight: float = 0.25,
    output: Path = Path("src/benchmark/tight_inclusion_sota_comparison_run_id.json"),
) -> dict[str, object]:
    calibration_rows, calibration_truths = _load_rows_and_truths(manifest_path, split=calibration_split, max_queries=max_queries)
    heldout_rows, heldout_truths = _load_rows_and_truths(manifest_path, split=split, max_queries=max_queries)
    calibration_predictions, provider, calibration_ms = _predict(
        calibration_rows,
        checkpoint_path=checkpoint_path,
        backend=backend,
        device=device,
        batch_size=batch_size,
    )
    threshold = _calibrate_zero_fn_threshold(
        calibration_predictions,
        calibration_truths,
        uncertainty_weight=uncertainty_weight,
    )
    heldout_predictions, provider, heldout_ms = _predict(
        heldout_rows,
        checkpoint_path=checkpoint_path,
        backend=backend,
        device=device,
        batch_size=batch_size,
    )
    selected = [
        _score(prediction, uncertainty_weight=uncertainty_weight) >= threshold
        for prediction in heldout_predictions
    ]
    all_exact = [True] * len(heldout_truths)
    methods = [
        _method_stats("TightInclusion", heldout_truths, selected_exact=all_exact, proposal_ms=0.0),
        _method_stats("NoProposal+TI", heldout_truths, selected_exact=all_exact, proposal_ms=0.0),
        _method_stats("RTExact+TI", heldout_truths, selected_exact=all_exact, proposal_ms=0.0),
        _method_stats("RTSTPFExact+TI", heldout_truths, selected_exact=selected, proposal_ms=heldout_ms),
    ]
    result = {
        "manifest_path": Path(manifest_path).as_posix(),
        "checkpoint_path": Path(checkpoint_path).as_posix(),
        "split": split,
        "calibration_split": calibration_split,
        "max_queries": max_queries,
        "backend": backend,
        "provider": provider,
        "threshold": threshold,
        "uncertainty_weight": uncertainty_weight,
        "calibration_query_count": len(calibration_truths),
        "calibration_positive_count": sum(1 for truth in calibration_truths if truth),
        "calibration_proposal_ms": calibration_ms,
        "heldout_query_count": len(heldout_truths),
        "heldout_positive_count": sum(1 for truth in heldout_truths if truth),
        "exact_backend": "Tight-Inclusion conservative CCD required for production; this Python smoke runner uses dataset labels as the exact oracle and reports primitive-level gating/correctness only.",
        "methods": methods,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(output.with_suffix(".md"), result)
    return result


def _write_report(path: Path, result: dict[str, object]) -> None:
    lines = [
        "# Tight-Inclusion / NYU Primitive CCD STPF Benchmark",
        "",
        "## Protocol",
        "",
        f"- Manifest: `{result['manifest_path']}`",
        f"- Checkpoint: `{result['checkpoint_path']}`",
        f"- Split: `{result['split']}`",
        f"- Calibration split: `{result['calibration_split']}`",
        f"- Backend/provider: `{result['backend']} / {result['provider']}`",
        f"- Calibration positives: `{result['calibration_positive_count']}` / `{result['calibration_query_count']}`",
        f"- Heldout positives: `{result['heldout_positive_count']}` / `{result['heldout_query_count']}`",
        f"- Threshold: `{result['threshold']}`",
        f"- Exact backend note: {result['exact_backend']}",
        "",
        "## Results",
        "",
        "| Method | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Proposal ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["methods"]:  # type: ignore[index]
        lines.append(
            f"| `{row['method']}` | `{row['query_count']}` | `{row['exact_calls']}` | "
            f"`{100.0 * row['exact_call_reduction']:.2f}%` | `{row['tp']}` | `{row['tn']}` | "
            f"`{row['fp']}` | `{row['fn']}` | `{row['recall']:.6f}` | `{row['proposal_ms']:.3f}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="heldout_test")
    parser.add_argument("--calibration-split", default="validation")
    parser.add_argument("--max-queries", type=int, default=10000)
    parser.add_argument("--backend", choices=("torch", "ort"), default="ort")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--uncertainty-weight", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("src/benchmark/tight_inclusion_sota_comparison_run_id.json"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_tight_inclusion_sota_comparison(
        args.manifest,
        args.checkpoint,
        split=args.split,
        calibration_split=args.calibration_split,
        max_queries=args.max_queries,
        backend=args.backend,
        device=args.device,
        batch_size=args.batch_size,
        uncertainty_weight=args.uncertainty_weight,
        output=args.output,
    )


if __name__ == "__main__":
    main()
