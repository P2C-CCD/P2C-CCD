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

from .high_density_mesh_training_benchmark import (
    _load_abc_assets,
    _load_fusion360_assets,
    _load_thingi10k_assets,
    _make_pairs,
    _scale_workload_costs,
    _subset_workload,
)
from .large_dense_complex_mesh_cases import (
    LargeDenseComplexMeshCasesConfig,
    _build_dataset_from_cases as _build_large_dataset_from_cases,
    _make_heavy_cross_pairs,
    _make_heavy_intra_pairs,
    _split_pairs as _split_large_pairs,
    _unique_face_counts,
)
from .multi_dense_mesh_contact_pairs import (
    MultiDenseMeshContactPairsConfig,
    _build_dataset_from_case_pairs,
    _load_large_face_abc_assets,
    _make_cross_pairs,
    _rename_assets,
    _split_pairs as _split_multi_pairs,
    _unique_assets_from_pairs,
)
from .tight_inclusion_rtstpf_benchmark import _shard_feature_arrays, _unique_stream_mask
from .trained_stpf_high_density import (
    HighDensityMethodMetrics,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
)


DEFAULT_CHECKPOINT = Path(
    "src/outputs/stpf_training/rtstpf_paper_dataset_v2_paper_full_run_id/model_state.pt"
)
DEFAULT_PAPER_FULL_SHARDS = Path(
    "src/datasets/training/rtstpf_paper_dataset_v2/shards/rtstpf_paper_dataset_v2_paper_full_run_id"
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


def _score(outputs: dict[str, np.ndarray], *, uncertainty_weight: float) -> np.ndarray:
    return outputs["priority_score"] + float(uncertainty_weight) * outputs["uncertainty_score"]


def _iter_shards(shard_root: Path, split: str) -> list[Path]:
    split_dir = shard_root / split
    if not split_dir.exists():
        raise FileNotFoundError(split_dir)
    paths = sorted(split_dir.glob("chunk_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no chunk_*.npz under {split_dir}")
    return paths


def _calibrate_shard_threshold(
    runtime,
    shard_root: Path,
    *,
    split: str,
    batch_size: int,
    uncertainty_weight: float,
) -> dict[str, Any]:
    min_positive_score: float | None = None
    query_count = 0
    positive_count = 0
    previous_key: tuple[str, int] | None = None
    started = time.perf_counter()
    for shard_path in _iter_shards(shard_root, split):
        with np.load(shard_path, allow_pickle=False) as chunk:
            base_mask = np.ones(len(chunk["features"]), dtype=np.bool_)
            unique_mask, previous_key, query_count, _ = _unique_stream_mask(
                chunk["csv_paths"],
                chunk["source_query_indices"],
                base_mask,
                previous_key=previous_key,
                max_queries=None,
                emitted_queries=query_count,
            )
            if not np.any(unique_mask):
                continue
            feature_arrays = {
                key: value[unique_mask] if hasattr(value, "__getitem__") and getattr(value, "ndim", 0) > 0 else value
                for key, value in _shard_feature_arrays(chunk).items()
            }
            outputs = batched_stpf_inference_ort_arrays(runtime, feature_arrays, batch_size=batch_size)
            truths = np.asarray(chunk["ground_truth"][unique_mask], dtype=np.bool_)
            positive_count += int(np.count_nonzero(truths))
            if np.any(truths):
                scores = _score(outputs, uncertainty_weight=uncertainty_weight)
                local_min = float(np.min(scores[truths]))
                min_positive_score = local_min if min_positive_score is None else min(min_positive_score, local_min)
    threshold = -1.0e30 if min_positive_score is None else float(min_positive_score)
    return {
        "split": split,
        "query_count": query_count,
        "positive_count": positive_count,
        "threshold": threshold,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
    }


def _benchmark_shard_split(
    runtime,
    shard_root: Path,
    *,
    split: str,
    threshold: float,
    batch_size: int,
    uncertainty_weight: float,
) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    exact_calls = 0
    query_count = 0
    positive_count = 0
    previous_key: tuple[str, int] | None = None
    by_source: dict[str, dict[str, int]] = {}
    started = time.perf_counter()
    for shard_path in _iter_shards(shard_root, split):
        with np.load(shard_path, allow_pickle=False) as chunk:
            base_mask = np.ones(len(chunk["features"]), dtype=np.bool_)
            unique_mask, previous_key, query_count, _ = _unique_stream_mask(
                chunk["csv_paths"],
                chunk["source_query_indices"],
                base_mask,
                previous_key=previous_key,
                max_queries=None,
                emitted_queries=query_count,
            )
            if not np.any(unique_mask):
                continue
            feature_arrays = {
                key: value[unique_mask] if hasattr(value, "__getitem__") and getattr(value, "ndim", 0) > 0 else value
                for key, value in _shard_feature_arrays(chunk).items()
            }
            outputs = batched_stpf_inference_ort_arrays(runtime, feature_arrays, batch_size=batch_size)
            scores = _score(outputs, uncertainty_weight=uncertainty_weight)
            selected = scores >= float(threshold)
            truths = np.asarray(chunk["ground_truth"][unique_mask], dtype=np.bool_)
            sources = chunk["source_types"][unique_mask]
            exact_calls += int(np.count_nonzero(selected))
            positive_count += int(np.count_nonzero(truths))
            tp += int(np.count_nonzero(selected & truths))
            fp += int(np.count_nonzero(selected & ~truths))
            fn += int(np.count_nonzero(~selected & truths))
            tn += int(np.count_nonzero(~selected & ~truths))
            for source in np.unique(sources):
                source_mask = sources == source
                source_truth = truths[source_mask]
                source_selected = selected[source_mask]
                row = by_source.setdefault(str(source), {"queries": 0, "positives": 0, "exact_calls": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0})
                row["queries"] += int(np.count_nonzero(source_mask))
                row["positives"] += int(np.count_nonzero(source_truth))
                row["exact_calls"] += int(np.count_nonzero(source_selected))
                row["tp"] += int(np.count_nonzero(source_selected & source_truth))
                row["fp"] += int(np.count_nonzero(source_selected & ~source_truth))
                row["fn"] += int(np.count_nonzero(~source_selected & source_truth))
                row["tn"] += int(np.count_nonzero(~source_selected & ~source_truth))
    return {
        "split": split,
        "query_count": query_count,
        "positive_count": positive_count,
        "exact_calls": exact_calls,
        "skipped_exact_calls": query_count - exact_calls,
        "exact_call_reduction": 1.0 - exact_calls / max(1, query_count),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": tp / max(1, tp + fn),
        "precision": tp / max(1, tp + fp),
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "by_source": by_source,
    }


def _metric_dict(metric: HighDensityMethodMetrics) -> dict[str, Any]:
    return asdict(metric)


def _reduction(trained: HighDensityMethodMetrics, baseline: HighDensityMethodMetrics) -> float:
    return 1.0 - trained.exact_work_units / max(1.0e-9, baseline.exact_work_units)


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _run_multi_dense(model, *, device: str, batch_size: int) -> dict[str, Any]:
    model.to(device)
    model.eval()
    cfg = MultiDenseMeshContactPairsConfig()
    abc_top = _rename_assets(_load_abc_assets(Path(cfg.abc_root), cfg.asset_limit_per_source), "ABC top-face")
    abc_large = _load_large_face_abc_assets(cfg)
    fusion = _rename_assets(_load_fusion360_assets(Path(cfg.fusion360_root), cfg.asset_limit_per_source), "Fusion 360 Gallery")
    thingi = _rename_assets(_load_thingi10k_assets(Path(cfg.thingi10k_root), cfg.asset_limit_per_source), "Thingi10K")
    pairs_by_case = {
        "ABC-largeface-intra": _make_pairs(_rename_assets(abc_large, "ABC-largeface-intra"), cfg.pair_limit_per_case),
        "ABC-topface-intra": _make_pairs(_rename_assets(abc_top, "ABC-topface-intra"), cfg.pair_limit_per_case),
        "Fusion360-intra": _make_pairs(_rename_assets(fusion, "Fusion360-intra"), cfg.pair_limit_per_case),
        "Thingi10K-intra": _make_pairs(_rename_assets(thingi, "Thingi10K-intra"), cfg.pair_limit_per_case),
        "ABCtop-Fusion360-cross": _make_cross_pairs("ABCtop-Fusion360-cross", abc_top, fusion, limit=cfg.pair_limit_per_case),
        "ABCtop-Thingi10K-cross": _make_cross_pairs("ABCtop-Thingi10K-cross", abc_top, thingi, limit=cfg.pair_limit_per_case),
        "Fusion360-Thingi10K-cross": _make_cross_pairs("Fusion360-Thingi10K-cross", fusion, thingi, limit=cfg.pair_limit_per_case),
    }
    pairs_by_case = {case_name: pairs for case_name, pairs in pairs_by_case.items() if len(pairs) >= 2}
    train_pairs_by_case: dict[str, tuple] = {}
    eval_pairs_by_case: dict[str, tuple] = {}
    for offset, (case_name, pairs) in enumerate(pairs_by_case.items()):
        train_pairs, eval_pairs = _split_multi_pairs(pairs, train_fraction=cfg.train_fraction, seed=cfg.seed + offset)
        train_pairs_by_case[case_name] = train_pairs
        eval_pairs_by_case[case_name] = eval_pairs
    eval_dataset, eval_case_by_query_id, eval_costs = _build_dataset_from_case_pairs(
        eval_pairs_by_case,
        first_sample_id=11_000_001,
        samples_per_pair=cfg.samples_per_pair,
    )
    eval_workload = _scale_workload_costs(
        build_high_density_stpf_workload(eval_dataset, cfg.high_density, name=f"{cfg.run_name}_eval_checkpoint"),
        eval_costs,
    )
    baseline = benchmark_no_proposal_on_high_density_workload(eval_workload)
    trained = benchmark_stpf_on_high_density_workload(
        eval_workload,
        model=model,
        device=device,
        proposal_batch_size=batch_size,
        method_name="MultiDenseMesh-RTSTPFExact-PaperFullCheckpoint",
    )
    cases: list[dict[str, Any]] = []
    for case_name in sorted(eval_pairs_by_case):
        query_ids = {sample.query_id for sample in eval_dataset.samples if eval_case_by_query_id[sample.query_id] == case_name}
        case_workload = _subset_workload(eval_workload, query_ids, f"{cfg.run_name}_{case_name}_eval_checkpoint")
        case_baseline = benchmark_no_proposal_on_high_density_workload(case_workload)
        case_trained = benchmark_stpf_on_high_density_workload(
            case_workload,
            model=model,
            device=device,
            proposal_batch_size=batch_size,
            method_name=f"{case_name}-RTSTPFExact-PaperFullCheckpoint",
        )
        case_assets = _unique_assets_from_pairs([*train_pairs_by_case[case_name], *eval_pairs_by_case[case_name]])
        face_counts = sorted(asset.face_count for asset in case_assets)
        cases.append(
            {
                "case_name": case_name,
                "eval_queries": len(query_ids),
                "eval_candidates": case_workload.candidate_count,
                "face_min": face_counts[0],
                "face_median": face_counts[len(face_counts) // 2],
                "face_max": face_counts[-1],
                "no_proposal": _metric_dict(case_baseline),
                "rtstpf": _metric_dict(case_trained),
                "exact_work_reduction": _reduction(case_trained, case_baseline),
            }
        )
    return {
        "benchmark": "multi_dense_mesh_contact_pairs",
        "density": eval_workload.avg_candidates_per_query,
        "eval_queries": eval_workload.query_count,
        "eval_candidates": eval_workload.candidate_count,
        "no_proposal": _metric_dict(baseline),
        "rtstpf": _metric_dict(trained),
        "exact_work_reduction": _reduction(trained, baseline),
        "case_results": cases,
    }


def _run_large_dense(model, *, device: str, batch_size: int) -> dict[str, Any]:
    model.to(device)
    model.eval()
    cfg = LargeDenseComplexMeshCasesConfig()
    abc = _rename_assets(_load_abc_assets(Path(cfg.abc_root), cfg.asset_limit_per_source), "ABC megaface")
    fusion = _rename_assets(_load_fusion360_assets(Path(cfg.fusion360_root), cfg.asset_limit_per_source), "Fusion 360 dense")
    thingi = tuple(
        sorted(
            _rename_assets(_load_thingi10k_assets(Path(cfg.thingi10k_root), cfg.asset_limit_per_source), "Thingi10K dirty"),
            key=lambda asset: (-asset.dirty_score, -asset.face_count, asset.asset_id),
        )
    )
    pairs_by_case = {
        "L1-ABC-megaface-intra": _make_heavy_intra_pairs("L1-ABC-megaface-intra", abc, limit=cfg.pair_limit_per_case),
        "L2-ABCmegaface-Fusiondense-cross": _make_heavy_cross_pairs(
            "L2-ABCmegaface-Fusiondense-cross", abc, fusion, limit=cfg.pair_limit_per_case
        ),
        "L3-ABCmegaface-Thingi10Kdirty-cross": _make_heavy_cross_pairs(
            "L3-ABCmegaface-Thingi10Kdirty-cross", abc, thingi, limit=cfg.pair_limit_per_case
        ),
    }
    train_pairs_by_case: dict[str, tuple] = {}
    eval_pairs_by_case: dict[str, tuple] = {}
    for offset, (case_name, pairs) in enumerate(pairs_by_case.items()):
        train_pairs, eval_pairs = _split_large_pairs(pairs, train_fraction=cfg.train_fraction, seed=cfg.seed + offset)
        train_pairs_by_case[case_name] = train_pairs
        eval_pairs_by_case[case_name] = eval_pairs
    eval_dataset, eval_case_by_query_id, eval_costs = _build_large_dataset_from_cases(
        eval_pairs_by_case,
        first_sample_id=13_000_001,
        samples_per_pair=cfg.samples_per_pair,
    )
    eval_workload = _scale_workload_costs(
        build_high_density_stpf_workload(eval_dataset, cfg.high_density, name=f"{cfg.run_name}_eval_checkpoint"),
        eval_costs,
    )
    baseline = benchmark_no_proposal_on_high_density_workload(eval_workload)
    trained = benchmark_stpf_on_high_density_workload(
        eval_workload,
        model=model,
        device=device,
        proposal_batch_size=batch_size,
        method_name="LargeDenseComplex-RTSTPFExact-PaperFullCheckpoint",
    )
    cases: list[dict[str, Any]] = []
    for case_name in sorted(eval_pairs_by_case):
        query_ids = {sample.query_id for sample in eval_dataset.samples if eval_case_by_query_id[sample.query_id] == case_name}
        case_workload = _subset_workload(eval_workload, query_ids, f"{cfg.run_name}_{case_name}_eval_checkpoint")
        case_baseline = benchmark_no_proposal_on_high_density_workload(case_workload)
        case_trained = benchmark_stpf_on_high_density_workload(
            case_workload,
            model=model,
            device=device,
            proposal_batch_size=batch_size,
            method_name=f"{case_name}-RTSTPFExact-PaperFullCheckpoint",
        )
        face_counts = _unique_face_counts([*train_pairs_by_case[case_name], *eval_pairs_by_case[case_name]])
        cases.append(
            {
                "case_name": case_name,
                "eval_queries": len(query_ids),
                "eval_candidates": case_workload.candidate_count,
                "face_min": face_counts[0],
                "face_median": face_counts[len(face_counts) // 2],
                "face_max": face_counts[-1],
                "no_proposal": _metric_dict(case_baseline),
                "rtstpf": _metric_dict(case_trained),
                "exact_work_reduction": _reduction(case_trained, case_baseline),
            }
        )
    return {
        "benchmark": "large_dense_complex_mesh_cases",
        "density": eval_workload.avg_candidates_per_query,
        "eval_queries": eval_workload.query_count,
        "eval_candidates": eval_workload.candidate_count,
        "no_proposal": _metric_dict(baseline),
        "rtstpf": _metric_dict(trained),
        "exact_work_reduction": _reduction(trained, baseline),
        "case_results": cases,
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    shard = result["paper_full_shard"]
    dense = result["dense"]
    lines = [
        "# RTSTPFExact Paper-Full Checkpoint complete Benchmark",
        "",
        "## Protocol",
        "",
        f"- Checkpoint: `{result['checkpoint_path']}`",
        f"- ONNX: `{result['onnx_path']}`",
        f"- ORT provider: `{result['ort_provider']}`",
        f"- Device: `{result['device']}`",
        "- `RTSTPFExact` uses only learned STPF; collisiondescriptionConclusionshould stilldescription exact/fallback conservative guarantee. ",
        "",
        "## Paper-Full Heldout Shard",
        "",
        f"- Shard root: `{shard['shard_root']}`",
        f"- Calibration split: `{shard['calibration']['split']}`",
        f"- Heldout split: `{shard['heldout']['split']}`",
        f"- Threshold: `{shard['calibration']['threshold']}`",
        "",
        "| Split | Queries | Positives | Exact calls | Reduction | TP | TN | FP | FN | Recall | Wall ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (shard["calibration"], shard["heldout"]):
        if "exact_calls" not in row:
            lines.append(
                f"| `{row['split']}` | `{row['query_count']}` | `{row['positive_count']}` | `n/a` | `n/a` | "
                f"`n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `{row['wall_ms']:.3f}` |"
            )
            continue
        lines.append(
            f"| `{row['split']}` | `{row['query_count']}` | `{row['positive_count']}` | `{row['exact_calls']}` | "
            f"`{_pct(row['exact_call_reduction'])}` | `{row['tp']}` | `{row['tn']}` | `{row['fp']}` | `{row['fn']}` | "
            f"`{row['recall']:.6f}` | `{row['wall_ms']:.3f}` |"
        )
    lines.extend(["", "### Heldout By Source", "", "| Source | Queries | Positives | Exact calls | FN | Recall |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for source, row in sorted(shard["heldout"]["by_source"].items()):
        recall = row["tp"] / max(1, row["tp"] + row["fn"])
        lines.append(
            f"| `{source}` | `{row['queries']}` | `{row['positives']}` | `{row['exact_calls']}` | `{row['fn']}` | `{recall:.6f}` |"
        )
    lines.extend(
        [
            "",
            "## Dense / High-Cost Workloads",
            "",
            "| Benchmark | Density | Eval queries | Eval candidates | NoProposal work | RTSTPF work | Reduction | Exact calls | FN | Proposal ms | Total ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dense:
        baseline = row["no_proposal"]
        rtstpf = row["rtstpf"]
        lines.append(
            f"| `{row['benchmark']}` | `{row['density']:.0f}` | `{row['eval_queries']}` | `{row['eval_candidates']}` | "
            f"`{baseline['exact_work_units']:.1f}` | `{rtstpf['exact_work_units']:.1f}` | `{_pct(row['exact_work_reduction'])}` | "
            f"`{rtstpf['exact_call_count']}` | `{rtstpf['fn_count']}` | `{rtstpf['proposal_wall_ms']:.3f}` | `{rtstpf['total_wall_ms']:.3f}` |"
        )
    lines.extend(["", "## Dense Case Breakdown", ""])
    for row in dense:
        lines.extend(
            [
                f"### {row['benchmark']}",
                "",
                "| Case | Queries | Candidates | Face min/median/max | NoProposal work | RTSTPF work | Reduction | Exact calls | FN |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for case in row["case_results"]:
            baseline = case["no_proposal"]
            rtstpf = case["rtstpf"]
            lines.append(
                f"| `{case['case_name']}` | `{case['eval_queries']}` | `{case['eval_candidates']}` | "
                f"`{case['face_min']}/{case['face_median']}/{case['face_max']}` | "
                f"`{baseline['exact_work_units']:.1f}` | `{rtstpf['exact_work_units']:.1f}` | "
                f"`{_pct(case['exact_work_reduction'])}` | `{rtstpf['exact_call_count']}` | `{rtstpf['fn_count']}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Conclusion",
            "",
            "- in `paper_full heldout` on, zero-FN threshold isdescription, descriptionused for correctness-preserving gating description. ",
            "- in dense/high-cost workload on, new checkpoint descriptionreduction exact work; this isthis paperMethoddescriptionadvantagescene. ",
            "- ifdescriptionadvantagedescription wall-time description, description C++/CUDA level proposal feature construction + scheduling, avoid Python runner overheaddescription. ",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_complete_checkpoint_benchmark(
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    paper_full_shard_root: Path = DEFAULT_PAPER_FULL_SHARDS,
    output_prefix: Path = Path("src/benchmark/rtstpf_paper_full_checkpoint_complete_benchmark_run_id"),
    device: str = "cuda",
    batch_size: int = 32768,
    uncertainty_weight: float = 0.25,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    model = _load_model(checkpoint_path, device=device)
    onnx_path = ensure_stpf_model_onnx(model, checkpoint_path=checkpoint_path, model_tag=checkpoint_path.stem)
    # ONNX export uses model.cpu(); move it back before PyTorch dense benchmarks.
    model.to(device)
    model.eval()
    runtime = create_ort_inference_session(
        onnx_path,
        requested_device=device,
        prefer_tensorrt=True,
        allow_cuda_fallback=True,
        allow_cpu_fallback=True,
    )
    calibration = _calibrate_shard_threshold(
        runtime,
        paper_full_shard_root,
        split="validation",
        batch_size=batch_size,
        uncertainty_weight=uncertainty_weight,
    )
    heldout = _benchmark_shard_split(
        runtime,
        paper_full_shard_root,
        split="heldout_test",
        threshold=float(calibration["threshold"]),
        batch_size=batch_size,
        uncertainty_weight=uncertainty_weight,
    )
    dense = [
        _run_multi_dense(model, device=device, batch_size=batch_size),
        _run_large_dense(model, device=device, batch_size=batch_size),
    ]
    result = {
        "checkpoint_path": checkpoint_path.as_posix(),
        "onnx_path": Path(onnx_path).as_posix(),
        "ort_provider": runtime.provider_name,
        "device": device,
        "batch_size": batch_size,
        "uncertainty_weight": uncertainty_weight,
        "paper_full_shard": {
            "shard_root": paper_full_shard_root.as_posix(),
            "calibration": calibration,
            "heldout": heldout,
        },
        "dense": dense,
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(md_path, result)
    return {**result, "summary_json": json_path.as_posix(), "report_path": md_path.as_posix()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--paper-full-shard-root", type=Path, default=DEFAULT_PAPER_FULL_SHARDS)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("src/benchmark/rtstpf_paper_full_checkpoint_complete_benchmark_run_id"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--uncertainty-weight", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    result = run_complete_checkpoint_benchmark(
        checkpoint_path=args.checkpoint,
        paper_full_shard_root=args.paper_full_shard_root,
        output_prefix=args.output_prefix,
        device=args.device,
        batch_size=args.batch_size,
        uncertainty_weight=args.uncertainty_weight,
    )
    print(json.dumps({"report_path": result["report_path"], "elapsed_s": time.perf_counter() - started}, ensure_ascii=False))


if __name__ == "__main__":
    main()
