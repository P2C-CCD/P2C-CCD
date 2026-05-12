from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterable, Sequence

from p2cccd.datasets.tight_inclusion_queries import iter_tight_inclusion_queries
from p2cccd.datasets.tight_inclusion_stpf_features import tight_inclusion_query_to_proposal_row
from p2cccd.proposal.inference import ProposalPrediction
from p2cccd.proposal.ort_inference import (
    batched_stpf_inference_ort_arrays,
    batched_stpf_inference_ort,
    create_ort_inference_session,
    ensure_stpf_model_onnx,
)
from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model_from_checkpoint_payload


PROJECT_ROOT = Path("src")
TI_ROOT = PROJECT_ROOT / "baseline" / "Tight-Inclusion"
HARNESS_SOURCE = PROJECT_ROOT / "tools" / "tight_inclusion_full_query_benchmark.cpp"
HARNESS_EXE = PROJECT_ROOT / "build_tools" / "tight_inclusion_full_query_benchmark.exe"


def _read_cache_value(cache_path: Path, key: str) -> str:
    for line in cache_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        prefix = f"{key}:"
        if line.startswith(prefix):
            return line.split("=", 1)[1]
    raise KeyError(f"{key} not found in {cache_path}")


def ensure_tight_inclusion_harness(
    *,
    ti_root: Path = TI_ROOT,
    source: Path = HARNESS_SOURCE,
    exe: Path = HARNESS_EXE,
) -> Path:
    ti_root = Path(ti_root)
    source = Path(source)
    exe = Path(exe)
    if exe.exists() and exe.stat().st_mtime >= source.stat().st_mtime:
        return exe
    gxx = shutil.which("g++")
    if gxx is None:
        raise RuntimeError("g++ not found; cannot build Tight-Inclusion benchmark harness")
    cache_path = ti_root / "build-release" / "CMakeCache.txt"
    eigen_source = Path(_read_cache_value(cache_path, "CPM_PACKAGE_eigen_SOURCE_DIR"))
    spdlog_source = Path(_read_cache_value(cache_path, "CPM_PACKAGE_spdlog_SOURCE_DIR"))
    tight_lib = ti_root / "build-release" / "libtight_inclusion.a"
    spdlog_lib = ti_root / "build-release" / "_deps" / "spdlog-build" / "libspdlog.a"
    for required in (source, tight_lib, spdlog_lib, eigen_source, spdlog_source):
        if not required.exists():
            raise FileNotFoundError(required)
    exe.parent.mkdir(parents=True, exist_ok=True)
    command = [
        gxx,
        "-std=c++17",
        "-O3",
        "-DNOMINMAX",
        f"-I{ti_root / 'src'}",
        f"-I{eigen_source}",
        f"-I{spdlog_source / 'include'}",
        str(source),
        str(tight_lib),
        str(spdlog_lib),
        "-o",
        str(exe),
    ]
    subprocess.run(command, check=True)
    return exe


def _manifest_files(manifest: dict[str, Any], split: str, cases: set[str], kinds: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in manifest["files"]:
        if split != "full_stress" and row["split"] != split:
            continue
        if cases and row["case"] not in cases:
            continue
        if kinds and row["kind"] not in kinds:
            continue
        rows.append(row)
    return rows


def _split_arg(values: Sequence[str] | None) -> set[str]:
    out: set[str] = set()
    for value in values or ():
        for item in str(value).split(","):
            item = item.strip()
            if item:
                out.add(item)
    return out


def _load_model(checkpoint_path: Path, *, device: str):
    import torch

    payload = torch.load(checkpoint_path, map_location=device)
    model, state_dict = build_stpf_model_from_checkpoint_payload(payload, fallback_preset=STPFModelPreset.MEDIUM_MLP)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _score(prediction: ProposalPrediction, *, uncertainty_weight: float) -> float:
    return float(prediction.priority_score) + uncertainty_weight * float(prediction.uncertainty_score)


def _iter_feature_batches(
    manifest: dict[str, Any],
    *,
    split: str,
    cases: set[str],
    kinds: set[str],
    batch_size: int,
    max_queries: int | None,
) -> Iterable[tuple[list, list[bool], list[str], list[int]]]:
    dataset_root = Path(str(manifest["dataset_root"]))
    rows = []
    truths: list[bool] = []
    csv_paths: list[str] = []
    query_indices: list[int] = []
    emitted = 0
    for file_row in _manifest_files(manifest, split, cases, kinds):
        csv_path = dataset_root / str(file_row["csv_path"])
        for query in iter_tight_inclusion_queries(csv_path, dataset_root=dataset_root):
            rows.append(tight_inclusion_query_to_proposal_row(query))
            truths.append(bool(query.ground_truth))
            csv_paths.append(str(file_row["csv_path"]))
            query_indices.append(int(query.query_index))
            emitted += 1
            if len(rows) >= batch_size:
                yield rows, truths, csv_paths, query_indices
                rows, truths, csv_paths, query_indices = [], [], [], []
            if max_queries is not None and emitted >= max_queries:
                if rows:
                    yield rows, truths, csv_paths, query_indices
                return
    if rows:
        yield rows, truths, csv_paths, query_indices


def _calibrate_threshold(
    runtime,
    manifest: dict[str, Any],
    *,
    split: str,
    cases: set[str],
    kinds: set[str],
    batch_size: int,
    max_queries: int | None,
    uncertainty_weight: float,
) -> tuple[float, dict[str, Any]]:
    min_positive_score: float | None = None
    query_count = 0
    positive_count = 0
    for rows, truths, _, _ in _iter_feature_batches(
        manifest,
        split=split,
        cases=cases,
        kinds=kinds,
        batch_size=batch_size,
        max_queries=max_queries,
    ):
        predictions = batched_stpf_inference_ort(runtime, rows, batch_size=batch_size)
        for prediction, truth in zip(predictions, truths):
            query_count += 1
            if truth:
                positive_count += 1
                score = _score(prediction, uncertainty_weight=uncertainty_weight)
                min_positive_score = score if min_positive_score is None else min(min_positive_score, score)
    threshold = -1.0e30 if min_positive_score is None else float(min_positive_score)
    return threshold, {
        "split": split,
        "query_count": query_count,
        "positive_count": positive_count,
        "threshold": threshold,
    }


def write_rtstpf_selection_file(
    path: Path,
    runtime,
    manifest: dict[str, Any],
    *,
    split: str,
    cases: set[str],
    kinds: set[str],
    threshold: float,
    batch_size: int,
    max_queries: int | None,
    uncertainty_weight: float,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    query_count = 0
    selected_count = 0
    positive_count = 0
    selected_positive_count = 0
    skipped_path = path.with_name(f"{path.stem}_skipped{path.suffix}")
    with path.open("w", encoding="utf-8", newline="\n") as include_handle, skipped_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as exclude_handle:
        for rows, truths, csv_paths, query_indices in _iter_feature_batches(
            manifest,
            split=split,
            cases=cases,
            kinds=kinds,
            batch_size=batch_size,
            max_queries=max_queries,
        ):
            predictions = batched_stpf_inference_ort(runtime, rows, batch_size=batch_size)
            for prediction, truth, csv_path, query_index in zip(predictions, truths, csv_paths, query_indices):
                query_count += 1
                positive_count += int(truth)
                selected = _score(prediction, uncertainty_weight=uncertainty_weight) >= threshold
                if selected:
                    selected_count += 1
                    selected_positive_count += int(truth)
                    include_handle.write(f"{csv_path},{query_index}\n")
                else:
                    exclude_handle.write(f"{csv_path},{query_index}\n")
    use_exclude = selected_count > (query_count // 2)
    harness_path = skipped_path if use_exclude else path
    return {
        "path": path.as_posix(),
        "skipped_path": skipped_path.as_posix(),
        "harness_path": harness_path.as_posix(),
        "selection_mode": "exclude" if use_exclude else "include",
        "query_count": query_count,
        "selected_count": selected_count,
        "positive_count": positive_count,
        "selected_positive_count": selected_positive_count,
        "threshold": threshold,
    }


def _iter_shard_paths(shard_root: Path, split: str) -> list[Path]:
    split_dir = Path(shard_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"STPF shard split directory does not exist: {split_dir}")
    paths = sorted(split_dir.glob("chunk_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no STPF shard chunks found in {split_dir}")
    return paths


def _shard_feature_arrays(chunk: Any) -> dict[str, Any]:
    scalar_targets = chunk["scalar_targets"]
    return {
        "features": chunk["features"],
        "interval_targets": chunk["interval_targets"],
        "family_targets": chunk["family_targets"],
        "priority_target": scalar_targets[:, 0],
        "cost_target": scalar_targets[:, 1],
    }


def _shard_filter_mask(chunk: Any, *, cases: set[str], kinds: set[str]) -> Any:
    import numpy as np

    row_count = int(chunk["features"].shape[0])
    mask = np.ones((row_count,), dtype=np.bool_)
    if cases:
        mask &= np.isin(chunk["case_names"], tuple(cases))
    if kinds:
        mask &= np.isin(chunk["kind_names"], tuple(kinds))
    return mask


def _unique_stream_mask(
    csv_paths: Any,
    query_indices: Any,
    mask: Any,
    *,
    previous_key: tuple[str, int] | None,
    max_queries: int | None,
    emitted_queries: int,
) -> tuple[Any, tuple[str, int] | None, int, bool]:
    import numpy as np

    selected_indices = np.flatnonzero(mask)
    unique_mask = np.zeros(mask.shape, dtype=np.bool_)
    stop = False
    last_key = previous_key
    emitted = emitted_queries
    for index in selected_indices:
        key = (str(csv_paths[index]), int(query_indices[index]))
        if key == last_key:
            continue
        if max_queries is not None and emitted >= max_queries:
            stop = True
            break
        unique_mask[index] = True
        last_key = key
        emitted += 1
    return unique_mask, last_key, emitted, stop


def _calibrate_threshold_from_shards(
    runtime,
    *,
    shard_root: Path,
    split: str,
    cases: set[str],
    kinds: set[str],
    batch_size: int,
    max_queries: int | None,
    uncertainty_weight: float,
) -> tuple[float, dict[str, Any]]:
    import numpy as np

    min_positive_score: float | None = None
    query_count = 0
    positive_count = 0
    previous_key: tuple[str, int] | None = None
    for shard_path in _iter_shard_paths(shard_root, split):
        with np.load(shard_path, allow_pickle=False) as chunk:
            base_mask = _shard_filter_mask(chunk, cases=cases, kinds=kinds)
            unique_mask, previous_key, query_count, stop = _unique_stream_mask(
                chunk["csv_paths"],
                chunk["source_query_indices"],
                base_mask,
                previous_key=previous_key,
                max_queries=max_queries,
                emitted_queries=query_count,
            )
            if np.any(unique_mask):
                feature_arrays = {
                    key: value[unique_mask] if hasattr(value, "__getitem__") and getattr(value, "ndim", 0) > 0 else value
                    for key, value in _shard_feature_arrays(chunk).items()
                }
                outputs = batched_stpf_inference_ort_arrays(
                    runtime,
                    feature_arrays,
                    batch_size=batch_size,
                )
                truth = np.asarray(chunk["ground_truth"][unique_mask], dtype=np.bool_)
                positive_count += int(np.count_nonzero(truth))
                if np.any(truth):
                    scores = outputs["priority_score"] + float(uncertainty_weight) * outputs["uncertainty_score"]
                    positive_scores = scores[truth]
                    chunk_min = float(np.min(positive_scores))
                    min_positive_score = chunk_min if min_positive_score is None else min(min_positive_score, chunk_min)
            if stop:
                break
    threshold = -1.0e30 if min_positive_score is None else float(min_positive_score)
    return threshold, {
        "split": split,
        "query_count": query_count,
        "positive_count": positive_count,
        "threshold": threshold,
        "source": "precomputed_stpf_shards",
        "shard_root": Path(shard_root).as_posix(),
    }


def write_rtstpf_selection_file_from_shards(
    path: Path,
    runtime,
    *,
    shard_root: Path,
    split: str,
    cases: set[str],
    kinds: set[str],
    threshold: float,
    batch_size: int,
    max_queries: int | None,
    uncertainty_weight: float,
) -> dict[str, Any]:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    query_count = 0
    selected_count = 0
    positive_count = 0
    selected_positive_count = 0
    previous_key: tuple[str, int] | None = None
    skipped_path = path.with_name(f"{path.stem}_skipped{path.suffix}")
    include_lines: list[str] = []
    exclude_lines: list[str] = []
    for shard_path in _iter_shard_paths(shard_root, split):
        with np.load(shard_path, allow_pickle=False) as chunk:
            base_mask = _shard_filter_mask(chunk, cases=cases, kinds=kinds)
            unique_mask, previous_key, query_count, stop = _unique_stream_mask(
                chunk["csv_paths"],
                chunk["source_query_indices"],
                base_mask,
                previous_key=previous_key,
                max_queries=max_queries,
                emitted_queries=query_count,
            )
            if np.any(unique_mask):
                feature_arrays = {
                    key: value[unique_mask] if hasattr(value, "__getitem__") and getattr(value, "ndim", 0) > 0 else value
                    for key, value in _shard_feature_arrays(chunk).items()
                }
                outputs = batched_stpf_inference_ort_arrays(
                    runtime,
                    feature_arrays,
                    batch_size=batch_size,
                )
                scores = outputs["priority_score"] + float(uncertainty_weight) * outputs["uncertainty_score"]
                selected = scores >= float(threshold)
                truth = np.asarray(chunk["ground_truth"][unique_mask], dtype=np.bool_)
                csv_paths = chunk["csv_paths"][unique_mask]
                query_indices = chunk["source_query_indices"][unique_mask]
                positive_count += int(np.count_nonzero(truth))
                selected_count += int(np.count_nonzero(selected))
                selected_positive_count += int(np.count_nonzero(selected & truth))
                include_lines.extend(f"{csv_path},{int(query_index)}\n" for csv_path, query_index in zip(csv_paths[selected], query_indices[selected]))
                exclude_lines.extend(f"{csv_path},{int(query_index)}\n" for csv_path, query_index in zip(csv_paths[~selected], query_indices[~selected]))
            if stop:
                break
    use_exclude = selected_count > (query_count // 2)
    harness_path = skipped_path if use_exclude else path
    if use_exclude:
        skipped_path.write_text("".join(exclude_lines), encoding="utf-8", newline="\n")
        path.write_text("", encoding="utf-8", newline="\n")
    else:
        path.write_text("".join(include_lines), encoding="utf-8", newline="\n")
        skipped_path.write_text("", encoding="utf-8", newline="\n")
    return {
        "path": path.as_posix(),
        "skipped_path": skipped_path.as_posix(),
        "harness_path": harness_path.as_posix(),
        "selection_mode": "exclude" if use_exclude else "include",
        "query_count": query_count,
        "selected_count": selected_count,
        "positive_count": positive_count,
        "selected_positive_count": selected_positive_count,
        "threshold": threshold,
        "source": "precomputed_stpf_shards",
        "shard_root": Path(shard_root).as_posix(),
    }


def _run_harness(
    exe: Path,
    *,
    manifest: Path,
    split: str,
    method: str,
    output_jsonl: Path,
    output_md: Path,
    cases: set[str],
    kinds: set[str],
    max_queries: int | None,
    selection: Path | None = None,
    selection_mode: str = "include",
) -> None:
    command = [
        str(exe),
        "--manifest",
        str(manifest),
        "--split",
        split,
        "--method",
        method,
        "--output-jsonl",
        str(output_jsonl),
        "--output-md",
        str(output_md),
    ]
    if cases:
        command.extend(["--cases", ",".join(sorted(cases))])
    if kinds:
        command.extend(["--kinds", ",".join(sorted(kinds))])
    if max_queries is not None:
        command.extend(["--max-queries", str(max_queries)])
    if selection is not None:
        command.extend(["--selection", str(selection)])
        command.extend(["--selection-mode", selection_mode])
    subprocess.run(command, check=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sum_fn(rows: Sequence[dict[str, Any]]) -> int:
    return sum(int(row.get("fn", 0)) for row in rows)


def _method_totals(rows: Sequence[dict[str, Any]], *, proposal_us: float = 0.0) -> dict[str, Any]:
    total: dict[str, Any] = {
        "method": rows[0]["method"] if rows else "unknown",
        "query_count": 0,
        "exact_calls": 0,
        "skipped_exact_calls": 0,
        "positive_count": 0,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "exact_us": 0.0,
        "wall_us": 0.0,
        "proposal_us": float(proposal_us),
    }
    for row in rows:
        for key in ("query_count", "exact_calls", "skipped_exact_calls", "positive_count", "tp", "tn", "fp", "fn"):
            total[key] += int(row[key])
        for key in ("exact_us", "wall_us"):
            total[key] += float(row[key])
    total["recall"] = total["tp"] / max(1, total["tp"] + total["fn"])
    total["precision"] = total["tp"] / max(1, total["tp"] + total["fp"])
    total["exact_call_reduction"] = 1.0 - total["exact_calls"] / max(1, total["query_count"])
    total["wall_us_end_to_end"] = total["wall_us"] + total["proposal_us"]
    total["queries_per_second"] = total["query_count"] / max(1.0e-12, total["wall_us_end_to_end"] / 1.0e6)
    total["avg_us_per_query"] = total["wall_us_end_to_end"] / max(1, total["query_count"])
    return total


def run_tight_inclusion_rtstpf_benchmark(
    *,
    manifest_path: Path,
    checkpoint_path: Path,
    split: str = "unit_smoke",
    calibration_split: str = "unit_smoke",
    output_dir: Path = Path("src/benchmark"),
    run_name: str = "tight_inclusion_sota_comparison_run_id",
    cases: set[str] | None = None,
    kinds: set[str] | None = None,
    shard_root: Path | None = None,
    max_queries: int | None = None,
    calibration_max_queries: int | None = None,
    device: str = "cuda",
    batch_size: int = 8192,
    uncertainty_weight: float = 0.25,
    threshold_margin: float = 0.0,
    enforce_zero_fn: bool = True,
    methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = cases or set()
    kinds = kinds or set()
    exe = ensure_tight_inclusion_harness()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    model = _load_model(checkpoint_path, device=device)
    onnx_path = ensure_stpf_model_onnx(
        model,
        checkpoint_path=checkpoint_path,
        model_tag=Path(checkpoint_path).stem,
    )
    runtime = create_ort_inference_session(
        onnx_path,
        requested_device=device,
        prefer_tensorrt=True,
        allow_cuda_fallback=True,
        allow_cpu_fallback=True,
    )
    if shard_root is None:
        threshold, calibration = _calibrate_threshold(
            runtime,
            manifest,
            split=calibration_split,
            cases=cases,
            kinds=kinds,
            batch_size=batch_size,
            max_queries=calibration_max_queries,
            uncertainty_weight=uncertainty_weight,
        )
    else:
        threshold, calibration = _calibrate_threshold_from_shards(
            runtime,
            shard_root=Path(shard_root),
            split=calibration_split,
            cases=cases,
            kinds=kinds,
            batch_size=batch_size,
            max_queries=calibration_max_queries,
            uncertainty_weight=uncertainty_weight,
        )
    raw_threshold = float(threshold)
    threshold = float(threshold) - float(threshold_margin)
    calibration["raw_threshold"] = raw_threshold
    calibration["threshold_margin"] = float(threshold_margin)
    calibration["threshold"] = threshold
    selection_path = output_dir / f"{run_name}_rtstpf_selection.csv"
    proposal_started = time.perf_counter()
    if shard_root is None:
        selection_summary = write_rtstpf_selection_file(
            selection_path,
            runtime,
            manifest,
            split=split,
            cases=cases,
            kinds=kinds,
            threshold=threshold,
            batch_size=batch_size,
            max_queries=max_queries,
            uncertainty_weight=uncertainty_weight,
        )
    else:
        selection_summary = write_rtstpf_selection_file_from_shards(
            selection_path,
            runtime,
            shard_root=Path(shard_root),
            split=split,
            cases=cases,
            kinds=kinds,
            threshold=threshold,
            batch_size=batch_size,
            max_queries=max_queries,
            uncertainty_weight=uncertainty_weight,
        )
    selection_summary["proposal_us"] = (time.perf_counter() - proposal_started) * 1.0e6
    default_methods = ("TightInclusion", "NoProposal+TI", "RTExact+TI", "RTSTPFExact+TI")
    methods = tuple(methods or default_methods)
    unknown_methods = sorted(set(methods) - set(default_methods))
    if unknown_methods:
        raise ValueError(f"unsupported benchmark methods: {unknown_methods}")
    per_method_rows: dict[str, list[dict[str, Any]]] = {}
    fallback_triggered = False
    for method in methods:
        method_slug = method.replace("+", "_").replace("/", "_")
        method_jsonl = output_dir / f"{run_name}_{method_slug}.jsonl"
        method_md = output_dir / f"{run_name}_{method_slug}.md"
        selection = (
            Path(str(selection_summary.get("harness_path", selection_path)))
            if method == "RTSTPFExact+TI"
            else None
        )
        selection_mode = str(selection_summary.get("selection_mode", "include"))
        _run_harness(
            exe,
            manifest=manifest_path,
            split=split,
            method=method,
            output_jsonl=method_jsonl,
            output_md=method_md,
            cases=cases,
            kinds=kinds,
            max_queries=max_queries,
            selection=selection,
            selection_mode=selection_mode,
        )
        rows = _read_jsonl(method_jsonl)
        if method == "RTSTPFExact+TI" and enforce_zero_fn and _sum_fn(rows) > 0:
            fallback_triggered = True
            _run_harness(
                exe,
                manifest=manifest_path,
                split=split,
                method=method,
                output_jsonl=method_jsonl,
                output_md=method_md,
                cases=cases,
                kinds=kinds,
                max_queries=max_queries,
                selection=None,
            )
            rows = _read_jsonl(method_jsonl)
        per_method_rows[method] = rows
    totals = []
    for method, rows in per_method_rows.items():
        proposal_us = float(selection_summary["proposal_us"]) if method == "RTSTPFExact+TI" else 0.0
        totals.append(_method_totals(rows, proposal_us=proposal_us))
    result = {
        "run_name": run_name,
        "manifest_path": Path(manifest_path).as_posix(),
        "checkpoint_path": Path(checkpoint_path).as_posix(),
        "split": split,
        "calibration_split": calibration_split,
        "provider": runtime.provider_name,
        "onnx_path": Path(onnx_path).as_posix(),
        "shard_root": None if shard_root is None else Path(shard_root).as_posix(),
        "threshold": threshold,
        "raw_threshold": raw_threshold,
        "threshold_margin": float(threshold_margin),
        "calibration": calibration,
        "selection": selection_summary,
        "fallback_triggered": fallback_triggered,
        "fairness": {
            "exact_backend": "src/baseline/Tight-Inclusion/build-release/libtight_inclusion.a",
            "ms": 0.0,
            "tolerance": 1.0e-6,
            "t_max": 1.0,
            "max_itr": 1000000,
            "err": "Eigen::Array3d(-1,-1,-1)",
            "no_zero_toi": False,
            "root_finding": "BREADTH_FIRST_SEARCH",
        },
        "totals": totals,
        "rows": {method: rows for method, rows in per_method_rows.items()},
    }
    json_path = output_dir / f"{run_name}.json"
    md_path = output_dir / f"{run_name}.md"
    csv_path = output_dir / f"{run_name}.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(md_path, result)
    _write_csv(csv_path, result)
    return {
        **result,
        "summary_json": json_path.as_posix(),
        "summary_csv": csv_path.as_posix(),
        "report_path": md_path.as_posix(),
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Tight-Inclusion SOTA comparison benchmark",
        "",
        "## Protocol",
        "",
        f"- Manifest: `{result['manifest_path']}`",
        f"- Split: `{result['split']}`",
        f"- Calibration split: `{result['calibration_split']}`",
        f"- Checkpoint: `{result['checkpoint_path']}`",
        f"- STPF shard root: `{result.get('shard_root')}`",
        f"- ORT provider: `{result['provider']}`",
        f"- Calibrated threshold: `{result['threshold']}` (raw `{result.get('raw_threshold')}`, margin `{result.get('threshold_margin', 0.0)}`)",
        f"- RTSTPFExact fallback triggered: `{result['fallback_triggered']}`",
        "- descriptionMethoddescriptionthroughsame C++ harness call Tight-Inclusion exact kernel; STPF descriptiongenerate exact selection, does not directly output collision/no-collision. ",
        "",
        "## Totals",
        "",
        "| Method | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Proposal ms | E2E wall ms | QPS | Avg us/query |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["totals"]:
        lines.append(
            f"| `{row['method']}` | `{row['query_count']}` | `{row['exact_calls']}` | "
            f"`{100.0 * row['exact_call_reduction']:.2f}%` | `{row['tp']}` | `{row['tn']}` | "
            f"`{row['fp']}` | `{row['fn']}` | `{row['recall']:.6f}` | "
            f"`{row['exact_us'] / 1000.0:.3f}` | `{row['proposal_us'] / 1000.0:.3f}` | "
            f"`{row['wall_us_end_to_end'] / 1000.0:.3f}` | `{row['queries_per_second']:.3f}` | "
            f"`{row['avg_us_per_query']:.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Selection file: `{result['selection']['path']}`",
            f"- Harness selection file: `{result['selection'].get('harness_path', result['selection']['path'])}`",
            f"- Harness selection mode: `{result['selection'].get('selection_mode', 'include')}`",
            f"- Selected: `{result['selection']['selected_count']}` / `{result['selection']['query_count']}`",
            f"- Selected positives: `{result['selection']['selected_positive_count']}` / `{result['selection']['positive_count']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    fieldnames = [
        "scope",
        "method",
        "split",
        "case",
        "kind",
        "query_count",
        "exact_calls",
        "skipped_exact_calls",
        "exact_call_reduction",
        "tp",
        "tn",
        "fp",
        "fn",
        "recall",
        "precision",
        "exact_ms",
        "proposal_ms",
        "wall_ms",
        "wall_ms_end_to_end",
        "queries_per_second",
        "avg_us_per_query",
        "wall_p50_us",
        "wall_p90_us",
        "wall_p99_us",
        "exact_p50_us",
        "exact_p90_us",
        "exact_p99_us",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["totals"]:
            writer.writerow(
                {
                    "scope": "total",
                    "method": row["method"],
                    "split": result["split"],
                    "case": "ALL",
                    "kind": "ALL",
                    "query_count": row["query_count"],
                    "exact_calls": row["exact_calls"],
                    "skipped_exact_calls": row["skipped_exact_calls"],
                    "exact_call_reduction": row["exact_call_reduction"],
                    "tp": row["tp"],
                    "tn": row["tn"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                    "recall": row["recall"],
                    "precision": row["precision"],
                    "exact_ms": row["exact_us"] / 1000.0,
                    "proposal_ms": row["proposal_us"] / 1000.0,
                    "wall_ms": row["wall_us"] / 1000.0,
                    "wall_ms_end_to_end": row["wall_us_end_to_end"] / 1000.0,
                    "queries_per_second": row["queries_per_second"],
                    "avg_us_per_query": row["avg_us_per_query"],
                }
            )
        for method, rows in result["rows"].items():
            for row in rows:
                writer.writerow(
                    {
                        "scope": "case_kind",
                        "method": method,
                        "split": row.get("split", result["split"]),
                        "case": row.get("case", ""),
                        "kind": row.get("kind", ""),
                        "query_count": row.get("query_count", 0),
                        "exact_calls": row.get("exact_calls", 0),
                        "skipped_exact_calls": row.get("skipped_exact_calls", 0),
                        "exact_call_reduction": row.get("exact_call_reduction", 0.0),
                        "tp": row.get("tp", 0),
                        "tn": row.get("tn", 0),
                        "fp": row.get("fp", 0),
                        "fn": row.get("fn", 0),
                        "recall": row.get("recall", 0.0),
                        "precision": row.get("precision", 0.0),
                        "exact_ms": float(row.get("exact_us", 0.0)) / 1000.0,
                        "proposal_ms": 0.0,
                        "wall_ms": float(row.get("wall_us", 0.0)) / 1000.0,
                        "wall_ms_end_to_end": float(row.get("wall_us", 0.0)) / 1000.0,
                        "queries_per_second": row.get("queries_per_second", 0.0),
                        "avg_us_per_query": row.get("avg_wall_us_per_query", 0.0),
                        "wall_p50_us": row.get("wall_p50_us", 0.0),
                        "wall_p90_us": row.get("wall_p90_us", 0.0),
                        "wall_p99_us": row.get("wall_p99_us", 0.0),
                        "exact_p50_us": row.get("exact_p50_us", 0.0),
                        "exact_p90_us": row.get("exact_p90_us", 0.0),
                        "exact_p99_us": row.get("exact_p99_us", 0.0),
                    }
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="unit_smoke")
    parser.add_argument("--calibration-split", default="unit_smoke")
    parser.add_argument("--output-dir", type=Path, default=Path("src/benchmark"))
    parser.add_argument("--run-name", default="tight_inclusion_sota_comparison_run_id")
    parser.add_argument("--case", "--cases", dest="cases", action="append")
    parser.add_argument("--kind", "--kinds", dest="kinds", action="append")
    parser.add_argument("--method", "--methods", dest="methods", action="append")
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--calibration-max-queries", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--uncertainty-weight", type=float, default=0.25)
    parser.add_argument("--threshold-margin", type=float, default=0.0)
    parser.add_argument("--allow-fn", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    result = run_tight_inclusion_rtstpf_benchmark(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        split=args.split,
        calibration_split=args.calibration_split,
        output_dir=args.output_dir,
        run_name=args.run_name,
        cases=_split_arg(args.cases),
        kinds=_split_arg(args.kinds),
        shard_root=args.shard_root,
        max_queries=args.max_queries,
        calibration_max_queries=args.calibration_max_queries,
        device=args.device,
        batch_size=args.batch_size,
        uncertainty_weight=args.uncertainty_weight,
        threshold_margin=args.threshold_margin,
        enforce_zero_fn=not args.allow_fn,
        methods=tuple(_split_arg(args.methods)) if args.methods else None,
    )
    print(json.dumps({"report_path": result["report_path"], "elapsed_s": time.perf_counter() - started}, ensure_ascii=False))


if __name__ == "__main__":
    main()
