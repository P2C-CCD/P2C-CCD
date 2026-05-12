from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .features import PROPOSAL_FEATURE_DIM, ProposalFeatureRow, validate_proposal_feature_row
from .inference import (
    DEFAULT_OOD_ABS_FEATURE_THRESHOLD,
    ProposalPrediction,
    dummy_proposal_policy,
    is_ood_feature_row,
    validate_proposal_prediction,
)


DEFAULT_ORT_OPSET_VERSION = 17
DEFAULT_ORT_ENGINE_CACHE_ROOT = Path("src/outputs/ort_engine_cache")
DEFAULT_STPF_ONNX_ROOT = Path("src/outputs/stpf_onnx")
_DLL_DIR_HANDLES: list[Any] = []
_REGISTERED_PATH_DIRS: set[str] = set()


def _candidate_runtime_dirs() -> tuple[Path, ...]:
    candidates: list[Path] = []
    env_names = ("P2CCCD_TENSORRT_ROOT", "TENSORRT_ROOT", "TENSORRT_HOME", "CUDA_PATH", "CUDA_HOME", "CONDA_PREFIX")
    env_paths = {name: os.environ.get(name) for name in env_names}

    trt_root = env_paths.get("P2CCCD_TENSORRT_ROOT") or env_paths.get("TENSORRT_ROOT") or env_paths.get("TENSORRT_HOME")
    if trt_root:
        candidates.append(Path(trt_root) / "bin")

    cuda_root = env_paths.get("CUDA_PATH") or env_paths.get("CUDA_HOME")
    if cuda_root:
        candidates.append(Path(cuda_root) / "bin")

    conda_prefix = env_paths.get("CONDA_PREFIX")
    if conda_prefix:
        conda_root = Path(conda_prefix)
        candidates.append(conda_root / "Library" / "bin")
        candidates.append(conda_root / "Lib" / "site-packages" / "torch" / "lib")

    return tuple(path for path in candidates if path.exists())


def _register_runtime_dll_directories() -> None:
    path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    updated_path_entries = list(path_entries)
    if not hasattr(os, "add_dll_directory"):
        for directory in _candidate_runtime_dirs():
            directory_str = str(directory)
            if directory_str in _REGISTERED_PATH_DIRS:
                continue
            updated_path_entries.insert(0, directory_str)
            _REGISTERED_PATH_DIRS.add(directory_str)
        os.environ["PATH"] = os.pathsep.join(updated_path_entries)
        return
    registered = {str(getattr(handle, "path", "")) for handle in _DLL_DIR_HANDLES}
    for directory in _candidate_runtime_dirs():
        directory_str = str(directory)
        if directory_str not in _REGISTERED_PATH_DIRS:
            updated_path_entries.insert(0, directory_str)
            _REGISTERED_PATH_DIRS.add(directory_str)
        if directory_str in registered:
            continue
        try:
            handle = os.add_dll_directory(directory_str)
        except OSError:
            continue
        _DLL_DIR_HANDLES.append(handle)
        registered.add(directory_str)
    os.environ["PATH"] = os.pathsep.join(updated_path_entries)


@dataclass(frozen=True, slots=True)
class ORTInferenceSession:
    session: Any
    onnx_path: Path
    provider_name: str
    provider_order: tuple[str, ...]


def _normalized_provider_names(providers: Sequence[Any]) -> tuple[str, ...]:
    names: list[str] = []
    for provider in providers:
        if isinstance(provider, tuple):
            names.append(str(provider[0]))
        else:
            names.append(str(provider))
    return tuple(names)


def _provider_attempts(
    *,
    requested_device: str | None,
    prefer_tensorrt: bool,
    allow_cuda_fallback: bool,
    allow_cpu_fallback: bool,
    engine_cache_root: Path,
) -> tuple[tuple[Any, ...], ...]:
    normalized_device = "cpu" if requested_device is None else str(requested_device).lower()
    if normalized_device.startswith("cpu"):
        return (("CPUExecutionProvider",),)

    engine_cache_root.mkdir(parents=True, exist_ok=True)
    tensorrt_provider = (
        "TensorrtExecutionProvider",
        {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(engine_cache_root),
            "trt_fp16_enable": True,
        },
    )
    cuda_provider = ("CUDAExecutionProvider", {})
    cpu_provider = "CPUExecutionProvider"

    attempts: list[tuple[Any, ...]] = []
    if prefer_tensorrt:
        providers: list[Any] = [tensorrt_provider]
        if allow_cuda_fallback:
            providers.append(cuda_provider)
        if allow_cpu_fallback:
            providers.append(cpu_provider)
        attempts.append(tuple(providers))
    if allow_cuda_fallback:
        providers = [cuda_provider]
        if allow_cpu_fallback:
            providers.append(cpu_provider)
        attempts.append(tuple(providers))
    if allow_cpu_fallback:
        attempts.append((cpu_provider,))
    if not attempts:
        attempts.append((cpu_provider,))
    return tuple(attempts)


def export_stpf_model_to_onnx(
    model: Any,
    output_path: str | Path,
    *,
    opset_version: int = DEFAULT_ORT_OPSET_VERSION,
) -> Path:
    import torch
    from torch import nn

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    class _ExportWrapper(nn.Module):
        def __init__(self, wrapped_model: Any):
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(self, features: torch.Tensor):
            result = self.wrapped_model(features)
            return (
                result.interval_logits,
                result.family_logits,
                result.priority_score,
                result.cost_score,
                result.uncertainty_score,
            )

    was_training = bool(getattr(model, "training", False))
    export_model = _ExportWrapper(model.cpu())
    export_model.eval()
    dummy_input = torch.zeros((1, PROPOSAL_FEATURE_DIM), dtype=torch.float32)
    try:
        with torch.no_grad():
            torch.onnx.export(
                export_model,
                dummy_input,
                output,
                input_names=["features"],
                output_names=[
                    "interval_logits",
                    "family_logits",
                    "priority_score",
                    "cost_score",
                    "uncertainty_score",
                ],
                dynamic_axes={
                    "features": {0: "batch"},
                    "interval_logits": {0: "batch"},
                    "family_logits": {0: "batch"},
                    "priority_score": {0: "batch"},
                    "cost_score": {0: "batch"},
                    "uncertainty_score": {0: "batch"},
                },
                opset_version=max(int(opset_version), 18),
                dynamo=False,
            )
    finally:
        if was_training:
            model.train()
    return output


def ensure_stpf_model_onnx(
    model: Any,
    *,
    output_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    model_tag: str,
    opset_version: int = DEFAULT_ORT_OPSET_VERSION,
) -> Path:
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    if output_path is None:
        onnx_name = f"{model_tag}.onnx"
        if checkpoint is not None:
            output = checkpoint.with_suffix(".onnx")
        else:
            output = DEFAULT_STPF_ONNX_ROOT / onnx_name
    else:
        output = Path(output_path)
    needs_export = not output.exists()
    if checkpoint is not None and checkpoint.exists() and output.exists():
        needs_export = checkpoint.stat().st_mtime > output.stat().st_mtime
    if needs_export:
        export_stpf_model_to_onnx(model, output, opset_version=opset_version)
    return output


def create_ort_inference_session(
    onnx_path: str | Path,
    *,
    requested_device: str | None,
    prefer_tensorrt: bool,
    allow_cuda_fallback: bool,
    allow_cpu_fallback: bool = True,
    engine_cache_root: str | Path = DEFAULT_ORT_ENGINE_CACHE_ROOT,
) -> ORTInferenceSession:
    _register_runtime_dll_directories()
    import onnxruntime as ort

    model_path = Path(onnx_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ORT model path does not exist: {model_path}")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    attempts = _provider_attempts(
        requested_device=requested_device,
        prefer_tensorrt=prefer_tensorrt,
        allow_cuda_fallback=allow_cuda_fallback,
        allow_cpu_fallback=allow_cpu_fallback,
        engine_cache_root=Path(engine_cache_root),
    )
    errors: list[str] = []
    for providers in attempts:
        try:
            session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=list(providers),
            )
            provider_names = _normalized_provider_names(providers)
            active_provider = session.get_providers()[0] if session.get_providers() else provider_names[0]
            return ORTInferenceSession(
                session=session,
                onnx_path=model_path,
                provider_name=str(active_provider),
                provider_order=provider_names,
            )
        except Exception as exc:
            errors.append(f"{_normalized_provider_names(providers)} -> {exc}")
    joined = "; ".join(errors) if errors else "no ORT provider attempts were made"
    raise RuntimeError(f"failed to initialize ORT STPF session for {model_path}: {joined}")


def batched_stpf_inference_ort(
    runtime: ORTInferenceSession,
    rows: Sequence[ProposalFeatureRow],
    *,
    batch_size: int = 1024,
    ood_abs_feature_threshold: float | None = DEFAULT_OOD_ABS_FEATURE_THRESHOLD,
) -> list[ProposalPrediction]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if ood_abs_feature_threshold is not None and ood_abs_feature_threshold <= 0.0:
        raise ValueError("ood_abs_feature_threshold must be positive when provided")
    if not rows:
        return []

    predictions: list[ProposalPrediction | None] = [None] * len(rows)
    inference_rows: list[ProposalFeatureRow] = []
    inference_indices: list[int] = []
    for index, row in enumerate(rows):
        validate_proposal_feature_row(row)
        if ood_abs_feature_threshold is not None and is_ood_feature_row(
            row,
            abs_feature_threshold=ood_abs_feature_threshold,
        ):
            predictions[index] = dummy_proposal_policy(
                [row],
                ood_abs_feature_threshold=ood_abs_feature_threshold,
            )[0]
        else:
            inference_rows.append(row)
            inference_indices.append(index)

    if inference_rows:
        input_name = runtime.session.get_inputs()[0].name
        output_names = [output.name for output in runtime.session.get_outputs()]
        for start in range(0, len(inference_rows), batch_size):
            batch_rows = inference_rows[start : start + batch_size]
            batch_indices = inference_indices[start : start + batch_size]
            features = np.asarray([row.features for row in batch_rows], dtype=np.float32)
            outputs = runtime.session.run(output_names, {input_name: features})
            interval_logits, family_logits, priority_score, cost_score, uncertainty_score = outputs
            interval_scores = np.exp(interval_logits - np.max(interval_logits, axis=1, keepdims=True))
            interval_scores /= np.maximum(interval_scores.sum(axis=1, keepdims=True), 1.0e-12)
            family_scores = _sigmoid_stable(family_logits)
            priority_values = np.asarray(priority_score, dtype=np.float32).reshape(-1)
            cost_values = np.asarray(cost_score, dtype=np.float32).reshape(-1)
            uncertainty_values = np.asarray(uncertainty_score, dtype=np.float32).reshape(-1)
            for local_index, row in enumerate(batch_rows):
                prediction = ProposalPrediction(
                    candidate_id=row.candidate_id,
                    interval_scores=[float(value) for value in interval_scores[local_index].tolist()],
                    family_scores=[float(value) for value in family_scores[local_index].tolist()],
                    priority_score=float(priority_values[local_index]),
                    cost_score=float(cost_values[local_index]),
                    uncertainty_score=float(uncertainty_values[local_index]),
                    source=f"stpf_ort:{runtime.provider_name}",
                )
                validate_proposal_prediction(prediction)
                predictions[batch_indices[local_index]] = prediction

    if any(prediction is None for prediction in predictions):
        raise RuntimeError("ORT proposal inference failed to produce one prediction per feature row")
    return [prediction for prediction in predictions if prediction is not None]


def _normalize_score_rows(
    values: np.ndarray,
    *,
    fallback_indices: tuple[int, ...],
) -> np.ndarray:
    cleaned = np.where(np.isfinite(values) & (values >= 0.0), values, 0.0).astype(np.float32, copy=False)
    totals = cleaned.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cleaned,
        np.maximum(totals, 1.0e-12),
        out=np.zeros_like(cleaned, dtype=np.float32),
        where=totals > 0.0,
    )
    zero_mask = np.squeeze(totals <= 0.0, axis=1)
    if np.any(zero_mask):
        normalized[zero_mask, :] = 0.0
        fallback_weight = np.float32(1.0 / max(1, len(fallback_indices)))
        for index in fallback_indices:
            normalized[zero_mask, index] = fallback_weight
    return normalized


def _sigmoid_stable(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float32)
    positive_mask = logits >= 0.0
    result = np.empty_like(logits, dtype=np.float32)
    result[positive_mask] = 1.0 / (1.0 + np.exp(-logits[positive_mask]))
    exp_values = np.exp(logits[~positive_mask])
    result[~positive_mask] = exp_values / (1.0 + exp_values)
    return result


def batched_stpf_inference_ort_arrays(
    runtime: ORTInferenceSession,
    feature_arrays: Mapping[str, np.ndarray],
    *,
    batch_size: int = 1024,
    ood_abs_feature_threshold: float | None = DEFAULT_OOD_ABS_FEATURE_THRESHOLD,
) -> dict[str, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if ood_abs_feature_threshold is not None and ood_abs_feature_threshold <= 0.0:
        raise ValueError("ood_abs_feature_threshold must be positive when provided")

    features = np.asarray(feature_arrays["features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != PROPOSAL_FEATURE_DIM:
        raise ValueError(
            f"feature_arrays['features'] must have shape [N, {PROPOSAL_FEATURE_DIM}]"
        )
    row_count = int(features.shape[0])
    if row_count == 0:
        return {
            "interval_scores": np.zeros((0, 8), dtype=np.float32),
            "family_scores": np.zeros((0, 8), dtype=np.float32),
            "priority_score": np.zeros((0,), dtype=np.float32),
            "cost_score": np.zeros((0,), dtype=np.float32),
            "uncertainty_score": np.zeros((0,), dtype=np.float32),
            "ood_mask": np.zeros((0,), dtype=np.bool_),
        }

    interval_targets = np.asarray(feature_arrays["interval_targets"], dtype=np.float32)
    family_targets = np.asarray(feature_arrays["family_targets"], dtype=np.float32)
    priority_targets = np.asarray(feature_arrays["priority_target"], dtype=np.float32).reshape(-1)
    cost_targets = np.asarray(feature_arrays["cost_target"], dtype=np.float32).reshape(-1)
    if interval_targets.shape != (row_count, 8):
        raise ValueError("feature_arrays['interval_targets'] must have shape [N, 8]")
    if family_targets.shape != (row_count, 8):
        raise ValueError("feature_arrays['family_targets'] must have shape [N, 8]")
    if priority_targets.shape != (row_count,):
        raise ValueError("feature_arrays['priority_target'] must have shape [N]")
    if cost_targets.shape != (row_count,):
        raise ValueError("feature_arrays['cost_target'] must have shape [N]")

    interval_scores = np.zeros((row_count, 8), dtype=np.float32)
    family_scores = np.zeros((row_count, 8), dtype=np.float32)
    priority_scores = np.zeros((row_count,), dtype=np.float32)
    cost_scores = np.zeros((row_count,), dtype=np.float32)
    uncertainty_scores = np.zeros((row_count,), dtype=np.float32)

    if ood_abs_feature_threshold is None:
        ood_mask = np.zeros((row_count,), dtype=np.bool_)
    else:
        finite_mask = np.all(np.isfinite(features), axis=1)
        bounded_mask = np.all(np.abs(features) <= float(ood_abs_feature_threshold), axis=1)
        ood_mask = ~(finite_mask & bounded_mask)

    if np.any(ood_mask):
        interval_scores[ood_mask, :] = _normalize_score_rows(
            interval_targets[ood_mask, :],
            fallback_indices=(0,),
        )
        family_scores[ood_mask, :] = _normalize_score_rows(
            family_targets[ood_mask, :],
            fallback_indices=(0, 1),
        )
        priority_scores[ood_mask] = np.maximum(priority_targets[ood_mask], 0.0)
        cost_scores[ood_mask] = np.maximum(cost_targets[ood_mask], 0.0)
        uncertainty_scores[ood_mask] = 1.0

    inference_indices = np.flatnonzero(~ood_mask)
    if inference_indices.size > 0:
        input_name = runtime.session.get_inputs()[0].name
        output_names = [output.name for output in runtime.session.get_outputs()]
        for start in range(0, int(inference_indices.size), batch_size):
            batch_indices = inference_indices[start : start + batch_size]
            batch_features = np.ascontiguousarray(features[batch_indices, :], dtype=np.float32)
            outputs = runtime.session.run(output_names, {input_name: batch_features})
            interval_logits, family_logits, priority_score, cost_score, uncertainty_score = outputs
            batch_interval = np.asarray(interval_logits, dtype=np.float32)
            batch_interval -= np.max(batch_interval, axis=1, keepdims=True)
            np.exp(batch_interval, out=batch_interval)
            batch_interval /= np.maximum(batch_interval.sum(axis=1, keepdims=True), 1.0e-12)
            batch_family = _sigmoid_stable(family_logits)
            interval_scores[batch_indices, :] = batch_interval
            family_scores[batch_indices, :] = batch_family
            priority_scores[batch_indices] = np.asarray(priority_score, dtype=np.float32).reshape(-1)
            cost_scores[batch_indices] = np.asarray(cost_score, dtype=np.float32).reshape(-1)
            uncertainty_scores[batch_indices] = np.asarray(
                uncertainty_score, dtype=np.float32
            ).reshape(-1)

    return {
        "interval_scores": interval_scores,
        "family_scores": family_scores,
        "priority_score": priority_scores,
        "cost_score": cost_scores,
        "uncertainty_score": uncertainty_scores,
        "ood_mask": ood_mask,
    }
