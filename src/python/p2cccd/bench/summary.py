from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import socket
import subprocess
from typing import Any, Mapping, Sequence

from p2cccd.contracts import BenchmarkRow, BenchmarkRowV2, BenchmarkRunMeta
from p2cccd.serialization import (
    write_benchmark_run_meta_json,
    write_benchmark_v2_csv,
    write_benchmark_v2_jsonl,
)
from p2cccd.validators import validate_benchmark_row_v2, validate_benchmark_run_meta


def benchmark_row_to_dict(row: BenchmarkRow | BenchmarkRowV2) -> dict[str, object]:
    return asdict(row)


def _plain_config_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _plain_config_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _plain_config_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain_config_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.value
    return value


def canonical_config_json(config: Any) -> str:
    return json.dumps(
        _plain_config_value(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def benchmark_config_hash(config: Any) -> str:
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def _run_command(args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _detect_nvidia_smi() -> dict[str, object]:
    output = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        )
    )
    if not output:
        return {}
    first_line = output.splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 4:
        return {}
    try:
        total_mb = int(float(parts[2]))
        free_mb = int(float(parts[3]))
    except ValueError:
        total_mb = 0
        free_mb = 0
    return {
        "gpu_name": parts[0] or "unknown",
        "driver_version": parts[1] or "unknown",
        "vram_total_mb": total_mb,
        "vram_free_mb": free_mb,
    }


def _detect_cuda_version() -> str:
    output = _run_command(("nvcc", "--version"))
    for line in output.splitlines():
        if "release" in line:
            release = line.split("release", 1)[1].split(",", 1)[0].strip()
            if release:
                return release
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path:
        return Path(cuda_path).name.replace("v", "") or "unknown"
    return "unknown"


def _detect_git_commit() -> str:
    output = _run_command(("git", "rev-parse", "--short=12", "HEAD"))
    return output.splitlines()[0] if output else "unknown"


def collect_benchmark_environment() -> dict[str, object]:
    nvidia = _detect_nvidia_smi()
    return {
        "git_commit": _detect_git_commit(),
        "host_name": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "gpu_name": str(nvidia.get("gpu_name", "unknown")),
        "driver_version": str(nvidia.get("driver_version", "unknown")),
        "cuda_version": _detect_cuda_version(),
        "optix_version": os.environ.get("OPTIX_VERSION", "unknown"),
        "vram_total_mb": int(nvidia.get("vram_total_mb", 0)),
        "vram_free_mb": int(nvidia.get("vram_free_mb", 0)),
    }


def create_benchmark_run_meta(
    *,
    dataset_name: str,
    scene_name: str,
    method_name: str,
    config: Any,
    seed: int = 0,
    run_id: str | None = None,
    row_count: int = 0,
    environment: Mapping[str, object] | None = None,
    notes: str = "",
    output_csv: str = "benchmark.csv",
    output_jsonl: str = "benchmark.jsonl",
    output_run_meta_json: str = "run_meta.json",
) -> BenchmarkRunMeta:
    config_json = canonical_config_json(config)
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    created_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    env = collect_benchmark_environment() if environment is None else dict(environment)
    final_run_id = run_id or f"{created_utc.replace(':', '').replace('-', '')}_{config_hash[:12]}"
    meta = BenchmarkRunMeta(
        run_id=final_run_id,
        created_utc=created_utc,
        dataset_name=dataset_name,
        scene_name=scene_name,
        method_name=method_name,
        config_hash=config_hash,
        config_json=config_json,
        seed=seed,
        row_count=row_count,
        git_commit=str(env.get("git_commit", "unknown")),
        host_name=str(env.get("host_name", socket.gethostname())),
        platform=str(env.get("platform", platform.platform())),
        python_version=str(env.get("python_version", platform.python_version())),
        gpu_name=str(env.get("gpu_name", "unknown")),
        driver_version=str(env.get("driver_version", "unknown")),
        cuda_version=str(env.get("cuda_version", "unknown")),
        optix_version=str(env.get("optix_version", "unknown")),
        vram_total_mb=int(env.get("vram_total_mb", 0)),
        vram_free_mb=int(env.get("vram_free_mb", 0)),
        output_csv=output_csv,
        output_jsonl=output_jsonl,
        output_run_meta_json=output_run_meta_json,
        notes=notes,
    )
    return validate_benchmark_run_meta(meta)


def latency_percentiles(
    latency_samples_ms: Sequence[float] | None,
    *,
    fallback_total_ms: float,
    fallback_query_count: int,
) -> dict[str, float]:
    samples = [float(value) for value in latency_samples_ms or [] if float(value) >= 0.0]
    if not samples and fallback_query_count > 0:
        samples = [float(fallback_total_ms) / float(fallback_query_count)]
    if not samples:
        samples = [0.0]
    samples.sort()

    def percentile(p: float) -> float:
        if len(samples) == 1:
            return samples[0]
        position = (len(samples) - 1) * p
        lower = int(position)
        upper = min(len(samples) - 1, lower + 1)
        weight = position - lower
        return samples[lower] * (1.0 - weight) + samples[upper] * weight

    return {
        "min": samples[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": samples[-1],
    }


def _family_exact_calls(
    family_exact_calls: Mapping[str, int] | None,
    *,
    fallback_total: int,
) -> dict[str, int]:
    calls = {str(key).lower(): int(value) for key, value in (family_exact_calls or {}).items()}
    point_triangle = max(0, calls.get("point_triangle", calls.get("pt", 0)))
    edge_edge = max(0, calls.get("edge_edge", calls.get("ee", 0)))
    conservative = max(0, calls.get("conservative", calls.get("swept_sphere_proxy", 0)))
    explicit_unknown = max(0, calls.get("unknown", 0))
    known = point_triangle + edge_edge + conservative + explicit_unknown
    unknown = explicit_unknown + max(0, fallback_total - known)
    total = point_triangle + edge_edge + conservative + unknown
    return {
        "point_triangle": point_triangle,
        "edge_edge": edge_edge,
        "conservative": conservative,
        "unknown": unknown,
        "total": total,
    }


def benchmark_row_v2_from_legacy(
    row: BenchmarkRow,
    meta: BenchmarkRunMeta,
    *,
    latency_samples_ms: Sequence[float] | None = None,
    family_exact_calls: Mapping[str, int] | None = None,
    candidate_inflation_ratio: float | None = None,
    undecided_to_resolved_ratio: float = 0.0,
    exact_queue_occupancy: float | None = None,
    rt_build_ms: float = 0.0,
    rt_update_ms: float = 0.0,
    rt_trace_ms: float | None = None,
    candidate_buffer_bandwidth_mb_s: float = 0.0,
    proposal_enqueue_dequeue_ms: float = 0.0,
    total_tail_latency_ms: float | None = None,
    vram_peak_mb: int = 0,
) -> BenchmarkRowV2:
    percentiles = latency_percentiles(
        latency_samples_ms,
        fallback_total_ms=row.total_ms,
        fallback_query_count=row.query_count,
    )
    fallback_exact_total = int(round(row.avg_exact_evals * row.query_count))
    family_calls = _family_exact_calls(family_exact_calls, fallback_total=fallback_exact_total)
    trace_ms = row.rt_ms if rt_trace_ms is None else rt_trace_ms
    candidate_total = max(0.0, float(row.avg_candidates) * float(row.query_count))
    inferred_inflation = (
        candidate_total / max(1.0, float(fallback_exact_total))
        if candidate_inflation_ratio is None
        else float(candidate_inflation_ratio)
    )
    inferred_occupancy = (
        min(1.0, float(fallback_exact_total) / max(1.0, candidate_total))
        if exact_queue_occupancy is None
        else float(exact_queue_occupancy)
    )
    row_v2 = BenchmarkRowV2(
        run_id=meta.run_id,
        dataset_name=meta.dataset_name,
        scene_name=meta.scene_name,
        method_name=meta.method_name,
        config_hash=meta.config_hash,
        seed=meta.seed,
        query_count=row.query_count,
        fn_count=row.fn_count,
        fp_count=row.fp_count,
        candidate_recall=row.candidate_recall,
        avg_candidates=row.avg_candidates,
        avg_exact_evals=row.avg_exact_evals,
        avg_subdivision_depth=row.avg_subdivision_depth,
        fallback_ratio=row.fallback_ratio,
        candidate_inflation_ratio=inferred_inflation,
        undecided_to_resolved_ratio=float(undecided_to_resolved_ratio),
        exact_queue_occupancy=inferred_occupancy,
        rt_build_ms=rt_build_ms,
        rt_update_ms=rt_update_ms,
        rt_trace_ms=trace_ms,
        rt_ms=rt_build_ms + rt_update_ms + trace_ms,
        proposal_ms=row.proposal_ms,
        exact_ms=row.exact_ms,
        total_ms=row.total_ms,
        latency_min_ms=percentiles["min"],
        latency_p50_ms=percentiles["p50"],
        latency_p90_ms=percentiles["p90"],
        latency_p95_ms=percentiles["p95"],
        latency_p99_ms=percentiles["p99"],
        latency_max_ms=percentiles["max"],
        qps=row.qps,
        family_point_triangle_exact_calls=family_calls["point_triangle"],
        family_edge_edge_exact_calls=family_calls["edge_edge"],
        family_conservative_exact_calls=family_calls["conservative"],
        family_unknown_exact_calls=family_calls["unknown"],
        exact_calls_total=family_calls["total"],
        candidate_buffer_bandwidth_mb_s=float(candidate_buffer_bandwidth_mb_s),
        proposal_enqueue_dequeue_ms=float(proposal_enqueue_dequeue_ms),
        total_tail_latency_ms=percentiles["p99"] if total_tail_latency_ms is None else float(total_tail_latency_ms),
        vram_peak_mb=vram_peak_mb,
    )
    return validate_benchmark_row_v2(row_v2)


class BenchmarkExportPaths:
    def __init__(self, run_dir: Path, csv_path: Path, jsonl_path: Path, run_meta_path: Path) -> None:
        self.run_dir = run_dir
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.run_meta_path = run_meta_path


def export_benchmark_run(
    run_dir: str | Path,
    meta: BenchmarkRunMeta,
    rows: Sequence[BenchmarkRowV2],
) -> BenchmarkExportPaths:
    if not rows:
        raise ValueError("cannot export a benchmark run without BenchmarkRowV2 rows")
    output_dir = Path(run_dir)
    csv_path = output_dir / meta.output_csv
    jsonl_path = output_dir / meta.output_jsonl
    run_meta_path = output_dir / meta.output_run_meta_json
    checked_rows = tuple(validate_benchmark_row_v2(row) for row in rows)
    for row in checked_rows:
        if row.run_id != meta.run_id:
            raise ValueError("BenchmarkRowV2.run_id must match BenchmarkRunMeta.run_id")
        if row.config_hash != meta.config_hash:
            raise ValueError("BenchmarkRowV2.config_hash must match BenchmarkRunMeta.config_hash")
    final_meta = replace(meta, row_count=len(checked_rows))
    validate_benchmark_run_meta(final_meta)
    write_benchmark_v2_csv(csv_path, checked_rows)
    write_benchmark_v2_jsonl(jsonl_path, checked_rows)
    write_benchmark_run_meta_json(run_meta_path, final_meta)
    return BenchmarkExportPaths(output_dir, csv_path, jsonl_path, run_meta_path)
