from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import CONTRACT_SCHEMA_VERSION, MAX_FAMILY_SCORES, MAX_INTERVAL_SCORES


@dataclass(frozen=True, slots=True)
class EpsilonConfig:
    eps_time: float
    eps_space: float
    eps_proxy: float
    eps_cert: float


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    max_interval_bins: int
    max_family_scores: int
    enable_audit_log: bool


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    write_csv: bool
    write_jsonl: bool


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    schema_version: int
    epsilon: EpsilonConfig
    runtime: RuntimeLimits
    benchmark: BenchmarkConfig


def _require_mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be an object")
    return data


def _require_keys(data: dict[str, Any], name: str, keys: set[str]) -> None:
    missing = sorted(keys - set(data.keys()))
    extra = sorted(set(data.keys()) - keys)
    if missing:
        raise ValueError(f"{name} missing required field(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"{name} has unknown field(s): {', '.join(extra)}")


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _bounded_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def runtime_config_from_dict(data: dict[str, Any]) -> RuntimeConfig:
    root = _require_mapping(data, "runtime config")
    _require_keys(root, "runtime config", {"schema_version", "epsilon", "runtime", "benchmark"})
    if root["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise ValueError("runtime config schema_version is unsupported")

    epsilon_data = _require_mapping(root["epsilon"], "epsilon")
    _require_keys(epsilon_data, "epsilon", {"eps_time", "eps_space", "eps_proxy", "eps_cert"})
    epsilon = EpsilonConfig(
        eps_time=_positive_float(epsilon_data["eps_time"], "epsilon.eps_time"),
        eps_space=_positive_float(epsilon_data["eps_space"], "epsilon.eps_space"),
        eps_proxy=_positive_float(epsilon_data["eps_proxy"], "epsilon.eps_proxy"),
        eps_cert=_positive_float(epsilon_data["eps_cert"], "epsilon.eps_cert"),
    )

    runtime_data = _require_mapping(root["runtime"], "runtime")
    _require_keys(
        runtime_data,
        "runtime",
        {"max_interval_bins", "max_family_scores", "enable_audit_log"},
    )
    runtime = RuntimeLimits(
        max_interval_bins=_bounded_int(
            runtime_data["max_interval_bins"],
            "runtime.max_interval_bins",
            minimum=1,
            maximum=MAX_INTERVAL_SCORES,
        ),
        max_family_scores=_bounded_int(
            runtime_data["max_family_scores"],
            "runtime.max_family_scores",
            minimum=1,
            maximum=MAX_FAMILY_SCORES,
        ),
        enable_audit_log=_bool(runtime_data["enable_audit_log"], "runtime.enable_audit_log"),
    )

    benchmark_data = _require_mapping(root["benchmark"], "benchmark")
    _require_keys(benchmark_data, "benchmark", {"write_csv", "write_jsonl"})
    benchmark = BenchmarkConfig(
        write_csv=_bool(benchmark_data["write_csv"], "benchmark.write_csv"),
        write_jsonl=_bool(benchmark_data["write_jsonl"], "benchmark.write_jsonl"),
    )

    return RuntimeConfig(
        schema_version=CONTRACT_SCHEMA_VERSION,
        epsilon=epsilon,
        runtime=runtime,
        benchmark=benchmark,
    )


def load_runtime_config_dict(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    return runtime_config_from_dict(load_runtime_config_dict(path))


def validate_runtime_config(config: RuntimeConfig | dict[str, Any]) -> RuntimeConfig:
    if isinstance(config, RuntimeConfig):
        return runtime_config_from_dict(
            {
                "schema_version": config.schema_version,
                "epsilon": {
                    "eps_time": config.epsilon.eps_time,
                    "eps_space": config.epsilon.eps_space,
                    "eps_proxy": config.epsilon.eps_proxy,
                    "eps_cert": config.epsilon.eps_cert,
                },
                "runtime": {
                    "max_interval_bins": config.runtime.max_interval_bins,
                    "max_family_scores": config.runtime.max_family_scores,
                    "enable_audit_log": config.runtime.enable_audit_log,
                },
                "benchmark": {
                    "write_csv": config.benchmark.write_csv,
                    "write_jsonl": config.benchmark.write_jsonl,
                },
            }
        )
    return runtime_config_from_dict(config)
