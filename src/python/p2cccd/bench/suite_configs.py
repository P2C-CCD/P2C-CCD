from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .suite_runner import BenchmarkSuiteConfig, load_benchmark_suite_config


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteConfigInfo:
    name: str
    path: Path
    suite_name: str
    suite_type: str
    case_count: int


def bundled_suite_config_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "benchmark_suites"


def discover_bundled_suite_configs(config_dir: str | Path | None = None) -> tuple[BenchmarkSuiteConfigInfo, ...]:
    root = Path(config_dir) if config_dir is not None else bundled_suite_config_dir()
    infos: list[BenchmarkSuiteConfigInfo] = []
    for path in sorted(root.glob("*.json")):
        config = load_benchmark_suite_config(path)
        infos.append(
            BenchmarkSuiteConfigInfo(
                name=path.name,
                path=path,
                suite_name=config.suite_name,
                suite_type=config.suite_type,
                case_count=len(config.cases),
            )
        )
    return tuple(infos)


def load_bundled_suite_config(name: str, config_dir: str | Path | None = None) -> BenchmarkSuiteConfig:
    root = Path(config_dir) if config_dir is not None else bundled_suite_config_dir()
    candidate = root / name
    if candidate.suffix != ".json":
        candidate = candidate.with_suffix(".json")
    return load_benchmark_suite_config(candidate)
