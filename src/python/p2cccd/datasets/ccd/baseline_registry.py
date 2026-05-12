from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BaselineSourceSpec:
    name: str
    directory_name: str
    priority: str
    role: str
    official_url: str
    required_for_first_layer: bool
    expected_markers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BaselineSourceStatus:
    spec: BaselineSourceSpec
    path: Path
    available: bool
    missing_markers: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.spec.name

    def to_manifest_entry(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "directory_name": self.spec.directory_name,
            "priority": self.spec.priority,
            "role": self.spec.role,
            "official_url": self.spec.official_url,
            "required_for_first_layer": self.spec.required_for_first_layer,
            "available": self.available,
            "path": str(self.path),
            "missing_markers": list(self.missing_markers),
        }


FIRST_LAYER_SOURCES: tuple[BaselineSourceSpec, ...] = (
    BaselineSourceSpec(
        name="Tight Inclusion",
        directory_name="Tight-Inclusion",
        priority="P0",
        role="Conservative CCD reference implementation and correctness comparison source.",
        official_url="https://continuous-collision-detection.github.io/tight_inclusion/",
        required_for_first_layer=True,
        expected_markers=("README.md", "LICENSE", "src/tight_inclusion/ccd.hpp"),
    ),
    BaselineSourceSpec(
        name="CCD-Wrapper",
        directory_name="CCD-Wrapper",
        priority="P0",
        role="External CCD benchmark harness and method organization reference.",
        official_url="https://github.com/Continuous-Collision-Detection/CCD-Wrapper",
        required_for_first_layer=True,
        expected_markers=("README.md", "LICENSE", "src/ccd.hpp", "src/benchmark.cpp"),
    ),
    BaselineSourceSpec(
        name="Scalable CCD",
        directory_name="Scalable-CCD",
        priority="P0",
        role="Scalable broad/narrow phase CCD algorithm and benchmark reference.",
        official_url="https://continuous-collision-detection.github.io/scalable_ccd/",
        required_for_first_layer=True,
        expected_markers=("README.md", "LICENSE", "src/scalable_ccd"),
    ),
    BaselineSourceSpec(
        name="Sample Scalable CCD Data",
        directory_name="Sample-Scalable-CCD-Data",
        priority="P0",
        role="Ground-truth sample CCD query data used as the first runnable external correctness input.",
        official_url="https://github.com/Continuous-Collision-Detection/Sample-Scalable-CCD-Data",
        required_for_first_layer=True,
        expected_markers=("README.md", "LICENSE", "cloth-funnel/queries/227vf.csv"),
    ),
    BaselineSourceSpec(
        name="Exact Root Parity CCD",
        directory_name="Exact-Root-Parity-CCD",
        priority="P1",
        role="Exact narrow-phase reference for deeper exact CCD cross-checks.",
        official_url="https://continuous-collision-detection.github.io/root_parity/",
        required_for_first_layer=False,
        expected_markers=("README.md", "LICENSE", "src/ccd.hpp"),
    ),
    BaselineSourceSpec(
        name="Rigid IPC scenes",
        directory_name="rigid-ipc",
        priority="P1",
        role="Complex rigid-body scene correctness extension once scenes are downloaded.",
        official_url="https://ipc-sim.github.io/rigid-ipc/",
        required_for_first_layer=False,
        expected_markers=("README.md", "LICENSE", "fixtures", "meshes"),
    ),
)


def default_baseline_root() -> Path:
    return Path(__file__).resolve().parents[4] / "baseline"


def _missing_markers(source_path: Path, markers: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(marker for marker in markers if not (source_path / marker).exists())


def discover_first_layer_sources(baseline_root: str | Path | None = None) -> tuple[BaselineSourceStatus, ...]:
    root = Path(baseline_root) if baseline_root is not None else default_baseline_root()
    statuses: list[BaselineSourceStatus] = []
    for spec in FIRST_LAYER_SOURCES:
        path = root / spec.directory_name
        missing = _missing_markers(path, spec.expected_markers)
        statuses.append(
            BaselineSourceStatus(
                spec=spec,
                path=path,
                available=path.exists() and len(missing) == 0,
                missing_markers=missing,
            )
        )
    return tuple(statuses)


def build_first_layer_manifest(baseline_root: str | Path | None = None) -> dict[str, Any]:
    statuses = discover_first_layer_sources(baseline_root)
    required_statuses = [status for status in statuses if status.spec.required_for_first_layer]
    return {
        "schema_version": 1,
        "layer": "2.1 correctness_ccd",
        "baseline_root": str(Path(baseline_root) if baseline_root is not None else default_baseline_root()),
        "required_available": all(status.available for status in required_statuses),
        "sources": [status.to_manifest_entry() for status in statuses],
    }
