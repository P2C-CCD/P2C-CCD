from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from p2cccd.datasets.ccd.contracts import SourceLicense

from .contracts import CAD_ADAPTER_SCHEMA_VERSION, StepNativeAsset, StepPreprocessRecord
from .mesh_io import stable_asset_id


BETTER_STEP_SOURCE_NAME = "Better STEP"
STEP_EXTENSIONS = (".step", ".stp")

_PRODUCT_RE = re.compile(r"PRODUCT\s*\(\s*'([^']*)'", re.IGNORECASE)
_FILE_SCHEMA_RE = re.compile(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']*)'", re.IGNORECASE)
_UNIT_RE = re.compile(r"(SI_UNIT|NAMED_UNIT|CONVERSION_BASED_UNIT)\s*\(([^;]*)", re.IGNORECASE)


def default_better_step_root() -> Path:
    return Path(__file__).resolve().parents[5] / "datasets" / "better_step"


def is_step_path(path: Path) -> bool:
    return path.suffix.lower() in STEP_EXTENSIONS


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _sidecar_metadata(path: Path) -> dict[str, Any]:
    for candidate in (
        path.with_suffix(".json"),
        path.parent / f"{path.stem}.metadata.json",
        path.parent / "metadata" / f"{path.stem}.json",
    ):
        data = _load_json_if_exists(candidate)
        if data:
            return {**data, "metadata_path": str(candidate)}
    return {}


def _dedup(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = value.strip()
        if clean and clean not in seen:
            result.append(clean)
            seen.add(clean)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class BetterSTEPAdapter:
    root: Path | None = None

    @property
    def source_root(self) -> Path:
        return self.root if self.root is not None else default_better_step_root()

    def require_available(self) -> None:
        self.license().require_available()
        if not self.source_root.exists():
            raise FileNotFoundError(f"Better STEP root not found: {self.source_root}")

    def license(self) -> SourceLicense:
        return SourceLicense(
            name=BETTER_STEP_SOURCE_NAME,
            license_path=self.source_root / "LICENSE",
            url="https://github.com/saali14/better-step",
            terms="CAD-native STEP preprocessing bridge; preserve upstream tool and dataset terms before meshing.",
        )

    def list_step_paths(self, *, limit: int | None = None) -> tuple[Path, ...]:
        self.require_available()
        paths: list[Path] = []
        for path in sorted(self.source_root.rglob("*")):
            if path.is_file() and is_step_path(path):
                paths.append(path)
                if limit is not None and len(paths) >= limit:
                    break
        return tuple(paths)

    def load_asset(self, path: str | Path) -> StepNativeAsset:
        self.require_available()
        step_path = Path(path)
        if not step_path.is_absolute():
            step_path = self.source_root / step_path
        if not step_path.exists():
            raise FileNotFoundError(f"Better STEP asset not found: {step_path}")
        if not is_step_path(step_path):
            raise ValueError(f"unsupported STEP asset extension: {step_path.suffix}")

        product_names: list[str] = []
        schema_names: list[str] = []
        unit_names: list[str] = []
        entity_count = 0
        with step_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line.startswith("#"):
                    entity_count += 1
                product_names.extend(match.group(1) for match in _PRODUCT_RE.finditer(line))
                schema_names.extend(match.group(1) for match in _FILE_SCHEMA_RE.finditer(line))
                unit_names.extend(match.group(1).upper() for match in _UNIT_RE.finditer(line))

        sidecar = _sidecar_metadata(step_path)
        return StepNativeAsset(
            schema_version=CAD_ADAPTER_SCHEMA_VERSION,
            source_name=BETTER_STEP_SOURCE_NAME,
            asset_id=stable_asset_id(step_path, self.source_root),
            step_path=step_path,
            file_size_bytes=step_path.stat().st_size,
            entity_count=entity_count,
            product_names=_dedup(product_names),
            schema_names=_dedup(schema_names),
            unit_names=_dedup(unit_names),
            sidecar_metadata=sidecar,
            metadata={
                "source_relative_path": step_path.relative_to(self.source_root).as_posix(),
                "cad_native_format": step_path.suffix.lower().lstrip("."),
            },
        )

    def list_assets(self, *, limit: int | None = None) -> tuple[StepNativeAsset, ...]:
        return tuple(self.load_asset(path) for path in self.list_step_paths(limit=limit))

    def generate_preprocess_records(
        self,
        *,
        assets: tuple[StepNativeAsset, ...] | None = None,
        target_mesh_format: str = "obj",
        limit: int | None = None,
    ) -> tuple[StepPreprocessRecord, ...]:
        asset_list = list(assets) if assets is not None else list(self.list_assets(limit=limit))
        records: list[StepPreprocessRecord] = []
        for asset in asset_list:
            output_stem = asset.sidecar_metadata.get("output_stem")
            if not isinstance(output_stem, str) or not output_stem:
                output_stem = asset.step_path.with_suffix("").name
            records.append(
                StepPreprocessRecord(
                    schema_version=CAD_ADAPTER_SCHEMA_VERSION,
                    source_name=BETTER_STEP_SOURCE_NAME,
                    record_id=f"{asset.asset_id}:{target_mesh_format}",
                    asset=asset,
                    output_stem=output_stem,
                    target_mesh_format=target_mesh_format,
                    metadata={
                        "preprocess_stage": "cad_native_to_mesh_bridge",
                        "requires_meshing_backend": True,
                        "entity_count": asset.entity_count,
                    },
                )
            )
            if limit is not None and len(records) >= limit:
                break
        return tuple(records)
