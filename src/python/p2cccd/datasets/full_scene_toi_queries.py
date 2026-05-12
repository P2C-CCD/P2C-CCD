from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .tight_inclusion_queries import (
    TIGHT_INCLUSION_QUERY_ROWS,
    TightInclusionPrimitiveQuery,
    normalize_query_kind,
    parse_query_lines,
)


FULL_SCENE_NAMES = (
    "armadillo-rollers",
    "cloth-ball",
    "cloth-funnel",
    "n-body-simulation",
    "puffer-ball",
    "rod-twist",
)


@dataclass(frozen=True, slots=True)
class FullSceneTOIFile:
    scene_name: str
    kind: str
    query_csv_path: Path
    label_json_path: Path
    query_count: int
    positive_count: int
    byte_size: int


def infer_full_scene_kind(path: Path) -> str:
    stem = path.stem
    if stem.endswith("vf"):
        return "vertex-face"
    if stem.endswith("ee"):
        return "edge-edge"
    raise ValueError(f"cannot infer full-scene query kind from path: {path}")


def label_path_for_full_scene_query(csv_path: Path) -> Path:
    scene_dir = csv_path.parent.parent
    return scene_dir / "mma_bool" / f"{csv_path.stem}_mma_bool.json"


def _load_bool_labels(path: Path) -> list[bool]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError(f"expected list of booleans in {path}")
    labels: list[bool] = []
    for index, value in enumerate(values):
        if not isinstance(value, bool):
            raise ValueError(f"{path}:{index} expected bool, got {type(value).__name__}")
        labels.append(bool(value))
    return labels


def inspect_full_scene_toi_file(csv_path: Path, *, dataset_root: Path) -> FullSceneTOIFile:
    path = Path(csv_path)
    if path.name.startswith("._"):
        raise ValueError(f"AppleDouble metadata file is not a query CSV: {path}")
    scene_name = path.parent.parent.name
    kind = normalize_query_kind(infer_full_scene_kind(path))
    label_path = label_path_for_full_scene_query(path)
    if not label_path.exists():
        raise FileNotFoundError(f"missing full-scene label json for {path}: {label_path}")
    labels = _load_bool_labels(label_path)
    line_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            line_count += chunk.count(b"\n")
    if line_count % TIGHT_INCLUSION_QUERY_ROWS != 0:
        raise ValueError(f"{path} line_count={line_count} is not divisible by 8")
    query_count = line_count // TIGHT_INCLUSION_QUERY_ROWS
    if query_count != len(labels):
        raise ValueError(f"{path} query_count={query_count} but label_count={len(labels)}")
    return FullSceneTOIFile(
        scene_name=scene_name,
        kind=kind,
        query_csv_path=path,
        label_json_path=label_path,
        query_count=query_count,
        positive_count=sum(1 for item in labels if item),
        byte_size=path.stat().st_size + label_path.stat().st_size,
    )


def discover_full_scene_toi_files(dataset_root: Path) -> tuple[FullSceneTOIFile, ...]:
    root = Path(dataset_root)
    rows: list[FullSceneTOIFile] = []
    for scene_name in FULL_SCENE_NAMES:
        query_dir = root / scene_name / "queries"
        if not query_dir.exists():
            continue
        for csv_path in sorted(query_dir.glob("*.csv")):
            if csv_path.name.startswith("._"):
                continue
            rows.append(inspect_full_scene_toi_file(csv_path, dataset_root=root))
    return tuple(rows)


def iter_full_scene_toi_queries(
    csv_path: Path,
    *,
    dataset_root: Path,
) -> Iterator[TightInclusionPrimitiveQuery]:
    path = Path(csv_path)
    scene_name = path.parent.parent.name
    kind = normalize_query_kind(infer_full_scene_kind(path))
    labels = _load_bool_labels(label_path_for_full_scene_query(path))
    with path.open("r", encoding="utf-8") as handle:
        query_lines: list[str] = []
        query_index = 0
        first_line = 1
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            query_lines.append(stripped)
            if len(query_lines) == TIGHT_INCLUSION_QUERY_ROWS:
                if query_index >= len(labels):
                    raise ValueError(f"{path}:{line_number} has more query blocks than labels")
                truth = 1 if labels[query_index] else 0
                yield parse_query_lines(
                    [f"{item},{truth}" for item in query_lines],
                    case_name=scene_name,
                    kind=kind,
                    csv_path=path,
                    query_index=query_index,
                    first_line_number=first_line,
                )
                query_index += 1
                query_lines = []
                first_line = line_number + 1
        if query_lines:
            raise ValueError(f"{path} has trailing partial query with {len(query_lines)} rows")
    if query_index != len(labels):
        raise ValueError(f"{path} query_count={query_index} but label_count={len(labels)}")


__all__ = [
    "FULL_SCENE_NAMES",
    "FullSceneTOIFile",
    "discover_full_scene_toi_files",
    "infer_full_scene_kind",
    "inspect_full_scene_toi_file",
    "iter_full_scene_toi_queries",
    "label_path_for_full_scene_query",
]
