from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from .tight_inclusion_queries import (
    TIGHT_INCLUSION_QUERY_ROWS,
    TightInclusionPrimitiveQuery,
    normalize_query_kind,
    parse_rational_vertex_row,
)


FullSceneQueryKind = Literal["vertex-face", "edge-edge"]


@dataclass(frozen=True, slots=True)
class FullSceneCCDQueryFile:
    scene_name: str
    kind: FullSceneQueryKind
    step_id: int
    csv_path: Path
    mma_bool_path: Path
    box_path: Path | None
    query_count: int
    positive_count: int
    byte_size: int


def _kind_from_stem(stem: str) -> FullSceneQueryKind:
    if stem.endswith("vf"):
        return "vertex-face"
    if stem.endswith("ee"):
        return "edge-edge"
    raise ValueError(f"cannot infer full-scene query kind from stem: {stem}")


def _step_from_stem(stem: str) -> int:
    suffix_len = 2 if stem.endswith(("vf", "ee")) else 0
    token = stem[:-suffix_len] if suffix_len else stem
    if not token.isdigit():
        raise ValueError(f"cannot infer full-scene step id from stem: {stem}")
    return int(token)


def _suffix_for_kind(kind: FullSceneQueryKind) -> str:
    return "vf" if kind == "vertex-face" else "ee"


def _load_bool_labels(path: Path) -> list[bool]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected boolean list in {path}")
    labels: list[bool] = []
    for index, value in enumerate(data):
        if not isinstance(value, bool):
            raise ValueError(f"{path}:{index} expected bool, got {type(value).__name__}")
        labels.append(bool(value))
    return labels


def discover_full_scene_query_files(dataset_root: Path) -> tuple[FullSceneCCDQueryFile, ...]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(root)
    found: list[FullSceneCCDQueryFile] = []
    for scene_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        query_dir = scene_dir / "queries"
        bool_dir = scene_dir / "mma_bool"
        box_dir = scene_dir / "boxes"
        if not query_dir.exists() or not bool_dir.exists():
            continue
        for csv_path in sorted(query_dir.glob("*.csv")):
            if csv_path.name.startswith("._"):
                continue
            stem = csv_path.stem
            kind = _kind_from_stem(stem)
            suffix = _suffix_for_kind(kind)
            step_id = _step_from_stem(stem)
            bool_path = bool_dir / f"{step_id}{suffix}_mma_bool.json"
            if not bool_path.exists():
                raise FileNotFoundError(f"missing full-scene mma_bool file for {csv_path}: {bool_path}")
            box_path = box_dir / f"{step_id}{suffix}.json"
            labels = _load_bool_labels(bool_path)
            found.append(
                FullSceneCCDQueryFile(
                    scene_name=scene_dir.name,
                    kind=kind,
                    step_id=step_id,
                    csv_path=csv_path,
                    mma_bool_path=bool_path,
                    box_path=box_path if box_path.exists() else None,
                    query_count=len(labels),
                    positive_count=sum(1 for value in labels if value),
                    byte_size=csv_path.stat().st_size,
                )
            )
    return tuple(found)


def iter_full_scene_queries(
    query_file: FullSceneCCDQueryFile,
    *,
    dataset_root: Path | None = None,
) -> Iterator[TightInclusionPrimitiveQuery]:
    labels = _load_bool_labels(query_file.mma_bool_path)
    query_lines: list[str] = []
    query_index = 0
    with query_file.csv_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            query_lines.append(line)
            if len(query_lines) == TIGHT_INCLUSION_QUERY_ROWS:
                if query_index >= len(labels):
                    raise ValueError(f"{query_file.csv_path} has more CSV queries than mma_bool labels")
                yield _parse_full_scene_query_lines(
                    query_lines,
                    query_file=query_file,
                    query_index=query_index,
                    ground_truth=labels[query_index],
                    first_line_number=line_number - TIGHT_INCLUSION_QUERY_ROWS + 1,
                    dataset_root=dataset_root,
                )
                query_lines = []
                query_index += 1
    if query_lines:
        raise ValueError(f"{query_file.csv_path} has trailing rows not divisible by 8")
    if query_index != len(labels):
        raise ValueError(
            f"{query_file.csv_path} query_count={query_index} does not match mma_bool labels={len(labels)}"
        )


def _parse_full_scene_query_lines(
    lines: list[str],
    *,
    query_file: FullSceneCCDQueryFile,
    query_index: int,
    ground_truth: bool,
    first_line_number: int,
    dataset_root: Path | None,
) -> TightInclusionPrimitiveQuery:
    vertices: list[np.ndarray] = []
    numerators: list[np.ndarray] = []
    denominators: list[np.ndarray] = []
    for offset, line in enumerate(lines):
        # Full-scene rows contain six rational columns; append the external
        # label so we can reuse the Tight-Inclusion rational parser.
        row_text = line.strip()
        truth = "1" if ground_truth else "0"
        vertex, nums, dens, _ = parse_rational_vertex_row(
            f"{row_text},{truth}",
            line_number=first_line_number + offset,
        )
        vertices.append(vertex)
        numerators.append(nums)
        denominators.append(dens)
    root = Path(dataset_root).resolve() if dataset_root is not None else None
    csv_path = query_file.csv_path
    if root is not None:
        try:
            csv_path = csv_path.resolve().relative_to(root)
        except ValueError:
            pass
    return TightInclusionPrimitiveQuery(
        case_name=query_file.scene_name,
        kind=normalize_query_kind(query_file.kind),
        csv_path=csv_path,
        query_index=query_file.step_id * 10_000_000 + query_index,
        vertices_t0_t1=np.vstack(vertices).astype(np.float64, copy=False),
        ground_truth=bool(ground_truth),
        numerators=np.vstack(numerators).astype(object, copy=False),
        denominators=np.vstack(denominators).astype(object, copy=False),
    )


__all__ = [
    "FullSceneCCDQueryFile",
    "discover_full_scene_query_files",
    "iter_full_scene_queries",
]
