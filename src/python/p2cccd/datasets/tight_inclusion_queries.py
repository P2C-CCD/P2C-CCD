from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import itertools
import math
import random
from typing import Iterable, Iterator, Literal, Sequence

import numpy as np


TightInclusionQueryKind = Literal["vertex-face", "edge-edge"]
DEFAULT_FILE_SPLIT_SEED = 424242
TIGHT_INCLUSION_QUERY_ROWS = 8
TIGHT_INCLUSION_CSV_COLUMNS = 7


@dataclass(frozen=True, slots=True)
class TightInclusionCSVFile:
    case_name: str
    kind: TightInclusionQueryKind
    csv_path: Path
    line_count: int
    query_count: int
    byte_size: int


@dataclass(frozen=True, slots=True)
class TightInclusionPrimitiveQuery:
    case_name: str
    kind: TightInclusionQueryKind
    csv_path: Path
    query_index: int
    vertices_t0_t1: np.ndarray
    ground_truth: bool
    numerators: np.ndarray
    denominators: np.ndarray

    @property
    def vertices_t0(self) -> np.ndarray:
        return self.vertices_t0_t1[:4]

    @property
    def vertices_t1(self) -> np.ndarray:
        return self.vertices_t0_t1[4:]

    @property
    def rational_magnitude_features(self) -> np.ndarray:
        nums = np.asarray([_log1p_abs_integer(value) for value in self.numerators.flat], dtype=np.float64)
        dens = np.asarray([_log1p_abs_integer(value) for value in self.denominators.flat], dtype=np.float64)
        return np.asarray(
            [
                float(nums.max(initial=0.0)),
                float(dens.max(initial=0.0)),
                float(nums.mean() if nums.size else 0.0),
                float(dens.mean() if dens.size else 0.0),
            ],
            dtype=np.float64,
        )


def default_tight_inclusion_dataset_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return project_root / "baseline" / "datasets" / "continuous-collision-detection"
    return Path("src/baseline/datasets/continuous-collision-detection")


def normalize_query_kind(kind: str) -> TightInclusionQueryKind:
    if kind in {"vertex-face", "vf"}:
        return "vertex-face"
    if kind in {"edge-edge", "ee"}:
        return "edge-edge"
    raise ValueError(f"unsupported Tight-Inclusion query kind: {kind}")


def infer_query_kind(csv_path: Path) -> TightInclusionQueryKind:
    parts = set(csv_path.parts)
    if "vertex-face" in parts:
        return "vertex-face"
    if "edge-edge" in parts:
        return "edge-edge"
    name = csv_path.name
    if name.startswith("vertex-face"):
        return "vertex-face"
    if name.startswith("edge-edge"):
        return "edge-edge"
    raise ValueError(f"cannot infer Tight-Inclusion query kind from path: {csv_path}")


def infer_case_name(csv_path: Path, dataset_root: Path | None = None) -> str:
    path = csv_path.resolve()
    root = dataset_root.resolve() if dataset_root is not None else None
    if root is not None:
        try:
            rel = path.relative_to(root)
            return rel.parts[0]
        except ValueError:
            pass
    for parent in csv_path.parents:
        if parent.name in {"vertex-face", "edge-edge"}:
            return parent.parent.name
    raise ValueError(f"cannot infer Tight-Inclusion case name from path: {csv_path}")


def _log1p_abs_integer(value: object) -> float:
    integer = abs(int(value))
    try:
        return math.log1p(float(integer))
    except OverflowError:
        # Very large exact rationals are rare but valid. Decimal digit count
        # gives a stable log-scale feature without truncating the original int.
        return len(str(integer)) * math.log(10.0)


def parse_rational_vertex_row(line: str, *, line_number: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    text = line.strip()
    if not text:
        raise ValueError("empty Tight-Inclusion CSV row")
    columns = text.split(",")
    if len(columns) != TIGHT_INCLUSION_CSV_COLUMNS:
        location = f" at line {line_number}" if line_number is not None else ""
        raise ValueError(f"expected 7 columns{location}, got {len(columns)}")
    try:
        values = [int(item) for item in columns]
    except ValueError as exc:
        location = f" at line {line_number}" if line_number is not None else ""
        raise ValueError(f"non-integer rational CSV value{location}: {text}") from exc
    denominators = np.asarray([values[1], values[3], values[5]], dtype=object)
    if np.any(denominators == 0):
        location = f" at line {line_number}" if line_number is not None else ""
        raise ValueError(f"zero denominator in Tight-Inclusion CSV row{location}")
    truth_value = values[6]
    if truth_value not in (0, 1):
        location = f" at line {line_number}" if line_number is not None else ""
        raise ValueError(f"ground-truth value must be 0 or 1{location}, got {truth_value}")
    numerators = np.asarray([values[0], values[2], values[4]], dtype=object)
    vertex = numerators.astype(np.float64) / denominators.astype(np.float64)
    return vertex, numerators, denominators, bool(truth_value)


def parse_query_lines(
    lines: Sequence[str],
    *,
    case_name: str,
    kind: TightInclusionQueryKind,
    csv_path: Path,
    query_index: int,
    first_line_number: int = 1,
) -> TightInclusionPrimitiveQuery:
    if len(lines) != TIGHT_INCLUSION_QUERY_ROWS:
        raise ValueError(f"Tight-Inclusion query must contain 8 rows, got {len(lines)}")
    vertices: list[np.ndarray] = []
    numerators: list[np.ndarray] = []
    denominators: list[np.ndarray] = []
    truths: list[bool] = []
    for offset, line in enumerate(lines):
        vertex, nums, dens, truth = parse_rational_vertex_row(line, line_number=first_line_number + offset)
        vertices.append(vertex)
        numerators.append(nums)
        denominators.append(dens)
        truths.append(truth)
    if len(set(truths)) != 1:
        raise ValueError(f"inconsistent truth labels in {csv_path}:{query_index}")
    return TightInclusionPrimitiveQuery(
        case_name=case_name,
        kind=kind,
        csv_path=csv_path,
        query_index=query_index,
        vertices_t0_t1=np.vstack(vertices).astype(np.float64, copy=False),
        ground_truth=truths[0],
        numerators=np.vstack(numerators).astype(object, copy=False),
        denominators=np.vstack(denominators).astype(object, copy=False),
    )


def count_csv_lines(csv_path: Path) -> int:
    line_count = 0
    with csv_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            line_count += chunk.count(b"\n")
    return line_count


def inspect_tight_inclusion_csv(csv_path: Path, *, dataset_root: Path | None = None) -> TightInclusionCSVFile:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    line_count = count_csv_lines(path)
    if line_count % TIGHT_INCLUSION_QUERY_ROWS != 0:
        raise ValueError(f"{path} line_count={line_count} is not divisible by 8")
    return TightInclusionCSVFile(
        case_name=infer_case_name(path, dataset_root),
        kind=infer_query_kind(path),
        csv_path=path,
        line_count=line_count,
        query_count=line_count // TIGHT_INCLUSION_QUERY_ROWS,
        byte_size=path.stat().st_size,
    )


def discover_tight_inclusion_csv_files(
    dataset_root: Path,
    *,
    cases: Iterable[str] | None = None,
    kinds: Iterable[str] = ("vertex-face", "edge-edge"),
    inspect: bool = False,
) -> tuple[TightInclusionCSVFile, ...]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(root)
    case_filter = set(cases) if cases is not None else None
    normalized_kinds = tuple(normalize_query_kind(kind) for kind in kinds)
    found: list[TightInclusionCSVFile] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        if case_filter is not None and case_dir.name not in case_filter:
            continue
        if case_dir.name in {"nyu-full-dataset-archives", "Sample-Queries", "visualization"}:
            continue
        for kind in normalized_kinds:
            kind_dir = case_dir / kind
            if not kind_dir.exists():
                continue
            for csv_path in sorted(kind_dir.glob("*.csv")):
                if inspect:
                    found.append(inspect_tight_inclusion_csv(csv_path, dataset_root=root))
                else:
                    found.append(
                        TightInclusionCSVFile(
                            case_name=case_dir.name,
                            kind=kind,
                            csv_path=csv_path,
                            line_count=0,
                            query_count=0,
                            byte_size=csv_path.stat().st_size,
                        )
                    )
    return tuple(found)


def read_tight_inclusion_query(
    csv_path: Path,
    query_index: int,
    *,
    dataset_root: Path | None = None,
) -> TightInclusionPrimitiveQuery:
    if query_index < 0:
        raise ValueError("query_index must be non-negative")
    path = Path(csv_path)
    first = query_index * TIGHT_INCLUSION_QUERY_ROWS
    with path.open("r", encoding="ascii") as handle:
        lines = list(itertools.islice(handle, first, first + TIGHT_INCLUSION_QUERY_ROWS))
    if len(lines) != TIGHT_INCLUSION_QUERY_ROWS:
        raise ValueError(f"query_index={query_index} is out of range for {path}")
    return parse_query_lines(
        lines,
        case_name=infer_case_name(path, dataset_root),
        kind=infer_query_kind(path),
        csv_path=path,
        query_index=query_index,
        first_line_number=first + 1,
    )


def iter_tight_inclusion_queries(
    csv_path: Path,
    *,
    dataset_root: Path | None = None,
    start_query_index: int = 0,
    limit: int | None = None,
) -> Iterator[TightInclusionPrimitiveQuery]:
    if start_query_index < 0:
        raise ValueError("start_query_index must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    path = Path(csv_path)
    case_name = infer_case_name(path, dataset_root)
    kind = infer_query_kind(path)
    emitted = 0
    buffer: list[str] = []
    query_index = 0
    with path.open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            buffer.append(line)
            if len(buffer) != TIGHT_INCLUSION_QUERY_ROWS:
                continue
            if query_index >= start_query_index:
                if limit is not None and emitted >= limit:
                    return
                yield parse_query_lines(
                    buffer,
                    case_name=case_name,
                    kind=kind,
                    csv_path=path,
                    query_index=query_index,
                    first_line_number=line_number - TIGHT_INCLUSION_QUERY_ROWS + 1,
                )
                emitted += 1
            buffer = []
            query_index += 1
    if buffer:
        raise ValueError(f"{path} has a truncated query block with {len(buffer)} leftover rows")


def iter_dataset_queries(
    dataset_root: Path,
    *,
    cases: Iterable[str] | None = None,
    kinds: Iterable[str] = ("vertex-face", "edge-edge"),
    limit_per_file: int | None = None,
) -> Iterator[TightInclusionPrimitiveQuery]:
    for csv_file in discover_tight_inclusion_csv_files(dataset_root, cases=cases, kinds=kinds, inspect=False):
        yield from iter_tight_inclusion_queries(csv_file.csv_path, dataset_root=dataset_root, limit=limit_per_file)


@dataclass(frozen=True, slots=True)
class TightInclusionFileSplit:
    train: tuple[TightInclusionCSVFile, ...]
    validation: tuple[TightInclusionCSVFile, ...]
    heldout_test: tuple[TightInclusionCSVFile, ...]
    unit_smoke: tuple[TightInclusionCSVFile, ...]
    full_stress: tuple[TightInclusionCSVFile, ...]


def build_file_level_split(
    files: Sequence[TightInclusionCSVFile],
    *,
    seed: int = DEFAULT_FILE_SPLIT_SEED,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.10,
    heldout_fraction: float = 0.20,
    unit_smoke_case: str = "unit-tests",
) -> TightInclusionFileSplit:
    total_fraction = train_fraction + validation_fraction + heldout_fraction
    if abs(total_fraction - 1.0) > 1.0e-8:
        raise ValueError("train/validation/heldout fractions must sum to 1")
    grouped: dict[tuple[str, TightInclusionQueryKind], list[TightInclusionCSVFile]] = {}
    unit_smoke: list[TightInclusionCSVFile] = []
    for item in files:
        if item.case_name == unit_smoke_case:
            unit_smoke.append(item)
            continue
        grouped.setdefault((item.case_name, item.kind), []).append(item)
    train: list[TightInclusionCSVFile] = []
    validation: list[TightInclusionCSVFile] = []
    heldout: list[TightInclusionCSVFile] = []
    for key, group in sorted(grouped.items()):
        shuffled = list(group)
        group_seed = seed + int(hashlib.sha1(f"{key[0]}:{key[1]}".encode("utf-8")).hexdigest()[:8], 16)
        random.Random(group_seed).shuffle(shuffled)
        n = len(shuffled)
        if n <= 1:
            heldout.extend(shuffled)
            continue
        train_count = int(round(n * train_fraction))
        val_count = int(round(n * validation_fraction))
        train_count = max(1, min(train_count, n - 1))
        val_count = max(0, min(val_count, n - train_count - 1))
        train.extend(shuffled[:train_count])
        validation.extend(shuffled[train_count : train_count + val_count])
        heldout.extend(shuffled[train_count + val_count :])
    return TightInclusionFileSplit(
        train=tuple(sorted(train, key=lambda item: str(item.csv_path))),
        validation=tuple(sorted(validation, key=lambda item: str(item.csv_path))),
        heldout_test=tuple(sorted(heldout, key=lambda item: str(item.csv_path))),
        unit_smoke=tuple(sorted(unit_smoke, key=lambda item: str(item.csv_path))),
        full_stress=tuple(sorted(files, key=lambda item: str(item.csv_path))),
    )


__all__ = [
    "TIGHT_INCLUSION_CSV_COLUMNS",
    "TIGHT_INCLUSION_QUERY_ROWS",
    "TightInclusionCSVFile",
    "TightInclusionFileSplit",
    "TightInclusionPrimitiveQuery",
    "TightInclusionQueryKind",
    "build_file_level_split",
    "count_csv_lines",
    "default_tight_inclusion_dataset_root",
    "discover_tight_inclusion_csv_files",
    "infer_case_name",
    "infer_query_kind",
    "inspect_tight_inclusion_csv",
    "iter_dataset_queries",
    "iter_tight_inclusion_queries",
    "normalize_query_kind",
    "parse_query_lines",
    "parse_rational_vertex_row",
    "read_tight_inclusion_query",
]
