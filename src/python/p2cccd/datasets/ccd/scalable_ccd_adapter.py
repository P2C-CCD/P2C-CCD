from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .baseline_registry import default_baseline_root
from .contracts import (
    CCD_ADAPTER_SCHEMA_VERSION,
    CCDQueryFamily,
    DatasetQueryBatch,
    DatasetScene,
    ExternalCCDQuery,
    SourceLicense,
    Vec3,
)


SCALABLE_SAMPLE_SOURCE_NAME = "Sample-Scalable-CCD-Data"


@dataclass(frozen=True, slots=True)
class ScalableSampleBatchInfo:
    scene_name: str
    step: int
    family: CCDQueryFamily
    query_csv: Path
    bool_json: Path | None
    boxes_json: Path | None
    roots_archive: Path | None
    query_count: int
    collision_count: int | None
    box_pair_count: int | None

    @property
    def batch_id(self) -> str:
        return f"{self.scene_name}:{self.step}{self.family.value}"


def _rational_pair_to_float(numerator: str, denominator: str) -> float:
    den = int(denominator)
    if den == 0:
        raise ValueError("rational coordinate denominator cannot be zero")
    return float(Fraction(int(numerator), den))


def _parse_vertex(row: list[str]) -> Vec3:
    if len(row) != 6:
        raise ValueError(f"expected 6 rational columns per vertex row, got {len(row)}")
    return (
        _rational_pair_to_float(row[0], row[1]),
        _rational_pair_to_float(row[2], row[3]),
        _rational_pair_to_float(row[4], row[5]),
    )


def _read_bool_json(path: Path | None) -> list[bool]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected boolean list in {path}")
    return [bool(item) for item in data]


def _read_boxes_json(path: Path | None) -> list[tuple[int, int]]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    boxes: list[tuple[int, int]] = []
    for item in data:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"expected box-pair entries in {path}")
        boxes.append((int(item[0]), int(item[1])))
    return boxes


def _count_query_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        row_count = sum(1 for row in csv.reader(handle) if row)
    if row_count % 8 != 0:
        raise ValueError(f"query CSV row count must be divisible by 8: {path}")
    return row_count // 8


def _parse_query_filename(path: Path) -> tuple[int, CCDQueryFamily]:
    stem = path.stem
    if stem.endswith("vf"):
        return int(stem[:-2]), CCDQueryFamily.VERTEX_FACE
    if stem.endswith("ee"):
        return int(stem[:-2]), CCDQueryFamily.EDGE_EDGE
    raise ValueError(f"unsupported scalable CCD query file name: {path.name}")


class ScalableCCDSampleAdapter:
    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            self.root = default_baseline_root() / SCALABLE_SAMPLE_SOURCE_NAME
        else:
            self.root = Path(root)

    def require_available(self) -> None:
        self.license().require_available()
        if not self.root.exists():
            raise FileNotFoundError(f"Scalable CCD sample data not found: {self.root}")
        if not (self.root / "README.md").exists():
            raise FileNotFoundError(f"Scalable CCD sample README missing: {self.root}")

    def license(self) -> SourceLicense:
        return SourceLicense(
            name=SCALABLE_SAMPLE_SOURCE_NAME,
            license_path=self.root / "LICENSE",
            url="https://github.com/Continuous-Collision-Detection/Sample-Scalable-CCD-Data",
            terms="Use as external CCD query data; preserve upstream dataset license and labels.",
        )

    def list_scenes(self) -> tuple[DatasetScene, ...]:
        self.require_available()
        scenes: list[DatasetScene] = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue
            queries_dir = path / "queries"
            if not queries_dir.exists():
                continue
            frames = tuple(sorted((path / "frames").glob("*"))) if (path / "frames").exists() else ()
            scenes.append(
                DatasetScene(
                    schema_version=CCD_ADAPTER_SCHEMA_VERSION,
                    source_name=SCALABLE_SAMPLE_SOURCE_NAME,
                    scene_name=path.name,
                    scene_path=path,
                    frames=frames,
                    metadata={
                        "query_file_count": len(tuple(queries_dir.glob("*.csv"))),
                        "frame_count": len(frames),
                    },
                )
            )
        return tuple(scenes)

    def list_query_batches(self, scene_name: str | None = None) -> tuple[ScalableSampleBatchInfo, ...]:
        self.require_available()
        scene_paths = [self.root / scene_name] if scene_name is not None else [scene.scene_path for scene in self.list_scenes()]
        batches: list[ScalableSampleBatchInfo] = []
        for scene_path in scene_paths:
            queries_dir = scene_path / "queries"
            if not queries_dir.exists():
                continue
            for query_csv in sorted(queries_dir.glob("*.csv")):
                step, family = _parse_query_filename(query_csv)
                stem = query_csv.stem
                bool_json = scene_path / "mma_bool" / f"{stem}_mma_bool.json"
                boxes_json = scene_path / "boxes" / f"{stem}.json"
                roots_archive = scene_path / "roots" / f"{stem}_roots.tar.gz"
                labels = _read_bool_json(bool_json)
                boxes = _read_boxes_json(boxes_json)
                batches.append(
                    ScalableSampleBatchInfo(
                        scene_name=scene_path.name,
                        step=step,
                        family=family,
                        query_csv=query_csv,
                        bool_json=bool_json if bool_json.exists() else None,
                        boxes_json=boxes_json if boxes_json.exists() else None,
                        roots_archive=roots_archive if roots_archive.exists() else None,
                        query_count=_count_query_csv_rows(query_csv),
                        collision_count=sum(labels) if labels else None,
                        box_pair_count=len(boxes) if boxes else None,
                    )
                )
        return tuple(batches)

    def load_query_batch(
        self,
        scene_name: str,
        *,
        family: CCDQueryFamily | str,
        step: int,
        limit: int | None = None,
    ) -> DatasetQueryBatch:
        self.require_available()
        query_family = family if isinstance(family, CCDQueryFamily) else CCDQueryFamily(str(family))
        stem = f"{step}{query_family.value}"
        scene_path = self.root / scene_name
        query_csv = scene_path / "queries" / f"{stem}.csv"
        if not query_csv.exists():
            raise FileNotFoundError(f"Scalable CCD query CSV not found: {query_csv}")
        labels = _read_bool_json(scene_path / "mma_bool" / f"{stem}_mma_bool.json")
        boxes = _read_boxes_json(scene_path / "boxes" / f"{stem}.json")
        batch_id = f"{scene_name}:{stem}"
        queries: list[ExternalCCDQuery] = []

        with query_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            current_rows: list[Vec3] = []
            for row in reader:
                if not row:
                    continue
                current_rows.append(_parse_vertex(row))
                if len(current_rows) != 8:
                    continue
                source_index = len(queries)
                queries.append(
                    ExternalCCDQuery(
                        schema_version=CCD_ADAPTER_SCHEMA_VERSION,
                        source_name=SCALABLE_SAMPLE_SOURCE_NAME,
                        scene_name=scene_name,
                        batch_id=batch_id,
                        query_id=source_index,
                        source_query_index=source_index,
                        family=query_family,
                        vertices_t0=tuple(current_rows[:4]),  # type: ignore[arg-type]
                        vertices_t1=tuple(current_rows[4:]),  # type: ignore[arg-type]
                        ground_truth_collides=labels[source_index] if source_index < len(labels) else None,
                        box_pair=boxes[source_index] if source_index < len(boxes) else None,
                        metadata={
                            "query_csv": str(query_csv),
                            "witness_family": query_family.p2cccd_witness_family,
                        },
                    )
                )
                current_rows = []
                if limit is not None and len(queries) >= limit:
                    break
            if current_rows:
                raise ValueError(f"incomplete query group in {query_csv}")

        frame_t0 = scene_path / "frames" / f"{step}.ply"
        frame_t1 = scene_path / "frames" / f"{step + 1}.ply"
        return DatasetQueryBatch(
            schema_version=CCD_ADAPTER_SCHEMA_VERSION,
            source_name=SCALABLE_SAMPLE_SOURCE_NAME,
            scene_name=scene_name,
            batch_id=batch_id,
            family=query_family,
            queries=tuple(queries),
            frame_t0=frame_t0 if frame_t0.exists() else None,
            frame_t1=frame_t1 if frame_t1.exists() else None,
            metadata={
                "query_csv": str(query_csv),
                "known_label_count": sum(1 for query in queries if query.ground_truth_collides is not None),
                "collision_count": sum(1 for query in queries if query.ground_truth_collides is True),
                "box_pair_count": len(boxes),
                "limit": limit,
            },
        )
