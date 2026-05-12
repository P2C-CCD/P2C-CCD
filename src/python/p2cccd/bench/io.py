from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from p2cccd.contracts import BenchmarkRowV2, BenchmarkRunMeta
from p2cccd.serialization import from_json, read_jsonl
from p2cccd.validators import validate_benchmark_row_v2, validate_benchmark_run_meta

from .summary import BenchmarkExportPaths, export_benchmark_run


@dataclass(frozen=True, slots=True)
class BenchmarkRunData:
    run_dir: Path
    meta: BenchmarkRunMeta
    rows: tuple[BenchmarkRowV2, ...]


def read_benchmark_run(run_dir: str | Path) -> BenchmarkRunData:
    root = Path(run_dir)
    meta_path = root / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing benchmark run_meta.json: {meta_path}")
    meta = validate_benchmark_run_meta(from_json(BenchmarkRunMeta, meta_path.read_text(encoding="utf-8")))
    rows_path = root / meta.output_jsonl
    rows = tuple(validate_benchmark_row_v2(row) for row in read_jsonl(rows_path, BenchmarkRowV2))
    for row in rows:
        if row.run_id != meta.run_id:
            raise ValueError("BenchmarkRowV2.run_id must match BenchmarkRunMeta.run_id")
        if row.config_hash != meta.config_hash:
            raise ValueError("BenchmarkRowV2.config_hash must match BenchmarkRunMeta.config_hash")
    return BenchmarkRunData(run_dir=root, meta=meta, rows=rows)


def write_benchmark_run(
    run_dir: str | Path,
    meta: BenchmarkRunMeta,
    rows: Sequence[BenchmarkRowV2],
) -> BenchmarkExportPaths:
    return export_benchmark_run(run_dir, meta, rows)


def discover_benchmark_run_dirs(root: str | Path) -> tuple[Path, ...]:
    base = Path(root)
    if not base.exists():
        return ()
    return tuple(sorted(path.parent for path in base.rglob("run_meta.json")))
