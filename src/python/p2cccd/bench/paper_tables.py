from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from p2cccd.contracts import BenchmarkRowV2


@dataclass(frozen=True, slots=True)
class PaperTableRow:
    method_name: str
    dataset_name: str
    scene_name: str
    query_count: int
    fn_count: int
    candidate_recall: float
    total_ms: float
    qps: float
    latency_p95_ms: float
    exact_calls_total: int


def paper_table_rows_from_benchmark_rows(rows: Sequence[BenchmarkRowV2]) -> tuple[PaperTableRow, ...]:
    return tuple(
        PaperTableRow(
            method_name=row.method_name,
            dataset_name=row.dataset_name,
            scene_name=row.scene_name,
            query_count=row.query_count,
            fn_count=row.fn_count,
            candidate_recall=row.candidate_recall,
            total_ms=row.total_ms,
            qps=row.qps,
            latency_p95_ms=row.latency_p95_ms,
            exact_calls_total=row.exact_calls_total,
        )
        for row in rows
    )


def format_paper_table_markdown(rows: Sequence[PaperTableRow], *, include_scene: bool = False) -> str:
    headers = ["method", "dataset"]
    if include_scene:
        headers.append("scene")
    headers.extend(("queries", "FN", "recall", "total ms", "qps", "p95 ms", "exact calls"))
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values: list[str] = [row.method_name, row.dataset_name]
        if include_scene:
            values.append(row.scene_name)
        values.extend(
            (
                str(row.query_count),
                str(row.fn_count),
                f"{row.candidate_recall:.6f}",
                f"{row.total_ms:.6f}",
                f"{row.qps:.3f}",
                f"{row.latency_p95_ms:.6f}",
                str(row.exact_calls_total),
            )
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
