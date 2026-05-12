from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Mapping, Sequence

from p2cccd.contracts import CandidateRecord, CertificateResult, CertificateStatus, ExactWorkItem


@dataclass(frozen=True, slots=True)
class CandidateDensityBin:
    key: str
    candidate_count: int
    hit_count: int
    density: float


@dataclass(frozen=True, slots=True)
class ExactWorkSummary:
    work_item_count: int
    max_depth: int
    avg_priority: float
    source_counts: dict[str, int]
    family_mask_counts: dict[int, int]


@dataclass(frozen=True, slots=True)
class CertificateTraceSummary:
    certificate_count: int
    collision_count: int
    separation_count: int
    undecided_count: int
    min_toi_upper: float
    min_safe_margin_lb: float


def candidate_density_by_slab(candidates: Sequence[CandidateRecord]) -> tuple[CandidateDensityBin, ...]:
    bins: dict[int, list[CandidateRecord]] = {}
    for candidate in candidates:
        bins.setdefault(candidate.slab_id, []).append(candidate)
    result: list[CandidateDensityBin] = []
    for slab_id, rows in sorted(bins.items()):
        hit_count = sum(max(0, int(candidate.rt_hit_count)) for candidate in rows)
        result.append(
            CandidateDensityBin(
                key=f"slab:{slab_id}",
                candidate_count=len(rows),
                hit_count=hit_count,
                density=hit_count / max(1, len(rows)),
            )
        )
    return tuple(result)


def summarize_exact_work(work_items: Sequence[ExactWorkItem]) -> ExactWorkSummary:
    source_counts: dict[str, int] = {}
    family_counts: dict[int, int] = {}
    for item in work_items:
        source = item.source.name.lower() if hasattr(item.source, "name") else str(item.source)
        source_counts[source] = source_counts.get(source, 0) + 1
        family_counts[item.feature_family_mask] = family_counts.get(item.feature_family_mask, 0) + 1
    return ExactWorkSummary(
        work_item_count=len(work_items),
        max_depth=max((int(item.depth) for item in work_items), default=0),
        avg_priority=sum(float(item.priority_score) for item in work_items) / max(1, len(work_items)),
        source_counts=source_counts,
        family_mask_counts=family_counts,
    )


def summarize_certificate_trace(certificates: Sequence[CertificateResult]) -> CertificateTraceSummary:
    collision_count = sum(1 for certificate in certificates if certificate.status is CertificateStatus.COLLISION)
    separation_count = sum(1 for certificate in certificates if certificate.status is CertificateStatus.SEPARATION)
    undecided_count = sum(1 for certificate in certificates if certificate.status is CertificateStatus.UNDECIDED)
    return CertificateTraceSummary(
        certificate_count=len(certificates),
        collision_count=collision_count,
        separation_count=separation_count,
        undecided_count=undecided_count,
        min_toi_upper=min((float(certificate.toi_upper) for certificate in certificates), default=1.0),
        min_safe_margin_lb=min((float(certificate.safe_margin_lb) for certificate in certificates), default=0.0),
    )


def _bar_svg(values: Mapping[str, float], *, title: str, width: int = 720, bar_height: int = 26) -> str:
    if width <= 0:
        raise ValueError("SVG width must be positive")
    max_value = max(values.values(), default=1.0)
    max_value = max(1.0e-12, max_value)
    height = 56 + max(1, len(values)) * (bar_height + 12)
    rows = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<text x="16" y="28" font-size="20" font-family="Georgia,serif">{escape(title)}</text>',
    ]
    y = 48
    for label, value in values.items():
        bar_width = int((width - 220) * (float(value) / max_value))
        rows.append(f'<text x="16" y="{y + 18}" font-size="13" font-family="Consolas,monospace">{escape(label)}</text>')
        rows.append(f'<rect x="180" y="{y}" width="{max(1, bar_width)}" height="{bar_height}" fill="#4f7cac" rx="5"/>')
        rows.append(f'<text x="{190 + max(1, bar_width)}" y="{y + 18}" font-size="13" font-family="Consolas,monospace">{float(value):.3f}</text>')
        y += bar_height + 12
    rows.append("</svg>")
    return "\n".join(rows)


def render_candidate_density_svg(candidates: Sequence[CandidateRecord]) -> str:
    bins = candidate_density_by_slab(candidates)
    return _bar_svg({entry.key: float(entry.candidate_count) for entry in bins}, title="Candidate Density By Slab")


def render_exact_work_svg(work_items: Sequence[ExactWorkItem]) -> str:
    summary = summarize_exact_work(work_items)
    return _bar_svg(summary.source_counts, title="Exact Work Items By Source")


def render_certificate_trace_svg(certificates: Sequence[CertificateResult]) -> str:
    summary = summarize_certificate_trace(certificates)
    return _bar_svg(
        {
            "collision": float(summary.collision_count),
            "separation": float(summary.separation_count),
            "undecided": float(summary.undecided_count),
        },
        title="Certificate Trace Status",
    )


def write_pipeline_debug_html(
    path: str | Path,
    *,
    candidates: Sequence[CandidateRecord],
    work_items: Sequence[ExactWorkItem],
    certificates: Sequence[CertificateResult],
    title: str = "P2CCCD Pipeline Debug View",
) -> Path:
    candidate_svg = render_candidate_density_svg(candidates)
    work_svg = render_exact_work_svg(work_items)
    certificate_svg = render_certificate_trace_svg(certificates)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            (
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\">",
                f"<title>{escape(title)}</title>",
                "<style>body{font-family:Georgia,serif;margin:32px;background:#f6f0e4;color:#162015}"
                "section{background:#fffaf0;border:1px solid #d8c7aa;border-radius:18px;padding:18px;margin:18px 0}"
                "svg{width:100%;height:auto}</style>",
                "</head><body>",
                f"<h1>{escape(title)}</h1>",
                f"<p>candidates={len(candidates)}, work_items={len(work_items)}, certificates={len(certificates)}</p>",
                f"<section>{candidate_svg}</section>",
                f"<section>{work_svg}</section>",
                f"<section>{certificate_svg}</section>",
                "</body></html>",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return output
