from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = "scene_object_envelope_motion_trap_native_small_run_id"
BENCH = ROOT / "src" / "benchmark" / RUN
COMPARISON = BENCH / f"{RUN}_strong_native_comparison.csv"
MAIN_TABLE = BENCH / f"{RUN}_main_scheduler_table.csv"
OUT_STEM = BENCH / "p3_motion_trap_native_scheduler_ablation_run_id"

SELECTED = [
    ("AllExact+TI", "all-exact"),
    ("FrozenLearnedAnyHit+TI", "frozen learned"),
    ("LearnedResidualAnyHit+TI", "learned residual"),
    ("RandomAnyHit+TI", "random"),
    ("ProximityHeuristicAnyHit+TI", "proximity"),
    ("MotionHeuristicAnyHit+TI", "motion"),
    ("BestFixedHeuristicOracle+TI", "best fixed heuristic oracle"),
    ("FairFrontierLearnedResidualAnyHit+TI", "fair-frontier learned residual"),
    ("FairFrontierProximityAnyHit+TI", "fair-frontier proximity"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in rows}


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else default


def first_float(row: dict[str, str], keys: tuple[str, ...], default: float = 0.0) -> float:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return float(row[key])
    return default


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def main() -> None:
    comparison = index(read_rows(COMPARISON))
    main_table = index(read_rows(MAIN_TABLE))
    out_rows: list[dict[str, object]] = []

    for method, label in SELECTED:
        row = comparison.get(method) or main_table[method]
        detail = main_table.get(method, row)
        out_rows.append(
            {
                "method": method,
                "label": label,
                "exact_calls": int(round(as_float(row, "exact_calls"))),
                "call_reduction_vs_all_exact": first_float(
                    row,
                    ("exact_call_reduction_vs_dense", "call_reduction_vs_all_exact"),
                ),
                "mean_first_hit_exact_calls": as_float(detail, "positive_first_hit_mean_exact_calls"),
                "positive_proposal_hits": int(round(as_float(row, "positive_proposal_hits"))),
                "positive_groups": int(round(as_float(row, "positive_groups"))),
                "native_exact_backend_ms": as_float(row, "native_exact_backend_ms"),
                "scheduler_backend_ms": first_float(row, ("total_wall_ms", "scheduler_backend_ms")),
                "fn": int(round(as_float(row, "fn"))),
            }
        )

    with OUT_STEM.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    OUT_STEM.with_suffix(".json").write_text(json.dumps(out_rows, indent=2), encoding="utf-8")

    lines = [
        "# P3 Motion-trap Native Scheduler Ablation",
        "",
        "This ablation reuses the completed native scene/object motion-trap benchmark and",
        "tables scheduler variants under the same held-out object-envelope protocol.",
        "The learned rows use frozen checkpoints/scores; every final decision is still",
        "native Tight-Inclusion exact CCD or conservative fallback.",
        "",
        "| Method | Exact calls | Call reduction | Mean first-hit calls | Pos. hits/groups | Native exact ms | Scheduler backend ms | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in out_rows:
        lines.append(
            "| {label} | {exact_calls:,} | {call_red} | {first:.3f} | {hits}/{groups} | "
            "{native:.3f} | {backend:.3f} | {fn} |".format(
                label=row["label"],
                exact_calls=row["exact_calls"],
                call_red=fmt_pct(float(row["call_reduction_vs_all_exact"])),
                first=float(row["mean_first_hit_exact_calls"]),
                hits=row["positive_proposal_hits"],
                groups=row["positive_groups"],
                native=float(row["native_exact_backend_ms"]),
                backend=float(row["scheduler_backend_ms"]),
                fn=row["fn"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Learned residual reaches one certified hit per positive group on the held-out motion-trap case, reducing 110,464 all-exact native calls to 16 calls with FN=0.",
            "- This is an exact-call scheduler ablation, not a backend wall-time dominance claim: the first learned hits can be harder native TI calls than the first simple-heuristic hits.",
            "- Fair-frontier rows share the same native frontier/top-K/partial-sort budget and show that the learned residual rule matches the best fixed proximity row in exact calls while preserving the certificate contract.",
            "",
        ]
    )
    OUT_STEM.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"csv": str(OUT_STEM.with_suffix(".csv")), "md": str(OUT_STEM.with_suffix(".md"))}, indent=2))


if __name__ == "__main__":
    main()
