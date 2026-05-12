from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN = "native_ti_heldout_dense_group_run_id"
BENCH = ROOT / "src" / "benchmark" / RUN
SOURCE = BENCH / f"{RUN}.json"
OUT_STEM = BENCH / "heldout_native_ti_mixed_run_id"

METHOD_ORDER = [
    "AllExact+TI",
    "Random+TI",
    "FrozenLearned+TI",
    "SingleHeuristicProximity+TI",
    "HeuristicMotionHigh+TI",
    "BestFixedHeuristicOracle+TI",
]


def normalize_method(method: str) -> str:
    if method.startswith("BestFixedHeuristicOracle+TI"):
        return "BestFixedHeuristicOracle+TI"
    return method


def method_label(method: str) -> str:
    labels = {
        "AllExact+TI": "all-exact",
        "Random+TI": "random",
        "FrozenLearned+TI": "frozen STPF",
        "SingleHeuristicProximity+TI": "fixed proximity",
        "HeuristicMotionHigh+TI": "predeclared motion-high",
        "BestFixedHeuristicOracle+TI": "best fixed heuristic oracle",
    }
    return labels[normalize_method(method)]


def collect_methods(split: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in split["methods"]:
        rows[normalize_method(row["method"])] = row
    for row in split["all_heuristics"]:
        rows[normalize_method(row["method"])] = row
    return rows


def format_pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def compact_split_name(name: str) -> str:
    if name == "group_heldout":
        return "group-heldout"
    if name.startswith("source_heldout_"):
        return name.replace("source_heldout_", "source-heldout ")
    return name.replace("_", "-")


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    out_rows: list[dict[str, Any]] = []
    for split in payload["splits"]:
        methods = collect_methods(split)
        for method in METHOD_ORDER:
            row = methods[method]
            out_rows.append(
                {
                    "split": split["split"],
                    "split_label": compact_split_name(split["split"]),
                    "method": method,
                    "method_label": method_label(method),
                    "groups": int(row["group_count"]),
                    "candidates": int(row["candidate_count"]),
                    "positive_groups": int(row["positive_group_count"]),
                    "negative_groups": int(row["group_count"]) - int(row["positive_group_count"]),
                    "exact_calls": int(row["exact_calls"]),
                    "call_reduction": float(row["exact_call_reduction"]),
                    "tp": int(row["tp"]),
                    "tn": int(row["tn"]),
                    "fp": int(row["fp"]),
                    "fn": int(row["fn"]),
                    "first_positive_rank_mean": float(row["first_positive_rank_mean"]),
                    "exact_ms": float(row["exact_ms"]),
                    "wall_ms": float(row["wall_ms"]),
                }
            )

    with OUT_STEM.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    OUT_STEM.with_suffix(".json").write_text(json.dumps(out_rows, indent=2), encoding="utf-8")

    lines = [
        "# Public held-out native TI mixed benchmark",
        "",
        "This table reorganizes the completed frozen-checkpoint native TI held-out",
        "benchmark so the deployable fixed `MotionHigh` rule is visible rather than",
        "hidden only inside the retrospective heuristic-oracle row.  All rows use",
        "native Tight-Inclusion certificates and the same mixed positive/negative",
        "hard-near-miss groups; FP entries are native TI positives in nominal",
        "near-miss groups, not learned classifier false positives.",
        "",
        "| Split | Method | Pos./neg. groups | Exact calls | Call red. | TP/TN/FP/FN | First-hit rank | Wall ms |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in out_rows:
        lines.append(
            "| {split} | {method} | {pos}/{neg} | {calls:,} | {red} | {tp}/{tn}/{fp}/{fn} | {rank:.3f} | {wall:.3f} |".format(
                split=row["split_label"],
                method=row["method_label"],
                pos=row["positive_groups"],
                neg=row["negative_groups"],
                calls=row["exact_calls"],
                red=format_pct(row["call_reduction"]),
                tp=row["tp"],
                tn=row["tn"],
                fp=row["fp"],
                fn=row["fn"],
                rank=row["first_positive_rank_mean"],
                wall=row["wall_ms"],
            )
        )
    lines.extend(
        [
            "",
            "## Main reading",
            "",
            "- On group-heldout mixed groups, frozen STPF reduces exact calls from 65,536 to 31,445, beating random, fixed proximity, and the predeclared motion-high rule while preserving FN=0.",
            "- On source-heldout ShapeNetCore, frozen STPF reduces exact calls from 32,768 to 18,118 and is effectively tied with predeclared motion-high / best fixed heuristic oracle; this supports cross-source stability, not universal dominance over every hand rule.",
            "- The table is therefore the main learned-vs-fixed-rule mixed evidence, while scene/object envelope rows remain a certificate/fallback audit.",
            "",
        ]
    )
    OUT_STEM.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"csv": str(OUT_STEM.with_suffix(".csv")), "md": str(OUT_STEM.with_suffix(".md"))}, indent=2))


if __name__ == "__main__":
    main()
