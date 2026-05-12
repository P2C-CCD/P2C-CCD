from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "src" / "MyDemo" / "paper_aris_ccf_a_cases_run_id" / "car_wall_impact"
BENCH_PATH = ROOT / "src" / "benchmark" / "car_wall_impact_dense_wall_patch_rtstpf_training_benchmark_run_id.json"
RELEASE_CASE_FIG_DIR = ROOT / "assets" / "figures" / "cases"
ANALYSIS_DIR = CASE_DIR / "analysis"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_benchmark_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [{key: str(value) for key, value in row.items()} for row in data.get("rows", [])]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else float("nan")


def _heldout_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("split") == "heldout"]


def _method_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in rows}


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "jpg"):
        fig.savefig(ANALYSIS_DIR / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.06)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _release_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_time_series(metrics: dict) -> dict[str, np.ndarray]:
    bm = metrics["benchmark_metrics"]
    ref = bm["mujoco_reference"]
    total_contacts = np.asarray(ref["contact_counts_by_frame"], dtype=np.float64)
    vehicle_contacts = np.asarray(ref["vehicle_brick_contact_counts_by_frame"], dtype=np.float64)
    frame_count = int(metrics["frame_count"])
    duration = float(metrics["duration_seconds"])
    times = np.linspace(0.0, duration, frame_count)
    toi = float(bm["toi_seconds"])
    impact_speed = float(bm["vehicle_impact_speed_mps"])

    # This is an interpretable kinematic proxy: it uses the reported vehicle
    # speed and certified TOI to show approach-to-contact timing. It is not a
    # measured signed distance field.
    gap_proxy = np.maximum((toi - times) * impact_speed, 0.0)

    audit = bm["p2cccd_swept_ccd_audit"]
    first_seg = int(audit["first_toi_segment"])
    hit_count = int(audit["collision_segments"])
    segment_times = times[:-1]
    hit_window = np.zeros_like(segment_times)
    hit_window[first_seg : min(len(hit_window), first_seg + hit_count)] = 1.0

    return {
        "times": times,
        "segment_times": segment_times,
        "total_contacts": total_contacts,
        "vehicle_contacts": vehicle_contacts,
        "gap_proxy": gap_proxy,
        "hit_window": hit_window,
    }


def plot_contact_timeline(metrics: dict, heldout: list[dict[str, str]]) -> None:
    bm = metrics["benchmark_metrics"]
    audit = bm["p2cccd_swept_ccd_audit"]
    ts = build_time_series(metrics)
    lookup = _method_lookup(heldout)
    toi = float(bm["toi_seconds"])

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2))
    ax = axes[0, 0]
    ax.plot(ts["times"], ts["gap_proxy"], color="#1f77b4", linewidth=2.3)
    ax.axvline(toi, color="#d62728", linestyle="--", linewidth=1.8, label=f"certified TOI={toi:.3f}s")
    ax.set_title("Approach-to-wall distance proxy")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("front gap proxy (m)")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[0, 1]
    ax.plot(ts["times"], ts["total_contacts"], label="MuJoCo all contacts", color="#4c78a8", linewidth=2.0)
    ax.plot(ts["times"], ts["vehicle_contacts"], label="vehicle-brick contacts", color="#f58518", linewidth=2.0)
    ax.axvline(toi, color="#d62728", linestyle="--", linewidth=1.6)
    ax.set_title("Reference contact pressure over replay")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("contact count")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.fill_between(ts["segment_times"], 0, ts["hit_window"], step="post", color="#2ca02c", alpha=0.45)
    ax.axvline(toi, color="#d62728", linestyle="--", linewidth=1.6)
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1], labels=["separation", "certified hit"])
    ax.set_title("P2CCCD swept-certificate hit window")
    ax.set_xlabel("time (s)")

    ax = axes[1, 1]
    methods = ["NoProposal", "RTExact", "RTSTPFExact", "Random-STPF"]
    exact_calls = [_float(lookup[m], "exact_calls") for m in methods]
    colors = ["#6b7280", "#4c78a8", "#2ca02c", "#f58518"]
    bars = ax.bar(methods, exact_calls, color=colors)
    ax.set_yscale("log")
    ax.set_title("Heldout exact calls after scheduling")
    ax.set_ylabel("exact calls (log)")
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, exact_calls):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.08, f"{int(value):,}", ha="center", va="bottom", fontsize=10)

    fig.suptitle(
        f"Car-wall impact contact timeline: FN={int(bm['fn'])}, "
        f"candidate primitives={int(audit['candidate_count']):,}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "car_wall_contact_timeline")
    plt.close(fig)


def plot_energy_and_work(metrics: dict, heldout: list[dict[str, str]]) -> None:
    bm = metrics["benchmark_metrics"]
    audit = bm["p2cccd_swept_ccd_audit"]
    lookup = _method_lookup(heldout)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2))
    ax = axes[0, 0]
    energy_names = ["pre-impact KE", "exit KE", "absorbed"]
    energy_values = [
        float(bm["vehicle_kinetic_energy_pre_j"]),
        float(bm["vehicle_kinetic_energy_exit_j"]),
        float(bm["absorbed_energy_j"]),
    ]
    bars = ax.bar(energy_names, np.asarray(energy_values) / 1000.0, color=["#4c78a8", "#72b7b2", "#e45756"])
    ax.set_ylabel("energy (kJ)")
    ax.set_title("Vehicle energy budget")
    ax.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, energy_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value / 1000.0 * 1.02, f"{value/1000.0:.1f}", ha="center", va="bottom", fontsize=10)

    ax = axes[0, 1]
    family = audit["exact_family_counts"]
    family_names = ["car V - brick F", "brick V - car F", "edge - edge"]
    family_values = [
        int(family["car_vertex_vs_brick_triangle"]),
        int(family["brick_vertex_vs_car_triangle"]),
        int(family["car_edge_vs_brick_edge"]),
    ]
    ax.pie(
        family_values,
        labels=family_names,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        colors=["#59a14f", "#edc948", "#b07aa1"],
        textprops={"fontsize": 10},
    )
    ax.set_title("Exact primitive family mix")

    ax = axes[1, 0]
    methods = ["NoProposal", "RTSTPFExact", "Random-STPF"]
    work = [_float(lookup[m], "exact_work") for m in methods]
    wall = [_float(lookup[m], "wall_ms") for m in methods]
    x = np.arange(len(methods))
    width = 0.38
    ax2 = ax.twinx()
    b1 = ax.bar(x - width / 2, work, width=width, color="#4c78a8", label="exact work")
    b2 = ax2.bar(x + width / 2, wall, width=width, color="#f58518", alpha=0.82, label="wall ms")
    ax.set_xticks(x, methods, rotation=12)
    ax.set_ylabel("exact work units")
    ax2.set_ylabel("wall time (ms)")
    ax.set_title("Heldout work reduction vs proposal overhead")
    ax.legend([b1, b2], ["exact work", "wall time"], frameon=False, loc="upper right")

    ax = axes[1, 1]
    labels = ["bricks", "displaced", "max disp. (m)", "max speed (m/s)", "max contacts"]
    vals = [
        float(bm["brick_count"]),
        float(bm["displaced_brick_count"]),
        float(bm["max_brick_displacement_m"]),
        float(bm["max_brick_speed_mps"]),
        float(bm["mujoco_reference"]["max_contact_count"]),
    ]
    ax.barh(labels, vals, color=["#9ca3af", "#60a5fa", "#34d399", "#fbbf24", "#f87171"])
    ax.set_xscale("log")
    ax.set_title("Rigid wall response summary")
    ax.set_xlabel("value (log scale)")
    for i, value in enumerate(vals):
        ax.text(value * 1.05, i, f"{value:.2f}" if value < 100 else f"{value:.0f}", va="center", fontsize=10)

    rt = lookup["RTSTPFExact"]
    fig.suptitle(
        f"Car-wall impact energy and exact-work audit: "
        f"RTSTPF work reduction={100*_float(rt, 'exact_work_reduction'):.2f}%, "
        f"call reduction={100*_float(rt, 'exact_call_reduction'):.2f}%",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "car_wall_energy_and_work")
    plt.close(fig)


def write_analysis_tables(metrics: dict, heldout: list[dict[str, str]]) -> None:
    bm = metrics["benchmark_metrics"]
    audit = bm["p2cccd_swept_ccd_audit"]
    ts = build_time_series(metrics)

    rows = []
    for idx, t in enumerate(ts["times"]):
        rows.append(
            [
                idx,
                f"{float(t):.9f}",
                f"{float(ts['gap_proxy'][idx]):.9f}",
                int(ts["total_contacts"][idx]),
                int(ts["vehicle_contacts"][idx]),
            ]
        )
    _write_csv(
        ANALYSIS_DIR / "car_wall_contact_timeseries.csv",
        ["frame", "time_s", "front_gap_proxy_m", "mujoco_total_contacts", "mujoco_vehicle_brick_contacts"],
        rows,
    )

    _write_csv(
        ANALYSIS_DIR / "car_wall_heldout_method_summary.csv",
        [
            "method",
            "rows",
            "groups",
            "exact_calls",
            "exact_call_reduction",
            "exact_work",
            "exact_work_reduction",
            "wall_ms",
            "proposal_ms",
            "fn",
            "recall",
            "precision",
            "provider",
        ],
        [
            [
                row["method"],
                row["rows"],
                row["groups"],
                row["exact_calls"],
                row["exact_call_reduction"],
                row["exact_work"],
                row["exact_work_reduction"],
                row["wall_ms"],
                row["proposal_ms"],
                row["fn"],
                row["recall"],
                row["precision"],
                row["provider"],
            ]
            for row in heldout
        ],
    )

    summary = {
        "source_metrics": _release_path(CASE_DIR / "metrics.json"),
        "source_benchmark_report": _release_path(BENCH_PATH),
        "outputs": {
            "contact_timeline": "analysis/car_wall_contact_timeline.{png,pdf,jpg}",
            "energy_and_work": "analysis/car_wall_energy_and_work.{png,pdf,jpg}",
            "timeseries_csv": "analysis/car_wall_contact_timeseries.csv",
            "heldout_method_csv": "analysis/car_wall_heldout_method_summary.csv",
        },
        "data_boundary": {
            "gap_proxy": "front_gap_proxy_m is reconstructed from reported vehicle impact speed and certified TOI; it is not a measured signed-distance field.",
            "mujoco_contacts": "MuJoCo contact counts are reference/control contacts and are not counted as P2CCCD false negatives.",
            "training_benchmark": "RTSTPF exact-call reductions come from the learned dense-wall-patch benchmark CSV, not from the physical replay exact-fallback count.",
        },
        "key_numbers": {
            "toi_seconds": bm["toi_seconds"],
            "fn": bm["fn"],
            "p2cccd_replay_candidate_count": audit["candidate_count"],
            "p2cccd_replay_exact_fallback_count": audit["exact_fallback_count"],
            "replay_collision_segments": audit["collision_segments"],
            "mujoco_max_contact_count": bm["mujoco_reference"]["max_contact_count"],
            "mujoco_max_vehicle_brick_contact_count": bm["mujoco_reference"]["max_vehicle_brick_reference_contact_count"],
            "vehicle_kinetic_energy_pre_j": bm["vehicle_kinetic_energy_pre_j"],
            "vehicle_kinetic_energy_exit_j": bm["vehicle_kinetic_energy_exit_j"],
            "absorbed_energy_j": bm["absorbed_energy_j"],
            "max_brick_displacement_m": bm["max_brick_displacement_m"],
            "max_brick_speed_mps": bm["max_brick_speed_mps"],
        },
    }
    (ANALYSIS_DIR / "car_wall_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# Car-Wall Impact Analysis Curves

This folder consolidates the data-driven analysis figures for the car-wall impact case.

## Generated figures

- `car_wall_contact_timeline.png/pdf/jpg`: approach-to-contact proxy, MuJoCo reference contact counts, P2CCCD certified-hit window, and heldout exact-call comparison.
- `car_wall_energy_and_work.png/pdf/jpg`: vehicle energy budget, exact primitive family mix, heldout exact-work/wall-time comparison, and rigid-wall response summary.
- `paper_car_wall_impact_analysis.*`: copy of the paper-style visual analysis panel when available.

## Data sources

- Case metrics: `{_release_path(CASE_DIR / 'metrics.json')}`
- Heldout benchmark report: `{_release_path(BENCH_PATH)}`

## Safe interpretation

- The contact-count curves are from MuJoCo reference replay and show physical contact pressure.
- P2CCCD correctness metrics come from adjacent-frame swept CCD over the rendered car mesh and brick box meshes.
- `front_gap_proxy_m` is reconstructed from vehicle speed and certified TOI; it is an explanatory timing proxy, not a measured signed distance field.
- RTSTPF exact-call and exact-work reductions are from the learned dense-wall-patch benchmark CSV.

## Key numbers

- Certified TOI: `{bm['toi_seconds']:.6f}` s.
- P2CCCD replay candidate primitives: `{audit['candidate_count']:,}`.
- P2CCCD replay exact fallback primitives: `{audit['exact_fallback_count']:,}`.
- Replay collision segments: `{audit['collision_segments']}`.
- FN: `{bm['fn']}`.
- Vehicle kinetic energy: `{bm['vehicle_kinetic_energy_pre_j']/1000:.2f}` kJ before impact, `{bm['vehicle_kinetic_energy_exit_j']/1000:.2f}` kJ after wall exit, `{bm['absorbed_energy_j']/1000:.2f}` kJ absorbed.
- Displaced bricks: `{bm['displaced_brick_count']}` / `{bm['brick_count']}`.
- Max brick displacement: `{bm['max_brick_displacement_m']:.3f}` m.
- Max brick speed: `{bm['max_brick_speed_mps']:.3f}` m/s.
"""
    (ANALYSIS_DIR / "analysis_report.md").write_text(report, encoding="utf-8")


def copy_release_visual_panel() -> None:
    for src_name, out_name in (("Car_wall.jpg", "release_car_wall_impact_visual.jpg"),):
        src = RELEASE_CASE_FIG_DIR / src_name
        if src.exists():
            shutil.copy2(src, ANALYSIS_DIR / out_name)


def plot_release_benchmark_summary(heldout: list[dict[str, str]]) -> None:
    lookup = _method_lookup(heldout)
    methods = [name for name in ("NoProposal", "RTExact", "RTSTPFExact", "Random-STPF") if name in lookup]
    if not methods:
        raise RuntimeError(f"no supported heldout method rows found in {BENCH_PATH}")

    exact_calls = [_float(lookup[m], "exact_calls") for m in methods]
    exact_work = [_float(lookup[m], "exact_work") for m in methods]
    wall_ms = [_float(lookup[m], "wall_ms") for m in methods]
    colors = ["#6b7280", "#4c78a8", "#2ca02c", "#f58518"][: len(methods)]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    for ax, values, title, ylabel in (
        (axes[0], exact_calls, "Exact calls", "calls"),
        (axes[1], exact_work, "Exact work", "work units"),
        (axes[2], wall_ms, "Wall time", "ms"),
    ):
        bars = ax.bar(methods, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=18)
        for bar, value in zip(bars, values):
            label = f"{int(value):,}" if value >= 100 else f"{value:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.02, label, ha="center", va="bottom", fontsize=9)

    fig.suptitle("Car-wall held-out dense wall patch benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "car_wall_release_benchmark_summary")
    plt.close(fig)


def write_release_benchmark_tables(heldout: list[dict[str, str]]) -> None:
    _write_csv(
        ANALYSIS_DIR / "car_wall_heldout_method_summary.csv",
        [
            "method",
            "rows",
            "groups",
            "exact_calls",
            "exact_call_reduction",
            "exact_work",
            "exact_work_reduction",
            "wall_ms",
            "proposal_ms",
            "fn",
            "recall",
            "precision",
            "provider",
        ],
        [
            [
                row.get("method", ""),
                row.get("rows", ""),
                row.get("groups", ""),
                row.get("exact_calls", ""),
                row.get("exact_call_reduction", ""),
                row.get("exact_work", ""),
                row.get("exact_work_reduction", ""),
                row.get("wall_ms", ""),
                row.get("proposal_ms", ""),
                row.get("fn", ""),
                row.get("recall", ""),
                row.get("precision", ""),
                row.get("provider", ""),
            ]
            for row in heldout
        ],
    )

    summary = {
        "source_benchmark_report": _release_path(BENCH_PATH),
        "release_visual": _release_path(RELEASE_CASE_FIG_DIR / "Car_wall.jpg"),
        "outputs": {
            "benchmark_summary": "analysis/car_wall_release_benchmark_summary.{png,pdf,jpg}",
            "heldout_method_csv": "analysis/car_wall_heldout_method_summary.csv",
        },
        "data_boundary": {
            "release_mode": "The public release bundles the held-out benchmark report and representative case visual. Full physical replay metrics are optional external rerun artifacts.",
        },
    }
    (ANALYSIS_DIR / "car_wall_release_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# Car-Wall Held-Out Benchmark Summary

This folder contains release-local summary outputs generated from the bundled
car-wall benchmark report.

## Generated outputs

- `car_wall_release_benchmark_summary.png/pdf/jpg`: exact-call, exact-work, and wall-time summary.
- `car_wall_heldout_method_summary.csv`: held-out benchmark rows.
- `release_car_wall_impact_visual.jpg`: copy of the bundled representative visual when available.

## Data sources

- Heldout benchmark report: `{_release_path(BENCH_PATH)}`
- Representative visual: `{_release_path(RELEASE_CASE_FIG_DIR / 'Car_wall.jpg')}`

## Boundary

The public release includes the benchmark summary and representative figure. Full
physics replay artifacts remain optional external reruns.
"""
    (ANALYSIS_DIR / "analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    heldout = _heldout_rows(_load_benchmark_rows(BENCH_PATH))
    if not heldout:
        raise RuntimeError(f"no heldout rows found in {BENCH_PATH}")
    metrics_path = CASE_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = _load_json(metrics_path)
        plot_contact_timeline(metrics, heldout)
        plot_energy_and_work(metrics, heldout)
        write_analysis_tables(metrics, heldout)
    else:
        plot_release_benchmark_summary(heldout)
        write_release_benchmark_tables(heldout)
    copy_release_visual_panel()
    print(ANALYSIS_DIR)


if __name__ == "__main__":
    main()
