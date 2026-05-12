from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "src" / "tools"
CASE_DIR = ROOT / "src" / "MyDemo" / "paper_aris_ccf_a_cases_run_id" / "car_wall_impact"
ANALYSIS_DIR = CASE_DIR / "analysis"
METRICS_PATH = CASE_DIR / "metrics.json"

sys.path.insert(0, str(TOOLS_DIR))
import render_aris_real_mesh_physics_cases as aris_cases  # noqa: E402
from generate_car_wall_brick_spatial_curves import rebuild_brick_centers, upsert_section  # noqa: E402


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.7,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "jpg"):
        fig.savefig(ANALYSIS_DIR / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.06)


def _load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def _hit_segment_mask(times: np.ndarray, metrics: dict) -> np.ndarray:
    audit = metrics["benchmark_metrics"]["p2cccd_swept_ccd_audit"]
    first = int(audit["first_toi_segment"])
    count = int(audit["collision_segments"])
    mask = np.zeros(len(times), dtype=bool)
    if len(times):
        lo = max(0, first)
        hi = min(len(times), first + count + 1)
        mask[lo:hi] = True
    return mask


def build_detection_proxy_points(
    metrics: dict,
    times: np.ndarray,
    physical_times: np.ndarray,
    centers: np.ndarray,
) -> list[dict[str, object]]:
    """Build time-indexed proxy detection points from vehicle-brick contacts.

    The case stores MuJoCo vehicle-brick contact brick IDs by frame, but not
    exact contact manifold coordinates.  We therefore use contacted brick
    centers as reproducible region proxies and label them explicitly.
    """

    ref = metrics["benchmark_metrics"]["mujoco_reference"]
    vehicle_bricks_by_frame = ref["vehicle_bricks_by_frame"]
    vehicle_counts = np.asarray(ref["vehicle_brick_contact_counts_by_frame"], dtype=np.int64)
    total_counts = np.asarray(ref["contact_counts_by_frame"], dtype=np.int64)
    hit_mask = _hit_segment_mask(times, metrics)

    rows: list[dict[str, object]] = []
    for frame_idx, brick_ids in enumerate(vehicle_bricks_by_frame):
        if frame_idx >= len(times):
            break
        for brick_id_raw in brick_ids:
            brick_id = int(brick_id_raw)
            if brick_id < 0 or brick_id >= centers.shape[1]:
                continue
            row, col = divmod(brick_id, 12)
            p = centers[frame_idx, brick_id]
            rows.append(
                {
                    "frame": int(frame_idx),
                    "render_time_s": float(times[frame_idx]),
                    "physical_time_s": float(physical_times[frame_idx]),
                    "brick_id": brick_id,
                    "row": row,
                    "col": col,
                    "x_m": float(p[0]),
                    "y_m": float(p[1]),
                    "z_m": float(p[2]),
                    "vehicle_brick_contacts_this_frame": int(vehicle_counts[frame_idx]) if frame_idx < len(vehicle_counts) else 0,
                    "all_mujoco_contacts_this_frame": int(total_counts[frame_idx]) if frame_idx < len(total_counts) else 0,
                    "in_p2cccd_certified_hit_window": bool(hit_mask[frame_idx]),
                    "point_semantics": "contacted_brick_center_proxy",
                }
            )
    return rows


def write_proxy_csv(rows: list[dict[str, object]]) -> Path:
    path = ANALYSIS_DIR / "car_wall_detection_proxy_points.csv"
    fields = [
        "frame",
        "render_time_s",
        "physical_time_s",
        "brick_id",
        "row",
        "col",
        "x_m",
        "y_m",
        "z_m",
        "vehicle_brick_contacts_this_frame",
        "all_mujoco_contacts_this_frame",
        "in_p2cccd_certified_hit_window",
        "point_semantics",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _arrays(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    if not rows:
        return {k: np.asarray([]) for k in ("t", "physical_t", "brick", "x", "y", "z", "hit", "count")}
    return {
        "t": np.asarray([r["render_time_s"] for r in rows], dtype=np.float64),
        "physical_t": np.asarray([r["physical_time_s"] for r in rows], dtype=np.float64),
        "brick": np.asarray([r["brick_id"] for r in rows], dtype=np.int64),
        "x": np.asarray([r["x_m"] for r in rows], dtype=np.float64),
        "y": np.asarray([r["y_m"] for r in rows], dtype=np.float64),
        "z": np.asarray([r["z_m"] for r in rows], dtype=np.float64),
        "hit": np.asarray([r["in_p2cccd_certified_hit_window"] for r in rows], dtype=bool),
        "count": np.asarray([r["vehicle_brick_contacts_this_frame"] for r in rows], dtype=np.int64),
    }


def plot_detection_scatter(
    rows: list[dict[str, object]],
    times: np.ndarray,
    metrics: dict,
) -> None:
    data = _arrays(rows)
    ref = metrics["benchmark_metrics"]["mujoco_reference"]
    total_counts = np.asarray(ref["contact_counts_by_frame"], dtype=np.float64)
    vehicle_counts = np.asarray(ref["vehicle_brick_contact_counts_by_frame"], dtype=np.float64)
    hit_mask = _hit_segment_mask(times, metrics)
    toi = float(metrics["benchmark_metrics"]["toi_seconds"])

    fig = plt.figure(figsize=(13.8, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 1.0])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_id = fig.add_subplot(gs[0, 1])
    ax_counts = fig.add_subplot(gs[0, 2])

    if len(rows):
        scatter = ax3d.scatter(
            data["x"],
            data["y"],
            data["z"],
            c=data["t"],
            cmap="viridis",
            s=np.clip(16 + 2.5 * data["count"], 18, 120),
            alpha=0.72,
            edgecolors="none",
        )
        ax3d.scatter(
            data["x"][data["hit"]],
            data["y"][data["hit"]],
            data["z"][data["hit"]],
            s=92,
            facecolors="none",
            edgecolors="#dc2626",
            linewidths=1.5,
            label="P2CCCD certified-hit window",
        )
        cb = fig.colorbar(scatter, ax=ax3d, shrink=0.72, pad=0.02)
        cb.set_label("render time (s)")
    ax3d.set_xlabel("x impact direction (m)")
    ax3d.set_ylabel("y wall width (m)")
    ax3d.set_zlabel("z height (m)")
    ax3d.view_init(elev=22, azim=-58)
    ax3d.legend(frameon=False, loc="upper left", fontsize=7)
    ax3d.text2D(
        0.5,
        -0.08,
        "Contact-region proxy points in 3D",
        transform=ax3d.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )

    if len(rows):
        ax_id.scatter(
            data["t"],
            data["brick"],
            c=np.where(data["hit"], "#dc2626", "#2563eb"),
            s=np.clip(12 + 2.2 * data["count"], 16, 95),
            alpha=0.72,
            edgecolors="none",
        )
    ax_id.axvline(toi, color="#dc2626", linestyle="--", linewidth=1.6, label=f"TOI={toi:.3f}s")
    ax_id.set_xlabel("render time (s)")
    ax_id.set_ylabel("brick ID")
    ax_id.legend(frameon=False, loc="upper left")
    ax_id.text(
        0.5,
        -0.12,
        "Contacted brick IDs over time",
        transform=ax_id.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )

    ax_counts.plot(times, total_counts, color="#4c78a8", linewidth=2.0, label="all MuJoCo contacts")
    ax_counts.plot(times, vehicle_counts, color="#f58518", linewidth=2.0, label="vehicle-brick contacts")
    if hit_mask.any():
        ax_counts.fill_between(
            times,
            0,
            np.maximum(total_counts.max(), 1.0),
            where=hit_mask,
            color="#dc2626",
            alpha=0.13,
            step="mid",
            label="P2CCCD certified-hit window",
        )
    ax_counts.axvline(toi, color="#dc2626", linestyle="--", linewidth=1.6)
    ax_counts.set_xlabel("render time (s)")
    ax_counts.set_ylabel("contact count")
    ax_counts.legend(frameon=False, loc="upper right")
    ax_counts.text(
        0.5,
        -0.12,
        "Contact pressure and certified-hit window",
        transform=ax_counts.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )

    _save(fig, "car_wall_detection_proxy_scatter")
    plt.close(fig)


def plot_detection_time_planes(rows: list[dict[str, object]], metrics: dict) -> None:
    data = _arrays(rows)
    toi = float(metrics["benchmark_metrics"]["toi_seconds"])
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)

    if len(rows):
        axes[0].scatter(data["t"], data["x"], c=data["z"], cmap="plasma", s=26, alpha=0.72, edgecolors="none")
        axes[1].scatter(data["t"], data["y"], c=data["z"], cmap="plasma", s=26, alpha=0.72, edgecolors="none")
        sc = axes[2].scatter(data["t"], data["z"], c=data["x"], cmap="viridis", s=26, alpha=0.72, edgecolors="none")
        cb = fig.colorbar(sc, ax=axes[2], shrink=0.85)
        cb.set_label("x (m)")

    labels = [("x impact direction (m)", "(a) x vs time"), ("y lateral (m)", "(b) y vs time"), ("z height (m)", "(c) height vs time")]
    for ax, (ylabel, title) in zip(axes, labels):
        ax.axvline(toi, color="#dc2626", linestyle="--", linewidth=1.4)
        ax.set_xlabel("render time (s)")
        ax.set_ylabel(ylabel)
        ax.text(
            0.5,
            -0.28,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )

    _save(fig, "car_wall_detection_proxy_time_scatter")
    plt.close(fig)


def write_summary(rows: list[dict[str, object]], csv_path: Path) -> dict[str, object]:
    data = _arrays(rows)
    unique_bricks = sorted(set(int(v) for v in data["brick"])) if len(rows) else []
    summary = {
        "source_case": str(CASE_DIR),
        "point_semantics": "MuJoCo vehicle-contacted brick centers used as proxy contact-region detection points",
        "not_exact_ccd_intersection_points": True,
        "proxy_point_count": int(len(rows)),
        "unique_contacted_bricks": int(len(unique_bricks)),
        "first_proxy_time_s": float(data["t"].min()) if len(rows) else None,
        "last_proxy_time_s": float(data["t"].max()) if len(rows) else None,
        "csv": str(csv_path),
        "figures": [
            "car_wall_detection_proxy_scatter.png/pdf/jpg",
            "car_wall_detection_proxy_time_scatter.png/pdf/jpg",
        ],
    }
    (ANALYSIS_DIR / "car_wall_detection_proxy_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def update_reports(summary: dict[str, object]) -> None:
    section = f"""- `car_wall_detection_proxy_scatter.png/pdf/jpg`: 3D scatter of time-colored vehicle-brick contact-region proxy points, contacted brick IDs over time, and contact-count timeline.
- `car_wall_detection_proxy_time_scatter.png/pdf/jpg`: x/y/z proxy contact coordinates against render time.
- `car_wall_detection_proxy_points.csv`: per-frame contacted-brick proxy coordinates with certified-hit-window flags.
- `car_wall_detection_proxy_summary.json`: scatter-plot metadata.

These points are contacted brick-center proxies from the MuJoCo reference replay, not exact CCD primitive intersection coordinates. They are used to visualize where the contact/detection region moves over time. P2CCCD correctness remains governed by the exact swept CCD audit in `metrics.json`.

Proxy points: `{summary['proxy_point_count']}` across `{summary['unique_contacted_bricks']}` contacted bricks, from `t={summary['first_proxy_time_s']:.3f}s` to `t={summary['last_proxy_time_s']:.3f}s`.
"""
    upsert_section(ANALYSIS_DIR / "analysis_report.md", "Detection proxy scatter plots", section)
    upsert_section(CASE_DIR / "case_report.md", "Detection proxy scatter plots", section)
    report = f"""# Car-Wall Detection Proxy Scatter Plots

The car-wall case does not store every exact primitive intersection coordinate. This analysis therefore plots vehicle-contacted brick centers as region-level proxy detection points over time.

## Outputs

- `car_wall_detection_proxy_scatter.png/pdf/jpg`
- `car_wall_detection_proxy_time_scatter.png/pdf/jpg`
- `car_wall_detection_proxy_points.csv`
- `car_wall_detection_proxy_summary.json`

## Counts

- Proxy points: `{summary['proxy_point_count']}`
- Unique contacted bricks: `{summary['unique_contacted_bricks']}`
- Time span: `{summary['first_proxy_time_s']:.6f}` s to `{summary['last_proxy_time_s']:.6f}` s

## Safe interpretation

Use these plots as contact-region visualization. Do not describe them as exact CCD intersection points unless a future runner exports primitive-level intersection coordinates.
"""
    (ANALYSIS_DIR / "detection_proxy_scatter_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    metrics = _load_metrics()
    times, physical_times, centers, _ = rebuild_brick_centers(metrics)
    rows = build_detection_proxy_points(metrics, times, physical_times, centers)
    csv_path = write_proxy_csv(rows)
    plot_detection_scatter(rows, times, metrics)
    plot_detection_time_planes(rows, metrics)
    summary = write_summary(rows, csv_path)
    update_reports(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
