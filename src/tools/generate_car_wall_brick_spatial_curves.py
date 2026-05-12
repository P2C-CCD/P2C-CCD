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


def _make_car_asset() -> aris_cases.MeshAsset:
    path = aris_cases.find_asset("02958343", rank=0, max_bytes=85_000_000)
    vertices, faces, stats = aris_cases.load_mesh_preview(path)
    display_vertices, display_faces, display_method = aris_cases.choose_display_mesh("car", vertices, faces)
    return aris_cases.MeshAsset(
        "ShapeNet car",
        "car",
        path,
        vertices,
        faces,
        stats,
        display_vertices=display_vertices,
        display_faces=display_faces,
        display_shell_method=display_method,
    )


def rebuild_brick_centers(metrics: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Rebuild the exact MuJoCo brick-wall trajectory used by the case.

    The published metrics store contact counts but not every brick center.
    This function reuses the original deterministic case generator and only
    recomputes the brick center paths needed for trajectory plots.
    """

    bm = metrics["benchmark_metrics"]
    frame_count = int(metrics["frame_count"])
    times = np.linspace(0.0, float(metrics["duration_seconds"]), frame_count, dtype=np.float64)
    times[int(np.argmin(np.abs(times - aris_cases.CONTACT_T)))] = aris_cases.CONTACT_T

    car_asset = _make_car_asset()
    _, brick_trajectory, metadata, _, _, physical_times = aris_cases.simulate_mujoco_brick_wall_impact(
        car_asset=car_asset,
        scale_car=4.55,
        times=times,
        output_path=ANALYSIS_DIR / "_regenerated_car_wall_mujoco_bricks_for_trajectory.obj",
        vehicle_mass_kg=float(bm["vehicle_mass_kg"]),
        vehicle_impact_speed_mps=float(bm["vehicle_impact_speed_mps"]),
        vehicle_exit_speed_mps=float(bm["vehicle_exit_speed_mps"]),
        render_speed_scale=float(bm["render_speed_scale"]),
    )

    brick_count = int(metadata["brick_count"])
    centers = brick_trajectory.reshape(len(times), brick_count, 8, 3).mean(axis=2)
    return times, physical_times, centers, metadata


def select_bricks(centers: np.ndarray, count: int = 8) -> np.ndarray:
    displacement = np.linalg.norm(centers[-1] - centers[0], axis=1)
    top = np.argsort(displacement)[::-1]

    selected: list[int] = []
    rows = 13
    cols = 12
    used_rows: set[int] = set()
    for idx in top:
        row = int(idx // cols)
        if row not in used_rows or len(selected) >= rows:
            selected.append(int(idx))
            used_rows.add(row)
        if len(selected) == count:
            break
    if len(selected) < count:
        for idx in top:
            if int(idx) not in selected:
                selected.append(int(idx))
            if len(selected) == count:
                break
    return np.asarray(selected, dtype=np.int64)


def _equalize_3d_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    radius = max(radius, 0.25)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 0.55))
    except Exception:
        pass


def plot_spatial_trajectories(times: np.ndarray, centers: np.ndarray, selected: np.ndarray) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))
    selected_centers = centers[:, selected, :]
    displacement = np.linalg.norm(selected_centers - selected_centers[0:1], axis=2)

    fig = plt.figure(figsize=(10.6, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.0, 1.0])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[1, 1])

    for local_i, brick_id in enumerate(selected):
        curve = centers[:, brick_id, :]
        row, col = divmod(int(brick_id), 12)
        label = f"b{brick_id} (r{row},c{col})"
        color = colors[local_i]
        ax3d.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color, linewidth=2.1, label=label)
        ax3d.scatter(curve[0, 0], curve[0, 1], curve[0, 2], color=color, marker="o", s=34)
        ax3d.scatter(curve[-1, 0], curve[-1, 1], curve[-1, 2], color=color, marker="^", s=48)
        ax_top.plot(curve[:, 0], curve[:, 1], color=color, linewidth=2.0, label=label)
        ax_top.scatter([curve[0, 0]], [curve[0, 1]], color=color, marker="o", s=26)
        ax_top.scatter([curve[-1, 0]], [curve[-1, 1]], color=color, marker="^", s=36)
        ax_side.plot(times, displacement[:, local_i], color=color, linewidth=2.0, label=label)

    wall_x0 = float(np.median(centers[0, :, 0]))
    y0, y1 = float(centers[0, :, 1].min()), float(centers[0, :, 1].max())
    z0, z1 = float(centers[0, :, 2].min()), float(centers[0, :, 2].max())
    yy, zz = np.meshgrid(np.linspace(y0, y1, 2), np.linspace(z0, z1, 2))
    xx = np.full_like(yy, wall_x0)
    ax3d.plot_surface(xx, yy, zz, color="#b45309", alpha=0.12, linewidth=0, shade=False)

    _equalize_3d_axes(ax3d, selected_centers.reshape(-1, 3))
    ax3d.view_init(elev=22, azim=-58)
    ax3d.set_title("3D brick center trajectories")
    ax3d.set_xlabel("x: impact direction (m)")
    ax3d.set_ylabel("y: wall width (m)")
    ax3d.set_zlabel("z: height (m)")
    ax3d.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=7)

    ax_top.axvline(wall_x0, color="#92400e", linestyle="--", linewidth=1.4, label="initial wall plane")
    ax_top.set_title("Top view: impact direction vs lateral spread")
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.set_aspect("equal", adjustable="box")

    ax_side.axvline(float(times[np.argmin(np.abs(times - aris_cases.CONTACT_T))]), color="#d62728", linestyle="--", linewidth=1.5)
    ax_side.set_title("Selected brick displacement over time")
    ax_side.set_xlabel("render time (s)")
    ax_side.set_ylabel("displacement from initial center (m)")

    fig.tight_layout()
    _save(fig, "car_wall_selected_brick_spatial_trajectories")
    plt.close(fig)


def plot_projection_grid(times: np.ndarray, centers: np.ndarray, selected: np.ndarray) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), constrained_layout=True)

    for local_i, brick_id in enumerate(selected):
        curve = centers[:, brick_id, :]
        color = colors[local_i]
        label = f"b{int(brick_id)}"
        axes[0].plot(curve[:, 0], curve[:, 1], color=color, linewidth=2.0, label=label)
        axes[0].scatter(curve[0, 0], curve[0, 1], color=color, marker="o", s=24)
        axes[0].scatter(curve[-1, 0], curve[-1, 1], color=color, marker="^", s=34)
        axes[1].plot(curve[:, 0], curve[:, 2], color=color, linewidth=2.0)
        axes[1].scatter(curve[0, 0], curve[0, 2], color=color, marker="o", s=24)
        axes[1].scatter(curve[-1, 0], curve[-1, 2], color=color, marker="^", s=34)
        axes[2].plot(times, curve[:, 2], color=color, linewidth=2.0)

    axes[0].set_title("XY projection")
    axes[0].set_xlabel("x impact direction (m)")
    axes[0].set_ylabel("y lateral (m)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(frameon=False, ncol=2, fontsize=7)

    axes[1].set_title("XZ projection")
    axes[1].set_xlabel("x impact direction (m)")
    axes[1].set_ylabel("z height (m)")

    axes[2].set_title("Height curves")
    axes[2].set_xlabel("render time (s)")
    axes[2].set_ylabel("z height (m)")
    axes[2].axvline(float(times[np.argmin(np.abs(times - aris_cases.CONTACT_T))]), color="#d62728", linestyle="--", linewidth=1.5)

    _save(fig, "car_wall_selected_brick_projection_curves")
    plt.close(fig)


def write_tables(
    times: np.ndarray,
    physical_times: np.ndarray,
    centers: np.ndarray,
    selected: np.ndarray,
    metadata: dict[str, object],
) -> dict[str, object]:
    dt_render = np.diff(times)
    dt_physical = np.diff(physical_times)
    velocity_render = np.zeros_like(centers)
    velocity_physical = np.zeros_like(centers)
    if len(times) > 1:
        velocity_render[1:] = np.diff(centers, axis=0) / np.maximum(dt_render[:, None, None], 1.0e-8)
        velocity_physical[1:] = np.diff(centers, axis=0) / np.maximum(dt_physical[:, None, None], 1.0e-8)
        velocity_physical[np.asarray(physical_times) <= 0.0] = 0.0

    csv_path = ANALYSIS_DIR / "car_wall_selected_brick_trajectories.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "render_time_s",
                "physical_time_s",
                "brick_id",
                "row",
                "col",
                "x_m",
                "y_m",
                "z_m",
                "displacement_m",
                "render_speed_mps",
                "physical_speed_mps",
            ]
        )
        for frame_idx, (render_t, physical_t) in enumerate(zip(times, physical_times)):
            for brick_id in selected:
                row, col = divmod(int(brick_id), 12)
                p = centers[frame_idx, brick_id]
                disp = float(np.linalg.norm(p - centers[0, brick_id]))
                v_render = float(np.linalg.norm(velocity_render[frame_idx, brick_id]))
                v_physical = float(np.linalg.norm(velocity_physical[frame_idx, brick_id]))
                writer.writerow(
                    [
                        frame_idx,
                        f"{float(render_t):.9f}",
                        f"{float(physical_t):.9f}",
                        int(brick_id),
                        row,
                        col,
                        f"{float(p[0]):.9f}",
                        f"{float(p[1]):.9f}",
                        f"{float(p[2]):.9f}",
                        f"{disp:.9f}",
                        f"{v_render:.9f}",
                        f"{v_physical:.9f}",
                    ]
                )

    summary_rows = []
    for brick_id in selected:
        row, col = divmod(int(brick_id), 12)
        displacement_series = np.linalg.norm(centers[:, brick_id] - centers[0, brick_id], axis=1)
        physical_speed_series = np.linalg.norm(velocity_physical[:, brick_id], axis=1)
        summary_rows.append(
            {
                "brick_id": int(brick_id),
                "row": row,
                "col": col,
                "initial_center_m": [float(v) for v in centers[0, brick_id]],
                "final_center_m": [float(v) for v in centers[-1, brick_id]],
                "max_displacement_m": float(displacement_series.max()),
                "max_physical_speed_mps": float(physical_speed_series.max()),
            }
        )

    summary = {
        "source_case": str(CASE_DIR),
        "source_generator": "src/tools/render_aris_real_mesh_physics_cases.py::simulate_mujoco_brick_wall_impact",
        "brick_count": int(metadata["brick_count"]),
        "selected_brick_count": int(len(selected)),
        "selected_bricks": summary_rows,
        "global_max_displacement_m": float(np.linalg.norm(centers[-1] - centers[0], axis=1).max()),
        "global_max_height_m": float(centers[:, :, 2].max()),
        "csv": str(csv_path),
        "figures": [
            "car_wall_selected_brick_spatial_trajectories.png/pdf/jpg",
            "car_wall_selected_brick_projection_curves.png/pdf/jpg",
        ],
    }
    (ANALYSIS_DIR / "car_wall_selected_brick_trajectory_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def upsert_section(path: Path, heading: str, content: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = f"## {heading}"
    if marker in old:
        prefix, rest = old.split(marker, 1)
        next_idx = rest.find("\n## ")
        if next_idx >= 0:
            suffix = rest[next_idx:]
        else:
            suffix = ""
        new_text = prefix.rstrip() + "\n\n" + marker + "\n\n" + content.strip() + "\n" + suffix
    else:
        new_text = old.rstrip() + "\n\n" + marker + "\n\n" + content.strip() + "\n"
    path.write_text(new_text, encoding="utf-8")


def update_reports(summary: dict[str, object]) -> None:
    selected = summary["selected_bricks"]
    assert isinstance(selected, list)
    top = selected[0]
    section = f"""- `car_wall_selected_brick_spatial_trajectories.png/pdf/jpg`: 8 selected high-displacement brick-center trajectories in 3D, with start/end markers and initial wall plane.
- `car_wall_selected_brick_projection_curves.png/pdf/jpg`: XY projection, XZ projection, and height-time curves for the same bricks.
- `car_wall_selected_brick_trajectories.csv`: per-frame center position, displacement, and speed for selected bricks.
- `car_wall_selected_brick_trajectory_summary.json`: trajectory metadata and selected-brick statistics.

The selected bricks are chosen by largest rigid-body displacement while preserving row diversity. The most displaced selected brick is `b{top['brick_id']}` at row `{top['row']}`, column `{top['col']}`, with max displacement `{top['max_displacement_m']:.3f}` m and max physical speed `{top['max_physical_speed_mps']:.3f}` m/s.
"""
    upsert_section(ANALYSIS_DIR / "analysis_report.md", "Brick spatial trajectory curves", section)
    upsert_section(CASE_DIR / "case_report.md", "Brick spatial trajectory curves", section)

    report = f"""# Car-Wall Brick Spatial Trajectory Curves

This report records the spatial center curves for selected bricks in the car-wall impact case.

## Source

- Case: `{CASE_DIR}`
- Generator: `render_aris_real_mesh_physics_cases.py::simulate_mujoco_brick_wall_impact`
- Brick count: `{summary['brick_count']}`
- Selected brick count: `{summary['selected_brick_count']}`

## Outputs

- `car_wall_selected_brick_spatial_trajectories.png/pdf/jpg`
- `car_wall_selected_brick_projection_curves.png/pdf/jpg`
- `car_wall_selected_brick_trajectories.csv`
- `car_wall_selected_brick_trajectory_summary.json`

## Interpretation

The curves are rigid brick center trajectories from the deterministic MuJoCo replay used by the case. They are not contact classifier outputs. P2CCCD correctness remains reported by the swept CCD audit in `metrics.json`.
"""
    (ANALYSIS_DIR / "brick_spatial_trajectory_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    metrics = _load_metrics()
    times, physical_times, centers, metadata = rebuild_brick_centers(metrics)
    selected = select_bricks(centers, count=8)
    plot_spatial_trajectories(times, centers, selected)
    plot_projection_grid(times, centers, selected)
    summary = write_tables(times, physical_times, centers, selected, metadata)
    update_reports(summary)
    print(json.dumps({"selected": [int(v) for v in selected], "analysis_dir": str(ANALYSIS_DIR)}, indent=2))


if __name__ == "__main__":
    main()
