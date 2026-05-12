from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

import standard_graphics_model_collision_suite as suite


CASE_DIR = (
    suite.ROOT / "src"
    / "MyDemo"
    / suite.RUN_TAG
    / "classic_models_cornell_room_drop"
)
FONT_SCALE = 2.0
AXIS_TITLE_SCALE = 0.9
FIG3_STYLE_HEIGHT_PNG_ONLY = os.environ.get("P2CCCD_HEIGHT_CURVES_FIG3_STYLE_ONLY", "0") == "1"


def fs(value: float) -> float:
    return value * FONT_SCALE


def axis_fs(value: float) -> float:
    return value * FONT_SCALE * AXIS_TITLE_SCALE


def short_body_name(name: str) -> str:
    replacements = {
        "Stanford Bunny": "Bunny",
        "Stanford Dragon": "Dragon",
        "Spot the Cow": "Spot",
        "Utah Teapot": "Teapot",
    }
    return replacements.get(name, name)


def reconstruct_classic_drop_bodies():
    specs = suite.standard_model_specs()
    suite.validate_inputs(specs)
    rng = np.random.default_rng(suite.SEED)
    assets = {
        spec.key: suite.load_standard_mesh_asset(
            spec.name,
            "classic_dynamic_mesh",
            spec.path,
            up_axis="y",
            max_collision_faces=24_000,
            max_display_faces=12_000,
        )
        for spec in specs
    }
    physical_props = suite.physical_properties_for_assets(assets, specs)
    object_specs = [
        (assets[spec.key], spec.color, physical_props[spec.key].mass_kg, spec.scale, spec.xy)
        for spec in specs
    ]
    ground = suite.generated_dense_ground_asset(
        "Dense Cornell-room collision floor",
        "ground",
        suite.GENERATED_ASSET_ROOT / "classic_models_dense_floor.obj",
        (8.8, 6.4, 0.08),
        nx=128,
        ny=96,
    )
    bodies, metrics, _ = suite.make_standard_drop_bodies(object_specs, ground, rng)
    props_by_name = {prop.model_name: prop for prop in physical_props.values()}
    suite.attach_physical_metadata_to_bodies(bodies, props_by_name)
    dynamic_bodies = [body for body in bodies if body.asset.category == "classic_dynamic_mesh"]
    return dynamic_bodies, metrics


def write_displacement_csv(bodies, csv_path: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "case": "classic_models_cornell_room_drop",
        "definition": "displacement_m = ||center_of-mass_position(t) - center_of-mass_position(0)|| in scene meters",
        "body_count": len(bodies),
        "bodies": [],
    }
    for body in bodies:
        if body.trajectory_times is None or body.trajectory_positions is None:
            continue
        times = np.asarray(body.trajectory_times, dtype=np.float64)
        positions = np.asarray(body.trajectory_positions, dtype=np.float64)
        velocities = (
            np.asarray(body.trajectory_velocities, dtype=np.float64)
            if body.trajectory_velocities is not None
            else np.gradient(positions, times, axis=0)
        )
        p0 = positions[0]
        delta = positions - p0
        displacement = np.linalg.norm(delta, axis=1)
        horizontal = np.linalg.norm(delta[:, :2], axis=1)
        vertical = delta[:, 2]
        bottom = positions[:, 2] + float(suite.local_vertices(body.asset, body.scale, body.yaw)[:, 2].min())
        speed = np.linalg.norm(velocities, axis=1)
        metadata = body.metadata or {}
        material = str(metadata.get("material", "unknown"))
        density = float(metadata.get("density_kg_m3", 0.0))
        mass = float(metadata.get("mass_kg", body.mass))
        first_contact = metadata.get("first_ground_contact_time")
        body_summary = {
            "name": body.asset.name,
            "material": material,
            "density_kg_m3": density,
            "mass_kg": mass,
            "max_displacement_m": float(np.max(displacement)),
            "final_displacement_m": float(displacement[-1]),
            "initial_bottom_height_m": float(bottom[0]),
            "min_bottom_height_m": float(np.min(bottom)),
            "final_bottom_height_m": float(bottom[-1]),
            "max_speed_mps": float(np.max(speed)),
            "first_ground_contact_time_s": None if first_contact is None else float(first_contact),
        }
        summary["bodies"].append(body_summary)
        for i, t in enumerate(times):
            rows.append(
                {
                    "time_s": float(t),
                    "body": body.asset.name,
                    "material": material,
                    "density_kg_m3": density,
                    "mass_kg": mass,
                    "x_m": float(positions[i, 0]),
                    "y_m": float(positions[i, 1]),
                    "z_m": float(positions[i, 2]),
                    "displacement_m": float(displacement[i]),
                    "horizontal_displacement_m": float(horizontal[i]),
                    "vertical_displacement_m": float(vertical[i]),
                    "bottom_height_m": float(bottom[i]),
                    "speed_mps": float(speed[i]),
                }
            )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def plot_height_curves(
    bodies,
    output_stems: list[Path],
    benchmark_metrics: dict[str, object],
    *,
    save_pdf: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.3,
            "ytick.major.width": 1.3,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax, info_ax) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.15),
        gridspec_kw={"width_ratios": [4.5, 1.28], "wspace": 0.24},
    )
    for body in bodies:
        if body.trajectory_times is None or body.trajectory_positions is None:
            continue
        times = np.asarray(body.trajectory_times, dtype=np.float64)
        positions = np.asarray(body.trajectory_positions, dtype=np.float64)
        bottom = positions[:, 2] + float(suite.local_vertices(body.asset, body.scale, body.yaw)[:, 2].min())
        color = tuple(channel / 255.0 for channel in body.color)
        ax.plot(times, bottom, label=short_body_name(body.asset.name), color=color, linewidth=1.8)
        first_contact = (body.metadata or {}).get("first_ground_contact_time")
        if first_contact is not None:
            fc = float(first_contact)
            ax.scatter([fc], [float(np.interp(fc, times, bottom))], s=26, color=color, zorder=4)
    ax.axhline(0.0, color="#444444", linewidth=1.25, linestyle="--", label="floor")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Height-time", fontweight="normal", pad=4)
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(
        ncol=2,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        handlelength=1.8,
        handletextpad=0.35,
        columnspacing=0.70,
        borderpad=0.20,
        labelspacing=0.20,
    )

    dense_pairs = float(benchmark_metrics.get("dense_no_proposal_object_ground_pair_budget", 0.0))
    rtstpf_calls = float(benchmark_metrics.get("rtstpf_exact_call_budget", 0.0))
    reduction = 0.0 if dense_pairs <= 0.0 else max(0.0, 1.0 - rtstpf_calls / dense_pairs)
    factor = 0.0 if rtstpf_calls <= 0.0 else dense_pairs / rtstpf_calls
    first_contact = benchmark_metrics.get("first_ground_contact_time", None)
    last_contact = benchmark_metrics.get("last_first_ground_contact_time", None)

    values = np.asarray([dense_pairs, rtstpf_calls], dtype=np.float64)
    labels = ["No\nProposal", "P2C\nCCD"]
    colors = ["#9aa4af", "#1a7f37"]
    bars = info_ax.bar(labels, values, color=colors, width=0.62)
    info_ax.set_yscale("log")
    info_ax.set_title("Exact checks", fontweight="normal", pad=4)
    info_ax.set_ylabel("Count (log)", labelpad=8)
    info_ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    info_ax.spines["top"].set_visible(False)
    info_ax.spines["right"].set_visible(False)
    for bar, text in zip(bars, [f"{dense_pairs / 1.0e9:.2f}B", f"{rtstpf_calls / 1.0e3:.0f}K"]):
        is_large = bar.get_height() > 1.0e8
        info_ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() * (0.72 if is_large else 1.35),
            text,
            ha="center",
            va="top" if is_large else "bottom",
            fontsize=10,
            color="#20262e",
            fontweight="bold",
        )
    info_ax.text(
        0.52,
        0.72,
        f"{int(reduction * 100.0)}%\nless",
        transform=info_ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#1a7f37",
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.20, top=0.88, wspace=0.24)
    for output_stem in output_stems:
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_stem.with_suffix(".png"))
        if save_pdf:
            fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    bodies, metrics = reconstruct_classic_drop_bodies()
    if FIG3_STYLE_HEIGHT_PNG_ONLY:
        plot_height_curves(
            bodies,
            [CASE_DIR / "height_time_curves"],
            metrics,
            save_pdf=False,
        )
        print(f"wrote {CASE_DIR / 'height_time_curves.png'}")
        return

    csv_path = CASE_DIR / "displacement_time_curves.csv"
    summary = write_displacement_csv(bodies, csv_path)
    summary["source_metrics"] = metrics
    summary_path = CASE_DIR / "displacement_time_curves_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_height_curves(
        bodies,
        [
            CASE_DIR / "height_time_curves",
            CASE_DIR / "displacement_time_curves",
        ],
        metrics,
    )
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {CASE_DIR / 'height_time_curves.png'}")
    print(f"wrote {CASE_DIR / 'height_time_curves.pdf'}")


if __name__ == "__main__":
    main()
