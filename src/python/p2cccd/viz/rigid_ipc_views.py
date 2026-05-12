from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

from p2cccd.datasets.ccd import DatasetQueryBatch, RigidIPCScene
from p2cccd.datasets.ccd.contracts import Vec3


@dataclass(frozen=True, slots=True)
class RigidIPCVisualizationSummary:
    scene_name: str
    body_count: int
    moving_body_count: int
    mesh_body_count: int
    query_count: int
    bounds_min: Vec3
    bounds_max: Vec3


def summarize_rigid_ipc_visualization(
    scene: RigidIPCScene,
    batch: DatasetQueryBatch | None = None,
) -> RigidIPCVisualizationSummary:
    if not scene.bodies:
        raise ValueError("Rigid-IPC scene must contain at least one body")
    bounds_min = tuple(min(body.position[index] - body.radius for body in scene.bodies) for index in range(3))
    bounds_max = tuple(max(body.position[index] + body.radius for body in scene.bodies) for index in range(3))
    return RigidIPCVisualizationSummary(
        scene_name=scene.scene_name,
        body_count=scene.body_count,
        moving_body_count=scene.moving_body_count,
        mesh_body_count=sum(1 for body in scene.bodies if body.mesh is not None),
        query_count=0 if batch is None else batch.query_count,
        bounds_min=bounds_min,  # type: ignore[arg-type]
        bounds_max=bounds_max,  # type: ignore[arg-type]
    )


def _norm(vec: Vec3) -> float:
    return (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]) ** 0.5


def _body_pair_edges(batch: DatasetQueryBatch | None) -> tuple[tuple[int, int], ...]:
    if batch is None:
        return ()
    edges: list[tuple[int, int]] = []
    for query in batch.queries:
        if query.box_pair is None:
            continue
        edges.append((int(query.box_pair[0]), int(query.box_pair[1])))
    return tuple(edges)


def render_rigid_ipc_scene_svg(
    scene: RigidIPCScene,
    batch: DatasetQueryBatch | None = None,
    *,
    width: int = 980,
    height: int = 680,
) -> str:
    if width <= 0 or height <= 0:
        raise ValueError("SVG dimensions must be positive")
    summary = summarize_rigid_ipc_visualization(scene, batch)
    margin = 72.0
    span_x = max(1.0e-9, summary.bounds_max[0] - summary.bounds_min[0])
    span_y = max(1.0e-9, summary.bounds_max[1] - summary.bounds_min[1])
    scale = min((width - 2.0 * margin) / span_x, (height - 2.0 * margin) / span_y)
    scale = max(1.0e-9, scale)

    def project(point: Vec3) -> tuple[float, float]:
        x = margin + (point[0] - summary.bounds_min[0]) * scale
        y = height - margin - (point[1] - summary.bounds_min[1]) * scale
        return x, y

    body_by_id = {body.body_id: body for body in scene.bodies}
    rows: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(scene.scene_name)} Rigid-IPC scene">',
        "<defs>",
        '<marker id="rigid-ipc-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#bd5a2a"/>',
        "</marker>",
        "</defs>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f5efe3"/>',
        f'<text x="28" y="36" font-size="24" font-family="Georgia,serif" fill="#1d2a22">{escape(scene.scene_name)}</text>',
        (
            f'<text x="28" y="62" font-size="13" font-family="Consolas,monospace" fill="#526156">'
            f"bodies={summary.body_count}, moving={summary.moving_body_count}, "
            f"mesh_bodies={summary.mesh_body_count}, query_pairs={summary.query_count}</text>"
        ),
    ]

    for body_a_id, body_b_id in _body_pair_edges(batch):
        body_a = body_by_id.get(body_a_id)
        body_b = body_by_id.get(body_b_id)
        if body_a is None or body_b is None:
            continue
        ax, ay = project(body_a.position)
        bx, by = project(body_b.position)
        rows.append(
            f'<line x1="{ax:.2f}" y1="{ay:.2f}" x2="{bx:.2f}" y2="{by:.2f}" '
            'stroke="#345b82" stroke-width="2.5" stroke-dasharray="8 7" opacity="0.68"/>'
        )

    for body in scene.bodies:
        cx, cy = project(body.position)
        radius = max(7.0, min(44.0, body.radius * scale))
        velocity_end = (
            body.position[0] + body.linear_velocity[0] * scene.timestep,
            body.position[1] + body.linear_velocity[1] * scene.timestep,
            body.position[2] + body.linear_velocity[2] * scene.timestep,
        )
        vx, vy = project(velocity_end)
        moving = _norm(body.linear_velocity) > 0.0 or not body.is_fixed
        fill = "#d96f32" if moving else "#5f7666"
        stroke = "#522b17" if moving else "#26352d"
        rows.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2.2" opacity="0.88"/>'
        )
        if _norm(body.linear_velocity) > 0.0:
            rows.append(
                f'<line x1="{cx:.2f}" y1="{cy:.2f}" x2="{vx:.2f}" y2="{vy:.2f}" '
                'stroke="#bd5a2a" stroke-width="3.0" marker-end="url(#rigid-ipc-arrow)"/>'
            )
        label = f"#{body.body_id} {body.mesh or body.body_type}"
        rows.append(
            f'<text x="{cx + radius + 5.0:.2f}" y="{cy + 4.0:.2f}" '
            f'font-size="12" font-family="Consolas,monospace" fill="#17231b">{escape(label)}</text>'
        )

    legend_y = height - 32
    rows.extend(
        (
            f'<circle cx="28" cy="{legend_y}" r="8" fill="#d96f32"/>',
            f'<text x="44" y="{legend_y + 5}" font-size="12" font-family="Consolas,monospace">moving/dynamic body</text>',
            f'<circle cx="210" cy="{legend_y}" r="8" fill="#5f7666"/>',
            f'<text x="226" y="{legend_y + 5}" font-size="12" font-family="Consolas,monospace">fixed/static body</text>',
            f'<line x1="390" y1="{legend_y}" x2="450" y2="{legend_y}" stroke="#345b82" stroke-width="2.5" stroke-dasharray="8 7"/>',
            f'<text x="462" y="{legend_y + 5}" font-size="12" font-family="Consolas,monospace">generated body-pair query</text>',
            "</svg>",
        )
    )
    return "\n".join(rows)


def write_rigid_ipc_scene_debug_html(
    path: str | Path,
    *,
    scene: RigidIPCScene,
    batch: DatasetQueryBatch | None = None,
    title: str = "Rigid-IPC Scene Debug View",
) -> Path:
    summary = summarize_rigid_ipc_visualization(scene, batch)
    svg = render_rigid_ipc_scene_svg(scene, batch)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            (
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\">",
                f"<title>{escape(title)}</title>",
                "<style>",
                "body{margin:32px;background:#efe7d7;color:#18251c;font-family:Georgia,serif}",
                "main{max-width:1100px;margin:auto}",
                "section{background:#fffaf0;border:1px solid #d4c2a3;border-radius:20px;padding:20px;margin:18px 0}",
                "svg{width:100%;height:auto}",
                "code{font-family:Consolas,monospace}",
                "</style></head><body><main>",
                f"<h1>{escape(title)}</h1>",
                f"<p><code>scene={escape(summary.scene_name)}</code></p>",
                (
                    "<p>"
                    f"bodies={summary.body_count}, moving={summary.moving_body_count}, "
                    f"mesh_bodies={summary.mesh_body_count}, query_pairs={summary.query_count}"
                    "</p>"
                ),
                f"<section>{svg}</section>",
                "</main></body></html>",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return output
