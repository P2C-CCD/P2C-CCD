from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = ROOT / "src" / "benchmark"
DEMO_ROOT = ROOT / "src" / "MyDemo"
TI_DATA_ROOT = ROOT / "src" / "baseline" / "datasets" / "continuous-collision-detection"


def find_tool(env_var: str, executable: str) -> str:
    configured = os.environ.get(env_var)
    if configured:
        return configured
    discovered = shutil.which(executable)
    return discovered or executable


FFMPEG = find_tool("P2CCCD_FFMPEG", "ffmpeg")


def run_ffmpeg(args: list[str]) -> None:
    completed = subprocess.run(args, check=False, capture_output=True)
    if completed.returncode != 0:
        stdout = completed.stdout.decode("utf-8", errors="replace")
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")


def encode_frames(frame_dir: Path, output_mp4: Path, fps: int = 24) -> None:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    temp_mp4 = output_mp4.with_suffix(".h264.tmp.mp4")
    if temp_mp4.exists():
        temp_mp4.unlink()
    run_ffmpeg(
        [
            FFMPEG,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temp_mp4),
        ]
    )
    temp_mp4.replace(output_mp4)


def transcode_to_h264(mp4_path: Path) -> Path:
    if not mp4_path.exists():
        raise FileNotFoundError(mp4_path)
    legacy = mp4_path.with_name(mp4_path.stem + "_legacy_mp4v.mp4")
    if not legacy.exists():
        shutil.copy2(mp4_path, legacy)
    temp_mp4 = mp4_path.with_suffix(".h264.tmp.mp4")
    if temp_mp4.exists():
        temp_mp4.unlink()
    run_ffmpeg(
        [
            FFMPEG,
            "-y",
            "-i",
            str(legacy),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(temp_mp4),
        ]
    )
    temp_mp4.replace(mp4_path)
    return legacy


def safe_font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "segoeui.ttf", "calibri.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def parse_rational_vertex(row: str) -> tuple[tuple[float, float, float], bool]:
    x_num, x_den, y_num, y_den, z_num, z_den, truth = row.strip().split(",")
    return (
        (
            float(x_num) / float(x_den),
            float(y_num) / float(y_den),
            float(z_num) / float(z_den),
        ),
        truth.strip() not in {"0", "false", "False", ""},
    )


def load_ti_query_blocks(targets: dict[str, set[int]]) -> dict[tuple[str, int], dict[str, object]]:
    loaded: dict[tuple[str, int], dict[str, object]] = {}
    for rel_csv, query_indices in targets.items():
        csv_path = TI_DATA_ROOT / rel_csv
        if not csv_path.exists():
            continue
        wanted = set(query_indices)
        max_query = max(wanted)
        current_query: list[str] = []
        with csv_path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                query_index = line_index // 8
                if query_index > max_query:
                    break
                if query_index in wanted:
                    current_query.append(line)
                    if len(current_query) == 8:
                        vertices: list[tuple[float, float, float]] = []
                        labels: list[bool] = []
                        for raw in current_query:
                            vertex, truth = parse_rational_vertex(raw)
                            vertices.append(vertex)
                            labels.append(truth)
                        loaded[(rel_csv, query_index)] = {
                            "vertices": vertices,
                            "truth": any(labels),
                        }
                        current_query = []
                elif current_query:
                    current_query = []
    return loaded


def collect_representative_ti_queries(schedule_csv: Path, groups: int = 12) -> list[dict[str, object]]:
    rows: list[dict[str, str]] = []
    with schedule_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["group_id"]) > groups:
                break
            rows.append(row)

    targets: dict[str, set[int]] = {}
    for row in rows:
        targets.setdefault(row["csv_path"], set()).add(int(row["query_index"]))
    blocks = load_ti_query_blocks(targets)

    queries: list[dict[str, object]] = []
    for row in rows:
        query_index = int(row["query_index"])
        block = blocks.get((row["csv_path"], query_index))
        if block is None:
            continue
        vertices = block["vertices"]
        assert isinstance(vertices, list)
        query = {
            "group_id": int(row["group_id"]),
            "case": row["case"],
            "kind": row["kind"],
            "csv_path": row["csv_path"],
            "query_index": query_index,
            "score": float(row["score"]),
            "truth": bool(block["truth"]),
            "vertices": vertices,
        }
        query["toi_proxy"] = estimate_toi_proxy(query)
        queries.append(query)

    positives = [q for q in queries if q["truth"]]
    negatives = [q for q in queries if not q["truth"]]
    # Keep all positives from the early groups and only the most confident negatives.
    selected = positives[:8] + negatives[:36]
    if not positives and queries:
        selected = queries[:44]
    return selected


def vec_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_mul(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def vec_dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(max(0.0, vec_dot(a, a)))


def interp_vertex(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return (a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t, a[2] * (1.0 - t) + b[2] * t)


def interpolate_query_vertices(query: dict[str, object], t: float) -> list[tuple[float, float, float]]:
    vertices = query["vertices"]
    assert isinstance(vertices, list)
    if query["kind"] == "edge-edge":
        return [
            interp_vertex(vertices[0], vertices[4], t),
            interp_vertex(vertices[1], vertices[5], t),
            interp_vertex(vertices[2], vertices[6], t),
            interp_vertex(vertices[3], vertices[7], t),
        ]
    return [
        interp_vertex(vertices[0], vertices[4], t),
        interp_vertex(vertices[1], vertices[5], t),
        interp_vertex(vertices[2], vertices[6], t),
        interp_vertex(vertices[3], vertices[7], t),
    ]


def segment_segment_distance(
    p1: tuple[float, float, float],
    q1: tuple[float, float, float],
    p2: tuple[float, float, float],
    q2: tuple[float, float, float],
) -> float:
    # Ericson-style closest points between segments. Robust enough for visualization and TOI proxying.
    d1 = vec_sub(q1, p1)
    d2 = vec_sub(q2, p2)
    r = vec_sub(p1, p2)
    a = vec_dot(d1, d1)
    e = vec_dot(d2, d2)
    f = vec_dot(d2, r)
    eps = 1e-12
    if a <= eps and e <= eps:
        return vec_norm(vec_sub(p1, p2))
    if a <= eps:
        s = 0.0
        t = max(0.0, min(1.0, f / e))
    else:
        c = vec_dot(d1, r)
        if e <= eps:
            t = 0.0
            s = max(0.0, min(1.0, -c / a))
        else:
            b = vec_dot(d1, d2)
            denom = a * e - b * b
            s = max(0.0, min(1.0, (b * f - c * e) / denom)) if denom != 0.0 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))
    c1 = vec_add(p1, vec_mul(d1, s))
    c2 = vec_add(p2, vec_mul(d2, t))
    return vec_norm(vec_sub(c1, c2))


def point_triangle_distance(
    p: tuple[float, float, float],
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> float:
    # Conservative visualization proxy: minimum to triangle vertices/edges and projected plane if inside.
    ab = vec_sub(b, a)
    ac = vec_sub(c, a)
    ap = vec_sub(p, a)
    n = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    n_len = vec_norm(n)
    candidates = [
        vec_norm(vec_sub(p, a)),
        vec_norm(vec_sub(p, b)),
        vec_norm(vec_sub(p, c)),
        segment_segment_distance(p, p, a, b),
        segment_segment_distance(p, p, b, c),
        segment_segment_distance(p, p, c, a),
    ]
    if n_len > 1e-12:
        signed = vec_dot(ap, n) / n_len
        proj = vec_sub(p, vec_mul(n, signed / n_len))
        v0 = ac
        v1 = ab
        v2 = vec_sub(proj, a)
        dot00 = vec_dot(v0, v0)
        dot01 = vec_dot(v0, v1)
        dot02 = vec_dot(v0, v2)
        dot11 = vec_dot(v1, v1)
        dot12 = vec_dot(v1, v2)
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) > 1e-12:
            inv = 1.0 / denom
            u = (dot11 * dot02 - dot01 * dot12) * inv
            v = (dot00 * dot12 - dot01 * dot02) * inv
            if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
                candidates.append(abs(signed))
    return min(candidates)


def query_distance_proxy(query: dict[str, object], t: float) -> float:
    vertices = interpolate_query_vertices(query, t)
    if query["kind"] == "edge-edge":
        return segment_segment_distance(vertices[0], vertices[1], vertices[2], vertices[3])
    return point_triangle_distance(vertices[0], vertices[1], vertices[2], vertices[3])


def estimate_toi_proxy(query: dict[str, object]) -> float:
    samples = 81
    best_t, best_d = 0.0, float("inf")
    for i in range(samples):
        t = i / (samples - 1)
        d = query_distance_proxy(query, t)
        if d < best_d:
            best_t, best_d = t, d
    return best_t


def projection_basis(point: tuple[float, float, float], center: tuple[float, float, float], azimuth: float, elevation: float) -> tuple[float, float]:
    x, y, z = vec_sub(point, center)
    ca, sa = math.cos(azimuth), math.sin(azimuth)
    ce, se = math.cos(elevation), math.sin(elevation)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    yv = ce * yr - se * z
    return xr, yv


def compute_scene_frame(queries: list[dict[str, object]]) -> tuple[tuple[float, float, float], float]:
    points: list[tuple[float, float, float]] = []
    for query in queries:
        vertices = query["vertices"]
        assert isinstance(vertices, list)
        points.extend(vertices)
    center = tuple(sum(p[i] for p in points) / max(1, len(points)) for i in range(3))
    radius = max(vec_norm(vec_sub(p, center)) for p in points) if points else 1.0
    return center, max(radius, 1e-6)


def project_to_rect(
    point: tuple[float, float, float],
    center: tuple[float, float, float],
    radius: float,
    rect: tuple[int, int, int, int],
    azimuth: float = -0.70,
    elevation: float = 0.55,
) -> tuple[float, float]:
    x0, y0, x1, y1 = rect
    sx, sy = projection_basis(point, center, azimuth, elevation)
    scale = 0.42 * min(x1 - x0, y1 - y0) / radius
    return ((x0 + x1) * 0.5 + sx * scale, (y0 + y1) * 0.5 - sy * scale)


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    pts: list[tuple[float, float]],
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width, joint="curve")


def draw_query_geometry(
    draw: ImageDraw.ImageDraw,
    query: dict[str, object],
    t: float,
    rect: tuple[int, int, int, int],
    center: tuple[float, float, float],
    radius: float,
    primary: bool = False,
) -> None:
    truth = bool(query["truth"])
    toi = float(query.get("toi_proxy", 0.5))
    emphasis = max(0.0, 1.0 - abs(t - toi) / 0.12) if truth else 0.0
    alpha = 235 if primary or truth else 58
    width = 8 if primary else (5 if truth else 2)
    blue = (27, 133, 255, alpha)
    coral = (239, 93, 76, alpha)
    gray = (100, 116, 139, alpha)
    vertices = interpolate_query_vertices(query, t)
    p2 = [project_to_rect(v, center, radius, rect) for v in vertices]
    if query["kind"] == "edge-edge":
        color_a = blue if truth or primary else gray
        color_b = coral if truth or primary else gray
        draw_polyline(draw, [p2[0], p2[1]], color_a, width)
        draw_polyline(draw, [p2[2], p2[3]], color_b, width)
        for point, color in [(p2[0], color_a), (p2[1], color_a), (p2[2], color_b), (p2[3], color_b)]:
            r = 7 if primary else 4
            draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r], fill=color)
    else:
        tri_color = coral if truth or primary else gray
        vertex_color = blue if truth or primary else gray
        draw.polygon([p2[1], p2[2], p2[3]], fill=tri_color[:3] + (70 if not primary else 115,), outline=tri_color)
        r = 9 if primary else 5
        draw.ellipse([p2[0][0] - r, p2[0][1] - r, p2[0][0] + r, p2[0][1] + r], fill=vertex_color)
    if emphasis > 0.0:
        cx = sum(p[0] for p in p2) / 4.0
        cy = sum(p[1] for p in p2) / 4.0
        rr = (34 if primary else 20) + 18 * emphasis
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(34, 197, 94, int(220 * emphasis)), width=4)


def render_selected_real_ti_dense_group() -> dict[str, str]:
    case = "selected_real_ti_dense_group_large_run_id"
    bench_dir = BENCH_ROOT / case
    demo_dir = DEMO_ROOT / case
    frame_dir = demo_dir / "_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = bench_dir / "selected_real_ti_dense_group_1024x128_three_method.json"
    stats_path = bench_dir / "selected_real_ti_dense_group_1024x128_schedule_stats.json"
    schedule_path = bench_dir / "selected_real_ti_dense_group_1024x128_learned_schedule.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    methods = metrics["methods"]
    queries = collect_representative_ti_queries(schedule_path)
    positives = [q for q in queries if q["truth"]]
    primary_query = positives[0] if positives else (queries[0] if queries else None)
    global_center, global_radius = compute_scene_frame(queries)
    if primary_query is not None:
        zoom_center, zoom_radius = compute_scene_frame([primary_query])
    else:
        zoom_center, zoom_radius = global_center, global_radius

    w, h = 1920, 1080
    title_font = safe_font(48)
    subtitle_font = safe_font(28)
    label_font = safe_font(26)
    small_font = safe_font(20)
    colors = {
        "NoProposal+TI": (110, 110, 110),
        "Random+TI": (230, 155, 60),
        "RTSTPFExact+TI": (45, 170, 105),
    }
    max_calls = max(float(m["exact_calls"]) for m in methods)
    total_frames = 120
    frame_paths: list[Path] = []

    for idx in range(total_frames):
        t = idx / (total_frames - 1)
        img = Image.new("RGB", (w, h), (248, 250, 252))
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rectangle([0, 0, w, 160], fill=(14, 24, 38, 255))
        draw.text((58, 36), "Selected-real Tight-Inclusion Dense Group Scheduling", fill=(245, 248, 252), font=title_font)
        draw.text(
            (58, 96),
            "Actual NYU/Tight-Inclusion primitive CCD coordinates; rows 0-3 are t0, rows 4-7 are t1. STPF only schedules exact work.",
            fill=(195, 205, 220),
            font=subtitle_font,
        )

        global_rect = (58, 230, 950, 835)
        zoom_rect = (1020, 230, 1810, 835)
        draw.rounded_rectangle(global_rect, radius=22, fill=(241, 245, 249, 255), outline=(148, 163, 184, 150), width=2)
        draw.rounded_rectangle(zoom_rect, radius=22, fill=(255, 255, 255, 255), outline=(59, 130, 246, 180), width=3)
        draw.text((global_rect[0] + 28, global_rect[1] + 24), "Representative real primitive candidates", fill=(30, 41, 59), font=label_font)
        draw.text((zoom_rect[0] + 28, zoom_rect[1] + 24), "Local exact-query zoom", fill=(30, 41, 59), font=label_font)
        draw.text(
            (global_rect[0] + 28, global_rect[1] + 60),
            "gray: hard negatives | blue/red: scheduled positive exact hits",
            fill=(71, 85, 105),
            font=small_font,
        )
        if primary_query is not None:
            draw.text(
                (zoom_rect[0] + 28, zoom_rect[1] + 60),
                f"{primary_query['case']} / {primary_query['kind']} / query {primary_query['query_index']} / truth={int(bool(primary_query['truth']))}",
                fill=(71, 85, 105),
                font=small_font,
            )
        for q in queries:
            draw_query_geometry(draw, q, t, global_rect, global_center, global_radius, primary=False)
        if primary_query is not None:
            draw_query_geometry(draw, primary_query, t, zoom_rect, zoom_center, zoom_radius, primary=True)
            toi = float(primary_query.get("toi_proxy", 0.0))
            px0, py0 = zoom_rect[0] + 36, zoom_rect[3] - 56
            px1 = zoom_rect[1] + 680
            draw.line([(px0, py0), (px0 + 660, py0)], fill=(203, 213, 225, 255), width=6)
            draw.line([(px0, py0), (px0 + 660 * t, py0)], fill=(59, 130, 246, 255), width=6)
            draw.ellipse([px0 + 660 * toi - 8, py0 - 8, px0 + 660 * toi + 8, py0 + 8], fill=(34, 197, 94, 255))
            draw.text((px0, py0 + 14), f"animated time t={t:.2f}; green marker=min-distance TOI proxy {toi:.2f}", fill=(71, 85, 105), font=small_font)

        # Method bars.
        panel_x, panel_y = 58, 876
        draw.text((panel_x, panel_y - 46), "Dense group benchmark result", fill=(30, 41, 59), font=label_font)
        for j, m in enumerate(methods):
            name = m["method"]
            calls = float(m["exact_calls"])
            reduction = 100.0 * float(m["exact_call_reduction"])
            wall = float(m["wall_ms"])
            rank = float(m["first_positive_rank_mean"])
            x = panel_x + j * 612
            y = panel_y
            draw.text((x, y), name, fill=colors[name] + (255,), font=label_font)
            bar_w = int(500 * calls / max_calls)
            draw.rounded_rectangle([x, y + 42, x + 500, y + 82], radius=10, fill=(225, 231, 239, 255))
            draw.rounded_rectangle([x, y + 42, x + int(bar_w * (0.25 + 0.75 * t)), y + 82], radius=10, fill=colors[name] + (245,))
            draw.text(
                (x, y + 96),
                f"exact {int(calls):,}; reduction {reduction:.2f}%; rank {rank:.2f}; wall {wall:.1f} ms",
                fill=(64, 76, 92),
                font=small_font,
            )

        draw.rectangle([58, h - 48, w - 58, h - 16], outline=(148, 163, 184, 160), width=1)
        draw.text(
            (78, h - 42),
            "Scope: this NYU/TI benchmark stores primitive CCD queries rather than full object connectivity. The rendering therefore shows the real moving VF/EE primitives used by the exact certificate.",
            fill=(45, 55, 72),
            font=small_font,
        )

        path = frame_dir / f"frame_{idx:04d}.png"
        img.save(path)
        frame_paths.append(path)

    global_mp4 = demo_dir / "global.mp4"
    encode_frames(frame_dir, global_mp4)

    sheet = Image.new("RGB", (w * 2, h * 2), (248, 250, 252))
    for k, frame_idx in enumerate([0, 39, 79, 119]):
        sheet.paste(Image.open(frame_paths[frame_idx]), ((k % 2) * w, (k // 2) * h))
    contact_sheet = demo_dir / "contact_sheet.png"
    sheet.save(contact_sheet)

    report = [
        "# Selected-real TI Dense Group Primitive CCD Visualization",
        "",
        "This visualization renders actual primitive CCD geometry from the selected-real Tight-Inclusion / NYU dense-group benchmark.",
        "The source benchmark stores vertex-face and edge-edge primitive queries, not complete object mesh connectivity, so this is the physically faithful rendering available from this dataset.",
        "",
        f"- Source benchmark: `{metrics_path.as_posix()}`",
        f"- Schedule stats: `{stats_path.as_posix()}`",
        f"- Learned schedule: `{schedule_path.as_posix()}`",
        f"- MP4: `{global_mp4.as_posix()}`",
        f"- Contact sheet: `{contact_sheet.as_posix()}`",
        f"- Representative primitive queries rendered: `{len(queries)}`",
        "",
        "Rendering semantics:",
        "",
        "- Edge-edge query: two moving segment primitives, interpolated from rows 0-3 at `t0` to rows 4-7 at `t1`.",
        "- Vertex-face query: one moving vertex and one moving triangle, interpolated from rows 0-3 at `t0` to rows 4-7 at `t1`.",
        "- Green marker: sampled minimum-distance time proxy for visual emphasis only; the benchmark correctness still comes from native Tight-Inclusion exact evaluation.",
        "",
        "Scope: native Tight-Inclusion primitive exact payload. Full mesh object surfaces cannot be reconstructed from this benchmark alone because the dataset only stores primitive queries.",
    ]
    (demo_dir / "case_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"global_mp4": global_mp4.as_posix(), "contact_sheet": contact_sheet.as_posix()}


def main() -> None:
    selected = render_selected_real_ti_dense_group()
    transcoded: dict[str, str] = {}
    for case in ["object_object_dense_mesh_contact_run_id", "hard_negative_near_miss_dense_run_id"]:
        mp4 = DEMO_ROOT / case / "global.mp4"
        legacy = transcode_to_h264(mp4)
        transcoded[case] = legacy.as_posix()
    payload = {
        "selected_real_ti_dense_group_large_run_id": selected,
        "transcoded_legacy_sources": transcoded,
        "codec": "H.264 libx264, yuv420p, faststart",
    }
    out = DEMO_ROOT / "recent_three_cases_h264_rerender_run_id.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
