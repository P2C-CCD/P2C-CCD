from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = REPO_ROOT / "src/MyDemo/paper_true_mesh_surface_contact_abc_run_id"
METRICS_PATH = CASE_DIR / "metrics.json"
DEFAULT_OUTPUT = CASE_DIR / "paper_true_mesh_surface_contact_abc.mp4"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = CASE_DIR / "old" / f"hq_rerender_backup_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / path.name
    shutil.copy2(path, backup_path)
    return backup_path


def _load_asset_path() -> Path:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    raw = Path(metrics.get("asset_path") or metrics["asset"]["asset_path"])
    if raw.is_absolute():
        return raw
    return REPO_ROOT / raw


def _load_contact_meshes(target_diagonal: float = 3.65):
    import pyvista as pv

    mesh = pv.read(str(_load_asset_path())).triangulate().clean(tolerance=1e-8)
    points = np.asarray(mesh.points, dtype=np.float64)
    center = points.mean(axis=0)
    points = points - center
    diagonal = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    if diagonal <= 0:
        raise RuntimeError("invalid mesh diagonal")
    mesh.points = (points * (target_diagonal / diagonal)).astype(np.float32)

    mesh_a = mesh.copy(deep=True)
    mesh_b = mesh.copy(deep=True)
    mesh_b.points[:, 0] *= -1.0
    mesh_b.points[:, 1] *= -1.0
    return mesh_a, mesh_b


def _mesh_bounds_x(mesh) -> tuple[float, float, float]:
    xmin, xmax, *_ = mesh.bounds
    return float(xmin), float(xmax), float(xmax - xmin)


def _positions(mesh_a, mesh_b, t: float, toi: float = 0.5) -> tuple[float, float]:
    ax0, ax1, aw = _mesh_bounds_x(mesh_a)
    bx0, bx1, bw = _mesh_bounds_x(mesh_b)
    contact_a = -ax1
    contact_b = -bx0
    gap = 1.15 * max(aw, bw)
    if t <= toi:
        alpha = t / toi
        return contact_a - gap * (1.0 - alpha), contact_b + gap * (1.0 - alpha)
    beta = (t - toi) / max(1e-8, 1.0 - toi)
    # Elastic rebound: separate along the same approach axis after certificate time.
    return contact_a - 0.72 * gap * beta, contact_b + 0.72 * gap * beta


def _copy_with_x_offset(mesh, dx: float):
    out = mesh.copy(deep=True)
    out.points[:, 0] += dx
    return out


def _global_camera_state(mesh_a, mesh_b) -> tuple[np.ndarray, np.ndarray, float]:
    """Use one fixed camera for the entire clip so motion remains visible."""
    samples = []
    for t in np.linspace(0.0, 1.0, 9):
        dx_a, dx_b = _positions(mesh_a, mesh_b, float(t))
        samples.append(_copy_with_x_offset(mesh_a, dx_a).points)
        samples.append(_copy_with_x_offset(mesh_b, dx_b).points)
    all_points = np.vstack(samples)
    bounds_min = all_points.min(axis=0)
    bounds_max = all_points.max(axis=0)
    target = 0.5 * (bounds_min + bounds_max)
    extent = float(np.linalg.norm(bounds_max - bounds_min))
    camera = target + np.array([3.8, -4.8, 2.85]) * max(1.0, extent / 5.8)
    parallel_scale = 0.46 * max(1.0, extent)
    return target, camera, parallel_scale


def _render_scene(
    mesh_a,
    mesh_b,
    t: float,
    scene_size: tuple[int, int],
    camera_state: tuple[np.ndarray, np.ndarray, float],
) -> Image.Image:
    import pyvista as pv

    dx_a, dx_b = _positions(mesh_a, mesh_b, t)
    a = _copy_with_x_offset(mesh_a, dx_a)
    b = _copy_with_x_offset(mesh_b, dx_b)

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=scene_size)
    plotter.set_background("#f3f6f9")
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    plotter.add_mesh(
        a,
        color="#25a7e8",
        smooth_shading=True,
        specular=0.34,
        roughness=0.48,
        show_edges=False,
    )
    plotter.add_mesh(
        b,
        color="#ff705f",
        smooth_shading=True,
        specular=0.32,
        roughness=0.50,
        show_edges=False,
    )
    target, camera, parallel_scale = camera_state
    plotter.add_light(pv.Light(position=tuple(camera), focal_point=tuple(target), color="white", intensity=0.95))
    plotter.add_light(pv.Light(position=(-3.0, 3.0, 4.0), focal_point=tuple(target), color="white", intensity=0.32))
    plotter.camera_position = (tuple(camera.tolist()), tuple(target.tolist()), (0.0, 0.0, 1.0))
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = parallel_scale
    plotter.camera.clipping_range = (0.01, 10000.0)
    plotter.render()
    rgb = np.asarray(plotter.screenshot(return_img=True)[:, :, :3], dtype=np.uint8)
    plotter.close()
    return Image.fromarray(rgb, mode="RGB")


def _fit_contain(img: Image.Image, size: tuple[int, int], bg: tuple[int, int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, bg)
    scale = min(size[0] / img.width, size[1] / img.height)
    new_size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
    resized = img.resize(new_size, Image.Resampling.LANCZOS)
    canvas.paste(resized, ((size[0] - new_size[0]) // 2, (size[1] - new_size[1]) // 2))
    return canvas


def _draw_header(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, color: tuple[int, int, int], subtitle: str, calls: str, work: str) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0 + 24, y0 + 22, x1 - 24, y1 - 16), radius=20, fill=(232, 238, 244), outline=(166, 180, 194), width=2)
    draw.text((x0 + 46, y0 + 42), title, font=_font(44, bold=True), fill=color)
    draw.text((x0 + 46, y0 + 94), subtitle, font=_font(24), fill=(65, 76, 88))
    draw.text((x0 + 46, y0 + 132), calls, font=_font(23), fill=(38, 48, 60))
    draw.text((x0 + 330, y0 + 132), work, font=_font(23), fill=(38, 48, 60))
    draw.text((x0 + 46, y0 + 166), "FN 0", font=_font(23, bold=True), fill=(38, 48, 60))


def _draw_candidate_grid(draw: ImageDraw.ImageDraw, x0: int, y0: int, w: int, h: int, method: str) -> None:
    draw.text((x0 + 26, y0 - 30), "candidate grid: slabs x patch-pairs", font=_font(20), fill=(106, 116, 128))
    if method == "rtstpf":
        draw.rectangle((x0 + 50, y0 + 22, x0 + 162, y0 + h - 8), outline=(0, 164, 216), width=4)
        draw.rectangle((x0 + 330, y0 + h - 34, x0 + 430, y0 + h - 24), fill=(245, 145, 20), outline=(210, 105, 10))
        return
    slabs = 8
    slab_w = (w - 80) / slabs
    for i in range(slabs):
        left = int(x0 + 40 + i * slab_w)
        right = int(x0 + 40 + (i + 1) * slab_w - 7)
        fill = (26, 178, 138) if i == 1 else (239, 146, 18)
        draw.rectangle((left, y0 + 24, right, y0 + h - 16), fill=fill, outline=(204, 119, 12), width=2)
        for yy in range(y0 + 42, y0 + h - 22, 22):
            draw.line((left + 3, yy, right - 3, yy), fill=(104, 90, 55), width=1)
    selected_left = int(x0 + 40 + slab_w)
    selected_right = int(x0 + 40 + 2 * slab_w - 7)
    draw.rectangle((selected_left - 3, y0 + 18, selected_right + 3, y0 + h - 10), outline=(0, 180, 210), width=4)


def _compose_frame(scene: Image.Image, frame_index: int, frame_count: int, fps: int) -> Image.Image:
    width, height = 2880, 1620
    panel_w = width // 3
    header_h = 230
    footer_h = 255
    scene_h = height - header_h - footer_h
    bg = (244, 247, 250)
    frame = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(frame)

    panel_scene = _fit_contain(scene, (panel_w, scene_h), bg)
    methods = [
        ("RTSTPFExact", (23, 158, 92), "learned STPF proposal + exact certificate", "exact calls 1/128", "work 335.2", "rtstpf"),
        ("RTExact", (58, 131, 218), "RT candidates direct to exact", "exact calls 128/128", "work 270,442.9", "exact"),
        ("NoProposal", (226, 87, 76), "safety fallback exact queue", "exact calls 128/128", "work 270,442.9", "exact"),
    ]
    for col, (title, color, subtitle, calls, work, mode) in enumerate(methods):
        x = col * panel_w
        if col > 0:
            draw.line((x, header_h, x, height - 12), fill=(198, 205, 213), width=4)
        _draw_header(draw, (x, 0, x + panel_w, header_h), title, color, subtitle, calls, work)
        frame.paste(panel_scene, (x, header_h))
        _draw_candidate_grid(draw, x + 12, header_h + scene_h + 42, panel_w - 24, footer_h - 88, mode)

    progress = frame_index / max(1, frame_count - 1)
    y = height - 42
    draw.rectangle((92, y, width - 92, y + 10), fill=(206, 213, 221))
    draw.rectangle((92, y, int(92 + (width - 184) * progress), y + 10), fill=(255, 122, 68))
    px = int(92 + (width - 184) * progress)
    draw.ellipse((px - 14, y - 10, px + 14, y + 18), fill=(255, 122, 68), outline=(110, 118, 128), width=3)
    t_seconds = progress * 5.0
    draw.text((34, height - 84), f"t={t_seconds:.2f}s | TOI=2.50s", font=_font(24), fill=(87, 98, 110))
    return frame


def _encode_frames(frame_dir: Path, output: Path, fps: int, crf: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def render(output: Path, fps: int, frame_count: int, keep_frames: bool, crf: int) -> dict[str, str]:
    mesh_a, mesh_b = _load_contact_meshes()
    camera_state = _global_camera_state(mesh_a, mesh_b)
    backup_path = _backup(output)
    frame_dir = output.parent / "hq_rerender_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    scene_size = (1920, 2360)
    preview_path = output.with_name("paper_true_mesh_surface_contact_abc_hq_preview.png")
    for i in range(frame_count):
        t = i / max(1, frame_count - 1)
        scene = _render_scene(mesh_a, mesh_b, t, scene_size, camera_state)
        frame = _compose_frame(scene, i, frame_count, fps)
        if i == frame_count // 2:
            frame.save(preview_path)
        frame.save(frame_dir / f"frame_{i:05d}.png", quality=96)

    _encode_frames(frame_dir, output, fps, crf)
    if not keep_frames:
        shutil.rmtree(frame_dir)

    return {
        "output": str(output),
        "preview": str(preview_path),
        "backup": str(backup_path) if backup_path else "",
        "frames": str(frame_dir) if keep_frames else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frame-count", type=int, default=121)
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    result = render(args.output, args.fps, args.frame_count, args.keep_frames, args.crf)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
