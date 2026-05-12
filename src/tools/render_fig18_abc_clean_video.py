from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = (
    REPO_ROOT
    / "src/MyDemo/paper_true_mesh_surface_contact_abc_run_id"
)
GLOBAL_MP4 = CASE_DIR / "paper_true_mesh_surface_contact_abc.mp4"
ZOOM_MP4 = CASE_DIR / "collision_zoom_wireframe.mp4"
OUT_DIR = CASE_DIR / "fig18_clean_video"
DEFAULT_OUTPUT = OUT_DIR / "fig18_abc_real_mesh_contact_clean_2x2.mp4"


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded: {path}")
    return frames, fps


def crop_rel(frame: np.ndarray, crop: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = crop
    return frame[
        max(0, int(round(y0 * h))) : min(h, int(round(y1 * h))),
        max(0, int(round(x0 * w))) : min(w, int(round(x1 * w))),
    ]


def mask_box_rel(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    fill: tuple[int, int, int] = (63, 74, 87),
) -> None:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    left = max(0, min(w - 1, int(round(x0 * w))))
    top = max(0, min(h - 1, int(round(y0 * h))))
    right = max(left + 1, min(w, int(round(x1 * w))))
    bottom = max(top + 1, min(h, int(round(y1 * h))))
    frame[top:bottom, left:right] = np.array(fill, dtype=np.uint8)


def clean_global(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Remove text overlays while preserving the lower candidate-grid windows."""
    clean = crop_rel(frame, (0.0, 0.17, 1.0, 0.925)).copy()
    # Per-column candidate captions, axis labels, and lower player/watermark text.
    # The colored candidate-grid bars are intentionally kept.
    for col in range(3):
        x0 = col / 3.0
        mask_box_rel(clean, (x0 + 0.002, 0.700, x0 + 0.320, 0.775))
        mask_box_rel(clean, (x0 + 0.010, 0.900, x0 + 0.330, 1.0))
    mask_box_rel(clean, (0.0, 0.900, 1.0, 1.0))
    panel = fit_cover(clean, target_size)
    mask_box_rel(panel, (0.0, 0.730, 1.0, 0.845))
    for col in range(3):
        x0 = col / 3.0
        mask_box_rel(panel, (x0 + 0.002, 0.790, x0 + 0.325, 0.850))
        mask_box_rel(panel, (x0 + 0.002, 0.944, x0 + 0.325, 1.0))
    return panel


def clean_zoom(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    clean = crop_rel(frame, (0.0, 0.13, 1.0, 0.94)).copy()
    # Remove upper-left numeric overlay and lower-left watermark text; keep the
    # bottom candidate/TOI indicator region.
    mask_box_rel(clean, (0.0, 0.0, 0.34, 0.115), fill=(16, 28, 41))
    mask_box_rel(clean, (0.0, 0.875, 0.46, 1.0), fill=(16, 28, 41))
    mask_box_rel(clean, (0.45, 0.930, 0.56, 1.0), fill=(16, 28, 41))
    panel = fit_cover(clean, target_size)
    mask_box_rel(panel, (0.0, 0.0, 0.34, 0.13), fill=(16, 28, 41))
    mask_box_rel(panel, (0.0, 0.900, 0.50, 1.0), fill=(16, 28, 41))
    return panel


def fit_cover(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size
    h, w = frame.shape[:2]
    scale = max(target_w / w, target_h / h)
    resized = cv2.resize(
        frame,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    rh, rw = resized.shape[:2]
    x0 = max(0, (rw - target_w) // 2)
    y0 = max(0, (rh - target_h) // 2)
    return resized[y0 : y0 + target_h, x0 : x0 + target_w]


def staged_global_index(progress: float, stage_start: float, span: float, n: int) -> int:
    ratio = min(0.999, max(0.0, stage_start + span * progress))
    return min(n - 1, int(round(ratio * (n - 1))))


def build_frames(output_dir: Path, fps: float, duration_s: float) -> None:
    global_frames, global_fps = read_video(GLOBAL_MP4)
    zoom_frames, zoom_fps = read_video(ZOOM_MP4)
    fps = fps or min(global_fps, zoom_fps, 24.0)
    total = max(1, int(round(duration_s * fps)))
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_size = (960, 540)
    canvas_w = panel_size[0] * 2
    canvas_h = panel_size[1] * 2
    gap_color = np.array([255, 255, 255], dtype=np.uint8)

    for i in range(total):
        p = i / max(1, total - 1)
        top_left = clean_global(
            global_frames[staged_global_index(p, 0.00, 0.22, len(global_frames))],
            panel_size,
        )
        top_right = clean_global(
            global_frames[staged_global_index(p, 0.32, 0.22, len(global_frames))],
            panel_size,
        )
        bottom_left = clean_zoom(
            zoom_frames[min(len(zoom_frames) - 1, int(round(p * (len(zoom_frames) - 1))))],
            panel_size,
        )
        bottom_right = clean_global(
            global_frames[staged_global_index(p, 0.68, 0.24, len(global_frames))],
            panel_size,
        )

        canvas = np.empty((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = gap_color
        canvas[0 : panel_size[1], 0 : panel_size[0]] = top_left
        canvas[0 : panel_size[1], panel_size[0] : canvas_w] = top_right
        canvas[panel_size[1] : canvas_h, 0 : panel_size[0]] = bottom_left
        canvas[panel_size[1] : canvas_h, panel_size[0] : canvas_w] = bottom_right

        out_path = output_dir / f"frame_{i:05d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def encode_mp4(frame_dir: Path, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-i",
        str(frame_dir / "frame_%05d.png"),
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
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    frame_dir = OUT_DIR / "fig18_clean_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    build_frames(frame_dir, args.fps, args.duration)
    encode_mp4(frame_dir, args.output, args.fps)
    if not args.keep_frames:
        shutil.rmtree(frame_dir)
    print(args.output)


if __name__ == "__main__":
    main()
