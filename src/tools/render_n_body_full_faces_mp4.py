from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

import run_scalable_ccd_scene_supplementary as scenevis


SCENE = "n-body-simulation"
P_DRIVE_ROOT = Path(os.environ.get("P2CCCD_ASCII_ROOT", str(Path(__file__).resolve().parents[2])))


def _find_group() -> object:
    for group in scenevis.discover_groups(scenevis.SOURCE_ROOT):
        if group.scene == SCENE:
            return group
    raise RuntimeError(f"could not find Scalable-CCD group for {SCENE}")


def _write_progress(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _p_drive_path(path: Path) -> str:
    try:
        return str(P_DRIVE_ROOT / path.resolve().relative_to(scenevis.ROOT.resolve())).replace("/", "\\")
    except ValueError:
        return str(path)


def _read_png_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"failed to decode rendered frame {path}")
    return frame


def _encode_mp4(frame_dir: Path, frame_count: int, output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.mp4")
    first = _read_png_bgr(frame_dir / "frame_0000.png")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(_p_drive_path(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {tmp_path}")
    for i in range(frame_count):
        writer.write(_read_png_bgr(frame_dir / f"frame_{i:04d}.png"))
    writer.release()

    command = [
        "ffmpeg",
        "-y",
        "-i",
        _p_drive_path(tmp_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-crf",
        "18",
        _p_drive_path(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        tmp_path.replace(output_path)
    elif tmp_path.exists():
        tmp_path.unlink()


def render_full_faces_mp4(frame_count: int, fps: int, force: bool) -> dict[str, object]:
    started = time.perf_counter()
    output_dir = scenevis.MYDEMO_DIR
    frame_dir = output_dir / "n_body_simulation_full_faces_frames"
    output_path = output_dir / "n_body_simulation_full_faces.mp4"
    manifest_path = output_dir / "n_body_simulation_full_faces_render_manifest.json"

    if force and frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    group = _find_group()
    sequence = scenevis.scene_frame_sequence(SCENE, int(group.timestep))
    if len(sequence) < 3:
        raise RuntimeError(f"{SCENE} does not have a full frame sequence")

    sampled_indices = np.rint(np.linspace(0, len(sequence) - 1, frame_count)).astype(np.int64)
    unique_indices = sorted(set(int(index) for index in sampled_indices))

    vertices_by_index: dict[int, np.ndarray] = {}
    contacts_by_index: dict[int, np.ndarray] = {}
    faces: np.ndarray | None = None
    min_corner: np.ndarray | None = None
    max_corner: np.ndarray | None = None
    for index in unique_indices:
        vertices, loaded_faces = scenevis.load_mesh(sequence[index], SCENE)
        vertices_by_index[index] = vertices
        number = scenevis.frame_number(sequence[index])
        contacts_by_index[index] = (
            scenevis.read_scene_step_contacts(SCENE, int(number))
            if number is not None
            else np.zeros((0, 3), dtype=np.float64)
        )
        if faces is None:
            faces = loaded_faces
        min_corner = vertices.min(axis=0) if min_corner is None else np.minimum(min_corner, vertices.min(axis=0))
        max_corner = vertices.max(axis=0) if max_corner is None else np.maximum(max_corner, vertices.max(axis=0))

    assert faces is not None and min_corner is not None and max_corner is not None
    first_vertices = vertices_by_index[unique_indices[0]]
    object_ids = scenevis.face_object_ids(first_vertices, faces)
    center = 0.5 * (min_corner + max_corner)
    projected_bounds = [scenevis.project(vertices, center, 1.0, (1280, 720))[0] for vertices in vertices_by_index.values()]
    all_xy = np.vstack(projected_bounds)
    span = np.maximum(all_xy.max(axis=0) - all_xy.min(axis=0), 1.0)
    scale = min(1040.0 / span[0], 470.0 / span[1]) * scenevis.SCENE_ZOOM_MULTIPLIER.get(SCENE, 1.0)

    source_frame_numbers = []
    for out_index, source_index_raw in enumerate(sampled_indices):
        source_index = int(source_index_raw)
        frame_path = frame_dir / f"frame_{out_index:04d}.png"
        source_frame_numbers.append(scenevis.frame_number(sequence[source_index]))
        if frame_path.exists() and not force:
            continue
        frame = scenevis.render_mesh_frame(
            SCENE,
            float(out_index) / float(max(1, frame_count - 1)),
            vertices_by_index[source_index],
            faces,
            center,
            scale,
            contacts_by_index[source_index],
            object_ids=object_ids,
        )
        frame.save(frame_path)
        if out_index == 0 or (out_index + 1) % 10 == 0 or out_index + 1 == frame_count:
            _write_progress(
                manifest_path,
                {
                    "status": "rendering_frames",
                    "scene": SCENE,
                    "rendered_frames": out_index + 1,
                    "target_frames": frame_count,
                    "fps": fps,
                    "frame_dir": str(frame_dir),
                    "output_mp4": str(output_path),
                    "source_frame_count": len(sequence),
                    "unique_source_indices": len(unique_indices),
                    "render_faces": int(faces.shape[0]),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )

    _write_progress(
        manifest_path,
        {
            "status": "encoding_mp4",
            "scene": SCENE,
            "rendered_frames": frame_count,
            "target_frames": frame_count,
            "fps": fps,
            "frame_dir": str(frame_dir),
            "output_mp4": str(output_path),
            "source_frame_count": len(sequence),
            "unique_source_indices": len(unique_indices),
            "source_frame_numbers": source_frame_numbers,
            "render_faces": int(faces.shape[0]),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    _encode_mp4(frame_dir, frame_count, output_path, fps)

    result = {
        "status": "complete",
        "scene": SCENE,
        "mp4": str(output_path),
        "frame_dir": str(frame_dir),
        "manifest": str(manifest_path),
        "frame_count": frame_count,
        "fps": fps,
        "source_frame_count": len(sequence),
        "unique_source_indices": len(unique_indices),
        "render_faces": int(faces.shape[0]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_progress(manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Render full-face n-body Scalable-CCD MP4 without the global face cap.")
    parser.add_argument("--frame-count", type=int, default=scenevis.VISUALIZATION_FRAME_COUNT)
    parser.add_argument("--fps", type=int, default=scenevis.VISUALIZATION_FPS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(render_full_faces_mp4(args.frame_count, args.fps, args.force), indent=2), flush=True)


if __name__ == "__main__":
    main()
