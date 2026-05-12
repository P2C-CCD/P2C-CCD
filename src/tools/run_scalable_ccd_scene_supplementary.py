from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "src" / "python"
TOOLS_DIR = Path(__file__).resolve().parent
for import_root in (PYTHON_ROOT, TOOLS_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from convert_scalable_ccd_sample_to_p2c_groups import (  # noqa: E402
    RUN_NAME as CONVERSION_RUN_NAME,
    concatenate_arrays,
    discover_groups,
    rel,
    write_npz,
)
from p2cccd.bench.tight_inclusion_stpf_training import run_tight_inclusion_stpf_training  # noqa: E402
from p2cccd.proposal.policy_head_selection import (  # noqa: E402
    RTSTPFPolicyHead,
    score_rtstpf_candidates,
)
from p2cccd.proposal.stpf_model import (  # noqa: E402
    STPFModelPreset,
    build_stpf_model_from_checkpoint_payload,
)


RUN_NAME = "scalable_ccd_scene_supplementary_training_run_id"
CONVERTED_SHARD_DIR = ROOT / "src" / "datasets" / "training" / "scalable_ccd_scene_groups" / "shards" / CONVERSION_RUN_NAME
SPLIT_SHARD_DIR = ROOT / "src" / "datasets" / "training" / "scalable_ccd_scene_groups" / "shards" / RUN_NAME
BENCHMARK_DIR = ROOT / "src" / "benchmark" / RUN_NAME
OUTPUT_DIR = ROOT / "src" / "outputs" / "stpf_training"
MYDEMO_DIR = ROOT / "src" / "MyDemo" / RUN_NAME
SOURCE_ROOT = ROOT / "src" / "baseline" / "Sample-Scalable-CCD-Data"
FULL_DATASET_ROOT = ROOT / "src" / "baseline" / "datasets" / "continuous-collision-detection"
TRAIN_SCENES = {"armadillo-rollers", "cloth-ball", "n-body-simulation", "puffer-ball"}
VALIDATION_SCENES = {"cloth-funnel"}
HELDOUT_SCENES = {"rod-twist"}
VISUALIZATION_FRAME_COUNT = 240
VISUALIZATION_FPS = 60
MAX_RENDER_FACES = 120000
MAX_SPLAT_VERTICES = 90000
SCENE_MAX_RENDER_FACES: dict[str, int | None] = {
    # n-body frames contain 146,480 faces.  The global 120k cap creates visible
    # holes in ball surfaces, so render this scene with the complete PLY mesh.
    "n-body-simulation": None,
}
SCENE_ZOOM_MULTIPLIER = {
    "puffer-ball": 5.5,
}


@dataclass
class MethodRow:
    split: str
    method: str
    groups: int
    candidates: int
    positive_groups: int
    positives: int
    exact_calls: float
    exact_work: float
    call_reduction: float
    work_reduction: float
    first_positive_rank_mean: float
    fn: int
    timing_scope: str


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"].item()))
    return arrays, metadata


def write_split_shards(converted_dir: Path, split_dir: Path) -> dict[str, Any]:
    groups_dir = converted_dir / "groups"
    if not groups_dir.exists():
        raise FileNotFoundError(groups_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    split_chunks: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "validation": [], "heldout_test": [], "all_scene": []}
    split_group_files: dict[str, list[str]] = {key: [] for key in split_chunks}
    per_group: list[dict[str, Any]] = []
    for npz_path in sorted(groups_dir.glob("*.npz")):
        arrays, metadata = load_npz(npz_path)
        scene = str(metadata["scene"])
        if scene in TRAIN_SCENES:
            split = "train"
        elif scene in VALIDATION_SCENES:
            split = "validation"
        elif scene in HELDOUT_SCENES:
            split = "heldout_test"
        else:
            split = "heldout_test"
        split_chunks[split].append(arrays)
        split_chunks["all_scene"].append(arrays)
        split_group_files[split].append(rel(npz_path))
        split_group_files["all_scene"].append(rel(npz_path))
        per_group.append(
            {
                "scene": scene,
                "kind": metadata["kind"],
                "timestep": metadata["timestep"],
                "row_count": metadata["row_count"],
                "positive_count": metadata["positive_count"],
                "split": split,
                "source_npz": rel(npz_path),
            }
        )
    chunks: list[dict[str, Any]] = []
    for split, arrays_list in split_chunks.items():
        if not arrays_list:
            continue
        arrays = concatenate_arrays(arrays_list)
        shard_path = split_dir / f"{split}.npz"
        metadata = {
            "schema_version": 1,
            "run_name": RUN_NAME,
            "source_conversion": CONVERSION_RUN_NAME,
            "split": split,
            "split_names": [split],
            "row_count": int(arrays["features"].shape[0]),
            "feature_dim": int(arrays["features"].shape[1]),
            "interval_bins": int(arrays["interval_targets"].shape[1]),
            "family_count": int(arrays["family_targets"].shape[1]),
            "grouping_scope": "scene_step_kind",
            "comparison_scope": "scene-level supplementary converted P2C candidate groups",
            "source_group_npz": split_group_files[split],
        }
        write_npz(shard_path, arrays, metadata)
        chunks.append(
            {
                "split": split,
                "path": shard_path.as_posix(),
                "row_count": int(arrays["features"].shape[0]),
                "positive_count": int(np.count_nonzero(arrays["ground_truth"])),
            }
        )
    manifest = {
        "schema_version": 1,
        "run_name": RUN_NAME,
        "source_conversion": CONVERSION_RUN_NAME,
        "converted_dir": rel(converted_dir),
        "split_policy": {
            "train_scenes": sorted(TRAIN_SCENES),
            "validation_scenes": sorted(VALIDATION_SCENES),
            "heldout_scenes": sorted(HELDOUT_SCENES),
        },
        "chunks": chunks,
        "groups": per_group,
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def group_slices(query_ids: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(query_ids, kind="stable")
    sorted_q = query_ids[order]
    slices: list[np.ndarray] = []
    start = 0
    n = int(order.shape[0])
    while start < n:
        end = start + 1
        qid = sorted_q[start]
        while end < n and sorted_q[end] == qid:
            end += 1
        slices.append(order[start:end])
        start = end
    return slices


def exact_calls_from_scores(labels: np.ndarray, costs: np.ndarray, query_ids: np.ndarray, scores: np.ndarray) -> tuple[int, float, list[int]]:
    exact_calls = 0
    exact_work = 0.0
    first_ranks: list[int] = []
    for idx in group_slices(query_ids):
        local_scores = scores[idx]
        local_order = idx[np.argsort(-local_scores, kind="stable")]
        local_labels = labels[local_order]
        if np.any(local_labels):
            first = int(np.flatnonzero(local_labels)[0]) + 1
            exact_calls += first
            exact_work += float(np.sum(costs[local_order[:first]], dtype=np.float64))
            first_ranks.append(first)
        else:
            exact_calls += int(local_order.shape[0])
            exact_work += float(np.sum(costs[local_order], dtype=np.float64))
    return exact_calls, exact_work, first_ranks


def benchmark_scores(split: str, arrays: dict[str, np.ndarray], score_map: dict[str, np.ndarray]) -> list[MethodRow]:
    labels = np.asarray(arrays["ground_truth"], dtype=np.bool_)
    costs = np.asarray(arrays["costs"], dtype=np.float64)
    query_ids = np.asarray(arrays["ids"][:, 1], dtype=np.uint64)
    candidate_count = int(labels.shape[0])
    total_work = float(np.sum(costs, dtype=np.float64))
    group_count = len(np.unique(query_ids))
    positive_groups = 0
    for idx in group_slices(query_ids):
        positive_groups += int(np.any(labels[idx]))
    positive_count = int(np.count_nonzero(labels))
    rows = [
        MethodRow(
            split=split,
            method="NoProposalAllExact",
            groups=group_count,
            candidates=candidate_count,
            positive_groups=positive_groups,
            positives=positive_count,
            exact_calls=float(candidate_count),
            exact_work=total_work,
            call_reduction=0.0,
            work_reduction=0.0,
            first_positive_rank_mean=0.0,
            fn=0,
            timing_scope="converted_candidate_row_all_exact",
        )
    ]
    for method, scores in score_map.items():
        exact_calls, exact_work, ranks = exact_calls_from_scores(labels, costs, query_ids, np.asarray(scores, dtype=np.float64))
        rows.append(
            MethodRow(
                split=split,
                method=method,
                groups=group_count,
                candidates=candidate_count,
                positive_groups=positive_groups,
                positives=positive_count,
                exact_calls=float(exact_calls),
                exact_work=float(exact_work),
                call_reduction=1.0 - exact_calls / max(1, candidate_count),
                work_reduction=1.0 - exact_work / max(1.0e-12, total_work),
                first_positive_rank_mean=float(np.mean(ranks)) if ranks else 0.0,
                fn=0,
                timing_scope="converted_candidate_row_group_early_stop",
            )
        )
    return rows


def trained_scores(checkpoint: Path, arrays: dict[str, np.ndarray], device: str) -> np.ndarray:
    import torch

    payload = torch.load(checkpoint, map_location=device)
    model, state_dict = build_stpf_model_from_checkpoint_payload(payload, fallback_preset=STPFModelPreset.TINY_MLP)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    features = np.asarray(arrays["features"], dtype=np.float32)
    priority_scores = np.zeros(features.shape[0], dtype=np.float32)
    cost_scores = np.zeros(features.shape[0], dtype=np.float32)
    uncertainty_scores = np.zeros(features.shape[0], dtype=np.float32)
    batch_size = 65536
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
            out = model(batch)
            stop = start + int(batch.shape[0])
            priority_scores[start:stop] = out.priority_score.detach().cpu().numpy()
            cost_scores[start:stop] = out.cost_score.detach().cpu().numpy()
            uncertainty_scores[start:stop] = out.uncertainty_score.detach().cpu().numpy()
    return score_rtstpf_candidates(
        {
            "priority_score": priority_scores,
            "cost_score": cost_scores,
            "uncertainty_score": uncertainty_scores,
        },
        {"features": features},
        head=RTSTPFPolicyHead.COST_AWARE,
    )


def write_benchmark(rows: list[MethodRow], path_prefix: Path, metadata: dict[str, Any]) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    row_dicts = [row.__dict__ for row in rows]
    with (path_prefix.with_suffix(".csv")).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)
    payload = {"metadata": metadata, "rows": row_dicts}
    path_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    table_rows = [
        [
            row.split,
            row.method,
            str(row.groups),
            str(row.candidates),
            str(row.positives),
            f"{row.exact_calls:.0f}",
            f"{100.0 * row.call_reduction:.3f}%",
            f"{100.0 * row.work_reduction:.3f}%",
            f"{row.first_positive_rank_mean:.3f}",
            str(row.fn),
        ]
        for row in rows
    ]
    lines = [
        "# Scalable-CCD Scene-level Supplementary Training Benchmark",
        "",
        f"Run identifier: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        "",
        "## Scope",
        "",
        "- Benchmark Inputisdescription Sample-Scalable-CCD-Data convertdescriptionto P2C candidate groups. ",
        "- descriptionand Scalable-CCD native kernel / simulator wall time direct comparison. ",
        "- description sample scenes descriptionsplit candidate descriptionis positive, group early-stop descriptionconnectdescription scene-level data source, descriptionas learned scheduling description SOTA description. ",
        "",
        "## Paths",
        "",
        f"- Split shards: `{metadata['split_shard_dir']}`",
        f"- Training report: `{metadata['training_report']}`",
        f"- Model state: `{metadata['model_state_path']}`",
        f"- Visualization: `{metadata['visualization_dir']}`",
        "",
        "## Results",
        "",
        "| Split | Method | Groups | Candidates | Positives | Exact calls | Call reduction | Work reduction | First-positive rank | FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in table_rows)
    lines.extend(
        [
            "",
            "## Scope Note",
            "",
            "This supplementary should be cited as a scene-data conversion and P2C replay compatibility result. It should not be worded as P2C-CCD beating Scalable-CCD native runtime.",
            "",
        ]
    )
    path_prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def load_mesh(path: Path, scene: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    try:
        mesh = trimesh.load(path, process=False)
    except UnicodeDecodeError:
        raw = path.read_bytes()
        marker = b"end_header"
        header_end = raw.find(marker)
        if header_end < 0:
            raise
        line_end = raw.find(b"\n", header_end)
        if line_end < 0:
            raise
        header = raw[: line_end + 1].decode("latin-1").encode("utf-8")
        mesh = trimesh.load(io.BytesIO(header + raw[line_end + 1 :]), file_type="ply", process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return vertices, limit_faces(faces, scene)


def scene_max_render_faces(scene: str | None) -> int | None:
    if scene is not None and scene in SCENE_MAX_RENDER_FACES:
        return SCENE_MAX_RENDER_FACES[scene]
    return MAX_RENDER_FACES


def limit_faces(faces: np.ndarray, scene: str | None = None) -> np.ndarray:
    max_faces = scene_max_render_faces(scene)
    if max_faces is None or faces.shape[0] <= max_faces:
        return faces
    indices = np.linspace(0, faces.shape[0] - 1, max_faces, dtype=np.int64)
    return faces[indices]


def load_mesh_pair(frame0: Path, frame1: Path, scene: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v0, faces = load_mesh(frame0, scene)
    v1, _ = load_mesh(frame1, scene)
    if v0.shape != v1.shape:
        v1 = v0.copy()
    return v0, v1, faces


def face_object_ids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return a two-object visual partition for paper figures.

    Scalable-CCD sample frames store scene meshes as a single PLY.  When mesh
    connectivity separates objects, use connected components.  If the frame is a
    single connected cloth/shell, split by the dominant spatial axis so the two
    sides remain visually distinguishable instead of rendering a single blue
    blob.
    """

    parent = np.arange(vertices.shape[0], dtype=np.int64)
    rank = np.zeros(vertices.shape[0], dtype=np.uint8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b, c in faces:
        union(int(a), int(b))
        union(int(b), int(c))
    roots = np.fromiter((find(int(v)) for v in range(vertices.shape[0])), dtype=np.int64, count=vertices.shape[0])
    face_roots = roots[faces[:, 0]]
    unique, counts = np.unique(face_roots, return_counts=True)
    if unique.shape[0] >= 2:
        largest = unique[np.argmax(counts)]
        return (face_roots != largest).astype(np.uint8)

    centroids = vertices[faces].mean(axis=1)
    centered = centroids - centroids.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coord = centered @ vh[0]
    threshold = float(np.median(coord))
    return (coord > threshold).astype(np.uint8)


def frame_number(path: Path) -> int | None:
    if path.name.startswith("._") or path.name == ".DS_Store":
        return None
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


def scene_frame_sequence(scene: str, preferred_timestep: int) -> list[Path]:
    frames_dir = FULL_DATASET_ROOT / scene / "frames"
    if not frames_dir.exists():
        frames_dir = SOURCE_ROOT / scene / "frames"
    numbered = [(number, path) for path in frames_dir.glob("*.ply") if (number := frame_number(path)) is not None]
    numbered.sort(key=lambda item: item[0])
    if len(numbered) <= 2:
        return [path for _, path in numbered]
    numbers = np.asarray([number for number, _ in numbered], dtype=np.int64)
    start_index = int(np.argmin(np.abs(numbers - int(preferred_timestep))))
    window_count = min(len(numbered), VISUALIZATION_FRAME_COUNT)
    if start_index + window_count <= len(numbered):
        begin = start_index
    else:
        begin = max(0, len(numbered) - window_count)
    return [path for _, path in numbered[begin : begin + window_count]]


def closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(1.0e-12, d1 - d3)
        return a + v * ab
    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(1.0e-12, d2 - d6)
        return a + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / max(1.0e-12, (d4 - d3) + (d5 - d6))
        return b + w * (c - b)
    denom = max(1.0e-12, va + vb + vc)
    v = vb / denom
    w = vc / denom
    return a + ab * v + ac * w


def closest_points_between_segments(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denom = a * c - b * b
    if denom < 1.0e-12:
        s = 0.0
        t = np.clip(e / max(c, 1.0e-12), 0.0, 1.0)
    else:
        s = np.clip((b * e - c * d) / denom, 0.0, 1.0)
        t = (b * s + e) / max(c, 1.0e-12)
        if t < 0.0:
            t = 0.0
            s = np.clip(-d / max(a, 1.0e-12), 0.0, 1.0)
        elif t > 1.0:
            t = 1.0
            s = np.clip((b - d) / max(a, 1.0e-12), 0.0, 1.0)
    return a0 + s * u, b0 + t * v


def approximate_query_contact(points_t0: np.ndarray, points_t1: np.ndarray, kind: str) -> np.ndarray:
    best_distance = math.inf
    best_point = points_t0.mean(axis=0)
    for alpha in np.linspace(0.0, 1.0, 9):
        points = (1.0 - float(alpha)) * points_t0 + float(alpha) * points_t1
        if kind == "ee":
            ca, cb = closest_points_between_segments(points[0], points[1], points[2], points[3])
            point = 0.5 * (ca + cb)
            distance = float(np.linalg.norm(ca - cb))
        else:
            tri_point = closest_point_on_triangle(points[0], points[1], points[2], points[3])
            point = 0.5 * (points[0] + tri_point)
            distance = float(np.linalg.norm(points[0] - tri_point))
        if distance < best_distance:
            best_distance = distance
            best_point = point
    return best_point


def camera_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = np.asarray([0.58, -0.72, 0.38], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(np.asarray([0.0, 0.0, 1.0]), forward)
    right /= max(1.0e-12, np.linalg.norm(right))
    up = np.cross(forward, right)
    up /= max(1.0e-12, np.linalg.norm(up))
    return right, up, forward


def project(points: np.ndarray, center: np.ndarray, scale: float, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    right, up, forward = camera_basis()
    centered = points - center
    x = centered @ right
    y = centered @ up
    z = centered @ forward
    w, h = size
    return np.column_stack([w * 0.5 + x * scale, h * 0.56 - y * scale]), z


def render_mesh_frame(
    scene: str,
    t_label: float,
    vertices: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    scale: float,
    contacts: np.ndarray,
    object_ids: np.ndarray | None = None,
    output_size: tuple[int, int] = (1280, 720),
) -> Image.Image:
    import cv2

    xy, depth = project(vertices, center, scale, output_size)
    face_xy = xy[faces]
    face_depth = depth[faces].mean(axis=1)
    image_array = np.full((output_size[1], output_size[0], 3), (246, 248, 249), dtype=np.uint8)
    splat_step = max(1, int(math.ceil(vertices.shape[0] / MAX_SPLAT_VERTICES)))
    splat_xy = np.rint(xy[::splat_step]).astype(np.int64)
    valid = (
        (splat_xy[:, 0] >= 1)
        & (splat_xy[:, 0] < output_size[0] - 1)
        & (splat_xy[:, 1] >= 1)
        & (splat_xy[:, 1] < output_size[1] - 1)
    )
    splat_xy = splat_xy[valid]
    splat_color = np.asarray([118, 176, 217], dtype=np.uint8)
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        image_array[splat_xy[:, 1] + dy, splat_xy[:, 0] + dx] = splat_color
    light = np.asarray([0.2, -0.35, 0.91], dtype=np.float64)
    light /= np.linalg.norm(light)
    order = np.argsort(face_depth)
    object_palette = (
        np.asarray([46, 132, 218], dtype=np.float64),
        np.asarray([232, 111, 81], dtype=np.float64),
    )
    rim_palette = (
        (16, 76, 142),
        (164, 62, 45),
    )
    wire_step = max(1, int(math.ceil(order.shape[0] / 26000)))
    for draw_count, face_index in enumerate(order):
        pts = face_xy[face_index]
        if pts[:, 0].max() < -80 or pts[:, 0].min() > output_size[0] + 80 or pts[:, 1].max() < -80 or pts[:, 1].min() > output_size[1] + 80:
            continue
        tri = vertices[faces[face_index]]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nrm = np.linalg.norm(normal)
        shade = 0.62 if nrm < 1.0e-12 else float(np.clip(0.48 + 0.52 * abs(np.dot(normal / nrm, light)), 0.38, 1.0))
        obj = int(object_ids[face_index]) if object_ids is not None and face_index < object_ids.shape[0] else 0
        base_color = object_palette[obj % 2]
        base = base_color * shade + np.asarray([245, 247, 250], dtype=np.float64) * (1.0 - shade) * 0.28
        fill = tuple(int(np.clip(v, 0, 255)) for v in base)
        polygon = np.rint(pts).astype(np.int32)
        cv2.fillConvexPoly(image_array, polygon, fill, lineType=cv2.LINE_AA)
        if draw_count % wire_step == 0:
            cv2.polylines(image_array, [polygon], True, rim_palette[obj % 2], thickness=1, lineType=cv2.LINE_AA)
    image = Image.fromarray(image_array, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle([18, 18, output_size[0] - 18, 86], fill=(255, 255, 255, 218), outline=(180, 188, 198, 230), width=1)
    draw.text((38, 34), f"Scalable-CCD Scene Supplementary: {scene}", fill=(28, 36, 46, 255))
    draw.text((38, 60), "full-dataset real mesh frames; visualization only, no native runtime comparison", fill=(86, 96, 108, 255))
    if contacts.size:
        contact_xy, _ = project(contacts, center, scale, output_size)
        if contact_xy.shape[0] >= 2:
            hull = contact_xy[: min(36, contact_xy.shape[0])]
            cx, cy = hull.mean(axis=0)
            radius = float(np.percentile(np.linalg.norm(hull - np.array([cx, cy]), axis=1), 90))
            radius = float(np.clip(radius + 18.0, 18.0, 78.0))
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(255, 205, 84, 36), outline=(255, 167, 38, 190), width=3)
        for i, (x, y) in enumerate(contact_xy[:64]):
            r = 8 if i < 16 else 5
            draw.line([x - r * 1.7, y, x + r * 1.7, y], fill=(255, 236, 170, 230), width=2)
            draw.line([x, y - r * 1.7, x, y + r * 1.7], fill=(255, 236, 170, 230), width=2)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 139, 42, 190), outline=(70, 40, 20, 210), width=2)
    draw.rounded_rectangle([38, output_size[1] - 70, 378, output_size[1] - 24], radius=10, fill=(255, 255, 255, 210), outline=(195, 205, 215, 190), width=1)
    draw.rectangle([56, output_size[1] - 56, 78, output_size[1] - 40], fill=(46, 132, 218, 235))
    draw.text((86, output_size[1] - 59), "object / component A", fill=(54, 64, 76, 255))
    draw.rectangle([230, output_size[1] - 56, 252, output_size[1] - 40], fill=(232, 111, 81, 235))
    draw.text((260, output_size[1] - 59), "object / component B", fill=(54, 64, 76, 255))
    draw.text((38, output_size[1] - 98), f"sequence t={t_label:.2f} | orange halo/rings: positive CCD witness region", fill=(54, 64, 76, 255))
    return image


def render_pair_interpolated_frame(
    scene: str,
    t: float,
    v0: np.ndarray,
    v1: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    scale: float,
    contacts: np.ndarray,
    object_ids: np.ndarray | None = None,
) -> Image.Image:
    vertices = (1.0 - t) * v0 + t * v1
    pair_contacts = (1.0 - t) * contacts[:, 0, :] + t * contacts[:, 1, :] if contacts.size else np.zeros((0, 3), dtype=np.float64)
    return render_mesh_frame(scene, t, vertices, faces, center, scale, pair_contacts, object_ids=object_ids)


def read_contact_points(query_csv: Path, bool_json: Path, limit: int = 48) -> np.ndarray:
    from convert_scalable_ccd_sample_to_p2c_groups import iter_query_vertices

    labels = [bool(x) for x in json.loads(bool_json.read_text(encoding="utf-8"))]
    points: list[np.ndarray] = []
    for index, (p0, p1, _) in enumerate(iter_query_vertices(query_csv)):
        if index < len(labels) and labels[index]:
            kind = "ee" if query_csv.stem.endswith("ee") else "vf"
            point = approximate_query_contact(p0, p1, kind)
            points.append(np.stack([point, point], axis=0))
        if len(points) >= limit:
            break
    if not points:
        return np.zeros((0, 2, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def read_scene_step_contacts(scene: str, timestep: int, limit: int = 48) -> np.ndarray:
    points: list[np.ndarray] = []
    for kind in ("ee", "vf"):
        query_csv = FULL_DATASET_ROOT / scene / "queries" / f"{timestep}{kind}.csv"
        bool_json = FULL_DATASET_ROOT / scene / "mma_bool" / f"{timestep}{kind}_mma_bool.json"
        if not query_csv.exists() or not bool_json.exists():
            continue
        labels = [bool(x) for x in json.loads(bool_json.read_text(encoding="utf-8"))]
        from convert_scalable_ccd_sample_to_p2c_groups import iter_query_vertices

        for index, (p0, p1, _) in enumerate(iter_query_vertices(query_csv)):
            if index < len(labels) and labels[index]:
                points.append(approximate_query_contact(p0, p1, kind))
                if len(points) >= limit:
                    return np.asarray(points, dtype=np.float64)
    if points:
        return np.asarray(points, dtype=np.float64)
    return np.zeros((0, 3), dtype=np.float64)


def write_mp4(path: Path, frames: list[Image.Image], fps: int = VISUALIZATION_FPS) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(frames[0].convert("RGB"))
    h, w = first.shape[:2]
    tmp_path = path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {tmp_path}")
    for frame in frames:
        rgb = np.asarray(frame.convert("RGB"))
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(tmp_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-crf",
        "18",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        tmp_path.replace(path)
    elif tmp_path.exists():
        tmp_path.unlink()


def render_visualizations(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = discover_groups(SOURCE_ROOT)
    selected: dict[str, Any] = {}
    for group in groups:
        selected.setdefault(group.scene, group)
    mp4_paths: list[str] = []
    sheet_frames: list[Image.Image] = []
    sequence_metadata: dict[str, Any] = {}
    for scene, group in selected.items():
        if group.frame0 is None or group.frame1 is None:
            continue
        sequence = scene_frame_sequence(scene, int(group.timestep))
        if len(sequence) > 2:
            sampled_indices = np.rint(np.linspace(0, len(sequence) - 1, VISUALIZATION_FRAME_COUNT)).astype(np.int64)
            unique_indices = sorted(set(int(index) for index in sampled_indices))
            vertices_by_index: dict[int, np.ndarray] = {}
            contacts_by_index: dict[int, np.ndarray] = {}
            faces: np.ndarray | None = None
            min_corner: np.ndarray | None = None
            max_corner: np.ndarray | None = None
            for index in unique_indices:
                vertices, loaded_faces = load_mesh(sequence[index], scene)
                vertices_by_index[index] = vertices
                number = frame_number(sequence[index])
                contacts_by_index[index] = read_scene_step_contacts(scene, int(number)) if number is not None else np.zeros((0, 3), dtype=np.float64)
                if faces is None:
                    faces = loaded_faces
                min_corner = vertices.min(axis=0) if min_corner is None else np.minimum(min_corner, vertices.min(axis=0))
                max_corner = vertices.max(axis=0) if max_corner is None else np.maximum(max_corner, vertices.max(axis=0))
            assert faces is not None and min_corner is not None and max_corner is not None
            first_vertices = vertices_by_index[unique_indices[0]]
            object_ids = face_object_ids(first_vertices, faces)
            center = 0.5 * (min_corner + max_corner)
            projected_bounds: list[np.ndarray] = []
            for vertices in vertices_by_index.values():
                projected_bounds.append(project(vertices, center, 1.0, (1280, 720))[0])
            all_xy = np.vstack(projected_bounds)
            span = np.maximum(all_xy.max(axis=0) - all_xy.min(axis=0), 1.0)
            scale = min(1040.0 / span[0], 470.0 / span[1]) * SCENE_ZOOM_MULTIPLIER.get(scene, 1.0)
            frames = [
                render_mesh_frame(
                    scene,
                    float(out_index) / float(max(1, VISUALIZATION_FRAME_COUNT - 1)),
                    vertices_by_index[int(source_index)],
                    faces,
                    center,
                    scale,
                    contacts_by_index[int(source_index)],
                    object_ids=object_ids,
                )
                for out_index, source_index in enumerate(sampled_indices)
            ]
            sequence_metadata[scene] = {
                "mode": "full_dataset_real_frames",
                "contact_mode": "per-rendered-frame nearest-point positive CCD contacts from matching full-dataset queries",
                "zoom_multiplier": SCENE_ZOOM_MULTIPLIER.get(scene, 1.0),
                "source_frame_count": len(sequence),
                "rendered_frame_count": VISUALIZATION_FRAME_COUNT,
                "first_frame": rel(sequence[0]),
                "last_frame": rel(sequence[-1]),
            }
        else:
            contacts = read_contact_points(group.query_csv, group.bool_json)
            v0, v1, faces = load_mesh_pair(group.frame0, group.frame1, scene)
            object_ids = face_object_ids(v0, faces)
            all_vertices = np.vstack([v0, v1])
            center = 0.5 * (all_vertices.min(axis=0) + all_vertices.max(axis=0))
            xy, _ = project(all_vertices, center, 1.0, (1280, 720))
            span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1.0)
            scale = min(1040.0 / span[0], 470.0 / span[1]) * SCENE_ZOOM_MULTIPLIER.get(scene, 1.0)
            frames = [
                render_pair_interpolated_frame(scene, float(t), v0, v1, faces, center, scale, contacts, object_ids=object_ids)
                for t in np.linspace(0.0, 1.0, VISUALIZATION_FRAME_COUNT)
            ]
            sequence_metadata[scene] = {
                "mode": "sample_two_frame_interpolation",
                "contact_mode": "nearest-point positive CCD contacts from sample queries",
                "zoom_multiplier": SCENE_ZOOM_MULTIPLIER.get(scene, 1.0),
                "source_frame_count": len(sequence),
                "rendered_frame_count": VISUALIZATION_FRAME_COUNT,
                "first_frame": rel(group.frame0),
                "last_frame": rel(group.frame1),
            }
        mp4_path = output_dir / f"{safe_name(scene)}.mp4"
        write_mp4(mp4_path, frames)
        mp4_paths.append(rel(mp4_path))
        sheet_frames.append(frames[len(frames) // 2].resize((640, 360)))
    if sheet_frames:
        cols = 2
        rows = int(math.ceil(len(sheet_frames) / cols))
        sheet = Image.new("RGB", (cols * 640, rows * 360), (246, 248, 249))
        for i, frame in enumerate(sheet_frames):
            sheet.paste(frame, ((i % cols) * 640, (i // cols) * 360))
        sheet_path = output_dir / "contact_sheet.png"
        sheet.save(sheet_path)
    else:
        sheet_path = output_dir / "contact_sheet.png"
    return {
        "visualization_dir": rel(output_dir),
        "mp4_paths": mp4_paths,
        "contact_sheet": rel(sheet_path),
        "sequence_metadata": sequence_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, benchmark, and visualize Scalable-CCD converted P2C scene groups.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--model-preset", default="tiny_mlp")
    args = parser.parse_args()
    started = time.perf_counter()

    if args.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                args.device = "cpu"
        except Exception:
            args.device = "cpu"

    manifest = write_split_shards(CONVERTED_SHARD_DIR, SPLIT_SHARD_DIR)
    training = run_tight_inclusion_stpf_training(
        SPLIT_SHARD_DIR,
        run_name=RUN_NAME,
        report_name=f"{RUN_NAME}_training",
        output_dir=OUTPUT_DIR,
        report_dir=BENCHMARK_DIR,
        model_preset=args.model_preset,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_split="train",
        validation_split="validation",
        uncertainty_weight=0.25,
    )
    checkpoint = Path(str(training["model_state_path"]))
    benchmark_rows: list[MethodRow] = []
    for split in ("train", "validation", "heldout_test", "all_scene"):
        shard = SPLIT_SHARD_DIR / f"{split}.npz"
        arrays, _ = load_npz(shard)
        labels = np.asarray(arrays["ground_truth"], dtype=np.bool_)
        costs = np.asarray(arrays["costs"], dtype=np.float64)
        rng = np.random.default_rng(fixed_seed)
        random_scores = rng.random(labels.shape[0])
        score_map = {
            "TrainedSTPFGroupEarlyStop": trained_scores(checkpoint, arrays, args.device),
            "HeuristicProximityGroupEarlyStop": np.asarray(arrays["features"][:, 22], dtype=np.float64),
            "RandomUniformExpectedOneSeed": random_scores,
            "OraclePositiveFirst": labels.astype(np.float64) + 1.0e-6 / np.maximum(costs, 1.0e-9),
        }
        benchmark_rows.extend(benchmark_scores(split, arrays, score_map))
    visual = render_visualizations(MYDEMO_DIR)
    metadata = {
        "run_name": RUN_NAME,
        "converted_shard_dir": rel(CONVERTED_SHARD_DIR),
        "split_shard_dir": rel(SPLIT_SHARD_DIR),
        "training_report": rel(Path(str(training["report_path"]))),
        "model_state_path": rel(checkpoint),
        "manifest": manifest,
        "visualization_dir": visual["visualization_dir"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_benchmark(benchmark_rows, BENCHMARK_DIR / RUN_NAME, metadata)
    summary = {
        **metadata,
        "benchmark_report": rel(BENCHMARK_DIR / f"{RUN_NAME}.md"),
        "benchmark_json": rel(BENCHMARK_DIR / f"{RUN_NAME}.json"),
        "benchmark_csv": rel(BENCHMARK_DIR / f"{RUN_NAME}.csv"),
        **visual,
    }
    (BENCHMARK_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
