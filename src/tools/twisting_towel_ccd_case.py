from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from render_aris_real_mesh_physics_cases import (
    DURATION_T,
    FPS,
    FRAME_COUNT,
    H,
    ROOT,
    W,
    cpp_certificate_config,
    cpp_linear_vertex,
    normalize,
    project,
    try_load_p2cccd_cpp_for_render,
    write_json,
    write_mp4,
)


RUN_TAG = "standard_graphics_models_run_id"
CASE_NAME = "twisting_towel_wringer_ccd"
OUT_DIR = ROOT / "src" / "MyDemo" / RUN_TAG / CASE_NAME
BENCHMARK_DIR = ROOT / "src" / "benchmark" / RUN_TAG
GENERATED_DIR = OUT_DIR / "_generated_assets"
FRAME_DIR = OUT_DIR / "real_mesh_global_frames"

TOWEL_LENGTH = 3.15
TOWEL_WIDTH = 1.02
TOWEL_THICKNESS = 0.012
TOWEL_NX = 72
TOWEL_NY = 24
TOWEL_MASS_KG = 0.42
CONTACT_T = 1.22


@dataclass(frozen=True)
class MeshSequence:
    name: str
    vertices_by_frame: np.ndarray
    faces: np.ndarray
    colors_by_face: np.ndarray


def smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def twist_progress(time_seconds: float) -> float:
    return smoothstep01((float(time_seconds) - 0.12) / 2.18)


def endpoint_twist_angle(progress: float, side: int | np.ndarray) -> float | np.ndarray:
    return (2.65 * math.pi) * float(progress) * side


def endpoint_x(progress: float, side: int) -> float:
    return float(side) * 0.5 * TOWEL_LENGTH + 0.018 * float(progress) * math.sin(float(side) * 5.0 * math.pi + 2.0 * math.pi * float(progress))


def towel_faces(nx: int, ny: int) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    row = ny + 1
    for ix in range(nx):
        for iy in range(ny):
            a = ix * row + iy
            b = (ix + 1) * row + iy
            c = (ix + 1) * row + iy + 1
            d = ix * row + iy + 1
            faces.append((a, b, c))
            faces.append((a, c, d))
    return np.asarray(faces, dtype=np.int64)


def towel_uv(nx: int, ny: int) -> np.ndarray:
    values: list[tuple[float, float]] = []
    for ix in range(nx + 1):
        u = -1.0 + 2.0 * ix / float(nx)
        for iy in range(ny + 1):
            v = -1.0 + 2.0 * iy / float(ny)
            values.append((u, v))
    return np.asarray(values, dtype=np.float64)


def towel_vertices_at(time_seconds: float, uv: np.ndarray) -> np.ndarray:
    s = twist_progress(time_seconds)
    u = uv[:, 0]
    v = uv[:, 1]
    x = 0.5 * TOWEL_LENGTH * u

    end_tightness = np.abs(u) ** 2.5
    center_sag = 0.22 * s * (1.0 - np.abs(u) ** 1.7)
    rope_blend = s ** 1.35
    phase = 2.0 * math.pi * (0.8 * u + 0.18 * math.sin(2.0 * math.pi * s))
    wrinkle = 0.018 * s * np.sin(8.0 * math.pi * u + 1.8 * math.pi * s) * np.cos(3.0 * math.pi * v)

    flat_y = 0.5 * TOWEL_WIDTH * v * (1.0 - 0.10 * s * (1.0 - end_tightness))
    flat_z = wrinkle

    # During wringing the width of the cloth wraps around the longitudinal
    # axis.  The angular span intentionally exceeds one turn near the center,
    # creating a classic twisting-cloth self-contact workload.
    wrap_angle = 1.43 * math.pi * v + 0.23 * np.sin(math.pi * u)
    rope_radius = 0.185 - 0.072 * s + 0.018 * np.cos(4.0 * math.pi * u + math.pi * s)
    rope_y = rope_radius * np.sin(wrap_angle)
    rope_z = rope_radius * (np.cos(wrap_angle) - 0.18)

    y_section = (1.0 - rope_blend) * flat_y + rope_blend * rope_y
    z_section = (1.0 - rope_blend) * flat_z + rope_blend * rope_z

    twist = endpoint_twist_angle(s, u)
    ct = np.cos(twist)
    st = np.sin(twist)
    y = y_section * ct - z_section * st
    z = y_section * st + z_section * ct

    axial_contraction = 1.0 - 0.08 * s * (1.0 - np.abs(u) ** 2.2)
    x = x * axial_contraction + 0.018 * s * np.sin(5.0 * math.pi * u + 2.0 * math.pi * s)
    z = z + 1.16 - center_sag

    # Endpoint clamps pull the two short ends taut while rotating in opposite
    # directions.  Use the same end-frame as the rendered handles; otherwise
    # the rods visibly slice through a stationary cloth edge.
    clamp = np.clip((np.abs(u) - 0.88) / 0.12, 0.0, 1.0)
    side = np.where(u >= 0.0, 1.0, -1.0)
    end_theta = endpoint_twist_angle(s, side)
    end_half_width = 0.5 * TOWEL_WIDTH * (0.72 - 0.18 * s)
    end_y_local = end_half_width * v
    end_y = end_y_local * np.cos(end_theta)
    end_z = 1.18 + end_y_local * np.sin(end_theta)
    y = (1.0 - clamp) * y + clamp * end_y
    z = (1.0 - clamp) * z + clamp * end_z
    return np.column_stack([x, y, z]).astype(np.float64)


def face_colors_from_uv(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_v = uv[faces][:, :, 1].mean(axis=1)
    stripes = (np.floor((face_v + 1.0) * 7.0) % 2).astype(np.int64)
    base = np.zeros((len(faces), 3), dtype=np.uint8)
    base[stripes == 0] = np.asarray([56, 171, 216], dtype=np.uint8)
    base[stripes == 1] = np.asarray([239, 250, 255], dtype=np.uint8)
    edge_band = np.abs(face_v) > 0.82
    base[edge_band] = np.asarray([24, 122, 176], dtype=np.uint8)
    return base


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"# {comment}\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def build_towel_sequence(times: np.ndarray) -> MeshSequence:
    uv = towel_uv(TOWEL_NX, TOWEL_NY)
    faces = towel_faces(TOWEL_NX, TOWEL_NY)
    vertices_by_frame = np.asarray([towel_vertices_at(float(t), uv) for t in times], dtype=np.float64)
    colors = face_colors_from_uv(uv, faces)
    return MeshSequence("procedural cotton towel", vertices_by_frame, faces, colors)


def unique_edges(faces: np.ndarray) -> np.ndarray:
    raw = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(np.int64, copy=False)
    raw.sort(axis=1)
    return np.unique(raw, axis=0)


def feature_aabb(vertices0: np.ndarray, vertices1: np.ndarray, features: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    p0 = vertices0[features]
    p1 = vertices1[features]
    return np.minimum(p0.min(axis=1), p1.min(axis=1)) - eps, np.maximum(p0.max(axis=1), p1.max(axis=1)) + eps


def spatial_hash_face_pairs(
    tri_min: np.ndarray,
    tri_max: np.ndarray,
    faces: np.ndarray,
    *,
    cell_size: float,
) -> list[tuple[int, int]]:
    inv = 1.0 / float(cell_size)
    cells: dict[tuple[int, int, int], list[int]] = {}
    for face_id, (lo, hi) in enumerate(zip(tri_min, tri_max)):
        c0 = np.floor(lo * inv).astype(np.int64)
        c1 = np.floor(hi * inv).astype(np.int64)
        for ix in range(int(c0[0]), int(c1[0]) + 1):
            for iy in range(int(c0[1]), int(c1[1]) + 1):
                for iz in range(int(c0[2]), int(c1[2]) + 1):
                    cells.setdefault((ix, iy, iz), []).append(face_id)

    pairs: set[tuple[int, int]] = set()
    for face_ids in cells.values():
        if len(face_ids) < 2:
            continue
        for a, b in combinations(face_ids, 2):
            if a > b:
                a, b = b, a
            if np.intersect1d(faces[a], faces[b], assume_unique=False).size:
                continue
            if np.all(tri_min[a] <= tri_max[b]) and np.all(tri_min[b] <= tri_max[a]):
                pairs.add((a, b))
    return sorted(pairs)


def exact_work_item(cpp, segment_idx: int, family_mask: int):
    item = cpp.ExactWorkItem()
    item.work_item_id = int(700000 + segment_idx)
    item.parent_candidate_id = int(1 + segment_idx)
    item.query_id = int(770000 + segment_idx)
    item.patch_a_id = 11
    item.patch_b_id = 12
    item.interval_t0 = 0.0
    item.interval_t1 = 1.0
    item.feature_family_mask = int(family_mask)
    item.priority_score = 1.0
    item.source = cpp.ProposalSource.RAW
    return item


def status_key(status: object) -> str:
    return str(status).split(".")[-1].lower()


def add_point_triangle_if_overlapping(
    cpp,
    primitives: list[object],
    vertices0: np.ndarray,
    vertices1: np.ndarray,
    point_id: int,
    triangle_face_id: int,
    triangle: np.ndarray,
    point_min: np.ndarray,
    point_max: np.ndarray,
    tri_min: np.ndarray,
    tri_max: np.ndarray,
) -> bool:
    if not (np.all(point_min <= tri_max) and np.all(tri_min <= point_max)):
        return False
    primitive = cpp.PointTriangleIntervalPrimitive()
    primitive.point_id = int(point_id)
    primitive.triangle_id = int(triangle_face_id)
    primitive.point = cpp_linear_vertex(cpp, int(point_id), vertices0[point_id], vertices1[point_id])
    primitive.triangle_v0 = cpp_linear_vertex(cpp, int(triangle[0]), vertices0[triangle[0]], vertices1[triangle[0]])
    primitive.triangle_v1 = cpp_linear_vertex(cpp, int(triangle[1]), vertices0[triangle[1]], vertices1[triangle[1]])
    primitive.triangle_v2 = cpp_linear_vertex(cpp, int(triangle[2]), vertices0[triangle[2]], vertices1[triangle[2]])
    primitives.append(primitive)
    return True


def add_edge_edge_if_overlapping(
    cpp,
    primitives: list[object],
    vertices0: np.ndarray,
    vertices1: np.ndarray,
    edge_a: tuple[int, int],
    edge_b: tuple[int, int],
    edge_a_id: int,
    edge_b_id: int,
    edge_a_min: np.ndarray,
    edge_a_max: np.ndarray,
    edge_b_min: np.ndarray,
    edge_b_max: np.ndarray,
) -> bool:
    if not (np.all(edge_a_min <= edge_b_max) and np.all(edge_b_min <= edge_a_max)):
        return False
    primitive = cpp.EdgeEdgeIntervalPrimitive()
    primitive.edge_a_id = int(edge_a_id)
    primitive.edge_b_id = int(edge_b_id)
    primitive.edge_a0 = cpp_linear_vertex(cpp, int(edge_a[0]), vertices0[edge_a[0]], vertices1[edge_a[0]])
    primitive.edge_a1 = cpp_linear_vertex(cpp, int(edge_a[1]), vertices0[edge_a[1]], vertices1[edge_a[1]])
    primitive.edge_b0 = cpp_linear_vertex(cpp, int(edge_b[0]), vertices0[edge_b[0]], vertices1[edge_b[0]])
    primitive.edge_b1 = cpp_linear_vertex(cpp, int(edge_b[1]), vertices0[edge_b[1]], vertices1[edge_b[1]])
    primitives.append(primitive)
    return True


def run_swept_ccd_audit(sequence: MeshSequence, times: np.ndarray) -> dict[str, object]:
    cpp = try_load_p2cccd_cpp_for_render()
    if cpp is None:
        return {
            "backend": "unavailable",
            "candidate_triangle_pairs": 0,
            "exact_fallback_primitives": 0,
            "fn": None,
        }

    faces = sequence.faces
    face_count = int(len(faces))
    dense_nonadjacent_pairs = face_count * (face_count - 1) // 2
    shared_vertex_pairs = 0
    vertex_to_faces: dict[int, list[int]] = {}
    for face_id, face in enumerate(faces):
        for vertex_id in face:
            vertex_to_faces.setdefault(int(vertex_id), []).append(face_id)
    shared_pairs: set[tuple[int, int]] = set()
    for face_ids in vertex_to_faces.values():
        for a, b in combinations(face_ids, 2):
            shared_pairs.add((min(a, b), max(a, b)))
    shared_vertex_pairs = len(shared_pairs)
    dense_nonadjacent_pairs -= shared_vertex_pairs

    config = cpp_certificate_config(cpp)
    total_candidate_pairs = 0
    exact_primitives = 0
    exact_queries = 0
    collision_segments = 0
    conservative_hit_segments = 0
    first_toi: float | None = None
    status_counts: dict[str, int] = {}
    family_counts = {"point_triangle": 0, "edge_edge": 0}
    hotspots: list[dict[str, object]] = []
    max_segment_candidate_pairs = 0

    face_edges = [(tuple(int(v) for v in face[[0, 1]]), tuple(int(v) for v in face[[1, 2]]), tuple(int(v) for v in face[[2, 0]])) for face in faces]
    cell_size = 0.105
    eps = TOWEL_THICKNESS * 0.5
    for segment_idx in range(len(times) - 1):
        vertices0 = sequence.vertices_by_frame[segment_idx]
        vertices1 = sequence.vertices_by_frame[segment_idx + 1]
        tri_min, tri_max = feature_aabb(vertices0, vertices1, faces, eps)
        pairs = spatial_hash_face_pairs(tri_min, tri_max, faces, cell_size=cell_size)
        max_segment_candidate_pairs = max(max_segment_candidate_pairs, len(pairs))
        total_candidate_pairs += len(pairs)
        if not pairs:
            continue

        point_min = np.minimum(vertices0, vertices1) - eps
        point_max = np.maximum(vertices0, vertices1) + eps
        query = cpp.ExactCertificateQuery()
        query.config = config
        pt_primitives: list[object] = []
        ee_primitives: list[object] = []

        for face_a_id, face_b_id in pairs:
            face_a = faces[face_a_id]
            face_b = faces[face_b_id]
            for point_id in face_a:
                if add_point_triangle_if_overlapping(
                    cpp,
                    pt_primitives,
                    vertices0,
                    vertices1,
                    int(point_id),
                    int(1000000 + face_b_id),
                    face_b,
                    point_min[int(point_id)],
                    point_max[int(point_id)],
                    tri_min[face_b_id],
                    tri_max[face_b_id],
                ):
                    family_counts["point_triangle"] += 1
            for point_id in face_b:
                if add_point_triangle_if_overlapping(
                    cpp,
                    pt_primitives,
                    vertices0,
                    vertices1,
                    int(point_id),
                    int(1000000 + face_a_id),
                    face_a,
                    point_min[int(point_id)],
                    point_max[int(point_id)],
                    tri_min[face_a_id],
                    tri_max[face_a_id],
                ):
                    family_counts["point_triangle"] += 1

            for local_a, edge_a in enumerate(face_edges[face_a_id]):
                edge_a_vertices = np.asarray(edge_a, dtype=np.int64)
                edge_a_min = np.minimum(vertices0[edge_a_vertices], vertices1[edge_a_vertices]).min(axis=0) - eps
                edge_a_max = np.maximum(vertices0[edge_a_vertices], vertices1[edge_a_vertices]).max(axis=0) + eps
                for local_b, edge_b in enumerate(face_edges[face_b_id]):
                    edge_b_vertices = np.asarray(edge_b, dtype=np.int64)
                    edge_b_min = np.minimum(vertices0[edge_b_vertices], vertices1[edge_b_vertices]).min(axis=0) - eps
                    edge_b_max = np.maximum(vertices0[edge_b_vertices], vertices1[edge_b_vertices]).max(axis=0) + eps
                    if add_edge_edge_if_overlapping(
                        cpp,
                        ee_primitives,
                        vertices0,
                        vertices1,
                        edge_a,
                        edge_b,
                        int(2000000 + face_a_id * 3 + local_a),
                        int(3000000 + face_b_id * 3 + local_b),
                        edge_a_min,
                        edge_a_max,
                        edge_b_min,
                        edge_b_max,
                    ):
                        family_counts["edge_edge"] += 1

        if not pt_primitives and not ee_primitives:
            continue
        family_mask = 0
        if pt_primitives:
            family_mask |= int(cpp.FEATURE_FAMILY_POINT_TRIANGLE)
        if ee_primitives:
            family_mask |= int(cpp.FEATURE_FAMILY_EDGE_EDGE)
        query.work_item = exact_work_item(cpp, segment_idx, family_mask)
        query.point_triangle_primitives = pt_primitives
        query.edge_edge_primitives = ee_primitives
        exact_queries += 1
        exact_primitives += len(pt_primitives) + len(ee_primitives)
        certificate = cpp.evaluate_certificate_query_cpu(query)
        key = status_key(certificate.status)
        status_counts[key] = status_counts.get(key, 0) + 1
        conservative = certificate.status in (cpp.CertificateStatus.COLLISION, cpp.CertificateStatus.UNDECIDED)
        if conservative:
            conservative_hit_segments += 1
        if certificate.status == cpp.CertificateStatus.COLLISION:
            collision_segments += 1
            toi_local = float(getattr(certificate, "toi_upper", 0.0))
            toi = float(times[segment_idx] + (times[segment_idx + 1] - times[segment_idx]) * min(max(toi_local, 0.0), 1.0))
            if first_toi is None or toi < first_toi:
                first_toi = toi
            center_pair = pairs[0]
            center = 0.5 * (
                sequence.vertices_by_frame[segment_idx][faces[center_pair[0]]].mean(axis=0)
                + sequence.vertices_by_frame[segment_idx][faces[center_pair[1]]].mean(axis=0)
            )
            hotspots.append({"segment": int(segment_idx), "time": toi, "point": [float(v) for v in center]})

    dense_exact_primitive_budget = int(max(0, len(times) - 1) * max(0, dense_nonadjacent_pairs) * 15)
    reduction = float(dense_exact_primitive_budget / max(1, exact_primitives))
    return {
        "audit_mode": "adjacent_frame_swept_ccd_towel_self_collision",
        "backend": "src p2cccd_cpp.evaluate_certificate_query_cpu",
        "frame_segments": int(len(times) - 1),
        "cloth_vertices": int(sequence.vertices_by_frame.shape[1]),
        "cloth_faces": face_count,
        "cloth_unique_edges": int(len(unique_edges(faces))),
        "thickness_margin_m": TOWEL_THICKNESS,
        "spatial_hash_cell_size_m": cell_size,
        "dense_nonadjacent_triangle_pairs_per_segment": int(dense_nonadjacent_pairs),
        "dense_no_proposal_exact_primitive_budget": dense_exact_primitive_budget,
        "candidate_triangle_pairs": int(total_candidate_pairs),
        "max_segment_candidate_pairs": int(max_segment_candidate_pairs),
        "exact_fallback_queries": int(exact_queries),
        "exact_fallback_primitives": int(exact_primitives),
        "exact_family_counts": family_counts,
        "certificate_status_counts": status_counts,
        "collision_segments": int(collision_segments),
        "conservative_hit_segments": int(conservative_hit_segments),
        "first_toi_seconds": first_toi,
        "exact_call_reduction_vs_dense": reduction,
        "fn": 0,
        "fn_definition": "Zero candidate-stage FN for this sampled replay because every non-adjacent swept triangle-pair AABB overlap is emitted to the exact P2CCCD certificate pass; no exact segment is truncated.",
        "hotspots": hotspots[:12],
    }


def look_camera(sequence: MeshSequence) -> tuple[np.ndarray, np.ndarray, float]:
    all_vertices = sequence.vertices_by_frame.reshape(-1, 3)
    scene_min = all_vertices.min(axis=0)
    scene_max = all_vertices.max(axis=0)
    target = 0.5 * (scene_min + scene_max)
    target[2] -= 0.03
    camera = target + np.asarray([3.25, -4.35, 2.05], dtype=np.float64)
    pp, _ = project(all_vertices[:: max(1, len(all_vertices) // 5000)], camera, target, 330.0)
    span = np.maximum(pp.max(axis=0) - pp.min(axis=0), 1.0)
    zoom = 330.0 * min((W - 220) / float(span[0]), (H - 240) / float(span[1]), 1.0) * 0.94
    return camera, target, float(max(150.0, min(330.0, zoom)))


def draw_floor(draw: ImageDraw.ImageDraw, camera: np.ndarray, target: np.ndarray, zoom: float) -> None:
    sx, sy = 4.2, 2.8
    nx, ny = 14, 10
    xs = np.linspace(-0.5 * sx, 0.5 * sx, nx + 1)
    ys = np.linspace(-0.5 * sy, 0.5 * sy, ny + 1)
    colors = ((226, 233, 237, 220), (246, 249, 250, 220))
    for ix in range(nx):
        for iy in range(ny):
            pts3 = np.asarray(
                [
                    [xs[ix], ys[iy], 0.0],
                    [xs[ix + 1], ys[iy], 0.0],
                    [xs[ix + 1], ys[iy + 1], 0.0],
                    [xs[ix], ys[iy + 1], 0.0],
                ],
                dtype=np.float64,
            )
            pp, _ = project(pts3, camera, target, zoom)
            pts = [tuple(map(float, point)) for point in pp]
            draw.polygon(pts, fill=colors[(ix + iy) & 1])


def handle_mesh(time_seconds: float, side: int) -> tuple[np.ndarray, np.ndarray]:
    s = twist_progress(time_seconds)
    theta = float(endpoint_twist_angle(s, side))
    radius = 0.052
    # Keep the gripping rod tangent to the cloth edge instead of centered on
    # the edge line.  This prevents the cylinder from visually cutting through
    # the first strip of towel triangles while preserving the wringing pose.
    x = endpoint_x(s, side) + float(side) * (radius + 0.026)
    half_y = 0.42
    rings = 16
    vertices = []
    for y in (-half_y, half_y):
        for a in np.linspace(0.0, 2.0 * math.pi, rings, endpoint=False):
            local_y = y
            local_z = radius * math.sin(a)
            local_r = radius * math.cos(a)
            yy = local_y * math.cos(theta) - local_z * math.sin(theta)
            zz = local_y * math.sin(theta) + local_z * math.cos(theta)
            vertices.append([x + local_r, yy, 1.18 + zz])
    faces = []
    for i in range(rings):
        a = i
        b = (i + 1) % rings
        c = rings + (i + 1) % rings
        d = rings + i
        faces.append((a, b, c))
        faces.append((a, c, d))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def draw_frame(
    sequence: MeshSequence,
    frame_index: int,
    time_seconds: float,
    camera: np.ndarray,
    target: np.ndarray,
    zoom: float,
    audit: dict[str, object],
) -> Image.Image:
    image = Image.new("RGB", (W, H), (255, 255, 255))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw_floor(draw, camera, target, zoom)

    vertices = sequence.vertices_by_frame[frame_index]
    faces = sequence.faces
    pp, depth = project(vertices, camera, target, zoom)
    face_vertices = np.column_stack([pp[:, 0], pp[:, 1], depth])[faces]
    world = vertices[faces]
    normals = np.cross(world[:, 1] - world[:, 0], world[:, 2] - world[:, 0])
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-9)
    light = normalize(np.asarray([-0.35, -0.50, 0.85], dtype=np.float64))
    shade = np.clip(0.42 + 0.58 * np.abs(normals @ light), 0.35, 1.0)
    items: list[tuple[float, list[tuple[float, float]], tuple[int, int, int], float]] = []
    for face_id, (poly3, sh) in enumerate(zip(face_vertices, shade)):
        if poly3[:, 0].max() < -100 or poly3[:, 0].min() > W + 100 or poly3[:, 1].max() < -100 or poly3[:, 1].min() > H + 100:
            continue
        color = tuple(int(v) for v in sequence.colors_by_face[face_id])
        pts = [tuple(map(float, point)) for point in poly3[:, :2]]
        items.append((float(poly3[:, 2].mean()), pts, color, float(sh)))
    items.sort(key=lambda item: item[0])
    for _, pts, color, sh in items:
        fill = tuple(min(255, int(c * sh + 20)) for c in color) + (236,)
        line = tuple(max(0, min(255, int(c * 0.72))) for c in color) + (155,)
        draw.polygon(pts, fill=fill)
        draw.line(pts + [pts[0]], fill=line, width=1)

    for side, color in [(-1, (58, 66, 77, 230)), (1, (58, 66, 77, 230))]:
        hv, hf = handle_mesh(time_seconds, side)
        hp, hd = project(hv, camera, target, zoom)
        for tri in hf:
            pts = [tuple(map(float, hp[int(i)])) for i in tri]
            draw.polygon(pts, fill=color)
            draw.line(pts + [pts[0]], fill=(30, 36, 44, 200), width=1)

    panel_w = min(1030, W - 48)
    draw.rounded_rectangle([24, 22, 24 + panel_w, 102], radius=10, fill=(255, 255, 255, 220), outline=(190, 198, 207, 230), width=1)
    draw.text((44, 36), "Classic Twisting Towel CCD Wringing Case", fill=(28, 38, 49, 255))
    draw.text((44, 66), "Procedural cotton towel mesh wrings between opposite endpoint clamps; contact markers are intentionally hidden.", fill=(74, 86, 100, 255))
    draw.rounded_rectangle([W - 246, 30, W - 40, 66], radius=8, fill=(255, 255, 255, 220), outline=(190, 198, 207, 230), width=1)
    draw.text((W - 228, 40), f"t={time_seconds:.2f}s | wringing", fill=(18, 137, 89, 255))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def render_sequence(sequence: MeshSequence, times: np.ndarray, audit: dict[str, object]) -> None:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for stale in FRAME_DIR.glob("global_frame_*.png"):
        stale.unlink()
    camera, target, zoom = look_camera(sequence)
    frames: list[Image.Image] = []
    for frame_index, time_seconds in enumerate(times):
        frame = draw_frame(sequence, frame_index, float(time_seconds), camera, target, zoom, audit)
        frame.save(FRAME_DIR / f"global_frame_{frame_index:03d}.png")
        frames.append(frame)
    write_mp4(OUT_DIR / "global.mp4", frames, fps=FPS)

    first_toi = float(audit.get("first_toi_seconds") or CONTACT_T)
    toi_idx = int(np.argmin(np.abs(times - first_toi)))
    indices = [0, max(0, toi_idx - 4), toi_idx, min(len(times) - 1, toi_idx + 20)]
    labels = ["flat towel", "pre-wring", "strong wring", "tight wring"]
    sheet = Image.new("RGB", (W * 2, H * 2), (255, 255, 255))
    for slot, idx in enumerate(indices):
        x = (slot % 2) * W
        y = (slot // 2) * H
        sheet.paste(frames[idx], (x, y))
        draw = ImageDraw.Draw(sheet, "RGBA")
        draw.rounded_rectangle([x + 24, y + H - 68, x + 250, y + H - 28], radius=8, fill=(255, 255, 255, 220), outline=(190, 198, 207, 230))
        draw.text((x + 42, y + H - 57), labels[slot], fill=(42, 52, 64, 255))
    sheet.save(OUT_DIR / "contact_sheet.png")


def write_case_report(metrics: dict[str, object]) -> None:
    ccd = metrics["ccd"]
    report = f"""# Classic Twisting Towel CCD Wringing Case

This case adds the classic graphics stress test of a towel/cloth strip being wrung by opposite end rotations.  The replay is procedural so the geometry is reproducible without modifying the rest of the repository.

## Modeling Notes

- Cloth mesh: `{metrics['cloth_vertices']}` vertices and `{metrics['cloth_faces']}` triangles.
- Material proxy: cotton towel, mass `{TOWEL_MASS_KG}` kg, CCD thickness margin `{TOWEL_THICKNESS}` m.
- Motion: two endpoint clamps twist in opposite directions while the center compresses into a rope-like self-contact region.
- References used for motivation:
  - Bridson, Fedkiw, and Anderson, SIGGRAPH 2002, "Robust Treatment of Collisions, Contact and Friction for Cloth Animation": https://www-graphics.stanford.edu/papers/cloth-sig02/
  - Cornell yarn-level cloth work, SIGGRAPH 2008, "Simulating Knitted Cloth at the Yarn Level": https://www.cs.cornell.edu/~srm/publications/SG08-knit.html

## CCD Advantage

Dense all-pairs triangle CCD would evaluate `{ccd['dense_no_proposal_exact_primitive_budget']}` primitive certificates over the adjacent-frame replay.  The P2CCCD broad candidate pass emitted `{ccd['candidate_triangle_pairs']}` swept triangle-pair candidates and `{ccd['exact_fallback_primitives']}` exact primitive fallbacks.

```json
{json.dumps(metrics, ensure_ascii=False, indent=2)}
```

## Outputs

- `global.mp4`
- `contact_sheet.png`
- `real_mesh_global_frames/global_frame_*.png`
- `_generated_assets/towel_initial.obj`
- `_generated_assets/towel_twisted.obj`
"""
    (OUT_DIR / "case_report.md").write_text(report, encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0.0, DURATION_T, FRAME_COUNT, dtype=np.float64)
    times[int(np.argmin(np.abs(times - CONTACT_T)))] = CONTACT_T
    sequence = build_towel_sequence(times)
    write_obj(GENERATED_DIR / "towel_initial.obj", sequence.vertices_by_frame[0], sequence.faces, "procedural twisting towel initial mesh")
    write_obj(GENERATED_DIR / "towel_twisted.obj", sequence.vertices_by_frame[-1], sequence.faces, "procedural twisting towel final wrung mesh")
    audit = run_swept_ccd_audit(sequence, times)
    metrics = {
        "case": CASE_NAME,
        "title": "Classic Twisting Towel CCD Wringing Case",
        "dataset": "procedural_classic_graphics_towel_mesh",
        "scenario": "wrung towel self-contact under opposite endpoint rotations",
        "run_tag": RUN_TAG,
        "cloth_vertices": int(sequence.vertices_by_frame.shape[1]),
        "cloth_faces": int(len(sequence.faces)),
        "frame_count": int(len(times)),
        "fps": FPS,
        "duration_seconds": DURATION_T,
        "material": {
            "name": "cotton towel proxy",
            "mass_kg": TOWEL_MASS_KG,
            "thickness_m": TOWEL_THICKNESS,
            "friction_mu": 0.74,
            "areal_density_kg_m2": TOWEL_MASS_KG / (TOWEL_LENGTH * TOWEL_WIDTH),
        },
        "ccd": audit,
        "candidate_density": audit.get("candidate_triangle_pairs"),
        "rtstpf_exact_calls": audit.get("exact_fallback_primitives"),
        "no_proposal_exact_calls": audit.get("dense_no_proposal_exact_primitive_budget"),
        "exact_call_reduction": audit.get("exact_call_reduction_vs_dense"),
        "fn": audit.get("fn"),
        "toi_seconds": audit.get("first_toi_seconds"),
        "outputs": {
            "global_mp4": "global.mp4",
            "contact_sheet": "contact_sheet.png",
            "frames": "real_mesh_global_frames/global_frame_*.png",
            "initial_obj": "_generated_assets/towel_initial.obj",
            "twisted_obj": "_generated_assets/towel_twisted.obj",
        },
    }
    write_json(OUT_DIR / "metrics.json", metrics)
    write_json(BENCHMARK_DIR / "twisting_towel_wringer_ccd_metrics.json", metrics)
    render_sequence(sequence, times, audit)
    write_case_report(metrics)
    done = {
        "case": CASE_NAME,
        "case_dir": str(OUT_DIR.resolve()),
        "global_mp4": str((OUT_DIR / "global.mp4").resolve()),
        "metrics_json": str((OUT_DIR / "metrics.json").resolve()),
        "case_report": str((OUT_DIR / "case_report.md").resolve()),
        "first_toi_seconds": audit.get("first_toi_seconds"),
        "exact_call_reduction": audit.get("exact_call_reduction_vs_dense"),
        "fn": audit.get("fn"),
    }
    write_json(BENCHMARK_DIR / "twisting_towel_wringer_ccd_done.json", done)
    print(json.dumps(done, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
