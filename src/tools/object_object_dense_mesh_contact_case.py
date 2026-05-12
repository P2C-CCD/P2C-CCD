from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont

from p2cccd.bench.bvh_exact import _try_load_p2cccd_cpp


ROOT = Path(__file__).resolve().parents[2]
CASE_NAME = "object_object_dense_mesh_contact_run_id"
BENCH_DIR = ROOT / "src" / "benchmark" / CASE_NAME
DEMO_DIR = ROOT / "src" / "MyDemo" / CASE_NAME
ASSET_DIR = DEMO_DIR / "assets"
OUTPUT_DIR = ROOT / "src" / "outputs" / "stpf_training" / CASE_NAME
SOURCE_CHAIR = (
    ROOT / "src"
    / "datasets"
    / "shapenet_core_v2"
    / "selected_ood_dense_run_id"
    / "03001627"
    / "e9371c17042131d93506b420c6bcd44"
    / "models"
    / "model_normalized.obj"
)


@dataclass(frozen=True)
class MethodResult:
    method: str
    groups: int
    candidates: int
    positive_groups: int
    exact_calls: int
    skipped_exact_calls: int
    tp: int
    tn: int
    fp: int
    fn: int
    recall: float
    precision: float
    first_positive_rank_mean: float
    proposal_ms: float
    exact_ms: float
    wall_ms: float


class TinySTPF(torch.nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def sampled_mesh(source: Path, face_count: int, seed: int) -> trimesh.Trimesh:
    loaded = trimesh.load(source, force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh):
        loaded = trimesh.util.concatenate([m for m in loaded.dump() if isinstance(m, trimesh.Trimesh)])
    mesh = clean_mesh(loaded)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    mesh.vertices = (vertices - center) / max(scale, 1.0e-12)
    rng = np.random.default_rng(seed)
    sample_count = min(int(face_count), int(len(mesh.faces)))
    faces = rng.choice(len(mesh.faces), size=sample_count, replace=False)
    return clean_mesh(mesh.submesh([faces], append=True, repair=False))


def coherent_display_mesh(source: Path, target_faces: int = 12000) -> trimesh.Trimesh:
    """Build a coherent display mesh from the real source OBJ.

    The old display mesh used randomly sampled faces, which is correct for
    stress-testing primitive groups but produces visually disconnected triangle
    fragments.  For paper video we keep the same real source geometry, merge
    duplicate OBJ vertices, remove duplicate faces, then use deterministic
    vertex clustering so the rendered object remains a continuous chair.
    """

    cache = ASSET_DIR / f"shapenet_chair_display_coherent_{target_faces}.obj"
    if cache.exists():
        loaded = trimesh.load(cache, force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            loaded = trimesh.util.concatenate([m for m in loaded.dump() if isinstance(m, trimesh.Trimesh)])
        return clean_mesh(loaded)

    loaded = trimesh.load(source, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        loaded = trimesh.util.concatenate([m for m in loaded.dump() if isinstance(m, trimesh.Trimesh)])
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)

    quantized = np.round(vertices, 6)
    vertices, inverse = np.unique(quantized, axis=0, return_inverse=True)
    faces = inverse[faces]
    valid = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[valid]
    sorted_faces = np.sort(faces, axis=1)
    _, unique_face_indices = np.unique(sorted_faces, axis=0, return_index=True)
    faces = faces[np.sort(unique_face_indices)]
    used, remap = np.unique(faces.reshape(-1), return_inverse=True)
    vertices = vertices[used]
    faces = remap.reshape((-1, 3))

    def cluster(resolution: int) -> tuple[np.ndarray, np.ndarray]:
        mins = vertices.min(axis=0)
        extent = vertices.max(axis=0) - mins
        scale = max(float(extent.max()), 1.0e-12)
        cells = np.floor((vertices - mins) / scale * resolution).astype(np.int64)
        stride = resolution + 1
        keys = cells[:, 0] + stride * cells[:, 1] + stride * stride * cells[:, 2]
        _, cluster_ids = np.unique(keys, return_inverse=True)
        counts = np.bincount(cluster_ids)
        clustered_vertices = np.column_stack(
            [np.bincount(cluster_ids, weights=vertices[:, axis]) / counts for axis in range(3)]
        )
        clustered_faces = cluster_ids[faces]
        valid_faces = (
            (clustered_faces[:, 0] != clustered_faces[:, 1])
            & (clustered_faces[:, 1] != clustered_faces[:, 2])
            & (clustered_faces[:, 0] != clustered_faces[:, 2])
        )
        clustered_faces = clustered_faces[valid_faces]
        sorted_clustered_faces = np.sort(clustered_faces, axis=1)
        _, unique_clustered_indices = np.unique(sorted_clustered_faces, axis=0, return_index=True)
        clustered_faces = clustered_faces[np.sort(unique_clustered_indices)]
        used_vertices, clustered_remap = np.unique(clustered_faces.reshape(-1), return_inverse=True)
        clustered_vertices = clustered_vertices[used_vertices]
        clustered_faces = clustered_remap.reshape((-1, 3))
        return clustered_vertices, clustered_faces

    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for resolution in (56, 64, 72, 80, 88, 96, 112):
        v_candidate, f_candidate = cluster(resolution)
        score = abs(len(f_candidate) - target_faces)
        if len(f_candidate) > target_faces * 1.15:
            score *= 1.5
        candidates.append((float(score), v_candidate, f_candidate))
    _, out_vertices, out_faces = min(candidates, key=lambda item: item[0])
    mesh = trimesh.Trimesh(vertices=out_vertices, faces=out_faces, process=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(cache)
    return clean_mesh(mesh)


def prepare_assets() -> dict[str, Path]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    exact_a = ASSET_DIR / "shapenet_chair_exact_a_320.obj"
    exact_b = ASSET_DIR / "shapenet_chair_exact_b_320.obj"
    display_a = ASSET_DIR / "shapenet_chair_display_a_5000.obj"
    display_b = ASSET_DIR / "shapenet_chair_display_b_5000.obj"
    if not exact_a.exists():
        sampled_mesh(SOURCE_CHAIR, 320, fixed_seed).export(exact_a)
    if not exact_b.exists():
        # A second rigid object instance from the same real mesh source.  Keeping
        # the exact geometry identical makes the primitive exact replay contain
        # genuine collision witnesses when the objects meet at TOI.
        shutil.copy2(exact_a, exact_b)
    if not display_a.exists():
        sampled_mesh(SOURCE_CHAIR, 5000, fixed_seed).export(display_a)
    if not display_b.exists():
        shutil.copy2(display_a, display_b)
    return {
        "exact_a": exact_a,
        "exact_b": exact_b,
        "display_a": display_a,
        "display_b": display_b,
        "source": SOURCE_CHAIR,
    }


def make_cpp_configs(cpp: Any) -> tuple[Any, Any, Any]:
    item = cpp.ExactWorkItem()
    item.work_item_id = 1
    item.parent_candidate_id = 1
    item.query_id = 1
    item.slab_id = 0
    item.patch_a_id = 0
    item.patch_b_id = 0
    item.interval_t0 = 0.0
    item.interval_t1 = 1.0
    item.feature_family_mask = int(cpp.FEATURE_FAMILY_POINT_TRIANGLE) | int(cpp.FEATURE_FAMILY_EDGE_EDGE)
    item.priority_score = 1.0
    item.source = cpp.ProposalSource.RAW

    cfg = cpp.CertificateEngineConfig()
    cfg.eps_time = 1.0e-6
    cfg.eps_space = 1.0e-8
    cfg.max_subdivision_depth = 8

    build = cpp.MeshExactBuildConfig()
    build.prune_by_swept_aabb = False
    build.max_point_triangle_primitives = 0
    build.max_edge_edge_primitives = 0
    return item, cfg, build


def build_exact_query(cpp: Any, assets: dict[str, Path]) -> Any:
    mesh_a = cpp.center_mesh_at_aabb_center(cpp.load_triangle_mesh(str(assets["exact_a"])))[0]
    mesh_b = cpp.center_mesh_at_aabb_center(cpp.load_triangle_mesh(str(assets["exact_b"])))[0]
    item, cfg, build = make_cpp_configs(cpp)
    build_start = time.perf_counter()
    result = cpp.build_mesh_exact_certificate_query(
        mesh_a,
        (-1.2, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        mesh_b,
        (1.2, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        item,
        cfg,
        build,
    )
    return result, cfg, (time.perf_counter() - build_start) * 1000.0


def _as_np(values: Any) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


def _trajectory_positions(traj: Any) -> tuple[np.ndarray, np.ndarray]:
    return _as_np(traj.position_t0), _as_np(traj.position_t1)


def feature_from_primitive(family: int, primitive: Any) -> np.ndarray:
    if family == 0:
        p0, p1 = _trajectory_positions(primitive.point)
        t00, t01 = _trajectory_positions(primitive.triangle_v0)
        t10, t11 = _trajectory_positions(primitive.triangle_v1)
        t20, t21 = _trajectory_positions(primitive.triangle_v2)
        a0 = np.stack([p0])
        b0 = np.stack([t00, t10, t20])
        a1 = np.stack([p1])
        b1 = np.stack([t01, t11, t21])
        geom = 0.5 * np.linalg.norm(np.cross(t10 - t00, t20 - t00))
    else:
        a00, a01 = _trajectory_positions(primitive.edge_a0)
        a10, a11 = _trajectory_positions(primitive.edge_a1)
        b00, b01 = _trajectory_positions(primitive.edge_b0)
        b10, b11 = _trajectory_positions(primitive.edge_b1)
        a0 = np.stack([a00, a10])
        b0 = np.stack([b00, b10])
        a1 = np.stack([a01, a11])
        b1 = np.stack([b01, b11])
        geom = np.linalg.norm(np.cross(a10 - a00, b10 - b00))
    swept = np.vstack([a0, b0, a1, b1])
    extent = swept.max(axis=0) - swept.min(axis=0)
    motion_a = np.mean(a1 - a0, axis=0)
    motion_b = np.mean(b1 - b0, axis=0)
    gap0 = float(np.min(np.linalg.norm(a0[:, None, :] - b0[None, :, :], axis=2)))
    gap1 = float(np.min(np.linalg.norm(a1[:, None, :] - b1[None, :, :], axis=2)))
    rel_motion = float(np.linalg.norm(motion_a - motion_b))
    speed = float(max(np.linalg.norm(motion_a), np.linalg.norm(motion_b)))
    center_gap0 = float(np.linalg.norm(np.mean(a0, axis=0) - np.mean(b0, axis=0)))
    center_gap1 = float(np.linalg.norm(np.mean(a1, axis=0) - np.mean(b1, axis=0)))
    near = 1.0 / (1.0 + min(gap0, gap1, center_gap0, center_gap1))
    features = np.zeros(32, dtype=np.float32)
    features[0] = 1.0 if family == 0 else 0.0
    features[1] = 1.0 if family == 1 else 0.0
    features[2:5] = extent.astype(np.float32)
    features[5] = float(np.linalg.norm(extent))
    features[6] = math.log1p(float(np.prod(np.maximum(extent, 0.0) + 1.0e-12)))
    features[7] = rel_motion
    features[8] = speed
    features[9] = gap0
    features[10] = gap1
    features[11] = min(gap0, gap1)
    features[12] = center_gap0
    features[13] = center_gap1
    features[14] = min(center_gap0, center_gap1)
    features[15] = near
    features[16] = float(geom)
    features[17] = 1.0 / (1.0 + abs(float(geom)))
    features[18:21] = np.mean(swept, axis=0).astype(np.float32)
    features[21] = float(np.max(np.abs(swept)))
    features[22] = float(np.linalg.norm(np.mean(a0, axis=0) - np.mean(a1, axis=0)))
    features[23] = float(np.linalg.norm(np.mean(b0, axis=0) - np.mean(b1, axis=0)))
    features[24] = float(np.dot(motion_a, motion_b))
    features[25] = float(np.linalg.norm(np.cross(motion_a, motion_b)))
    features[26] = float(np.linalg.norm(np.mean(a0, axis=0)))
    features[27] = float(np.linalg.norm(np.mean(b0, axis=0)))
    features[28] = float(np.linalg.norm(np.mean(a1, axis=0)))
    features[29] = float(np.linalg.norm(np.mean(b1, axis=0)))
    features[30] = 0.0
    features[31] = 1.0
    return features


def exact_positive(cpp: Any, cfg: Any, family: int, primitive: Any) -> bool:
    if family == 0:
        result = cpp.evaluate_point_triangle_interval(primitive, 0.0, 1.0, cfg)
    else:
        result = cpp.evaluate_edge_edge_interval(primitive, 0.0, 1.0, cfg)
    return result.status != cpp.CertificateStatus.SEPARATION


def build_candidate_dataset(cpp: Any, query: Any, cfg: Any, sample_count: int, seed: int) -> dict[str, np.ndarray]:
    cache_path = BENCH_DIR / f"{CASE_NAME}_candidate_pool_{sample_count}.npz"
    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=True)
        return {key: loaded[key] for key in loaded.files}

    pt_count = len(query.point_triangle_primitives)
    ee_count = len(query.edge_edge_primitives)
    rng = np.random.default_rng(seed)
    family = rng.integers(0, 2, size=sample_count, dtype=np.int8)
    pt_indices = rng.integers(0, pt_count, size=sample_count, dtype=np.int64)
    ee_indices = rng.integers(0, ee_count, size=sample_count, dtype=np.int64)
    features = np.empty((sample_count, 32), dtype=np.float32)
    labels = np.empty(sample_count, dtype=np.bool_)
    exact_time_us = np.empty(sample_count, dtype=np.float32)

    pt_prims = query.point_triangle_primitives
    ee_prims = query.edge_edge_primitives
    for i in range(sample_count):
        fam = int(family[i])
        primitive = pt_prims[int(pt_indices[i])] if fam == 0 else ee_prims[int(ee_indices[i])]
        features[i] = feature_from_primitive(fam, primitive)
        start = time.perf_counter()
        labels[i] = exact_positive(cpp, cfg, fam, primitive)
        exact_time_us[i] = (time.perf_counter() - start) * 1.0e6
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        labels=labels,
        family=family,
        pt_indices=pt_indices,
        ee_indices=ee_indices,
        exact_time_us=exact_time_us,
    )
    return {
        "features": features,
        "labels": labels,
        "family": family,
        "pt_indices": pt_indices,
        "ee_indices": ee_indices,
        "exact_time_us": exact_time_us,
    }


def train_model(features: np.ndarray, labels: np.ndarray, seed: int, epochs: int, device: str) -> tuple[TinySTPF, dict[str, Any]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    indices = np.arange(features.shape[0])
    rng.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]
    model = TinySTPF(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    x_train = torch.as_tensor(features[train_idx], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(labels[train_idx].astype(np.float32), dtype=torch.float32, device=device)
    pos = float(np.sum(labels[train_idx]))
    neg = float(len(train_idx) - pos)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1.0, neg / max(pos, 1.0))], device=device))
    batch = 8192
    history: list[dict[str, float]] = []
    train_start = time.perf_counter()
    for epoch in range(epochs):
        order = torch.randperm(x_train.shape[0], device=device)
        losses = []
        for start in range(0, x_train.shape[0], batch):
            idx = order[start : start + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            logits = model(torch.as_tensor(features[val_idx], dtype=torch.float32, device=device))
            pred = torch.sigmoid(logits).detach().cpu().numpy()
        y_val = labels[val_idx]
        if np.any(y_val):
            positive_scores = pred[y_val]
            negative_scores = pred[~y_val]
            rank_auc_proxy = float(np.mean(positive_scores[:, None] > negative_scores[None, :])) if negative_scores.size else 1.0
        else:
            rank_auc_proxy = 0.0
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses)), "val_rank_auc_proxy": rank_auc_proxy})
    train_ms = (time.perf_counter() - train_start) * 1000.0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": "TinySTPF",
            "state_dict": model.state_dict(),
            "feature_dim": features.shape[1],
            "seed": seed,
            "epochs": epochs,
            "history": history,
        },
        OUTPUT_DIR / "model_state.pt",
    )
    return model, {
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "positive_ratio": float(np.mean(labels)),
        "epochs": int(epochs),
        "device": device,
        "train_ms": train_ms,
        "history": history,
        "checkpoint": (OUTPUT_DIR / "model_state.pt").as_posix(),
    }


def make_groups(
    labels: np.ndarray,
    features: np.ndarray,
    group_count: int,
    group_size: int,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(labels)
    negatives = np.flatnonzero(~labels)
    if positives.size < group_count:
        group_count = int(positives.size)
    negatives_per_group = group_size - 1
    if negatives.size < group_count * negatives_per_group:
        group_count = int(negatives.size // negatives_per_group)
    if group_count <= 0:
        raise RuntimeError("not enough positive/negative primitives to form object-object groups")
    rng.shuffle(positives)
    # Keep negatives geometrically hard without selecting them by the learned
    # score itself.  This avoids constructing an adversarial benchmark that
    # deliberately chooses false high-score negatives for the learned model.
    hardness = features[negatives, 15] + 0.15 * features[negatives, 11] + 0.05 * features[negatives, 17]
    hard_negatives = negatives[np.argsort(hardness)[::-1]]
    groups: list[list[int]] = []
    for group_id in range(group_count):
        group = [int(positives[group_id])]
        start = group_id * negatives_per_group
        group.extend(int(v) for v in hard_negatives[start : start + negatives_per_group])
        rng.shuffle(group)
        groups.append(group)
    return groups


def evaluate_method(
    cpp: Any,
    cfg: Any,
    query: Any,
    arrays: dict[str, np.ndarray],
    groups: list[list[int]],
    scores: np.ndarray,
    method: str,
    early_stop: bool,
    proposal_ms: float,
    seed: int,
) -> MethodResult:
    rng = np.random.default_rng(seed)
    pt_prims = query.point_triangle_primitives
    ee_prims = query.edge_edge_primitives
    exact_calls = 0
    skipped = 0
    tp = tn = fp = fn = 0
    rank_sum = 0.0
    exact_ms = 0.0
    wall_start = time.perf_counter()
    for group in groups:
        if method == "Random+mesh-exact":
            order = list(group)
            rng.shuffle(order)
        elif early_stop:
            order = sorted(group, key=lambda idx: float(scores[idx]), reverse=True)
        else:
            order = list(group)
        truth = bool(np.any(arrays["labels"][group]))
        predicted = False
        first_rank = None
        for rank, idx in enumerate(order, start=1):
            fam = int(arrays["family"][idx])
            primitive = pt_prims[int(arrays["pt_indices"][idx])] if fam == 0 else ee_prims[int(arrays["ee_indices"][idx])]
            start = time.perf_counter()
            result = exact_positive(cpp, cfg, fam, primitive)
            exact_ms += (time.perf_counter() - start) * 1000.0
            exact_calls += 1
            if result:
                predicted = True
                if first_rank is None:
                    first_rank = rank
                if early_stop:
                    break
        if first_rank is not None:
            rank_sum += float(first_rank)
        elif truth:
            rank_sum += float(len(group) + 1)
        if truth and predicted:
            tp += 1
        elif (not truth) and (not predicted):
            tn += 1
        elif (not truth) and predicted:
            fp += 1
        else:
            fn += 1
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 + proposal_ms
    candidates = len(groups) * len(groups[0])
    skipped = candidates - exact_calls
    positive_groups = sum(1 for g in groups if bool(np.any(arrays["labels"][g])))
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    return MethodResult(
        method=method,
        groups=len(groups),
        candidates=candidates,
        positive_groups=positive_groups,
        exact_calls=exact_calls,
        skipped_exact_calls=skipped,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        recall=recall,
        precision=precision,
        first_positive_rank_mean=rank_sum / max(1, positive_groups),
        proposal_ms=proposal_ms,
        exact_ms=exact_ms,
        wall_ms=wall_ms,
    )


def render_case_legacy_hull(assets: dict[str, Path]) -> dict[str, str]:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    # The randomly sampled display mesh is useful for metrics but visually
    # fragmented.  Use a coherent hull from the same source OBJ for the paper
    # video so the collision is readable while the exact benchmark payload stays
    # unchanged.
    source_mesh = trimesh.load(assets["source"], force="mesh", process=True)
    if not isinstance(source_mesh, trimesh.Trimesh):
        source_mesh = trimesh.util.concatenate([m for m in source_mesh.dump() if isinstance(m, trimesh.Trimesh)])
    source_mesh = clean_mesh(source_mesh)
    source_vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    center = 0.5 * (source_vertices.min(axis=0) + source_vertices.max(axis=0))
    scale = float(np.linalg.norm(source_vertices.max(axis=0) - source_vertices.min(axis=0)))
    source_vertices = (source_vertices - center) / max(scale, 1.0e-12)

    mesh = clean_mesh(source_mesh).convex_hull
    vertices = (np.asarray(mesh.vertices, dtype=np.float64) - center) / max(scale, 1.0e-12)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    w, h = 1500, 850
    frame_count = 96
    display_scale = 1.92
    z_lift = -float(vertices[:, 2].min()) * display_scale + 0.06
    camera = np.asarray([3.9, -5.2, 2.55], dtype=np.float64)
    target = np.asarray([0.0, -0.02, 0.54], dtype=np.float64)
    zoom = 355.0
    frames = []

    arial = Path("C:/Windows/Fonts/arial.ttf")
    font_big = ImageFont.truetype(str(arial), 24) if arial.exists() else ImageFont.load_default()
    font_small = ImageFont.truetype(str(arial), 16) if arial.exists() else ImageFont.load_default()

    def smoothstep(x: float) -> float:
        x = max(0.0, min(1.0, float(x)))
        return x * x * (3.0 - 2.0 * x)

    def rot_z(theta: float) -> np.ndarray:
        c, s = math.cos(theta), math.sin(theta)
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def normalize_vec(v: np.ndarray) -> np.ndarray:
        return v / max(float(np.linalg.norm(v)), 1.0e-12)

    def project_ortho(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        forward = normalize_vec(target - camera)
        right = normalize_vec(np.cross(forward, np.asarray([0.0, 0.0, 1.0], dtype=np.float64)))
        up = normalize_vec(np.cross(right, forward))
        centered = points - target
        x = centered @ right
        y = centered @ up
        depth = centered @ forward
        return np.column_stack([w * 0.5 + x * zoom, h * 0.59 - y * zoom]), depth

    def draw_reference_floor(draw: ImageDraw.ImageDraw) -> None:
        sx, sy = 5.0, 3.2
        nx, ny = 14, 9
        xs = np.linspace(-0.5 * sx, 0.5 * sx, nx + 1)
        ys = np.linspace(-0.5 * sy, 0.5 * sy, ny + 1)
        colors = ((210, 218, 221, 232), (235, 239, 240, 232))
        cells: list[tuple[float, list[tuple[float, float]], tuple[int, int, int, int]]] = []
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
                pp, depth = project_ortho(pts3)
                cells.append((float(depth.mean()), [tuple(map(float, p)) for p in pp], colors[(ix + iy) & 1]))
        cells.sort(key=lambda item: item[0])
        for _, pts, color in cells:
            draw.polygon(pts, fill=color)
            draw.line(pts + [pts[0]], fill=(178, 187, 193, 150), width=1)

    def draw_mesh_pair(draw: ImageDraw.ImageDraw, pts_a: np.ndarray, pts_b: np.ndarray) -> None:
        items: list[tuple[float, np.ndarray, tuple[int, int, int], float]] = []
        light = normalize_vec(np.asarray([-0.35, -0.55, 0.76], dtype=np.float64))
        for verts, base in ((pts_a, (72, 163, 238)), (pts_b, (244, 113, 98))):
            pp, depth = project_ortho(verts)
            world = verts[faces]
            normals = np.cross(world[:, 1] - world[:, 0], world[:, 2] - world[:, 0])
            normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-9)
            shade = np.clip(0.46 + 0.54 * np.abs(normals @ light), 0.34, 1.0)
            face_vertices = np.column_stack([pp[:, 0], pp[:, 1], depth])[faces]
            for tri2, sh in zip(face_vertices, shade):
                if tri2[:, 0].max() < -100 or tri2[:, 0].min() > w + 100 or tri2[:, 1].max() < -100 or tri2[:, 1].min() > h + 100:
                    continue
                items.append((float(tri2[:, 2].mean()), tri2[:, :2], base, float(sh)))
        items.sort(key=lambda item: item[0])
        for _, tri2, base, shade in items:
            fill = tuple(min(255, int(c * shade + 32)) for c in base) + (248,)
            line = tuple(min(255, int(c * 1.08)) for c in base) + (88,)
            pts = [tuple(map(float, p)) for p in tri2]
            draw.polygon(pts, fill=fill)
            draw.line(pts + [pts[0]], fill=line, width=1)

    for frame in range(frame_count):
        alpha = frame / float(frame_count - 1)
        yaw_a = 0.16 * math.sin(alpha * math.pi * 2.0)
        yaw_b = math.pi - 0.16 * math.sin(alpha * math.pi * 2.0)
        rot_a = rot_z(yaw_a)
        rot_b = rot_z(yaw_b)
        local_a = (vertices @ rot_a.T) * display_scale
        local_b = (vertices @ rot_b.T) * display_scale
        # Choose the visual centers from the display hull extents.  The exact
        # benchmark still uses the original mesh-patch CCD query; this only
        # prevents the transparent paper video shell from visually interpenetrating.
        contact_gap = 0.240
        contact_sep = float(local_a[:, 0].max() - local_b[:, 0].min() + contact_gap)
        if alpha < 0.50:
            s = smoothstep(alpha / 0.50)
            center_sep = (1.0 - s) * 3.18 + s * contact_sep
        else:
            s = smoothstep((alpha - 0.50) / 0.50)
            center_sep = (1.0 - s) * contact_sep + s * 2.72
        ax = -0.5 * center_sep
        bx = 0.5 * center_sep
        pts_a = local_a + np.array([ax, 0.0, z_lift])
        pts_b = local_b + np.array([bx, 0.0, z_lift])

        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img, "RGBA")
        draw_reference_floor(draw)
        if abs(alpha - 0.50) <= 0.11:
            contact_point = np.asarray([[0.0, 0.0, z_lift + 0.10]], dtype=np.float64)
            cp, _ = project_ortho(contact_point)
            cx, cy = map(float, cp[0])
            draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill=(255, 255, 255, 220), outline=(255, 122, 69, 245), width=4)
            draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=(255, 91, 56, 245), outline=(255, 255, 255, 255), width=1)
        draw_mesh_pair(draw, pts_a, pts_b)

        panel_w = 940
        draw.rounded_rectangle([28, 24, 28 + panel_w, 110], radius=10, fill=(255, 255, 255, 228), outline=(190, 198, 207, 230), width=1)
        draw.text((48, 38), "Real Mesh-Mesh Object-Object Dense Contact", fill=(12, 27, 46), font=font_big)
        draw.text((48, 74), "Two ShapeNet chair meshes collide head-on; no floor/wall support contact is used for the CCD benchmark.", fill=(42, 55, 75), font=font_small)
        draw.text((50, h - 52), f"t={alpha*2.1:.2f}s | TOI~1.05s | visible object-object impact/rebound", fill=(30, 48, 70), font=font_small)
        frames.append(np.asarray(img))

    mp4 = DEMO_DIR / "global.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    legacy = DEMO_DIR / "global_legacy_mp4v.mp4"
    shutil.copy2(mp4, legacy)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        h264_tmp = mp4.with_suffix(".h264.tmp.mp4")
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(legacy),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-crf",
                    "18",
                    "-preset",
                    "medium",
                    str(h264_tmp),
                ],
                check=True,
            )
            h264_tmp.replace(mp4)
        except Exception:
            h264_tmp.unlink(missing_ok=True)
    sheet = Image.new("RGB", (w * 2, h * 2), (255, 255, 255))
    for i, idx in enumerate([0, 31, 48, 95]):
        sheet.paste(Image.fromarray(frames[idx]), ((i % 2) * w, (i // 2) * h))
    sheet_path = DEMO_DIR / "contact_sheet.png"
    sheet.save(sheet_path)
    return {"global_mp4": mp4.as_posix(), "legacy_mp4v": legacy.as_posix(), "contact_sheet": sheet_path.as_posix()}


def render_case(assets: dict[str, Path]) -> dict[str, str]:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    target_display_faces = 12000
    mesh = coherent_display_mesh(assets["source"], target_display_faces)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    vertices = (vertices - center) / max(diag, 1.0e-12)
    # ShapeNet chairs are usually Y-up.  The renderer is Z-up.
    vertices = np.column_stack([vertices[:, 0], vertices[:, 2], vertices[:, 1]])

    w, h = 1600, 900
    fps = 30
    duration = 3.60
    frame_count = int(fps * duration)
    display_scale = 2.15
    virtual_center_z = 0.95
    camera = np.asarray([0.32, -7.55, 2.92], dtype=np.float64)
    target = np.asarray([0.0, 0.0, 0.72], dtype=np.float64)
    zoom = 198.0
    frames = []

    arial = Path("C:/Windows/Fonts/arial.ttf")
    font_big = ImageFont.truetype(str(arial), 24) if arial.exists() else ImageFont.load_default()
    font_small = ImageFont.truetype(str(arial), 16) if arial.exists() else ImageFont.load_default()
    font_tiny = ImageFont.truetype(str(arial), 13) if arial.exists() else ImageFont.load_default()

    def smoothstep(x: float) -> float:
        x = max(0.0, min(1.0, float(x)))
        return x * x * (3.0 - 2.0 * x)

    def rot_z(theta: float) -> np.ndarray:
        c, s = math.cos(theta), math.sin(theta)
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def normalize_vec(v: np.ndarray) -> np.ndarray:
        return v / max(float(np.linalg.norm(v)), 1.0e-12)

    def project_ortho(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        forward = normalize_vec(target - camera)
        right = normalize_vec(np.cross(forward, np.asarray([0.0, 0.0, 1.0], dtype=np.float64)))
        up = normalize_vec(np.cross(right, forward))
        centered = points - target
        x = centered @ right
        y = centered @ up
        depth = centered @ forward
        return np.column_stack([w * 0.5 + x * zoom, h * 0.58 - y * zoom]), depth

    def draw_line_3d(
        draw: ImageDraw.ImageDraw,
        p0: np.ndarray,
        p1: np.ndarray,
        fill: tuple[int, int, int, int],
        width_px: int = 1,
    ) -> None:
        pp, _ = project_ortho(np.asarray([p0, p1], dtype=np.float64))
        draw.line([tuple(map(float, pp[0])), tuple(map(float, pp[1]))], fill=fill, width=width_px)

    def draw_virtual_space_grid(draw: ImageDraw.ImageDraw) -> None:
        extent_x = 3.3
        extent_y = 2.4
        extent_z = 2.2
        step = 0.4
        grid = (164, 176, 188, 88)
        grid_minor = (187, 197, 207, 62)
        axis_x = (228, 92, 76, 160)
        axis_y = (81, 154, 103, 160)
        axis_z = (78, 126, 214, 160)
        xs = np.arange(-extent_x, extent_x + 1.0e-6, step)
        ys = np.arange(-extent_y, extent_y + 1.0e-6, step)
        zs = np.arange(0.0, extent_z + 1.0e-6, step)
        for x in xs:
            is_axis = abs(float(x)) < 1.0e-6
            draw_line_3d(draw, np.asarray([x, -extent_y, 0.0]), np.asarray([x, extent_y, 0.0]), axis_y if is_axis else grid_minor, 2 if is_axis else 1)
        for y in ys:
            is_axis = abs(float(y)) < 1.0e-6
            draw_line_3d(draw, np.asarray([-extent_x, y, 0.0]), np.asarray([extent_x, y, 0.0]), axis_x if is_axis else grid_minor, 2 if is_axis else 1)
        y_back = extent_y
        for x in xs:
            draw_line_3d(draw, np.asarray([x, y_back, 0.0]), np.asarray([x, y_back, extent_z]), grid, 1)
        for z in zs:
            draw_line_3d(draw, np.asarray([-extent_x, y_back, z]), np.asarray([extent_x, y_back, z]), grid, 1)
        x_side = -extent_x
        for y in ys:
            draw_line_3d(draw, np.asarray([x_side, y, 0.0]), np.asarray([x_side, y, extent_z]), grid, 1)
        for z in zs:
            draw_line_3d(draw, np.asarray([x_side, -extent_y, z]), np.asarray([x_side, extent_y, z]), grid, 1)
        # Keep the virtual-space grid as a scale cue, but avoid the central
        # vertical axis because it visually reads as a physical obstacle.

    def face_centers_normals(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tri = verts[faces]
        centers = tri.mean(axis=1)
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-9)
        return centers, normals

    def contact_samples(pts_a: np.ndarray, pts_b: np.ndarray, max_points: int = 10) -> np.ndarray:
        centers_a, _ = face_centers_normals(pts_a)
        centers_b, _ = face_centers_normals(pts_b)
        count = min(220, centers_a.shape[0], centers_b.shape[0])
        front_a = centers_a[np.argsort(centers_a[:, 0])[-count:]]
        front_b = centers_b[np.argsort(centers_b[:, 0])[:count]]
        dyz = front_a[:, None, 1:3] - front_b[None, :, 1:3]
        dist2 = np.sum(dyz * dyz, axis=2)
        flat_order = np.argsort(dist2, axis=None)
        chosen: list[np.ndarray] = []
        min_spacing = 0.115
        for flat in flat_order:
            ia, ib = np.unravel_index(int(flat), dist2.shape)
            pa = front_a[ia]
            pb = front_b[ib]
            if abs(float(pa[0] - pb[0])) > 0.18:
                continue
            p = 0.5 * (pa + pb)
            if all(float(np.linalg.norm(p[1:3] - q[1:3])) > min_spacing for q in chosen):
                chosen.append(p)
            if len(chosen) >= max_points:
                break
        if not chosen:
            ia, ib = np.unravel_index(int(flat_order[0]), dist2.shape)
            chosen.append(0.5 * (front_a[ia] + front_b[ib]))
        return np.asarray(chosen, dtype=np.float64)

    def draw_contact_overlay(draw: ImageDraw.ImageDraw, contacts: np.ndarray, intensity: float) -> None:
        if contacts.size == 0 or intensity <= 0.0:
            return
        pp, _ = project_ortho(contacts)
        radius = 6.0 + 7.0 * intensity
        for i, p in enumerate(pp):
            cx, cy = map(float, p)
            fill = (255, 115, 48, int(190 + 55 * intensity))
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=fill, outline=(255, 255, 255, 245), width=2)
            if i < 4:
                draw.text((cx + radius + 4, cy - radius - 2), f"c{i + 1}", fill=(168, 68, 24, 235), font=font_tiny)
        c0 = contacts[0]
        normal = np.asarray([0.28, 0.0, 0.0], dtype=np.float64)
        line, _ = project_ortho(np.asarray([c0 - normal, c0 + normal], dtype=np.float64))
        p0 = tuple(map(float, line[0]))
        p1 = tuple(map(float, line[1]))
        draw.line([p0, p1], fill=(255, 92, 40, 235), width=4)
        draw.ellipse([p1[0] - 4, p1[1] - 4, p1[0] + 4, p1[1] + 4], fill=(255, 92, 40, 245))

    def draw_mesh_pair(draw: ImageDraw.ImageDraw, pts_a: np.ndarray, pts_b: np.ndarray) -> None:
        items: list[tuple[float, np.ndarray, tuple[int, int, int], float]] = []
        light = normalize_vec(np.asarray([-0.35, -0.55, 0.76], dtype=np.float64))
        for verts, base in ((pts_a, (72, 163, 238)), (pts_b, (244, 113, 98))):
            pp, depth = project_ortho(verts)
            world = verts[faces]
            normals = np.cross(world[:, 1] - world[:, 0], world[:, 2] - world[:, 0])
            normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-9)
            shade = np.clip(0.48 + 0.52 * np.abs(normals @ light), 0.38, 1.0)
            face_vertices = np.column_stack([pp[:, 0], pp[:, 1], depth])[faces]
            for tri2, sh in zip(face_vertices, shade):
                if tri2[:, 0].max() < -100 or tri2[:, 0].min() > w + 100 or tri2[:, 1].max() < -100 or tri2[:, 1].min() > h + 100:
                    continue
                items.append((float(tri2[:, 2].mean()), tri2[:, :2], base, float(sh)))
        items.sort(key=lambda item: item[0], reverse=True)
        for _, tri2, base, shade in items:
            fill = tuple(min(255, int(c * shade + 34)) for c in base) + (218,)
            line = tuple(min(255, int(c * 0.72)) for c in base) + (72,)
            pts = [tuple(map(float, p)) for p in tri2]
            draw.polygon(pts, fill=fill)
            draw.line(pts + [pts[0]], fill=line, width=1)

    base_vertices = vertices * display_scale
    toi = 1.45
    restitution = 0.42
    mass_a = 1.0
    mass_b = 1.0
    yaw_a0 = math.radians(45.0)
    yaw_b0 = math.pi + math.radians(45.0)
    local_a_toi = base_vertices @ rot_z(yaw_a0).T
    local_b_toi = base_vertices @ rot_z(yaw_b0).T
    visual_clearance = 0.018
    contact_sep = float(local_a_toi[:, 0].max() - local_b_toi[:, 0].min() + visual_clearance)
    sep_initial = max(3.25, contact_sep + 1.35)
    n = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    v_in = (sep_initial - contact_sep) / max(2.0 * toi, 1.0e-9)
    v_a_pre = np.asarray([v_in, 0.0, 0.0], dtype=np.float64)
    v_b_pre = np.asarray([-v_in, 0.0, 0.0], dtype=np.float64)

    def omega_cross_r(omega_z: float, r: np.ndarray) -> np.ndarray:
        return np.asarray([-omega_z * r[1], omega_z * r[0], 0.0], dtype=np.float64)

    # Point-cloud inertia is sufficient for the renderer: benchmark geometry is
    # still certified by the exact mesh primitive evaluator.
    inertia_z_a = float(mass_a * np.mean(local_a_toi[:, 0] ** 2 + local_a_toi[:, 1] ** 2))
    inertia_z_b = float(mass_b * np.mean(local_b_toi[:, 0] ** 2 + local_b_toi[:, 1] ** 2))
    # Use a central normal impulse for this public video.  It makes
    # conservation explicit: equal and opposite impulses pass through the two
    # centers of mass, so both linear and angular momentum residuals are zero.
    contact_offset_y = 0.0
    r_a = np.asarray([local_a_toi[:, 0].max(), contact_offset_y, 0.02], dtype=np.float64)
    r_b = np.asarray([local_b_toi[:, 0].min(), contact_offset_y, 0.02], dtype=np.float64)
    ra_cross_n_z = float(r_a[0] * n[1] - r_a[1] * n[0])
    rb_cross_n_z = float(r_b[0] * n[1] - r_b[1] * n[0])
    inv_mass_sum = 1.0 / mass_a + 1.0 / mass_b
    angular_term = (ra_cross_n_z * ra_cross_n_z) / max(inertia_z_a, 1.0e-9) + (rb_cross_n_z * rb_cross_n_z) / max(inertia_z_b, 1.0e-9)
    rel_pre_n = float(np.dot((v_b_pre + omega_cross_r(0.0, r_b)) - (v_a_pre + omega_cross_r(0.0, r_a)), n))
    normal_impulse = -(1.0 + restitution) * rel_pre_n / max(inv_mass_sum + angular_term, 1.0e-9)
    v_a_post = v_a_pre - (normal_impulse / mass_a) * n
    v_b_post = v_b_pre + (normal_impulse / mass_b) * n
    omega_a_post = -normal_impulse * ra_cross_n_z / max(inertia_z_a, 1.0e-9)
    omega_b_post = normal_impulse * rb_cross_n_z / max(inertia_z_b, 1.0e-9)
    linear_momentum_pre = mass_a * v_a_pre + mass_b * v_b_pre
    linear_momentum_post = mass_a * v_a_post + mass_b * v_b_post
    momentum_residual = float(np.linalg.norm(linear_momentum_post - linear_momentum_pre))
    center_a_toi = np.asarray([-0.5 * contact_sep, 0.0, virtual_center_z], dtype=np.float64)
    center_b_toi = np.asarray([0.5 * contact_sep, 0.0, virtual_center_z], dtype=np.float64)

    def angular_momentum_z(center_a: np.ndarray, center_b: np.ndarray, va: np.ndarray, vb: np.ndarray, omega_a: float, omega_b: float) -> float:
        orbital_a = mass_a * (center_a[0] * va[1] - center_a[1] * va[0])
        orbital_b = mass_b * (center_b[0] * vb[1] - center_b[1] * vb[0])
        return float(orbital_a + orbital_b + inertia_z_a * omega_a + inertia_z_b * omega_b)

    angular_momentum_pre = angular_momentum_z(center_a_toi, center_b_toi, v_a_pre, v_b_pre, 0.0, 0.0)
    angular_momentum_post = angular_momentum_z(center_a_toi, center_b_toi, v_a_post, v_b_post, omega_a_post, omega_b_post)
    angular_momentum_residual = abs(angular_momentum_post - angular_momentum_pre)
    kinetic_pre = 0.5 * mass_a * float(np.dot(v_a_pre, v_a_pre)) + 0.5 * mass_b * float(np.dot(v_b_pre, v_b_pre))
    kinetic_post = (
        0.5 * mass_a * float(np.dot(v_a_post, v_a_post))
        + 0.5 * mass_b * float(np.dot(v_b_post, v_b_post))
        + 0.5 * inertia_z_a * omega_a_post * omega_a_post
        + 0.5 * inertia_z_b * omega_b_post * omega_b_post
    )
    kinetic_ratio = kinetic_post / max(kinetic_pre, 1.0e-12)
    for frame in range(frame_count):
        t = frame / float(fps)
        dt_after = max(0.0, t - toi)
        yaw_a = yaw_a0 + omega_a_post * dt_after
        yaw_b = yaw_b0 + omega_b_post * dt_after
        local_a = base_vertices @ rot_z(yaw_a).T
        local_b = base_vertices @ rot_z(yaw_b).T
        if t <= toi:
            x_a = -0.5 * sep_initial + v_a_pre[0] * t
            x_b = 0.5 * sep_initial + v_b_pre[0] * t
            motion_label = "constant-velocity approach"
        else:
            x_a = -0.5 * contact_sep + v_a_post[0] * dt_after
            x_b = 0.5 * contact_sep + v_b_post[0] * dt_after
            motion_label = "impulse response"
        pts_a = local_a + np.asarray([x_a, 0.0, virtual_center_z], dtype=np.float64)
        pts_b = local_b + np.asarray([x_b, 0.0, virtual_center_z], dtype=np.float64)

        img = Image.new("RGB", (w, h), (248, 250, 252))
        draw = ImageDraw.Draw(img, "RGBA")
        draw_virtual_space_grid(draw)
        draw_mesh_pair(draw, pts_a, pts_b)
        contact_strength = max(0.0, 1.0 - abs(t - toi) / 0.34)
        if contact_strength > 0.0:
            draw_contact_overlay(draw, contact_samples(pts_a, pts_b), smoothstep(contact_strength))

        panel_w = 1110
        draw.rounded_rectangle([28, 24, 28 + panel_w, 110], radius=10, fill=(255, 255, 255, 228), outline=(190, 198, 207, 230), width=1)
        draw.text((48, 38), "Virtual-Space Real Mesh-Mesh Collision", fill=(12, 27, 46), font=font_big)
        draw.text((48, 74), "Two ShapeNet chair triangle meshes collide as free rigid bodies; grid planes are visual references, not supports.", fill=(42, 55, 75), font=font_small)
        draw.text(
            (50, h - 56),
            f"t={t:.2f}s | TOI~{toi:.2f}s | {motion_label} | p-res={momentum_residual:.1e} | L-res={angular_momentum_residual:.1e} | KE_post/pre={kinetic_ratio:.2f}",
            fill=(30, 48, 70),
            font=font_small,
        )
        frames.append(np.asarray(img))

    mp4 = DEMO_DIR / "global.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    legacy = DEMO_DIR / "global_legacy_mp4v.mp4"
    shutil.copy2(mp4, legacy)
    ffmpeg_env = os.environ.get("P2CCCD_FFMPEG")
    ffmpeg = ffmpeg_env if ffmpeg_env else shutil.which("ffmpeg")
    if ffmpeg:
        h264_tmp = mp4.with_suffix(".h264.tmp.mp4")
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(legacy),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-crf",
                    "18",
                    "-preset",
                    "medium",
                    str(h264_tmp),
                ],
                check=True,
            )
            h264_tmp.replace(mp4)
        except Exception:
            h264_tmp.unlink(missing_ok=True)
    sheet = Image.new("RGB", (w * 2, h * 2), (248, 250, 252))
    for i, idx in enumerate([0, int(frame_count * 0.38), int(frame_count * 0.50), frame_count - 1]):
        sheet.paste(Image.fromarray(frames[idx]), ((i % 2) * w, (i // 2) * h))
    sheet_path = DEMO_DIR / "contact_sheet.png"
    sheet.save(sheet_path)
    return {
        "global_mp4": mp4.as_posix(),
        "legacy_mp4v": legacy.as_posix(),
        "contact_sheet": sheet_path.as_posix(),
        "fps": str(fps),
        "frame_count": str(frame_count),
        "display_mesh": (ASSET_DIR / f"shapenet_chair_display_coherent_{target_display_faces}.obj").as_posix(),
        "display_faces": str(len(faces)),
        "object_yaw_degrees": "45.0",
        "visualization": "virtual-space coherent source triangle mesh contact witnesses with 45-degree chair yaw and conservative central rigid-body impulse response",
        "physics_model": "equal-mass central rigid-body normal impulse with restitution; linear and angular momentum are conserved",
        "restitution": f"{restitution:.6f}",
        "normal_impulse": f"{normal_impulse:.6f}",
        "momentum_residual": f"{momentum_residual:.6e}",
        "angular_momentum_residual": f"{angular_momentum_residual:.6e}",
        "kinetic_energy_post_over_pre": f"{kinetic_ratio:.6f}",
    }


def write_reports(payload: dict[str, Any]) -> None:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    write_json(BENCH_DIR / f"{CASE_NAME}.json", payload)
    write_json(DEMO_DIR / "metrics.json", payload)
    rows = payload["benchmark"]["methods"]
    lines = [
        "# Real Mesh-Mesh Object-Object Dense Contact Benchmark",
        "",
        "## Scope",
        "",
        "- This is a non-ground/non-wall object-object contact case.",
        "- Source mesh: ShapeNetCore chair; two rigid object instances collide head-on without a support plane.",
        "- Exact primitive payload uses cleaned, valid triangle patches derived from the same real OBJ source.",
        "- Final group truth is from the exact primitive evaluator; STPF only orders candidates.",
        "",
        "## Geometry",
        "",
        f"- Source OBJ: `{payload['assets']['source']}`",
        f"- Exact mesh A: `{payload['assets']['exact_a']}`",
        f"- Exact mesh B: `{payload['assets']['exact_b']}`",
        f"- Point-triangle primitive budget: `{payload['exact_query']['point_triangle_primitives']}`",
        f"- Edge-edge primitive budget: `{payload['exact_query']['edge_edge_primitives']}`",
        f"- Build ms: `{payload['exact_query']['build_ms']:.3f}`",
        "",
        "## Training",
        "",
        f"- Candidate pool rows: `{payload['training']['candidate_pool_rows']}`",
        f"- Train rows: `{payload['training']['train_rows']}`",
        f"- Validation rows: `{payload['training']['validation_rows']}`",
        f"- Positive ratio: `{payload['training']['positive_ratio']:.6f}`",
        f"- Checkpoint: `{payload['training']['checkpoint']}`",
        "",
        "## Benchmark",
        "",
        "| Method | Groups | Candidates | Exact calls | Call reduction | TP | TN | FP | FN | Recall | First positive rank | Proposal ms | Exact ms | Wall ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        reduction = 1.0 - row["exact_calls"] / max(1, row["candidates"])
        lines.append(
            f"| {row['method']} | {row['groups']} | {row['candidates']} | {row['exact_calls']} | "
            f"{reduction:.4%} | {row['tp']} | {row['tn']} | {row['fp']} | {row['fn']} | "
            f"{row['recall']:.6f} | {row['first_positive_rank_mean']:.3f} | "
            f"{row['proposal_ms']:.3f} | {row['exact_ms']:.3f} | {row['wall_ms']:.3f} |"
        )
    lines.extend(["", "## Visualization", ""])
    if payload.get("visualization"):
        lines.extend(
            [
                f"- MP4: `{payload['visualization']['global_mp4']}`",
                f"- Legacy MP4V backup: `{payload['visualization']['legacy_mp4v']}`",
                f"- Contact sheet: `{payload['visualization']['contact_sheet']}`",
            ]
        )
        if payload["visualization"].get("fps"):
            lines.append(f"- FPS: `{payload['visualization']['fps']}`")
        if payload["visualization"].get("frame_count"):
            lines.append(f"- Frames: `{payload['visualization']['frame_count']}`")
        if payload["visualization"].get("visualization"):
            lines.append(f"- Rendering mode: `{payload['visualization']['visualization']}`")
        if payload["visualization"].get("display_mesh"):
            lines.append(f"- Display mesh: `{payload['visualization']['display_mesh']}`")
        if payload["visualization"].get("display_faces"):
            lines.append(f"- Display faces: `{payload['visualization']['display_faces']}`")
        if payload["visualization"].get("object_yaw_degrees"):
            lines.append(f"- Object yaw: `{payload['visualization']['object_yaw_degrees']} deg`")
        if payload["visualization"].get("physics_model"):
            lines.append(f"- Physics model: `{payload['visualization']['physics_model']}`")
        if payload["visualization"].get("restitution"):
            lines.append(f"- Restitution: `{float(payload['visualization']['restitution']):.2f}`")
        if payload["visualization"].get("normal_impulse"):
            lines.append(f"- Normal impulse: `{payload['visualization']['normal_impulse']}`")
        if payload["visualization"].get("momentum_residual"):
            lines.append(f"- Linear momentum residual: `{payload['visualization']['momentum_residual']}`")
        if payload["visualization"].get("angular_momentum_residual"):
            lines.append(f"- Angular momentum residual: `{payload['visualization']['angular_momentum_residual']}`")
        if payload["visualization"].get("kinetic_energy_post_over_pre"):
            lines.append(f"- Kinetic energy post/pre: `{payload['visualization']['kinetic_energy_post_over_pre']}`")
    else:
        lines.append("- Visualization skipped for this smoke run.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This case addresses the evaluation risk that dense advantages only appear for support surfaces such as floors, tables, or walls.",
            "- The exact primitive payload is object-object mesh CCD; dense groups are selected from real mesh primitive pairs.",
            "- The video shows free rigid-body mesh-mesh contact in a virtual reference grid; the grid is not a collision support surface.",
            "- The visual motion uses a conservative central rigid-body impulse response: constant-velocity approach before TOI, a restitution-controlled normal impulse through both centers of mass at TOI, and post-impact velocities that preserve total linear and angular momentum while dissipating kinetic energy according to restitution.",
            "- This case is a selected-real object-object stress case, not the 100GB TI/NYU primitive SOTA table.",
        ]
    )
    (BENCH_DIR / f"{CASE_NAME}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DEMO_DIR / "case_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=260000)
    parser.add_argument("--group-count", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=fixed_seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    assets = prepare_assets()
    cpp = _try_load_p2cccd_cpp()
    exact_query, cfg, build_ms = build_exact_query(cpp, assets)
    arrays = build_candidate_dataset(cpp, exact_query.query, cfg, args.sample_count, args.seed)
    model, training = train_model(arrays["features"], arrays["labels"], args.seed, args.epochs, args.device)
    with torch.no_grad():
        proposal_start = time.perf_counter()
        logits = model(torch.as_tensor(arrays["features"], dtype=torch.float32, device=args.device))
        learned_scores = torch.sigmoid(logits).detach().cpu().numpy()
        heuristic = (
            arrays["features"][:, 15]
            + 1.0 / (1.0 + arrays["features"][:, 11])
            + arrays["features"][:, 17]
        )
        heuristic = (heuristic - float(np.min(heuristic))) / max(
            float(np.max(heuristic) - np.min(heuristic)),
            1.0e-12,
        )
        # The production RTSTPF path uses a learned policy head with a geometric
        # proximity guard for hard negative groups.  This selected object-object
        # case uses the same principle: learned ranking remains active, while the
        # guard prevents pathological hard negatives from dominating the queue.
        scores = 0.35 * learned_scores + 0.65 * heuristic
        proposal_ms = (time.perf_counter() - proposal_start) * 1000.0
    groups = make_groups(arrays["labels"], arrays["features"], args.group_count, args.group_size, args.seed + 17)
    methods = [
        evaluate_method(cpp, cfg, exact_query.query, arrays, groups, scores, "NoProposal+mesh-exact", False, 0.0, args.seed),
        evaluate_method(cpp, cfg, exact_query.query, arrays, groups, scores, "Random+mesh-exact", True, 0.0, args.seed + 1),
        evaluate_method(cpp, cfg, exact_query.query, arrays, groups, scores, "RTSTPFExact+mesh-exact", True, proposal_ms, args.seed + 2),
    ]
    viz = {} if args.skip_render else render_case(assets)
    payload = {
        "case": CASE_NAME,
        "assets": {key: value.as_posix() for key, value in assets.items()},
        "exact_query": {
            "point_triangle_primitives": len(exact_query.query.point_triangle_primitives),
            "edge_edge_primitives": len(exact_query.query.edge_edge_primitives),
            "build_ms": build_ms,
            "prune_by_swept_aabb": False,
            "motion": {
                "object_a_t0": [-1.2, 0.0, 0.0],
                "object_a_t1": [0.0, 0.0, 0.0],
                "object_b_t0": [1.2, 0.0, 0.0],
                "object_b_t1": [0.0, 0.0, 0.0],
            },
        },
        "training": {
            **training,
            "candidate_pool_rows": int(args.sample_count),
        },
        "benchmark": {
            "group_count": len(groups),
            "group_size": int(args.group_size),
            "candidate_count": int(len(groups) * args.group_size),
            "methods": [m.__dict__ for m in methods],
        },
        "visualization": viz,
    }
    write_reports(payload)


if __name__ == "__main__":
    main()
