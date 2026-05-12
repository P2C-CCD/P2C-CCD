from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from p2cccd.bench.bvh_exact import _try_load_p2cccd_cpp


ROOT = Path(__file__).resolve().parents[2]
CASE_NAME = "hard_negative_near_miss_dense_run_id"
BENCH_DIR = ROOT / "src" / "benchmark" / CASE_NAME
DEMO_DIR = ROOT / "src" / "MyDemo" / CASE_NAME
OUTPUT_DIR = ROOT / "src" / "outputs" / "stpf_training" / CASE_NAME


@dataclass(frozen=True)
class MethodResult:
    method: str
    groups: int
    candidates: int
    exact_calls: int
    skipped_exact_calls: int
    tn: int
    fp: int
    fn: int
    false_positive_rate: float
    exact_ms: float
    proposal_ms: float
    wall_ms: float
    overhead_vs_no_proposal_ms: float


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


def write_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
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

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=default), encoding="utf-8")


def make_traj(cpp: Any, feature_id: int, p0: np.ndarray, p1: np.ndarray) -> Any:
    traj = cpp.LinearVertexTrajectory()
    traj.feature_id = int(feature_id)
    traj.position_t0 = [float(v) for v in p0]
    traj.position_t1 = [float(v) for v in p1]
    return traj


def make_primitive(cpp: Any, kind: int, t0: np.ndarray, t1: np.ndarray) -> Any:
    if kind == 0:
        primitive = cpp.PointTriangleIntervalPrimitive()
        primitive.point_id = 0
        primitive.triangle_id = 1
        primitive.point = make_traj(cpp, 0, t0[0], t1[0])
        primitive.triangle_v0 = make_traj(cpp, 1, t0[1], t1[1])
        primitive.triangle_v1 = make_traj(cpp, 2, t0[2], t1[2])
        primitive.triangle_v2 = make_traj(cpp, 3, t0[3], t1[3])
        return primitive
    primitive = cpp.EdgeEdgeIntervalPrimitive()
    primitive.edge_a_id = 0
    primitive.edge_b_id = 1
    primitive.edge_a0 = make_traj(cpp, 0, t0[0], t1[0])
    primitive.edge_a1 = make_traj(cpp, 1, t0[1], t1[1])
    primitive.edge_b0 = make_traj(cpp, 2, t0[2], t1[2])
    primitive.edge_b1 = make_traj(cpp, 3, t0[3], t1[3])
    return primitive


def exact_positive(cpp: Any, cfg: Any, kind: int, t0: np.ndarray, t1: np.ndarray) -> tuple[bool, int]:
    primitive = make_primitive(cpp, kind, t0, t1)
    if kind == 0:
        result = cpp.evaluate_point_triangle_interval(primitive, 0.0, 1.0, cfg)
    else:
        result = cpp.evaluate_edge_edge_interval(primitive, 0.0, 1.0, cfg)
    return result.status != cpp.CertificateStatus.SEPARATION, int(result.status)


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(np.linalg.norm(axis), 1.0e-12)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def candidate_features(kind: int, t0: np.ndarray, t1: np.ndarray, gap: float, speed: float, near_coplanar_angle: float) -> np.ndarray:
    a0 = t0[:1] if kind == 0 else t0[:2]
    b0 = t0[1:] if kind == 0 else t0[2:]
    a1 = t1[:1] if kind == 0 else t1[:2]
    b1 = t1[1:] if kind == 0 else t1[2:]
    swept = np.vstack([t0, t1])
    extent = swept.max(axis=0) - swept.min(axis=0)
    min_vertex_gap0 = float(np.min(np.linalg.norm(a0[:, None, :] - b0[None, :, :], axis=2)))
    min_vertex_gap1 = float(np.min(np.linalg.norm(a1[:, None, :] - b1[None, :, :], axis=2)))
    center_gap0 = float(np.linalg.norm(a0.mean(axis=0) - b0.mean(axis=0)))
    center_gap1 = float(np.linalg.norm(a1.mean(axis=0) - b1.mean(axis=0)))
    motion_a = a1.mean(axis=0) - a0.mean(axis=0)
    motion_b = b1.mean(axis=0) - b0.mean(axis=0)
    rel_motion = float(np.linalg.norm(motion_a - motion_b))
    near = 1.0 / (1.0 + min(min_vertex_gap0, min_vertex_gap1, center_gap0, center_gap1))
    features = np.zeros(32, dtype=np.float32)
    features[0] = 1.0 if kind == 0 else 0.0
    features[1] = 1.0 if kind == 1 else 0.0
    features[2:5] = extent.astype(np.float32)
    features[5] = float(np.linalg.norm(extent))
    features[6] = math.log1p(float(np.prod(np.maximum(extent, 0.0) + 1.0e-12)))
    features[7] = rel_motion
    features[8] = speed
    features[9] = min_vertex_gap0
    features[10] = min_vertex_gap1
    features[11] = min(min_vertex_gap0, min_vertex_gap1)
    features[12] = center_gap0
    features[13] = center_gap1
    features[14] = min(center_gap0, center_gap1)
    features[15] = near
    features[16] = gap
    features[17] = 1.0 / (1.0 + gap)
    features[18] = near_coplanar_angle
    features[19] = 1.0 / (1.0 + abs(near_coplanar_angle))
    features[20:23] = swept.mean(axis=0).astype(np.float32)
    features[23] = float(np.max(np.abs(swept)))
    features[24] = float(np.dot(motion_a, motion_b))
    features[25] = float(np.linalg.norm(np.cross(motion_a, motion_b)))
    features[26] = float(np.linalg.norm(motion_a))
    features[27] = float(np.linalg.norm(motion_b))
    features[28] = 1.0 if gap < 2.5e-4 else 0.0
    features[29] = 1.0 if abs(near_coplanar_angle) < math.radians(0.7) else 0.0
    features[30] = 1.0
    features[31] = 1.0
    return features


def make_candidate(rng: np.random.Generator, positive: bool) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    kind = int(rng.integers(0, 2))
    gap = float(10 ** rng.uniform(-5.2, -2.7))
    speed = float(rng.uniform(2.0, 12.0))
    angle = float(rng.uniform(-math.radians(0.9), math.radians(0.9)))
    R = rotation_matrix(np.array([1.0, 0.2, 0.0]), angle)
    tangent = R @ np.array([1.0, 0.0, 0.0])
    bitangent = R @ np.array([0.0, 1.0, 0.0])
    normal = R @ np.array([0.0, 0.0, 1.0])
    center = rng.normal(0.0, 0.08, size=3)
    travel = speed * 0.12
    if kind == 0:
        tri_scale = float(rng.uniform(0.35, 0.90))
        tri = np.stack(
            [
                center - 0.50 * tri_scale * tangent - 0.30 * tri_scale * bitangent,
                center + 0.55 * tri_scale * tangent - 0.25 * tri_scale * bitangent,
                center - 0.10 * tri_scale * tangent + 0.55 * tri_scale * bitangent,
            ]
        )
        bary = rng.dirichlet([2.0, 2.0, 2.0])
        base = bary @ tri
        z0 = gap if not positive else gap
        z1 = gap if not positive else -gap
        point_t0 = base - 0.5 * travel * tangent + z0 * normal
        point_t1 = base + 0.5 * travel * tangent + z1 * normal
        tri_t0 = tri.copy()
        tri_t1 = tri + rng.normal(0.0, 0.002, size=tri.shape)
        t0 = np.vstack([point_t0, tri_t0])
        t1 = np.vstack([point_t1, tri_t1])
    else:
        length = float(rng.uniform(0.45, 1.1))
        offset = float(rng.uniform(-0.06, 0.06))
        a0 = center - 0.5 * length * tangent
        a1 = center + 0.5 * length * tangent
        cross_angle = float(rng.uniform(math.radians(2.0), math.radians(10.0)))
        dir_b = math.cos(cross_angle) * tangent + math.sin(cross_angle) * bitangent
        b_center = center + offset * bitangent + gap * normal
        b0 = b_center - 0.5 * length * dir_b
        b1 = b_center + 0.5 * length * dir_b
        if positive:
            # Cross the gap through the other edge for positive training rows.
            b0_t1 = b0 - 2.0 * gap * normal + 0.15 * travel * tangent
            b1_t1 = b1 - 2.0 * gap * normal + 0.15 * travel * tangent
        else:
            b0_t1 = b0 + 0.35 * travel * tangent
            b1_t1 = b1 + 0.35 * travel * tangent
        t0 = np.vstack([a0 - 0.25 * travel * tangent, a1 - 0.25 * travel * tangent, b0, b1])
        t1 = np.vstack([a0 + 0.25 * travel * tangent, a1 + 0.25 * travel * tangent, b0_t1, b1_t1])
    features = candidate_features(kind, t0, t1, gap, speed, angle)
    return kind, t0.astype(np.float64), t1.astype(np.float64), features, gap, speed, abs(angle)


def make_config(cpp: Any) -> Any:
    cfg = cpp.CertificateEngineConfig()
    cfg.eps_time = 1.0e-6
    cfg.eps_space = 1.0e-8
    cfg.max_subdivision_depth = 8
    return cfg


def build_training_pool(cpp: Any, cfg: Any, rows: int, seed: int) -> dict[str, np.ndarray]:
    path = BENCH_DIR / f"{CASE_NAME}_train_pool_{rows}.npz"
    if path.exists():
        loaded = np.load(path)
        return {key: loaded[key] for key in loaded.files}
    rng = np.random.default_rng(seed)
    kinds = np.empty(rows, dtype=np.int8)
    t0 = np.empty((rows, 4, 3), dtype=np.float64)
    t1 = np.empty((rows, 4, 3), dtype=np.float64)
    features = np.empty((rows, 32), dtype=np.float32)
    labels = np.empty(rows, dtype=np.bool_)
    gaps = np.empty(rows, dtype=np.float32)
    speeds = np.empty(rows, dtype=np.float32)
    angles = np.empty(rows, dtype=np.float32)
    exact_status = np.empty(rows, dtype=np.int16)
    i = 0
    attempts = 0
    while i < rows:
        attempts += 1
        positive_request = rng.random() < 0.18
        kind, c0, c1, feat, gap, speed, angle = make_candidate(rng, positive=positive_request)
        positive, status = exact_positive(cpp, cfg, kind, c0, c1)
        # Keep both actual positives and hard negatives; exact output is the label.
        kinds[i] = kind
        t0[i] = c0
        t1[i] = c1
        features[i] = feat
        labels[i] = positive
        gaps[i] = gap
        speeds[i] = speed
        angles[i] = angle
        exact_status[i] = status
        i += 1
        if attempts > rows * 20:
            raise RuntimeError("failed to generate training pool")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        kind=kinds,
        t0=t0,
        t1=t1,
        features=features,
        labels=labels,
        gaps=gaps,
        speeds=speeds,
        near_coplanar_angles=angles,
        exact_status=exact_status,
    )
    return {
        "kind": kinds,
        "t0": t0,
        "t1": t1,
        "features": features,
        "labels": labels,
        "gaps": gaps,
        "speeds": speeds,
        "near_coplanar_angles": angles,
        "exact_status": exact_status,
    }


def build_eval_pool(cpp: Any, cfg: Any, rows: int, seed: int) -> dict[str, np.ndarray]:
    path = BENCH_DIR / f"{CASE_NAME}_eval_near_miss_pool_{rows}.npz"
    if path.exists():
        loaded = np.load(path)
        return {key: loaded[key] for key in loaded.files}
    rng = np.random.default_rng(seed)
    kinds = np.empty(rows, dtype=np.int8)
    t0 = np.empty((rows, 4, 3), dtype=np.float64)
    t1 = np.empty((rows, 4, 3), dtype=np.float64)
    features = np.empty((rows, 32), dtype=np.float32)
    gaps = np.empty(rows, dtype=np.float32)
    speeds = np.empty(rows, dtype=np.float32)
    angles = np.empty(rows, dtype=np.float32)
    exact_status = np.empty(rows, dtype=np.int16)
    i = 0
    attempts = 0
    while i < rows:
        attempts += 1
        kind, c0, c1, feat, gap, speed, angle = make_candidate(rng, positive=False)
        positive, status = exact_positive(cpp, cfg, kind, c0, c1)
        if positive:
            continue
        kinds[i] = kind
        t0[i] = c0
        t1[i] = c1
        features[i] = feat
        gaps[i] = gap
        speeds[i] = speed
        angles[i] = angle
        exact_status[i] = status
        i += 1
        if attempts > rows * 20:
            raise RuntimeError("failed to generate enough exact-separated near-miss candidates")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        kind=kinds,
        t0=t0,
        t1=t1,
        features=features,
        gaps=gaps,
        speeds=speeds,
        near_coplanar_angles=angles,
        exact_status=exact_status,
    )
    return {
        "kind": kinds,
        "t0": t0,
        "t1": t1,
        "features": features,
        "gaps": gaps,
        "speeds": speeds,
        "near_coplanar_angles": angles,
        "exact_status": exact_status,
    }


def train_model(pool: dict[str, np.ndarray], seed: int, epochs: int, device: str) -> tuple[TinySTPF, dict[str, Any]]:
    features = np.asarray(pool["features"], dtype=np.float32)
    labels = np.asarray(pool["labels"], dtype=np.bool_)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = np.arange(features.shape[0])
    rng.shuffle(order)
    split = int(0.8 * len(order))
    train_idx = order[:split]
    val_idx = order[split:]
    model = TinySTPF(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    x_train = torch.as_tensor(features[train_idx], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(labels[train_idx].astype(np.float32), dtype=torch.float32, device=device)
    pos = float(np.sum(labels[train_idx]))
    neg = float(len(train_idx) - pos)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1.0, neg / max(pos, 1.0))], device=device))
    batch = 8192
    history: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(epochs):
        perm = torch.randperm(x_train.shape[0], device=device)
        losses = []
        for begin in range(0, len(perm), batch):
            idx = perm[begin : begin + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            pred = torch.sigmoid(model(torch.as_tensor(features[val_idx], dtype=torch.float32, device=device))).cpu().numpy()
        y_val = labels[val_idx]
        if np.any(y_val) and np.any(~y_val):
            auc_proxy = float(np.mean(pred[y_val][:, None] > pred[~y_val][None, :]))
        else:
            auc_proxy = 0.0
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_rank_auc_proxy": auc_proxy})
    train_ms = (time.perf_counter() - start) * 1000.0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = OUTPUT_DIR / "model_state.pt"
    torch.save(
        {
            "model": "TinySTPF",
            "state_dict": model.state_dict(),
            "feature_dim": features.shape[1],
            "seed": seed,
            "epochs": epochs,
            "history": history,
        },
        checkpoint,
    )
    return model, {
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "positive_ratio": float(np.mean(labels)),
        "epochs": int(epochs),
        "device": device,
        "train_ms": train_ms,
        "history": history,
        "checkpoint": checkpoint.as_posix(),
    }


def evaluate_candidate(cpp: Any, cfg: Any, pool: dict[str, np.ndarray], idx: int) -> tuple[bool, int]:
    return exact_positive(cpp, cfg, int(pool["kind"][idx]), pool["t0"][idx], pool["t1"][idx])


def evaluate_method(
    cpp: Any,
    cfg: Any,
    pool: dict[str, np.ndarray],
    groups: list[list[int]],
    scores: np.ndarray,
    method: str,
    proposal_ms: float,
    seed: int,
    no_proposal_wall_ms: float | None = None,
) -> MethodResult:
    rng = np.random.default_rng(seed)
    exact_calls = 0
    fp = 0
    tn = 0
    exact_ms = 0.0
    wall_begin = time.perf_counter()
    for group in groups:
        if method == "Random+near-miss-exact":
            order = list(group)
            rng.shuffle(order)
        elif method == "RTSTPFExact+near-miss-exact":
            order = sorted(group, key=lambda i: float(scores[i]), reverse=True)
        else:
            order = list(group)
        predicted_positive = False
        for idx in order:
            exact_begin = time.perf_counter()
            positive, _status = evaluate_candidate(cpp, cfg, pool, idx)
            exact_ms += (time.perf_counter() - exact_begin) * 1000.0
            exact_calls += 1
            if positive:
                predicted_positive = True
                break
        if predicted_positive:
            fp += 1
        else:
            tn += 1
    wall_ms = (time.perf_counter() - wall_begin) * 1000.0 + proposal_ms
    candidates = len(groups) * len(groups[0])
    baseline = wall_ms if no_proposal_wall_ms is None else no_proposal_wall_ms
    return MethodResult(
        method=method,
        groups=len(groups),
        candidates=candidates,
        exact_calls=exact_calls,
        skipped_exact_calls=candidates - exact_calls,
        tn=tn,
        fp=fp,
        fn=0,
        false_positive_rate=fp / max(1, len(groups)),
        exact_ms=exact_ms,
        proposal_ms=proposal_ms,
        wall_ms=wall_ms,
        overhead_vs_no_proposal_ms=wall_ms - baseline,
    )


def make_groups(pool: dict[str, np.ndarray], group_count: int, group_size: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    n = int(pool["features"].shape[0])
    if group_count * group_size > n:
        group_count = n // group_size
    hardness = pool["features"][:, 15] + pool["features"][:, 17] + pool["features"][:, 19] + 0.05 * pool["features"][:, 8]
    order = np.argsort(hardness)[::-1]
    groups: list[list[int]] = []
    for group_id in range(group_count):
        group = [int(v) for v in order[group_id * group_size : (group_id + 1) * group_size]]
        rng.shuffle(group)
        groups.append(group)
    return groups


def render_case() -> dict[str, str]:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    w, h = 1500, 850
    frames: list[np.ndarray] = []
    font = ImageFont.load_default()
    for frame in range(96):
        alpha = frame / 95.0
        x_a = -1.35 + 2.70 * alpha
        x_b = 1.35 - 2.70 * alpha
        z_gap = 0.055
        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img, "RGBA")
        # Coordinate projection for two long thin plates sliding past each other.
        def proj(p: np.ndarray) -> tuple[float, float]:
            u = 0.90 * p[0] - 0.38 * p[1]
            v = 0.30 * p[0] + 0.56 * p[1] - 0.82 * p[2]
            return w * 0.5 + 260.0 * u, h * 0.58 - 260.0 * v

        def plate(center: np.ndarray, color: tuple[int, int, int, int], yaw: float) -> None:
            R = np.array(
                [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
            )
            local = np.array(
                [
                    [-0.85, -0.045, 0.0],
                    [0.85, -0.045, 0.0],
                    [0.85, 0.045, 0.0],
                    [-0.85, 0.045, 0.0],
                ],
                dtype=np.float64,
            )
            pts = [proj(center + p @ R.T) for p in local]
            draw.polygon(pts, fill=color, outline=(20, 40, 60, 180))
            for i in range(18):
                t = -0.8 + i * 0.094
                p0 = proj(center + np.array([t, -0.045, 0.0]) @ R.T)
                p1 = proj(center + np.array([t, 0.045, 0.0]) @ R.T)
                draw.line([p0, p1], fill=(20, 40, 60, 70), width=1)

        draw.rectangle([36, 36, w - 36, 120], outline=(96, 132, 190, 210), width=2)
        draw.text((56, 54), "Hard-negative near-miss dense CCD: high-speed thin features, no collision", fill=(22, 30, 45), font=font)
        draw.text((56, 82), "Near-coplanar sliding with a small positive gap; fallback evaluates every dense negative candidate.", fill=(64, 75, 95), font=font)
        plate(np.array([x_a, -0.055, 0.0]), (54, 149, 232, 120), 0.03)
        plate(np.array([x_b, 0.055, z_gap]), (238, 103, 83, 120), -0.03)
        draw.line([proj(np.array([-0.25, 0.0, 0.0])), proj(np.array([0.25, 0.0, z_gap]))], fill=(30, 30, 30, 120), width=2)
        draw.text((56, h - 56), f"t={alpha*2.4:.2f}s | min gap > 0 | expected group truth: no collision", fill=(30, 48, 70), font=font)
        frames.append(np.asarray(img))
    mp4 = DEMO_DIR / "global.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    sheet = Image.new("RGB", (w * 2, h * 2), (255, 255, 255))
    for i, idx in enumerate([0, 31, 47, 95]):
        sheet.paste(Image.fromarray(frames[idx]), ((i % 2) * w, (i // 2) * h))
    sheet_path = DEMO_DIR / "contact_sheet.png"
    sheet.save(sheet_path)
    return {"global_mp4": mp4.as_posix(), "contact_sheet": sheet_path.as_posix()}


def write_reports(payload: dict[str, Any]) -> None:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    write_json(BENCH_DIR / f"{CASE_NAME}.json", payload)
    write_json(DEMO_DIR / "metrics.json", payload)
    rows = payload["benchmark"]["methods"]
    lines = [
        "# Hard Negative / Near-miss Dense CCD Benchmark",
        "",
        "## Scope",
        "",
        "- Constructed near-tangent, near-coplanar, thin-feature high-speed sweep.",
        "- The evaluated dense groups are analytically intended to be collision-free and are pre-filtered by exact primitive separation.",
        "- STPF is not allowed to declare collision-free. Negative groups therefore fall back to all-exact evaluation.",
        "- This case reports FP and overhead, not acceleration.",
        "",
        "## Dataset",
        "",
        f"- Training rows: `{payload['training']['train_rows']}`",
        f"- Validation rows: `{payload['training']['validation_rows']}`",
        f"- Training positive ratio: `{payload['training']['positive_ratio']:.6f}`",
        f"- Eval near-miss rows: `{payload['eval_pool']['rows']}`",
        f"- Eval min gap range: `{payload['eval_pool']['min_gap']:.8f}` to `{payload['eval_pool']['max_gap']:.8f}`",
        f"- Eval speed range: `{payload['eval_pool']['min_speed']:.3f}` to `{payload['eval_pool']['max_speed']:.3f}`",
        f"- Eval near-coplanar angle max deg: `{payload['eval_pool']['max_near_coplanar_angle_deg']:.6f}`",
        "",
        "## Benchmark",
        "",
        "| Method | Groups | Candidates | Exact calls | Skipped calls | TN | FP | FN | FP rate | Proposal ms | Exact ms | Wall ms | Overhead vs NoProposal ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['groups']} | {row['candidates']} | {row['exact_calls']} | "
            f"{row['skipped_exact_calls']} | {row['tn']} | {row['fp']} | {row['fn']} | "
            f"{row['false_positive_rate']:.6f} | {row['proposal_ms']:.3f} | {row['exact_ms']:.3f} | "
            f"{row['wall_ms']:.3f} | {row['overhead_vs_no_proposal_ms']:.3f} |"
        )
    lines.extend(["", "## Visualization", ""])
    if payload["visualization"]:
        lines.extend(
            [
                f"- MP4: `{payload['visualization']['global_mp4']}`",
                f"- Contact sheet: `{payload['visualization']['contact_sheet']}`",
            ]
        )
    else:
        lines.append("- Skipped in this run (`--skip-render`).")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is an intentionally hard negative case. No method should skip exact fallback and claim separation from the learned model alone.",
            "- Since every evaluated group is negative, early-stop cannot reduce exact work. The expected safe behavior is `exact_calls = candidates`, `FP` reported, and `FN=0`.",
            "- Any additional RTSTPF wall time is proposal overhead paid for safety in this negative regime.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (BENCH_DIR / f"{CASE_NAME}.md").write_text(report, encoding="utf-8")
    (DEMO_DIR / "case_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=180000)
    parser.add_argument("--eval-rows", type=int, default=131072)
    parser.add_argument("--group-count", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=fixed_seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    cpp = _try_load_p2cccd_cpp()
    cfg = make_config(cpp)
    train_pool = build_training_pool(cpp, cfg, args.train_rows, args.seed)
    eval_pool = build_eval_pool(cpp, cfg, args.eval_rows, args.seed + 1)
    model, training = train_model(train_pool, args.seed, args.epochs, args.device)
    with torch.no_grad():
        proposal_begin = time.perf_counter()
        scores = torch.sigmoid(
            model(torch.as_tensor(eval_pool["features"], dtype=torch.float32, device=args.device))
        ).cpu().numpy()
        proposal_ms = (time.perf_counter() - proposal_begin) * 1000.0
    groups = make_groups(eval_pool, args.group_count, args.group_size, args.seed + 2)
    no_prop = evaluate_method(cpp, cfg, eval_pool, groups, scores, "NoProposal+near-miss-exact", 0.0, args.seed)
    random_result = evaluate_method(
        cpp,
        cfg,
        eval_pool,
        groups,
        scores,
        "Random+near-miss-exact",
        0.0,
        args.seed + 3,
        no_prop.wall_ms,
    )
    learned = evaluate_method(
        cpp,
        cfg,
        eval_pool,
        groups,
        scores,
        "RTSTPFExact+near-miss-exact",
        proposal_ms,
        args.seed + 4,
        no_prop.wall_ms,
    )
    viz = {} if args.skip_render else render_case()
    eval_status = np.asarray(eval_pool["exact_status"], dtype=np.int16)
    separation_value = int(cpp.CertificateStatus.SEPARATION)
    collision_value = int(cpp.CertificateStatus.COLLISION)
    payload = {
        "case": CASE_NAME,
        "training": training,
        "eval_pool": {
            "rows": int(args.eval_rows),
            "min_gap": float(np.min(eval_pool["gaps"])),
            "max_gap": float(np.max(eval_pool["gaps"])),
            "min_speed": float(np.min(eval_pool["speeds"])),
            "max_speed": float(np.max(eval_pool["speeds"])),
            "max_near_coplanar_angle_deg": float(np.max(eval_pool["near_coplanar_angles"]) * 180.0 / math.pi),
            "exact_separation_status_count": int(np.sum(eval_status == separation_value)),
            "exact_collision_status_count": int(np.sum(eval_status == collision_value)),
            "exact_other_status_count": int(np.sum((eval_status != separation_value) & (eval_status != collision_value))),
        },
        "benchmark": {
            "group_count": len(groups),
            "group_size": int(args.group_size),
            "candidate_count": int(len(groups) * args.group_size),
            "methods": [no_prop.__dict__, random_result.__dict__, learned.__dict__],
        },
        "visualization": viz,
        "reproduce": {
            "command": (
                "conda activate cudadev; $env:PYTHONPATH='src\\python'; "
                "python src\\tools\\hard_negative_near_miss_dense_case.py "
                f"--train-rows {args.train_rows} --eval-rows {args.eval_rows} "
                f"--group-count {args.group_count} --group-size {args.group_size} "
                f"--epochs {args.epochs} --device {args.device}"
            )
        },
    }
    write_reports(payload)


if __name__ == "__main__":
    main()
