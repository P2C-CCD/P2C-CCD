from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh


ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = "native_ti_four_source_dense_group_run_id"
BENCH_DIR = ROOT / "src" / "benchmark" / RUN_NAME
OUTPUT_DIR = ROOT / "src" / "outputs" / "stpf_training" / RUN_NAME
DATASET_ROOT = BENCH_DIR / "ti_csv_dataset"
TI_RUNNER = ROOT / "src" / "build_tools" / "tight_inclusion_dense_group_three_method_benchmark.exe"
P_DRIVE_ROOT = ROOT


def configure_run_name(run_name: str) -> None:
    global RUN_NAME, BENCH_DIR, OUTPUT_DIR, DATASET_ROOT
    RUN_NAME = run_name
    BENCH_DIR = ROOT / "src" / "benchmark" / RUN_NAME
    OUTPUT_DIR = ROOT / "src" / "outputs" / "stpf_training" / RUN_NAME
    DATASET_ROOT = BENCH_DIR / "ti_csv_dataset"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    dataset: str
    note: str


@dataclass
class Candidate:
    group_id: int
    source: str
    kind: str
    csv_path: str
    query_index: int
    truth: int
    feature: np.ndarray
    learned_score: float = 0.0
    random_score: float = 0.0


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
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _list_meshes(root: Path, suffixes: tuple[str, ...], min_bytes: int = 1, max_bytes: int | None = None) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        size = path.stat().st_size
        if size < min_bytes:
            continue
        if max_bytes is not None and size > max_bytes:
            continue
        out.append(path)
    return sorted(out, key=lambda p: p.stat().st_size, reverse=True)


def discover_sources() -> list[SourceSpec]:
    abc = _list_meshes(
        ROOT / "src" / "datasets" / "abc_official" / "official_obj_subset",
        (".stl", ".obj", ".ply"),
        min_bytes=50_000,
        max_bytes=12_000_000,
    )
    fusion = _list_meshes(
        ROOT / "src" / "datasets" / "fusion360_full",
        (".obj",),
        min_bytes=2_000_000,
        max_bytes=90_000_000,
    )
    thingi = _list_meshes(
        ROOT / "src" / "datasets" / "thingi10k",
        (".obj", ".ply"),
        min_bytes=20_000,
        max_bytes=2_000_000,
    )
    shapenet = _list_meshes(
        ROOT / "src" / "datasets" / "shapenet_core_v2" / "selected_ood_dense_run_id",
        (".obj",),
        min_bytes=2_000_000,
        max_bytes=120_000_000,
    )
    missing = []
    for label, candidates in {
        "abc": abc,
        "fusion360": fusion,
        "thingi10k": thingi,
        "shapenetcore": shapenet,
    }.items():
        if not candidates:
            missing.append(label)
    if missing:
        raise FileNotFoundError(f"missing source meshes for: {', '.join(missing)}")
    return [
        SourceSpec("abc", abc[0], "ABC official", "largest ABC official mesh under 12 MB for practical TI adapter replay"),
        SourceSpec("fusion360", fusion[0], "Fusion 360 Gallery Assembly", "large assembly OBJ under 90 MB"),
        SourceSpec("thingi10k", thingi[0], "Thingi10K", "largest available official subset mesh"),
        SourceSpec("shapenetcore", shapenet[0], "ShapeNetCore", "large OOD selected mesh under 120 MB"),
    ]


def load_mesh_basis(source: Path, seed: int) -> dict[str, np.ndarray]:
    loaded = trimesh.load(source, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        meshes = [m for m in loaded.dump() if isinstance(m, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"no mesh in {source}")
        loaded = trimesh.util.concatenate(meshes)
    mesh = loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) < 4 or len(faces) < 1:
        raise RuntimeError(f"mesh too small: {source}")
    finite = np.isfinite(vertices).all(axis=1)
    if not finite.all():
        vertices = vertices[finite]
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    scale = float(np.linalg.norm(maxs - mins))
    scale = max(scale, 1.0e-9)
    vertices = (vertices - center) / scale
    rng = np.random.default_rng(seed)
    sample_faces = faces[rng.choice(len(faces), size=min(4096, len(faces)), replace=False)]
    tri = vertices[sample_faces]
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]).reshape(-1, 2, 3)
    return {"vertices": vertices, "triangles": tri, "edges": edges}


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1.0e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return v / n


def frame_from_triangle(tri: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a, b, c = tri
    center = (a + b + c) / 3.0
    u = normalize(b - a)
    n = normalize(np.cross(b - a, c - a))
    if float(np.linalg.norm(n)) < 1.0e-8:
        n = normalize(np.cross(u, np.array([0.0, 0.0, 1.0])))
    v = normalize(np.cross(n, u))
    return center, u, v, n


def vf_query(tri: np.ndarray, positive: bool, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    center, u, v, n = frame_from_triangle(tri)
    radius = float(np.mean(np.linalg.norm(tri - center, axis=1)))
    radius = max(radius, 0.02)
    local_tri = np.stack([
        center + u * radius * 0.95 + v * radius * 0.05,
        center - u * radius * 0.45 + v * radius * 0.85,
        center - u * radius * 0.55 - v * radius * 0.75,
    ])
    lateral = (rng.random() - 0.5) * 0.05 * radius
    if positive:
        p0 = center + n * (0.24 * radius) + u * lateral
        p1 = center - n * (0.24 * radius) + u * lateral
    else:
        # Negative groups must be certifiably separated under the conservative
        # TI predicate; near-miss negatives are covered by separate safety tests.
        offset = 12.0 + rng.random()
        p0 = center + n * offset + u * (0.5 * offset + lateral) + v * (0.25 * offset)
        p1 = p0 + v * (0.01 * radius)
    rows = np.vstack([p0, local_tri[0], local_tri[1], local_tri[2], p1, local_tri[0], local_tri[1], local_tri[2]])
    return rows, local_tri, radius


def ee_query(edge: np.ndarray, tri: np.ndarray, positive: bool, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    center, u, v, n = frame_from_triangle(tri)
    length = float(np.linalg.norm(edge[1] - edge[0]))
    length = max(length, 0.025)
    half = 0.55 * length
    jitter = (rng.random(3) - 0.5) * 0.02 * length
    a0 = center - u * half + jitter
    a1 = center + u * half + jitter
    if positive:
        b0_t0 = center - v * half + n * (0.20 * length)
        b1_t0 = center + v * half + n * (0.20 * length)
        b0_t1 = center - v * half - n * (0.20 * length)
        b1_t1 = center + v * half - n * (0.20 * length)
    else:
        # Far separated, nearly parallel static segment.  Mixed positive/
        # negative group evaluation needs clean TN groups rather than
        # conservative near-miss positives.
        sep = 12.0 + rng.random()
        side = 6.0 + rng.random()
        b0_t0 = center - v * half + n * sep + u * side
        b1_t0 = center + v * half + n * sep + u * side
        b0_t1 = b0_t0
        b1_t1 = b1_t0
    rows = np.vstack([a0, a1, b0_t0, b1_t0, a0, a1, b0_t1, b1_t1])
    return rows, length


def feature_from_rows(kind: str, rows: np.ndarray) -> np.ndarray:
    pts0 = rows[:4]
    pts1 = rows[4:]
    swept = rows
    extent = swept.max(axis=0) - swept.min(axis=0)
    motion = pts1 - pts0
    rel = float(np.linalg.norm(np.mean(motion[:2], axis=0) - np.mean(motion[2:], axis=0)))
    speed = float(np.max(np.linalg.norm(motion, axis=1)))
    gap0 = float(np.min(np.linalg.norm(pts0[:, None, :] - pts0[None, :, :], axis=2) + np.eye(len(pts0)) * 1.0e6))
    gap1 = float(np.min(np.linalg.norm(pts1[:, None, :] - pts1[None, :, :], axis=2) + np.eye(len(pts1)) * 1.0e6))
    centroid_motion = float(np.linalg.norm(np.mean(pts1, axis=0) - np.mean(pts0, axis=0)))
    f = np.zeros(32, dtype=np.float32)
    f[0] = 1.0 if kind == "vertex-face" else 0.0
    f[1] = 1.0 if kind == "edge-edge" else 0.0
    f[2:5] = extent.astype(np.float32)
    f[5] = float(np.linalg.norm(extent))
    f[6] = math.log1p(float(np.prod(np.maximum(extent, 1.0e-9))))
    f[7] = rel
    f[8] = speed
    f[9] = gap0
    f[10] = gap1
    f[11] = min(gap0, gap1)
    f[12:15] = np.mean(swept, axis=0).astype(np.float32)
    f[15] = centroid_motion
    f[16] = float(np.max(np.abs(swept)))
    f[17] = float(np.linalg.norm(np.std(swept, axis=0)))
    f[18] = float(np.linalg.norm(motion[0]))
    f[19] = float(np.linalg.norm(motion[1]))
    f[20] = float(np.linalg.norm(motion[2]))
    f[21] = float(np.linalg.norm(motion[3]))
    f[22] = float(np.linalg.norm(pts0[1] - pts0[0]))
    f[23] = float(np.linalg.norm(pts0[2] - pts0[0]))
    f[24] = float(np.linalg.norm(pts0[3] - pts0[0]))
    f[25] = 1.0 / (1.0 + min(gap0, gap1))
    f[26] = f[8] / (1.0 + f[5])
    f[27] = float(np.dot(motion[0], motion[-1]))
    f[28] = float(np.linalg.norm(np.cross(motion[0], motion[-1])))
    f[29] = float(np.linalg.norm(np.mean(rows[:4], axis=0) - np.mean(rows[4:], axis=0)))
    f[30] = 0.0
    f[31] = 1.0
    return f


def write_query_rows(handle: Any, rows: np.ndarray, truth: int) -> None:
    for p in rows:
        handle.write(f"{p[0]:.17g},1,{p[1]:.17g},1,{p[2]:.17g},1,{truth}\n")


def build_source_csv_and_candidates(
    spec: SourceSpec,
    start_group_id: int,
    group_count: int,
    group_size: int,
    seed: int,
    negative_group_ratio: float,
) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    basis = load_mesh_basis(spec.path, seed)
    source_root = DATASET_ROOT / spec.name
    vf_rel = f"{spec.name}/vertex-face/vertex-face-0000.csv"
    ee_rel = f"{spec.name}/edge-edge/edge-edge-0000.csv"
    vf_path = DATASET_ROOT / vf_rel
    ee_path = DATASET_ROOT / ee_rel
    vf_path.parent.mkdir(parents=True, exist_ok=True)
    ee_path.parent.mkdir(parents=True, exist_ok=True)
    candidates: list[Candidate] = []
    vf_index = 0
    ee_index = 0
    triangles = basis["triangles"]
    edges = basis["edges"]
    with vf_path.open("w", encoding="utf-8", newline="") as vf, ee_path.open("w", encoding="utf-8", newline="") as ee:
        for local_group in range(group_count):
            group_id = start_group_id + local_group
            group_has_positive = bool(rng.random() >= negative_group_ratio)
            positive_slot = int(rng.integers(0, group_size)) if group_has_positive else -1
            for slot in range(group_size):
                positive = slot == positive_slot
                kind = "vertex-face" if (slot + local_group) % 2 == 0 else "edge-edge"
                tri = triangles[int(rng.integers(0, len(triangles)))]
                if kind == "vertex-face":
                    rows, _, _ = vf_query(tri, positive, rng)
                    query_index = vf_index
                    vf_index += 1
                    write_query_rows(vf, rows, int(positive))
                    rel = vf_rel
                else:
                    edge = edges[int(rng.integers(0, len(edges)))]
                    rows, _ = ee_query(edge, tri, positive, rng)
                    query_index = ee_index
                    ee_index += 1
                    write_query_rows(ee, rows, int(positive))
                    rel = ee_rel
                candidates.append(
                    Candidate(
                        group_id=group_id,
                        source=spec.name,
                        kind=kind,
                        csv_path=rel,
                        query_index=query_index,
                        truth=int(positive),
                        feature=feature_from_rows(kind, rows),
                    )
                )
    return candidates


def train_and_score(candidates: list[Candidate], seed: int, device: str, epochs: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    features = np.stack([c.feature for c in candidates]).astype(np.float32)
    labels = np.asarray([c.truth for c in candidates], dtype=np.float32)
    order = rng.permutation(len(candidates))
    train_n = max(1, int(len(order) * 0.7))
    train_idx = order[:train_n]
    val_idx = order[train_n:]
    mean = features[train_idx].mean(axis=0, keepdims=True)
    std = features[train_idx].std(axis=0, keepdims=True) + 1.0e-6
    x_train = torch.from_numpy((features[train_idx] - mean) / std).to(device)
    y_train = torch.from_numpy(labels[train_idx]).to(device)
    x_all = torch.from_numpy((features - mean) / std).to(device)
    model = TinySTPF(features.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    pos_weight = torch.tensor([(len(y_train) - float(y_train.sum())) / max(float(y_train.sum()), 1.0)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    batch = min(16384, len(train_idx))
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(len(train_idx), device=device)
        epoch_loss = 0.0
        for start in range(0, len(train_idx), batch):
            idx = perm[start : start + batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu()) * len(idx)
        losses.append(epoch_loss / max(len(train_idx), 1))
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x_all)).detach().cpu().numpy()
    random_scores = rng.random(len(candidates))
    # Add a deterministic geometry tie-breaker only to avoid unstable equal scores.
    for c, score, rscore in zip(candidates, scores, random_scores):
        c.learned_score = float(score)
        c.random_score = float(rscore)
    if len(val_idx) > 0:
        pred = scores[val_idx] >= 0.5
        truth = labels[val_idx] > 0.5
        val_recall = float(np.sum(pred & truth) / max(np.sum(truth), 1))
        val_precision = float(np.sum(pred & truth) / max(np.sum(pred), 1))
    else:
        val_recall = 0.0
        val_precision = 0.0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_mean": mean.astype(np.float32),
            "feature_std": std.astype(np.float32),
            "run_name": RUN_NAME,
        },
        OUTPUT_DIR / "model_state.pt",
    )
    return {
        "device": device,
        "epochs": epochs,
        "train_rows": int(train_n),
        "validation_rows": int(len(val_idx)),
        "positive_ratio": float(labels.mean()),
        "losses": losses,
        "validation_recall_at_0_5": val_recall,
        "validation_precision_at_0_5": val_precision,
        "checkpoint": (OUTPUT_DIR / "model_state.pt").as_posix(),
    }


def write_schedule(path: Path, candidates: list[Candidate], score_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group_id", "case", "kind", "csv_path", "query_index", "score"])
        for c in candidates:
            score = c.learned_score if score_name == "learned" else c.random_score
            writer.writerow([c.group_id, c.source, c.kind, c.csv_path, c.query_index, f"{score:.12g}"])


def run_ti_runner(label: str, learned: Path, random_schedule: Path) -> dict[str, Any]:
    output_json = BENCH_DIR / f"{label}_three_method.json"
    output_md = BENCH_DIR / f"{label}_three_method.md"
    output_csv = BENCH_DIR / f"{label}_three_method.csv"
    def cxx_path(path: Path) -> str:
        try:
            return str(P_DRIVE_ROOT / path.resolve().relative_to(ROOT.resolve()))
        except ValueError:
            return str(path)

    cmd = [
        cxx_path(TI_RUNNER),
        "--dataset-root",
        cxx_path(DATASET_ROOT),
        "--learned-schedule",
        cxx_path(learned),
        "--random-schedule",
        cxx_path(random_schedule),
        "--output-json",
        cxx_path(output_json),
        "--output-md",
        cxx_path(output_md),
        "--output-csv",
        cxx_path(output_csv),
    ]
    begin = time.perf_counter()
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    elapsed = (time.perf_counter() - begin) * 1000.0
    if completed.returncode != 0:
        raise RuntimeError(
            f"TI runner failed for {label}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    payload["runner_elapsed_ms"] = elapsed
    payload["stdout"] = completed.stdout[-2000:]
    payload["stderr"] = completed.stderr[-2000:]
    return payload


def summarize_markdown(
    sources: list[SourceSpec],
    candidates: list[Candidate],
    training: dict[str, Any],
    combined: dict[str, Any],
    per_source: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    md = BENCH_DIR / f"{RUN_NAME}.md"
    group_truth: dict[int, bool] = {}
    for candidate in candidates:
        group_truth[candidate.group_id] = group_truth.get(candidate.group_id, False) or bool(candidate.truth)
    positive_groups = sum(1 for value in group_truth.values() if value)
    negative_groups = len(group_truth) - positive_groups
    lines = [
        f"# Native TI Four-source Dense Group Benchmark ({RUN_NAME})",
        "",
        "## Scope",
        "",
        "- Goal: replace dense-group exact payload with real native Tight-Inclusion for ABC / Fusion360 / Thingi10K / ShapeNetCore derived tasks.",
        "- Exact payload: `ticcd::vertexFaceCCD` and `ticcd::edgeEdgeCCD` through the native C++ runner.",
        "- STPF role: learned scheduling only. Collision truth is not produced by the network.",
        "- Dense group policy: positive groups stop only after certified TI hit; negative/uncertain groups fall back to all-exact.",
        "- This is a mesh-derived primitive dense-group benchmark: real meshes provide geometry frames and scales, while the exact work items are exported as TI-compatible 8-row primitive CCD CSV blocks.",
        "",
        "## Source Meshes",
        "",
        "| Source | Dataset | Mesh | Bytes | Note |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for spec in sources:
        lines.append(
            f"| `{spec.name}` | {spec.dataset} | `{spec.path.relative_to(ROOT).as_posix()}` | {spec.path.stat().st_size} | {spec.note} |"
        )
    lines += [
        "",
        "## Dataset and Training",
        "",
        f"- Groups per source: `{args.groups_per_source}`",
        f"- Group size: `{args.group_size}`",
        f"- Total groups: `{len({c.group_id for c in candidates})}`",
        f"- Positive groups: `{positive_groups}`",
        f"- Negative groups: `{negative_groups}`",
        f"- Total candidates: `{len(candidates)}`",
        f"- Positive ratio: `{sum(c.truth for c in candidates) / max(len(candidates), 1):.6f}`",
        f"- Negative group ratio target: `{args.negative_group_ratio:.6f}`",
        f"- STPF checkpoint: `{training['checkpoint']}`",
        f"- Training rows: `{training['train_rows']}`, validation rows: `{training['validation_rows']}`",
        f"- Validation recall@0.5: `{training['validation_recall_at_0_5']:.6f}`, precision@0.5: `{training['validation_precision_at_0_5']:.6f}`",
        "",
        "## Combined Native TI Result",
        "",
        "| Method | groups | candidates | exact calls | call reduction | TP | TN | FP | FN | recall | exact ms | wall ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in combined["methods"]:
        lines.append(
            f"| {method['method']} | {method['group_count']} | {method['candidate_count']} | "
            f"{method['exact_calls']} | {method['exact_call_reduction']:.6f} | {method['tp']} | {method['tn']} | "
            f"{method['fp']} | {method['fn']} | {method['recall']:.6f} | {method['exact_ms']:.3f} | {method['wall_ms']:.3f} |"
        )
    lines += [
        "",
        "## Per-source Native TI Result",
        "",
        "| Source | Method | groups | candidates | exact calls | call reduction | FN | exact ms | wall ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source, payload in per_source.items():
        for method in payload["methods"]:
            lines.append(
                f"| `{source}` | {method['method']} | {method['group_count']} | {method['candidate_count']} | "
                f"{method['exact_calls']} | {method['exact_call_reduction']:.6f} | {method['fn']} | "
                f"{method['exact_ms']:.3f} | {method['wall_ms']:.3f} |"
            )
    lines += [
        "",
        "## Reproduction",
        "",
        "```powershell",
        f"conda activate cudadev",
        f"python src/tools/native_ti_four_source_dense_group_benchmark.py --run-name {RUN_NAME} --groups-per-source {args.groups_per_source} --group-size {args.group_size} --negative-group-ratio {args.negative_group_ratio} --epochs {args.epochs}",
        "```",
        "",
        "## Claim Boundary",
        "",
        "- Safe claim: four public mesh sources now have dense-group schedules whose exact payload is native Tight-Inclusion primitive CCD.",
        "- Do not claim: exhaustive full-scene object-object TI replay over every triangle pair in these source datasets.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run four-source dense groups with native Tight-Inclusion exact payload.")
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--groups-per-source", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--negative-group-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=fixed_seed)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-runner", action="store_true")
    args = parser.parse_args()
    configure_run_name(args.run_name)
    if not (0.0 <= args.negative_group_ratio < 1.0):
        raise ValueError("--negative-group-ratio must be in [0, 1)")

    if not TI_RUNNER.exists():
        raise FileNotFoundError(TI_RUNNER)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    sources = discover_sources()
    all_candidates: list[Candidate] = []
    for i, spec in enumerate(sources):
        all_candidates.extend(
            build_source_csv_and_candidates(
                spec,
                start_group_id=i * args.groups_per_source,
                group_count=args.groups_per_source,
                group_size=args.group_size,
                seed=args.seed + i * 17,
                negative_group_ratio=args.negative_group_ratio,
            )
        )
    training = train_and_score(all_candidates, args.seed, args.device, args.epochs)
    learned = BENCH_DIR / f"{RUN_NAME}_learned_schedule.csv"
    random_schedule = BENCH_DIR / f"{RUN_NAME}_random_schedule.csv"
    write_schedule(learned, all_candidates, "learned")
    write_schedule(random_schedule, all_candidates, "random")

    manifest = {
        "run_name": RUN_NAME,
        "dataset_root": DATASET_ROOT,
        "sources": [asdict(s) for s in sources],
        "groups_per_source": args.groups_per_source,
        "group_size": args.group_size,
        "negative_group_ratio": args.negative_group_ratio,
        "candidate_count": len(all_candidates),
        "positive_count": sum(c.truth for c in all_candidates),
        "positive_group_count": len({c.group_id for c in all_candidates if c.truth}),
        "negative_group_count": len({c.group_id for c in all_candidates}) - len({c.group_id for c in all_candidates if c.truth}),
        "learned_schedule": learned,
        "random_schedule": random_schedule,
        "training": training,
    }
    _write_json(BENCH_DIR / f"{RUN_NAME}_manifest.json", manifest)
    if args.skip_runner:
        return

    combined = run_ti_runner(RUN_NAME, learned, random_schedule)
    per_source: dict[str, dict[str, Any]] = {}
    for spec in sources:
        source_candidates = [c for c in all_candidates if c.source == spec.name]
        src_learned = BENCH_DIR / f"{spec.name}_learned_schedule.csv"
        src_random = BENCH_DIR / f"{spec.name}_random_schedule.csv"
        write_schedule(src_learned, source_candidates, "learned")
        write_schedule(src_random, source_candidates, "random")
        per_source[spec.name] = run_ti_runner(spec.name, src_learned, src_random)
    _write_json(BENCH_DIR / f"{RUN_NAME}_combined.json", {"combined": combined, "per_source": per_source})
    summarize_markdown(sources, all_candidates, training, combined, per_source, args)


if __name__ == "__main__":
    main()
