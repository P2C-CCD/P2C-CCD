from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = ROOT / "src" / "baseline" / "Sample-Scalable-CCD-Data"
RUN_NAME = "scalable_ccd_sample_scene_candidate_groups_run_id"
DEFAULT_SHARD_DIR = ROOT / "src" / "datasets" / "training" / "scalable_ccd_scene_groups" / "shards" / RUN_NAME
DEFAULT_BENCHMARK_DIR = ROOT / "src" / "benchmark" / RUN_NAME
STPF_TARGET_MASK_ALL = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
FEATURE_DIM = 32
INTERVAL_BINS = 8
FAMILY_BINS = 8


SCENE_ORDER = (
    "armadillo-rollers",
    "cloth-ball",
    "cloth-funnel",
    "n-body-simulation",
    "puffer-ball",
    "rod-twist",
)


@dataclass(frozen=True)
class GroupSpec:
    scene: str
    scene_id: int
    stem: str
    timestep: int
    kind: str
    query_csv: Path
    boxes_json: Path
    bool_json: Path
    frame0: Path | None
    frame1: Path | None


@dataclass
class GroupSummary:
    scene: str
    kind: str
    timestep: int
    query_count: int
    positives: int
    negatives: int
    positive_fraction: float
    frame0: str
    frame1: str
    query_csv: str
    boxes_json: str
    bool_json: str
    first_positive_rank_heuristic: int
    heuristic_call_reduction: float
    heuristic_work_reduction: float
    oracle_call_reduction: float
    random_expected_call_reduction: float


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()


def string_array(value: str, count: int) -> np.ndarray:
    return np.full(count, value, dtype=f"<U{max(1, len(value))}")


def rational_row_to_xyz(row: list[str]) -> tuple[np.ndarray, float]:
    if len(row) != 6:
        raise ValueError(f"Expected 6 rational columns, got {len(row)}")
    values: list[float] = []
    max_bits = 0
    for i in range(0, 6, 2):
        numerator = int(row[i])
        denominator = int(row[i + 1])
        if denominator == 0:
            raise ZeroDivisionError("Rational denominator is zero")
        values.append(float(numerator) / float(denominator))
        max_bits = max(max_bits, abs(numerator).bit_length(), abs(denominator).bit_length())
    return np.asarray(values, dtype=np.float64), float(max_bits)


def iter_query_vertices(path: Path) -> Iterable[tuple[np.ndarray, np.ndarray, float]]:
    pending: list[np.ndarray] = []
    max_bits = 0.0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            xyz, bits = rational_row_to_xyz(row)
            pending.append(xyz)
            max_bits = max(max_bits, bits)
            if len(pending) == 8:
                block = np.asarray(pending, dtype=np.float64)
                yield block[:4], block[4:], max_bits
                pending.clear()
                max_bits = 0.0
    if pending:
        raise ValueError(f"{path} has {len(pending)} trailing rows; query CSV row count must be divisible by 8")


def segment_segment_distance(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    # Ericson-style closest segment distance, used only for schedule features.
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denom = a * c - b * b
    s_d = denom
    t_d = denom
    if denom < 1.0e-14:
        s_n = 0.0
        s_d = 1.0
        t_n = e
        t_d = c
    else:
        s_n = b * e - c * d
        t_n = a * e - b * d
        if s_n < 0.0:
            s_n = 0.0
            t_n = e
            t_d = c
        elif s_n > s_d:
            s_n = s_d
            t_n = e + b
            t_d = c
    if t_n < 0.0:
        t_n = 0.0
        if -d < 0.0:
            s_n = 0.0
        elif -d > a:
            s_n = s_d
        else:
            s_n = -d
            s_d = a
    elif t_n > t_d:
        t_n = t_d
        if -d + b < 0.0:
            s_n = 0.0
        elif -d + b > a:
            s_n = s_d
        else:
            s_n = -d + b
            s_d = a
    sc = 0.0 if abs(s_n) < 1.0e-14 else s_n / max(s_d, 1.0e-14)
    tc = 0.0 if abs(t_n) < 1.0e-14 else t_n / max(t_d, 1.0e-14)
    d_p = w + sc * u - tc * v
    return float(np.linalg.norm(d_p))


def point_triangle_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))
    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return float(np.linalg.norm(ap - v * ab))
    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return float(np.linalg.norm(ap - w * ac))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(bp + w * (c - b)))
    normal = np.cross(ab, ac)
    norm = float(np.linalg.norm(normal))
    if norm < 1.0e-14:
        return min(float(np.linalg.norm(ap)), float(np.linalg.norm(bp)), float(np.linalg.norm(cp)))
    return abs(float(np.dot(ap, normal))) / norm


def primitive_distance(points: np.ndarray, kind: str) -> float:
    if kind == "ee":
        return segment_segment_distance(points[0], points[1], points[2], points[3])
    return point_triangle_distance(points[0], points[1], points[2], points[3])


def primitive_sizes(points: np.ndarray, kind: str) -> tuple[float, float, float]:
    if kind == "ee":
        len_a = float(np.linalg.norm(points[1] - points[0]))
        len_b = float(np.linalg.norm(points[3] - points[2]))
        degenerate = float(len_a < 1.0e-8 or len_b < 1.0e-8)
        return len_a, len_b, degenerate
    tri_area = 0.5 * float(np.linalg.norm(np.cross(points[2] - points[1], points[3] - points[1])))
    vertex_to_centroid = float(np.linalg.norm(points[0] - points[1:].mean(axis=0)))
    degenerate = float(tri_area < 1.0e-10)
    return vertex_to_centroid, tri_area, degenerate


def sampled_min_distance(points_t0: np.ndarray, points_t1: np.ndarray, kind: str) -> tuple[float, float]:
    best_distance = math.inf
    best_t = 0.0
    for t in np.linspace(0.0, 1.0, 9):
        points = (1.0 - float(t)) * points_t0 + float(t) * points_t1
        distance = primitive_distance(points, kind)
        if distance < best_distance:
            best_distance = distance
            best_t = float(t)
    return float(best_distance), best_t


def feature_row(
    *,
    points_t0: np.ndarray,
    points_t1: np.ndarray,
    kind: str,
    scene_id: int,
    group_size: int,
    timestep: int,
    max_timestep: int,
    local_index: int,
    rational_bits: float,
    box_pair: list[int] | tuple[int, int],
) -> tuple[np.ndarray, float, float, float]:
    velocities = points_t1 - points_t0
    all_points = np.vstack([points_t0, points_t1])
    swept_min = all_points.min(axis=0)
    swept_max = all_points.max(axis=0)
    extent = np.maximum(swept_max - swept_min, 1.0e-12)
    diag = float(np.linalg.norm(extent))
    volume = float(np.prod(extent))
    speed = np.linalg.norm(velocities, axis=1)
    mean_speed = float(np.mean(speed))
    max_speed = float(np.max(speed))
    centroid_motion = float(np.linalg.norm(points_t1.mean(axis=0) - points_t0.mean(axis=0)))
    min_gap, closest_t = sampled_min_distance(points_t0, points_t1, kind)
    size_a, size_b, degenerate = primitive_sizes(points_t0, kind)
    gap_scale = max(1.0e-6, 0.25 * (size_a + size_b) + 0.1 * diag)
    risk_score = math.exp(-min_gap / gap_scale)
    box_a = int(box_pair[0]) if len(box_pair) >= 1 else 0
    box_b = int(box_pair[1]) if len(box_pair) >= 2 else 0

    f = np.zeros(FEATURE_DIM, dtype=np.float32)
    f[0] = 1.0 if kind == "ee" else 0.0
    f[1] = 1.0 if kind == "vf" else 0.0
    f[2 + scene_id] = 1.0
    f[8] = math.log2(max(2, group_size)) / 18.0
    f[9] = float(timestep) / float(max(1, max_timestep))
    f[10] = math.log1p(min_gap)
    f[11] = closest_t
    f[12] = math.log1p(mean_speed)
    f[13] = math.log1p(max_speed)
    f[14] = math.log1p(diag)
    f[15] = math.log1p(volume)
    f[16:19] = np.log1p(extent).astype(np.float32)
    f[19] = math.log1p(centroid_motion)
    f[20] = math.log1p(abs(size_a))
    f[21] = math.log1p(abs(size_b))
    f[22] = risk_score
    f[23] = float(local_index) / float(max(1, group_size - 1))
    f[24] = math.log1p(abs(box_a % 100000)) / 12.0
    f[25] = math.log1p(abs(box_b % 100000)) / 12.0
    f[26] = float(swept_min[2])
    f[27] = float(swept_max[2])
    f[28] = 1.0 if SCENE_ORDER[scene_id] in {"cloth-ball", "cloth-funnel", "rod-twist"} else 0.0
    f[29] = degenerate
    f[30] = rational_bits / 64.0
    f[31] = 1.0

    exact_cost = (1.25 if kind == "ee" else 1.0) * (1.0 + 0.25 * degenerate) * (1.0 + 0.15 * math.log1p(max_speed))
    return f, float(exact_cost), float(risk_score), float(closest_t)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_frame(scene_dir: Path, timestep: int) -> Path | None:
    frames_dir = scene_dir / "frames"
    candidates = sorted(frames_dir.glob(f"*{timestep}.*"))
    if candidates:
        return candidates[0]
    return None


def discover_groups(source_root: Path, *, max_query_files: int = 0) -> list[GroupSpec]:
    groups: list[GroupSpec] = []
    scene_dirs = [source_root / name for name in SCENE_ORDER if (source_root / name).is_dir()]
    extra_scene_dirs = sorted([p for p in source_root.iterdir() if p.is_dir() and p.name not in SCENE_ORDER and not p.name.startswith(".")])
    for scene_dir in scene_dirs + extra_scene_dirs:
        scene_id = SCENE_ORDER.index(scene_dir.name) if scene_dir.name in SCENE_ORDER else len(SCENE_ORDER)
        for query_csv in sorted((scene_dir / "queries").glob("*.csv")):
            stem = query_csv.stem
            match = re.match(r"(\d+)(ee|vf)$", stem)
            if not match:
                continue
            timestep = int(match.group(1))
            kind = match.group(2)
            boxes_json = scene_dir / "boxes" / f"{stem}.json"
            bool_json = scene_dir / "mma_bool" / f"{stem}_mma_bool.json"
            groups.append(
                GroupSpec(
                    scene=scene_dir.name,
                    scene_id=scene_id,
                    stem=stem,
                    timestep=timestep,
                    kind=kind,
                    query_csv=query_csv,
                    boxes_json=boxes_json,
                    bool_json=bool_json,
                    frame0=find_frame(scene_dir, timestep),
                    frame1=find_frame(scene_dir, timestep + 1),
                )
            )
    if max_query_files > 0:
        return groups[:max_query_files]
    return groups


def interval_bin(closest_t: float, positive: bool) -> int:
    if not positive:
        return 0
    return max(0, min(INTERVAL_BINS - 1, int(math.floor(max(0.0, min(0.999, closest_t)) * INTERVAL_BINS))))


def family_index(spec: GroupSpec) -> int:
    if spec.kind == "ee":
        return 0
    if spec.kind == "vf":
        return 1
    return min(7, 2 + spec.scene_id)


def schedule_stats(labels: np.ndarray, costs: np.ndarray, scores: np.ndarray) -> tuple[int, float, int, float, float, float]:
    n = int(labels.shape[0])
    positives = int(np.count_nonzero(labels))
    all_work = float(np.sum(costs, dtype=np.float64))
    if positives == 0:
        return n, all_work, 0, 0.0, 0.0, 0.0
    order = np.argsort(-scores, kind="mergesort")
    local_labels = labels[order]
    first = int(np.flatnonzero(local_labels)[0]) + 1
    work = float(np.sum(costs[order[:first]], dtype=np.float64))
    oracle_reduction = 1.0 - 1.0 / max(1, n)
    random_expected_calls = (n + 1.0) / (positives + 1.0)
    random_expected_reduction = 1.0 - random_expected_calls / max(1, n)
    return first, work, first, 1.0 - first / max(1, n), 1.0 - work / max(1.0e-12, all_work), random_expected_reduction


def convert_group(spec: GroupSpec, *, max_timestep: int, unique_offset: int) -> tuple[dict[str, np.ndarray], GroupSummary, dict[str, Any]]:
    if not spec.query_csv.exists():
        raise FileNotFoundError(spec.query_csv)
    if not spec.boxes_json.exists():
        raise FileNotFoundError(spec.boxes_json)
    if not spec.bool_json.exists():
        raise FileNotFoundError(spec.bool_json)

    labels_list = [bool(value) for value in load_json(spec.bool_json)]
    boxes = load_json(spec.boxes_json)
    if len(boxes) != len(labels_list):
        raise ValueError(f"Box/label count mismatch for {spec.stem}: {len(boxes)} boxes vs {len(labels_list)} labels")

    n = len(labels_list)
    features = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    costs = np.zeros(n, dtype=np.float64)
    labels = np.asarray(labels_list, dtype=np.bool_)
    interval_targets = np.zeros((n, INTERVAL_BINS), dtype=np.float32)
    family_targets = np.zeros((n, FAMILY_BINS), dtype=np.float32)
    scalar_targets = np.zeros((n, 3), dtype=np.float32)
    ids = np.zeros((n, 9), dtype=np.uint64)
    case_names = string_array(spec.scene, n)
    kind_names = string_array(spec.kind, n)
    csv_paths = string_array(rel(spec.query_csv), n)
    risk_scores = np.zeros(n, dtype=np.float64)

    group_id = unique_offset + 1
    for local_index, (points_t0, points_t1, rational_bits) in enumerate(iter_query_vertices(spec.query_csv)):
        if local_index >= n:
            raise ValueError(f"{spec.query_csv} contains more queries than labels")
        f, exact_cost, risk_score, closest_t = feature_row(
            points_t0=points_t0,
            points_t1=points_t1,
            kind=spec.kind,
            scene_id=spec.scene_id,
            group_size=n,
            timestep=spec.timestep,
            max_timestep=max_timestep,
            local_index=local_index,
            rational_bits=rational_bits,
            box_pair=boxes[local_index],
        )
        positive = bool(labels[local_index])
        features[local_index] = f
        costs[local_index] = exact_cost
        risk_scores[local_index] = risk_score
        interval_targets[local_index, interval_bin(closest_t, positive)] = 1.0
        family_targets[local_index, family_index(spec)] = 1.0
        scalar_targets[local_index, 0] = 1.0 if positive else min(0.55, 0.55 * risk_score)
        scalar_targets[local_index, 1] = float(exact_cost)
        scalar_targets[local_index, 2] = 0.08 if positive else 0.22
        ids[local_index, 0] = 1
        ids[local_index, 1] = group_id
        ids[local_index, 2] = unique_offset * 1_000_000 + local_index + 1
        ids[local_index, 3] = spec.timestep
        ids[local_index, 4] = spec.scene_id + 1
        ids[local_index, 5] = int(boxes[local_index][0]) if boxes[local_index] else 0
        ids[local_index, 6] = spec.scene_id + 1
        ids[local_index, 7] = int(boxes[local_index][1]) if len(boxes[local_index]) > 1 else 0
        ids[local_index, 8] = STPF_TARGET_MASK_ALL
    if local_index + 1 != n:
        raise ValueError(f"{spec.query_csv} has {local_index + 1} queries but {n} labels")

    first_rank, _, _, call_reduction, work_reduction, random_reduction = schedule_stats(labels, costs, risk_scores)
    positives = int(np.count_nonzero(labels))
    summary = GroupSummary(
        scene=spec.scene,
        kind=spec.kind,
        timestep=spec.timestep,
        query_count=n,
        positives=positives,
        negatives=n - positives,
        positive_fraction=float(positives / max(1, n)),
        frame0=rel(spec.frame0),
        frame1=rel(spec.frame1),
        query_csv=rel(spec.query_csv),
        boxes_json=rel(spec.boxes_json),
        bool_json=rel(spec.bool_json),
        first_positive_rank_heuristic=first_rank if positives else 0,
        heuristic_call_reduction=call_reduction,
        heuristic_work_reduction=work_reduction,
        oracle_call_reduction=(1.0 - 1.0 / max(1, n)) if positives else 0.0,
        random_expected_call_reduction=random_reduction,
    )
    arrays = {
        "ids": ids,
        "features": features,
        "interval_targets": interval_targets,
        "family_targets": family_targets,
        "scalar_targets": scalar_targets,
        "ground_truth": labels,
        "costs": costs,
        "case_names": case_names,
        "kind_names": kind_names,
        "csv_paths": csv_paths,
        "source_query_indices": ids[:, 2].astype(np.uint64),
    }
    metadata = {
        "schema_version": 1,
        "scene": spec.scene,
        "kind": spec.kind,
        "timestep": spec.timestep,
        "row_count": n,
        "feature_dim": FEATURE_DIM,
        "interval_bins": INTERVAL_BINS,
        "family_count": FAMILY_BINS,
        "split_names": ["scene_eval"],
        "query_count": n,
        "positive_count": positives,
        "source_query_csv": rel(spec.query_csv),
        "source_boxes_json": rel(spec.boxes_json),
        "source_mma_bool_json": rel(spec.bool_json),
        "frame0": rel(spec.frame0),
        "frame1": rel(spec.frame1),
        "grouping_scope": "scene_step_kind",
        "comparison_scope": "converted P2C candidate groups only; no Scalable-CCD native kernel time comparison",
    }
    return arrays, summary, metadata


def concatenate_arrays(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        raise ValueError("No chunks to concatenate")
    keys = chunks[0].keys()
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}


def write_npz(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, ensure_ascii=False), dtype=np.str_)
    np.savez_compressed(path, **payload)


def write_csv(path: Path, summaries: list[GroupSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(GroupSummary.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def write_report(
    path: Path,
    *,
    source_root: Path,
    shard_dir: Path,
    benchmark_dir: Path,
    combined_path: Path,
    summaries: list[GroupSummary],
    elapsed_s: float,
) -> dict[str, Any]:
    total_candidates = sum(s.query_count for s in summaries)
    total_positives = sum(s.positives for s in summaries)
    total_negatives = total_candidates - total_positives
    positive_groups = sum(1 for s in summaries if s.positives > 0)
    all_exact_calls = total_candidates
    heuristic_calls = sum(s.first_positive_rank_heuristic if s.positives > 0 else s.query_count for s in summaries)
    oracle_calls = sum(1 if s.positives > 0 else s.query_count for s in summaries)
    random_expected_calls = sum(
        ((s.query_count + 1.0) / (s.positives + 1.0)) if s.positives > 0 else float(s.query_count)
        for s in summaries
    )
    payload = {
        "run_name": RUN_NAME,
        "source_root": rel(source_root),
        "shard_dir": rel(shard_dir),
        "benchmark_dir": rel(benchmark_dir),
        "combined_npz": rel(combined_path),
        "scene_step_kind_groups": len(summaries),
        "candidate_rows": total_candidates,
        "positive_rows": total_positives,
        "negative_rows": total_negatives,
        "positive_groups": positive_groups,
        "heuristic_exact_calls": heuristic_calls,
        "heuristic_call_reduction": 1.0 - heuristic_calls / max(1, all_exact_calls),
        "oracle_exact_calls": oracle_calls,
        "oracle_call_reduction": 1.0 - oracle_calls / max(1, all_exact_calls),
        "random_expected_exact_calls": random_expected_calls,
        "random_expected_call_reduction": 1.0 - random_expected_calls / max(1, all_exact_calls),
        "elapsed_seconds": elapsed_s,
    }
    rows = [
        [
            s.scene,
            s.kind.upper(),
            str(s.timestep),
            str(s.query_count),
            str(s.positives),
            f"{100.0 * s.positive_fraction:.3f}%",
            str(s.first_positive_rank_heuristic),
            f"{100.0 * s.heuristic_call_reduction:.3f}%",
            f"{100.0 * s.random_expected_call_reduction:.3f}%",
            f"{100.0 * s.oracle_call_reduction:.3f}%",
        ]
        for s in summaries
    ]
    lines = [
        "# Scalable-CCD sample converted to P2C scene-level candidate groups",
        "",
        f"Run identifier: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        "",
        "## Scope",
        "",
        "- descriptionreport `Sample-Scalable-CCD-Data`  full-scene time-step query files convertas P2C candidate-row groups. ",
        "- Group granularity is `scene + timestep + primitive kind`, descriptionscenewhendescriptionall EE or VF primitive queries asdescription P2C candidate group. ",
        "- descriptioncompare Scalable-CCD native kernel time, description scene simulator runtime and P2C candidate-row replay runtime descriptioninsamedescription. ",
        "- `mma_bool` as ground truth label; `boxes` asoriginal broad-phase intersecting box ids; query CSV  rational coordinates generate 32 description STPF-compatible feature rows. ",
        "",
        "## Outputs",
        "",
        f"- Combined shard: `{rel(combined_path)}`",
        f"- Per-group shards: `{rel(shard_dir / 'groups')}`",
        f"- Manifest: `{rel(shard_dir / 'manifest.json')}`",
        f"- Summary CSV: `{rel(benchmark_dir / (RUN_NAME + '.csv'))}`",
        f"- Summary JSON: `{rel(benchmark_dir / (RUN_NAME + '.json'))}`",
        "",
        "## Overall",
        "",
        *markdown_table(
            ["Metric", "Value"],
            [
                ["source scenes", str(len({s.scene for s in summaries}))],
                ["scene-step-kind groups", str(len(summaries))],
                ["candidate rows", str(total_candidates)],
                ["positive rows", str(total_positives)],
                ["negative rows", str(total_negatives)],
                ["positive groups", str(positive_groups)],
                ["heuristic schedule exact calls", str(heuristic_calls)],
                ["heuristic call reduction", f"{100.0 * payload['heuristic_call_reduction']:.3f}%"],
                ["random expected call reduction", f"{100.0 * payload['random_expected_call_reduction']:.3f}%"],
                ["oracle lower-bound call reduction", f"{100.0 * payload['oracle_call_reduction']:.3f}%"],
            ],
        ),
        "",
        "## Per Scene-step-kind Group",
        "",
        *markdown_table(
            [
                "Scene",
                "Kind",
                "Step",
                "Candidates",
                "Positive",
                "Positive %",
                "Heuristic first rank",
                "Heuristic call red.",
                "Random expected red.",
                "Oracle red.",
            ],
            rows,
        ),
        "",
        "## Interpretation",
        "",
        "- this supplementary descriptionInputis converted candidate groups, rather than Scalable-CCD original pipeline time. ",
        "- this sample description scene-step-kind group  positive fraction descriptionhigh, therefore early-stop reduction descriptionanddescription; descriptionas learned scheduling better than SOTA description. ",
        "- `heuristic` descriptionis motion/proximity feature  sanity diagnostic, usedescription converted scene groups has scheduling pressure; descriptionis notthis paperdescription SOTA Methoddescription. ",
        "- description TOG writedescription, description supplementary: descriptionthis paperdescriptionconnectdescription Scalable-CCD full-scene data source, descriptionby scene time-step description P2C candidate groups. ",
        "- ifdescriptionenterdescription, underdescriptionindescription converted groups onfixed STPF checkpoint / validation-selected schedule, descriptionusesame exact certificate policy perform replay. ",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "& 'python' src\\tools\\convert_scalable_ccd_sample_to_p2c_groups.py",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Sample-Scalable-CCD-Data full-scene queries to P2C candidate groups.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--max-query-files", type=int, default=0, help="Debug limit for smoke conversion.")
    args = parser.parse_args()

    started = time.perf_counter()
    source_root = args.source_root.resolve()
    shard_dir = args.shard_dir.resolve()
    benchmark_dir = args.benchmark_dir.resolve()
    groups = discover_groups(source_root, max_query_files=args.max_query_files)
    if not groups:
        raise FileNotFoundError(f"No Scalable-CCD query CSV files found under {source_root}")
    max_timestep = max(group.timestep for group in groups)

    chunks: list[dict[str, np.ndarray]] = []
    summaries: list[GroupSummary] = []
    manifest_groups: list[dict[str, Any]] = []
    per_group_dir = shard_dir / "groups"
    per_group_dir.mkdir(parents=True, exist_ok=True)
    for group_index, spec in enumerate(groups):
        arrays, summary, metadata = convert_group(spec, max_timestep=max_timestep, unique_offset=group_index)
        group_path = per_group_dir / f"{group_index:03d}_{safe_name(spec.scene)}_{spec.stem}.npz"
        write_npz(group_path, arrays, metadata)
        chunks.append(arrays)
        summaries.append(summary)
        manifest_groups.append({**metadata, "npz": rel(group_path)})

    combined = concatenate_arrays(chunks)
    combined_path = shard_dir / "scene_eval.npz"
    combined_metadata = {
        "schema_version": 1,
        "run_name": RUN_NAME,
        "source_root": rel(source_root),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "grouping_scope": "scene_step_kind",
        "row_count": int(combined["features"].shape[0]),
        "feature_dim": FEATURE_DIM,
        "interval_bins": INTERVAL_BINS,
        "family_count": FAMILY_BINS,
        "split_names": ["scene_eval"],
        "scene_count": len({s.scene for s in summaries}),
        "scene_step_kind_groups": len(summaries),
        "candidate_rows": int(combined["features"].shape[0]),
        "comparison_scope": "converted P2C candidate groups only; no direct Scalable-CCD kernel time comparison",
    }
    write_npz(combined_path, combined, combined_metadata)

    manifest = {
        **combined_metadata,
        "combined_npz": rel(combined_path),
        "groups": manifest_groups,
    }
    (shard_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(benchmark_dir / f"{RUN_NAME}.csv", summaries)
    elapsed_s = time.perf_counter() - started
    report_payload = write_report(
        benchmark_dir / f"{RUN_NAME}.md",
        source_root=source_root,
        shard_dir=shard_dir,
        benchmark_dir=benchmark_dir,
        combined_path=combined_path,
        summaries=summaries,
        elapsed_s=elapsed_s,
    )
    (benchmark_dir / f"{RUN_NAME}.json").write_text(
        json.dumps({**report_payload, "groups": [summary.__dict__ for summary in summaries]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
