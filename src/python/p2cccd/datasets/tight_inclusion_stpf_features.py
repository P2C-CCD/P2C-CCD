from __future__ import annotations

from pathlib import Path
import hashlib
import math

import numpy as np

from p2cccd.proposal.features import (
    PROPOSAL_FAMILY_COUNT,
    PROPOSAL_FEATURE_DIM,
    PROPOSAL_INTERVAL_BIN_COUNT,
    TARGET_COST,
    TARGET_FAMILY,
    TARGET_INTERVAL,
    TARGET_PRIORITY,
    TARGET_UNCERTAINTY,
    ProposalFeatureRow,
    validate_proposal_feature_row,
)

from .tight_inclusion_queries import TightInclusionPrimitiveQuery


def _safe_norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _bbox(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=np.float64)
    return values.min(axis=0), values.max(axis=0)


def _bbox_gap(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_min, lhs_max = _bbox(lhs)
    rhs_min, rhs_max = _bbox(rhs)
    sep = np.maximum(np.maximum(lhs_min - rhs_max, rhs_min - lhs_max), 0.0)
    return _safe_norm(sep)


def _bbox_overlap_ratio(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_min, lhs_max = _bbox(lhs)
    rhs_min, rhs_max = _bbox(rhs)
    inter = np.maximum(np.minimum(lhs_max, rhs_max) - np.maximum(lhs_min, rhs_min), 0.0)
    lhs_extent = np.maximum(lhs_max - lhs_min, 0.0)
    rhs_extent = np.maximum(rhs_max - rhs_min, 0.0)
    denom = float(np.prod(lhs_extent + rhs_extent + 1.0e-12))
    return _clamp01(float(np.prod(inter)) / max(denom, 1.0e-12))


def _min_vertex_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    diffs = lhs[:, None, :] - rhs[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    return float(np.min(distances))


def _edge_length(points: np.ndarray, a: int, b: int) -> float:
    return _safe_norm(points[b] - points[a])


def _triangle_area(points: np.ndarray, a: int, b: int, c: int) -> float:
    return 0.5 * _safe_norm(np.cross(points[b] - points[a], points[c] - points[a]))


def _query_id(query: TightInclusionPrimitiveQuery) -> int:
    payload = f"{query.csv_path.as_posix()}:{query.query_index}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") or 1


def _case_hash_unit(case_name: str) -> float:
    value = int.from_bytes(hashlib.blake2b(case_name.encode("utf-8"), digest_size=4).digest(), "little")
    return float(value % 1000) / 999.0


def _case_is_handcrafted(case_name: str) -> float:
    return 1.0 if case_name.startswith("erleben-") or case_name == "unit-tests" else 0.0


def _primitive_groups(query: TightInclusionPrimitiveQuery, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if query.kind == "vertex-face":
        return vertices[:1], vertices[1:4]
    return vertices[:2], vertices[2:4]


def _geometry_features(query: TightInclusionPrimitiveQuery, vertices: np.ndarray) -> tuple[float, float, float, float]:
    if query.kind == "vertex-face":
        edge0 = _edge_length(vertices, 1, 2)
        edge1 = _edge_length(vertices, 2, 3)
        area_or_cross = _triangle_area(vertices, 1, 2, 3)
        degeneracy = 1.0 / (1.0 + 1000.0 * area_or_cross)
        return edge0, edge1, area_or_cross, _clamp01(degeneracy)
    edge0 = _edge_length(vertices, 0, 1)
    edge1 = _edge_length(vertices, 2, 3)
    cross_norm = _safe_norm(np.cross(vertices[1] - vertices[0], vertices[3] - vertices[2]))
    degeneracy = 1.0 / (1.0 + 1000.0 * min(edge0, edge1, cross_norm))
    return edge0, edge1, cross_norm, _clamp01(degeneracy)


def _interval_index(query: TightInclusionPrimitiveQuery, proximity_t0: float, proximity_t1: float) -> int:
    if query.ground_truth:
        return PROPOSAL_INTERVAL_BIN_COUNT // 2
    return 0 if proximity_t0 <= proximity_t1 else PROPOSAL_INTERVAL_BIN_COUNT - 1


def tight_inclusion_query_to_proposal_row(
    query: TightInclusionPrimitiveQuery,
    *,
    query_id: int | None = None,
    candidate_id: int | None = None,
) -> ProposalFeatureRow:
    t0 = np.asarray(query.vertices_t0, dtype=np.float64)
    t1 = np.asarray(query.vertices_t1, dtype=np.float64)
    swept = np.asarray(query.vertices_t0_t1, dtype=np.float64)
    swept_min, swept_max = _bbox(swept)
    swept_extent = np.maximum(swept_max - swept_min, 0.0)
    swept_diag = _safe_norm(swept_extent)
    swept_volume_log = math.log1p(float(np.prod(swept_extent + 1.0e-12)))
    motion = t1 - t0
    speeds = np.linalg.norm(motion, axis=1)
    a0, b0 = _primitive_groups(query, t0)
    a1, b1 = _primitive_groups(query, t1)
    gap_t0 = _bbox_gap(a0, b0)
    gap_t1 = _bbox_gap(a1, b1)
    prox_t0 = _min_vertex_distance(a0, b0)
    prox_t1 = _min_vertex_distance(a1, b1)
    overlap_t0 = _bbox_overlap_ratio(a0, b0)
    overlap_t1 = _bbox_overlap_ratio(a1, b1)
    geom0_t0, geom1_t0, geom_area_t0, deg_t0 = _geometry_features(query, t0)
    geom0_t1, geom1_t1, geom_area_t1, deg_t1 = _geometry_features(query, t1)
    rational = query.rational_magnitude_features
    coord_scale = math.log1p(float(np.max(np.abs(swept))) if swept.size else 0.0)
    mean_a_motion = np.mean(a1, axis=0) - np.mean(a0, axis=0)
    mean_b_motion = np.mean(b1, axis=0) - np.mean(b0, axis=0)
    relative_motion = _safe_norm(mean_a_motion - mean_b_motion)
    near_score = _clamp01(1.0 / (1.0 + min(gap_t0, gap_t1, prox_t0, prox_t1)))
    degeneracy = max(deg_t0, deg_t1)
    denominator_scale = _clamp01(float(rational[1]) / 64.0)

    features = [0.0] * PROPOSAL_FEATURE_DIM
    features[0] = 1.0 if query.kind == "vertex-face" else 0.0
    features[1] = 1.0 if query.kind == "edge-edge" else 0.0
    features[2] = coord_scale
    features[3] = float(rational[1])
    features[4] = float(rational[0])
    features[5] = float(swept_extent[0])
    features[6] = float(swept_extent[1])
    features[7] = float(swept_extent[2])
    features[8] = swept_diag
    features[9] = swept_volume_log
    features[10] = _safe_norm(np.mean(motion, axis=0))
    features[11] = float(np.max(speeds)) if speeds.size else 0.0
    features[12] = float(np.mean(speeds)) if speeds.size else 0.0
    features[13] = relative_motion
    features[14] = gap_t0
    features[15] = gap_t1
    features[16] = min(gap_t0, gap_t1)
    features[17] = geom0_t0
    features[18] = geom1_t0
    features[19] = geom_area_t0
    features[20] = geom0_t1
    features[21] = geom1_t1
    features[22] = geom_area_t1
    features[23] = degeneracy
    features[24] = prox_t0
    features[25] = prox_t1
    features[26] = float(np.min(speeds)) if speeds.size else 0.0
    features[27] = overlap_t0
    features[28] = overlap_t1
    features[29] = _case_hash_unit(query.case_name)
    features[30] = _case_is_handcrafted(query.case_name)
    features[31] = 1.0

    interval_targets = [0.0] * PROPOSAL_INTERVAL_BIN_COUNT
    interval_targets[_interval_index(query, prox_t0, prox_t1)] = 1.0
    family_targets = [0.0] * PROPOSAL_FAMILY_COUNT
    family_targets[0 if query.kind == "vertex-face" else 1] = 1.0
    priority = 1.0 if query.ground_truth else max(0.05, 0.75 * near_score + 0.25 * max(overlap_t0, overlap_t1))
    uncertainty = _clamp01(0.20 * denominator_scale + 0.35 * degeneracy + 0.35 * near_score + (0.10 if query.ground_truth else 0.0))
    cost = 1.0 + 2.0 * near_score + 1.5 * uncertainty + (0.5 if query.kind == "edge-edge" else 0.0)

    return validate_proposal_feature_row(
        ProposalFeatureRow(
            query_id=_query_id(query) if query_id is None else int(query_id),
            candidate_id=(query.query_index + 1) if candidate_id is None else int(candidate_id),
            slab_id=0,
            object_a_id=0,
            patch_a_id=0,
            object_b_id=1,
            patch_b_id=0,
            features=[float(value) for value in features],
            interval_targets=interval_targets,
            family_targets=family_targets,
            priority_target=float(priority),
            cost_target=float(cost),
            uncertainty_target=float(uncertainty),
            target_mask=TARGET_INTERVAL | TARGET_FAMILY | TARGET_PRIORITY | TARGET_COST | TARGET_UNCERTAINTY,
        )
    )


__all__ = ["tight_inclusion_query_to_proposal_row"]
