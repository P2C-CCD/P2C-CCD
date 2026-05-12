from __future__ import annotations

from dataclasses import dataclass
import math

from p2cccd.contracts import ProxyType
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

from .oracle import ExactOracleTrace, evaluate_swept_sphere_oracle
from .samplers import (
    DEFAULT_SPLITS,
    SPLIT_OOD,
    SPLIT_ROBOT_LINK,
    MotionDiscPairSample,
    PairFamily,
    generate_mesh_pair_motion_samples,
    generate_robot_link_pair_motion_samples,
)


DATASET_ROW_SCHEMA_VERSION = 1
DATASET_SHARD_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DatasetGenerationConfig:
    mesh_count_per_split: int = 8
    robot_link_count: int = 8
    seed: int = 13
    include_robot_links: bool = True


@dataclass(slots=True)
class GeneratedDataset:
    rows: list[ProposalFeatureRow]
    samples: list[MotionDiscPairSample]
    traces: list[ExactOracleTrace]
    split_names: tuple[str, ...]


def _sub(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def _norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(max(0.0, value[0] * value[0] + value[1] * value[1] + value[2] * value[2]))


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(hi, max(lo, value))


def _interval_bin(t: float) -> int:
    clamped = min(1.0 - 1.0e-12, max(0.0, t))
    return int(clamped * PROPOSAL_INTERVAL_BIN_COUNT)


def _aabb_extent(radius: float) -> float:
    return 2.0 * radius


def _sphere_volume(radius: float) -> float:
    return 4.0 / 3.0 * math.pi * radius * radius * radius


def _sphere_surface(radius: float) -> float:
    return 4.0 * math.pi * radius * radius


def _proxy_type_value(proxy_type: ProxyType) -> float:
    return float(int(proxy_type))


def proposal_row_from_oracle_trace(
    sample: MotionDiscPairSample,
    trace: ExactOracleTrace,
) -> ProposalFeatureRow:
    radius_sum = sample.radius_a + sample.radius_b
    center_distance_t0 = _norm(_sub(sample.center_a_t0, sample.center_b_t0))
    center_distance_t1 = _norm(_sub(sample.center_a_t1, sample.center_b_t1))
    velocity_a = _norm(_sub(sample.center_a_t1, sample.center_a_t0))
    velocity_b = _norm(_sub(sample.center_b_t1, sample.center_b_t0))
    min_center_distance = trace.min_distance
    overlap_depth = max(0.0, radius_sum - trace.min_distance)
    normalized_overlap = _clamp(overlap_depth / max(radius_sum, 1.0e-6))
    near_contact = _clamp(1.0 - max(0.0, trace.safe_margin) / max(radius_sum, 1.0e-6))

    features = [0.0] * PROPOSAL_FEATURE_DIM
    features[0] = 0.0
    features[1] = 1.0
    features[2] = 1.0
    features[3] = 2.0 if trace.collided else (1.0 if near_contact > 0.5 else 0.0)
    features[4] = _proxy_type_value(sample.proxy_type_a)
    features[5] = _proxy_type_value(sample.proxy_type_b)
    features[6] = float(velocity_a)
    features[7] = float(velocity_b)
    features[8] = float(radius_sum)
    features[9] = float(sample.hardness)
    features[10] = _aabb_extent(sample.radius_a)
    features[11] = _aabb_extent(sample.radius_a)
    features[12] = _aabb_extent(sample.radius_a)
    features[13] = _aabb_extent(sample.radius_b)
    features[14] = _aabb_extent(sample.radius_b)
    features[15] = _aabb_extent(sample.radius_b)
    features[16] = _sphere_volume(sample.radius_a)
    features[17] = _sphere_volume(sample.radius_b)
    features[18] = float(overlap_depth)
    features[19] = float(normalized_overlap)
    features[20] = float(min_center_distance)
    features[21] = _sphere_surface(sample.radius_a)
    features[22] = _sphere_surface(sample.radius_b)
    features[23] = float(sample.radius_a)
    features[24] = float(sample.radius_b)
    features[25] = float(velocity_a)
    features[26] = float(velocity_b)
    features[27] = 0.0
    features[28] = 0.0
    features[29] = float(sample.hardness)
    features[30] = float(near_contact)
    features[31] = float(features[3])

    row = ProposalFeatureRow(
        schema_version=DATASET_ROW_SCHEMA_VERSION,
        query_id=sample.query_id,
        candidate_id=sample.candidate_id,
        slab_id=sample.slab_id,
        object_a_id=sample.object_a_id,
        patch_a_id=sample.patch_a_id,
        object_b_id=sample.object_b_id,
        patch_b_id=sample.patch_b_id,
        features=features,
        priority_target=float(_clamp(0.25 * near_contact + 0.55 * float(trace.collided) + 0.20 * sample.hardness)),
        cost_target=float(trace.exact_cost),
        uncertainty_target=float(1.0 if sample.ood else _clamp(0.15 + 0.55 * sample.hardness + 0.25 * near_contact)),
        target_mask=TARGET_INTERVAL | TARGET_FAMILY | TARGET_PRIORITY | TARGET_COST | TARGET_UNCERTAINTY,
    )
    interval_time = trace.toi if trace.collided else trace.closest_time
    row.interval_targets[_interval_bin(interval_time)] = 1.0
    if sample.family == PairFamily.ROBOT_LINK_PAIR:
        row.family_targets[0] = 0.0
        row.family_targets[1] = 1.0
    else:
        row.family_targets[0] = 1.0
        row.family_targets[1] = 1.0
    for index in range(2, PROPOSAL_FAMILY_COUNT):
        row.family_targets[index] = 0.0
    return validate_proposal_feature_row(row)


def generate_exact_oracle_dataset(config: DatasetGenerationConfig) -> GeneratedDataset:
    if config.mesh_count_per_split < 0:
        raise ValueError("mesh_count_per_split must be non-negative")
    if config.robot_link_count < 0:
        raise ValueError("robot_link_count must be non-negative")

    samples = generate_mesh_pair_motion_samples(
        count_per_split=config.mesh_count_per_split,
        seed=config.seed,
        splits=DEFAULT_SPLITS,
        first_sample_id=1,
    )
    split_names = list(DEFAULT_SPLITS)
    if config.include_robot_links:
        robot_samples = generate_robot_link_pair_motion_samples(
            count=config.robot_link_count,
            seed=config.seed + 17,
            first_sample_id=500000,
        )
        samples.extend(robot_samples)
        split_names.append(SPLIT_ROBOT_LINK)

    traces = [evaluate_swept_sphere_oracle(sample) for sample in samples]
    rows = [proposal_row_from_oracle_trace(sample, trace) for sample, trace in zip(samples, traces)]
    return GeneratedDataset(rows=rows, samples=samples, traces=traces, split_names=tuple(split_names))


def split_ids_for_samples(samples: list[MotionDiscPairSample], split_names: tuple[str, ...]) -> list[int]:
    split_to_id = {name: index for index, name in enumerate(split_names)}
    return [split_to_id[sample.split] for sample in samples]


def is_ood_split(split: str) -> bool:
    return split == SPLIT_OOD
