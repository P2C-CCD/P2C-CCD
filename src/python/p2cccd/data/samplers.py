from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
import math
import random
from typing import Iterable

from p2cccd.contracts import ProxyType


DATASET_SCHEMA_VERSION = 1


class PairFamily(IntEnum):
    MESH_PAIR = 1
    ROBOT_LINK_PAIR = 2


SPLIT_EASY_NEGATIVE = "easy_negatives"
SPLIT_NEAR_CONTACT = "near_contact_hard_negatives"
SPLIT_GRAZING = "grazing_contacts"
SPLIT_MULTI_CONTACT = "multiple_contact_intervals"
SPLIT_OOD = "ood_meshes"
SPLIT_ROBOT_LINK = "robot_link_validation"

DEFAULT_SPLITS = (
    SPLIT_EASY_NEGATIVE,
    SPLIT_NEAR_CONTACT,
    SPLIT_GRAZING,
    SPLIT_MULTI_CONTACT,
    SPLIT_OOD,
)


@dataclass(frozen=True, slots=True)
class MotionDiscPairSample:
    sample_id: int
    query_id: int
    candidate_id: int
    split: str
    family: PairFamily
    object_a_id: int
    patch_a_id: int
    object_b_id: int
    patch_b_id: int
    slab_id: int
    center_a_t0: tuple[float, float, float]
    center_a_t1: tuple[float, float, float]
    center_b_t0: tuple[float, float, float]
    center_b_t1: tuple[float, float, float]
    radius_a: float
    radius_b: float
    proxy_type_a: ProxyType
    proxy_type_b: ProxyType
    hardness: float
    ood: bool = False
    mass_a: float | None = None
    mass_b: float | None = None
    restitution: float = 1.0


def _jitter(rng: random.Random, scale: float) -> float:
    return rng.uniform(-scale, scale)


def _vec(x: float, y: float, z: float = 0.0) -> tuple[float, float, float]:
    return (float(x), float(y), float(z))


def _proxy_mass_from_radius(radius: float) -> float:
    clamped_radius = max(1.0e-6, float(radius))
    return (4.0 / 3.0) * math.pi * clamped_radius * clamped_radius * clamped_radius


def _sample_for_split(
    *,
    sample_id: int,
    split: str,
    rng: random.Random,
    family: PairFamily,
) -> MotionDiscPairSample:
    radius_a = rng.uniform(0.18, 0.36)
    radius_b = rng.uniform(0.16, 0.34)
    combined_radius = radius_a + radius_b
    base_y = _jitter(rng, 0.12)

    if split == SPLIT_EASY_NEGATIVE:
        start_gap = combined_radius + rng.uniform(1.6, 3.2)
        end_gap = combined_radius + rng.uniform(1.0, 2.4)
        hardness = 0.05
    elif split == SPLIT_NEAR_CONTACT:
        start_gap = combined_radius + rng.uniform(0.10, 0.28)
        end_gap = combined_radius + rng.uniform(0.03, 0.12)
        hardness = 0.55
    elif split == SPLIT_GRAZING:
        start_gap = combined_radius + rng.uniform(0.08, 0.18)
        end_gap = combined_radius - rng.uniform(0.005, 0.035)
        hardness = 0.82
    elif split == SPLIT_MULTI_CONTACT:
        start_gap = combined_radius + rng.uniform(0.35, 0.65)
        end_gap = -(combined_radius + rng.uniform(0.18, 0.42))
        hardness = 0.92
    elif split == SPLIT_OOD:
        radius_a = rng.uniform(0.65, 1.2)
        radius_b = rng.uniform(0.55, 1.1)
        combined_radius = radius_a + radius_b
        start_gap = combined_radius + rng.uniform(4.0, 7.0)
        end_gap = combined_radius - rng.uniform(0.2, 0.7)
        hardness = 1.0
    else:
        raise ValueError(f"unknown mesh-pair split: {split}")

    center_a_t0 = _vec(0.0 + _jitter(rng, 0.05), base_y)
    center_a_t1 = _vec(0.55 + _jitter(rng, 0.09), base_y + _jitter(rng, 0.08))
    center_b_t0 = _vec(start_gap, base_y + _jitter(rng, 0.08))
    center_b_t1 = _vec(end_gap, base_y + _jitter(rng, 0.08))
    if split == SPLIT_MULTI_CONTACT:
        center_a_t1 = _vec(0.15 + _jitter(rng, 0.05), base_y)
        center_b_t1 = _vec(end_gap, base_y + _jitter(rng, 0.04))

    patch_offset = 1000 if family == PairFamily.ROBOT_LINK_PAIR else 0
    return MotionDiscPairSample(
        sample_id=sample_id,
        query_id=100000 + sample_id,
        candidate_id=200000 + sample_id,
        split=split,
        family=family,
        object_a_id=10 if family == PairFamily.MESH_PAIR else 110,
        patch_a_id=patch_offset + 1 + (sample_id % 7),
        object_b_id=20 if family == PairFamily.MESH_PAIR else 120,
        patch_b_id=patch_offset + 101 + (sample_id % 11),
        slab_id=sample_id % 8,
        center_a_t0=center_a_t0,
        center_a_t1=center_a_t1,
        center_b_t0=center_b_t0,
        center_b_t1=center_b_t1,
        radius_a=radius_a,
        radius_b=radius_b,
        proxy_type_a=ProxyType.SWEPT_AABB,
        proxy_type_b=ProxyType.CAPSULE if family == PairFamily.ROBOT_LINK_PAIR else ProxyType.SWEPT_AABB,
        hardness=hardness,
        ood=split == SPLIT_OOD,
        mass_a=_proxy_mass_from_radius(radius_a),
        mass_b=_proxy_mass_from_radius(radius_b),
        restitution=1.0,
    )


def generate_mesh_pair_motion_samples(
    *,
    count_per_split: int,
    seed: int = 1,
    splits: Iterable[str] = DEFAULT_SPLITS,
    first_sample_id: int = 1,
) -> list[MotionDiscPairSample]:
    if count_per_split < 0:
        raise ValueError("count_per_split must be non-negative")
    rng = random.Random(seed)
    samples: list[MotionDiscPairSample] = []
    sample_id = first_sample_id
    for split in splits:
        for _ in range(count_per_split):
            samples.append(
                _sample_for_split(
                    sample_id=sample_id,
                    split=split,
                    rng=rng,
                    family=PairFamily.MESH_PAIR,
                )
            )
            sample_id += 1
    return samples


def generate_robot_link_pair_motion_samples(
    *,
    count: int,
    seed: int = 1009,
    first_sample_id: int = 500000,
) -> list[MotionDiscPairSample]:
    if count < 0:
        raise ValueError("count must be non-negative")
    rng = random.Random(seed)
    samples: list[MotionDiscPairSample] = []
    for index in range(count):
        sample = _sample_for_split(
            sample_id=first_sample_id + index,
            split=SPLIT_NEAR_CONTACT if index % 2 == 0 else SPLIT_GRAZING,
            rng=rng,
            family=PairFamily.ROBOT_LINK_PAIR,
        )
        samples.append(
            replace(
                sample,
                split=SPLIT_ROBOT_LINK,
                proxy_type_a=ProxyType.CAPSULE,
                proxy_type_b=ProxyType.CAPSULE,
                hardness=min(1.0, sample.hardness + 0.08),
            )
        )
    return samples


def sample_path_length(sample: MotionDiscPairSample) -> float:
    def dist(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> float:
        return math.sqrt(sum((a - b) * (a - b) for a, b in zip(lhs, rhs)))

    return dist(sample.center_a_t0, sample.center_a_t1) + dist(sample.center_b_t0, sample.center_b_t1)
