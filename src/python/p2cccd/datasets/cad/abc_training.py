from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from p2cccd.contracts import ProxyType
from p2cccd.data import GeneratedDataset, proposal_row_from_oracle_trace
from p2cccd.data.oracle import evaluate_swept_sphere_oracle
from p2cccd.data.response import proxy_mass_from_radius
from p2cccd.data.samplers import (
    MotionDiscPairSample,
    PairFamily,
    SPLIT_EASY_NEGATIVE,
    SPLIT_GRAZING,
    SPLIT_MULTI_CONTACT,
    SPLIT_NEAR_CONTACT,
)

from .abc_adapter import ABCDatasetAdapter, default_abc_root
from .contracts import CadMeshAsset, CadMeshPair


ABC_TRAINING_SCHEMA_VERSION = 1
ABC_DEMO_SUBSET_DIRNAME = "demo_subset_generated"


def _stable_u32(token: str) -> int:
    return int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16)


def _vec(x: float, y: float, z: float = 0.0) -> tuple[float, float, float]:
    return (float(x), float(y), float(z))


def _box_obj(size_x: float, size_y: float, size_z: float) -> str:
    hx = 0.5 * size_x
    hy = 0.5 * size_y
    hz = 0.5 * size_z
    vertices = (
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    )
    faces = (
        (1, 2, 3),
        (1, 3, 4),
        (5, 6, 7),
        (5, 7, 8),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 4, 8),
        (3, 8, 7),
        (4, 1, 5),
        (4, 5, 8),
    )
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    return "\n".join(lines) + "\n"


def _pyramid_obj(size_x: float, size_y: float, size_z: float) -> str:
    hx = 0.5 * size_x
    hy = 0.5 * size_y
    base_z = -0.5 * size_z
    apex_z = 0.5 * size_z
    vertices = (
        (-hx, -hy, base_z),
        (hx, -hy, base_z),
        (hx, hy, base_z),
        (-hx, hy, base_z),
        (0.0, 0.0, apex_z),
    )
    faces = (
        (1, 2, 3),
        (1, 3, 4),
        (1, 2, 5),
        (2, 3, 5),
        (3, 4, 5),
        (4, 1, 5),
    )
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    return "\n".join(lines) + "\n"


def _wedge_obj(size_x: float, size_y: float, size_z: float) -> str:
    hx = 0.5 * size_x
    hy = 0.5 * size_y
    hz = 0.5 * size_z
    vertices = (
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
    )
    faces = (
        (1, 2, 3),
        (1, 3, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 6),
        (3, 4, 5),
        (3, 5, 6),
        (4, 1, 5),
    )
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    return "\n".join(lines) + "\n"


def _write_demo_asset(
    root: Path,
    *,
    index: int,
    family: str,
    dims: tuple[float, float, float],
    patch_count: int,
) -> Path:
    asset_dir = root / ABC_DEMO_SUBSET_DIRNAME / family
    asset_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = asset_dir / f"{family}_{index:03d}.obj"
    if family == "box":
        text = _box_obj(*dims)
    elif family == "pyramid":
        text = _pyramid_obj(*dims)
    else:
        text = _wedge_obj(*dims)
    mesh_path.write_text(text, encoding="utf-8")
    (mesh_path.with_suffix(".json")).write_text(
        json.dumps(
            {
                "demo_subset": True,
                "asset_family": family,
                "asset_index": index,
                "dims": list(dims),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (mesh_path.with_suffix(".patch.json")).write_text(
        json.dumps(
            {
                "patches": [{"patch_id": patch_id + 1} for patch_id in range(patch_count)],
                "patch_count": patch_count,
                "demo_subset": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return mesh_path


@dataclass(frozen=True, slots=True)
class ABCProxyDatasetConfig:
    root: Path | None = None
    allow_demo_bootstrap: bool = True
    demo_asset_count: int = 18
    asset_limit: int = 18
    pair_limit: int = 96
    train_fraction: float = 0.75
    seed: int = 44


@dataclass(frozen=True, slots=True)
class ABCProxyDatasetBundle:
    source_root: Path
    used_demo_subset: bool
    assets: tuple[CadMeshAsset, ...]
    train_pairs: tuple[CadMeshPair, ...]
    eval_pairs: tuple[CadMeshPair, ...]
    train_dataset: GeneratedDataset
    eval_dataset: GeneratedDataset


def bootstrap_abc_demo_subset(
    root: Path | None = None,
    *,
    asset_count: int = 18,
) -> Path:
    if asset_count < 6:
        raise ValueError("asset_count must be at least 6")
    source_root = root if root is not None else default_abc_root()
    source_root.mkdir(parents=True, exist_ok=True)
    license_path = source_root / "LICENSE"
    if not license_path.exists():
        license_path.write_text(
            (
                "Local adapter-compatible CAD demo subset for ABC pipeline development.\n"
                "This directory is not an official redistribution of the ABC Dataset.\n"
                "Replace it with a user-provided official ABC root before publishable experiments.\n"
            ),
            encoding="utf-8",
        )
    readme_path = source_root / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            (
                "# Local ABC-Compatible Demo Subset\n\n"
                "This root contains locally generated OBJ meshes and patch sidecars so the\n"
                "ABC ingestion, shard generation, and STPF training pipeline can run without\n"
                "downloading the multi-GB official ABC object chunks.\n"
            ),
            encoding="utf-8",
        )

    families = ("box", "pyramid", "wedge")
    for index in range(asset_count):
        family = families[index % len(families)]
        scale = 0.45 + 0.08 * index
        aspect = 0.75 + 0.11 * (index % 5)
        dims = (
            scale,
            scale * aspect,
            scale * (0.6 + 0.07 * ((index + 2) % 4)),
        )
        patch_count = 2 + (index % 5)
        _write_demo_asset(source_root, index=index, family=family, dims=dims, patch_count=patch_count)
    return source_root


def _supported_mesh_count(root: Path) -> int:
    adapter = ABCDatasetAdapter(root)
    return len(adapter.list_mesh_paths(limit=1_000_000))


def _has_demo_subset(root: Path) -> bool:
    return (root / ABC_DEMO_SUBSET_DIRNAME).exists()


def _ensure_abc_root(config: ABCProxyDatasetConfig) -> tuple[Path, bool]:
    source_root = config.root if config.root is not None else default_abc_root()
    if source_root.exists() and _supported_mesh_count(source_root) > 0:
        return source_root, _has_demo_subset(source_root)
    if not config.allow_demo_bootstrap:
        raise FileNotFoundError(
            f"ABC source root {source_root} has no supported mesh assets and demo bootstrap is disabled"
        )
    bootstrap_abc_demo_subset(source_root, asset_count=config.demo_asset_count)
    return source_root, True


def _radius_from_asset(asset: CadMeshAsset) -> float:
    return max(0.05, 0.5 * max(asset.stats.diagonal, 0.1))


def _patch_id(asset: CadMeshAsset, local_index: int) -> int:
    patch_count = max(1, int(asset.patch_metadata.get("patch_count", 0)) or len(asset.patch_metadata.get("patches", [])) or 1)
    return 1 + (local_index % patch_count)


def _object_id(asset: CadMeshAsset) -> int:
    return 100000 + (_stable_u32(asset.asset_id) % 900000)


def _pair_motion_scale(pair: CadMeshPair) -> float:
    return max(0.1, 0.5 * (pair.asset_a.stats.diagonal + pair.asset_b.stats.diagonal))


def _sample_for_variant(
    pair: CadMeshPair,
    *,
    sample_id: int,
    variant_index: int,
    split: str,
    hardness: float,
) -> MotionDiscPairSample:
    radius_a = _radius_from_asset(pair.asset_a)
    radius_b = _radius_from_asset(pair.asset_b)
    radius_sum = radius_a + radius_b
    scale = _pair_motion_scale(pair)
    pair_token = _stable_u32(pair.pair_id)
    lateral = ((pair_token % 200) / 199.0 - 0.5) * 0.18 * scale
    z_bias = (((pair_token // 200) % 200) / 199.0 - 0.5) * 0.12 * scale

    if split == SPLIT_EASY_NEGATIVE:
        start_gap = radius_sum + 1.9 * scale
        end_gap = radius_sum + 0.95 * scale
        ax1 = 0.30 * scale
    elif split == SPLIT_NEAR_CONTACT:
        start_gap = radius_sum + 0.28 * scale
        end_gap = radius_sum + 0.06 * scale
        ax1 = 0.24 * scale
    elif split == SPLIT_GRAZING:
        start_gap = radius_sum + 0.18 * scale
        end_gap = radius_sum - 0.015 * scale
        ax1 = 0.18 * scale
    else:
        start_gap = radius_sum + 0.42 * scale
        end_gap = -(radius_sum + 0.22 * scale)
        ax1 = 0.10 * scale

    center_a_t0 = _vec(-0.10 * scale, lateral, z_bias)
    center_a_t1 = _vec(ax1, lateral + 0.03 * scale * ((variant_index % 2) * 2 - 1), z_bias)
    center_b_t0 = _vec(start_gap, lateral - 0.02 * scale, z_bias)
    center_b_t1 = _vec(end_gap, lateral + 0.02 * scale, z_bias)

    return MotionDiscPairSample(
        sample_id=sample_id,
        query_id=700000 + sample_id,
        candidate_id=900000 + sample_id,
        split=split,
        family=PairFamily.MESH_PAIR,
        object_a_id=_object_id(pair.asset_a),
        patch_a_id=_patch_id(pair.asset_a, variant_index),
        object_b_id=_object_id(pair.asset_b),
        patch_b_id=_patch_id(pair.asset_b, variant_index + 1),
        slab_id=variant_index,
        center_a_t0=center_a_t0,
        center_a_t1=center_a_t1,
        center_b_t0=center_b_t0,
        center_b_t1=center_b_t1,
        radius_a=radius_a,
        radius_b=radius_b,
        proxy_type_a=ProxyType.SWEPT_AABB,
        proxy_type_b=ProxyType.SWEPT_AABB,
        hardness=max(0.0, min(1.0, hardness)),
        ood=False,
        mass_a=proxy_mass_from_radius(radius_a),
        mass_b=proxy_mass_from_radius(radius_b),
        restitution=1.0,
    )


def _samples_from_pairs(
    pairs: Sequence[CadMeshPair],
    *,
    first_sample_id: int,
) -> list[MotionDiscPairSample]:
    samples: list[MotionDiscPairSample] = []
    sample_id = first_sample_id
    for pair in pairs:
        pair_hardness = float(pair.hardness_score)
        variants = (
            (SPLIT_EASY_NEGATIVE, 0.10 + 0.35 * pair_hardness),
            (SPLIT_NEAR_CONTACT, 0.35 + 0.45 * pair_hardness),
            (SPLIT_GRAZING, 0.65 + 0.30 * pair_hardness),
            (SPLIT_MULTI_CONTACT, 0.80 + 0.20 * pair_hardness),
        )
        for variant_index, (split, hardness) in enumerate(variants):
            samples.append(
                _sample_for_variant(
                    pair,
                    sample_id=sample_id,
                    variant_index=variant_index,
                    split=split,
                    hardness=hardness,
                )
            )
            sample_id += 1
    return samples


def _dataset_from_samples(samples: Sequence[MotionDiscPairSample]) -> GeneratedDataset:
    sample_list = list(samples)
    traces = [evaluate_swept_sphere_oracle(sample) for sample in sample_list]
    rows = [proposal_row_from_oracle_trace(sample, trace) for sample, trace in zip(sample_list, traces)]
    split_names = (
        SPLIT_EASY_NEGATIVE,
        SPLIT_NEAR_CONTACT,
        SPLIT_GRAZING,
        SPLIT_MULTI_CONTACT,
    )
    return GeneratedDataset(
        rows=rows,
        samples=sample_list,
        traces=traces,
        split_names=split_names,
    )


def generate_abc_proxy_datasets(
    config: ABCProxyDatasetConfig | None = None,
) -> ABCProxyDatasetBundle:
    cfg = config or ABCProxyDatasetConfig()
    if cfg.asset_limit < 2:
        raise ValueError("asset_limit must be at least 2")
    if cfg.pair_limit < 2:
        raise ValueError("pair_limit must be at least 2")
    if not 0.0 < cfg.train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")

    source_root, used_demo_subset = _ensure_abc_root(cfg)
    adapter = ABCDatasetAdapter(source_root)
    assets = adapter.list_assets(limit=cfg.asset_limit)
    if len(assets) < 2:
        raise ValueError("ABC root must contain at least two mesh assets")
    pairs = list(adapter.industrial_hard_negative_pairs(limit=cfg.pair_limit))
    if len(pairs) < 2:
        raise ValueError("ABC root must produce at least two CAD mesh pairs")

    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    train_pair_count = max(1, min(len(pairs) - 1, int(round(len(pairs) * cfg.train_fraction))))
    train_pairs = tuple(sorted(pairs[:train_pair_count], key=lambda pair: pair.pair_id))
    eval_pairs = tuple(sorted(pairs[train_pair_count:], key=lambda pair: pair.pair_id))

    train_dataset = _dataset_from_samples(_samples_from_pairs(train_pairs, first_sample_id=1))
    eval_dataset = _dataset_from_samples(_samples_from_pairs(eval_pairs, first_sample_id=1_000_001))
    return ABCProxyDatasetBundle(
        source_root=source_root,
        used_demo_subset=used_demo_subset,
        assets=tuple(sorted(assets, key=lambda asset: asset.asset_id)),
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


__all__ = [
    "ABC_DEMO_SUBSET_DIRNAME",
    "ABCProxyDatasetBundle",
    "ABCProxyDatasetConfig",
    "ABC_TRAINING_SCHEMA_VERSION",
    "bootstrap_abc_demo_subset",
    "generate_abc_proxy_datasets",
]
