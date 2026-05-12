from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from p2cccd.contracts import ProxyType
from p2cccd.data import GeneratedDataset, proposal_row_from_oracle_trace, proxy_mass_from_radius
from p2cccd.data.oracle import evaluate_swept_sphere_oracle
from p2cccd.data.samplers import (
    MotionDiscPairSample,
    PairFamily,
    SPLIT_EASY_NEGATIVE,
    SPLIT_GRAZING,
    SPLIT_MULTI_CONTACT,
    SPLIT_NEAR_CONTACT,
    SPLIT_OOD,
)

from .contracts import OBJECT_ADAPTER_SCHEMA_VERSION, ObjectMeshAsset
from .thingi10k_adapter import THINGI10K_SOURCE_NAME, Thingi10KAdapter, default_thingi10k_root


THINGI10K_TRAINING_SCHEMA_VERSION = 1


def default_thingi10k_cache_dir(root: Path | None = None) -> Path:
    source_root = root if root is not None else default_thingi10k_root()
    return source_root / ".thingi10k_cache"


def _stable_u32(token: str) -> int:
    return int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(hi, max(lo, value))


def _vec(x: float, y: float, z: float = 0.0) -> tuple[float, float, float]:
    return (float(x), float(y), float(z))


def _thingi10k_module() -> Any:
    try:
        import thingi10k  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - runtime guidance
        raise RuntimeError("thingi10k Python package is required; install it in the active environment") from exc
    return thingi10k


def _json_ready_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, Path):
            cleaned[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def _dirty_score_from_entry(entry: dict[str, Any]) -> float:
    score = 0.0
    if bool(entry.get("self_intersecting", False)):
        score += 0.35
    if not bool(entry.get("solid", True)):
        score += 0.15
    if not bool(entry.get("closed", True)):
        score += 0.10
    if not bool(entry.get("vertex_manifold", True)):
        score += 0.10
    if not bool(entry.get("edge_manifold", True)):
        score += 0.10
    if not bool(entry.get("oriented", True)):
        score += 0.10
    if not bool(entry.get("PWN", True)):
        score += 0.10
    components = int(entry.get("num_components", 1) or 1)
    score += min(0.10, max(0, components - 1) * 0.02)
    return _clamp(score)


def _write_obj(path: Path, vertices, facets) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for vertex in vertices:
        lines.append(f"v {float(vertex[0]):.9f} {float(vertex[1]):.9f} {float(vertex[2]):.9f}")
    for face in facets:
        indices = [int(index) + 1 for index in face if int(index) >= 0]
        if len(indices) < 3:
            continue
        if len(indices) == 3:
            lines.append(f"f {indices[0]} {indices[1]} {indices[2]}")
        else:
            anchor = indices[0]
            for offset in range(1, len(indices) - 1):
                lines.append(f"f {anchor} {indices[offset]} {indices[offset + 1]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class Thingi10KOfficialSubsetConfig:
    root: Path | None = None
    cache_dir: Path | None = None
    asset_limit: int = 96
    min_facets: int = 48
    max_facets: int = 1200
    force_redownload: bool = False
    prefer_unique_thing_ids: bool = True


@dataclass(frozen=True, slots=True)
class Thingi10KObjectPair:
    pair_id: str
    asset_a: ObjectMeshAsset
    asset_b: ObjectMeshAsset
    pair_score: float
    dirty_score: float


@dataclass(frozen=True, slots=True)
class Thingi10KProxyDatasetConfig:
    subset: Thingi10KOfficialSubsetConfig = Thingi10KOfficialSubsetConfig()
    train_fraction: float = 0.75
    train_pair_limit: int = 320
    eval_pair_limit: int = 128
    ood_dirty_threshold: float = 0.55
    seed: int = 424242


@dataclass(frozen=True, slots=True)
class Thingi10KProxyDatasetBundle:
    source_root: Path
    assets: tuple[ObjectMeshAsset, ...]
    train_pairs: tuple[Thingi10KObjectPair, ...]
    eval_pairs: tuple[Thingi10KObjectPair, ...]
    train_dataset: GeneratedDataset
    eval_dataset: GeneratedDataset


def prepare_thingi10k_official_subset(
    config: Thingi10KOfficialSubsetConfig | None = None,
) -> Path:
    cfg = config or Thingi10KOfficialSubsetConfig()
    if cfg.asset_limit <= 0:
        raise ValueError("asset_limit must be positive")
    if cfg.min_facets <= 0 or cfg.max_facets < cfg.min_facets:
        raise ValueError("invalid min/max facet range for Thingi10K subset")

    source_root = cfg.root if cfg.root is not None else default_thingi10k_root()
    cache_dir = cfg.cache_dir if cfg.cache_dir is not None else default_thingi10k_cache_dir(source_root)
    subset_manifest_path = source_root / "official_subset_manifest.json"
    existing_objs = tuple(sorted((source_root / "official_subset").rglob("*.obj"))) if source_root.exists() else ()
    if subset_manifest_path.exists() and len(existing_objs) >= cfg.asset_limit:
        return source_root

    thingi10k = _thingi10k_module()
    source_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    thingi10k.init(cache_dir=str(cache_dir), force_redownload=cfg.force_redownload)
    dataset = thingi10k.dataset(num_facets=(cfg.min_facets, cfg.max_facets))

    candidates: list[dict[str, Any]] = []
    best_by_thing: dict[int, dict[str, Any]] = {}
    for index in range(len(dataset)):
        entry = _json_ready_row(dataset[index])
        entry["dirty_score"] = _dirty_score_from_entry(entry)
        if cfg.prefer_unique_thing_ids:
            thing_id = int(entry.get("thing_id", -1))
            incumbent = best_by_thing.get(thing_id)
            if incumbent is None or (
                float(entry["dirty_score"]),
                -int(entry.get("num_facets", 0)),
                -int(entry.get("file_id", 0)),
            ) > (
                float(incumbent["dirty_score"]),
                -int(incumbent.get("num_facets", 0)),
                -int(incumbent.get("file_id", 0)),
            ):
                best_by_thing[thing_id] = entry
        else:
            candidates.append(entry)
    if cfg.prefer_unique_thing_ids:
        candidates = list(best_by_thing.values())
    candidates.sort(
        key=lambda row: (
            -float(row["dirty_score"]),
            int(row.get("num_facets", 0)),
            int(row.get("file_id", 0)),
        )
    )
    selected = candidates[: cfg.asset_limit]
    subset_dir = source_root / "official_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)

    materialized: list[dict[str, Any]] = []
    for rank, row in enumerate(selected):
        category = str(row.get("category", "unknown") or "unknown").strip().replace("/", "_")
        stem = f"{int(row['file_id'])}_{_stable_u32(str(row.get('name', 'thingi10k'))):08x}"
        mesh_path = subset_dir / category / f"{stem}.obj"
        if not mesh_path.exists():
            vertices, facets = thingi10k.load_file(str(row["file_path"]))
            _write_obj(mesh_path, vertices, facets)
        sidecar = {
            **row,
            "selection_rank": rank,
            "source_dataset": THINGI10K_SOURCE_NAME,
            "source_variant": "npz",
            "source_cache_dir": str(cache_dir),
            "source_repository": "https://github.com/Thingi10K/Thingi10K",
        }
        mesh_path.with_suffix(".json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        materialized.append(
            {
                "file_id": int(row["file_id"]),
                "thing_id": int(row.get("thing_id", -1)),
                "mesh_path": str(mesh_path.relative_to(source_root).as_posix()),
                "category": category,
                "num_vertices": int(row.get("num_vertices", 0)),
                "num_facets": int(row.get("num_facets", 0)),
                "dirty_score": float(row["dirty_score"]),
                "license": str(row.get("license", "")),
            }
        )

    license_path = source_root / "LICENSE"
    license_path.write_text(
        (
            "Thingi10K official subset materialized for local P2CCCD experiments.\n"
            "Source repository: https://github.com/Thingi10K/Thingi10K\n"
            "Source package: thingi10k\n"
            "Each mesh sidecar preserves original dataset metadata and per-entry license strings.\n"
            "Do not redistribute without respecting the upstream dataset and per-model license terms.\n"
        ),
        encoding="utf-8",
    )
    subset_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": THINGI10K_TRAINING_SCHEMA_VERSION,
                "source_name": THINGI10K_SOURCE_NAME,
                "source_repository": "https://github.com/Thingi10K/Thingi10K",
                "cache_dir": str(cache_dir),
                "asset_limit": cfg.asset_limit,
                "min_facets": cfg.min_facets,
                "max_facets": cfg.max_facets,
                "prefer_unique_thing_ids": cfg.prefer_unique_thing_ids,
                "asset_count": len(materialized),
                "assets": materialized,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return source_root


def _radius_from_asset(asset: ObjectMeshAsset) -> float:
    return max(0.05, 0.5 * max(asset.stats.diagonal, 0.1))


def _object_id(asset: ObjectMeshAsset) -> int:
    return 300000 + (_stable_u32(asset.object_id) % 600000)


def _pair_scale(asset_a: ObjectMeshAsset, asset_b: ObjectMeshAsset) -> float:
    return max(0.1, 0.5 * (asset_a.stats.diagonal + asset_b.stats.diagonal))


def _pair_score(asset_a: ObjectMeshAsset, asset_b: ObjectMeshAsset) -> float:
    diag_a = max(asset_a.stats.diagonal, 1.0e-12)
    diag_b = max(asset_b.stats.diagonal, 1.0e-12)
    scale_similarity = math.exp(-abs(math.log(diag_a / diag_b)))
    dirty = max(asset_a.dirty_score, asset_b.dirty_score)
    return _clamp(0.65 * dirty + 0.35 * scale_similarity)


def _make_pairs(assets: Sequence[ObjectMeshAsset], *, limit: int) -> tuple[Thingi10KObjectPair, ...]:
    pairs: list[Thingi10KObjectPair] = []
    for asset_a, asset_b in combinations(assets, 2):
        pair_score = _pair_score(asset_a, asset_b)
        pairs.append(
            Thingi10KObjectPair(
                pair_id=f"{asset_a.object_id}__{asset_b.object_id}",
                asset_a=asset_a,
                asset_b=asset_b,
                pair_score=pair_score,
                dirty_score=max(asset_a.dirty_score, asset_b.dirty_score),
            )
        )
    pairs.sort(key=lambda pair: (-pair.pair_score, -pair.dirty_score, pair.pair_id))
    return tuple(pairs[:limit])


def _sample_for_variant(
    pair: Thingi10KObjectPair,
    *,
    sample_id: int,
    variant_index: int,
    split: str,
    hardness: float,
    ood: bool,
) -> MotionDiscPairSample:
    radius_a = _radius_from_asset(pair.asset_a)
    radius_b = _radius_from_asset(pair.asset_b)
    radius_sum = radius_a + radius_b
    scale = _pair_scale(pair.asset_a, pair.asset_b)
    lateral = (((_stable_u32(pair.pair_id) + 17 * variant_index) % 200) / 199.0 - 0.5) * 0.18 * scale
    z_bias = (((_stable_u32(pair.pair_id) + 37 * variant_index) % 200) / 199.0 - 0.5) * 0.10 * scale
    dirty = pair.dirty_score
    if split == SPLIT_EASY_NEGATIVE:
        start_gap = radius_sum + (1.2 + 0.6 * dirty) * scale
        end_gap = radius_sum + (0.55 + 0.25 * dirty) * scale
        ax1 = (0.18 + 0.12 * dirty) * scale
    elif split == SPLIT_NEAR_CONTACT:
        start_gap = radius_sum + (0.28 + 0.08 * dirty) * scale
        end_gap = radius_sum + (0.03 + 0.03 * dirty) * scale
        ax1 = (0.20 + 0.08 * dirty) * scale
    elif split == SPLIT_GRAZING:
        start_gap = radius_sum + (0.18 + 0.06 * dirty) * scale
        end_gap = radius_sum - (0.01 + 0.02 * dirty) * scale
        ax1 = (0.14 + 0.07 * dirty) * scale
    elif split == SPLIT_MULTI_CONTACT:
        start_gap = radius_sum + (0.35 + 0.08 * dirty) * scale
        end_gap = -(radius_sum + (0.16 + 0.08 * dirty) * scale)
        ax1 = (0.08 + 0.04 * dirty) * scale
    elif split == SPLIT_OOD:
        start_gap = radius_sum + (0.22 + 0.18 * dirty) * scale
        end_gap = radius_sum - (0.12 + 0.12 * dirty) * scale
        ax1 = (0.26 + 0.10 * dirty) * scale
    else:
        raise ValueError(f"unsupported Thingi10K split: {split}")
    center_a_t0 = _vec(-0.12 * scale, lateral, z_bias)
    center_a_t1 = _vec(ax1, lateral + 0.03 * scale * ((variant_index % 2) * 2 - 1), z_bias)
    center_b_t0 = _vec(start_gap, lateral - 0.02 * scale, z_bias)
    center_b_t1 = _vec(end_gap, lateral + 0.02 * scale, z_bias)
    return MotionDiscPairSample(
        sample_id=sample_id,
        query_id=9_000_000 + sample_id,
        candidate_id=9_200_000 + sample_id,
        split=split,
        family=PairFamily.MESH_PAIR,
        object_a_id=_object_id(pair.asset_a),
        patch_a_id=1,
        object_b_id=_object_id(pair.asset_b),
        patch_b_id=1,
        slab_id=variant_index,
        center_a_t0=center_a_t0,
        center_a_t1=center_a_t1,
        center_b_t0=center_b_t0,
        center_b_t1=center_b_t1,
        radius_a=radius_a,
        radius_b=radius_b,
        proxy_type_a=ProxyType.SWEPT_AABB,
        proxy_type_b=ProxyType.SWEPT_AABB,
        hardness=_clamp(hardness),
        ood=ood,
        mass_a=proxy_mass_from_radius(radius_a),
        mass_b=proxy_mass_from_radius(radius_b),
        restitution=1.0,
    )


def _samples_from_pairs(
    pairs: Sequence[Thingi10KObjectPair],
    *,
    first_sample_id: int,
    ood_dirty_threshold: float,
) -> list[MotionDiscPairSample]:
    samples: list[MotionDiscPairSample] = []
    sample_id = first_sample_id
    for pair in pairs:
        dirty = pair.dirty_score
        scale_similarity = math.exp(
            -abs(math.log(max(pair.asset_a.stats.diagonal, 1.0e-12) / max(pair.asset_b.stats.diagonal, 1.0e-12)))
        )
        base_hardness = _clamp(0.25 + 0.45 * dirty + 0.30 * scale_similarity)
        variant_specs = (
            (SPLIT_EASY_NEGATIVE, _clamp(base_hardness - 0.20), False),
            (SPLIT_NEAR_CONTACT, _clamp(base_hardness + 0.10), dirty >= ood_dirty_threshold),
            (SPLIT_GRAZING, _clamp(base_hardness + 0.20), dirty >= ood_dirty_threshold),
            (SPLIT_MULTI_CONTACT if dirty < ood_dirty_threshold else SPLIT_OOD, _clamp(base_hardness + 0.30), dirty >= ood_dirty_threshold),
        )
        for variant_index, (split, hardness, ood) in enumerate(variant_specs):
            samples.append(
                _sample_for_variant(
                    pair,
                    sample_id=sample_id,
                    variant_index=variant_index,
                    split=split,
                    hardness=hardness,
                    ood=ood,
                )
            )
            sample_id += 1
    return samples


def _dataset_from_samples(samples: Sequence[MotionDiscPairSample]) -> GeneratedDataset:
    split_names = tuple(
        split
        for split in (
            SPLIT_EASY_NEGATIVE,
            SPLIT_NEAR_CONTACT,
            SPLIT_GRAZING,
            SPLIT_MULTI_CONTACT,
            SPLIT_OOD,
        )
        if any(sample.split == split for sample in samples)
    )
    traces = [evaluate_swept_sphere_oracle(sample) for sample in samples]
    rows = [proposal_row_from_oracle_trace(sample, trace) for sample, trace in zip(samples, traces)]
    return GeneratedDataset(rows=rows, samples=list(samples), traces=traces, split_names=split_names)


def generate_thingi10k_proxy_datasets(
    config: Thingi10KProxyDatasetConfig | None = None,
) -> Thingi10KProxyDatasetBundle:
    cfg = config or Thingi10KProxyDatasetConfig()
    if cfg.train_pair_limit <= 0 or cfg.eval_pair_limit <= 0:
        raise ValueError("train_pair_limit and eval_pair_limit must be positive")
    if cfg.train_fraction <= 0.0 or cfg.train_fraction >= 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    source_root = prepare_thingi10k_official_subset(cfg.subset)
    adapter = Thingi10KAdapter(source_root)
    assets = adapter.list_assets(limit=cfg.subset.asset_limit)
    split_index = max(2, min(len(assets) - 2, int(round(len(assets) * cfg.train_fraction))))
    train_assets = assets[:split_index]
    eval_assets = assets[split_index:]
    train_pairs = _make_pairs(train_assets, limit=cfg.train_pair_limit)
    eval_pairs = _make_pairs(eval_assets, limit=cfg.eval_pair_limit)
    train_dataset = _dataset_from_samples(
        _samples_from_pairs(train_pairs, first_sample_id=1, ood_dirty_threshold=cfg.ood_dirty_threshold)
    )
    eval_dataset = _dataset_from_samples(
        _samples_from_pairs(eval_pairs, first_sample_id=1_000_001, ood_dirty_threshold=cfg.ood_dirty_threshold)
    )
    return Thingi10KProxyDatasetBundle(
        source_root=source_root,
        assets=assets,
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


__all__ = [
    "THINGI10K_TRAINING_SCHEMA_VERSION",
    "Thingi10KOfficialSubsetConfig",
    "Thingi10KObjectPair",
    "Thingi10KProxyDatasetBundle",
    "Thingi10KProxyDatasetConfig",
    "default_thingi10k_cache_dir",
    "generate_thingi10k_proxy_datasets",
    "prepare_thingi10k_official_subset",
]
