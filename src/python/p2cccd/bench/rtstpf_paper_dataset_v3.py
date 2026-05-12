from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from p2cccd.data.dataset import DATASET_SHARD_SCHEMA_VERSION
from p2cccd.datasets.tight_inclusion_stpf_features import tight_inclusion_query_to_proposal_row
from p2cccd.proposal.features import ProposalFeatureRow, validate_proposal_feature_row

from .high_density_mesh_training_benchmark import (
    MeshDensityAsset,
    MeshDensityPair,
    _dataset_from_samples,
    _load_abc_assets,
    _load_fusion360_assets,
    _load_thingi10k_assets,
    _sample_from_pair,
    _scale_workload_costs,
)
from .large_dense_complex_mesh_cases import _make_heavy_cross_pairs, _make_heavy_intra_pairs
from .multi_dense_mesh_contact_pairs import MultiDenseMeshContactPairsConfig, _load_large_face_abc_assets, _rename_assets
from .rtstpf_paper_dataset_v2 import (
    SOURCE_FULL_SCENE,
    SOURCE_ORIGINAL,
    SOURCE_ROUNDED,
    PaperDatasetV2Record,
    _discover_full_scene_tree,
    _discover_tight_inclusion_tree,
    _iter_queries_for_manifest_file,
    _records_to_npz,
    _summarize_rows,
    write_manifest,
)
from .trained_stpf_high_density import HighDensitySTPFConfig, build_high_density_stpf_workload


MANIFEST_SCHEMA_VERSION = 3
DATASET_NAME = "rtstpf_paper_dataset_v3"

SOURCE_DENSE_ABC_MEGAFACE = "dense_abc_megaface"
SOURCE_DENSE_FUSION360_ASSEMBLY = "dense_fusion360_assembly"
SOURCE_DENSE_THINGI10K_DIRTY = "dense_thingi10k_dirty"
SOURCE_DENSE_REAL_MESH_CONTACT = "dense_real_mesh_contact"

DENSE_SOURCES = (
    SOURCE_DENSE_ABC_MEGAFACE,
    SOURCE_DENSE_FUSION360_ASSEMBLY,
    SOURCE_DENSE_THINGI10K_DIRTY,
    SOURCE_DENSE_REAL_MESH_CONTACT,
)

FILE_SOURCES = (SOURCE_ORIGINAL, SOURCE_ROUNDED, SOURCE_FULL_SCENE)

PRESET_ROW_LIMITS: dict[str, dict[str, int | None]] = {
    "smoke": {"train": 120_000, "validation": 24_000, "heldout_test": 24_000},
    "large": {"train": 20_000_000, "validation": 2_000_000, "heldout_test": 2_000_000},
    "paper_full": {"train": 80_000_000, "validation": 8_000_000, "heldout_test": 8_000_000},
}

SOURCE_QUOTAS = {
    SOURCE_ORIGINAL: 0.30,
    SOURCE_ROUNDED: 0.20,
    SOURCE_FULL_SCENE: 0.15,
    SOURCE_DENSE_ABC_MEGAFACE: 0.10,
    SOURCE_DENSE_FUSION360_ASSEMBLY: 0.10,
    SOURCE_DENSE_THINGI10K_DIRTY: 0.075,
    SOURCE_DENSE_REAL_MESH_CONTACT: 0.075,
}

MAX_POSITIVE_RATIO = {
    SOURCE_ORIGINAL: 0.55,
    SOURCE_ROUNDED: 0.55,
    SOURCE_FULL_SCENE: 0.50,
    SOURCE_DENSE_ABC_MEGAFACE: 0.35,
    SOURCE_DENSE_FUSION360_ASSEMBLY: 0.35,
    SOURCE_DENSE_THINGI10K_DIRTY: 0.35,
    SOURCE_DENSE_REAL_MESH_CONTACT: 0.35,
}


@dataclass(frozen=True, slots=True)
class DenseSourceConfig:
    abc_root: Path = Path("src/datasets/abc_official")
    fusion360_root: Path = Path("src/datasets/fusion360")
    thingi10k_root: Path = Path("src/datasets/thingi10k")
    asset_limit_per_source: int = 192
    pair_limit_per_source: int = 4096
    samples_per_pair: int = 4
    seed: int = 424242
    high_density: HighDensitySTPFConfig = HighDensitySTPFConfig(
        slab_count=16,
        patches_per_object=12,
        representative_attempt_limit=3,
        uncertainty_fallback_threshold=0.75,
        narrow_interval_min_cost_scale=0.18,
        interval_miss_penalty_scale=0.22,
        full_exact_cost_scale=1.0,
    )


def _stable_unit_interval(text: str) -> float:
    value = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)
    return float(value) / float(0xFFFFFFFFFFFF)


def _split_sequence(keys: Sequence[str], *, train: float = 0.70, validation: float = 0.10) -> dict[str, str]:
    unique_keys = sorted(set(keys), key=lambda key: (_stable_unit_interval(key), key))
    if not unique_keys:
        return {}
    if len(unique_keys) == 1:
        return {unique_keys[0]: "train"}
    train_count = max(1, int(round(len(unique_keys) * train)))
    validation_count = max(1, int(round(len(unique_keys) * validation))) if len(unique_keys) >= 3 else 0
    if train_count + validation_count >= len(unique_keys):
        train_count = max(1, len(unique_keys) - 1 - validation_count)
    if train_count + validation_count >= len(unique_keys):
        validation_count = 0
        train_count = max(1, len(unique_keys) - 1)
    out: dict[str, str] = {}
    for index, key in enumerate(unique_keys):
        if index < train_count:
            out[key] = "train"
        elif index < train_count + validation_count:
            out[key] = "validation"
        else:
            out[key] = "heldout_test"
    return out


def _apply_scene_level_splits(rows: list[dict[str, object]]) -> None:
    keys_by_source: dict[str, list[str]] = {}
    for row in rows:
        case = str(row["case"])
        if case == "unit-tests":
            row["split"] = "unit_smoke"
            row["split_key"] = f"{row['source_type']}/{case}"
            continue
        key = f"{row['source_type']}/{case}"
        keys_by_source.setdefault(str(row["source_type"]), []).append(key)
        row["split_key"] = key
    split_by_key: dict[str, str] = {}
    for source, keys in keys_by_source.items():
        split_by_key.update(_split_sequence(keys))
    for row in rows:
        if row.get("split") == "unit_smoke":
            continue
        row["split"] = split_by_key[str(row["split_key"])]


def build_rtstpf_paper_dataset_v3_manifest(
    dataset_root: Path,
    *,
    inspect_counts: bool = False,
    seed: int = 424242,
) -> dict[str, object]:
    root = Path(dataset_root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    rows: list[dict[str, object]] = []
    rows.extend(
        _discover_tight_inclusion_tree(
            root,
            source_type=SOURCE_ORIGINAL,
            source_root=root,
            manifest_root=root,
            inspect_counts=inspect_counts,
        )
    )
    rounded_root = root / "rounded_ground_truth"
    if rounded_root.exists():
        rows.extend(
            _discover_tight_inclusion_tree(
                root,
                source_type=SOURCE_ROUNDED,
                source_root=rounded_root,
                manifest_root=root,
                inspect_counts=inspect_counts,
            )
        )
    rows.extend(_discover_full_scene_tree(root, inspect_counts=inspect_counts))
    _apply_scene_level_splits(rows)
    rows.sort(key=lambda row: (str(row["source_type"]), str(row["case"]), str(row["kind"]), str(row["csv_path"])))
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "dataset_root": root.as_posix(),
        "seed": int(seed),
        "split_policy": {
            "granularity": "object_or_scene_level",
            "file_sources": "source_type/case; no same simulation case across train/validation/heldout",
            "dense_sources": "asset-level; a dense pair is used only when both CAD assets belong to the same split",
            "train": 0.70,
            "validation": 0.10,
            "heldout_test": 0.20,
        },
        "sampling_policy": {
            "target_rows": PRESET_ROW_LIMITS,
            "source_quotas": SOURCE_QUOTAS,
            "max_positive_ratio": MAX_POSITIVE_RATIO,
            "positive_oversample": 1,
            "full_scene_positive_cap": "full_scene_toi is capped to avoid dominating the training prior",
            "hard_negatives": "negative rows with high geometric proximity/cost/uncertainty are retained and retargeted instead of being discarded",
        },
        "dense_sources": list(DENSE_SOURCES),
        "sources": list(FILE_SOURCES) + list(DENSE_SOURCES),
        "summary": _summarize_rows(rows),
        "files": rows,
    }


def _copy_row(row: ProposalFeatureRow) -> ProposalFeatureRow:
    return ProposalFeatureRow(
        schema_version=row.schema_version,
        query_id=row.query_id,
        candidate_id=row.candidate_id,
        slab_id=row.slab_id,
        object_a_id=row.object_a_id,
        patch_a_id=row.patch_a_id,
        object_b_id=row.object_b_id,
        patch_b_id=row.patch_b_id,
        features=[float(value) for value in row.features],
        interval_targets=[float(value) for value in row.interval_targets],
        family_targets=[float(value) for value in row.family_targets],
        priority_target=float(row.priority_target),
        cost_target=float(row.cost_target),
        uncertainty_target=float(row.uncertainty_target),
        target_mask=int(row.target_mask),
    )


def _is_hard_negative(row: ProposalFeatureRow) -> bool:
    if row.priority_target >= 0.30 or row.uncertainty_target >= 0.35 or row.cost_target >= 2.0:
        return True
    features = row.features
    proximity = min(abs(features[index]) for index in (14, 15, 16, 24, 25) if index < len(features))
    return proximity < 1.0e-3


def _retarget_for_v3(row: ProposalFeatureRow, *, ground_truth: bool, hard_negative: bool) -> ProposalFeatureRow:
    out = _copy_row(row)
    if ground_truth:
        out.priority_target = 1.0
        # The runtime score is priority + uncertainty_weight * uncertainty.
        # Positive CCD contacts must stay above the conservative zero-FN gate even
        # when the MLP is under-trained or evaluated on a held-out dense source.
        out.uncertainty_target = max(0.55, float(out.uncertainty_target))
        out.cost_target = max(0.75, float(out.cost_target))
    elif hard_negative:
        out.priority_target = min(0.40, max(0.18, float(out.priority_target)))
        out.uncertainty_target = min(0.35, max(0.12, float(out.uncertainty_target)))
        out.cost_target = max(0.35, float(out.cost_target))
    else:
        out.priority_target = min(0.12, float(out.priority_target))
        out.uncertainty_target = min(0.18, float(out.uncertainty_target))
        out.cost_target = min(0.35, max(0.05, float(out.cost_target)))
    return validate_proposal_feature_row(out)


def _manifest_rows_for_split_source(manifest: dict[str, object], split: str, source_type: str) -> list[dict[str, object]]:
    return [
        row
        for row in manifest["files"]  # type: ignore[index]
        if str(row["split"]) == split and str(row["source_type"]) == source_type
    ]


def _file_group_key(row: dict[str, object]) -> tuple[str, str]:
    return (str(row["case"]), str(row["kind"]))


def _iter_file_group_records(
    root: Path,
    rows: Sequence[dict[str, object]],
) -> Iterator[PaperDatasetV2Record]:
    for file_row in rows:
        for query in _iter_queries_for_manifest_file(root, file_row):
            proposal_row = tight_inclusion_query_to_proposal_row(query)
            hard_negative = (not query.ground_truth) and _is_hard_negative(proposal_row)
            proposal_row = _retarget_for_v3(
                proposal_row,
                ground_truth=bool(query.ground_truth),
                hard_negative=hard_negative,
            )
            yield PaperDatasetV2Record(
                row=proposal_row,
                source_type=str(file_row["source_type"]),
                case_name=query.case_name,
                kind=query.kind,
                ground_truth=bool(query.ground_truth),
                csv_path=str(file_row["csv_path"]),
                query_index=int(query.query_index),
            )


def _select_with_positive_cap(
    record: PaperDatasetV2Record,
    *,
    source_type: str,
    row_budget: int,
    emitted: int,
    positive_count: int,
) -> bool:
    if not record.ground_truth:
        return True
    max_positive_rows = int(float(row_budget) * float(MAX_POSITIVE_RATIO.get(source_type, 0.50)))
    if max_positive_rows <= 0:
        return False
    return positive_count < max_positive_rows and emitted < row_budget


def _iter_file_source_records(
    manifest: dict[str, object],
    *,
    split: str,
    source_type: str,
    row_budget: int,
) -> Iterator[PaperDatasetV2Record]:
    root = Path(str(manifest["dataset_root"]))
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in _manifest_rows_for_split_source(manifest, split, source_type):
        groups.setdefault(_file_group_key(row), []).append(row)
    iterators: list[Iterator[PaperDatasetV2Record]] = []
    for key in sorted(groups):
        file_rows = sorted(groups[key], key=lambda row: str(row["csv_path"]))
        iterators.append(_iter_file_group_records(root, file_rows))
    emitted = 0
    positive_count = 0
    scanned = 0
    scan_budget = max(row_budget * 20, row_budget + 50_000)
    cursor = 0
    while iterators and emitted < row_budget and scanned < scan_budget:
        cursor %= len(iterators)
        iterator = iterators[cursor]
        try:
            record = next(iterator)
        except StopIteration:
            iterators.pop(cursor)
            continue
        scanned += 1
        if _select_with_positive_cap(
            record,
            source_type=source_type,
            row_budget=row_budget,
            emitted=emitted,
            positive_count=positive_count,
        ):
            emitted += 1
            positive_count += int(record.ground_truth)
            yield record
        cursor += 1


def _asset_split_map(assets: Sequence[MeshDensityAsset], source_type: str) -> dict[str, str]:
    keys = [f"{source_type}/{asset.asset_id}/{asset.asset_path}" for asset in assets]
    split_by_key = _split_sequence(keys)
    return {asset.asset_path: split_by_key[f"{source_type}/{asset.asset_id}/{asset.asset_path}"] for asset in assets}


def _load_dense_pairs(source_type: str, cfg: DenseSourceConfig) -> tuple[MeshDensityPair, ...]:
    if source_type == SOURCE_DENSE_ABC_MEGAFACE:
        abc = _rename_assets(_load_abc_assets(cfg.abc_root, cfg.asset_limit_per_source), "ABC megaface")
        return _make_heavy_intra_pairs("ABC-megaface-v3", abc, limit=cfg.pair_limit_per_source)
    if source_type == SOURCE_DENSE_FUSION360_ASSEMBLY:
        fusion = _rename_assets(_load_fusion360_assets(cfg.fusion360_root, cfg.asset_limit_per_source), "Fusion360 assembly")
        return _make_heavy_intra_pairs("Fusion360-assembly-v3", fusion, limit=cfg.pair_limit_per_source)
    if source_type == SOURCE_DENSE_THINGI10K_DIRTY:
        thingi = tuple(
            sorted(
                _rename_assets(_load_thingi10k_assets(cfg.thingi10k_root, cfg.asset_limit_per_source), "Thingi10K dirty"),
                key=lambda asset: (-asset.dirty_score, -asset.face_count, asset.asset_id),
            )
        )
        return _make_heavy_intra_pairs("Thingi10K-dirty-v3", thingi, limit=cfg.pair_limit_per_source)
    if source_type == SOURCE_DENSE_REAL_MESH_CONTACT:
        large_cfg = MultiDenseMeshContactPairsConfig(
            abc_root=cfg.abc_root.as_posix(),
            fusion360_root=cfg.fusion360_root.as_posix(),
            thingi10k_root=cfg.thingi10k_root.as_posix(),
            asset_limit_per_source=cfg.asset_limit_per_source,
        )
        abc_large = _rename_assets(_load_large_face_abc_assets(large_cfg), "Real mesh contact")
        abc_mega = _rename_assets(_load_abc_assets(cfg.abc_root, cfg.asset_limit_per_source), "ABC megaface")
        return _make_heavy_cross_pairs(
            "real-mesh-contact-v3",
            abc_large,
            abc_mega,
            limit=cfg.pair_limit_per_source,
        )
    raise ValueError(f"unsupported dense source {source_type!r}")


def _filter_dense_pairs_by_split(
    pairs: Sequence[MeshDensityPair],
    *,
    source_type: str,
    split: str,
) -> tuple[MeshDensityPair, ...]:
    assets: dict[str, MeshDensityAsset] = {}
    for pair in pairs:
        assets[pair.asset_a.asset_path] = pair.asset_a
        assets[pair.asset_b.asset_path] = pair.asset_b
    split_by_asset = _asset_split_map(tuple(assets.values()), source_type)
    filtered = [
        pair
        for pair in pairs
        if split_by_asset.get(pair.asset_a.asset_path) == split and split_by_asset.get(pair.asset_b.asset_path) == split
    ]
    if not filtered and pairs:
        # Small smoke runs or cross-source dense pairs may not have a pair whose
        # two assets both fall into the same split. Fall back to pair-level
        # assignment so every source has calibration coverage. Large/paper runs
        # still prefer asset-level split whenever it yields valid pairs.
        pair_keys = [
            f"{source_type}/{pair.pair_id}/{pair.asset_a.asset_path}/{pair.asset_b.asset_path}"
            for pair in pairs
        ]
        split_by_pair = _split_sequence(pair_keys)
        filtered = [
            pair
            for pair, key in zip(pairs, pair_keys)
            if split_by_pair.get(key) == split
        ]
    return tuple(filtered)


def _dense_case_name(source_type: str) -> str:
    return {
        SOURCE_DENSE_ABC_MEGAFACE: "ABC megaface dense",
        SOURCE_DENSE_FUSION360_ASSEMBLY: "Fusion360 assembly dense",
        SOURCE_DENSE_THINGI10K_DIRTY: "Thingi10K dirty mesh dense",
        SOURCE_DENSE_REAL_MESH_CONTACT: "real mesh contact dense",
    }[source_type]


def _dense_record_from_row(
    *,
    source_type: str,
    pair: MeshDensityPair,
    row: ProposalFeatureRow,
    ground_truth: bool,
    hard_negative: bool,
) -> PaperDatasetV2Record:
    proposal_row = _retarget_for_v3(row, ground_truth=ground_truth, hard_negative=hard_negative)
    pair_key = f"{Path(pair.asset_a.asset_path).stem}__{Path(pair.asset_b.asset_path).stem}"
    return PaperDatasetV2Record(
        row=proposal_row,
        source_type=source_type,
        case_name=_dense_case_name(source_type),
        kind="mesh-contact-candidate",
        ground_truth=ground_truth,
        csv_path=f"dense://{source_type}/{pair_key}",
        query_index=int(row.candidate_id),
    )


def _iter_dense_source_records(
    *,
    split: str,
    source_type: str,
    row_budget: int,
    cfg: DenseSourceConfig,
    pairs: Sequence[MeshDensityPair] | None = None,
) -> Iterator[PaperDatasetV2Record]:
    pairs = _filter_dense_pairs_by_split(
        tuple(pairs) if pairs is not None else _load_dense_pairs(source_type, cfg),
        source_type=source_type,
        split=split,
    )
    emitted = 0
    positive_count = 0
    scanned = 0
    scan_budget = max(row_budget * 20, row_budget + 50_000)
    sample_id = 30_000_000 + int(_stable_unit_interval(source_type) * 1_000_000)
    variant_index = 0
    for pair in pairs:
        for _ in range(cfg.samples_per_pair):
            sample = _sample_from_pair(pair, sample_id=sample_id, variant_index=variant_index)
            dataset = _dataset_from_samples([sample])
            workload = _scale_workload_costs(
                build_high_density_stpf_workload(dataset, cfg.high_density, name=f"{source_type}_{split}"),
                {sample.query_id: pair.cost_scale},
            )
            trace = workload.traces_by_query_id[sample.query_id]
            for row in workload.rows:
                if scanned >= scan_budget:
                    return
                info = workload.candidate_infos[int(row.candidate_id)]
                ground_truth = bool(info.slab_overlap_contact or (trace.collided and info.preferred_representative))
                hard_negative = (not ground_truth) and (
                    bool(info.slab_contains_reference_time)
                    or bool(info.preferred_representative)
                    or float(info.patch_match_score) >= 0.50
                )
                record = _dense_record_from_row(
                    source_type=source_type,
                    pair=pair,
                    row=row,
                    ground_truth=ground_truth,
                    hard_negative=hard_negative,
                )
                scanned += 1
                if _select_with_positive_cap(
                    record,
                    source_type=source_type,
                    row_budget=row_budget,
                    emitted=emitted,
                    positive_count=positive_count,
                ):
                    emitted += 1
                    positive_count += int(record.ground_truth)
                    yield record
                    if emitted >= row_budget:
                        return
            sample_id += 1
            variant_index += 1


def _source_budgets(row_limit: int) -> dict[str, int]:
    ordered_sources = list(SOURCE_QUOTAS)
    budgets: dict[str, int] = {}
    assigned = 0
    for source in ordered_sources[:-1]:
        value = int(round(float(row_limit) * SOURCE_QUOTAS[source]))
        budgets[source] = value
        assigned += value
    budgets[ordered_sources[-1]] = max(0, int(row_limit) - assigned)
    return budgets


def _iter_records_v3(
    manifest: dict[str, object],
    *,
    split: str,
    row_limit: int,
    dense_config: DenseSourceConfig,
    dense_pair_cache: dict[str, tuple[MeshDensityPair, ...]],
) -> Iterator[PaperDatasetV2Record]:
    for source_type, budget in _source_budgets(row_limit).items():
        if budget <= 0:
            continue
        if source_type in FILE_SOURCES:
            yield from _iter_file_source_records(
                manifest,
                split=split,
                source_type=source_type,
                row_budget=budget,
            )
        else:
            yield from _iter_dense_source_records(
                split=split,
                source_type=source_type,
                row_budget=budget,
                cfg=dense_config,
                pairs=dense_pair_cache.get(source_type),
            )


def _record_stats(records: Sequence[PaperDatasetV2Record]) -> dict[str, int]:
    rows = list(records)
    return {
        "row_count": len(rows),
        "positive_count": sum(1 for record in rows if record.ground_truth),
        "negative_count": sum(1 for record in rows if not record.ground_truth),
        "hard_negative_count": sum(
            1
            for record in rows
            if (not record.ground_truth)
            and (record.row.priority_target >= 0.18 or record.row.uncertainty_target >= 0.12)
        ),
    }


def build_rtstpf_paper_dataset_v3_shards(
    manifest_path: Path,
    *,
    output_dir: Path,
    preset: str = "smoke",
    splits: Sequence[str] = ("train", "validation", "heldout_test"),
    chunk_rows: int = 500_000,
    resume: bool = True,
    dense_config: DenseSourceConfig | None = None,
) -> dict[str, object]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if preset not in PRESET_ROW_LIMITS:
        raise ValueError(f"unsupported preset {preset!r}; expected {sorted(PRESET_ROW_LIMITS)}")
    cfg = dense_config or DenseSourceConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, object]] = []
    row_limits = PRESET_ROW_LIMITS[preset]
    stats_by_split: dict[str, dict[str, int]] = {}
    dense_pair_cache = {source_type: _load_dense_pairs(source_type, cfg) for source_type in DENSE_SOURCES}
    for split in splits:
        split_limit = row_limits.get(split)
        if split_limit is None:
            raise ValueError("v3 shard writer requires finite row limits")
        buffer: list[PaperDatasetV2Record] = []
        chunk_index = 0
        split_stats = {"row_count": 0, "positive_count": 0, "negative_count": 0, "hard_negative_count": 0}
        for record in _iter_records_v3(
            manifest,
            split=split,
            row_limit=int(split_limit),
            dense_config=cfg,
            dense_pair_cache=dense_pair_cache,
        ):
            buffer.append(record)
            if len(buffer) >= chunk_rows:
                path = output_dir / split / f"chunk_{chunk_index:06d}.npz"
                existed = path.exists()
                chunk_stats = _record_stats(buffer)
                if not (resume and existed):
                    _records_to_npz(
                        path,
                        buffer,
                        metadata={
                            "source": DATASET_NAME,
                            "split": split,
                            "preset": preset,
                            "chunk_index": chunk_index,
                            "sampling_policy": "scene_asset_level_balanced_hard_negative",
                            **chunk_stats,
                        },
                    )
                chunks.append(
                    {
                        "split": split,
                        "path": path.as_posix(),
                        "row_count": len(buffer),
                        "resumed": bool(resume and existed),
                        **chunk_stats,
                    }
                )
                for key, value in chunk_stats.items():
                    split_stats[key] += int(value)
                buffer = []
                chunk_index += 1
        if buffer:
            path = output_dir / split / f"chunk_{chunk_index:06d}.npz"
            existed = path.exists()
            chunk_stats = _record_stats(buffer)
            if not (resume and existed):
                _records_to_npz(
                    path,
                    buffer,
                    metadata={
                        "source": DATASET_NAME,
                        "split": split,
                        "preset": preset,
                        "chunk_index": chunk_index,
                        "sampling_policy": "scene_asset_level_balanced_hard_negative",
                        **chunk_stats,
                    },
                )
            chunks.append(
                {
                    "split": split,
                    "path": path.as_posix(),
                    "row_count": len(buffer),
                    "resumed": bool(resume and existed),
                    **chunk_stats,
                }
            )
            for key, value in chunk_stats.items():
                split_stats[key] += int(value)
        stats_by_split[split] = split_stats
    shard_manifest = {
        "schema_version": DATASET_SHARD_SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "preset": preset,
        "manifest_path": Path(manifest_path).as_posix(),
        "output_dir": output_dir.as_posix(),
        "chunk_rows": int(chunk_rows),
        "source_quotas": SOURCE_QUOTAS,
        "max_positive_ratio": MAX_POSITIVE_RATIO,
        "dense_config": {
            "abc_root": cfg.abc_root.as_posix(),
            "fusion360_root": cfg.fusion360_root.as_posix(),
            "thingi10k_root": cfg.thingi10k_root.as_posix(),
            "asset_limit_per_source": cfg.asset_limit_per_source,
            "pair_limit_per_source": cfg.pair_limit_per_source,
            "samples_per_pair": cfg.samples_per_pair,
            "high_density": {
                "slab_count": cfg.high_density.slab_count,
                "patches_per_object": cfg.high_density.patches_per_object,
            },
        },
        "chunks": chunks,
        "row_counts_by_split": {
            split: sum(int(chunk["row_count"]) for chunk in chunks if chunk["split"] == split)
            for split in splits
        },
        "stats_by_split": stats_by_split,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(shard_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return shard_manifest


def write_design_report(path: Path, manifest: dict[str, object], *, shard_plan: dict[str, object] | None = None) -> Path:
    summary = manifest["summary"]
    lines = [
        "# RTSTPFExact Paper Dataset v3 design report",
        "",
        "## Objective",
        "",
        "- Objectivedescription: `80M train + 8M validation + 8M heldout`. ",
        "- split descriptionto object/scene-level, avoid the same CAD object orsame simulation case leakage. ",
        "- full-scene positive description; use source quota + positive cap + hard negatives keep conservative scheduling generalization. ",
        "- dense sources coverage `ABC megaface`, `Fusion360 assembly`, `Thingi10K dirty mesh`, `real mesh contact`. ",
        "- descriptionObjectiveis zero-FN threshold underdescription exact-call reduction, rather than fallback all. ",
        "",
        "## description",
        "",
        f"- Dataset root: `{manifest['dataset_root']}`",
        f"- File count: `{summary['file_count']}`",
        f"- Bytes indexed: `{summary['total_bytes']}`",
        f"- Known queries: `{summary['known_query_count']}`",
        f"- Known positives: `{summary['known_positive_count']}`",
        f"- Known positive ratio: `{summary['known_positive_ratio']}`",
        "",
        "## Source Quota",
        "",
        "| Source | Quota | Max positive ratio |",
        "| --- | ---: | ---: |",
    ]
    for source, quota in SOURCE_QUOTAS.items():
        lines.append(f"| `{source}` | `{100.0 * quota:.1f}%` | `{100.0 * MAX_POSITIVE_RATIO[source]:.1f}%` |")
    lines.extend(["", "## Scene-level Split", "", "| Split | Files | Bytes | Known queries | Known positives |", "| --- | ---: | ---: | ---: | ---: |"])
    for split, row in sorted(summary["by_split"].items()):
        lines.append(
            f"| `{split}` | `{row['files']}` | `{row['bytes']}` | `{row['known_queries']}` | `{row['known_positive']}` |"
        )
    if shard_plan is not None:
        lines.extend(
            [
                "",
                "## descriptiongenerate Shard",
                "",
                f"- Output dir: `{shard_plan['output_dir']}`",
                f"- Preset: `{shard_plan['preset']}`",
                f"- Chunk rows: `{shard_plan['chunk_rows']}`",
                "",
                "| Split | Rows | Positives | Negatives | Hard negatives |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for split, stats in sorted(shard_plan["stats_by_split"].items()):
            lines.append(
                f"| `{split}` | `{stats['row_count']}` | `{stats['positive_count']}` | "
                f"`{stats['negative_count']}` | `{stats['hard_negative_count']}` |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--root", type=Path, required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--report", type=Path, required=True)
    manifest_parser.add_argument("--inspect-counts", action="store_true")
    shard_parser = subparsers.add_parser("shards")
    shard_parser.add_argument("--manifest", type=Path, required=True)
    shard_parser.add_argument("--output-dir", type=Path, required=True)
    shard_parser.add_argument("--preset", default="smoke", choices=sorted(PRESET_ROW_LIMITS))
    shard_parser.add_argument("--chunk-rows", type=int, default=500_000)
    shard_parser.add_argument("--report", type=Path, required=True)
    shard_parser.add_argument("--dense-asset-limit", type=int, default=192)
    shard_parser.add_argument("--dense-pair-limit", type=int, default=4096)
    shard_parser.add_argument("--dense-samples-per-pair", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "manifest":
        manifest = build_rtstpf_paper_dataset_v3_manifest(args.root, inspect_counts=args.inspect_counts)
        write_manifest(args.output, manifest)
        write_design_report(args.report, manifest)
        return
    if args.cmd == "shards":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        dense_config = DenseSourceConfig(
            asset_limit_per_source=int(args.dense_asset_limit),
            pair_limit_per_source=int(args.dense_pair_limit),
            samples_per_pair=int(args.dense_samples_per_pair),
        )
        shard_plan = build_rtstpf_paper_dataset_v3_shards(
            args.manifest,
            output_dir=args.output_dir,
            preset=args.preset,
            chunk_rows=args.chunk_rows,
            dense_config=dense_config,
        )
        write_design_report(args.report, manifest, shard_plan=shard_plan)
        return
    raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
