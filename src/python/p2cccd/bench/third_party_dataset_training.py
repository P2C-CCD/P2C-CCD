from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from itertools import combinations
from pathlib import Path
from typing import Sequence

from p2cccd.contracts import ProxyType
from p2cccd.data import GeneratedDataset, proposal_row_from_oracle_trace, write_npz_shard, default_metadata
from p2cccd.data.oracle import evaluate_swept_sphere_oracle
from p2cccd.data.response import proxy_mass_from_radius
from p2cccd.data.samplers import MotionDiscPairSample, PairFamily
from p2cccd.datasets.cad import BetterSTEPAdapter, Fusion360GalleryAdapter
from p2cccd.datasets.cad.contracts import CadAssemblyMotionSample, StepNativeAsset
from p2cccd.datasets.objects import ShapeNetAdapter
from p2cccd.datasets.objects.contracts import ObjectMotionSample
from p2cccd.proposal.stpf_model import STPFModelPreset
from p2cccd.proposal.training import STPFTrainingConfig, evaluate_stpf_model
from p2cccd.proposal.training_runner import STPFTrainingRunConfig, STPFTrainingRunResult, run_stpf_training


FUSION360_ASSEMBLY_SOURCE_URL = (
    "https://fusion-360-gallery-dataset.s3-us-west-2.amazonaws.com/assembly/a1.0.0/a1.0.0_00.7z"
)
BETTER_STEP_FRDR_URL = "https://www.frdr-dfdr.ca/repo/dataset/d54b95e0-bc14-4236-b50b-922e5bf4ba7d"
SHAPENET_HF_URL = "https://huggingface.co/datasets/ShapeNet/ShapeNetCore"


def _default_training() -> STPFTrainingConfig:
    return STPFTrainingConfig(
        epochs=4,
        batch_size=2048,
        learning_rate=1.0e-3,
        seed=424242,
        device="cuda",
        validation_fraction=0.0,
        model_preset=STPFModelPreset.MEDIUM_MLP,
    )


@dataclass(frozen=True, slots=True)
class ThirdPartyTrainingConfig:
    source_name: str
    root: str
    run_name: str
    sample_limit: int = 1000
    train_fraction: float = 0.8
    seed: int = 424242
    training: STPFTrainingConfig = _default_training()
    shard_root: str = "src/datasets/training/third_party/shards"
    training_output_dir: str = "src/outputs/stpf_training"
    benchmark_output_dir: str = "src/benchmark"


@dataclass(frozen=True, slots=True)
class ThirdPartyTrainingResult:
    config: ThirdPartyTrainingConfig
    dataset: GeneratedDataset
    train_dataset: GeneratedDataset
    eval_dataset: GeneratedDataset
    training_run: STPFTrainingRunResult
    source_stats: dict[str, object]
    base_eval_metrics: object
    shard_manifest_path: Path
    report_path: Path
    summary_json_path: Path


def _stable_u32(token: str) -> int:
    return int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16)


def _radius_from_diagonal(diagonal: float) -> float:
    return max(0.05, 0.5 * max(float(diagonal), 0.1))


def _make_generated_dataset(samples: Sequence[MotionDiscPairSample], split_names: tuple[str, ...]) -> GeneratedDataset:
    sample_list = list(samples)
    traces = [evaluate_swept_sphere_oracle(sample) for sample in sample_list]
    rows = [proposal_row_from_oracle_trace(sample, trace) for sample, trace in zip(sample_list, traces)]
    return GeneratedDataset(rows=rows, samples=sample_list, traces=traces, split_names=split_names)


def _sample_from_fusion_motion(motion: CadAssemblyMotionSample, index: int) -> MotionDiscPairSample:
    asset_a = motion.pair.asset_a
    asset_b = motion.pair.asset_b
    radius_a = _radius_from_diagonal(asset_a.stats.diagonal)
    radius_b = _radius_from_diagonal(asset_b.stats.diagonal)
    return MotionDiscPairSample(
        sample_id=index + 1,
        query_id=8_100_000 + index + 1,
        candidate_id=8_200_000 + index + 1,
        split="fusion360_assembly",
        family=PairFamily.MESH_PAIR,
        object_a_id=400_000 + (_stable_u32(asset_a.asset_id) % 500_000),
        patch_a_id=1 + (_stable_u32(str(asset_a.asset_path)) % max(1, asset_a.stats.face_count)),
        object_b_id=400_000 + (_stable_u32(asset_b.asset_id) % 500_000),
        patch_b_id=1 + (_stable_u32(str(asset_b.asset_path)) % max(1, asset_b.stats.face_count)),
        slab_id=index % 8,
        center_a_t0=motion.center_a_t0,
        center_a_t1=motion.center_a_t1,
        center_b_t0=motion.center_b_t0,
        center_b_t1=motion.center_b_t1,
        radius_a=radius_a,
        radius_b=radius_b,
        proxy_type_a=ProxyType.SWEPT_AABB,
        proxy_type_b=ProxyType.SWEPT_AABB,
        hardness=max(0.0, min(1.0, float(motion.pair.hardness_score))),
        ood=False,
        mass_a=proxy_mass_from_radius(radius_a),
        mass_b=proxy_mass_from_radius(radius_b),
        restitution=1.0,
    )


def dataset_from_fusion360(config: ThirdPartyTrainingConfig) -> tuple[GeneratedDataset, dict[str, object]]:
    adapter = Fusion360GalleryAdapter(Path(config.root))
    sequences = adapter.list_sequences()
    motions = adapter.generate_assembly_motion_samples(limit=config.sample_limit)
    samples = [_sample_from_fusion_motion(motion, index) for index, motion in enumerate(motions)]
    stats = {
        "source_url": FUSION360_ASSEMBLY_SOURCE_URL,
        "sequence_count": len(sequences),
        "motion_sample_count": len(motions),
        "mesh_asset_count": sum(len(sequence.assets) for sequence in sequences),
        "root": str(Path(config.root)),
    }
    return _make_generated_dataset(samples, ("fusion360_assembly",)), stats


def _sample_from_step_pair(asset_a: StepNativeAsset, asset_b: StepNativeAsset, index: int) -> MotionDiscPairSample:
    entity_a = max(1, asset_a.entity_count)
    entity_b = max(1, asset_b.entity_count)
    radius_a = max(0.05, min(2.0, 0.05 + math.log1p(max(1, asset_a.file_size_bytes)) / 16.0))
    radius_b = max(0.05, min(2.0, 0.05 + math.log1p(max(1, asset_b.file_size_bytes)) / 16.0))
    hardness = math.exp(-abs(math.log(entity_a / entity_b)))
    start_gap = radius_a + radius_b + 0.25 + 0.05 * (index % 7)
    end_gap = radius_a + radius_b - 0.02 * (1 + (index % 3))
    lateral = 0.01 * (index % 11)
    return MotionDiscPairSample(
        sample_id=index + 1,
        query_id=8_300_000 + index + 1,
        candidate_id=8_400_000 + index + 1,
        split="better_step_native_proxy",
        family=PairFamily.MESH_PAIR,
        object_a_id=500_000 + (_stable_u32(asset_a.asset_id) % 400_000),
        patch_a_id=1,
        object_b_id=500_000 + (_stable_u32(asset_b.asset_id) % 400_000),
        patch_b_id=1,
        slab_id=index % 8,
        center_a_t0=(0.0, lateral, 0.0),
        center_a_t1=(0.2, lateral, 0.0),
        center_b_t0=(start_gap, lateral, 0.0),
        center_b_t1=(end_gap, lateral, 0.0),
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


def dataset_from_better_step(config: ThirdPartyTrainingConfig) -> tuple[GeneratedDataset, dict[str, object]]:
    adapter = BetterSTEPAdapter(Path(config.root))
    assets = adapter.list_assets(limit=max(2, config.sample_limit + 1))
    pairs = list(combinations(assets, 2))[: config.sample_limit]
    samples = [_sample_from_step_pair(asset_a, asset_b, index) for index, (asset_a, asset_b) in enumerate(pairs)]
    stats = {
        "source_url": BETTER_STEP_FRDR_URL,
        "asset_count": len(assets),
        "pair_count": len(pairs),
        "root": str(Path(config.root)),
        "note": "STEP-native metadata proxy training; mesh-level training needs a STEP meshing backend or Better STEP HDF5 geometry reader.",
    }
    return _make_generated_dataset(samples, ("better_step_native_proxy",)), stats


def _sample_from_object_motion(motion: ObjectMotionSample, index: int) -> MotionDiscPairSample:
    asset_a = motion.asset_a
    asset_b = motion.asset_b
    radius_a = _radius_from_diagonal(asset_a.stats.diagonal)
    radius_b = _radius_from_diagonal(asset_b.stats.diagonal)
    hardness = max(0.0, min(1.0, 0.25 + 0.5 * max(asset_a.dirty_score, asset_b.dirty_score)))
    return MotionDiscPairSample(
        sample_id=index + 1,
        query_id=8_500_000 + index + 1,
        candidate_id=8_600_000 + index + 1,
        split="shapenet_ood_object_pair",
        family=PairFamily.MESH_PAIR,
        object_a_id=600_000 + (_stable_u32(asset_a.object_id) % 300_000),
        patch_a_id=1,
        object_b_id=600_000 + (_stable_u32(asset_b.object_id) % 300_000),
        patch_b_id=1,
        slab_id=index % 8,
        center_a_t0=motion.center_a_t0,
        center_a_t1=motion.center_a_t1,
        center_b_t0=motion.center_b_t0,
        center_b_t1=motion.center_b_t1,
        radius_a=radius_a,
        radius_b=radius_b,
        proxy_type_a=ProxyType.SWEPT_AABB,
        proxy_type_b=ProxyType.SWEPT_AABB,
        hardness=hardness,
        ood=True,
        mass_a=proxy_mass_from_radius(radius_a),
        mass_b=proxy_mass_from_radius(radius_b),
        restitution=1.0,
    )


def dataset_from_shapenet(config: ThirdPartyTrainingConfig) -> tuple[GeneratedDataset, dict[str, object]]:
    adapter = ShapeNetAdapter(Path(config.root))
    assets = adapter.list_assets(limit=max(2, config.sample_limit + 1))
    motions = adapter.generate_ood_subset_samples(assets=assets, limit=config.sample_limit)
    samples = [_sample_from_object_motion(motion, index) for index, motion in enumerate(motions)]
    stats = {
        "source_url": SHAPENET_HF_URL,
        "asset_count": len(assets),
        "motion_sample_count": len(motions),
        "root": str(Path(config.root)),
    }
    return _make_generated_dataset(samples, ("shapenet_ood_object_pair",)), stats


def _split_dataset(dataset: GeneratedDataset, train_fraction: float, seed: int) -> tuple[GeneratedDataset, GeneratedDataset]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    indices = list(range(len(dataset.samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_count = max(1, min(len(indices) - 1, int(round(len(indices) * train_fraction))))
    train_ids = set(indices[:train_count])

    def subset(selected: set[int]) -> GeneratedDataset:
        rows = [row for index, row in enumerate(dataset.rows) if index in selected]
        samples = [sample for index, sample in enumerate(dataset.samples) if index in selected]
        traces = [trace for index, trace in enumerate(dataset.traces) if index in selected]
        return GeneratedDataset(rows=rows, samples=samples, traces=traces, split_names=dataset.split_names)

    return subset(train_ids), subset(set(indices[train_count:]))


def _dataset_for_source(config: ThirdPartyTrainingConfig) -> tuple[GeneratedDataset, dict[str, object]]:
    key = config.source_name.lower().replace("-", "_")
    if key in {"fusion360", "fusion_360", "fusion_360_gallery"}:
        return dataset_from_fusion360(config)
    if key in {"better_step", "betterstep", "step"}:
        return dataset_from_better_step(config)
    if key in {"shapenet", "shapenetcore"}:
        return dataset_from_shapenet(config)
    raise ValueError(f"unsupported third-party training source: {config.source_name}")


def run_third_party_training(config: ThirdPartyTrainingConfig) -> ThirdPartyTrainingResult:
    dataset, source_stats = _dataset_for_source(config)
    if len(dataset.rows) < 2:
        raise ValueError(f"{config.source_name} produced fewer than two training rows")
    train_dataset, eval_dataset = _split_dataset(dataset, config.train_fraction, config.seed)

    shard_dir = Path(config.shard_root) / config.run_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_npz_shard(
        shard_dir / "train.npz",
        train_dataset,
        metadata={**default_metadata(train_dataset, seed=config.seed, source=config.source_name), "dataset_role": "train"},
    )
    write_npz_shard(
        shard_dir / "eval.npz",
        eval_dataset,
        metadata={**default_metadata(eval_dataset, seed=config.seed + 1, source=config.source_name), "dataset_role": "eval"},
    )

    training_run = run_stpf_training(
        train_dataset.rows,
        STPFTrainingRunConfig(
            training=config.training,
            output_dir=config.training_output_dir,
            run_name=config.run_name,
        ),
        validation_rows=eval_dataset.rows,
    )
    base_eval_metrics = evaluate_stpf_model(
        training_run.result.model,
        eval_dataset.rows,
        config.training,
        epoch=config.training.epochs,
        split=f"{config.source_name}_eval",
    )

    manifest = {
        "config": asdict(config),
        "source_stats": source_stats,
        "train_query_count": len(train_dataset.samples),
        "eval_query_count": len(eval_dataset.samples),
        "train_row_count": len(train_dataset.rows),
        "eval_row_count": len(eval_dataset.rows),
        "checkpoint_path": str(training_run.artifacts.model_state_path),
        "final_train_loss": training_run.final_train_loss,
        "final_validation_loss": training_run.final_validation_loss,
        "base_eval_metrics": asdict(base_eval_metrics),
    }
    shard_manifest_path = shard_dir / "manifest.json"
    shard_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    output_root = Path(config.benchmark_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_root / f"{config.run_name}_summary.json"
    report_path = output_root / f"{config.run_name}_report.md"
    summary_json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, manifest)
    return ThirdPartyTrainingResult(
        config=config,
        dataset=dataset,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_run=training_run,
        source_stats=source_stats,
        base_eval_metrics=base_eval_metrics,
        shard_manifest_path=shard_manifest_path,
        report_path=report_path,
        summary_json_path=summary_json_path,
    )


def _write_report(path: Path, manifest: dict[str, object]) -> None:
    source_stats = manifest["source_stats"]
    config = manifest["config"]
    metrics = manifest["base_eval_metrics"]
    assert isinstance(source_stats, dict)
    assert isinstance(config, dict)
    assert isinstance(metrics, dict)
    lines = [
        f"# {config['source_name']} third-party dataset STPF training report",
        "",
        "## Source",
        "",
        f"- root: `{config['root']}`",
        f"- source url: `{source_stats.get('source_url', 'n/a')}`",
        f"- sample limit: `{config['sample_limit']}`",
        f"- source stats: `{json.dumps(source_stats, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Training",
        "",
        f"- train queries: `{manifest['train_query_count']}`",
        f"- eval queries: `{manifest['eval_query_count']}`",
        f"- checkpoint: `{manifest['checkpoint_path']}`",
        f"- final train loss: `{float(manifest['final_train_loss']):.6f}`",
        f"- final validation loss: `{float(manifest['final_validation_loss']):.6f}`",
        f"- eval interval top1 recall: `{float(metrics['interval_top1_recall']):.4f}`",
        f"- eval family top2 recall: `{float(metrics['family_top2_recall']):.4f}`",
        f"- eval estimated exact work reduction: `{float(metrics['estimated_exact_work_reduction']):.4f}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "BETTER_STEP_FRDR_URL",
    "FUSION360_ASSEMBLY_SOURCE_URL",
    "SHAPENET_HF_URL",
    "ThirdPartyTrainingConfig",
    "ThirdPartyTrainingResult",
    "dataset_from_better_step",
    "dataset_from_fusion360",
    "dataset_from_shapenet",
    "run_third_party_training",
]
