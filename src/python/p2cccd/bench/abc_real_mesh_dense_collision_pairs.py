from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import random
from pathlib import Path

from p2cccd.data import default_metadata, write_npz_shard
from p2cccd.datasets.cad import ABCDatasetAdapter
from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model
from p2cccd.proposal.training import STPFTrainingConfig
from p2cccd.proposal.training_runner import STPFTrainingRunConfig, STPFTrainingRunResult, run_stpf_training

from .high_density_mesh_training_benchmark import (
    MeshDensityAsset,
    MeshDensityPair,
    _asset_from_cad,
    _dataset_from_samples,
    _make_pairs,
    _sample_from_pair,
    _scale_workload_costs,
)
from .trained_stpf_high_density import (
    HighDensityMethodMetrics,
    HighDensitySTPFConfig,
    HighDensitySTPFWorkload,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
)


def _default_high_density() -> HighDensitySTPFConfig:
    return HighDensitySTPFConfig(
        slab_count=16,
        patches_per_object=12,
        representative_attempt_limit=3,
        uncertainty_fallback_threshold=0.75,
        narrow_interval_min_cost_scale=0.18,
        interval_miss_penalty_scale=0.22,
        full_exact_cost_scale=1.0,
    )


def _default_training() -> STPFTrainingConfig:
    return STPFTrainingConfig(
        epochs=6,
        batch_size=8192,
        learning_rate=8.0e-4,
        seed=424242,
        device="cuda",
        validation_fraction=0.0,
        model_preset=STPFModelPreset.MEDIUM_MLP,
    )


@dataclass(frozen=True, slots=True)
class ABCRealMeshDenseCollisionPairConfig:
    abc_root: str = "src/datasets/abc_official"
    min_faces_per_mesh: int = 10_000
    max_faces_per_mesh: int = 20_000
    asset_limit: int = 20
    pair_limit: int = 24
    samples_per_pair: int = 12
    train_fraction: float = 0.75
    seed: int = 424242
    high_density: HighDensitySTPFConfig = field(default_factory=_default_high_density)
    training: STPFTrainingConfig = field(default_factory=_default_training)
    run_name: str = "abc_real_mesh_dense_collision_pairs_run_id"
    shard_root: str = "src/datasets/training/abc_real_mesh_dense_collision_pairs/shards"
    training_output_dir: str = "src/outputs/stpf_training"
    benchmark_output_dir: str = "src/benchmark"


@dataclass(frozen=True, slots=True)
class ABCRealMeshDenseCollisionPairResult:
    config: ABCRealMeshDenseCollisionPairConfig
    assets: tuple[MeshDensityAsset, ...]
    train_pairs: tuple[MeshDensityPair, ...]
    eval_pairs: tuple[MeshDensityPair, ...]
    train_workload: HighDensitySTPFWorkload
    eval_workload: HighDensitySTPFWorkload
    training_run: STPFTrainingRunResult
    no_proposal: HighDensityMethodMetrics
    random_stpf: HighDensityMethodMetrics
    trained_stpf: HighDensityMethodMetrics
    report_path: Path
    summary_json_path: Path


def _load_large_face_abc_assets(config: ABCRealMeshDenseCollisionPairConfig) -> tuple[MeshDensityAsset, ...]:
    adapter = ABCDatasetAdapter(Path(config.abc_root))
    assets = [
        _asset_from_cad("ABC real mesh large-face", asset)
        for asset in adapter.list_assets(limit=None)
        if config.min_faces_per_mesh <= int(asset.stats.face_count) <= config.max_faces_per_mesh
    ]
    assets.sort(key=lambda asset: (-asset.face_count, -asset.vertex_count, asset.asset_id))
    return tuple(assets[: config.asset_limit])


def _split_pairs(
    pairs: tuple[MeshDensityPair, ...],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[tuple[MeshDensityPair, ...], tuple[MeshDensityPair, ...]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_fraction))))
    return tuple(shuffled[:train_count]), tuple(shuffled[train_count:])


def _build_dataset_and_costs(
    pairs: tuple[MeshDensityPair, ...],
    *,
    first_sample_id: int,
    samples_per_pair: int,
) -> tuple[object, dict[int, float]]:
    samples = []
    cost_scale_by_query_id: dict[int, float] = {}
    sample_id = first_sample_id
    for pair_index, pair in enumerate(pairs):
        for local_index in range(samples_per_pair):
            variant_index = pair_index * samples_per_pair + local_index
            sample = _sample_from_pair(pair, sample_id=sample_id, variant_index=variant_index)
            samples.append(sample)
            cost_scale_by_query_id[sample.query_id] = pair.cost_scale
            sample_id += 1
    return _dataset_from_samples(samples), cost_scale_by_query_id


def _collision_query_count(workload: HighDensitySTPFWorkload) -> int:
    return sum(1 for trace in workload.traces_by_query_id.values() if trace.collided)


def _collision_candidate_count(workload: HighDensitySTPFWorkload) -> int:
    return sum(1 for info in workload.candidate_infos.values() if info.slab_overlap_contact)


def _reduction(trained: HighDensityMethodMetrics, baseline: HighDensityMethodMetrics) -> float:
    return 1.0 - trained.exact_work_units / max(1.0e-9, baseline.exact_work_units)


def run_abc_real_mesh_dense_collision_pair_benchmark(
    config: ABCRealMeshDenseCollisionPairConfig | None = None,
) -> ABCRealMeshDenseCollisionPairResult:
    cfg = config or ABCRealMeshDenseCollisionPairConfig()
    if cfg.asset_limit < 2:
        raise ValueError("asset_limit must be at least 2")
    if cfg.pair_limit < 2:
        raise ValueError("pair_limit must be at least 2")
    if cfg.samples_per_pair < 1:
        raise ValueError("samples_per_pair must be positive")

    assets = _load_large_face_abc_assets(cfg)
    if len(assets) < 2:
        raise ValueError("ABC large-face selection produced fewer than two assets")
    pairs = _make_pairs(assets, cfg.pair_limit)
    if len(pairs) < 2:
        raise ValueError("ABC large-face pair generation produced fewer than two pairs")
    train_pairs, eval_pairs = _split_pairs(pairs, train_fraction=cfg.train_fraction, seed=cfg.seed)

    train_dataset, train_costs = _build_dataset_and_costs(
        train_pairs,
        first_sample_id=9_100_001,
        samples_per_pair=cfg.samples_per_pair,
    )
    eval_dataset, eval_costs = _build_dataset_and_costs(
        eval_pairs,
        first_sample_id=9_500_001,
        samples_per_pair=cfg.samples_per_pair,
    )
    train_workload = _scale_workload_costs(
        build_high_density_stpf_workload(
            train_dataset,
            cfg.high_density,
            name=f"{cfg.run_name}_train",
        ),
        train_costs,
    )
    eval_workload = _scale_workload_costs(
        build_high_density_stpf_workload(
            eval_dataset,
            cfg.high_density,
            name=f"{cfg.run_name}_eval",
        ),
        eval_costs,
    )

    shard_dir = Path(cfg.shard_root) / cfg.run_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_npz_shard(
        shard_dir / "train.npz",
        train_dataset,
        metadata={
            **default_metadata(train_dataset, seed=cfg.seed, source="abc_real_mesh_dense_collision_pairs"),
            "dataset_role": "train",
        },
    )
    write_npz_shard(
        shard_dir / "eval.npz",
        eval_dataset,
        metadata={
            **default_metadata(eval_dataset, seed=cfg.seed + 1, source="abc_real_mesh_dense_collision_pairs"),
            "dataset_role": "eval",
        },
    )

    training_run = run_stpf_training(
        train_workload.rows,
        STPFTrainingRunConfig(
            training=cfg.training,
            output_dir=cfg.training_output_dir,
            run_name=cfg.run_name,
        ),
        validation_rows=eval_workload.rows,
    )
    trained_model = training_run.result.model
    trained_model.to(cfg.training.device)
    trained_model.eval()
    random_model = build_stpf_model(cfg.training.model_preset)
    random_model.to(cfg.training.device)
    random_model.eval()

    no_proposal = benchmark_no_proposal_on_high_density_workload(eval_workload)
    random_stpf = benchmark_stpf_on_high_density_workload(
        eval_workload,
        model=random_model,
        device=cfg.training.device,
        proposal_batch_size=cfg.training.batch_size,
        method_name="ABCRealMeshDense-RTSTPFExact-Random",
    )
    trained_stpf = benchmark_stpf_on_high_density_workload(
        eval_workload,
        model=trained_model,
        device=cfg.training.device,
        proposal_batch_size=cfg.training.batch_size,
        method_name="ABCRealMeshDense-RTSTPFExact-Trained",
    )

    output_root = Path(cfg.benchmark_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / f"{cfg.run_name}.md"
    summary_json_path = output_root / f"{cfg.run_name}.json"
    result = ABCRealMeshDenseCollisionPairResult(
        config=cfg,
        assets=assets,
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        train_workload=train_workload,
        eval_workload=eval_workload,
        training_run=training_run,
        no_proposal=no_proposal,
        random_stpf=random_stpf,
        trained_stpf=trained_stpf,
        report_path=report_path,
        summary_json_path=summary_json_path,
    )
    _write_summary_json(summary_json_path, result, shard_dir)
    _write_report(report_path, result, shard_dir)
    return result


def _metric_dict(metric: HighDensityMethodMetrics) -> dict[str, object]:
    return asdict(metric)


def _write_summary_json(path: Path, result: ABCRealMeshDenseCollisionPairResult, shard_dir: Path) -> None:
    payload = {
        "config": asdict(result.config),
        "asset_count": len(result.assets),
        "asset_face_counts": [asset.face_count for asset in result.assets],
        "train_pair_count": len(result.train_pairs),
        "eval_pair_count": len(result.eval_pairs),
        "train_query_count": result.train_workload.query_count,
        "eval_query_count": result.eval_workload.query_count,
        "train_candidate_count": result.train_workload.candidate_count,
        "eval_candidate_count": result.eval_workload.candidate_count,
        "eval_collision_query_count": _collision_query_count(result.eval_workload),
        "eval_collision_candidate_count": _collision_candidate_count(result.eval_workload),
        "checkpoint_path": str(result.training_run.artifacts.model_state_path),
        "shard_dir": str(shard_dir),
        "final_train_loss": result.training_run.final_train_loss,
        "final_validation_loss": result.training_run.final_validation_loss,
        "no_proposal": _metric_dict(result.no_proposal),
        "random_stpf": _metric_dict(result.random_stpf),
        "trained_stpf": _metric_dict(result.trained_stpf),
        "trained_exact_work_reduction_vs_no_proposal": _reduction(result.trained_stpf, result.no_proposal),
        "notes": [
            "This is a dense patch/candidate benchmark built from real ABC large-face meshes.",
            "The current C++ mesh exact builder does not crop by patch ids, so exact work is primitive-weighted and calibrated by real mesh face counts, not a true patch-submesh exact wall-time result.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _write_report(path: Path, result: ABCRealMeshDenseCollisionPairResult, shard_dir: Path) -> None:
    face_counts = sorted(asset.face_count for asset in result.assets)
    lines = [
        "# ABC Real Mesh Dense Collision-Pair Benchmark",
        "",
        "## Protocol",
        "",
        "- Objective: descriptionperform `ABC real mesh exact large-face small-batch`,  whole-mesh single-candidate case description patch/candidate collision pairs. ",
        "- description: official ABC real mesh, description `10000~20000 faces/mesh` description CAD mesh. ",
        f"- Dense candidate description: `{result.config.high_density.slab_count} slabs x {result.config.high_density.patches_per_object} patches x {result.config.high_density.patches_per_object} patches = {result.eval_workload.avg_candidates_per_query:.0f} candidates/query`. ",
        "- RTSTPFExact constraint: uses only learned STPF, descriptionuse dummy policy. ",
        "- description: current C++ mesh exact builder descriptionby patch id description; descriptionreport exact work isbyreal mesh face count calibrated dense candidate work, is not final patch-submesh exact wall time. ",
        "",
        "## description",
        "",
        f"- ABC root: `{result.config.abc_root}`",
        f"- Selected assets: `{len(result.assets)}`",
        f"- Face count min/median/max: `{face_counts[0]}` / `{face_counts[len(face_counts) // 2]}` / `{face_counts[-1]}`",
        f"- Train pairs / eval pairs: `{len(result.train_pairs)}` / `{len(result.eval_pairs)}`",
        f"- Train queries / eval queries: `{result.train_workload.query_count}` / `{result.eval_workload.query_count}`",
        f"- Train candidates / eval candidates: `{result.train_workload.candidate_count}` / `{result.eval_workload.candidate_count}`",
        f"- Eval collision queries: `{_collision_query_count(result.eval_workload)}`",
        f"- Eval collision-overlap candidate pairs: `{_collision_candidate_count(result.eval_workload)}`",
        f"- Dataset shard dir: `{shard_dir}`",
        "",
        "## description",
        "",
        f"- Checkpoint: `{result.training_run.artifacts.model_state_path}`",
        f"- Model preset: `{result.config.training.model_preset}`",
        f"- Device: `{result.config.training.device}`",
        f"- Epochs: `{result.config.training.epochs}`",
        f"- Batch size: `{result.config.training.batch_size}`",
        f"- Final train loss: `{result.training_run.final_train_loss:.6f}`",
        f"- Final validation loss: `{result.training_run.final_validation_loss:.6f}`",
        "",
        "## Benchmark",
        "",
        "| Method | Candidates/query | Exact calls | Fallback calls | Exact work | Proposal ms | Scheduling ms | Total ms | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in (result.no_proposal, result.random_stpf, result.trained_stpf):
        lines.append(
            f"| `{metric.method_name}` | `{_fmt(metric.avg_candidates_per_query, 1)}` | "
            f"`{metric.exact_call_count}` | `{metric.fallback_call_count}` | "
            f"`{_fmt(metric.exact_work_units, 1)}` | `{_fmt(metric.proposal_wall_ms, 3)}` | "
            f"`{_fmt(metric.scheduling_wall_ms, 3)}` | `{_fmt(metric.total_wall_ms, 3)}` | `{metric.fn_count}` |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- this run eval constructdescription `{result.eval_workload.candidate_count}`  patch/candidate pairs, descriptionisdescription small-batch  `4`  work items. ",
            f"- Trained STPF description NoProposal  exact work reduction as `{_pct(_reduction(result.trained_stpf, result.no_proposal))}`. ",
            f"- Trained STPF final FN as `{result.trained_stpf.fn_count}`. ",
            "- this case descriptionwithasdescriptionin `C/K description, candidate false-positive high`  ABC real-mesh dense evidence. ",
            "- ifdescriptionthisdescriptionlevelasdescriptionreal wall-time description, underdescription C++ `BuildMeshExactCertificateQuery` support `patch_a_id/patch_b_id` description/description, ordescriptionin RT broad phase descriptionconnectOutput BVH leaf/triangle-cluster pair afterconnect submesh exact certificate. ",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "ABCRealMeshDenseCollisionPairConfig",
    "ABCRealMeshDenseCollisionPairResult",
    "run_abc_real_mesh_dense_collision_pair_benchmark",
]
