from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
import json
import math
import random
from pathlib import Path

from p2cccd.contracts import ProxyType
from p2cccd.data import GeneratedDataset, default_metadata, proposal_row_from_oracle_trace, write_npz_shard
from p2cccd.data.oracle import evaluate_swept_sphere_oracle
from p2cccd.data.samplers import MotionDiscPairSample, PairFamily, sample_path_length
from p2cccd.bench.trained_stpf_high_density import (
    HighDensitySTPFConfig,
    build_high_density_stpf_workload,
    workload_to_shard_dataset,
)


RUN_NAME = "common_modeling_high_density_scenarios_run_id"
DEFAULT_DATASET_ROOT = Path("src/datasets/training/common_modeling_high_density/shards") / RUN_NAME
DEFAULT_BENCHMARK_ROOT = Path("src/datasets/benchmark/physics_dense_contact_bench") / RUN_NAME
DEFAULT_REPORT_PATH = Path("src/benchmark") / f"{RUN_NAME}.md"
DEFAULT_JSON_PATH = Path("src/benchmark") / f"{RUN_NAME}.json"
DEFAULT_CSV_PATH = Path("src/benchmark") / f"{RUN_NAME}_summary.csv"


SCENARIO_HIGH_SPEED = "high_speed_object_head_on"
SCENARIO_FLEX_WALL = "flexible_body_wall_impact"
SCENARIO_ROTATING_SURFACE = "complex_surface_rotating_intersection"
SCENARIO_THIN_SLOT = "thin_feature_slot_insertion"


@dataclass(frozen=True, slots=True)
class CommonModelingScenarioConfig:
    run_name: str = RUN_NAME
    seed: int = 428
    train_per_scenario: int = 80
    eval_per_scenario: int = 24
    slab_count: int = 24
    patches_per_object: int = 12
    dataset_root: Path = DEFAULT_DATASET_ROOT
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT
    report_path: Path = DEFAULT_REPORT_PATH
    json_path: Path = DEFAULT_JSON_PATH
    csv_path: Path = DEFAULT_CSV_PATH

    @property
    def candidates_per_query(self) -> int:
        return int(self.slab_count * self.patches_per_object * self.patches_per_object)


@dataclass(frozen=True, slots=True)
class ScenarioSummary:
    scenario: str
    role: str
    queries: int
    positives: int
    negatives: int
    dense_candidates: int
    candidates_per_query: int
    mean_path_length: float
    mean_speed_proxy: float
    mean_exact_cost: float
    mean_hardness: float


@dataclass(frozen=True, slots=True)
class CommonModelingScenarioResult:
    config: CommonModelingScenarioConfig
    dataset_root: Path
    benchmark_root: Path
    base_train_path: Path
    base_eval_path: Path
    dense_train_path: Path
    dense_eval_path: Path
    manifest_path: Path
    benchmark_manifest_path: Path
    summary_rows: tuple[ScenarioSummary, ...]


def _vec(x: float, y: float, z: float = 0.0) -> tuple[float, float, float]:
    return (float(x), float(y), float(z))


def _add(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _mul(value: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    return (value[0] * scalar, value[1] * scalar, value[2] * scalar)


def _proxy_mass(radius: float) -> float:
    r = max(1.0e-6, float(radius))
    return (4.0 / 3.0) * math.pi * r * r * r


def _sample(
    *,
    sample_id: int,
    scenario: str,
    object_a_id: int,
    patch_a_id: int,
    object_b_id: int,
    patch_b_id: int,
    center_a_t0: tuple[float, float, float],
    center_a_t1: tuple[float, float, float],
    center_b_t0: tuple[float, float, float],
    center_b_t1: tuple[float, float, float],
    radius_a: float,
    radius_b: float,
    hardness: float,
    proxy_type_a: ProxyType = ProxyType.SWEPT_AABB,
    proxy_type_b: ProxyType = ProxyType.SWEPT_AABB,
    ood: bool = False,
) -> MotionDiscPairSample:
    return MotionDiscPairSample(
        sample_id=int(sample_id),
        query_id=10_000_000 + int(sample_id),
        candidate_id=20_000_000 + int(sample_id),
        split=scenario,
        family=PairFamily.MESH_PAIR,
        object_a_id=int(object_a_id),
        patch_a_id=int(patch_a_id),
        object_b_id=int(object_b_id),
        patch_b_id=int(patch_b_id),
        slab_id=int(sample_id % 24),
        center_a_t0=center_a_t0,
        center_a_t1=center_a_t1,
        center_b_t0=center_b_t0,
        center_b_t1=center_b_t1,
        radius_a=float(radius_a),
        radius_b=float(radius_b),
        proxy_type_a=proxy_type_a,
        proxy_type_b=proxy_type_b,
        hardness=float(max(0.0, min(1.0, hardness))),
        ood=ood,
        mass_a=_proxy_mass(radius_a),
        mass_b=_proxy_mass(radius_b),
        restitution=0.88 if scenario == SCENARIO_FLEX_WALL else 0.96,
    )


def _make_high_speed_head_on(
    *,
    rng: random.Random,
    count: int,
    first_sample_id: int,
) -> list[MotionDiscPairSample]:
    samples: list[MotionDiscPairSample] = []
    for i in range(count):
        positive = (i % 4) != 0
        radius_a = rng.uniform(0.18, 0.38)
        radius_b = rng.uniform(0.18, 0.42)
        radius_sum = radius_a + radius_b
        speed = rng.uniform(8.0, 18.0)
        lateral = rng.uniform(-0.25, 0.25)
        if not positive:
            lateral = (1.08 + rng.uniform(0.03, 0.16)) * radius_sum * (1.0 if i % 2 == 0 else -1.0)
        a0 = _vec(-0.55 * speed, lateral, rng.uniform(-0.05, 0.05))
        a1 = _vec(0.55 * speed, lateral + rng.uniform(-0.04, 0.04), rng.uniform(-0.05, 0.05))
        b0 = _vec(0.55 * speed, -0.35 * lateral, rng.uniform(-0.05, 0.05))
        b1 = _vec(-0.55 * speed, -0.35 * lateral + rng.uniform(-0.04, 0.04), rng.uniform(-0.05, 0.05))
        samples.append(
            _sample(
                sample_id=first_sample_id + i,
                scenario=SCENARIO_HIGH_SPEED,
                object_a_id=101,
                patch_a_id=1 + (i % 19),
                object_b_id=102,
                patch_b_id=101 + (i % 23),
                center_a_t0=a0,
                center_a_t1=a1,
                center_b_t0=b0,
                center_b_t1=b1,
                radius_a=radius_a,
                radius_b=radius_b,
                hardness=0.86 if positive else 0.72,
                proxy_type_a=ProxyType.CAPSULE,
                proxy_type_b=ProxyType.CAPSULE,
            )
        )
    return samples


def _make_flexible_body_wall(
    *,
    rng: random.Random,
    count: int,
    first_sample_id: int,
) -> list[MotionDiscPairSample]:
    samples: list[MotionDiscPairSample] = []
    grid = max(1, int(math.ceil(math.sqrt(count))))
    for i in range(count):
        positive = (i % 5) != 0
        row = i // grid
        col = i % grid
        y = (col - 0.5 * grid) * 0.12 + rng.uniform(-0.018, 0.018)
        z = (row - 0.5 * grid) * 0.12 + rng.uniform(-0.018, 0.018)
        cloth_radius = rng.uniform(0.025, 0.055)
        wall_radius = rng.uniform(0.10, 0.18)
        radius_sum = cloth_radius + wall_radius
        start_x = -rng.uniform(1.0, 2.4)
        end_x = rng.uniform(0.02, 0.22) if positive else -(radius_sum + rng.uniform(0.015, 0.12))
        wall_center = _vec(0.0, y + rng.uniform(-0.035, 0.035), z + rng.uniform(-0.035, 0.035))
        samples.append(
            _sample(
                sample_id=first_sample_id + i,
                scenario=SCENARIO_FLEX_WALL,
                object_a_id=201,
                patch_a_id=1 + i,
                object_b_id=202,
                patch_b_id=10_000 + (row * grid + col),
                center_a_t0=_vec(start_x, y, z),
                center_a_t1=_vec(end_x, y + rng.uniform(-0.08, 0.08), z + rng.uniform(-0.08, 0.08)),
                center_b_t0=wall_center,
                center_b_t1=wall_center,
                radius_a=cloth_radius,
                radius_b=wall_radius,
                hardness=0.78 if positive else 0.62,
                proxy_type_a=ProxyType.CAPSULE,
                proxy_type_b=ProxyType.SWEPT_AABB,
                ood=True,
            )
        )
    return samples


def _make_rotating_surface_intersection(
    *,
    rng: random.Random,
    count: int,
    first_sample_id: int,
) -> list[MotionDiscPairSample]:
    samples: list[MotionDiscPairSample] = []
    for i in range(count):
        positive = (i % 4) != 1
        radius_a = rng.uniform(0.08, 0.18)
        radius_b = rng.uniform(0.08, 0.22)
        orbit = rng.uniform(0.55, 1.35)
        theta0 = rng.uniform(-math.pi, math.pi)
        angular_sweep = rng.uniform(1.2 * math.pi, 2.2 * math.pi)
        theta1 = theta0 + angular_sweep
        a0 = _vec(orbit * math.cos(theta0), orbit * math.sin(theta0), rng.uniform(-0.08, 0.08))
        a1 = _vec(orbit * math.cos(theta1), orbit * math.sin(theta1), rng.uniform(-0.08, 0.08))
        chord_mid = _mul(_add(a0, a1), 0.5)
        miss_gap = radius_a + radius_b + rng.uniform(0.03, 0.20)
        normal = _vec(-math.sin(theta0), math.cos(theta0), 0.0)
        b_center = chord_mid if positive else _add(chord_mid, _mul(normal, miss_gap))
        b_drift = _vec(rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), rng.uniform(-0.04, 0.04))
        samples.append(
            _sample(
                sample_id=first_sample_id + i,
                scenario=SCENARIO_ROTATING_SURFACE,
                object_a_id=301,
                patch_a_id=1 + (i % 31),
                object_b_id=302,
                patch_b_id=1001 + (i % 37),
                center_a_t0=a0,
                center_a_t1=a1,
                center_b_t0=b_center,
                center_b_t1=_add(b_center, b_drift),
                radius_a=radius_a,
                radius_b=radius_b,
                hardness=0.94 if positive else 0.78,
                proxy_type_a=ProxyType.SWEPT_AABB,
                proxy_type_b=ProxyType.SWEPT_AABB,
                ood=True,
            )
        )
    return samples


def _make_thin_feature_slot_insertion(
    *,
    rng: random.Random,
    count: int,
    first_sample_id: int,
) -> list[MotionDiscPairSample]:
    samples: list[MotionDiscPairSample] = []
    for i in range(count):
        positive = (i % 3) != 0
        tool_radius = rng.uniform(0.035, 0.075)
        lip_radius = rng.uniform(0.035, 0.065)
        clearance = rng.uniform(-0.035, 0.025) if positive else rng.uniform(0.04, 0.11)
        y = clearance + rng.uniform(-0.015, 0.015)
        z = rng.uniform(-0.05, 0.05)
        start_x = -rng.uniform(1.5, 3.0)
        end_x = rng.uniform(0.6, 1.2)
        lip_center = _vec(0.0, 0.0, z + rng.uniform(-0.012, 0.012))
        samples.append(
            _sample(
                sample_id=first_sample_id + i,
                scenario=SCENARIO_THIN_SLOT,
                object_a_id=401,
                patch_a_id=1 + (i % 41),
                object_b_id=402,
                patch_b_id=2001 + (i % 43),
                center_a_t0=_vec(start_x, y, z),
                center_a_t1=_vec(end_x, y + rng.uniform(-0.025, 0.025), z),
                center_b_t0=lip_center,
                center_b_t1=lip_center,
                radius_a=tool_radius,
                radius_b=lip_radius,
                hardness=0.90 if positive else 0.76,
                proxy_type_a=ProxyType.CAPSULE,
                proxy_type_b=ProxyType.CAPSULE,
            )
        )
    return samples


def _build_samples(*, cfg: CommonModelingScenarioConfig, role: str) -> list[MotionDiscPairSample]:
    count = cfg.train_per_scenario if role == "train" else cfg.eval_per_scenario
    seed_shift = 0 if role == "train" else 100_000
    rng = random.Random(cfg.seed + seed_shift)
    first = 7_000_000 if role == "train" else 8_000_000
    samples: list[MotionDiscPairSample] = []
    builders = (
        _make_high_speed_head_on,
        _make_flexible_body_wall,
        _make_rotating_surface_intersection,
        _make_thin_feature_slot_insertion,
    )
    for scenario_index, builder in enumerate(builders):
        samples.extend(builder(rng=rng, count=count, first_sample_id=first + scenario_index * 100_000))
    return samples


def _dataset_from_samples(samples: list[MotionDiscPairSample]) -> GeneratedDataset:
    traces = [evaluate_swept_sphere_oracle(sample) for sample in samples]
    rows = [proposal_row_from_oracle_trace(sample, trace) for sample, trace in zip(samples, traces)]
    split_names = tuple(
        scenario
        for scenario in (SCENARIO_HIGH_SPEED, SCENARIO_FLEX_WALL, SCENARIO_ROTATING_SURFACE, SCENARIO_THIN_SLOT)
        if any(sample.split == scenario for sample in samples)
    )
    return GeneratedDataset(rows=rows, samples=samples, traces=traces, split_names=split_names)


def _summaries(
    *,
    dataset: GeneratedDataset,
    role: str,
    candidates_per_query: int,
) -> list[ScenarioSummary]:
    rows: list[ScenarioSummary] = []
    for scenario in dataset.split_names:
        scenario_pairs = [
            (sample, trace)
            for sample, trace in zip(dataset.samples, dataset.traces)
            if sample.split == scenario
        ]
        if not scenario_pairs:
            continue
        queries = len(scenario_pairs)
        positives = sum(1 for _, trace in scenario_pairs if trace.collided)
        mean_path = sum(sample_path_length(sample) for sample, _ in scenario_pairs) / float(queries)
        mean_speed = mean_path
        mean_cost = sum(trace.exact_cost for _, trace in scenario_pairs) / float(queries)
        mean_hardness = sum(sample.hardness for sample, _ in scenario_pairs) / float(queries)
        rows.append(
            ScenarioSummary(
                scenario=scenario,
                role=role,
                queries=queries,
                positives=positives,
                negatives=queries - positives,
                dense_candidates=queries * candidates_per_query,
                candidates_per_query=candidates_per_query,
                mean_path_length=mean_path,
                mean_speed_proxy=mean_speed,
                mean_exact_cost=mean_cost,
                mean_hardness=mean_hardness,
            )
        )
    return rows


def _write_summary_csv(path: Path, rows: list[ScenarioSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            for key in ("mean_path_length", "mean_speed_proxy", "mean_exact_cost", "mean_hardness"):
                data[key] = f"{data[key]:.6f}"
            writer.writerow(data)


def _write_report(path: Path, result: CommonModelingScenarioResult) -> None:
    cfg = result.config
    lines = [
        "# Common Modeling High-Density / High-Speed Collision Dataset",
        "",
        f"Run label: `run_id`",
        "",
        "## 1. Objective",
        "",
        "constructdescription/descriptionsceneinhigh-density, highdescription CCD descriptionand benchmark dataset, used fordescription ABC/Fusion/Thingi10K descriptionphysicsdescriptionModel hard cases. ",
        "",
        "containsscene: ",
        "",
        "- `high_speed_object_head_on`: highdescription/description. ",
        "- `flexible_body_wall_impact`: description/patch highdescription. ",
        "- `complex_surface_rotating_intersection`: description patch description. ",
        "- `thin_feature_slot_insertion`: description, description, descriptioncontact. ",
        "",
        "## 2. description",
        "",
        f"- train per scenario: `{cfg.train_per_scenario}`",
        f"- eval per scenario: `{cfg.eval_per_scenario}`",
        f"- slab count: `{cfg.slab_count}`",
        f"- patches/object: `{cfg.patches_per_object}`",
        f"- candidates/query: `{cfg.candidates_per_query}`",
        f"- dataset root: `{result.dataset_root.as_posix()}`",
        f"- benchmark root: `{result.benchmark_root.as_posix()}`",
        "",
        "## 3. Output files",
        "",
        f"- base train: `{result.base_train_path.as_posix()}`",
        f"- base eval: `{result.base_eval_path.as_posix()}`",
        f"- dense train: `{result.dense_train_path.as_posix()}`",
        f"- dense eval: `{result.dense_eval_path.as_posix()}`",
        f"- manifest: `{result.manifest_path.as_posix()}`",
        f"- benchmark manifest: `{result.benchmark_manifest_path.as_posix()}`",
        f"- summary csv: `{cfg.csv_path.as_posix()}`",
        "",
        "## 4. scenestatistics",
        "",
        "| Scenario | Role | Queries | Positives | Negatives | Dense candidates | Mean speed proxy | Mean exact cost |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.summary_rows:
        lines.append(
            "| "
            f"`{row.scenario}` | `{row.role}` | `{row.queries}` | `{row.positives}` | `{row.negatives}` | "
            f"`{row.dense_candidates}` | `{row.mean_speed_proxy:.3f}` | `{row.mean_exact_cost:.3f}` |"
        )
    lines.extend(
        [
            "",
            "## 5. descriptionusedescription",
            "",
            "description shard anddescriptionhas STPF description, descriptionasdescription, descriptionwithdescription v3 afterdescription. ",
            "",
            "descriptionusedescription: ",
            "",
            "- highdescription: test STPF description TOI, description contact interval description. ",
            "- description: testsame wall/cloth group indescriptioncandidate scheduling. ",
            "- description: testhighdescription/description broad-phase candidate inflation. ",
            "- description: test narrow clearance, near miss hard negatives and uncertainty fallback. ",
            "",
            "## 6. correctnessdescription",
            "",
            "currentdatasetdescriptionuse analytic swept-sphere/capsule proxy oracle generatedescription, description STPF scheduling; final paper correctness description exact certificate / Tight-Inclusion / conservative fallback description. ",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_common_modeling_high_density_scenarios(
    cfg: CommonModelingScenarioConfig | None = None,
) -> CommonModelingScenarioResult:
    config = cfg or CommonModelingScenarioConfig()
    dataset_root = config.dataset_root
    benchmark_root = config.benchmark_root
    dataset_root.mkdir(parents=True, exist_ok=True)
    benchmark_root.mkdir(parents=True, exist_ok=True)

    train_dataset = _dataset_from_samples(_build_samples(cfg=config, role="train"))
    eval_dataset = _dataset_from_samples(_build_samples(cfg=config, role="eval"))
    dense_cfg = HighDensitySTPFConfig(
        slab_count=config.slab_count,
        patches_per_object=config.patches_per_object,
        representative_attempt_limit=3,
        uncertainty_fallback_threshold=0.75,
    )
    dense_train = build_high_density_stpf_workload(train_dataset, dense_cfg, name=f"{config.run_name}_dense_train")
    dense_eval = build_high_density_stpf_workload(eval_dataset, dense_cfg, name=f"{config.run_name}_dense_eval")

    base_train_path = dataset_root / "base_train.npz"
    base_eval_path = dataset_root / "base_eval.npz"
    dense_train_path = dataset_root / "dense_train.npz"
    dense_eval_path = dataset_root / "dense_eval.npz"

    write_npz_shard(
        base_train_path,
        train_dataset,
        metadata={
            **default_metadata(train_dataset, seed=config.seed, source=config.run_name),
            "dataset_role": "base_train",
        },
    )
    write_npz_shard(
        base_eval_path,
        eval_dataset,
        metadata={
            **default_metadata(eval_dataset, seed=config.seed + 100_000, source=config.run_name),
            "dataset_role": "base_eval",
        },
    )
    write_npz_shard(
        dense_train_path,
        workload_to_shard_dataset(dense_train),
        metadata={
            **default_metadata(workload_to_shard_dataset(dense_train), seed=config.seed, source=config.run_name),
            "dataset_role": "dense_train",
            "slab_count": config.slab_count,
            "patches_per_object": config.patches_per_object,
            "candidates_per_query": config.candidates_per_query,
        },
    )
    write_npz_shard(
        dense_eval_path,
        workload_to_shard_dataset(dense_eval),
        metadata={
            **default_metadata(workload_to_shard_dataset(dense_eval), seed=config.seed + 100_000, source=config.run_name),
            "dataset_role": "dense_eval",
            "slab_count": config.slab_count,
            "patches_per_object": config.patches_per_object,
            "candidates_per_query": config.candidates_per_query,
        },
    )

    summary_rows = tuple(
        _summaries(dataset=train_dataset, role="train", candidates_per_query=config.candidates_per_query)
        + _summaries(dataset=eval_dataset, role="eval", candidates_per_query=config.candidates_per_query)
    )
    _write_summary_csv(config.csv_path, list(summary_rows))

    manifest = {
        "run_name": config.run_name,
        "schema": "common_modeling_high_density_scenarios.v1",
        "config": {
            **asdict(config),
            "dataset_root": config.dataset_root.as_posix(),
            "benchmark_root": config.benchmark_root.as_posix(),
            "report_path": config.report_path.as_posix(),
            "json_path": config.json_path.as_posix(),
            "csv_path": config.csv_path.as_posix(),
        },
        "paths": {
            "base_train": base_train_path.as_posix(),
            "base_eval": base_eval_path.as_posix(),
            "dense_train": dense_train_path.as_posix(),
            "dense_eval": dense_eval_path.as_posix(),
            "summary_csv": config.csv_path.as_posix(),
        },
        "summary": [asdict(row) for row in summary_rows],
    }
    manifest_path = dataset_root / "manifest.json"
    benchmark_manifest_path = benchmark_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    benchmark_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    config.json_path.parent.mkdir(parents=True, exist_ok=True)
    config.json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    result = CommonModelingScenarioResult(
        config=config,
        dataset_root=dataset_root,
        benchmark_root=benchmark_root,
        base_train_path=base_train_path,
        base_eval_path=base_eval_path,
        dense_train_path=dense_train_path,
        dense_eval_path=dense_eval_path,
        manifest_path=manifest_path,
        benchmark_manifest_path=benchmark_manifest_path,
        summary_rows=summary_rows,
    )
    _write_report(config.report_path, result)
    return result


def main() -> None:
    result = run_common_modeling_high_density_scenarios()
    print(json.dumps({"report": result.config.report_path.as_posix(), "manifest": result.manifest_path.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
