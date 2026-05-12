from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench import (  # noqa: E402
    BenchmarkSuiteCaseConfig,
    BenchmarkSuiteConfig,
    BenchmarkSuiteDatasetConfig,
    run_benchmark_suite,
)
from p2cccd.data import (  # noqa: E402
    DatasetGenerationConfig,
    generate_exact_oracle_dataset,
    generate_mesh_pair_motion_samples,
    generate_robot_link_pair_motion_samples,
)


def _sample_signature(sample) -> tuple[object, ...]:
    return (
        sample.sample_id,
        sample.query_id,
        sample.candidate_id,
        sample.split,
        int(sample.family),
        sample.object_a_id,
        sample.patch_a_id,
        sample.object_b_id,
        sample.patch_b_id,
        sample.slab_id,
        tuple(round(value, 12) for value in sample.center_a_t0),
        tuple(round(value, 12) for value in sample.center_a_t1),
        tuple(round(value, 12) for value in sample.center_b_t0),
        tuple(round(value, 12) for value in sample.center_b_t1),
        round(sample.radius_a, 12),
        round(sample.radius_b, 12),
        int(sample.proxy_type_a),
        int(sample.proxy_type_b),
        round(sample.hardness, 12),
        sample.ood,
    )


def _dataset_signature(config: DatasetGenerationConfig) -> tuple[tuple[object, ...], ...]:
    dataset = generate_exact_oracle_dataset(config)
    signature: list[tuple[object, ...]] = []
    for row, sample, trace in zip(dataset.rows, dataset.samples, dataset.traces):
        signature.append(
            (
                _sample_signature(sample),
                row.query_id,
                row.candidate_id,
                tuple(round(value, 12) for value in row.features),
                tuple(round(value, 12) for value in row.interval_targets),
                tuple(round(value, 12) for value in row.family_targets),
                round(row.priority_target, 12),
                round(row.cost_target, 12),
                round(row.uncertainty_target, 12),
                trace.collided,
                round(trace.toi, 12),
                round(trace.closest_time, 12),
                round(trace.min_distance, 12),
                round(trace.safe_margin, 12),
                trace.exact_cost,
            )
        )
    return tuple(signature)


def _suite_row_order_signature(result) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            row.dataset_name,
            row.scene_name,
            row.method_name,
            row.seed,
            row.query_count,
            row.fn_count,
            row.fp_count,
            round(row.candidate_recall, 12),
            round(row.avg_candidates, 12),
            round(row.avg_exact_evals, 12),
            row.exact_calls_total,
        )
        for row in result.rows
    )


def test_sampled_motion_reproducibility_by_seed() -> None:
    mesh_a = generate_mesh_pair_motion_samples(count_per_split=2, seed=101)
    mesh_b = generate_mesh_pair_motion_samples(count_per_split=2, seed=101)
    mesh_c = generate_mesh_pair_motion_samples(count_per_split=2, seed=102)
    robot_a = generate_robot_link_pair_motion_samples(count=3, seed=201)
    robot_b = generate_robot_link_pair_motion_samples(count=3, seed=201)

    assert [_sample_signature(sample) for sample in mesh_a] == [_sample_signature(sample) for sample in mesh_b]
    assert [_sample_signature(sample) for sample in mesh_a] != [_sample_signature(sample) for sample in mesh_c]
    assert [_sample_signature(sample) for sample in robot_a] == [_sample_signature(sample) for sample in robot_b]


def test_data_generation_reproducibility_by_seed() -> None:
    config_a = DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=2, seed=303)
    config_b = DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=2, seed=303)
    config_c = DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=2, seed=304)

    assert _dataset_signature(config_a) == _dataset_signature(config_b)
    assert _dataset_signature(config_a) != _dataset_signature(config_c)


def test_benchmark_query_ordering_reproducibility_by_seed(tmp_path: Path) -> None:
    suite = BenchmarkSuiteConfig(
        schema_version=1,
        suite_name="seed_reproducibility_smoke",
        suite_type="correctness",
        seed=509,
        output_root=str(tmp_path),
        notes="seed reproducibility smoke",
        cases=(
            BenchmarkSuiteCaseConfig(
                name="pure_exact",
                method="PureExactCPU",
                dataset=BenchmarkSuiteDatasetConfig(
                    mesh_count_per_split=1,
                    robot_link_count=1,
                    include_robot_links=True,
                ),
            ),
            BenchmarkSuiteCaseConfig(
                name="rt_exact",
                method="RTExact",
                dataset=BenchmarkSuiteDatasetConfig(
                    mesh_count_per_split=1,
                    robot_link_count=1,
                    include_robot_links=True,
                ),
                config={"backend_name": "cpu_reference_rt"},
            ),
        ),
    )

    first = run_benchmark_suite(suite, run_id="seed_reproducibility", export=False)
    second = run_benchmark_suite(suite, run_id="seed_reproducibility", export=False)

    assert _suite_row_order_signature(first) == _suite_row_order_signature(second)
