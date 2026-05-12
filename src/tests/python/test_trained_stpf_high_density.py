from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench.trained_stpf_high_density import (  # noqa: E402
    HighDensitySTPFConfig,
    build_high_density_stpf_workload,
    run_trained_stpf_high_density_experiment,
    workload_to_shard_dataset,
)
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset  # noqa: E402


def test_high_density_workload_expands_candidates_per_query() -> None:
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=1, robot_link_count=0, seed=17)
    )
    workload = build_high_density_stpf_workload(
        dataset,
        HighDensitySTPFConfig(slab_count=4, patches_per_object=3),
    )

    assert workload.query_count == len(dataset.samples)
    assert workload.avg_candidates_per_query > 1.0
    assert workload.candidate_count == workload.query_count * 4 * 3 * 3
    shard_dataset = workload_to_shard_dataset(workload)
    assert len(shard_dataset.rows) == workload.candidate_count
    assert len(shard_dataset.samples) == workload.candidate_count
    assert len(shard_dataset.traces) == workload.candidate_count


def test_trained_stpf_high_density_experiment_runs_and_reduces_vs_no_proposal(tmp_path: Path) -> None:
    train_dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=0, seed=21)
    )
    eval_dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=1, robot_link_count=0, seed=23)
    )
    result = run_trained_stpf_high_density_experiment(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=HighDensitySTPFConfig(slab_count=4, patches_per_object=3, representative_attempt_limit=2),
        training_output_dir=str(tmp_path),
        run_name="smoke",
        training_device="cpu",
        benchmark_device="cpu",
        epochs=2,
        batch_size=64,
        learning_rate=2.0e-3,
        seed=23,
    )

    assert result.eval_workload.avg_candidates_per_query > 1.0
    assert result.baseline.exact_work_units > result.trained_stpf.exact_work_units
    assert result.trained_stpf.fn_count == 0
    assert result.random_stpf.fn_count == 0
