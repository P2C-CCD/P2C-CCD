from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench.t0_synthetic_proxy import (  # noqa: E402
    T0SyntheticProxyExperimentConfig,
    run_t0_synthetic_proxy_experiment,
    write_t0_synthetic_proxy_report,
    write_t0_synthetic_proxy_summary_json,
)
from p2cccd.bench.trained_stpf_high_density import HighDensitySTPFConfig  # noqa: E402
from p2cccd.proposal.training import STPFTrainingConfig  # noqa: E402


def test_t0_synthetic_proxy_experiment_runs_and_writes_outputs(tmp_path: Path) -> None:
    config = T0SyntheticProxyExperimentConfig(
        train_mesh_count_per_split=4,
        train_robot_link_count=2,
        eval_mesh_count_per_split=2,
        eval_robot_link_count=1,
        seed=31,
        high_density=HighDensitySTPFConfig(slab_count=4, patches_per_object=3, representative_attempt_limit=2),
        training=STPFTrainingConfig(
            epochs=2,
            batch_size=128,
            learning_rate=2.0e-3,
            seed=31,
            device="cpu",
            validation_fraction=0.0,
        ),
        shard_root=str(tmp_path / "shards"),
        training_output_dir=str(tmp_path / "training"),
        run_name="smoke_t0",
        benchmark_device="cpu",
    )
    result = run_t0_synthetic_proxy_experiment(config)

    report_path = write_t0_synthetic_proxy_report(tmp_path / "report.md", result)
    summary_json_path = write_t0_synthetic_proxy_summary_json(tmp_path / "summary.json", result)

    assert result.mixed_train_row_count > len(result.base_train_dataset.rows)
    assert result.dense_eval_workload.avg_candidates_per_query > 1.0
    assert result.trained_stpf.fn_count == 0
    assert result.baseline.exact_work_units > result.trained_stpf.exact_work_units
    assert report_path.exists()
    assert summary_json_path.exists()
    assert (result.artifacts.shard_dir / "base_train.npz").exists()
    assert (result.artifacts.shard_dir / "dense_train.npz").exists()
