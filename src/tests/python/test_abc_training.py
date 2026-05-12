from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench.abc_training import (  # noqa: E402
    ABCTrainingExperimentConfig,
    run_abc_training_experiment,
    write_abc_training_report,
    write_abc_training_summary_json,
)
from p2cccd.bench.trained_stpf_high_density import HighDensitySTPFConfig  # noqa: E402
from p2cccd.datasets.cad import (  # noqa: E402
    ABCProxyDatasetConfig,
    bootstrap_abc_demo_subset,
    generate_abc_proxy_datasets,
)
from p2cccd.proposal.training import STPFTrainingConfig  # noqa: E402


def test_bootstrap_abc_demo_subset_and_generate_proxy_datasets(tmp_path: Path) -> None:
    root = tmp_path / "abc"
    bootstrap_abc_demo_subset(root, asset_count=8)
    bundle = generate_abc_proxy_datasets(
        ABCProxyDatasetConfig(
            root=root,
            allow_demo_bootstrap=False,
            asset_limit=8,
            pair_limit=12,
            train_fraction=0.75,
            seed=19,
        )
    )

    assert bundle.used_demo_subset is True
    assert len(bundle.assets) == 8
    assert len(bundle.train_pairs) > 0
    assert len(bundle.eval_pairs) > 0
    assert len(bundle.train_dataset.rows) == len(bundle.train_dataset.samples)
    assert len(bundle.eval_dataset.rows) == len(bundle.eval_dataset.samples)


def test_abc_adapter_supports_relative_root_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("abc_relative")
    bootstrap_abc_demo_subset(root, asset_count=8)
    bundle = generate_abc_proxy_datasets(
        ABCProxyDatasetConfig(
            root=root,
            allow_demo_bootstrap=False,
            asset_limit=8,
            pair_limit=10,
            train_fraction=0.7,
            seed=29,
        )
    )

    assert len(bundle.assets) == 8
    assert bundle.source_root == root


def test_abc_training_experiment_runs_and_writes_outputs(tmp_path: Path) -> None:
    root = tmp_path / "abc"
    cfg = ABCTrainingExperimentConfig(
        dataset=ABCProxyDatasetConfig(
            root=root,
            allow_demo_bootstrap=True,
            demo_asset_count=10,
            asset_limit=10,
            pair_limit=16,
            train_fraction=0.75,
            seed=23,
        ),
        high_density=HighDensitySTPFConfig(slab_count=4, patches_per_object=3, representative_attempt_limit=2),
        training=STPFTrainingConfig(
            epochs=2,
            batch_size=128,
            learning_rate=2.0e-3,
            seed=23,
            device="cpu",
            validation_fraction=0.0,
        ),
        shard_root=str(tmp_path / "shards"),
        training_output_dir=str(tmp_path / "training"),
        run_name="abc_smoke",
        benchmark_device="cpu",
    )
    result = run_abc_training_experiment(cfg)
    report_path = write_abc_training_report(tmp_path / "abc_report.md", result)
    summary_json_path = write_abc_training_summary_json(tmp_path / "abc_summary.json", result)

    assert result.bundle.used_demo_subset is True
    assert result.mixed_train_row_count > len(result.bundle.train_dataset.rows)
    assert result.dense_eval_workload.avg_candidates_per_query > 1.0
    assert result.trained_stpf.fn_count == 0
    assert result.baseline.exact_work_units > result.trained_stpf.exact_work_units
    assert report_path.exists()
    assert summary_json_path.exists()
    assert (result.artifacts.shard_dir / "base_train.npz").exists()
    assert (result.artifacts.shard_dir / "dense_train.npz").exists()
