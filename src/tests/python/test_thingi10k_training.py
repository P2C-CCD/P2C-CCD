from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench.thingi10k_training import (  # noqa: E402
    Thingi10KTrainingExperimentConfig,
    run_thingi10k_training_experiment,
    write_thingi10k_training_report,
    write_thingi10k_training_summary_json,
)
from p2cccd.bench.trained_stpf_high_density import HighDensitySTPFConfig  # noqa: E402
from p2cccd.datasets.objects import (  # noqa: E402
    Thingi10KOfficialSubsetConfig,
    Thingi10KProxyDatasetConfig,
    default_thingi10k_root,
    generate_thingi10k_proxy_datasets,
    prepare_thingi10k_official_subset,
)
from p2cccd.proposal.training import STPFTrainingConfig  # noqa: E402


def _require_local_thingi10k() -> Path:
    root = default_thingi10k_root()
    if not (root / "official_subset_manifest.json").exists():
        pytest.skip("local Thingi10K official subset is not materialized")
    return root


def test_prepare_thingi10k_official_subset_uses_local_manifest() -> None:
    root = _require_local_thingi10k()
    output_root = prepare_thingi10k_official_subset(
        Thingi10KOfficialSubsetConfig(root=root, asset_limit=8, min_facets=48, max_facets=800)
    )

    assert output_root == root
    assert (root / "official_subset_manifest.json").exists()


def test_generate_thingi10k_proxy_datasets_from_local_subset() -> None:
    root = _require_local_thingi10k()
    bundle = generate_thingi10k_proxy_datasets(
        Thingi10KProxyDatasetConfig(
            subset=Thingi10KOfficialSubsetConfig(root=root, asset_limit=12, min_facets=48, max_facets=800),
            train_fraction=0.75,
            train_pair_limit=12,
            eval_pair_limit=6,
            seed=fixed_seed,
        )
    )

    assert len(bundle.assets) == 12
    assert len(bundle.train_pairs) > 0
    assert len(bundle.eval_pairs) > 0
    assert len(bundle.train_dataset.rows) == len(bundle.train_dataset.samples)
    assert len(bundle.eval_dataset.rows) == len(bundle.eval_dataset.samples)


def test_thingi10k_training_experiment_runs_and_writes_outputs(tmp_path: Path) -> None:
    root = _require_local_thingi10k()
    cfg = Thingi10KTrainingExperimentConfig(
        dataset=Thingi10KProxyDatasetConfig(
            subset=Thingi10KOfficialSubsetConfig(root=root, asset_limit=12, min_facets=48, max_facets=800),
            train_fraction=0.75,
            train_pair_limit=12,
            eval_pair_limit=6,
            seed=fixed_seed,
        ),
        high_density=HighDensitySTPFConfig(slab_count=4, patches_per_object=3, representative_attempt_limit=2),
        training=STPFTrainingConfig(
            epochs=2,
            batch_size=128,
            learning_rate=1.0e-3,
            seed=fixed_seed,
            device="cpu",
            validation_fraction=0.0,
        ),
        shard_root=str(tmp_path / "shards"),
        training_output_dir=str(tmp_path / "training"),
        run_name="thingi10k_training_smoke",
        benchmark_device="cpu",
    )
    result = run_thingi10k_training_experiment(cfg)
    report_path = write_thingi10k_training_report(tmp_path / "thingi10k_training.md", result)
    summary_json_path = write_thingi10k_training_summary_json(tmp_path / "thingi10k_training.json", result)

    assert result.mixed_train_row_count > len(result.bundle.train_dataset.rows)
    assert result.dense_eval_workload.avg_candidates_per_query == 36.0
    assert result.trained_stpf.fn_count == 0
    assert result.baseline.exact_work_units > result.trained_stpf.exact_work_units
    assert report_path.exists()
    assert summary_json_path.exists()
    assert (result.artifacts.shard_dir / "base_train.npz").exists()
    assert (result.artifacts.shard_dir / "dense_eval.npz").exists()

