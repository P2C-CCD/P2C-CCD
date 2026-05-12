from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench.thingi10k_paper_benchmark import (  # noqa: E402
    Thingi10KPaperBenchmarkConfig,
    build_thingi10k_paper_benchmark_dataset,
    run_thingi10k_paper_benchmark,
    write_thingi10k_paper_benchmark_report,
    write_thingi10k_paper_benchmark_summary_json,
)
from p2cccd.bench.trained_stpf_high_density import HighDensitySTPFConfig  # noqa: E402
from p2cccd.datasets.objects import Thingi10KOfficialSubsetConfig, Thingi10KProxyDatasetConfig, default_thingi10k_root  # noqa: E402
from p2cccd.proposal.stpf_model import build_stpf_model  # noqa: E402


def _require_local_thingi10k() -> Path:
    root = default_thingi10k_root()
    if not (root / "official_subset_manifest.json").exists():
        pytest.skip("local Thingi10K official subset is not materialized")
    return root


def test_thingi10k_paper_benchmark_dataset_is_held_out_and_exported(tmp_path: Path) -> None:
    root = _require_local_thingi10k()
    cfg = Thingi10KPaperBenchmarkConfig(
        dataset=Thingi10KProxyDatasetConfig(
            subset=Thingi10KOfficialSubsetConfig(root=root, asset_limit=16, min_facets=48, max_facets=800),
            train_fraction=0.75,
            train_pair_limit=16,
            eval_pair_limit=8,
            seed=fixed_seed,
        ),
        benchmark_dataset_dir=str(tmp_path / "datasets"),
        benchmark_output_dir=str(tmp_path / "benchmark"),
        run_name="thingi10k_paper_dataset_smoke",
        model_device="cpu",
    )
    bundle, dataset = build_thingi10k_paper_benchmark_dataset(cfg)

    assert len(bundle.assets) == 16
    assert dataset.asset_count == 16
    assert dataset.pair_count == len(bundle.eval_pairs)
    assert len(dataset.generated_dataset.samples) == len(bundle.eval_dataset.samples)
    assert dataset.dataset_npz_path.exists()
    assert dataset.dataset_manifest_path.exists()


def test_thingi10k_paper_benchmark_runs_with_cpu_reference_rt(tmp_path: Path) -> None:
    root = _require_local_thingi10k()
    model = build_stpf_model()
    checkpoint_path = tmp_path / "thingi10k_stpf_state.pt"
    torch.save(model.state_dict(), checkpoint_path)

    cfg = Thingi10KPaperBenchmarkConfig(
        dataset=Thingi10KProxyDatasetConfig(
            subset=Thingi10KOfficialSubsetConfig(root=root, asset_limit=16, min_facets=48, max_facets=800),
            train_fraction=0.75,
            train_pair_limit=16,
            eval_pair_limit=8,
            seed=fixed_seed,
        ),
        rt_backend_name="cpu_reference_rt",
        model_checkpoint_path=str(checkpoint_path),
        model_device="cpu",
        include_random_stpf=True,
        hard_case_enabled=True,
        hard_case=HighDensitySTPFConfig(slab_count=4, patches_per_object=3, representative_attempt_limit=2),
        benchmark_dataset_dir=str(tmp_path / "datasets"),
        benchmark_output_dir=str(tmp_path / "benchmark"),
        run_name="thingi10k_paper_benchmark_smoke",
    )
    result = run_thingi10k_paper_benchmark(cfg)
    report_path = write_thingi10k_paper_benchmark_report(tmp_path / "thingi10k_paper_benchmark.md", result)
    summary_json_path = write_thingi10k_paper_benchmark_summary_json(tmp_path / "thingi10k_paper_benchmark.json", result)

    assert result.rtstpf_trained.benchmark.fn_count == 0
    assert result.rtstpf_trained.benchmark.candidate_recall == 1.0
    assert result.no_proposal.benchmark.query_count == len(result.dataset.generated_dataset.samples)
    assert result.rtstpf_random is not None
    assert result.hard_case_no_proposal is not None
    assert result.hard_case_trained is not None
    assert result.hard_case_no_proposal.avg_candidates_per_query == 36.0
    assert report_path.exists()
    assert summary_json_path.exists()
