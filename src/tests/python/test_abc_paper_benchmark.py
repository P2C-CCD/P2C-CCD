from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench.trained_stpf_high_density import HighDensitySTPFConfig  # noqa: E402
from p2cccd.bench.abc_paper_benchmark import (  # noqa: E402
    ABCPaperBenchmarkConfig,
    build_abc_paper_benchmark_dataset,
    run_abc_paper_benchmark,
    write_abc_paper_benchmark_report,
    write_abc_paper_benchmark_summary_json,
)
from p2cccd.datasets.cad import abc_official as abc_official_mod  # noqa: E402
from p2cccd.datasets.cad import bootstrap_abc_demo_subset  # noqa: E402
from p2cccd.proposal.stpf_model import build_stpf_model  # noqa: E402


def test_abc_paper_benchmark_dataset_is_held_out_and_exported(tmp_path: Path) -> None:
    root = tmp_path / "abc"
    bootstrap_abc_demo_subset(root, asset_count=18)
    cfg = ABCPaperBenchmarkConfig(
        root=root,
        allow_demo_bootstrap=False,
        benchmark_asset_offset=9,
        benchmark_asset_count=6,
        pair_limit=12,
        seed=31,
        benchmark_dataset_dir=str(tmp_path / "datasets"),
        benchmark_output_dir=str(tmp_path / "benchmark"),
        run_name="abc_paper_dataset_smoke",
    )
    dataset = build_abc_paper_benchmark_dataset(cfg)

    assert dataset.used_demo_subset is True
    assert dataset.asset_offset == 9
    assert dataset.asset_count == 6
    assert dataset.pair_count == 12
    assert len(dataset.generated_dataset.samples) == 48
    assert dataset.dataset_npz_path.exists()
    assert dataset.dataset_manifest_path.exists()


def test_abc_paper_benchmark_runs_with_cpu_reference_rt(tmp_path: Path) -> None:
    root = tmp_path / "abc"
    bootstrap_abc_demo_subset(root, asset_count=18)
    model = build_stpf_model()
    checkpoint_path = tmp_path / "stpf_state.pt"
    torch.save(model.state_dict(), checkpoint_path)

    cfg = ABCPaperBenchmarkConfig(
        root=root,
        allow_demo_bootstrap=False,
        benchmark_asset_offset=9,
        benchmark_asset_count=6,
        pair_limit=10,
        seed=41,
        rt_backend_name="cpu_reference_rt",
        model_checkpoint_path=str(checkpoint_path),
        model_device="cpu",
        benchmark_dataset_dir=str(tmp_path / "datasets"),
        benchmark_output_dir=str(tmp_path / "benchmark"),
        run_name="abc_paper_benchmark_smoke",
    )
    result = run_abc_paper_benchmark(cfg)
    report_path = write_abc_paper_benchmark_report(tmp_path / "abc_paper_benchmark.md", result)
    summary_json_path = write_abc_paper_benchmark_summary_json(tmp_path / "abc_paper_benchmark.json", result)

    assert result.rtstpf_trained.benchmark.fn_count == 0
    assert result.rtstpf_trained.benchmark.candidate_recall == 1.0
    assert result.no_proposal.benchmark.query_count == len(result.dataset.generated_dataset.samples)
    assert result.rtstpf_random is not None
    assert report_path.exists()
    assert summary_json_path.exists()


def test_abc_paper_benchmark_hard_case_metrics_exist(tmp_path: Path) -> None:
    root = tmp_path / "abc"
    bootstrap_abc_demo_subset(root, asset_count=18)
    model = build_stpf_model()
    checkpoint_path = tmp_path / "stpf_state.pt"
    torch.save(model.state_dict(), checkpoint_path)

    cfg = ABCPaperBenchmarkConfig(
        root=root,
        allow_demo_bootstrap=False,
        benchmark_asset_offset=9,
        benchmark_asset_count=6,
        pair_limit=8,
        seed=53,
        rt_backend_name="cpu_reference_rt",
        model_checkpoint_path=str(checkpoint_path),
        model_device="cpu",
        hard_case_enabled=True,
        hard_case=HighDensitySTPFConfig(slab_count=4, patches_per_object=3, representative_attempt_limit=2),
        benchmark_dataset_dir=str(tmp_path / "datasets"),
        benchmark_output_dir=str(tmp_path / "benchmark"),
        run_name="abc_paper_benchmark_hard_case_smoke",
    )
    result = run_abc_paper_benchmark(cfg)

    assert result.hard_case_no_proposal is not None
    assert result.hard_case_trained is not None
    assert result.hard_case_no_proposal.avg_candidates_per_query == 36.0
    assert result.hard_case_trained.fn_count == 0


def test_fetch_abc_official_obj_chunks_parses_manifest(monkeypatch) -> None:
    def fake_read_text(url: str) -> str:
        if url == abc_official_mod.ABC_OFFICIAL_SIZE_YML_URL:
            return "\n".join(
                [
                    "abc_0015_obj_v00.7z: 5432922997",
                    "abc_0001_obj_v00.7z: 6938954286",
                ]
            )
        if url == abc_official_mod.ABC_OFFICIAL_OBJ_V00_URL:
            return "\n".join(
                [
                    "https://example.com/0015 abc_0015_obj_v00.7z",
                    "https://example.com/0001 abc_0001_obj_v00.7z",
                ]
            )
        raise AssertionError(url)

    monkeypatch.setattr(abc_official_mod, "_read_text", fake_read_text)
    chunks = abc_official_mod.fetch_abc_official_obj_chunks()

    assert len(chunks) == 2
    assert chunks[0].chunk_name == "abc_0015_obj_v00.7z"
    assert chunks[0].size_bytes == 5432922997
