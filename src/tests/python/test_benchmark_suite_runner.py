from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd import BenchmarkRunMeta, from_json, read_jsonl  # noqa: E402
from p2cccd.bench import (  # noqa: E402
    BenchmarkSuiteConfig,
    BenchmarkSuiteDatasetConfig,
    BenchmarkSuiteCaseConfig,
    load_benchmark_suite_config,
    run_benchmark_suite,
    run_benchmark_suite_from_config_path,
    validate_benchmark_suite_config,
)
from p2cccd.bench.run_suite import main as run_suite_main  # noqa: E402
from p2cccd.contracts import BenchmarkRowV2  # noqa: E402


CONFIG_ROOT = PROJECT_ROOT / "configs" / "benchmark_suites"


def _environment() -> dict[str, object]:
    return {
        "git_commit": "abc123",
        "host_name": "suite-host",
        "platform": "suite-platform",
        "python_version": "3.12.0",
        "gpu_name": "Suite GPU",
        "driver_version": "555.00",
        "cuda_version": "12.6",
        "optix_version": "8.0",
        "vram_total_mb": 24000,
        "vram_free_mb": 16000,
    }


def test_bundled_benchmark_suite_configs_load_and_validate() -> None:
    for name in (
        "correctness.json",
        "correctness_external.json",
        "ci_minimal_cpu.json",
        "performance.json",
        "ablation.json",
        "ood_stress.json",
        "rt_style_reproduction.json",
        "learned_style_comparison.json",
        "curobo_downstream.json",
    ):
        config = load_benchmark_suite_config(CONFIG_ROOT / name)
        assert validate_benchmark_suite_config(config) is config
        assert config.cases
        assert config.output_root.endswith("outputs/benchmark_suites")


def test_benchmark_suite_runner_exports_v2_files(tmp_path: Path) -> None:
    suite = BenchmarkSuiteConfig(
        schema_version=1,
        suite_name="unit_suite",
        suite_type="correctness",
        seed=7,
        output_root=str(tmp_path),
        notes="unit",
        cases=(
            BenchmarkSuiteCaseConfig(
                name="pure_exact",
                method="PureExactCPU",
                dataset=BenchmarkSuiteDatasetConfig(
                    mesh_count_per_split=1,
                    robot_link_count=0,
                    include_robot_links=False,
                ),
            ),
            BenchmarkSuiteCaseConfig(
                name="patch_ablation",
                method="PatchGranularityAblation",
                dataset=BenchmarkSuiteDatasetConfig(
                    mesh_count_per_split=1,
                    robot_link_count=0,
                    include_robot_links=False,
                ),
                config={
                    "options": [
                        {"name": "coarse", "patches_per_object": 1},
                        {"name": "fine", "patches_per_object": 2}
                    ]
                },
            ),
        ),
    )

    result = run_benchmark_suite(
        suite,
        environment=_environment(),
        run_id="suite_unit_001",
    )

    assert len(result.rows) == 3
    assert len(result.case_results) == 2
    assert result.export_paths is not None
    assert result.export_paths.csv_path.exists()
    assert result.export_paths.jsonl_path.exists()
    assert result.export_paths.run_meta_path.exists()
    assert {row.method_name for row in result.rows} == {
        "PureExactCPU",
        "PatchGranularityAblation:coarse",
        "PatchGranularityAblation:fine",
    }
    assert all(row.fn_count == 0 for row in result.rows)
    assert all(row.candidate_recall == 1.0 for row in result.rows)

    loaded_rows = list(read_jsonl(result.export_paths.jsonl_path, BenchmarkRowV2))
    loaded_meta = from_json(BenchmarkRunMeta, result.export_paths.run_meta_path.read_text(encoding="utf-8"))

    assert loaded_rows == list(result.rows)
    assert loaded_meta.row_count == len(result.rows)
    assert loaded_meta.gpu_name == "Suite GPU"


def test_benchmark_suite_config_path_and_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "suite.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_name": "cli_suite",
                "suite_type": "ablation",
                "seed": 11,
                "output_root": str(tmp_path / "from_config"),
                "cases": [
                    {
                        "name": "slab_proxy",
                        "method": "SlabProxyAblation",
                        "dataset": {
                            "kind": "internal_generated",
                            "mesh_count_per_split": 1,
                            "robot_link_count": 0,
                            "include_robot_links": False
                        },
                        "config": {
                            "options": [
                                {
                                    "name": "slab1",
                                    "slab_count": 1,
                                    "proxy_type_a": "SWEPT_AABB",
                                    "proxy_type_b": "SWEPT_AABB"
                                },
                                {
                                    "name": "slab2_capsule",
                                    "slab_count": 2,
                                    "proxy_type_a": "CAPSULE",
                                    "proxy_type_b": "CAPSULE"
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_benchmark_suite_from_config_path(
        config_path,
        output_root=tmp_path / "api",
        environment=_environment(),
        run_id="suite_api_001",
    )

    assert len(result.rows) == 2
    assert result.export_paths is not None
    assert result.export_paths.csv_path.exists()

    exit_code = run_suite_main(
        [
            "--config",
            str(config_path),
            "--output-root",
            str(tmp_path / "cli"),
            "--run-id",
            "suite_cli_001",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "cli" / "cli_suite" / "suite_cli_001" / "benchmark.csv").exists()
