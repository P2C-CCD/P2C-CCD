from __future__ import annotations

import argparse
from pathlib import Path

from .suite_runner import run_benchmark_suite_from_config_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a P2CCCD benchmark suite config.")
    parser.add_argument("--config", required=True, help="Path to a benchmark suite JSON config.")
    parser.add_argument("--output-root", default=None, help="Override suite output root.")
    parser.add_argument("--run-id", default=None, help="Optional deterministic run id.")
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Run the suite without writing benchmark.csv, benchmark.jsonl, and run_meta.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_benchmark_suite_from_config_path(
        args.config,
        output_root=args.output_root,
        run_id=args.run_id,
        export=not args.no_export,
    )
    print(f"suite={result.suite.suite_name}")
    print(f"run_id={result.meta_run_id}")
    print(f"rows={len(result.rows)}")
    if result.export_paths is not None:
        print(f"run_dir={Path(result.export_paths.run_dir)}")
        print(f"csv={Path(result.export_paths.csv_path)}")
        print(f"jsonl={Path(result.export_paths.jsonl_path)}")
        print(f"run_meta={Path(result.export_paths.run_meta_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
