from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import DatasetGenerationConfig, generate_exact_oracle_dataset
from .metrics import compute_label_metrics
from .shards import default_metadata, write_npz_shard


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a P2CCCD proposal dataset shard.")
    parser.add_argument("output", type=Path, help="Output .npz shard path.")
    parser.add_argument("--mesh-count-per-split", type=int, default=8)
    parser.add_argument("--robot-link-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-robot-links", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = DatasetGenerationConfig(
        mesh_count_per_split=args.mesh_count_per_split,
        robot_link_count=args.robot_link_count,
        seed=args.seed,
        include_robot_links=not args.no_robot_links,
    )
    dataset = generate_exact_oracle_dataset(config)
    metadata = default_metadata(dataset, seed=args.seed)
    write_npz_shard(args.output, dataset, metadata=metadata)
    metrics = compute_label_metrics(dataset.rows)
    print(f"wrote {args.output.resolve()}")
    print(
        "rows="
        f"{metrics.row_count}, positives={metrics.positive_count}, "
        f"positive_ratio={metrics.positive_ratio:.3f}, "
        f"mean_cost={metrics.mean_cost_target:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
