from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from p2cccd.bench.tight_inclusion_dense_group_real_exact import _make_groups, _write_schedule


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build matched learned/random Tight-Inclusion dense-group schedules."
    )
    parser.add_argument(
        "--shard",
        type=Path,
        default=Path(
            "src/datasets/training/tight_inclusion_nyu/shards/"
            "tight_inclusion_nyu_large_run_id/heldout_test/chunk_000002.npz"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "src/outputs/stpf_training/tight_inclusion_nyu_large_run_id/model_state.pt"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--group-count", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--positives-per-group", type=int, default=1)
    parser.add_argument("--seed", type=int, default=fixed_seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    learned_rows, learned_stats = _make_groups(
        shard=args.shard,
        checkpoint=args.checkpoint,
        group_count=args.group_count,
        group_size=args.group_size,
        positives_per_group=args.positives_per_group,
        seed=args.seed,
        source_name="tight_inclusion_nyu_large_selected_real_dense_group",
        device=args.device,
        random_schedule=False,
    )
    random_rows, random_stats = _make_groups(
        shard=args.shard,
        checkpoint=args.checkpoint,
        group_count=args.group_count,
        group_size=args.group_size,
        positives_per_group=args.positives_per_group,
        seed=args.seed,
        source_name="tight_inclusion_nyu_large_selected_real_dense_group",
        device=args.device,
        random_schedule=True,
    )

    learned_csv = args.output_dir / f"{args.run_name}_learned_schedule.csv"
    random_csv = args.output_dir / f"{args.run_name}_random_schedule.csv"
    _write_schedule(learned_csv, learned_rows)
    _write_schedule(random_csv, random_rows)
    object.__setattr__(learned_stats, "schedule_csv", learned_csv.as_posix())
    object.__setattr__(random_stats, "schedule_csv", random_csv.as_posix())

    payload = {
        "run_name": args.run_name,
        "shard": args.shard.as_posix(),
        "checkpoint": args.checkpoint.as_posix(),
        "group_count": args.group_count,
        "group_size": args.group_size,
        "candidate_count": args.group_count * args.group_size,
        "positives_per_group": args.positives_per_group,
        "seed": args.seed,
        "device": args.device,
        "learned": asdict(learned_stats),
        "random": asdict(random_stats),
    }
    (args.output_dir / f"{args.run_name}_schedule_stats.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
