from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iter_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "entries" in payload:
        entries = payload["entries"]
    else:
        entries = []
        entries.extend(payload.get("evidence", []))
        entries.extend(payload.get("artifacts", []))
    if not isinstance(entries, list):
        raise ValueError("manifest entries must be a list")
    return entries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(path: Path, *, repo_root: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")

    failures = 0
    for entry in _iter_entries(payload):
        rel = entry.get("path")
        expected_hash = str(entry.get("sha256", "")).lower()
        expected_bytes = int(entry.get("bytes", -1))
        if not rel or not expected_hash:
            print(f"FAIL missing path/hash in entry: {entry}")
            failures += 1
            continue

        artifact = repo_root / str(rel)
        if not artifact.exists():
            print(f"FAIL missing: {rel}")
            failures += 1
            continue

        actual_bytes = artifact.stat().st_size
        actual_hash = _sha256(artifact)
        if actual_bytes != expected_bytes or actual_hash != expected_hash:
            print(
                "FAIL mismatch: "
                f"{rel} bytes {actual_bytes} != {expected_bytes} "
                f"or sha256 {actual_hash} != {expected_hash}"
            )
            failures += 1
        else:
            print(f"OK {rel}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify P2CCCD artifact manifest hashes.")
    parser.add_argument("manifest", type=Path, help="Path to a JSON artifact manifest.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_repo_root(),
        help="Repository root. Defaults to the root inferred from this script.",
    )
    args = parser.parse_args()
    return 1 if verify_manifest(args.manifest, repo_root=args.repo_root.resolve()) else 0


if __name__ == "__main__":
    raise SystemExit(main())

