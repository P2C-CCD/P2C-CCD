from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "artifacts" / "release_case_manifest.json"
MODEL_ARTIFACT_MANIFEST = REPO_ROOT / "src" / "docs" / "model_artifacts_manifest.json"


def iter_release_paths(case: dict) -> Iterable[tuple[str, str]]:
    release_paths = case.get("release_paths", {})
    for group, paths in release_paths.items():
        for path in paths:
            yield group, path


def validate_path(path_text: str) -> tuple[bool, str]:
    path = Path(path_text)
    if path.is_absolute():
        return False, "absolute paths are not allowed in the release manifest"
    normalized = path_text.replace("\\", "/")
    if normalized.startswith("Paper/") or "/Paper/" in normalized:
        return False, "manifest path points outside the public release tree"

    target = REPO_ROOT / path
    if not target.exists():
        return False, "missing"
    if target.is_file() and target.stat().st_size <= 0:
        return False, "empty file"
    if target.is_dir() and not any(child.is_file() for child in target.rglob("*")):
        return False, "empty directory"
    return True, "ok"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_artifacts() -> list[str]:
    if not MODEL_ARTIFACT_MANIFEST.exists():
        return [f"missing {MODEL_ARTIFACT_MANIFEST.relative_to(REPO_ROOT)}"]
    manifest = json.loads(MODEL_ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    for entry in manifest.get("entries", []):
        rel_path = entry.get("path", "")
        target = REPO_ROOT / rel_path
        if not target.exists():
            failures.append(f"{rel_path} (missing)")
            continue
        expected_size = int(entry.get("bytes", -1))
        if target.stat().st_size != expected_size:
            failures.append(
                f"{rel_path} (size {target.stat().st_size}, expected {expected_size})"
            )
            continue
        expected_sha = str(entry.get("sha256", "")).lower()
        actual_sha = sha256_file(target)
        if actual_sha != expected_sha:
            failures.append(f"{rel_path} (sha256 mismatch)")
    return failures


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    total_paths = 0

    print(f"release root: {REPO_ROOT}")
    print(f"manifest: {MANIFEST.relative_to(REPO_ROOT)}")
    print()

    for case in manifest.get("cases", []):
        case_id = case.get("id", "<missing-id>")
        case_title = case.get("title", case_id)
        case_paths = list(iter_release_paths(case))
        total_paths += len(case_paths)
        case_failures = []
        for group, path_text in case_paths:
            ok, reason = validate_path(path_text)
            if not ok:
                case_failures.append(f"{group}: {path_text} ({reason})")
        status = "PASS" if not case_failures else "FAIL"
        print(f"[{status}] {case_id}: {case_title} ({len(case_paths)} paths)")
        for failure in case_failures:
            print(f"  - {failure}")
        failures.extend(f"{case_id}: {failure}" for failure in case_failures)

    print()
    print(f"checked cases: {len(manifest.get('cases', []))}")
    print(f"checked paths: {total_paths}")
    artifact_failures = validate_model_artifacts()
    if artifact_failures:
        print(f"artifact failures: {len(artifact_failures)}")
        for failure in artifact_failures:
            print(f"  - {failure}")
        failures.extend(f"model artifact: {failure}" for failure in artifact_failures)
    else:
        artifact_count = len(
            json.loads(MODEL_ARTIFACT_MANIFEST.read_text(encoding="utf-8")).get(
                "entries", []
            )
        )
        print(f"checked model/data artifacts: {artifact_count}")
    if failures:
        print(f"failures: {len(failures)}")
        return 1
    print("all release-local paper case assets and compact model/data artifacts are present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
