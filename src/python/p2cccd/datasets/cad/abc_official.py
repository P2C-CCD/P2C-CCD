from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
import urllib.request

from .abc_adapter import ABC_SOURCE_NAME


ABC_OFFICIAL_SIZE_YML_URL = "https://deep-geometry.github.io/abc-dataset/data/size.yml"
ABC_OFFICIAL_OBJ_V00_URL = "https://deep-geometry.github.io/abc-dataset/data/obj_v00.txt"
ABC_OFFICIAL_STL2_V00_URL = "https://deep-geometry.github.io/abc-dataset/data/stl2_v00.txt"


def _default_dataset_root(name: str) -> Path:
    file_path = Path(__file__).resolve()
    project_local = file_path.parents[4] / "datasets" / name
    legacy_root = file_path.parents[5] / "datasets" / name
    return project_local if project_local.exists() else legacy_root


def default_abc_official_root() -> Path:
    return _default_dataset_root("abc_official")


@dataclass(frozen=True, slots=True)
class ABCOfficialObjChunk:
    chunk_name: str
    url: str
    size_bytes: int
    mesh_variant: str


def _read_text(url: str) -> str:
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def _existing_mesh_paths(root: Path, mesh_variant: str) -> tuple[Path, ...]:
    variant = str(mesh_variant).lower()
    extension = ".obj" if variant == "obj" else ".stl"
    return tuple(sorted(root.rglob(f"*{extension}")))


def fetch_abc_official_mesh_chunks(mesh_variant: str = "obj") -> tuple[ABCOfficialObjChunk, ...]:
    variant = str(mesh_variant).lower()
    if variant not in {"obj", "stl2"}:
        raise ValueError("mesh_variant must be 'obj' or 'stl2'")
    size_text = _read_text(ABC_OFFICIAL_SIZE_YML_URL)
    url_text = _read_text(ABC_OFFICIAL_OBJ_V00_URL if variant == "obj" else ABC_OFFICIAL_STL2_V00_URL)
    size_by_name: dict[str, int] = {}
    for raw_line in size_text.splitlines():
        line = raw_line.strip()
        match = re.match(rf"^(abc_\d{{4}}_{variant}_v00\.7z):\s*(\d+)$", line)
        if match is None:
            continue
        size_by_name[str(match.group(1))] = int(match.group(2))

    chunks: list[ABCOfficialObjChunk] = []
    for raw_line in url_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        url, chunk_name = parts
        size_bytes = size_by_name.get(chunk_name)
        if size_bytes is None:
            continue
        chunks.append(
            ABCOfficialObjChunk(
                chunk_name=chunk_name,
                url=url,
                size_bytes=size_bytes,
                mesh_variant=variant,
            )
        )
    if not chunks:
        raise RuntimeError("failed to fetch official ABC obj chunk manifest")
    chunks.sort(key=lambda item: (item.size_bytes, item.chunk_name))
    return tuple(chunks)


def fetch_abc_official_obj_chunks() -> tuple[ABCOfficialObjChunk, ...]:
    return fetch_abc_official_mesh_chunks("obj")


def _local_chunk_from_manifest(root: Path, mesh_variant: str, chunk_name: str | None) -> ABCOfficialObjChunk | None:
    manifest_path = root / "official_subset_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    local_variant = str(data.get("mesh_variant", "")).lower()
    local_chunk_name = str(data.get("chunk_name", ""))
    if local_variant != str(mesh_variant).lower() or not local_chunk_name:
        return None
    if chunk_name is not None and chunk_name != local_chunk_name:
        return None
    archive_path = root / "_official_archives" / local_chunk_name
    if not archive_path.exists():
        archive_hint = data.get("archive_path")
        if isinstance(archive_hint, str) and archive_hint:
            hint_path = Path(archive_hint)
            if hint_path.exists():
                archive_path = hint_path
    if not archive_path.exists():
        return None
    size_bytes = int(data.get("archive_size_bytes", archive_path.stat().st_size))
    url = str(data.get("chunk_url", ""))
    return ABCOfficialObjChunk(
        chunk_name=local_chunk_name,
        url=url,
        size_bytes=size_bytes,
        mesh_variant=local_variant,
    )


def _ensure_py7zr():
    try:
        import py7zr  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "py7zr is required to extract official ABC .7z archives. "
            "Install it in the active environment before preparing the official ABC root."
        ) from exc
    return py7zr


def _download_file(url: str, output_path: Path, *, expected_size: int | None = None, max_retries: int = 8) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    offset = output_path.stat().st_size if output_path.exists() else 0
    if expected_size is not None and offset >= expected_size:
        return output_path
    retries = 0
    while True:
        if expected_size is not None and offset >= expected_size:
            break
        request = urllib.request.Request(url)
        if offset > 0:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request) as response:
                status = getattr(response, "status", None)
                if offset > 0 and status == 200:
                    output_path.unlink(missing_ok=True)
                    offset = 0
                mode = "ab" if offset > 0 else "wb"
                with output_path.open(mode) as handle:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        offset += len(chunk)
                retries = 0
        except Exception as exc:  # pragma: no cover - exercised in real download path
            retries += 1
            if retries > max_retries:
                raise RuntimeError(f"failed to download {url} after {max_retries} retries") from exc
            time.sleep(min(30.0, 2.0 * retries))
            if not output_path.exists() and offset > 0:
                offset = 0
            elif output_path.exists():
                offset = output_path.stat().st_size
            continue
        if expected_size is None:
            break
    return output_path


def _write_root_metadata(
    root: Path,
    *,
    chunk: ABCOfficialObjChunk,
    extracted_mesh_count: int,
    archive_path: Path,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    license_path = root / "LICENSE"
    license_path.write_text(
        (
            f"{ABC_SOURCE_NAME}\n"
            "Official subset extracted from upstream ABC archive.\n"
            f"Source: {chunk.url}\n"
            "Use remains subject to upstream ABC dataset terms.\n"
        ),
        encoding="utf-8",
    )
    readme_path = root / "README.md"
    readme_path.write_text(
        (
            "# Official ABC Minimal Root\n\n"
            "This root stores a minimal subset extracted from an official upstream ABC mesh chunk.\n\n"
            f"- Chunk: `{chunk.chunk_name}`\n"
            f"- Mesh variant: `{chunk.mesh_variant}`\n"
            f"- Source URL: `{chunk.url}`\n"
            f"- Archive size bytes: `{chunk.size_bytes}`\n"
            f"- Extracted mesh count: `{extracted_mesh_count}`\n"
            f"- Archive path: `{archive_path}`\n"
        ),
        encoding="utf-8",
    )
    manifest_path = root / "official_subset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "chunk_name": chunk.chunk_name,
                "chunk_url": chunk.url,
                "mesh_variant": chunk.mesh_variant,
                "archive_size_bytes": chunk.size_bytes,
                "extracted_mesh_count": extracted_mesh_count,
                "archive_path": str(archive_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def prepare_official_abc_minimal_root(
    root: Path | None = None,
    *,
    asset_limit: int = 64,
    mesh_variant: str = "obj",
    chunk_name: str | None = None,
    archive_dir: Path | None = None,
    keep_archive: bool = True,
) -> Path:
    if asset_limit <= 0:
        raise ValueError("asset_limit must be positive")
    root_path = root if root is not None else default_abc_official_root()
    mesh_paths = _existing_mesh_paths(root_path, mesh_variant)
    if len(mesh_paths) >= asset_limit:
        return root_path

    variant = str(mesh_variant).lower()
    if variant not in {"obj", "stl2"}:
        raise ValueError("mesh_variant must be 'obj' or 'stl2'")
    selected = _local_chunk_from_manifest(root_path, variant, chunk_name)
    if selected is None:
        try:
            chunks = fetch_abc_official_mesh_chunks(variant)
        except Exception as exc:
            raise RuntimeError(
                "failed to fetch official ABC chunk manifest and no compatible local archive metadata was found"
            ) from exc
        if chunk_name is None:
            selected = chunks[0]
        else:
            matches = [chunk for chunk in chunks if chunk.chunk_name == chunk_name]
            if not matches:
                raise ValueError(f"unknown ABC obj chunk: {chunk_name}")
            selected = matches[0]

    archive_root = archive_dir if archive_dir is not None else (root_path / "_official_archives")
    archive_path = archive_root / selected.chunk_name
    if archive_path.exists():
        if archive_path.stat().st_size < int(selected.size_bytes):
            if not selected.url:
                raise RuntimeError(
                    f"local ABC archive {archive_path} is incomplete and no download URL is available"
                )
            _download_file(selected.url, archive_path, expected_size=selected.size_bytes)
    else:
        if not selected.url:
            raise RuntimeError(f"local ABC archive {archive_path} is missing and no download URL is available")
        _download_file(selected.url, archive_path, expected_size=selected.size_bytes)

    py7zr = _ensure_py7zr()
    extract_root = root_path / "official_obj_subset"
    extract_root.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        extension = ".obj" if variant == "obj" else ".stl"
        names = [name for name in archive.getnames() if name.lower().endswith(extension)]
        if len(names) < asset_limit:
            raise RuntimeError(
                f"official ABC archive {selected.chunk_name} only exposes {len(names)} mesh files, "
                f"below requested asset_limit={asset_limit}"
            )
        names.sort()
        target_names = names[:asset_limit]
        archive.extract(path=extract_root, targets=target_names)

    extracted_count = len(tuple(sorted(extract_root.rglob("*.obj")))) + len(tuple(sorted(extract_root.rglob("*.stl"))))
    _write_root_metadata(
        root_path,
        chunk=selected,
        extracted_mesh_count=extracted_count,
        archive_path=archive_path,
    )
    if not keep_archive and archive_path.exists():
        archive_path.unlink()
    return root_path


__all__ = [
    "ABC_OFFICIAL_OBJ_V00_URL",
    "ABC_OFFICIAL_SIZE_YML_URL",
    "ABC_OFFICIAL_STL2_V00_URL",
    "ABCOfficialObjChunk",
    "default_abc_official_root",
    "fetch_abc_official_mesh_chunks",
    "fetch_abc_official_obj_chunks",
    "prepare_official_abc_minimal_root",
]
