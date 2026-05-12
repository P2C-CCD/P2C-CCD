from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.datasets.cad import ABCDatasetAdapter, Fusion360GalleryAdapter  # noqa: E402
from p2cccd.datasets.ccd import (  # noqa: E402
    CCDWrapperAdapter,
    RootParityAdapter,
    RigidIPCSceneAdapter,
    ScalableCCDSampleAdapter,
    SourceLicense,
    TightInclusionAdapter,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder", encoding="utf-8")


def test_source_license_requires_license_file_and_source_metadata(tmp_path: Path) -> None:
    license_path = tmp_path / "LICENSE"
    license_path.write_text("unit license", encoding="utf-8")

    valid = SourceLicense(
        name="Unit Source",
        license_path=license_path,
        url="https://example.invalid/unit",
        terms="unit terms",
    )

    assert valid.require_available() is valid

    with pytest.raises(ValueError):
        SourceLicense(name="", license_path=license_path, url=valid.url, terms=valid.terms).require_available()
    with pytest.raises(ValueError):
        SourceLicense(name=valid.name, license_path=license_path, url="", terms=valid.terms).require_available()
    with pytest.raises(ValueError):
        SourceLicense(name=valid.name, license_path=license_path, url=valid.url, terms="").require_available()
    with pytest.raises(FileNotFoundError):
        SourceLicense(
            name=valid.name,
            license_path=tmp_path / "MISSING_LICENSE",
            url=valid.url,
            terms=valid.terms,
        ).require_available()


def test_external_ccd_adapters_refuse_to_run_without_license_files(tmp_path: Path) -> None:
    tight_root = tmp_path / "Tight-Inclusion"
    _touch(tight_root / "src" / "tight_inclusion" / "ccd.hpp")

    wrapper_root = tmp_path / "CCD-Wrapper"
    _touch(wrapper_root / "src" / "benchmark.cpp")

    root_parity_root = tmp_path / "Exact-Root-Parity-CCD"
    _touch(root_parity_root / "src" / "ccd.hpp")

    scalable_root = tmp_path / "Sample-Scalable-CCD-Data"
    _touch(scalable_root / "README.md")

    rigid_root = tmp_path / "rigid-ipc"
    (rigid_root / "fixtures").mkdir(parents=True)
    (rigid_root / "meshes").mkdir(parents=True)

    for adapter in (
        TightInclusionAdapter(tight_root),
        CCDWrapperAdapter(wrapper_root),
        RootParityAdapter(root_parity_root),
        ScalableCCDSampleAdapter(scalable_root),
        RigidIPCSceneAdapter(rigid_root),
    ):
        assert not adapter.license().available()
        with pytest.raises(FileNotFoundError):
            adapter.require_available()


def test_cad_adapters_refuse_to_run_without_license_files(tmp_path: Path) -> None:
    abc_root = tmp_path / "abc"
    abc_root.mkdir()
    _touch(abc_root / "part.obj")

    fusion_root = tmp_path / "fusion360"
    fusion_root.mkdir()
    _touch(fusion_root / "assembly" / "part.obj")

    abc = ABCDatasetAdapter(abc_root)
    fusion = Fusion360GalleryAdapter(fusion_root)

    assert not abc.license().available()
    assert not fusion.license().available()
    with pytest.raises(FileNotFoundError):
        abc.list_mesh_paths()
    with pytest.raises(FileNotFoundError):
        fusion.list_sequences()
