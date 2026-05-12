from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest


_RELEASE_ROOT = Path(__file__).resolve().parents[3]
_PYTEST_TEMP_ROOT = _RELEASE_ROOT / ".pytest_tmp"


def pytest_configure(config) -> None:
    _PYTEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    temp_root = str(_PYTEST_TEMP_ROOT)
    os.environ["TMP"] = temp_root
    os.environ["TEMP"] = temp_root
    os.environ["TMPDIR"] = temp_root
    tempfile.tempdir = temp_root


@pytest.fixture
def tmp_path() -> Path:
    path = _PYTEST_TEMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
