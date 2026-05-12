from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .baseline_registry import default_baseline_root
from .contracts import SourceLicense


@dataclass(frozen=True, slots=True)
class RootParityAdapter:
    root: Path | None = None

    @property
    def source_root(self) -> Path:
        return self.root if self.root is not None else default_baseline_root() / "Exact-Root-Parity-CCD"

    def require_available(self) -> None:
        self.license().require_available()
        if not (self.source_root / "src" / "ccd.hpp").exists():
            raise FileNotFoundError(f"Exact Root Parity CCD source tree is incomplete: {self.source_root}")

    def license(self) -> SourceLicense:
        return SourceLicense(
            name="Exact Root Parity CCD",
            license_path=self.source_root / "LICENSE",
            url="https://github.com/Continuous-Collision-Detection/Exact-Root-Parity-CCD",
            terms="Use as P1 exact narrow-phase reference; not a P2CCCD runtime dependency.",
        )

    def reference_entry_points(self) -> dict[str, str]:
        self.require_available()
        return {
            "ccd_header": str(self.source_root / "src" / "ccd.hpp"),
            "ccd_source": str(self.source_root / "src" / "ccd.cpp"),
            "test": str(self.source_root / "tests" / "test_ccd.cpp"),
        }
