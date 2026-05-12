from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .baseline_registry import default_baseline_root
from .contracts import SourceLicense


@dataclass(frozen=True, slots=True)
class TightInclusionAdapter:
    root: Path | None = None

    @property
    def source_root(self) -> Path:
        return self.root if self.root is not None else default_baseline_root() / "Tight-Inclusion"

    def require_available(self) -> None:
        self.license().require_available()
        if not (self.source_root / "src" / "tight_inclusion" / "ccd.hpp").exists():
            raise FileNotFoundError(f"Tight Inclusion source tree is incomplete: {self.source_root}")

    def license(self) -> SourceLicense:
        return SourceLicense(
            name="Tight Inclusion",
            license_path=self.source_root / "LICENSE",
            url="https://github.com/Continuous-Collision-Detection/Tight-Inclusion",
            terms="Use as a conservative CCD reference implementation; do not vendor without license review.",
        )

    def reference_entry_points(self) -> dict[str, str]:
        self.require_available()
        return {
            "double_ccd_header": str(self.source_root / "src" / "tight_inclusion" / "ccd.hpp"),
            "double_ccd_source": str(self.source_root / "src" / "tight_inclusion" / "ccd.cpp"),
            "rational_ccd_header": str(self.source_root / "src" / "tight_inclusion" / "rational" / "ccd.hpp"),
            "app_entry": str(self.source_root / "app" / "main.cpp"),
        }
