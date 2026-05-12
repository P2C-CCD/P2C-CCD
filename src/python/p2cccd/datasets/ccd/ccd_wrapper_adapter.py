from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .baseline_registry import default_baseline_root
from .contracts import SourceLicense


@dataclass(frozen=True, slots=True)
class CCDWrapperAdapter:
    root: Path | None = None

    @property
    def source_root(self) -> Path:
        return self.root if self.root is not None else default_baseline_root() / "CCD-Wrapper"

    def require_available(self) -> None:
        self.license().require_available()
        if not (self.source_root / "src" / "benchmark.cpp").exists():
            raise FileNotFoundError(f"CCD-Wrapper source tree is incomplete: {self.source_root}")

    def license(self) -> SourceLicense:
        return SourceLicense(
            name="CCD-Wrapper",
            license_path=self.source_root / "LICENSE",
            url="https://github.com/Continuous-Collision-Detection/CCD-Wrapper",
            terms="Use as benchmark harness and method organization reference; keep third-party method licenses separate.",
        )

    def benchmark_entry_points(self) -> dict[str, str]:
        self.require_available()
        return {
            "benchmark": str(self.source_root / "src" / "benchmark.cpp"),
            "ccd_api": str(self.source_root / "src" / "ccd.hpp"),
            "query_dump_test": str(self.source_root / "tests" / "dump_queries.hpp"),
        }

    def method_recipe_names(self) -> tuple[str, ...]:
        self.require_available()
        recipes_dir = self.source_root / "cmake" / "recipes"
        return tuple(sorted(path.stem for path in recipes_dir.glob("*.cmake")))
