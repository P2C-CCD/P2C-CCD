from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from p2cccd.datasets.ccd.contracts import SourceLicense

from .common import approach_motion_sample, load_json_if_exists, mesh_paths_under, object_asset_from_mesh
from .contracts import ObjectMeshAsset, ObjectMotionSample


THINGI10K_SOURCE_NAME = "Thingi10K"


def default_thingi10k_root() -> Path:
    return Path(__file__).resolve().parents[4] / "datasets" / "thingi10k"


def _dirty_score(path: Path) -> float:
    metadata = load_json_if_exists(path.with_suffix(".json"))
    score = 0.0
    lower_name = path.name.lower()
    if any(token in lower_name for token in ("dirty", "broken", "nonmanifold", "bad")):
        score += 0.35
    if metadata.get("dirty") is True or metadata.get("non_manifold") is True:
        score += 0.45
    if metadata.get("self_intersections") is True:
        score += 0.20
    return min(1.0, score)


@dataclass(frozen=True, slots=True)
class Thingi10KAdapter:
    root: Path | None = None

    @property
    def source_root(self) -> Path:
        return self.root if self.root is not None else default_thingi10k_root()

    def require_available(self) -> None:
        self.license().require_available()
        if not self.source_root.exists():
            raise FileNotFoundError(f"Thingi10K root not found: {self.source_root}")

    def license(self) -> SourceLicense:
        return SourceLicense(
            name=THINGI10K_SOURCE_NAME,
            license_path=self.source_root / "LICENSE",
            url="https://ten-thousand-models.appspot.com/",
            terms="Dirty/OOD mesh stress testing; preserve Thingi10K source license and mesh metadata.",
        )

    def list_mesh_paths(self, *, limit: int | None = None) -> tuple[Path, ...]:
        self.require_available()
        return mesh_paths_under(self.source_root, limit=limit)

    def load_asset(self, path: str | Path) -> ObjectMeshAsset:
        self.require_available()
        mesh_path = Path(path)
        if not mesh_path.is_absolute():
            mesh_path = self.source_root / mesh_path
        if not mesh_path.exists():
            raise FileNotFoundError(f"Thingi10K mesh not found: {mesh_path}")
        metadata = load_json_if_exists(mesh_path.with_suffix(".json"))
        return object_asset_from_mesh(
            source_name=THINGI10K_SOURCE_NAME,
            root=self.source_root,
            mesh_path=mesh_path,
            object_name=mesh_path.stem,
            category=str(metadata.get("category", "dirty_mesh_ood")),
            dirty_score=_dirty_score(mesh_path),
            metadata={
                "ood_role": "proxy_inflation_fallback_exact_robustness",
                **metadata,
            },
        )

    def list_assets(self, *, limit: int | None = None) -> tuple[ObjectMeshAsset, ...]:
        assets = [self.load_asset(path) for path in self.list_mesh_paths(limit=limit)]
        assets.sort(key=lambda asset: (-asset.dirty_score, asset.object_id))
        return tuple(assets)

    def generate_ood_stress_samples(
        self,
        *,
        assets: Sequence[ObjectMeshAsset] | None = None,
        limit: int | None = None,
        proxy_inflation_scale: float = 1.5,
    ) -> tuple[ObjectMotionSample, ...]:
        if proxy_inflation_scale < 1.0:
            raise ValueError("proxy_inflation_scale must be >= 1")
        asset_list = list(assets) if assets is not None else list(self.list_assets())
        samples: list[ObjectMotionSample] = []
        for index, asset_a in enumerate(asset_list):
            for asset_b in asset_list[index + 1 :]:
                samples.append(
                    approach_motion_sample(
                        source_name=THINGI10K_SOURCE_NAME,
                        sample_id=f"thingi10k_ood:{len(samples)}",
                        asset_a=asset_a,
                        asset_b=asset_b,
                        motion_type="dirty_mesh_ood_approach",
                        approach_fraction=0.25,
                        metadata={
                            "proxy_inflation_scale": proxy_inflation_scale,
                            "fallback_expected": asset_a.dirty_score > 0.0 or asset_b.dirty_score > 0.0,
                            "exact_robustness_stress": True,
                        },
                    )
                )
                if limit is not None and len(samples) >= limit:
                    return tuple(samples)
        return tuple(samples)
