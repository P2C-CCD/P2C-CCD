from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Sequence

import numpy as np

from .oracle import ExactOracleTrace, evaluate_swept_sphere_oracle
from .samplers import MotionDiscPairSample


@dataclass(frozen=True, slots=True)
class WarpAvailability:
    installed: bool
    module_name: str = "warp"
    reason: str = ""


@dataclass(frozen=True, slots=True)
class WarpOracleResult:
    traces: list[ExactOracleTrace]
    backend: str
    warp_available: WarpAvailability


def detect_warp() -> WarpAvailability:
    spec = importlib.util.find_spec("warp")
    if spec is None:
        return WarpAvailability(installed=False, reason="Python module 'warp' is not importable")
    return WarpAvailability(installed=True, reason="Python module 'warp' is importable")


def samples_to_warp_arrays(samples: Sequence[MotionDiscPairSample]) -> dict[str, np.ndarray]:
    return {
        "sample_id": np.asarray([sample.sample_id for sample in samples], dtype=np.int64),
        "center_a_t0": np.asarray([sample.center_a_t0 for sample in samples], dtype=np.float32),
        "center_a_t1": np.asarray([sample.center_a_t1 for sample in samples], dtype=np.float32),
        "center_b_t0": np.asarray([sample.center_b_t0 for sample in samples], dtype=np.float32),
        "center_b_t1": np.asarray([sample.center_b_t1 for sample in samples], dtype=np.float32),
        "radius_sum": np.asarray(
            [sample.radius_a + sample.radius_b for sample in samples], dtype=np.float32
        ),
        "hardness": np.asarray([sample.hardness for sample in samples], dtype=np.float32),
        "ood": np.asarray([sample.ood for sample in samples], dtype=np.bool_),
    }


def evaluate_swept_sphere_oracle_with_optional_warp(
    samples: Sequence[MotionDiscPairSample],
    *,
    prefer_warp: bool = False,
    require_warp: bool = False,
) -> WarpOracleResult:
    availability = detect_warp()
    if require_warp and not availability.installed:
        raise RuntimeError(availability.reason)

    # The helper deliberately keeps the CPU oracle as the safety reference.
    # Future Warp kernels can consume samples_to_warp_arrays without changing
    # the dataset schema or generated labels.
    traces = [evaluate_swept_sphere_oracle(sample) for sample in samples]
    backend = "cpu_reference"
    if prefer_warp and availability.installed:
        backend = "cpu_reference_with_warp_ready_arrays"
    return WarpOracleResult(traces=traces, backend=backend, warp_available=availability)
