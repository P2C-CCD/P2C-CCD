from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from .samplers import MotionDiscPairSample


ResponseMode = Literal["raw", "stop_at_toi", "bounce"]


def _add(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _sub(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def _scale(vec: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    return (vec[0] * scalar, vec[1] * scalar, vec[2] * scalar)


def _dot(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> float:
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]


def _norm(vec: tuple[float, float, float]) -> float:
    return math.sqrt(max(0.0, _dot(vec, vec)))


def _normalize(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _norm(vec)
    if length <= 1.0e-12:
        return (1.0, 0.0, 0.0)
    inv = 1.0 / length
    return (vec[0] * inv, vec[1] * inv, vec[2] * inv)


def _lerp(
    lhs: tuple[float, float, float],
    rhs: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    return (
        lhs[0] + (rhs[0] - lhs[0]) * t,
        lhs[1] + (rhs[1] - lhs[1]) * t,
        lhs[2] + (rhs[2] - lhs[2]) * t,
    )


def proxy_mass_from_radius(
    radius: float,
    *,
    density: float = 1.0,
    min_mass: float = 1.0e-6,
) -> float:
    clamped_radius = max(1.0e-6, float(radius))
    clamped_density = max(1.0e-9, float(density))
    mass = clamped_density * (4.0 / 3.0) * math.pi * clamped_radius * clamped_radius * clamped_radius
    return max(min_mass, mass)


@dataclass(frozen=True, slots=True)
class ElasticImpactResponse:
    collided: bool
    toi: float
    mass_a: float
    mass_b: float
    restitution: float
    normal: tuple[float, float, float]
    center_a_t0: tuple[float, float, float]
    center_b_t0: tuple[float, float, float]
    center_a_t1: tuple[float, float, float]
    center_b_t1: tuple[float, float, float]
    center_a_toi: tuple[float, float, float]
    center_b_toi: tuple[float, float, float]
    velocity_a_pre: tuple[float, float, float]
    velocity_b_pre: tuple[float, float, float]
    velocity_a_post: tuple[float, float, float]
    velocity_b_post: tuple[float, float, float]


def build_elastic_impact_response(
    *,
    center_a_t0: tuple[float, float, float],
    center_a_t1: tuple[float, float, float],
    center_b_t0: tuple[float, float, float],
    center_b_t1: tuple[float, float, float],
    toi: float,
    collided: bool,
    mass_a: float,
    mass_b: float,
    restitution: float = 1.0,
) -> ElasticImpactResponse:
    clamped_toi = min(1.0, max(0.0, float(toi)))
    clamped_mass_a = max(1.0e-6, float(mass_a))
    clamped_mass_b = max(1.0e-6, float(mass_b))
    clamped_restitution = min(1.0, max(0.0, float(restitution)))
    velocity_a_pre = _sub(center_a_t1, center_a_t0)
    velocity_b_pre = _sub(center_b_t1, center_b_t0)
    center_a_toi = _lerp(center_a_t0, center_a_t1, clamped_toi)
    center_b_toi = _lerp(center_b_t0, center_b_t1, clamped_toi)
    normal = _normalize(_sub(center_b_toi, center_a_toi))
    velocity_a_post = velocity_a_pre
    velocity_b_post = velocity_b_pre
    if collided:
        relative_pre = _sub(velocity_a_pre, velocity_b_pre)
        closing_speed = _dot(relative_pre, normal)
        if closing_speed > 0.0:
            impulse = -((1.0 + clamped_restitution) * closing_speed) / (
                (1.0 / clamped_mass_a) + (1.0 / clamped_mass_b)
            )
            velocity_a_post = _add(velocity_a_pre, _scale(normal, impulse / clamped_mass_a))
            velocity_b_post = _sub(velocity_b_pre, _scale(normal, impulse / clamped_mass_b))
    return ElasticImpactResponse(
        collided=bool(collided),
        toi=clamped_toi,
        mass_a=clamped_mass_a,
        mass_b=clamped_mass_b,
        restitution=clamped_restitution,
        normal=normal,
        center_a_t0=center_a_t0,
        center_b_t0=center_b_t0,
        center_a_t1=center_a_t1,
        center_b_t1=center_b_t1,
        center_a_toi=center_a_toi,
        center_b_toi=center_b_toi,
        velocity_a_pre=velocity_a_pre,
        velocity_b_pre=velocity_b_pre,
        velocity_a_post=velocity_a_post,
        velocity_b_post=velocity_b_post,
    )


def build_sample_elastic_impact_response(
    sample: MotionDiscPairSample,
    *,
    toi: float,
    collided: bool,
    restitution: float | None = None,
    mass_a: float | None = None,
    mass_b: float | None = None,
) -> ElasticImpactResponse:
    resolved_mass_a = float(sample.mass_a) if sample.mass_a is not None else proxy_mass_from_radius(sample.radius_a)
    resolved_mass_b = float(sample.mass_b) if sample.mass_b is not None else proxy_mass_from_radius(sample.radius_b)
    return build_elastic_impact_response(
        center_a_t0=sample.center_a_t0,
        center_a_t1=sample.center_a_t1,
        center_b_t0=sample.center_b_t0,
        center_b_t1=sample.center_b_t1,
        toi=toi,
        collided=collided,
        mass_a=resolved_mass_a if mass_a is None else float(mass_a),
        mass_b=resolved_mass_b if mass_b is None else float(mass_b),
        restitution=float(sample.restitution) if restitution is None else float(restitution),
    )


def replay_positions_at_time(
    response: ElasticImpactResponse,
    t: float,
    *,
    mode: ResponseMode = "bounce",
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    clamped_t = min(1.0, max(0.0, float(t)))
    if mode == "raw":
        return (
            _lerp(response.center_a_t0, response.center_a_t1, clamped_t),
            _lerp(response.center_b_t0, response.center_b_t1, clamped_t),
        )
    if mode == "stop_at_toi" or not response.collided:
        stop_t = min(clamped_t, response.toi)
        return (
            _lerp(response.center_a_t0, response.center_a_t1, stop_t),
            _lerp(response.center_b_t0, response.center_b_t1, stop_t),
        )
    if clamped_t <= response.toi:
        return (
            _lerp(response.center_a_t0, response.center_a_t1, clamped_t),
            _lerp(response.center_b_t0, response.center_b_t1, clamped_t),
        )
    dt = clamped_t - response.toi
    return (
        _add(response.center_a_toi, _scale(response.velocity_a_post, dt)),
        _add(response.center_b_toi, _scale(response.velocity_b_post, dt)),
    )


def momentum(response: ElasticImpactResponse, *, post_impact: bool = False) -> tuple[float, float, float]:
    velocity_a = response.velocity_a_post if post_impact else response.velocity_a_pre
    velocity_b = response.velocity_b_post if post_impact else response.velocity_b_pre
    return _add(_scale(velocity_a, response.mass_a), _scale(velocity_b, response.mass_b))


def kinetic_energy(response: ElasticImpactResponse, *, post_impact: bool = False) -> float:
    velocity_a = response.velocity_a_post if post_impact else response.velocity_a_pre
    velocity_b = response.velocity_b_post if post_impact else response.velocity_b_pre
    return 0.5 * response.mass_a * _dot(velocity_a, velocity_a) + 0.5 * response.mass_b * _dot(
        velocity_b,
        velocity_b,
    )


__all__ = [
    "ElasticImpactResponse",
    "ResponseMode",
    "build_elastic_impact_response",
    "build_sample_elastic_impact_response",
    "kinetic_energy",
    "momentum",
    "proxy_mass_from_radius",
    "replay_positions_at_time",
]
