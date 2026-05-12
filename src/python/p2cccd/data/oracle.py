from __future__ import annotations

from dataclasses import dataclass
import math

from .samplers import MotionDiscPairSample, sample_path_length


@dataclass(frozen=True, slots=True)
class ExactOracleTrace:
    sample_id: int
    collided: bool
    toi: float
    closest_time: float
    min_distance: float
    safe_margin: float
    exact_cost: float
    contact_interval_t0: float
    contact_interval_t1: float


def _sub(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def _dot(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> float:
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]


def _norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(max(0.0, _dot(value, value)))


def _lerp(lhs: tuple[float, float, float], rhs: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return (
        lhs[0] + (rhs[0] - lhs[0]) * t,
        lhs[1] + (rhs[1] - lhs[1]) * t,
        lhs[2] + (rhs[2] - lhs[2]) * t,
    )


def _distance_at(sample: MotionDiscPairSample, t: float) -> float:
    a = _lerp(sample.center_a_t0, sample.center_a_t1, t)
    b = _lerp(sample.center_b_t0, sample.center_b_t1, t)
    return _norm(_sub(a, b))


def evaluate_swept_sphere_oracle(sample: MotionDiscPairSample) -> ExactOracleTrace:
    """Analytic CCD oracle for two linearly moving bounding spheres.

    This is the baseline dataset oracle until pybind exposes the C++ exact feature
    engine. It is conservative for the generated proxy samples because every
    sampled mesh/link pair is represented by its enclosing moving disc/sphere.
    """

    radius_sum = sample.radius_a + sample.radius_b
    d0 = _sub(sample.center_a_t0, sample.center_b_t0)
    d1 = _sub(sample.center_a_t1, sample.center_b_t1)
    v = _sub(d1, d0)

    a = _dot(v, v)
    b = 2.0 * _dot(d0, v)
    c = _dot(d0, d0) - radius_sum * radius_sum

    if a <= 1.0e-14:
        closest_time = 0.0
    else:
        closest_time = min(1.0, max(0.0, -b / (2.0 * a)))
    min_distance = _distance_at(sample, closest_time)

    roots: list[float] = []
    if c <= 0.0:
        roots.append(0.0)
    elif a > 1.0e-14:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            sqrt_disc = math.sqrt(discriminant)
            for root in ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)):
                if 0.0 <= root <= 1.0:
                    roots.append(root)

    collided = len(roots) > 0
    toi = min(roots) if collided else 1.0
    safe_margin = min_distance - radius_sum

    contact_interval_t0 = toi if collided else closest_time
    contact_interval_t1 = contact_interval_t0
    if collided and a > 1.0e-14:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            sqrt_disc = math.sqrt(discriminant)
            enter = max(0.0, min(1.0, (-b - sqrt_disc) / (2.0 * a)))
            exit_ = max(0.0, min(1.0, (-b + sqrt_disc) / (2.0 * a)))
            contact_interval_t0 = min(enter, exit_)
            contact_interval_t1 = max(enter, exit_)

    near_contact = max(0.0, 1.0 - max(0.0, safe_margin) / max(radius_sum, 1.0e-6))
    exact_cost = 1.0 + 8.0 * sample.hardness + 5.0 * near_contact + (4.0 if collided else 0.0)
    exact_cost += 0.25 * sample_path_length(sample)

    return ExactOracleTrace(
        sample_id=sample.sample_id,
        collided=collided,
        toi=float(toi),
        closest_time=float(closest_time),
        min_distance=float(min_distance),
        safe_margin=float(safe_margin),
        exact_cost=float(exact_cost),
        contact_interval_t0=float(contact_interval_t0),
        contact_interval_t1=float(contact_interval_t1),
    )
