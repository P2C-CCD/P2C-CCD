from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.data.response import (  # noqa: E402
    build_elastic_impact_response,
    kinetic_energy,
    momentum,
    replay_positions_at_time,
)


def test_equal_mass_head_on_elastic_collision_swaps_velocity() -> None:
    response = build_elastic_impact_response(
        center_a_t0=(0.0, 0.0, 0.0),
        center_a_t1=(1.0, 0.0, 0.0),
        center_b_t0=(1.0, 0.0, 0.0),
        center_b_t1=(1.0, 0.0, 0.0),
        toi=1.0,
        collided=True,
        mass_a=1.0,
        mass_b=1.0,
        restitution=1.0,
    )
    assert response.velocity_a_post == pytest.approx((0.0, 0.0, 0.0))
    assert response.velocity_b_post == pytest.approx((1.0, 0.0, 0.0))
    assert momentum(response) == pytest.approx(momentum(response, post_impact=True))
    assert kinetic_energy(response) == pytest.approx(kinetic_energy(response, post_impact=True))


def test_unequal_mass_elastic_collision_conserves_momentum_and_energy() -> None:
    response = build_elastic_impact_response(
        center_a_t0=(0.0, 0.0, 0.0),
        center_a_t1=(2.0, 0.0, 0.0),
        center_b_t0=(3.0, 0.0, 0.0),
        center_b_t1=(2.5, 0.0, 0.0),
        toi=0.5,
        collided=True,
        mass_a=2.0,
        mass_b=5.0,
        restitution=1.0,
    )
    before_p = momentum(response)
    after_p = momentum(response, post_impact=True)
    assert after_p == pytest.approx(before_p)
    assert kinetic_energy(response, post_impact=True) == pytest.approx(kinetic_energy(response))


def test_bounce_replay_is_continuous_at_toi() -> None:
    response = build_elastic_impact_response(
        center_a_t0=(0.0, 0.0, 0.0),
        center_a_t1=(1.0, 0.0, 0.0),
        center_b_t0=(1.0, 0.0, 0.0),
        center_b_t1=(0.0, 0.0, 0.0),
        toi=0.5,
        collided=True,
        mass_a=1.0,
        mass_b=1.0,
        restitution=1.0,
    )
    pre = replay_positions_at_time(response, 0.5 - 1.0e-9, mode="bounce")
    at = replay_positions_at_time(response, 0.5, mode="bounce")
    post = replay_positions_at_time(response, 0.5 + 1.0e-9, mode="bounce")
    assert at[0] == pytest.approx(response.center_a_toi)
    assert at[1] == pytest.approx(response.center_b_toi)
    assert math.dist(pre[0], at[0]) < 1.0e-6
    assert math.dist(post[0], at[0]) < 1.0e-6
