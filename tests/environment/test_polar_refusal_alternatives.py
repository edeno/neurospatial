"""Each Cartesian-only refusal's suggested alternative runs on a polar environment.

A refusal names the code to use instead (``CartesianOnlyError.example``). These
tests run that code on a polar surround with a row outside its ranges and check
the answer: the containing bin or value inside, ``-1``, ``False`` or NaN
outside.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from environment_helpers import call_unbound

import neurospatial
from neurospatial import CartesianOnlyError, Environment
from neurospatial.environment.core import CARTESIAN_OPERATIONS
from neurospatial.geometry._lookup import locate

# (distance cm, angle rad); the second angle is a whole turn past the range,
# and the last row lies beyond the 40 cm surround.
POINTS = np.array([[15.0, 0.3], [25.0, -2.0 + 2 * np.pi], [41.0, 0.0]])
# A polar sample sequence that leaves the surround once.
SAMPLES = np.array(
    [[5.0, 0.1], [15.0, 0.1], [41.0, 0.0], [15.0, 0.9], [25.0, 0.9], [35.0, 0.9]]
)


def _surround() -> Environment:
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 40.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=10.0,
        angle_bin_size=np.pi / 4,
        units="cm",
    )


def _run(code: str, names: dict[str, Any]) -> Any:
    """Run an alternative: ``; ``-separated statements, the last an expression."""
    *statements, expression = code.split("; ")
    for statement in statements:
        exec(statement, names)
    return eval(expression, names)


def _containing_bins(env: Environment, rows: np.ndarray) -> np.ndarray:
    """Polar containment, the lookup ``points=`` queries use; -1 outside."""
    return locate(env.binning, env.arena, rows, call="points=")


def _recording() -> dict[str, Any]:
    """An allocentric pass by one object, for the egocentric alternatives."""
    times = np.arange(0.0, 20.0, 0.1)
    positions = np.c_[np.linspace(5.0, 95.0, times.size), np.full(times.size, 50.0)]
    room = Environment.from_samples(
        np.random.default_rng(0).uniform(0, 100, (2000, 2)), bin_size=5.0
    )
    return {
        "room": room,
        "times": times,
        "positions": positions,
        "headings": np.zeros(times.size),
        "object_positions": np.array([[50.0, 50.0], [90.0, 10.0]]),
        "spike_times": np.array([1.0, 2.0]),
    }


def _check_interpolate(env, result, names):
    bins = _containing_bins(env, POINTS)
    expected = np.where(bins >= 0, names["field"][bins], np.nan)
    np.testing.assert_array_equal(result, expected)
    assert np.isnan(result[-1])


def _check_rebin(env, result, names):
    assert result.coordinate_kind == "polar"
    assert result.n_bins == 8
    assert result.dimension_ranges == env.dimension_ranges


def _check_mask(predicate: Callable[[np.ndarray], np.ndarray]):
    def check(env, result, names):
        keep = predicate(env.bin_centers)
        assert 0 < keep.sum() < env.n_bins
        np.testing.assert_array_equal(result, names["field"][keep])

    return check


def _check_transform(env, result, names):
    shifted = names["positions"] + np.array([10.0, 0.0])
    offsets = names["object_positions"][None] - shifted[:, None]
    distance = np.hypot(offsets[..., 0], offsets[..., 1])
    bearing = np.angle(
        np.exp(
            1j
            * (
                np.arctan2(offsets[..., 1], offsets[..., 0])
                - names["headings"][:, None]
            )
        )
    )
    np.testing.assert_allclose(result[..., 0], distance)
    np.testing.assert_allclose(result[..., 1], bearing, atol=1e-12)


def _check_distance(env, result, names):
    assert result.shape == (env.n_bins,)
    assert result[0] == 0.0
    assert np.all(np.isfinite(result)) and np.all(result[1:] > 0)


CHECKS = {
    "interpolate": (lambda: {"positions": POINTS}, _check_interpolate),
    "rebin": (dict, _check_rebin),
    "subset": (dict, _check_mask(lambda centers: centers[:, 0] < 20.0)),
    "clip": (dict, _check_mask(lambda centers: np.abs(centers[:, 1]) < np.pi / 2)),
    "apply_transform": (_recording, _check_transform),
}


def _names(env: Environment, extra: dict[str, Any]) -> dict[str, Any]:
    names = {
        "np": np,
        "neurospatial": neurospatial,
        "Environment": Environment,
        "env": env,
        "field": np.arange(env.n_bins, dtype=np.float64) + 1.0,
    }
    names.update(extra)
    return names


def _member_refusal(env: Environment, name: str) -> CartesianOnlyError:
    with pytest.raises(CartesianOnlyError) as refused:
        call_unbound(env, name)
    return refused.value


def test_every_member_refusal_has_a_checked_alternative():
    env = _surround()
    assert all(
        _member_refusal(env, name).example is not None for name in CARTESIAN_OPERATIONS
    )
    assert set(CHECKS) == CARTESIAN_OPERATIONS


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_member_alternative_gives_the_right_answer(name):
    env = _surround()
    refusal = _member_refusal(env, name)
    assert refusal.example is not None
    assert refusal.example in str(refusal)  # the message shows the same code
    make_names, check = CHECKS[name]
    names = _names(env, make_names())
    check(env, _run(refusal.example, names), names)


def _check_start_occupancy(env, result, names):
    bins = _containing_bins(env, SAMPLES)
    expected = np.zeros(env.n_bins)
    # Each interval counts at its start; the sample beyond 40 cm adds no time.
    np.add.at(expected, bins[:-1][bins[:-1] >= 0], 0.1)
    np.testing.assert_allclose(result, expected)
    assert result.sum() == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("call", "check", "extra"),
    [
        (
            lambda env: env.distance_to(bins=[0], metric="euclidean"),
            _check_distance,
            {},
        ),
        (
            lambda env: env.occupancy(
                np.arange(SAMPLES.shape[0]) * 0.1,
                SAMPLES,
                time_allocation="linear",
            ),
            _check_start_occupancy,
            {"times": np.arange(SAMPLES.shape[0]) * 0.1, "positions": SAMPLES},
        ),
    ],
)
def test_mode_alternative_gives_the_right_answer(call, check, extra):
    env = _surround()
    with pytest.raises(CartesianOnlyError) as refused:
        call(env)
    example = refused.value.example
    assert example is not None and example in str(refused.value)
    names = _names(env, extra)
    check(env, _run(example, names), names)


def test_nearest_bin_alone_would_put_the_outside_row_in_an_edge_bin():
    """The interpolate alternative uses containment, not the nearest center."""
    env = _surround()
    assert _containing_bins(env, POINTS)[-1] == -1
    assert env.nearest_bin(POINTS)[-1] >= 0
