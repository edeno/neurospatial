"""Headings that look like degrees but are read as radians warn."""

from __future__ import annotations

import warnings

import numpy as np
import pytest


@pytest.fixture(scope="module")
def session() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Five minutes at 30 Hz: times, positions (cm), headings (rad), spikes."""
    rng = np.random.default_rng(0)
    times = np.arange(0.0, 300.0, 1 / 30)
    positions = rng.uniform(0.0, 100.0, (times.size, 2))
    headings = rng.uniform(-np.pi, np.pi, times.size)
    spikes = np.sort(np.random.default_rng(1).uniform(0.0, times[-1], 300))
    return times, positions, headings, spikes


@pytest.mark.parametrize(
    "angles",
    [
        np.array([-np.pi, 0.0, np.pi]),
        np.array([0.0, 1.0, 2 * np.pi]),
        np.array([np.nan, 3.0, -3.0]),
        np.array([np.nan, np.nan]),
        np.linspace(-500.0, 500.0, 11),  # unwrapped radians over many turns
    ],
    ids=["signed", "zero-to-two-pi", "with-nan", "all-nan", "beyond-360"],
)
def test_radian_ranges_do_not_warn(angles) -> None:
    from neurospatial.encoding._validation import warn_if_angles_look_like_degrees

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_angles_look_like_degrees(angles, name="headings", context="f")


@pytest.mark.parametrize(
    "angles", [np.array([0.0, 90.0, 359.0]), np.array([-180.0, 0.0, 179.0])]
)
def test_degree_ranges_warn(angles) -> None:
    from neurospatial.encoding._validation import warn_if_angles_look_like_degrees

    with pytest.warns(UserWarning, match=r"np\.radians\(headings\)"):
        warn_if_angles_look_like_degrees(angles, name="headings", context="f")


def test_directional_rate_warns_for_degrees_read_as_radians(session) -> None:
    """Degrees read as radians wrap into a meaningless map; the call says so."""
    from neurospatial.encoding import compute_directional_rate

    times, _, headings, spikes = session
    with pytest.warns(UserWarning, match='angle_unit="deg"'):
        compute_directional_rate(spikes, times, np.degrees(headings))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_directional_rate(spikes, times, np.degrees(headings), angle_unit="deg")
        compute_directional_rate(spikes, times, headings)


def test_egocentric_entry_point_warns_for_degree_headings(session) -> None:
    from neurospatial.encoding.egocentric import compute_egocentric_rate

    times, positions, headings, spikes = session
    with pytest.warns(UserWarning, match=r"np\.radians\(headings\)"):
        compute_egocentric_rate(
            None, spikes, times, positions, np.degrees(headings), [[50.0, 50.0]]
        )


def test_shuffle_warns_once_for_degree_headings(session) -> None:
    from neurospatial.encoding import head_direction_cell_significance

    times, _, headings, spikes = session
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        head_direction_cell_significance(
            [spikes], times, np.degrees(headings), n_shuffles=5, rng=0
        )
    messages = [str(w.message) for w in caught if "look like degrees" in str(w.message)]
    assert len(messages) == 1
