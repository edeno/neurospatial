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


@pytest.fixture(scope="module")
def turning_headings() -> np.ndarray:
    """Ten minutes at 30 Hz of a head turning at about 100° per second.

    Unwrapped, the heading wanders over several turns; wrapped, it crosses
    the ±π seam many times.
    """
    steps = np.random.default_rng(4).normal(0.0, np.radians(100.0) / 30, 18000)
    return np.cumsum(steps)


def test_unwrapped_radians_do_not_warn(turning_headings) -> None:
    """Unwrapped radians stay continuous: no step between samples exceeds π."""
    from neurospatial.encoding._validation import warn_if_angles_look_like_degrees

    assert 2 * np.pi < np.abs(turning_headings).max() <= 360.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_angles_look_like_degrees(turning_headings, name="headings", context="f")


@pytest.mark.parametrize(
    "to_degrees",
    [
        lambda h: np.degrees(h) % 360.0,  # [0, 360)
        lambda h: np.degrees(np.angle(np.exp(1j * h))),  # (-180, 180]
        # Confined to 20°-340° at the same turning speed: never crosses 0°.
        lambda h: np.degrees(np.pi + 2.8 * np.sin(h / 2.8)),
    ],
    ids=["zero-to-360", "signed-180", "no-wrap"],
)
def test_degree_sessions_warn(turning_headings, to_degrees) -> None:
    """Degree headings jump by more than π between samples, wrapped or not."""
    from neurospatial.encoding._validation import warn_if_angles_look_like_degrees

    with pytest.warns(UserWarning, match="look like degrees"):
        warn_if_angles_look_like_degrees(
            to_degrees(turning_headings), name="headings", context="f"
        )


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
            None,
            spikes,
            times,
            positions,
            np.degrees(headings),
            [[50.0, 50.0]],
            distance_range=(0.0, 50.0),
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


def test_all_nan_headings_warning_names_headings(session) -> None:
    """With no finite heading, the fix concerns headings, not positions."""
    from neurospatial.encoding import compute_directional_rate

    times, _, headings, spikes = session
    with pytest.warns(UserWarning, match="excluded ALL") as caught:
        compute_directional_rate(spikes, times, np.full_like(headings, np.nan))
    message = next(str(w.message) for w in caught if "excluded ALL" in str(w.message))
    assert "NaN headings" in message
    assert "positions" not in message
