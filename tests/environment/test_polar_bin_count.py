"""Polar factories retain requested cell counts when division rounds upward."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding._egocentric_counts import compute_egocentric_occupancy
from neurospatial.encoding.egocentric import compute_egocentric_rate


@pytest.mark.parametrize("n_angle", [1, 2, 6, 8, 31, 61, 122])
def test_exact_angular_division_keeps_the_requested_count(n_angle):
    env = Environment.from_polar_egocentric(
        distance_range=(0.0, 10.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=10.0,
        angle_bin_size=2 * np.pi / n_angle,
    )
    assert env.grid_shape == (1, n_angle)
    edges = np.linspace(-np.pi, np.pi, n_angle + 1)
    np.testing.assert_array_equal(env.grid_edges[1], edges)
    np.testing.assert_array_equal(env.bin_centers[:, 1], (edges[:-1] + edges[1:]) / 2)


@pytest.mark.parametrize("span,n_distance", [(1.0, 49), (10.0, 61), (100.0, 29)])
def test_exact_radial_division_keeps_the_requested_count(span, n_distance):
    env = Environment.from_polar_egocentric(
        distance_range=(0.0, span),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=span / n_distance,
        angle_bin_size=2 * np.pi,
    )
    assert env.grid_shape == (n_distance, 1)
    np.testing.assert_array_equal(
        env.grid_edges[0], np.linspace(0.0, span, n_distance + 1)
    )


def test_nonintegral_spans_still_round_up_to_cover_the_range():
    env = Environment.from_polar_egocentric(
        distance_range=(0.0, 10.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=3.0,
        angle_bin_size=1.0,
    )
    assert env.grid_shape == (4, 7)
    assert env.grid_edges[0][-1] == 10.0
    assert env.grid_edges[1][-1] == np.pi


def test_resolvable_excess_above_an_integer_still_adds_a_bin():
    env = Environment.from_polar_egocentric(
        distance_range=(0.0, 61.0 + 1e-8),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=1.0,
        angle_bin_size=2 * np.pi,
    )
    assert env.grid_shape == (62, 1)


def test_sixty_one_direction_occupancy_matches_the_published_bin_centers():
    """An object ahead occupies the zero-angle cell in the returned environment."""
    occupancy, env = compute_egocentric_occupancy(
        np.array([0.0, 0.1]),
        np.zeros((2, 2)),
        np.zeros(2),
        np.array([[5.0, 0.0]]),
        distance_range=(0.0, 10.0),
        n_distance_bins=1,
        n_direction_bins=61,
    )
    assert env.n_bins == occupancy.size == 61
    occupied = np.flatnonzero(occupancy)
    np.testing.assert_array_equal(occupied, [30])
    np.testing.assert_allclose(env.bin_centers[occupied], [[5.0, 0.0]], atol=1e-15)
    np.testing.assert_allclose(occupancy[occupied], [0.1])


def test_sixty_one_direction_rate_peak_matches_the_firing_object_bearing():
    result = compute_egocentric_rate(
        None,
        np.array([0.05]),
        np.array([0.0, 0.1]),
        np.zeros((2, 2)),
        np.zeros(2),
        np.array([[5.0, 0.0]]),
        distance_range=(0.0, 10.0),
        distance_bin_size=10.0,
        direction_bin_size=2 * np.pi / 61,
        method="binned",
    )
    assert result.env.n_bins == result.firing_rate.size == 61
    peak = int(np.nanargmax(result.firing_rate))
    np.testing.assert_allclose(result.env.bin_centers[peak], [5.0, 0.0], atol=1e-15)
    assert result.firing_rate[peak] == pytest.approx(10.0)
