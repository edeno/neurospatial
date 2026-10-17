"""Polar environments locate ``(distance, angle)`` rows in the cell that holds them.

An angle names a direction, so an angle a whole number of turns outside the
angle range finds the same bin as the angle inside it. Rows beyond the
distance range or in an excluded sector of a partial window have no bin.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment, Regions

TURN = 2 * np.pi


def _circle() -> Environment:
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 40.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=10.0,
        angle_bin_size=np.pi / 4,
        units="cm",
    )


def _frontal() -> Environment:
    """Only the half-plane ahead of the animal: -π/2 to π/2."""
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 40.0),
        angle_range=(-np.pi / 2, np.pi / 2),
        distance_bin_size=10.0,
        angle_bin_size=np.pi / 4,
        circular_angle=False,
        units="cm",
    )


@pytest.mark.parametrize("build", [_circle, _frontal], ids=["circle", "frontal"])
@pytest.mark.parametrize("turns", [-2, -1, 1, 3])
def test_points_queries_wrap_whole_turns(build, turns):
    env = build()
    for index, center in enumerate(env.bin_centers):
        wrapped = center + np.array([0.0, turns * TURN])
        np.testing.assert_array_equal(
            env.distance_to(points=[wrapped]), env.distance_to(bins=[index])
        )


def test_the_seam_angle_is_one_direction_on_a_full_circle():
    env = _circle()
    np.testing.assert_array_equal(
        env.distance_to(points=[[15.0, np.pi]]),
        env.distance_to(points=[[15.0, -np.pi]]),
    )


def test_point_regions_wrap_angles():
    regions = Regions()
    regions.add("ahead", point=(25.0, 0.4))
    regions.add("ahead_turned", point=(25.0, 0.4 + TURN))
    env = _circle().with_regions(regions)
    (bin_index,) = env.bins_in_region("ahead")
    np.testing.assert_array_equal(env.bins_in_region("ahead_turned"), [bin_index])


def test_an_excluded_sector_stays_outside_a_partial_window():
    env = _frontal()
    for angle in (np.pi, -3 * np.pi / 4, np.pi + TURN):
        with pytest.raises(ValueError, match="outside the environment"):
            env.distance_to(points=[[15.0, angle]])
