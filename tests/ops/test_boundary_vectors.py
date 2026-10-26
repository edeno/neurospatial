"""Distances to walls are measured to the declared walls, ray by ray."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial._exceptions import CartesianOnlyError
from neurospatial.arena import OpenArena
from neurospatial.ops import (
    boundary_directions,
    compute_boundary_distances,
    nearest_boundary,
)

SIDE = 100.0  # cm


@pytest.fixture(scope="module")
def box_env() -> Environment:
    return Environment.from_polygon(box(0, 0, SIDE, SIDE), bin_size=5.0, units="cm")


def _distance_to_box(positions: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Exact distance from each position along each angle to the box sides."""
    cos, sin = np.cos(angles)[None, :], np.sin(angles)[None, :]
    x, y = positions[:, :1], positions[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        to_x = np.where(cos > 0, (SIDE - x) / cos, np.where(cos < 0, -x / cos, np.inf))
        to_y = np.where(sin > 0, (SIDE - y) / sin, np.where(sin < 0, -y / sin, np.inf))
    return np.minimum(to_x, to_y)


def test_directions_are_sector_centers():
    np.testing.assert_allclose(
        np.degrees(boundary_directions(n_directions=4)), [-135.0, -45.0, 45.0, 135.0]
    )


def test_allocentric_distances_match_the_box_geometry(box_env):
    positions = np.random.default_rng(0).uniform(5, 95, (50, 2))

    distances, directions = compute_boundary_distances(
        box_env, positions, direction_frame="allocentric", n_directions=24
    )

    np.testing.assert_allclose(distances, _distance_to_box(positions, directions))


def test_egocentric_rays_turn_with_the_heading(box_env):
    """Egocentric 0 is straight ahead: facing North, it measures to y = 100."""
    rng = np.random.default_rng(1)
    positions = rng.uniform(5, 95, (50, 2))
    headings = rng.uniform(-np.pi, np.pi, 50)

    distances, directions = compute_boundary_distances(
        box_env, positions, headings, direction_frame="egocentric", n_directions=24
    )

    allocentric = headings[:, None] + directions[None, :]
    expected = np.stack(
        [
            _distance_to_box(position[None, :], angles)[0]
            for position, angles in zip(positions, allocentric, strict=True)
        ]
    )
    np.testing.assert_allclose(distances, expected)


def test_an_interior_wall_stops_the_ray():
    arena = OpenArena(box(0, 0, SIDE, SIDE), walls=(LineString([(50, 0), (50, 60)]),))
    positions = np.random.default_rng(2).uniform(0, SIDE, (4000, 2))
    env = Environment.from_samples(positions, bin_size=5.0, arena=arena)

    distances, directions = compute_boundary_distances(
        env,
        np.array([[30.0, 30.0], [30.0, 80.0]]),
        direction_frame="allocentric",
        n_directions=4,
    )
    east = np.argmin(np.abs(directions - np.pi / 4))  # 45 degrees, up and right

    # Below the wall's end the ray stops at x = 50 (without the wall it would
    # run 70 * sqrt(2) to the corner); above it, it reaches y = 100.
    assert distances[0, east] == pytest.approx(20 * np.sqrt(2))
    assert distances[1, east] == pytest.approx(20 * np.sqrt(2))  # y = 100 first


def test_physical_walls_ignore_an_unvisited_half_and_analysis_edges_do_not():
    """An unvisited half of the box is not a wall unless the bins are asked."""
    visited = np.random.default_rng(3).uniform([0, 0], [50, SIDE], (4000, 2))
    env = Environment.from_samples(
        visited, bin_size=5.0, arena=OpenArena(box(0, 0, SIDE, SIDE))
    )
    position = np.array([[25.0, 50.0]])

    physical, directions = compute_boundary_distances(
        env, position, direction_frame="allocentric", n_directions=4
    )
    analysis, _ = compute_boundary_distances(
        env,
        position,
        direction_frame="allocentric",
        n_directions=4,
        boundary_source="analysis",
    )
    northeast = np.argmin(np.abs(directions - np.pi / 4))

    # 45 degrees from (25, 50): the box side at y = 100, or the visited edge
    # at x = 50.
    assert physical[0, northeast] == pytest.approx(50 * np.sqrt(2))
    assert analysis[0, northeast] == pytest.approx(25 * np.sqrt(2))


def test_missing_samples_are_nan_and_unreached_walls_are_inf(box_env):
    positions = np.array([[np.nan, 50.0], [200.0, 50.0], [50.0, 50.0], [50.0, 50.0]])
    headings = np.array([0.0, 0.0, np.nan, 0.0])

    distances, _ = compute_boundary_distances(
        box_env,
        positions,
        headings,
        direction_frame="egocentric",
        n_directions=8,
        max_distance=40.0,
    )

    assert np.isnan(distances[:3]).all()
    assert np.isinf(distances[3]).all()  # every wall is 50 cm or more away


@pytest.mark.parametrize(
    ("headings", "frame", "message"),
    [
        (np.zeros(1), "allocentric", "does not use headings"),
        (None, "egocentric", "needs headings"),
    ],
)
def test_headings_must_match_the_direction_frame(box_env, headings, frame, message):
    with pytest.raises(ValueError, match=message):
        compute_boundary_distances(
            box_env, np.array([[50.0, 50.0]]), headings, direction_frame=frame
        )


@pytest.mark.parametrize(
    ("measure", "options"),
    [
        (compute_boundary_distances, {"direction_frame": "allocentric"}),
        (nearest_boundary, {}),
    ],
    ids=["compute_boundary_distances", "nearest_boundary"],
)
def test_wrong_position_columns_name_the_function_called(box_env, measure, options):
    with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\) got positions"):
        measure(box_env, np.zeros((2, 3)), **options)


def test_nearest_boundary_points_to_the_closest_side(box_env):
    positions = np.array([[20.0, 50.0], [50.0, 90.0], [np.nan, 50.0]])

    distances, directions = nearest_boundary(box_env, positions)

    np.testing.assert_allclose(distances[:2], [20.0, 10.0])
    np.testing.assert_allclose(directions[:2], [np.pi, np.pi / 2])
    assert np.isnan(distances[2]) and np.isnan(directions[2])


def test_polar_environments_are_refused():
    polar = Environment.from_polar_egocentric(
        distance_range=(0.0, 40.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=10.0,
        angle_bin_size=np.pi / 4,
    )

    with pytest.raises(CartesianOnlyError):
        nearest_boundary(polar, np.array([[10.0, 0.0]]))
