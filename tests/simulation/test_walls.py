"""Walls keep a simulated walker inside the arena, even in sharp corners."""

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from neurospatial import Environment
from neurospatial.simulation import simulate_trajectory_ou
from neurospatial.simulation._walls import Walls


def _wedge(degrees):
    """A 100 cm triangle whose corner at the origin has the given angle."""
    angle = np.deg2rad(degrees)
    return Polygon([(0, 0), (100, 0), (100 * np.cos(angle), 100 * np.sin(angle))])


@pytest.mark.parametrize("degrees", [8.0, 20.0])
def test_a_step_into_a_sharp_corner_bounces_out(degrees):
    """Mirrored off each side in turn, the step leaves the corner in one move."""
    import shapely

    wedge = _wedge(degrees)
    walls = Walls.from_geometry(wedge.exterior)
    angle = np.deg2rad(degrees)
    # Inside the corner, 3 cm out along its bisector, heading into the corner.
    bisector = np.array([np.cos(angle / 2), np.sin(angle / 2)])
    position = 3.0 * bisector
    unit = -bisector
    moved, heading = walls.move(position, 5.0 * unit, unit, "reflect")
    # The walker takes the whole step along its new heading, staying inside.
    np.testing.assert_allclose(moved - position, 5.0 * heading, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(heading), 1.0)
    assert shapely.contains_xy(wedge, *moved)


def test_fast_walks_in_a_sharp_corner_never_stall():
    """A blocked step would freeze the walker: every reflected step moves it."""
    import shapely

    wedge = _wedge(8.0)
    env = Environment.from_polygon(wedge, bin_size=2.0, units="cm")
    positions, _times = simulate_trajectory_ou(
        env,
        duration=300.0,
        dt=0.05,
        speed_mean=60.0,
        speed_std=30.0,
        speed_units="cm",
        seed=0,
    )
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    assert np.all(steps > 0)
    assert shapely.covers(wedge, shapely.points(positions)).all()


def test_enclosing_walls_of_a_track_without_planar_cells_are_none():
    """A track's bins are lengths, not areas: there are no cells to enclose."""
    track = Environment.linear_track(
        node_positions=[(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)],
        bin_size=5.0,
        units="cm",
    )
    assert Walls.enclosing(track) is None


def test_enclosing_walls_of_a_box():
    env = Environment.from_polygon(box(0, 0, 10, 10), bin_size=2.0, units="cm")
    walls = Walls.enclosing(env)
    assert walls is not None
    assert len(walls.starts) == 4
