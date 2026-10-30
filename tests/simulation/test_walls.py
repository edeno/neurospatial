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


def _box_with_wall():
    from shapely.geometry import LineString

    from neurospatial.geometry import OpenArena

    outline = box(0, 0, 100, 100)
    wall = LineString([(50, 0), (50, 70)])
    return Environment.from_polygon(
        outline, bin_size=2.0, units="cm", arena=OpenArena(outline, walls=(wall,))
    )


def _rotated_box():
    from shapely import affinity

    rotated = affinity.translate(
        affinity.rotate(box(0, 0, 100, 100), 30, (0, 0)), 7, -2
    )
    return Environment.from_polygon(rotated, bin_size=2.0, units="cm")


@pytest.mark.parametrize(
    ("make_env", "boundary_source"),
    [
        (_box_with_wall, "physical"),
        # Bin edges run along this outline, so it is also the analysis wall.
        (_box_with_wall, "analysis"),
        (_rotated_box, "physical"),
    ],
)
def test_ray_distances_match_the_analysis_ray_casts(make_env, boundary_source):
    """Boundary-vector cells and the analysis measure the same wall distances,
    including from observers standing on the outline, a corner or a wall."""
    import shapely

    from neurospatial.geometry._boundaries import valid_observers
    from neurospatial.ops.boundary_vectors import (
        boundary_directions,
        compute_boundary_distances,
    )
    from neurospatial.simulation.models import BoundaryVectorCellModel

    env = make_env()
    model = BoundaryVectorCellModel(
        env,
        preferred_distance=10.0,
        preferred_direction=0.0,
        boundary_source=boundary_source,
    )
    rng = np.random.default_rng(0)
    # Points on the outline, between and at its corners.
    outline = shapely.get_coordinates(
        shapely.get_exterior_ring(env.arena.polygonal_domain())
    )
    pieces = rng.integers(0, len(outline) - 1, 20)
    on_outline = outline[pieces] + rng.uniform(0, 1, (20, 1)) * (
        outline[pieces + 1] - outline[pieces]
    )
    positions = np.concatenate(
        [
            # Off the bin lattice: a ray through a lattice corner of the
            # analysis walls is a tie the two casts may break differently.
            env.bin_centers[rng.choice(env.n_bins, 20)]
            + rng.uniform(-0.9, 0.9, (20, 2)),
            on_outline,
            outline,
            [[50.0, 30.0], [50.0, 70.0]],  # on the internal wall and its end
        ]
    )
    positions = positions[
        valid_observers(env.binning, env.arena, positions, boundary_source)
    ]
    assert len(positions) >= 30  # observers on walls are part of the comparison
    directions = boundary_directions(n_directions=36)
    expected, _ = compute_boundary_distances(
        env,
        positions,
        direction_frame="allocentric",
        n_directions=36,
        boundary_source=boundary_source,
    )
    found = model._walls.ray_distances(positions, directions, model._on_floor)
    np.testing.assert_allclose(found, expected, rtol=0, atol=1e-9)


def test_boundary_vector_rate_is_continuous_at_the_wall():
    """A cell tuned to the south wall fires at the wall as just inside it."""
    from neurospatial.simulation.models import BoundaryVectorCellModel

    env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")
    model = BoundaryVectorCellModel(
        env, preferred_distance=10.0, preferred_direction=-np.pi / 2
    )
    rates = model.firing_rate(np.zeros(2), np.array([[60.0, 0.0], [60.0, 1e-9]]))
    np.testing.assert_allclose(rates[0], rates[1], rtol=1e-6)
