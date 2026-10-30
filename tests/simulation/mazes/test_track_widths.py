"""Every simulated maze's track has a width, so positions can be assigned to bins.

A track without ``track_width`` refuses position lookups. Mazes with one
corridor width pass it; mazes whose corridors differ pass the widest one, so
a sample within half the widest width of the centerline counts as on the
track.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial.simulation import simulate_session
from neurospatial.simulation.mazes._geometry import corridor_width, widest_corridor

_TRACK_MAZES = [
    "make_crossword_maze",
    "make_hamlet_maze",
    "make_hampton_court_maze",
    "make_small_hex_maze",
    "make_honeycomb_maze",
    "make_linear_track",
    "make_plus_maze",
    "make_radial_arm_maze",
    "make_rat_hexmaze",
    "make_repeated_t_maze",
    "make_repeated_y_maze",
    "make_sungod_maze",
    "make_t_maze",
    "make_w_maze",
    "make_y_maze",
]


@pytest.fixture
def track(default_maze):
    return default_maze.env_track


@pytest.mark.parametrize("maze_name", _TRACK_MAZES, indirect=True)
def test_every_maze_track_has_a_width(track):
    assert track.arena.domain_known
    assert track.arena.width > 0


@pytest.mark.parametrize(
    "maze_name",
    ["make_small_hex_maze", "make_sungod_maze", "make_honeycomb_maze"],
    indirect=True,
)
def test_mazes_without_one_corridor_width_simulate_and_look_up(track):
    session = simulate_session(
        track,
        duration=5.0,
        n_cells=3,
        trajectory_method="laps",
        show_progress=False,
        seed=0,
    )
    bins = track.bin_at(session.positions)
    assert (bins >= 0).all()
    np.testing.assert_allclose(
        track.occupancy(session.times, session.positions).sum(),
        session.times[-1] - session.times[0],
    )
    # A sample far off every corridor is not on the track.
    assert track.bin_at(np.array([[1000.0, 1000.0]]))[0] == -1


@pytest.mark.parametrize("maze_name", ["make_honeycomb_maze"], indirect=True)
def test_honeycomb_track_is_one_platform_wide(track):
    # Platforms are hexagons 11.5 cm along each side, which is their corner
    # radius, so a platform is 23 cm corner to corner.
    assert track.arena.width == pytest.approx(2 * 11.5)


def test_corridor_width_is_the_short_side_of_a_rotated_corridor():
    from shapely.affinity import rotate

    assert corridor_width(rotate(box(0.0, 0.0, 50.0, 8.0), 30.0)) == pytest.approx(8.0)


def test_widest_corridor_takes_the_widest_mid_segment_clearance():
    walkable = box(0.0, 0.0, 50.0, 8.0).union(box(50.0, -6.0, 100.0, 14.0))
    segments = [((0.0, 4.0), (50.0, 4.0)), ((50.0, 4.0), (100.0, 4.0))]
    assert widest_corridor(walkable, segments) == pytest.approx(20.0)
