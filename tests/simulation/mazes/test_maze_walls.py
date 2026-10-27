"""Which maze edges are walls, and wall-based analyses on every maze.

Elevated mazes (open tracks and platforms) have edges the animal can see
past: their environments declare no physical walls, so wall-based analyses
refuse them. Walled mazes declare their outline as walls, and every wall
piece and every gaze hit on a wall belongs to a bin.
"""

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.encoding import border_score
from neurospatial.encoding._physical_border import border_geometry
from neurospatial.geometry import wall_hit_bins
from neurospatial.ops.visibility import compute_viewed_location
from neurospatial.simulation import mazes

ELEVATED = [
    "make_barnes_maze",
    "make_cheeseboard_maze",
    "make_crossword_maze",
    "make_honeycomb_maze",
    "make_linear_track",
    "make_plus_maze",
    "make_radial_arm_maze",
    "make_t_maze",
    "make_w_maze",
    "make_y_maze",
]
WALLED = [
    "make_hamlet_maze",
    "make_hampton_court_maze",
    "make_rat_hexmaze",
    "make_repeated_t_maze",
    "make_repeated_y_maze",
    "make_small_hex_maze",
    "make_square_maze",
    "make_sungod_maze",
    "make_watermaze",
]


def test_every_maze_is_classified():
    factories = {name for name in dir(mazes) if name.startswith("make_")}
    factories -= {
        "make_buffered_line",
        "make_circular_arena",
        "make_corridor_polygon",
        "make_star_graph",
    }
    assert factories == set(ELEVATED) | set(WALLED)


@pytest.fixture(scope="module", params=ELEVATED)
def elevated(request):
    return getattr(mazes, request.param)(bin_size=4.0).env_2d


@pytest.fixture(scope="module", params=WALLED)
def walled(request):
    return getattr(mazes, request.param)(bin_size=4.0).env_2d


def test_elevated_maze_declares_no_walls(elevated):
    assert elevated.arena.outline is None
    assert elevated.arena.walls == ()
    # The floor still decides which positions are on the maze.
    assert (elevated.bin_at(elevated.bin_centers) >= 0).all()


def test_wall_analyses_refuse_an_elevated_maze(elevated):
    rates = np.ones(elevated.n_bins)
    with pytest.raises(NotImplementedError, match="Physical boundaries are unknown"):
        border_score(elevated, rates)
    with pytest.raises(NotImplementedError, match="Physical boundaries are unknown"):
        compute_viewed_location(
            elevated.bin_centers[:1], np.zeros(1), method="ray_cast", env=elevated
        )


def test_walled_maze_walls_and_gaze_hits_all_have_bins(walled):
    border_geometry(walled, "physical")
    rng = np.random.default_rng(0)
    origins = walled.bin_centers[rng.integers(0, walled.n_bins, 500)]
    hits = compute_viewed_location(
        origins, rng.uniform(-np.pi, np.pi, 500), method="ray_cast", env=walled
    )
    assert np.isfinite(hits).all()
    bins = walled.bin_at(np.nextafter(hits, origins))
    bins = wall_hit_bins(walled.binning, walled.arena, hits, origins, bins, "physical")
    assert (bins >= 0).all()


@pytest.mark.parametrize(
    "factory", ["make_t_maze", "make_w_maze", "make_crossword_maze"]
)
@pytest.mark.parametrize("bin_size", [2.0, 4.0])
def test_corridors_whose_walls_cross_bin_centers_own_every_wall(factory, bin_size):
    """Walled versions of the narrow-corridor mazes, at bin sizes that put
    rows of bin centers on the corridor walls."""
    floor = getattr(mazes, factory)(bin_size=bin_size).env_2d.arena.polygon
    env = Environment.from_polygon(floor, bin_size=bin_size, units="cm")
    geometry = border_geometry(env, "physical")
    assert geometry.wall_bins.any()


def test_a_walled_rebuild_of_an_elevated_maze_keeps_its_bins():
    """The documented recipe for a walled version gives the same bins."""
    maze = mazes.make_t_maze(bin_size=2.0)
    walled = Environment.from_polygon(
        maze.env_2d.arena.polygon, bin_size=2.0, units="cm"
    ).with_regions(maze.env_2d.regions)
    np.testing.assert_array_equal(walled.bin_centers, maze.env_2d.bin_centers)
    assert walled.arena.outline == "wall"


def test_a_square_box_is_walled():
    env = mazes.make_square_maze(bin_size=4.0).env_2d
    assert env.arena.outline == "wall"
    assert env.arena.polygon.equals(box(*env.arena.polygon.bounds))
