"""What each maze's edges are, and boundary analyses on every maze.

Elevated mazes (open tracks and platforms) are bounded by drop edges: the
animal cannot step off, but it sees past them. Walled mazes are bounded by
opaque walls. The Sungod maze has opaque external walls and transparent walls
between its arms. Movement analyses (border scores) use every edge; gaze
stops only at opaque walls.
"""

import numpy as np
import pytest
import shapely
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.encoding import border_score
from neurospatial.encoding._physical_border import border_geometry
from neurospatial.geometry import boundary_geometry, cell_polygons, wall_hit_bins
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
    "make_w_maze",
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
    "make_t_maze",
    "make_watermaze",
    "make_y_maze",
]


def test_every_maze_is_classified():
    factories = {name for name in dir(mazes) if name.startswith("make_")}
    factories -= {
        "make_buffered_line",
        "make_circular_arena",
        "make_corridor_polygon",
    }
    assert factories == set(ELEVATED) | set(WALLED)


@pytest.fixture(scope="module", params=ELEVATED)
def elevated(request):
    return getattr(mazes, request.param)(bin_size=4.0).env_2d


@pytest.fixture(scope="module", params=WALLED)
def walled(request):
    return getattr(mazes, request.param)(bin_size=4.0).env_2d


def test_elevated_maze_edges_are_drops(elevated):
    assert elevated.arena.outline == "drop"
    assert elevated.arena.walls == ()
    # The floor still decides which positions are on the maze.
    assert (elevated.bin_at(elevated.bin_centers) >= 0).all()


def test_elevated_maze_bounds_movement_but_not_gaze(elevated):
    """Border scores use the drop edges; a gaze leaving the maze sees no wall."""
    border_geometry(elevated, "physical")
    assert np.isfinite(border_score(elevated, np.ones(elevated.n_bins)))
    rng = np.random.default_rng(0)
    origins = elevated.bin_centers[rng.integers(0, elevated.n_bins, 200)]
    viewed = compute_viewed_location(
        origins, rng.uniform(-np.pi, np.pi, 200), method="ray_cast", env=elevated
    )
    assert np.isnan(viewed).all()


def test_walled_maze_outline_is_opaque(walled):
    assert walled.arena.outline == "wall"


def test_walled_maze_walls_and_gaze_hits_all_have_bins(walled):
    border_geometry(walled, "physical")
    rng = np.random.default_rng(0)
    origins = walled.bin_centers[rng.integers(0, walled.n_bins, 500)]
    hits = compute_viewed_location(
        origins, rng.uniform(-np.pi, np.pi, 500), method="ray_cast", env=walled
    )
    seen = np.isfinite(hits).all(axis=1)
    # Only the Sungod maze lets a gaze out (through its clear arm walls).
    if walled.name != "sungod_maze":
        assert seen.all()
    bins = walled.bin_at(np.nextafter(hits[seen], origins[seen]))
    bins = wall_hit_bins(
        walled.binning, walled.arena, hits[seen], origins[seen], bins, "physical"
    )
    assert (bins >= 0).all()


def test_sungod_walls_between_arms_are_transparent():
    """Gillespie et al. (2021): opaque external walls, transparent walls
    between arms. A gaze from one arm toward its neighbour passes the clear
    wall that stops the animal."""
    env = mazes.make_sungod_maze(bin_size=2.0, include_track=False).env_2d
    arena = env.arena
    assert arena.outline == "wall"
    assert set(arena.wall_kinds) == {"transparent_wall"}
    # Seven gaps between eight arms, one wall on each side of each gap.
    assert len(set(arena.wall_names)) == 14
    assert "arm_1_wall_facing_arm_2" in arena.wall_names
    assert "arm_8_wall_facing_arm_7" in arena.wall_names
    gaze = boundary_geometry(env.binning, arena, "physical", blocking="gaze")
    movement = boundary_geometry(env.binning, arena, "physical", blocking="movement")
    # The floor's whole outline bounds movement; the dividers do not bound gaze.
    assert movement.length == pytest.approx(arena.polygon.exterior.length)
    dividers = sum(wall.length for wall in arena.walls)
    assert gaze.length == pytest.approx(movement.length - dividers)
    assert gaze.intersection(arena.walls[0]).length == pytest.approx(0, abs=1e-9)


def test_sungod_bins_are_unchanged_by_its_dividers():
    """The dividers lie on the outline, so declaring them moves no bin."""
    maze = mazes.make_sungod_maze(bin_size=2.0, include_track=False).env_2d
    plain = Environment.from_polygon(
        maze.arena.polygon, bin_size=2.0, connect_diagonal_neighbors=True
    )
    np.testing.assert_array_equal(maze.bin_centers, plain.bin_centers)


@pytest.mark.parametrize("factory", ["make_w_maze", "make_crossword_maze"])
@pytest.mark.parametrize("bin_size", [2.0, 4.0])
def test_corridors_whose_walls_cross_bin_centers_own_every_wall(factory, bin_size):
    """Walled versions of the narrow-corridor mazes, at bin sizes that put
    rows of bin centers on the corridor walls."""
    floor = getattr(mazes, factory)(bin_size=bin_size).env_2d.arena.polygon
    env = Environment.from_polygon(floor, bin_size=bin_size, units="cm")
    geometry = border_geometry(env, "physical")
    assert geometry.wall_bins.any()


@pytest.mark.parametrize("bin_size", [2.0, 4.0])
def test_t_maze_walls_crossing_bin_centers_are_owned(bin_size):
    env = mazes.make_t_maze(bin_size=bin_size).env_2d
    assert border_geometry(env, "physical").wall_bins.any()


def test_a_walled_rebuild_of_an_elevated_maze_keeps_its_bins():
    """The documented recipe for a walled version gives the same bins."""
    maze = mazes.make_plus_maze(bin_size=2.0)
    walled = Environment.from_polygon(
        maze.env_2d.arena.polygon, bin_size=2.0, units="cm"
    ).with_regions(maze.env_2d.regions)
    np.testing.assert_array_equal(walled.bin_centers, maze.env_2d.bin_centers)
    assert walled.arena.outline == "wall"


def test_a_square_box_is_walled():
    env = mazes.make_square_maze(bin_size=4.0).env_2d
    assert env.arena.outline == "wall"
    assert env.arena.polygon.equals(box(*env.arena.polygon.bounds))


@pytest.mark.parametrize(
    ("factory", "bin_size"),
    [("make_hampton_court_maze", 8.0), ("make_repeated_t_maze", 5.0)],
)
def test_bins_coarser_than_the_walls_do_not_join_through_them(factory, bin_size):
    """Partitions thinner than a bin still separate the alleys beside them.

    Each open pair of bins must be joined inside the floor by the parts of
    their two cells, without passing through a third bin: the part of the
    two cells on the floor that holds one center also holds the other.
    """
    env = getattr(mazes, factory)(bin_size=bin_size, include_track=False).env_2d
    floor = env.arena.polygonal_domain()
    cells = np.asarray(cell_polygons(env.binning), dtype=object)
    centers = shapely.points(env.bin_centers)
    pairs = env.edge_pairs
    whole = shapely.covers(floor, cells)
    for u, v in pairs[~(whole[pairs[:, 0]] & whole[pairs[:, 1]])]:
        joined = shapely.union_all([cells[u], cells[v]]).intersection(floor)
        # Absorb rounding where the two cells meet.
        parts = shapely.get_parts(joined.buffer(1e-9))
        (part,) = parts[shapely.covers(parts, centers[u])]
        assert part.covers(centers[v]), (u, v)
