"""The default object-vector distance range covers every position in the environment."""

import numpy as np
import pytest
import shapely

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.encoding import compute_object_vector_rate
from neurospatial.geometry import cell_polygons
from neurospatial.layout import create_layout

BOX = shapely.box(0, 0, 100, 100)
# A wall from the south side up to y = 80: the only way round is its north end.
WALL_END = np.array([50.0, 80.0])
CIRCLE = shapely.Point(0, 0).buffer(40)


def _box():
    return Environment.from_polygon(BOX, bin_size=2.0, units="cm")


def _walled_box():
    wall = shapely.LineString([(50, 0), tuple(WALL_END)])
    return Environment.from_polygon(
        BOX, bin_size=5.0, units="cm", arena=OpenArena(BOX, walls={"wall": wall})
    )


def _circle():
    return Environment.from_polygon(CIRCLE, bin_size=4.0, units="cm")


def _hexagons():
    layout = create_layout(
        "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0, 40), (0, 40)]
    )
    return Environment.from_layout(layout, units="cm")


def _mesh():
    layout = create_layout(
        "TriangularMesh", boundary_polygon=shapely.box(0, 0, 48, 48), point_spacing=6.0
    )
    return Environment.from_layout(layout, units="cm")


def _track():
    return Environment.linear_track(
        node_positions=[(0, 0), (100, 0), (100, 60)],
        bin_size=7.0,
        track_width=10.0,
        units="cm",
    )


LAYOUTS = {
    "box": _box,
    "walled box": _walled_box,
    "circle": _circle,
    "hexagons": _hexagons,
    "mesh": _mesh,
    "track": _track,
}


def _positions_inside(env, rng):
    """Positions inside ``env``: scattered, and at the far corners of every bin.

    The corners are each cell's vertices moved a millionth of the way to the
    bin center, and points just inside the arena's box and outline, where
    the distance to an object is largest.
    """
    low, high = env.arena.domain_bounds()
    candidates = [rng.uniform(low, high, (1500, 2))]
    edge = np.linspace(0.0, 1.0, 201)
    for x, y in ((edge, 0.0 * edge), (edge, 1.0 + 0.0 * edge)):
        for u, v in ((x, y), (y, x)):
            candidates.append(low + np.column_stack([u, v]) * (high - low))
    domain = env.arena.polygonal_domain()
    if domain is not None:
        outline = shapely.get_coordinates(shapely.segmentize(domain.boundary, 0.5))
        center = np.asarray(domain.centroid.coords[0])
        candidates.append(outline + 1e-9 * (center - outline))
    if env.binning.linearization is None:
        vertices, owner = shapely.get_coordinates(
            np.asarray(cell_polygons(env.binning), dtype=object), return_index=True
        )
        candidates.append(vertices + 1e-6 * (env.bin_centers[owner] - vertices))
    candidates = np.vstack(candidates)
    return candidates[env.bin_at(candidates) >= 0]


def _object_at_a_corner(env, positions):
    """An in-environment position at the south-west extreme: the far corner
    of an edge bin, next to the wall or outline."""
    return positions[np.argmin(positions.sum(axis=1))][None]


def _map(env, positions, objects, metric, *, ring=None):
    """The map of one spike; ``ring`` is the ring width, None for the 5 cm default.

    Whole rings round the range up, so narrow rings leave the least slack.
    """
    times = np.arange(len(positions)) * 0.02
    return compute_object_vector_rate(
        env,
        np.array([0.011]),
        times,
        positions,
        objects,
        metric=metric,
        distance_bin_size=ring,
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("metric", ["euclidean", "geodesic"])
def test_no_position_inside_is_beyond_the_default_range(layout, metric):
    """With the object in a corner, the far corners of the far bins are in range."""
    env = LAYOUTS[layout]()
    positions = _positions_inside(env, np.random.default_rng(0))
    objects = _object_at_a_corner(env, positions)

    result = _map(env, positions, objects, metric, ring=0.5)

    settings = result.analysis_settings
    assert settings["time_beyond_distance_range"] == 0.0
    assert settings["fraction_of_time_in_distance_range"] == pytest.approx(1.0)
    # No path is shorter than the straight line.
    straight = np.linalg.norm(positions - objects[0], axis=1).max()
    assert result.distance_range[1] > straight


def test_geodesic_default_range_covers_the_path_round_a_wall():
    """An object against a wall: the animal behind the wall is in range.

    The shortest path from the far corner behind the wall goes round the
    wall's free end, so its length is known exactly.
    """
    env = _walled_box()
    positions = _positions_inside(env, np.random.default_rng(1))
    objects = np.array([[49.9, 0.1]])  # west face of the wall, south-wall corner
    behind = positions[positions[:, 0] > 50.0]
    round_the_wall = np.linalg.norm(objects[0] - WALL_END) + np.linalg.norm(
        behind - WALL_END, axis=1
    )
    assert round_the_wall.max() > np.hypot(100.0, 100.0)  # farther than any line

    result = _map(env, positions, objects, "geodesic", ring=0.5)

    assert result.analysis_settings["time_beyond_distance_range"] == 0.0
    assert result.distance_range[1] > round_the_wall.max()


def test_euclidean_default_range_covers_an_object_outside_the_environment():
    """A cue outside the arena is farther than the arena's diagonal."""
    env = _box()
    positions = _positions_inside(env, np.random.default_rng(2))
    objects = np.array([[150.0, 50.0]])
    farthest = np.hypot(150.0, 50.0)  # from the corners (0, 0) and (0, 100)

    result = _map(env, positions, objects, "euclidean")

    assert result.analysis_settings["time_beyond_distance_range"] == 0.0
    assert farthest < result.distance_range[1] <= farthest + 5.0


def test_euclidean_default_range_is_the_same_wherever_the_object_is_inside():
    """Moving the object inside the arena keeps the grid: the box diagonal."""
    env = _box()
    positions = _positions_inside(env, np.random.default_rng(3))
    ranges = {
        _map(env, positions, np.array([place]), "euclidean").distance_range
        for place in ([0.1, 0.1], [50.0, 50.0], [80.0, 3.0])
    }
    assert ranges == {(0.0, 145.0)}  # the 141.4 cm diagonal in whole 5 cm rings


def test_geodesic_default_range_of_a_track_is_the_longest_run_along_it():
    """On a track the range covers the run from the object to the far end."""
    env = _track()
    positions = np.array([[1.0, 3.0], [50.0, -4.0], [104.0, 30.0], [100.0, 64.9]])
    assert (env.bin_at(positions) >= 0).all()
    objects = np.array([[1.0, 3.0]])  # 1 cm along the first arm
    longest_run = 99.0 + 60.0  # to the end of the second arm

    result = _map(env, positions, objects, "geodesic")

    assert result.analysis_settings["time_beyond_distance_range"] == 0.0
    assert result.distance_range == (0.0, 160.0)
    assert longest_run < result.distance_range[1]


def _wells(max_distance):
    """Three bins found by nearest center, 20 cm apart along an L."""
    from neurospatial.binning import (
        BinConnections,
        Binning,
        MissingFaces,
        NearestBinLocator,
        SpatialBins,
    )

    centers = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0]])
    return Environment.from_bins(
        Binning(
            kind="wells",
            bins=SpatialBins.from_arrays(
                centers=centers, sizes=np.ones(3), measure_dim=2
            ),
            connections=BinConnections.from_arrays(
                pairs=np.array([[0, 1], [1, 2]]),
                pair_distances=np.array([20.0, 20.0]),
            ),
            locator=NearestBinLocator(centers, max_distance=max_distance),
            faces=MissingFaces("Only the well centers were measured."),
        ),
        units="cm",
    )


@pytest.mark.parametrize("metric", ["euclidean", "geodesic"])
def test_nearest_center_bins_are_covered_out_to_their_radius(metric):
    """Bins found by nearest center hold positions up to ``max_distance`` away."""
    env = _wells(max_distance=4.0)
    positions = np.array([[0.0, 0.0], [20.0, 23.9], [22.8, 22.8], [-2.8, -2.8]])
    assert (env.bin_at(positions) >= 0).all()
    objects = np.array([[-2.8, -2.8]])

    result = _map(env, positions, objects, metric, ring=0.5)

    assert result.analysis_settings["time_beyond_distance_range"] == 0.0


@pytest.mark.parametrize(
    ("metric", "expected_range"),
    [("euclidean", (0.0, 30.0)), ("geodesic", (0.0, 45.0))],
)
@pytest.mark.parametrize("max_distance", [np.inf])
def test_bins_with_no_lookup_radius_are_covered_at_their_centers(
    metric, expected_range, max_distance
):
    """With no limit on how far a bin's positions reach, the default range
    covers the bin centers, and time farther out is reported, not refused.

    Euclidean: the 28.3 cm diagonal of the centers' box. Geodesic: 40 cm
    along the two connections plus 1.4 cm from the first center to the
    object.
    """
    env = _wells(max_distance=max_distance)
    positions = np.array(
        [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [20.0, 300.0], [20.0, 20.0]]
    )
    assert (env.bin_at(positions) >= 0).all()
    objects = np.array([[1.0, 1.0]])

    result = _map(env, positions, objects, metric)

    assert result.distance_range == expected_range
    # Four 0.02 s intervals; the one that starts 300 cm out is beyond.
    settings = result.analysis_settings
    assert settings["time_beyond_distance_range"] == pytest.approx(0.02)
    assert settings["fraction_of_time_in_distance_range"] == pytest.approx(0.75)


def _session_with_a_moving_object(n_samples=400):
    """Positions all over the 100 cm box and an object jittering at its center."""
    rng = np.random.default_rng(6)
    positions = rng.uniform(1.0, 99.0, (n_samples, 2))
    objects = np.array([50.0, 50.0]) + rng.normal(0.0, 1.0, (n_samples, 1, 2))
    return positions, objects


@pytest.mark.parametrize("stray", [(-500.0, -500.0), (1e5, 1e5)])
def test_one_stray_sample_of_a_moving_object_does_not_set_the_range(stray):
    """A tracking glitch far outside the arena is one sample out of range.

    The default range of an object that moves is the arena's diagonal
    whatever its positions, so one stray sample cannot stretch the grid; the
    time at that sample is reported as beyond the range.
    """
    env = _box()
    positions, objects = _session_with_a_moving_object()
    clean = _map(env, positions, objects, "euclidean")
    objects[123, 0] = stray

    result = _map(env, positions, objects, "euclidean")

    assert result.distance_range == clean.distance_range == (0.0, 145.0)
    assert result.n_distance_bins == clean.n_distance_bins == 29
    settings = result.analysis_settings
    assert settings["time_beyond_distance_range"] == pytest.approx(0.02)
    assert settings["fraction_of_time_in_distance_range"] == pytest.approx(398 / 399)


def test_a_moving_object_outside_the_arena_is_reported_not_covered():
    """An object that moves outside the arena needs an explicit range.

    At 100 cm east of the east wall it is beyond the 145 cm default from the
    western part of the box; that time is reported, and an explicit range
    covers it.
    """
    env = _box()
    positions, objects = _session_with_a_moving_object()
    objects[:, 0, 0] += 150.0

    result = _map(env, positions, objects, "euclidean")

    assert result.distance_range == (0.0, 145.0)
    assert 0.2 < result.analysis_settings["fraction_of_time_in_distance_range"] < 0.9
    times = np.arange(len(positions)) * 0.02
    covered = compute_object_vector_rate(
        env,
        np.array([0.011]),
        times,
        positions,
        objects,
        distance_range=(0.0, 215.0),
    )
    assert covered.analysis_settings["time_beyond_distance_range"] == 0.0
