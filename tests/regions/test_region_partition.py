"""Regions that tile space split the points on their shared edges, one each.

A point exactly on a region's edge belongs to the region it would enter if
nudged a hair north-east. Polygons that share edges therefore partition the
points and bins between them, and each query of membership applies this one
rule.
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
import shapely.affinity
from numpy.testing import assert_array_equal
from shapely.geometry import LineString, Polygon, box

from neurospatial import Environment
from neurospatial.ops import regions_to_mask
from neurospatial.regions import (
    Region,
    Regions,
    points_in_any_region,
    regions_containing_points,
)

# Four tiles around a shared vertex at the origin. The spokes between them run
# at about 60° (slope 7/4), about 120° (slope -7/4), 45° and 0°: the slopes
# of hexagonal, diagonal and axis-aligned edges.
_SPOKE_ENDS = {
    "60": (4.0, 7.0),
    "120": (-4.0, 7.0),
    "45": (-6.0, -6.0),
    "0": (8.0, 0.0),
}
_TILES = {
    "north": [(0.0, 0.0), (4.0, 7.0), (0.0, 9.0), (-4.0, 7.0)],
    "west": [(0.0, 0.0), (-4.0, 7.0), (-7.0, 1.0), (-6.0, -6.0)],
    "south": [(0.0, 0.0), (-6.0, -6.0), (2.0, -8.0), (8.0, 0.0)],
    "east": [(0.0, 0.0), (8.0, 0.0), (9.0, 3.0), (4.0, 7.0)],
}
# The outline of the four tiles together.
_OUTLINE = [
    (4.0, 7.0),
    (0.0, 9.0),
    (-4.0, 7.0),
    (-7.0, 1.0),
    (-6.0, -6.0),
    (2.0, -8.0),
    (8.0, 0.0),
    (9.0, 3.0),
]


def _lattice(step: float = 0.125) -> np.ndarray:
    """Points every ``step`` over the tiles; many lie exactly on the spokes."""
    axis = np.arange(-10.0, 10.0 + step, step)
    return np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)


def _rotated(coords, angle: float) -> list[tuple[float, float]]:
    rotated = shapely.affinity.rotate(
        Polygon(coords), angle, origin=(0, 0), use_radians=True
    )
    return list(rotated.exterior.coords)[:-1]


def _tiling(angle: float) -> tuple[Regions, Region]:
    """The four tiles and their outline, rotated by ``angle`` radians.

    Each tile is rotated vertex by vertex with the same arithmetic, so tiles
    that share a vertex share its floating-point coordinates.
    """
    tiles = Regions()
    for name, coords in _TILES.items():
        tiles.add(name, polygon=Polygon(_rotated(coords, angle)))
    outline = Region("outline", "polygon", Polygon(_rotated(_OUTLINE, angle)))
    return tiles, outline


def _on_spokes(points: np.ndarray) -> np.ndarray:
    spokes = shapely.MultiLineString(
        [LineString([(0.0, 0.0), end]) for end in _SPOKE_ENDS.values()]
    )
    return shapely.intersects_xy(spokes, points[:, 0], points[:, 1])


@pytest.mark.parametrize("angle", [0.0, 0.3, np.pi / 6, 1.0])
def test_tiles_partition_the_points_their_outline_holds(angle):
    """Each point in the outline is in exactly one tile, also on shared edges."""
    tiles, outline = _tiling(angle)
    origin = np.array([[0.0, 0.0]])
    points = np.vstack([_lattice(), origin])
    if angle != 0.0:
        points = np.vstack([points, np.array(_rotated(_SPOKE_ENDS.values(), angle))])
    counts = regions_containing_points(points, tiles).to_numpy().sum(axis=1)
    in_outline = points_in_any_region(points, Regions([outline]))
    assert_array_equal(counts, in_outline.astype(int))
    # The shared vertex belongs to exactly one tile.
    assert counts[len(_lattice())] == 1


def test_the_lattice_lands_exactly_on_every_spoke():
    """The partition test above checks points on each shared edge."""
    points = _lattice()
    on = _on_spokes(points)
    for end in _SPOKE_ENDS.values():
        spoke = LineString([(0.0, 0.0), end])
        assert shapely.intersects_xy(spoke, points[on, 0], points[on, 1]).sum() > 4


def test_a_point_on_a_shared_edge_goes_to_the_tile_north_east_of_it():
    """Nudged north-east, a point on a shared edge enters one tile: that one."""
    tiles, _ = _tiling(0.0)
    points = np.array(
        [
            [0.5, 0.875],  # on the 60° spoke: east of it is "east"
            [-0.5, 0.875],  # on the 120° spoke: north-east of it is "north"
            [-1.0, -1.0],  # on the 45° spoke: nudged at 37°, it falls "south"
            [2.0, 0.0],  # on the 0° spoke: north of it is "east"
            [0.0, 0.0],  # the shared vertex: north-east of it is "east"
        ]
    )
    assert regions_containing_points(points, tiles, return_dataframe=False) == [
        ["east"],
        ["north"],
        ["south"],
        ["east"],
        ["east"],
    ]


def test_rectangles_are_closed_on_south_and_west_and_open_on_north_and_east():
    """A box holds its south and west edges, as a bin [a, b) holds a."""
    regions = Regions([Region("box", "polygon", box(0.0, 0.0, 10.0, 10.0))])
    points = np.array(
        [[0.0, 5.0], [5.0, 0.0], [0.0, 0.0], [10.0, 5.0], [5.0, 10.0], [10.0, 0.0]]
    )
    assert_array_equal(
        points_in_any_region(points, regions),
        [True, True, True, False, False, False],
    )


def test_the_rule_does_not_depend_on_the_coordinate_scale():
    """Far-from-origin and tiny copies of the tiling split each point alike.

    Scales are powers of two, so points on the spokes stay exactly on them.
    """
    tiles, outline = _tiling(0.0)
    lattice = _lattice(0.5)
    inside = shapely.contains_xy(outline.data, lattice[:, 0], lattice[:, 1])
    points = np.vstack([lattice[inside], [[0.0, 0.0]]])
    expected = regions_containing_points(points, tiles).to_numpy()
    assert_array_equal(expected.sum(axis=1), 1)
    for scale, offset in [(2.0**17, 2.0**25), (2.0**-40, 0.0), (1.0, 2.0**20)]:
        moved = Regions(
            Region(
                name,
                "polygon",
                shapely.affinity.translate(
                    shapely.affinity.scale(region.data, scale, scale, origin=(0, 0)),
                    offset,
                    offset,
                ),
            )
            for name, region in tiles.items()
        )
        found = regions_containing_points(points * scale + offset, moved)
        assert_array_equal(found.to_numpy(), expected)


def test_a_ring_and_its_hole_split_the_ring_boundary():
    hole = box(3.0, 3.0, 7.0, 7.0)
    regions = Regions()
    regions.add("ring", polygon=box(0.0, 0.0, 10.0, 10.0).difference(hole))
    regions.add("hole", polygon=hole)
    points = _lattice(0.5)
    points = points[(points >= 0).all(axis=1) & (points < 10).all(axis=1)]
    counts = regions_containing_points(points, regions).to_numpy().sum(axis=1)
    assert_array_equal(counts, np.ones(len(points), dtype=int))


class TestPointRegionRim:
    """A point region with ``point_radius`` is a disc with the same tie rule."""

    def test_rim_is_held_on_the_half_facing_away_from_north_east(self):
        well = Regions([Region("well", "point", np.array([50.0, 50.0]))])
        # 3-4-5 triangles: every probe is exactly 5 cm from the well.
        rim = np.array([[47.0, 46.0], [53.0, 54.0], [46.0, 53.0], [54.0, 47.0]])
        assert_array_equal(
            points_in_any_region(rim, well, point_radius=5.0),
            [True, False, True, False],
        )

    def test_touching_discs_share_their_contact_point_once(self):
        wells = Regions()
        wells.add("west", point=(-1.0, 0.0))
        wells.add("east", point=(1.0, 0.0))
        found = regions_containing_points(
            np.array([[0.0, 0.0]]), wells, point_radius=1.0, return_dataframe=False
        )
        assert found == [["east"]]


def test_environment_bins_on_shared_edges_go_to_one_tile():
    """Bin centers on the lattice split between the tiles like the points do."""
    tiles, outline = _tiling(0.0)
    points = _lattice(0.5)
    env = Environment.from_samples(points, bin_size=0.5).with_regions(tiles)
    assert np.isin(env.bin_centers, points).all()
    centers = env.bin_centers
    assert _on_spokes(centers).sum() > 10
    membership = env.region_membership()
    in_outline = points_in_any_region(centers, Regions([outline]))
    assert_array_equal(membership.sum(axis=1), in_outline.astype(int))
    expected = regions_containing_points(centers, tiles).to_numpy()
    assert_array_equal(membership, expected)
    for column, name in enumerate(tiles):
        assert_array_equal(
            np.flatnonzero(membership[:, column]), env.bins_in_region(name)
        )
        assert_array_equal(regions_to_mask(env, name), membership[:, column])
