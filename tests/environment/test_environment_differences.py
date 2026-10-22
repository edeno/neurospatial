"""Environment comparison separates bin meaning from topology.

Decoding and rate comparisons ask the same question of two environments: do
their bins mean the same places, and is the space connected the same way? A
door or barrier experiment keeps the bins but changes connectivity, so the two
answers are reported separately.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import NoReturn
from uuid import uuid4

import networkx as nx
import numpy as np
import pytest
import shapely
from environment_helpers import renumbered, renumbered_cells
from shapely.geometry import LineString, Point, Polygon, box

from neurospatial import Environment, IncompatibleEnvironmentError
from neurospatial.arena import OpenArena
from neurospatial.binning import (
    BinConnections,
    Binning,
    FaceGeometry,
    KnownFaces,
    MeshLocator,
    MissingFaces,
    NearestBinLocator,
    SpatialBins,
    TriangleShapes,
)
from neurospatial.decoding.posterior import decode_position
from neurospatial.encoding.spatial import SpatialRatesResult
from neurospatial.environment import (
    _environment_differences,
    _fields,
    _mapped_bin_difference,
    _mapped_topology_difference,
)
from neurospatial.environment._comparison import _conductances, _ownership_keys
from neurospatial.geometry import MissingCellGeometryError
from neurospatial.layout import create_layout
from neurospatial.ops import AffineND


def _grid(*, offset: float = 0.0, diagonals: bool = True) -> Environment:
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool),
        grid_edges=(offset + np.arange(3.0) * 10,) * 2,
        connect_diagonal_neighbors=diagonals,
    )


def _samples() -> np.ndarray:
    return np.random.default_rng(0).uniform(0, 30, (500, 2))


def _t_maze() -> Environment:
    graph = nx.Graph()
    graph.add_nodes_from(
        [
            (0, {"pos": (0.0, 0.0)}),
            (1, {"pos": (0.0, 40.0)}),
            (2, {"pos": (-30.0, 40.0)}),
            (3, {"pos": (30.0, 40.0)}),
        ]
    )
    graph.add_edge(0, 1, distance=40.0)
    graph.add_edge(1, 2, distance=30.0)
    graph.add_edge(1, 3, distance=30.0)
    return Environment.from_graph(
        graph=graph,
        edge_order=[(0, 1), (1, 2), (1, 3)],
        edge_spacing=10.0,
        bin_size=5.0,
    )


def test_an_environment_does_not_differ_from_itself() -> None:
    env = _grid()
    assert _environment_differences(env, env) == (None, None)


def test_relabelled_copy_with_the_same_labels_shares_geometry() -> None:
    env = _grid()
    copy = env.with_metadata()
    assert copy is not env
    assert copy._geometry is env._geometry
    assert _environment_differences(env, copy) == (None, None)


@pytest.mark.parametrize("label", ["units", "frame"])
def test_different_units_or_frame_change_bin_meaning(label: str) -> None:
    env = _grid().with_metadata(units="cm", frame="room")
    relabelled = env.with_metadata(**{label: "mm" if label == "units" else "camera"})
    differences = _environment_differences(env, relabelled)
    assert differences.bins == "units or coordinate frames"
    assert differences.topology is None


def test_polar_and_cartesian_bins_differ_in_coordinate_system() -> None:
    polar = Environment.from_polar_egocentric(
        distance_range=(0.0, 20.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=10.0,
        angle_bin_size=np.pi / 2,
    )
    assert _environment_differences(polar, _grid()).bins == "coordinate systems"


def test_bins_with_equal_centers_and_different_areas_differ_in_measures() -> None:
    small = _clipped_cell(box(0.25, 0.25, 0.75, 0.75))
    large = _clipped_cell(box(0.1, 0.1, 0.9, 0.9))
    np.testing.assert_array_equal(small.bin_centers, large.bin_centers)
    assert _environment_differences(small, large) == ("bin measures", None)


def test_different_connectivity_is_topology_not_bins() -> None:
    differences = _environment_differences(_grid(), _grid(diagonals=False))
    assert differences.bins is None
    assert differences.topology == "weighted connectivity"


def test_shifted_bins_differ_in_bin_coordinates() -> None:
    differences = _environment_differences(_grid(), _grid(offset=1000.0))
    assert differences.bins == "bin coordinates or ordering"


def test_bin_and_topology_differences_are_both_reported() -> None:
    differences = _environment_differences(
        _grid(), _grid(offset=1000.0, diagonals=False)
    )
    assert differences.bins == "bin coordinates or ordering"
    assert differences.topology == "weighted connectivity"


def _clipped_cell(clip: Polygon, walls: tuple[LineString, ...] = ()) -> Environment:
    parent = Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool),
        grid_edges=(np.arange(2.0),) * 2,
        arena=OpenArena(box(0, 0, 1, 1), walls=walls),
    )
    return parent.clip(clip)


_SQUARE = box(0, 0.25, 1, 0.75)
_DIAMOND = Polygon([(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)])


def test_polygon_cells_with_equal_centers_and_areas_differ_in_bins() -> None:
    square, diamond = _clipped_cell(_SQUARE), _clipped_cell(_DIAMOND)
    np.testing.assert_array_equal(square.bin_centers, diamond.bin_centers)
    np.testing.assert_array_equal(square.bin_sizes, diamond.bin_sizes)
    assert _environment_differences(square, diamond) == (
        "polygon cell geometry",
        None,
    )


def test_a_wall_is_topology_and_does_not_hide_a_cell_difference() -> None:
    wall = (LineString([(0, 0.9), (0.2, 0.9)]),)
    assert _environment_differences(
        _clipped_cell(_SQUARE), _clipped_cell(_SQUARE, walls=wall)
    ) == (None, "continuous arena or physical walls")
    assert _environment_differences(
        _clipped_cell(_SQUARE), _clipped_cell(_DIAMOND, walls=wall)
    ) == ("polygon cell geometry", "continuous arena or physical walls")


def test_independently_built_environments_from_the_same_samples_agree() -> None:
    first = Environment.from_samples(_samples(), bin_size=3.0)
    second = Environment.from_samples(_samples(), bin_size=3.0)
    assert first._geometry is not second._geometry
    assert _environment_differences(first, second) == (None, None)


def _mesh() -> Environment:
    return Environment.from_layout(
        create_layout(
            "TriangularMesh",
            boundary_polygon=Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),
            point_spacing=3.0,
        )
    )


def _hexagons() -> Environment:
    return Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=3.0, dimension_ranges=[(0, 20), (0, 20)]
        )
    )


# Comparing builds the mesh's faces, which warn when many triangles are skewed;
# whether this square-grid Delaunay mesh is skewed depends on the platform.
_MESH_WARNING = "ignore:Triangular mesh is non-K-orthogonal:UserWarning"


def _polar() -> Environment:
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 20.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 4,
    )


def _grid_3d() -> Environment:
    positions = np.random.default_rng(1).uniform(0, 10, (3000, 3))
    return Environment.from_samples(positions, bin_size=2.0)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(_mesh, id="mesh", marks=pytest.mark.filterwarnings(_MESH_WARNING)),
        pytest.param(_hexagons, id="hexagons"),
        pytest.param(_polar, id="polar grid"),
        pytest.param(_grid_3d, id="3-D grid"),
    ],
)
def test_independently_built_cells_agree(build) -> None:
    first, second = build(), build()
    assert first._geometry is not second._geometry
    assert _environment_differences(first, second) == (None, None)


def _two_triangles(points: list[tuple[float, float]], triangles) -> Environment:
    """Two triangles sharing an edge, in a fixed arena, with their own faces."""
    points_array = np.asarray(points, dtype=float)
    triangles_array = np.asarray(triangles)
    corners = points_array[triangles_array]
    centers = corners.mean(axis=1)
    sides = corners[:, 1:] - corners[:, :1]
    areas = (
        np.abs(sides[:, 0, 0] * sides[:, 1, 1] - sides[:, 0, 1] * sides[:, 1, 0]) / 2
    )
    distance = np.hypot(*(centers[1] - centers[0]))
    pairs = np.array([[0, 1]])
    return Environment.from_bins(
        Binning(
            "TriangularMesh",
            SpatialBins.from_arrays(centers, areas, 2),
            BinConnections.from_arrays(pairs, [distance]),
            MeshLocator(points_array, triangles_array, np.arange(2), None),
            KnownFaces(
                FaceGeometry(pairs, np.array([6 * np.sqrt(2)]), [distance], areas)
            ),
            shapes=TriangleShapes(points_array, triangles_array),
        ),
        arena=OpenArena(box(-5, -5, 12, 12)),
    )


def test_triangles_with_equal_centroids_areas_and_faces_differ_in_cell_shapes() -> None:
    # The second mesh is the first reflected through the midpoint of its two
    # centroids, so its triangles swap places: each bin keeps its centroid,
    # area, shared edge and neighbor distance, but is a different triangle.
    first = _two_triangles([(0, 0), (6, 0), (0, 6), (9, 3)], [[0, 1, 2], [1, 2, 3]])
    second = _two_triangles([(1, 5), (7, -1), (-2, 2), (7, 5)], [[0, 1, 2], [3, 0, 1]])
    np.testing.assert_array_equal(first.bin_centers, second.bin_centers)
    np.testing.assert_array_equal(first.bin_sizes, second.bin_sizes)
    np.testing.assert_array_equal(
        first.binning.connections.pair_distances,
        second.binning.connections.pair_distances,
    )
    assert _environment_differences(first, second) == ("cell shapes", None)


def test_the_same_triangles_listed_differently_have_the_same_cells() -> None:
    # Vertex order within a triangle and the numbering of mesh points do not
    # change the cell.
    first = _two_triangles([(0, 0), (6, 0), (0, 6), (9, 3)], [[0, 1, 2], [1, 2, 3]])
    relisted = _two_triangles([(9, 3), (0, 6), (6, 0), (0, 0)], [[3, 1, 2], [0, 2, 1]])
    assert _environment_differences(first, relisted) == (None, None)


def test_hexagons_with_equal_centers_and_areas_differ_in_cell_shapes() -> None:
    # One hexagon moved to the origin, then turned a quarter turn about it:
    # the same center and area, but a flat-topped instead of a pointy-topped cell.
    single = _hexagons().subset(mask=np.arange(_hexagons().n_bins) == 5)
    x, y = single.bin_centers[0]
    at_origin = single.apply_transform(
        AffineND(np.array([[1.0, 0, -x], [0, 1, -y], [0, 0, 1]]))
    )
    turned = at_origin.apply_transform(
        AffineND(np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    )
    np.testing.assert_array_equal(at_origin.bin_centers, turned.bin_centers)
    np.testing.assert_array_equal(at_origin.bin_sizes, turned.bin_sizes)
    assert _environment_differences(at_origin, turned).bins == "cell shapes"


_ROOM = OpenArena(box(-20, -20, 30, 30))


def _one_cell(x_edges: list[float], y_edges: list[float]) -> Environment:
    return Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool),
        grid_edges=(np.array(x_edges), np.array(y_edges)),
        arena=_ROOM,
    )


def test_grid_cells_with_equal_centers_and_areas_differ_in_cell_shapes() -> None:
    square, wide = _one_cell([0, 10], [0, 10]), _one_cell([-5, 15], [2.5, 7.5])
    np.testing.assert_array_equal(square.bin_centers, wide.bin_centers)
    np.testing.assert_array_equal(square.bin_sizes, wide.bin_sizes)
    assert _environment_differences(square, wide) == ("cell shapes", None)


def test_a_rotated_grid_cell_differs_from_an_axis_aligned_one() -> None:
    square = _one_cell([-5, 5], [-5, 5])
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    diamond = square.apply_transform(
        AffineND(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
    )
    np.testing.assert_array_equal(square.bin_centers, diamond.bin_centers)
    np.testing.assert_array_equal(square.bin_sizes, diamond.bin_sizes)
    assert _environment_differences(square, diamond).bins == "cell shapes"


def _clipped_grid() -> Environment:
    return Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool), grid_edges=(np.arange(5.0),) * 2
    ).clip(box(0, 0, 3.5, 3.5))


def _without_cells(env: Environment, *, size_scale: float = 1.0) -> Environment:
    """``env``'s bins and connections as explicit bins that record no cells."""
    binning = renumbered(env, np.arange(env.n_bins)).binning
    bins = binning.bins
    return Environment.from_bins(
        replace(
            binning,
            bins=SpatialBins.from_arrays(
                bins.centers, bins.sizes * size_scale, bins.measure_dims
            ),
            faces=MissingFaces("explicit bins without cells"),
        ),
        units=env.units,
        frame=env.frame,
        arena=env.arena,
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(_clipped_grid, id="polygon cells"),
        pytest.param(_grid, id="grid"),
        pytest.param(_mesh, id="mesh", marks=pytest.mark.filterwarnings(_MESH_WARNING)),
    ],
)
def test_bins_without_cells_are_compared_by_center_and_measure(build) -> None:
    # Only one environment records cells, so there are no cells to compare;
    # the verdict comes from the bin centers and measures.
    with_cells = build()
    same = _without_cells(with_cells)
    larger = _without_cells(with_cells, size_scale=2.0)
    assert same.binning.shapes is None
    identity = np.column_stack([np.arange(with_cells.n_bins)] * 2)
    for first, second in ((with_cells, same), (same, with_cells)):
        assert _environment_differences(first, second).bins is None
        assert _mapped_bin_difference(first, second, identity) is None
    assert _environment_differences(with_cells, larger).bins == "bin measures"
    assert _mapped_bin_difference(with_cells, larger, identity) == "bin measures"
    reversed_order = np.column_stack([identity[:, 0], identity[::-1, 1]])
    assert _mapped_bin_difference(with_cells, same, reversed_order) == (
        "bin coordinates"
    )


_FACES = "smoothing conductances between neighboring bins"


def _three_bins_in_a_row(second_face_area: float) -> Environment:
    """Bins 10 units apart whose second face has the given area."""
    centers = np.array([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)])
    sizes = np.full(3, 100.0)
    pairs = np.array([[0, 1], [1, 2]])
    return Environment.from_bins(
        Binning(
            "explicit",
            SpatialBins.from_arrays(centers, sizes, 2),
            BinConnections.from_arrays(pairs, [10.0, 10.0]),
            NearestBinLocator(centers, max_distance=5.0),
            KnownFaces(
                FaceGeometry(pairs, [10.0, second_face_area], [10.0, 10.0], sizes)
            ),
        ),
        arena=OpenArena(box(-5, -5, 25, 5)),
    )


def test_different_faces_between_the_same_bins_are_a_topology_difference() -> None:
    # Same bins, neighbors, distances and arena; a narrower second face
    # passes less of a field between bins 1 and 2 when smoothing.
    wide, narrow = _three_bins_in_a_row(10.0), _three_bins_in_a_row(2.0)
    field = np.array([0.0, 1.0, 0.0])
    smoothed = [
        env.diffuse(field, bandwidth=5.0, mode="average") for env in (wide, narrow)
    ]
    assert not np.allclose(*smoothed)
    assert _environment_differences(wide, narrow) == (None, _FACES)


def test_faces_present_in_only_one_environment_are_a_topology_difference() -> None:
    wide = _three_bins_in_a_row(10.0)
    without = Environment.from_bins(
        replace(wide.binning, faces=MissingFaces("no cell faces for a test")),
        arena=wide.arena,
    )
    assert _environment_differences(wide, without) == (None, _FACES)


def test_two_environments_that_cannot_smooth_have_the_same_conductances() -> None:
    def without_faces() -> Environment:
        wide = _three_bins_in_a_row(10.0)
        return Environment.from_bins(
            replace(wide.binning, faces=MissingFaces("no cell faces for a test")),
            arena=wide.arena,
        )

    first, second = without_faces(), without_faces()
    assert first._geometry is not second._geometry
    assert _environment_differences(first, second) == (None, None)
    identity = np.column_stack([np.arange(3), np.arange(3)])
    assert _mapped_topology_difference(first, second, identity) is None


def _sheared_combined_grids() -> Environment:
    def grid(offset: float) -> Environment:
        return Environment.from_grid_mask(
            np.ones((2, 2), dtype=bool),
            grid_edges=(offset + np.arange(3.0) * 10,) * 2,
            units="cm",
        )

    combined = Environment.combine({"a": grid(0.0), "b": grid(40.0)})
    shear = AffineND(np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    return combined.apply_transform(shear, units="cm")


def test_two_sheared_combined_environments_cannot_smooth_alike() -> None:
    # A shear is not a similarity transform, so the combined conductances
    # cannot be carried into the new coordinates.
    first, second = _sheared_combined_grids(), _sheared_combined_grids()
    assert _conductances(first) is None
    assert _environment_differences(first, second) == (None, None)
    with pytest.raises(NotImplementedError, match="similarity transform"):
        first.smooth(np.ones(first.n_bins), bandwidth=5.0, mode="average")


def test_a_combined_environment_with_a_child_without_faces_cannot_smooth() -> None:
    # The combined conductances need every child's faces; one child without
    # them leaves the whole environment unable to smooth, as MissingFaces does.
    def combined() -> Environment:
        first = _grid()
        second = _grid(offset=40.0)
        without = Environment.from_bins(
            replace(second.binning, faces=MissingFaces("no cell faces for a test")),
            arena=second.arena,
        )
        return Environment.combine({"a": first, "b": without})

    first, second = combined(), combined()
    assert _conductances(first) is None
    assert _environment_differences(first, second) == (None, None)
    with pytest.raises(MissingCellGeometryError, match="no cell faces for a test"):
        first.smooth(np.ones(first.n_bins), bandwidth=5.0, mode="average")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (NotImplementedError("a bug"), NotImplementedError),
        (ValueError("bad faces"), ValueError),
    ],
)
def test_an_error_other_than_missing_cell_geometry_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected: type[Exception]
) -> None:
    def broken(*args: object, **kwargs: object) -> NoReturn:
        raise error

    monkeypatch.setattr(_fields, "_build_diffusion_geometry", broken)
    with pytest.raises(expected, match=str(error)):
        _environment_differences(_grid(), _grid())


_IDENTITY_3 = np.column_stack([np.arange(3), np.arange(3)])


def test_different_faces_differ_through_an_identity_mapping() -> None:
    # A bin_mapping must not hide what comparing without one reports.
    wide, narrow = _three_bins_in_a_row(10.0), _three_bins_in_a_row(2.0)
    assert _mapped_topology_difference(wide, narrow, _IDENTITY_3) == _FACES


def test_a_different_face_to_an_unmapped_neighbor_is_a_difference() -> None:
    # Bin 1 smooths with unmapped bin 2 through a wider face in one of them.
    wide, narrow = _three_bins_in_a_row(10.0), _three_bins_in_a_row(2.0)
    pairs = np.column_stack([np.arange(2), np.arange(2)])
    assert _mapped_topology_difference(wide, narrow, pairs) == _FACES


@pytest.mark.parametrize("bins", [[0, 1], [1, 2], [0, 2]])
def test_a_partial_mapping_between_the_same_faces_has_no_difference(
    bins: list[int],
) -> None:
    pairs = np.column_stack([bins, bins])
    assert (
        _mapped_topology_difference(
            _three_bins_in_a_row(2.0), _three_bins_in_a_row(2.0), pairs
        )
        is None
    )


def test_faces_present_in_only_one_environment_differ_through_a_mapping() -> None:
    wide = _three_bins_in_a_row(10.0)
    without = Environment.from_bins(
        replace(wide.binning, faces=MissingFaces("no cell faces for a test")),
        arena=wide.arena,
    )
    assert _mapped_topology_difference(wide, without, _IDENTITY_3) == _FACES


def _rotated_grid() -> Environment:
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    return Environment.from_samples(_samples(), bin_size=3.0).apply_transform(
        AffineND(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
    )


_ROUND_TRIP_LAYOUTS = [
    pytest.param(
        lambda: Environment.from_samples(_samples(), bin_size=3.0), id="samples"
    ),
    pytest.param(
        lambda: Environment.from_polygon(Point(15, 15).buffer(15), bin_size=2.0),
        id="polygon",
    ),
    pytest.param(_t_maze, id="track"),
    pytest.param(_mesh, id="mesh", marks=pytest.mark.filterwarnings(_MESH_WARNING)),
    pytest.param(_hexagons, id="hexagons"),
    pytest.param(_rotated_grid, id="rotated grid"),
]


@pytest.mark.parametrize("build", _ROUND_TRIP_LAYOUTS)
def test_nwb_restored_environment_equals_its_original(build, tmp_path) -> None:
    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    env = build().with_metadata(units="cm", frame="room")

    nwbfile = pynwb.NWBFile(
        session_description="environment comparison round trip",
        identifier=str(uuid4()),
        session_start_time=datetime.now().astimezone(),
    )
    write_environment(nwbfile, env)
    path = tmp_path / "environment.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        restored = read_environment(io.read())

    assert restored._geometry is not env._geometry
    assert _environment_differences(env, restored) == (None, None)


# ---------------------------------------------------------------------------
# Topology through a one-to-one pairing of bins
# ---------------------------------------------------------------------------


def _corridor(*walls: LineString) -> Environment:
    """Five 10-unit bins in a row; a wall at x closes the pair it crosses."""
    return Environment.from_grid_mask(
        np.ones((5, 1), dtype=bool),
        grid_edges=(np.arange(6.0) * 10, np.arange(2.0) * 10),
        arena=OpenArena(box(0, 0, 50, 10), walls=walls),
    )


def _wall_at(x: float) -> LineString:
    return LineString([(x, 0), (x, 10)])


_ORDER = np.array([3, 0, 4, 1, 2])
_RENUMBERING = np.column_stack([_ORDER, np.arange(5)])


def test_renumbered_bins_have_the_same_topology_through_the_mapping() -> None:
    corridor = _corridor()
    copy = renumbered(corridor, _ORDER)
    assert _environment_differences(corridor, copy).topology is not None
    assert _mapped_topology_difference(corridor, copy, _RENUMBERING) is None


def test_a_changed_distance_between_mapped_bins_is_a_topology_difference() -> None:
    corridor = _corridor()
    distances = corridor.binning.connections.pair_distances.copy()
    distances[0] = 11.0
    copy = renumbered(corridor, _ORDER, distances)
    assert (
        _mapped_topology_difference(corridor, copy, _RENUMBERING)
        == "weighted connectivity"
    )


_OUTSIDE = "connections from mapped bins to bins outside the mapping"


@pytest.mark.parametrize("subset_side", ["first", "second"])
def test_a_subset_whose_border_bins_had_outside_neighbors_differs(
    subset_side: str,
) -> None:
    # Bins 1 and 3 of the corridor neighbor bin 2, which the subset drops, so
    # they were smoothed over fewer neighbors in the subset.
    corridor = _corridor()
    kept = np.array([True, True, False, True, True])
    pairs = np.column_stack([np.flatnonzero(kept), np.arange(4)])
    subset = corridor.subset(mask=kept)
    if subset_side == "first":
        first, second, pairs = subset, corridor, pairs[:, ::-1]
    else:
        first, second = corridor, subset
    assert _mapped_topology_difference(first, second, pairs) == _OUTSIDE


_PARTIAL = np.array([0, 1, 3, 4])


@pytest.mark.parametrize("copy", ["same", "with_metadata"])
def test_a_partial_mapping_between_identical_environments_has_no_difference(
    copy: str,
) -> None:
    # Bins 1 and 3 neighbor unmapped bin 2 in both environments, at the same
    # place and distance, so every mapped bin has the same neighbors.
    corridor = _corridor()
    other = corridor if copy == "same" else corridor.with_metadata(name="copy")
    pairs = np.column_stack([_PARTIAL, _PARTIAL])
    assert _mapped_topology_difference(corridor, other, pairs) is None


def test_an_outside_neighbor_at_another_distance_is_a_difference() -> None:
    corridor = _corridor()
    distances = corridor.binning.connections.pair_distances.copy()
    changed = np.flatnonzero(np.all(np.sort(corridor.edge_pairs) == [1, 2], axis=1))
    distances[changed] = 11.0
    longer = renumbered(corridor, np.arange(5), distances)
    pairs = np.column_stack([_PARTIAL, _PARTIAL])
    assert _mapped_topology_difference(corridor, longer, pairs) == _OUTSIDE


def _three_points_and_a_neighbor(neighbor: tuple[float, float]) -> Environment:
    """Bins 0-1-2 in a row; bin 3 at ``neighbor`` joins bin 1, 10 units away."""
    centers = np.array([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), neighbor])
    return Environment.from_bins(
        Binning(
            "explicit",
            SpatialBins.from_arrays(centers, np.ones(4), 2),
            BinConnections.from_arrays([[0, 1], [1, 2], [1, 3]], [10.0] * 3),
            NearestBinLocator(centers, max_distance=1.0),
            MissingFaces("point bins for a test"),
        )
    )


def test_an_outside_neighbor_at_another_place_is_a_difference() -> None:
    above = _three_points_and_a_neighbor((10.0, 10.0))
    pairs = np.column_stack([np.arange(3), np.arange(3)])
    same_place = _three_points_and_a_neighbor((10.0, 10.0))
    below = _three_points_and_a_neighbor((10.0, -10.0))
    assert _mapped_topology_difference(above, same_place, pairs) is None
    assert _mapped_topology_difference(above, below, pairs) == _OUTSIDE


def test_a_whole_connected_component_has_the_same_topology() -> None:
    # Two corridors of three bins with a gap between them; the second is a
    # separate connected component with no edges leaving it.
    two_pieces = Environment.from_grid_mask(
        np.array([True, True, True, False, True, True, True])[:, None],
        grid_edges=(np.arange(8.0) * 10, np.arange(2.0) * 10),
    )
    assert two_pieces.n_bins == 6
    kept = np.array([False, False, False, True, True, True])
    piece = two_pieces.subset(mask=kept)
    pairs = np.column_stack([np.flatnonzero(kept), np.arange(3)])
    assert _mapped_topology_difference(two_pieces, piece, pairs) is None


def test_a_barrier_between_mapped_bins_is_a_topology_difference() -> None:
    kept = np.array([True, True, True, False, False])
    pairs = np.column_stack([np.flatnonzero(kept), np.arange(3)])
    subset = _corridor(_wall_at(20.0)).subset(mask=kept)
    assert (
        _mapped_topology_difference(_corridor(), subset, pairs)
        == "weighted connectivity"
    )


def test_a_wall_that_closes_no_pair_is_still_a_topology_difference() -> None:
    # A wall along the corridor's side crosses no line between bin centers.
    walled = _corridor(LineString([(1, 8), (4, 8)]))
    np.testing.assert_array_equal(walled.edge_pairs, _corridor().edge_pairs)
    identity = np.column_stack([np.arange(5), np.arange(5)])
    assert (
        _mapped_topology_difference(_corridor(), walled, identity)
        == "continuous arena or physical walls"
    )


# ---------------------------------------------------------------------------
# Boundary ownership does not depend on bin numbering
# ---------------------------------------------------------------------------

# Parent cells 12 to 15 of the wall-split arena are cut in two by the wall;
# their pieces are bins 12 to 19, and pieces 12 and 13 share the wall
# segment x = 12.5, 0 <= y <= 5.
_SPLIT_PIECES = ([12, 13], [14, 15], [16, 17], [18, 19])


def _wall_split_arena() -> Environment:
    return Environment.from_samples(
        np.random.default_rng(0).uniform(0, 30, (4000, 2)),
        bin_size=5.0,
        units="cm",
        arena=OpenArena(box(0, 0, 30, 30), walls=[LineString([(12.5, 0), (12.5, 20)])]),
        wall_fit="split",
    )


def _order_preserving_permutation(n_bins: int) -> np.ndarray:
    """A seeded bin order that keeps each split cell's pieces in order."""
    order = np.random.default_rng(1).permutation(n_bins)
    for pieces in _SPLIT_PIECES:
        positions = np.flatnonzero(np.isin(order, pieces))
        order[positions] = np.sort(order[positions])
    return order


def _renumbering(order: np.ndarray) -> np.ndarray:
    return np.column_stack([order, np.arange(len(order))])


def test_the_wall_split_arena_cuts_four_cells_in_two() -> None:
    env = _wall_split_arena()
    sources = env.clip_mapping.parent_bin_indices
    assert env.n_bins == 40
    np.testing.assert_array_equal(
        np.flatnonzero(np.bincount(sources) > 1), [12, 13, 14, 15]
    )
    for parent, pieces in zip([12, 13, 14, 15], _SPLIT_PIECES, strict=True):
        np.testing.assert_array_equal(np.flatnonzero(sources == parent), pieces)


def test_a_piece_cedes_the_line_it_shares_with_an_earlier_piece() -> None:
    env = _wall_split_arena()
    rows = np.arange(env.n_bins)
    cells = np.empty(env.n_bins, dtype=object)
    cells[:] = env.binning.shapes.polygons
    _, keys, ceded = _ownership_keys(env.binning.locator, rows, cells)
    np.testing.assert_array_equal(keys, env.clip_mapping.parent_bin_indices)
    # The second piece of each cut cell cedes its stretch of the wall.
    later = np.array([second for _, second in _SPLIT_PIECES])
    for piece, low in zip(later, [0.0, 5.0, 10.0, 15.0], strict=True):
        assert shapely.equals(ceded[piece], LineString([(12.5, low), (12.5, low + 5)]))
    # Every other bin cedes nothing, and nothing equals nothing.
    others = ~np.isin(rows, later)
    assert np.all(shapely.is_empty(ceded[others]))
    assert np.all(shapely.equals(ceded[others], ceded[others][::-1]))


def test_an_order_preserving_renumbering_of_split_cells_is_the_same_cells() -> None:
    env = _wall_split_arena()
    order = _order_preserving_permutation(env.n_bins)
    copy = renumbered_cells(env, order)
    assert _mapped_bin_difference(env, copy, _renumbering(order)) is None


def test_swapping_two_touching_pieces_changes_boundary_ownership() -> None:
    env = _wall_split_arena()
    order = np.arange(env.n_bins)
    order[[12, 13]] = [13, 12]
    copy = renumbered_cells(env, order)
    # The point on the line between the pieces moves to the other piece.
    on_the_line = [[12.5, 2.0]]
    assert order[copy.bin_at(on_the_line)[0]] != env.bin_at(on_the_line)[0]
    assert (
        _mapped_bin_difference(env, copy, _renumbering(order))
        == "bin-boundary ownership"
    )


def test_a_renumbered_clip_that_kept_every_cell_whole_is_the_same_cells() -> None:
    disk = Environment.from_grid_mask(
        np.ones((6, 6), dtype=bool), grid_edges=(np.arange(7.0) * 5,) * 2, units="cm"
    ).clip(Point(15, 15).buffer(13))
    order = np.random.default_rng(2).permutation(disk.n_bins)
    copy = renumbered_cells(disk, order)
    assert _mapped_bin_difference(disk, copy, _renumbering(order)) is None


def test_a_subset_of_a_split_clip_maps_onto_its_parent() -> None:
    env = _wall_split_arena()
    kept = np.flatnonzero(np.arange(env.n_bins) != 20)
    part = env.subset(mask=np.arange(env.n_bins) != 20)
    pairs = np.column_stack([kept, np.arange(len(kept))])
    assert _mapped_bin_difference(env, part, pairs) is None


def test_independently_built_split_arenas_agree() -> None:
    first, second = _wall_split_arena(), _wall_split_arena()
    assert first._geometry is not second._geometry
    assert _environment_differences(first, second) == (None, None)


# ---------------------------------------------------------------------------
# Tracks are compared by their geometry
# ---------------------------------------------------------------------------


def _track(
    nodes: list[tuple[float, float]],
    edges: list[tuple[int, int]],
    spacing: float = 0.0,
    bin_size: float = 10.0,
) -> Environment:
    graph = nx.Graph()
    for node, pos in enumerate(nodes):
        graph.add_node(node, pos=pos)
    for u, v in edges:
        graph.add_edge(u, v, distance=float(np.hypot(*np.subtract(nodes[u], nodes[v]))))
    return Environment.from_graph(
        graph=graph,
        edge_order=edges,
        edge_spacing=spacing,
        bin_size=bin_size,
        track_width=2.0,
        units="cm",
    )


def _crossing_l_tracks() -> tuple[Environment, Environment]:
    """Two L-shaped tracks whose two bins have the same centers and lengths."""
    edges = [(0, 1), (1, 2)]
    return (
        _track([(-5.0, 0.0), (5.0, 0.0), (5.0, 10.0)], edges),
        _track([(0.0, -5.0), (0.0, 5.0), (10.0, 5.0)], edges),
    )


def test_tracks_with_equal_bin_centers_along_different_segments_differ() -> None:
    # (3, 0) lies on the first track's first segment and on neither segment
    # of the second track, so the bins do not mean the same places.
    first, second = _crossing_l_tracks()
    np.testing.assert_array_equal(first.bin_centers, second.bin_centers)
    np.testing.assert_array_equal(first.bin_sizes, second.bin_sizes)
    np.testing.assert_array_equal(first.bin_at([[3.0, 0.0]]), [0])
    np.testing.assert_array_equal(second.bin_at([[3.0, 0.0]]), [-1])
    assert _environment_differences(first, second).bins == "track geometry"


def test_a_track_with_different_edge_spacing_differs_in_track_geometry() -> None:
    # The gap between segments shifts every later bin's linear coordinate.
    nodes = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    edges = [(0, 1), (1, 2)]
    touching = _track(nodes, edges, spacing=0.0, bin_size=2.0)
    spaced = _track(nodes, edges, spacing=5.0, bin_size=2.0)
    np.testing.assert_array_equal(touching.bin_centers, spaced.bin_centers)
    np.testing.assert_array_equal(touching.bin_sizes, spaced.bin_sizes)
    assert _environment_differences(touching, spaced) == ("track geometry", None)


def test_identically_built_tracks_agree() -> None:
    first, second = _t_maze(), _t_maze()
    assert first._geometry is not second._geometry
    assert _environment_differences(first, second) == (None, None)


def test_a_track_subset_maps_onto_its_parent() -> None:
    track = _t_maze()
    part = track.subset(mask=np.arange(track.n_bins) != 2)
    kept = np.flatnonzero(np.arange(track.n_bins) != 2)
    pairs = np.column_stack([kept, np.arange(len(kept))])
    assert _mapped_bin_difference(track, part, pairs) is None
    mapped = part.bin_mapping_from(track).pairs
    np.testing.assert_array_equal(mapped[np.argsort(mapped[:, 1])], pairs)


def test_explicit_point_bins_are_compared_by_center_and_measure() -> None:
    # Explicit point bins record no track, so a copy of a track's centers,
    # lengths and measured dimensions is the same bins.
    track, _ = _crossing_l_tracks()
    bins = track.binning.bins
    points = Environment.from_bins(
        Binning(
            "explicit",
            SpatialBins.from_arrays(bins.centers, bins.sizes, bins.measure_dims),
            track.binning.connections,
            NearestBinLocator(bins.centers, max_distance=5.0),
            MissingFaces("explicit point bins without cells"),
        ),
        units="cm",
        arena=track.arena,
    )
    assert points.binning.linearization is None
    assert _environment_differences(track, points).bins is None


def test_decoding_refuses_rates_from_a_different_track() -> None:
    decoding_track, rates_track = _crossing_l_tracks()
    rates = SpatialRatesResult(
        firing_rates=np.array([[1.0, 2.0]]),
        occupancy=np.ones(2),
        env=rates_track,
        method="binned",
        bandwidth=5.0,
    )
    with pytest.raises(IncompatibleEnvironmentError, match="track geometry"):
        decode_position(decoding_track, np.array([[0], [1]]), rates, dt=0.1)
