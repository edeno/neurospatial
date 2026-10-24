"""Geometry records refuse inconsistent states when they are constructed."""

from dataclasses import fields, replace

import networkx as nx
import numpy as np
import pytest
from environment_helpers import publish_binning

from neurospatial import Environment
from neurospatial.arena import EgocentricSurround, MaskArena, TrackArena
from neurospatial.binning import (
    BinConnections,
    Binning,
    FaceGeometry,
    FaceNormals,
    FramedGrid,
    HexagonShapes,
    HexLocator,
    KnownFaces,
    MissingFaces,
    RectilinearGrid,
    RectilinearLocator,
    SpatialBins,
    TrackLinearization,
    TriangleShapes,
)
from neurospatial.geometry import GridIndex, NativeFaces, _CombinedArena
from neurospatial.geometry._codecs import decode, encode


def _track_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(1.0, 0.0))
    graph.add_edge(0, 1, edge_id=0)
    return graph


GRID_RECORDS = {
    "RectilinearGrid": lambda edges, shape, mask: RectilinearGrid(edges, shape, mask),
    "RectilinearLocator": lambda edges, shape, mask: RectilinearLocator(
        edges, shape, mask
    ),
    "MaskArena": lambda edges, shape, mask: MaskArena(edges, mask),
    "TrackLinearization": lambda edges, shape, mask: TrackLinearization(
        _track_graph(), [(0, 1)], 0.0, edges, mask
    ),
}


@pytest.mark.parametrize("record", GRID_RECORDS)
@pytest.mark.parametrize(
    ("edges", "shape", "mask"),
    [
        ((np.array([3.0, 2.0, 1.0]),), (2,), np.ones(2, bool)),  # decreasing
        ((np.array([0.0, np.nan, 2.0]),), (2,), np.ones(2, bool)),  # not finite
        ((np.array([0.0, 1.0, 2.0]),), (2,), np.ones((2, 2), bool)),  # mask shape
        ((np.array([0.0, 1.0, 2.0]),), (2,), np.ones(2, int)),  # not boolean
    ],
    ids=["decreasing", "nan", "mask-shape", "int-mask"],
)
def test_grid_records_refuse_inconsistent_cells(record, edges, shape, mask):
    with pytest.raises(ValueError, match=record):
        GRID_RECORDS[record](edges, shape, mask)


@pytest.mark.parametrize("record", ["RectilinearGrid", "RectilinearLocator"])
def test_grid_records_refuse_a_shape_that_disagrees_with_the_edges(record):
    with pytest.raises(ValueError, match="edges\\[0\\]"):
        GRID_RECORDS[record]((np.array([0.0, 1.0, 2.0]),), (5,), np.ones(5, bool))


def test_grid_records_own_frozen_copies():
    edges, mask = np.array([0.0, 1.0, 2.0]), np.ones(2, bool)
    grid = RectilinearGrid((edges,), (np.int64(2),), mask)
    assert grid.shape == (2,) and type(grid.shape[0]) is int
    assert not grid.edges[0].flags.writeable and not grid.active_mask.flags.writeable
    assert edges.flags.writeable and grid.active_mask is not mask


def test_framed_grid_refuses_a_one_dimensional_source():
    grid = RectilinearGrid((np.array([0.0, 1.0]),), (1,), np.ones(1, bool))
    with pytest.raises(ValueError, match="2-D or 3-D"):
        FramedGrid(grid, np.eye(2))


def _track(graph=None, edge_order=((0, 1),), spacing=0.0, edges=None, mask=None):
    return TrackLinearization(
        _track_graph() if graph is None else graph,
        edge_order,
        spacing,
        (np.array([0.0, 1.0]),) if edges is None else edges,
        np.ones(1, bool) if mask is None else mask,
    )


@pytest.mark.parametrize(
    ("changes", "pattern"),
    [
        ({"edges": (np.array([0.0, 1.0]), np.array([2.0, 1.0]))}, "one linear axis"),
        ({"edge_order": ((5, 9),)}, r"edge_order\[0\]"),
        ({"edge_order": ((0, 1, 2),)}, r"edge_order\[0\]"),
        ({"spacing": -3.0}, "non-negative"),
        ({"spacing": float("nan")}, "non-negative"),
        ({"spacing": (1.0, 2.0)}, "one gap per pair"),
        ({"spacing": "1.0"}, "edge_spacing"),
        ({"graph": nx.DiGraph(_track_graph())}, "undirected"),
    ],
    ids=["two-axes", "not-an-edge", "triple", "negative", "nan", "count", "str", "di"],
)
def test_track_linearization_refuses_inconsistent_numbers(changes, pattern):
    with pytest.raises(ValueError, match=pattern):
        _track(**changes)


def test_track_linearization_names_a_node_without_a_position():
    graph = _track_graph()
    del graph.nodes[0]["pos"]
    with pytest.raises(ValueError, match="node 0 has no numeric 'pos'"):
        _track(graph=graph)


def test_track_linearization_owns_valid_spacing():
    graph = _track_graph()
    graph.add_node(2, pos=(2.0, 0.0))
    graph.add_edge(1, 2, edge_id=1)
    track = _track(
        graph=graph,
        edge_order=[(0, 1), (1, 2)],
        spacing=[np.float32(0.5)],
        edges=(np.array([0.0, 1.0, 1.5, 2.5]),),
        mask=np.array([True, False, True]),
    )
    assert track.edge_spacing == (0.5,) and track.edge_order == ((0, 1), (1, 2))
    assert _track(spacing=np.int64(2)).edge_spacing == 2.0


def test_grid_index_owns_a_frozen_mask():
    mask = np.ones(2, bool)
    index = GridIndex((np.arange(3.0),), (2,), mask)
    assert not index.active_mask.flags.writeable and index.active_mask is not mask
    assert mask.flags.writeable
    with pytest.raises(ValueError, match="GridIndex"):
        GridIndex((np.arange(5.0),), (9,), mask)
    with pytest.raises(ValueError, match="GridIndex"):
        GridIndex((), (3,), mask)


@pytest.mark.parametrize(
    "build",
    [
        lambda: HexagonShapes(np.zeros((2, 2)), -1.0, 0.0),
        lambda: HexagonShapes(np.zeros((2, 2)), 1.0, float("nan")),
        lambda: HexagonShapes(np.zeros((2, 3)), 1.0, 0.0),
        lambda: TriangleShapes(np.zeros((3, 2)), np.array([[0, 1, 99]])),
        lambda: TriangleShapes(np.zeros((3, 2)), np.array([[0, 1]])),
        lambda: HexLocator(0.0, 0.0, -1.0, (1, 2), np.arange(2)),
        lambda: HexLocator(0.0, float("inf"), 1.0, (1, 2), np.arange(2)),
        lambda: HexLocator(0.0, 0.0, 1.0, (3, 3), np.arange(2)),
        lambda: HexLocator(0.0, 0.0, 1.0, (1, 2), np.array([0, 0])),
        lambda: HexLocator(0.0, 0.0, 1.0, (2,), np.arange(2)),
    ],
    ids=[
        "hex-radius",
        "hex-orientation",
        "hex-3d-centers",
        "triangle-index",
        "triangle-shape",
        "locator-radius",
        "locator-offset",
        "locator-mapping-size",
        "locator-duplicate-bin",
        "locator-shape",
    ],
)
def test_cell_shape_records_refuse_impossible_geometry(build):
    with pytest.raises(ValueError, match=r"Hexagon|Triangle|HexLocator"):
        build()


def test_chained_track_arena_transforms_stay_similarities():
    track = TrackArena(np.array([[[0.0, 0.0], [10.0, 0.0]]]), 2.0)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shear = np.array([[1.0, 3.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = track.transformed(rotation)
    assert rotated.transformed(rotation).as_track() is not None
    with pytest.raises(ValueError, match="similarity"):
        rotated.transformed(shear)
    with pytest.raises(ValueError, match="similarity"):
        _CombinedArena((track,)).transformed(shear)


def test_rectilinear_locator_numbers_bins_by_its_mask():
    """The active-bin numbering follows the mask, so lookup and centers agree."""
    edges = (np.array([0.0, 1.0, 2.0]),)
    locator = RectilinearLocator(edges, (2,), np.array([True, False]))
    np.testing.assert_array_equal(locator.active_centers().ravel(), [0.5])
    np.testing.assert_array_equal(locator.locate(np.array([[0.5], [1.5]])), [0, -1])
    with pytest.raises(TypeError):
        RectilinearLocator(edges, (2,), np.array([True, False]), np.array([-1, 0]))


def test_rectilinear_locator_codec_refuses_the_old_record_format():
    locator = RectilinearLocator(
        (np.array([0.0, 1.0, 2.0]),), (2,), np.array([True, True])
    )
    meta, arrays = encode(locator, "part")
    assert meta["codec_version"] == 2
    assert set(arrays) == {"part/edges_0", "part/active_mask"}
    with pytest.raises(ValueError, match="codec_version 1"):
        decode({**meta, "codec_version": 1}, arrays, "part")


def _two_bins(**changes):
    edges = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))
    parts = {
        "kind": "pair",
        "bins": SpatialBins.from_arrays([[0.5, 0.5], [1.5, 0.5]], [1.0, 1.0], 2),
        "connections": BinConnections.from_arrays([[0, 1]], [1.0]),
        "locator": RectilinearLocator(edges, (2, 1), np.ones((2, 1), bool)),
        "faces": KnownFaces(
            FaceGeometry(np.array([[0, 1]]), np.ones(1), np.ones(1), np.ones(2))
        ),
        "grid": RectilinearGrid(edges, (2, 1), np.ones((2, 1), bool)),
    }
    return Binning(**(parts | changes))


@pytest.mark.parametrize(
    ("changes", "error", "pattern"),
    [
        ({"kind": 5}, TypeError, "kind"),
        (
            {"connections": BinConnections.from_arrays([[0, 7]], [1.0])},
            ValueError,
            r"pairs\(0, 1\) must be in \[0, 2\)",
        ),
        (
            {
                "faces": KnownFaces(
                    FaceGeometry(np.array([[0, 1]]), np.ones(1), np.ones(1), np.ones(3))
                )
            },
            ValueError,
            r"faces\.volumes",
        ),
        (
            {
                "faces": KnownFaces(
                    FaceGeometry(np.array([[1, 0]]), np.ones(1), np.ones(1), np.ones(2))
                )
            },
            ValueError,
            r"faces\.edges",
        ),
        (
            {"face_normals": FaceNormals(np.array([[1.0, 0.0, 0.0]]))},
            ValueError,
            "face_normals",
        ),
    ],
    ids=["kind", "pair-bin", "volumes", "edges", "normals"],
)
def test_binning_refuses_records_that_disagree_in_shape(changes, error, pattern):
    _two_bins()  # the unchanged records are consistent
    with pytest.raises(error, match=pattern):
        _two_bins(**changes)


def test_binning_holds_one_grid_description():
    """A binning's grid is axis-aligned or framed, never both."""
    binning = _two_bins()
    assert binning.rectilinear is binning.grid and binning.grid_frame is None
    with pytest.raises(TypeError):
        _two_bins(rectilinear=binning.grid)
    with pytest.raises(TypeError):
        _two_bins(grid_frame=None)


def test_binning_codec_refuses_the_two_grid_field_format():
    meta, arrays = encode(_two_bins(), "binning")
    assert meta["codec_version"] == 6 and meta["grid"]["codec"] == "RectilinearGrid"
    assert "rectilinear" not in meta and "grid_frame" not in meta
    with pytest.raises(ValueError, match="codec_version 5"):
        decode({**meta, "codec_version": 5}, arrays, "binning")


def test_missing_faces_reason_is_a_string():
    with pytest.raises(TypeError, match="reason"):
        MissingFaces(42)


def test_native_faces_follow_the_pairs_of_the_binning_that_holds_them():
    """A native face recipe keeps no pairs of its own to disagree with."""
    rng = np.random.default_rng(0)
    env = Environment.from_samples(rng.uniform(0, 10, (2000, 2)), bin_size=2.0)
    binning = env.binning
    assert isinstance(binning.faces, NativeFaces)
    assert {item.name for item in fields(NativeFaces)}.isdisjoint(
        {"bins", "connections"}
    )
    keep = np.arange(len(binning.connections.pairs)) % 2 == 0
    closed = replace(
        binning, connections=binning.connections.closed(keep), face_normals=None
    )
    faces = closed.build_faces()
    np.testing.assert_array_equal(faces.edges, closed.connections.pairs)
    np.testing.assert_array_equal(faces.volumes, closed.bins.sizes)
    published = Environment.from_bins(closed, arena=env.arena)
    np.testing.assert_array_equal(published.edge_pairs, closed.connections.pairs)


def test_a_partial_angle_window_cannot_wrap():
    with pytest.raises(ValueError, match="less than a full turn"):
        EgocentricSurround((0.0, 10.0), (0.0, 0.1), True)
    assert EgocentricSurround((0.0, 10.0), (-np.pi, np.pi), True).circular


def _polar(circular_angle=None):
    return Environment.from_polar_egocentric(
        distance_range=(0, 10),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5,
        angle_bin_size=np.pi / 2,
        circular_angle=circular_angle,
    )


def _publish_polar(binning, arena):
    return publish_binning(binning, coordinate_kind="polar", arena=arena)


def test_seam_pairs_exist_exactly_when_the_arena_angle_wraps():
    wrapped, unwrapped = _polar(), _polar(circular_angle=False)
    assert wrapped.arena.circular and not unwrapped.arena.circular
    _publish_polar(wrapped.binning, wrapped.arena)  # consistent records publish
    with pytest.raises(ValueError, match="across the angular seam"):
        _publish_polar(
            replace(wrapped.binning, faces=MissingFaces("seam check only")),
            unwrapped.arena,
        )
    with pytest.raises(ValueError, match="not paired across the seam"):
        _publish_polar(
            replace(unwrapped.binning, faces=MissingFaces("seam check only")),
            wrapped.arena,
        )
    with pytest.raises(ValueError, match="polar face recipe"):
        _publish_polar(
            replace(unwrapped.binning, faces=wrapped.binning.faces),
            unwrapped.arena,
        )
