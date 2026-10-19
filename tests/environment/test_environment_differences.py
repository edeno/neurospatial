"""Environment comparison separates bin meaning from topology.

Decoding and rate comparisons ask the same question of two environments: do
their bins mean the same places, and is the space connected the same way? A
door or barrier experiment keeps the bins but changes connectivity, so the two
answers are reported separately.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.binning import (
    BinConnections,
    Binning,
    MissingFaces,
    NearestBinLocator,
    SpatialBins,
)
from neurospatial.environment import (
    _environment_differences,
    _mapped_topology_difference,
)


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


@pytest.mark.parametrize("layout", ["samples", "polygon", "track"])
def test_nwb_restored_environment_equals_its_original(layout: str, tmp_path) -> None:
    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    if layout == "samples":
        env = Environment.from_samples(_samples(), bin_size=3.0)
    elif layout == "polygon":
        env = Environment.from_polygon(Point(15, 15).buffer(15), bin_size=2.0)
    else:
        env = _t_maze()
    env = env.with_metadata(units="cm", frame="room")

    nwbfile = pynwb.NWBFile(
        session_description="environment comparison round trip",
        identifier=str(uuid4()),
        session_start_time=datetime.now().astimezone(),
    )
    write_environment(nwbfile, env)
    path = tmp_path / f"{layout}.nwb"
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


def _renumbered(env: Environment, order, distances=None) -> Environment:
    """``env``'s bins and connections, with new bin ``i`` = old bin ``order[i]``."""
    binning = env.binning
    centers = binning.bins.centers[order]
    connections = binning.connections
    return Environment.from_bins(
        Binning(
            "explicit",
            SpatialBins.from_arrays(
                centers, binning.bins.sizes[order], binning.bins.measure_dims[order]
            ),
            BinConnections.from_arrays(
                np.argsort(order)[connections.pairs],
                connections.pair_distances if distances is None else distances,
            ),
            NearestBinLocator(centers, max_distance=5.0),
            MissingFaces("renumbered copy for a test"),
        ),
        arena=env.arena,
    )


_ORDER = np.array([3, 0, 4, 1, 2])
_RENUMBERING = np.column_stack([_ORDER, np.arange(5)])


def test_renumbered_bins_have_the_same_topology_through_the_mapping() -> None:
    corridor = _corridor()
    renumbered = _renumbered(corridor, _ORDER)
    assert _environment_differences(corridor, renumbered).topology is not None
    assert _mapped_topology_difference(corridor, renumbered, _RENUMBERING) is None


def test_a_changed_distance_between_mapped_bins_is_a_topology_difference() -> None:
    corridor = _corridor()
    distances = corridor.binning.connections.pair_distances.copy()
    distances[0] = 11.0
    renumbered = _renumbered(corridor, _ORDER, distances)
    assert (
        _mapped_topology_difference(corridor, renumbered, _RENUMBERING)
        == "weighted connectivity"
    )


def test_a_subset_keeps_the_topology_among_its_bins() -> None:
    corridor = _corridor()
    kept = np.array([True, True, False, True, True])
    pairs = np.column_stack([np.flatnonzero(kept), np.arange(4)])
    subset = corridor.subset(mask=kept)
    assert _mapped_topology_difference(corridor, subset, pairs) is None


def test_a_barrier_between_mapped_bins_is_a_topology_difference() -> None:
    kept = np.array([True, True, True, False, False])
    pairs = np.column_stack([np.flatnonzero(kept), np.arange(3)])
    subset = _corridor(_wall_at(20.0)).subset(mask=kept)
    assert (
        _mapped_topology_difference(_corridor(), subset, pairs)
        == "weighted connectivity"
    )


def test_a_wall_outside_the_mapped_bins_is_still_a_topology_difference() -> None:
    kept = np.array([True, True, True, False, False])
    pairs = np.column_stack([np.flatnonzero(kept), np.arange(3)])
    subset = _corridor(_wall_at(40.0)).subset(mask=kept)
    assert subset.connectivity.has_edge(0, 1) and subset.connectivity.has_edge(1, 2)
    assert (
        _mapped_topology_difference(_corridor(), subset, pairs)
        == "continuous arena or physical walls"
    )
