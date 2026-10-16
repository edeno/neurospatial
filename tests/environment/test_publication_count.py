"""Every factory, reader and derivation publishes its environment exactly once."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pytest
from environment_helpers import record_publications
from shapely.geometry import Point, Polygon, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.binning import KnownFaces
from neurospatial.layout import create_layout
from neurospatial.ops.transforms import translate

_POSITIONS = np.random.default_rng(0).uniform(0, 20, (500, 2))


def _track_graph():
    graph = nx.Graph()
    for node, pos in enumerate([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]):
        graph.add_node(node, pos=pos)
    graph.add_edge(0, 1, distance=10.0)
    graph.add_edge(1, 2, distance=10.0)
    return graph


def _grid():
    return Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool), grid_edges=(np.arange(11.0),) * 2
    )


FACTORIES = {
    "from_samples": lambda: Environment.from_samples(_POSITIONS, bin_size=2.0),
    "from_samples_arena": lambda: Environment.from_samples(
        _POSITIONS, bin_size=2.0, arena=OpenArena(box(0, 0, 12, 20))
    ),
    "open_field": lambda: Environment.open_field(_POSITIONS, bin_size=2.0),
    "hexagonal": lambda: Environment.from_samples(
        _POSITIONS, bin_size=3.0, layout="Hexagonal"
    ),
    "from_polygon": lambda: Environment.from_polygon(
        Point(10, 10).buffer(10), bin_size=2.0
    ),
    "from_polygon_arena": lambda: Environment.from_polygon(
        box(0, 0, 10, 10), bin_size=1.0, arena=OpenArena(box(0, 0, 6, 10))
    ),
    "from_grid_mask": _grid,
    "from_grid_mask_arena": lambda: Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool),
        grid_edges=(np.arange(11.0),) * 2,
        arena=OpenArena(box(0, 0, 6, 10)),
    ),
    "from_pixel_mask": lambda: Environment.from_pixel_mask(
        np.ones((6, 6), dtype=bool), pixel_size=2.0
    ),
    "from_graph": lambda: Environment.from_graph(
        _track_graph(), edge_order=[(0, 1), (1, 2)], edge_spacing=0.0, bin_size=1.0
    ),
    "linear_track": lambda: Environment.linear_track(
        endpoints=[(0.0, 0.0), (10.0, 0.0)], bin_size=1.0
    ),
    "maze": lambda: Environment.maze(
        kind="t",
        node_positions={"s": (0, 0), "j": (0, 10), "l": (-10, 10), "r": (10, 10)},
        bin_size=2.0,
    ),
    "from_layout_mesh": lambda: Environment.from_layout(
        create_layout(
            "TriangularMesh",
            boundary_polygon=Polygon([(0, 0), (8, 0), (8, 8), (0, 8)]),
            point_spacing=2.0,
        )
    ),
    "from_polar_egocentric": lambda: Environment.from_polar_egocentric(
        distance_range=(0.0, 10.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=2.0,
        angle_bin_size=np.pi / 4,
    ),
}


@pytest.mark.parametrize("factory", sorted(FACTORIES))
def test_each_factory_publishes_once(factory, monkeypatch):
    published = record_publications(monkeypatch)
    env = FACTORIES[factory]()
    assert published == [env]


def test_from_bins_and_combine_publish_once(monkeypatch):
    left = _grid()
    right = left.apply_transform(translate(20.0, 0.0))
    snapshot = replace(left.binning, faces=KnownFaces(left.binning.faces.build()))
    published = record_publications(monkeypatch)
    from_bins = Environment.from_bins(snapshot, arena=left.arena)
    combined = Environment.combine({"left": left, "right": right})
    assert published == [from_bins, combined]


DERIVATIONS = {
    "subset": lambda env: env.subset(bins=env.bin_centers[:, 0] < 5),
    "clip": lambda env: env.clip(box(1, 1, 6, 6)),
    "rebin_factor": lambda env: env.rebin(2),
    "rebin_bin_size": lambda env: env.rebin(bin_size=2.0),
    "apply_transform": lambda env: env.apply_transform(translate(1.0, 2.0)),
    "with_metadata": lambda env: env.with_metadata(name="renamed"),
}


@pytest.mark.parametrize("derivation", sorted(DERIVATIONS))
def test_each_derivation_publishes_at_most_once(derivation, monkeypatch):
    parent = _grid()
    published = record_publications(monkeypatch)
    child = DERIVATIONS[derivation](parent)
    # Relabelling shares the parent's published geometry.
    expected = [] if derivation == "with_metadata" else [child]
    assert published == expected


@pytest.mark.parametrize("how", ["dict", "file", "nwb"])
def test_readers_publish_once(how, monkeypatch, tmp_path):
    env = _grid()
    if how == "nwb":
        pynwb = pytest.importorskip("pynwb")
        from neurospatial.io.nwb import read_environment, write_environment

        nwbfile = pynwb.NWBFile(
            session_description="count",
            identifier="count",
            session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        write_environment(nwbfile, env)
    published = record_publications(monkeypatch)
    if how == "dict":
        restored = Environment.from_dict(env.to_dict())
    elif how == "file":
        env.to_file(tmp_path / "grid")
        restored = Environment.from_file(tmp_path / "grid")
    else:
        restored = read_environment(nwbfile)
    assert published == [restored]


def _arena_grid(arena):
    return Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool), grid_edges=(np.arange(11.0),) * 2, arena=arena
    )


def test_arena_cropping_matches_subset_and_keeps_factory_provenance():
    cropping = OpenArena(box(0, 0, 6, 10))
    cropped = _arena_grid(cropping)
    full = _grid()
    keep = full.bin_centers[:, 0] < 6
    expected = full.subset(bins=keep)
    assert cropped.n_bins == int(keep.sum()) < full.n_bins
    np.testing.assert_array_equal(cropped.bin_centers, expected.bin_centers)
    np.testing.assert_array_equal(cropped.edge_pairs, expected.edge_pairs)
    np.testing.assert_array_equal(cropped.bin_sizes, expected.bin_sizes)
    np.testing.assert_array_equal(cropped.active_mask, expected.active_mask)
    assert cropped.grid_shape == expected.grid_shape
    assert cropped.dimension_ranges == expected.dimension_ranges
    assert list(cropped.connectivity.edges(data=True)) == list(
        expected.connectivity.edges(data=True)
    )
    # The recorded build is the factory's, whether or not the arena drops bins.
    uncropped = _arena_grid(OpenArena(box(0, 0, 10, 10)))
    assert uncropped.n_bins == full.n_bins
    for env in (cropped, uncropped):
        assert env.layout_type == full.layout_type == "MaskedGrid"
        assert env._provenance == full._provenance
        assert env.layout_parameters == full.layout_parameters
