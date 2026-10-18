"""Factory and persistence paths publish labels with their geometry."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment, Region, Regions
from neurospatial.arena import OpenArena
from neurospatial.layout import create_layout


def _create(kind, **metadata):
    positions = np.array([[0.0, 0.0], [0.0, 3.0], [3.0, 0.0], [3.0, 3.0]])
    if kind == "samples":
        return Environment.from_samples(
            positions, bin_size=1.0, infer_active_bins=False, **metadata
        )
    if kind == "open_field":
        return Environment.open_field(
            positions, bin_size=1.0, infer_active_bins=False, **metadata
        )
    if kind == "linear_track":
        return Environment.linear_track(
            endpoints=[(0, 0), (4, 0)], bin_size=1, track_width=1.0, **metadata
        )
    if kind == "maze":
        return Environment.maze(
            kind="T",
            node_positions={"c": (0, 0), "s": (0, -2), "l": (-2, 0), "r": (2, 0)},
            bin_size=1,
            track_width=1.0,
            **metadata,
        )
    if kind == "graph":
        import networkx as nx

        graph = nx.Graph()
        graph.add_node(0, pos=(0, 0))
        graph.add_node(1, pos=(4, 0))
        graph.add_edge(0, 1, distance=4.0)
        return Environment.from_graph(
            graph,
            [(0, 1)],
            edge_spacing=0.0,
            bin_size=1.0,
            track_width=1.0,
            **metadata,
        )
    if kind == "polygon":
        return Environment.from_polygon(box(0, 0, 3, 3), bin_size=1.0, **metadata)
    if kind == "grid_mask":
        return Environment.from_grid_mask(
            np.ones((3, 3), dtype=bool), (np.arange(4.0), np.arange(4.0)), **metadata
        )
    if kind == "pixel_mask":
        return Environment.from_pixel_mask(
            np.ones((3, 3), dtype=bool), pixel_size=1.0, **metadata
        )
    if kind == "layout":
        return Environment.from_layout(
            create_layout(
                "MaskedGrid",
                active_mask=np.ones((3, 3), dtype=bool),
                grid_edges=(np.arange(4.0), np.arange(4.0)),
            ),
            **metadata,
        )
    return Environment.from_polar_egocentric(
        distance_range=(1, 3),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=1.0,
        angle_bin_size=np.pi,
        **metadata,
    )


_FACTORIES = [
    "samples",
    "open_field",
    "linear_track",
    "maze",
    "graph",
    "polygon",
    "grid_mask",
    "pixel_mask",
    "layout",
    "polar_public",
]


@pytest.mark.parametrize("kind", _FACTORIES)
def test_all_factories_publish_metadata_without_coordinate_conversion(kind):
    baseline = _create(kind)
    env = _create(kind, name="declared", units="m", frame="camera")
    assert (env.name, env.units, env.frame) == ("declared", "m", "camera")
    np.testing.assert_array_equal(env.bin_centers, baseline.bin_centers)
    np.testing.assert_array_equal(env.bin_sizes, baseline.bin_sizes)
    np.testing.assert_array_equal(
        env.bin_at(env.bin_centers), baseline.bin_at(baseline.bin_centers)
    )


@pytest.mark.parametrize(
    "kind", [kind for kind in _FACTORIES if kind != "polar_public"]
)
def test_every_cartesian_factory_snapshots_regions(kind):
    regions = Regions([Region("goal", "point", [1.0, 0.0])])
    env = _create(kind, regions=regions)
    regions.remove("goal")
    assert env.regions.list_names() == ["goal"]
    np.testing.assert_array_equal(env.regions["goal"].data, [1.0, 0.0])
    np.testing.assert_array_equal(env.bin_centers, _create(kind).bin_centers)


def test_explicit_arena_center_selection_retains_requested_name_labels_and_regions():
    regions = Regions([Region("goal", "point", [0.5, 0.5])])
    env = Environment.from_layout(
        create_layout(
            "MaskedGrid",
            active_mask=np.ones((3, 3), dtype=bool),
            grid_edges=(np.arange(4.0), np.arange(4.0)),
        ),
        arena=OpenArena(box(0, 0, 2, 2)),
        regions=regions,
        name="selected",
        units="cm",
        frame="world",
    )
    assert env.n_bins == 4
    assert (env.name, env.units, env.frame) == ("selected", "cm", "world")
    assert env.regions.list_names() == ["goal"]


def test_sample_factory_snapshots_regions_without_passing_them_to_layout_builder():
    regions = Regions([Region("goal", "point", [0.5, 0.5])])
    env = Environment.from_samples(
        np.array([[0.0, 0.0], [2.0, 2.0]]),
        bin_size=1.0,
        regions=regions,
        infer_active_bins=False,
    )
    regions.remove("goal")
    assert env.regions.list_names() == ["goal"]


@pytest.mark.parametrize("kind", ["grid_mask", "polar_create", "graph"])
@pytest.mark.parametrize("path", ["dict", "file"])
def test_persistence_restores_labels_through_constructor(kind, path, tmp_path):
    env = _create(kind, name="saved", units="mm", frame="camera")
    if path == "dict":
        restored = Environment.from_dict(env.to_dict())
    else:
        file_path = tmp_path / "environment.npz"
        env.to_file(file_path)
        restored = Environment.from_file(file_path)
    assert type(restored) is type(env)
    assert (restored.name, restored.units, restored.frame) == ("saved", "mm", "camera")
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)


@pytest.mark.parametrize("kind", ["grid_mask", "polar_create", "graph"])
def test_factory_preserves_advisory_unit_validation(kind):
    with pytest.warns(UserWarning, match="standard"):
        env = _create(kind, units="lab distance")
    assert env.units == "lab distance"


@pytest.mark.parametrize("kind", ["grid_mask", "polar_create", "graph"])
def test_nwb_roundtrip_preserves_constructor_metadata(kind):
    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    env = _create(kind, name="saved", units="mm", frame="camera")
    nwb = pynwb.NWBFile(
        "metadata test", "metadata test", datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    write_environment(nwb, env)
    restored = read_environment(nwb)
    assert type(restored) is type(env)
    assert (restored.name, restored.units, restored.frame) == ("saved", "mm", "camera")
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)


@pytest.mark.parametrize("attribute", ["name", "units", "frame"])
def test_factory_rejects_invalid_label_types(attribute):
    with pytest.raises(TypeError):
        _create("grid_mask", **{attribute: 1})
