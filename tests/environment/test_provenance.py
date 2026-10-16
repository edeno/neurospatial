"""Recorded build provenance holds scalar choices, never the input data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.io import from_dict, from_file, to_dict, to_file
from neurospatial.layout import create_layout

_SCALARS = (bool, int, float, str, type(None))


def _track():
    graph = nx.Graph()
    for node, pos in enumerate([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]):
        graph.add_node(node, pos=pos)
    graph.add_edge(0, 1, distance=10.0)
    graph.add_edge(1, 2, distance=10.0)
    return Environment.from_graph(
        graph, edge_order=[(0, 1), (1, 2)], edge_spacing=0.0, bin_size=1.0
    )


FACTORIES = {
    "from_samples": lambda: Environment.from_samples(
        np.random.default_rng(0).uniform(0, 20, (500, 2)), bin_size=(2.0, 4.0)
    ),
    "hexagonal": lambda: Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=3.0, dimension_ranges=[(0, 12), (0, 12)]
        )
    ),
    "from_polygon": lambda: Environment.from_polygon(
        Point(10, 10).buffer(10), bin_size=2.0
    ),
    "from_grid_mask": lambda: Environment.from_grid_mask(
        np.ones((6, 6), dtype=bool),
        grid_edges=(np.arange(7.0),) * 2,
        arena=OpenArena(box(0, 0, 4, 6)),
    ),
    "from_pixel_mask": lambda: Environment.from_pixel_mask(
        np.ones((6, 6), dtype=bool), pixel_size=(2.0, 1.0)
    ),
    "from_graph": _track,
    "mesh": lambda: Environment.from_layout(
        create_layout(
            "TriangularMesh",
            boundary_polygon=Polygon([(0, 0), (8, 0), (8, 8), (0, 8)]),
            point_spacing=2.0,
        )
    ),
    "mesh_with_hole": lambda: Environment.from_layout(
        create_layout(
            "TriangularMesh",
            boundary_polygon=Polygon(
                [(0, 0), (8, 0), (8, 8), (0, 8)], [[(3, 3), (5, 3), (5, 5), (3, 5)]]
            ),
            point_spacing=2.0,
        )
    ),
    "grid_from_ranges": lambda: Environment.from_layout(
        create_layout("RegularGrid", bin_size=3.0, dimension_ranges=[(0, 12), (0, 12)])
    ),
    "from_polar_egocentric": lambda: Environment.from_polar_egocentric(
        distance_range=(0.0, 10.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=2.0,
        angle_bin_size=np.pi / 4,
    ),
}


def _is_choice(value):
    if isinstance(value, tuple):
        return all(isinstance(item, _SCALARS) for item in value)
    return isinstance(value, _SCALARS)


@pytest.mark.parametrize("factory", sorted(FACTORIES))
def test_layout_parameters_hold_only_scalar_choices(factory):
    env = FACTORIES[factory]()
    params = env.layout_parameters
    assert all(_is_choice(value) for value in params.values()), params
    # Data arguments are not choices, not even as a None placeholder.
    for data in (
        "positions",
        "active_mask",
        "grid_edges",
        "image_mask",
        "polygon",
        "boundary_polygon",
        "graph_definition",
        "edge_order",
        "dimension_ranges",
        "boundary_exterior_coords",
        "boundary_interior_coords_list",
    ):
        assert data not in params
    json.dumps(params)


def test_recorded_choices_keep_their_values():
    samples = FACTORIES["from_samples"]()
    assert samples.layout_parameters["bin_size"] == (2.0, 4.0)
    assert samples.layout_parameters["connect_diagonal_neighbors"] is True
    pixels = FACTORIES["from_pixel_mask"]()
    assert pixels.layout_parameters["pixel_size"] == (2.0, 1.0)
    polar = FACTORIES["from_polar_egocentric"]()
    assert polar.layout_parameters["circular_angle"] is True


@pytest.mark.parametrize("container", [list, tuple, np.array])
def test_per_axis_choices_are_recorded_whatever_the_container(container):
    positions = np.random.default_rng(0).uniform(0, 20, (500, 2))
    samples = Environment.from_samples(positions, bin_size=container([2.0, 4.0]))
    assert samples.layout_parameters["bin_size"] == (2.0, 4.0)
    pixels = Environment.from_pixel_mask(
        np.ones((4, 4), dtype=bool), pixel_size=container([2.0, 1.0])
    )
    assert pixels.layout_parameters["pixel_size"] == (2.0, 1.0)


def test_one_dimensional_masks_are_data_not_choices():
    env = Environment.from_grid_mask(
        np.array([True, True, False, True]), grid_edges=(np.arange(5.0),)
    )
    assert "active_mask" not in env.layout_parameters


def test_track_geometry_is_owned_by_the_binning_not_provenance():
    env = _track()
    assert env.layout_parameters == {"edge_spacing": 0.0, "bin_size": 1.0}
    linearization = env.binning.linearization
    assert list(linearization.graph.edges) == [(0, 1), (1, 2)]
    assert tuple(linearization.edge_order) == ((0, 1), (1, 2))


def _nwb_round_trip(env):
    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    nwbfile = pynwb.NWBFile(
        session_description="provenance",
        identifier="provenance",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    write_environment(nwbfile, env)
    return read_environment(nwbfile)


@pytest.mark.parametrize("how", ["dict", "json", "file", "nwb"])
@pytest.mark.parametrize("factory", sorted(FACTORIES))
def test_round_trips_reproduce_environment_and_provenance(factory, how, tmp_path):
    env = FACTORIES[factory]()
    if how == "dict":
        restored = from_dict(to_dict(env))
    elif how == "json":
        restored = from_dict(json.loads(json.dumps(to_dict(env))))
    elif how == "file":
        to_file(env, tmp_path / "env")
        restored = from_file(tmp_path / "env")
    else:
        restored = _nwb_round_trip(env)
    assert restored.layout_type == env.layout_type
    assert restored._provenance == env._provenance
    assert restored.layout_parameters == env.layout_parameters
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)
    np.testing.assert_array_equal(restored.edge_pairs, env.edge_pairs)
    np.testing.assert_array_equal(restored.bin_sizes, env.bin_sizes)


def _saved_size(n_samples: int, directory: Path) -> int:
    rng = np.random.default_rng(n_samples)
    corners = np.array([[0.0, 0.0], [100.0, 100.0]])
    positions = np.vstack([corners, rng.uniform(0, 100, (n_samples, 2))])
    env = Environment.from_samples(positions, bin_size=5.0, infer_active_bins=False)
    path = directory / f"env{n_samples}"
    to_file(env, path)
    return sum(file.stat().st_size for file in directory.glob(f"env{n_samples}.*"))


def test_saved_file_size_does_not_grow_with_trajectory_length(tmp_path):
    short, long = _saved_size(10, tmp_path), _saved_size(100_000, tmp_path)
    # Identical geometry; only timestamps and the like may differ.
    assert abs(long - short) <= 0.01 * short


def test_a_layout_records_the_same_choices_with_or_without_optional_data():
    mesh = FACTORIES["mesh"]().layout_parameters
    assert mesh == FACTORIES["mesh_with_hole"]().layout_parameters
    assert mesh == {"point_spacing": 2.0}
    samples = Environment.from_samples(
        np.random.default_rng(0).uniform(0, 12, (500, 2)),
        bin_size=3.0,
        layout="Hexagonal",
    )
    assert FACTORIES["hexagonal"]().layout_parameters.keys() == (
        samples.layout_parameters.keys()
    )


def test_custom_engine_keeps_every_recorded_choice_including_none():
    from neurospatial.layout.engines.masked_grid import MaskedGridLayout

    class CustomEngine:
        """A custom engine that declares no data arguments."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

    inner = MaskedGridLayout()
    inner.build(
        active_mask=np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    inner._build_params_used["smoothing_choice"] = None
    params = Environment.from_layout(CustomEngine(inner)).layout_parameters
    assert params["smoothing_choice"] is None
    assert params["connect_diagonal_neighbors"] is True
