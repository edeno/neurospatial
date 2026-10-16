"""Place-field components follow retained graph edges across grid contexts."""

import numpy as np
from environment_helpers import publish_binning, publish_environment

from neurospatial import Environment
from neurospatial.encoding.spatial import _extract_connected_component_graph


def test_hex_logical_diagonals_do_not_join_unconnected_fields():
    env = Environment.from_layout(
        "Hexagonal", {"hexagon_width": 4.0, "dimension_ranges": [(0, 12), (0, 12)]}
    )
    mask = np.zeros(env.n_bins, dtype=bool)
    mask[[0, 5]] = True
    assert not env.connectivity.has_edge(0, 5)
    np.testing.assert_array_equal(_extract_connected_component_graph(0, mask, env), [0])


def test_legacy_grid_segmentation_follows_graph_despite_provenance():
    from dataclasses import replace

    from neurospatial import Environment

    original = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool),
        grid_edges=(np.arange(3.0),) * 2,
        connect_diagonal_neighbors=True,
    )
    bins = replace(original._binning, grid_spec=None)
    legacy = publish_binning(
        bins,
        connectivity=original.connectivity,
        dimension_ranges=original.dimension_ranges,
        layout_type="MaskedGrid",
        category="grid",
        build_params={"connect_diagonal_neighbors": True},
        is_linearized_track=False,
    )
    env = publish_environment(layout=legacy, layout_type_used="MaskedGrid")
    # A detached provenance export cannot change the retained graph.
    env.layout_parameters["connect_diagonal_neighbors"] = False
    mask = np.array([True, False, False, True])
    np.testing.assert_array_equal(
        _extract_connected_component_graph(0, mask, env), [0, 3]
    )


def test_explicit_bins_segment_using_supplied_diagonal_connections():
    from dataclasses import replace

    from neurospatial.binning import KnownFaces

    original = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool),
        grid_edges=(np.arange(3.0),) * 2,
        connect_diagonal_neighbors=True,
    )
    env = Environment.from_bins(
        replace(original._binning, faces=KnownFaces(original._face_geometry))
    )
    assert env._binning.grid_spec.connect_diagonal
    mask = np.array([True, False, False, True])
    np.testing.assert_array_equal(
        _extract_connected_component_graph(0, mask, env), [0, 3]
    )


def test_pickle_preserves_segmentation_connections():
    import pickle

    original = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool),
        grid_edges=(np.arange(3.0),) * 2,
        connect_diagonal_neighbors=True,
    )
    restored = pickle.loads(pickle.dumps(original))
    mask = np.array([True, False, False, True])
    np.testing.assert_array_equal(
        _extract_connected_component_graph(0, mask, restored), [0, 3]
    )


def test_from_layout_segmentation_keeps_resolved_default():
    env = Environment.from_layout(
        "MaskedGrid",
        {
            "active_mask": np.ones((2, 2), dtype=bool),
            "grid_edges": (np.arange(3.0),) * 2,
        },
    )
    assert "connect_diagonal_neighbors" not in env.layout_parameters
    mask = np.array([True, False, False, True])
    np.testing.assert_array_equal(
        _extract_connected_component_graph(0, mask, env), [0, 3]
    )


def test_supplied_arena_keeps_resolved_segmentation_default():
    from shapely.geometry import box

    from neurospatial.arena import OpenArena

    env = Environment.from_layout(
        "MaskedGrid",
        {
            "active_mask": np.ones((2, 2), dtype=bool),
            "grid_edges": (np.arange(3.0),) * 2,
        },
        arena=OpenArena(box(-1, -1, 3, 3)),
    )
    np.testing.assert_array_equal(
        _extract_connected_component_graph(
            0, np.array([True, False, False, True]), env
        ),
        [0, 3],
    )
