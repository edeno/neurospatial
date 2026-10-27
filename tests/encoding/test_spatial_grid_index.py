"""Place-field components follow retained graph edges across grid contexts."""

import numpy as np
from environment_helpers import publish_binning, publish_environment

from neurospatial import Environment
from neurospatial.encoding._field_metrics import _seed_component
from neurospatial.environment._fields import face_geometry
from neurospatial.layout import create_layout


def test_hex_logical_diagonals_do_not_join_unconnected_fields():
    env = Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0, 12), (0, 12)]
        )
    )
    mask = np.zeros(env.n_bins, dtype=bool)
    mask[[0, 5]] = True
    assert not env.connectivity.has_edge(0, 5)
    np.testing.assert_array_equal(_seed_component(0, mask, env), [0])


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
        dimension_ranges=original.dimension_ranges,
        layout_type="MaskedGrid",
        build_params={"connect_diagonal_neighbors": True},
    )
    env = publish_environment(layout=legacy, layout_type_used="MaskedGrid")
    # A detached provenance export cannot change the retained graph.
    env.layout_parameters["connect_diagonal_neighbors"] = False
    mask = np.array([True, False, False, True])
    np.testing.assert_array_equal(_seed_component(0, mask, env), [0, 3])


def test_explicit_bins_segment_using_supplied_diagonal_connections():
    from dataclasses import replace

    from neurospatial.binning import KnownFaces

    original = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool),
        grid_edges=(np.arange(3.0),) * 2,
        connect_diagonal_neighbors=True,
    )
    env = Environment.from_bins(
        replace(
            original._binning,
            faces=KnownFaces(face_geometry(original._cache, original._binning)),
        )
    )
    assert env._binning.grid_spec.connect_diagonal
    mask = np.array([True, False, False, True])
    np.testing.assert_array_equal(_seed_component(0, mask, env), [0, 3])


def test_pickle_preserves_segmentation_connections():
    import pickle

    original = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool),
        grid_edges=(np.arange(3.0),) * 2,
        connect_diagonal_neighbors=True,
    )
    restored = pickle.loads(pickle.dumps(original))
    mask = np.array([True, False, False, True])
    np.testing.assert_array_equal(_seed_component(0, mask, restored), [0, 3])


def test_from_layout_segmentation_keeps_resolved_default():
    env = Environment.from_layout(
        create_layout(
            "MaskedGrid",
            active_mask=np.ones((2, 2), dtype=bool),
            grid_edges=(np.arange(3.0),) * 2,
        )
    )
    # The engine's record keeps the default it resolved.
    assert env.layout_parameters["connect_diagonal_neighbors"] is True
    mask = np.array([True, False, False, True])
    np.testing.assert_array_equal(_seed_component(0, mask, env), [0, 3])


def test_supplied_arena_keeps_resolved_segmentation_default():
    from shapely.geometry import box

    from neurospatial.arena import OpenArena

    env = Environment.from_layout(
        create_layout(
            "MaskedGrid",
            active_mask=np.ones((2, 2), dtype=bool),
            grid_edges=(np.arange(3.0),) * 2,
        ),
        arena=OpenArena(box(-1, -1, 3, 3)),
    )
    np.testing.assert_array_equal(
        _seed_component(0, np.array([True, False, False, True]), env),
        [0, 3],
    )
