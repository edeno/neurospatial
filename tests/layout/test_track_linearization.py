"""Track projections retain absolute coordinates, gaps and owned geometry."""

from __future__ import annotations

import copy
import pickle
import warnings
from dataclasses import FrozenInstanceError, replace

import networkx as nx
import numpy as np
import pytest

from neurospatial.geometry._binning import TrackLinearization


def _assert_previous_projection(component, layout, prefix, recorded):
    for operation in ("to_linear", "linear_to_nd", "linear_wrong_shape"):
        method = "linear_to_nd" if operation == "linear_wrong_shape" else operation
        key = f"{prefix}/{operation}"
        points = recorded[f"{key}/points"].copy()
        unchanged = points.copy()
        for target in (component, layout):
            if f"{key}/error_type" in recorded:
                assert recorded[f"{key}/error_type"].item() == "ValueError"
                with pytest.raises(ValueError) as exc:
                    getattr(target, method)(points)
                assert str(exc.value) == recorded[f"{key}/error_message"].item()
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RuntimeWarning)
                    actual = getattr(target, method)(points)
                np.testing.assert_array_equal(actual, recorded[f"{key}/values"])
                messages = [
                    f"{warning.category.__name__}: {warning.message}"
                    for warning in caught
                ]
                assert messages == recorded[f"{key}/warnings"].tolist()
            np.testing.assert_array_equal(points, unchanged)


@pytest.mark.parametrize("built_environment", ["track"], indirect=True)
def test_track_linearization_matches_previous(built_environment, locator_parity):
    """On- and off-track points retain their recorded branching-track projections."""
    layout = _native_oracle(built_environment)
    component = built_environment._binning.linearization
    assert isinstance(component, TrackLinearization)
    assert built_environment._binning.linearization is component
    assert component is built_environment._binning.linearization
    _assert_previous_projection(component, layout, "track/projection", locator_parity)


@pytest.mark.parametrize("spacing_kind", ["scalar", "list"])
def test_gapped_track_linearization_matches_previous(
    graph_layout_with_gap, locator_parity, spacing_kind
):
    """Gap and outside queries retain their previous endpoint projections."""
    layout = graph_layout_with_gap
    if spacing_kind == "list":
        params = layout._build_params_used
        layout.build(
            graph_definition=params["graph_definition"],
            edge_order=params["edge_order"],
            edge_spacing=[10.0, 10.0],
            bin_size=params["bin_size"],
        )
    _assert_previous_projection(
        layout.linearization(), layout, "gapped_track/projection", locator_parity
    )
    assert np.any(~layout.linearization().active_mask)
    np.testing.assert_allclose(
        layout.to_linear(layout.bin_centers), layout.linear_bin_centers_
    )
    np.testing.assert_allclose(
        layout.linear_to_nd(layout.linear_bin_centers_), layout.bin_centers
    )


def test_track_linearization_owns_constructor_geometry(
    graph_layout_with_gap, locator_parity
):
    """Caller graph, order, gaps and grid buffers cannot alter a stored projection."""
    layout = graph_layout_with_gap
    graph = copy.deepcopy(layout._build_params_used["graph_definition"])
    order = list(layout._build_params_used["edge_order"])
    spacing = np.full(len(order) - 1, layout._build_params_used["edge_spacing"])
    edges = tuple(edge.copy() for edge in layout.grid_edges)
    mask = layout.active_mask.copy()
    component = TrackLinearization(graph, order, spacing, edges, mask)
    query = locator_parity["gapped_track/projection/to_linear/points"]
    expected = component.to_linear(query)
    old_order, old_spacing = tuple(order), tuple(spacing)
    old_edges, old_mask = tuple(edge.copy() for edge in edges), mask.copy()

    node = next(iter(graph.nodes))
    graph.nodes[node]["pos"] = (-100.0, -100.0)
    graph.edges[order[0]]["distance"] = 999.0
    order.reverse()
    spacing[:] = 100
    for edge in edges:
        assert edge.flags.writeable
        edge[:] = -1
    assert mask.flags.writeable
    mask[:] = False

    assert tuple(component.edge_order) == old_order
    assert tuple(component.edge_spacing) == old_spacing
    assert component.graph is not graph
    assert nx.is_frozen(component.graph)
    for actual, original in zip(component.grid_edges, old_edges, strict=True):
        np.testing.assert_array_equal(actual, original)
        assert not actual.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            actual[0] = 0
    np.testing.assert_array_equal(component.active_mask, old_mask)
    assert not component.active_mask.flags.writeable
    np.testing.assert_array_equal(component.to_linear(query), expected)
    with pytest.raises(nx.NetworkXError):
        component.graph.add_node("new")


@pytest.mark.parametrize(
    "restore", [copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_track_linearization_is_identity_hashed_and_read_only(
    graph_layout_with_gap, locator_parity, restore
):
    component = graph_layout_with_gap.linearization()
    equal_values = replace(component)
    assert component != equal_values
    assert len({component, equal_values}) == 2
    with pytest.raises(FrozenInstanceError):
        component.edge_spacing = 0.0
    restored = restore(component)
    assert restored != component
    assert nx.is_frozen(restored.graph)
    for array in (*restored.grid_edges, restored.active_mask):
        assert not array.flags.writeable
    _assert_previous_projection(
        restored, graph_layout_with_gap, "gapped_track/projection", locator_parity
    )


def test_legacy_track_layout_rebuilds_missing_component(
    graph_layout_with_gap, locator_parity
):
    """Layouts saved before the component existed can recreate their projection."""
    layout = graph_layout_with_gap
    original = layout.linearization()
    del layout._linearization
    restored = layout.linearization()
    assert restored is not original
    assert restored is layout.linearization()
    _assert_previous_projection(
        restored, layout, "gapped_track/projection", locator_parity
    )


@pytest.mark.parametrize(
    "restore",
    [lambda x: x, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))],
    ids=["original", "deepcopy", "pickle"],
)
@pytest.mark.parametrize("attributes", ["node", "edge", "graph"])
def test_stored_track_graph_attributes_cannot_change_projections(
    graph_layout_with_gap, locator_parity, restore, attributes
):
    """A published graph protects attributes as well as NetworkX topology."""
    component = restore(graph_layout_with_gap.linearization())
    points = locator_parity["gapped_track/projection/to_linear/points"]
    expected = component.to_linear(points)
    if attributes == "node":
        node = next(iter(component.graph.nodes))
        mapping, key, value = component.graph.nodes[node], "pos", (100.0, 0.0)
    elif attributes == "edge":
        edge = next(iter(component.graph.edges))
        mapping, key, value = component.graph.edges[edge], "distance", 999.0
    else:
        mapping, key, value = component.graph.graph, "description", "changed"
    with pytest.raises(TypeError):
        mapping[key] = value
    np.testing.assert_array_equal(component.to_linear(points), expected)
    with pytest.raises(nx.NetworkXError):
        component.graph.add_node("new")


def _native_oracle(env):
    if env._is_polar:
        from neurospatial.layout.engines._polar import _PolarGridLayout

        layout = _PolarGridLayout()
        layout.build(**env.layout_parameters)
        return layout
    from neurospatial import Environment

    return Environment._create_layout(env.layout_type, env.layout_parameters)
