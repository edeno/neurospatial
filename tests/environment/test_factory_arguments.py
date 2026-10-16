"""Factory arguments: data positional, options keyword-only and never ignored."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.layout.engines.masked_grid import MaskedGridLayout


@pytest.fixture(scope="module")
def positions():
    """Samples of a 30 x 30 cm arena with an unvisited 10 x 10 cm center."""
    samples = np.random.default_rng(0).uniform(0, 30, (6000, 2))
    keep = ~np.all((samples > 10) & (samples < 20), axis=1)
    return samples[keep]


def _same_bins(first, second):
    np.testing.assert_array_equal(first.bin_centers, second.bin_centers)
    np.testing.assert_array_equal(first.edge_pairs, second.edge_pairs)


# Each factory's data arguments, in order. Everything else is keyword-only.
_DATA_ARGUMENTS = {
    "from_samples": ("positions",),
    "open_field": ("positions",),
    "from_graph": ("graph", "edge_order"),
    "linear_track": (),
    "maze": (),
    "from_polygon": ("polygon",),
    "from_grid_mask": ("active_mask", "grid_edges"),
    "from_pixel_mask": ("image_mask",),
    "from_bins": ("binning",),
    "from_layout": ("layout",),
    "combine": ("components",),
    "from_polar_egocentric": (),
}
_LABELS = ("name", "units", "frame", "regions")


@pytest.mark.parametrize(("factory", "data"), _DATA_ARGUMENTS.items())
def test_factories_take_only_their_data_positionally(factory, data):
    parameters = inspect.signature(getattr(Environment, factory)).parameters
    positional = tuple(
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    )
    assert positional == data
    assert all(
        parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        for parameter in parameters.values()
    )


@pytest.mark.parametrize(
    "factory",
    [
        factory
        for factory in _DATA_ARGUMENTS
        if factory not in ("combine", "from_polar_egocentric")
    ],
)
def test_cartesian_factories_accept_the_same_labels(factory):
    parameters = inspect.signature(getattr(Environment, factory)).parameters
    for label in _LABELS:
        assert parameters[label].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("factory", ["from_samples", "open_field"])
@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("dilate", True),
        ("fill_holes", True),
        ("close_gaps", True),
        ("add_boundary_bins", True),
        # Passing a regular grid's default value is still an explicit choice
        # that a hexagonal layout would ignore.
        ("connect_diagonal_neighbors", True),
        ("dilate", False),
    ],
)
def test_regular_grid_option_with_hexagonal_layout_is_refused(
    factory, option, value, positions
):
    with pytest.raises(ValueError, match=option) as caught:
        getattr(Environment, factory)(
            positions, bin_size=3.0, layout="Hexagonal", **{option: value}
        )
    message = str(caught.value)
    assert f"{factory}(layout='Hexagonal')" in message
    assert "Fix: remove" in message
    assert "layout='RegularGrid'" in message


def test_refusal_lists_every_inapplicable_option(positions):
    with pytest.raises(ValueError, match="does not use dilate, close_gaps"):
        Environment.from_samples(
            positions, bin_size=3.0, layout="Hexagonal", dilate=True, close_gaps=True
        )


@pytest.mark.parametrize("factory", ["from_samples", "open_field"])
def test_hexagonal_layout_needs_no_regular_grid_option(factory, positions):
    hexagons = getattr(Environment, factory)(
        positions, bin_size=3.0, layout="Hexagonal"
    )
    assert hexagons.layout_type == "Hexagonal"
    assert "fill_holes" not in hexagons.layout_parameters


def test_omitted_regular_grid_options_use_their_documented_defaults(positions):
    omitted = Environment.from_samples(positions, bin_size=3.0)
    explicit = Environment.from_samples(
        positions,
        bin_size=3.0,
        dilate=False,
        fill_holes=False,
        close_gaps=False,
        add_boundary_bins=False,
        connect_diagonal_neighbors=True,
    )
    _same_bins(omitted, explicit)
    assert omitted.layout_parameters == explicit.layout_parameters


def test_open_field_fills_holes_unless_asked_not_to(positions):
    filled = Environment.open_field(positions, bin_size=3.0)
    unfilled = Environment.open_field(positions, bin_size=3.0, fill_holes=False)
    _same_bins(
        filled, Environment.from_samples(positions, bin_size=3.0, fill_holes=True)
    )
    _same_bins(unfilled, Environment.from_samples(positions, bin_size=3.0))
    # The unvisited center is a hole: only the filled arena covers it.
    center = np.array([[15.0, 15.0]])
    assert filled.contains(center)[0]
    assert not unfilled.contains(center)[0]
    assert filled.layout_parameters["fill_holes"] is True
    assert unfilled.layout_parameters["fill_holes"] is False


def test_dimension_ranges_sets_the_tiled_extent(positions):
    env = Environment.from_samples(
        positions,
        bin_size=3.0,
        dimension_ranges=[(0.0, 30.0), (0.0, 30.0)],
        infer_active_bins=False,
    )
    assert env.n_bins == 100
    np.testing.assert_allclose(env.dimension_ranges, [(0.0, 30.0), (0.0, 30.0)])


def _l_track_graph():
    import networkx as nx

    graph = nx.Graph()
    for node, pos in enumerate([(0.0, 0.0), (20.0, 0.0), (20.0, 20.0)]):
        graph.add_node(node, pos=pos)
    graph.add_edge(0, 1, distance=20.0)
    graph.add_edge(1, 2, distance=20.0)
    return graph


def test_track_factories_share_the_edge_spacing_default():
    graph = _l_track_graph()
    from_graph = Environment.from_graph(graph, [(0, 1), (1, 2)], bin_size=2.0)
    track = Environment.linear_track(
        node_positions=[(0.0, 0.0), (20.0, 0.0), (20.0, 20.0)], bin_size=2.0
    )
    _same_bins(from_graph, track)
    # Edges placed end to end: the linear coordinate runs 0..40 with no gap.
    np.testing.assert_allclose(
        from_graph.to_linear(np.array([[20.0, 1.0]])), [21.0], atol=1e-12
    )
    assert from_graph.layout_parameters["edge_spacing"] == 0.0


@pytest.mark.parametrize("layout", ["TriangularMesh", MaskedGridLayout])
def test_from_layout_refuses_a_layout_name_or_class(layout):
    with pytest.raises(TypeError, match="takes a built layout engine") as caught:
        Environment.from_layout(layout)
    assert "create_layout(" in str(caught.value)


def test_from_layout_owns_a_copy_of_the_engine_bins():
    engine = MaskedGridLayout()
    engine.build(
        active_mask=np.ones((2, 3), dtype=bool),
        grid_edges=(np.arange(3.0), np.arange(4.0)),
    )
    env = Environment.from_layout(engine, name="arena", units="cm")
    centers = env.bin_centers.copy()
    engine.bin_centers[:] = -1.0
    engine.active_mask[:] = False
    np.testing.assert_array_equal(env.bin_centers, centers)
    assert (env.name, env.units, env.n_bins) == ("arena", "cm", 6)
