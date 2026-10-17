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
    "from_nwb": ("nwbfile",),
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
        if factory not in ("combine", "from_polar_egocentric", "from_nwb")
    ],
)
def test_cartesian_factories_accept_the_same_labels(factory):
    parameters = inspect.signature(getattr(Environment, factory)).parameters
    for label in _LABELS:
        assert parameters[label].kind is inspect.Parameter.KEYWORD_ONLY


_REGULAR_GRID_ONLY = (
    "dilate",
    "fill_holes",
    "close_gaps",
    "add_boundary_bins",
    "connect_diagonal_neighbors",
)
_MASK_OPTIONS = ("bin_count_threshold", "dilate", "fill_holes", "close_gaps")


def _default(factory, option):
    return inspect.signature(getattr(Environment, factory)).parameters[option].default


# Values that ask for nothing: no clean-up, no threshold, the neighbors a
# layout has anyway. A layout that ignores an option accepts these and the
# factory's default.
_NO_OP = {
    "bin_count_threshold": 0,
    "dilate": False,
    "fill_holes": False,
    "close_gaps": False,
    "add_boundary_bins": False,
    "connect_diagonal_neighbors": True,
}


def _changed(factory, option):
    """A value that is neither the factory's default nor the no-op value."""
    return 5 if option == "bin_count_threshold" else not _NO_OP[option]


def _refusable(options):
    """(factory, option) pairs for which some value would be refused."""
    return [
        (factory, option)
        for factory in ("from_samples", "open_field")
        for option in options
        if _changed(factory, option) != _default(factory, option)
    ]


@pytest.mark.parametrize("factory", ["from_samples", "open_field"])
def test_sample_option_signatures_show_their_real_defaults(factory):
    defaults = {
        option: _default(factory, option)
        for option in ("bin_count_threshold", *_REGULAR_GRID_ONLY)
    }
    assert defaults == {
        "bin_count_threshold": 0,
        "dilate": False,
        "fill_holes": factory == "open_field",
        "close_gaps": False,
        "add_boundary_bins": False,
        "connect_diagonal_neighbors": True,
    }


@pytest.mark.parametrize(("factory", "option"), _refusable(_REGULAR_GRID_ONLY))
def test_regular_grid_option_with_hexagonal_layout_is_refused(
    factory, option, positions
):
    value = _changed(factory, option)
    with pytest.raises(ValueError, match=f"{option}={value!r}") as caught:
        getattr(Environment, factory)(
            positions, bin_size=3.0, layout="Hexagonal", **{option: value}
        )
    message = str(caught.value)
    assert f"{factory}(layout='Hexagonal') cannot apply" in message
    assert "Fix: remove" in message
    assert "layout='RegularGrid'" in message


@pytest.mark.parametrize("factory", ["from_samples", "open_field"])
def test_hexagonal_layout_accepts_the_regular_grid_defaults(factory, positions):
    """A parameter set can switch layouts without editing the grid options."""
    defaults = {option: _default(factory, option) for option in _REGULAR_GRID_ONLY}
    explicit = getattr(Environment, factory)(
        positions, bin_size=3.0, layout="Hexagonal", **defaults
    )
    omitted = getattr(Environment, factory)(positions, bin_size=3.0, layout="Hexagonal")
    _same_bins(explicit, omitted)
    assert explicit.layout_parameters == omitted.layout_parameters


@pytest.mark.parametrize(
    "setting", [{"layout": "Hexagonal"}, {"infer_active_bins": False}]
)
def test_open_field_accepts_fill_holes_false_where_nothing_is_filled(
    setting, positions
):
    """``fill_holes=False`` describes what these bins are, so it is no error."""
    unfilled = Environment.open_field(
        positions, bin_size=3.0, fill_holes=False, **setting
    )
    default = Environment.open_field(positions, bin_size=3.0, **setting)
    _same_bins(unfilled, default)
    assert (
        unfilled.n_bins
        == Environment.from_samples(positions, bin_size=3.0, **setting).n_bins
    )


@pytest.mark.parametrize("factory", ["from_samples", "open_field"])
def test_unused_mask_options_are_recorded_at_their_no_op_values(factory, positions):
    """Without inferred bins nothing is thresholded, dilated, filled or closed.

    ``layout_parameters`` describes the bins that were built, so it must not
    report ``open_field``'s ``fill_holes=True`` default for an all-active grid.
    """
    env = getattr(Environment, factory)(
        positions, bin_size=3.0, infer_active_bins=False
    )
    recorded = {option: env.layout_parameters[option] for option in _MASK_OPTIONS}
    assert recorded == {
        "bin_count_threshold": 0,
        "dilate": False,
        "fill_holes": False,
        "close_gaps": False,
    }
    unfilled = Environment.from_samples(
        positions, bin_size=3.0, infer_active_bins=False
    )
    _same_bins(env, unfilled)
    assert env.layout_parameters == unfilled.layout_parameters


def test_refusal_lists_every_inapplicable_option(positions):
    with pytest.raises(ValueError, match="cannot apply dilate=True, close_gaps=True"):
        Environment.from_samples(
            positions,
            bin_size=3.0,
            layout="Hexagonal",
            dilate=True,
            fill_holes=False,
            close_gaps=True,
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


def test_edge_spacing_on_a_one_edge_track_is_refused():
    """A gap sits between two edges, so a straight track has nowhere to put it."""
    with pytest.raises(ValueError, match="no gap to fill") as caught:
        Environment.linear_track(
            endpoints=[(0.0, 0.0), (100.0, 0.0)], bin_size=2.0, edge_spacing=5.0
        )
    assert "remove edge_spacing" in str(caught.value).split("Fix:")[1]
    straight = Environment.linear_track(
        endpoints=[(0.0, 0.0), (100.0, 0.0)], bin_size=2.0, edge_spacing=0.0
    )
    assert straight.layout_parameters["edge_spacing"] == 0.0


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


@pytest.mark.parametrize(("factory", "option"), _refusable(_MASK_OPTIONS))
def test_mask_option_without_inferred_bins_is_refused(factory, option, positions):
    value = _changed(factory, option)
    with pytest.raises(ValueError, match=f"{option}={value!r}") as caught:
        getattr(Environment, factory)(
            positions, bin_size=3.0, infer_active_bins=False, **{option: value}
        )
    message = str(caught.value)
    assert f"{factory}(infer_active_bins=False) cannot apply" in message
    assert "every bin in the extent is active" in message
    assert "infer_active_bins=True" in message.split("Fix:")[1]


@pytest.mark.parametrize("factory", ["from_samples", "open_field"])
def test_mask_option_defaults_without_inferred_bins_are_accepted(factory, positions):
    defaults = {option: _default(factory, option) for option in _MASK_OPTIONS}
    explicit = getattr(Environment, factory)(
        positions, bin_size=3.0, infer_active_bins=False, **defaults
    )
    omitted = getattr(Environment, factory)(
        positions, bin_size=3.0, infer_active_bins=False
    )
    _same_bins(explicit, omitted)


@pytest.mark.parametrize("layout", ["RegularGrid", "Hexagonal"])
def test_without_inferred_bins_every_bin_in_the_extent_is_active(layout, positions):
    every_bin = Environment.from_samples(
        positions, bin_size=3.0, layout=layout, infer_active_bins=False
    )
    inferred = Environment.from_samples(positions, bin_size=3.0, layout=layout)
    # The unvisited 10 x 10 cm center is kept only when bins are not inferred.
    center = np.array([[15.0, 15.0]])
    assert every_bin.contains(center)[0]
    assert not inferred.contains(center)[0]
    assert every_bin.layout_parameters["bin_count_threshold"] == 0


def test_options_that_apply_without_inferred_bins_are_accepted(positions):
    env = Environment.from_samples(
        positions,
        bin_size=3.0,
        infer_active_bins=False,
        add_boundary_bins=True,
        connect_diagonal_neighbors=False,
    )
    assert env.layout_parameters["add_boundary_bins"] is True
    assert env.layout_parameters["connect_diagonal_neighbors"] is False


@pytest.mark.parametrize("factory", ["from_samples", "open_field"])
@pytest.mark.parametrize("layout", ["TriangularMesh", "MaskedGrid", "Graph"])
def test_layout_without_sample_tiling_is_refused(factory, layout, positions):
    with pytest.raises(ValueError, match=f"cannot build layout='{layout}'") as caught:
        getattr(Environment, factory)(positions, bin_size=3.0, layout=layout)
    message = str(caught.value)
    assert "layout='RegularGrid' and layout='Hexagonal'" in message
    assert "Environment.from_layout" in message.split("Fix:")[1]


def test_from_layout_refuses_an_unbuilt_engine():
    from neurospatial import LayoutNotBuiltError

    with pytest.raises(LayoutNotBuiltError, match="not built yet"):
        Environment.from_layout(MaskedGridLayout())


# The option each factory took first by position on earlier versions; the
# others never took an option by position.
_FORMER_FIRST_POSITIONAL_OPTION = {
    "from_samples": "bin_size",
    "open_field": "bin_size",
    "from_graph": "edge_spacing",
    "maze": "kind",
    "from_polygon": "bin_size",
    "from_pixel_mask": "pixel_size",
    "from_polar_egocentric": "distance_range",
}


@pytest.mark.parametrize(("factory", "data"), _DATA_ARGUMENTS.items())
def test_positional_option_guard_names_the_keyword_or_lists_the_options(factory, data):
    method = getattr(Environment, factory)
    # The guard wraps the factory, yet the published signature is the
    # factory's own: data positional, every option keyword-only.
    wrapper = inspect.signature(method.__func__, follow_wrapped=False)
    assert [p.kind for p in wrapper.parameters.values()] == [
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    ]
    parameters = inspect.signature(method).parameters
    options = [name for name in parameters if name not in data]
    assert all(
        parameters[name].kind is inspect.Parameter.KEYWORD_ONLY for name in options
    )
    assert method.__doc__ == method.__func__.__wrapped__.__doc__

    # One value too many; the guard refuses before the factory reads anything.
    with pytest.raises(TypeError, match="positional argument") as caught:
        method(*([None] * len(data)), 7)
    fix = str(caught.value).split("Fix: ")[1]
    former = _FORMER_FIRST_POSITIONAL_OPTION.get(factory)
    if factory == "from_layout":
        assert "create_layout(" in fix
    elif former is not None:
        assert fix.startswith(f"pass {former}=7 by keyword")
    else:
        assert fix.startswith("options are keyword-only; pass them by name:")
        assert all(f"{name}=" in fix for name in options)
        assert "=7" not in fix


def _graph():
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_edge(0, 1, distance=10.0)
    return graph


@pytest.mark.parametrize(
    ("call", "advice"),
    [
        (
            lambda p: Environment.from_samples(p, 4.0),
            "pass bin_size=4.0 by keyword, as in "
            "Environment.from_samples(positions, bin_size=4.0)",
        ),
        (
            lambda p: Environment.open_field(p, 4.0, name="arena"),
            "Environment.open_field(positions, bin_size=4.0, name='arena')",
        ),
        # The track factory took edge_spacing before bin_size.
        (
            lambda p: Environment.from_graph(_graph(), [(0, 1)], 0.0, 2.0),
            "pass edge_spacing=0.0, bin_size=2.0 by keyword",
        ),
        (
            lambda p: Environment.maze("plus", bin_size=2.0),
            "Environment.maze(kind='plus', bin_size=2.0)",
        ),
        (
            lambda p: Environment.from_polar_egocentric((0, 50), (-1, 1), 5.0, 0.5),
            "distance_range=(0, 50), angle_range=(-1, 1), distance_bin_size=5.0, "
            "angle_bin_size=0.5",
        ),
        (
            lambda p: Environment.from_layout("TriangularMesh", {"point_spacing": 2}),
            "Environment.from_layout(create_layout('TriangularMesh'",
        ),
    ],
)
def test_option_passed_by_position_names_the_keyword(call, advice, positions):
    with pytest.raises(TypeError, match="positional argument") as caught:
        call(positions)
    message = str(caught.value)
    assert "Why: options are keyword-only" in message
    assert advice in message.split("Fix: ")[1]


def test_long_values_are_named_by_type_in_the_advice(positions):
    with pytest.raises(TypeError) as caught:
        Environment.from_samples(positions, np.full(2, 4.0) + np.arange(30)[:2])
    assert "bin_size=array([4., 5.])" in str(caught.value)
    with pytest.raises(TypeError) as caught:
        Environment.from_samples(positions, positions)
    assert "bin_size=<ndarray>" in str(caught.value)


def test_other_argument_errors_keep_their_python_message(positions):
    with pytest.raises(TypeError, match="Did you mean 'bin_size'"):
        Environment.from_samples(positions, bin_sizee=4.0)
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
        Environment.from_samples(positions)
    # A TypeError raised while the factory runs is not rewritten.
    with pytest.raises(TypeError, match="bin_size must be a numeric value"):
        Environment.from_samples(positions, bin_size="four")


@pytest.mark.parametrize(
    "build",
    [
        lambda diagonal: Environment.from_samples(
            np.linspace(0.0, 30.0, 200)[:, None],
            bin_size=3.0,
            connect_diagonal_neighbors=diagonal,
        ),
        lambda diagonal: Environment.from_grid_mask(
            np.ones(6, dtype=bool),
            (np.arange(7.0),),
            connect_diagonal_neighbors=diagonal,
        ),
    ],
    ids=["1-D samples", "1-D mask"],
)
def test_diagonal_choice_on_a_1d_grid_is_recorded_as_no_op(build):
    """A 1-D grid, and every grid derived from it, has no diagonal neighbors."""
    connected, unconnected = build(True), build(False)
    _same_bins(connected, unconnected)
    assert connected.layout_parameters["connect_diagonal_neighbors"] is False
    assert connected.layout_parameters == unconnected.layout_parameters
    _same_bins(connected.rebin(bin_size=1.0), unconnected.rebin(bin_size=1.0))


def test_diagonal_choice_is_kept_where_it_acts(positions):
    with_diagonals = Environment.from_samples(positions, bin_size=3.0)
    without = Environment.from_samples(
        positions, bin_size=3.0, connect_diagonal_neighbors=False
    )
    assert with_diagonals.layout_parameters["connect_diagonal_neighbors"] is True
    assert len(with_diagonals.edge_pairs) > len(without.edge_pairs)


@pytest.mark.parametrize(
    ("ranges", "rule"),
    [
        ((0.0, 30.0), "which is not 2 (min, max) pair(s)"),
        ([(0.0, np.inf), (0.0, 30.0)], "has a value that is not finite"),
        ([(30.0, 0.0), (0.0, 30.0)], "axis 0 does not have its minimum below"),
        ([(0.0, 30.0), (15.0, 15.0)], "axis 1 does not have its minimum below"),
    ],
)
def test_dimension_ranges_refusal_names_the_rule_it_breaks(positions, ranges, rule):
    with pytest.raises(ValueError, match=r"dimension_ranges") as caught:
        Environment.from_samples(positions, bin_size=3.0, dimension_ranges=ranges)
    assert rule in str(caught.value).splitlines()[0]
