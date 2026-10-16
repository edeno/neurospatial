"""Factory arguments: data positional, options keyword-only and never ignored."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture(scope="module")
def positions():
    """Samples of a 30 x 30 cm arena with an unvisited 10 x 10 cm center."""
    samples = np.random.default_rng(0).uniform(0, 30, (6000, 2))
    keep = ~np.all((samples > 10) & (samples < 20), axis=1)
    return samples[keep]


def _same_bins(first, second):
    np.testing.assert_array_equal(first.bin_centers, second.bin_centers)
    np.testing.assert_array_equal(first.edge_pairs, second.edge_pairs)


@pytest.mark.parametrize("factory", ["from_samples", "open_field"])
def test_bin_size_is_keyword_only(factory, positions):
    with pytest.raises(TypeError, match="positional"):
        getattr(Environment, factory)(positions, 3.0)


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
