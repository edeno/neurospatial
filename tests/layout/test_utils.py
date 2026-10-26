import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

from neurospatial.layout._plotting import _generic_graph_plot
from neurospatial.layout.helpers.utils import (
    _infer_dimension_ranges_from_samples,
    get_centers,
    get_n_bins,
)


def test_get_centers_basic():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    centers = get_centers(edges)
    np.testing.assert_allclose(centers, [0.5, 1.5, 2.5])


def test_get_centers_single_bin():
    edges = np.array([2.0, 5.0])
    centers = get_centers(edges)
    np.testing.assert_allclose(centers, [3.5])


def test_get_n_bins_basic():
    data = np.array([[0, 0], [2, 2], [4, 4]])
    bins = get_n_bins(data, bin_size=2)
    np.testing.assert_array_equal(bins, [2, 2])


def test_get_n_bins_with_dimension_range():
    data = np.array([[0, 0], [2, 2], [4, 4]])
    bins = get_n_bins(data, bin_size=[2, 2], dimension_range=[(0, 5), (0, 5)])
    np.testing.assert_array_equal(bins, [3, 3])


def test_get_n_bins_zero_extent():
    data = np.array([[1, 1], [1, 1]])
    bins = get_n_bins(data, bin_size=1)
    np.testing.assert_array_equal(bins, [1, 1])


def test_get_n_bins_invalid_bin_size():
    data = np.array([[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        get_n_bins(data, bin_size=0)


def test_get_n_bins_invalid_dimension_range():
    data = np.array([[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        get_n_bins(data, bin_size=1, dimension_range=[(0,)])


def test_get_n_bins_large_extent_no_int32_overflow():
    # extent 5000 / bin_size 1e-6 -> 5e9 bins, which overflows int32.
    # Pass dimension_range so no giant positions array is materialized.
    dummy_positions = np.zeros((1, 1), dtype=np.float64)
    bins = get_n_bins(dummy_positions, bin_size=1e-6, dimension_range=[(0.0, 5000.0)])
    assert bins.dtype == np.int64
    assert bins[0] == 5_000_000_000
    assert bins[0] > 0  # never a wrapped/negative int32


def test_get_n_bins_overflow_guard_raises():
    # extent / bin_size implying > 2**63 bins must raise, not silently wrap.
    dummy_positions = np.zeros((1, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="overflow"):
        get_n_bins(dummy_positions, bin_size=1e-300, dimension_range=[(0.0, 1.0)])


def test_infer_dimension_ranges_from_samples_basic():
    data = np.array([[0, 1], [2, 3], [4, 5]])
    result = _infer_dimension_ranges_from_samples(data)
    assert np.allclose(result[0], (0, 4))
    assert np.allclose(result[1], (1, 5))


def test_infer_dimension_ranges_from_samples_with_buffer():
    data = np.array([[1, 2], [3, 4]])
    result = _infer_dimension_ranges_from_samples(data, buffer_around_data=1)
    assert np.allclose(result[0], (0, 4))
    assert np.allclose(result[1], (1, 5))


def test_infer_dimension_ranges_from_samples_buffer_sequence():
    data = np.array([[1, 2], [3, 4]])
    result = _infer_dimension_ranges_from_samples(data, buffer_around_data=[0.5, 1.5])
    assert np.allclose(result[0], (0.5, 3.5))
    assert np.allclose(result[1], (0.5, 5.5))


def test_infer_dimension_ranges_from_samples_point_data():
    data = np.array([[2, 3], [2, 3]])
    result = _infer_dimension_ranges_from_samples(data)
    assert np.allclose(result[0], (1.5, 2.5))
    assert np.allclose(result[1], (2.5, 3.5))


def test_infer_dimension_ranges_from_samples_all_nan():
    data = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    with pytest.raises(ValueError):
        _infer_dimension_ranges_from_samples(data)


def test_infer_dimension_ranges_from_samples_wrong_buffer_length():
    data = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        _infer_dimension_ranges_from_samples(data, buffer_around_data=[1])


def test_infer_dimension_ranges_from_samples_not_2d():
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        _infer_dimension_ranges_from_samples(data)


def test_generic_graph_plot_2d(monkeypatch):
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(1, 1))
    G.add_edge(0, 1)
    # Patch plt.figure to avoid opening a window
    monkeypatch.setattr(plt, "figure", lambda *a, **k: plt.Figure())
    ax = _generic_graph_plot(G, "Test2D")
    assert hasattr(ax, "set_aspect")


def test_generic_graph_plot_empty_graph():
    G = nx.Graph()
    with pytest.raises(ValueError):
        _generic_graph_plot(G, "Empty")


@pytest.mark.parametrize("missing", [np.nan, np.inf])
def test_no_finite_position_says_nan_or_infinite(missing):
    """Infinite samples are missing like NaN, so the message names both."""
    from neurospatial import Environment

    with pytest.raises(ValueError, match="NaN or infinite") as caught:
        Environment.from_samples(np.full((10, 2), missing), bin_size=1.0)
    assert "all 10 rows" in str(caught.value)


def test_get_n_bins_counts_a_reversed_range_like_the_grid_does():
    """A (max, min) pair is ordered, as the grid builder orders it."""
    forward = get_n_bins(
        np.zeros((2, 2)), bin_size=2.0, dimension_range=[(0, 10), (0, 6)]
    )
    reversed_ = get_n_bins(
        np.zeros((2, 2)), bin_size=2.0, dimension_range=[(10, 0), (0, 6)]
    )
    np.testing.assert_array_equal(reversed_, forward)
    np.testing.assert_array_equal(forward, [5, 3])


def test_get_n_bins_ignores_rows_that_are_not_finite():
    positions = np.array([[0.0, 0.0], [np.inf, 1.0], [np.nan, 2.0], [10.0, 5.0]])
    np.testing.assert_array_equal(get_n_bins(positions, bin_size=2.0), [5, 3])


@pytest.mark.parametrize(
    "dimension_range", [[(0, np.inf), (0, 1)], [(np.nan, 1), (0, 1)]]
)
def test_get_n_bins_refuses_a_range_that_is_not_finite(dimension_range):
    with pytest.raises(ValueError, match=r"not finite[\s\S]*Fix:"):
        get_n_bins(np.zeros((2, 2)), bin_size=1.0, dimension_range=dimension_range)


def test_get_n_bins_refuses_positions_with_no_finite_row():
    with pytest.raises(ValueError, match=r"no row[\s\S]*Fix:"):
        get_n_bins(np.full((3, 2), np.nan), bin_size=1.0)
