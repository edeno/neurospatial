"""Posterior image pixels and overlays share the decoder clock."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import DecodingResult


@pytest.fixture
def plot_env():
    return Environment.from_samples(np.linspace(0, 100, 21)[:, None], bin_size=5.0)


@pytest.mark.parametrize(
    "times",
    [
        np.array([10.05, 10.15, 10.25, 10.35]),
        np.array([10.05, 10.15]),
        np.array([10.05]),
    ],
)
def test_continuous_image_columns_center_on_timestamps(plot_env, times):
    posterior = np.zeros((len(times), plot_env.n_bins))
    posterior[:, 4] = 1
    result = DecodingResult(posterior, plot_env, times)
    ax = result.plot(show_map=True)
    extent = ax.images[0].get_extent()
    edges = np.linspace(extent[0], extent[1], len(times) + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    np.testing.assert_allclose(centers, times, rtol=0, atol=1e-12)
    width = times[1] - times[0] if len(times) > 1 else 1.0
    assert extent[1] - extent[0] == pytest.approx(len(times) * width)
    assert extent[2:] == [-0.5, plot_env.n_bins - 0.5]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), times)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), result.map_estimate)
    np.testing.assert_array_equal(ax.images[0].get_array().T, posterior)
    plt.close(ax.figure)


def test_gapped_clock_keeps_centered_bin_indices_and_markers(plot_env):
    times = np.array([10.05, 10.15, 20.05, 20.15])
    result = DecodingResult(
        np.ones((4, plot_env.n_bins)) / plot_env.n_bins, plot_env, times
    )
    ax = result.plot(show_map=True)
    extent = ax.images[0].get_extent()
    assert extent[:2] == [-0.5, 3.5]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), np.arange(4))
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [1.5, 1.5])
    assert "recording gaps" in ax.get_xlabel()
    plt.close(ax.figure)


def test_explicit_extent_override_is_retained(plot_env):
    result = DecodingResult(
        np.ones((2, plot_env.n_bins)) / plot_env.n_bins, plot_env, np.array([1.0, 2.0])
    )
    extent = [0.0, 5.0, -1.0, 22.0]
    ax = result.plot(extent=extent)
    assert ax.images[0].get_extent() == extent
    plt.close(ax.figure)
