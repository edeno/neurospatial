"""Native one-dimensional rate maps plot in physical coordinates."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial import compute_spatial_rates
from neurospatial.simulation import linear_track_session


@pytest.fixture(scope="module")
def track_rates():
    sim = linear_track_session(duration=60, n_place_cells=3, seed=0)
    return compute_spatial_rates(
        sim.env, sim.spike_times, sim.times, sim.positions, unit_ids=sim.unit_ids
    )


@pytest.mark.parametrize("single", [False, True])
def test_native_track_rate_plot_uses_physical_bin_centers(track_rates, single):
    rates = track_rates
    ax = rates[0].plot() if single else rates.plot(idx=0)
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), rates.env.bin_centers[:, 0])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), rates.firing_rates[0])
    assert ax.get_xlabel() == "Position (cm)"
    assert ax.get_ylabel() == "Firing Rate (Hz)"
    plt.close(ax.figure)


def test_native_track_field_preserves_nan_breaks_and_labels(track_rates):
    field = np.arange(track_rates.env.n_bins, dtype=float)
    field[2:4] = np.nan
    ax = track_rates.env.plot_field(field, colorbar_label="Probability")
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), field)
    assert ax.get_ylabel() == "Probability"
    assert len(ax.figure.axes) == 1
    plt.close(ax.figure)
