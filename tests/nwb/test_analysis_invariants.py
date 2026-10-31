"""Computed fields keep their scientific meaning through real NWB files."""

import numpy as np
import pytest

pytest.importorskip("pynwb")

from neurospatial import compute_spatial_rates
from neurospatial.decoding import BayesianDecoder
from neurospatial.encoding import SpikeTrains
from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates


def test_disk_roundtrip_preserves_computed_decode(
    analysis_invariant_recording, analysis_invariant_method, empty_nwb, tmp_path
):
    """All estimators/layouts retain labelled predictions after closing the file."""
    from pynwb import NWBHDF5IO

    env, spikes, times, positions, coverage = analysis_invariant_recording
    rates = compute_spatial_rates(
        env,
        spikes,
        times,
        positions,
        spike_window=coverage,
        **analysis_invariant_method,
    )
    expected = BayesianDecoder.from_rates(rates, dt=0.1).predict(
        spikes, times, spike_window=coverage
    )
    write_spatial_rates(empty_nwb, rates)
    path = tmp_path / "analysis.nwb"
    with NWBHDF5IO(path, "w") as io:
        io.write(empty_nwb)
    with NWBHDF5IO(path, "r") as io:
        restored = read_spatial_rates(io.read())

    np.testing.assert_array_equal(restored.unit_ids, rates.unit_ids)
    np.testing.assert_array_equal(restored.firing_rates, rates.firing_rates)
    np.testing.assert_array_equal(restored.occupancy, rates.occupancy)
    np.testing.assert_array_equal(restored.spike_window, coverage)
    np.testing.assert_allclose(restored.env.bin_centers, env.bin_centers)
    np.testing.assert_allclose(restored.env.bin_sizes, env.bin_sizes)
    assert restored.env.units == env.units
    assert restored.env.frame == env.frame
    assert restored.method == rates.method
    assert restored.analysis_settings == rates.analysis_settings
    np.testing.assert_array_equal(restored.spatial_support, rates.spatial_support)
    reordered = SpikeTrains(
        [spikes.sel(unit_id=label) for label in spikes.unit_ids[[2, 0, 1]]],
        unit_ids=spikes.unit_ids[[2, 0, 1]],
    )
    actual = BayesianDecoder.from_rates(restored, dt=0.1).predict(
        reordered, times, spike_window=restored.spike_window
    )
    assert np.isfinite(actual.posterior).all()
    assert np.ptp(actual.posterior) > 0.01
    np.testing.assert_array_equal(actual.times, expected.times)
    np.testing.assert_allclose(actual.posterior, expected.posterior)
    np.testing.assert_allclose(actual.mean_position, expected.mean_position)
