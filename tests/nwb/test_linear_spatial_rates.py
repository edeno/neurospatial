"""An actual NWB file preserves linearly allocated rates and their support."""

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment, compute_spatial_rates
from neurospatial.decoding import BayesianDecoder
from neurospatial.encoding import SpikeTrains

pynwb = pytest.importorskip("pynwb")


def test_linear_rates_file_preserves_support_and_decoder(empty_nwb, tmp_path):
    from neurospatial.io.nwb import read_place_field, write_spatial_rates

    env = Environment.from_grid_mask(
        np.ones((2, 1), bool),
        grid_edges=(np.arange(3.0), np.array([0.0, 1.0])),
    ).clip(box(0.25, 0, 1.75, 1))
    spikes = SpikeTrains([np.array([1.0]), np.array([2.0])], unit_ids=[9, 3])
    rates = compute_spatial_rates(
        env,
        spikes,
        np.array([0.0, 3.0]),
        np.array([[-0.5, 0.5], [2.5, 0.5]]),
        method="binned",
        bandwidth=0,
        time_allocation="linear",
        max_gap=None,
    )
    np.testing.assert_allclose(rates.occupancy, [0.75, 0.75])
    prediction_times = np.arange(0.0, 3.25, 0.25)
    expected = BayesianDecoder.from_rates(rates, dt=0.25).predict(
        spikes, prediction_times
    )
    write_spatial_rates(empty_nwb, rates)
    path = tmp_path / "linear-rates.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(empty_nwb)
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        restored = read_place_field(io.read())
    np.testing.assert_array_equal(restored.unit_ids, [9, 3])
    np.testing.assert_array_equal(restored.firing_rates, rates.firing_rates)
    np.testing.assert_array_equal(restored.occupancy, rates.occupancy)
    assert restored.analysis_settings["time_allocation"] == "linear"
    np.testing.assert_allclose(
        restored.analysis_settings["observed_intervals"], [[0.75, 2.25]]
    )
    np.testing.assert_array_equal(restored.env.clip_mapping.parent_bin_indices, [0, 1])
    actual = BayesianDecoder.from_rates(restored, dt=0.25).predict(
        spikes, prediction_times
    )
    np.testing.assert_array_equal(actual.times, expected.times)
    np.testing.assert_array_equal(actual.posterior, expected.posterior)
