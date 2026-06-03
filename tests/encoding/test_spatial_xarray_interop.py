"""Tests for SpatialRatesResult xarray export.

Covers ``SpatialRatesResult.to_xarray()`` dims/coords/values and the
actionable ImportError raised when xarray is unavailable.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding.spatial import SpatialRatesResult


@pytest.fixture(scope="module")
def rates_result() -> SpatialRatesResult:
    """A small 3-neuron SpatialRatesResult with a known rate matrix."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 50, (500, 2))
    env = Environment.from_samples(positions, bin_size=5.0)
    n_neurons = 3
    firing_rates = rng.uniform(0.0, 10.0, (n_neurons, env.n_bins))
    occupancy = rng.uniform(0.5, 2.0, env.n_bins)
    return SpatialRatesResult(
        firing_rates=firing_rates,
        occupancy=occupancy,
        env=env,
        smoothing_method="binned",
        bandwidth=5.0,
    )


def test_spatial_rates_to_xarray_dims(rates_result):
    """Dims ('neuron','bin'); coords integer indices; values match rates."""
    pytest.importorskip("xarray")

    da = rates_result.to_xarray()

    rates = np.asarray(rates_result.firing_rates)
    n_neurons, n_bins = rates.shape

    assert da.dims == ("neuron", "bin")
    np.testing.assert_array_equal(da.coords["neuron"].values, np.arange(n_neurons))
    np.testing.assert_array_equal(da.coords["bin"].values, np.arange(n_bins))
    np.testing.assert_array_equal(da.values, rates)


def test_spatial_rates_to_xarray_without_xarray_raises(rates_result, monkeypatch):
    """A failing xarray import raises a clear, actionable ImportError."""
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "xarray":
            raise ImportError("No module named 'xarray'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match="neurospatial\\[xarray\\]"):
        rates_result.to_xarray()
