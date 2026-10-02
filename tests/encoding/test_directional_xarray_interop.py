"""xarray / NetCDF export of directional population rates.

Runs in the dedicated xarray CI job (``.github/workflows/test_xarray.yml``),
which installs the optional ``xarray`` extra.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.encoding.directional import compute_directional_rates


@pytest.fixture(scope="module")
def unsmoothed_rates():
    """Directional rates for two units with no smoothing (``bandwidth=None``)."""
    rng = np.random.default_rng(0)
    times = np.linspace(0.0, 60.0, 3000)
    headings = np.unwrap(np.cumsum(rng.normal(0.0, 0.2, times.size)))
    headings = np.angle(np.exp(1j * headings))
    spike_trains = [np.sort(rng.uniform(0.0, 60.0, n)) for n in (120, 80)]
    return compute_directional_rates(spike_trains, times, headings, bandwidth=None)


def test_none_bandwidth_netcdf_roundtrip(unsmoothed_rates, tmp_path):
    """bandwidth=None is omitted from attrs, so the Dataset writes to NetCDF."""
    xr = pytest.importorskip("xarray")
    assert unsmoothed_rates.bandwidth is None

    ds = unsmoothed_rates.to_xarray()
    assert "bandwidth" not in ds.attrs

    path = tmp_path / "directional.nc"
    ds.to_netcdf(path, engine="scipy")
    loaded = xr.load_dataset(path, engine="scipy")

    np.testing.assert_array_equal(
        loaded["firing_rate"].values, np.asarray(unsmoothed_rates.firing_rates)
    )
    np.testing.assert_array_equal(loaded["occupancy"].values, ds["occupancy"].values)
    np.testing.assert_array_equal(loaded["unit_id"].values, ds["unit_id"].values)
    np.testing.assert_array_equal(
        loaded["bin_center_angle"].values, ds["bin_center_angle"].values
    )


def test_bandwidth_attr_kept_when_smoothed():
    """A smoothed result still records its bandwidth."""
    pytest.importorskip("xarray")
    rng = np.random.default_rng(1)
    times = np.linspace(0.0, 30.0, 1500)
    headings = rng.uniform(-np.pi, np.pi, times.size)
    trains = [np.sort(rng.uniform(0.0, 30.0, 60))]

    result = compute_directional_rates(trains, times, headings, bandwidth=0.3)

    assert result.to_xarray().attrs["bandwidth"] == 0.3
