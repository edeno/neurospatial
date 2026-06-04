"""Tests for population result xarray export (labeled ``xr.Dataset``).

Covers ``SpatialRatesResult.to_xarray()`` (and the directional / view /
egocentric population results) producing a labeled :class:`xarray.Dataset`
with dims ``("unit_id", "bin")``, real ``unit_id`` index coords,
``bin_center_*`` non-index coords, an ``occupancy`` data var, provenance
attrs, duplicate-id validation, and the actionable ImportError raised when
xarray is unavailable.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding.directional import DirectionalRatesResult
from neurospatial.encoding.egocentric import EgocentricRatesResult
from neurospatial.encoding.spatial import SpatialRatesResult
from neurospatial.encoding.view import ViewRatesResult


@pytest.fixture(scope="module")
def rates_result() -> SpatialRatesResult:
    """A small 3-neuron SpatialRatesResult with a known rate matrix."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 50, (500, 2))
    env = Environment.from_samples(positions, bin_size=5.0)
    env.units = "cm"
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


def test_spatial_rates_to_xarray_is_dataset(rates_result):
    """Returns an xr.Dataset with dims unit_id/bin and matching firing_rate."""
    xr = pytest.importorskip("xarray")

    ds = rates_result.to_xarray()

    rates = np.asarray(rates_result.firing_rates)
    n_neurons, n_bins = rates.shape

    assert isinstance(ds, xr.Dataset)
    assert "unit_id" in ds.dims
    assert "bin" in ds.dims
    assert ds["firing_rate"].dims == ("unit_id", "bin")
    assert ds.sizes["unit_id"] == n_neurons
    assert ds.sizes["bin"] == n_bins
    np.testing.assert_array_equal(ds["firing_rate"].values, rates)


def test_spatial_rates_unit_id_index_coord(rates_result):
    """unit_id is the index coord = real unit_ids; .sel selects that unit."""
    pytest.importorskip("xarray")

    ds = rates_result.to_xarray()
    rates = np.asarray(rates_result.firing_rates)

    np.testing.assert_array_equal(ds.coords["unit_id"].values, rates_result.unit_ids)
    # Label-based selection picks the right row.
    uid = rates_result.unit_ids[1]
    sel = ds.sel(unit_id=uid)
    np.testing.assert_array_equal(sel["firing_rate"].values, rates[1])


def test_spatial_rates_bin_center_coords(rates_result):
    """bin_center_x / bin_center_y are non-index coords on bin."""
    pytest.importorskip("xarray")

    ds = rates_result.to_xarray()
    bin_centers = rates_result.env.bin_centers

    assert "bin_center_x" in ds.coords
    assert "bin_center_y" in ds.coords
    # Non-index: bin_center_x is NOT a dimension.
    assert "bin_center_x" not in ds.dims
    assert ds.coords["bin_center_x"].dims == ("bin",)
    np.testing.assert_array_equal(ds.coords["bin_center_x"].values, bin_centers[:, 0])
    np.testing.assert_array_equal(ds.coords["bin_center_y"].values, bin_centers[:, 1])


def test_spatial_rates_occupancy_data_var(rates_result):
    """occupancy is a data var on ('bin',)."""
    pytest.importorskip("xarray")

    ds = rates_result.to_xarray()
    assert "occupancy" in ds.data_vars
    assert ds["occupancy"].dims == ("bin",)
    np.testing.assert_array_equal(
        ds["occupancy"].values, np.asarray(rates_result.occupancy)
    )


def test_spatial_rates_attrs(rates_result):
    """attrs carry units, bandwidth, env fingerprint and software_version."""
    pytest.importorskip("xarray")

    ds = rates_result.to_xarray()
    assert ds.attrs["units"] == "cm"
    assert ds.attrs["bandwidth"] == 5.0
    assert "Environment" in ds.attrs["env"]
    assert isinstance(ds.attrs["software_version"], str)
    assert ds.attrs["software_version"]


def test_spatial_rates_duplicate_unit_ids_raise(rates_result):
    """Duplicate unit_ids -> ValueError naming the offending labels."""
    pytest.importorskip("xarray")

    rates = np.asarray(rates_result.firing_rates)
    dup = SpatialRatesResult(
        firing_rates=rates,
        occupancy=rates_result.occupancy,
        env=rates_result.env,
        smoothing_method="binned",
        bandwidth=5.0,
        unit_ids=np.array([7, 7, 9]),
    )
    with pytest.raises(ValueError, match="duplicated"):
        dup.to_xarray()


def test_spatial_rates_string_unit_ids_select_by_label(rates_result):
    """String unit_ids select by label."""
    pytest.importorskip("xarray")

    rates = np.asarray(rates_result.firing_rates)
    res = SpatialRatesResult(
        firing_rates=rates,
        occupancy=rates_result.occupancy,
        env=rates_result.env,
        smoothing_method="binned",
        bandwidth=5.0,
        unit_ids=np.array(["a", "b", "c"]),
    )
    ds = res.to_xarray()
    sel = ds.sel(unit_id="b")
    np.testing.assert_array_equal(sel["firing_rate"].values, rates[1])


def test_directional_rates_to_xarray():
    """DirectionalRatesResult.to_xarray -> Dataset with bin_center_angle."""
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(1)
    n_neurons, n_bins = 3, 60
    bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    result = DirectionalRatesResult(
        firing_rates=rng.random((n_neurons, n_bins)) * 10,
        occupancy=np.ones(n_bins) * 0.5,
        bin_centers=bin_centers,
        bin_size=np.pi / 30,
        bandwidth=None,
    )
    ds = result.to_xarray()
    assert isinstance(ds, xr.Dataset)
    assert ds["firing_rate"].dims == ("unit_id", "bin")
    assert "bin_center_angle" in ds.coords
    np.testing.assert_array_equal(ds.coords["bin_center_angle"].values, bin_centers)
    assert ds.attrs["units"] == "radians"
    assert "occupancy" in ds.data_vars


def test_view_rates_to_xarray():
    """ViewRatesResult.to_xarray -> Dataset with Cartesian bin coords."""
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(2)
    positions = rng.uniform(0, 50, (300, 2))
    env = Environment.from_samples(positions, bin_size=5.0)
    env.units = "cm"
    n_neurons = 3
    result = ViewRatesResult(
        firing_rates=rng.uniform(0, 10, (n_neurons, env.n_bins)),
        occupancy=rng.uniform(0.5, 2.0, env.n_bins),
        env=env,
        gaze_model="fixed_distance",
        view_distance=20.0,
        smoothing_method="binned",
        bandwidth=5.0,
    )
    ds = result.to_xarray()
    assert isinstance(ds, xr.Dataset)
    assert ds["firing_rate"].dims == ("unit_id", "bin")
    assert "bin_center_x" in ds.coords
    assert "bin_center_y" in ds.coords
    assert ds.attrs["units"] == "cm"


def test_egocentric_rates_to_xarray_polar_coords():
    """Egocentric polar result uses bin_center_distance / bin_center_angle."""
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(3)
    env = Environment.from_polar_egocentric(
        distance_range=(0.0, 50.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=10.0,
        angle_bin_size=np.pi / 6,
    )
    n_neurons = 3
    result = EgocentricRatesResult(
        firing_rates=rng.uniform(0, 10, (n_neurons, env.n_bins)),
        occupancy=rng.uniform(0.5, 2.0, env.n_bins),
        env=env,
        distance_range=(0.0, 50.0),
        n_distance_bins=5,
        n_direction_bins=12,
    )
    ds = result.to_xarray()
    assert isinstance(ds, xr.Dataset)
    assert ds["firing_rate"].dims == ("unit_id", "bin")
    # Polar coords, NOT x/y.
    assert "bin_center_distance" in ds.coords
    assert "bin_center_angle" in ds.coords
    assert "bin_center_x" not in ds.coords
    np.testing.assert_array_equal(
        ds.coords["bin_center_distance"].values, env.bin_centers[:, 0]
    )
    np.testing.assert_array_equal(
        ds.coords["bin_center_angle"].values, env.bin_centers[:, 1]
    )


def test_spatial_rates_occupancy_length_mismatch_raises(rates_result):
    """A present occupancy whose length != n_bins raises ValueError.

    Guards against silently dropping the occupancy data var and emitting a
    structurally-incomplete Dataset.
    """
    pytest.importorskip("xarray")

    rates = np.asarray(rates_result.firing_rates)
    n_bins = rates.shape[1]
    bad = SpatialRatesResult(
        firing_rates=rates,
        # occupancy present but one element too short
        occupancy=np.ones(n_bins - 1),
        env=rates_result.env,
        smoothing_method="binned",
        bandwidth=5.0,
    )
    with pytest.raises(ValueError, match="occupancy length does not match"):
        bad.to_xarray()


def test_directional_bin_centers_length_mismatch_raises():
    """A present bin_centers whose length != n_bins raises ValueError."""
    pytest.importorskip("xarray")

    rng = np.random.default_rng(11)
    n_neurons, n_bins = 3, 60
    result = DirectionalRatesResult(
        firing_rates=rng.random((n_neurons, n_bins)) * 10,
        occupancy=np.ones(n_bins) * 0.5,
        # bin_centers present but one element too short
        bin_centers=np.linspace(0, 2 * np.pi, n_bins - 1, endpoint=False),
        bin_size=np.pi / 30,
        bandwidth=None,
    )
    with pytest.raises(ValueError, match="bin_centers length does not match"):
        result.to_xarray()


def test_build_population_dataset_env_bin_center_mismatch_raises():
    """An env whose bin_centers length != n_bins raises ValueError."""
    pytest.importorskip("xarray")
    from neurospatial._results import build_population_dataset

    rng = np.random.default_rng(12)
    positions = rng.uniform(0, 50, (300, 2))
    env = Environment.from_samples(positions, bin_size=5.0)
    # Claim more bins than the env actually has -> bin-center length mismatch.
    n_bins = env.n_bins + 1
    rates = rng.uniform(0, 10, (3, n_bins))
    with pytest.raises(ValueError, match="bin_centers length does not match"):
        build_population_dataset(rates, np.arange(3), env=env)


def test_spatial_rates_units_none_omits_none_string():
    """When env.units is None the units attr is absent (never the str 'None')."""
    pytest.importorskip("xarray")

    rng = np.random.default_rng(13)
    positions = rng.uniform(0, 50, (300, 2))
    env = Environment.from_samples(positions, bin_size=5.0)
    env.units = None
    result = SpatialRatesResult(
        firing_rates=rng.uniform(0, 10, (3, env.n_bins)),
        occupancy=rng.uniform(0.5, 2.0, env.n_bins),
        env=env,
        smoothing_method="binned",
        bandwidth=5.0,
    )
    ds = result.to_xarray()
    assert ds.attrs.get("units", "") != "None"
    assert "units" not in ds.attrs


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
