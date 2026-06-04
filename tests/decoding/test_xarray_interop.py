"""Tests for DecodingResult xarray export and ground-truth error alignment.

Covers ``DecodingResult.to_xarray()`` (dims/coords, times=None integer-index
fallback, missing-xarray ImportError) and ``DecodingResult.error_against()``
(alignment matches a hand-rolled interpolation + decoding_error).
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from neurospatial.decoding import DecodingResult
from neurospatial.decoding.metrics import decoding_error


def _delta_posterior(env, bin_indices):
    """Build a one-hot posterior placing all mass on the given bins."""
    posterior = np.zeros((len(bin_indices), env.n_bins), dtype=np.float64)
    posterior[np.arange(len(bin_indices)), bin_indices] = 1.0
    return posterior


class TestToXarrayDims:
    """to_xarray() dims, coords, and values."""

    def test_decoding_result_to_xarray_is_dataset(self, small_2d_env):
        """Returns xr.Dataset; dims ('time','bin'); posterior matches."""
        xr = pytest.importorskip("xarray")
        n_time = 7
        posterior = np.random.default_rng(0).random((n_time, small_2d_env.n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)
        times = np.linspace(0.0, 3.0, n_time)
        result = DecodingResult(posterior=posterior, env=small_2d_env, times=times)

        ds = result.to_xarray()

        assert isinstance(ds, xr.Dataset)
        assert ds["posterior"].dims == ("time", "bin")
        # No unit_id axis on a decode result.
        assert "unit_id" not in ds.dims
        np.testing.assert_array_equal(ds.coords["time"].values, times)
        np.testing.assert_array_equal(
            ds.coords["bin"].values, np.arange(small_2d_env.n_bins)
        )
        np.testing.assert_array_equal(ds["posterior"].values, posterior)

    def test_decode_bin_center_coords(self, small_2d_env):
        """bin_center_x / bin_center_y are non-index coords on bin."""
        pytest.importorskip("xarray")
        n_time = 4
        posterior = np.ones((n_time, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        ds = result.to_xarray()

        assert "bin_center_x" in ds.coords
        assert "bin_center_y" in ds.coords
        assert ds.coords["bin_center_x"].dims == ("bin",)
        np.testing.assert_array_equal(
            ds.coords["bin_center_x"].values, small_2d_env.bin_centers[:, 0]
        )

    def test_decode_attrs(self, small_2d_env):
        """attrs carry units, env fingerprint, and software_version."""
        pytest.importorskip("xarray")
        n_time = 3
        posterior = np.ones((n_time, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        ds = result.to_xarray()

        assert "units" in ds.attrs
        assert "Environment" in ds.attrs["env"]
        assert ds.attrs["software_version"]

    def test_to_xarray_times_none(self, small_2d_env):
        """times=None -> time coord is integer index np.arange(n_time)."""
        pytest.importorskip("xarray")
        n_time = 6
        posterior = np.ones((n_time, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)
        assert result.times is None

        ds = result.to_xarray()

        assert ds["posterior"].dims == ("time", "bin")
        np.testing.assert_array_equal(ds.coords["time"].values, np.arange(n_time))
        np.testing.assert_array_equal(
            ds.coords["bin"].values, np.arange(small_2d_env.n_bins)
        )
        np.testing.assert_array_equal(ds["posterior"].values, posterior)

    def test_to_xarray_without_xarray_raises(self, small_2d_env, monkeypatch):
        """A failing xarray import raises a clear, actionable ImportError."""
        posterior = np.ones((3, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "xarray":
                raise ImportError("No module named 'xarray'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        with pytest.raises(ImportError, match="neurospatial\\[xarray\\]"):
            result.to_xarray()


class TestErrorAgainst:
    """error_against() aligns the decode grid to ground truth."""

    def test_error_against_matches_manual(self, small_2d_env):
        """error_against equals hand-aligned interp + decoding_error."""
        n_time = 5
        # Place MAP on distinct, known bins.
        bin_indices = [0, 2, 4, 6, 8]
        posterior = _delta_posterior(small_2d_env, bin_indices)
        decode_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = DecodingResult(
            posterior=posterior, env=small_2d_env, times=decode_times
        )

        # Ground truth on a coarser grid that must be interpolated.
        true_times = np.array([0.0, 0.5, 1.0])
        true_positions = np.array(
            [[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]], dtype=np.float64
        )

        errors = result.error_against(true_times, true_positions)

        # Hand-rolled alignment: interpolate each coordinate onto decode times.
        n_dims = small_2d_env.n_dims
        aligned = np.empty((n_time, n_dims), dtype=np.float64)
        for d in range(n_dims):
            aligned[:, d] = np.interp(decode_times, true_times, true_positions[:, d])
        expected = decoding_error(
            result.map_position, aligned, env=small_2d_env, metric="euclidean"
        )

        np.testing.assert_allclose(errors, expected)
        assert errors.shape == (n_time,)

    def test_error_against_all_nan_row_is_nan(self, small_2d_env):
        """An all-NaN posterior row yields NaN error, not a finite bin-0 error.

        Regression test: argmax over an all-NaN row returns bin 0, which would
        otherwise produce a finite (wrong) error. Undecodable rows must be NaN.
        Finite rows are unaffected.
        """
        n_bins = small_2d_env.n_bins
        posterior = np.empty((2, n_bins), dtype=np.float64)
        posterior[0] = np.nan  # undecodable row
        posterior[1] = 0.0
        posterior[1, 0] = 1.0  # finite row, MAP on bin 0
        decode_times = np.array([0.0, 1.0])
        result = DecodingResult(
            posterior=posterior, env=small_2d_env, times=decode_times
        )

        # Ground truth fixed at bin 0's center: the finite row has error 0.
        bin0 = small_2d_env.bin_centers[0]
        true_times = np.array([0.0, 1.0])
        true_positions = np.vstack([bin0, bin0])

        errors = result.error_against(true_times, true_positions)

        assert np.isnan(errors[0])  # undecodable -> NaN (not finite bin-0 error)
        assert np.isfinite(errors[1])
        np.testing.assert_allclose(errors[1], 0.0)

    def test_error_against_times_none_raises(self, small_2d_env):
        """error_against requires decode times; times=None -> ValueError."""
        posterior = np.ones((3, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        true_times = np.array([0.0, 1.0])
        true_positions = np.array([[0.0, 0.0], [10.0, 10.0]])

        with pytest.raises(ValueError, match="times=None"):
            result.error_against(true_times, true_positions)

    def test_error_against_shape_mismatch_raises(self, small_2d_env):
        """Mismatched true_positions shape raises ValueError."""
        posterior = np.ones((3, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(
            posterior=posterior,
            env=small_2d_env,
            times=np.array([0.0, 0.5, 1.0]),
        )

        true_times = np.array([0.0, 1.0])
        # Wrong number of dims (3 columns for a 2D env).
        true_positions = np.zeros((2, 3))

        with pytest.raises(ValueError, match="true_positions must have shape"):
            result.error_against(true_times, true_positions)

    def test_error_against_rejects_unsorted_true_times(self, small_2d_env):
        """Unsorted true_times raises ValueError (interp would silently lie)."""
        posterior = np.ones((3, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(
            posterior=posterior,
            env=small_2d_env,
            times=np.array([0.0, 0.5, 1.0]),
        )

        # Descending true_times: np.interp would return garbage silently.
        true_times = np.array([1.0, 0.0])
        true_positions = np.array([[10.0, 10.0], [0.0, 0.0]])

        with pytest.raises(ValueError, match="must be strictly increasing"):
            result.error_against(true_times, true_positions)

    def test_error_against_rejects_duplicate_true_times(self, small_2d_env):
        """Duplicate true_times raise ValueError (interp resolves ties arbitrarily)."""
        posterior = np.ones((3, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(
            posterior=posterior,
            env=small_2d_env,
            times=np.array([0.0, 0.5, 1.0]),
        )

        # Duplicated x-value at t=0.0: np.interp would resolve the tie
        # arbitrarily and silently mis-align the ground truth.
        true_times = np.array([0.0, 0.0, 1.0])
        true_positions = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])

        with pytest.raises(ValueError, match="must be strictly increasing"):
            result.error_against(true_times, true_positions)

    def test_error_against_strictly_increasing_true_times_ok(self, small_2d_env):
        """Strictly increasing true_times still decode without error."""
        posterior = np.ones((3, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(
            posterior=posterior,
            env=small_2d_env,
            times=np.array([0.0, 0.5, 1.0]),
        )

        true_times = np.array([0.0, 0.5, 1.0])
        true_positions = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])

        errors = result.error_against(true_times, true_positions)
        assert errors.shape == (3,)
        assert np.isfinite(errors).all()

    def test_error_against_rejects_nan_self_times(self, small_2d_env):
        """A NaN in self.times raises ValueError (not a silent NaN error)."""
        posterior = np.ones((3, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(
            posterior=posterior,
            env=small_2d_env,
            times=np.array([0.0, np.nan, 1.0]),
        )

        true_times = np.array([0.0, 1.0])
        true_positions = np.array([[0.0, 0.0], [10.0, 10.0]])

        with pytest.raises(ValueError, match="times"):
            result.error_against(true_times, true_positions)

    def test_error_against_rejects_nan_true_times(self, small_2d_env):
        """A NaN in true_times raises ValueError (not a silent NaN error)."""
        posterior = np.ones((3, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(
            posterior=posterior,
            env=small_2d_env,
            times=np.array([0.0, 0.5, 1.0]),
        )

        true_times = np.array([0.0, np.nan])
        true_positions = np.array([[0.0, 0.0], [10.0, 10.0]])

        with pytest.raises(ValueError, match="true_times"):
            result.error_against(true_times, true_positions)

    def test_error_against_rejects_inf_true_positions(self, small_2d_env):
        """A non-finite true_positions value raises ValueError."""
        posterior = np.ones((3, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(
            posterior=posterior,
            env=small_2d_env,
            times=np.array([0.0, 0.5, 1.0]),
        )

        true_times = np.array([0.0, 1.0])
        true_positions = np.array([[0.0, 0.0], [np.inf, 10.0]])

        with pytest.raises(ValueError, match="true_positions"):
            result.error_against(true_times, true_positions)

    def test_error_against_geodesic_smoke(self, linear_track_1d_env):
        """Geodesic metric returns a finite per-time error of the right shape."""
        env = linear_track_1d_env
        n_time = 4
        bin_indices = list(range(n_time))
        posterior = _delta_posterior(env, bin_indices)
        decode_times = np.linspace(0.0, 1.0, n_time)
        result = DecodingResult(posterior=posterior, env=env, times=decode_times)

        # Ground truth spanning the 1D track endpoints.
        true_times = np.array([0.0, 1.0])
        true_positions = np.array([[0.0], [10.0]], dtype=np.float64)

        errors = result.error_against(true_times, true_positions, metric="geodesic")

        assert errors.shape == (n_time,)
        assert np.all(np.isfinite(errors))
