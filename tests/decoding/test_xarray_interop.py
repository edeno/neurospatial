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

    def test_decoding_result_to_xarray_dims(self, small_2d_env):
        """Returns dims ('time','bin'); time coord matches times; values match."""
        pytest.importorskip("xarray")
        n_time = 7
        posterior = np.random.default_rng(0).random((n_time, small_2d_env.n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)
        times = np.linspace(0.0, 3.0, n_time)
        result = DecodingResult(posterior=posterior, env=small_2d_env, times=times)

        da = result.to_xarray()

        assert da.dims == ("time", "bin")
        np.testing.assert_array_equal(da.coords["time"].values, times)
        np.testing.assert_array_equal(
            da.coords["bin"].values, np.arange(small_2d_env.n_bins)
        )
        np.testing.assert_array_equal(da.values, posterior)

    def test_to_xarray_times_none(self, small_2d_env):
        """times=None -> time coord is integer index np.arange(n_time)."""
        pytest.importorskip("xarray")
        n_time = 6
        posterior = np.ones((n_time, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)
        assert result.times is None

        da = result.to_xarray()

        assert da.dims == ("time", "bin")
        np.testing.assert_array_equal(da.coords["time"].values, np.arange(n_time))
        np.testing.assert_array_equal(
            da.coords["bin"].values, np.arange(small_2d_env.n_bins)
        )
        np.testing.assert_array_equal(da.values, posterior)

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

        with pytest.raises(ValueError, match="must be sorted ascending"):
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
