"""Tests for memory-safe summary decoding (Task 2.1).

Covers:
- ``decode_position`` gains ``dtype`` and ``time_chunk`` keyword-only params
  WITHOUT changing its return contract (``.posterior`` stays a real ndarray).
- New ``decode_position_summary`` -> ``DecodingSummary`` streams over time and
  never materializes the full ``(n_time, n_bins)`` posterior.
- ``DecodingSummary`` terminal verbs (``to_dataframe``/``summary``/``plot``/
  ``to_xarray``).
- ``decode_session_summary`` golden-path wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_decode_inputs(env, *, n_time=120, n_neurons=8, seed=0):
    """Build deterministic decode inputs on a given environment."""
    rng = np.random.default_rng(seed)
    spike_counts = rng.poisson(1.5, (n_time, n_neurons)).astype(np.int64)
    encoding_models = rng.uniform(0.5, 12.0, (n_neurons, env.n_bins))
    return spike_counts, encoding_models


class TestDtypeParam:
    """decode_position(..., dtype=...) controls stored posterior dtype."""

    def test_float32_posterior_dtype(self, small_2d_env):
        from neurospatial.decoding import decode_position

        spike_counts, encoding_models = _make_decode_inputs(small_2d_env)
        result = decode_position(
            small_2d_env, spike_counts, encoding_models, dt=0.025, dtype=np.float32
        )
        assert result.posterior.dtype == np.float32
        assert isinstance(result.posterior, np.ndarray)

    def test_float32_parity_with_float64(self, small_2d_env):
        from neurospatial.decoding import decode_position

        spike_counts, encoding_models = _make_decode_inputs(small_2d_env)
        res64 = decode_position(small_2d_env, spike_counts, encoding_models, dt=0.025)
        res32 = decode_position(
            small_2d_env, spike_counts, encoding_models, dt=0.025, dtype=np.float32
        )
        np.testing.assert_allclose(
            res32.posterior, res64.posterior, rtol=1e-5, atol=1e-6
        )

    def test_float32_methods_still_run(self, small_2d_env):
        from neurospatial.decoding import decode_position

        spike_counts, encoding_models = _make_decode_inputs(small_2d_env)
        times = np.arange(spike_counts.shape[0]) * 0.025
        result = decode_position(
            small_2d_env,
            spike_counts,
            encoding_models,
            dt=0.025,
            dtype=np.float32,
            times=times,
        )
        # All standard methods/properties run unchanged on a float32 posterior.
        assert result.map_estimate.shape == (spike_counts.shape[0],)
        assert result.map_position.shape[0] == spike_counts.shape[0]
        assert result.mean_position.shape[0] == spike_counts.shape[0]
        assert result.posterior_entropy.shape == (spike_counts.shape[0],)
        df = result.to_dataframe()
        assert len(df) == spike_counts.shape[0]
        assert isinstance(result.summary(), dict)

    def test_invalid_dtype_raises(self, small_2d_env):
        from neurospatial.decoding import decode_position

        spike_counts, encoding_models = _make_decode_inputs(small_2d_env)
        with pytest.raises(ValueError, match="dtype"):
            decode_position(
                small_2d_env,
                spike_counts,
                encoding_models,
                dt=0.025,
                dtype=np.float16,  # type: ignore[arg-type]
            )


class TestTimeChunkParam:
    """decode_position(..., time_chunk=k) is byte-for-byte chunked decode."""

    @pytest.mark.parametrize("k", [1, 7, 40, 120, 1000])
    def test_time_chunk_parity(self, small_2d_env, k):
        from neurospatial.decoding import decode_position

        spike_counts, encoding_models = _make_decode_inputs(small_2d_env, n_time=120)
        ref = decode_position(small_2d_env, spike_counts, encoding_models, dt=0.025)
        chunked = decode_position(
            small_2d_env,
            spike_counts,
            encoding_models,
            dt=0.025,
            time_chunk=k,
        )
        np.testing.assert_allclose(chunked.posterior, ref.posterior, rtol=0, atol=0)

    def test_time_chunk_with_prior(self, small_2d_env):
        from neurospatial.decoding import decode_position

        spike_counts, encoding_models = _make_decode_inputs(small_2d_env)
        prior = np.linspace(1.0, 2.0, small_2d_env.n_bins)
        ref = decode_position(
            small_2d_env, spike_counts, encoding_models, dt=0.025, prior=prior
        )
        chunked = decode_position(
            small_2d_env,
            spike_counts,
            encoding_models,
            dt=0.025,
            prior=prior,
            time_chunk=13,
        )
        np.testing.assert_allclose(chunked.posterior, ref.posterior, rtol=0, atol=0)


class TestNormalizeDegenerateChunking:
    """normalize_to_posterior chunked path handles degenerate rows identically."""

    def test_all_neg_inf_rows_chunked(self):
        from neurospatial.decoding.posterior import normalize_to_posterior

        ll = np.array(
            [
                [-1.0, -2.0, -0.5],
                [-np.inf, -np.inf, -np.inf],  # all -inf -> uniform
                [-0.2, -0.3, -0.1],
            ]
        )
        ref = normalize_to_posterior(ll)
        chunked = normalize_to_posterior(ll, time_chunk=1)
        np.testing.assert_allclose(chunked, ref, rtol=0, atol=0)
        # All-(-inf) row becomes uniform.
        np.testing.assert_allclose(chunked[1], np.full(3, 1 / 3))

    def test_nan_rows_chunked_nan_mode(self):
        from neurospatial.decoding.posterior import normalize_to_posterior

        ll = np.array(
            [
                [-1.0, -2.0, -0.5],
                [np.nan, -0.3, -0.1],  # NaN -> degenerate
                [-0.2, -0.3, -0.1],
            ]
        )
        ref = normalize_to_posterior(ll, handle_degenerate="nan")
        chunked = normalize_to_posterior(ll, handle_degenerate="nan", time_chunk=2)
        # Row 1 is NaN in both.
        assert np.isnan(chunked[1]).all()
        np.testing.assert_array_equal(np.isnan(chunked), np.isnan(ref))
        np.testing.assert_allclose(chunked[[0, 2]], ref[[0, 2]], rtol=0, atol=0)


class TestDecodePositionSummaryParity:
    """decode_position_summary reductions match the full posterior reductions."""

    def test_map_position_exact_parity(self, small_2d_env):
        from neurospatial.decoding import decode_position, decode_position_summary

        spike_counts, encoding_models = _make_decode_inputs(small_2d_env)
        full = decode_position(small_2d_env, spike_counts, encoding_models, dt=0.025)
        summ = decode_position_summary(
            small_2d_env, spike_counts, encoding_models, dt=0.025
        )
        np.testing.assert_array_equal(summ.map_bin, full.map_estimate)
        np.testing.assert_array_equal(summ.map_position, full.map_position)

    def test_mean_and_entropy_parity(self, small_2d_env):
        from neurospatial.decoding import decode_position, decode_position_summary

        spike_counts, encoding_models = _make_decode_inputs(small_2d_env)
        full = decode_position(small_2d_env, spike_counts, encoding_models, dt=0.025)
        summ = decode_position_summary(
            small_2d_env, spike_counts, encoding_models, dt=0.025
        )
        np.testing.assert_allclose(
            summ.mean_position, full.mean_position, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            summ.posterior_entropy, full.posterior_entropy, rtol=1e-9, atol=1e-9
        )
        expected_peak = full.posterior.max(axis=1)
        np.testing.assert_allclose(summ.peak_prob, expected_peak, rtol=1e-9, atol=1e-9)

    def test_times_carried(self, small_2d_env):
        from neurospatial.decoding import decode_position_summary

        spike_counts, encoding_models = _make_decode_inputs(small_2d_env)
        times = np.arange(spike_counts.shape[0]) * 0.025
        summ = decode_position_summary(
            small_2d_env, spike_counts, encoding_models, dt=0.025, times=times
        )
        np.testing.assert_array_equal(summ.times, times)


class TestDecodingSummaryTerminalVerbs:
    """DecodingSummary implements to_dataframe / summary / plot / to_xarray."""

    def _summary(self, env, *, with_times=True):
        from neurospatial.decoding import decode_position_summary

        spike_counts, encoding_models = _make_decode_inputs(env)
        times = np.arange(spike_counts.shape[0]) * 0.025 if with_times else None
        return decode_position_summary(
            env, spike_counts, encoding_models, dt=0.025, times=times
        ), spike_counts.shape[0]

    def test_to_dataframe_columns(self, small_2d_env):
        summ, n_time = self._summary(small_2d_env)
        df = summ.to_dataframe()
        assert len(df) == n_time
        for col in [
            "time",
            "map_bin",
            "map_x",
            "map_y",
            "mean_x",
            "mean_y",
            "posterior_entropy",
            "peak_prob",
        ]:
            assert col in df.columns, col

    def test_to_dataframe_no_time_column_when_none(self, small_2d_env):
        summ, _ = self._summary(small_2d_env, with_times=False)
        df = summ.to_dataframe()
        assert "time" not in df.columns

    def test_to_dataframe_1d_uses_x(self, small_1d_env):
        summ, _ = self._summary(small_1d_env)
        df = summ.to_dataframe()
        assert "map_x" in df.columns
        assert "map_y" not in df.columns

    def test_summary_keys(self, small_2d_env):
        summ, n_time = self._summary(small_2d_env)
        s = summ.summary()
        assert s["n_time_bins"] == n_time
        assert s["n_bins"] == small_2d_env.n_bins
        assert "mean_entropy" in s
        assert "max_entropy" in s
        # all scalar
        for v in s.values():
            assert np.isscalar(v) or isinstance(v, (int, float))

    def test_plot_returns_axes(self, small_2d_env):
        import matplotlib

        matplotlib.use("Agg")
        summ, _ = self._summary(small_2d_env)
        ax = summ.plot()
        from matplotlib.axes import Axes

        assert isinstance(ax, Axes)

    def test_to_xarray_dims(self, small_2d_env):
        pytest.importorskip("xarray")
        summ, n_time = self._summary(small_2d_env)
        ds = summ.to_xarray()
        # Only a time dim; NO bin dim.
        assert "bin" not in ds.dims
        assert set(ds.dims) == {"time"}
        assert ds.sizes["time"] == n_time
        for var in [
            "map_bin",
            "map_x",
            "map_y",
            "mean_x",
            "mean_y",
            "posterior_entropy",
            "peak_prob",
        ]:
            assert var in ds.data_vars, var
        assert "software_version" in ds.attrs
        assert "env" in ds.attrs

    def test_to_xarray_time_fallback(self, small_2d_env):
        pytest.importorskip("xarray")
        summ, n_time = self._summary(small_2d_env, with_times=False)
        ds = summ.to_xarray()
        np.testing.assert_array_equal(ds["time"].values, np.arange(n_time))


class TestDecodeSessionSummary:
    """decode_session_summary mirrors decode_session glue."""

    def _session_inputs(self, env, seed=0):
        rng = np.random.default_rng(seed)
        n_frames = 400
        times = np.linspace(0.0, 10.0, n_frames)
        # Random walk positions inside env bounds.
        lo = env.bin_centers.min(axis=0)
        hi = env.bin_centers.max(axis=0)
        positions = rng.uniform(lo, hi, (n_frames, env.n_dims))
        # 6 neurons, a handful of spikes each within the window.
        spike_times = [
            np.sort(rng.uniform(0.0, 10.0, rng.integers(20, 60))) for _ in range(6)
        ]
        return spike_times, times, positions

    def test_session_summary_matches_session_map(self, small_2d_env):
        from neurospatial.decoding import decode_session, decode_session_summary

        spike_times, times, positions = self._session_inputs(small_2d_env)
        full = decode_session(small_2d_env, spike_times, times, positions, dt=0.1)
        summ = decode_session_summary(
            small_2d_env, spike_times, times, positions, dt=0.1
        )
        np.testing.assert_array_equal(summ.map_position, full.map_position)
        np.testing.assert_array_equal(summ.map_bin, full.map_estimate)

    def test_session_unchanged(self, small_2d_env):
        """decode_session must still return a DecodingResult with a real posterior."""
        from neurospatial.decoding import DecodingResult, decode_session

        spike_times, times, positions = self._session_inputs(small_2d_env)
        full = decode_session(small_2d_env, spike_times, times, positions, dt=0.1)
        assert isinstance(full, DecodingResult)
        assert isinstance(full.posterior, np.ndarray)
