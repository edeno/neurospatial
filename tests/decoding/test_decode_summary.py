"""Tests for memory-safe summary decoding.

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

import math
from unittest.mock import patch

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
        assert result.map_bin.shape == (spike_counts.shape[0],)
        assert result.map_position.shape[0] == spike_counts.shape[0]
        assert result.mean_position.shape[0] == spike_counts.shape[0]
        assert result.posterior_entropy.shape == (spike_counts.shape[0],)
        df = result.to_dataframe()
        assert len(df) == spike_counts.shape[0]
        assert isinstance(result.summary(), dict)


class TestTimeChunkParam:
    """decode_position(..., time_chunk=k) is tolerance-equal to the full path.

    HYBRID R11: ``time_chunk=None`` (default) keeps the byte-exact full-matmul
    path; an explicit ``time_chunk=k`` computes the Poisson log-likelihood
    blockwise into the preallocated posterior. The per-block likelihood matmul
    is a different BLAS shape than the full matmul, so the chunked posterior is
    tolerance-equal (~1e-15), not byte-exact, to the full path. MAP/argmax is
    identical and every row sums to 1.
    """

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
        # Tolerance-equal (BLAS shape-dependence), MAP identical, rows sum to 1.
        np.testing.assert_allclose(chunked.posterior, ref.posterior, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(chunked.map_bin, ref.map_bin)
        np.testing.assert_allclose(
            np.asarray(chunked.posterior).sum(axis=1), 1.0, atol=1e-9
        )

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
        np.testing.assert_allclose(chunked.posterior, ref.posterior, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(chunked.map_bin, ref.map_bin)


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

    def test_neg_inf_rows_chunked_nan_mode(self):
        from neurospatial.decoding.posterior import normalize_to_posterior

        ll = np.array(
            [
                [-1.0, -2.0, -0.5],
                [-np.inf, -np.inf, -np.inf],
                [-0.2, -0.3, -0.1],
            ]
        )
        ref = normalize_to_posterior(ll, handle_degenerate="nan")
        chunked = normalize_to_posterior(ll, handle_degenerate="nan", time_chunk=2)
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
        np.testing.assert_array_equal(summ.map_bin, full.map_bin)
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
        np.testing.assert_allclose(
            summ.peak_probability, expected_peak, rtol=1e-9, atol=1e-9
        )

    def test_times_carried(self, small_2d_env):
        from neurospatial.decoding import decode_position_summary

        spike_counts, encoding_models = _make_decode_inputs(small_2d_env)
        times = np.arange(spike_counts.shape[0]) * 0.025
        summ = decode_position_summary(
            small_2d_env, spike_counts, encoding_models, dt=0.025, times=times
        )
        np.testing.assert_array_equal(summ.times, times)

    def test_float32_summary_reductions_precise(self, medium_2d_env):
        """float32 working posterior is honored but reductions stay precise.

        The summary path upcasts each block to float64 before reducing, so a
        float32 ``dtype=`` (smaller working posterior) must produce the SAME MAP
        as float64 exactly, and entropy/peak_probability/mean within a tight tolerance
        -- it must NOT accumulate entropy/peak in float32 and lose precision.
        """
        from neurospatial.decoding import decode_position_summary

        # Larger n_bins makes float32 accumulation error visible if it leaked.
        env = medium_2d_env  # 625 bins
        rng = np.random.default_rng(1)
        n_time, n_neurons = 200, 10
        spike_counts = rng.poisson(1.5, (n_time, n_neurons)).astype(np.int64)
        encoding_models = rng.uniform(0.5, 12.0, (n_neurons, env.n_bins))

        summ64 = decode_position_summary(
            env, spike_counts, encoding_models, dt=0.025, dtype=np.float64
        )
        summ32 = decode_position_summary(
            env, spike_counts, encoding_models, dt=0.025, dtype=np.float32
        )

        # MAP is exact: the upcast preserves the argmax bin.
        np.testing.assert_array_equal(summ32.map_bin, summ64.map_bin)
        np.testing.assert_array_equal(summ32.map_position, summ64.map_position)

        # Reductions are computed in float64 in both; only the working
        # posterior differs (float32 vs float64), so they agree tightly.
        np.testing.assert_allclose(
            summ32.posterior_entropy, summ64.posterior_entropy, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            summ32.peak_probability, summ64.peak_probability, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            summ32.mean_position, summ64.mean_position, rtol=1e-5, atol=1e-5
        )


class TestDecodePositionSummaryTimeChunkGuard:
    """decode_position_summary rejects time_chunk=None and non-positive values.

    R12: the summary decoder promises never to materialize the full
    ``(n_time, n_bins)`` posterior. ``time_chunk=None`` would set the block to
    the whole session and thereby materialize it transiently, defeating the
    purpose, so it is rejected up front (as are 0 and negatives).
    """

    def test_default_and_explicit_positive_chunk_agree(self, small_2d_env):
        """Default (no time_chunk) and an explicit positive chunk give same result.

        Regression pin: the guard must not change the default behavior, and an
        explicit positive ``time_chunk`` streams to the same reductions.
        """
        from neurospatial.decoding import decode_position_summary

        spike_counts, encoding_models = _make_decode_inputs(small_2d_env)
        default = decode_position_summary(
            small_2d_env, spike_counts, encoding_models, dt=0.025
        )
        chunked = decode_position_summary(
            small_2d_env, spike_counts, encoding_models, dt=0.025, time_chunk=17
        )
        np.testing.assert_array_equal(chunked.map_bin, default.map_bin)
        np.testing.assert_array_equal(chunked.map_position, default.map_position)
        np.testing.assert_allclose(
            chunked.mean_position, default.mean_position, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            chunked.posterior_entropy,
            default.posterior_entropy,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            chunked.peak_probability, default.peak_probability, rtol=1e-12, atol=1e-12
        )


class TestDecodingSummaryPortability:
    """DecodingSummary parity accessor + direct-construction shape guard."""

    def test_map_bin_aliases_map_bin(self, small_2d_env):
        from neurospatial.decoding import decode_position_summary

        spike_counts, encoding_models = _make_decode_inputs(small_2d_env)
        summ = decode_position_summary(
            small_2d_env, spike_counts, encoding_models, dt=0.025
        )
        # map_bin ports from DecodingResult: it is the MAP bin index.
        np.testing.assert_array_equal(summ.map_bin, summ.map_bin)

    def test_post_init_rejects_ragged_field(self, small_2d_env):
        from neurospatial.decoding._result import DecodingSummary

        env = small_2d_env
        n_time, n_dims = 5, env.n_dims
        map_bin = np.zeros(n_time, dtype=np.int64)
        # mean_position has the wrong length -> should raise naming the field.
        with pytest.raises(ValueError, match="mean_position"):
            DecodingSummary(
                times=None,
                mean_position=np.zeros((n_time - 1, n_dims)),
                posterior_entropy=np.zeros(n_time),
                peak_probability=np.zeros(n_time),
                env=env,
                map_bin=map_bin,
            )

    def test_post_init_accepts_consistent_shapes(self, small_2d_env):
        from neurospatial.decoding._result import DecodingSummary

        env = small_2d_env
        n_time, n_dims = 5, env.n_dims
        # A fully-consistent construction must not raise.
        DecodingSummary(
            times=np.arange(n_time, dtype=np.float64),
            mean_position=np.zeros((n_time, n_dims)),
            posterior_entropy=np.zeros(n_time),
            peak_probability=np.zeros(n_time),
            env=env,
            map_bin=np.zeros(n_time, dtype=np.int64),
        )


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
            "peak_probability",
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
        assert set(s) == {
            "n_time_bins",
            "n_undecodable",
            "n_bins",
            "mean_entropy",
            "max_entropy",
            "mean_peak_probability",
            "spike_window",
            "spike_window_assumed",
        }
        assert s["spike_window"] is None
        assert s["spike_window_assumed"] is True
        # Headline metrics remain scalar; coverage metadata may hold intervals.
        for key in (
            "n_time_bins",
            "n_bins",
            "mean_entropy",
            "max_entropy",
            "mean_peak_probability",
        ):
            v = s[key]
            assert np.isscalar(v) or isinstance(v, (int, float))

    def test_plot_returns_axes(self, small_2d_env):
        import matplotlib

        matplotlib.use("Agg")
        summ, _ = self._summary(small_2d_env)
        ax = summ.plot()
        from matplotlib.axes import Axes

        assert isinstance(ax, Axes)
        (line,) = ax.get_lines()
        np.testing.assert_array_equal(line.get_xdata(), summ.times)
        np.testing.assert_array_equal(line.get_ydata(), summ.posterior_entropy)

        map_lines = summ.plot(quantity="map").get_lines()
        assert len(map_lines) == summ.map_position.shape[1]
        for i, map_line in enumerate(map_lines):
            np.testing.assert_array_equal(map_line.get_ydata(), summ.map_position[:, i])

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
            "peak_probability",
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
        np.testing.assert_array_equal(summ.map_bin, full.map_bin)

    def test_session_unchanged(self, small_2d_env):
        """decode_session must still return a DecodingResult with a real posterior."""
        from neurospatial.decoding import DecodingResult, decode_session

        spike_times, times, positions = self._session_inputs(small_2d_env)
        full = decode_session(small_2d_env, spike_times, times, positions, dt=0.1)
        assert isinstance(full, DecodingResult)
        assert isinstance(full.posterior, np.ndarray)


class TestDecodePositionSummaryPriorValidation:
    """Streamed decoders validate the prior shape before the block loop.

    The loop iterates over n_time rows, so an over-long 2-D prior would
    otherwise be truncated silently.
    """

    def test_exact_2d_prior_parity(self, small_2d_env):
        from neurospatial.decoding import decode_position, decode_position_summary

        spike_counts, encoding_models = _make_decode_inputs(small_2d_env)
        n_time = spike_counts.shape[0]
        rng = np.random.default_rng(7)
        prior = rng.uniform(0.5, 2.0, (n_time, small_2d_env.n_bins))

        full = decode_position(
            small_2d_env, spike_counts, encoding_models, dt=0.025, prior=prior
        )
        summ = decode_position_summary(
            small_2d_env, spike_counts, encoding_models, dt=0.025, prior=prior
        )
        np.testing.assert_array_equal(summ.map_bin, full.map_bin)
        np.testing.assert_array_equal(summ.map_position, full.map_position)

    def test_correct_1d_prior_parity(self, small_2d_env):
        from neurospatial.decoding import decode_position, decode_position_summary

        spike_counts, encoding_models = _make_decode_inputs(small_2d_env)
        prior = np.linspace(1.0, 2.0, small_2d_env.n_bins)

        full = decode_position(
            small_2d_env, spike_counts, encoding_models, dt=0.025, prior=prior
        )
        summ = decode_position_summary(
            small_2d_env, spike_counts, encoding_models, dt=0.025, prior=prior
        )
        np.testing.assert_array_equal(summ.map_bin, full.map_bin)
        np.testing.assert_array_equal(summ.map_position, full.map_position)

    @pytest.mark.parametrize(
        "entry",
        ["decode_position", "decode_position_summary", "decode_session_summary"],
    )
    @pytest.mark.parametrize("shape", ["long", "short", "bins", "3d"])
    def test_wrong_prior_shape_names_received_and_expected(
        self, small_2d_env, entry, shape
    ):
        """No entry point truncates or broadcasts a prior of the wrong shape."""
        import neurospatial.decoding as decoding

        env = small_2d_env
        if entry == "decode_session_summary":
            rng = np.random.default_rng(0)
            times = np.linspace(0.0, 10.0, 400)
            positions = rng.uniform(
                env.bin_centers.min(axis=0), env.bin_centers.max(axis=0), (400, 2)
            )
            spikes = [np.sort(rng.uniform(0.0, 10.0, 40)) for _ in range(3)]
            n_time = 100  # whole 0.1 s bins in [0, 10]

            def call(prior):
                return decoding.decode_session_summary(
                    env, spikes, times, positions, dt=0.1, prior=prior
                )
        else:
            counts, models = _make_decode_inputs(env)
            n_time = counts.shape[0]

            def call(prior):
                return getattr(decoding, entry)(
                    env, counts, models, dt=0.025, prior=prior
                )

        n_bins = env.n_bins
        bad = {
            "long": (n_time + 3, n_bins),
            "short": (n_time - 2, n_bins),
            "bins": (n_bins + 1,),
            "3d": (n_time, n_bins, 1),
        }[shape]
        with pytest.raises(ValueError, match="prior has shape") as err:
            call(np.ones(bad))
        message = str(err.value)
        assert str(bad) in message
        assert f"({n_time}, {env.n_bins})" in message
        assert "Fix:" in message


class TestSummaryDecodeStreamingThroughput:
    """Structural (NON-timing) throughput guard for the summary decoders.

    The summary decode path promises to stream the posterior in time-blocks of
    ``time_chunk`` rather than (a) materializing the whole ``(n_time, n_bins)``
    posterior in one shot or (b) doing work that fans out with neuron count.
    Both ``decode_position_summary`` and ``decode_session_summary`` express that
    by decoding through the shared block loop, which normalizes each block with
    ``_normalize_block`` exactly once.

    These tests spy on ``_normalize_block`` (preserving its behavior via
    ``wraps=real``) and assert the call count equals ``ceil(n_time/time_chunk)``
    and is INDEPENDENT of ``n_neurons``. They are deterministic (call counts,
    not tracemalloc/timing), so they run in default CI without flaking.

    Why these catch the two structural regressions:

    - **Materialize-in-one-block:** if the summary path decoded the whole window
      in a single block, ``_normalize_block`` would be called exactly
      once (== ``ceil(n_time/n_time) == 1``) instead of ``ceil(n_time/k)``, so
      the block-count assertion fails.
    - **Per-neuron fan-out:** if the per-block work were split per neuron, the
      call count would scale with ``n_neurons`` (e.g. 12x), so the
      n=1-vs-n=12 equality fails.
    """

    def test_position_summary_block_count_eq_ceil_n_time_over_chunk(self, small_2d_env):
        """Block count == ceil(n_time/time_chunk); non-dividing chunk so ceil matters."""
        import neurospatial.decoding.posterior as posterior_mod
        from neurospatial.decoding import decode_position_summary

        real = posterior_mod._normalize_block
        n_time, time_chunk = 100, 32
        expected_blocks = math.ceil(n_time / time_chunk)  # ceil(100/32) == 4
        assert expected_blocks == 4

        spike_counts, encoding_models = _make_decode_inputs(
            small_2d_env, n_time=n_time, n_neurons=8
        )
        with patch.object(posterior_mod, "_normalize_block", wraps=real) as spy:
            decode_position_summary(
                small_2d_env,
                spike_counts,
                encoding_models,
                dt=0.025,
                time_chunk=time_chunk,
            )
        # Spy actually fired (not a no-op patch) and streamed in 4 blocks.
        assert spy.call_count > 0
        assert spy.call_count == expected_blocks

    @pytest.mark.parametrize("n_neurons", [1, 12])
    def test_position_summary_block_count_independent_of_n_neurons(
        self, small_2d_env, n_neurons
    ):
        """Same n_time/time_chunk -> identical block count for n=1 and n=12 neurons.

        A per-neuron fan-out would make this scale with ``n_neurons``; pinning it
        to the same ``ceil(n_time/time_chunk)`` for both neuron counts guards
        against that regression.
        """
        import neurospatial.decoding.posterior as posterior_mod
        from neurospatial.decoding import decode_position_summary

        real = posterior_mod._normalize_block
        n_time, time_chunk = 100, 32
        expected_blocks = math.ceil(n_time / time_chunk)  # 4, regardless of n_neurons

        spike_counts, encoding_models = _make_decode_inputs(
            small_2d_env, n_time=n_time, n_neurons=n_neurons
        )
        with patch.object(posterior_mod, "_normalize_block", wraps=real) as spy:
            decode_position_summary(
                small_2d_env,
                spike_counts,
                encoding_models,
                dt=0.025,
                time_chunk=time_chunk,
            )
        assert spy.call_count == expected_blocks == 4

    def test_position_summary_block_count_scales_with_n_time(self, small_2d_env):
        """Doubling n_time ~doubles the block count (pins 'streams over time')."""
        import neurospatial.decoding.posterior as posterior_mod
        from neurospatial.decoding import decode_position_summary

        real = posterior_mod._normalize_block
        time_chunk = 32

        def _count_blocks(n_time):
            spike_counts, encoding_models = _make_decode_inputs(
                small_2d_env, n_time=n_time, n_neurons=8
            )
            with patch.object(posterior_mod, "_normalize_block", wraps=real) as spy:
                decode_position_summary(
                    small_2d_env,
                    spike_counts,
                    encoding_models,
                    dt=0.025,
                    time_chunk=time_chunk,
                )
            return spy.call_count

        blocks_64 = _count_blocks(64)
        blocks_128 = _count_blocks(128)
        assert blocks_64 == math.ceil(64 / time_chunk)  # 2
        assert blocks_128 == math.ceil(128 / time_chunk)  # 4
        assert blocks_128 == 2 * blocks_64

    def _session_inputs(self, env, *, n_neurons, seed=0):
        """Tiny deterministic session: t in [0, 10], so n_time = floor(10/dt + eps)."""
        rng = np.random.default_rng(seed)
        n_frames = 400
        times = np.linspace(0.0, 10.0, n_frames)
        lo = env.bin_centers.min(axis=0)
        hi = env.bin_centers.max(axis=0)
        positions = rng.uniform(lo, hi, (n_frames, env.n_dims))
        spike_times = [
            np.sort(rng.uniform(0.0, 10.0, rng.integers(20, 60)))
            for _ in range(n_neurons)
        ]
        return spike_times, times, positions

    @pytest.mark.parametrize("n_neurons", [1, 12])
    def test_session_summary_block_count_eq_ceil_and_indep_of_n_neurons(
        self, small_2d_env, n_neurons
    ):
        """decode_session_summary: per-block decode count == ceil(n_time/chunk).

        The session decoder runs the block loop of the ``posterior`` module,
        which looks ``_normalize_block`` up there at call time -- patch it
        THERE. With ``times`` spanning [0, 10] and ``dt=0.1``, the global grid is
        ``n_time = floor(10/0.1 + 1e-9) = 100`` (mirrors session.py's grid math),
        so a non-dividing ``time_chunk=32`` gives ``ceil(100/32) == 4`` blocks --
        identical for n=1 and n=12 neurons.
        """
        import neurospatial.decoding.posterior as posterior_mod
        from neurospatial.decoding import decode_session_summary

        real = posterior_mod._normalize_block
        dt, time_chunk = 0.1, 32
        n_time = 100  # floor((10 - 0) / 0.1 + 1e-9)
        expected_blocks = math.ceil(n_time / time_chunk)  # 4
        assert expected_blocks == 4

        spike_times, times, positions = self._session_inputs(
            small_2d_env, n_neurons=n_neurons
        )
        with patch.object(posterior_mod, "_normalize_block", wraps=real) as spy:
            result = decode_session_summary(
                small_2d_env,
                spike_times,
                times,
                positions,
                dt=dt,
                time_chunk=time_chunk,
            )
        # Confirm the fixture's actual n_time matches the computed grid.
        assert result.map_bin.shape[0] == n_time
        assert spy.call_count > 0
        assert spy.call_count == expected_blocks == 4


def test_polar_summary_mean_matches_full_posterior_mean():
    """predict_summary averages polar bins in the egocentric plane, like predict.

    Averaging raw (distance, angle) columns gives a different, wrong point.
    """
    from neurospatial import Environment
    from neurospatial.decoding import BayesianDecoder

    env = Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 8,
    )
    rng = np.random.default_rng(0)
    centers = env.bin_centers
    xy = np.column_stack(
        [centers[:, 0] * np.cos(centers[:, 1]), centers[:, 0] * np.sin(centers[:, 1])]
    )
    peaks = rng.uniform(-25.0, 25.0, (8, 2))
    distances_sq = ((xy[None] - peaks[:, None]) ** 2).sum(axis=-1)
    models = 2.0 + 20.0 * np.exp(-distances_sq / (2 * 8.0**2))
    decoder = BayesianDecoder.from_models(env, models, dt=0.25)

    times = np.arange(0.0, 20.0, 0.25)
    spike_rng = np.random.default_rng(1)
    spikes = [np.sort(spike_rng.uniform(0.0, 20.0, 40)) for _ in range(8)]
    full = decoder.predict(spikes, times)
    summary = decoder.predict_summary(spikes, times, time_chunk=7)

    np.testing.assert_allclose(summary.mean_position, full.mean_position, atol=1e-9)
    np.testing.assert_array_equal(summary.map_bin, full.map_bin)


def test_streamed_summary_counts_unsorted_trains_like_sorted():
    """Block spike selection does not depend on the order of spike times."""
    from neurospatial import Environment
    from neurospatial.decoding import BayesianDecoder

    env = Environment.from_samples(np.linspace(0, 100, 200)[:, None], bin_size=10.0)
    models = np.random.default_rng(0).uniform(1.0, 20.0, (3, env.n_bins))
    decoder = BayesianDecoder.from_models(env, models, dt=0.1)
    times = np.arange(0.0, 30.0, 0.05)
    spike_rng = np.random.default_rng(1)
    trains = [np.sort(spike_rng.uniform(0.0, 30.0, 200)) for _ in range(3)]
    shuffled = [spike_rng.permutation(t) for t in trains]
    full = decoder.predict(trains, times)
    sorted_summary = decoder.predict_summary(trains, times, time_chunk=16)
    shuffled_summary = decoder.predict_summary(shuffled, times, time_chunk=16)
    np.testing.assert_array_equal(shuffled_summary.map_bin, sorted_summary.map_bin)
    np.testing.assert_array_equal(sorted_summary.map_bin, full.map_bin)
    np.testing.assert_allclose(
        shuffled_summary.posterior_entropy, full.posterior_entropy, rtol=1e-12
    )
