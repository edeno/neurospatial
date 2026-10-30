"""Tests for the immutable ``BayesianDecoder`` fit/predict wrapper.

The headline acceptance test is **byte-exact parity** with the functional
``decode_session`` core: a decoder that fits its encoding models and then
predicts must reproduce ``decode_session``'s posterior exactly. The wrapper is a
thin, frozen convenience layer -- it must not re-implement decoding.

Tests
-----
1. PARITY (headline): fit -> predict posterior byte-equals decode_session.
2. predict_summary MAP == predict MAP (streaming summary matches dense).
3. score: median/mean reductions match error_against; unknown metric raises.
4. train/test epoch split: fit(epochs=...) restricts the encoding models.
5. Unfitted predict/predict_summary/score raise a clear RuntimeError.
6. Immutability: fit returns a new object; original stays unfitted; frozen.
7. SpikeTrains (a spike group) input yields the plain-list posterior.
8. Held-out accuracy in an open field and, geodesically, on a W-maze.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial import Environment
from neurospatial.decoding import (
    BayesianDecoder,
    DecodingResult,
    DecodingSummary,
    decode_session,
)
from neurospatial.decoding.session import _decode_with_models
from neurospatial.encoding import SpikeTrains

# ---------------------------------------------------------------------------
# Helpers -- small, fast simulation (dt large for speed)
# ---------------------------------------------------------------------------


def _make_sim(
    *,
    n_neurons: int = 8,
    duration: float = 40.0,
    seed: int = 0,
) -> tuple[Environment, list[np.ndarray], np.ndarray, np.ndarray]:
    """Build a tiny open-field simulation (2-D, 50 cm, 5 cm bins)."""
    from neurospatial.simulation import (
        PlaceCellModel,
        generate_poisson_spikes,
        simulate_trajectory_ou,
    )

    rng = np.random.default_rng(seed)

    sample_positions = np.random.default_rng(seed).uniform(0.0, 50.0, (400, 2))
    env = Environment.from_samples(sample_positions, bin_size=5.0)
    env = env.with_metadata(units="cm")

    positions, times = simulate_trajectory_ou(
        env, duration=duration, seed=seed, speed_units="cm"
    )

    centers = rng.uniform(5.0, 45.0, (n_neurons, 2))
    spike_times: list[np.ndarray] = []
    for i in range(n_neurons):
        cell = PlaceCellModel(
            env,
            center=centers[i],
            width=12.0,
            max_rate=30.0,
            seed=int(rng.integers(0, 2**31)),
        )
        rates = cell.firing_rate(times, positions)
        spikes = generate_poisson_spikes(rates, times, seed=int(rng.integers(0, 2**31)))
        spike_times.append(spikes)

    return env, spike_times, times, positions


@pytest.fixture(scope="module")
def sim() -> tuple[Environment, list[np.ndarray], np.ndarray, np.ndarray]:
    """Module-scoped simulation reused across tests (cheap fit/predict at dt=0.5)."""
    return _make_sim()


def _small_env() -> Environment:
    """Tiny 2-D env for fast construction / validation tests."""
    positions = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    return Environment.from_samples(positions, bin_size=5.0)


def _make_linear_sim(
    seed: int = 0,
) -> tuple[Environment, list[np.ndarray], np.ndarray, np.ndarray]:
    """1-D linearized-track sim (back-and-forth), for graph/geodesic decoding."""
    from neurospatial.simulation import PlaceCellModel, generate_poisson_spikes

    env = Environment.linear_track(
        endpoints=[(0.0, 0.0), (100.0, 0.0)], bin_size=5.0, track_width=5.0
    )
    env = env.with_metadata(units="cm")

    times = np.linspace(0.0, 20.0, 1000)
    x = 50.0 + 50.0 * np.sin(2 * np.pi * times / 20.0)
    positions = np.column_stack([x, np.zeros_like(x)])

    rng = np.random.default_rng(seed)
    centers = np.column_stack([np.linspace(5.0, 95.0, 6), np.zeros(6)])
    spikes: list[np.ndarray] = []
    for i in range(6):
        cell = PlaceCellModel(env, center=centers[i], width=15.0, max_rate=30.0, seed=i)
        rates = cell.firing_rate(times, positions)
        spikes.append(
            generate_poisson_spikes(rates, times, seed=int(rng.integers(0, 2**31)))
        )
    return env, spikes, times, positions


# ---------------------------------------------------------------------------
# 1. PARITY (headline)
# ---------------------------------------------------------------------------


class TestParity:
    def test_posterior_byte_equal_to_decode_session(self, sim) -> None:
        """fit -> predict posterior is byte-identical to decode_session."""
        env, spikes, times, positions = sim

        dec = (
            BayesianDecoder(env, dt=0.5, bandwidth=5.0)
            .fit(spikes, times, positions)
            .predict(spikes, times)
        )
        ref = decode_session(env, spikes, times, positions, dt=0.5, bandwidth=5.0)

        assert isinstance(dec, DecodingResult)
        assert_array_equal(dec.posterior, ref.posterior)

    def test_map_bins_byte_equal(self, sim) -> None:
        """MAP position and MAP bin index match decode_session exactly."""
        env, spikes, times, positions = sim

        dec = (
            BayesianDecoder(env, dt=0.5, bandwidth=5.0)
            .fit(spikes, times, positions)
            .predict(spikes, times)
        )
        ref = decode_session(env, spikes, times, positions, dt=0.5, bandwidth=5.0)

        assert_array_equal(dec.map_position, ref.map_position)
        assert_array_equal(dec.map_bin, ref.map_bin)

    def test_times_grid_matches(self, sim) -> None:
        """Decode time-bin centers match decode_session."""
        env, spikes, times, positions = sim
        fit = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
        dec = fit.predict(spikes, times)
        ref = decode_session(env, spikes, times, positions, dt=0.5)
        assert_array_equal(dec.times, ref.times)


# ---------------------------------------------------------------------------
# 2. predict_summary MAP == predict MAP
# ---------------------------------------------------------------------------


def test_predict_summary_map_matches_predict(sim) -> None:
    """Streaming summary MAP equals the dense-posterior MAP."""
    env, spikes, times, positions = sim
    fit = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)

    summary = fit.predict_summary(spikes, times)
    dense = fit.predict(spikes, times)

    assert isinstance(summary, DecodingSummary)
    assert_array_equal(summary.map_bin, dense.map_bin)
    assert_array_equal(summary.map_position, dense.map_position)


# ---------------------------------------------------------------------------
# 3. score
# ---------------------------------------------------------------------------


class TestScore:
    def test_median_error_matches_error_against(self, sim) -> None:
        env, spikes, times, positions = sim
        fit = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)

        score = fit.score(spikes, times, positions, metric="median_error")
        errors = fit.predict(spikes, times).error_against(times, positions)
        assert score == pytest.approx(float(np.nanmedian(errors)))

    def test_mean_error_matches_error_against(self, sim) -> None:
        env, spikes, times, positions = sim
        fit = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)

        score = fit.score(spikes, times, positions, metric="mean_error")
        errors = fit.predict(spikes, times).error_against(times, positions)
        assert score == pytest.approx(float(np.nanmean(errors)))

    def test_default_metric_is_median(self, sim) -> None:
        env, spikes, times, positions = sim
        fit = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
        assert fit.score(spikes, times, positions) == pytest.approx(
            fit.score(spikes, times, positions, metric="median_error")
        )

    def test_unknown_metric_raises(self, sim) -> None:
        env, spikes, times, positions = sim
        fit = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
        with pytest.raises(ValueError, match="metric"):
            fit.score(spikes, times, positions, metric="rmse")


# ---------------------------------------------------------------------------
# 4. train/test epoch split
# ---------------------------------------------------------------------------


def test_fit_epochs_matches_compute_spatial_rates(continuous_recording):
    from neurospatial.encoding import compute_spatial_rates

    r = continuous_recording
    trains = [r.spike_times + 0.04 * u for u in range(5)]
    fitted = BayesianDecoder(r.env, method="binned").fit(
        trains, r.times, r.positions, epochs=[(0, 100)]
    )
    expected = compute_spatial_rates(
        r.env,
        trains,
        r.times,
        r.positions,
        method="binned",
        bandwidth=None,
        min_occupancy=None,
        max_gap=0.5,
        epochs=[(0, 100)],
    )
    np.testing.assert_array_equal(fitted.encoding_models, expected.firing_rates)
    with pytest.raises(TypeError, match="epoch"):
        BayesianDecoder(r.env).fit(trains, r.times, r.positions, epoch=(0, 100))


def test_predict_forwards_max_gap_and_windows(two_epoch_recording):
    r = two_epoch_recording
    trains = [r.spike_times + 0.04 * u for u in range(5)]
    default = BayesianDecoder(r.env, method="binned").fit(trains, r.times, r.positions)
    wide = BayesianDecoder(r.env, method="binned", max_gap=2000.0).fit(
        trains, r.times, r.positions
    )
    assert len(default.predict(trains, r.times).times) == 7998
    assert len(wide.predict(trains, r.times).times) == 47999
    predicted = default.predict(trains, r.times, epochs=[(1100, 1200)])
    summary = default.predict_summary(
        trains, r.times, epochs=[(1100, 1200)], time_chunk=1000
    )
    assert len(predicted.times) == 3999
    np.testing.assert_array_equal(summary.times, predicted.times)
    np.testing.assert_array_equal(summary.map_bin, predicted.map_bin)
    score = default.score(trains, r.times, r.positions, epochs=[(1100, 1200)])
    expected = np.nanmedian(predicted.error_against(r.times, r.positions))
    assert score == pytest.approx(expected)


def test_epochs_restrict_encoding(sim) -> None:
    """Analysis windows train on original samples and preserve held-out parity."""
    from neurospatial.encoding import compute_spatial_rates

    env, spikes, times, positions = sim
    mid = float(times[len(times) // 2])
    epoch = (float(times[0]), mid)
    fitted = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions, epochs=epoch)
    models = compute_spatial_rates(
        env,
        spikes,
        times,
        positions,
        epochs=epoch,
        bandwidth=None,
        method="diffusion_kde",
        min_occupancy=None,
    ).firing_rates
    assert_array_equal(fitted.encoding_models, models)
    held_out = (mid, float(times[-1]))
    predicted = fitted.predict(spikes, times, epochs=held_out)
    reference = _decode_with_models(
        env,
        spikes,
        times,
        models,
        dt=0.5,
        epochs=held_out,
        spatial_support=fitted.spatial_support,
    )
    assert_array_equal(predicted.posterior, reference.posterior)
    full = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
    assert not np.array_equal(fitted.encoding_models, full.encoding_models)


def test_epochs_keep_speed_aligned_to_original_samples(sim) -> None:
    """Epoch gating keeps a full-length supplied speed aligned with tracking."""
    from neurospatial.encoding import compute_spatial_rates

    env, spikes, times, positions = sim
    epoch = (float(times[0]), float(times[len(times) // 2]))
    speed = np.full(times.shape[0], 20.0, dtype=np.float64)
    fitted = BayesianDecoder(env, dt=0.5).fit(
        spikes, times, positions, epochs=epoch, speed=speed, min_speed=5.0
    )
    expected = compute_spatial_rates(
        env,
        spikes,
        times,
        positions,
        epochs=epoch,
        speed=speed,
        min_speed=5.0,
        bandwidth=None,
        method="diffusion_kde",
        min_occupancy=None,
    ).firing_rates
    assert fitted.is_fitted
    assert_array_equal(fitted.encoding_models, expected)


# ---------------------------------------------------------------------------
# 5. Unfitted raises
# ---------------------------------------------------------------------------


class TestUnfitted:
    def test_predict_raises(self, sim) -> None:
        env, spikes, times, _ = sim
        with pytest.raises(RuntimeError, match="not fitted"):
            BayesianDecoder(env).predict(spikes, times)

    def test_predict_summary_raises(self, sim) -> None:
        env, spikes, times, _ = sim
        with pytest.raises(RuntimeError, match="not fitted"):
            BayesianDecoder(env).predict_summary(spikes, times)

    def test_score_raises(self, sim) -> None:
        env, spikes, times, positions = sim
        with pytest.raises(RuntimeError, match="not fitted"):
            BayesianDecoder(env).score(spikes, times, positions)

    def test_discarded_fit_error_shows_reassignment(self, sim) -> None:
        # fit() returns a new decoder, so a bare `decoder.fit(...)` leaves
        # `decoder` unfitted. Repeating ".fit(...)" would not help; the error
        # must show how to keep the fitted result.
        env, spikes, times, positions = sim
        decoder = BayesianDecoder(env, dt=0.5)
        decoder.fit(spikes, times, positions)
        with pytest.raises(RuntimeError, match=r"decoder = decoder\.fit\(") as exc:
            decoder.predict(spikes, times)
        assert "returns a new" in str(exc.value)


# ---------------------------------------------------------------------------
# 6. Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_fit_returns_new_object_original_unfitted(self, sim) -> None:
        env, spikes, times, positions = sim
        original = BayesianDecoder(env, dt=0.5)
        fitted = original.fit(spikes, times, positions)

        assert fitted is not original
        assert original.encoding_models is None  # original still unfitted
        assert fitted.encoding_models is not None

    def test_fit_preserves_config(self, sim) -> None:
        env, spikes, times, positions = sim
        original = BayesianDecoder(env, dt=0.5, bandwidth=7.0, min_occupancy=0.1)
        fitted = original.fit(spikes, times, positions)
        assert fitted.dt == 0.5
        assert fitted.bandwidth == 7.0
        assert fitted.min_occupancy == 0.1

    def test_frozen_rebinding_raises(self, sim) -> None:
        env, *_ = sim
        dec = BayesianDecoder(env)
        with pytest.raises(dataclasses.FrozenInstanceError):
            dec.dt = 0.1  # type: ignore[misc]

    def test_from_rates_decoder_isolated_from_rate_result(self, sim) -> None:
        # A fitted decoder is a snapshot: the rate maps it was built from are
        # read-only, and editing their recorded settings does not reach it.
        from neurospatial.encoding import compute_spatial_rates

        env, spikes, times, positions = sim
        rates = compute_spatial_rates(env, spikes, times, positions)
        decoder = BayesianDecoder.from_rates(rates, dt=0.5)
        with pytest.raises(ValueError, match="read-only"):
            np.asarray(rates.firing_rates)[:] = 1.0
        rates.analysis_settings["method"] = "edited"
        assert decoder.analysis_settings["method"] == "diffusion_kde"

    def test_fitted_state_is_read_only(self, sim) -> None:
        env, spikes, times, positions = sim
        fitted = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
        for name in ("encoding_models", "unit_ids", "spatial_support"):
            assert getattr(fitted, name).flags.writeable is False, name
        with pytest.raises(ValueError, match="read-only"):
            fitted.encoding_models[0, 0] = 0.0

    @pytest.mark.parametrize("how", ["pickle", "deepcopy"])
    def test_copied_decoder_stays_read_only(self, sim, how) -> None:
        import copy
        import pickle

        env, spikes, times, positions = sim
        fitted = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
        copied = (
            pickle.loads(pickle.dumps(fitted))
            if how == "pickle"
            else copy.deepcopy(fitted)
        )
        assert_array_equal(copied.encoding_models, fitted.encoding_models)
        for name in ("encoding_models", "unit_ids", "spatial_support"):
            assert getattr(copied, name).flags.writeable is False, name


# ---------------------------------------------------------------------------
# 7. SpikeTrainsLike input
# ---------------------------------------------------------------------------


def test_spiketrains_input_parity(sim) -> None:
    """A SpikeTrains group flows through fit/predict like a plain list."""
    env, spikes, times, positions = sim
    st = SpikeTrains(spikes, unit_ids=np.arange(10, 10 + len(spikes)))

    dec_group = (
        BayesianDecoder(env, dt=0.5).fit(st, times, positions).predict(st, times)
    )
    dec_list = (
        BayesianDecoder(env, dt=0.5)
        .fit(spikes, times, positions)
        .predict(spikes, times)
    )
    assert_array_equal(dec_group.posterior, dec_list.posterior)


def test_spiketrains_unit_ids_captured(sim) -> None:
    """fit captures unit_ids from a SpikeTrains group for introspection."""
    env, spikes, times, positions = sim
    ids = np.arange(100, 100 + len(spikes))
    st = SpikeTrains(spikes, unit_ids=ids)
    fit = BayesianDecoder(env, dt=0.5).fit(st, times, positions)
    assert_array_equal(fit.unit_ids, ids)


def test_plain_list_unit_ids_default_to_arange(sim) -> None:
    """A plain-list input (no ids) fits with default arange unit_ids."""
    env, spikes, times, positions = sim
    fit = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
    assert_array_equal(fit.unit_ids, np.arange(len(spikes)))


# ---------------------------------------------------------------------------
# 8. Held-out accuracy
# ---------------------------------------------------------------------------


def _chance_error(env, result, times, positions, metric):
    """Median error after permuting the MAP positions in time."""
    from neurospatial.decoding import decoding_error

    actual = np.column_stack(
        [np.interp(result.times, times, positions[:, d]) for d in range(2)]
    )
    shuffled = np.random.default_rng(3).permutation(result.map_position)
    return float(np.nanmedian(decoding_error(env, shuffled, actual, metric=metric)))


def test_held_out_open_field_accuracy() -> None:
    """Fit on the first half, decode the second: error well below chance."""
    env, spikes, times, positions = _make_sim(n_neurons=12, duration=120.0)
    half = times[len(times) // 2]
    test = (half, times[-1] + 1e-9)
    decoder = BayesianDecoder(env, dt=0.5).fit(
        spikes, times, positions, epochs=(times[0], half)
    )
    result = decoder.predict(spikes, times, epochs=test)
    error = float(
        np.nanmedian(result.error_against(times, positions, metric="euclidean"))
    )
    chance = _chance_error(env, result, times, positions, "euclidean")
    assert error < 10.0, error
    assert error < 0.4 * chance, (error, chance)


def test_held_out_maze_geodesic_accuracy() -> None:
    """On a W-maze track, held-out geodesic error is about one bin."""
    from neurospatial.simulation import (
        PlaceCellModel,
        generate_poisson_spikes,
        simulate_trajectory_ou,
    )
    from neurospatial.simulation.mazes import make_w_maze

    env = make_w_maze(bin_size=4.0).env_track
    assert env.is_linearized_track
    positions, times = simulate_trajectory_ou(
        env, duration=300.0, seed=0, speed_units="cm"
    )
    centers = env.bin_centers[np.linspace(0, env.n_bins - 1, 30).astype(int)]
    spikes = [
        generate_poisson_spikes(
            PlaceCellModel(env, center=c, width=8.0, max_rate=25.0, seed=i).firing_rate(
                times,
                positions,
            ),
            times,
            seed=100 + i,
        )
        for i, c in enumerate(centers)
    ]
    half = times[len(times) // 2]
    test = (half, times[-1] + 1e-9)
    decoder = BayesianDecoder(env, dt=0.25).fit(
        spikes, times, positions, epochs=(times[0], half)
    )
    error = decoder.score(spikes, times, positions, epochs=test, distance="geodesic")
    result = decoder.predict(spikes, times, epochs=test)
    chance = _chance_error(env, result, times, positions, "geodesic")
    bin_size = float(np.median(env.bin_sizes))
    assert error <= 2 * bin_size, error
    assert error < 0.2 * chance, (error, chance)


# ---------------------------------------------------------------------------
# 9. score: undecodable-bin handling + distance= (FIX 1)
# ---------------------------------------------------------------------------


def _fitted_minimal(env: Environment | None = None) -> BayesianDecoder:
    """A fitted decoder built from hand-made models."""
    env = env if env is not None else _small_env()
    models = np.ones((2, env.n_bins))
    return BayesianDecoder.from_models(env, models, unit_ids=np.arange(2))


def _result_with_nan_rows(
    env: Environment, nan_rows: list[int], n_time: int = 4
) -> DecodingResult:
    """A DecodingResult whose listed posterior rows are entirely NaN."""
    n_bins = env.n_bins
    posterior = np.full((n_time, n_bins), 1.0 / n_bins)
    for r in nan_rows:
        posterior[r] = np.nan
    decode_times = np.linspace(0.0, 0.75, n_time)
    return DecodingResult(posterior=posterior, env=env, times=decode_times)


class TestScoreUndecodable:
    _TRUE_TIMES = np.array([0.0, 1.0])
    _TRUE_POS = np.array([[0.0, 0.0], [10.0, 10.0]])

    def test_partially_undecodable_warns_and_scores_survivors(self, monkeypatch):
        dec = _fitted_minimal()
        result = _result_with_nan_rows(dec.env, nan_rows=[1])
        monkeypatch.setattr(
            BayesianDecoder, "predict", lambda self, s, t, **kwargs: result
        )

        with pytest.warns(UserWarning, match="undecodable"):
            score = dec.score([np.array([0.1])] * 2, self._TRUE_TIMES, self._TRUE_POS)

        errors = result.error_against(self._TRUE_TIMES, self._TRUE_POS)
        assert np.isnan(errors[1])  # the undecodable row is dropped by nanmedian
        assert score == pytest.approx(float(np.nanmedian(errors)))
        assert np.isfinite(score)

    def test_all_undecodable_raises_not_nan(self, monkeypatch):
        dec = _fitted_minimal()
        result = _result_with_nan_rows(dec.env, nan_rows=[0, 1, 2, 3])
        monkeypatch.setattr(
            BayesianDecoder, "predict", lambda self, s, t, **kwargs: result
        )

        with pytest.raises(ValueError, match="could not decode any time bin"):
            dec.score([np.array([0.1])] * 2, self._TRUE_TIMES, self._TRUE_POS)

    def test_all_decodable_no_warning_and_equals_nanmedian(self, sim):
        env, spikes, times, positions = sim
        fit = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            score = fit.score(spikes, times, positions)
        assert not any("undecodable" in str(w.message) for w in rec)

        errors = fit.predict(spikes, times).error_against(times, positions)
        assert score == pytest.approx(float(np.nanmedian(errors)))

    def test_distance_geodesic_forwards_on_graph_env(self):
        env, spikes, times, positions = _make_linear_sim()
        fit = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)

        score = fit.score(spikes, times, positions, distance="geodesic")
        errors = fit.predict(spikes, times).error_against(
            times, positions, metric="geodesic"
        )
        assert score == pytest.approx(float(np.nanmedian(errors)))
        assert np.isfinite(score)

    def test_unknown_metric_raises_before_predict(self, monkeypatch):
        dec = _fitted_minimal()

        def _spy(self, *a, **k):
            raise AssertionError("predict must not run when metric is invalid")

        monkeypatch.setattr(BayesianDecoder, "predict", _spy)
        with pytest.raises(ValueError, match="metric"):
            dec.score(
                [np.array([0.1])] * 2, self._TRUE_TIMES, self._TRUE_POS, metric="rmse"
            )

    def test_unknown_distance_raises_before_predict(self, monkeypatch):
        dec = _fitted_minimal()

        def _spy(self, *a, **k):
            raise AssertionError("predict must not run when distance is invalid")

        monkeypatch.setattr(BayesianDecoder, "predict", _spy)
        with pytest.raises(ValueError, match="distance"):
            dec.score(
                [np.array([0.1])] * 2,
                self._TRUE_TIMES,
                self._TRUE_POS,
                distance="manhattan",
            )


# ---------------------------------------------------------------------------
# 10. Construction-time validation via __post_init__ (FIX 2)
# ---------------------------------------------------------------------------


class TestConstructionValidation:
    def test_negative_dt_raises(self):
        env = _small_env()
        with pytest.raises(ValueError, match="dt"):
            BayesianDecoder(env, dt=-1.0)

    def test_zero_dt_raises(self):
        env = _small_env()
        with pytest.raises(ValueError, match="dt"):
            BayesianDecoder(env, dt=0.0)

    def test_bad_dtype_raises(self):
        env = _small_env()
        with pytest.raises(ValueError, match="dtype"):
            BayesianDecoder(env, dtype=np.float16)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
    def test_from_models_refuses_invalid_rates_on_support(self, bad):
        """Every allowed bin needs a finite, non-negative rate for every unit."""
        env = _small_env()
        models = np.full((2, env.n_bins), 5.0)
        models[1, 0] = bad
        with pytest.raises(ValueError, match="spatial_support") as err:
            BayesianDecoder.from_models(env, models)
        assert "Fix:" in str(err.value)
        support = np.ones(env.n_bins, dtype=bool)
        support[0] = False
        if np.isnan(bad) or np.isinf(bad):
            # Outside the supported bins a missing rate is allowed.
            decoder = BayesianDecoder.from_models(env, models, spatial_support=support)
            assert decoder.is_fitted
        else:
            with pytest.raises(ValueError, match="negative"):
                BayesianDecoder.from_models(env, models, spatial_support=support)

    def test_encoding_models_wrong_nbins_raises(self):
        env = _small_env()
        bad = np.zeros((3, env.n_bins + 1))  # wrong bin axis
        with pytest.raises(ValueError, match="bins"):
            BayesianDecoder.from_models(env, bad, unit_ids=np.array([0, 1, 2]))

    def test_encoding_models_without_unit_ids_pair_by_position(self):
        env = _small_env()
        decoder = BayesianDecoder.from_models(env, np.ones((2, env.n_bins)))
        assert_array_equal(decoder.unit_ids, [0, 1])
        assert decoder._unit_ids_generated is True

    def test_encoding_models_unit_count_mismatch_raises(self):
        env = _small_env()
        models = np.ones((2, env.n_bins))
        with pytest.raises(ValueError, match="unit"):
            BayesianDecoder.from_models(env, models, unit_ids=np.array([0, 1, 2]))

    def test_unfitted_construct_ok(self):
        env = _small_env()
        dec = BayesianDecoder(env)
        assert dec.encoding_models is None

    def test_real_fit_construct_ok(self, sim):
        env, spikes, times, positions = sim
        fit = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
        assert fit.encoding_models is not None
        assert fit.encoding_models.shape[1] == env.n_bins

    def test_constructor_takes_configuration_only(self):
        # Fitted state comes from fit, from_rates or from_models; the
        # constructor shows only the settings a scientist chooses.
        params = list(inspect.signature(BayesianDecoder).parameters)
        assert [name for name in params if not name.startswith("_")] == [
            "env",
            "dt",
            "bandwidth",
            "method",
            "min_occupancy",
            "penalty",
            "rank",
            "max_gap",
            "dtype",
            "warn_on_drop",
        ]
        assert [name for name in params if name.startswith("_")] == ["_fit"]

    def test_replace_keeps_fitted_models(self, sim):
        env, spikes, times, positions = sim
        fitted = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
        coarser = dataclasses.replace(fitted, dt=1.0)
        assert coarser.dt == 1.0
        assert coarser.encoding_models is fitted.encoding_models
        assert coarser.predict(spikes, times).posterior.shape[1] == env.n_bins

    def test_from_models_keeps_a_read_only_copy(self):
        env = _small_env()
        models = np.ones((2, env.n_bins))
        decoder = BayesianDecoder.from_models(env, models)
        models[:] = 0.0
        assert_array_equal(decoder.encoding_models, 1.0)
        assert decoder.encoding_models.flags.writeable is False

    def test_from_models_reproduces_fitted_decoder(self, sim):
        env, spikes, times, positions = sim
        fitted = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
        rebuilt = BayesianDecoder.from_models(
            env,
            fitted.encoding_models,
            spatial_support=fitted.spatial_support,
            dt=0.5,
        )
        assert_array_equal(
            rebuilt.predict(spikes, times).posterior,
            fitted.predict(spikes, times).posterior,
        )


# ---------------------------------------------------------------------------
# 11. is_fitted read-only property (FIX 3)
# ---------------------------------------------------------------------------


class TestIsFitted:
    def test_false_when_unfitted(self):
        assert BayesianDecoder(_small_env()).is_fitted is False

    def test_true_after_fit(self, sim):
        env, spikes, times, positions = sim
        fit = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
        assert fit.is_fitted is True


# ---------------------------------------------------------------------------
# 12. warn_on_drop config knob (FIX 4)
# ---------------------------------------------------------------------------


def test_warn_on_drop_false_silences_out_of_window_warning(sim):
    env, spikes, times, positions = sim
    fit = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
    fit_silent = BayesianDecoder(env, dt=0.5, warn_on_drop=False).fit(
        spikes, times, positions
    )

    # Out-of-window spikes (ms vs s): >50% fall outside the decode window.
    bad_spikes = [s * 1000.0 for s in spikes]
    msg = "fell outside the decode time window"

    with warnings.catch_warnings(record=True) as rec_default:
        warnings.simplefilter("always")
        fit.predict(bad_spikes, times)
    assert any(msg in str(w.message) for w in rec_default)

    with warnings.catch_warnings(record=True) as rec_silent:
        warnings.simplefilter("always")
        fit_silent.predict(bad_spikes, times)
    assert not any(msg in str(w.message) for w in rec_silent)


# ---------------------------------------------------------------------------
# 13. Errors for training windows that exclude every interval
# ---------------------------------------------------------------------------


def test_fit_epochs_excluding_all_intervals_reports_fix(sim):
    env, spikes, times, positions = sim
    epoch = (float(times[0]) - 5.0, float(times[0]) - 1.0)
    with (
        pytest.warns(UserWarning, match="epochs"),
        pytest.raises(ValueError, match="no occupied bin") as caught,
    ):
        BayesianDecoder(env, dt=0.5).fit(spikes, times, positions, epochs=epoch)
    assert "epochs" in str(caught.value)
    assert any(line.startswith("Fix:") for line in str(caught.value).splitlines())


# ---------------------------------------------------------------------------
# Pairing spike trains with encoding models (population identity)
# ---------------------------------------------------------------------------

_FIT_IDS = np.arange(10, 18)


@pytest.fixture(scope="module")
def labelled_fit(sim, make_spike_group):
    """Decoder fitted on a labelled 8-unit group (ids 10..17), plus the sim."""
    env, spikes, times, positions = sim
    group = make_spike_group(spikes, _FIT_IDS)
    decoder = BayesianDecoder(env, dt=0.5).fit(group, times, positions)
    return decoder, spikes, times


class TestUnitAlignment:
    def test_predict_aligns_reordered_labels(
        self, labelled_fit, make_spike_group
    ) -> None:
        decoder, spikes, times = labelled_fit
        in_order = decoder.predict(make_spike_group(spikes, _FIT_IDS), times)
        reversed_group = make_spike_group(spikes[::-1], _FIT_IDS[::-1])
        reordered = decoder.predict(reversed_group, times)
        np.testing.assert_allclose(reordered.posterior, in_order.posterior, atol=1e-12)

    @pytest.mark.parametrize("call", ["predict", "predict_summary", "score"])
    def test_predict_label_mismatch_lists_labels(
        self, sim, labelled_fit, make_spike_group, call
    ) -> None:
        _, _, _, positions = sim
        decoder, spikes, times = labelled_fit
        shifted = make_spike_group(spikes, _FIT_IDS + 1)
        args = (shifted, times, positions) if call == "score" else (shifted, times)
        with pytest.raises(ValueError) as excinfo:
            getattr(decoder, call)(*args)
        message = str(excinfo.value)
        assert "missing: [10]" in message
        assert "unexpected: [18]" in message
        assert "\nFix:" in message

    def test_constructor_labels_are_caller_supplied(
        self, sim, make_spike_group
    ) -> None:
        env, spikes, times, positions = sim
        a, b = spikes[0], spikes[1]
        fitted = BayesianDecoder(env, dt=0.5).fit([a, b], times, positions)
        assert fitted._unit_ids_generated is True

        decoder = BayesianDecoder.from_models(
            env,
            fitted.encoding_models,
            unit_ids=[10, 20],
            spatial_support=fitted.spatial_support,
            dt=0.5,
        )
        assert decoder._unit_ids_generated is False
        assert dataclasses.replace(decoder, dt=0.05)._unit_ids_generated is False
        # Unit 20's spikes must meet unit 20's model whatever the input order.
        in_order = decoder.predict(make_spike_group([a, b], [10, 20]), times)
        swapped = decoder.predict(make_spike_group([b, a], [20, 10]), times)
        np.testing.assert_allclose(swapped.posterior, in_order.posterior, atol=1e-12)

    def test_positional_count_mismatch_raises(self, sim) -> None:
        env, spikes, times, positions = sim
        decoder = BayesianDecoder(env, dt=0.5).fit(spikes[:3], times, positions)
        with pytest.raises(
            ValueError, match=r"Got 2 spike trains .* 3 units"
        ) as excinfo:
            decoder.predict(spikes[:2], times)
        assert "\nFix:" in str(excinfo.value)

    def test_duplicate_input_labels_raise(self, labelled_fit, make_spike_group) -> None:
        decoder, spikes, times = labelled_fit
        ids = _FIT_IDS.copy()
        ids[1] = 10
        with pytest.raises(ValueError, match=r"\[10\]"):
            decoder.predict(make_spike_group(spikes, ids), times)

    def test_duplicate_fitted_labels_raise(self, sim, make_spike_group) -> None:
        env, spikes, times, positions = sim
        fitted = BayesianDecoder(env, dt=0.5).fit(spikes[:2], times, positions)
        with pytest.raises(ValueError, match=r"unique.*\[10\]"):
            BayesianDecoder.from_models(
                env,
                fitted.encoding_models,
                unit_ids=[10, 10],
                spatial_support=fitted.spatial_support,
                dt=0.5,
            )
        with pytest.raises(ValueError, match=r"unique.*\[10\]"):
            BayesianDecoder(env, dt=0.5).fit(
                make_spike_group(spikes[:3], [10, 10, 12]), times, positions
            )

    def test_duplicate_input_error_names_the_method(
        self, labelled_fit, make_spike_group
    ) -> None:
        decoder, spikes, times = labelled_fit
        ids = _FIT_IDS.copy()
        ids[1] = 10
        with pytest.raises(ValueError, match=r"BayesianDecoder\.predict_summary"):
            decoder.predict_summary(make_spike_group(spikes, ids), times)

    def test_predict_plain_arrays_stay_positional(self, sim) -> None:
        env, spikes, times, positions = sim
        decoder = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
        ref = _decode_with_models(
            env,
            spikes,
            times,
            decoder.encoding_models,
            dt=0.5,
            spatial_support=decoder.spatial_support,
        )
        assert_array_equal(decoder.predict(spikes, times).posterior, ref.posterior)


def test_positions_required_where_used(sim):
    from neurospatial.decoding import decode_session_summary
    from neurospatial.encoding import compute_spatial_rate, compute_spatial_rates

    for function in [
        BayesianDecoder.fit,
        BayesianDecoder.score,
        decode_session,
        decode_session_summary,
        compute_spatial_rate,
        compute_spatial_rates,
    ]:
        signature = inspect.signature(function)
        assert signature.parameters["positions"].default is inspect.Parameter.empty
    for function in [decode_session, decode_session_summary]:
        assert "encoding_models" not in inspect.signature(function).parameters
    assert list(inspect.signature(BayesianDecoder.predict).parameters) == [
        "self",
        "spike_times",
        "times",
        "epochs",
        "break_before",
        "spike_window",
    ]
    assert list(inspect.signature(BayesianDecoder.predict_summary).parameters) == [
        "self",
        "spike_times",
        "times",
        "epochs",
        "break_before",
        "spike_window",
        "time_chunk",
    ]
    env, spikes, times, _ = sim
    with pytest.raises(TypeError, match="positions"):
        decode_session(env, spikes, times)


def test_predict_matches_decode_session(sim):
    env, spikes, times, positions = sim
    decoder = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
    np.testing.assert_array_equal(
        decoder.predict(spikes, times).posterior,
        decode_session(env, spikes, times, positions, dt=0.5).posterior,
    )


def test_fit_unit_ids_enable_label_alignment(sim, make_spike_group):
    env, spikes, times, positions = sim
    trains = spikes[:3]
    decoder = BayesianDecoder(env, dt=0.5).fit(
        trains, times, positions, unit_ids=[10, 11, 12]
    )
    reordered = make_spike_group(trains[::-1], index=[12, 11, 10])
    np.testing.assert_array_equal(
        decoder.predict(reordered, times).posterior,
        decoder.predict(make_spike_group(trains, index=[10, 11, 12]), times).posterior,
    )
    with pytest.raises(ValueError) as caught:
        decoder.predict(make_spike_group(trains, index=[10, 11, 13]), times)
    assert "missing: [12]" in str(caught.value)
    assert "unexpected: [13]" in str(caught.value)


@pytest.mark.parametrize("labelled_side", ["fit", "predict"])
@pytest.mark.parametrize("method", ["predict", "predict_summary", "score"])
def test_one_sided_unit_labels_raise(sim, make_spike_group, labelled_side, method):
    """Labels on only one side cannot pair trains with models.

    Positional pairing of a labelled group listed in another order would
    decode with the wrong models, silently.
    """
    env, spikes, times, positions = sim
    trains = spikes[:3]
    labels = [10, 11, 12]
    decoder = BayesianDecoder(env, dt=0.5)
    if labelled_side == "fit":
        decoder = decoder.fit(make_spike_group(trains, index=labels), times, positions)
        test_input = trains[::-1]
    else:
        decoder = decoder.fit(trains, times, positions)
        test_input = make_spike_group(trains[::-1], index=labels[::-1])
    args = (test_input, times, positions) if method == "score" else (test_input, times)
    with pytest.raises(ValueError, match="unit labels") as caught:
        getattr(decoder, method)(*args)
    assert "Fix:" in str(caught.value)


def test_fit_unit_ids_must_match_group_labels(sim, make_spike_group):
    env, spikes, times, positions = sim
    group = make_spike_group(spikes[:2], index=[10, 20])
    decoder = BayesianDecoder(env, dt=0.5)
    with pytest.raises(ValueError) as caught:
        decoder.fit(group, times, positions, unit_ids=[20, 10])
    for text in ["[20, 10]", "[10, 20]", "Fix:"]:
        assert text in str(caught.value)
    accepted = decoder.fit(group, times, positions, unit_ids=[10, 20])
    np.testing.assert_array_equal(accepted.unit_ids, [10, 20])
    supplied = decoder.fit(spikes[:2], times, positions, unit_ids=[20, 10])
    np.testing.assert_array_equal(supplied.unit_ids, [20, 10])
    with pytest.raises(ValueError, match="unique"):
        decoder.fit(spikes[:3], times, positions, unit_ids=[3, 3, 7])


def test_from_rates_matches_fit(two_epoch_recording):
    from neurospatial.encoding import compute_spatial_rates

    r = two_epoch_recording
    trains = [np.arange(0, 100, 0.4), np.arange(1100, 1200, 0.7)]
    rates = compute_spatial_rates(
        r.env,
        trains,
        r.times,
        r.positions,
        spike_window=[(0, 100), (1100, 1200)],
    )
    from_rates = BayesianDecoder.from_rates(rates)
    fitted = BayesianDecoder(r.env).fit(
        trains, r.times, r.positions, spike_window=[(0, 100), (1100, 1200)]
    )
    np.testing.assert_allclose(
        from_rates.predict(trains, r.times).posterior,
        fitted.predict(trains, r.times).posterior,
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_array_equal(from_rates.spike_window, rates.spike_window)
    assert from_rates.is_fitted


def test_from_rates_label_alignment(sim, make_spike_group):
    from neurospatial.encoding import compute_spatial_rates

    env, spikes, times, positions = sim
    trains = spikes[:3]
    rates = compute_spatial_rates(env, trains, times, positions)
    decoder = BayesianDecoder.from_rates(rates, dt=0.5)
    with pytest.raises(ValueError, match="unit labels"):
        decoder.predict(make_spike_group(trains, index=[3, 7, 9]), times)
    labelled = compute_spatial_rates(
        env,
        make_spike_group(trains, index=[10, 11, 12]),
        times,
        positions,
    )
    decoder = BayesianDecoder.from_rates(labelled, dt=0.5)
    np.testing.assert_array_equal(
        decoder.predict(
            make_spike_group(trains[::-1], index=[12, 11, 10]), times
        ).posterior,
        decoder.predict(make_spike_group(trains, index=[10, 11, 12]), times).posterior,
    )
    with pytest.raises(ValueError) as caught:
        decoder.predict(make_spike_group(trains, index=[10, 11, 13]), times)
    assert "missing: [12]" in str(caught.value)
    assert "unexpected: [13]" in str(caught.value)


def test_from_rates_constructor_labels(sim, make_spike_group):
    from neurospatial.encoding import SpatialRatesResult

    env, spikes, times, _ = sim
    bins = np.arange(env.n_bins)
    maps = np.vstack([1 + bins, 1 + bins[::-1]])
    rates = SpatialRatesResult(
        maps, np.ones(env.n_bins), env, "binned", 5, unit_ids=[10, 20]
    )
    decoder = BayesianDecoder.from_rates(rates, dt=0.5)
    a = make_spike_group(spikes[:2], index=[10, 20])
    b = make_spike_group(spikes[:2][::-1], index=[20, 10])
    np.testing.assert_array_equal(
        decoder.predict(a, times).posterior, decoder.predict(b, times).posterior
    )
    assert not rates._unit_ids_generated
    assert not rates[0]._unit_ids_generated
    assert not dataclasses.replace(rates)._unit_ids_generated
    generated = SpatialRatesResult(maps, np.ones(env.n_bins), env, "binned", 5)
    assert generated._unit_ids_generated
    assert generated[0]._unit_ids_generated
    assert dataclasses.replace(generated)._unit_ids_generated
    decoder = BayesianDecoder.from_rates(generated, dt=0.5)
    with pytest.raises(ValueError, match="unit labels"):
        decoder.predict(a, times)


@pytest.mark.parametrize(
    "method, options",
    [
        ("binned", {"bandwidth": 5}),
        ("diffusion_kde", {"bandwidth": 5}),
        ("glm", {"rank": 8, "penalty": 1.0}),
    ],
)
def test_spatial_unit_identity_provenance_all_methods(sim, method, options):
    from neurospatial.encoding import compute_spatial_rates

    env, spikes, times, positions = sim
    unlabelled = compute_spatial_rates(
        env, spikes[:2], times, positions, method=method, **options
    )
    labelled = compute_spatial_rates(
        env, spikes[:2], times, positions, method=method, unit_ids=[7, 9], **options
    )
    assert unlabelled._unit_ids_generated and unlabelled[0]._unit_ids_generated
    assert not labelled._unit_ids_generated and not labelled[0]._unit_ids_generated


def test_from_rates_rejects_other_types():
    from neurospatial.encoding import compute_directional_rates

    times = np.arange(300) / 30
    rates = compute_directional_rates([times[::10]], times, np.sin(times))
    with pytest.raises(TypeError, match="compute_spatial_rates"):
        BayesianDecoder.from_rates(rates)


def test_from_rates_nan_bins_are_excluded_from_spatial_support(sim):
    from neurospatial.decoding import bin_spikes_in_time, decode_position
    from neurospatial.encoding import SpatialRatesResult

    env, spikes, times, _ = sim
    maps = np.ones((2, env.n_bins))
    maps[0, 0] = np.nan
    rates = SpatialRatesResult(maps, np.ones(env.n_bins), env, "binned", 5)
    decoder = BayesianDecoder.from_rates(rates, dt=0.5)
    actual = decoder.predict(spikes[:2], times)
    np.testing.assert_array_equal(actual.posterior[:, 0], 0.0)
    counts, centers = bin_spikes_in_time(
        spikes[:2], dt=0.5, t_start=times[0], t_stop=times[-1]
    )
    expected = decode_position(env, counts, rates, dt=0.5, times=centers)
    np.testing.assert_array_equal(actual.posterior, expected.posterior)
    assert np.isnan(maps[0, 0])


@pytest.mark.parametrize(
    "fit_kwargs",
    [
        pytest.param({"min_speed": 1e6}, id="speed-gate-excludes-all"),
    ],
)
def test_fit_without_surviving_data_raises(sim, fit_kwargs) -> None:
    """A model fit on no data would decode every time bin as uniform."""
    env, spike_times, times, positions = sim
    with (
        warnings.catch_warnings(),
        pytest.raises(ValueError, match=r"no occupied bin") as exc,
    ):
        warnings.simplefilter("ignore")
        BayesianDecoder(env, dt=0.5, bandwidth=7.0).fit(
            spike_times, times, positions, **fit_kwargs
        )
    assert "Fix:" in str(exc.value)


def test_min_occupancy_masking_every_bin_raises(sim) -> None:
    env, spike_times, times, positions = sim
    with (
        warnings.catch_warnings(),
        pytest.raises(ValueError, match=r"min_occupancy"),
    ):
        warnings.simplefilter("ignore")
        BayesianDecoder(env, dt=0.5, bandwidth=7.0, min_occupancy=1e6).fit(
            spike_times, times, positions
        )


def test_fitted_decoder_refuses_a_different_environment(sim):
    """Equal bin counts do not make the fitted maps valid on another arena."""
    from neurospatial import IncompatibleEnvironmentError

    env, spikes, times, positions = sim
    fitted = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
    shifted = Environment.from_samples(
        env.bin_centers + 1000.0, bin_size=5.0
    ).with_metadata(units="cm")
    assert shifted.n_bins == env.n_bins
    with pytest.raises(IncompatibleEnvironmentError, match="env cannot change") as err:
        dataclasses.replace(fitted, env=shifted)
    assert "from_models" in str(err.value)
    # An equivalent copy of the training environment is accepted.
    copy = dataclasses.replace(fitted, env=env.with_metadata())
    assert_array_equal(
        copy.predict(spikes, times).posterior, fitted.predict(spikes, times).posterior
    )
