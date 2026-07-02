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
4. train/test epoch split: fit(epoch=...) restricts the encoding models.
5. Unfitted predict/predict_summary/score raise a clear RuntimeError.
6. Immutability: fit returns a new object; original stays unfitted; frozen.
7. SpikeTrains (a SpikeTrainsLike group) input yields the plain-list posterior.
8. Linearized-track smoke: fit + predict runs on a 1-D track env.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial import Environment
from neurospatial.behavior import restrict, restrict_spike_trains
from neurospatial.decoding import (
    BayesianDecoder,
    DecodingResult,
    DecodingSummary,
    decode_session,
)
from neurospatial.decoding.session import _build_encoding_model
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
    env.units = "cm"

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
        rates = cell.firing_rate(positions, times)
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

    env = Environment.linear_track(endpoints=[(0.0, 0.0), (100.0, 0.0)], bin_size=5.0)
    env.units = "cm"

    times = np.linspace(0.0, 20.0, 1000)
    x = 50.0 + 50.0 * np.sin(2 * np.pi * times / 20.0)
    positions = np.column_stack([x, np.zeros_like(x)])

    rng = np.random.default_rng(seed)
    centers = np.column_stack([np.linspace(5.0, 95.0, 6), np.zeros(6)])
    spikes: list[np.ndarray] = []
    for i in range(6):
        cell = PlaceCellModel(env, center=centers[i], width=15.0, max_rate=30.0, seed=i)
        rates = cell.firing_rate(positions, times)
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

    def test_map_estimates_byte_equal(self, sim) -> None:
        """MAP position and MAP bin index match decode_session exactly."""
        env, spikes, times, positions = sim

        dec = (
            BayesianDecoder(env, dt=0.5, bandwidth=5.0)
            .fit(spikes, times, positions)
            .predict(spikes, times)
        )
        ref = decode_session(env, spikes, times, positions, dt=0.5, bandwidth=5.0)

        assert_array_equal(dec.map_position, ref.map_position)
        assert_array_equal(dec.map_estimate, ref.map_estimate)

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
    assert_array_equal(summary.map_estimate, dense.map_estimate)
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


def test_epoch_restricts_encoding(sim) -> None:
    """fit(epoch=train) builds models from restricted train data only."""
    env, spikes, times, positions = sim

    t_mid = float(times[len(times) // 2])
    train_epoch = (float(times[0]), t_mid)
    test_epoch = (t_mid, float(times[-1]))

    fit = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions, epoch=train_epoch)

    # Reference encoding models built from the restricted train data, via the
    # SAME restrict + _build_encoding_model path fit uses internally.
    t_train, pos_train = restrict(times, positions, epochs=train_epoch)
    trains_train = restrict_spike_trains(spikes, train_epoch)
    models = _build_encoding_model(
        env,
        trains_train,
        t_train,
        pos_train,
        dt=0.5,
        bandwidth=5.0,
        smoothing_method="diffusion_kde",
        min_occupancy=0.0,
        max_gap=0.5,
        encoding_models=None,
        warn_on_drop=True,
        dtype=np.float64,
    )[1]

    assert_array_equal(fit.encoding_models, models)

    # Predict on the held-out test slice equals decode_session with those models.
    t_test, _pos_test = restrict(times, positions, epochs=test_epoch)
    trains_test = restrict_spike_trains(spikes, test_epoch)
    pred = fit.predict(trains_test, t_test)
    ref = decode_session(env, trains_test, t_test, encoding_models=models, dt=0.5)
    assert_array_equal(pred.posterior, ref.posterior)

    # Restricting to train changes the models vs the full-session fit.
    full = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
    assert not np.array_equal(fit.encoding_models, full.encoding_models)


def test_epoch_with_speed_restricts_speed(sim) -> None:
    """fit(epoch=..., speed=...) slices the time-aligned speed by the epoch too.

    Regression: the epoch branch restricted times/positions/trains but passed
    the full-length ``speed`` straight to the encoder, raising a length
    mismatch. ``speed`` is aligned to ``times`` and must be sliced identically.
    """
    env, spikes, times, positions = sim
    t_mid = float(times[len(times) // 2])
    train_epoch = (float(times[0]), t_mid)

    # Full-length speed aligned to `times` (a plausible per-sample speed track).
    speed = np.full(times.shape[0], 20.0, dtype=np.float64)

    fit = BayesianDecoder(env, dt=0.5).fit(
        spikes, times, positions, epoch=train_epoch, speed=speed, min_speed=5.0
    )
    assert fit.is_fitted

    # Equivalent to hand-restricting all three (times, positions, speed) first.
    t_tr, pos_tr, speed_tr = restrict(times, positions, speed, epochs=train_epoch)
    ref_models = _build_encoding_model(
        env,
        restrict_spike_trains(spikes, train_epoch),
        t_tr,
        pos_tr,
        dt=0.5,
        bandwidth=5.0,
        smoothing_method="diffusion_kde",
        min_occupancy=0.0,
        speed=speed_tr,
        min_speed=5.0,
        max_gap=0.5,
        encoding_models=None,
        warn_on_drop=True,
        dtype=np.float64,
    )[1]
    assert_array_equal(fit.encoding_models, ref_models)


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
# 8. Linearized-track differentiator (smoke)
# ---------------------------------------------------------------------------


def test_linearized_track_smoke() -> None:
    """fit + predict runs on a 1-D linearized track env (env-based decode)."""
    env, spikes, times, positions = _make_linear_sim()
    assert env.is_linearized_track

    fit = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
    result = fit.predict(spikes, times)

    assert isinstance(result, DecodingResult)
    assert result.posterior.shape[1] == env.n_bins
    assert result.posterior.shape[0] == result.times.shape[0]


# ---------------------------------------------------------------------------
# 9. score: undecodable-bin handling + distance= (FIX 1)
# ---------------------------------------------------------------------------


def _fitted_minimal(env: Environment | None = None) -> BayesianDecoder:
    """A fitted decoder built by directly injecting valid fitted state."""
    env = env if env is not None else _small_env()
    models = np.ones((2, env.n_bins))
    return BayesianDecoder(env, encoding_models=models, unit_ids=np.arange(2))


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
        monkeypatch.setattr(BayesianDecoder, "predict", lambda self, s, t: result)

        with pytest.warns(UserWarning, match="undecodable"):
            score = dec.score([np.array([0.1])] * 2, self._TRUE_TIMES, self._TRUE_POS)

        errors = result.error_against(self._TRUE_TIMES, self._TRUE_POS)
        assert np.isnan(errors[1])  # the undecodable row is dropped by nanmedian
        assert score == pytest.approx(float(np.nanmedian(errors)))
        assert np.isfinite(score)

    def test_all_undecodable_raises_not_nan(self, monkeypatch):
        dec = _fitted_minimal()
        result = _result_with_nan_rows(dec.env, nan_rows=[0, 1, 2, 3])
        monkeypatch.setattr(BayesianDecoder, "predict", lambda self, s, t: result)

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

    def test_encoding_models_wrong_nbins_raises(self):
        env = _small_env()
        bad = np.zeros((3, env.n_bins + 1))  # wrong bin axis
        with pytest.raises(ValueError, match="bins"):
            BayesianDecoder(env, encoding_models=bad, unit_ids=np.array([0, 1, 2]))

    def test_encoding_models_without_unit_ids_raises(self):
        env = _small_env()
        models = np.ones((2, env.n_bins))
        with pytest.raises(ValueError, match="unit_ids"):
            BayesianDecoder(env, encoding_models=models)

    def test_encoding_models_unit_count_mismatch_raises(self):
        env = _small_env()
        models = np.ones((2, env.n_bins))
        with pytest.raises(ValueError, match="unit"):
            BayesianDecoder(env, encoding_models=models, unit_ids=np.array([0, 1, 2]))

    def test_unfitted_construct_ok(self):
        env = _small_env()
        dec = BayesianDecoder(env)
        assert dec.encoding_models is None

    def test_real_fit_construct_ok(self, sim):
        env, spikes, times, positions = sim
        fit = BayesianDecoder(env, dt=0.5).fit(spikes, times, positions)
        assert fit.encoding_models is not None
        assert fit.encoding_models.shape[1] == env.n_bins


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
# 13. Error provenance for fit(epoch=...) (FIX 5)
# ---------------------------------------------------------------------------


def test_fit_epoch_too_small_names_bayesian_decoder(sim):
    env, spikes, times, positions = sim
    empty_epoch = (float(times[0]) - 5.0, float(times[0]) - 1.0)  # selects 0 samples
    with pytest.raises(ValueError, match=r"BayesianDecoder\.fit"):
        BayesianDecoder(env, dt=0.5).fit(spikes, times, positions, epoch=empty_epoch)
