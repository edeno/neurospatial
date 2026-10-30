"""Tests for replay_significance and the weighted-correlation score."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from replay_helpers import wrapped_bump

from neurospatial import Environment
from neurospatial.replay import (
    LinearFitResult,
    WeightedCorrelationResult,
    detect_trajectory_radon,
    fit_linear_trajectory,
    replay_significance,
    weighted_correlation,
)


def _track(length: float = 100.0, bin_size: float = 4.0) -> Environment:
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [length, 0.0]], bin_size=bin_size, units="cm"
    )


def _sweep(env: Environment, n_time: int, step: int = 1) -> np.ndarray:
    """A posterior that moves ``step`` bins along the track per time bin."""
    posterior = np.zeros((n_time, env.n_bins))
    posterior[np.arange(n_time), step * np.arange(n_time)] = 1.0
    return posterior


def _direct_correlation(weights, times, x):
    """The formula of Wu & Foster (2014), summed term by term."""
    pairs = [(i, j) for i in range(len(times)) for j in range(len(x))]
    w = weights / weights.sum()
    mean_t = sum(w[i, j] * times[i] for i, j in pairs)
    mean_x = sum(w[i, j] * x[j] for i, j in pairs)
    cov = sum(w[i, j] * (times[i] - mean_t) * (x[j] - mean_x) for i, j in pairs)
    var_t = sum(w[i, j] * (times[i] - mean_t) ** 2 for i, j in pairs)
    var_x = sum(w[i, j] * (x[j] - mean_x) ** 2 for i, j in pairs)
    return cov / np.sqrt(var_t * var_x)


class TestWeightedCorrelation:
    def test_matches_a_direct_double_sum(self):
        env = _track()
        rng = np.random.default_rng(0)
        posterior = rng.dirichlet(np.ones(env.n_bins) * 0.3, size=7)
        times = np.cumsum(rng.uniform(0.01, 0.03, 7))
        expected = _direct_correlation(posterior, times, env.bin_centers[:, 0])

        result = weighted_correlation(env, posterior, times)
        assert isinstance(result, WeightedCorrelationResult)
        assert result.score == pytest.approx(expected, rel=1e-12)

    def test_sign_follows_direction_along_the_track(self):
        env = _track()
        times = np.arange(10) * 0.02
        forward = _sweep(env, 10)
        assert weighted_correlation(env, forward, times).score == pytest.approx(1.0)
        assert weighted_correlation(env, forward[::-1], times).score == pytest.approx(
            -1.0
        )

    def test_evenly_spaced_times_match_the_bin_index(self):
        """Papers use the time-bin index; the correlation ignores the time scale."""
        env = _track()
        posterior = np.random.default_rng(1).dirichlet(np.ones(env.n_bins), size=6)
        by_time = weighted_correlation(env, posterior, 0.3 + 0.02 * np.arange(6))
        by_index = weighted_correlation(env, posterior, np.arange(6.0))
        assert by_time.score == pytest.approx(by_index.score, rel=1e-12)

    def test_row_normalization(self):
        """per_time_bin weighs every time bin alike; none keeps each row's mass."""
        env = _track()
        times = np.arange(4) * 0.02
        posterior = _sweep(env, 4)
        posterior[3] = 1.0 / env.n_bins  # a flat last time bin
        light = posterior.copy()
        light[3] *= 0.1  # e.g. a joint-decode row holding little mass

        x = env.bin_centers[:, 0]

        equal = weighted_correlation(env, light, times).score
        assert equal == pytest.approx(_direct_correlation(posterior, times, x))
        as_given = weighted_correlation(env, light, times, row_normalization="none")
        assert as_given.score == pytest.approx(_direct_correlation(light, times, x))
        assert as_given.score != pytest.approx(equal)

    def test_rows_that_sum_below_one_are_accepted(self):
        """One template's share of a joint decode."""
        env = _track()
        times = np.arange(10) * 0.02
        result = weighted_correlation(
            env, 0.3 * _sweep(env, 10), times, row_normalization="none"
        )
        assert result.score == pytest.approx(1.0)

    def test_stationary_posterior_is_undefined(self):
        env = _track()
        posterior = np.zeros((5, env.n_bins))
        posterior[:, 4] = 1.0
        assert np.isnan(weighted_correlation(env, posterior, np.arange(5) * 0.02).score)

    @pytest.mark.parametrize(
        ("make_posterior", "n_times", "match"),
        [
            (lambda env: np.full((3, env.n_bins), np.nan), 3, "NaN, Inf"),
            (lambda env: -np.ones((3, env.n_bins)), 3, "negative"),
            (lambda env: np.zeros((3, env.n_bins)), 3, "no mass"),
            (lambda env: np.ones((3, 2)), 3, "one column per"),
            (lambda env: np.ones((3, env.n_bins)), 4, "one finite time"),
        ],
    )
    def test_refusals(self, make_posterior, n_times, match):
        env = _track()
        with pytest.raises(ValueError, match=match):
            weighted_correlation(env, make_posterior(env), np.arange(float(n_times)))

    def test_unknown_row_normalization(self):
        env = _track()
        with pytest.raises(ValueError, match="row_normalization"):
            weighted_correlation(
                env,
                _sweep(env, 3),
                np.arange(3.0),
                row_normalization="x",  # type: ignore[arg-type]
            )

    def test_open_arena_is_refused(self):
        arena = Environment.from_samples(
            np.random.default_rng(0).uniform(0, 50, (500, 2)), bin_size=10.0
        )
        posterior = np.full((3, arena.n_bins), 1.0 / arena.n_bins)
        with pytest.raises(ValueError, match="linearized track"):
            weighted_correlation(arena, posterior, np.arange(3.0))


class TestReplaySignificanceAnyScore:
    def test_batched_scores_equal_scoring_each_shuffle(self):
        """The batched paths score shuffles exactly as one call each would."""
        from neurospatial.replay.significance import _scorer_for

        env = _track()
        times = np.arange(8) * 0.02
        rng = np.random.default_rng(2)
        posteriors = [
            rng.dirichlet(np.ones(env.n_bins) * 0.3, size=8) for _ in range(5)
        ]
        for score, options in (
            (weighted_correlation, {}),
            (weighted_correlation, {"row_normalization": "none"}),
            (detect_trajectory_radon, {"band": 10.0}),
        ):
            scorer = _scorer_for(score, env, posteriors[0], times, dict(options))
            batched = scorer.scores(iter(posteriors), len(posteriors))
            one_by_one = [score(env, p, times, **options).score for p in posteriors]
            assert_allclose(batched, one_by_one, rtol=1e-12)
            assert scorer.observe().score == pytest.approx(one_by_one[0], rel=1e-12)

    def test_weighted_correlation_sweep_is_significant(self):
        env = _track()
        times = np.arange(20) * 0.02
        result = replay_significance(
            env,
            _sweep(env, 20),
            times,
            score=weighted_correlation,
            n_shuffles=199,
            rng=0,
        )
        assert isinstance(result.observed, WeightedCorrelationResult)
        assert result.p_value == pytest.approx(1 / 200)
        assert result.summary()["score"] == pytest.approx(1.0)

    def test_reverse_sweep_counts_by_default(self):
        """The weighted correlation is tested as |r| unless told otherwise."""
        env = _track()
        times = np.arange(20) * 0.02
        reverse = _sweep(env, 20)[::-1]
        common = {"score": weighted_correlation, "n_shuffles": 199, "rng": 0}
        default = replay_significance(env, reverse, times, **common)
        assert default.use_absolute
        assert default.p_value == pytest.approx(1 / 200)
        signed = replay_significance(env, reverse, times, use_absolute=False, **common)
        assert signed.p_value > 0.9
        assert replay_significance(
            env, reverse, times, use_absolute=False, alternative="less", **common
        ).p_value == pytest.approx(1 / 200)
        assert not replay_significance(env, reverse, times, n_shuffles=9).use_absolute

    def test_two_sided(self):
        env = _track()
        times = np.arange(20) * 0.02
        result = replay_significance(
            env,
            _sweep(env, 20),
            times,
            score=weighted_correlation,
            use_absolute=False,
            alternative="two-sided",
            n_shuffles=199,
            rng=0,
        )
        assert result.p_value == pytest.approx(2 / 200)
        assert result.summary()["alternative"] == "two-sided"

    def test_weighted_correlation_ties_count(self):
        """Sharp posteriors on a coarse track tie often; ties raise the p-value.

        Without rounding, 41 of these 1485 shuffles tied with the event in
        exact arithmetic would score 1e-16 below it and p would be 0.7225.
        """
        env = _track(bin_size=10.0)
        posterior = np.zeros((4, env.n_bins))
        posterior[np.arange(4), [9, 6, 6, 8]] = 1.0
        result = replay_significance(
            env,
            posterior,
            np.arange(4) * 0.02,
            score=weighted_correlation,
            n_shuffles=2000,
            rng=7,
        )
        test = result.shuffles["column_cycle"]
        null = test.null_scores[np.isfinite(test.null_scores)]  # NaN: no variation
        at_least = int(np.sum(null >= test.observed_score))
        assert at_least == int(np.sum(null >= test.observed_score - 1e-12))
        assert result.p_value == pytest.approx((at_least + 1) / (null.size + 1))
        assert result.p_value == pytest.approx(0.743, abs=5e-4)

    def test_batches_larger_than_one_block(self):
        from neurospatial.replay.significance import _scorer_for

        env = _track()
        times = np.arange(6) * 0.02
        rng = np.random.default_rng(3)
        posteriors = [rng.dirichlet(np.ones(env.n_bins), size=6) for _ in range(300)]
        scorer = _scorer_for(weighted_correlation, env, posteriors[0], times, {})
        batched = scorer.scores(iter(posteriors), len(posteriors))
        one_by_one = [weighted_correlation(env, p, times).score for p in posteriors]
        assert_allclose(batched, one_by_one, rtol=1e-12)

    def test_partial_of_a_batched_score_stays_batched(self):
        import functools

        from neurospatial.replay.significance import _RadonScorer, _unwrap_partial

        env = _track()
        posterior = wrapped_bump(env, 20.0 + 6.0 * np.arange(8), 6.0)
        times = np.arange(8) * 0.02
        score, options = _unwrap_partial(
            functools.partial(detect_trajectory_radon, band=10.0), {}
        )
        assert score is detect_trajectory_radon and options == {"band": 10.0}
        from neurospatial.replay.significance import _scorer_for

        assert isinstance(
            _scorer_for(score, env, posterior, times, options), _RadonScorer
        )
        via_partial = replay_significance(
            env,
            posterior,
            times,
            score=functools.partial(detect_trajectory_radon, band=10.0),
            n_shuffles=30,
            rng=1,
        )
        via_options = replay_significance(
            env, posterior, times, score_options={"band": 10.0}, n_shuffles=30, rng=1
        )
        assert_allclose(
            via_partial.shuffles["column_cycle"].null_scores,
            via_options.shuffles["column_cycle"].null_scores,
        )
        with pytest.raises(TypeError, match="positional"):
            replay_significance(
                env,
                posterior,
                times,
                score=functools.partial(detect_trajectory_radon, env),
                n_shuffles=5,
            )

    def test_sampled_scores_are_reproducible_with_a_seed(self):
        env = _track()
        posterior = wrapped_bump(env, 20.0 + 6.0 * np.arange(8), 8.0)
        times = np.arange(8) * 0.02
        runs = [
            replay_significance(
                env,
                posterior,
                times,
                score=fit_linear_trajectory,
                score_options={"n_samples": 50},
                n_shuffles=10,
                rng=4,
            )
            for _ in range(2)
        ]
        assert runs[0].observed.score == runs[1].observed.score
        assert_allclose(
            runs[0].shuffles["column_cycle"].null_scores,
            runs[1].shuffles["column_cycle"].null_scores,
        )

    def test_any_function_with_a_score(self):
        env = _track()
        times = np.arange(20) * 0.02
        result = replay_significance(
            env,
            _sweep(env, 20),
            times,
            score=fit_linear_trajectory,
            score_options={"method": "map"},
            n_shuffles=49,
            rng=0,
        )
        assert isinstance(result.observed, LinearFitResult)
        assert result.observed.score == result.observed.r_squared
        assert result.p_value == pytest.approx(1 / 50)

    def test_a_score_without_score_attribute_is_refused(self):
        env = _track()
        with pytest.raises(TypeError, match="no score attribute"):
            replay_significance(
                env,
                _sweep(env, 5),
                np.arange(5.0),
                score=lambda *args: 1.0,
                n_shuffles=5,
            )

    def test_unknown_score_option_is_refused(self):
        env = _track()
        with pytest.raises(TypeError, match="detect_trajectory_radon does not take"):
            replay_significance(
                env, _sweep(env, 5), np.arange(5.0), score_options={"bandwidth": 3.0}
            )

    def test_undefined_observed_score_is_refused(self):
        env = _track()
        posterior = np.zeros((5, env.n_bins))
        posterior[:, 3] = 1.0
        with pytest.raises(ValueError, match="score is nan"):
            replay_significance(
                env,
                posterior,
                np.arange(5.0),
                score=weighted_correlation,
                n_shuffles=5,
            )

    def test_weighted_correlation_p_values_are_uniform_without_replay(self):
        """Independent wrapped bumps: P(p <= 0.1) is about 0.1."""
        env = _track()
        rng = np.random.default_rng(12)
        times = np.arange(8) * 0.02
        p_values = np.array(
            [
                replay_significance(
                    env,
                    wrapped_bump(env, rng.uniform(0, 100, len(times)), 6.0),
                    times,
                    score=weighted_correlation,
                    use_absolute=True,
                    n_shuffles=39,
                    rng=rng,
                ).p_value
                for _ in range(200)
            ]
        )
        # Under the null, P(p <= 0.1) = 0.1: 20 of 200 expected (sd 4.2).
        assert 8 <= np.sum(p_values <= 0.1) <= 32
        assert 0.4 < np.mean(p_values) < 0.6


class TestNullSelection:
    def test_one_null_by_name(self):
        env = _track()
        result = replay_significance(
            env,
            _sweep(env, 10),
            np.arange(10) * 0.02,
            nulls="column_cycle",
            n_shuffles=9,
        )
        assert set(result.shuffles) == {"column_cycle"}

    def test_result_compares_by_identity(self):
        env = _track()
        result = replay_significance(
            env, _sweep(env, 10), np.arange(10) * 0.02, n_shuffles=9
        )
        assert result == result
        assert hash(result) == hash(result)
        assert isinstance(result.shuffles["column_cycle"].observed_score, float)

    def test_nulls_run_only_when_asked(self):
        env = _track()
        result = replay_significance(
            env,
            _sweep(env, 10),
            np.arange(10) * 0.02,
            nulls=["column_cycle"],
            n_shuffles=9,
        )
        assert set(result.shuffles) == {"column_cycle"}

    def test_unknown_null(self):
        env = _track()
        with pytest.raises(ValueError, match="nulls must name"):
            replay_significance(
                env, _sweep(env, 5), np.arange(5.0), nulls=["time_swap"], n_shuffles=5
            )

    def test_requested_null_without_its_inputs(self):
        env = _track()
        with pytest.raises(ValueError, match="without their inputs"):
            replay_significance(
                env,
                _sweep(env, 5),
                np.arange(5.0),
                nulls=["pseudo_event"],
                n_shuffles=5,
            )

    def test_unknown_alternative(self):
        env = _track()
        with pytest.raises(ValueError, match="alternative"):
            replay_significance(
                env,
                _sweep(env, 5),
                np.arange(5.0),
                alternative="above",  # type: ignore[arg-type]
                n_shuffles=5,
            )


class TestReplaySignificance:
    """The replay line of Davidson et al. tested against their three shuffles."""

    @pytest.fixture(scope="class")
    def place_code(self):
        """30 place cells on a 200 cm track in 4 cm bins, and a replay sweep."""
        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=4.0, units="cm"
        )
        centers = env.bin_centers[:, 0]
        fields = np.linspace(0.0, 200.0, 30)
        rates = 0.5 + 25.0 * np.exp(
            -0.5 * ((centers[None] - fields[:, None]) / 12.0) ** 2
        )
        return env, rates

    def test_tirole_nulls_reject_a_planted_sweep(self, place_code):
        """Spike-train shift, place-field shift and column cycle (Tirole 2022)."""
        from neurospatial.decoding import decode_position

        env, rates = place_code
        dt = 0.02
        times = np.arange(15) * dt
        path = 30.0 + 500.0 * times
        true_bins = np.abs(env.bin_centers[:, 0][None] - path[:, None]).argmin(axis=1)
        counts = np.random.default_rng(7).poisson(rates[:, true_bins].T * dt * 2)
        posterior = decode_position(env, counts, rates, dt=dt).posterior

        result = replay_significance(
            env,
            posterior,
            times,
            score=weighted_correlation,
            nulls=["spike_train_circular", "place_field_circular", "column_cycle"],
            spike_counts=counts,
            encoding_models=rates,
            dt=dt,
            n_shuffles=199,
            rng=0,
        )
        assert set(result.shuffles) == {
            "spike_train_circular",
            "place_field_circular",
            "column_cycle",
        }
        assert result.p_value == pytest.approx(1 / 200)

    def test_default_nulls_stay_davidsons(self, place_code):
        from neurospatial.decoding import decode_position

        env, rates = place_code
        counts = np.random.default_rng(2).poisson(1.0, (10, rates.shape[0]))
        posterior = decode_position(env, counts, rates, dt=0.02).posterior
        result = replay_significance(
            env,
            posterior,
            np.arange(10) * 0.02,
            spike_counts=counts,
            encoding_models=rates,
            dt=0.02,
            n_shuffles=9,
            rng=0,
        )
        assert set(result.shuffles) == {"column_cycle", "unit_identity"}

    def test_tirole_nulls_together_are_calibrated_without_replay(self, place_code):
        """Spikes from random positions: all three nulls together keep alpha.

        One decoding null alone rejects such events too often (about 0.16 at
        a nominal 0.1), which is why Tirole et al. require all three.
        """
        from neurospatial.decoding import decode_position

        env, rates = place_code
        dt = 0.02
        times = np.arange(10) * dt
        position_rng = np.random.default_rng(13)
        spike_rng = np.random.default_rng(14)
        shuffle_rng = np.random.default_rng(15)
        p_values = []
        for _ in range(300):
            bins = position_rng.integers(0, env.n_bins, len(times))
            counts = spike_rng.poisson(rates[:, bins].T * dt * 2)
            posterior = decode_position(env, counts, rates, dt=dt).posterior
            p_values.append(
                replay_significance(
                    env,
                    posterior,
                    times,
                    score=weighted_correlation,
                    nulls=[
                        "spike_train_circular",
                        "place_field_circular",
                        "column_cycle",
                    ],
                    spike_counts=counts,
                    encoding_models=rates,
                    dt=dt,
                    n_shuffles=39,
                    rng=shuffle_rng,
                ).p_value
            )
        # 30 of 300 expected (sd 5.2); one null alone gives about 47.
        assert 16 <= np.sum(np.asarray(p_values) <= 0.1) <= 42

    def test_time_bin_permutation_is_calibrated_without_replay(self, place_code):
        """Independent time bins are exchangeable, so this null is exact."""
        env, _ = place_code
        rng = np.random.default_rng(16)
        times = np.arange(10) * 0.02
        p_values = np.array(
            [
                replay_significance(
                    env,
                    wrapped_bump(env, rng.uniform(0, 200, len(times)), 8.0),
                    times,
                    score=weighted_correlation,
                    nulls="time_bin_permutation",
                    n_shuffles=39,
                    rng=rng,
                ).p_value
                for _ in range(300)
            ]
        )
        assert 16 <= np.sum(p_values <= 0.1) <= 44

    def test_permuting_posterior_rows_equals_decoding_permuted_counts(self, place_code):
        from neurospatial.decoding import decode_position

        env, rates = place_code
        counts = np.random.default_rng(3).poisson(1.0, (12, rates.shape[0]))
        order = np.random.default_rng(4).permutation(12)
        posterior = decode_position(env, counts, rates, dt=0.02).posterior
        permuted = decode_position(env, counts[order], rates, dt=0.02).posterior
        np.testing.assert_allclose(posterior[order], permuted, atol=1e-12)

    def test_time_bin_permutation_needs_only_the_posterior(self, place_code):
        env, _ = place_code
        posterior = wrapped_bump(env, 20.0 + 8.0 * np.arange(12), 6.0)
        result = replay_significance(
            env,
            posterior,
            np.arange(12) * 0.02,
            score=weighted_correlation,
            nulls="time_bin_permutation",
            n_shuffles=199,
            rng=0,
        )
        assert set(result.shuffles) == {"time_bin_permutation"}
        assert result.p_value == pytest.approx(1 / 200)

    def test_place_field_shift_keeps_bins_without_training_data_out(self, place_code):
        from neurospatial.decoding import decode_position
        from neurospatial.replay import shuffle_place_fields_circular
        from neurospatial.replay.significance import _supported_rates

        env, rates = place_code
        partial = rates.copy()
        partial[:, :3] = np.nan  # the animal never reached the first 12 cm
        counts = np.random.default_rng(2).poisson(1.0, (10, rates.shape[0]))
        with pytest.warns(UserWarning, match="zero posterior probability"):
            posterior = decode_position(env, counts, partial, dt=0.02).posterior

        supported, support_prior = _supported_rates(env, partial, None)
        unshifted = decode_position(
            env, counts, np.nan_to_num(supported), dt=0.02, prior=support_prior
        ).posterior
        np.testing.assert_allclose(unshifted, posterior, atol=1e-12)
        for maps in shuffle_place_fields_circular(env, supported, n_shuffles=5, rng=0):
            assert np.all(np.isnan(maps[:, :3]))
            assert np.isfinite(maps[:, 3:]).all()

        with pytest.warns(UserWarning, match="zero posterior probability"):
            result = replay_significance(
                env,
                posterior,
                np.arange(10) * 0.02,
                nulls="place_field_circular",
                spike_counts=counts,
                encoding_models=partial,
                dt=0.02,
                n_shuffles=5,
                rng=0,
            )
        assert set(result.shuffles) == {"place_field_circular"}

    def test_partial_decoding_inputs_name_every_decoding_null(self, place_code):
        env, rates = place_code
        posterior = wrapped_bump(env, 20.0 + 8.0 * np.arange(6), 6.0)
        with pytest.raises(
            ValueError, match="spike_train_circular, place_field_circular"
        ):
            replay_significance(
                env,
                posterior,
                np.arange(6) * 0.02,
                nulls="spike_train_circular",
                spike_counts=np.ones((6, rates.shape[0]), dtype=int),
                encoding_models=rates,
                n_shuffles=5,
            )

    def test_decoding_nulls_need_their_inputs(self, place_code):
        env, _ = place_code
        posterior = wrapped_bump(env, 20.0 + 8.0 * np.arange(6), 6.0)
        for name in ("spike_train_circular", "place_field_circular"):
            with pytest.raises(ValueError, match="without their inputs"):
                replay_significance(
                    env, posterior, np.arange(6) * 0.02, nulls=name, n_shuffles=5
                )

    def test_weighted_correlation_under_every_shuffle(self, place_code):
        from neurospatial.decoding import decode_position

        env, rates = place_code
        dt = 0.02
        times = np.arange(15) * dt
        path = 170.0 - 500.0 * times  # a reverse sweep
        true_bins = np.abs(env.bin_centers[:, 0][None] - path[:, None]).argmin(axis=1)
        counts = np.random.default_rng(7).poisson(rates[:, true_bins].T * dt * 2)
        posterior = decode_position(env, counts, rates, dt=dt).posterior
        noise = np.random.default_rng(8).poisson(
            rates[:, np.random.default_rng(9).integers(0, env.n_bins, 300)].T * dt
        )
        pool = decode_position(env, noise, rates, dt=dt).posterior

        result = replay_significance(
            env,
            posterior,
            times,
            score=weighted_correlation,
            spike_counts=counts,
            encoding_models=rates,
            dt=dt,
            pseudo_event_pool=pool,
            n_shuffles=199,
            rng=0,
        )
        assert set(result.shuffles) == {"column_cycle", "unit_identity", "pseudo_event"}
        assert result.observed.score < -0.8
        assert result.p_value == pytest.approx(1 / 200)

    def test_planted_replay_is_significant_under_every_shuffle(self, place_code):
        from neurospatial.decoding import decode_position
        from neurospatial.replay import replay_significance

        env, rates = place_code
        dt = 0.02
        times = np.arange(15) * dt
        path = 30.0 + 500.0 * times  # cm, 500 cm/s
        true_bins = np.abs(env.bin_centers[:, 0][None] - path[:, None]).argmin(axis=1)
        counts = np.random.default_rng(7).poisson(rates[:, true_bins].T * dt * 2)
        posterior = decode_position(env, counts, rates, dt=dt).posterior
        noise = np.random.default_rng(8).poisson(
            rates[:, np.random.default_rng(9).integers(0, env.n_bins, 300)].T * dt
        )
        pool = decode_position(env, noise, rates, dt=dt).posterior

        result = replay_significance(
            env,
            posterior,
            times,
            spike_counts=counts,
            encoding_models=rates,
            dt=dt,
            pseudo_event_pool=pool,
            n_shuffles=199,
            rng=0,
        )

        assert set(result.shuffles) == {"column_cycle", "unit_identity", "pseudo_event"}
        assert result.p_value == pytest.approx(1 / 200)
        assert result.p_value == max(t.p_value for t in result.shuffles.values())
        assert result.observed.speed == pytest.approx(500.0, rel=0.1)
        for test in result.shuffles.values():
            assert test.n_shuffles == 199
            assert test.observed_score == result.observed.score
            assert np.all(test.null_scores < result.observed.score)
        summary = result.summary()
        assert (
            summary["unit_identity_p_value"] == result.shuffles["unit_identity"].p_value
        )

    def test_column_cycle_p_values_are_uniform_without_replay(self):
        """Events of independent random positions are significant at about alpha.

        Each row is a bump at a random position, wrapped around the track ends,
        so a circular shift of a row is as likely as the row itself.
        """
        from neurospatial.replay import replay_significance

        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=4.0, units="cm"
        )
        rng = np.random.default_rng(11)
        times = np.arange(8) * 0.02
        p_values = np.array(
            [
                replay_significance(
                    env,
                    wrapped_bump(env, rng.uniform(0, 100, len(times)), 6.0),
                    times,
                    n_shuffles=39,
                    rng=rng,
                ).p_value
                for _ in range(200)
            ]
        )

        # Under the null, P(p <= 0.1) = 0.1: 20 of 200 expected (sd 4.2).
        assert 8 <= np.sum(p_values <= 0.1) <= 32
        assert 0.4 < np.mean(p_values) < 0.6

    def test_shuffles_tied_with_the_observed_score_count_as_ties(self):
        """Scores equal up to rounding compare equal, so ties raise the p-value.

        Every row is a delta within the band of a stationary line, so the
        observed score and many shuffles' scores are exactly 1.
        """
        from neurospatial.replay import replay_significance

        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
        )
        times = np.arange(3) * 0.02
        posterior = np.zeros((3, env.n_bins))
        posterior[:, 10] = 1.0
        result = replay_significance(env, posterior, times, n_shuffles=200, rng=0)
        null = result.shuffles["column_cycle"].null_scores

        assert result.observed.score == 1.0
        assert null.max() <= 1.0
        tied = int(np.sum(null == 1.0))
        assert tied > 0
        assert result.p_value == pytest.approx((tied + 1) / 201)

    def test_seeded_results_repeat(self, place_code):
        from neurospatial.replay import replay_significance

        env, _ = place_code
        posterior = wrapped_bump(env, 20.0 + 10.0 * np.arange(10), 8.0)
        times = np.arange(10) * 0.02
        first = replay_significance(env, posterior, times, n_shuffles=50, rng=3)
        second = replay_significance(env, posterior, times, n_shuffles=50, rng=3)

        np.testing.assert_array_equal(
            first.shuffles["column_cycle"].null_scores,
            second.shuffles["column_cycle"].null_scores,
        )

    def test_unit_identity_needs_the_decode_of_the_posterior(self, place_code):
        from neurospatial.decoding import decode_position
        from neurospatial.replay import replay_significance

        env, rates = place_code
        counts = np.random.default_rng(2).poisson(1.0, (10, rates.shape[0]))
        posterior = decode_position(env, counts, rates, dt=0.02).posterior
        times = np.arange(10) * 0.02

        with pytest.raises(ValueError, match="does not reproduce posterior"):
            replay_significance(
                env,
                posterior,
                times,
                spike_counts=counts,
                encoding_models=rates,
                dt=0.05,
                n_shuffles=10,
            )
        with pytest.raises(ValueError, match="encoding_models, dt are missing"):
            replay_significance(
                env, posterior, times, spike_counts=counts, n_shuffles=10
            )

    def test_pseudo_event_pool_must_be_posterior_rows(self, place_code):
        from neurospatial.replay import replay_significance

        env, _ = place_code
        posterior = wrapped_bump(env, 20.0 + 10.0 * np.arange(10), 8.0)
        times = np.arange(10) * 0.02

        with pytest.raises(ValueError, match=r"pseudo_event_pool has shape \(4, 3\)"):
            replay_significance(
                env, posterior, times, pseudo_event_pool=np.ones((4, 3)), n_shuffles=10
            )
        with pytest.raises(ValueError, match="do not sum to 1"):
            replay_significance(
                env,
                posterior,
                times,
                pseudo_event_pool=np.ones((4, env.n_bins)),
                n_shuffles=10,
            )

    @pytest.mark.parametrize("n_shuffles", [0, -5, 2.5, True])
    def test_n_shuffles_must_be_a_positive_integer(self, place_code, n_shuffles):
        from neurospatial.replay import replay_significance

        env, _ = place_code
        posterior = wrapped_bump(env, 20.0 + 10.0 * np.arange(10), 8.0)
        with pytest.raises(ValueError, match="positive integer"):
            replay_significance(
                env, posterior, np.arange(10) * 0.02, n_shuffles=n_shuffles
            )


def test_spike_counts_in_score_options_are_refused_with_row_moving_nulls():
    """The counts would stay with the original rows while the null moves them."""
    from neurospatial.replay import fit_linear_trajectory

    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=4.0, units="cm"
    )
    posterior = np.zeros((10, env.n_bins))
    posterior[np.arange(10), 2 * np.arange(10)] = 1.0
    spikes = np.ones(10, dtype=int)
    with pytest.raises(ValueError, match="change which time bins have spikes"):
        replay_significance(
            env,
            posterior,
            np.arange(10) * 0.02,
            score=fit_linear_trajectory,
            score_options={"method": "map", "spike_counts": spikes},
            nulls="time_bin_permutation",
            n_shuffles=5,
        )
    # Column cycle keeps each row in its time bin, so it is allowed.
    result = replay_significance(
        env,
        posterior,
        np.arange(10) * 0.02,
        score=fit_linear_trajectory,
        score_options={"method": "map", "spike_counts": spikes},
        nulls="column_cycle",
        n_shuffles=5,
        rng=0,
    )
    assert set(result.shuffles) == {"column_cycle"}
