"""Tests for the published replay method recipes."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import bin_spikes_in_time, decode_position
from neurospatial.replay import (
    ReplayCriterion,
    ReplayMethod,
    compare_replay_methods,
    detect_replay,
    weighted_correlation,
)
from neurospatial.replay.methods import (
    DAVIDSON_2009,
    KARLSSON_FRANK_2009,
    MICHON_2019,
    PUBLISHED_METHODS,
    SILVA_2015,
    TIROLE_2022,
    WU_2017,
)
from neurospatial.simulation import simulate_replay_event

DT = 0.02


@pytest.fixture(scope="module")
def place_code():
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=4.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    rates = 0.5 + 25.0 * np.exp(
        -0.5 * ((x - np.linspace(0.0, 200.0, 40)[:, None]) / 10.0) ** 2
    )
    return env, rates


def _decoded(env, rates, kind, seed, **kwargs):
    event = simulate_replay_event(
        env, rates, kind=kind, rate_gain=4.0, seed=seed, **kwargs
    )
    counts, grid = bin_spikes_in_time(
        list(event.spike_times), dt=DT, t_start=0.0, t_stop=event.duration
    )
    return decode_position(env, counts, rates, dt=DT, times=grid), counts


def _fast(method):
    """The method with few shuffles, for tests."""
    import dataclasses

    return dataclasses.replace(method, n_shuffles=199)


class TestCriterion:
    @pytest.mark.parametrize(
        ("comparison", "value", "expected"),
        [("<", 0.39, True), ("<", 0.4, False), ("<=", 0.4, True), (">", 0.4, False)],
    )
    def test_comparisons(self, comparison, value, expected):
        assert ReplayCriterion("q", comparison, 0.4).evaluate(value, np.nan) is expected

    def test_relative_to_track_length_and_absolute(self):
        jump = ReplayCriterion("max_map_jump", "<", 0.4, relative_to="track_length")
        assert jump.evaluate(79.0, 200.0) and not jump.evaluate(81.0, 200.0)
        correlation = ReplayCriterion("score", ">", 0.6, absolute=True)
        assert correlation.evaluate(-0.7, np.nan)
        assert correlation.label == "|score| > 0.6"

    def test_nan_fails(self):
        assert not ReplayCriterion("q", ">", 0.0).evaluate(np.nan, 1.0)

    def test_unknown_comparison(self):
        with pytest.raises(ValueError, match="comparison"):
            ReplayCriterion("q", "==", 1.0)  # type: ignore[arg-type]


class TestPublishedMethods:
    def test_every_recipe_names_its_paper_and_notes(self):
        names = [method.name for method in PUBLISHED_METHODS]
        assert len(set(names)) == len(names)
        for method in PUBLISHED_METHODS:
            assert method.reference and method.notes

    def test_davidson_detects_a_planted_sweep(self, place_code):
        env, rates = place_code
        sweep, counts = _decoded(env, rates, "sweep", seed=1, speed=900.0)
        noise = [
            _decoded(env, rates, "random", seed=50 + s)[0].posterior for s in range(5)
        ]
        import dataclasses

        result = detect_replay(
            env,
            sweep.posterior,
            sweep.times,
            method=dataclasses.replace(DAVIDSON_2009, n_shuffles=199),
            spike_counts=counts,
            encoding_models=rates,
            dt=DT,
            pseudo_event_pool=np.vstack(noise),
            rng=0,
        )
        assert result.is_replay
        assert result.p_value == pytest.approx(1 / 200)

    @pytest.mark.parametrize(
        "method", [KARLSSON_FRANK_2009, WU_2017, TIROLE_2022], ids=lambda m: m.name
    )
    def test_spike_methods_detect_a_sweep_and_reject_a_random_event(
        self, place_code, method
    ):
        env, rates = place_code
        method = _fast(method)
        if method.name == "karlsson_frank_2009":
            import dataclasses

            method = dataclasses.replace(
                method,
                score_options={"method": "sample", "n_samples_per_time_bin": 200},
            )
        sweep, counts = _decoded(env, rates, "sweep", seed=2, speed=900.0)
        result = detect_replay(
            env,
            sweep.posterior,
            sweep.times,
            method=method,
            spike_counts=counts,
            encoding_models=rates,
            dt=DT,
            rng=0,
        )
        assert result.is_replay, result.summary()
        verdicts = []
        for seed in range(6):
            noise, noise_counts = _decoded(env, rates, "random", seed=200 + seed)
            verdicts.append(
                detect_replay(
                    env,
                    noise.posterior,
                    noise.times,
                    method=method,
                    spike_counts=noise_counts,
                    encoding_models=rates,
                    dt=DT,
                    rng=seed,
                ).is_replay
            )
        assert sum(verdicts) <= 1

    def test_silva_thresholds_correlation_and_jump(self, place_code):
        env, _ = place_code
        posterior = np.zeros((20, env.n_bins))
        posterior[np.arange(20), np.arange(20)] = 1.0
        times = np.arange(20) * DT
        assert detect_replay(env, posterior, times, method=SILVA_2015).is_replay
        jumpy = posterior.copy()
        jumpy[10:] = np.roll(posterior[10:], 30, axis=1)  # a 120 cm jump
        result = detect_replay(env, jumpy, times, method=SILVA_2015)
        assert not result.is_replay
        assert not result.criteria["max_map_jump < 0.4 x track_length"]

    def test_karlsson_drops_time_bins_without_spikes(self, place_code):
        env, rates = place_code
        sweep, counts = _decoded(env, rates, "sweep", seed=3, speed=900.0)
        silent = counts.copy()
        silent[4] = 0  # this time bin is now empty: its row must be ignored
        polluted = sweep.posterior.copy()
        polluted[4] = 1.0 / env.n_bins
        import dataclasses

        method = dataclasses.replace(
            _fast(KARLSSON_FRANK_2009),
            score_options={"method": "sample", "n_samples_per_time_bin": 100},
        )
        kept = np.ones(len(counts), dtype=bool)
        kept[4] = False
        dropped = detect_replay(
            env, polluted, sweep.times, method=method, spike_counts=silent, rng=0
        )
        by_hand = detect_replay(
            env,
            sweep.posterior[kept],
            sweep.times[kept],
            method=dataclasses.replace(method, empty_time_bins="keep", criteria=()),
            rng=0,
        )
        assert dropped.score == by_hand.score

    def test_tirole_zeroes_empty_time_bins_before_scoring(self, place_code):
        env, rates = place_code
        sweep, counts = _decoded(env, rates, "sweep", seed=4, speed=900.0)
        silent = counts.copy()
        silent[3] = 0
        posterior = decode_position(env, silent, rates, dt=DT).posterior
        zeroed = posterior.copy()
        zeroed[3] = 0.0
        expected = abs(
            weighted_correlation(
                env, zeroed, sweep.times, row_normalization="none"
            ).score
        )
        result = detect_replay(
            env,
            posterior,
            sweep.times,
            method=_fast(TIROLE_2022),
            spike_counts=silent,
            encoding_models=rates,
            dt=DT,
            rng=0,
        )
        assert abs(result.score) == pytest.approx(expected)

    def test_methods_that_need_counts_say_so(self, place_code):
        env, _ = place_code
        posterior = np.full((5, env.n_bins), 1.0 / env.n_bins)
        for method in (KARLSSON_FRANK_2009, WU_2017, TIROLE_2022):
            with pytest.raises(ValueError, match="needs spike_counts"):
                detect_replay(env, posterior, np.arange(5) * DT, method=method)

    def test_recipes_pickle(self):
        for method in PUBLISHED_METHODS:
            for clone in (pickle.loads(pickle.dumps(method)), copy.deepcopy(method)):
                assert clone == method


class TestCompare:
    def test_table_has_one_row_per_event_and_method(self, place_code):
        env, rates = place_code
        events, counts = [], []
        for kind, seed in (("sweep", 5), ("random", 6), ("random", 7)):
            decoded, event_counts = _decoded(
                env,
                rates,
                kind,
                seed=seed,
                **({"speed": 900.0} if kind == "sweep" else {}),
            )
            events.append(decoded)
            counts.append(event_counts)
        table = compare_replay_methods(
            env,
            events,
            methods=[_fast(SILVA_2015), _fast(MICHON_2019)],
            spike_counts=counts,
            rng=0,
        )
        assert len(table) == 6
        assert set(table["method"]) == {"silva_2015", "michon_2019"}
        sweep_rows = table[table["event"] == 0]
        assert sweep_rows["is_replay"].all()

    def test_events_need_times(self, place_code):
        env, rates = place_code
        decoded, _ = _decoded(env, rates, "random", seed=8)
        untimed = decode_position(
            env, np.zeros((3, rates.shape[0]), dtype=int), rates, dt=DT
        )
        with pytest.raises(ValueError, match="has no times"):
            compare_replay_methods(env, [decoded, untimed], methods=[SILVA_2015])


def test_a_method_without_a_score_cannot_test_one():
    with pytest.raises(ValueError, match="without a score"):
        ReplayMethod(name="x", reference="y", score=None, nulls=("column_cycle",))
