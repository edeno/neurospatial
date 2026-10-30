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
            replay_method=dataclasses.replace(DAVIDSON_2009, n_shuffles=199),
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
            replay_method=method,
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
                    replay_method=method,
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
        assert detect_replay(env, posterior, times, replay_method=SILVA_2015).is_replay
        jumpy = posterior.copy()
        jumpy[10:] = np.roll(posterior[10:], 30, axis=1)  # a 120 cm jump
        result = detect_replay(env, jumpy, times, replay_method=SILVA_2015)
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
            env, polluted, sweep.times, replay_method=method, spike_counts=silent, rng=0
        )
        by_hand = detect_replay(
            env,
            sweep.posterior[kept],
            sweep.times[kept],
            replay_method=dataclasses.replace(
                method, empty_time_bins="keep", criteria=()
            ),
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
            replay_method=_fast(TIROLE_2022),
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
                detect_replay(env, posterior, np.arange(5) * DT, replay_method=method)

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
            replay_methods=[_fast(SILVA_2015), _fast(MICHON_2019)],
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
            compare_replay_methods(env, [decoded, untimed], replay_methods=[SILVA_2015])


def test_a_method_without_a_score_cannot_test_one():
    with pytest.raises(ValueError, match="without a score"):
        ReplayMethod(name="x", reference="y", score=None, nulls=("column_cycle",))


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (
            DAVIDSON_2009,
            {
                "nulls": ("column_cycle", "unit_identity", "pseudo_event"),
                "n_shuffles": 1500,
                "alpha": 0.01,
                "empty_time_bins": "keep",
            },
        ),
        (
            KARLSSON_FRANK_2009,
            {
                "nulls": ("time_bin_permutation",),
                "n_shuffles": 10_000,
                "alpha": 0.05,
                "empty_time_bins": "drop",
            },
        ),
        (
            WU_2017,
            {
                "nulls": ("time_bin_permutation",),
                "n_shuffles": 1000,
                "alpha": 0.05,
                "empty_time_bins": "drop",
            },
        ),
        (SILVA_2015, {"nulls": (), "empty_time_bins": "keep"}),
        (
            MICHON_2019,
            {"nulls": ("pseudo_event",), "n_shuffles": 500, "alpha": 0.01},
        ),
        (
            TIROLE_2022,
            {
                "nulls": (
                    "spike_train_circular",
                    "place_field_circular",
                    "column_cycle",
                ),
                "n_shuffles": 1000,
                "alpha": 0.05,
                "use_absolute": True,
                "empty_time_bins": "zero",
            },
        ),
    ],
    ids=lambda value: getattr(value, "name", ""),
)
def test_recipe_parameters_match_the_papers(method, expected):
    for name, value in expected.items():
        assert getattr(method, name) == value, name


class TestMethodValidation:
    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("nulls", ("column_cycle", "time_reversal"), "unknown nulls"),
            ("alternative", "up", "alternative"),
            ("alpha", 1.5, "alpha"),
            ("n_shuffles", 0, "n_shuffles"),
        ],
    )
    def test_invalid_fields_are_refused(self, field, value, match):
        with pytest.raises(ValueError, match=match):
            ReplayMethod(
                name="x", reference="y", score=weighted_correlation, **{field: value}
            )


class TestDetection:
    def test_result_pickles_with_read_only_mappings(self, place_code):
        env, _ = place_code
        posterior = np.zeros((20, env.n_bins))
        posterior[np.arange(20), np.arange(20)] = 1.0
        result = detect_replay(
            env, posterior, np.arange(20) * DT, replay_method=SILVA_2015
        )
        for clone in (pickle.loads(pickle.dumps(result)), copy.deepcopy(result)):
            np.testing.assert_equal(clone.summary(), result.summary())
            assert dict(clone.values) == dict(result.values)
            with pytest.raises(TypeError):
                clone.criteria["x"] = True  # type: ignore[index]

    def test_active_cell_criterion(self, place_code):
        env, _ = place_code
        posterior = np.zeros((20, env.n_bins))
        posterior[np.arange(20), np.arange(20)] = 1.0
        method = ReplayMethod(
            name="x",
            reference="y",
            score=weighted_correlation,
            criteria=(ReplayCriterion("n_active_cells", ">=", 5),),
        )
        counts = np.zeros((20, 8), dtype=int)
        counts[:, :3] = 1
        few = detect_replay(
            env,
            posterior,
            np.arange(20) * DT,
            replay_method=method,
            spike_counts=counts,
        )
        assert not few.is_replay
        assert few.values["n_active_cells >= 5"] == 3.0
        counts[:, :5] = 1
        assert detect_replay(
            env,
            posterior,
            np.arange(20) * DT,
            replay_method=method,
            spike_counts=counts,
        ).is_replay

    def test_zeroing_with_posterior_only_nulls(self, place_code):
        env, _ = place_code
        posterior = np.zeros((20, env.n_bins))
        posterior[np.arange(20), np.arange(20)] = 1.0
        counts = np.ones((20, 4), dtype=int)
        counts[3] = 0
        times = np.arange(20) * DT
        method = ReplayMethod(
            name="x",
            reference="y",
            score=weighted_correlation,
            score_options={"row_normalization": "none"},
            nulls=("column_cycle",),
            n_shuffles=99,
            empty_time_bins="zero",
        )
        result = detect_replay(
            env, posterior, times, replay_method=method, spike_counts=counts, rng=0
        )
        zeroed = posterior.copy()
        zeroed[3] = 0.0
        expected = weighted_correlation(
            env, zeroed, times, row_normalization="none"
        ).score
        assert result.score == pytest.approx(expected)


class TestJointDecode:
    @pytest.fixture(scope="class")
    def two_tracks(self, place_code):
        from neurospatial.decoding import decode_jointly

        env_a, rates_a = place_code
        env_b = Environment.linear_track(
            endpoints=[[0.0, 0.0], [160.0, 0.0]], bin_size=4.0, units="cm"
        )
        x = env_b.bin_centers[:, 0]
        centers = np.random.default_rng(0).permutation(np.linspace(0.0, 160.0, 40))
        rates_b = 0.5 + 25.0 * np.exp(-0.5 * ((x - centers[:, None]) / 10.0) ** 2)
        envs = {"a": env_a, "b": env_b}
        models = {"a": rates_a, "b": rates_b}
        event = simulate_replay_event(
            env_a, rates_a, kind="sweep", rate_gain=4.0, seed=9, speed=900.0
        )
        counts, grid = bin_spikes_in_time(
            list(event.spike_times), dt=DT, t_start=0.0, t_stop=event.duration
        )
        share = decode_jointly(envs, counts, models, dt=DT).joint("a")
        return envs, models, counts, grid, share

    def test_tirole_on_one_tracks_share_of_a_joint_decode(self, two_tracks):
        envs, models, counts, grid, share = two_tracks
        result = detect_replay(
            envs,
            share,
            grid,
            replay_method=_fast(TIROLE_2022),
            spike_counts=counts,
            encoding_models=models,
            dt=DT,
            template="a",
            rng=0,
        )
        assert result.is_replay, result.summary()

    def test_one_tracks_rate_maps_do_not_reproduce_a_joint_share(self, two_tracks):
        envs, models, counts, grid, share = two_tracks
        with pytest.raises(ValueError, match="does not reproduce posterior"):
            detect_replay(
                envs["a"],
                share,
                grid,
                replay_method=_fast(TIROLE_2022),
                spike_counts=counts,
                encoding_models=models["a"],
                dt=DT,
                rng=0,
            )

    def test_a_mapping_of_environments_needs_a_template(self, two_tracks):
        envs, _, _, grid, share = two_tracks
        with pytest.raises(ValueError, match="needs template"):
            detect_replay(envs, share, grid, replay_method=SILVA_2015)


class TestCompareStreamsAndPool:
    @pytest.fixture(scope="class")
    def events(self, place_code):
        env, rates = place_code
        events = [
            _decoded(env, rates, "sweep", seed=10, speed=900.0)[0],
            _decoded(env, rates, "random", seed=11)[0],
            _decoded(env, rates, "random", seed=12)[0],
        ]
        return env, events

    def test_pseudo_event_pool_is_every_event(self, events, monkeypatch):
        import neurospatial.replay.methods as methods_module

        env, decoded = events
        pools = []
        detect = methods_module.detect_replay

        def spy(*args, **kwargs):
            pools.append(kwargs["pseudo_event_pool"])
            return detect(*args, **kwargs)

        monkeypatch.setattr(methods_module, "detect_replay", spy)
        compare_replay_methods(env, decoded, replay_methods=[_fast(MICHON_2019)], rng=0)
        expected = np.vstack([event.posterior for event in decoded])
        assert len(pools) == len(decoded)
        for pool in pools:
            np.testing.assert_array_equal(pool, expected)

    def test_reordering_methods_does_not_change_results(self, events):
        import dataclasses

        env, decoded = events
        cycle = dataclasses.replace(
            SILVA_2015, name="silva_cycle", nulls=("column_cycle",), n_shuffles=99
        )
        michon = _fast(MICHON_2019)
        forward = compare_replay_methods(
            env, decoded, replay_methods=[cycle, michon], rng=0
        )
        backward = compare_replay_methods(
            env, decoded, replay_methods=[michon, cycle], rng=0
        )
        key = ["event", "method"]
        forward = forward.sort_values(key).reset_index(drop=True)
        backward = backward.sort_values(key).reset_index(drop=True)[forward.columns]
        assert forward.equals(backward)

    def test_an_event_too_short_to_score_gets_a_row(self, place_code, events):
        env, rates = place_code
        _, decoded = events
        short = decode_position(
            env, np.ones((1, rates.shape[0]), dtype=int), rates, dt=DT, times=[0.0]
        )
        table = compare_replay_methods(
            env, [decoded[0], short], replay_methods=[SILVA_2015]
        )
        row = table[table["event"] == 1].iloc[0]
        assert not row["is_replay"] and np.isnan(row["score"])
        assert row["error"] == "fewer than two time bins to score"
        assert table[table["event"] == 0].iloc[0]["error"] == ""
