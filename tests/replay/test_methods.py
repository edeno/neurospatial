"""Tests for the published replay method recipes."""

from __future__ import annotations

import copy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import (
    DecodingResult,
    JointDecodingResult,
    bin_spikes_in_time,
    decode_position,
)
from neurospatial.replay import (
    RankOrderMethod,
    ReplayCriterion,
    ReplayMethod,
    ReplaySegmentation,
    compare_replay_methods,
    detect_rank_order_replay,
    detect_replay,
    weighted_correlation,
)
from neurospatial.replay.methods import (
    CAREY_2019,
    DAVIDSON_2009,
    DIBA_BUZSAKI_2007,
    FOSTER_WILSON_2006,
    IGATA_2021,
    KARLSSON_FRANK_2009,
    MALLORY_2025,
    MICHON_2019,
    PFEIFFER_FOSTER_2013,
    PFEIFFER_FOSTER_2015,
    PUBLISHED_METHODS,
    SILVA_2015,
    TIROLE_2022,
    WIDLOSKI_FOSTER_2022,
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


def _one_hot(env, bins):
    return np.eye(env.n_bins)[list(bins)]


def _steps(times_or_n, dt=0.005):
    return np.arange(times_or_n) * dt


def _with_segmentation(method, **changes):
    import dataclasses

    return dataclasses.replace(
        method, segmentation=dataclasses.replace(method.segmentation, **changes)
    )


class TestSegmentationRecipes:
    def test_pfeiffer_tests_the_longest_segment_only(self, place_code):
        """12 stationary bins, a jump, then 11 bins covering 40 cm.

        Pfeiffer & Foster (2013) truncate to the longest sequence, which
        travels nowhere, so the event fails; testing any segment, the
        sweep passes.
        """
        env, _ = place_code
        posterior = _one_hot(env, [10] * 12 + list(range(30, 41)))
        times = _steps(23)
        longest = detect_replay(
            env, posterior, times, replay_method=PFEIFFER_FOSTER_2013
        )
        assert not longest.is_replay
        assert longest.segments.rows(longest.segment) == slice(0, 12)
        assert longest.values["segment_start_to_end_distance >= 40"] == 0.0

        any_segment = detect_replay(
            env,
            posterior,
            times,
            replay_method=_with_segmentation(PFEIFFER_FOSTER_2013, choose="any"),
        )
        assert any_segment.is_replay
        assert any_segment.segments.rows(any_segment.segment) == slice(12, 23)
        assert any_segment.values[
            "segment_start_to_end_distance >= 40"
        ] == pytest.approx(40.0)

    def test_pfeiffer_2015_needs_80_cm(self, place_code):
        env, _ = place_code
        # 11 time bins either way; means move 8 cm (80 cm) or 4 cm (40 cm).
        passes = detect_replay(
            env,
            _one_hot(env, range(0, 22, 2)),
            _steps(11),
            replay_method=PFEIFFER_FOSTER_2015,
        )
        fails = detect_replay(
            env,
            _one_hot(env, range(11)),
            _steps(11),
            replay_method=PFEIFFER_FOSTER_2015,
        )
        assert passes.is_replay
        assert fails.segment == 0 and not fails.is_replay
        assert fails.values["segment_start_to_end_distance >= 80"] == pytest.approx(
            40.0
        )

    def test_pfeiffer_2015_follows_the_posterior_mean(self, place_code):
        """The MAP alternates between two bins 112 cm apart; the mean glides.

        Row t holds 0.52 in one of bins 2t and 2t + 28 and 0.48 in the
        other, so the mean moves about 8 cm per step and the MAP jumps.
        """
        env, _ = place_code
        posterior = np.zeros((11, env.n_bins))
        for t in range(11):
            near, far = (2 * t, 2 * t + 28) if t % 2 == 0 else (2 * t + 28, 2 * t)
            posterior[t, near], posterior[t, far] = 0.52, 0.48
        result = detect_replay(
            env, posterior, _steps(11), replay_method=PFEIFFER_FOSTER_2015
        )
        assert result.is_replay
        assert result.values["segment_start_to_end_distance >= 80"] == pytest.approx(
            80.0
        )
        by_map = _with_segmentation(
            PFEIFFER_FOSTER_2015,
            options={**PFEIFFER_FOSTER_2015.segmentation.options, "method": "map"},
        )
        assert not detect_replay(
            env, posterior, _steps(11), replay_method=by_map
        ).is_replay

    def test_carey_needs_50_ms(self, place_code):
        env, _ = place_code
        for n_time_bins, expected in ((10, False), (11, True)):
            result = detect_replay(
                env,
                _one_hot(env, range(n_time_bins)),
                _steps(n_time_bins),
                replay_method=CAREY_2019,
                spike_counts=np.ones((n_time_bins, 6), dtype=int),
            )
            assert result.is_replay is expected

    @pytest.mark.parametrize(("n_time_bins", "expected"), [(21, False), (22, True)])
    def test_widloski_duration_is_strictly_above_100_ms(
        self, place_code, n_time_bins, expected
    ):
        # 21 bins on 5 ms steps last exactly 100 ms, which the code rejects.
        env, _ = place_code
        result = detect_replay(
            env,
            _one_hot(env, range(n_time_bins)),
            _steps(n_time_bins),
            replay_method=WIDLOSKI_FOSTER_2022,
        )
        assert result.is_replay is expected
        assert result.criteria["segment_dispersion > 12"]

    def test_widloski_tests_any_segment_and_takes_the_speed_rule(self, place_code):
        """A stationary 29-bin piece and a dispersed 25-bin sweep.

        The longest segment is stationary; the event is replay because the
        sweep passes. Marking most of the sweep as running leaves two
        15 ms pieces 100 ms apart, too far to merge, so it is not.
        """
        env, _ = place_code
        posterior = _one_hot(env, [5] * 30 + list(range(20, 45)))
        times = _steps(55)
        result = detect_replay(
            env, posterior, times, replay_method=WIDLOSKI_FOSTER_2022
        )
        assert result.is_replay
        assert result.segments.n_segments == 2
        assert result.segments.rows(result.segment) == slice(30, 55)

        running = np.ones(55, dtype=bool)
        running[34:52] = False
        slowed = detect_replay(
            env,
            posterior,
            times,
            replay_method=WIDLOSKI_FOSTER_2022,
            valid_time_bins=running,
        )
        assert not slowed.is_replay

    def test_carey_needs_four_active_cells_per_time_bin(self, place_code):
        env, _ = place_code
        posterior = _one_hot(env, range(24))
        counts = np.ones((24, 6), dtype=int)
        with pytest.raises(ValueError, match="needs spike_counts"):
            detect_replay(env, posterior, _steps(24), replay_method=CAREY_2019)

        counts[9, 3:] = 0  # 3 active cells
        split = detect_replay(
            env, posterior, _steps(24), replay_method=CAREY_2019, spike_counts=counts
        )
        assert split.is_replay
        assert split.segments.segments.tolist() == [[10, 24]]

        counts[19, 3:] = 0
        broken = detect_replay(
            env, posterior, _steps(24), replay_method=CAREY_2019, spike_counts=counts
        )
        assert not broken.is_replay and broken.segment is None

    def test_igata_p_is_the_share_of_surrogates_with_a_trajectory(self, monkeypatch):
        """On a 10 m track, few shifted-field surrogates hold a trajectory.

        The test statistic is detection itself: each surrogate scores 1
        when it has a trajectory event, so p = (k + 1) / (n + 1) over the
        k surrogates that do.
        """
        import neurospatial.replay.significance as significance

        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [1000.0, 0.0]], bin_size=4.0, units="cm"
        )
        x = env.bin_centers[:, 0]
        rates = 0.5 + 25.0 * np.exp(
            -0.5 * ((x - np.linspace(0.0, 1000.0, 200)[:, None]) / 10.0) ** 2
        )
        sweep, counts = _decoded(env, rates, "sweep", seed=1, speed=900.0)
        tests = []
        original = significance.replay_significance

        def spy(*args, **kwargs):
            tests.append(original(*args, **kwargs))
            return tests[-1]

        monkeypatch.setattr(significance, "replay_significance", spy)
        result = detect_replay(
            env,
            sweep.posterior,
            sweep.times,
            replay_method=_fast(IGATA_2021),
            spike_counts=counts,
            encoding_models=rates,
            dt=DT,
            rng=1,
        )
        null = tests[0].shuffles["place_field_circular"].null_scores
        assert set(np.unique(null)) <= {0.0, 1.0}
        assert result.p_value == pytest.approx((null.sum() + 1) / (len(null) + 1))
        assert result.criteria["segment_path_length > 20"]
        assert result.is_replay and result.p_value < 0.05

    def test_an_undetected_event_has_p_one_without_shuffles(
        self, place_code, monkeypatch
    ):
        import neurospatial.replay.significance as significance

        def refuse(*args, **kwargs):
            raise AssertionError("no shuffles are needed")

        monkeypatch.setattr(significance, "replay_significance", refuse)
        env, rates = place_code
        posterior = np.full((10, env.n_bins), 1.0 / env.n_bins)
        result = detect_replay(
            env,
            posterior,
            np.arange(10) * DT,
            replay_method=IGATA_2021,
            spike_counts=np.ones((10, rates.shape[0]), dtype=int),
            encoding_models=rates,
            dt=DT,
        )
        assert result.p_value == 1.0 and not result.is_replay

    def test_score_and_criteria_apply_to_the_tested_segment(self, place_code):
        env, _ = place_code
        method = ReplayMethod(
            name="segment_then_correlate",
            reference="test",
            score=weighted_correlation,
            segmentation=ReplaySegmentation({"max_jump": 20.0, "min_time_bins": 5}),
            criteria=(ReplayCriterion("duration", "<", 1.0),),
        )
        bins = [45, 2, 40, 2, *range(10, 20)]
        posterior = _one_hot(env, bins)
        times = np.arange(len(bins)) * DT
        result = detect_replay(env, posterior, times, replay_method=method)
        assert result.segments.rows(result.segment) == slice(4, 14)
        assert result.score == pytest.approx(1.0)
        assert result.values["duration < 1"] == pytest.approx(9 * DT)
        whole = weighted_correlation(env, posterior, times).score
        assert whole < 0.9

    def test_thresholds_in_cm_need_an_environment_in_cm(self):
        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [2.0, 0.0]], bin_size=0.04, units="m"
        )
        with pytest.raises(ValueError, match="thresholds in 'cm'"):
            detect_replay(
                env,
                _one_hot(env, range(12)),
                _steps(12),
                replay_method=PFEIFFER_FOSTER_2013,
            )
        unitless = Environment.linear_track(
            endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=4.0
        )
        with pytest.raises(ValueError, match="has no units"):
            detect_replay(
                unitless,
                _one_hot(unitless, range(12)),
                _steps(12),
                replay_method=PFEIFFER_FOSTER_2013,
            )

    def test_empty_time_bins_are_invalid_rows_not_deleted(self, place_code):
        """Four empty bins decoded 160 cm away sit between two sweeps.

        Deleting them would join the sweeps into one segment; as invalid
        rows they split it, and the segments index the caller's rows.
        """
        env, _ = place_code
        posterior = _one_hot(env, [0, 1, 2, 3, 45, 45, 45, 45, 4, 5, 6, 7])
        counts = np.ones((12, 3), dtype=int)
        counts[4:8] = 0
        method = ReplayMethod(
            name="drop_then_segment",
            reference="test",
            score=None,
            empty_time_bins="drop",
            segmentation=ReplaySegmentation({"max_jump": 20.0}),
        )
        result = detect_replay(
            env, posterior, _steps(12), replay_method=method, spike_counts=counts
        )
        assert result.segments.n_time_bins == 12
        assert result.segments.segments.tolist() == [[0, 4], [8, 12]]

    def test_samples_per_time_bin_count_the_segment(self, place_code):
        env, _ = place_code
        seen = []

        def score(env, posterior, times, n_samples):
            seen.append((n_samples, len(posterior)))
            return SimpleNamespace(score=0.0)

        method = ReplayMethod(
            name="sampled",
            reference="test",
            score=score,
            score_options={"n_samples_per_time_bin": 10},
            segmentation=ReplaySegmentation({"max_jump": 20.0}),
        )
        posterior = _one_hot(env, [40, 41, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        detect_replay(env, posterior, _steps(11), replay_method=method)
        assert seen == [(90, 9)]

    def test_misaligned_spike_counts_are_refused(self, place_code):
        env, _ = place_code
        with pytest.raises(ValueError, match="spike_counts has 23 rows"):
            detect_replay(
                env,
                _one_hot(env, range(24)),
                _steps(24),
                replay_method=CAREY_2019,
                spike_counts=np.ones((23, 6), dtype=int),
            )

    def test_detection_test_refuses_missing_inputs_on_any_event(self, place_code):
        # A flat posterior holds no trajectory; the missing rate maps are
        # still refused, as they are for an event that holds one.
        env, _ = place_code
        with pytest.raises(ValueError, match="without their inputs"):
            detect_replay(
                env,
                np.full((10, env.n_bins), 1.0 / env.n_bins),
                np.arange(10) * DT,
                replay_method=IGATA_2021,
            )

    def test_detection_test_with_any_segment(self, place_code):
        """A stationary piece is longest; the sweep after it meets 20 cm."""
        env, _ = place_code
        method = ReplayMethod(
            name="any_detected",
            reference="test",
            score=None,
            nulls=("column_cycle",),
            n_shuffles=19,
            segmentation=ReplaySegmentation({"max_jump": 20.0}, choose="any"),
            criteria=(ReplayCriterion("segment_path_length", ">", 20.0),),
        )
        posterior = _one_hot(env, [10] * 12 + list(range(30, 38)))
        any_segment = detect_replay(
            env, posterior, _steps(20), replay_method=method, rng=0
        )
        longest = detect_replay(
            env,
            posterior,
            _steps(20),
            replay_method=_with_segmentation(method, choose="longest"),
            rng=0,
        )
        assert any_segment.segments.rows(any_segment.segment) == slice(12, 20)
        assert any_segment.p_value < 1.0
        assert longest.p_value == 1.0 and not longest.is_replay

    def test_chance_counts_every_template_of_a_joint_decode(self, place_code):
        """Peak mass 0.02 on one track of two: 1.5 x chance is 0.015."""
        env, rates = place_code
        method = ReplayMethod(
            name="chance",
            reference="test",
            score=None,
            segmentation=ReplaySegmentation(
                {"max_jump": 20.0, "min_peak_probability": 1.5},
                relative_to={"min_peak_probability": "chance"},
            ),
        )
        share = 0.02 * _one_hot(env, range(10))
        times = _steps(10)
        joint = detect_replay(
            {"a": env, "b": env}, share, times, replay_method=method, template="a"
        )
        alone = detect_replay(env, share, times, replay_method=method)
        assert joint.is_replay and not alone.is_replay
        both = detect_replay(
            env,
            share,
            times,
            replay_method=method,
            template="a",
            encoding_models={"a": rates, "b": rates},
        )
        assert both.is_replay
        with pytest.raises(ValueError, match="every template's rate maps"):
            detect_replay(env, share, times, replay_method=method, template="a")

    def test_methods_without_segmentation_ignore_valid_time_bins(self, place_code):
        env, _ = place_code
        posterior = _one_hot(env, range(20))
        times = np.arange(20) * DT
        plain = detect_replay(env, posterior, times, replay_method=SILVA_2015)
        masked = detect_replay(
            env,
            posterior,
            times,
            replay_method=SILVA_2015,
            valid_time_bins=np.zeros(20, dtype=bool),
        )
        np.testing.assert_equal(plain.summary(), masked.summary())
        assert "n_segments" not in plain.summary()

    def test_a_detection_test_refuses_valid_bins_with_row_moving_nulls(
        self, place_code
    ):
        env, _ = place_code
        method = ReplayMethod(
            name="x",
            reference="y",
            score=None,
            nulls=("time_bin_permutation",),
            segmentation=ReplaySegmentation({"max_jump": 20.0}),
        )
        with pytest.raises(ValueError, match="valid_time_bins cannot be used"):
            detect_replay(
                env,
                _one_hot(env, range(12)),
                _steps(12),
                replay_method=method,
                valid_time_bins=np.ones(12, dtype=bool),
            )

    def test_result_with_segments_pickles(self, place_code):
        env, _ = place_code
        result = detect_replay(
            env,
            _one_hot(env, range(14)),
            _steps(14),
            replay_method=PFEIFFER_FOSTER_2013,
        )
        clone = pickle.loads(pickle.dumps(result))
        np.testing.assert_equal(clone.summary(), result.summary())
        assert clone.segment == 0
        assert not clone.segments.segments.flags.writeable
        assert result.summary()["segment_end_time"] == pytest.approx(13 * 0.005)

    def test_compare_passes_each_events_valid_time_bins(self, place_code):
        env, rates = place_code
        events = [
            decode_position(
                env,
                np.zeros((22, rates.shape[0]), dtype=int),
                rates,
                dt=0.005,
                times=_steps(22),
            )
        ]
        # A flat posterior has no trajectory with or without the mask.
        table = compare_replay_methods(
            env,
            events,
            replay_methods=[WIDLOSKI_FOSTER_2022, SILVA_2015],
            valid_time_bins=[np.ones(22, dtype=bool)],
        )
        assert list(table["n_segments"].iloc[:1]) == [0]
        assert np.isnan(table["n_segments"].iloc[1])
        with pytest.raises(ValueError, match="valid_time_bins has 2 entries"):
            compare_replay_methods(
                env,
                events,
                replay_methods=[WIDLOSKI_FOSTER_2022],
                valid_time_bins=[np.ones(22, dtype=bool)] * 2,
            )


def test_criteria_treat_a_rounded_threshold_as_equal(place_code):
    """21 bins at t0 + 0.005 k last 100 ms up to rounding: not > 100 ms."""
    env, _ = place_code
    times = 100.0 + _steps(21)
    assert times[-1] - times[0] != 0.1
    result = detect_replay(
        env, _one_hot(env, range(21)), times, replay_method=WIDLOSKI_FOSTER_2022
    )
    assert not result.criteria["segment_duration > 0.1"]
    assert ReplayCriterion("q", ">=", 0.1).evaluate(0.1 - 1e-12, np.nan)


class TestSegmentationValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"options": {"max_jump": 20.0, "jump": 1}}, "does not take"),
            (
                {"options": {"max_jump": 20.0, "valid_time_bins": None}},
                "does not take",
            ),
            ({"options": {"min_time_bins": 5}}, "needs max_jump"),
            (
                {
                    "options": {"max_jump": 0.4},
                    "relative_to": {"min_duration": "chance"},
                },
                "not in options",
            ),
            (
                {"options": {"max_jump": 0.4}, "relative_to": {"max_jump": "arena"}},
                "relative_to values",
            ),
            ({"options": {"max_jump": 20.0}, "choose": "all"}, "choose"),
            (
                {"options": {"max_jump": 20.0}, "min_active_cells_per_time_bin": 0},
                "positive integer",
            ),
        ],
    )
    def test_invalid_segmentations_are_refused(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ReplaySegmentation(**kwargs)

    def test_relative_options_scale_with_the_environment(self, place_code):
        env, _ = place_code
        segmentation = ReplaySegmentation(
            {"max_jump": 0.4, "min_peak_probability": 5.0},
            relative_to={"max_jump": "track_length", "min_peak_probability": "chance"},
        )
        options = segmentation.resolved_options(env)
        assert options["max_jump"] == pytest.approx(0.4 * 200.0)
        assert options["min_peak_probability"] == pytest.approx(5.0 / env.n_bins)
        arena = Environment.from_samples(
            np.random.default_rng(0).uniform(0, 50, (500, 2)), bin_size=5.0, units="cm"
        )
        with pytest.raises(ValueError, match="track"):
            segmentation.resolved_options(arena)

    def test_segment_criteria_need_a_segmentation(self):
        with pytest.raises(ValueError, match="no segmentation"):
            ReplayMethod(
                name="x",
                reference="y",
                score=None,
                criteria=(ReplayCriterion("segment_path_length", ">", 20.0),),
            )

    def test_detection_tests_are_one_sided(self):
        with pytest.raises(ValueError, match="one-sided"):
            ReplayMethod(
                name="x",
                reference="y",
                score=None,
                nulls=("column_cycle",),
                alternative="two-sided",
                segmentation=ReplaySegmentation({"max_jump": 20.0}),
            )

    def test_active_cell_rule_refuses_nulls_that_change_spikes(self):
        with pytest.raises(ValueError, match="active-cell"):
            ReplayMethod(
                name="x",
                reference="y",
                score=None,
                nulls=("spike_train_circular",),
                segmentation=ReplaySegmentation(
                    {"max_jump": 20.0}, min_active_cells_per_time_bin=4
                ),
            )


@pytest.mark.parametrize(
    ("method", "options", "criteria", "choose"),
    [
        (
            PFEIFFER_FOSTER_2013,
            {"max_jump": 20.0, "method": "map", "min_time_bins": 11},
            ["segment_start_to_end_distance >= 40"],
            "longest",
        ),
        (
            PFEIFFER_FOSTER_2015,
            {"max_jump": 50.0, "method": "expected", "min_time_bins": 11},
            ["segment_start_to_end_distance >= 80"],
            "longest",
        ),
        (
            CAREY_2019,
            {"max_jump": 40.0, "method": "map", "min_time_bins": 10},
            ["segment_duration >= 0.05"],
            "any",
        ),
        (
            IGATA_2021,
            {
                "max_jump": 40.0,
                "method": "map",
                "max_gap_bins": 1,
                "min_peak_probability": 5.0,
                "min_time_bins": 5,
            },
            ["segment_path_length > 20"],
            "longest",
        ),
        (
            WIDLOSKI_FOSTER_2022,
            {
                "max_jump": 20.0,
                "method": "expected",
                "jump_rule": "drop_earlier",
                "max_spread": 10.0,
                "merge_max_time_gap": 0.05,
                "merge_max_distance": 20.0,
            },
            ["segment_duration > 0.1", "segment_dispersion > 12"],
            "any",
        ),
    ],
    ids=lambda value: getattr(value, "name", ""),
)
def test_segmentation_recipes_match_the_papers(method, options, criteria, choose):
    assert dict(method.segmentation.options) == options
    assert [c.label for c in method.criteria] == criteria
    assert method.segmentation.choose == choose
    assert method.units == "cm"


@pytest.fixture(scope="module")
def direction_track():
    """A 200 cm track in 2 cm bins, as Mallory et al. (2025) decoded."""
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=2.0, units="cm"
    )


def _joint_event(env, outbound, inbound, p_outbound):
    """A joint decode whose conditional posteriors are given row by row.

    ``outbound`` and ``inbound`` are (n_time_bins, n_bins) conditional
    posteriors and ``p_outbound`` the outbound template's probability in
    each time bin.
    """
    p_outbound = np.broadcast_to(np.asarray(p_outbound, dtype=float), len(outbound))
    times = _steps(len(outbound))
    return JointDecodingResult(
        template_names=("outbound", "inbound"),
        template_probability=np.column_stack([p_outbound, 1.0 - p_outbound]),
        conditional={
            "outbound": DecodingResult(posterior=outbound, env=env, times=times),
            "inbound": DecodingResult(posterior=inbound, env=env, times=times),
        },
    )


def _parked(env, n_time_bins, bin_index=50):
    return _one_hot(env, [bin_index] * n_time_bins)


class TestMallory:
    def test_detects_a_sweep_in_the_template_with_more_posterior(self, direction_track):
        env = direction_track
        event = _joint_event(
            env, _parked(env, 60), _one_hot(env, range(10, 70)), p_outbound=0.1
        )
        result = detect_replay(
            env,
            event,
            _steps(60),
            replay_method=MALLORY_2025,
            spike_counts=np.ones((60, 12), dtype=int),
        )
        assert result.is_replay and result.template == "inbound"
        assert result.values["template_fraction > 0.665"] == pytest.approx(0.9)
        assert result.values["segment_template_fraction >= 0.5"] == pytest.approx(0.9)
        assert result.summary()["template"] == "inbound"

    def test_needs_two_thirds_of_the_posterior_in_one_template(self, direction_track):
        env = direction_track
        event = _joint_event(
            env, _one_hot(env, range(10, 70)), _parked(env, 60), p_outbound=0.6
        )
        result = detect_replay(
            env,
            event,
            _steps(60),
            replay_method=MALLORY_2025,
            spike_counts=np.ones((60, 12), dtype=int),
        )
        assert result.template == "outbound" and not result.is_replay
        assert not result.criteria["template_fraction > 0.665"]
        assert result.criteria["|score| > 0.6"]

    def test_rejects_a_segment_with_more_posterior_in_the_other_template(
        self, direction_track
    ):
        """Outbound holds most of the event, inbound most of the segment.

        60 blurred time bins put 95% of their posterior in the outbound
        template but no bin above 5 times chance, so they are not in the
        segment; the 30-bin sweep after them holds 40% outbound.
        """
        env = direction_track
        flat = np.full((60, env.n_bins), 1.0 / env.n_bins)
        outbound = np.vstack([flat, _one_hot(env, range(20, 50))])
        inbound = np.vstack([flat, _parked(env, 30)])
        p_outbound = np.r_[np.full(60, 0.95), np.full(30, 0.4)]
        result = detect_replay(
            env,
            _joint_event(env, outbound, inbound, p_outbound),
            _steps(90),
            replay_method=MALLORY_2025,
            spike_counts=np.ones((90, 12), dtype=int),
        )
        assert result.template == "outbound"
        assert result.segments.rows(result.segment) == slice(60, 90)
        assert result.criteria["template_fraction > 0.665"]
        assert result.values["segment_template_fraction >= 0.5"] == pytest.approx(0.4)
        assert result.criteria["|score| > 0.6"]
        assert not result.is_replay

    def test_a_joint_decode_needs_a_template_or_a_choice_rule(self, direction_track):
        env = direction_track
        event = _joint_event(
            env, _one_hot(env, range(10, 70)), _parked(env, 60), p_outbound=0.9
        )
        with pytest.raises(ValueError, match="needs template="):
            detect_replay(env, event, _steps(60), replay_method=SILVA_2015)
        named = detect_replay(
            env, event, _steps(60), replay_method=SILVA_2015, template="outbound"
        )
        assert named.is_replay and named.template == "outbound"

    def test_template_criteria_need_a_joint_decode(self, direction_track):
        env = direction_track
        with pytest.raises(ValueError, match="compare templates"):
            detect_replay(
                env,
                _one_hot(env, range(10, 70)),
                _steps(60),
                replay_method=MALLORY_2025,
                spike_counts=np.ones((60, 12), dtype=int),
            )

    def test_active_cells_count_the_whole_event(self, place_code):
        """Cells 1 and 2 spike only before the segment and still count."""
        env, _ = place_code
        method = ReplayMethod(
            name="cells",
            reference="test",
            score=None,
            segmentation=ReplaySegmentation({"max_jump": 20.0}),
            criteria=(ReplayCriterion("n_active_cells", ">=", 3),),
        )
        counts = np.zeros((14, 3), dtype=int)
        counts[:, 0] = 1
        counts[:2, 1:] = 1
        result = detect_replay(
            env,
            _one_hot(env, [45, 45, *range(12)]),
            _steps(14),
            replay_method=method,
            spike_counts=counts,
        )
        assert result.segments.rows(result.segment) == slice(2, 14)
        assert result.values["n_active_cells >= 3"] == 3.0

    def test_compare_takes_joint_decodes(self, direction_track):
        env = direction_track
        events = [
            _joint_event(
                env, _one_hot(env, range(10, 70)), _parked(env, 60), p_outbound=0.9
            ),
            _joint_event(
                env, _parked(env, 60), _one_hot(env, range(10, 70)), p_outbound=0.6
            ),
        ]
        counts = [np.ones((60, 12), dtype=int)] * 2
        table = compare_replay_methods(
            env, events, replay_methods=[MALLORY_2025], spike_counts=counts
        )
        assert table["template"].tolist() == ["outbound", "outbound"]
        assert table["is_replay"].tolist() == [True, False]
        with pytest.raises(ValueError, match="cannot draw from joint decodes"):
            compare_replay_methods(
                env, events, replay_methods=[MICHON_2019], spike_counts=counts
            )

    def test_metrics_use_the_segments_trajectory_time_bins(self, direction_track):
        """A blurred row inside the sweep has its MAP at the far end.

        Merging bridges it; the code measures the jump and range on the
        trajectory rows, so the row's MAP does not count.
        """
        env = direction_track
        inbound = _one_hot(env, range(10, 70))
        inbound[30] = (1.0 - 0.0109) / (env.n_bins - 1)
        inbound[30, 95] = 0.0109
        event = _joint_event(env, _parked(env, 60), inbound, p_outbound=0.1)
        counts = np.ones((60, 12), dtype=int)
        result = detect_replay(
            env, event, _steps(60), replay_method=MALLORY_2025, spike_counts=counts
        )
        assert result.segments.rows(result.segment) == slice(0, 60)
        assert not result.segments.trajectory_time_bins[30]
        assert result.is_replay
        assert result.values["max_map_jump < 0.4 x track_length"] == pytest.approx(4.0)
        spanned = detect_replay(
            env,
            event,
            _steps(60),
            replay_method=_with_segmentation(MALLORY_2025, rows="segment"),
            spike_counts=counts,
        )
        assert not spanned.is_replay
        assert not spanned.criteria["max_map_jump < 0.4 x track_length"]

    def test_track_ends_are_left_out_when_choosing_the_template(self, direction_track):
        """55% of the posterior is outbound, but within 20 cm of the start."""
        import dataclasses

        env = direction_track
        event = _joint_event(
            env, _parked(env, 60, bin_index=2), _one_hot(env, range(10, 70)), 0.55
        )
        counts = np.ones((60, 12), dtype=int)
        masked = detect_replay(
            env, event, _steps(60), replay_method=MALLORY_2025, spike_counts=counts
        )
        unmasked = detect_replay(
            env,
            event,
            _steps(60),
            replay_method=dataclasses.replace(MALLORY_2025, exclude_end_distance=0.0),
            spike_counts=counts,
        )
        assert masked.template == "inbound"
        assert masked.values["template_fraction > 0.665"] == pytest.approx(1.0)
        # Unmasked, outbound holds 55% of the segment's bins, so inbound's
        # segment is rejected.
        assert not masked.criteria["segment_template_fraction >= 0.5"]
        assert unmasked.template == "outbound"

    def test_the_second_template_wins_a_tie(self, direction_track):
        env = direction_track
        event = _joint_event(env, _parked(env, 60), _parked(env, 60, 60), 0.5)
        result = detect_replay(
            env,
            event,
            _steps(60),
            replay_method=MALLORY_2025,
            spike_counts=np.ones((60, 12), dtype=int),
        )
        assert result.template == "inbound"

    def test_chance_counts_both_directions(self, direction_track):
        # Peak 0.03 in the tested share: above 5 / 200, below 5 / 100.
        env = direction_track
        event = _joint_event(
            env, _parked(env, 60), _one_hot(env, range(10, 70)), p_outbound=0.97
        )
        result = detect_replay(
            env,
            event,
            _steps(60),
            replay_method=MALLORY_2025,
            spike_counts=np.ones((60, 12), dtype=int),
            template="inbound",
        )
        assert result.template == "inbound"
        assert result.segments.rows(result.segment) == slice(0, 60)

    def test_a_named_template_must_exist(self, direction_track):
        env = direction_track
        event = _joint_event(env, _parked(env, 60), _parked(env, 60, 60), 0.5)
        with pytest.raises(ValueError, match="'left' is not a template"):
            detect_replay(
                env,
                event,
                _steps(60),
                replay_method=MALLORY_2025,
                spike_counts=np.ones((60, 12), dtype=int),
                template="left",
            )

    def test_env_must_be_the_decoded_environment(self, direction_track):
        env = direction_track
        event = _joint_event(
            env, _parked(env, 60), _one_hot(env, range(10, 70)), p_outbound=0.1
        )
        other = Environment.linear_track(
            endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=1.0, units="cm"
        )
        assert other.n_bins == env.n_bins
        with pytest.raises(ValueError, match="not the environment template"):
            detect_replay(
                other,
                event,
                _steps(60),
                replay_method=MALLORY_2025,
                spike_counts=np.ones((60, 12), dtype=int),
            )

    def test_compare_lets_mallory_choose_and_others_take_template(
        self, direction_track
    ):
        env = direction_track
        event = _joint_event(
            env, _parked(env, 60), _one_hot(env, range(10, 70)), p_outbound=0.1
        )
        table = compare_replay_methods(
            env,
            [event],
            replay_methods=[MALLORY_2025, SILVA_2015],
            spike_counts=[np.ones((60, 12), dtype=int)],
            template="outbound",
        )
        assert table["template"].tolist() == ["inbound", "outbound"]
        assert table["is_replay"].tolist() == [True, False]

    def test_result_with_template_pickles(self, direction_track):
        env = direction_track
        event = _joint_event(
            env, _parked(env, 60), _one_hot(env, range(10, 70)), p_outbound=0.1
        )
        result = detect_replay(
            env,
            event,
            _steps(60),
            replay_method=MALLORY_2025,
            spike_counts=np.ones((60, 12), dtype=int),
        )
        clone = pickle.loads(pickle.dumps(result))
        assert clone.template == "inbound"
        np.testing.assert_equal(clone.summary(), result.summary())

    def test_recipe_matches_the_code(self):
        assert MALLORY_2025.segmentation.rows == "trajectory"
        assert dict(MALLORY_2025.segmentation.options) == {
            "max_jump": 0.4,
            "method": "map",
            "min_peak_probability": 5.0,
            "merge_max_time_gap": 0.05,
            "merge_max_distance": 0.4,
            "min_duration": 0.03,
        }
        assert [c.label for c in MALLORY_2025.criteria] == [
            "segment_template_fraction >= 0.5",
            "template_fraction > 0.665",
            "|score| > 0.6",
            "max_map_jump < 0.4 x track_length",
            "map_range > 0.2 x track_length",
            "n_active_cells >= 10",
        ]
        assert MALLORY_2025.template_choice == "most_posterior"
        assert MALLORY_2025.exclude_end_distance == 20.0


def _firing(cells, n_cells=40, start=1.0, step=0.01, spikes_per_cell=1):
    """Spike trains in which ``cells`` fire once (or a burst) in turn."""
    trains = [np.array([]) for _ in range(n_cells)]
    for k, cell in enumerate(cells):
        trains[cell] = start + step * k + 0.002 * np.arange(spikes_per_cell)
    return trains


class TestRankOrderMethods:
    @pytest.mark.parametrize("order", [1, -1], ids=["forward", "reverse"])
    def test_diba_detects_forward_and_reverse_sequences(self, place_code, order):
        env, rates = place_code
        cells = list(range(10, 25))[::order]  # 15 of 40 cells
        result = detect_rank_order_replay(
            env,
            _firing(cells),
            (1.0, 1.2),
            rates,
            rank_order_method=DIBA_BUZSAKI_2007,
            rng=0,
        )
        assert result.is_replay
        assert result.score == pytest.approx(order)
        assert result.p_value == pytest.approx(2 / 501)

    def test_diba_rejects_a_shuffled_order(self, place_code):
        env, rates = place_code
        cells = list(np.random.default_rng(3).permutation(np.arange(10, 25)))
        result = detect_rank_order_replay(
            env,
            _firing(cells),
            (1.0, 1.2),
            rates,
            rank_order_method=DIBA_BUZSAKI_2007,
            rng=0,
        )
        assert abs(result.score) < 0.4 and not result.is_replay

    def test_diba_needs_30_percent_of_the_cells(self, place_code):
        env, rates = place_code
        result = detect_rank_order_replay(
            env,
            _firing(range(10, 18)),  # 8 of 40 cells
            (1.0, 1.2),
            rates,
            rank_order_method=DIBA_BUZSAKI_2007,
            rng=0,
        )
        assert result.p_value < 0.1
        assert result.values["active_fraction >= 0.3"] == pytest.approx(0.2)
        assert not result.is_replay

    def test_foster_uses_every_spike_and_the_analytic_p(self, place_code):
        env, rates = place_code
        result = detect_rank_order_replay(
            env,
            _firing(range(10, 24), spikes_per_cell=2),  # 14 of 40 cells
            (1.0, 1.2),
            rates,
            rank_order_method=FOSTER_WILSON_2006,
        )
        assert result.is_replay
        assert result.criteria["active_fraction >= 0.333333"]
        assert result.p_value < 1e-6

    def test_too_few_cells_is_not_replay(self, place_code):
        env, rates = place_code
        result = detect_rank_order_replay(
            env,
            _firing([10, 11]),
            (1.0, 1.2),
            rates,
            rank_order_method=DIBA_BUZSAKI_2007,
        )
        assert np.isnan(result.score) and np.isnan(result.p_value)
        assert not result.is_replay

    def test_compare_scores_spikes_over_each_events_windows(self, place_code):
        env, rates = place_code
        event = simulate_replay_event(
            env, rates, kind="sweep", rate_gain=4.0, seed=1, speed=900.0
        )
        counts, grid = bin_spikes_in_time(
            list(event.spike_times), dt=DT, t_start=0.0, t_stop=event.duration
        )
        decoded = decode_position(env, counts, rates, dt=DT, times=grid)
        table = compare_replay_methods(
            env,
            [decoded],
            replay_methods=[SILVA_2015, DIBA_BUZSAKI_2007],
            spike_times=list(event.spike_times),
            encoding_models=rates,
            dt=DT,
            rng=0,
        )
        assert table["method"].tolist() == ["silva_2015", "diba_buzsaki_2007"]
        assert table["is_replay"].all()
        expected = detect_rank_order_replay(
            env,
            list(event.spike_times),
            (grid[0] - DT / 2, grid[-1] + DT / 2),
            rates,
            rank_order_method=DIBA_BUZSAKI_2007,
            rng=0,
        )
        assert table["score"].iloc[1] == pytest.approx(expected.score)
        with pytest.raises(ValueError, match="need \\['spike_times'\\]"):
            compare_replay_methods(
                env,
                [decoded],
                replay_methods=[DIBA_BUZSAKI_2007],
                encoding_models=rates,
                dt=DT,
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"test": "analytic", "alternative": "greater"}, "two-sided"),
            ({"test": "bootstrap"}, "test must be"),
            ({"min_active_cells": 2}, "min_active_cells"),
            ({"min_active_fraction": 1.5}, "min_active_fraction"),
            ({"n_shuffles": 0}, "n_shuffles"),
        ],
    )
    def test_invalid_rank_order_methods_are_refused(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            RankOrderMethod(name="x", reference="y", **kwargs)

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            (
                FOSTER_WILSON_2006,
                {
                    "spike_statistic": "all_spikes",
                    "field_order": "peak",
                    "min_active_fraction": 1 / 3,
                    "test": "analytic",
                    "alpha": 0.05,
                },
            ),
            (
                DIBA_BUZSAKI_2007,
                {
                    "spike_statistic": "first",
                    "field_order": "peak",
                    "min_active_cells": 5,
                    "min_active_fraction": 0.3,
                    "test": "cell_order",
                    "alternative": "two-sided",
                    "n_shuffles": 500,
                    "alpha": 0.1,
                },
            ),
        ],
        ids=lambda value: getattr(value, "name", ""),
    )
    def test_rank_order_recipes_match_the_papers(self, method, expected):
        for name, value in expected.items():
            assert getattr(method, name) == value, name
