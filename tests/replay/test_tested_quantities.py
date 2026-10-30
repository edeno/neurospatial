"""Trajectory quantities tested against shuffles (Widloski & Foster 2025)."""

from __future__ import annotations

import dataclasses
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import bin_spikes_in_time, decode_position
from neurospatial.replay import (
    ReplayCriterion,
    ReplayMethod,
    ReplaySegmentation,
    compare_replay_methods,
    detect_replay,
    trajectory_metrics,
    weighted_correlation,
)
from neurospatial.replay.methods import WIDLOSKI_FOSTER_2022, WIDLOSKI_FOSTER_2025
from neurospatial.simulation import simulate_replay_event
from neurospatial.stats.shuffle import compute_shuffle_pvalue

DT = 0.02
TESTED = {"mean_spread": "less", "mean_mean_jump": "less"}


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


@pytest.fixture(scope="module")
def sweep(place_code):
    """A 900 cm/s sweep decoded in 20 ms bins: posterior, times and counts."""
    env, rates = place_code
    event = simulate_replay_event(
        env, rates, kind="sweep", speed=900.0, rate_gain=4.0, seed=1
    )
    counts, grid = bin_spikes_in_time(
        list(event.spike_times), dt=DT, t_start=0.0, t_stop=event.duration
    )
    decoded = decode_position(env, counts, rates, dt=DT, times=grid)
    return decoded.posterior, decoded.times, counts


def _method(**changes):
    fields = {
        "name": "tested",
        "reference": "test",
        "score": None,
        "nulls": ("unit_identity",),
        "n_shuffles": 100,
        "tested_quantities": TESTED,
    }
    return ReplayMethod(**{**fields, **changes})


@pytest.fixture
def spy(monkeypatch):
    """Every replay_significance call detect_replay makes, with its result."""
    import neurospatial.replay.significance as significance

    calls = []
    original = significance.replay_significance

    def recorded(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result

    monkeypatch.setattr(significance, "replay_significance", recorded)
    return calls


def _detect(place_code, event, method, **kwargs):
    env, rates = place_code
    posterior, times, counts = event
    return detect_replay(
        env,
        posterior,
        times,
        replay_method=method,
        spike_counts=counts,
        encoding_models=rates,
        dt=DT,
        **{"rng": 0, **kwargs},
    )


class TestShuffleTest:
    def test_a_sweep_is_sharper_and_steps_less_than_every_shuffle(
        self, place_code, sweep
    ):
        result = _detect(place_code, sweep, _method())
        assert dict(result.tested_p_values) == pytest.approx(
            {"mean_spread": 1 / 101, "mean_mean_jump": 1 / 101}
        )
        assert result.p_value == pytest.approx(1 / 101)
        assert result.is_replay and result.n_shuffles_used == 100
        metrics = trajectory_metrics(place_code[0], sweep[0], sweep[1])
        assert result.tested_values["mean_spread"] == pytest.approx(metrics.mean_spread)
        summary = result.summary()
        assert summary["p_value[mean_mean_jump]"] == pytest.approx(1 / 101)
        assert summary["mean_mean_jump"] == pytest.approx(metrics.mean_mean_jump)

    def test_events_decoded_from_shuffled_cells_are_rarely_replay(
        self, place_code, sweep
    ):
        """A sweep's counts with the cells permuted is itself a shuffle."""
        env, rates = place_code
        _, times, counts = sweep
        p_values = []
        for seed in range(20):
            order = np.random.default_rng(100 + seed).permutation(counts.shape[1])
            shuffled = counts[:, order]
            posterior = decode_position(env, shuffled, rates, dt=DT).posterior
            result = _detect(
                place_code,
                (posterior, times, shuffled),
                _method(n_shuffles=99),
                rng=seed,
            )
            p_values.append(result.p_value)
        p_values = np.asarray(p_values)
        assert np.sum(p_values < 0.05) <= 2
        assert np.median(p_values) > 0.2

    def test_each_tail_is_counted_from_the_same_shuffles(self, place_code, sweep, spy):
        tails = {
            "mean_spread": "less",
            "mean_mean_jump": "greater",
            "dispersion": "two-sided",
        }
        result = _detect(place_code, sweep, _method(tested_quantities=tails))
        nulls = {
            kwargs["score_options"]["quantity"]: result_.shuffles["unit_identity"]
            for _, kwargs, result_ in spy
        }
        assert result.tested_p_values["mean_spread"] == pytest.approx(1 / 101)
        # Every shuffle jumps more than the sweep: none as small, all as large.
        assert result.tested_p_values["mean_mean_jump"] == pytest.approx(1.0)
        dispersion = nulls["dispersion"]
        assert result.tested_p_values["dispersion"] == pytest.approx(
            compute_shuffle_pvalue(
                dispersion.observed_score, dispersion.null_scores, tail="two-sided"
            )
        )
        assert result.p_value == pytest.approx(1.0) and not result.is_replay

    def test_the_score_and_every_quantity_see_the_same_shuffles(
        self, place_code, sweep, spy
    ):
        """A score equal to a tested quantity gets the quantity's null scores.

        The score takes an ``rng``, which the test would otherwise draw from
        the shuffles' stream before drawing the shuffles.
        """

        def mean_spread(env, posterior, times, *, rng=None):
            spread = trajectory_metrics(env, posterior, times).mean_spread
            return SimpleNamespace(score=spread)

        method = _method(score=mean_spread, alternative="less")
        result = _detect(place_code, sweep, method)
        seeds = {kwargs["rng"] for _, kwargs, _ in spy}
        assert len(spy) == 3 and len(seeds) == 1
        score_test, *quantity_tests = (result_ for _, _, result_ in spy)
        spread_test = next(
            result_
            for (_, kwargs, result_) in spy
            if kwargs["score_options"].get("quantity") == "mean_spread"
        )
        assert spread_test in quantity_tests
        np.testing.assert_array_equal(
            score_test.shuffles["unit_identity"].null_scores,
            spread_test.shuffles["unit_identity"].null_scores,
        )
        assert result.p_value == pytest.approx(
            max(score_test.p_value, *result.tested_p_values.values())
        )

    def test_a_seed_reproduces_the_test(self, place_code, sweep):
        method = _method(tested_quantities={"dispersion": "two-sided"})
        first = _detect(place_code, sweep, method, rng=3)
        again = _detect(place_code, sweep, method, rng=3)
        assert first.tested_p_values == again.tested_p_values

    def test_an_undefined_quantity_gives_no_verdict(self, place_code, sweep):
        posterior, times, counts = sweep
        one_bin = (posterior[:1], times[:1], counts[:1])
        result = _detect(place_code, one_bin, _method())
        assert np.isnan(result.tested_values["mean_mean_jump"])
        assert np.isnan(result.p_value) and not result.is_replay

    def test_compare_reports_each_quantity(self, place_code):
        from neurospatial.decoding import DecodingResult

        env, _ = place_code
        method = _method(
            nulls=("time_bin_permutation",),
            n_shuffles=99,
            tested_quantities={"mean_mean_jump": "less", "mean_spread": "less"},
        )
        event = DecodingResult(
            posterior=np.eye(env.n_bins)[10:20], env=env, times=np.arange(10) * DT
        )
        table = compare_replay_methods(env, [event], replay_methods=[method], rng=0)
        row = table.iloc[0]
        assert row["p_value[mean_mean_jump]"] == pytest.approx(1 / 100)
        assert row["mean_mean_jump"] == pytest.approx(4.0)
        # A point posterior stays one in every permutation: no evidence.
        assert row["mean_spread"] == pytest.approx(0.0, abs=1e-9)
        assert row["p_value[mean_spread]"] == 1.0
        assert row["p_value"] == 1.0
        assert row["error"] == "" and not row["is_replay"]

    def test_an_undefined_quantity_is_an_error_in_the_table(self):
        """The range along a branched track is undefined (NaN)."""
        from neurospatial.decoding import DecodingResult

        t_maze = Environment.maze(
            kind="t",
            node_positions={
                "stem": (0.0, 0.0),
                "junction": (0.0, 50.0),
                "left": (-50.0, 50.0),
                "right": (50.0, 50.0),
            },
            bin_size=10.0,
            track_width=10.0,
            units="cm",
        )
        bins = t_maze.bin_at(np.array([[0.0, 5.0], [0.0, 25.0], [0.0, 45.0]]))
        event = DecodingResult(
            posterior=np.eye(t_maze.n_bins)[bins], env=t_maze, times=np.arange(3) * DT
        )
        method = _method(
            nulls=("time_bin_permutation",), tested_quantities={"mean_range": "greater"}
        )
        table = compare_replay_methods(t_maze, [event], replay_methods=[method])
        assert np.isnan(table["p_value"].iloc[0]) and not table["is_replay"].iloc[0]
        assert "tested quantity is undefined" in table["error"].iloc[0]

    def test_methods_and_results_pickle(self, place_code, sweep):
        method = _method()
        assert pickle.loads(pickle.dumps(method)).tested_quantities == TESTED
        result = _detect(place_code, sweep, method)
        restored = pickle.loads(pickle.dumps(result))
        assert restored.tested_p_values == result.tested_p_values
        with pytest.raises(TypeError):
            method.tested_quantities["dispersion"] = "less"  # type: ignore[index]


class TestSegments:
    def test_the_segment_rows_are_tested_without_segmenting_shuffles(
        self, place_code, sweep, spy
    ):
        """Six rows at the sweep's end, then the sweep from its start.

        The jump back to the start ends the first segment; the sweep's rows
        are tested, and the shuffles are those rows decoded again.
        """
        env, _ = place_code
        posterior, times, counts = sweep
        stay = np.tile(posterior[-1], (6, 1))
        stay_counts = np.tile(counts[-1], (6, 1))
        event = (
            np.vstack([stay, posterior]),
            np.arange(len(posterior) + 6) * DT,
            np.vstack([stay_counts, counts]),
        )
        method = _method(segmentation=ReplaySegmentation({"max_jump": 40.0}))
        result = _detect(place_code, event, method)
        rows = result.segments.rows(result.segment)
        assert rows == slice(6, 6 + len(posterior))
        for args, _, _ in spy:
            np.testing.assert_array_equal(args[1], event[0][rows])
        assert result.tested_values["mean_spread"] == pytest.approx(
            trajectory_metrics(env, posterior, times).mean_spread
        )
        assert result.tested_p_values["mean_spread"] == pytest.approx(1 / 101)

    def test_any_segment_tests_the_longest_first_and_reports_the_first_significant(
        self, place_code
    ):
        """A 30-bin stationary segment cannot step less than its permutations.

        Every time-bin permutation of a stationary segment is the segment
        itself, so its jump p-value is 1; the 25-bin sweep after it passes.
        """
        env, _ = place_code
        posterior = np.eye(env.n_bins)[[5] * 30 + list(range(20, 45))]
        times = np.arange(55) * 0.005
        method = _method(
            nulls=("time_bin_permutation",),
            n_shuffles=99,
            tested_quantities={"mean_mean_jump": "less"},
            segmentation=ReplaySegmentation({"max_jump": 20.0}, choose="any"),
        )
        result = detect_replay(env, posterior, times, replay_method=method, rng=0)
        assert result.segments.n_segments == 2
        assert result.segments.rows(result.segment) == slice(30, 55)
        assert result.is_replay and result.p_value == pytest.approx(1 / 100)

        longest = dataclasses.replace(
            method, segmentation=ReplaySegmentation({"max_jump": 20.0})
        )
        stuck = detect_replay(env, posterior, times, replay_method=longest, rng=0)
        assert stuck.segments.rows(stuck.segment) == slice(0, 30)
        assert stuck.p_value == 1.0 and not stuck.is_replay

        # Without a significant segment, the longest is reported.
        still = np.eye(env.n_bins)[[5] * 30 + [20] * 25]
        neither = detect_replay(env, still, times, replay_method=method, rng=0)
        assert neither.segments.rows(neither.segment) == slice(0, 30)
        assert not neither.is_replay

    def test_any_segment_only_tests_segments_that_meet_the_criteria(
        self, place_code, spy
    ):
        env, _ = place_code
        posterior = np.eye(env.n_bins)[[5] * 30 + list(range(20, 45))]
        method = _method(
            nulls=("time_bin_permutation",),
            n_shuffles=19,
            tested_quantities={"mean_mean_jump": "less"},
            segmentation=ReplaySegmentation({"max_jump": 20.0}, choose="any"),
            criteria=(ReplayCriterion("segment_dispersion", ">", 12.0),),
        )
        detect_replay(env, posterior, np.arange(55) * 0.005, replay_method=method)
        assert len(spy) == 1
        np.testing.assert_array_equal(spy[0][0][1], posterior[30:])

    def test_dropped_time_bins_are_left_out_of_the_tested_rows(self, place_code, sweep):
        posterior, times, counts = sweep
        silent = counts.copy()
        silent[3] = 0
        kept = np.arange(len(times)) != 3
        dropped = _detect(
            place_code,
            (posterior, times, silent),
            _method(empty_time_bins="drop", tested_quantities={"dispersion": "less"}),
        )
        assert dropped.tested_values["dispersion"] == pytest.approx(
            trajectory_metrics(place_code[0], posterior[kept], times[kept]).dispersion
        )

    def test_a_joint_decode_tests_the_chosen_templates_segment(self, place_code):
        """Outbound holds 80% of a sweep; inbound 20% of a parked position."""
        from neurospatial.decoding import DecodingResult, JointDecodingResult

        env, _ = place_code
        times = np.arange(20) * 0.005
        event = JointDecodingResult(
            template_names=("outbound", "inbound"),
            template_probability=np.column_stack([np.full(20, 0.8), np.full(20, 0.2)]),
            conditional={
                "outbound": DecodingResult(
                    posterior=np.eye(env.n_bins)[10:30], env=env, times=times
                ),
                "inbound": DecodingResult(
                    posterior=np.eye(env.n_bins)[[40] * 20], env=env, times=times
                ),
            },
        )
        method = _method(
            nulls=("time_bin_permutation",),
            n_shuffles=99,
            tested_quantities={"mean_mean_jump": "less"},
            segmentation=ReplaySegmentation({"max_jump": 20.0}),
            criteria=(ReplayCriterion("segment_template_fraction", ">=", 0.5),),
            template_choice="most_posterior",
        )
        result = detect_replay(env, event, times, replay_method=method, rng=0)
        assert result.template == "outbound"
        assert result.values["segment_template_fraction >= 0.5"] == pytest.approx(0.8)
        assert result.tested_values["mean_mean_jump"] == pytest.approx(4.0)
        assert result.p_value == pytest.approx(1 / 100) and result.is_replay


class TestValidation:
    @pytest.mark.parametrize(
        "quantity",
        ["score", "segment_dispersion", "duration", "n_time_bins", "empty_fraction"],
    )
    def test_only_quantities_a_shuffle_changes_can_be_tested(self, quantity):
        with pytest.raises(ValueError, match="a shuffle test cannot compare"):
            _method(tested_quantities={quantity: "less"})

    def test_a_tail_must_be_named(self):
        with pytest.raises(ValueError, match="tails must be one of"):
            _method(tested_quantities={"mean_spread": "lower"})

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"nulls": ()}, "need nulls"),
            ({"empty_time_bins": "zero"}, "empty_time_bins='zero'"),
            (
                {"score": weighted_correlation, "template_choice": "max_score"},
                "template_choice='max_score'",
            ),
            (
                {
                    "score": weighted_correlation,
                    "segmentation": ReplaySegmentation(
                        {"max_jump": 20.0}, choose="any"
                    ),
                },
                "choose='any'",
            ),
        ],
    )
    def test_unsupported_settings_are_refused(self, changes, match):
        with pytest.raises(ValueError, match=match):
            _method(**changes)

    def test_a_score_and_a_segmentation_are_not_needed(self):
        assert _method().segmentation is None
        tested = _method(
            score=weighted_correlation,
            segmentation=ReplaySegmentation({"max_jump": 20.0}),
        )
        assert tested.tested_quantities == TESTED


class TestWidloskiFoster2025:
    @pytest.fixture(scope="class")
    def windows(self):
        """80 ms windows stepped by 5 ms over a 100 cm/s sweep in 2 cm bins."""
        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=2.0, units="cm"
        )
        x = env.bin_centers[:, 0]
        rates = 0.5 + 25.0 * np.exp(
            -0.5 * ((x - np.linspace(0.0, 200.0, 60)[:, None]) / 10.0) ** 2
        )
        event = simulate_replay_event(
            env,
            rates,
            kind="sweep",
            speed=100.0,
            duration=0.8,
            start_position=40.0,
            seed=1,
        )
        counts, grid = bin_spikes_in_time(
            list(event.spike_times), dt=0.005, t_start=0.0, t_stop=event.duration
        )
        summed = np.lib.stride_tricks.sliding_window_view(counts, 16, axis=0)
        windows = summed.sum(axis=-1)
        centers = (grid[:-15] + grid[15:]) / 2
        posterior = decode_position(env, windows, rates, dt=0.08).posterior
        return env, rates, posterior, centers, windows

    def test_the_recipe_is_the_2022_detection_with_a_shuffle_test(self):
        changed = {
            item.name
            for item in dataclasses.fields(WIDLOSKI_FOSTER_2025)
            if getattr(WIDLOSKI_FOSTER_2025, item.name)
            != getattr(WIDLOSKI_FOSTER_2022, item.name)
        }
        assert changed == {
            "name",
            "reference",
            "nulls",
            "n_shuffles",
            "tested_quantities",
            "decoding",
            "notes",
        }

    def test_a_slow_sweep_is_replay(self, windows):
        env, rates, posterior, times, counts = windows
        result = detect_replay(
            env,
            posterior,
            times,
            replay_method=WIDLOSKI_FOSTER_2025,
            spike_counts=counts,
            encoding_models=rates,
            dt=0.08,
            valid_time_bins=np.ones(len(times), dtype=bool),
            rng=0,
        )
        assert result.segments.n_segments == 1
        assert result.criteria == {
            "segment_duration > 0.1": True,
            "segment_dispersion > 12": True,
        }
        assert dict(result.tested_p_values) == pytest.approx(
            {"mean_spread": 1 / 101, "mean_mean_jump": 1 / 101}
        )
        assert result.is_replay

    def test_running_bins_are_not_replay(self, windows):
        env, rates, posterior, times, counts = windows
        result = detect_replay(
            env,
            posterior,
            times,
            replay_method=WIDLOSKI_FOSTER_2025,
            spike_counts=counts,
            encoding_models=rates,
            dt=0.08,
            valid_time_bins=np.zeros(len(times), dtype=bool),
            rng=0,
        )
        assert result.segment is None and not result.is_replay
        assert np.isnan(result.tested_p_values["mean_spread"])
