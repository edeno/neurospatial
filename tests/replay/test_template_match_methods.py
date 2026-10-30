"""Template-matching recipes: verdicts, comparison tables, calibration, data-set tests."""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from neurospatial import Environment
from neurospatial.decoding import DecodingResult
from neurospatial.replay import (
    TemplateMatchMethod,
    apply_alpha,
    compare_replay_methods,
    detect_template_match_replay,
    matched_alpha,
    template_match_dataset_test,
)
from neurospatial.replay.methods import (
    DIBA_BUZSAKI_2007,
    JI_WILSON_2007,
    LEE_WILSON_2002,
)

DT = 0.02
STEP = 0.06  # seconds between letters, above Lee & Wilson's 50 ms bursts


@pytest.fixture(scope="module")
def track():
    """A 240 cm track in 2 cm bins with 12 cells whose fields peak 20 cm apart."""
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [240.0, 0.0]], bin_size=2.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    rates = 0.5 + 20.0 * np.exp(
        -0.5 * ((x - (9.0 + 20.0 * np.arange(12))[:, None]) / 6.0) ** 2
    )
    return env, rates


def _session(orders, n_cells=12, step=STEP, gap=2.0, start=1.0):
    """One event per order, ``gap`` seconds apart; the k-th cell at k * step."""
    trains = [[] for _ in range(n_cells)]
    intervals = []
    for index, order in enumerate(orders):
        onset = start + gap * index
        for k, cell in enumerate(order):
            trains[cell].append(onset + step * k)
        intervals.append((onset, onset + step * len(order)))
    return [np.sort(np.array(t)) for t in trains], np.array(intervals)


def _decoded(env, intervals):
    """Events holding only times, whose decoding span is each interval."""
    events = []
    for start, stop in intervals:
        n_bins = round((stop - start) / DT)
        times = start + DT / 2 + DT * np.arange(n_bins)
        events.append(
            DecodingResult(
                posterior=np.full((n_bins, env.n_bins), 1.0 / env.n_bins),
                env=env,
                times=times,
            )
        )
    return events


class TestMethodSettings:
    def test_each_statistic_takes_its_own_options(self):
        with pytest.raises(ValueError, match="needs"):
            TemplateMatchMethod(name="x", reference="r", statistic="best_match")
        with pytest.raises(ValueError, match="takes no"):
            dataclasses.replace(JI_WILSON_2007, max_isi=0.05)
        with pytest.raises(ValueError, match="max_isi"):
            dataclasses.replace(LEE_WILSON_2002, max_isi=0.2)

    @pytest.mark.parametrize(
        "change",
        [
            {"p_comparison": "<<"},
            {"min_active_cells": 1},
            {"exact_max_letters": 11},
            {"alpha": 1.0},
            {"template_choice": "most_active"},
            {"min_template_cells": 1},
        ],
    )
    def test_bad_settings_are_refused(self, change):
        with pytest.raises(ValueError):
            dataclasses.replace(JI_WILSON_2007, **change)

    def test_the_active_cell_criterion(self):
        assert [c.label for c in JI_WILSON_2007.criteria] == ["n_active_cells >= 4"]


class TestVerdicts:
    def test_lee_counts_p_equal_to_one_in_24(self, track):
        """A perfect four-letter word: p = 1/24 is replay (p <= 1/24)."""
        env, rates = track
        spikes, intervals = _session([[0, 3, 5, 9]])
        result = detect_template_match_replay(
            env, spikes, intervals[0], rates, template_match_method=LEE_WILSON_2002
        )
        assert result.p_value == pytest.approx(1 / 24, rel=1e-12)
        assert result.is_replay and result.score == 4.0
        strict = dataclasses.replace(LEE_WILSON_2002, p_comparison="<")
        assert not detect_template_match_replay(
            env, spikes, intervals[0], rates, template_match_method=strict
        ).is_replay

    def test_ji_requires_p_below_alpha(self, track):
        """At 5 cells, 1 opposite pair gives p = 5/120; at alpha = 5/120 it fails."""
        env, rates = track
        spikes, intervals = _session([[0, 2, 5, 4, 6]], step=0.04)
        result = detect_template_match_replay(
            env, spikes, intervals[0], rates, template_match_method=JI_WILSON_2007
        )
        assert result.is_replay and result.score == pytest.approx(0.8)
        at_p = dataclasses.replace(JI_WILSON_2007, alpha=5 / 120)
        assert not detect_template_match_replay(
            env, spikes, intervals[0], rates, template_match_method=at_p
        ).is_replay

    def test_three_active_cells_are_never_ji_replay(self, track):
        env, rates = track
        spikes, intervals = _session([[0, 1, 2]], step=0.04)
        loose = dataclasses.replace(JI_WILSON_2007, min_active_cells=2)
        result = detect_template_match_replay(
            env, spikes, intervals[0], rates, template_match_method=loose
        )
        assert result.p_value == pytest.approx(1 / 6) and not result.is_replay
        default = detect_template_match_replay(
            env, spikes, intervals[0], rates, template_match_method=JI_WILSON_2007
        )
        assert np.isnan(default.score) and not default.is_replay

    def test_ji_tests_the_template_with_the_larger_index(self, track):
        env, rates = track
        spikes, intervals = _session([[9, 7, 6, 4, 2, 1]], step=0.04)
        result = detect_template_match_replay(
            env,
            spikes,
            intervals[0],
            {"out": rates, "in": rates[:, ::-1]},
            template_match_method=JI_WILSON_2007,
        )
        assert result.template == "in" and result.score == 1.0

    def test_lee_needs_a_template_with_several(self, track):
        env, rates = track
        spikes, intervals = _session([[0, 1]])
        with pytest.raises(ValueError, match="template="):
            detect_template_match_replay(
                env,
                spikes,
                intervals[0],
                {"out": rates, "in": rates[:, ::-1]},
                template_match_method=LEE_WILSON_2002,
            )

    def test_small_templates_are_refused(self, track):
        env, rates = track
        maps = rates.copy()
        maps[4:] = np.nan
        spikes, intervals = _session([[0, 1, 2, 3]])
        with pytest.raises(ValueError, match="fewer than method"):
            detect_template_match_replay(
                env, spikes, intervals[0], maps, template_match_method=JI_WILSON_2007
            )


class TestCompareReplayMethods:
    def test_recipes_run_with_the_other_methods(self, track):
        """Forward, scrambled and too-small events on one table."""
        env, rates = track
        forward = list(range(10))
        scrambled = [5, 0, 9, 3, 7, 1, 8, 2, 6, 4]
        spikes, intervals = _session([forward, scrambled, [3, 6]])
        table = compare_replay_methods(
            env,
            _decoded(env, intervals),
            replay_methods=[LEE_WILSON_2002, JI_WILSON_2007, DIBA_BUZSAKI_2007],
            spike_times=spikes,
            encoding_models=rates,
            dt=DT,
            rng=0,
        ).set_index(["event", "method"])
        for method in ("lee_wilson_2002", "ji_wilson_2007"):
            assert table.loc[(0, method), "is_replay"]
            assert not table.loc[(1, method), "is_replay"]
        assert table.loc[(0, "lee_wilson_2002"), "score"] == 10.0
        assert table.loc[(0, "ji_wilson_2007"), "score"] == 1.0
        assert table.loc[(2, "ji_wilson_2007"), "error"] == (
            "fewer than 4 active template cells"
        )
        assert not table.loc[(2, "ji_wilson_2007"), "passes[n_active_cells >= 4]"]
        assert table.loc[(2, "lee_wilson_2002"), "error"] == ""
        # The verdicts equal the per-event function's on the decoding span.
        single = detect_template_match_replay(
            env, spikes, intervals[1], rates, template_match_method=JI_WILSON_2007
        )
        assert table.loc[(1, "ji_wilson_2007"), "p_value"] == single.p_value

    def test_spike_times_are_required(self, track):
        env, rates = track
        _, intervals = _session([[0, 1, 2, 3]])
        with pytest.raises(ValueError, match="spike_times"):
            compare_replay_methods(
                env,
                _decoded(env, intervals),
                replay_methods=[JI_WILSON_2007],
                encoding_models=rates,
                dt=DT,
            )


class TestCalibration:
    def test_lee_tables_keep_p_equal_to_alpha(self, track):
        """matched_alpha and apply_alpha read Lee's p <= 1/24 verdicts."""
        env, rates = track
        orders = [[0, 1, 2, 3]] * 2 + [[1, 0, 2, 3]] * 8 + [[0, 1, 2, 3, 4]] * 2
        spikes, intervals = _session(orders)
        table = compare_replay_methods(
            env,
            _decoded(env, intervals),
            replay_methods=[LEE_WILSON_2002],
            spike_times=spikes,
            encoding_models=rates,
            dt=DT,
            rng=0,
        )
        assert table["is_replay"].sum() == 4
        matched = matched_alpha(table, replay_methods=[LEE_WILSON_2002], target=0.2)
        row = matched.iloc[0]
        # Two events at p = 1/120 may pass, the two at 1/24 may not.
        assert 1 / 120 <= row["alpha"] < 1 / 24
        assert row["rate"] == pytest.approx(2 / 12)
        applied = apply_alpha(
            table,
            alpha={"lee_wilson_2002": row["alpha"]},
            replay_methods=[LEE_WILSON_2002],
        )
        assert applied["is_replay"].sum() == 2


# Data-set tests --------------------------------------------------------------


def _by_template(table):
    """Rows by template; the rows of every template together as "all"."""
    return table.assign(template=table["template"].fillna("all")).set_index("template")


def _lee_dataset(n_pairs, n_pair_matches, n_low, n_low_matches):
    """Pairs of cells 0 and 1, and four-letter words of cells 2-5."""
    orders = [[0, 1]] * n_pair_matches + [[1, 0]] * (n_pairs - n_pair_matches)
    orders += [[2, 3, 4, 5]] * n_low_matches + [[3, 2, 4, 5]] * (n_low - n_low_matches)
    return _session(orders, n_cells=6, gap=0.5)


class TestLeeDatasetTest:
    def test_the_papers_post_sleep_counts(self, track):
        """655 of 1255 pairs: Z = 1.55; 35 of 270 low-probability words: Z =
        7.23, exact p = 3.8e-9 (Lee & Wilson 2002, p. 1187).
        """
        env, rates = track
        spikes, intervals = _lee_dataset(1255, 655, 270, 35)
        method = dataclasses.replace(LEE_WILSON_2002, n_shuffles=20)
        result = template_match_dataset_test(
            env, spikes, intervals, rates[:6], template_match_method=method, rng=0
        )
        table = result.groups_table().set_index("group")
        assert table.loc["pair", ["n_trials", "n_matches"]].tolist() == [1255, 655]
        assert table.loc["pair", "z_score"] == pytest.approx(1.55, abs=0.005)
        assert table.loc["pair", "p_value"] == pytest.approx(
            stats.norm.sf(1.5525), abs=1e-3
        )
        low = table.loc["low_probability"]
        assert (low["n_trials"], low["n_matches"]) == (270, 35)
        assert low["z_score"] == pytest.approx(7.23, abs=0.005)
        assert low["p_value"] == pytest.approx(3.76e-9, rel=0.01)
        assert low["z_score_upper"] == pytest.approx(low["z_score"])
        assert table.loc["triplet", "n_trials"] == 0
        # The shuffle z is the observed Z in standard deviations of the shuffles.
        shuffled = result.shuffle_statistics
        assert result.observed_statistic == pytest.approx(low["z_score"])
        assert result.shuffle_z_score == pytest.approx(
            (low["z_score"] - shuffled.mean()) / shuffled.std()
        )

    def test_the_upper_bound_uses_each_words_attainable_size(self, track):
        """Five distinct letters: the largest achievable p below 1/24 is 1/120 + ..."""
        from neurospatial.replay.sequence_match import _match_keys

        env, rates = track
        spikes, intervals = _session([[0, 1, 2, 4, 3]] * 30, n_cells=6, gap=0.5)
        method = dataclasses.replace(LEE_WILSON_2002, n_shuffles=5)
        result = template_match_dataset_test(
            env, spikes, intervals, rates[:6], template_match_method=method, rng=0
        )
        low = result.groups_table().set_index("group").loc["low_probability"]
        keys = np.sort(
            _match_keys(
                np.array(list(itertools.permutations(range(5)))), "interruptions_first"
            )
        )
        shares = np.array([np.mean(keys >= key) for key in np.unique(keys)])
        size = shares[shares <= 1 / 24].max()
        assert size < 1 / 24
        expected = 30 * size
        assert low["z_score_upper"] == pytest.approx(
            (low["n_matches"] - expected) / np.sqrt(30 * size * (1 - size))
        )
        assert low["z_score_upper"] > low["z_score"]

    def test_a_tie_never_helps_a_match_in_the_shuffles(self, track):
        """Four cells fire at the same time in each word. Ties are ordered
        against the template, so no word matches it, nor any shuffled
        template: every shuffle's Z equals the observed Z.
        """
        env, rates = track
        onsets = 1.0 + 0.5 * np.arange(30)
        spikes = [onsets.copy() for _ in range(4)]
        intervals = np.column_stack([onsets, onsets + 0.05])
        method = dataclasses.replace(LEE_WILSON_2002, n_shuffles=100)
        result = template_match_dataset_test(
            env, spikes, intervals, rates[:4], template_match_method=method, rng=0
        )
        low = result.groups_table().set_index("group").loc["low_probability"]
        assert (low["n_trials"], low["n_matches"]) == (30, 0)
        np.testing.assert_array_equal(
            result.shuffle_statistics, np.full(100, result.observed_statistic)
        )

    def test_several_templates_need_their_own_words(self, track):
        env, rates = track
        spikes, intervals = _session([[0, 1, 2, 3]])
        models = {"out": rates, "in": rates[:, ::-1]}
        with pytest.raises(ValueError, match="map each template"):
            template_match_dataset_test(
                env, spikes, intervals, models, template_match_method=LEE_WILSON_2002
            )
        result = template_match_dataset_test(
            env,
            spikes,
            {"out": intervals, "in": intervals},
            models,
            template_match_method=dataclasses.replace(LEE_WILSON_2002, n_shuffles=3),
            rng=0,
        )
        table = result.groups_table()
        low = _by_template(table[table["group"] == "low_probability"])
        assert low.loc["out", "n_matches"] == 1 and low.loc["in", "n_matches"] == 0
        assert low.loc["all", "n_trials"] == 2


class TestJiDatasetTest:
    def test_the_chance_test(self, track):
        """a = the sum of each frame's exact size at its number of cells."""
        env, rates = track
        frames = [list(range(m)) for m in (4, 5, 6, 7, 8)] + [[1, 0, 3, 2, 5, 4]]
        spikes, intervals = _session(frames, step=0.04)
        method = dataclasses.replace(JI_WILSON_2007, n_shuffles=10)
        result = template_match_dataset_test(
            env, spikes, intervals, rates, template_match_method=method, rng=0
        )
        row = result.groups_table().iloc[0]
        sizes = [1 / 24, 5 / 120, 20 / 720, 174 / 5040, 1230 / 40320, 20 / 720]
        assert row["n_trials"] == 6 and row["n_matches"] == 5
        assert row["expected"] == pytest.approx(sum(sizes), rel=1e-12)
        a = sum(sizes)
        assert row["z_score"] == pytest.approx((5 - a) / np.sqrt(a))
        assert row["p_value"] == pytest.approx(stats.norm.sf((5 - a) / np.sqrt(a)))
        # The exact tail of independent trials, by enumeration.
        exact = sum(
            np.prod([q if hit else 1 - q for q, hit in zip(sizes, hits, strict=True)])
            for hits in itertools.product([0, 1], repeat=6)
            if sum(hits) >= 5
        )
        assert row["p_value_exact"] == pytest.approx(exact, rel=1e-9)

    def test_every_shuffle_assigns_frames_again(self, track):
        """Six-cell frames in field order against two templates (the order and
        its reverse): a shuffle replays when either shuffled template matches,
        1 - (1 - 20/720)^2 of the time; keeping the observed template would
        give 20/720.
        """
        env, rates = track
        spikes, intervals = _session([list(range(6))] * 3, n_cells=6, step=0.04)
        method = dataclasses.replace(JI_WILSON_2007, n_shuffles=3000)
        result = template_match_dataset_test(
            env,
            spikes,
            intervals,
            {"out": rates[:6], "in": rates[:6][:, ::-1]},
            template_match_method=dataclasses.replace(method, min_template_cells=None),
            rng=1,
        )
        assert set(np.unique(result.shuffle_statistics)) <= {0.0, 3.0}
        rate = np.mean(result.shuffle_statistics == 3.0)
        expected = 1 - (1 - 20 / 720) ** 2
        assert abs(rate - expected) < 4 * np.sqrt(expected * (1 - expected) / 3000)
        assert result.observed_statistic == 3.0
        table = _by_template(result.groups_table())
        assert table.loc["out", "n_trials"] == 3 and table.loc["in", "n_trials"] == 0

    def test_frames_must_be_one_array(self, track):
        env, rates = track
        spikes, intervals = _session([[0, 1, 2, 3]])
        with pytest.raises(ValueError, match="one array"):
            template_match_dataset_test(
                env,
                spikes,
                {"out": intervals},
                {"out": rates},
                template_match_method=JI_WILSON_2007,
            )


def test_the_summary_reports_the_shuffled_group(track):
    env, rates = track
    spikes, intervals = _session([list(range(6))] * 4, step=0.04)
    result = template_match_dataset_test(
        env,
        spikes,
        intervals,
        rates,
        template_match_method=dataclasses.replace(JI_WILSON_2007, n_shuffles=50),
        rng=0,
    )
    summary = result.summary()
    assert summary["group"] == "candidate_frames" and summary["n_matches"] == 4
    assert isinstance(result.groups_table(), pd.DataFrame)
    assert not result.shuffle_statistics.flags.writeable
