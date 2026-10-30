"""Tests for rank_order_correlation and rank_order_significance."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from neurospatial import Environment
from neurospatial.replay import (
    RankOrderResult,
    rank_order_correlation,
    rank_order_significance,
)

INTERVAL = (1.0, 1.2)


@pytest.fixture
def track():
    """A 120 cm track and 8 cells with fields evenly spaced along it."""
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [120.0, 0.0]], bin_size=5.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    fields = np.linspace(10.0, 110.0, 8)
    rates = 20.0 * np.exp(-0.5 * ((x - fields[:, None]) / 8.0) ** 2)
    return env, rates, fields


def _in_field_order(n_cells: int, order=None) -> list[np.ndarray]:
    """One burst per cell, 15 ms apart, in ``order`` (default: field order)."""
    order = np.arange(n_cells) if order is None else np.asarray(order)
    trains = [np.empty(0)] * n_cells
    for slot, cell in enumerate(order):
        start = INTERVAL[0] + 0.015 * slot
        trains[cell] = np.array([start, start + 0.004, start + 0.009])
    return trains


class TestCorrelation:
    def test_forward_and_reverse_sequences(self, track):
        env, rates, _ = track
        forward = rank_order_correlation(env, _in_field_order(8), INTERVAL, rates)
        reverse = rank_order_correlation(
            env, _in_field_order(8, order=np.arange(8)[::-1]), INTERVAL, rates
        )
        assert forward.score == pytest.approx(1.0)
        assert reverse.score == pytest.approx(-1.0)
        assert forward.n_active_cells == 8

    @pytest.mark.parametrize(
        ("statistic", "summarize"),
        [("first", np.min), ("median", np.median), ("mean", np.mean)],
    )
    def test_matches_spearman_of_each_cells_spike_time(
        self, track, statistic, summarize
    ):
        env, rates, _ = track
        rng = np.random.default_rng(0)
        trains = [np.sort(rng.uniform(*INTERVAL, rng.integers(1, 6))) for _ in range(8)]
        result = rank_order_correlation(
            env, trains, INTERVAL, rates, spike_statistic=statistic
        )
        peaks = env.bin_centers[np.argmax(rates, axis=1), 0]
        expected = stats.spearmanr(peaks, [summarize(t) for t in trains])
        assert result.score == pytest.approx(expected.statistic, rel=1e-12)

    def test_all_spikes_pairs_every_spike_with_its_field(self, track):
        env, rates, _ = track
        rng = np.random.default_rng(1)
        trains = [np.sort(rng.uniform(*INTERVAL, rng.integers(1, 6))) for _ in range(8)]
        result = rank_order_correlation(
            env, trains, INTERVAL, rates, spike_statistic="all_spikes"
        )
        peaks = env.bin_centers[np.argmax(rates, axis=1), 0]
        x = np.concatenate(
            [np.full(t.size, p) for t, p in zip(trains, peaks, strict=True)]
        )
        expected = stats.spearmanr(x, np.concatenate(trains))
        assert result.score == pytest.approx(expected.statistic, rel=1e-12)

    def test_the_statistic_changes_the_order(self, track):
        """A cell that fires first once but mostly late moves with the statistic."""
        env, rates, _ = track
        trains = _in_field_order(6)[:6]
        trains[5] = np.array([1.0001, 1.19, 1.195])  # field last, first spike first
        rates6 = rates[:6]
        first = rank_order_correlation(
            env, trains, INTERVAL, rates6, spike_statistic="first"
        ).score
        median = rank_order_correlation(
            env, trains, INTERVAL, rates6, spike_statistic="median"
        ).score
        assert median == pytest.approx(1.0)
        assert first < median

    def test_cell_numbering_does_not_change_the_score(self, track):
        """The RnR_methods toolbox correlates sort indices, which depends on it."""
        env, rates, _ = track
        rng = np.random.default_rng(2)
        trains = [np.sort(rng.uniform(*INTERVAL, 3)) for _ in range(8)]
        score = rank_order_correlation(env, trains, INTERVAL, rates).score
        for _ in range(5):
            relabel = rng.permutation(8)
            again = rank_order_correlation(
                env, [trains[i] for i in relabel], INTERVAL, rates[relabel]
            ).score
            assert again == pytest.approx(score, rel=1e-12)

    def test_field_center_differs_from_peak_for_a_skewed_field(self, track):
        env, rates, _ = track
        x = env.bin_centers[:, 0]
        skewed = rates.copy()
        # Cell 0 peaks at 10 cm but carries a long tail to 115 cm.
        skewed[0] = 20.0 * np.exp(-0.5 * ((x - 10.0) / 4.0) ** 2) + np.where(
            x > 60, 15.0, 0.0
        )
        trains = _in_field_order(8)
        peak = rank_order_correlation(env, trains, INTERVAL, skewed).score
        center = rank_order_correlation(
            env, trains, INTERVAL, skewed, field_order="center"
        ).score
        assert peak == pytest.approx(1.0)
        assert center < peak

    def test_only_cells_with_a_field_and_a_spike_in_the_event_count(self, track):
        env, rates, _ = track
        trains = _in_field_order(8)
        trains[2] = np.array([0.5, 1.2])  # spikes before and at stop: outside
        no_field = rates.copy()
        no_field[4] = np.nan  # no training data for cell 4
        result = rank_order_correlation(env, trains, INTERVAL, no_field)
        assert result.n_active_cells == 6
        assert list(result.active_units) == [0, 1, 3, 5, 6, 7]
        assert result.score == pytest.approx(1.0)

    def test_too_few_active_cells_is_nan(self, track):
        env, rates, _ = track
        trains = _in_field_order(8)
        for cell in (0, 1, 2, 3):
            trains[cell] = np.empty(0)
        result = rank_order_correlation(env, trains, INTERVAL, rates)
        assert result.n_active_cells == 4
        assert np.isnan(result.score)
        loose = rank_order_correlation(env, trains, INTERVAL, rates, min_active_cells=4)
        assert loose.score == pytest.approx(1.0)

    def test_result_is_read_only_and_pickles(self, track):
        env, rates, _ = track
        result = rank_order_correlation(env, _in_field_order(8), INTERVAL, rates)
        assert isinstance(result, RankOrderResult)
        with pytest.raises(ValueError, match="read-only"):
            result.active_units[0] = 3
        for clone in (pickle.loads(pickle.dumps(result)), copy.deepcopy(result)):
            assert clone.score == result.score
            assert not clone.active_units.flags.writeable

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"spike_statistic": "last"}, "spike_statistic"),
            ({"field_order": "mode"}, "field_order"),
            ({"min_active_cells": 2}, "at least 3"),
            ({"interval": (1.2, 1.0)}, "start < stop"),
            ({"interval": (1.0, np.nan)}, "start < stop"),
        ],
    )
    def test_refusals(self, track, kwargs, match):
        env, rates, _ = track
        interval = kwargs.pop("interval", INTERVAL)
        with pytest.raises(ValueError, match=match):
            rank_order_correlation(env, _in_field_order(8), interval, rates, **kwargs)

    def test_cell_count_must_match(self, track):
        env, rates, _ = track
        with pytest.raises(ValueError, match="spike_times has 7 cells"):
            rank_order_correlation(env, _in_field_order(7), INTERVAL, rates)

    def test_negative_rates_are_refused(self, track):
        env, rates, _ = track
        bad = rates.copy()
        bad[2, 3] = -1.0
        with pytest.raises(ValueError, match="negative rates"):
            rank_order_correlation(env, _in_field_order(8), INTERVAL, bad)

    def test_rate_result_from_another_environment_is_refused(self):
        from neurospatial._exceptions import IncompatibleEnvironmentError
        from neurospatial.encoding import compute_spatial_rates

        forward = Environment.linear_track(
            endpoints=[[0.0, 0.0], [120.0, 0.0]], bin_size=10.0, units="cm"
        ).with_track_width(10.0)
        mirrored = Environment.linear_track(
            endpoints=[[120.0, 0.0], [0.0, 0.0]], bin_size=10.0, units="cm"
        ).with_track_width(10.0)
        times = np.arange(0.0, 100.0, 0.02)
        positions = np.c_[60 + 59 * np.sin(times / 3), 0 * times]
        rng = np.random.default_rng(0)
        spikes = [np.sort(rng.uniform(0, 100.0, 200)) for _ in range(5)]
        rates = compute_spatial_rates(forward, spikes, times, positions)
        with pytest.raises(IncompatibleEnvironmentError, match="differs from env"):
            rank_order_correlation(mirrored, _in_field_order(5), INTERVAL, rates)

    def test_track_numbered_out_of_order(self):
        """Field positions follow the track, not the bin numbering."""
        from environment_helpers import renumbered, strips_1d

        env = strips_1d(np.arange(0.0, 121.0, 5.0))
        order = np.random.default_rng(3).permutation(env.n_bins)
        shuffled = renumbered(env, order)
        x = env.bin_centers[:, 0]
        rates = 20.0 * np.exp(
            -0.5 * ((x - np.linspace(10.0, 110.0, 8)[:, None]) / 8.0) ** 2
        )
        result = rank_order_correlation(
            shuffled, _in_field_order(8), INTERVAL, rates[:, order]
        )
        assert result.score == pytest.approx(1.0)

    def test_rate_maps_must_be_on_env(self, track):
        env, rates, _ = track
        with pytest.raises(ValueError, match="one rate map per cell on env"):
            rank_order_correlation(env, _in_field_order(8), INTERVAL, rates[:, :5])


class TestAnalyticPValue:
    def test_exact_for_few_cells(self, track):
        """rho = 0.9 with five cells: 12 of 120 orderings are as extreme."""
        from itertools import permutations

        env, rates, _ = track
        rates5 = rates[:5]
        order = [0, 2, 1, 3, 4]  # one adjacent swap: rho = 0.9
        trains = _in_field_order(5, order)
        result = rank_order_correlation(env, trains, INTERVAL, rates5)
        assert result.score == pytest.approx(0.9)
        rhos = [
            stats.spearmanr(np.arange(5), q).statistic for q in permutations(range(5))
        ]
        expected = np.mean(np.abs(np.round(rhos, 12)) >= 0.9)
        assert result.analytic_p_value == pytest.approx(expected)
        assert result.analytic_p_value > 0.05  # the t-approximation gives 0.037

    def test_t_approximation_above_nine_pairs(self):
        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=5.0, units="cm"
        )
        x = env.bin_centers[:, 0]
        fields = np.linspace(10.0, 190.0, 12)
        rates = 20.0 * np.exp(-0.5 * ((x - fields[:, None]) / 6.0) ** 2)
        rng = np.random.default_rng(9)
        trains = [np.sort(rng.uniform(*INTERVAL, 2)) for _ in range(12)]
        result = rank_order_correlation(env, trains, INTERVAL, rates)
        peaks = env.bin_centers[np.argmax(rates, axis=1), 0]
        expected = stats.spearmanr(peaks, [np.median(t) for t in trains])
        assert result.analytic_p_value == pytest.approx(expected.pvalue, rel=1e-9)

    def test_perfect_order_of_many_cells_is_not_zero(self):
        import math

        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=5.0, units="cm"
        )
        x = env.bin_centers[:, 0]
        rates = 20.0 * np.exp(
            -0.5 * ((x - np.linspace(10.0, 190.0, 12)[:, None]) / 6.0) ** 2
        )
        result = rank_order_correlation(env, _in_field_order(12), INTERVAL, rates)
        assert result.score == pytest.approx(1.0)
        assert result.analytic_p_value == pytest.approx(2 / math.factorial(12))

    def test_calibrated_for_median_spike_times(self, track):
        """Random firing order: P(p < 0.05) is at most about 0.05 (discrete)."""
        env, rates, _ = track
        rng = np.random.default_rng(11)
        p_values = [
            rank_order_correlation(
                env,
                [np.sort(rng.uniform(*INTERVAL, 3)) for _ in range(5)],
                INTERVAL,
                rates[:5],
            ).analytic_p_value
            for _ in range(600)
        ]
        # 30 of 600 expected at most (sd 5.3).
        assert np.sum(np.asarray(p_values) < 0.05) <= 46


class TestSignificance:
    def test_cell_order_test_is_calibrated_for_bursty_all_spikes(self, track):
        """Bursts move with their cell, so all_spikes keeps alpha under this test."""
        env, rates, _ = track
        rng = np.random.default_rng(12)

        def bursty():
            return [
                np.sort(
                    rng.uniform(1.0, 1.15) + rng.uniform(0, 0.01, rng.integers(2, 7))
                )
                for _ in range(8)
            ]

        p_values = np.array(
            [
                rank_order_significance(
                    env,
                    bursty(),
                    INTERVAL,
                    rates,
                    spike_statistic="all_spikes",
                    n_shuffles=99,
                    rng=rng,
                ).p_value
                for _ in range(300)
            ]
        )
        # P(p <= 0.1) = 0.1 under the null: 30 of 300 expected (sd 5.2).
        assert 15 <= np.sum(p_values <= 0.1) <= 45

    def test_perfect_sequence_is_significant_two_sided(self, track):
        env, rates, _ = track
        for order in (np.arange(8), np.arange(8)[::-1]):
            result = rank_order_significance(
                env, _in_field_order(8, order), INTERVAL, rates, n_shuffles=999, rng=0
            )
            assert result.p_value == pytest.approx(2 / 1000)
            assert result.shuffle.shuffle_type == "cell_order"

    def test_one_sided_alternatives(self, track):
        env, rates, _ = track
        reverse = _in_field_order(8, np.arange(8)[::-1])
        greater = rank_order_significance(
            env, reverse, INTERVAL, rates, alternative="greater", n_shuffles=199, rng=0
        )
        less = rank_order_significance(
            env, reverse, INTERVAL, rates, alternative="less", n_shuffles=199, rng=0
        )
        assert greater.p_value > 0.9
        assert less.p_value == pytest.approx(1 / 200)

    def test_null_equals_spearman_of_each_permutation(self, track):
        from neurospatial.replay.rank_order import _event_pairs, _permuted_correlations

        env, rates, _ = track
        rng = np.random.default_rng(4)
        trains = [np.sort(rng.uniform(*INTERVAL, rng.integers(1, 4))) for _ in range(8)]
        for statistic in ("median", "all_spikes"):
            pairs = _event_pairs(
                env,
                trains,
                INTERVAL,
                rates,
                spike_statistic=statistic,
                field_order="peak",
                min_active_cells=5,
                operation="test",
            )
            permutations = np.array([rng.permutation(8) for _ in range(20)])
            expected = [
                stats.spearmanr(
                    pairs.field_positions[p][pairs.cell], pairs.times
                ).statistic
                for p in permutations
            ]
            assert_allclose(_permuted_correlations(pairs, permutations), expected)

    def test_random_firing_order_is_significant_at_about_alpha(self, track):
        env, rates, _ = track
        rng = np.random.default_rng(5)
        p_values = np.array(
            [
                rank_order_significance(
                    env,
                    [np.array([rng.uniform(*INTERVAL)]) for _ in range(8)],
                    INTERVAL,
                    rates,
                    n_shuffles=99,
                    rng=rng,
                ).p_value
                for _ in range(300)
            ]
        )
        # P(p <= 0.1) = 0.1 under the null: 30 of 300 expected (sd 5.2).
        assert 15 <= np.sum(p_values <= 0.1) <= 45

    def test_seeded_results_repeat(self, track):
        env, rates, _ = track
        rng = np.random.default_rng(6)
        trains = [np.array([rng.uniform(*INTERVAL)]) for _ in range(8)]
        first = rank_order_significance(
            env, trains, INTERVAL, rates, n_shuffles=50, rng=3
        )
        second = rank_order_significance(
            env, trains, INTERVAL, rates, n_shuffles=50, rng=3
        )
        assert_allclose(first.shuffle.null_scores, second.shuffle.null_scores)

    def test_summary(self, track):
        env, rates, _ = track
        result = rank_order_significance(
            env, _in_field_order(8), INTERVAL, rates, n_shuffles=49, rng=0
        )
        summary = result.summary()
        assert summary["score"] == pytest.approx(1.0)
        assert summary["p_value"] == result.p_value
        assert "z_score" in summary

    def test_undefined_correlation_is_refused(self, track):
        env, rates, _ = track
        trains = _in_field_order(8)
        for cell in range(4):
            trains[cell] = np.empty(0)
        with pytest.raises(ValueError, match="undefined"):
            rank_order_significance(env, trains, INTERVAL, rates, n_shuffles=9)

    @pytest.mark.parametrize("n_shuffles", [0, 2.5, True])
    def test_n_shuffles_must_be_a_positive_integer(self, track, n_shuffles):
        env, rates, _ = track
        with pytest.raises(ValueError, match="positive integer"):
            rank_order_significance(
                env, _in_field_order(8), INTERVAL, rates, n_shuffles=n_shuffles
            )

    def test_unknown_alternative(self, track):
        env, rates, _ = track
        with pytest.raises(ValueError, match="alternative"):
            rank_order_significance(
                env,
                _in_field_order(8),
                INTERVAL,
                rates,
                alternative="above",  # type: ignore[arg-type]
                n_shuffles=9,
            )
