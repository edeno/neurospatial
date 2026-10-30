"""Tests for the pairwise skew-bias sequence correlation (Liu et al. 2023)."""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import bin_spikes_in_time, decode_position
from neurospatial.replay import (
    BiasSimilarityResult,
    PairwiseBiasResult,
    bias_similarity,
    bias_similarity_significance,
    compare_replay_methods,
    detect_bias_sequence_replay,
    matched_alpha,
    mean_pairwise_bias,
    pairwise_bias_matrix,
)
from neurospatial.replay.methods import LIU_2023_BIAS
from neurospatial.simulation import simulate_replay_event


def _brute_force_bias(spike_times, start, stop):
    """S(i, j) = 2 n(i before j) / (n_i n_j) - 1 over every spike pair."""
    trains = [t[(t >= start) & (t < stop)] for t in spike_times]
    n = len(trains)
    bias = np.full((n, n), np.nan)
    for i, j in itertools.product(range(n), repeat=2):
        if trains[i].size == 0 or trains[j].size == 0:
            continue
        before = sum(
            1.0 if a < b else 0.5 if a == b else 0.0
            for a in trains[i]
            for b in trains[j]
        )
        bias[i, j] = 2.0 * before / (trains[i].size * trains[j].size) - 1.0
    return bias


def _random_trains(rng, n_units=7, rate=8, duration=1.0, decimals=3):
    """Spike trains on a 1 ms grid, so some spikes of different units tie."""
    return [
        np.sort(np.round(rng.uniform(0, duration, rng.poisson(rate)), decimals))
        for _ in range(n_units)
    ]


def _ordered_template(n_units):
    """Every unit fires before every later unit: S(i, j) = 1 for i < j."""
    return np.sign(np.subtract.outer(np.arange(n_units), np.arange(n_units))).T * 1.0


class TestPairwiseBiasMatrix:
    @pytest.mark.parametrize("seed", range(4))
    def test_matches_brute_force_with_ties_and_silent_units(self, seed):
        rng = np.random.default_rng(seed)
        trains = _random_trains(rng)
        trains[2] = np.array([])  # a silent unit
        result = pairwise_bias_matrix(trains, (0.1, 0.9), unit_ids=list("abcdefg"))
        assert isinstance(result, PairwiseBiasResult)
        np.testing.assert_allclose(
            result.bias, _brute_force_bias(trains, 0.1, 0.9), atol=1e-12
        )
        assert np.isnan(result.bias[2]).all() and np.isnan(result.bias[:, 2]).all()
        assert list(result.unit_ids) == list("abcdefg")
        assert result.summary()["n_active_cells"] == int(np.sum(result.n_spikes > 0))

    @pytest.mark.parametrize("seed", range(4))
    def test_range_is_minus_one_to_one_and_skew_symmetric(self, seed):
        """2B - 1, not the printed 2B - 2: a pair spans [-1, 1], 0 at chance."""
        bias = pairwise_bias_matrix(
            _random_trains(np.random.default_rng(10 + seed), n_units=12), (0.0, 1.0)
        ).bias
        finite = bias[np.isfinite(bias)]
        assert finite.min() >= -1.0 and finite.max() <= 1.0
        np.testing.assert_allclose(bias, -bias.T, atol=1e-12)

    def test_extremes_and_chance(self):
        ordered = [np.array([0.01, 0.02]), np.array([0.05, 0.06])]
        interleaved = [np.array([0.01, 0.04]), np.array([0.02, 0.03])]
        assert pairwise_bias_matrix(ordered, (0.0, 1.0)).bias[0, 1] == 1.0
        assert pairwise_bias_matrix(ordered, (0.0, 1.0)).bias[1, 0] == -1.0
        assert pairwise_bias_matrix(interleaved, (0.0, 1.0)).bias[0, 1] == 0.0

    def test_interval_is_half_open(self):
        trains = [np.array([0.0, 0.5]), np.array([0.2, 1.0])]
        result = pairwise_bias_matrix(trains, (0.0, 1.0))
        assert list(result.n_spikes) == [2, 1]

    def test_refusals(self):
        trains = [np.array([0.1]), np.array([0.2])]
        with pytest.raises(ValueError, match="interval"):
            pairwise_bias_matrix(trains, (1.0, 0.0))
        with pytest.raises(ValueError, match="unit_ids"):
            pairwise_bias_matrix(trains, (0.0, 1.0), unit_ids=[1, 2, 3])


class TestMeanPairwiseBias:
    def test_mean_over_events_where_both_units_fired(self):
        rng = np.random.default_rng(5)
        trains = _random_trains(rng, n_units=6, rate=20, duration=2.0)
        trains[4] = trains[4][trains[4] < 0.5]  # unit 4 fires in the first cycle only
        cycles = np.array([[0.0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0]])
        stack = np.stack([_brute_force_bias(trains, *cycle) for cycle in cycles])
        expected = np.nanmean(stack, axis=0)  # NaN where both never fired
        np.testing.assert_allclose(
            mean_pairwise_bias(trains, cycles), expected, atol=1e-12
        )

    def test_refuses_bad_intervals(self):
        with pytest.raises(ValueError, match="intervals"):
            mean_pairwise_bias([np.array([0.1])], np.array([0.0, 1.0]))


class TestBiasSimilarity:
    def test_cosine_over_pairs_defined_in_both(self):
        template = _ordered_template(4)
        event = np.full((4, 4), np.nan)
        event[0, 1], event[1, 0] = 0.5, -0.5
        event[2, 3], event[3, 2] = -1.0, 1.0
        event[0, 0] = event[1, 1] = event[2, 2] = event[3, 3] = 0.0
        # Pairs (0, 1) and (2, 3): (1 * 0.5 + 1 * -1) / (sqrt(2) * sqrt(1.25))
        expected = -0.5 / (np.sqrt(2.0) * np.sqrt(1.25))
        assert bias_similarity(template, event) == pytest.approx(expected)
        assert bias_similarity(template, template) == 1.0
        assert bias_similarity(template, -template) == -1.0

    def test_undefined_without_common_pairs(self):
        template = np.full((3, 3), np.nan)
        assert np.isnan(bias_similarity(template, _ordered_template(3)))

    @pytest.mark.parametrize(
        "matrix, problem",
        [
            (np.ones((2, 3)), "shape"),
            (np.array([[0.0, 2.0], [-2.0, 0.0]]), r"\[-1, 1\]"),
            (np.array([[0.0, 0.5], [0.5, 0.0]]), "skew-symmetric"),
        ],
    )
    def test_refuses_matrices_that_are_not_skew_bias(self, matrix, problem):
        with pytest.raises(ValueError, match=problem):
            bias_similarity(matrix, matrix)


class TestSignificance:
    def test_shuffles_keep_spike_times_and_counts(self):
        trains = [np.array([0.01 * (i + 1)]) for i in range(10)]
        result = bias_similarity_significance(
            trains, (0.0, 0.2), _ordered_template(10), n_shuffles=50, rng=0
        )
        assert isinstance(result, BiasSimilarityResult)
        assert result.similarity == 1.0
        assert result.p_value == pytest.approx(1 / 51)
        assert result.n_pairs == 45 and result.n_active_cells == 10
        assert result.null_similarities.shape == (50,)
        assert np.all(np.abs(result.null_similarities) <= 1.0)

    def test_false_positive_rate_is_held_at_alpha(self):
        """Random firing order: the permutation test rejects at most alpha.

        With labels exchangeable given the spike times, P(p < alpha) <=
        alpha; 400 events bound the rate within 3 binomial SDs.
        """
        rng = np.random.default_rng(7)
        n_units, n_events = 10, 400
        template = _ordered_template(n_units)
        rejected = 0
        for event in range(n_events):
            trains = [rng.uniform(0.0, 0.1, rng.integers(1, 4)) for _ in range(n_units)]
            p = bias_similarity_significance(
                trains, (0.0, 0.1), template, n_shuffles=100, rng=event
            ).p_value
            rejected += p < 0.05
        bound = 0.05 * n_events + 3 * np.sqrt(n_events * 0.05 * 0.95)
        assert rejected <= bound

    def test_absolute_test_needs_greater(self):
        with pytest.raises(ValueError, match="alternative='greater'"):
            bias_similarity_significance(
                [np.array([0.1]), np.array([0.2])],
                (0.0, 1.0),
                _ordered_template(2),
                alternative="two-sided",
            )

    def test_unit_count_must_match_the_template(self):
        with pytest.raises(ValueError, match="template_bias"):
            bias_similarity_significance(
                [np.array([0.1]), np.array([0.2])], (0.0, 1.0), _ordered_template(3)
            )


# --------------------------------------------------------------------------
# The recipe on simulated sessions
# --------------------------------------------------------------------------

N_CELLS = 30
DT = 0.02


@pytest.fixture(scope="module")
def session():
    """Theta-like forward sweeps while running, then candidate sleep events.

    The template is the mean skew bias over 20 running sweeps. Sleep events
    are two forward sweeps, two reverse sweeps and four random events.
    """
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [150.0, 0.0]], bin_size=5.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    rates = 0.2 + 30.0 * np.exp(
        -0.5 * ((x - np.linspace(0.0, 150.0, N_CELLS)[:, None]) / 8.0) ** 2
    )
    spikes = [[] for _ in range(N_CELLS)]
    cycles = []
    offset = 0.0
    for seed in range(20):
        event = simulate_replay_event(
            env,
            rates,
            kind="sweep",
            speed=600.0,
            duration=0.24,
            rate_gain=2.0,
            seed=seed,
        )
        for cell, train in enumerate(event.spike_times):
            spikes[cell].append(train + offset)
        cycles.append((offset, offset + event.duration))
        offset += 1.0
    kinds = [
        "sweep",
        "sweep",
        "reverse",
        "reverse",
        "random",
        "random",
        "random",
        "random",
    ]
    intervals = []
    for index, kind in enumerate(kinds):
        options = (
            {"kind": "random"}
            if kind == "random"
            else {
                "kind": "sweep",
                "speed": 600.0 if kind == "sweep" else -600.0,
            }
        )
        event = simulate_replay_event(
            env, rates, duration=0.24, rate_gain=4.0, seed=100 + index, **options
        )
        for cell, train in enumerate(event.spike_times):
            spikes[cell].append(train + offset)
        intervals.append((offset, offset + event.duration))
        offset += 1.0
    spike_times = [np.concatenate(trains) for trains in spikes]
    template = mean_pairwise_bias(spike_times, np.array(cycles))
    events = []
    for start, stop in intervals:
        counts, times = bin_spikes_in_time(
            spike_times, dt=DT, t_start=start, t_stop=stop
        )
        events.append(decode_position(env, counts, rates, dt=DT, times=times))
    return env, spike_times, template, events, np.array(kinds), intervals


def test_recipe_finds_forward_and_reverse_sequences_in_compare(session):
    env, spike_times, template, events, kinds, _ = session
    method = LIU_2023_BIAS
    table = compare_replay_methods(
        env,
        events,
        replay_methods=[method],
        spike_times=spike_times,
        template_bias=template,
        dt=DT,
        rng=0,
    )
    assert (table["error"] == "").all()
    rows = table.set_index("event")
    assert rows["is_replay"][kinds != "random"].all()
    assert (rows["score"][kinds == "sweep"] > 0.5).all()
    assert (rows["score"][kinds == "reverse"] < -0.5).all()
    assert rows["is_replay"][kinds == "random"].sum() <= 1
    assert np.allclose(rows["p_value"][kinds != "random"], 1 / 101)
    # The table's verdicts can be recalibrated like any p-value test.
    calibrated = matched_alpha(table, replay_methods=[method], target=0.5)
    assert calibrated.loc[0, "method"] == method.name


def test_compare_matches_detecting_each_event(session):
    env, spike_times, template, events, _, _ = session
    table = compare_replay_methods(
        env,
        events[:2],
        replay_methods=[LIU_2023_BIAS],
        spike_times=spike_times,
        template_bias=template,
        dt=DT,
        rng=0,
    )
    times = events[1].times
    direct = detect_bias_sequence_replay(
        spike_times,
        (times[0] - DT / 2, times[-1] + DT / 2),
        template,
        bias_sequence_method=LIU_2023_BIAS,
    )
    assert table.loc[1, "score"] == direct.score
    assert table.loc[1, "passes[n_active_cells >= 2]"]


def test_compare_needs_the_template(session):
    env, spike_times, _, events, _, _ = session
    with pytest.raises(ValueError, match="template_bias"):
        compare_replay_methods(
            env, events, replay_methods=[LIU_2023_BIAS], spike_times=spike_times, dt=DT
        )


def test_event_without_common_pairs_is_an_error_row(session):
    env, spike_times, template, events, _, _ = session
    silent = np.full_like(template, np.nan)
    table = compare_replay_methods(
        env,
        events[:1],
        replay_methods=[dataclasses.replace(LIU_2023_BIAS, n_shuffles=10)],
        spike_times=spike_times,
        template_bias=silent,
        dt=DT,
        rng=0,
    )
    assert not table.loc[0, "is_replay"]
    assert "undefined" in table.loc[0, "error"]
