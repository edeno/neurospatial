"""Pairwise firing-order score and test of Gupta et al. (2010)."""

from __future__ import annotations

import copy
import dataclasses
import itertools
import pickle
import zlib

import numpy as np
import pytest
from replay_helpers import template_stream

from neurospatial import Environment
from neurospatial.decoding import DecodingResult
from neurospatial.replay import (
    PairwiseOrderMethod,
    compare_replay_methods,
    detect_pairwise_order_replay,
    pairwise_order_score,
    pairwise_order_significance,
)
from neurospatial.replay import pairwise_order as module
from neurospatial.replay.methods import GUPTA_2010


@pytest.fixture(scope="module")
def track():
    """A 200 cm track in 10 cm bins: bin ``b`` is centered at 10 b + 5 cm."""
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=10.0, units="cm"
    )


def _rates(env, field_bins):
    """One rate map per cell, 10 Hz in each of its field bins, 0 elsewhere."""
    rates = np.zeros((len(field_bins), env.n_bins))
    for cell, bins in enumerate(field_bins):
        rates[cell, list(bins)] = 10.0
    return rates


def _brute_force(trains, centers, lag, sign=1.0, groups=None):
    """Largest score over one field per cell, from a double loop over spikes.

    ``centers[u]`` lists cell ``u``'s field centers (empty for none). With
    ``groups``, pairs of spikes from cells in one group are skipped.
    """
    spikes = [
        (t, u) for u, train in enumerate(trains) for t in train if len(centers[u])
    ]
    scored = sorted({u for _, u in spikes})
    best = -np.inf
    for choice in itertools.product(*(centers[u] for u in scored)):
        position = dict(zip(scored, choice, strict=True))
        total = 0.0
        for (ti, ui), (tj, uj) in itertools.combinations(spikes, 2):
            if groups is not None and groups[ui] == groups[uj]:
                continue
            if abs(tj - ti) < lag:
                total += np.sign(tj - ti) * np.sign(position[uj] - position[ui])
        best = max(best, sign * total)
    return 0.0 if best == -np.inf else best


class TestScore:
    def test_three_cells_in_order(self, track):
        """Spikes at 0, 10, 20 ms from fields at 15, 25, 35 cm: S = 3."""
        rates = _rates(track, [[1], [2], [3]])
        trains = [np.array([1.00]), np.array([1.01]), np.array([1.02])]
        forward = pairwise_order_score(track, trains, (1.0, 1.1), rates)
        backward = pairwise_order_score(
            track, trains, (1.0, 1.1), rates, direction="backward"
        )
        assert (forward.score, forward.n_pairs) == (3, 3)
        assert backward.score == -3
        np.testing.assert_array_equal(forward.assigned_positions, [15.0, 25.0, 35.0])

    def test_pairs_80_ms_apart_or_more_are_not_compared(self, track):
        rates = _rates(track, [[1], [2], [3]])
        # 0.25 and 0.33 are 80 ms apart up to rounding above 0.08.
        late = [np.array([0.25]), np.array([0.26]), np.array([0.33])]
        result = pairwise_order_score(track, late, (0.2, 0.5), rates)
        assert (result.score, result.n_pairs) == (2, 2)
        # Exactly 0.08 s apart in binary floating point: not compared.
        exact = [np.array([0.0]), np.array([0.0625]), np.array([0.08])]
        assert pairwise_order_score(track, exact, (0.0, 0.5), rates).n_pairs == 2
        far = [np.array([1.00]), np.array([1.01]), np.array([1.11])]
        result = pairwise_order_score(track, far, (1.0, 1.2), rates)
        assert (result.score, result.n_pairs) == (1, 1)
        everything = pairwise_order_score(
            track, far, (1.0, 1.2), rates, max_time_lag=np.inf
        )
        assert everything.score == 3

    def test_matches_a_brute_force_pair_count(self, track):
        """Several spikes per cell, ties in time and field position."""
        rng = np.random.default_rng(3)
        field_bins = [[b] for b in rng.integers(0, 20, 12)]
        rates = _rates(track, field_bins)
        centers = [[10.0 * b[0] + 5.0] for b in field_bins]
        for seed in range(5):
            spikes = np.random.default_rng(seed)
            trains = [
                np.round(np.sort(spikes.uniform(0.0, 0.3, spikes.poisson(3))), 2)
                for _ in range(12)
            ]
            for direction, sign in (("forward", 1.0), ("backward", -1.0)):
                result = pairwise_order_score(
                    track, trains, (0.0, 0.5), rates, direction=direction
                )
                assert result.score == _brute_force(trains, centers, 0.08, sign)

    def test_one_spike_per_cell_without_a_lag_is_kendalls_s(self, track):
        rng = np.random.default_rng(0)
        order = rng.permutation(15)
        rates = _rates(track, [[b] for b in range(15)])
        trains = [np.array([1.0 + 0.01 * order[cell]]) for cell in range(15)]
        result = pairwise_order_score(
            track, trains, (1.0, 2.0), rates, max_time_lag=np.inf
        )
        concordant_minus_discordant = sum(
            np.sign(order[j] - order[i])
            for i, j in itertools.combinations(range(15), 2)
        )
        assert result.score == concordant_minus_discordant
        assert result.n_pairs == 105

    def test_cells_without_a_field_or_spike_add_nothing(self, track):
        rates = _rates(track, [[1], [], [3], [5]])
        trains = [np.array([1.00]), np.array([1.01]), np.array([1.02]), np.array([])]
        result = pairwise_order_score(track, trains, (1.0, 1.1), rates)
        assert (result.score, result.n_pairs, result.n_active_cells) == (1, 1, 3)
        np.testing.assert_array_equal(result.scored_units, [0, 2])


class TestMultipleFields:
    def test_a_cell_takes_the_field_that_fits_the_sequence(self, track):
        """Fields at 15 and 95 cm, firing between cells at 85 and 105 cm."""
        rates = _rates(track, [[8], [1, 9], [10]])
        trains = [np.array([1.00]), np.array([1.01]), np.array([1.02])]
        result = pairwise_order_score(track, trains, (1.0, 1.1), rates)
        assert result.score == 3
        np.testing.assert_array_equal(result.assigned_positions, [85.0, 95.0, 105.0])
        backward = pairwise_order_score(
            track, trains, (1.0, 1.1), rates, direction="backward"
        )
        # Backward the cell takes 15 cm: before both others, +1 - 1 - 1.
        assert backward.score == -1
        assert backward.assigned_positions[1] == 15.0

    def test_matches_a_brute_force_search(self, track):
        """Four multi-field cells among eight, many spikes each."""
        field_bins = [[2, 12], [5], [8, 15, 18], [0], [10, 3], [14], [6, 17], [11]]
        rates = _rates(track, field_bins)
        centers = [sorted(10.0 * b + 5.0 for b in bins) for bins in field_bins]
        for seed in range(4):
            rng = np.random.default_rng(seed)
            trains = [
                np.sort(rng.uniform(0.0, 0.2, rng.poisson(3))) for _ in field_bins
            ]
            for direction, sign in (("forward", 1.0), ("backward", -1.0)):
                result = pairwise_order_score(
                    track, trains, (0.0, 0.2), rates, direction=direction
                )
                assert result.score == _brute_force(trains, centers, 0.08, sign)

    def test_too_many_combinations_are_refused(self, track):
        rates = _rates(track, [[2 * k, 2 * k + 10] for k in range(5)] * 4)
        trains = [np.array([1.0 + 0.001 * k]) for k in range(20)]
        with pytest.raises(ValueError, match="1,048,576 combinations"):
            pairwise_order_score(track, trains, (1.0, 1.1), rates)
        # The same cells without their spikes are not searched.
        quiet = trains[:16] + [np.array([])] * 4
        pairwise_order_score(track, quiet, (1.0, 1.1), rates)


class TestFields:
    def test_threshold_is_a_fraction_of_the_population_peak(self, track):
        """A cell peaking at 4% of the highest rate has no field."""
        rates = np.zeros((3, track.n_bins))
        rates[0, 1] = 100.0
        rates[1, 2] = 4.0
        rates[2, 3] = 6.0
        trains = [np.array([1.00]), np.array([1.01]), np.array([1.02])]
        result = pairwise_order_score(track, trains, (1.0, 1.1), rates)
        np.testing.assert_array_equal(result.scored_units, [0, 2])

    def test_runs_one_bin_apart_merge(self, track):
        rates = np.zeros((3, track.n_bins))
        rates[0, [2, 4]] = 10.0  # one bin apart: one field centered at 35 cm
        rates[1, [2, 5]] = 10.0  # two bins apart: fields at 25 and 55 cm
        rates[2, 19] = 10.0
        trains = [np.array([1.00]), np.array([1.01]), np.array([1.02])]
        result = pairwise_order_score(track, trains, (1.0, 1.1), rates)
        assert result.assigned_positions[0] == 35.0
        assert result.assigned_positions[1] in (25.0, 55.0)
        merged = pairwise_order_score(
            track, trains, (1.0, 1.1), rates, merge_gap_bins=2
        )
        # Rate-weighted over 25-55 cm: the gap bins have no rate.
        assert merged.assigned_positions[1] == 40.0

    def test_peak_and_rate_weighted_centers(self, track):
        rates = np.zeros((2, track.n_bins))
        rates[0, [1, 2, 3]] = [2.0, 10.0, 8.0]
        rates[1, 10] = 10.0
        trains = [np.array([1.00]), np.array([1.01])]
        weighted = pairwise_order_score(track, trains, (1.0, 1.1), rates)
        assert weighted.assigned_positions[0] == pytest.approx(
            (15 * 2 + 25 * 10 + 35 * 8) / 20
        )
        peak = pairwise_order_score(
            track, trains, (1.0, 1.1), rates, field_center="peak"
        )
        assert peak.assigned_positions[0] == 25.0

    def test_bins_without_data_hold_no_field(self, track):
        rates = _rates(track, [[1], [2], [3]])
        rates[1] = np.nan
        trains = [np.array([1.00]), np.array([1.01]), np.array([1.02])]
        result = pairwise_order_score(track, trains, (1.0, 1.1), rates)
        np.testing.assert_array_equal(result.scored_units, [0, 2])


def _recorded(monkeypatch, name):
    """Record the arguments and results of a private function of the module."""
    calls = []
    original = getattr(module, name)

    def spy(*args):
        value = original(*args)
        calls.append((args, value))
        return value

    monkeypatch.setattr(module, name, spy)
    return calls


class TestNulls:
    @pytest.fixture
    def event(self, track):
        """Ten cells, three with two fields, two without a field, one silent."""
        field_bins = [[1], [3, 12], [5], [7, 16], [], [9], [11, 2], [13], [], [15]]
        rng = np.random.default_rng(11)
        trains = [np.sort(rng.uniform(1.0, 1.15, rng.poisson(3))) for _ in field_bins]
        trains[4] = np.array([1.05, 1.06])  # active without a field
        trains[9] = np.array([])  # silent, with a field at 155 cm
        centers = [sorted(10.0 * b + 5.0 for b in bins) for bins in field_bins]
        return _rates(track, field_bins), trains, centers

    def test_spike_time_permutation_keeps_cells_counts_and_times(
        self, track, event, monkeypatch
    ):
        rates, trains, centers = event
        calls = _recorded(monkeypatch, "_order_weights")
        result = pairwise_order_significance(
            track,
            trains,
            (1.0, 1.15),
            rates,
            nulls=["spike_time_permutation"],
            n_shuffles=20,
            rng=1,
        )
        active = [u for u, train in enumerate(trains) if len(train)]
        scored = [u for u in active if centers[u]]
        # Scored spike times, and each scored cell's count by active index.
        times = np.sort(np.concatenate([trains[u] for u in scored]))
        counts = np.repeat(
            [active.index(u) for u in scored], [len(trains[u]) for u in scored]
        )
        null = result.shuffles["spike_time_permutation"].null_scores
        assert len(calls) == 21
        changed = 0
        for k, (args, _) in enumerate(calls[1:]):
            labels = args[3]
            # Each cell keeps its number of spikes, on the scored spike times.
            np.testing.assert_array_equal(np.sort(labels), counts)
            shuffled = [[] for _ in trains]
            for t, label in zip(times, labels, strict=True):
                shuffled[active[label]].append(t)
            changed += any(not np.array_equal(shuffled[u], trains[u]) for u in scored)
            # Each shuffle is scored with its own best fields.
            assert null[k] == _brute_force(shuffled, centers, 0.08)
        assert changed == 20

    def test_cell_identity_keeps_the_trains_and_permutes_fields(
        self, track, event, monkeypatch
    ):
        rates, trains, centers = event
        calls = _recorded(monkeypatch, "_best_assignment")
        result = pairwise_order_significance(
            track,
            trains,
            (1.0, 1.15),
            rates,
            nulls=("cell_identity",),
            n_shuffles=30,
            rng=2,
        )
        observed_weights = calls[0][0][0]
        active = [u for u, train in enumerate(trains) if len(train)]
        assert not centers[4] and 4 in active  # an active cell without a field
        field_sets = [tuple(c) for c in centers if len(c)]
        null = result.shuffles["cell_identity"].null_scores
        silent_field_taken = 0
        for k, ((weights, fields, _), _) in enumerate(calls[1:]):
            np.testing.assert_array_equal(weights, observed_weights)
            given = [tuple(f) for f in fields]
            # Active cells with a field take distinct field sets of cells
            # with a field; cells without a field stay without one.
            remaining = list(field_sets)
            for unit, field_set in zip(active, given, strict=True):
                if centers[unit]:
                    remaining.remove(field_set)
                else:
                    assert field_set == ()
            shuffled_centers = [[] for _ in trains]
            for unit, field_set in zip(active, given, strict=True):
                shuffled_centers[unit] = list(field_set)
            assert null[k] == _brute_force(trains, shuffled_centers, 0.08)
            silent_field_taken += (155.0,) in given
        assert len(calls) == 31
        # A silent cell's fields are drawn too, not only the active cells'.
        assert silent_field_taken > 0

    def test_cells_without_a_field_change_neither_null(self, track, event):
        """Fieldless cells, silent or active, leave every shuffle unchanged."""
        rates, trains, centers = event
        with_field = [u for u, c in enumerate(centers) if len(c)]
        field_only = pairwise_order_significance(
            track,
            [trains[u] for u in with_field],
            (1.0, 1.15),
            rates[with_field],
            n_shuffles=50,
            rng=3,
        )
        # The fixture's two fieldless cells (one active), and 20 silent ones.
        padded = pairwise_order_significance(
            track,
            [*trains, *[np.array([])] * 20],
            (1.0, 1.15),
            np.vstack([rates, np.zeros((20, track.n_bins))]),
            n_shuffles=50,
            rng=3,
        )
        assert padded.observed.score == field_only.observed.score
        for null in ("spike_time_permutation", "cell_identity"):
            np.testing.assert_array_equal(
                padded.shuffles[null].null_scores,
                field_only.shuffles[null].null_scores,
            )

    def test_each_null_alone_holds_alpha_with_fieldless_cells(self):
        """Random firing order, with silent and active cells without a field.

        Cells without a field once joined the cell-identity pool, which made
        most shuffles score 0 and called about a third of these events
        significant at alpha = 0.05.
        """
        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=3.5, units="cm"
        )
        x = env.bin_centers[:, 0]
        rng = np.random.default_rng(0)
        n_events = 300
        rejected = dict.fromkeys(("spike_time_permutation", "cell_identity"), 0)
        for seed in range(n_events):
            centers = rng.uniform(10.0, 190.0, 12)
            rates = np.vstack(
                [
                    20.0 * np.exp(-0.5 * ((x - centers[:, None]) / 5.0) ** 2),
                    np.zeros((42, env.n_bins)),
                ]
            )
            trains = [
                *[
                    np.sort(rng.uniform(0.0, 0.15, rng.integers(1, 4)))
                    for _ in range(12)
                ],
                *[np.sort(rng.uniform(0.0, 0.15, 3)) for _ in range(6)],
                *[np.array([])] * 36,
            ]
            result = pairwise_order_significance(
                env, trains, (0.0, 0.15), rates, n_shuffles=100, rng=seed
            )
            for null, test in result.shuffles.items():
                rejected[null] += test.p_value < 0.05
        # Binomial(300, 0.05) lies in [6, 30] with probability above 99%.
        for null, count in rejected.items():
            assert 6 <= count <= 30, (null, count)

    def test_nulls_do_not_depend_on_each_other(self, track, event):
        rates, trains, _ = event
        both = pairwise_order_significance(track, trains, (1.0, 1.15), rates, rng=5)
        alone = pairwise_order_significance(
            track, trains, (1.0, 1.15), rates, nulls=("cell_identity",), rng=5
        )
        np.testing.assert_array_equal(
            both.shuffles["cell_identity"].null_scores,
            alone.shuffles["cell_identity"].null_scores,
        )
        assert both.p_value == max(t.p_value for t in both.shuffles.values())


class TestUnitGroups:
    """Only pairs of spikes from cells in different groups are compared."""

    @pytest.fixture
    def event(self, track):
        """Ten cells on four tetrodes, two with two fields, one without."""
        field_bins = [[1], [3, 12], [5], [7], [], [9], [11, 2], [13], [15], [17]]
        rng = np.random.default_rng(21)
        trains = [np.sort(rng.uniform(1.0, 1.15, rng.poisson(3))) for _ in field_bins]
        trains[4] = np.array([1.05, 1.06])  # active without a field
        centers = [sorted(10.0 * b + 5.0 for b in bins) for bins in field_bins]
        tetrodes = np.array(["a", "a", "a", "b", "b", "c", "c", "c", "d", "d"])
        return _rates(track, field_bins), trains, centers, tetrodes

    def test_score_and_pairs_match_a_hand_filtered_count(self, track, event):
        rates, trains, centers, tetrodes = event
        for direction, sign in (("forward", 1.0), ("backward", -1.0)):
            result = pairwise_order_score(
                track,
                trains,
                (1.0, 1.15),
                rates,
                direction=direction,
                unit_groups=tetrodes,
            )
            assert result.score == _brute_force(
                trains, centers, 0.08, sign, groups=tetrodes
            )
        spikes = [(t, u) for u, train in enumerate(trains) for t in train]
        n_pairs = sum(
            bool(
                abs(tj - ti) < 0.08
                and len(centers[ui]) > 0
                and len(centers[uj]) > 0
                and tetrodes[ui] != tetrodes[uj]
            )
            for (ti, ui), (tj, uj) in itertools.combinations(spikes, 2)
        )
        assert result.n_pairs == n_pairs
        everything = pairwise_order_score(track, trains, (1.0, 1.15), rates)
        assert n_pairs < everything.n_pairs

    def test_one_group_per_cell_changes_nothing(self, track, event):
        rates, trains, _, _ = event
        kwargs = {"n_shuffles": 25, "rng": 3}
        grouped = pairwise_order_significance(
            track, trains, (1.0, 1.15), rates, unit_groups=np.arange(10), **kwargs
        )
        plain = pairwise_order_significance(track, trains, (1.0, 1.15), rates, **kwargs)
        assert grouped.observed.summary() == plain.observed.summary()
        for null in plain.shuffles:
            np.testing.assert_array_equal(
                grouped.shuffles[null].null_scores, plain.shuffles[null].null_scores
            )

    def test_one_group_for_every_cell_compares_nothing(self, track, event):
        rates, trains, _, _ = event
        result = pairwise_order_score(
            track, trains, (1.0, 1.15), rates, unit_groups=["a"] * 10
        )
        assert (result.score, result.n_pairs) == (0, 0)

    def test_spike_time_permutation_keeps_each_trains_group(
        self, track, event, monkeypatch
    ):
        """A shuffled spike counts with the group of the cell it now belongs to."""
        rates, trains, centers, tetrodes = event
        calls = _recorded(monkeypatch, "_order_weights")
        result = pairwise_order_significance(
            track,
            trains,
            (1.0, 1.15),
            rates,
            nulls=["spike_time_permutation"],
            n_shuffles=15,
            unit_groups=tetrodes,
            rng=4,
        )
        active = [u for u, train in enumerate(trains) if len(train)]
        scored = [u for u in active if centers[u]]
        times = np.sort(np.concatenate([trains[u] for u in scored]))
        null = result.shuffles["spike_time_permutation"].null_scores
        for k, (args, _) in enumerate(calls[1:]):
            shuffled = [[] for _ in trains]
            for t, label in zip(times, args[3], strict=True):
                shuffled[active[label]].append(t)
            assert null[k] == _brute_force(shuffled, centers, 0.08, groups=tetrodes)

    def test_cell_identity_moves_fields_not_groups(self, track, event, monkeypatch):
        rates, trains, _, tetrodes = event
        calls = _recorded(monkeypatch, "_best_assignment")
        result = pairwise_order_significance(
            track,
            trains,
            (1.0, 1.15),
            rates,
            nulls=["cell_identity"],
            n_shuffles=15,
            unit_groups=tetrodes,
            rng=5,
        )
        active = [u for u, train in enumerate(trains) if len(train)]
        null = result.shuffles["cell_identity"].null_scores
        for k, ((_, fields, _), _) in enumerate(calls[1:]):
            shuffled_centers = [[] for _ in trains]
            for unit, field_set in zip(active, fields, strict=True):
                shuffled_centers[unit] = list(field_set)
            assert null[k] == _brute_force(
                trains, shuffled_centers, 0.08, groups=tetrodes
            )

    def test_detection_compares_only_pairs_across_groups(self):
        env, rates = _place_code()
        trains = _sweep(30, np.random.default_rng(6))
        tetrodes = np.arange(30) // 4
        result = detect_pairwise_order_replay(
            env,
            trains,
            (0.9, 1.2),
            rates,
            pairwise_order_method=GUPTA_2010,
            unit_groups=tetrodes,
            rng=0,
        )
        forward = pairwise_order_score(
            env, trains, (0.9, 1.2), rates, unit_groups=tetrodes
        )
        assert result.score == forward.score
        assert result.is_replay
        plain = pairwise_order_score(env, trains, (0.9, 1.2), rates)
        assert forward.score < plain.score

    @pytest.mark.parametrize("groups", [np.zeros(9), np.zeros((10, 1))])
    def test_one_group_per_cell_is_required(self, track, event, groups):
        rates, trains, _, _ = event
        with pytest.raises(ValueError, match="one group per cell"):
            pairwise_order_score(track, trains, (1.0, 1.15), rates, unit_groups=groups)


def _place_code(n_cells=30):
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=4.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    centers = np.linspace(5.0, 195.0, n_cells)
    rates = 20.0 * np.exp(-0.5 * ((x - centers[:, None]) / 8.0) ** 2)
    return env, rates


def _sweep(n_cells, rng, start=1.0, step=0.004, reverse=False):
    order = np.arange(n_cells)[::-1] if reverse else np.arange(n_cells)
    trains = [np.array([])] * n_cells
    for k, cell in enumerate(order):
        n = rng.poisson(1.5) + 1
        trains[cell] = np.sort(start + step * k + rng.normal(0.0, 0.003, n))
    return trains


def _random_event(n_cells, rng, start=1.0, duration=0.15):
    return [
        np.sort(rng.uniform(start, start + duration, rng.poisson(1.0)))
        for _ in range(n_cells)
    ]


class TestDetection:
    def test_sweeps_are_replay_in_their_direction(self):
        env, rates = _place_code()
        rng = np.random.default_rng(0)
        forward = detect_pairwise_order_replay(
            env,
            _sweep(30, rng),
            (0.9, 1.2),
            rates,
            pairwise_order_method=GUPTA_2010,
            rng=0,
        )
        backward = detect_pairwise_order_replay(
            env,
            _sweep(30, rng, reverse=True),
            (0.9, 1.2),
            rates,
            pairwise_order_method=GUPTA_2010,
            rng=0,
        )
        assert forward.is_replay and forward.score > 0
        assert backward.is_replay and backward.score < 0
        assert forward.p_value == pytest.approx(1 / 301)
        assert forward.n_shuffles_used == 300

    def test_random_events_are_rarely_replay(self):
        """Per test about alpha; over two directions at most about 2 alpha."""
        env, rates = _place_code()
        forward_only = dataclasses.replace(
            GUPTA_2010, name="forward_only", directions=("forward",)
        )
        rng = np.random.default_rng(7)
        n_events = 100
        forward = both = 0
        for seed in range(n_events):
            trains = _random_event(30, rng)
            forward += detect_pairwise_order_replay(
                env,
                trains,
                (1.0, 1.15),
                rates,
                pairwise_order_method=forward_only,
                rng=seed,
            ).is_replay
            both += detect_pairwise_order_replay(
                env,
                trains,
                (1.0, 1.15),
                rates,
                pairwise_order_method=GUPTA_2010,
                rng=seed,
            ).is_replay
        # Binomial(100, 0.05) exceeds 12 with probability below 0.1%.
        assert forward / n_events <= 0.12
        assert both / n_events <= 0.2

    def test_the_test_with_the_smallest_p_is_reported(self):
        """A sweep in the order of template "right" only."""
        env, rates = _place_code()
        right = rates[::-1].copy()
        rng = np.random.default_rng(3)
        trains = _sweep(30, rng, reverse=True)
        result = detect_pairwise_order_replay(
            env,
            trains,
            (0.9, 1.2),
            {"left": rates, "right": right},
            pairwise_order_method=GUPTA_2010,
            rng=0,
        )
        # Reversed on "left" is forward on "right"; "left" backward ties it,
        # and "left" comes first.
        assert result.template == "left" and result.score < 0
        forward_only = dataclasses.replace(
            GUPTA_2010, name="forward", directions=("forward",)
        )
        result = detect_pairwise_order_replay(
            env,
            trains,
            (0.9, 1.2),
            {"left": rates, "right": right},
            pairwise_order_method=forward_only,
            rng=0,
        )
        assert result.template == "right" and result.is_replay and result.score > 0

    def test_p_value_is_that_of_the_tests_own_stream(self, track):
        rates = _rates(track, [[b] for b in range(0, 20, 2)])
        rng = np.random.default_rng(4)
        trains = [np.sort(rng.uniform(1.0, 1.1, 2)) for _ in range(10)]
        method = dataclasses.replace(
            GUPTA_2010, name="forward", directions=("forward",), n_shuffles=50
        )
        result = detect_pairwise_order_replay(
            track, trains, (1.0, 1.1), rates, pairwise_order_method=method, rng=9
        )
        expected = pairwise_order_significance(
            track,
            trains,
            (1.0, 1.1),
            rates,
            n_shuffles=50,
            rng=template_stream(9, "forward"),
        )
        assert result.p_value == expected.p_value
        assert result.score == expected.observed.score

    def test_fewer_than_three_active_cells_are_not_replay(self, track):
        rates = _rates(track, [[1], [2], [3]])
        trains = [np.array([1.00, 1.02]), np.array([1.01, 1.03]), np.array([])]
        method = dataclasses.replace(GUPTA_2010, name="alpha_one", alpha=0.999)
        result = detect_pairwise_order_replay(
            track, trains, (1.0, 1.1), rates, pairwise_order_method=method, rng=0
        )
        assert result.criteria == {"n_active_cells >= 3": False}
        assert result.values == {"n_active_cells >= 3": 2.0}
        assert not result.is_replay

    def test_template_needs_a_mapping(self, track):
        rates = _rates(track, [[1], [2], [3]])
        trains = [np.array([1.0])] * 3
        with pytest.raises(ValueError, match="needs a mapping"):
            detect_pairwise_order_replay(
                track,
                trains,
                (1.0, 1.1),
                rates,
                pairwise_order_method=GUPTA_2010,
                template="left",
            )
        with pytest.raises(ValueError, match="not one of"):
            detect_pairwise_order_replay(
                track,
                trains,
                (1.0, 1.1),
                {"left": rates},
                pairwise_order_method=GUPTA_2010,
                template="right",
            )


class TestCompare:
    def test_recipe_runs_through_compare_replay_methods(self):
        env, rates = _place_code()
        rng = np.random.default_rng(2)
        sweep = _sweep(30, rng, start=1.0)
        noise = _random_event(30, rng, start=3.0)
        trains = [np.concatenate([a, b]) for a, b in zip(sweep, noise, strict=True)]
        dt = 0.02
        events = [
            DecodingResult(
                posterior=np.full((7, env.n_bins), 1.0 / env.n_bins),
                env=env,
                times=start + dt / 2 + dt * np.arange(7),
            )
            for start in (0.98, 3.0)
        ]
        table = compare_replay_methods(
            env,
            events,
            replay_methods=[GUPTA_2010],
            spike_times=trains,
            encoding_models={"left": rates, "right": rates[::-1].copy()},
            dt=dt,
            rng=0,
        )
        assert table["is_replay"].tolist() == [True, False]
        assert table["error"].tolist() == ["", ""]
        assert table["passes[n_active_cells >= 3]"].tolist() == [True, True]
        # Each row is detect_pairwise_order_replay over the event's decoding
        # windows, with the (event, method) stream.
        for index, event in enumerate(events):
            stream = np.random.default_rng(
                np.random.SeedSequence(
                    table.attrs["rng_entropy"],
                    spawn_key=(index, zlib.crc32(b"gupta_2010")),
                )
            )
            expected = detect_pairwise_order_replay(
                env,
                trains,
                (event.times[0] - dt / 2, event.times[-1] + dt / 2),
                {"left": rates, "right": rates[::-1].copy()},
                pairwise_order_method=GUPTA_2010,
                rng=stream,
            )
            row = table.loc[index]
            assert (row["score"], row["p_value"], row["template"]) == (
                expected.score,
                expected.p_value,
                expected.template,
            )

    def test_a_search_too_large_is_an_error_row(self, track):
        rates = _rates(track, [[2 * k, 2 * k + 10] for k in range(5)] * 4)
        trains = [np.array([1.0 + 0.001 * k]) for k in range(20)]
        event = DecodingResult(
            posterior=np.full((3, track.n_bins), 1.0 / track.n_bins),
            env=track,
            times=np.array([1.01, 1.03, 1.05]),
        )
        table = compare_replay_methods(
            track,
            [event],
            replay_methods=[GUPTA_2010],
            spike_times=trains,
            encoding_models=rates,
            dt=0.02,
        )
        assert table.loc[0, "error"].startswith("too many combinations")
        assert not table.loc[0, "is_replay"]
        assert table.loc[0, "passes[n_active_cells >= 3]"] is False

    def test_spike_inputs_are_required(self, track):
        event = DecodingResult(
            posterior=np.full((3, track.n_bins), 1.0 / track.n_bins),
            env=track,
            times=np.array([1.01, 1.03, 1.05]),
        )
        with pytest.raises(ValueError, match="gupta_2010"):
            compare_replay_methods(track, [event], replay_methods=[GUPTA_2010])


class TestMethod:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"field_threshold": 0.0},
            {"field_threshold": 1.0},
            {"merge_gap_bins": -1},
            {"field_center": "center"},
            {"max_time_lag": 0.0},
            {"directions": ()},
            {"directions": ("forward", "forward")},
            {"directions": ("sideways",)},
            {"nulls": ("unit_identity",)},
            {"n_shuffles": 0},
            {"alpha": 1.0},
            {"min_active_cells": 1},
        ],
    )
    def test_settings_that_cannot_run_are_refused(self, kwargs):
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            PairwiseOrderMethod(name="x", reference="y", **kwargs)

    def test_recipe_pickles_and_hashes(self):
        for clone in (
            pickle.loads(pickle.dumps(GUPTA_2010)),
            copy.deepcopy(GUPTA_2010),
        ):
            assert clone == GUPTA_2010
            assert hash(clone) == hash(GUPTA_2010)

    def test_results_pickle(self, track):
        rates = _rates(track, [[1], [2], [3]])
        trains = [np.array([1.00]), np.array([1.01]), np.array([1.02])]
        result = pairwise_order_significance(
            track, trains, (1.0, 1.1), rates, n_shuffles=10, rng=0
        )
        clone = pickle.loads(pickle.dumps(result))
        assert clone.p_value == result.p_value
        assert clone.observed.score == result.observed.score
        assert not clone.observed.assigned_positions.flags.writeable
        assert (
            clone.summary()["cell_identity_p_value"]
            == result.shuffles["cell_identity"].p_value
        )
