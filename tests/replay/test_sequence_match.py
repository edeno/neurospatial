"""Lee & Wilson (2002) best matches and Ji & Wilson (2007) matching indices.

The papers' worked numbers are reproduced through the public functions, and
the probabilities are checked against brute-force enumeration written from
the papers' definitions.
"""

from __future__ import annotations

import itertools
import math
import pickle
from fractions import Fraction

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.replay import (
    best_sequence_match,
    burst_onset_times,
    sequence_matching_index,
    spike_words,
)

LETTER_STEP = 0.06  # seconds between letters, above the 50 ms burst interval


@pytest.fixture(scope="module")
def track():
    """A 400 cm track in 2 cm bins with 20 cells whose fields peak 20 cm apart."""
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [400.0, 0.0]], bin_size=2.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    peaks = 9.0 + 20.0 * np.arange(20)
    rates = 15.0 * np.exp(-0.5 * ((x - peaks[:, None]) / 6.0) ** 2)
    return env, rates


def _letters(cells, n_cells=20, start=1.0, step=LETTER_STEP):
    """One spike per letter: the k-th letter of ``cells`` at start + k * step."""
    trains = [[] for _ in range(n_cells)]
    for k, cell in enumerate(cells):
        trains[cell].append(start + step * k)
    return [np.array(train) for train in trains]


def _word(text, alphabet="123456789A"):
    """Template ranks of a word written in Lee & Wilson's letters."""
    return [alphabet.index(letter) for letter in text]


# Brute force from the paper's definitions -----------------------------------


def _lis(values):
    best = [1] * len(values)
    for j in range(len(values)):
        for i in range(j):
            if values[i] < values[j]:
                best[j] = max(best[j], best[i] + 1)
    return max(best, default=0)


def _contains(word, x, y):
    width = x + y
    return any(_lis(word[s : s + width]) >= x for s in range(len(word) - width + 1))


def _match_list(k):
    """Lee & Wilson's list: (K,0) > (K,1) > (K-1,0) > ... > (2,0)."""
    return [(x, x - d) for d in range(k, 1, -1) for x in range(k, d - 1, -1)]


def _brute_best(word, k):
    for rank, (x, y) in enumerate(_match_list(k)):
        if _contains(word, x, y):
            return rank, (x, y)
    return len(_match_list(k)), (0, 0)


def _brute_p(word, k):
    observed, _ = _brute_best(word, k)
    orders = list(itertools.permutations(word))
    better = sum(_brute_best(list(order), k)[0] <= observed for order in orders)
    return Fraction(better, len(orders))


# Letters and words -----------------------------------------------------------


class TestBursts:
    def test_a_chain_of_short_intervals_is_one_letter(self):
        """30 ms apart: one burst though it lasts 90 ms (the chain rule)."""
        onsets = burst_onset_times(
            [np.array([1.0, 1.03, 1.06, 1.09, 1.3])], max_isi=0.05
        )
        np.testing.assert_allclose(onsets[0], [1.0, 1.3])

    def test_an_interval_equal_to_max_isi_starts_a_new_letter(self):
        """ISI < max_isi joins a burst; 1.15 - 1.1 is 0.04999999999999982 in
        floating point, equal to 50 ms within round-off.
        """
        exact = burst_onset_times([np.array([0.0, 0.25, 0.5])], max_isi=0.25)[0]
        np.testing.assert_array_equal(exact, [0.0, 0.25, 0.5])
        rounded = burst_onset_times([np.array([1.1, 1.15])], max_isi=0.05)[0]
        np.testing.assert_array_equal(rounded, [1.1, 1.15])

    def test_max_isi_zero_keeps_every_spike(self):
        train = np.array([1.0, 1.001, 1.002])
        np.testing.assert_array_equal(burst_onset_times([train], max_isi=0.0)[0], train)

    def test_a_burst_begun_before_the_event_adds_no_letter(self, track):
        """A burst from 0.98 s runs into the event; its later spike is no letter."""
        env, rates = track
        trains = _letters([0, 1, 2, 3])
        trains[0] = np.array([0.98, 1.01])
        match = best_sequence_match(env, trains, (1.0, 1.3), rates)
        assert match.n_letters == 3
        assert 0 not in match.letter_units.tolist()


class TestSpikeWords:
    def test_words_split_at_gaps_longer_than_max_gap(self):
        """Letters 100 ms apart stay in one word; 101 ms apart, they split."""
        spikes = [np.array([1.0, 1.201]), np.array([1.1])]
        words = spike_words(spikes, max_isi=0.05, max_gap=0.1)
        table = words.words_table()
        assert table["n_letters"].tolist() == [2, 1]
        np.testing.assert_allclose(words.intervals, [[1.0, 1.15], [1.201, 1.251]])

    def test_each_interval_holds_exactly_its_words_letters(self):
        rng = np.random.default_rng(3)
        spikes = [np.sort(rng.uniform(0, 20, 60)) for _ in range(6)]
        words = spike_words(spikes, max_isi=0.05, max_gap=0.1)
        for index, (start, stop) in enumerate(words.intervals):
            inside = (words.letter_times >= start) & (words.letter_times < stop)
            np.testing.assert_array_equal(
                np.flatnonzero(inside), np.flatnonzero(words.word_ids == index)
            )
        assert np.all(words.intervals[1:, 0] >= words.intervals[:-1, 1])

    def test_cells_and_epochs_select_letters(self):
        spikes = [np.array([1.0, 5.0]), np.array([1.05, 5.05]), np.array([1.02])]
        words = spike_words(
            spikes,
            max_isi=0.0,
            max_gap=0.1,
            cells=np.array([True, True, False]),
            epochs=np.array([[0.0, 1.03], [1.04, 6.0]]),
        )
        # 1.0 (epoch 1) and 1.05 (epoch 2) are 50 ms apart but in two epochs.
        assert words.letter_units.tolist() == [0, 1, 0, 1]
        assert words.words_table()["n_letters"].tolist() == [1, 1, 2]
        assert words.intervals[0, 1] == pytest.approx(1.03)

    def test_max_isi_above_max_gap_is_refused(self):
        with pytest.raises(ValueError, match="max_isi <= max_gap"):
            spike_words([np.array([1.0])], max_isi=0.2, max_gap=0.1)

    def test_words_are_candidate_events_for_the_score(self, track):
        """Scoring each word's interval gives the word's letters back."""
        env, rates = track
        spikes = _letters([3, 1, 4, 6, 9, 2, 5], step=0.07)
        words = spike_words(spikes, max_isi=0.05, max_gap=0.1)
        assert words.n_words == 1
        match = best_sequence_match(env, spikes, words.intervals[0], rates)
        np.testing.assert_array_equal(match.letter_units, words.letter_units)


# Lee & Wilson's best match -----------------------------------------------------


class TestBestMatch:
    def test_the_papers_worked_example(self, track):
        """325789A against 123456789A: (6, 0) and p = (12 + 1) / 7! (p. 1185)."""
        from neurospatial.replay.sequence_match import (
            _dense,
            _exact_distribution,
            _match_keys,
        )

        env, rates = track
        cells = _word("325789A")
        maps = rates[:10]
        match = best_sequence_match(env, _letters(cells, n_cells=10), (1.0, 2.0), maps)
        assert (match.matched_letters, match.interruptions) == (6, 0)
        assert match.match_probability == pytest.approx(13 / 5040, rel=1e-12)
        assert match.exact and match.n_letters == 7
        # 12 orderings have a (6, 0) match and one a better one.
        dense = _dense(np.array(cells))
        distribution = _exact_distribution(tuple(np.sort(dense)), "interruptions_first")
        key = int(_match_keys(dense[None, :], "interruptions_first")[0])
        at_six = distribution.count(key) - distribution.count(key + 1)
        assert (at_six, distribution.count(key + 1)) == (12, 1)

    @pytest.mark.parametrize(
        ("text", "template", "expected"),
        [
            ("11377", "123456789", (3, 0)),
            ("13436892", "123456789", (6, 1)),
            ("1234", "123456789A", (4, 0)),
            ("4321", "123456789A", (0, 0)),
        ],
    )
    def test_the_papers_match_definition(self, track, text, template, expected):
        """Examples of p. 1192; a reversed word has no match and p = 1."""
        env, rates = track
        maps = rates[: len(template)]
        cells = _word(text)
        match = best_sequence_match(
            env, _letters(cells, n_cells=len(template)), (1.0, 2.0), maps
        )
        assert (match.matched_letters, match.interruptions) == expected
        assert match.match_probability == pytest.approx(
            float(_brute_p(cells, len(template))), rel=1e-12
        )
        if expected == (0, 0):
            assert match.match_probability == 1.0

    def test_a_perfect_four_letter_word_has_p_one_in_24(self, track):
        env, rates = track
        match = best_sequence_match(env, _letters([0, 1, 2, 3]), (1.0, 2.0), rates)
        assert match.match_probability == pytest.approx(1 / 24, rel=1e-12)
        assert match.trial_group == "low_probability"
        assert match.best_possible_probability == pytest.approx(1 / 24, rel=1e-12)

    @pytest.mark.parametrize("seed", range(6))
    def test_exact_probabilities_equal_brute_force(self, track, seed):
        """Random words with repeated letters, every ordering enumerated."""
        env, rates = track
        rng = np.random.default_rng(seed)
        cells = rng.integers(0, 6, size=int(rng.integers(3, 7))).tolist()
        match = best_sequence_match(
            env, _letters(cells, n_cells=6), (1.0, 2.0), rates[:6]
        )
        _, best = _brute_best(cells, 6)
        assert (match.matched_letters, match.interruptions) == best
        assert match.match_probability == pytest.approx(
            float(_brute_p(cells, 6)), rel=1e-12
        )

    @pytest.mark.parametrize("match_order", ["interruptions_first", "length_first"])
    @pytest.mark.parametrize(
        "pattern",
        [
            (0, 0, 0, 1, 2),
            (0, 1, 2, 2, 2),  # the mirror of the one above
            (0, 0, 1, 1, 1, 2, 3),
            (0, 1, 1, 2, 3, 3, 3),
            (0, 1, 2, 3, 4, 5),
        ],
    )
    def test_the_exact_distribution_counts_every_ordering(self, pattern, match_order):
        """Distinct words are scored once and counted for each ordering of a
        cell's letters, and mirrored patterns share one computation; the
        counts equal those over all n! orderings.
        """
        from neurospatial.replay.sequence_match import (
            _exact_distribution,
            _match_keys,
        )

        orderings = np.array(list(itertools.permutations(pattern)))
        keys = _match_keys(orderings, match_order)
        values = np.unique(keys)
        distribution = _exact_distribution(pattern, match_order)
        assert distribution.total == math.factorial(len(pattern))
        np.testing.assert_array_equal(distribution.keys, values)
        np.testing.assert_array_equal(
            distribution.at_least, [np.sum(keys >= value) for value in values]
        )

    def test_length_first_ranks_more_ordered_letters_higher(self, track):
        """5 6 7 8 1 2 9: (4, 0) has fewer interruptions, (5, 2) more letters."""
        env, rates = track
        spikes = _letters([4, 5, 6, 7, 0, 1, 8], n_cells=9)
        default = best_sequence_match(env, spikes, (1.0, 2.0), rates[:9])
        longest = best_sequence_match(
            env, spikes, (1.0, 2.0), rates[:9], match_order="length_first"
        )
        assert (default.matched_letters, default.interruptions) == (4, 0)
        assert (longest.matched_letters, longest.interruptions) == (5, 2)

    def test_relabeling_cells_consistently_leaves_the_match_unchanged(self, track):
        env, rates = track
        cells = [2, 1, 4, 6, 7, 3, 9]
        spikes = _letters(cells, n_cells=10)
        maps = rates[:10]
        order = np.random.default_rng(0).permutation(10)
        relabeled = best_sequence_match(
            env, [spikes[i] for i in order], (1.0, 2.0), maps[order]
        )
        original = best_sequence_match(env, spikes, (1.0, 2.0), maps)
        assert relabeled.summary() == original.summary()

    def test_simultaneous_letters_never_help_a_match(self, track):
        env, rates = track
        spikes = [np.array([1.0]), np.array([1.0])] + [np.array([])] * 18
        match = best_sequence_match(env, spikes, (1.0, 2.0), rates)
        assert match.matched_letters == 0 and match.match_probability == 1.0

    def test_sampled_probability_agrees_with_exact(self, track):
        env, rates = track
        cells = [1, 0, 2, 4, 3, 5, 7, 6]
        spikes = _letters(cells, n_cells=10)
        exact = best_sequence_match(env, spikes, (1.0, 2.0), rates[:10])
        sampled = best_sequence_match(
            env,
            spikes,
            (1.0, 2.0),
            rates[:10],
            exact_max_letters=7,
            n_samples=20_000,
            rng=0,
        )
        assert exact.exact and not sampled.exact
        assert sampled.standard_error > 0
        assert abs(sampled.match_probability - exact.match_probability) < (
            3 * sampled.standard_error + 1 / 20_001
        )

    def test_trial_groups(self, track):
        env, rates = track

        def group(cells):
            spikes = _letters(cells, n_cells=10)
            return best_sequence_match(env, spikes, (1.0, 2.0), rates[:10]).trial_group

        assert group([3, 5]) == "pair"
        assert group([5, 3, 4]) == "triplet"
        assert group([0, 0, 2, 3]) is None  # best possible 4/24 > 1/24
        assert group([4, 2, 1, 3]) == "low_probability"  # any order counts
        assert group([3, 3]) is None

    def test_one_letter_has_no_probability(self, track):
        env, rates = track
        match = best_sequence_match(env, _letters([4]), (1.0, 2.0), rates)
        assert np.isnan(match.match_probability)

    def test_tied_field_peaks_are_refused(self, track):
        env, rates = track
        maps = np.vstack([rates[:4], rates[2]])
        with pytest.raises(ValueError, match="same position"):
            best_sequence_match(env, _letters([0, 1], n_cells=5), (1.0, 2.0), maps)

    def test_result_arrays_are_read_only_and_pickle(self, track):
        env, rates = track
        match = best_sequence_match(env, _letters([0, 2, 1]), (1.0, 2.0), rates)
        assert not match.letter_units.flags.writeable
        copy = pickle.loads(pickle.dumps(match))
        assert not copy.letter_units.flags.writeable
        assert copy.summary() == match.summary()


# Ji & Wilson's matching index ------------------------------------------------


# Supplementary Table 1: (M, I, p, m, n). The paper enumerated orders below
# 11 cells and sampled 10,000 above.
SUPPLEMENTARY_TABLE_1 = [
    (4, 1.0, 0.042, 6, 0),
    (5, 0.8, 0.042, 9, 1),
    (6, 0.73, 0.028, 13, 2),
    (7, 0.62, 0.035, 17, 4),
    (8, 0.57, 0.031, 22, 6),
    (9, 0.50, 0.038, 27, 9),
    (10, 0.47, 0.036, 33, 12),
    (11, 0.42, 0.043, 39, 16),
    (13, 0.36, 0.049, 53, 25),
    (15, 0.33, 0.045, 70, 35),
]


def _cutoff(n_cells, alpha=0.05):
    """The most opposite pairs with p < alpha, from the exact p-values."""
    from neurospatial.replay.sequence_match import _inversion_p_values

    p_values = _inversion_p_values(n_cells)
    n = int(np.flatnonzero(p_values < alpha)[-1])
    return n, float(p_values[n])


class TestMatchingIndex:
    @pytest.mark.parametrize(
        ("m_cells", "index", "p", "same", "opposite"), SUPPLEMENTARY_TABLE_1
    )
    def test_supplementary_table_1(self, m_cells, index, p, same, opposite):
        n, exact = _cutoff(m_cells)
        pairs = m_cells * (m_cells - 1) // 2
        assert (pairs - n, n) == (same, opposite)
        assert round((pairs - 2 * n) / pairs, 2) == index
        if m_cells < 11:
            assert round(exact, 3) == p
        else:  # within three standard errors of 10,000 samples
            assert abs(exact - p) < 3 * np.sqrt(p * (1 - p) / 10_000)

    def test_the_14_cell_cutoff_moves(self):
        """Ji & Wilson's 30 opposite pairs at 14 cells is p = 0.0505 exactly."""
        from neurospatial.replay.sequence_match import _inversion_p_values

        assert _cutoff(14)[0] == 29
        assert _inversion_p_values(14)[30] == pytest.approx(0.0505, abs=5e-5)

    @pytest.mark.parametrize("m_cells", range(2, 8))
    def test_p_values_equal_brute_force(self, m_cells):
        from neurospatial.replay.sequence_match import _inversion_p_values

        counts = np.zeros(m_cells * (m_cells - 1) // 2 + 1)
        for order in itertools.permutations(range(m_cells)):
            inversions = sum(
                order[i] > order[j]
                for i in range(m_cells)
                for j in range(i + 1, m_cells)
            )
            counts[inversions] += 1
        np.testing.assert_allclose(
            _inversion_p_values(m_cells),
            np.cumsum(counts) / math.factorial(m_cells),
            rtol=1e-14,
        )

    @pytest.mark.parametrize(
        ("frame", "n_template", "p"),
        [
            ("02546", 7, 5 / 120),  # Suppl. Methods
            ("0132567", 8, 0.0014),  # p. 103
            ("01235", 6, 0.0083),  # p. 103
            ("023489567", 10, 0.0063),  # p. 104
            ("201345768", 10, 0.00043),  # Suppl. Fig. 5
            ("1024536", 7, 0.015),
        ],
    )
    def test_the_papers_examples(self, track, frame, n_template, p):
        env, rates = track
        cells = [int(c) for c in frame]
        spikes = _letters(cells, n_cells=n_template, step=0.04)
        result = sequence_matching_index(env, spikes, (1.0, 2.0), rates[:n_template])
        assert result.n_active_cells == len(cells)
        assert result.match_probability == pytest.approx(
            p, rel=0.05 if p != 5 / 120 else 1e-12
        )
        np.testing.assert_array_equal(result.firing_order, cells)

    def test_reversing_the_template_negates_the_index(self, track):
        env, rates = track
        spikes = _letters([3, 0, 5, 1, 7, 6, 2], n_cells=8, step=0.03)
        forward = sequence_matching_index(env, spikes, (1.0, 2.0), rates[:8])
        backward = sequence_matching_index(env, spikes, (1.0, 2.0), rates[:8][:, ::-1])
        assert backward.score == pytest.approx(-forward.score)
        assert forward.n_same_order_pairs == backward.n_opposite_order_pairs

    def test_the_firing_time_is_the_kernel_peak(self, track):
        """Three spikes early and one late: the peak sits in the cluster,
        between grid points of 1 ms.
        """
        env, rates = track
        spikes = [np.array([1.0, 1.0137, 1.0291, 1.3])] + [np.array([])] * 19
        result = sequence_matching_index(
            env, spikes, (1.0, 2.0), rates, bandwidth=0.05, min_active_cells=2
        )
        grid = np.arange(1.0, 1.3, 1e-6)
        density = np.exp(-0.5 * ((grid[:, None] - spikes[0]) / 0.05) ** 2).sum(axis=1)
        assert result.firing_times[0] == pytest.approx(
            grid[np.argmax(density)], abs=2e-6
        )

    def test_too_few_active_cells_has_no_index(self, track):
        env, rates = track
        result = sequence_matching_index(env, _letters([0, 1, 2]), (1.0, 2.0), rates)
        assert result.n_active_cells == 3 and np.isnan(result.score)
