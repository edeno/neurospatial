"""Nádasdy et al. (1999): recurring spike sequences and their surrogates."""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pytest

from neurospatial.replay import (
    RecurringSequenceMethod,
    recurring_sequences_significance,
    recurring_spike_sequences,
    surrogate_spike_trains,
)
from neurospatial.replay.methods import NADASDY_1999, PUBLISHED_METHODS


@pytest.fixture(scope="module")
def trains():
    """Five cells firing at random, about 3 Hz, for 100 s."""
    rng = np.random.default_rng(7)
    return [np.sort(rng.uniform(0.0, 100.0, rng.integers(250, 350))) for _ in range(5)]


class TestSurrogates:
    def test_interval_shuffle_keeps_each_trains_intervals(self, trains):
        surrogates = surrogate_spike_trains(trains, kind="isi_shuffle", rng=0)
        for original, surrogate in zip(trains, surrogates, strict=True):
            assert surrogate.size == original.size
            np.testing.assert_allclose(
                np.sort(np.diff(surrogate)), np.sort(np.diff(original)), atol=1e-9
            )
            assert surrogate[0] == original[0]
            assert surrogate[-1] == pytest.approx(original[-1])
            assert not np.allclose(surrogate, original)

    def test_jitter_moves_each_spike_later_by_up_to_max_jitter(self):
        """Spikes 100 ms apart keep their order, so each can be followed."""
        train = np.arange(200) * 0.1
        surrogate = surrogate_spike_trains([train], kind="spike_jitter", rng=0)[0]
        shift = surrogate - train
        assert shift.min() >= 0.0 and shift.max() < 0.05
        assert shift.max() > 0.045 and shift.min() < 0.005

    def test_across_train_shuffle_keeps_the_pooled_spikes(self, trains):
        surrogates = surrogate_spike_trains(trains, kind="across_trains", rng=0)
        np.testing.assert_array_equal(
            np.sort(np.concatenate(surrogates)), np.sort(np.concatenate(trains))
        )
        counts = [s.size for s in surrogates]
        assert counts != [t.size for t in trains]

    def test_theta_shuffle_keeps_each_spikes_phase(self, trains):
        rng = np.random.default_rng(1)
        peaks = np.cumsum(rng.uniform(0.11, 0.15, 800)) + 1.0
        surrogates = surrogate_spike_trains(
            trains, kind="theta_phase", theta_peak_times=peaks, rng=0
        )

        def phases(train):
            inside = (train >= peaks[0]) & (train < peaks[-1])
            cycle = np.searchsorted(peaks, train[inside], side="right") - 1
            return np.sort(
                (train[inside] - peaks[cycle]) / (peaks[cycle + 1] - peaks[cycle])
            ), train[~inside]

        for original, surrogate in zip(trains, surrogates, strict=True):
            assert surrogate.size == original.size
            original_phase, original_outside = phases(original)
            surrogate_phase, surrogate_outside = phases(surrogate)
            np.testing.assert_allclose(surrogate_phase, original_phase, atol=1e-9)
            np.testing.assert_array_equal(surrogate_outside, original_outside)
            assert not np.allclose(surrogate, original)

    def test_theta_shuffle_needs_peaks(self, trains):
        with pytest.raises(ValueError, match="theta_peak_times"):
            surrogate_spike_trains(trains, kind="theta_phase")


class TestSearch:
    def test_a_planted_sequence_recurs(self):
        starts = np.arange(10.0)
        spikes = [starts, starts + 0.03, starts + 0.08, np.array([20.0])]
        result = recurring_spike_sequences(spikes, reference_unit=0)
        assert result.complexity.tolist() == [3] * 10
        assert result.repetitions.tolist() == [9] * 10
        table = result.sequences_table()
        assert table.loc[0, "units"] == (1, 2)
        assert table.loc[0, "lags"] == pytest.approx((0.03, 0.08))

    def test_lags_must_agree_within_half_the_tolerance(self):
        """Cell 1 fires 30, 34 and 36 ms after the reference spikes: lags 4 or
        2 ms apart match (+/- 5 ms), 6 ms apart do not.
        """
        reference = np.array([0.0, 1.0, 2.0])
        spikes = [reference, np.array([0.03, 1.034, 2.036]), reference + 0.08]
        result = recurring_spike_sequences(spikes, reference_unit=0, tolerance=0.01)
        assert result.repetitions.tolist() == [1, 2, 1]

    def test_extra_spikes_do_not_prevent_a_match(self):
        reference = np.array([0.0, 1.0])
        spikes = [reference, np.array([0.03, 1.03, 1.05]), reference + 0.08]
        result = recurring_spike_sequences(spikes, reference_unit=0)
        # The second window holds the first's template plus a spike.
        assert result.complexity.tolist() == [3, 4]
        assert result.repetitions.tolist() == [1, 0]

    def test_each_template_spike_needs_its_own_exemplar_spike(self):
        """Cell 1 fires twice, 7 ms apart, after the first reference spike. One
        spike of cell 1 between them is within 5 ms of both lags but cannot be
        both template spikes; two spikes at the lags can.
        """
        spikes = [
            np.array([0.0, 1.0, 2.0]),
            np.array([0.020, 0.027, 1.0235, 2.019, 2.026]),
        ]
        result = recurring_spike_sequences(spikes, reference_unit=0)
        assert result.complexity.tolist() == [3, 3]
        # The window at 1 s (one spike) repeats neither; 0 s and 2 s repeat
        # each other.
        assert result.repetitions.tolist() == [1, 1]

    def test_repetitions_agree_with_a_bipartite_matching(self):
        """Dense random trains, so cells often give several spikes to a
        template: an exemplar repeats it when a maximum matching of template
        spikes to distinct exemplar spikes of the same cell, within +/- 5 ms
        of the lag, covers every template spike.
        """
        from scipy.sparse import csr_array
        from scipy.sparse.csgraph import maximum_bipartite_matching

        rng = np.random.default_rng(11)
        spikes = [np.sort(rng.uniform(0.0, 4.0, 40))] + [
            np.sort(rng.uniform(0.0, 4.0, 60)) for _ in range(2)
        ]
        window, half = 0.2, 0.005
        result = recurring_spike_sequences(
            spikes, reference_unit=0, window=window, tolerance=2 * half, burst_isi=0.0
        )
        table = result.sequences_table()
        reference = spikes[0]
        expected = []
        for row in table.itertuples():
            count = 0
            for other in reference[np.abs(reference - row.reference_time) >= window]:
                edges = np.array(
                    [
                        [
                            float(
                                cell == exemplar_cell and abs(t - other - lag) <= half
                            )
                            for exemplar_cell in (1, 2)
                            for t in spikes[exemplar_cell]
                        ]
                        for cell, lag in zip(row.units, row.lags, strict=True)
                    ]
                )
                matched = maximum_bipartite_matching(
                    csr_array(edges), perm_type="column"
                )
                count += bool(np.all(matched >= 0))
            expected.append(count)
        assert result.repetitions.tolist() == expected
        assert max(np.bincount(units).max() for units in table["units"]) >= 2

    def test_overlapping_windows_do_not_repeat_each_other(self):
        """Reference spikes 7 ms apart see the same spikes shifted by 7 ms."""
        spikes = [np.array([0.0, 0.007]), np.array([0.05]), np.array([0.09])]
        result = recurring_spike_sequences(spikes, reference_unit=0, tolerance=0.02)
        assert result.repetitions.tolist() == [0, 0]

    def test_bursts_are_one_spike(self):
        starts = np.arange(3.0)
        spikes = [np.sort(np.r_[starts, starts + 0.004]), starts + 0.03, starts + 0.08]
        result = recurring_spike_sequences(spikes, reference_unit=0)
        assert result.reference_times.tolist() == [0.0, 1.0, 2.0]

    def test_bad_reference_unit_is_refused(self, trains):
        with pytest.raises(ValueError, match="reference_unit"):
            recurring_spike_sequences(trains, reference_unit=5)


class TestSignificance:
    def test_a_repeated_sequence_beats_every_surrogate(self, trains):
        rng = np.random.default_rng(3)
        starts = np.sort(rng.uniform(0.0, 100.0, 40))
        planted = [
            np.sort(np.r_[train, starts + lag])
            for train, lag in zip(trains[:3], (0.0, 0.03, 0.08), strict=True)
        ]
        method = dataclasses.replace(
            NADASDY_1999, surrogates=("across_trains", "isi_shuffle"), n_surrogates=20
        )
        result = recurring_sequences_significance(
            planted, recurring_sequence_method=method, reference_unit=0, rng=0
        )
        assert result.p_values == {"across_trains": 1 / 21, "isi_shuffle": 1 / 21}
        assert result.p_value == 1 / 21
        assert not result.is_significant  # 1/21 is above alpha = 0.01
        assert (
            result.observed.n_recurring > result.surrogate_counts["across_trains"].max()
        )
        copy = pickle.loads(pickle.dumps(result))
        assert copy.summary() == result.summary()

    def test_each_surrogate_kind_has_its_own_stream(self, trains):
        both = dataclasses.replace(
            NADASDY_1999, surrogates=("across_trains", "spike_jitter"), n_surrogates=5
        )
        one = dataclasses.replace(both, surrogates=("spike_jitter",))
        first = recurring_sequences_significance(
            trains, recurring_sequence_method=both, reference_unit=0, rng=4
        )
        second = recurring_sequences_significance(
            trains, recurring_sequence_method=one, reference_unit=0, rng=4
        )
        np.testing.assert_array_equal(
            first.surrogate_counts["spike_jitter"],
            second.surrogate_counts["spike_jitter"],
        )

    def test_the_theta_surrogate_needs_peaks(self, trains):
        method = dataclasses.replace(NADASDY_1999, surrogates=("theta_phase",))
        with pytest.raises(ValueError, match="theta_peak_times"):
            recurring_sequences_significance(
                trains, recurring_sequence_method=method, reference_unit=0
            )

    def test_the_recipe_runs_on_sleep_without_theta_peaks(self, trains):
        """The paper's five-rat test shuffled across trains only."""
        method = dataclasses.replace(NADASDY_1999, n_surrogates=5)
        result = recurring_sequences_significance(
            trains, recurring_sequence_method=method, reference_unit=0, rng=0
        )
        assert list(result.p_values) == ["across_trains"]
        assert result.p_value == result.p_values["across_trains"]


def test_the_recipe_is_session_level():
    assert NADASDY_1999 not in PUBLISHED_METHODS
    assert (NADASDY_1999.window, NADASDY_1999.tolerance, NADASDY_1999.burst_isi) == (
        0.2,
        0.01,
        0.006,
    )
    assert NADASDY_1999.n_surrogates == 100 and NADASDY_1999.alpha == 0.01
    assert NADASDY_1999.max_jitter == 0.05
    assert NADASDY_1999.surrogates == ("across_trains",)
    with pytest.raises(ValueError, match="surrogates"):
        RecurringSequenceMethod(name="x", reference="r", surrogates=("shuffle",))
