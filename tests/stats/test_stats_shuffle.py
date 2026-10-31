"""Behavior tests for the shuffle generators and Monte Carlo p-value helpers.

Covers the temporal, trial and ISI shuffles in ``neurospatial.stats.shuffle``,
``ShuffleTestResult`` and its p-value and z-score helpers.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial.stats.shuffle import (
    ShuffleTestResult,
    compute_shuffle_pvalue,
    compute_shuffle_zscore,
    shuffle_spikes_isi,
    shuffle_time_bins,
    shuffle_time_bins_coherent,
    shuffle_trials,
)


class TestNewShuffleFunctions:
    """Tests for new shuffle functions."""

    def test_shuffle_trials_basic(self):
        """Test shuffle_trials shuffles trial labels."""
        trial_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        # Collect shuffled versions
        shuffled_list = list(shuffle_trials(trial_labels, n_shuffles=5, rng=42))

        assert len(shuffled_list) == 5
        for shuffled in shuffled_list:
            assert shuffled.shape == trial_labels.shape
            # Same values, just rearranged
            assert set(shuffled) == set(trial_labels)
            # Same counts per unique value
            for val in np.unique(trial_labels):
                assert np.sum(shuffled == val) == np.sum(trial_labels == val)

    def test_shuffle_trials_changes_order(self):
        """Test shuffle_trials actually changes order."""
        trial_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        # With enough shuffles, at least one should differ
        any_different = False
        for shuffled in shuffle_trials(trial_labels, n_shuffles=10, rng=42):
            if not np.array_equal(shuffled, trial_labels):
                any_different = True
                break
        assert any_different

    def test_shuffle_spikes_isi_basic(self):
        """Test shuffle_spikes_isi shuffles inter-spike intervals."""
        # Spike times for a single neuron
        spike_times = np.array([0.1, 0.15, 0.25, 0.4, 0.45, 0.7])

        shuffled_list = list(shuffle_spikes_isi(spike_times, n_shuffles=5, rng=42))

        assert len(shuffled_list) == 5
        for shuffled in shuffled_list:
            assert shuffled.shape == spike_times.shape
            # Should preserve number of spikes
            assert len(shuffled) == len(spike_times)
            # Should be sorted (times are ordered)
            assert np.all(np.diff(shuffled) >= 0)

    def test_shuffle_spikes_isi_preserves_isi_distribution(self):
        """Test shuffle_spikes_isi preserves ISI values."""
        spike_times = np.array([0.1, 0.15, 0.25, 0.4, 0.45, 0.7])
        original_isis = np.diff(spike_times)

        for shuffled in shuffle_spikes_isi(spike_times, n_shuffles=5, rng=42):
            shuffled_isis = np.diff(shuffled)
            # Same ISI values (just reordered)
            assert_allclose(sorted(shuffled_isis), sorted(original_isis))

    def test_shuffle_spikes_isi_preserves_first_spike(self):
        """Test shuffle_spikes_isi preserves first spike time."""
        spike_times = np.array([0.1, 0.15, 0.25, 0.4])

        for shuffled in shuffle_spikes_isi(spike_times, n_shuffles=5, rng=42):
            assert_allclose(shuffled[0], spike_times[0])


class TestShufflePairingDestruction:
    """A shuffle is only a valid null if it preserves the marginal AND destroys
    the pairing it is meant to test. Existing tests check preservation; these
    add the pairing-destruction half.
    """

    def test_isi_shuffle_destroys_isi_order(self) -> None:
        from neurospatial.stats.shuffle import shuffle_spikes_isi

        # A spike train with monotonically increasing ISIs. The shuffle permutes
        # the ISIs (preserving their multiset), so the shuffled ISI *order* must
        # be uncorrelated with the original. (Multiset preservation is covered by
        # existing tests; this covers order destruction.)
        n = 200
        event_times = np.cumsum(np.arange(1, n + 1)).astype(np.float64)
        original_isi = np.diff(event_times)

        abs_corrs = []
        for shuffled in shuffle_spikes_isi(event_times, n_shuffles=200, rng=3):
            shuffled_isi = np.diff(shuffled)
            abs_corrs.append(abs(np.corrcoef(original_isi, shuffled_isi)[0, 1]))

        # E[|r|] ~ sqrt(2 / (pi * (m-1))) ~ 0.057 for m=199; well under 0.1.
        assert np.mean(abs_corrs) < 0.1


# ---------------------------------------------------------------------------
# compute_shuffle_pvalue: non-finite null handling.
# ---------------------------------------------------------------------------


def test_compute_shuffle_pvalue_drops_nan_nulls():
    """Undefined (NaN) null scores are dropped (with a warning) and excluded from n."""
    null = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.warns(UserWarning, match="undefined \\(NaN\\)"):
        p = compute_shuffle_pvalue(10.0, null, tail="greater")
    # n reduced to 3 defined values, none >= 10 -> (0 + 1) / (3 + 1) = 0.25.
    assert_allclose(p, 0.25)


@pytest.mark.parametrize(
    ("tail", "expected"),
    [("greater", 501 / 1001), ("less", 501 / 1001), ("two-sided", 1.0)],
)
def test_compute_shuffle_pvalue_ranks_infinite_nulls(tail, expected):
    """Infinite null scores are extreme, defined scores: they are counted.

    Half the null is +inf and half 1.0. For an observed 10, (k + 1) / (n + 1)
    with n = 1000 gives k = 500 null scores >= 10 and 500 <= 10. Dropping
    the infinite scores would give 1 / 501 for "greater", a false positive.
    """
    null = np.r_[np.full(500, np.inf), np.ones(500)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = compute_shuffle_pvalue(10.0, null, tail=tail)
    assert p == pytest.approx(expected)


def test_compute_shuffle_pvalue_all_nan_raises():
    """An all-NaN null distribution raises a ValueError."""
    with pytest.raises(ValueError, match="no defined"):
        compute_shuffle_pvalue(5.0, np.full(4, np.nan))


def test_compute_shuffle_pvalue_nan_observed_raises():
    """A non-finite ``observed`` raises rather than silently returning 1/(n+1)."""
    null = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="observed must be a finite value"):
        compute_shuffle_pvalue(np.nan, null, tail="greater")
    with pytest.raises(ValueError, match="observed must be a finite value"):
        compute_shuffle_pvalue(np.inf, null, tail="greater")


def test_compute_shuffle_pvalue_finite_unchanged():
    """All-finite null reproduces the documented value (finite path untouched)."""
    null = np.array([1.0, 2.0, 3.0, 4.0])
    p = compute_shuffle_pvalue(10.0, null, tail="greater")
    assert_allclose(p, 0.2)


# ---------------------------------------------------------------------------
# shuffle_cell_identity: neuron-count agreement.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ShuffleTestResult: invariants, from_null and plot.
# ---------------------------------------------------------------------------


def test_from_null_leaves_out_undefined_scores_with_a_warning():
    """NaN shuffles are left out of p and z, with a warning naming the null."""

    null = np.array([np.nan, 1.0, 2.0, 3.0])
    with pytest.warns(UserWarning, match=r"time_bins: 1 of 4 shuffles"):
        result = ShuffleTestResult.from_null(2.5, null, shuffle_type="time_bins")
    defined = null[1:]
    assert result.p_value == compute_shuffle_pvalue(2.5, defined)
    assert result.z_score == compute_shuffle_zscore(2.5, defined)
    assert result.n_shuffles == 4
    assert result.n_shuffles_used == 3
    assert result.tail == "greater"


def test_from_null_decimals_ties_scores_equal_up_to_rounding():
    null = np.array([0.30000000000000004, 0.1, 0.2])
    exact = ShuffleTestResult.from_null(0.3, null, shuffle_type="x")
    rounded = ShuffleTestResult.from_null(0.3, null, shuffle_type="x", decimals=6)
    assert exact.p_value == pytest.approx(2 / 4)  # 0.30000000000000004 >= 0.3
    assert rounded.p_value == pytest.approx(2 / 4)
    below = ShuffleTestResult.from_null(
        0.30000000000000004, np.array([0.3, 0.1, 0.2]), shuffle_type="x"
    )
    assert below.p_value == pytest.approx(1 / 4)
    tied = ShuffleTestResult.from_null(
        0.30000000000000004, np.array([0.3, 0.1, 0.2]), shuffle_type="x", decimals=6
    )
    assert tied.p_value == pytest.approx(2 / 4)


def test_from_null_records_tail():
    result = ShuffleTestResult.from_null(
        -5.0, np.arange(9.0), shuffle_type="x", tail="less"
    )
    assert result.tail == "less"
    assert result.p_value == pytest.approx(1 / 10)
    assert result.summary()["tail"] == "less"


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"p_value": 0.0}, "p_value"),
        ({"p_value": 1.5}, "p_value"),
        ({"n_shuffles": 5}, "n_shuffles"),
        ({"null_scores": np.ones((2, 2)), "n_shuffles": 4}, "1-D"),
        ({"tail": "up"}, "tail"),
    ],
)
def test_shuffle_test_result_rejects_inconsistent_fields(changes, match):
    """The constructor and dataclasses.replace check the fields agree."""
    import dataclasses

    result = ShuffleTestResult.from_null(5.0, np.arange(3.0), shuffle_type="x")
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(result, **changes)


def test_shuffle_test_result_allows_nan_p_value_and_has_short_repr():
    result = ShuffleTestResult(
        observed_score=np.nan,
        null_scores=np.full(1000, 1.0),
        p_value=np.nan,
        z_score=np.nan,
        shuffle_type="x",
        n_shuffles=1000,
    )
    assert not result.is_significant()
    assert len(repr(result)) < 300


def test_shuffle_test_result_plot_with_infinite_nulls():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with pytest.warns(UserWarning, match="undefined"):
        result = ShuffleTestResult.from_null(
            3.0, np.array([1.0, 2.0, np.inf, np.nan]), shuffle_type="x"
        )
    ax = result.plot()
    assert "2 non-finite not drawn" in ax.get_title()
    assert "greater" in ax.get_title()
    plt.close("all")


def test_zscore_is_nan_with_infinite_null():
    """An infinite null score leaves the mean and SD undefined."""
    assert np.isnan(compute_shuffle_zscore(1.0, np.array([0.0, 1.0, np.inf])))
    assert compute_shuffle_zscore(2.0, np.array([0.0, np.nan, 2.0])) == pytest.approx(
        1.0
    )


# ---------------------------------------------------------------------------
# Generators: call-time validation, whole-trial and ISI shuffles.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda n: shuffle_time_bins(np.ones((3, 2)), n_shuffles=n),
        lambda n: shuffle_time_bins_coherent(np.ones((3, 2)), n_shuffles=n),
        lambda n: shuffle_trials(np.array([0, 0, 1]), n_shuffles=n),
        lambda n: shuffle_spikes_isi(np.array([0.1, 0.2, 0.4]), n_shuffles=n),
    ],
)
@pytest.mark.parametrize("n", [0, -3, 2.5, True])
def test_generators_refuse_bad_counts_at_call_time(make, n):
    """A count that would yield nothing raises before any draw is requested."""
    with pytest.raises(ValueError, match="positive integer"):
        make(n)


def test_shuffle_trials_moves_whole_trials():
    """Each trial keeps its samples together and takes another trial's label."""
    labels = np.repeat([10, 11, 12, 13], [3, 2, 4, 1])
    starts = np.array([0, 3, 5, 9])
    seen = set()
    for shuffled in shuffle_trials(labels, n_shuffles=200, rng=0):
        runs = np.split(shuffled, starts[1:])
        assert all(np.all(run == run[0]) for run in runs)
        run_labels = tuple(run[0] for run in runs)
        assert sorted(run_labels) == [10, 11, 12, 13]
        seen.add(run_labels)
    assert len(seen) == 24  # every ordering of the four trials appears


def test_shuffle_trials_trial_types_keep_type_counts():
    """With trial-type labels, the number of trials of each type is kept."""
    labels = np.array([0, 0, 1, 1, 1, 0, 1, 1])  # trials: 0, 1, 0, 1
    for shuffled in shuffle_trials(labels, n_shuffles=50, rng=1):
        boundaries = np.flatnonzero(np.diff(labels)) + 1
        run_labels = [run[0] for run in np.split(shuffled, boundaries)]
        assert all(np.all(run == run[0]) for run in np.split(shuffled, boundaries))
        assert sorted(run_labels) == [0, 0, 1, 1]


@pytest.mark.parametrize(
    "spikes",
    [np.array([0.5, 0.1, 0.9]), np.array([0.1, np.nan, 0.9]), np.ones((2, 2))],
)
def test_shuffle_spikes_isi_requires_sorted_finite_1d(spikes):
    with pytest.raises(ValueError, match="sorted, finite 1-D"):
        shuffle_spikes_isi(spikes)


def test_shuffle_spikes_isi_warns_when_null_is_degenerate():
    with pytest.warns(UserWarning, match="every shuffle equals the input"):
        draws = list(shuffle_spikes_isi(np.array([0.1, 0.4]), n_shuffles=3))
    assert all(np.array_equal(draw, [0.1, 0.4]) for draw in draws)


def test_shuffle_spikes_isi_keeps_span_and_interval_multiset():
    spikes = np.array([0.1, 0.15, 0.25, 0.4, 0.45, 0.7])
    for shuffled in shuffle_spikes_isi(spikes, n_shuffles=20, rng=3):
        assert shuffled[0] == spikes[0]
        assert shuffled[-1] == pytest.approx(spikes[-1], abs=1e-12)
        assert np.all(np.diff(shuffled) >= 0)
        assert_allclose(np.sort(np.diff(shuffled)), np.sort(np.diff(spikes)))


def test_circular_shift_warns_when_no_spike_is_in_the_windows():
    """Spikes in ms against windows in s all fall outside: say so."""
    from neurospatial.stats.shuffle import shuffle_spike_times_circular

    with pytest.warns(UserWarning, match="none of the 2 spikes"):
        draws = shuffle_spike_times_circular(
            np.array([500.0, 600.0]), np.array([[0.0, 100.0]]), n_shuffles=2, rng=0
        )
    assert all(draw.size == 0 for draw in draws)
