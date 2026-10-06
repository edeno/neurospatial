"""Recording gaps split sequences and gate every lagged transition interval."""

import numpy as np
import pytest


@pytest.mark.parametrize("lag, count", [(1, 9998), (3, 9994)])
def test_transitions_skip_pairs_across_pause(two_epoch_recording, lag, count):
    r = two_epoch_recording
    matrix = r.env.transitions(
        times=r.times,
        positions=r.positions,
        lag=lag,
        allow_teleports=True,
        normalize=False,
    )
    assert matrix.sum() == count
    # Independent per-pair reference gates every intervening interval.
    bins = r.env.bin_at(r.positions)
    expected = np.zeros(matrix.shape)
    for k in range(len(r.times) - lag):
        if np.all(np.diff(r.times[k : k + lag + 1]) <= 0.5):
            expected[bins[k], bins[k + lag]] += 1
    np.testing.assert_array_equal(matrix.toarray(), expected)


def test_same_bin_runs_split_at_pause(two_epoch_recording):
    r = two_epoch_recording
    positions = np.tile(r.env.bin_centers[0], (r.times.size, 1))
    result = r.env.bin_sequence_with_runs(r.times, positions)
    np.testing.assert_array_equal(result.bins, [0, 0])
    np.testing.assert_array_equal(result.run_starts, [0, 5000])
    np.testing.assert_array_equal(result.run_lengths, [5000, 5000])
    duration = (
        r.times[result.run_starts + result.run_lengths - 1] - r.times[result.run_starts]
    )
    assert np.max(duration) < 100
    np.testing.assert_array_equal(r.env.bin_sequence(r.times, positions), [0, 0])


def test_samples_in_no_run_are_dropped(small_2d_env):
    times = np.array([0.0, 0.1, 5.0, 10.0, 10.1])
    positions = small_2d_env.bin_centers[[0, 0, 1, 2, 2]]
    np.testing.assert_array_equal(
        small_2d_env.bin_sequence(times, positions, dedup=False), [0, 0, 2, 2]
    )
    result = small_2d_env.bin_sequence_with_runs(times, positions)
    np.testing.assert_array_equal(result.bins, [0, 2])
    np.testing.assert_array_equal(result.run_starts, [0, 3])
    np.testing.assert_array_equal(result.run_lengths, [2, 2])


def test_sequence_epochs_drop_excluded_samples(two_epoch_recording):
    r = two_epoch_recording
    actual = r.env.bin_sequence(
        r.times, r.positions, dedup=False, epochs=(1100.0, 1200.0)
    )
    np.testing.assert_array_equal(actual, r.env.bin_at(r.positions[5000:]))
    result = r.env.bin_sequence_with_runs(r.times, r.positions, epochs=(1100.0, 1200.0))
    assert np.all(result.run_starts >= 5000)
    assert np.sum(result.run_lengths) == 5000


def test_transition_epochs_preserve_original_alignment(two_epoch_recording):
    r = two_epoch_recording
    matrix = r.env.transitions(
        times=r.times,
        positions=r.positions,
        epochs=(1100.0, 1200.0),
        lag=3,
        allow_teleports=True,
        normalize=False,
    )
    expected = r.env.transitions(
        bins=r.env.bin_at(r.positions[5000:]),
        lag=3,
        allow_teleports=True,
        normalize=False,
    )
    assert matrix.sum() == 4997
    np.testing.assert_array_equal(matrix.toarray(), expected.toarray())


def test_raw_bin_transitions_have_no_time_gate(two_epoch_recording):
    r = two_epoch_recording
    bins = r.env.bin_at(r.positions)
    matrix = r.env.transitions(bins=bins, allow_teleports=True, normalize=False)
    assert matrix.sum() == 9999


def test_max_gap_none_preserves_unrestricted_sequence(two_epoch_recording):
    r = two_epoch_recording
    actual = r.env.bin_sequence(r.times, r.positions, dedup=False, max_gap=None)
    np.testing.assert_array_equal(actual, r.env.bin_at(r.positions))
    matrix = r.env.transitions(
        times=r.times,
        positions=r.positions,
        allow_teleports=True,
        normalize=False,
        max_gap=None,
    )
    assert matrix.sum() == 9999


@pytest.mark.parametrize(
    "name", ["bin_sequence", "bin_sequence_with_runs", "transitions"]
)
def test_invalid_epochs_still_raise_for_empty_inputs(small_2d_env, name):
    function = getattr(small_2d_env, name)
    args = (np.array([]), np.empty((0, 2)))
    with pytest.raises(ValueError, match="epochs"):
        if name == "transitions":
            function(times=args[0], positions=args[1], epochs=(1, 0))
        else:
            function(*args, epochs=(1, 0))
