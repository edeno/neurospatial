"""Independent window tiling and half-open spike-counting regressions."""

import numpy as np
import pytest

from neurospatial.decoding import bin_spikes_in_time


def helpers():
    from neurospatial.decoding._binning import (
        count_spikes_in_time_bins,
        time_bins_in_windows,
    )

    return time_bins_in_windows, count_spikes_in_time_bins


def test_epochs_tile_each_window():
    counts, centers = bin_spikes_in_time(
        [np.array([0.01, 1.5, 2.01])], 0.25, epochs=[(0, 1), (2, 3)]
    )
    np.testing.assert_array_equal(
        centers, np.r_[np.arange(0.125, 1, 0.25), np.arange(2.125, 3, 0.25)]
    )
    assert counts.shape == (8, 1)
    assert counts.sum() == 2
    with pytest.raises(ValueError, match="not both"):
        bin_spikes_in_time([], 0.25, t_start=0, epochs=(0, 1))


def test_time_bins_decimal_boundary():
    tile, count = helpers()
    left, right = tile(np.array([[0.1, 0.3]]), 0.1)
    np.testing.assert_array_equal(np.c_[left, right], [[0.1, 0.2], [0.2, 0.3]])
    np.testing.assert_array_equal(
        count([np.array([0.1, 0.2, 0.3])], left, right), [[1], [1]]
    )


@pytest.mark.parametrize("dt,stop,n", [(0.1, 0.3, 2), (0.025, 100.1, 4000)])
def test_time_bins_large_offset(dt, stop, n):
    tile, count = helpers()
    left, right = tile(np.array([[1e9 + 0.1, 1e9 + stop]]), dt)
    assert len(left) == n
    assert right[-1] == 1e9 + stop
    assert count([np.array([1e9 + stop])], left, right).sum() == 0


def test_bins_never_exceed_window():
    tile, count = helpers()
    rng = np.random.default_rng(41)
    for _ in range(2000):
        start = rng.choice([0, 1e3, 1e6, 1e9]) + rng.uniform(0, 10)
        dt = float(rng.choice([0.001, 0.002, 0.01, 0.025, 0.1, 0.2, 0.3]))
        n = int(rng.integers(1, 2001))
        stop = start + n * dt
        left, right = tile(np.array([[start, stop]]), dt)
        assert len(left) == n
        assert np.all(left < right)
        assert np.all(right <= stop)
        assert count([np.array([stop])], left, right).sum() == 0


def test_time_bins_reject_insufficient_precision():
    tile, _ = helpers()
    with pytest.raises(ValueError, match="Fix: subtract a time origin"):
        tile(np.array([[1e9, 1e9 + 1e-6]]), 2e-7)


@pytest.mark.parametrize("dt,n", [(5e-4, 20000), (2e-3, 5000)])
def test_time_bins_unix_epoch_timestamps(dt, n):
    tile, _ = helpers()
    left, right = tile(np.array([[1.7e9, 1.7e9 + 10]]), dt)
    assert len(left) == n
    assert np.all(right > left)
    assert np.all(right - left >= 0.99 * dt)


def test_partial_bin_dropped():
    tile, count = helpers()
    left, right = tile(np.array([[0, 1.05]]), 0.25)
    assert len(left) == 4 and right[-1] == 1.0
    assert count([np.array([0.99, 1, 1.04])], left, right).sum() == 1


def test_chunked_counts_equal_full():
    tile, count = helpers()
    left, right = tile(np.array([[0, 99.98], [1100, 1199.98]]), 0.025)
    rng = np.random.default_rng(5)
    trains = [np.sort(rng.uniform(-1, 1201, 4000)) for _ in range(5)]
    expected = np.array(
        [
            [np.count_nonzero((s >= a) & (s < b)) for s in trains]
            for a, b in zip(left, right, strict=True)
        ]
    )
    full = count(trains, left, right)
    np.testing.assert_array_equal(full, expected)
    blocks = []
    for start in range(0, len(left), 1000):
        stop = min(start + 1000, len(left))
        scoped = [s[(s >= left[start]) & (s < right[stop - 1])] for s in trains]
        blocks.append(count(scoped, left[start:stop], right[start:stop]))
    np.testing.assert_array_equal(np.concatenate(blocks), full)
    assert full.dtype == np.int64


def test_bins_are_half_open():
    counts, _ = bin_spikes_in_time(
        [np.array([0.0, 0.5, 1.0])], 0.5, t_start=0, t_stop=1
    )
    np.testing.assert_array_equal(counts, [[1], [1]])
