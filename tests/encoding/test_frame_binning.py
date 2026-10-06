"""Frame spike counts conserve only spikes in valid half-open intervals."""

import numpy as np
import pytest


@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_frame_counts_half_open_boundaries(offset):
    from neurospatial.encoding._binning import count_spikes_by_frame

    times = offset + np.array([0.1, 0.2, 0.3])
    spikes = np.r_[times[0] - 1, times, times[-1] + 1]
    counts = count_spikes_by_frame(
        spikes,
        times,
        np.array([0, 1, 2], dtype=np.intp),
        np.array([True, True]),
        3,
    )
    np.testing.assert_array_equal(counts, [1, 1, 0])
    assert counts.dtype == np.float64


def test_frame_counts_match_interval_reference_and_chunks():
    from neurospatial.encoding._binning import count_spikes_by_frame

    rng = np.random.default_rng(52)
    times = np.cumsum(rng.uniform(0.02, 1, 3000))
    bins = rng.integers(-1, 8, len(times), dtype=np.intp)
    mask = rng.random(len(times) - 1) > 0.2
    spikes = np.sort(rng.uniform(times[0] - 1, times[-1] + 1, 5000))
    expected = np.zeros(8)
    for k, (start, stop) in enumerate(zip(times[:-1], times[1:], strict=True)):
        if mask[k] and bins[k] >= 0:
            expected[bins[k]] += np.count_nonzero((spikes >= start) & (spikes < stop))
    counts = count_spikes_by_frame(spikes, times, bins, mask, 8)
    np.testing.assert_array_equal(counts, expected)
    chunked = sum(
        count_spikes_by_frame(chunk, times, bins, mask, 8)
        for chunk in np.array_split(spikes, 7)
    )
    np.testing.assert_array_equal(chunked, counts)


@pytest.mark.parametrize("spikes", [[], [-1, 3], [0, 0.5, 1]])
def test_frame_counts_exclude_invalid_intervals_and_bins(spikes):
    from neurospatial.encoding._binning import count_spikes_by_frame

    times = np.array([0.0, 0.5, 1.0, 1.5])
    counts = count_spikes_by_frame(
        np.asarray(spikes),
        times,
        np.array([0, -1, 1, 2], dtype=np.intp),
        np.array([False, True, True]),
        3,
    )
    expected = [0, int(1 in spikes), 0]
    np.testing.assert_array_equal(counts, expected)
