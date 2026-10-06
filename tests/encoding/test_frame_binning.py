"""Frame spike counts conserve only spikes in valid half-open intervals."""

from itertools import pairwise

import numpy as np
import pytest


@pytest.fixture
def short_frame_recording(continuous_recording):
    from dataclasses import replace

    return replace(
        continuous_recording,
        times=np.array([0.0, 0.5, 10.0, 10.5]),
        positions=np.tile([40.0, 40.0], (4, 1)),
        headings=np.zeros(4),
        spike_times=np.array([0.25, 0.5, 5.0, 10.25, 10.5]),
    )


def test_family_kernel_drops_pauses_and_final_sample(
    frame_family, short_frame_recording
):
    f, r = frame_family, short_frame_recording
    counts = f.count(*f.args(r, r.spike_times, kernel=True), **f.kernel_defaults)
    if f.name == "egocentric":
        counts = counts[0]
    assert counts.sum() == 2
    result = f.occupancy(*f.args(r, occupancy=True, kernel=True), **f.kernel_defaults)
    occupancy = result if f.name == "view" else result[0]
    assert occupancy.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("n_units", [0, 1, 3])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_family_kernel_shares_window_mask(
    frame_family, short_frame_recording, monkeypatch, n_units, n_jobs
):
    f, r = frame_family, short_frame_recording
    from neurospatial.environment import trajectory

    original = trajectory.interval_valid_mask
    masks = []

    def track_mask(*args, **kwargs):
        mask = original(*args, **kwargs)
        masks.append(mask)
        return mask

    monkeypatch.setattr(trajectory, "interval_valid_mask", track_mask)
    # Also support direct imports in the kernel module.
    monkeypatch.setattr(f.binning, "interval_valid_mask", track_mask, raising=False)
    window = np.array([[0.0, 0.5]])
    result = f.counts(
        *f.args(r, [r.spike_times] * n_units, kernel=True),
        **f.kernel_defaults,
        epochs=window,
        spike_window=window,
        n_jobs=n_jobs,
    )
    counts, occupancy = result[:2]
    assert len(masks) == 1
    np.testing.assert_array_equal(masks[0], [True, False, False])
    assert counts.shape == (n_units, len(occupancy))
    np.testing.assert_array_equal(counts.sum(axis=1), np.ones(n_units))
    assert occupancy.sum() == pytest.approx(0.5)


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
    for k, (start, stop) in enumerate(pairwise(times)):
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
