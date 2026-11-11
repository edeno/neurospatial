"""The line fit of Ólafsdóttir et al. (2016) as its Supplementary Software runs it.

``lineTraj_decode.m`` pads the posterior with zeros beyond the track, so a
time bin where the line has left the track adds nothing
(``off_track="zero"``), and discards lines on the posterior for fewer than a
third of its time bins (``min_on_track=1 / 3``).
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.replay import detect_trajectory_radon

DT = 0.02


@pytest.fixture(scope="module")
def track():
    """A 20 cm track in 2 cm bins."""
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [20.0, 0.0]], bin_size=2.0, units="cm"
    )


def _rows(n_bins, *bins):
    rows = np.zeros((len(bins), n_bins))
    for row, b in enumerate(bins):
        rows[row, b] = 1.0
    return rows


def _entry(result, speed, start):
    i = np.flatnonzero(np.isclose(result.sinogram_speeds, speed))
    j = np.flatnonzero(np.isclose(result.sinogram_start_positions, start))
    assert i.size == 1
    assert j.size == 1
    return result.sinogram[i[0], j[0]]


def test_a_time_bin_off_the_track_adds_nothing(track):
    # 25 cm/s from 18.2 cm keeps bin 9 ([18, 20] cm) in its 2 cm band for
    # rows 0-3 and leaves the track at row 4: 4 of 5 rows score 1.
    last = np.full(track.n_bins, 1.0 / track.n_bins)
    posterior = np.vstack([_rows(track.n_bins, 9, 9, 9, 9), last])
    times = np.arange(5) * DT
    options = {"band": 2.0, "max_speed": 100.0}
    zero = detect_trajectory_radon(track, posterior, times, off_track="zero", **options)
    median = detect_trajectory_radon(track, posterior, times, **options)
    assert _entry(zero, 25.0, 18.2) == pytest.approx(4.0 / 5, abs=1e-12)
    # Davidson's band median gives the off-track row the mass of a band, 0.2.
    assert _entry(median, 25.0, 18.2) == pytest.approx((4.0 + 0.2) / 5, abs=1e-9)


def test_lines_on_the_track_for_under_a_third_of_the_time_bins_are_not_searched(track):
    posterior = np.full((9, track.n_bins), 1.0 / track.n_bins)
    times = np.arange(9) * DT
    options = {"band": 2.0, "max_speed": 1000.0, "off_track": "zero"}
    every = detect_trajectory_radon(track, posterior, times, **options)
    third = detect_trajectory_radon(
        track, posterior, times, min_on_track=1 / 3, **options
    )
    position = (
        every.sinogram_start_positions[None, :, None]
        + every.sinogram_speeds[:, None, None] * times[None, None, :]
    )
    on_track = ((position >= 0.0) & (position <= 20.0)).sum(axis=2)
    searched = np.isfinite(third.sinogram)
    # Kept: on the track at 3 of 9 time bins or more, and searched without the rule.
    assert np.array_equal(searched, np.isfinite(every.sinogram) & (on_track >= 3))
    assert searched.sum() < np.isfinite(every.sinogram).sum()


def test_min_on_track_outside_zero_to_one_is_refused(track):
    posterior = np.full((3, track.n_bins), 1.0 / track.n_bins)
    with pytest.raises(ValueError, match=r"min_on_track=1\.5"):
        detect_trajectory_radon(track, posterior, np.arange(3) * DT, min_on_track=1.5)


def test_zero_off_track_needs_lines_that_can_leave_the_track(track):
    posterior = np.full((3, track.n_bins), 1.0 / track.n_bins)
    with pytest.raises(ValueError, match="off_track would change nothing"):
        detect_trajectory_radon(
            track, posterior, np.arange(3) * DT, off_track="zero", line_ends="on_track"
        )
