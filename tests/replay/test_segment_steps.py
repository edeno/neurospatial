"""replay_segments: bridging gaps and thresholds equal within rounding."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.replay import replay_segments

DT = 0.02


@pytest.fixture(scope="module")
def track():
    """10 bins of 10 cm: bin b is centered at 10 b + 5 cm."""
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    )


def _map_sequence(track, bins):
    return np.eye(track.n_bins)[list(bins)], np.arange(len(bins)) * DT


class TestBridgingGaps:
    """With max_gap_bins, a segment continues to the nearest bin within reach."""

    def test_a_steady_sweep_keeps_every_time_bin(self, track):
        posterior, times = _map_sequence(track, [1, 2, 3, 4, 5, 6])
        result = replay_segments(track, posterior, times, max_jump=25.0, max_gap_bins=1)
        positions = 10.0 * np.arange(1, 7) + 5.0
        np.testing.assert_array_equal(result.segments, [[0, 6]])
        assert result.trajectory_time_bins.all()
        np.testing.assert_allclose(result.path_lengths, [50.0])
        rms = np.sqrt(np.mean((positions - positions.mean()) ** 2))  # 17.08 cm
        np.testing.assert_allclose(result.dispersions, [rms])

    def test_a_zigzag_follows_every_step(self, track):
        """15 -> 35 -> 25 -> 45 cm: 20 + 10 + 20 cm, every step within 25 cm.

        Skipping the bin at 35 cm (15 -> 25 -> 45 cm) would measure 30 cm.
        """
        posterior, times = _map_sequence(track, [1, 3, 2, 4])
        result = replay_segments(track, posterior, times, max_jump=25.0, max_gap_bins=1)
        np.testing.assert_array_equal(result.segments, [[0, 4]])
        assert result.trajectory_time_bins.all()
        np.testing.assert_allclose(result.path_lengths, [50.0])

    def test_an_outlier_bin_is_skipped(self, track):
        """15 -> (85) -> 25 -> 35 cm: the segment skips the outlier."""
        posterior, times = _map_sequence(track, [1, 8, 2, 3])
        result = replay_segments(track, posterior, times, max_jump=25.0, max_gap_bins=1)
        np.testing.assert_array_equal(result.segments, [[0, 4]])
        np.testing.assert_array_equal(
            result.trajectory_time_bins, [True, False, True, True]
        )
        np.testing.assert_allclose(result.path_lengths, [20.0])


class TestThresholdsEqualWithinRounding:
    """A value within 1e-9 of a threshold counts as equal to it."""

    @staticmethod
    def _sweep_with_peak(track, peak):
        n = track.n_bins
        bins = [1, 2, 3, 4]
        posterior = np.full((len(bins), n), (1.0 - peak) / (n - 1))
        posterior[np.arange(len(bins)), bins] = peak
        return posterior, np.arange(len(bins)) * DT

    @pytest.mark.parametrize(
        ("excess", "n_segments"), [(1e-12, 0), (-1e-12, 0), (1e-6, 1)]
    )
    def test_peak_must_exceed_the_threshold(self, track, excess, n_segments):
        """min_peak_probability rejects peaks not above it; equal is not above."""
        threshold = 5.0 / track.n_bins
        posterior, times = self._sweep_with_peak(track, threshold + excess)
        result = replay_segments(
            track, posterior, times, max_jump=25.0, min_peak_probability=threshold
        )
        assert len(result.segments) == n_segments

    @pytest.mark.parametrize(
        ("max_jump", "n_segments"),
        [(10.0 + 1e-12, 0), (10.0 - 1e-12, 0), (10.0 + 1e-6, 1)],
    )
    def test_a_jump_of_max_jump_separates(self, track, max_jump, n_segments):
        """Steps of 10 cm: with max_jump equal to 10 cm, no two bins join."""
        posterior, times = _map_sequence(track, [1, 2, 3])
        result = replay_segments(track, posterior, times, max_jump=max_jump)
        assert len(result.segments) == n_segments
