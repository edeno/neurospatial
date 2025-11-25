"""Tests for Environment.bin_sequence() method.

This module tests the trajectory-to-bin-sequence conversion functionality,
including deduplication and run-length encoding.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial import Environment


class TestBinSequenceBasic:
    """Basic bin_sequence functionality tests."""

    def test_simple_trajectory(self):
        """Test basic trajectory mapping to bin sequence."""
        # Create simple 1D environment: 5 bins from 0 to 10
        env = Environment.from_samples(
            np.array([[0.0], [2.0], [4.0], [6.0], [8.0], [10.0]]),
            bin_size=2.5,
        )

        # Trajectory: move from bin 0 → bin 1 → bin 2
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        positions = np.array([[1.0], [3.0], [3.5], [5.0], [6.5], [7.0]])

        bins = env.bin_sequence(times, positions, dedup=False)

        assert bins.shape == (6,)
        assert bins.dtype == np.int32
        # All bins should be valid (>= 0)
        assert np.all(bins >= 0)

    def test_known_trajectory_bins(self):
        """Test trajectory with known expected bin sequence."""
        # Create 2D grid: 3x3 bins
        x = np.linspace(0, 10, 4)
        y = np.linspace(0, 10, 4)
        xx, yy = np.meshgrid(x, y)
        samples = np.column_stack([xx.ravel(), yy.ravel()])

        env = Environment.from_samples(samples, bin_size=3.5)

        # L-shaped trajectory: bottom-left → bottom-middle → middle-middle
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array(
            [
                [1.0, 1.0],  # Bottom-left bin
                [5.0, 1.0],  # Bottom-middle bin
                [5.0, 5.0],  # Middle-middle bin
            ]
        )

        bins = env.bin_sequence(times, positions, dedup=False)

        # All three positions should map to different bins
        assert len(bins) == 3
        assert len(np.unique(bins)) == 3, "Should visit 3 different bins"

    def test_empty_trajectory(self):
        """Test handling of empty input arrays."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0]]),
            bin_size=5.0,
        )

        times = np.array([])
        positions = np.array([]).reshape(0, 1)

        bins = env.bin_sequence(times, positions)

        assert bins.shape == (0,)
        assert bins.dtype == np.int32

    def test_single_sample(self):
        """Test trajectory with single sample."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0]]),
            bin_size=5.0,
        )

        times = np.array([0.0])
        positions = np.array([[2.5]])

        bins = env.bin_sequence(times, positions)

        assert bins.shape == (1,)
        assert bins[0] >= 0


class TestBinSequenceDeduplication:
    """Test deduplication behavior."""

    def test_dedup_consecutive_repeats(self):
        """Test that consecutive repeats are collapsed with dedup=True."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0], [15.0]]),
            bin_size=5.0,
        )

        # Trajectory: stay in same bin, then move, stay, move
        # Bin centers: [0, 5, 10, 15]
        times = np.arange(7, dtype=float)
        positions = np.array(
            [
                [0.5],  # Bin 0
                [1.0],  # Bin 0 (same)
                [1.5],  # Bin 0 (same)
                [5.5],  # Bin 1 (different)
                [6.0],  # Bin 1 (same)
                [10.5],  # Bin 2 (different)
                [11.0],  # Bin 2 (same)
            ]
        )

        bins_no_dedup = env.bin_sequence(times, positions, dedup=False)
        bins_dedup = env.bin_sequence(times, positions, dedup=True)

        # Without dedup: all 7 samples
        assert len(bins_no_dedup) == 7

        # With dedup: should collapse to 3 unique runs (bins 0, 1, 2)
        assert len(bins_dedup) == 3
        # All values should still be from the original bins
        assert set(bins_dedup).issubset(set(bins_no_dedup))

    def test_dedup_default_true(self):
        """Test that dedup=True is the default behavior."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0]]),
            bin_size=5.0,
        )

        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array([[2.0], [2.5], [7.0], [7.5]])

        bins_default = env.bin_sequence(times, positions)
        bins_explicit = env.bin_sequence(times, positions, dedup=True)

        assert_array_equal(bins_default, bins_explicit)

    def test_no_dedup_preserves_all_samples(self):
        """Test that dedup=False keeps all samples."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0]]),
            bin_size=5.0,
        )

        times = np.arange(10, dtype=float)
        positions = np.full((10, 1), 2.5)  # All in same bin

        bins = env.bin_sequence(times, positions, dedup=False)

        assert len(bins) == 10
        # All should be the same bin
        assert len(np.unique(bins)) == 1


class TestBinSequenceRuns:
    """Test run-length encoding functionality."""

    def test_return_runs_basic(self):
        """Test that return_runs=True returns run boundaries."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0], [15.0]]),
            bin_size=5.0,
        )

        # Three distinct runs: AAA, BB, CCCC
        # Bin centers: [0, 5, 10, 15]
        times = np.arange(9, dtype=float)
        positions = np.array(
            [
                [0.5],
                [1.0],
                [1.5],  # Run 1: bin 0 (3 samples, clearly < 2.5)
                [5.5],
                [6.0],  # Run 2: bin 1 (2 samples, clearly > 2.5 and < 7.5)
                [10.5],
                [11.0],
                [11.5],
                [12.0],  # Run 3: bin 2 (4 samples, clearly > 7.5 and < 12.5)
            ]
        )

        _bins, run_starts, run_ends = env.bin_sequence(
            times, positions, dedup=False, return_runs=True
        )

        # Should have 3 runs
        assert len(run_starts) == 3
        assert len(run_ends) == 3

        # First run: indices 0-2 (3 samples)
        assert run_starts[0] == 0
        assert run_ends[0] == 2

        # Second run: indices 3-4 (2 samples)
        assert run_starts[1] == 3
        assert run_ends[1] == 4

        # Third run: indices 5-8 (4 samples)
        assert run_starts[2] == 5
        assert run_ends[2] == 8

    def test_run_boundaries_with_dedup(self):
        """Test that runs work correctly with deduplication."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0]]),
            bin_size=5.0,
        )

        # Bin centers: [0, 5, 10]
        times = np.arange(6, dtype=float)
        positions = np.array(
            [
                [0.5],
                [1.0],  # Bin 0
                [5.5],
                [6.0],  # Bin 1
                [0.5],
                [1.0],  # Bin 0 again
            ]
        )

        bins, run_starts, run_ends = env.bin_sequence(
            times, positions, dedup=True, return_runs=True
        )

        # With dedup: should have 3 unique bins in sequence (0, 1, 0)
        assert len(bins) == 3

        # Should have 3 runs corresponding to the deduplicated sequence
        assert len(run_starts) == 3
        assert len(run_ends) == 3

    def test_run_duration_calculation(self):
        """Test that run boundaries enable duration calculations."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0]]),
            bin_size=5.0,
        )

        # Bin centers: [0, 5, 10]
        times = np.array([0.0, 1.0, 2.0, 5.0, 8.0, 9.0])
        positions = np.array(
            [
                [0.5],
                [1.0],
                [1.5],  # Run 1: bin 0 (0-2s)
                [5.5],
                [6.0],
                [6.5],  # Run 2: bin 1 (5-9s)
            ]
        )

        _bins, run_starts, run_ends = env.bin_sequence(
            times, positions, dedup=False, return_runs=True
        )

        # Calculate duration of first run
        duration_1 = times[run_ends[0]] - times[run_starts[0]]
        assert duration_1 == 2.0

        # Calculate duration of second run
        duration_2 = times[run_ends[1]] - times[run_starts[1]]
        assert duration_2 == 4.0

    def test_single_sample_run(self):
        """Test run encoding with single sample."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0]]),
            bin_size=5.0,
        )

        times = np.array([0.0])
        positions = np.array([[2.5]])

        _bins, run_starts, run_ends = env.bin_sequence(
            times, positions, return_runs=True
        )

        assert len(run_starts) == 1
        assert len(run_ends) == 1
        assert run_starts[0] == 0
        assert run_ends[0] == 0


class TestBinSequenceOutsideBehavior:
    """Test handling of samples outside environment."""

    def test_outside_value_default(self):
        """Test that outside samples default to -1."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0]]),
            bin_size=5.0,
        )

        times = np.arange(5, dtype=float)
        positions = np.array(
            [
                [2.5],  # Inside
                [50.0],  # Outside (far away)
                [2.5],  # Inside
                [-10.0],  # Outside (far away)
                [7.5],  # Inside
            ]
        )

        bins = env.bin_sequence(times, positions, dedup=False)

        # Outside samples should be -1
        assert bins[1] == -1
        assert bins[3] == -1
        # Inside samples should be >= 0
        assert bins[0] >= 0
        assert bins[2] >= 0
        assert bins[4] >= 0

    def test_outside_value_none_drops_samples(self):
        """Test that outside_value=None removes outside samples."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0]]),
            bin_size=5.0,
        )

        times = np.arange(5, dtype=float)
        positions = np.array(
            [
                [2.5],  # Inside
                [50.0],  # Outside
                [2.5],  # Inside
                [-10.0],  # Outside
                [7.5],  # Inside
            ]
        )

        bins = env.bin_sequence(times, positions, outside_value=None, dedup=False)

        # Should only have 3 samples (the inside ones)
        assert len(bins) == 3
        # All should be valid bins (>= 0)
        assert np.all(bins >= 0)

    def test_outside_splits_runs(self):
        """Test that outside samples split runs correctly."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0]]),
            bin_size=5.0,
        )

        # Bin centers: [0, 5, 10]
        times = np.arange(7, dtype=float)
        positions = np.array(
            [
                [0.5],
                [1.0],  # Run 1: bin 0 (indices 0-1)
                [50.0],  # Outside (splits runs, index 2)
                [5.5],
                [6.0],  # Run 2: bin 1 (indices 3-4)
                [50.0],  # Outside (splits runs, index 5)
                [0.5],  # Run 3: bin 0 again (index 6)
            ]
        )

        _bins, run_starts, run_ends = env.bin_sequence(
            times, positions, dedup=False, return_runs=True
        )

        # Should have 5 runs total:
        # Run 1: bin 0 (indices 0-1)
        # Run 2: outside/=-1 (index 2)
        # Run 3: bin 1 (indices 3-4)
        # Run 4: outside=-1 (index 5)
        # Run 5: bin 0 (index 6)
        assert len(run_starts) == 5
        assert len(run_ends) == 5


class TestBinSequenceValidation:
    """Test input validation."""

    def test_mismatched_lengths(self):
        """Test that mismatched times/positions raises ValueError."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0]]),
            bin_size=5.0,
        )

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[2.5], [7.5]])  # Only 2 positions

        with pytest.raises(ValueError, match=r"times and positions.*same length"):
            env.bin_sequence(times, positions)

    def test_wrong_position_dimensions(self):
        """Test that wrong position dimensions raises ValueError."""
        # Create 2D environment
        samples = np.array([[0, 0], [5, 0], [0, 5], [5, 5]])
        env = Environment.from_samples(samples, bin_size=5.0)

        times = np.array([0.0, 1.0, 2.0])
        positions_1d = np.array([[2.5], [3.5], [4.5]])  # Wrong: 1D not 2D

        with pytest.raises(ValueError, match=r"positions.*dimensions"):
            env.bin_sequence(times, positions_1d)

    def test_requires_fitted_environment(self):
        """Test that check_fitted decorator works."""
        # Create environment normally - it will be fitted
        # This test just verifies the decorator is applied
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0]]),
            bin_size=5.0,
        )

        times = np.array([0.0, 1.0])
        positions = np.array([[2.5], [7.5]])

        # Should work fine - environment is fitted
        bins = env.bin_sequence(times, positions)
        assert len(bins) > 0

        # Note: We can't easily test unfitted environment without
        # breaking encapsulation, but the @check_fitted decorator
        # ensures this method requires fitted state

    def test_non_monotonic_times_error(self):
        """Test error for non-monotonic timestamps."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0]]),
            bin_size=5.0,
        )

        # Non-monotonic times
        times = np.array([0.0, 2.0, 1.0, 3.0])  # 1.0 < 2.0 violates monotonicity
        positions = np.array([[2.5], [7.5], [2.5], [7.5]])

        # Should raise ValueError for non-monotonic times
        with pytest.raises(ValueError, match="monotonically increasing"):
            env.bin_sequence(times, positions)


class TestBinSequenceMultipleLayouts:
    """Test bin_sequence across different layout types."""

    def test_regular_grid_2d(self):
        """Test bin_sequence on 2D regular grid."""
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 20, (100, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        times = np.arange(50, dtype=float)
        positions = rng.uniform(0, 20, (50, 2))

        bins = env.bin_sequence(times, positions, dedup=False)

        assert bins.shape == (50,)
        assert bins.dtype == np.int32

    def test_polygon_layout(self):
        """Test bin_sequence on polygon layout."""
        from shapely.geometry import box

        # Create polygon-based environment
        rng = np.random.default_rng(42)
        polygon = box(0, 0, 10, 10)
        env = Environment.from_polygon(
            polygon=polygon,
            bin_size=2.5,
        )

        times = np.arange(20, dtype=float)
        positions = rng.uniform(0, 10, (20, 2))

        bins = env.bin_sequence(times, positions)

        # Length may be less than 20 if dedup=True (default)
        assert len(bins) <= 20
        # Some may be outside (-1) if they fall outside polygon
        assert bins.dtype == np.int32


class TestBinSequenceEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_samples_outside(self):
        """Test trajectory where all samples are outside environment."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0]]),
            bin_size=5.0,
        )

        times = np.arange(5, dtype=float)
        positions = np.full((5, 1), 100.0)  # All far outside

        bins = env.bin_sequence(times, positions, dedup=False)

        # All should be -1 (outside)
        assert np.all(bins == -1)

    def test_all_samples_outside_with_none(self):
        """Test all outside samples with outside_value=None."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0]]),
            bin_size=5.0,
        )

        times = np.arange(5, dtype=float)
        positions = np.full((5, 1), 100.0)  # All far outside

        bins = env.bin_sequence(times, positions, outside_value=None)

        # Should return empty array
        assert len(bins) == 0
        assert bins.dtype == np.int32

    def test_rapid_bin_changes(self):
        """Test trajectory with rapid bin changes (no consecutive repeats)."""
        env = Environment.from_samples(
            np.array([[0.0], [5.0], [10.0], [15.0], [20.0]]),
            bin_size=5.0,
        )

        times = np.arange(4, dtype=float)
        positions = np.array([[2.5], [7.5], [12.5], [17.5]])  # Different bin each time

        bins_no_dedup = env.bin_sequence(times, positions, dedup=False)
        bins_dedup = env.bin_sequence(times, positions, dedup=True)

        # No consecutive repeats, so dedup should have no effect
        assert_array_equal(bins_no_dedup, bins_dedup)

    def test_boundary_crossing(self):
        """Test trajectory that crosses bin boundaries."""
        env = Environment.from_samples(
            np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]),
            bin_size=5.0,
        )

        # Diagonal trajectory crossing multiple bins
        times = np.linspace(0, 1, 20)
        positions = np.column_stack(
            [
                np.linspace(1, 9, 20),
                np.linspace(1, 9, 20),
            ]
        )

        bins = env.bin_sequence(times, positions)

        # Should visit multiple bins along diagonal
        assert len(np.unique(bins)) > 1
