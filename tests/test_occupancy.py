"""Tests for Environment.occupancy() method."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment


class TestOccupancyBasic:
    """Basic occupancy computation tests."""

    def test_occupancy_simple_stationary(self):
        """Test occupancy with stationary samples in single bin."""
        # Create simple 2D grid environment
        data = np.array([[0, 0], [10, 10]])
        env = Environment.from_samples(data, bin_size=5.0)

        # Stationary at position [5, 5] for 10 seconds
        times = np.array([0.0, 10.0])
        positions = np.array([[5.0, 5.0], [5.0, 5.0]])

        # Use max_gap=None to count the full 10-second interval
        occ = env.occupancy(times, positions, max_gap=None)

        # Should have 10 seconds in the bin containing [5, 5]
        assert occ.shape == (env.n_bins,)
        assert_allclose(occ.sum(), 10.0, rtol=1e-6)
        assert np.all(occ >= 0)

    def test_occupancy_l_shaped_path(self):
        """Test occupancy on L-shaped trajectory with known durations."""
        # Create environment covering [0, 20] x [0, 20] with proper grid
        # Use more sample points to create a proper grid
        np.random.seed(42)
        grid_samples = np.random.uniform(0, 20, size=(100, 2))
        env = Environment.from_samples(grid_samples, bin_size=5.0)

        # L-shaped path: horizontal segment, then vertical segment
        # Segment 1: (5, 5) for 0-3 seconds
        # Segment 2: (15, 5) for 3-8 seconds
        # Segment 3: (15, 15) for 8-10 seconds
        times = np.array([0.0, 3.0, 8.0, 10.0])
        positions = np.array(
            [
                [5.0, 5.0],  # Start
                [5.0, 5.0],  # End of segment 1 (3 sec at bin A)
                [15.0, 5.0],  # End of segment 2 (5 sec at bin B)
                [15.0, 15.0],  # End of segment 3 (2 sec at bin C)
            ]
        )

        # Use max_gap=10.0 to allow all intervals
        occ = env.occupancy(times, positions, max_gap=10.0)

        # Total time should be 10 seconds
        assert_allclose(occ.sum(), 10.0, rtol=1e-6)

        # Find which bins were occupied
        occupied_bins = np.where(occ > 0)[0]
        assert len(occupied_bins) >= 1  # At least one bin visited

    def test_occupancy_empty_arrays(self):
        """Test occupancy with empty input arrays."""
        data = np.array([[0, 0], [10, 10]])
        env = Environment.from_samples(data, bin_size=2.0)

        times = np.array([])
        positions = np.empty((0, 2))

        occ = env.occupancy(times, positions)

        assert occ.shape == (env.n_bins,)
        assert_allclose(occ.sum(), 0.0)
        assert np.all(occ == 0.0)

    def test_occupancy_single_sample(self):
        """Test occupancy with single sample (no intervals)."""
        data = np.array([[0, 0], [10, 10]])
        env = Environment.from_samples(data, bin_size=2.0)

        times = np.array([0.0])
        positions = np.array([[5.0, 5.0]])

        occ = env.occupancy(times, positions)

        # No intervals means no occupancy
        assert occ.shape == (env.n_bins,)
        assert_allclose(occ.sum(), 0.0)


class TestOccupancyGapHandling:
    """Test max_gap parameter for handling large time gaps."""

    def test_occupancy_with_large_gaps(self):
        """Test that large gaps are excluded from occupancy."""
        data = np.array([[0, 0], [20, 20]])
        env = Environment.from_samples(data, bin_size=5.0)

        # Two segments with 100-second gap
        times = np.array([0.0, 5.0, 105.0, 110.0])
        positions = np.array(
            [
                [5.0, 5.0],
                [5.0, 5.0],
                [15.0, 15.0],
                [15.0, 15.0],
            ]
        )

        # With default max_gap=0.5, should only count 5s + 5s = 10s
        # The 100s gap should be excluded
        occ = env.occupancy(times, positions, max_gap=0.5)

        # Should exclude the 100-second gap
        assert occ.sum() < 15.0  # Much less than total span

    def test_occupancy_max_gap_none(self):
        """Test that max_gap=None includes all intervals."""
        data = np.array([[0, 0], [20, 20]])
        env = Environment.from_samples(data, bin_size=5.0)

        times = np.array([0.0, 5.0, 105.0, 110.0])
        positions = np.array(
            [
                [5.0, 5.0],
                [5.0, 5.0],
                [15.0, 15.0],
                [15.0, 15.0],
            ]
        )

        # With max_gap=None, should count everything
        occ = env.occupancy(times, positions, max_gap=None)

        # Total time: (5-0) + (105-5) + (110-105) = 110 seconds
        assert_allclose(occ.sum(), 110.0, rtol=1e-6)


class TestOccupancySpeedFiltering:
    """Test speed filtering functionality."""

    def test_occupancy_speed_threshold(self):
        """Test that slow periods are excluded when min_speed is set."""
        data = np.array([[0, 0], [20, 20]])
        env = Environment.from_samples(data, bin_size=5.0)

        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions = np.array(
            [
                [5.0, 5.0],
                [5.0, 5.0],  # Slow period
                [5.0, 5.0],
                [15.0, 15.0],  # Fast period
                [15.0, 15.0],
            ]
        )

        # Speed values: slow, slow, fast, slow
        speeds = np.array([0.5, 0.5, 15.0, 0.5, 0.5])

        # Filter out samples with speed < 2.0
        # Use max_gap=2.0 to allow 1-second intervals
        occ_filtered = env.occupancy(
            times, positions, speed=speeds, min_speed=2.0, max_gap=2.0
        )
        occ_all = env.occupancy(times, positions, max_gap=2.0)

        # Filtered occupancy should have less total time
        assert occ_filtered.sum() < occ_all.sum()

    def test_occupancy_speed_requires_speed_array(self):
        """Test that min_speed without speed array raises error or is ignored."""
        data = np.array([[0, 0], [10, 10]])
        env = Environment.from_samples(data, bin_size=2.0)

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])

        # Providing min_speed without speed should raise ValueError
        with pytest.raises(ValueError, match=r".*speed.*"):
            env.occupancy(times, positions, min_speed=2.0)


class TestOccupancySmoothing:
    """Test kernel smoothing functionality."""

    def test_occupancy_with_kernel_smoothing(self):
        """Test that kernel smoothing spreads occupancy to neighbors."""
        # Create environment with proper grid
        np.random.seed(42)
        grid_samples = np.random.uniform(0, 20, size=(200, 2))
        env = Environment.from_samples(grid_samples, bin_size=2.0)

        # Concentrate occupancy in center bin
        times = np.array([0.0, 10.0])
        positions = np.array([[10.0, 10.0], [10.0, 10.0]])

        # Use max_gap=None to count the full interval
        occ_raw = env.occupancy(times, positions, max_gap=None)
        occ_smoothed = env.occupancy(
            times, positions, kernel_bandwidth=3.0, max_gap=None
        )

        # Smoothing should spread mass to more bins
        assert (occ_smoothed > 0).sum() > (occ_raw > 0).sum()

        # But total mass should be conserved
        assert_allclose(occ_smoothed.sum(), occ_raw.sum(), rtol=1e-4)

    def test_occupancy_smoothing_mass_conservation(self):
        """Test that smoothing conserves total occupancy time."""
        # Create environment with proper grid
        np.random.seed(42)
        grid_samples = np.random.uniform(0, 20, size=(200, 2))
        env = Environment.from_samples(grid_samples, bin_size=2.0)

        times = np.linspace(0, 100, 1000)
        positions = np.random.uniform(5, 15, size=(1000, 2))

        # Use max_gap=1.0 to allow typical intervals (default 0.5 is too small)
        occ_raw = env.occupancy(times, positions, max_gap=1.0)
        occ_smoothed = env.occupancy(
            times, positions, kernel_bandwidth=2.0, max_gap=1.0
        )

        # Mass conservation
        assert_allclose(occ_smoothed.sum(), occ_raw.sum(), rtol=1e-3)


class TestOccupancyOutsideBehavior:
    """Test handling of samples outside environment bounds."""

    def test_occupancy_all_outside(self):
        """Test occupancy when all samples are outside environment."""
        data = np.array([[0, 0], [10, 10]])
        env = Environment.from_samples(data, bin_size=2.0)

        # All positions outside [0, 10] range
        times = np.array([0.0, 5.0, 10.0])
        positions = np.array(
            [
                [20.0, 20.0],
                [20.0, 20.0],
                [20.0, 20.0],
            ]
        )

        occ = env.occupancy(times, positions)

        # All bins should have zero occupancy
        assert_allclose(occ.sum(), 0.0)
        assert np.all(occ == 0.0)

    def test_occupancy_mixed_inside_outside(self):
        """Test occupancy with mix of inside and outside samples."""
        data = np.array([[0, 0], [10, 10]])
        env = Environment.from_samples(data, bin_size=2.0)

        times = np.array([0.0, 5.0, 10.0, 15.0])
        positions = np.array(
            [
                [5.0, 5.0],  # Inside
                [5.0, 5.0],  # Inside
                [20.0, 20.0],  # Outside
                [5.0, 5.0],  # Inside
            ]
        )

        # Use max_gap=None to count all intervals
        occ = env.occupancy(times, positions, max_gap=None)

        # Should only count time inside environment
        # Interval [0, 5] is inside, [5, 10] starts inside but ends outside,
        # [10, 15] starts outside (excluded)
        assert occ.shape == (env.n_bins,)
        assert occ.sum() > 0  # Some occupancy counted (at least the first interval)


class TestOccupancyValidation:
    """Test input validation and error handling."""

    def test_occupancy_mismatched_lengths(self):
        """Test that mismatched times/positions raises error."""
        data = np.array([[0, 0], [10, 10]])
        env = Environment.from_samples(data, bin_size=2.0)

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[5.0, 5.0], [5.0, 5.0]])  # Only 2 positions

        with pytest.raises(ValueError, match=r".*length.*"):
            env.occupancy(times, positions)

    def test_occupancy_wrong_dimensions(self):
        """Test that positions with wrong dimensions raises error."""
        data = np.array([[0, 0], [10, 10]])
        env = Environment.from_samples(data, bin_size=2.0)

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[5.0], [5.0], [5.0]])  # 1D instead of 2D

        with pytest.raises(ValueError, match=r".*dimension.*"):
            env.occupancy(times, positions)


class TestOccupancyMassConservation:
    """Property tests for mass conservation."""

    def test_occupancy_conserves_time(self):
        """Test that total occupancy equals total valid time."""
        data = np.array([[0, 0], [20, 20]])
        env = Environment.from_samples(data, bin_size=2.0)

        times = np.linspace(0, 50, 100)
        np.random.seed(42)
        positions = np.random.uniform(2, 18, size=(100, 2))

        occ = env.occupancy(times, positions, max_gap=1.0)

        # Compute expected total time (excluding large gaps)
        dt = np.diff(times)
        valid_dt = dt[dt <= 1.0]
        expected_time = valid_dt.sum()

        assert_allclose(occ.sum(), expected_time, rtol=1e-6)

    def test_occupancy_nonnegative(self):
        """Test that occupancy is always non-negative."""
        data = np.array([[0, 0], [20, 20]])
        env = Environment.from_samples(data, bin_size=2.0)

        times = np.linspace(0, 10, 50)
        np.random.seed(42)
        positions = np.random.uniform(0, 20, size=(50, 2))

        occ = env.occupancy(times, positions)

        assert np.all(occ >= 0)


class TestOccupancyPerformance:
    """Performance tests."""

    def test_occupancy_large_trajectory(self):
        """Test occupancy computation on large trajectory (performance check)."""
        data = np.array([[0, 0], [100, 100]])
        env = Environment.from_samples(data, bin_size=5.0)

        # 100k samples (not 1M to keep test fast, but validates scaling)
        n_samples = 100_000
        times = np.linspace(0, 1000, n_samples)
        np.random.seed(42)
        positions = np.random.uniform(10, 90, size=(n_samples, 2))

        # Should complete quickly
        import time

        start = time.time()
        occ = env.occupancy(times, positions)
        elapsed = time.time() - start

        # Verify result
        assert occ.shape == (env.n_bins,)
        assert occ.sum() > 0

        # Should be fast (100k samples << 1s, 1M would be ~10x slower)
        assert elapsed < 5.0  # Generous bound for CI


class TestOccupancyMultipleLayouts:
    """Test occupancy works across different layout types."""

    def test_occupancy_on_regular_grid(self):
        """Test occupancy on regular grid layout."""
        data = np.array([[0, 0], [20, 20]])
        env = Environment.from_samples(data, bin_size=5.0)

        times = np.array([0.0, 10.0])
        positions = np.array([[10.0, 10.0], [10.0, 10.0]])

        # Use max_gap=None to count the full interval
        occ = env.occupancy(times, positions, max_gap=None)
        assert occ.shape == (env.n_bins,)
        assert_allclose(occ.sum(), 10.0, rtol=1e-6)

    def test_occupancy_on_masked_grid(self):
        """Test occupancy on masked grid layout."""
        # Create L-shaped mask
        mask = np.zeros((10, 10), dtype=bool)
        mask[:5, :5] = True  # Bottom-left quadrant
        mask[5:, :5] = True  # Top-left quadrant

        # Create grid edges for 10x10 grid with 2.0 spacing
        grid_edges_x = np.linspace(0, 20, 11)
        grid_edges_y = np.linspace(0, 20, 11)

        env = Environment.from_mask(
            active_mask=mask, grid_edges=(grid_edges_x, grid_edges_y)
        )

        times = np.array([0.0, 10.0])
        positions = np.array([[5.0, 5.0], [5.0, 5.0]])

        # Use max_gap=None to count the full interval
        occ = env.occupancy(times, positions, max_gap=None)
        assert occ.shape == (env.n_bins,)
