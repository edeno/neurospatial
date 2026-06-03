"""Tests for Environment.occupancy() method."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment


class TestOccupancyBasic:
    """Basic occupancy computation tests.

    Note: test_occupancy_simple_stationary requires inline environment creation
    because it needs specific bin_size=5.0 on a 10x10 grid. The minimal_2d_grid_env
    fixture uses bin_size=2.0, which would create different bin structure.
    """

    def test_occupancy_simple_stationary(self):
        """Test occupancy with stationary samples in single bin."""
        # Dense grid env so position (5, 5) lands in an active bin
        # rather than being flagged out-of-mask by bin_at (M1 1.2).
        x = np.linspace(0.0, 10.0, 11)
        xx, yy = np.meshgrid(x, x)
        data = np.column_stack([xx.ravel(), yy.ravel()])
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

    def test_occupancy_l_shaped_path(self, minimal_20x20_grid_env):
        """Test occupancy on L-shaped trajectory with known durations."""
        env = minimal_20x20_grid_env

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

    def test_occupancy_empty_arrays(self, minimal_2d_grid_env):
        """Test occupancy with empty input arrays."""
        env = minimal_2d_grid_env

        times = np.array([])
        positions = np.empty((0, 2))

        occ = env.occupancy(times, positions)

        assert occ.shape == (env.n_bins,)
        assert_allclose(occ.sum(), 0.0)
        assert np.all(occ == 0.0)

    def test_occupancy_single_sample(self, minimal_2d_grid_env):
        """Test occupancy with single sample (no intervals)."""
        env = minimal_2d_grid_env

        times = np.array([0.0])
        positions = np.array([[5.0, 5.0]])

        occ = env.occupancy(times, positions)

        # No intervals means no occupancy
        assert occ.shape == (env.n_bins,)
        assert_allclose(occ.sum(), 0.0)


class TestOccupancyGapHandling:
    """Test max_gap parameter for handling large time gaps."""

    def test_occupancy_with_large_gaps(self, minimal_20x20_grid_env):
        """Test that large gaps are excluded from occupancy."""
        env = minimal_20x20_grid_env

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

    def test_occupancy_max_gap_none(self, minimal_20x20_grid_env):
        """Test that max_gap=None includes all intervals."""
        env = minimal_20x20_grid_env

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

    def test_occupancy_min_speed(self, minimal_20x20_grid_env):
        """Test that slow periods are excluded when min_speed is set."""
        env = minimal_20x20_grid_env

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

    def test_occupancy_speed_requires_speed_array(self, minimal_2d_grid_env):
        """Test that min_speed without speed array raises error or is ignored."""
        env = minimal_2d_grid_env

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])

        # Providing min_speed without speed should raise ValueError
        with pytest.raises(ValueError, match=r".*speed.*"):
            env.occupancy(times, positions, min_speed=2.0)


class TestOccupancySmoothing:
    """Test kernel smoothing functionality.

    Note: Both tests in this class require inline environment creation because
    they need bin_size=2.0 for proper kernel smoothing behavior:
    - test_occupancy_with_kernel_smoothing: bandwidth=3.0 spans ~1.5 bin widths
    - test_occupancy_smoothing_mass_conservation: bandwidth=2.0 is exactly 1 bin width
    The minimal_20x20_grid_env uses bin_size=5.0, which would produce different
    smoothing characteristics.
    """

    def test_occupancy_with_kernel_smoothing(self):
        """Test that kernel smoothing spreads occupancy to neighbors.

        Requires inline environment: needs bin_size=2.0 for proper smoothing
        behavior with bandwidth=3.0 (spreads to ~1.5 bin widths).
        """
        # Create environment with proper grid (bin_size=2.0 required for smoothing test)
        rng = np.random.default_rng(42)
        grid_samples = rng.uniform(0, 20, size=(200, 2))
        env = Environment.from_samples(grid_samples, bin_size=2.0)

        # Concentrate occupancy in center bin
        times = np.array([0.0, 10.0])
        positions = np.array([[10.0, 10.0], [10.0, 10.0]])

        # Use max_gap=None to count the full interval
        occ_raw = env.occupancy(times, positions, max_gap=None)
        occ_smoothed = env.occupancy(times, positions, bandwidth=3.0, max_gap=None)

        # Smoothing should spread mass to more bins
        assert (occ_smoothed > 0).sum() > (occ_raw > 0).sum()

        # But total mass should be conserved
        assert_allclose(occ_smoothed.sum(), occ_raw.sum(), rtol=1e-4)

    def test_occupancy_smoothing_mass_conservation(self):
        """Test that smoothing conserves total occupancy time.

        Requires inline environment: needs bin_size=2.0 and random samples
        to test mass conservation with bandwidth=2.0 (exactly 1 bin width).
        """
        # Create environment with proper grid (bin_size=2.0 required for smoothing test)
        rng = np.random.default_rng(42)
        grid_samples = rng.uniform(0, 20, size=(200, 2))
        env = Environment.from_samples(grid_samples, bin_size=2.0)

        times = np.linspace(0, 100, 1000)
        positions = rng.uniform(5, 15, size=(1000, 2))

        # Use max_gap=1.0 to allow typical intervals (default 0.5 is too small)
        occ_raw = env.occupancy(times, positions, max_gap=1.0)
        occ_smoothed = env.occupancy(times, positions, bandwidth=2.0, max_gap=1.0)

        # Mass conservation
        assert_allclose(occ_smoothed.sum(), occ_raw.sum(), rtol=1e-3)


class TestOccupancyOutsideBehavior:
    """Test handling of samples outside environment bounds."""

    def test_occupancy_all_outside(self, minimal_2d_grid_env):
        """Test occupancy when all samples are outside environment."""
        env = minimal_2d_grid_env

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
        """Test occupancy with mix of inside and outside samples.

        Builds a local dense 10x10 grid env so the (5, 5) "inside"
        samples actually land in an active bin. minimal_2d_grid_env
        only has two active bins (its corners), which under the M1 1.2
        ``bin_at`` semantics correctly flags (5, 5) as outside the
        active mask.
        """
        x = np.linspace(0.0, 10.0, 11)
        xx, yy = np.meshgrid(x, x)
        env = Environment.from_samples(
            np.column_stack([xx.ravel(), yy.ravel()]), bin_size=2.0
        )

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

    def test_occupancy_mismatched_lengths(self, minimal_2d_grid_env):
        """Test that mismatched times/positions raises error."""
        env = minimal_2d_grid_env

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[5.0, 5.0], [5.0, 5.0]])  # Only 2 positions

        with pytest.raises(ValueError, match=r".*length.*"):
            env.occupancy(times, positions)

    def test_occupancy_wrong_dimensions(self, minimal_2d_grid_env):
        """Test that positions with wrong dimensions raises error."""
        env = minimal_2d_grid_env

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[5.0], [5.0], [5.0]])  # 1D instead of 2D

        with pytest.raises(ValueError, match=r".*dimension.*"):
            env.occupancy(times, positions)


class TestOccupancyMassConservation:
    """Property tests for mass conservation."""

    def test_occupancy_conserves_time(self, minimal_20x20_grid_env):
        """Test that total occupancy equals total valid time."""
        env = minimal_20x20_grid_env
        rng = np.random.default_rng(42)

        times = np.linspace(0, 50, 100)
        positions = rng.uniform(2, 18, size=(100, 2))

        occ = env.occupancy(times, positions, max_gap=1.0)

        # Compute expected total time (excluding large gaps)
        dt = np.diff(times)
        valid_dt = dt[dt <= 1.0]
        expected_time = valid_dt.sum()

        assert_allclose(occ.sum(), expected_time, rtol=1e-6)

    def test_occupancy_nonnegative(self, minimal_20x20_grid_env):
        """Test that occupancy is always non-negative."""
        env = minimal_20x20_grid_env
        rng = np.random.default_rng(42)

        times = np.linspace(0, 10, 50)
        positions = rng.uniform(0, 20, size=(50, 2))

        occ = env.occupancy(times, positions)

        assert np.all(occ >= 0)


class TestOccupancyPerformance:
    """Performance tests.

    Note: test_occupancy_large_trajectory requires inline environment creation
    because it needs a 100x100 grid for performance testing. This is a unique
    large environment not available in shared fixtures, intentionally created
    to validate scaling behavior with 100k samples.
    """

    def test_occupancy_large_trajectory(self):
        """Test occupancy computation on large trajectory (performance check).

        Requires inline environment: 100x100 grid needed for performance testing
        (unique large environment size not available in fixtures).
        """
        # Dense 21x21 sample grid so the active mask covers every bin in
        # the (0..100, 0..100) region. The earlier two-corner construction
        # left the interior outside the active mask and `bin_at` (M1 1.2)
        # correctly excluded every random sample from occupancy.
        x = np.linspace(0.0, 100.0, 21)
        xx, yy = np.meshgrid(x, x)
        data = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(data, bin_size=5.0)

        # 100k samples (not 1M to keep test fast, but validates scaling)
        rng = np.random.default_rng(42)
        n_samples = 100_000
        times = np.linspace(0, 1000, n_samples)
        positions = rng.uniform(10, 90, size=(n_samples, 2))

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

    def test_occupancy_on_regular_grid(self, minimal_20x20_grid_env):
        """Test occupancy on regular grid layout."""
        env = minimal_20x20_grid_env

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

        env = Environment.from_grid_mask(
            active_mask=mask, grid_edges=(grid_edges_x, grid_edges_y)
        )

        times = np.array([0.0, 10.0])
        positions = np.array([[5.0, 5.0], [5.0, 5.0]])

        # Use max_gap=None to count the full interval
        occ = env.occupancy(times, positions, max_gap=None)
        assert occ.shape == (env.n_bins,)


class TestOccupancyReturnSeconds:
    """Test return_seconds parameter for time-weighted vs sample count occupancy."""

    def test_occupancy_return_seconds_true(self, minimal_20x20_grid_env):
        """Test that return_seconds=True returns time in seconds (time-weighted)."""
        env = minimal_20x20_grid_env

        # Create trajectory with varying time intervals at two distinct bins
        # Occupancy is assigned to the STARTING bin of each interval
        # So we need: [bin_a, bin_a, bin_b, bin_b] to get intervals starting at both bins
        bin_centers = env.bin_centers
        bin_a_pos = bin_centers[0]  # First bin
        bin_b_pos = bin_centers[-1] if env.n_bins > 1 else bin_centers[0]  # Last bin

        # 4 samples means 3 intervals:
        # Interval 0: starts at bin A (2 seconds)
        # Interval 1: starts at bin A (3 seconds)
        # Interval 2: starts at bin B (2 seconds)
        times = np.array([0.0, 2.0, 5.0, 7.0])
        positions = np.array([bin_a_pos, bin_a_pos, bin_b_pos, bin_b_pos])

        # With return_seconds=True, weight by time differences
        occ = env.occupancy(times, positions, return_seconds=True, max_gap=None)

        # Total time should be 2.0 + 3.0 + 2.0 = 7.0 seconds
        assert_allclose(occ.sum(), 7.0, rtol=1e-6)

        # Verify that bins have expected occupancy
        bin_a_idx = env.bin_at(np.array([bin_a_pos]))[0]
        bin_b_idx = env.bin_at(np.array([bin_b_pos]))[0]
        # Bin A has 2 intervals: 2 + 3 = 5 seconds
        # Bin B has 1 interval: 2 seconds
        assert_allclose(occ[bin_a_idx], 5.0, rtol=1e-6)
        assert_allclose(occ[bin_b_idx], 2.0, rtol=1e-6)

    def test_occupancy_return_seconds_false(self, minimal_20x20_grid_env):
        """Test that return_seconds=False returns sample counts (unweighted)."""
        env = minimal_20x20_grid_env

        # Create trajectory at two distinct bins (same structure as previous test)
        bin_centers = env.bin_centers
        bin_a_pos = bin_centers[0]
        bin_b_pos = bin_centers[-1] if env.n_bins > 1 else bin_centers[0]

        # Same trajectory structure as previous test
        # 4 samples means 3 intervals:
        # Interval 0: starts at bin A (count = 1)
        # Interval 1: starts at bin A (count = 1)
        # Interval 2: starts at bin B (count = 1)
        times = np.array([0.0, 2.0, 5.0, 7.0])
        positions = np.array([bin_a_pos, bin_a_pos, bin_b_pos, bin_b_pos])

        # With return_seconds=False, just count intervals starting at each bin
        occ = env.occupancy(times, positions, return_seconds=False, max_gap=None)

        # Total count should be 3 (number of intervals = n_samples - 1)
        assert_allclose(occ.sum(), 3.0, rtol=1e-6)

        # Verify counts per bin
        bin_a_idx = env.bin_at(np.array([bin_a_pos]))[0]
        bin_b_idx = env.bin_at(np.array([bin_b_pos]))[0]
        # Bin A has 2 intervals starting there
        # Bin B has 1 interval starting there
        assert_allclose(occ[bin_a_idx], 2.0, rtol=1e-6)
        assert_allclose(occ[bin_b_idx], 1.0, rtol=1e-6)

    def test_occupancy_return_seconds_stationary(self, minimal_20x20_grid_env):
        """Test return_seconds parameter with stationary samples."""
        env = minimal_20x20_grid_env

        # Stationary for 10 seconds with 5 samples
        times = np.array([0.0, 2.0, 4.0, 6.0, 10.0])
        positions = np.array(
            [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]
        )

        # With return_seconds=True: should be 10.0 seconds total
        occ_seconds = env.occupancy(times, positions, return_seconds=True, max_gap=None)
        assert_allclose(occ_seconds.sum(), 10.0, rtol=1e-6)

        # With return_seconds=False: should be 4.0 intervals total (n_samples - 1)
        occ_counts = env.occupancy(times, positions, return_seconds=False, max_gap=None)
        assert_allclose(occ_counts.sum(), 4.0, rtol=1e-6)

        # Ratio should be 10.0 / 4.0 = 2.5 (average time per interval)
        assert_allclose(occ_seconds.sum() / occ_counts.sum(), 2.5, rtol=1e-6)

    def test_occupancy_return_seconds_multiple_bins(self, minimal_20x20_grid_env):
        """Test return_seconds with trajectory visiting multiple bins."""
        env = minimal_20x20_grid_env

        # Visit 3 different bins with different durations
        times = np.array([0.0, 1.0, 4.0, 10.0])
        positions = np.array(
            [
                [5.0, 5.0],  # Bin A: 1 second
                [10.0, 10.0],  # Bin B: 3 seconds
                [15.0, 15.0],  # Bin C: 6 seconds
                [5.0, 5.0],  # Back to bin A (but this is last sample, no interval)
            ]
        )

        # With return_seconds=True: total = 1 + 3 + 6 = 10 seconds
        occ_seconds = env.occupancy(times, positions, return_seconds=True, max_gap=None)
        assert_allclose(occ_seconds.sum(), 10.0, rtol=1e-6)

        # With return_seconds=False: total = 3 intervals
        occ_counts = env.occupancy(times, positions, return_seconds=False, max_gap=None)
        assert_allclose(occ_counts.sum(), 3.0, rtol=1e-6)

    def test_occupancy_return_seconds_with_speed_filter(self, minimal_20x20_grid_env):
        """Test that return_seconds works correctly with speed filtering."""
        env = minimal_20x20_grid_env

        times = np.array([0.0, 1.0, 2.0, 5.0])
        positions = np.array([[5.0, 5.0], [5.0, 5.0], [10.0, 10.0], [10.0, 10.0]])

        # Speed: slow, slow, fast
        speeds = np.array([0.5, 0.5, 10.0, 5.0])

        # Filter out slow periods (speed < 2.0)
        # Only the interval [2.0, 5.0] should be counted (3 seconds at bin B)

        # With return_seconds=True and speed filter
        occ_seconds = env.occupancy(
            times,
            positions,
            speed=speeds,
            min_speed=2.0,
            return_seconds=True,
            max_gap=None,
        )

        # Should have 3 seconds total (only the fast interval)
        assert_allclose(occ_seconds.sum(), 3.0, rtol=1e-6)

        # With return_seconds=False and speed filter
        occ_counts = env.occupancy(
            times,
            positions,
            speed=speeds,
            min_speed=2.0,
            return_seconds=False,
            max_gap=None,
        )

        # Should have 1 interval counted
        assert_allclose(occ_counts.sum(), 1.0, rtol=1e-6)


class TestOccupancyAgreesWithBinSequenceOnOutsideSamples:
    """Regression for M1 1.2: trajectory functions must agree on out-of-env samples.

    Previously ``Environment.occupancy`` used ``map_points_to_bins`` which
    returns the nearest in-env bin index regardless of how far the input
    point is. ``Environment.bin_sequence`` used ``bin_at`` which returns
    -1 for points outside any active bin. The same trajectory passed
    through both methods produced inconsistent bin assignments: a
    tracking error at (1e6, 1e6) silently inflated occupancy at the
    env's edge.

    Both methods should now treat out-of-env samples the same way --
    -1 in the bin-sequence return, and excluded from the occupancy
    integral.
    """

    def _build_env(self):
        """Small open-field 10x10 cm Cartesian env (every grid bin active)."""
        from neurospatial import Environment

        # Dense grid of sample points so every bin in the 5x5 active region
        # carries at least one sample.
        x = np.linspace(0.0, 10.0, 11)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        return Environment.from_samples(positions, bin_size=2.0)

    def test_outside_sample_is_excluded_from_occupancy(self):
        """A sample just outside the active mask used to inflate edge occupancy.

        A tracking error at (13, 5) -- inside the env's bounding box but
        outside the 10x10 active mask -- was silently mapped by
        ``map_points_to_bins`` to the nearest in-env edge bin
        (the implicit "10x typical bin spacing" threshold doesn't catch
        2 cm of overshoot on a 2 cm-bin env). ``bin_sequence`` already
        used ``bin_at`` and correctly reported -1 for this sample.

        Now ``occupancy`` agrees: the interval starting at the
        out-of-env sample is dropped from the integral.
        """
        env = self._build_env()
        # Three timestamps, three samples. The middle one is just
        # outside the 10x10 active mask (overshoot at the right wall).
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array(
            [
                [5.0, 5.0],  # inside
                [13.0, 5.0],  # 3 cm outside the active mask
                [5.0, 5.0],  # inside
            ]
        )
        occ = env.occupancy(times, positions, max_gap=None)

        # bin_sequence agrees: the middle sample is -1.
        bin_seq = env.bin_sequence(times, positions, dedup=False, outside_value=-1)
        assert bin_seq[1] == -1

        # The interval (t=1 to t=2) starts at the out-of-env sample and
        # is therefore dropped. Total occupancy counts only the first
        # interval (t=0 to t=1, duration 1.0 s).
        assert occ.sum() == pytest.approx(1.0, abs=1e-9)

    def test_in_env_sample_consistent_between_occupancy_and_bin_sequence(self):
        """Every in-env sample maps to the same bin index in both methods."""
        env = self._build_env()
        rng = np.random.default_rng(0)
        times = np.linspace(0.0, 5.0, 50)
        # Synthetic trajectory squarely inside the env.
        positions = rng.uniform(2.0, 8.0, size=(50, 2))

        bin_seq = env.bin_sequence(times, positions, dedup=False, outside_value=-1)
        # Every assignment must be a valid in-env bin (never -1).
        assert (bin_seq >= 0).all()

        # Total occupancy time should equal sum of dt[:-1] for valid
        # intervals. With no out-of-env samples and no max_gap filter,
        # that's the full trajectory duration minus the last sample.
        occ = env.occupancy(times, positions, max_gap=None)
        expected_total = float(np.diff(times).sum())
        assert occ.sum() == pytest.approx(expected_total, rel=1e-6)


class TestOccupancyNonFiniteTimes:
    """Regression: non-finite timestamps raise a clear, non-contradictory error."""

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf])
    def test_occupancy_nonfinite_times_raises_clear_error(
        self, holed_grid_env, bad_value
    ):
        env = holed_grid_env
        center = env.bin_centers[0]
        positions = np.array([center, center, center], dtype=np.float64)
        times = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        times[1] = bad_value

        with pytest.raises(ValueError) as excinfo:
            env.occupancy(times, positions)

        message = str(excinfo.value)
        assert "times" in message
        assert "non-finite" in message
        assert "decreasing interval" not in message
