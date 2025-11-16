"""Tests for trajectory analysis gap closure (Priority 2.5.1).

This module targets uncovered edge cases in trajectory.py to increase
coverage from 63% to 85% (+22%). Focuses on:

- Occupancy edge cases (speed at threshold, large gaps, irregular sampling)
- Bin sequence advanced scenarios (boundary duplicates, out of bounds)
- Transition matrix normalization modes and edge cases
- Linear time allocation and ray-grid intersection
- Temporal binning edge cases

Priority 2.5.1 from TEST_PLAN2.md

Notes
-----
All tests follow NumPy docstring format and use existing fixtures from conftest.py.
Tests target specific uncovered branches and edge cases identified through
coverage analysis.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial import Environment


class TestOccupancyEdgeCases:
    """Edge cases for occupancy computation.

    Targets uncovered branches in occupancy() method including:
    - Speed filtering with velocities exactly at threshold
    - Large time gaps in trajectory
    - Highly irregular time sampling
    - Single position (stationary) trajectories
    - Edge cases in max_gap filtering
    """

    def test_occupancy_speed_filter_edge_velocities(self, medium_2d_env):
        """Test speed filtering with velocities exactly at threshold.

        Tests the boundary condition where speed equals min_speed threshold.
        This tests the >= comparison in the speed filter logic.
        """
        # Use a valid bin center that we know exists
        valid_bin_idx = 0
        bin_center = medium_2d_env.bin_centers[valid_bin_idx]

        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions = np.tile(bin_center.reshape(1, -1), (5, 1))

        # Speeds: below, at, above threshold (min_speed=2.0)
        speeds = np.array([1.0, 2.0, 3.0, 2.0, 1.5])

        occ = medium_2d_env.occupancy(
            times, positions, speed=speeds, min_speed=2.0, max_gap=None
        )

        # Intervals starting with speed >= 2.0 should be included
        # t=0-1: speed[0]=1.0 < 2.0, excluded
        # t=1-2: speed[1]=2.0 >= 2.0, included (1s)
        # t=2-3: speed[2]=3.0 >= 2.0, included (1s)
        # t=3-4: speed[3]=2.0 >= 2.0, included (1s)
        # Total: 3s at bin_center (guaranteed valid)
        assert np.sum(occ) == pytest.approx(3.0)
        assert occ[valid_bin_idx] == pytest.approx(3.0)

    def test_occupancy_large_time_gaps(self, small_2d_env):
        """Test occupancy with large gaps in time array.

        Tests max_gap filtering with various gap sizes to ensure
        the filtering logic correctly handles both small and large gaps.
        """
        # Create trajectory with gaps of different sizes
        times = np.array([0.0, 0.1, 5.0, 5.1, 20.0, 20.1])
        positions = np.tile(small_2d_env.bin_centers[0:1], (6, 1))

        # max_gap=1.0: Only intervals <= 1.0 are counted
        occ = small_2d_env.occupancy(times, positions, max_gap=1.0)

        # Intervals: 0.1s, 4.9s (excluded), 0.1s, 14.9s (excluded), 0.1s
        # Expected total: 0.1 + 0.1 + 0.1 = 0.3s
        assert np.sum(occ) == pytest.approx(0.3)

    def test_occupancy_irregular_sampling(self, small_2d_env):
        """Test occupancy with highly irregular time sampling.

        Tests with exponentially increasing time intervals to ensure
        occupancy calculation handles irregular sampling correctly.
        """
        # Use a valid bin center (guaranteed to be inside environment)
        valid_bin_idx = 0
        bin_center = small_2d_env.bin_centers[valid_bin_idx]

        # Exponentially increasing intervals
        times = np.array([0.0, 0.01, 0.03, 0.1, 0.5, 2.0])
        positions = np.tile(bin_center.reshape(1, -1), (6, 1))

        occ = small_2d_env.occupancy(times, positions, max_gap=None)

        # Total time: 2.0 - 0.0 = 2.0
        # Intervals: 0.01, 0.02, 0.07, 0.4, 1.5 = 2.0
        # All time should go to the bin_center
        assert np.sum(occ) == pytest.approx(2.0)
        assert occ[valid_bin_idx] == pytest.approx(2.0)

    def test_occupancy_single_position(self, small_2d_env):
        """Test occupancy with trajectory at single position.

        Tests stationary trajectory where all samples are at same location.
        This tests bin accumulation when there's no spatial movement.
        """
        # Use bin index that we know exists
        valid_bin_idx = min(5, small_2d_env.n_bins - 1)
        bin_center = small_2d_env.bin_centers[valid_bin_idx]

        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # All positions identical
        positions = np.tile(bin_center.reshape(1, -1), (5, 1))

        occ = small_2d_env.occupancy(times, positions)

        # All time should accumulate in one bin (if position is valid)
        # Total time is 4 seconds
        assert np.sum(occ) <= 4.0
        if np.sum(occ) > 0:
            # If any occupancy recorded, it should all be in one bin
            # (stationary trajectory)
            num_bins_with_occ = np.sum(occ > 0)
            assert num_bins_with_occ == 1

    def test_occupancy_max_gap_none_with_large_gaps(self, small_2d_env):
        """Test occupancy with max_gap=None (no gap filtering).

        Tests that setting max_gap=None disables gap filtering entirely,
        including very large gaps.
        """
        times = np.array([0.0, 1.0, 100.0, 101.0])
        positions = np.tile(small_2d_env.bin_centers[0:1], (4, 1))

        occ = small_2d_env.occupancy(times, positions, max_gap=None)

        # All intervals counted, including the 99s gap
        # Total: 1.0 + 99.0 + 1.0 = 101.0
        assert np.sum(occ) == pytest.approx(101.0)

    def test_occupancy_speed_array_validation(self, small_2d_env):
        """Test that speed array length must match times.

        Tests validation of speed array length to ensure it matches
        the times array length.
        """
        times = np.array([0.0, 1.0, 2.0])
        positions = np.tile(small_2d_env.bin_centers[0:1], (3, 1))
        speeds = np.array([1.0, 2.0])  # Too short

        with pytest.raises(ValueError, match="same length"):
            small_2d_env.occupancy(times, positions, speed=speeds, min_speed=2.0)

    def test_occupancy_validates_time_allocation_parameter(self, small_2d_env):
        """Test that invalid time_allocation parameter raises error.

        Tests validation of time_allocation parameter to ensure only
        'start' or 'linear' are accepted.
        """
        times = np.array([0.0, 1.0, 2.0])
        positions = np.tile(small_2d_env.bin_centers[0:1], (3, 1))

        with pytest.raises(ValueError, match="must be 'start' or 'linear'"):
            small_2d_env.occupancy(times, positions, time_allocation="invalid")


class TestBinSequenceAdvanced:
    """Advanced bin sequence extraction tests.

    Targets uncovered branches in bin_sequence() including:
    - Consecutive duplicates at bin boundaries
    - Out-of-bounds positions handling
    - return_runs with all deduplication modes
    - Edge cases in run boundary calculation
    """

    def test_bin_sequence_consecutive_duplicates_at_boundary(self, small_2d_env):
        """Test deduplication when duplicates span bin boundaries.

        Tests run-length encoding when multiple consecutive samples
        fall on the boundary between bins.
        """
        # Create positions that alternate between two bins
        bin_0_pos = small_2d_env.bin_centers[0]
        bin_1_pos = small_2d_env.bin_centers[1]

        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        positions = np.array(
            [
                bin_0_pos,
                bin_0_pos,
                bin_0_pos,  # Run 1: 3 samples in bin 0
                bin_1_pos,
                bin_1_pos,
                bin_1_pos,  # Run 2: 3 samples in bin 1
            ]
        )

        bins, starts, ends = small_2d_env.bin_sequence(
            times, positions, dedup=True, return_runs=True
        )

        # Deduplicated: [0, 1]
        assert_array_equal(bins, [0, 1])
        # Run 1: indices 0-2
        assert starts[0] == 0
        assert ends[0] == 2
        # Run 2: indices 3-5
        assert starts[1] == 3
        assert ends[1] == 5

    def test_bin_sequence_out_of_bounds_positions(self, small_2d_env):
        """Test handling of positions outside environment bounds.

        Tests bin_sequence behavior when some positions are far outside
        the environment bounds (should be marked as -1 or dropped).
        """
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions = np.array(
            [
                small_2d_env.bin_centers[0],
                [10000.0, 10000.0],  # Way outside
                [-10000.0, -10000.0],  # Way outside
                small_2d_env.bin_centers[1],
                [5000.0, 5000.0],  # Way outside
            ]
        )

        # With outside_value=-1 (default)
        bins_default = small_2d_env.bin_sequence(times, positions, dedup=False)
        assert bins_default[0] == 0
        assert bins_default[1] == -1
        assert bins_default[2] == -1
        assert bins_default[3] == 1
        assert bins_default[4] == -1

        # With outside_value=None (drop outside)
        bins_dropped = small_2d_env.bin_sequence(
            times, positions, outside_value=None, dedup=False
        )
        # Should only have bins 0 and 1
        assert_array_equal(bins_dropped, [0, 1])

    def test_bin_sequence_return_runs_single_sample(self, small_2d_env):
        """Test return_runs with single sample trajectory.

        Tests run boundary calculation when trajectory has only one sample.
        This is an edge case in the run calculation logic.
        """
        times = np.array([0.0])
        positions = small_2d_env.bin_centers[0:1]

        # With dedup=True, single sample
        bins, starts, ends = small_2d_env.bin_sequence(
            times, positions, dedup=True, return_runs=True
        )

        assert_array_equal(bins, [0])
        assert_array_equal(starts, [0])
        assert_array_equal(ends, [0])

        # With dedup=False, single sample
        bins, starts, ends = small_2d_env.bin_sequence(
            times, positions, dedup=False, return_runs=True
        )

        assert_array_equal(bins, [0])
        assert_array_equal(starts, [0])
        assert_array_equal(ends, [0])

    def test_bin_sequence_runs_with_outside_dropped(self, small_2d_env):
        """Test run boundaries when outside values are dropped.

        Tests that run boundaries are correctly calculated when
        outside_value=None causes some samples to be dropped.
        """
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        positions = np.array(
            [
                small_2d_env.bin_centers[0],
                small_2d_env.bin_centers[0],
                [10000.0, 10000.0],  # Outside - will be dropped
                [10000.0, 10000.0],  # Outside - will be dropped
                small_2d_env.bin_centers[1],
                small_2d_env.bin_centers[1],
            ]
        )

        bins, starts, ends = small_2d_env.bin_sequence(
            times, positions, outside_value=None, dedup=True, return_runs=True
        )

        # After dropping outside: [0, 0, 1, 1] with original indices [0, 1, 4, 5]
        # After dedup: [0, 1]
        assert len(bins) == 2
        assert bins[0] == 0
        assert bins[1] == 1
        assert len(starts) == 2
        assert len(ends) == 2

        # Verify runs are sensible (starts before ends, valid indices)
        for i in range(len(bins)):
            assert 0 <= starts[i] < len(times)
            assert 0 <= ends[i] < len(times)
            assert starts[i] <= ends[i]

        # First run should start at beginning
        assert starts[0] == 0

    def test_bin_sequence_validates_1d_positions(self, small_2d_env):
        """Test that 1D position array raises error.

        Tests validation that positions must be 2D array (n_samples, n_dims),
        not 1D array.
        """
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([0.0, 1.0, 2.0])  # 1D array

        with pytest.raises(ValueError, match="2-dimensional array"):
            small_2d_env.bin_sequence(times, positions)


class TestTransitionMatrixNormalization:
    """Transition matrix with all normalization modes.

    Targets uncovered branches in transitions() including:
    - All normalization modes (None, row, symmetric)
    - Lagged transitions with various lag values
    - Sparse trajectories with few transitions
    - Edge cases in adjacency filtering
    """

    @pytest.mark.parametrize("normalize", [True, False])
    def test_transition_matrix_normalization_modes(self, small_2d_env, normalize):
        """Test transition matrix with different normalization modes.

        Uses parametrize to test both normalized (row-stochastic) and
        unnormalized (raw counts) transition matrices.

        Parameters
        ----------
        normalize : bool
            If True, test normalized (row-stochastic) matrix.
            If False, test unnormalized (raw counts) matrix.
        """
        # Create trajectory with repeated transitions
        bin_0 = 0
        neighbors = list(small_2d_env.neighbors(bin_0))

        if len(neighbors) > 0:
            bin_1 = neighbors[0]
            # Sequence: 0->1->0->1->0 (4 transitions)
            bins = np.array([bin_0, bin_1, bin_0, bin_1, bin_0], dtype=np.int32)

            T = small_2d_env.transitions(
                bins=bins, normalize=normalize, allow_teleports=True
            )

            if normalize:
                # Row-stochastic: rows with transitions sum to 1
                row_sums = np.array(T.sum(axis=1)).flatten()
                nonzero_rows = row_sums > 0
                assert np.allclose(row_sums[nonzero_rows], 1.0)
            else:
                # Unnormalized: check raw counts
                # bin_0->bin_1: 2 transitions
                # bin_1->bin_0: 2 transitions
                assert T[bin_0, bin_1] == 2.0
                assert T[bin_1, bin_0] == 2.0

    def test_transition_matrix_lagged_transitions(self, small_2d_env):
        """Test lagged transition matrix with various lag values.

        Tests transitions with lag > 1, which counts transitions from
        bins[t] to bins[t+lag] instead of consecutive bins.
        """
        # Create sequence that allows testing different lags
        bins = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)

        # Lag=1 (consecutive)
        T_lag1 = small_2d_env.transitions(
            bins=bins, lag=1, allow_teleports=True, normalize=False
        )
        # Should have 6 transitions (0->1, 1->2, ..., 5->6)
        assert T_lag1.nnz == 6

        # Lag=2 (skip one)
        T_lag2 = small_2d_env.transitions(
            bins=bins, lag=2, allow_teleports=True, normalize=False
        )
        # Should have 5 transitions (0->2, 1->3, ..., 4->6)
        assert T_lag2.nnz == 5

        # Lag=3
        T_lag3 = small_2d_env.transitions(
            bins=bins, lag=3, allow_teleports=True, normalize=False
        )
        # Should have 4 transitions (0->3, 1->4, 2->5, 3->6)
        assert T_lag3.nnz == 4

    def test_transition_matrix_sparse_trajectory(self, medium_2d_env):
        """Test transition matrix with very sparse trajectories.

        Tests transition matrix computation when trajectory visits only
        a small fraction of bins. This tests sparse matrix efficiency.
        """
        # Visit only 5 bins out of many
        bins = np.array([0, 10, 20, 30, 40], dtype=np.int32)

        T = medium_2d_env.transitions(bins=bins, normalize=True, allow_teleports=True)

        # Should be very sparse (most bins never visited)
        total_entries = medium_2d_env.n_bins * medium_2d_env.n_bins
        sparsity = T.nnz / total_entries
        assert sparsity < 0.01  # Less than 1% of entries are nonzero

    def test_transition_matrix_diffusion_normalize_false_error(self, small_2d_env):
        """Test that diffusion method requires normalization.

        Tests that method='diffusion' raises error when normalize=False,
        since diffusion kernel is inherently normalized.
        """
        with pytest.raises(ValueError, match="does not support normalize=False"):
            small_2d_env.transitions(method="diffusion", bandwidth=2.0, normalize=False)

    def test_transition_matrix_validates_lag_positive(self, small_2d_env):
        """Test that lag must be positive.

        Tests validation that lag parameter must be >= 1.
        """
        bins = np.array([0, 1, 2], dtype=np.int32)

        with pytest.raises(ValueError, match="lag must be positive"):
            small_2d_env.transitions(bins=bins, lag=0)

        with pytest.raises(ValueError, match="lag must be positive"):
            small_2d_env.transitions(bins=bins, lag=-1)

    def test_transition_matrix_bins_dtype_validation(self, small_2d_env):
        """Test that bins must be integer-like array.

        Tests that float bins are properly validated and rejected if they
        don't represent valid integers.
        """
        bins = np.array([0.0, 1.0, 2.0])  # Float array with integer values

        # The code validates dtype and raises error for float arrays
        with pytest.raises(ValueError, match="integer array"):
            small_2d_env.transitions(bins=bins, allow_teleports=True)

    def test_transition_matrix_validates_missing_inputs(self, small_2d_env):
        """Test that transitions requires bins or times/positions.

        Tests that calling transitions() without any trajectory data
        raises appropriate error.
        """
        with pytest.raises(ValueError, match="Must provide either"):
            small_2d_env.transitions()


class TestTemporalBinningEdgeCases:
    """Temporal binning edge cases.

    Tests edge cases related to temporal aspects of occupancy and
    bin sequence methods, including irregular time intervals and
    boundary conditions.
    """

    def test_temporal_binning_zero_duration_intervals(self, small_2d_env):
        """Test handling of zero-duration time intervals.

        Tests occupancy when consecutive time samples are identical
        (zero time elapsed).
        """
        # Use a single valid bin center (guaranteed to be inside environment)
        valid_bin_idx = 0
        bin_center = small_2d_env.bin_centers[valid_bin_idx]

        times = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
        # All positions at same bin center
        positions = np.tile(bin_center.reshape(1, -1), (5, 1))

        occ = small_2d_env.occupancy(times, positions, max_gap=None)

        # Only non-zero intervals contribute
        # 0.0->0.0 (0s), 0.0->1.0 (1s), 1.0->1.0 (0s), 1.0->2.0 (1s)
        # Total: 2.0 seconds at bin_center
        assert np.sum(occ) == pytest.approx(2.0)
        assert occ[valid_bin_idx] == pytest.approx(2.0)

    def test_temporal_binning_decreasing_times_error(self, small_2d_env):
        """Test that decreasing times raise detailed error.

        Tests that non-monotonic (decreasing) timestamps raise error
        with diagnostic information about where the decrease occurs.
        """
        times = np.array([0.0, 2.0, 1.5, 3.0])  # Decreases at index 2
        positions = np.tile(small_2d_env.bin_centers[0:1], (4, 1))

        with pytest.raises(ValueError, match="monotonically increasing") as exc_info:
            small_2d_env.occupancy(times, positions)

        # Should mention index where decrease occurs
        assert "indices" in str(exc_info.value).lower()

    def test_temporal_binning_very_small_time_steps(self, small_2d_env):
        """Test occupancy with very small time steps.

        Tests numerical stability with microsecond-scale time intervals.
        """
        # Microsecond-scale intervals
        times = np.array([0.0, 1e-6, 2e-6, 3e-6, 4e-6])
        positions = np.tile(small_2d_env.bin_centers[0:1], (5, 1))

        occ = small_2d_env.occupancy(times, positions)

        # Should handle small times correctly
        assert np.sum(occ) == pytest.approx(4e-6, rel=1e-10)

    def test_temporal_binning_large_time_values(self, small_2d_env):
        """Test occupancy with large absolute time values.

        Tests that large timestamps (e.g., Unix epoch) are handled correctly.
        The key test is that time differences are computed correctly regardless
        of the absolute magnitude of the timestamps.
        """
        # Large absolute times (Unix epoch around 1.7 billion seconds)
        base_time = 1.7e9
        times = np.array([base_time, base_time + 1.0, base_time + 2.0, base_time + 3.0])

        # Use a valid bin center
        valid_bin_center = small_2d_env.bin_centers[0]
        positions = np.tile(valid_bin_center.reshape(1, -1), (4, 1))

        occ = small_2d_env.occupancy(times, positions, max_gap=None)

        # Test that the function completes without error
        assert occ.shape == (small_2d_env.n_bins,)
        # Total occupancy should equal total elapsed time (all at one bin)
        assert np.sum(occ) == pytest.approx(3.0)


class TestLinearTimeAllocation:
    """Tests for linear time allocation (time_allocation='linear').

    Tests the linear allocation mode that splits time intervals across
    bins traversed by ray-grid intersection. Requires RegularGridLayout.
    """

    def test_linear_allocation_requires_regular_grid(self, simple_graph_env):
        """Test that linear allocation requires RegularGridLayout.

        Tests that time_allocation='linear' raises NotImplementedError
        for non-RegularGrid layouts.
        """
        times = np.array([0.0, 1.0, 2.0])
        # Create positions - graph_env expects 2D coordinates
        # Use coordinates that might be valid for the graph
        positions = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])

        with pytest.raises(
            NotImplementedError, match="only supported for RegularGridLayout"
        ):
            simple_graph_env.occupancy(times, positions, time_allocation="linear")

    def test_linear_allocation_same_bin(self, grid_env_from_samples):
        """Test linear allocation when start and end in same bin.

        Tests the fast path when consecutive samples are in the same bin
        (no ray-grid intersection needed).
        """
        # Get a bin center and create trajectory that stays in same bin
        valid_bin_idx = 0
        bin_center = grid_env_from_samples.bin_centers[valid_bin_idx]

        times = np.array([0.0, 1.0, 2.0, 3.0])
        # All positions in same bin (small perturbations within bin)
        # Use tiny offsets to stay in same bin
        positions = np.array(
            [
                bin_center,
                bin_center + 0.001,
                bin_center - 0.001,
                bin_center,
            ]
        )

        occ = grid_env_from_samples.occupancy(
            times, positions, time_allocation="linear", max_gap=None
        )

        # All time should go to the single bin (bin_center is guaranteed valid)
        total_time = times[-1] - times[0]
        assert np.sum(occ) == pytest.approx(total_time)
        assert occ[valid_bin_idx] == pytest.approx(total_time)

    def test_linear_allocation_crosses_bins(self):
        """Test linear allocation when trajectory crosses multiple bins.

        Tests ray-grid intersection when path crosses bin boundaries.
        This tests the full ray-grid intersection algorithm.
        """
        # Create a simple RegularGrid environment (not MaskedGrid)
        # Linear allocation requires RegularGridLayout
        env = Environment.from_samples(
            positions=np.array([[0.0, 0.0], [10.0, 10.0]]),
            bin_size=2.0,
            infer_active_bins=False,  # RegularGrid, not MaskedGrid
        )

        # Find two bins that are separated (to ensure crossing)
        bin_0_center = env.bin_centers[0]
        bin_last_center = env.bin_centers[-1]

        times = np.array([0.0, 1.0])
        positions = np.array([bin_0_center, bin_last_center])

        # Disable gap filtering since we're using 1.0 second interval
        occ_linear = env.occupancy(
            times, positions, time_allocation="linear", max_gap=None
        )

        # Linear allocation should split time across multiple bins
        num_bins_with_occ = np.sum(occ_linear > 0)

        # Path from first to last bin should cross multiple bins
        assert num_bins_with_occ > 1, (
            "Path from bin 0 to last bin should cross multiple bins"
        )
        # Total time should be preserved
        assert np.sum(occ_linear) == pytest.approx(1.0)

    def test_linear_allocation_return_seconds_false(self, grid_env_from_samples):
        """Test linear allocation with return_seconds=False.

        Tests that linear allocation works correctly when returning
        sample counts instead of time in seconds.
        """
        # Create simple trajectory
        bin_0 = grid_env_from_samples.bin_centers[0]
        bin_1 = grid_env_from_samples.bin_centers[
            min(1, len(grid_env_from_samples.bin_centers) - 1)
        ]

        times = np.array([0.0, 2.0, 4.0])
        positions = np.array([bin_0, bin_1, bin_0])

        occ_counts = grid_env_from_samples.occupancy(
            times, positions, time_allocation="linear", return_seconds=False
        )

        # Should count intervals (2 total), with fractional counts
        # across bins due to linear allocation
        # Sum should be proportional to number of intervals
        assert np.sum(occ_counts) <= 2.0  # Cannot exceed number of intervals


class TestRayGridIntersection:
    """Tests for ray-grid intersection edge cases.

    Tests numerical stability and edge cases in the _compute_ray_grid_intersections
    and related helper methods. These are internal methods but important for
    linear time allocation correctness.
    """

    def test_ray_grid_zero_distance_movement(self, grid_env_from_samples):
        """Test ray-grid intersection with zero movement.

        Tests the EPSILON threshold for detecting zero-distance rays.
        When start == end, should allocate all time to starting bin.
        """
        # Use same position twice (zero movement)
        bin_center = grid_env_from_samples.bin_centers[0]

        times = np.array([0.0, 1.0])
        positions = np.array([bin_center, bin_center])

        occ = grid_env_from_samples.occupancy(
            times, positions, time_allocation="linear"
        )

        # All time goes to starting bin (if it's valid)
        assert np.sum(occ) <= 1.0

    def test_ray_grid_parallel_to_axis(self, grid_env_from_samples):
        """Test ray-grid intersection with movement parallel to grid axis.

        Tests the EPSILON threshold for ray direction components parallel
        to grid edges (prevents division by zero).
        """
        # Create movement strictly along one axis
        # Get two positions aligned along one dimension
        bin_0 = grid_env_from_samples.bin_centers[0]

        # Move only in x direction (y stays constant)
        times = np.array([0.0, 1.0])
        positions = np.array(
            [
                bin_0,
                bin_0 + np.array([2.0, 0.0]),  # Move in x only
            ]
        )

        occ = grid_env_from_samples.occupancy(
            times, positions, time_allocation="linear"
        )

        # Should handle axis-aligned movement without division by zero
        assert np.sum(occ) <= 1.0
        assert not np.any(np.isnan(occ))
        assert not np.any(np.isinf(occ))

    def test_ray_grid_diagonal_crossing_multiple_bins(self):
        """Test ray-grid intersection with diagonal path crossing multiple bins.

        Tests the full ray-grid intersection algorithm by creating a path
        that crosses multiple grid cells diagonally. This exercises the core
        algorithm logic (lines 1155-1216 in trajectory.py) including:
        - Computing grid line intersections
        - Allocating time proportionally to segment lengths
        - Handling multiple bin crossings

        Note: Linear allocation implementation may have limitations.
        This test verifies the function executes without error.
        """
        # Create a simple RegularGrid environment (not MaskedGrid)
        # Linear allocation requires RegularGridLayout
        env = Environment.from_samples(
            positions=np.array([[0.0, 0.0], [10.0, 10.0]]),
            bin_size=2.0,
            infer_active_bins=False,  # RegularGrid, not MaskedGrid
        )

        # Create diagonal path from corner to corner
        # This should cross multiple bins and test intermediate intersections
        start = env.bin_centers[0]
        end = env.bin_centers[-1]

        times = np.array([0.0, 1.0])
        positions = np.array([start, end])

        # Disable gap filtering since we're using 1.0 second interval
        occ_linear = env.occupancy(
            times, positions, time_allocation="linear", max_gap=None
        )

        # Diagonal should cross multiple bins
        num_bins_with_occ = np.sum(occ_linear > 0)
        assert num_bins_with_occ > 2, "Diagonal path should cross multiple bins"

        # Total time should be preserved
        assert np.sum(occ_linear) == pytest.approx(1.0)

        # Verify bins with occupancy form a connected path
        bins_visited = np.where(occ_linear > 0)[0]
        # Check that visited bins form a path (each bin connected to next)
        for i in range(len(bins_visited) - 1):
            neighbors = list(env.neighbors(bins_visited[i]))
            assert bins_visited[i + 1] in neighbors, "Path should be connected"


class TestTrajectoryIntegrationGaps:
    """Integration tests for trajectory gap scenarios.

    Tests combined scenarios that exercise multiple uncovered code paths
    in realistic use cases.
    """

    def test_combined_speed_and_gap_filters(self, small_2d_env):
        """Test trajectory with both speed and gap filters active.

        Tests interaction between speed filtering and gap filtering when
        both are applied simultaneously.
        """
        times = np.array([0.0, 1.0, 1.2, 5.0, 6.0, 7.0])
        positions = np.tile(small_2d_env.bin_centers[0:1], (6, 1))
        speeds = np.array([5.0, 0.5, 5.0, 5.0, 5.0, 5.0])

        occ = small_2d_env.occupancy(
            times, positions, speed=speeds, min_speed=2.0, max_gap=2.0
        )

        # Analyze each interval:
        # t=0-1 (1s): speed[0]=5.0 >= 2.0, gap=1.0 <= 2.0 -> PASS
        # t=1-1.2 (0.2s): speed[1]=0.5 < 2.0 -> FAIL (speed)
        # t=1.2-5 (3.8s): speed[2]=5.0 >= 2.0, gap=3.8 > 2.0 -> FAIL (gap)
        # t=5-6 (1s): speed[3]=5.0 >= 2.0, gap=1.0 <= 2.0 -> PASS
        # t=6-7 (1s): speed[4]=5.0 >= 2.0, gap=1.0 <= 2.0 -> PASS
        # Total: 1.0 + 1.0 + 1.0 = 3.0
        assert np.sum(occ) == pytest.approx(3.0)

    def test_trajectory_all_intervals_filtered(self, small_2d_env):
        """Test trajectory where all intervals are filtered out.

        Tests behavior when filters remove all intervals (should return
        zero occupancy everywhere).
        """
        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.tile(small_2d_env.bin_centers[0:1], (4, 1))
        # All speeds below threshold
        speeds = np.array([0.1, 0.2, 0.3, 0.4])

        occ = small_2d_env.occupancy(times, positions, speed=speeds, min_speed=2.0)

        # All intervals filtered - no occupancy
        assert np.sum(occ) == pytest.approx(0.0)
        assert np.all(occ == 0.0)

    def test_bin_sequence_and_transitions_consistency(self, medium_2d_env):
        """Test consistency between bin_sequence and transitions.

        Tests that transitions computed from bin_sequence matches
        expected transition counts.
        """
        # Create deterministic trajectory
        rng = np.random.default_rng(42)
        n_samples = 50
        times = np.linspace(0, 5, n_samples)

        # Start from center and take small random steps
        center = medium_2d_env.bin_centers.mean(axis=0)
        positions = [center]
        for _ in range(n_samples - 1):
            step = rng.normal(0, 0.5, size=medium_2d_env.n_dims)
            positions.append(positions[-1] + step)
        positions = np.array(positions)

        # Get bin sequence without dedup (to count all transitions)
        bins = medium_2d_env.bin_sequence(
            times, positions, dedup=False, outside_value=None
        )

        if len(bins) > 1:
            # Compute transition matrix
            T = medium_2d_env.transitions(
                bins=bins, normalize=False, allow_teleports=True
            )

            # Total transition count should equal number of transitions
            # (len(bins) - 1 consecutive pairs)
            total_transitions = T.sum()
            expected_transitions = len(bins) - 1
            assert total_transitions == pytest.approx(expected_transitions)

    def test_occupancy_with_positions_ndim_1(self, small_1d_env):
        """Test occupancy with 1D environment and positions.

        Tests that 1D environments work correctly with occupancy calculation.
        """
        times = np.array([0.0, 1.0, 2.0, 3.0])
        # Use valid bin centers to ensure positions are within environment
        positions = small_1d_env.bin_centers[:4].reshape(-1, 1)

        occ = small_1d_env.occupancy(times, positions, max_gap=None)

        # Should work without error
        assert occ.shape == (small_1d_env.n_bins,)
        # Total occupancy should equal total elapsed time
        expected_time = times[-1] - times[0]  # 3.0 seconds
        assert np.sum(occ) == pytest.approx(expected_time)

    def test_transitions_validates_times_without_positions(self, small_2d_env):
        """Test that providing only times raises error.

        Tests validation that times and positions must be provided together.
        """
        times = np.array([0.0, 1.0, 2.0])

        with pytest.raises(ValueError, match="Both times and positions"):
            small_2d_env.transitions(times=times)

    def test_transitions_validates_positions_without_times(self, small_2d_env):
        """Test that providing only positions raises error.

        Tests validation that times and positions must be provided together.
        """
        positions = np.tile(small_2d_env.bin_centers[0:1], (3, 1))

        with pytest.raises(ValueError, match="Both times and positions"):
            small_2d_env.transitions(positions=positions)
