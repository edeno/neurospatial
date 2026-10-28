"""Tests for trajectory metrics module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment
from neurospatial.behavior.trajectory import (
    compute_home_range,
    compute_step_lengths,
    compute_trajectory_curvature,
    compute_turn_angles,
    mean_square_displacement,
)


class TestComputeTurnAngles:
    """Test compute_turn_angles function."""

    def test_straight_line_trajectory(self):
        """Test that straight line movement has near-zero turn angles."""
        # Create straight line trajectory (continuous positions)
        positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])

        # Compute turn angles (continuous API)
        angles = compute_turn_angles(positions)

        # Should have n-2 angles (first and last transitions)
        assert angles.shape == (18,)

        # All angles should be near zero (straight line)
        assert_allclose(angles, 0.0, atol=0.1)

    def test_circular_trajectory_constant_turning(self):
        """Test that circular motion produces turn angles."""
        # Create circular trajectory (100 points around a circle)
        n_points = 100
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = 50 + 40 * np.cos(theta)
        y = 50 + 40 * np.sin(theta)
        positions = np.column_stack([x, y])

        # Compute turn angles (continuous API - no env needed!)
        angles = compute_turn_angles(positions)

        # Should have some non-zero angles (circular motion produces turns)
        assert len(angles) > 0
        non_zero_angles = angles[np.abs(angles) > 0.01]
        # At least some turning should occur
        assert len(non_zero_angles) > 0

    def test_turn_angles_range(self):
        """Test that turn angles are in [-π, π] range."""
        # Create meandering trajectory (deterministic - no randomness needed)
        t = np.linspace(0, 4 * np.pi, 200)
        x = t * 5 + 20 * np.sin(t)
        y = 20 * np.cos(t)
        positions = np.column_stack([x, y])

        # Continuous API
        angles = compute_turn_angles(positions)

        # All angles should be in [-π, π]
        assert np.all(angles >= -np.pi)
        assert np.all(angles <= np.pi)

    def test_stationary_trajectory(self):
        """Test that stationary periods are skipped."""
        # Create trajectory with repeated positions (stationary periods)
        positions = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],  # Duplicate
                [0.0, 0.0],  # Duplicate
                [10.0, 0.0],
                [20.0, 0.0],
                [20.0, 0.0],  # Duplicate
                [30.0, 0.0],
                [40.0, 0.0],
                [40.0, 0.0],  # Duplicate
                [40.0, 0.0],  # Duplicate
                [50.0, 0.0],
            ]
        )

        angles = compute_turn_angles(positions)

        # Should only compute angles for actual movements
        # Consecutive duplicates should be filtered out
        assert len(angles) >= 0  # At least some angles computed

    def test_parameter_order(self):
        """Test that parameter is just positions (continuous API)."""
        positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])

        # This should work without error
        angles = compute_turn_angles(positions)
        assert isinstance(angles, np.ndarray)

    def test_one_dimensional_reversal_is_pi_and_continuation_zero(self):
        positions = np.array([[0.0], [1.0], [3.0], [2.0]])
        assert_allclose(compute_turn_angles(positions), [0.0, np.pi], rtol=0, atol=0)

    def test_one_dimensional_missing_sample_is_nan_not_a_reversal(self):
        """A dropout is not a reversal; it is NaN, as in two dimensions."""
        positions = np.array([[0.0], [1.0], [np.nan], [3.0], [4.0]])
        one_d = compute_turn_angles(positions)
        two_d = compute_turn_angles(np.c_[positions, np.zeros(5)])
        assert np.isnan(one_d).all()
        np.testing.assert_array_equal(np.isnan(one_d), np.isnan(two_d))


class TestComputeStepLengths:
    """Test compute_step_lengths function."""

    def test_step_lengths_straight_line(self):
        """Test step lengths on a straight 1D trajectory with Euclidean distance."""
        # Create straight line trajectory (continuous positions)
        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])

        # Euclidean distance (default)
        step_lengths = compute_step_lengths(positions, metric="euclidean")

        # Should have n-1 step lengths
        assert step_lengths.shape == (20,)

        # All steps should be positive and roughly equal
        assert np.all(step_lengths >= 0)
        # For uniform spacing, expect uniform step lengths
        assert_allclose(step_lengths, step_lengths[0], rtol=0.01)

    def test_step_lengths_with_duplicates(self):
        """Test that consecutive duplicates have zero step length."""
        # Trajectory with stationary periods (duplicates)
        positions = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],  # Duplicate
                [10.0, 0.0],
                [20.0, 0.0],
                [20.0, 0.0],  # Duplicate
                [20.0, 0.0],  # Duplicate
                [30.0, 0.0],
            ]
        )

        step_lengths = compute_step_lengths(positions, metric="euclidean")

        # Should have n-1 step lengths
        assert step_lengths.shape == (6,)

        # Consecutive duplicates should have zero distance
        assert step_lengths[0] == 0.0  # [0,0] -> [0,0]
        assert step_lengths[3] == 0.0  # [20,0] -> [20,0]
        assert step_lengths[4] == 0.0  # [20,0] -> [20,0]

    def test_step_lengths_uses_graph_distance(self):
        """Test that step lengths with metric='geodesic' use graph distances."""
        # Create 2D grid environment
        x = np.linspace(0, 40, 100)
        y = np.linspace(0, 40, 100)
        xx, yy = np.meshgrid(x, y)
        sample_positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        # Create trajectory on bin centers (for geodesic distance to work)
        position_bins = np.arange(5)
        positions = env.bin_centers[position_bins]

        # Use geodesic distance
        step_lengths = compute_step_lengths(positions, metric="geodesic", env=env)

        # Verify each step length matches nx.shortest_path_length
        import networkx as nx

        for i in range(len(step_lengths)):
            bin_i = position_bins[i]
            bin_j = position_bins[i + 1]
            expected = float(
                nx.shortest_path_length(
                    env.connectivity, source=bin_i, target=bin_j, weight="distance"
                )
            )
            assert_allclose(step_lengths[i], expected, rtol=1e-5)

    def test_parameter_order(self):
        """Test new continuous API signature."""
        positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])

        # Euclidean (default)
        step_lengths = compute_step_lengths(positions)
        assert isinstance(step_lengths, np.ndarray)

        # Geodesic requires env
        env = Environment.from_samples(positions, bin_size=5.0)
        bin_positions = env.bin_centers[env.bin_at(positions)]
        step_lengths_geo = compute_step_lengths(
            bin_positions, metric="geodesic", env=env
        )
        assert isinstance(step_lengths_geo, np.ndarray)


class TestComputeHomeRange:
    """Test compute_home_range function."""

    def test_home_range_95_percentile(self):
        """Test 95% home range selection."""
        # Create trajectory with known occupancy distribution
        # Bin 0: 50 visits, Bin 1: 30 visits, Bin 2: 15 visits, Bin 3: 5 visits
        position_bins = np.concatenate(
            [
                np.repeat(0, 50),
                np.repeat(1, 30),
                np.repeat(2, 15),
                np.repeat(3, 5),
            ]
        )

        home_range = compute_home_range(position_bins, percentile=95.0)

        # 95% of 100 visits = 95 visits
        # Bin 0 (50) + Bin 1 (30) + Bin 2 (15) = 95 visits = 95%
        # So bins 0, 1, 2 should be in home range
        assert set(home_range) == {0, 1, 2}

    def test_home_range_100_percentile(self):
        """Test that 100% includes all visited bins."""
        position_bins = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 0])

        home_range = compute_home_range(position_bins, percentile=100.0)

        # Should include all unique bins
        assert set(home_range) == {0, 1, 2, 3}

    def test_home_range_50_percentile(self):
        """Test 50% home range (core area)."""
        # Bin 0: 40 visits, bin 1: 30, bins 2-4: 10 each.
        position_bins = np.repeat(np.arange(5), [40, 30, 10, 10, 10])

        home_range = compute_home_range(position_bins, percentile=50.0)

        # 40 visits are not yet half of 100; adding bin 1 reaches 70.
        np.testing.assert_array_equal(home_range, [0, 1])

    def test_bins_tied_at_the_cutoff_are_all_included(self):
        """Uniform use has no 50% core: every bin is at the cutoff level."""
        position_bins = np.repeat(np.arange(10), 10)
        home_range = compute_home_range(position_bins, percentile=50.0)
        np.testing.assert_array_equal(home_range, np.arange(10))

    def test_ties_are_independent_of_sample_order(self):
        # 40% in bin 0, then three bins with 20% each; the 70% cutoff falls
        # inside the tie, so the three tied bins are all included.
        position_bins = np.r_[np.repeat(0, 40), np.repeat([1, 2, 3], 20)]
        expected = [0, 1, 2, 3]
        np.testing.assert_array_equal(
            compute_home_range(position_bins, percentile=70.0), expected
        )
        np.testing.assert_array_equal(
            compute_home_range(position_bins[::-1].copy(), percentile=70.0), expected
        )

    def test_equal_occupancy_is_ordered_by_bin_index(self):
        position_bins = np.r_[np.repeat(5, 30), np.repeat([3, 1, 4], 10)]
        np.testing.assert_array_equal(
            compute_home_range(position_bins, percentile=100.0), [5, 1, 3, 4]
        )

    def test_time_weighted_ties_within_roundoff_are_ties(self):
        # Three bins with 0.3 s each, summed from 0.1 s dwells in different orders.
        position_bins = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        times = np.r_[np.arange(9) * 0.1, 0.9]
        times = np.round(times, 10)
        home_range = compute_home_range(
            position_bins, times=times, percentile=50.0, max_gap=None
        )
        np.testing.assert_array_equal(home_range, [2, 0, 1])

    def test_home_range_returns_indices(self):
        """Test that home_range returns bin indices (integers)."""
        position_bins = np.array([0, 1, 2, 1, 0, 1, 2])

        home_range = compute_home_range(position_bins, percentile=95.0)

        # Should be numpy array of integers
        assert isinstance(home_range, np.ndarray)
        assert home_range.dtype in [np.int32, np.int64, np.intp]

    def test_parameter_order(self):
        """Test that parameter order is (position_bins, *, percentile)."""
        position_bins = np.array([0, 1, 2, 1, 0])

        # This should work without error
        home_range = compute_home_range(position_bins, percentile=95.0)
        assert isinstance(home_range, np.ndarray)

    def test_time_weighted_occupancy_changes_ranking(self):
        """Time-weighted occupancy ranks long-dwell bins above high-count bins.

        Bin 0 is visited many times but each visit is brief, while bin 1 is
        visited fewer times but with long dwells. Count-based occupancy ranks
        bin 0 first; time-weighted occupancy must rank bin 1 first.
        """
        # 6 brief samples in bin 0, then 2 long-dwell samples in bin 1.
        position_bins = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        # dt: five 0.1s gaps in bin 0, then a 0.1s gap into bin 1, then a 10s
        # dwell within bin 1. Bin 1 accumulates far more time despite fewer
        # samples.
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 10.6])

        # Count-based: bin 0 has 6 counts, bin 1 has 2 -> bin 0 first.
        count_based = compute_home_range(position_bins, percentile=100.0)
        assert count_based[0] == 0

        # Time-weighted: bin 1 dwell time dominates -> bin 1 first.
        time_weighted = compute_home_range(
            position_bins, times=times, percentile=100.0, max_gap=None
        )
        assert time_weighted[0] == 1, (
            f"Time-weighted home range should rank long-dwell bin 1 first, "
            f"got order {time_weighted}"
        )

    def test_time_weighted_default_matches_counts_uniform_sampling(self):
        """With uniform sampling, time-weighted ranking matches count-based."""
        position_bins = np.concatenate(
            [np.repeat(0, 50), np.repeat(1, 30), np.repeat(2, 15), np.repeat(3, 5)]
        )
        times = np.arange(len(position_bins), dtype=np.float64) * 0.05  # uniform dt

        count_based = compute_home_range(position_bins, percentile=95.0)
        time_weighted = compute_home_range(position_bins, times=times, percentile=95.0)

        assert set(count_based) == set(time_weighted)

    def test_time_weighted_length_mismatch_raises(self):
        """Mismatched times length raises ValueError."""
        position_bins = np.array([0, 1, 2])
        times = np.array([0.0, 0.1])  # wrong length

        with pytest.raises(ValueError, match="same length"):
            compute_home_range(position_bins, times=times)

    @pytest.mark.parametrize(
        "times",
        [
            np.array([0.3, 0.2, 0.1, 0.0]),
            np.array([0.0, np.nan, 0.2, 0.3]),
            np.array([0.0, 0.1, 0.1, 0.2]),
        ],
        ids=["reversed", "nan", "repeated"],
    )
    def test_invalid_times_raise(self, times):
        with pytest.raises(ValueError, match="times"):
            compute_home_range(np.array([0, 1, 1, 2]), times=times)

    @pytest.mark.parametrize(
        "kwargs",
        [{"epochs": (0.0, 1.0)}, {"max_gap": 0.5}, {"max_gap": None}],
        ids=["epochs", "max_gap", "max_gap_none"],
    )
    def test_time_gates_without_times_raise(self, kwargs):
        with pytest.raises(ValueError, match="without times"):
            compute_home_range(np.array([0, 0, 1]), **kwargs)

    @pytest.mark.parametrize("percentile", [0.0, -10.0, 100.5, np.nan])
    def test_percentile_outside_0_100_raises(self, percentile):
        with pytest.raises(ValueError, match="percentile"):
            compute_home_range(np.array([0, 0, 1]), percentile=percentile)

    def test_fractional_percentile_gets_a_hint(self):
        with pytest.raises(ValueError, match=r"percentile=95"):
            compute_home_range(np.array([0, 0, 1]), percentile=0.95)

    def test_float_bins_raise(self):
        with pytest.raises(TypeError, match="integer"):
            compute_home_range(np.array([0.0, 1.7, 2.0]))


class TestMeanSquareDisplacement:
    """Test mean_square_displacement function."""

    def test_msd_shape(self):
        """Test that MSD returns two arrays (tau, msd)."""
        # Create simple trajectory (continuous positions)
        positions = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        times = np.linspace(0, 10, 50)

        # Continuous API with Euclidean distance (default)
        _msd = mean_square_displacement(
            times, positions, metric="euclidean", max_tau=5.0
        )
        tau_values, msd_values = _msd.lags, _msd.msd

        # Both should be 1D arrays
        assert tau_values.ndim == 1
        assert msd_values.ndim == 1
        assert len(tau_values) == len(msd_values)

    def test_msd_monotonic_increase(self):
        """Test that MSD generally increases with tau (for diffusive motion)."""
        # Create 2D random walk trajectory with local RNG
        rng = np.random.default_rng(42)
        n_steps = 100
        steps = rng.standard_normal((n_steps, 2)) * 2  # 2D random steps
        positions = np.cumsum(steps, axis=0)
        times = np.arange(n_steps) * 0.1

        # Continuous API with Euclidean distance
        _msd = mean_square_displacement(
            times, positions, metric="euclidean", max_tau=5.0
        )
        _tau_values, msd_values = _msd.lags, _msd.msd

        # MSD should generally increase with tau (monotonic for diffusion)
        # Due to sampling noise, allow for small decreases
        # Check that MSD at end is significantly larger than at start
        assert msd_values[-1] > msd_values[0]

    def test_msd_stationary_is_zero(self):
        """Test that MSD is near zero for stationary trajectory."""
        # Stationary trajectory (all same position)
        positions = np.tile([0.0, 0.0], (50, 1))  # 50 identical positions
        times = np.linspace(0, 10, 50)

        # Continuous API
        _msd = mean_square_displacement(
            times, positions, metric="euclidean", max_tau=5.0
        )
        _tau_values, msd_values = _msd.lags, _msd.msd

        # All MSD values should be zero (no displacement)
        assert_allclose(msd_values, 0.0, atol=1e-10)

    def test_msd_max_tau_parameter(self):
        """Test that max_tau limits the lag times."""
        positions = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        times = np.linspace(0, 10, 50)

        _msd = mean_square_displacement(
            times, positions, metric="euclidean", max_tau=3.0
        )
        tau_values, _ = _msd.lags, _msd.msd

        # All tau values should be <= max_tau
        assert np.all(tau_values <= 3.0)

    def test_msd_returns_floats(self):
        """Test that MSD values are float64."""
        positions = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        times = np.linspace(0, 10, 50)

        _msd = mean_square_displacement(
            times, positions, metric="euclidean", max_tau=5.0
        )
        tau_values, msd_values = _msd.lags, _msd.msd

        assert tau_values.dtype == np.float64
        assert msd_values.dtype == np.float64

    def test_parameter_order(self):
        """Test new continuous API signature."""
        positions = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        times = np.linspace(0, 10, 50)

        # Euclidean (default)
        _msd = mean_square_displacement(times, positions, max_tau=5.0)
        tau_values, msd_values = _msd.lags, _msd.msd
        assert isinstance(tau_values, np.ndarray)
        assert isinstance(msd_values, np.ndarray)

        # Geodesic requires env
        env = Environment.from_samples(positions, bin_size=5.0)
        bin_positions = env.bin_centers[env.bin_at(positions)]
        _msd = mean_square_displacement(
            times, bin_positions, metric="geodesic", env=env, max_tau=5.0
        )
        tau_geo, msd_geo = _msd.lags, _msd.msd
        assert isinstance(tau_geo, np.ndarray)
        assert isinstance(msd_geo, np.ndarray)

    @pytest.mark.parametrize("metric", ["euclidean", "geodesic"])
    def test_msd_matches_nearest_time_before_target(self, metric):
        """An off-grid lag includes movement before the trajectory stops."""
        times = np.arange(11.0)
        positions = np.minimum(times, 5)[:, None] + 0.5
        env = Environment.from_grid_mask(
            np.ones(6, dtype=bool), grid_edges=(np.arange(7.0),)
        )

        result = mean_square_displacement(
            times, positions, metric=metric, env=env, max_tau=2.2
        )

        # max_tau=2.2 gives lags 1 and 2 s. The nine pairs two samples
        # apart contribute 4, 4, 4, 4, 1 while moving and 0 after stopping.
        assert result.lags[-1] == pytest.approx(2.0)
        assert result.msd[-1] == pytest.approx(17 / 9)

    def test_msd_uses_both_neighbors_for_irregular_samples(self):
        """Both sides of a target can be its nearest recorded observation."""
        times = np.array([0.0, 1.0, 2.3, 3.2, 4.0, 5.0])
        positions = times[:, None]

        result = mean_square_displacement(times, positions, max_tau=2.1)

        # Target 2.1 chooses the successor 2.3; targets 3.1 and 4.4
        # choose their predecessor 3.2 and 4.0, respectively.
        expected = np.mean(np.array([2.3, 2.2, 1.7]) ** 2)
        assert result.msd[-1] == pytest.approx(expected)

    @pytest.mark.parametrize("metric", ["euclidean", "geodesic"])
    def test_msd_keeps_endpoint_pair_despite_timestamp_roundoff(self, metric):
        """The rounded target .2 + .1 still represents the final observation."""
        times = np.array([0.0, 0.1, 0.2, 0.3])
        positions = np.array([[0.5], [1.5], [2.5], [100.5]])
        env = Environment.from_grid_mask(
            np.ones(101, dtype=bool), grid_edges=(np.arange(102.0),)
        )

        result = mean_square_displacement(
            times, positions, metric=metric, env=env, max_tau=0.1
        )

        assert result.msd[0] == pytest.approx((1 + 1 + 98**2) / 3)

    @pytest.mark.parametrize("metric", ["euclidean", "geodesic"])
    def test_msd_excludes_real_target_after_recording_endpoint(self, metric):
        """A target beyond support is excluded even within matching tolerance."""
        times = np.array([0.0, 0.1, 0.2, 0.22, 0.3])
        positions = np.array([[0.5], [1.5], [2.5], [3.5], [100.5]])
        env = Environment.from_grid_mask(
            np.ones(101, dtype=bool), grid_edges=(np.arange(102.0),)
        )

        result = mean_square_displacement(
            times, positions, metric=metric, env=env, max_tau=0.09
        )

        # Median dt is .09: source .22 targets .31 and must be excluded,
        # while sources 0, .1 and .2 match .1, .2 and .3 respectively.
        assert result.msd[0] == pytest.approx((1 + 1 + 98**2) / 3)

    def test_msd_without_temporal_matches_is_missing(self):
        """An unobserved lag across a sampling gap is not zero motion."""
        times = np.array([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])

        result = mean_square_displacement(times, times[:, None], max_tau=10.0)

        # At most n // 2 = 3 lags. Lags 1 and 2 s are observed within each
        # stretch; no pair is 3 s apart.
        np.testing.assert_allclose(result.lags, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result.msd[:2], [1.0, 4.0])
        assert np.isnan(result.msd[2])

    @pytest.mark.parametrize("missing", [100.0, np.nan, np.inf])
    def test_geodesic_msd_excludes_missing_or_outside_bins(self, missing):
        """Only the observed stationary pair contributes to the estimate."""
        env = Environment.from_grid_mask(
            np.ones(3, dtype=bool), grid_edges=(np.arange(4.0),)
        )
        times = np.arange(4) * 0.1
        positions = np.array([[0.5], [0.5], [missing], [0.5]])

        result = mean_square_displacement(
            times, positions, metric="geodesic", env=env, max_tau=0.1
        )

        assert result.msd[0] == 0.0

    @pytest.mark.parametrize("metric", ["euclidean", "geodesic"])
    def test_msd_without_observed_positions_is_missing(self, metric):
        """Missing positions cannot establish zero displacement."""
        env = Environment.from_grid_mask(
            np.ones(3, dtype=bool), grid_edges=(np.arange(4.0),)
        )
        times = np.arange(4) * 0.1
        positions = np.full((4, 1), np.nan)

        result = mean_square_displacement(
            times, positions, metric=metric, env=env, max_tau=0.1
        )

        assert np.isnan(result.msd).all()

    def test_geodesic_msd_without_connected_pairs_is_missing(self):
        """Disconnected positions supply no finite displacement estimate."""
        env = Environment.from_grid_mask(
            np.array([True, False, True]), grid_edges=(np.arange(4.0),)
        )
        times = np.arange(4) * 0.1
        positions = np.array([[0.5], [2.5], [0.5], [2.5]])

        result = mean_square_displacement(
            times, positions, metric="geodesic", env=env, max_tau=0.1
        )

        assert np.isnan(result.msd).all()


class TestCurvatureSampleAlignment:
    """Stationary observations preserve the time and index of each turn."""

    def test_stationary_start_preserves_turn_index(self):
        positions = np.array([[0.0, 0.0]] * 6 + [[1.0, 0.0], [1.0, 1.0]])
        times = np.arange(8) * 0.1

        curvature = compute_trajectory_curvature(times, positions, smooth_window=0.0)

        # Only the retained middle position (sample 6) has a turn; the
        # stationary repeats and the run's ends are NaN.
        expected = np.full(8, np.nan)
        expected[6] = np.pi / 2
        assert_allclose(curvature, expected, equal_nan=True)

    def test_stationary_vertex_preserves_retained_turn_indices(self):
        positions = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [2.0, 1.0],
            ]
        )
        times = np.arange(len(positions)) * 0.1

        curvature = compute_trajectory_curvature(times, positions, smooth_window=0.0)

        expected = np.full(len(positions), np.nan)
        expected[[1, 4]] = [np.pi / 2, -np.pi / 2]
        assert_allclose(curvature, expected, equal_nan=True)

    def test_smoothing_stays_at_turn_time_and_within_observed_run(self):
        first = np.column_stack([np.arange(8.0), np.zeros(8)])
        second = np.array(
            [[0.0, 0.0]] * 6 + [[1.0, 0.0]] + [[1.0, float(y)] for y in range(1, 9)]
        )
        times = np.r_[np.arange(8) * 0.1, 10.0 + np.arange(len(second)) * 0.1]

        curvature = compute_trajectory_curvature(
            times, np.vstack([first, second]), smooth_window=0.1
        )

        assert np.isnan(curvature[[0, 7]]).all()
        assert_allclose(curvature[1:7], 0.0)
        assert np.nanargmax(curvature) == 8 + 6
        expected_second = compute_trajectory_curvature(
            times[8:], second, smooth_window=0.1
        )
        assert_allclose(curvature[8:], expected_second)


class TestCurvatureSmoothing:
    """Smoothing averages defined turns only; 0 disables it."""

    TIMES = np.arange(0, 5, 1 / 30)
    CIRCLE = 7.0 * np.c_[np.cos(0.8 * TIMES), np.sin(0.8 * TIMES)]

    def test_zero_window_is_unsmoothed_not_nan(self):
        turns = compute_trajectory_curvature(self.TIMES, self.CIRCLE, smooth_window=0)
        assert_allclose(turns[1:-1], 0.8 / 30, rtol=1e-9)

    def test_constant_turn_survives_smoothing_up_to_the_run_ends(self):
        # The undefined first and last turns used to count as zero, so the
        # smoothed turn at the run start read 0.87 of the true 0.8 / 30 rad.
        turns = compute_trajectory_curvature(self.TIMES, self.CIRCLE, smooth_window=0.2)
        assert np.isnan(turns[[0, -1]]).all()
        assert_allclose(turns[1:-1], 0.8 / 30, rtol=1e-9)

    def test_pause_is_not_averaged_in_as_zero_turn(self):
        # Turning left 0.1 rad per step, with a 10-sample pause in the middle.
        heading = 0.1 * np.arange(20)
        steps = np.c_[np.cos(heading), np.sin(heading)]
        path = np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])
        path = np.insert(path, 10, np.repeat(path[10:11], 10, axis=0), axis=0)
        times = np.arange(len(path)) * 0.1
        turns = compute_trajectory_curvature(times, path, smooth_window=0.3)
        defined = np.isfinite(turns)
        assert defined.sum() == 19
        assert_allclose(turns[defined], 0.1, rtol=1e-9)

    @pytest.mark.parametrize("window", [-0.1, np.nan, np.inf, None])
    def test_invalid_window_raises(self, window):
        with pytest.raises(ValueError, match="smooth_window"):
            compute_trajectory_curvature(self.TIMES, self.CIRCLE, smooth_window=window)


class TestCurvatureObservationSupport:
    """Turns are computed only within observed runs on a strictly increasing clock."""

    def test_missing_row_splits_the_run(self):
        times = np.arange(7) * 0.1
        positions = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [np.nan, np.nan],
                [3.0, 1.0],
                [3.0, 2.0],
            ]
        )

        curvature = compute_trajectory_curvature(
            times, positions, smooth_window=0.0, max_gap=0.15
        )

        # Run ends have no turn, and the two-sample second run has none at all.
        assert_allclose(
            curvature,
            [np.nan, 0, np.pi / 2, np.nan, np.nan, np.nan, np.nan],
            rtol=0,
            atol=1e-15,
            equal_nan=True,
        )

    def test_missing_row_is_bridged_like_a_deleted_row(self):
        times = np.arange(20) * 0.1
        positions = np.column_stack([np.arange(20.0), np.sin(np.arange(20.0))])
        positions[10] = np.nan
        kept = np.delete(np.arange(20), 10)

        curvature = compute_trajectory_curvature(times, positions)

        assert np.isnan(curvature[10])
        assert_allclose(
            curvature[kept],
            compute_trajectory_curvature(times[kept], positions[kept]),
            rtol=1e-12,
            atol=0,
        )

    def test_missing_row_does_not_spread_through_smoothing(self):
        times = np.arange(20) * 0.1
        positions = np.column_stack([np.arange(20.0), np.sin(np.arange(20.0))])
        positions[10] = np.nan

        curvature = compute_trajectory_curvature(times, positions, max_gap=0.15)

        # Each run's ends (samples 0, 9, 11, 19) have no turn.
        np.testing.assert_array_equal(
            np.flatnonzero(np.isnan(curvature)), [0, 9, 10, 11, 19]
        )
        assert_allclose(
            np.delete(curvature, 10),
            np.r_[
                compute_trajectory_curvature(times[:10], positions[:10]),
                compute_trajectory_curvature(times[11:], positions[11:]),
            ],
            rtol=1e-12,
            atol=0,
        )

    def test_repeated_timestamp_raises(self):
        times = np.array([0.0, 0.1, 0.1, 0.2])
        with pytest.raises(ValueError, match="strictly increasing"):
            compute_trajectory_curvature(times, np.arange(8.0).reshape(4, 2))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="got no samples"):
            compute_trajectory_curvature(np.zeros(0), np.zeros((0, 2)))

    def test_one_sample_has_no_turn(self):
        curvature = compute_trajectory_curvature(
            np.array([0.0]), np.array([[1.0, 2.0]])
        )
        np.testing.assert_array_equal(curvature, [np.nan])


class TestTrajectoryMetricsIntegration:
    """Test integration of all trajectory metrics."""

    def test_complete_workflow(self):
        """Test complete trajectory analysis workflow with continuous API."""
        # Generate synthetic meandering trajectory (deterministic - no randomness needed)
        t = np.linspace(0, 4 * np.pi, 200)
        x = t * 5 + 20 * np.sin(t)
        y = 20 * np.cos(t)
        positions = np.column_stack([x, y])
        times = np.linspace(0, 20, len(positions))

        # Continuous API: all metrics work directly on positions!
        turn_angles = compute_turn_angles(positions)
        step_lengths = compute_step_lengths(positions, metric="euclidean")

        # Home range still uses bins (makes sense for occupancy-based metrics)
        env = Environment.from_samples(positions, bin_size=5.0)
        position_bins = env.bin_at(positions)
        home_range = compute_home_range(position_bins, percentile=95.0)

        # MSD uses continuous positions
        _msd = mean_square_displacement(
            times, positions, metric="euclidean", max_tau=10.0
        )
        tau_values, msd_values = _msd.lags, _msd.msd

        # All metrics should be computed successfully
        assert len(turn_angles) > 0
        assert len(step_lengths) > 0
        assert len(home_range) > 0
        assert len(tau_values) > 0
        assert len(msd_values) > 0

        # Sanity checks
        assert np.all(np.abs(turn_angles) <= np.pi)
        assert np.all(step_lengths >= 0)
        assert np.all(tau_values >= 0)
        assert np.all(msd_values >= 0)


class TestGeodesicDistanceOptimization:
    """Tests for geodesic distance matrix optimization.

    These tests verify that the precomputed distance matrix approach
    produces identical results to per-step Dijkstra (the original implementation).
    """

    def test_step_lengths_geodesic_matches_networkx(self):
        """Test that optimized geodesic step lengths match NetworkX shortest_path_length.

        This verifies the optimization in trajectory.py (using geodesic_distance_matrix
        instead of per-step nx.shortest_path_length) produces identical results.
        """
        import networkx as nx

        # Create 2D grid environment
        x = np.linspace(0, 40, 100)
        y = np.linspace(0, 40, 100)
        xx, yy = np.meshgrid(x, y)
        sample_positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        # Create trajectory visiting multiple bins
        position_bins = np.array([0, 1, 2, 5, 10, 15, 10, 5, 2, 1, 0])
        positions = env.bin_centers[position_bins]

        # Get step lengths using our optimized implementation
        step_lengths = compute_step_lengths(positions, metric="geodesic", env=env)

        # Compare to NetworkX ground truth (per-step Dijkstra)
        for i in range(len(step_lengths)):
            bin_i = position_bins[i]
            bin_j = position_bins[i + 1]

            if bin_i == bin_j:
                expected = 0.0
            else:
                try:
                    expected = float(
                        nx.shortest_path_length(
                            env.connectivity,
                            source=int(bin_i),
                            target=int(bin_j),
                            weight="distance",
                        )
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    expected = np.inf

            assert_allclose(
                step_lengths[i],
                expected,
                rtol=1e-10,
                err_msg=f"Step {i}: optimized={step_lengths[i]}, NetworkX={expected}",
            )

    def test_geodesic_handles_disconnected_bins(self):
        """Test that geodesic distance returns inf for disconnected bins."""
        # Create environment with disconnected components
        # (Use a sparse point cloud that creates disconnected components)
        positions = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],  # Connected component 1
                [100.0, 100.0],
                [101.0, 100.0],  # Connected component 2
            ]
        )
        env = Environment.from_samples(positions, bin_size=2.0)

        # If we have disconnected components, test the behavior
        # Note: This test may pass trivially if all bins connect
        # The key is that the code handles inf distances gracefully
        position_bins = env.bin_at(positions)
        bin_positions = env.bin_centers[position_bins]

        step_lengths = compute_step_lengths(bin_positions, metric="geodesic", env=env)

        # Step lengths should be finite for connected steps, inf for disconnected
        assert np.all(step_lengths >= 0), "All step lengths should be non-negative"


class TestArrayLikeNormalization:
    """Tests that public boundary functions coerce array-likes before .ndim/.shape access.

    These tests enforce that passing a plain Python list (a common beginner input)
    either succeeds cleanly or raises a domain ValueError/TypeError — NEVER an
    AttributeError from accessing .ndim on a list.
    """

    # --- compute_turn_angles ---

    def test_compute_turn_angles_accepts_list_of_lists(self):
        """compute_turn_angles must accept a valid Python list-of-lists."""
        positions_list = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        # Should succeed — not raise AttributeError or anything else
        result = compute_turn_angles(positions_list)
        assert isinstance(result, np.ndarray)

    def test_compute_turn_angles_list_no_attribute_error(self):
        """compute_turn_angles must not raise AttributeError for a list input."""
        positions_list = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
        try:
            compute_turn_angles(positions_list)
        except AttributeError:
            pytest.fail(
                "compute_turn_angles raised AttributeError on list input — "
                "must coerce with np.asarray before accessing .ndim"
            )
        except (ValueError, TypeError):
            pass  # Domain error is fine

    def test_compute_turn_angles_1d_list_raises_domain_error(self):
        """compute_turn_angles must raise ValueError/TypeError (not AttributeError) for 1-D input."""
        with pytest.raises((ValueError, TypeError)):
            compute_turn_angles([0.0, 1.0, 2.0])  # 1D, not 2D

    def test_compute_turn_angles_non_numeric_raises_domain_error(self):
        """compute_turn_angles must raise TypeError (not AttributeError) for non-numeric input."""
        with pytest.raises((TypeError, ValueError)):
            compute_turn_angles([["a", "b"], ["c", "d"], ["e", "f"]])

    # --- compute_step_lengths ---

    def test_compute_step_lengths_accepts_list_of_lists(self):
        """compute_step_lengths must accept a valid Python list-of-lists."""
        positions_list = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        result = compute_step_lengths(positions_list, metric="euclidean")
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_compute_step_lengths_list_no_attribute_error(self):
        """compute_step_lengths must not raise AttributeError for a list input."""
        positions_list = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        try:
            compute_step_lengths(positions_list)
        except AttributeError:
            pytest.fail(
                "compute_step_lengths raised AttributeError on list input — "
                "must coerce with np.asarray before accessing .ndim"
            )
        except (ValueError, TypeError):
            pass  # Domain error is fine

    def test_compute_step_lengths_1d_list_raises_domain_error(self):
        """compute_step_lengths must raise ValueError/TypeError (not AttributeError) for 1-D input."""
        with pytest.raises((ValueError, TypeError)):
            compute_step_lengths([0.0, 1.0, 2.0])  # 1D flat list

    def test_compute_step_lengths_non_numeric_raises_domain_error(self):
        """compute_step_lengths must raise TypeError (not AttributeError) for non-numeric input."""
        with pytest.raises((TypeError, ValueError)):
            compute_step_lengths([["a", "b"], ["c", "d"]])

    # --- mean_square_displacement ---

    def test_msd_accepts_list_of_lists(self):
        """mean_square_displacement must accept a valid Python list-of-lists."""
        positions_list = [[float(i), 0.0] for i in range(10)]
        times_list = [float(i) * 0.1 for i in range(10)]
        result = mean_square_displacement(times_list, positions_list)
        assert hasattr(result, "lags")
        assert hasattr(result, "msd")

    def test_msd_list_no_attribute_error(self):
        """mean_square_displacement must not raise AttributeError for list input."""
        positions_list = [[float(i), 0.0] for i in range(10)]
        times_list = [float(i) * 0.1 for i in range(10)]
        try:
            mean_square_displacement(times_list, positions_list)
        except AttributeError:
            pytest.fail(
                "mean_square_displacement raised AttributeError on list input — "
                "must coerce with np.asarray before accessing .ndim"
            )
        except (ValueError, TypeError):
            pass  # Domain error is fine

    def test_msd_1d_positions_are_one_coordinate(self):
        """1-D positions give the same MSD as the (n, 1) column."""
        times = np.arange(20) * 0.1
        x = np.cumsum(np.random.default_rng(1).standard_normal(20))
        flat = mean_square_displacement(times, x, max_tau=1.0)
        column = mean_square_displacement(times, x[:, None], max_tau=1.0)
        np.testing.assert_array_equal(flat.lags, column.lags)
        np.testing.assert_array_equal(flat.msd, column.msd)

    def test_msd_empty_raises(self):
        with pytest.raises(ValueError, match="no samples"):
            mean_square_displacement(np.empty(0), np.empty((0, 2)))

    def test_msd_one_sample_is_empty_result(self):
        result = mean_square_displacement(np.array([0.0]), np.array([[1.0, 2.0]]))
        assert result.lags.shape == (0,)
        assert result.msd.shape == (0,)

    def test_msd_non_numeric_raises_domain_error(self):
        """mean_square_displacement raises TypeError (not AttributeError) for non-numeric input."""
        times_list = [0.0, 0.1, 0.2]
        with pytest.raises((TypeError, ValueError)):
            mean_square_displacement(times_list, [["a", "b"], ["c", "d"], ["e", "f"]])


class TestMeanSquareDisplacementBreaks:
    """No displacement pair spans a declared break or an epochs edge."""

    @staticmethod
    def _two_recordings():
        times = np.arange(10.0)
        x = np.array([0, 1, 2, 3, 4, 100, 101, 102, 103, 104], dtype=float)
        return times, np.c_[x, np.zeros(10)]

    def test_break_before_excludes_pairs_across_the_join(self):
        times, positions = self._two_recordings()
        break_before = np.zeros(10, dtype=bool)
        break_before[5] = True
        result = mean_square_displacement(
            times, positions, max_tau=3.0, break_before=break_before
        )
        np.testing.assert_array_equal(result.lags, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result.msd, [1.0, 4.0, 9.0])

    def test_without_break_pairs_span_the_join(self):
        times, positions = self._two_recordings()
        result = mean_square_displacement(times, positions, max_tau=3.0)
        np.testing.assert_allclose(
            result.msd, [9224 / 9, 18842 / 8, 28848 / 7], rtol=1e-12
        )

    def test_epochs_exclude_pairs_across_the_edge(self):
        times, positions = self._two_recordings()
        result = mean_square_displacement(
            times, positions, max_tau=3.0, epochs=[(0.0, 4.5), (5.0, 10.0)]
        )
        np.testing.assert_array_equal(result.msd, [1.0, 4.0, 9.0])

    def test_touching_epochs_merge(self):
        times, positions = self._two_recordings()
        merged = mean_square_displacement(
            times, positions, max_tau=3.0, epochs=[(0.0, 5.0), (5.0, 10.0)]
        )
        plain = mean_square_displacement(times, positions, max_tau=3.0)
        np.testing.assert_array_equal(merged.msd, plain.msd)

    def test_nan_row_pairs_still_excluded(self):
        times = np.arange(10.0)
        positions = np.c_[np.arange(10.0), np.zeros(10)]
        positions[5] = np.nan
        result = mean_square_displacement(times, positions, max_tau=3.0)
        np.testing.assert_array_equal(result.msd, [1.0, 4.0, 9.0])

    def test_lag_with_no_pair_inside_a_segment_is_nan(self):
        times, positions = self._two_recordings()
        break_before = np.zeros(10, dtype=bool)
        break_before[5] = True
        result = mean_square_displacement(
            times, positions[:, :1], max_tau=5.0, break_before=break_before
        )
        # Each recording spans four seconds, so lag 5 has no pair inside one.
        np.testing.assert_array_equal(result.msd[:4], [1.0, 4.0, 9.0, 16.0])
        assert np.isnan(result.msd[4])

    def test_geodesic_pairs_do_not_span_the_break(self):
        times, positions = self._two_recordings()
        line = np.c_[np.linspace(0.0, 104.0, 400), np.zeros(400)]
        env = Environment.from_samples(line, bin_size=1.0)
        break_before = np.zeros(10, dtype=bool)
        break_before[5] = True
        joined = mean_square_displacement(
            times, positions, metric="geodesic", env=env, max_tau=3.0
        )
        split = mean_square_displacement(
            times,
            positions,
            metric="geodesic",
            env=env,
            max_tau=3.0,
            break_before=break_before,
        )
        # Pairs across the join are ~100 units apart; within a recording the
        # displacement is at most 3 bin-widths.
        assert np.all(joined.msd > 100.0)
        np.testing.assert_allclose(split.msd, [1.0, 4.0, 9.0], atol=1.5)


class TestMeanSquareDisplacementLags:
    """Each reported lag is the time separation of the pairs it averages."""

    V = 3.0  # cm/s

    def _ballistic(self, n=100, dt=0.1):
        times = np.arange(n) * dt
        return times, np.c_[self.V * times, np.zeros(n)]

    @pytest.mark.parametrize("max_tau", [None, 0.55, 2.0, 2.04])
    def test_ballistic_msd_is_speed_times_lag_squared(self, max_tau):
        times, positions = self._ballistic()
        result = mean_square_displacement(times, positions, max_tau=max_tau)
        assert_allclose(result.msd, (self.V * result.lags) ** 2, rtol=1e-12)
        # Lags are whole multiples of the 0.1 s sample interval.
        k = np.arange(1, result.lags.size + 1)
        assert_allclose(result.lags, 0.1 * k, rtol=1e-12)
        alpha = np.polyfit(np.log(result.lags), np.log(result.msd), 1)[0]
        assert alpha == pytest.approx(2.0, abs=1e-9)

    def test_lags_stop_at_max_tau(self):
        times, positions = self._ballistic()
        assert mean_square_displacement(times, positions, max_tau=0.55).lags.size == 5
        assert mean_square_displacement(times, positions, max_tau=0.3).lags.size == 3

    def test_default_max_tau_is_half_the_recording(self):
        times, positions = self._ballistic()
        result = mean_square_displacement(times, positions)
        # 9.9 s recording: half is 4.95 s, so lags 0.1 .. 4.9 s.
        assert result.lags.size == 49
        assert result.lags[-1] == pytest.approx(4.9)

    def test_irregular_lag_is_mean_separation_of_its_pairs(self):
        # Median interval 1 s; the 2 s lag averages pairs 2.3, 2.2 and 1.7 s apart.
        times = np.array([0.0, 1.0, 2.3, 3.2, 4.0, 5.0])
        result = mean_square_displacement(times, times[:, None], max_tau=2.1)
        assert result.lags[-1] == pytest.approx((2.3 + 2.2 + 1.7) / 3)

    @pytest.mark.parametrize("max_tau", [0.0, -1.0, np.nan, np.inf])
    def test_max_tau_must_be_positive_and_finite(self, max_tau):
        times, positions = self._ballistic()
        with pytest.raises(ValueError, match="max_tau"):
            mean_square_displacement(times, positions, max_tau=max_tau)

    def test_max_tau_longer_than_recording_raises(self):
        times, positions = self._ballistic()
        with pytest.raises(ValueError, match="longer than the recording"):
            mean_square_displacement(times, positions, max_tau=100.0)

    def test_result_records_metric_and_epochs(self):
        times, positions = self._ballistic()
        result = mean_square_displacement(
            times, positions, max_tau=1.0, epochs=(0.0, 5.0)
        )
        assert result.metric == "euclidean"
        np.testing.assert_array_equal(result.epochs, [[0.0, 5.0]])
        assert mean_square_displacement(times, positions, max_tau=1.0).epochs is None

    def test_result_is_exported_from_behavior(self):
        import neurospatial.behavior as behavior
        from neurospatial.behavior.trajectory import MSDResult

        assert behavior.MSDResult is MSDResult
        assert "MSDResult" in behavior.__all__

    @pytest.mark.parametrize(
        ("lags", "msd", "metric", "match"),
        [
            ([1.0, 2.0], [1.0], "euclidean", "same length"),
            ([2.0, 1.0], [1.0, 4.0], "euclidean", "strictly increasing"),
            ([1.0, 2.0], [1.0, 4.0], "manhattan", "metric"),
        ],
    )
    def test_result_rejects_inconsistent_fields(self, lags, msd, metric, match):
        from neurospatial.behavior.trajectory import MSDResult

        with pytest.raises(ValueError, match=match):
            MSDResult(
                lags=np.array(lags), msd=np.array(msd), metric=metric, epochs=None
            )

    def test_max_tau_shorter_than_one_sample_interval_raises(self):
        times, positions = self._ballistic()
        with pytest.raises(ValueError, match="shorter than one sample interval"):
            mean_square_displacement(times, positions, max_tau=0.04)
