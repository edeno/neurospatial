"""Tests for trajectory metrics module."""

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment, InputError
from neurospatial.behavior.trajectory import (
    compute_home_range,
    compute_planar_turn_angles,
    mean_square_displacement,
    step_lengths,
)

# Equal one-unit bins: the home range ranks them by occupancy alone.
_HOME_ENV = Environment.from_samples(
    np.arange(1000.0)[:, None], bin_size=1.0, infer_active_bins=False
)


class TestComputeStepLengths:
    """Test step_lengths function."""

    def test_empty_positions_raise_a_clear_error(self):
        with pytest.raises(ValueError, match=r"step_lengths\(\) got no samples"):
            step_lengths(np.empty((0, 2)))

    def test_one_position_has_no_step(self):
        assert step_lengths(np.zeros((1, 2))).shape == (0,)

    @pytest.mark.parametrize(
        "function",
        [
            lambda **kw: step_lengths(np.zeros((3, 2)), **kw),
            lambda **kw: mean_square_displacement(
                np.arange(3.0), np.zeros((3, 2)), **kw
            ),
        ],
        ids=["step_lengths", "msd"],
    )
    def test_unknown_metric_and_geodesic_without_env_raise(self, function):
        with pytest.raises(ValueError, match="must be 'euclidean' or 'geodesic'"):
            function(metric="manhattan")
        with pytest.raises(ValueError, match=r"\(metric='geodesic'\) needs env="):
            function(metric="geodesic")

    def test_step_lengths_straight_line(self):
        """Test step lengths on a straight 1D trajectory with Euclidean distance."""
        # Create straight line trajectory (continuous positions)
        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])

        # Euclidean distance (default)
        steps = step_lengths(positions, metric="euclidean")

        # Should have n-1 step lengths
        assert steps.shape == (20,)

        # 21 evenly spaced samples over 100 cm: every step is 5 cm.
        assert_allclose(steps, 5.0, rtol=1e-12)

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

        steps = step_lengths(positions, metric="euclidean")

        # Should have n-1 step lengths
        assert steps.shape == (6,)

        # Consecutive duplicates should have zero distance
        assert steps[0] == 0.0  # [0,0] -> [0,0]
        assert steps[3] == 0.0  # [20,0] -> [20,0]
        assert steps[4] == 0.0  # [20,0] -> [20,0]

    def test_step_lengths_propagate_nan_to_both_adjacent_steps(self):
        positions = np.array([[0.0, 0.0], [np.nan, np.nan], [3.0, 4.0]])
        assert_allclose(
            step_lengths(positions),
            [np.nan, np.nan],
            rtol=0,
            atol=1e-15,
            equal_nan=True,
        )

    def test_geodesic_steps_in_an_open_field_are_straight_steps(self):
        """Inside an open field every straight step stays inside: Euclidean."""
        x = np.linspace(0, 40, 100)
        xx, yy = np.meshgrid(x, x)
        env = Environment.from_samples(
            np.column_stack([xx.ravel(), yy.ravel()]), bin_size=5.0
        )
        # Oblique steps across several bins, plus one inside a bin.
        positions = np.array(
            [[2.0, 3.0], [14.0, 8.0], [14.5, 8.2], [31.0, 37.0], [3.0, 2.0]]
        )

        steps = step_lengths(positions, metric="geodesic", env=env)

        assert_allclose(
            steps,
            np.linalg.norm(np.diff(positions, axis=0), axis=1),
            rtol=1e-12,
        )

    def test_parameter_order(self):
        """Test new continuous API signature."""
        positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])

        # Euclidean (default)
        steps = step_lengths(positions)
        assert isinstance(steps, np.ndarray)

        # Geodesic requires env
        env = Environment.from_samples(positions, bin_size=5.0)
        bin_positions = env.bin_centers[env.bin_at(positions)]
        steps_geo = step_lengths(bin_positions, metric="geodesic", env=env)
        assert isinstance(steps_geo, np.ndarray)


def _dwell_samples(
    position_bins: np.ndarray, env: Environment = _HOME_ENV, dt: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Times and positions that dwell ``dt`` at each bin centre in turn.

    Occupancy gives each interval to the bin of its first sample, so a
    closing sample repeats the last position: every listed sample then
    holds exactly ``dt``, and occupancy is proportional to sample counts.
    """
    positions = env.bin_centers[np.r_[position_bins, position_bins[-1]]]
    return np.arange(len(positions)) * dt, positions


def _home_range(position_bins, env: Environment = _HOME_ENV, **kwargs) -> np.ndarray:
    """compute_home_range of ``position_bins`` visited for 0.1 s per sample."""
    times, positions = _dwell_samples(np.asarray(position_bins), env)
    return compute_home_range(env, times, positions, **kwargs)


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

        home_range = _home_range(position_bins, percentile=95.0)

        # 95% of 100 visits = 95 visits
        # Bin 0 (50) + Bin 1 (30) + Bin 2 (15) = 95 visits = 95%
        # So bins 0, 1, 2 should be in home range
        assert set(home_range) == {0, 1, 2}

    def test_home_range_100_percentile(self):
        """Test that 100% includes all visited bins."""
        position_bins = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 0])

        home_range = _home_range(position_bins, percentile=100.0)

        # Should include all unique bins
        assert set(home_range) == {0, 1, 2, 3}

    def test_home_range_50_percentile(self):
        """Test 50% home range (core area)."""
        # Bin 0: 40 visits, bin 1: 30, bins 2-4: 10 each.
        position_bins = np.repeat(np.arange(5), [40, 30, 10, 10, 10])

        home_range = _home_range(position_bins, percentile=50.0)

        # 40 visits are not yet half of 100; adding bin 1 reaches 70.
        np.testing.assert_array_equal(home_range, [0, 1])

    def test_bins_tied_at_the_cutoff_are_all_included(self):
        """Uniform use has no 50% core: every bin is at the cutoff level."""
        position_bins = np.repeat(np.arange(10), 10)
        home_range = _home_range(position_bins, percentile=50.0)
        np.testing.assert_array_equal(home_range, np.arange(10))

    def test_ties_are_independent_of_sample_order(self):
        # 40% in bin 0, then three bins with 20% each; the 70% cutoff falls
        # inside the tie, so the three tied bins are all included.
        position_bins = np.r_[np.repeat(0, 40), np.repeat([1, 2, 3], 20)]
        expected = [0, 1, 2, 3]
        np.testing.assert_array_equal(
            _home_range(position_bins, percentile=70.0), expected
        )
        np.testing.assert_array_equal(
            _home_range(position_bins[::-1].copy(), percentile=70.0),
            expected,
        )

    def test_equal_occupancy_is_ordered_by_bin_index(self):
        position_bins = np.r_[np.repeat(5, 30), np.repeat([3, 1, 4], 10)]
        np.testing.assert_array_equal(
            _home_range(position_bins, percentile=100.0), [5, 1, 3, 4]
        )

    def test_time_weighted_ties_within_roundoff_are_ties(self):
        # Three bins with 0.3 s each, summed from 0.1 s dwells in different
        # orders (bin 2's last sample starts no interval): all tie at 50 %.
        position_bins = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        times = np.r_[np.arange(9) * 0.1, 0.9]
        times = np.round(times, 10)
        home_range = compute_home_range(
            _HOME_ENV,
            times,
            _HOME_ENV.bin_centers[position_bins],
            percentile=50.0,
            max_gap=None,
        )
        np.testing.assert_array_equal(home_range, [0, 1, 2])

    def test_home_range_returns_indices(self):
        """Test that home_range returns bin indices (integers)."""
        position_bins = np.array([0, 1, 2, 1, 0, 1, 2])

        home_range = _home_range(position_bins, percentile=95.0)

        # Should be numpy array of integers
        assert isinstance(home_range, np.ndarray)
        assert home_range.dtype in [np.int32, np.int64, np.intp]

    def test_parameter_order(self):
        """Test that parameter order is (env, times, positions, *, percentile)."""
        times, positions = _dwell_samples(np.array([0, 1, 2, 1, 0]))

        # This should work without error
        home_range = compute_home_range(_HOME_ENV, times, positions, percentile=95.0)
        assert isinstance(home_range, np.ndarray)

    def test_time_weighted_occupancy_changes_ranking(self):
        """Time-weighted occupancy ranks long-dwell bins above high-count bins.

        Bin 0 is visited many times but each visit is brief, while bin 1 is
        visited fewer times but with long dwells. Equal dwells rank bin 0
        first; the recorded times must rank bin 1 first.
        """
        # 6 brief samples in bin 0, then 2 long-dwell samples in bin 1.
        position_bins = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        # dt: five 0.1s gaps in bin 0, then a 0.1s gap into bin 1, then a 10s
        # dwell within bin 1. Bin 1 accumulates far more time despite fewer
        # samples.
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 10.6])

        # Equal dwell per sample: bin 0 has 6, bin 1 has 2 -> bin 0 first.
        count_based = _home_range(position_bins, percentile=100.0)
        assert count_based[0] == 0

        # Time-weighted: bin 1 dwell time dominates -> bin 1 first.
        time_weighted = compute_home_range(
            _HOME_ENV,
            times,
            _HOME_ENV.bin_centers[position_bins],
            percentile=100.0,
            max_gap=None,
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

        count_based = _home_range(position_bins, percentile=95.0)
        time_weighted = compute_home_range(
            _HOME_ENV, times, _HOME_ENV.bin_centers[position_bins], percentile=95.0
        )

        assert set(count_based) == set(time_weighted)

    def test_time_weighted_length_mismatch_raises(self):
        """Mismatched times length raises ValueError."""
        positions = _HOME_ENV.bin_centers[[0, 1, 2]]
        times = np.array([0.0, 0.1])  # wrong length

        with pytest.raises(ValueError, match="same length"):
            compute_home_range(_HOME_ENV, times, positions)

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
            compute_home_range(_HOME_ENV, times, _HOME_ENV.bin_centers[[0, 1, 1, 2]])

    @pytest.mark.parametrize("percentile", [0.0, -10.0, 100.5, np.nan])
    def test_percentile_outside_0_100_raises(self, percentile):
        with pytest.raises(ValueError, match="percentile"):
            _home_range(np.array([0, 0, 1]), percentile=percentile)

    def test_fractional_percentile_gets_a_hint(self):
        with pytest.raises(ValueError, match=r"percentile=95"):
            _home_range(np.array([0, 0, 1]), percentile=0.95)


class TestMeanSquareDisplacement:
    """Test mean_square_displacement function."""

    @pytest.mark.parametrize("missing_share", [0.0, 0.1])
    def test_msd_fft_path_equals_the_pair_average(self, missing_share):
        """Uniform sampling takes the FFT path; it must equal every-pair averaging."""
        rng = np.random.default_rng(11)
        n_samples, dt = 600, 0.02
        times = np.arange(n_samples) * dt
        positions = np.cumsum(rng.normal(0, 1.0, (n_samples, 2)), axis=0)
        positions[rng.random(n_samples) < missing_share] = np.nan

        result = mean_square_displacement(times, positions, max_tau=1.0)

        for lag, msd in zip(result.lags, result.msd, strict=True):
            k = round(lag / dt)
            squared = np.sum((positions[k:] - positions[:-k]) ** 2, axis=1)
            assert lag == pytest.approx(k * dt)
            assert msd == pytest.approx(np.nanmean(squared), rel=1e-10)
        assert len(result.lags) == round(1.0 / dt)

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

    def test_msd_of_a_random_walk_grows_linearly(self):
        """A 2-D random walk's MSD grows linearly: E[MSD(k steps)] = 8 k cm^2.

        Each step is Gaussian with standard deviation 2 cm per axis, so a
        displacement over k steps has expected squared length 2 * 4 * k, and
        so does every pair's in the time average. Averaged over 200 seeded
        walks, the MSD at each of the 50 lags lies within 4 standard errors
        (the spread across walks over sqrt(200)) of 8 k; a 4 SE miss has
        probability 6e-5 per lag.
        """
        rng = np.random.default_rng(42)
        n_walks, n_steps = 200, 100
        times = np.arange(n_steps) * 0.1
        msd = np.array(
            [
                mean_square_displacement(
                    times,
                    np.cumsum(rng.standard_normal((n_steps, 2)) * 2, axis=0),
                    metric="euclidean",
                    max_tau=5.0,
                ).msd
                for _ in range(n_walks)
            ]
        )
        k = np.arange(1, 51)
        mean = msd.mean(axis=0)
        standard_error = msd.std(axis=0, ddof=1) / np.sqrt(n_walks)
        assert np.all(np.abs(mean - 8.0 * k) < 4 * standard_error)

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

    @pytest.mark.parametrize("missing", [100.0, np.nan])
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
    def test_msd_without_observed_positions_raises(self, metric):
        """Missing positions cannot establish zero displacement."""
        env = Environment.from_grid_mask(
            np.ones(3, dtype=bool), grid_edges=(np.arange(4.0),)
        )
        times = np.arange(4) * 0.1
        positions = np.full((4, 1), np.nan)

        with pytest.raises(InputError, match="excluded ALL trajectory intervals"):
            mean_square_displacement(
                times, positions, metric=metric, env=env, max_tau=0.1
            )

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
        turn_angles = compute_planar_turn_angles(times, positions)
        steps = step_lengths(positions, metric="euclidean")

        # Home range ranks the environment's bins by time-weighted occupancy
        env = Environment.from_samples(positions, bin_size=5.0)
        home_range = compute_home_range(env, times, positions, percentile=95.0)

        # MSD uses continuous positions
        _msd = mean_square_displacement(
            times, positions, metric="euclidean", max_tau=10.0
        )
        tau_values, msd_values = _msd.lags, _msd.msd

        # All metrics should be computed successfully
        assert turn_angles.shape == (len(positions),)
        assert len(steps) > 0
        assert len(home_range) > 0
        assert len(tau_values) > 0
        assert len(msd_values) > 0

        # Sanity checks
        assert np.isnan(turn_angles[[0, -1]]).all()
        assert np.all(np.abs(turn_angles[1:-1]) <= np.pi)
        assert np.all(steps >= 0)
        assert np.all(tau_values >= 0)
        assert np.all(msd_values >= 0)


class TestGeodesicStepLengths:
    """Geodesic step lengths are shortest paths inside the environment."""

    def test_geodesic_step_across_a_wall_goes_around_it(self):
        """A step whose straight line crosses a wall is measured around it.

        A U of 1 cm corridors: the left arm x in [0, 1], the right arm
        x in [4, 5] (y in [0, 10]), joined by the top row y in [9, 10]. The
        step from (0.5, 0.5) to (4.5, 0.5) crosses the wall x in [1, 4],
        y in [0, 9]. The shortest path inside the U touches the wall's
        corners (1, 9) and (4, 9): 2 * hypot(0.5, 8.5) + 3 = 20.03 cm.
        The taut path through bin centers rounds each of the two corners at
        most one bin (1 cm) wide.
        """
        cells = np.array([(x + 0.5, y + 0.5) for x in range(5) for y in range(10)])
        u_shape = cells[(cells[:, 0] < 1) | (cells[:, 0] > 4) | (cells[:, 1] > 9)]
        env = Environment.from_samples(u_shape, bin_size=1.0)
        positions = np.array([[0.5, 0.5], [4.5, 0.5]])

        (step,) = step_lengths(positions, metric="geodesic", env=env)

        exact = 2 * np.hypot(0.5, 8.5) + 3.0
        assert exact <= step <= exact + 2 * 1.0
        # The bin-graph route (grid steps) is no shorter.
        assert step <= env.distance_between(points=positions) + 1e-12

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

        # 2 cm bins: the first three samples fall in bins centered at
        # (-0.5, 0) and (1.5, 0), the last two at (99.5, 100) and (101.5, 100).
        position_bins = env.bin_at(positions)
        bin_positions = env.bin_centers[position_bins]
        assert not nx.has_path(env.connectivity, position_bins[0], position_bins[3])

        steps = step_lengths(bin_positions, metric="geodesic", env=env)

        # Within a component the path is straight; across them there is none.
        np.testing.assert_array_equal(steps, [2.0, 0.0, np.inf, 2.0])


class TestArrayLikeNormalization:
    """Tests that public boundary functions coerce array-likes before .ndim/.shape access.

    These tests enforce that passing a plain Python list (a common beginner input)
    either succeeds cleanly or raises a domain ValueError/TypeError — NEVER an
    AttributeError from accessing .ndim on a list.
    """

    # --- step_lengths ---

    def test_compute_step_lengths_accepts_list_of_lists(self):
        """step_lengths must accept a valid Python list-of-lists."""
        positions_list = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        result = step_lengths(positions_list, metric="euclidean")
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_compute_step_lengths_list_no_attribute_error(self):
        """step_lengths must not raise AttributeError for a list input."""
        positions_list = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        try:
            step_lengths(positions_list)
        except AttributeError:
            pytest.fail(
                "step_lengths raised AttributeError on list input — "
                "must coerce with np.asarray before accessing .ndim"
            )
        except (ValueError, TypeError):
            pass  # Domain error is fine

    def test_compute_step_lengths_1d_list_raises_domain_error(self):
        """step_lengths must raise ValueError/TypeError (not AttributeError) for 1-D input."""
        with pytest.raises((ValueError, TypeError)):
            step_lengths([0.0, 1.0, 2.0])  # 1D flat list

    def test_compute_step_lengths_non_numeric_raises_domain_error(self):
        """step_lengths must raise TypeError (not AttributeError) for non-numeric input."""
        with pytest.raises((TypeError, ValueError)):
            step_lengths([["a", "b"], ["c", "d"]])

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
        np.testing.assert_allclose(result.msd, [1.0, 4.0, 9.0], rtol=1e-12)

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
        np.testing.assert_allclose(result.msd, [1.0, 4.0, 9.0], rtol=1e-12)

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
        np.testing.assert_allclose(result.msd, [1.0, 4.0, 9.0], rtol=1e-12)

    def test_lag_with_no_pair_inside_a_segment_is_nan(self):
        times, positions = self._two_recordings()
        break_before = np.zeros(10, dtype=bool)
        break_before[5] = True
        result = mean_square_displacement(
            times, positions[:, :1], max_tau=5.0, break_before=break_before
        )
        # Each recording spans four seconds, so lag 5 has no pair inside one.
        np.testing.assert_allclose(result.msd[:4], [1.0, 4.0, 9.0, 16.0], rtol=1e-12)
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
        # Along the straight line the geodesic distance is the straight one,
        # so the MSD is that of the x coordinates: within a recording a lag
        # of k seconds moves exactly k cm; joined, the pairs across the join
        # add their ~100 cm jumps.
        x = positions[:, 0]
        brute_force = [np.mean((x[k:] - x[:-k]) ** 2) for k in (1, 2, 3)]
        np.testing.assert_allclose(joined.msd, brute_force, rtol=1e-12)
        np.testing.assert_allclose(split.msd, [1.0, 4.0, 9.0], rtol=1e-12)


class TestMeanSquareDisplacementUniformSampling:
    """Uniformly sampled MSD equals the average over sample pairs ``k`` apart."""

    N = 301
    DT = 0.5

    @staticmethod
    def _pair_average(positions, group, n_lags):
        """Mean squared displacement over valid pairs in one group, per lag."""
        valid = np.isfinite(positions).all(axis=1) & (group >= 0)
        msd = np.full(n_lags, np.nan)
        for k in range(1, n_lags + 1):
            i = np.flatnonzero(valid[:-k] & valid[k:] & (group[:-k] == group[k:]))
            if i.size:
                msd[k - 1] = np.mean(np.sum((positions[i + k] - positions[i]) ** 2, 1))
        return msd

    def _walk(self, n_dims):
        rng = np.random.default_rng(n_dims)
        times = 100.0 + self.DT * np.arange(self.N)
        positions = 1e3 + np.cumsum(rng.normal(size=(self.N, n_dims)), axis=0)
        positions[rng.random(self.N) < 0.1] = np.nan
        return times, positions

    @pytest.mark.parametrize("n_dims", [1, 2, 3])
    def test_default_lags_with_missing_rows_and_breaks(self, n_dims):
        times, positions = self._walk(n_dims)
        break_before = np.zeros(self.N, dtype=bool)
        break_before[[50, 51, 200]] = True

        result = mean_square_displacement(times, positions, break_before=break_before)

        # The default max_tau is a quarter of the 150 s recording: 75 lags.
        n_lags = 75
        expected = self._pair_average(positions, np.cumsum(break_before), n_lags)
        np.testing.assert_array_equal(np.isnan(result.msd), np.isnan(expected))
        np.testing.assert_allclose(result.msd, expected, rtol=1e-10)
        np.testing.assert_allclose(
            result.lags, self.DT * np.arange(1, n_lags + 1), rtol=1e-14
        )

    def test_pairs_stay_inside_one_epoch(self):
        times, positions = self._walk(2)
        epochs = np.array([[100.0, 140.2], [140.2, 160.0], [170.0, 240.0]])

        result = mean_square_displacement(times, positions, max_tau=60.0, epochs=epochs)

        # Touching windows merge. Every interval between a pair's samples lies
        # in one window, so the sample at a window's stop still pairs.
        group = np.full(self.N, -1)
        group[(times >= 100.0) & (times <= 160.0)] = 0
        group[(times >= 170.0) & (times <= 240.0)] = 1
        expected = self._pair_average(positions, group, 120)
        np.testing.assert_allclose(result.msd, expected, rtol=1e-10)

    @pytest.mark.parametrize("seed", range(4))
    def test_value_at_a_lag_does_not_depend_on_max_tau(self, seed):
        times, positions = self._walk(2)
        rng = np.random.default_rng(seed)
        break_before = np.zeros(self.N, dtype=bool)
        break_before[rng.choice(np.arange(1, self.N), seed, replace=False)] = True
        epochs = np.array([[100.0, 180.0], [190.0, 250.0]]) if seed % 2 else None

        short, long = (
            mean_square_displacement(
                times,
                positions,
                max_tau=max_tau,
                epochs=epochs,
                break_before=break_before,
            )
            for max_tau in (10.0, 140.0)
        )

        # Bitwise: a longer max_tau only appends lags.
        n_lags = short.msd.size
        np.testing.assert_array_equal(short.msd, long.msd[:n_lags])
        np.testing.assert_array_equal(short.lags, long.lags[:n_lags])


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

    def test_default_max_tau_is_a_quarter_of_the_recording(self):
        times, positions = self._ballistic()
        result = mean_square_displacement(times, positions)
        # 9.9 s recording: a quarter is 2.475 s, so lags 0.1 .. 2.4 s.
        assert result.lags.size == 24
        assert result.lags[-1] == pytest.approx(2.4)
        explicit = mean_square_displacement(times, positions, max_tau=9.9 / 4)
        np.testing.assert_array_equal(result.msd, explicit.msd)

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


def test_home_range_ranks_bins_by_time_per_area():
    """A small bin used intensively ranks above a large bin used thinly.

    10 samples in a small triangle are more time per unit area than 12 in a
    large one, so the home range holding a third of the time is the small
    triangle; ranked by occupancy it was the large one.
    """
    from shapely.geometry import box

    from neurospatial.layout import create_layout

    env = Environment.from_layout(
        create_layout(
            "TriangularMesh", boundary_polygon=box(0, 0, 10, 10), point_spacing=2.0
        )
    )
    sizes = env.bin_sizes
    small, large = int(np.argmin(sizes)), np.argsort(sizes)[-2:]
    assert 10 / sizes[small] > 12 / sizes[large[0]]
    position_bins = np.repeat([small, large[0], large[1]], [10, 12, 8])
    # Bin centres locate back to their own bins.
    np.testing.assert_array_equal(env.bin_at(env.bin_centers), np.arange(env.n_bins))
    home_range = _home_range(position_bins, env, percentile=33.0)
    np.testing.assert_array_equal(home_range, [small])


def test_home_range_leaves_out_time_outside_the_environment():
    """Time outside the environment is in no bin and not in the total.

    0.3 s in bin 0 and 0.2 s in bin 1 (the step out of the environment adds
    nothing), then 1 s outside. Half of the 0.5 s observed is 0.25 s, held by
    bin 0 alone; counting the time outside, half would need both bins.
    """
    env = Environment.from_samples(
        np.arange(3.0)[:, None], bin_size=1.0, infer_active_bins=False
    )
    positions = np.r_[np.zeros(3), np.ones(3), np.full(10, 50.0)][:, None]
    times = np.arange(len(positions)) * 0.1
    assert env.bin_at(np.array([[50.0]]))[0] == -1
    home_range = compute_home_range(env, times, positions, percentile=50.0)
    np.testing.assert_array_equal(home_range, [0])
