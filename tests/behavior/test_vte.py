"""Tests for VTE (Vicarious Trial and Error) metrics module.

VTE is characterized by:
- Head sweeping (looking back and forth between options)
- Pausing/slowing at decision points

Key metrics:
- IdPhi (head_sweep_magnitude): Sum of absolute heading changes
- zIdPhi: Z-scored IdPhi relative to session baseline
- VTE index: Combined head sweep + slowness measure
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment, Regions

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_2d_environment():
    """Create a simple 2D environment for testing."""
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    xx, yy = np.meshgrid(x, y)
    sample_positions = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(sample_positions, bin_size=5.0)


@pytest.fixture
def t_maze_environment():
    """Create a T-maze environment with regions for testing.

    Layout:
        left (goal)  ---  center  ---  right (goal)
                            |
                            |
                          start
    """
    # Create T-maze shape
    # Vertical stem (y: 0 to 50)
    stem_x = np.linspace(45, 55, 5)
    stem_y = np.linspace(0, 50, 20)
    stem_xx, stem_yy = np.meshgrid(stem_x, stem_y)

    # Horizontal bar (y: 50 to 60)
    bar_x = np.linspace(0, 100, 40)
    bar_y = np.linspace(50, 60, 5)
    bar_xx, bar_yy = np.meshgrid(bar_x, bar_y)

    # Combine
    positions = np.vstack(
        [
            np.column_stack([stem_xx.ravel(), stem_yy.ravel()]),
            np.column_stack([bar_xx.ravel(), bar_yy.ravel()]),
        ]
    )

    env = Environment.from_samples(positions, bin_size=5.0)

    # Add regions (point regions - the bin containing each point)
    region_builder = Regions(env.regions.values())
    region_builder.add("start", point=(50.0, 5.0))
    region_builder.add("center", point=(50.0, 55.0))
    region_builder.add("left", point=(10.0, 55.0))
    region_builder.add("right", point=(90.0, 55.0))
    env = env.with_regions(region_builder)

    return env


def _zigzag(n_samples):
    """East-going zig-zag at 10 Hz: alternating y of 50 and 55 cm."""
    times = np.arange(n_samples) / 10
    x = 5.0 * np.arange(n_samples)
    y = np.where(np.arange(n_samples) % 2 == 0, 50.0, 55.0)
    return times, np.column_stack([x, y])


# =============================================================================
# Test wrap_angle()
# =============================================================================


class TestWrapAngle:
    """Test wrap_angle() function."""

    def test_wrap_angle_basic(self):
        """Test basic angle wrapping to (-pi, pi]."""
        from neurospatial.stats.circular import wrap_angle

        # Already in range
        assert_allclose(wrap_angle(np.array([0.0])), [0.0])
        assert_allclose(wrap_angle(np.array([np.pi / 2])), [np.pi / 2])
        assert_allclose(wrap_angle(np.array([-np.pi / 2])), [-np.pi / 2])

    def test_wrap_angle_positive_overflow(self):
        """Test wrapping angles > pi."""
        from neurospatial.stats.circular import wrap_angle

        # 3pi/2 should wrap to -pi/2
        result = wrap_angle(np.array([3 * np.pi / 2]))
        assert_allclose(result, [-np.pi / 2], atol=1e-10)

        # 2pi should wrap to 0 (or very close)
        result = wrap_angle(np.array([2 * np.pi]))
        assert_allclose(result, [0.0], atol=1e-10)

    def test_wrap_angle_negative_overflow(self):
        """Test wrapping angles < -pi."""
        from neurospatial.stats.circular import wrap_angle

        # -3pi/2 should wrap to pi/2
        result = wrap_angle(np.array([-3 * np.pi / 2]))
        assert_allclose(result, [np.pi / 2], atol=1e-10)

    def test_wrap_angle_pi_boundary(self):
        """Test that pi stays at pi (boundary case)."""
        from neurospatial.stats.circular import wrap_angle

        # (-pi, pi] is closed at +pi, so a half-turn either way is +pi.
        result = wrap_angle(np.array([np.pi, -np.pi]))
        assert_allclose(result, [np.pi, np.pi], atol=1e-10)

    def test_wrap_angle_array(self):
        """Test wrapping an array of angles."""
        from neurospatial.stats.circular import wrap_angle

        angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        result = wrap_angle(angles)
        expected = np.array([0, np.pi / 2, np.pi, -np.pi / 2, 0])
        assert_allclose(result, expected, atol=1e-10)


# =============================================================================
# Test head_sweep_magnitude() (IdPhi)
# =============================================================================


class TestHeadSweepMagnitude:
    """Test head_sweep_magnitude() function."""

    def test_head_sweep_straight_line(self):
        """Test that straight-line trajectory has low head sweep."""
        from neurospatial.behavior.vte import head_sweep_magnitude

        # Constant heading (0 radians = East)
        headings = np.zeros(20)
        result = head_sweep_magnitude(headings)

        # No heading changes -> head sweep = 0
        assert result == 0.0

    def test_head_sweep_scanning(self):
        """Test that scanning trajectory has high head sweep."""
        from neurospatial.behavior.vte import head_sweep_magnitude

        # Alternating headings: 0, pi/4, 0, pi/4, 0, pi/4 (looking back and forth)
        headings = np.array([0, np.pi / 4, 0, np.pi / 4, 0, np.pi / 4])
        result = head_sweep_magnitude(headings)

        # 5 transitions of pi/4 each = 5 * pi/4 = 5*pi/4 radians
        expected = 5 * np.pi / 4
        assert_allclose(result, expected, rtol=1e-10)

    def test_head_sweep_full_rotation(self):
        """Test head sweep for full rotation."""
        from neurospatial.behavior.vte import head_sweep_magnitude

        # Gradual rotation from 0 to 2pi (full circle)
        headings = np.linspace(0, 2 * np.pi, 9)  # 8 steps
        result = head_sweep_magnitude(headings)

        # Total rotation = 2pi, each step = pi/4, 8 steps
        expected = 2 * np.pi
        assert_allclose(result, expected, rtol=1e-10)

    def test_head_sweep_handles_nan(self):
        """Test that NaN values are filtered out."""
        from neurospatial.behavior.vte import head_sweep_magnitude

        # Headings with NaN (stationary periods)
        headings = np.array([0, np.nan, np.pi / 4, np.nan, 0])
        result = head_sweep_magnitude(headings)

        # Valid: [0, pi/4, 0] -> 2 transitions of pi/4 each = pi/2
        expected = np.pi / 2
        assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize(
        "headings",
        [np.array([0.0]), np.array([np.nan, np.nan, np.nan]), np.array([])],
        ids=["one", "all_nan", "empty"],
    )
    def test_head_sweep_fewer_than_2_samples_is_nan(self, headings):
        """Fewer than two observed headings measure no head sweep: NaN, not 0."""
        from neurospatial.behavior.vte import head_sweep_magnitude

        assert np.isnan(head_sweep_magnitude(headings))

    def test_head_sweep_wraps_angles(self):
        """Test that heading differences are properly wrapped."""
        from neurospatial.behavior.vte import head_sweep_magnitude

        # Jump from near pi to near -pi (should be small change, not 2pi)
        headings = np.array([0.9 * np.pi, -0.9 * np.pi])
        result = head_sweep_magnitude(headings)

        # The shortest path is 0.2*pi, not 1.8*pi
        expected = 0.2 * np.pi
        assert_allclose(result, expected, rtol=1e-10)


# =============================================================================
class TestHeadSweepFromPositions:
    """Test head_sweep_from_positions() function."""

    def test_straight_line_trajectory(self):
        """Test that straight trajectory has low head sweep."""
        from neurospatial.behavior.vte import head_sweep_from_positions

        # Straight line moving East (x increases, y constant)
        n_samples = 20
        times = np.linspace(0, 2, n_samples)  # 2 seconds
        positions = np.column_stack(
            [
                np.linspace(0, 100, n_samples),  # x: 0 to 100
                np.ones(n_samples) * 50,  # y: constant at 50
            ]
        )

        result = head_sweep_from_positions(times, positions, min_speed=5.0)

        # Straight line -> constant heading -> head sweep ≈ 0
        assert result < 0.1  # Allow small numerical error

    def test_zigzag_trajectory(self):
        """Test that zigzag trajectory has high head sweep."""
        from neurospatial.behavior.vte import head_sweep_from_positions

        # Zigzag: alternating East and Northeast
        n_samples = 21
        times = np.linspace(0, 2, n_samples)

        # Create zigzag positions
        x = np.linspace(0, 100, n_samples)
        y = np.zeros(n_samples)
        for i in range(n_samples):
            if i % 2 == 0:
                y[i] = 50
            else:
                y[i] = 55
        positions = np.column_stack([x, y])

        result = head_sweep_from_positions(times, positions, min_speed=0.1)

        # Zigzag should have significant head sweep
        assert result > 0.5  # At least some rotation

    def test_stationary_trajectory(self):
        """A still animal has no movement heading: NaN, without a warning."""
        from neurospatial.behavior.vte import head_sweep_from_positions

        n_samples = 20
        times = np.linspace(0, 2, n_samples)
        positions = np.full((n_samples, 2), 50.0)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = head_sweep_from_positions(times, positions, min_speed=5.0)

        assert np.isnan(result)

    def test_one_sample_is_nan(self):
        from neurospatial.behavior.vte import head_sweep_from_positions

        result = head_sweep_from_positions(
            np.array([0.0]), np.array([[1.0, 2.0]]), min_speed=1.0
        )
        assert np.isnan(result)

    def test_nan_row_is_bridged_like_a_deleted_row(self):
        """A lost frame inside max_gap is skipped as if its row were deleted."""
        from neurospatial.behavior.vte import head_sweep_from_positions

        times, positions = _zigzag(41)
        positions[20] = np.nan
        kept = np.delete(np.arange(41), 20)
        expected = head_sweep_from_positions(
            times[kept], positions[kept], min_speed=0.1
        )
        actual = head_sweep_from_positions(times, positions, min_speed=0.1)
        assert expected > 0
        assert_allclose(actual, expected, rtol=1e-12)

    def test_nan_row_beyond_max_gap_splits_runs(self):
        """A lost frame longer than max_gap: no heading change spans it."""
        from neurospatial.behavior.vte import head_sweep_from_positions

        times, positions = _zigzag(41)
        positions[20] = np.nan
        expected = head_sweep_from_positions(
            times[:20], positions[:20], min_speed=0.1
        ) + head_sweep_from_positions(times[21:], positions[21:], min_speed=0.1)
        actual = head_sweep_from_positions(
            times, positions, min_speed=0.1, max_gap=0.15
        )
        assert expected > 0
        assert_allclose(actual, expected, rtol=1e-12)

    def test_break_before_equals_separate_recordings(self):
        """Two recordings joined on one clock are analyzed separately."""
        from neurospatial.behavior.vte import head_sweep_from_positions

        times, positions = _zigzag(41)
        positions[20:] += np.array([-60.0, 30.0])  # put back elsewhere
        break_before = np.arange(41) == 20
        expected = head_sweep_from_positions(
            times[:20], positions[:20], min_speed=0.1
        ) + head_sweep_from_positions(times[20:], positions[20:], min_speed=0.1)
        actual = head_sweep_from_positions(
            times, positions, min_speed=0.1, break_before=break_before
        )
        undeclared = head_sweep_from_positions(times, positions, min_speed=0.1)
        assert actual == expected
        assert undeclared != expected

    def test_slow_run_adds_nothing(self):
        """A run with no observed heading contributes nothing, not zero turns."""
        from neurospatial.behavior.vte import head_sweep_from_positions

        times, positions = _zigzag(41)
        positions[20:] = positions[20]  # still after a 2 s break
        times = times.copy()
        times[20:] += 2.0
        moving = head_sweep_from_positions(times[:20], positions[:20], min_speed=0.1)
        both = head_sweep_from_positions(times, positions, min_speed=0.1)
        assert both == moving

    def test_repeated_timestamp_raises(self):
        from neurospatial.behavior.vte import head_sweep_from_positions

        times, positions = _zigzag(21)
        times = times.copy()
        times[5] = times[4]
        with pytest.raises(ValueError, match="strictly increasing"):
            head_sweep_from_positions(times, positions, min_speed=0.1)


# =============================================================================
# Test VTETrialResult Dataclass
# =============================================================================


class TestVTETrialResult:
    """Test VTETrialResult dataclass."""

    def test_idphi_alias(self):
        """Test that idphi property is alias for head_sweep_magnitude."""
        from neurospatial.behavior.vte import VTETrialResult

        result = VTETrialResult(
            head_sweep_magnitude=2.5,
            z_head_sweep=1.2,
            mean_speed=15.0,
            min_observed_speed=2.0,
            z_speed_inverse=0.8,
            vte_index=1.0,
            is_vte=True,
            window_start=0.0,
            window_end=1.0,
            status="ok",
        )

        assert result.idphi == result.head_sweep_magnitude

    def test_z_idphi_alias(self):
        """Test that z_idphi property is alias for z_head_sweep."""
        from neurospatial.behavior.vte import VTETrialResult

        result = VTETrialResult(
            head_sweep_magnitude=2.5,
            z_head_sweep=1.2,
            mean_speed=15.0,
            min_observed_speed=2.0,
            z_speed_inverse=0.8,
            vte_index=1.0,
            is_vte=True,
            window_start=0.0,
            window_end=1.0,
            status="ok",
        )

        assert result.z_idphi == result.z_head_sweep


def _two_trials():
    """Measured trials with head sweeps 1 and 2 rad and speeds 15 and 30 cm/s."""
    from neurospatial.behavior.vte import VTETrialResult

    return tuple(
        VTETrialResult(
            head_sweep_magnitude=head_sweep,
            z_head_sweep=None,
            mean_speed=speed,
            min_observed_speed=speed / 2,
            z_speed_inverse=None,
            vte_index=None,
            is_vte=None,
            window_start=0.0,
            window_end=1.0,
            status="ok",
        )
        for head_sweep, speed in [(1.0, 15.0), (2.0, 30.0)]
    )


class TestVTESessionResult:
    """Test VTESessionResult dataclass."""

    def test_session_statistics_are_derived_from_trials(self):
        """Mean and population std of the trials' observed metrics."""
        from neurospatial.behavior.vte import VTESessionResult

        session = VTESessionResult(trial_results=_two_trials())
        assert session.mean_head_sweep == 1.5
        assert session.std_head_sweep == 0.5
        assert session.mean_speed == 22.5
        assert session.std_speed == 7.5
        assert isinstance(session.trial_results, tuple)

    def test_mean_idphi_alias(self):
        """Test that mean_idphi property is alias for mean_head_sweep."""
        from neurospatial.behavior.vte import VTESessionResult

        session = VTESessionResult(trial_results=_two_trials())

        assert session.mean_idphi == session.mean_head_sweep == 1.5

    def test_std_idphi_alias(self):
        """Test that std_idphi property is alias for std_head_sweep."""
        from neurospatial.behavior.vte import VTESessionResult

        session = VTESessionResult(trial_results=_two_trials())

        assert session.std_idphi == session.std_head_sweep == 0.5

    def test_get_vte_trials(self):
        """Test get_vte_trials() returns only VTE trials."""
        from neurospatial.behavior.vte import VTESessionResult, VTETrialResult

        trial1 = VTETrialResult(
            head_sweep_magnitude=2.5,
            z_head_sweep=1.2,
            mean_speed=15.0,
            min_observed_speed=2.0,
            z_speed_inverse=0.8,
            vte_index=1.0,
            is_vte=True,
            window_start=0.0,
            window_end=1.0,
            status="ok",
        )
        trial2 = VTETrialResult(
            head_sweep_magnitude=0.5,
            z_head_sweep=-0.8,
            mean_speed=30.0,
            min_observed_speed=25.0,
            z_speed_inverse=-0.5,
            vte_index=-0.6,
            is_vte=False,
            window_start=1.0,
            window_end=2.0,
            status="ok",
        )
        trial3 = VTETrialResult(
            head_sweep_magnitude=3.0,
            z_head_sweep=1.5,
            mean_speed=10.0,
            min_observed_speed=1.0,
            z_speed_inverse=1.2,
            vte_index=1.3,
            is_vte=True,
            window_start=2.0,
            window_end=3.0,
            status="ok",
        )

        session = VTESessionResult(trial_results=[trial1, trial2, trial3])

        vte_trials = session.get_vte_trials()
        assert len(vte_trials) == 2
        assert all(t.is_vte for t in vte_trials)


class TestNormalizeVTEScores:
    """Test normalize_vte_scores() function."""

    def test_basic_z_scoring(self):
        """Test that z-scoring works correctly."""
        from neurospatial.behavior.vte import normalize_vte_scores

        head_sweeps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        speeds = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        z_head_sweeps, z_speed_inverse = normalize_vte_scores(head_sweeps, speeds)

        # Z-scores should have mean 0 and std 1
        assert_allclose(np.mean(z_head_sweeps), 0.0, atol=1e-10)
        assert_allclose(np.std(z_head_sweeps), 1.0, atol=1e-10)

        # Speed inverse should also be z-scored (higher speed -> lower z)
        assert_allclose(np.mean(z_speed_inverse), 0.0, atol=1e-10)
        assert_allclose(np.std(z_speed_inverse), 1.0, atol=1e-10)

    def test_speed_inverse_direction(self):
        """Test that slower speeds give higher z_speed_inverse."""
        from neurospatial.behavior.vte import normalize_vte_scores

        head_sweeps = np.array([1.0, 1.0, 1.0])
        speeds = np.array([10.0, 20.0, 30.0])

        with pytest.warns(UserWarning, match="No variation in head sweep"):
            _, z_speed_inverse = normalize_vte_scores(head_sweeps, speeds)

        # Slowest (10) should have highest z_speed_inverse
        assert z_speed_inverse[0] > z_speed_inverse[1] > z_speed_inverse[2]

    def test_length_mismatch_raises(self):
        """Test that mismatched array lengths raise ValueError."""
        from neurospatial.behavior.vte import normalize_vte_scores

        head_sweeps = np.array([1.0, 2.0, 3.0])
        speeds = np.array([10.0, 20.0])

        with pytest.raises(ValueError, match="same length"):
            normalize_vte_scores(head_sweeps, speeds)

    def test_zero_std_head_sweep_warns(self):
        """Test that zero std in head sweep triggers warning."""
        from neurospatial.behavior.vte import normalize_vte_scores

        # All identical head sweeps
        head_sweeps = np.array([2.0, 2.0, 2.0])
        speeds = np.array([10.0, 20.0, 30.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            z_head_sweeps, _ = normalize_vte_scores(head_sweeps, speeds)

            # Should warn about zero std
            assert len(w) == 1
            assert "variation" in str(w[0].message).lower()

            # Should return zeros, not NaN
            assert_allclose(z_head_sweeps, [0.0, 0.0, 0.0])

    def test_zero_std_speed_warns(self):
        """Test that zero std in speed triggers warning."""
        from neurospatial.behavior.vte import normalize_vte_scores

        # All identical speeds
        head_sweeps = np.array([1.0, 2.0, 3.0])
        speeds = np.array([20.0, 20.0, 20.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, z_speed_inverse = normalize_vte_scores(head_sweeps, speeds)

            # Should warn about zero std
            assert len(w) == 1
            assert "variation" in str(w[0].message).lower()

            # Should return zeros, not NaN
            assert_allclose(z_speed_inverse, [0.0, 0.0, 0.0])

    def test_missing_entries_are_left_out(self):
        """Mean and std use observed entries; a missing entry has a NaN z."""
        from neurospatial.behavior.vte import normalize_vte_scores

        z_head_sweeps, z_speed_inverse = normalize_vte_scores(
            [1.0, np.nan, 3.0], [10.0, 20.0, np.nan]
        )
        np.testing.assert_array_equal(z_head_sweeps, [-1.0, np.nan, 1.0])
        np.testing.assert_array_equal(z_speed_inverse, [1.0, -1.0, np.nan])

    def test_observed_entries_match_observed_only_input(self):
        from neurospatial.behavior.vte import normalize_vte_scores

        rng = np.random.default_rng(3)
        head_sweeps = rng.gamma(2.0, 1.0, 12)
        speeds = rng.gamma(5.0, 3.0, 12)
        missing = np.zeros(12, dtype=bool)
        missing[[2, 7]] = True
        expected_hs, _ = normalize_vte_scores(head_sweeps[~missing], speeds[~missing])
        _, expected_sp = normalize_vte_scores(head_sweeps, speeds)
        z_hs, z_sp = normalize_vte_scores(
            np.where(missing, np.nan, head_sweeps), speeds
        )
        np.testing.assert_array_equal(z_hs[~missing], expected_hs)
        assert np.isnan(z_hs[missing]).all()
        np.testing.assert_array_equal(z_sp, expected_sp)

    def test_no_observed_entries_is_all_nan_without_warning(self):
        from neurospatial.behavior.vte import normalize_vte_scores

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            z_hs, z_sp = normalize_vte_scores(
                np.full(3, np.nan), np.array([10.0, 20.0, 30.0])
            )
        assert np.isnan(z_hs).all()
        assert np.isfinite(z_sp).all()


# =============================================================================
# Test compute_vte_index()
# =============================================================================


class TestComputeVTEIndex:
    """Test compute_vte_index() function."""

    def test_equal_weighting(self):
        """Test VTE index with equal weighting (alpha=0.5)."""
        from neurospatial.behavior.vte import compute_vte_index

        z_head_sweep = 1.0
        z_speed_inv = 1.0

        result = compute_vte_index(z_head_sweep, z_speed_inv, alpha=0.5)

        # 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        assert result == 1.0

    def test_head_sweep_only(self):
        """Test VTE index with alpha=1.0 (head sweep only)."""
        from neurospatial.behavior.vte import compute_vte_index

        z_head_sweep = 2.0
        z_speed_inv = 0.5

        result = compute_vte_index(z_head_sweep, z_speed_inv, alpha=1.0)

        # 1.0 * 2.0 + 0.0 * 0.5 = 2.0
        assert result == 2.0

    def test_speed_only(self):
        """Test VTE index with alpha=0.0 (speed only)."""
        from neurospatial.behavior.vte import compute_vte_index

        z_head_sweep = 2.0
        z_speed_inv = 0.5

        result = compute_vte_index(z_head_sweep, z_speed_inv, alpha=0.0)

        # 0.0 * 2.0 + 1.0 * 0.5 = 0.5
        assert result == 0.5


# =============================================================================
# Test classify_vte()
# =============================================================================


class TestClassifyVTE:
    """Test classify_vte() function."""

    def test_above_threshold(self):
        """Test that index above threshold returns True."""
        from neurospatial.behavior.vte import classify_vte

        assert classify_vte(1.0, threshold=0.5) is True
        assert classify_vte(0.6, threshold=0.5) is True

    def test_below_threshold(self):
        """Test that index below threshold returns False."""
        from neurospatial.behavior.vte import classify_vte

        assert classify_vte(0.3, threshold=0.5) is False
        assert classify_vte(-0.5, threshold=0.5) is False

    def test_at_threshold(self):
        """Test that index at threshold returns False (> not >=)."""
        from neurospatial.behavior.vte import classify_vte

        assert classify_vte(0.5, threshold=0.5) is False


# =============================================================================
# Test compute_vte_trial()
# =============================================================================


class TestComputeVTETrial:
    """Test compute_vte_trial() function."""

    def test_single_trial_no_z_scores(self):
        """Test that single trial analysis has None for z-scores."""
        from neurospatial.behavior.vte import compute_vte_trial

        # Create a simple trajectory
        n_samples = 30
        times = np.linspace(0, 3, n_samples)
        positions = np.column_stack(
            [
                np.linspace(0, 50, n_samples),
                np.ones(n_samples) * 50,
            ]
        )

        result = compute_vte_trial(
            positions=positions,
            times=times,
            entry_time=2.0,
            window_duration=1.0,
            min_speed=1.0,
        )

        # Single trial: z-scores should be None
        assert result.z_head_sweep is None
        assert result.z_speed_inverse is None
        assert result.vte_index is None
        assert result.is_vte is None

        # But raw metrics should be computed
        assert result.head_sweep_magnitude >= 0
        assert result.mean_speed >= 0
        assert result.window_start == 1.0  # entry_time - window_duration
        assert result.window_end == 2.0  # entry_time

    def test_basic_metrics_computed(self):
        """Test that basic metrics are computed correctly."""
        from neurospatial.behavior.vte import compute_vte_trial

        # Create a trajectory with known properties
        n_samples = 50
        times = np.linspace(0, 5, n_samples)

        # Straight line trajectory (low head sweep)
        positions = np.column_stack(
            [
                np.linspace(0, 100, n_samples),
                np.ones(n_samples) * 50,
            ]
        )

        result = compute_vte_trial(
            positions=positions,
            times=times,
            entry_time=3.0,
            window_duration=1.0,
            min_speed=1.0,
        )

        # Straight line should have low head sweep
        assert result.head_sweep_magnitude < 0.5

        # Mean speed should be > 0 (we're moving)
        assert result.mean_speed > 0

    def test_one_sample_window_is_nan(self):
        """A window with one sample has no speed and no heading: NaN, not 0."""
        from neurospatial.behavior.vte import compute_vte_trial

        times, positions = _zigzag(30)
        result = compute_vte_trial(
            times, positions, entry_time=1.05, window_duration=0.1, min_speed=0.1
        )
        assert np.isnan(result.mean_speed)
        assert np.isnan(result.min_observed_speed)
        assert np.isnan(result.head_sweep_magnitude)

    def test_still_window_has_zero_speed_and_no_head_sweep(self):
        """A still animal moves at an observed 0 cm/s but faces no heading."""
        from neurospatial.behavior.vte import compute_vte_trial

        times = np.arange(30) / 10
        positions = np.full((30, 2), 50.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = compute_vte_trial(
                times, positions, entry_time=2.0, window_duration=1.0, min_speed=1.0
            )
        assert result.mean_speed == 0.0
        assert result.min_observed_speed == 0.0
        assert np.isnan(result.head_sweep_magnitude)

    def test_break_before_equals_separate_recording(self):
        """The window never reaches back across a declared recording break."""
        from neurospatial.behavior.vte import compute_vte_trial

        times, positions = _zigzag(41)
        positions[20:] += np.array([-60.0, 30.0])
        break_before = np.arange(41) == 20
        kwargs = {"entry_time": 3.5, "window_duration": 3.0, "min_speed": 0.1}
        declared = compute_vte_trial(
            times, positions, break_before=break_before, **kwargs
        )
        alone = compute_vte_trial(times[20:], positions[20:], **kwargs)
        assert declared == alone
        assert compute_vte_trial(times, positions, **kwargs) != alone

    def test_nan_row_in_window_is_missing(self):
        """A lost frame inside the window splits it; no velocity spans it."""
        from neurospatial.behavior.vte import (
            compute_vte_trial,
            head_sweep_from_positions,
        )

        times, positions = _zigzag(41)
        positions[25] = np.nan
        result = compute_vte_trial(
            times, positions, entry_time=3.5, window_duration=2.0, min_speed=0.1
        )
        window = slice(15, 35)
        assert result.head_sweep_magnitude == head_sweep_from_positions(
            times[window], positions[window], min_speed=0.1
        )
        assert np.isfinite(result.mean_speed)


# =============================================================================
# Test compute_vte_session()
# =============================================================================


class TestComputeVTESession:
    """Test compute_vte_session() function."""

    def test_high_head_sweep_low_speed_is_vte(self, t_maze_environment):
        """Test that high head sweep + low speed is classified as VTE."""
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.behavior.vte import compute_vte_session

        env = t_maze_environment

        # Create trials with different behaviors
        # Trial 1: High head sweep, low speed (VTE-like)
        # Trial 2: Low head sweep, high speed (non-VTE)

        # Build continuous position data for entire session
        n_per_trial = 60
        sample_rate = 30  # 30 Hz
        dt = 1.0 / sample_rate

        # Trial 1: Slow, head-scanning trajectory approaching center
        t1_times = np.arange(n_per_trial) * dt
        t1_x = np.linspace(50, 50, n_per_trial)  # Stay at x=50
        t1_y = np.linspace(30, 56, n_per_trial)  # Move up into the center

        # Add zigzag to create head scanning
        t1_x = t1_x + 5 * np.sin(np.linspace(0, 8 * np.pi, n_per_trial))

        # Trial 2: Fast, straight trajectory approaching center
        t2_times = t1_times[-1] + dt + np.arange(n_per_trial) * dt
        t2_x = np.linspace(50, 50, n_per_trial)
        t2_y = np.linspace(30, 56, n_per_trial)

        # Combine into session
        times = np.concatenate([t1_times, t2_times])
        positions = np.column_stack(
            [np.concatenate([t1_x, t2_x]), np.concatenate([t1_y, t2_y])]
        )

        # Define trials
        trials = [
            Trial(
                start_time=t1_times[0],
                end_time=t1_times[-1],
                start_region="start",
                end_region="center",
                success=True,
            ),
            Trial(
                start_time=t2_times[0],
                end_time=t2_times[-1],
                start_region="start",
                end_region="center",
                success=True,
            ),
        ]

        result = compute_vte_session(
            positions=positions,
            times=times,
            trials=trials,
            decision_region="center",
            env=env,
            window_duration=0.5,
            min_speed=1.0,
            alpha=0.5,
            vte_threshold=0.0,  # Low threshold to ensure some VTE classification
        )

        # Both trials reach the center
        assert len(result.trial_results) == 2
        scanning, straight = result.trial_results
        assert scanning.head_sweep_magnitude > straight.head_sweep_magnitude

        # Session statistics should be computed
        assert result.mean_head_sweep >= 0
        assert result.std_head_sweep >= 0
        assert result.mean_speed >= 0

    def test_no_valid_trials_returns_empty_result(self, t_maze_environment):
        """Test that no valid trials returns empty session result."""
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.behavior.vte import compute_vte_session

        env = t_maze_environment

        # Create trials that never reach decision region
        times = np.linspace(0, 2, 60)
        positions = np.column_stack(
            [
                np.ones(60) * 50,
                np.linspace(0, 20, 60),  # Never reaches center at y=55
            ]
        )

        trials = [
            Trial(
                start_time=0.0,
                end_time=2.0,
                start_region="start",
                end_region=None,
                success=False,
            ),
        ]

        # A trial that never enters the decision region must be skipped *and*
        # warned about (with its timing), not silently dropped.
        with pytest.warns(UserWarning, match=r"never enters decision region"):
            result = compute_vte_session(
                positions=positions,
                times=times,
                trials=trials,
                decision_region="center",
                env=env,
                window_duration=0.5,
                min_speed=1.0,
            )

        # The trial keeps its place, but nothing was measured, so no
        # statistic is 0.
        (trial,) = result.trial_results
        assert trial.status == "no_entry"
        assert np.isnan(trial.head_sweep_magnitude)
        assert trial.is_vte is None
        assert result.n_vte_trials == 0
        for value in (
            result.vte_fraction,
            result.mean_head_sweep,
            result.std_head_sweep,
            result.mean_speed,
            result.std_speed,
        ):
            assert np.isnan(value)

    def test_skipped_trial_warning_includes_timing(self, t_maze_environment):
        """The skip warning names the trial's start/end times."""
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.behavior.vte import compute_vte_session

        env = t_maze_environment

        times = np.linspace(0, 2, 60)
        positions = np.column_stack(
            [np.ones(60) * 50, np.linspace(0, 20, 60)]  # never reaches center
        )
        trials = [
            Trial(
                start_time=0.0,
                end_time=2.0,
                start_region="start",
                end_region=None,
                success=False,
            ),
        ]

        with pytest.warns(UserWarning) as record:
            compute_vte_session(
                positions=positions,
                times=times,
                trials=trials,
                decision_region="center",
                env=env,
                window_duration=0.5,
                min_speed=1.0,
            )

        messages = [str(w.message) for w in record]
        assert any("0.000" in m and "2.000" in m for m in messages), messages

    def test_vte_classification_consistency(self, t_maze_environment):
        """Test that VTE classification is consistent with metrics."""
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.behavior.vte import compute_vte_session

        env = t_maze_environment

        # Create multiple trials with known behavior
        n_per_trial = 60
        sample_rate = 30
        dt = 1.0 / sample_rate
        n_trials = 5

        all_times = []
        all_positions_x = []
        all_positions_y = []
        trials = []

        for i in range(n_trials):
            t_start = i * 3.0
            trial_times = t_start + np.arange(n_per_trial) * dt

            # Vary head sweep: even trials have more scanning
            if i % 2 == 0:
                # High head sweep (zigzag)
                x = 50 + 8 * np.sin(np.linspace(0, 10 * np.pi, n_per_trial))
            else:
                # Low head sweep (straight)
                x = np.ones(n_per_trial) * 50

            y = np.linspace(30, 56, n_per_trial)

            all_times.append(trial_times)
            all_positions_x.append(x)
            all_positions_y.append(y)

            trials.append(
                Trial(
                    start_time=trial_times[0],
                    end_time=trial_times[-1],
                    start_region="start",
                    end_region="center",
                    success=True,
                )
            )

        times = np.concatenate(all_times)
        positions = np.column_stack(
            [np.concatenate(all_positions_x), np.concatenate(all_positions_y)]
        )

        result = compute_vte_session(
            positions=positions,
            times=times,
            trials=trials,
            decision_region="center",
            env=env,
            window_duration=0.5,
            min_speed=0.5,
            alpha=0.5,
            vte_threshold=0.0,
        )

        # Check consistency: n_vte_trials matches is_vte count
        assert result.n_vte_trials == sum(1 for t in result.trial_results if t.is_vte)

        # Check consistency: vte_fraction is over the classified trials
        n_classified = sum(t.is_vte is not None for t in result.trial_results)
        assert n_classified == len(result.trial_results) == 5
        assert_allclose(result.vte_fraction, result.n_vte_trials / n_classified)


def test_compute_vte_session_env_first(t_maze_environment):
    """compute_vte_session is env-first and runs with positional env."""
    import inspect

    from neurospatial.behavior.segmentation import Trial
    from neurospatial.behavior.vte import compute_vte_session

    env = t_maze_environment

    # Signature: env must be the first positional parameter.
    params = list(inspect.signature(compute_vte_session).parameters)
    assert params[0] == "env"
    assert params[1] == "times"
    assert params[2] == "positions"

    # Build a slow head-scanning trajectory approaching the center region.
    n = 60
    dt = 1.0 / 30.0
    times = np.arange(n) * dt
    x = 50.0 + 5.0 * np.sin(np.linspace(0, 8 * np.pi, n))
    y = np.linspace(30.0, 56.0, n)  # into the center bin (y >= 52.5)
    positions = np.column_stack([x, y])

    trials = [
        Trial(
            start_time=times[0],
            end_time=times[-1],
            start_region="start",
            end_region="center",
            success=True,
        )
    ]

    # Positional (env, times, positions) call analyzes the trial. One trial
    # has no spread to z-score against, which warns.
    with pytest.warns(UserWarning, match="No variation"):
        result = compute_vte_session(
            env,
            times,
            positions,
            decision_region="center",
            trials=trials,
            window_duration=0.5,
            min_speed=1.0,
        )
    assert len(result.trial_results) == 1


@pytest.fixture
def zig_zag_then_run(simple_2d_environment):
    """An east-west zig-zag before 5 s, then a straight eastward run.

    The run starts at 5.0 s and enters the "decision" region at 5.267 s.
    Returns ``(env, times, positions)``.
    """
    from shapely.geometry import box

    env = simple_2d_environment
    region_builder = Regions(env.regions.values())
    region_builder.add("decision", polygon=box(46, 40, 60, 60))
    env = env.with_regions(region_builder)
    times = np.arange(0, 10, 1 / 30)
    phase = (times * 3.0 + 0.5) % 1.0
    zig_zag = 34 + 6 * (1 - np.abs(2 * phase - 1))  # x in [34, 40], 3 Hz
    run = np.minimum(40 + 30 * (times - 5.0), 90.0)  # east at 30 cm/s
    x = np.where(times < 5.0, zig_zag, run)
    positions = np.column_stack([x, np.full_like(times, 50.0)])
    return env, times, positions


def test_session_window_clamped_to_trial_start(zig_zag_then_run):
    """The pre-decision window never reaches back before the trial start.

    A 1 s window before the 5.267 s entry would otherwise cover the zig-zag
    that precedes the trial.
    """
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.behavior.vte import compute_vte_session

    env, times, positions = zig_zag_then_run

    with warnings.catch_warnings():
        # One trial: z-scores are undefined and warn; not under test here.
        warnings.simplefilter("ignore", UserWarning)
        result = compute_vte_session(
            env,
            times,
            positions,
            decision_region="decision",
            trials=[Trial(5.0, 9.9, "start", "decision", True)],
            window_duration=1.0,
            min_speed=5.0,
        )

    (trial,) = result.trial_results
    assert trial.window_start == 5.0
    assert_allclose(trial.window_end, 5.0 + 8 / 30, rtol=1e-12)
    assert abs(trial.head_sweep_magnitude) < 1e-12


def test_session_warns_when_trial_window_too_short(zig_zag_then_run):
    """A trial that enters the region within 2 samples of its start is skipped
    with a warning, not silently.
    """
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.behavior.vte import compute_vte_session

    env, times, positions = zig_zag_then_run

    with pytest.warns(UserWarning, match=r"only 2 observed sample\(s\)"):
        result = compute_vte_session(
            env,
            times,
            positions,
            decision_region="decision",
            trials=[Trial(5.2, 9.9, "start", "decision", True)],
            window_duration=1.0,
            min_speed=5.0,
        )

    (trial,) = result.trial_results
    assert trial.status == "too_few_samples"
    assert trial.window_start == 5.2
    assert_allclose(trial.window_end, 5.0 + 8 / 30, rtol=1e-12)
    assert np.isnan(trial.mean_speed)
    assert trial.is_vte is None


def _t_maze_session(amplitudes, *, creep=None):
    """Approaches up the T-maze stem into "center", one trial per entry.

    Each trial lasts 2 s at 30 Hz, starts 3 s after the previous one and
    rises from y = 30 to 56 cm (13 cm/s) with an east-west zig-zag of the
    given amplitude. With ``creep=i``, trial ``i`` instead creeps up from
    y = 50 cm at 1.5 cm/s, too slow for a movement heading at 5 cm/s.
    """
    from neurospatial.behavior.segmentation import Trial

    n = 60
    times, xs, ys, trials = [], [], [], []
    for i, amplitude in enumerate(amplitudes):
        t = 3.0 * i + np.arange(n) / 30
        if i == creep:
            x = np.full(n, 50.0)
            y = 50.0 + 1.5 * (t - t[0])
        else:
            x = 50.0 + amplitude * np.sin(np.linspace(0, 10 * np.pi, n))
            y = np.linspace(30.0, 56.0, n)
        times.append(t)
        xs.append(x)
        ys.append(y)
        trials.append(Trial(t[0], t[-1], "start", "center", True))
    positions = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
    return np.concatenate(times), positions, trials


def _first_center_sample(env, times, positions, *, after):
    in_center = np.isin(env.bin_at(positions), env.bins_in_region("center"))
    return int(np.flatnonzero(in_center & (times >= after))[0])


_SESSION_KWARGS = {
    "decision_region": "center",
    "window_duration": 0.5,
    "min_speed": 5.0,
    "vte_threshold": 0.0,
}


def test_session_trial_without_heading_is_unclassified(t_maze_environment):
    """A pause with no observed heading is not scored as "no head sweep".

    Its IdPhi is NaN, so it leaves the head-sweep z-scores of the other
    trials unchanged and is not classified; its slow speed is still measured.
    """
    from neurospatial.behavior.vte import compute_vte_session

    env = t_maze_environment
    times, positions, trials = _t_maze_session([2.0, 4.0, 0.0, 8.0], creep=2)
    result = compute_vte_session(
        env, times, positions, trials=trials, **_SESSION_KWARGS
    )
    moving = [0, 1, 3]
    without = compute_vte_session(
        env,
        times,
        positions,
        trials=[trials[i] for i in moving],
        **_SESSION_KWARGS,
    )

    assert len(result.trial_results) == 4
    paused = result.trial_results[2]
    assert np.isnan(paused.head_sweep_magnitude)
    assert np.isnan(paused.z_head_sweep)
    assert np.isnan(paused.vte_index)
    assert paused.is_vte is None
    assert_allclose(paused.mean_speed, 1.5, rtol=1e-9)
    z_speed = [t.z_speed_inverse for t in result.trial_results]
    assert np.argmax(z_speed) == 2

    for i, other in zip(moving, without.trial_results, strict=True):
        assert result.trial_results[i].head_sweep_magnitude > 0
        assert result.trial_results[i].z_head_sweep == other.z_head_sweep
        assert result.trial_results[i].is_vte is not None
    assert result.mean_head_sweep == without.mean_head_sweep
    assert result.std_head_sweep == without.std_head_sweep
    assert result.n_vte_trials == sum(t.is_vte is True for t in result.trial_results)
    assert result.vte_fraction == result.n_vte_trials / 3


def test_session_skips_trial_with_too_few_observed_samples(t_maze_environment):
    """Missing rows in the window do not count toward the 3-sample minimum."""
    from neurospatial.behavior.vte import compute_vte_session

    env = t_maze_environment
    times, positions, trials = _t_maze_session([0.0, 4.0, 8.0])
    entry = _first_center_sample(env, times, positions, after=3.0)
    positions[entry - 15 : entry - 2] = np.nan  # keep 2 of the 15 window rows

    with pytest.warns(UserWarning, match=r"Trial 1 .*only 2 observed sample\(s\)"):
        result = compute_vte_session(
            env, times, positions, trials=trials, **_SESSION_KWARGS
        )
    statuses = [t.status for t in result.trial_results]
    assert statuses == ["ok", "too_few_samples", "ok"]
    assert result.trial_results[1].is_vte is None


def test_session_break_before_ends_the_window(t_maze_environment):
    """A declared break in a window acts like the trial starting there."""
    from dataclasses import replace

    from neurospatial.behavior.segmentation import Trial
    from neurospatial.behavior.vte import compute_vte_session

    env = t_maze_environment
    times, positions, trials = _t_maze_session([0.0, 4.0, 8.0])
    entry = _first_center_sample(env, times, positions, after=3.0)
    first = entry - 6  # break 6 samples (0.2 s) before entry in trial 1
    break_before = np.arange(times.size) == first
    declared = compute_vte_session(
        env,
        times,
        positions,
        trials=trials,
        break_before=break_before,
        **_SESSION_KWARGS,
    )
    keep = (times < trials[1].start_time) | (times >= times[first])
    shortened = [
        trials[0],
        Trial(times[first], trials[1].end_time, "start", "center", True),
        trials[2],
    ]
    alone = compute_vte_session(
        env, times[keep], positions[keep], trials=shortened, **_SESSION_KWARGS
    )
    undeclared = compute_vte_session(
        env, times, positions, trials=trials, **_SESSION_KWARGS
    )

    for a, b in zip(declared.trial_results, alone.trial_results, strict=True):
        assert a == replace(b, window_start=a.window_start)
    assert declared.trial_results[1] != undeclared.trial_results[1]


@pytest.mark.parametrize("n_samples", [1, 30])
@pytest.mark.parametrize(
    "name", ["head_sweep_from_positions", "compute_vte_trial", "compute_vte_session"]
)
def test_one_dimensional_positions_raise(t_maze_environment, name, n_samples):
    """A heading needs x and y, so 1-D positions raise, naming the passed shape."""
    from neurospatial.behavior import vte

    times = np.arange(n_samples) / 10
    calls = {
        "head_sweep_from_positions": lambda p: vte.head_sweep_from_positions(
            times, p, min_speed=1.0
        ),
        "compute_vte_trial": lambda p: vte.compute_vte_trial(
            times, p, entry_time=2.0, window_duration=1.0, min_speed=1.0
        ),
        "compute_vte_session": lambda p: vte.compute_vte_session(
            t_maze_environment, times, p, trials=[], **_SESSION_KWARGS
        ),
    }
    match = rf"got shape \({n_samples},\)(.|\n)*heading .*needs both x and y"
    with pytest.raises(ValueError, match=match):
        calls[name](10 * times)


def test_session_summary_counts_classified_trials(t_maze_environment):
    """The text and the summary say which trials the VTE fraction is over."""
    from neurospatial.behavior.vte import compute_vte_session

    times, positions, trials = _t_maze_session([2.0, 4.0, 0.0, 8.0], creep=2)
    result = compute_vte_session(
        t_maze_environment, times, positions, trials=trials, **_SESSION_KWARGS
    )
    summary = result.summary()
    assert summary["n_trials"] == 4
    assert summary["n_classified"] == result.n_classified == 3
    assert summary["n_vte_trials"] == result.n_vte_trials
    first_line = str(result).splitlines()[0]
    assert first_line == (
        f"VTE session: {result.n_vte_trials}/3 classified trials VTE "
        f"({result.vte_fraction:.1%}), 1 unclassified"
    )


def test_session_text_without_classified_trials():
    from neurospatial.behavior.vte import VTESessionResult

    empty = VTESessionResult(trial_results=())
    text = str(empty)
    assert "nan%" not in text
    assert text.splitlines()[0] == "VTE session: no trial could be classified"
    assert empty.summary()["n_classified"] == 0


def test_no_variation_warning_points_at_the_call_site(t_maze_environment):
    """The zero-spread warning names the scientist's line, not library code."""
    from neurospatial.behavior.vte import compute_vte_session, normalize_vte_scores

    with pytest.warns(UserWarning, match="No variation in head sweep") as record:
        normalize_vte_scores(np.full(3, 2.0), np.array([10.0, 20.0, 30.0]))
    assert [w.filename for w in record] == [__file__]

    times, positions, trials = _t_maze_session([0.0, 0.0])
    with pytest.warns(UserWarning, match="No variation") as record:
        compute_vte_session(
            t_maze_environment, times, positions, trials=trials, **_SESSION_KWARGS
        )
    assert record
    assert {w.filename for w in record} == {__file__}


def test_session_entry_is_searched_inside_epochs(t_maze_environment):
    """Epochs (4, 10) s skip the 2.8 s entry and use the return at 7.9 s."""
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.behavior.vte import compute_vte_session

    # Up the stem into the center bin at 2.8 s, east to the right goal, then
    # west 2 cm per sample from x = 90 cm back into the center bin at 7.9 s.
    times = np.arange(101) / 10
    x = np.r_[np.full(30, 50.0), np.linspace(50, 90, 30), 90 - 2.0 * np.arange(41)]
    y = np.r_[np.linspace(5, 55, 30), np.full(71, 55.0)]
    positions = np.column_stack([x, y])
    trials = [
        Trial(
            start_time=0.0,
            end_time=10.0,
            start_region="start",
            end_region="left",
            success=True,
        )
    ]
    kwargs = {"decision_region": "center", "window_duration": 1.0, "min_speed": 5.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # one trial: no z-score spread
        whole = compute_vte_session(
            t_maze_environment, times, positions, trials=trials, **kwargs
        )
        late = compute_vte_session(
            t_maze_environment,
            times,
            positions,
            trials=trials,
            epochs=(4.0, 10.0),
            **kwargs,
        )
    assert_allclose(whole.trial_results[0].window_end, 2.8, rtol=1e-12)
    assert_allclose(late.trial_results[0].window_end, 7.9, rtol=1e-12)
    assert_allclose(late.trial_results[0].mean_speed, 20.0, rtol=1e-9)


def test_session_keeps_one_result_per_trial_in_input_order(t_maze_environment):
    """Unmeasured trials keep their place, with NaN metrics and a status."""
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.behavior.vte import compute_vte_session

    times, positions, trials = _t_maze_session([2.0, 4.0, 8.0])
    # Trial 1 never reaches the center: it stays low on the stem.
    in_trial_1 = (times >= trials[1].start_time) & (times <= trials[1].end_time)
    positions[in_trial_1, 1] = np.linspace(10.0, 30.0, in_trial_1.sum())
    # A trial on a millisecond clock has no samples on the seconds clock.
    ms_trial = Trial(
        start_time=1000.0,
        end_time=2000.0,
        start_region="start",
        end_region="center",
        success=True,
    )
    trials = [trials[0], ms_trial, trials[1], trials[2]]
    with pytest.warns(UserWarning) as record:
        result = compute_vte_session(
            t_maze_environment, times, positions, trials=trials, **_SESSION_KWARGS
        )
    messages = [str(w.message) for w in record]
    assert any("Trial 1 " in m and "no samples" in m and "clock" in m for m in messages)
    assert any("Trial 2 " in m and "never enters" in m for m in messages)

    assert [t.status for t in result.trial_results] == [
        "ok",
        "no_samples",
        "no_entry",
        "ok",
    ]
    for unmeasured in result.trial_results[1:3]:
        assert np.isnan(unmeasured.head_sweep_magnitude)
        assert np.isnan(unmeasured.window_end)
        assert np.isnan(unmeasured.vte_index)
        assert unmeasured.is_vte is None
    measured = [result.trial_results[i] for i in (0, 3)]
    assert measured[0].window_end < measured[1].window_end
    assert measured[0].head_sweep_magnitude < measured[1].head_sweep_magnitude
    assert result.n_classified == 2

    table = result.trials_table()
    assert table.index.name == "trial"
    assert table.index.tolist() == [0, 1, 2, 3]
    assert table["status"].tolist() == ["ok", "no_samples", "no_entry", "ok"]
    assert list(table.columns)[:3] == ["window_start", "window_end", "status"]
    assert list(table.columns)[-1] == "is_vte"


def test_trial_result_validates_status_and_window():
    from neurospatial.behavior.vte import VTETrialResult

    fields = {
        "head_sweep_magnitude": 1.0,
        "z_head_sweep": None,
        "mean_speed": 10.0,
        "min_observed_speed": 5.0,
        "z_speed_inverse": None,
        "vte_index": None,
        "is_vte": None,
        "window_start": 0.0,
        "window_end": 1.0,
        "status": "ok",
    }
    with pytest.raises(ValueError, match="status must be one of"):
        VTETrialResult(**{**fields, "status": "skipped"})
    with pytest.raises(ValueError, match="window_start"):
        VTETrialResult(**{**fields, "window_start": 2.0})
    with pytest.raises(ValueError, match="cannot be classified"):
        VTETrialResult(**{**fields, "status": "no_entry", "is_vte": False})
    plain = VTETrialResult(**{**fields, "mean_speed": np.float64(10.0)})
    assert type(plain.mean_speed) is float
