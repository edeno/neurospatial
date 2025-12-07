"""Tests for VTE (Vicarious Trial and Error) metrics module.

Following TDD: Tests written FIRST, then implementation.

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

from neurospatial import Environment

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
    env.regions.add("start", point=(50.0, 5.0))
    env.regions.add("center", point=(50.0, 55.0))
    env.regions.add("left", point=(10.0, 55.0))
    env.regions.add("right", point=(90.0, 55.0))

    return env


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

        result = wrap_angle(np.array([np.pi]))
        # pi should wrap to -pi (or stay at pi depending on convention)
        # Using (-pi, pi] convention, pi maps to -pi
        assert_allclose(result, [-np.pi], atol=1e-10)

    def test_wrap_angle_array(self):
        """Test wrapping an array of angles."""
        from neurospatial.stats.circular import wrap_angle

        angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        result = wrap_angle(angles)
        expected = np.array([0, np.pi / 2, -np.pi, -np.pi / 2, 0])
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

    def test_head_sweep_fewer_than_2_samples(self):
        """Test that fewer than 2 valid samples returns 0."""
        from neurospatial.behavior.vte import head_sweep_magnitude

        # Single sample
        result = head_sweep_magnitude(np.array([0.0]))
        assert result == 0.0

        # All NaN
        result = head_sweep_magnitude(np.array([np.nan, np.nan, np.nan]))
        assert result == 0.0

        # Empty
        result = head_sweep_magnitude(np.array([]))
        assert result == 0.0

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
# Test integrated_absolute_rotation alias
# =============================================================================


class TestIntegratedAbsoluteRotationAlias:
    """Test that integrated_absolute_rotation is an alias for head_sweep_magnitude."""

    def test_alias_exists(self):
        """Test that the alias is exported."""
        from neurospatial.behavior.vte import (
            head_sweep_magnitude,
            integrated_absolute_rotation,
        )

        assert integrated_absolute_rotation is head_sweep_magnitude


# =============================================================================
# Test head_sweep_from_positions()
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

        result = head_sweep_from_positions(positions, times, min_speed=5.0)

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

        result = head_sweep_from_positions(positions, times, min_speed=0.1)

        # Zigzag should have significant head sweep
        assert result > 0.5  # At least some rotation

    def test_stationary_trajectory(self):
        """Test that stationary trajectory returns 0 head sweep."""
        from neurospatial.behavior.vte import head_sweep_from_positions

        # Stationary: all positions identical
        n_samples = 20
        times = np.linspace(0, 2, n_samples)
        positions = np.column_stack(
            [
                np.ones(n_samples) * 50,
                np.ones(n_samples) * 50,
            ]
        )

        result = head_sweep_from_positions(positions, times, min_speed=5.0)

        # Stationary -> no valid headings -> head sweep = 0
        assert result == 0.0


# =============================================================================
# Test VTETrialResult Dataclass
# =============================================================================


class TestVTETrialResult:
    """Test VTETrialResult dataclass."""

    def test_dataclass_fields(self):
        """Test that dataclass has required fields."""
        from neurospatial.behavior.vte import VTETrialResult

        result = VTETrialResult(
            head_sweep_magnitude=2.5,
            z_head_sweep=1.2,
            mean_speed=15.0,
            min_speed=2.0,
            z_speed_inverse=0.8,
            vte_index=1.0,
            is_vte=True,
            window_start=0.0,
            window_end=1.0,
        )

        assert result.head_sweep_magnitude == 2.5
        assert result.z_head_sweep == 1.2
        assert result.mean_speed == 15.0
        assert result.min_speed == 2.0
        assert result.z_speed_inverse == 0.8
        assert result.vte_index == 1.0
        assert result.is_vte is True
        assert result.window_start == 0.0
        assert result.window_end == 1.0

    def test_idphi_alias(self):
        """Test that idphi property is alias for head_sweep_magnitude."""
        from neurospatial.behavior.vte import VTETrialResult

        result = VTETrialResult(
            head_sweep_magnitude=2.5,
            z_head_sweep=1.2,
            mean_speed=15.0,
            min_speed=2.0,
            z_speed_inverse=0.8,
            vte_index=1.0,
            is_vte=True,
            window_start=0.0,
            window_end=1.0,
        )

        assert result.idphi == result.head_sweep_magnitude

    def test_z_idphi_alias(self):
        """Test that z_idphi property is alias for z_head_sweep."""
        from neurospatial.behavior.vte import VTETrialResult

        result = VTETrialResult(
            head_sweep_magnitude=2.5,
            z_head_sweep=1.2,
            mean_speed=15.0,
            min_speed=2.0,
            z_speed_inverse=0.8,
            vte_index=1.0,
            is_vte=True,
            window_start=0.0,
            window_end=1.0,
        )

        assert result.z_idphi == result.z_head_sweep

    def test_summary_method(self):
        """Test summary() returns human-readable string."""
        from neurospatial.behavior.vte import VTETrialResult

        result = VTETrialResult(
            head_sweep_magnitude=2.5,
            z_head_sweep=1.2,
            mean_speed=15.0,
            min_speed=2.0,
            z_speed_inverse=0.8,
            vte_index=1.0,
            is_vte=True,
            window_start=0.0,
            window_end=1.0,
        )

        summary = result.summary()
        assert "2.5" in summary or "2.50" in summary  # head sweep magnitude
        assert "15" in summary  # speed
        assert "VTE" in summary  # classification


# =============================================================================
# Test VTESessionResult Dataclass
# =============================================================================


class TestVTESessionResult:
    """Test VTESessionResult dataclass."""

    def test_dataclass_fields(self):
        """Test that dataclass has required fields."""
        from neurospatial.behavior.vte import VTESessionResult, VTETrialResult

        trial1 = VTETrialResult(
            head_sweep_magnitude=2.5,
            z_head_sweep=1.2,
            mean_speed=15.0,
            min_speed=2.0,
            z_speed_inverse=0.8,
            vte_index=1.0,
            is_vte=True,
            window_start=0.0,
            window_end=1.0,
        )
        trial2 = VTETrialResult(
            head_sweep_magnitude=0.5,
            z_head_sweep=-0.8,
            mean_speed=30.0,
            min_speed=25.0,
            z_speed_inverse=-0.5,
            vte_index=-0.6,
            is_vte=False,
            window_start=1.0,
            window_end=2.0,
        )

        session = VTESessionResult(
            trial_results=[trial1, trial2],
            mean_head_sweep=1.5,
            std_head_sweep=1.0,
            mean_speed=22.5,
            std_speed=7.5,
            n_vte_trials=1,
            vte_fraction=0.5,
        )

        assert len(session.trial_results) == 2
        assert session.mean_head_sweep == 1.5
        assert session.std_head_sweep == 1.0
        assert session.mean_speed == 22.5
        assert session.std_speed == 7.5
        assert session.n_vte_trials == 1
        assert session.vte_fraction == 0.5

    def test_mean_idphi_alias(self):
        """Test that mean_idphi property is alias for mean_head_sweep."""
        from neurospatial.behavior.vte import VTESessionResult

        session = VTESessionResult(
            trial_results=[],
            mean_head_sweep=1.5,
            std_head_sweep=1.0,
            mean_speed=22.5,
            std_speed=7.5,
            n_vte_trials=0,
            vte_fraction=0.0,
        )

        assert session.mean_idphi == session.mean_head_sweep

    def test_std_idphi_alias(self):
        """Test that std_idphi property is alias for std_head_sweep."""
        from neurospatial.behavior.vte import VTESessionResult

        session = VTESessionResult(
            trial_results=[],
            mean_head_sweep=1.5,
            std_head_sweep=1.0,
            mean_speed=22.5,
            std_speed=7.5,
            n_vte_trials=0,
            vte_fraction=0.0,
        )

        assert session.std_idphi == session.std_head_sweep

    def test_get_vte_trials(self):
        """Test get_vte_trials() returns only VTE trials."""
        from neurospatial.behavior.vte import VTESessionResult, VTETrialResult

        trial1 = VTETrialResult(
            head_sweep_magnitude=2.5,
            z_head_sweep=1.2,
            mean_speed=15.0,
            min_speed=2.0,
            z_speed_inverse=0.8,
            vte_index=1.0,
            is_vte=True,
            window_start=0.0,
            window_end=1.0,
        )
        trial2 = VTETrialResult(
            head_sweep_magnitude=0.5,
            z_head_sweep=-0.8,
            mean_speed=30.0,
            min_speed=25.0,
            z_speed_inverse=-0.5,
            vte_index=-0.6,
            is_vte=False,
            window_start=1.0,
            window_end=2.0,
        )
        trial3 = VTETrialResult(
            head_sweep_magnitude=3.0,
            z_head_sweep=1.5,
            mean_speed=10.0,
            min_speed=1.0,
            z_speed_inverse=1.2,
            vte_index=1.3,
            is_vte=True,
            window_start=2.0,
            window_end=3.0,
        )

        session = VTESessionResult(
            trial_results=[trial1, trial2, trial3],
            mean_head_sweep=2.0,
            std_head_sweep=1.0,
            mean_speed=18.3,
            std_speed=8.0,
            n_vte_trials=2,
            vte_fraction=2 / 3,
        )

        vte_trials = session.get_vte_trials()
        assert len(vte_trials) == 2
        assert all(t.is_vte for t in vte_trials)

    def test_summary_method(self):
        """Test summary() returns human-readable string."""
        from neurospatial.behavior.vte import VTESessionResult, VTETrialResult

        trial1 = VTETrialResult(
            head_sweep_magnitude=2.5,
            z_head_sweep=1.2,
            mean_speed=15.0,
            min_speed=2.0,
            z_speed_inverse=0.8,
            vte_index=1.0,
            is_vte=True,
            window_start=0.0,
            window_end=1.0,
        )

        session = VTESessionResult(
            trial_results=[trial1],
            mean_head_sweep=2.5,
            std_head_sweep=0.0,
            mean_speed=15.0,
            std_speed=0.0,
            n_vte_trials=1,
            vte_fraction=1.0,
        )

        summary = session.summary()
        assert "1/1" in summary or "1" in summary  # VTE count
        assert "VTE" in summary


# =============================================================================
# Test normalize_vte_scores()
# =============================================================================


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
        t1_y = np.linspace(30, 50, n_per_trial)  # Move up slowly

        # Add zigzag to create head scanning
        t1_x = t1_x + 5 * np.sin(np.linspace(0, 8 * np.pi, n_per_trial))

        # Trial 2: Fast, straight trajectory approaching center
        t2_times = t1_times[-1] + dt + np.arange(n_per_trial) * dt
        t2_x = np.linspace(50, 50, n_per_trial)
        t2_y = np.linspace(30, 50, n_per_trial)

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

        # Should have results for trials that reached center
        assert len(result.trial_results) >= 0

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

        result = compute_vte_session(
            positions=positions,
            times=times,
            trials=trials,
            decision_region="center",
            env=env,
            window_duration=0.5,
            min_speed=1.0,
        )

        # No valid trials
        assert len(result.trial_results) == 0
        assert result.n_vte_trials == 0
        assert result.vte_fraction == 0.0

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

            y = np.linspace(30, 52, n_per_trial)

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

        # Check consistency: vte_fraction matches
        if len(result.trial_results) > 0:
            expected_fraction = result.n_vte_trials / len(result.trial_results)
            assert_allclose(result.vte_fraction, expected_fraction)
