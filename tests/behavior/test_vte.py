"""Tests for VTE (Vicarious Trial and Error) metrics module.

VTE is characterized by:
- Head sweeping (looking back and forth between options)
- Pausing/slowing at decision points

Key metrics:
- IdPhi (head_sweep_magnitude): Integrated absolute change in the
  orientation of motion, estimated as in the Redish lab's zIdPhi code
- zIdPhi: Z-scored IdPhi relative to session baseline
- VTE index: zIdPhi by default, optionally blended with slowness
"""

import functools
import itertools
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Intervals, Regions
from neurospatial.behavior.segmentation import Trial

from ._reference_turning import lab_rate, reference_head_sweep
from ._reference_velocity import reference_velocity

#: The two heading estimators: the Redish lab's (default) and neurospatial's.
METHODS = ("adaptive_window", "line_fit")

# =============================================================================
# Test helpers
# =============================================================================


def _zigzag(n_samples):
    """East-going zig-zag at 10 Hz: alternating y of 50 and 55 cm."""
    times = np.arange(n_samples) / 10
    x = 5.0 * np.arange(n_samples)
    y = np.where(np.arange(n_samples) % 2 == 0, 50.0, 55.0)
    return times, np.column_stack([x, y])


def _line_fit_head_sweep(times, positions, min_speed, sigma=0.1):
    """Line-fit IdPhi of one run, from the independent reference velocity.

    The heading is the direction of the Gaussian-weighted line-fit velocity
    (``sigma`` seconds) at each sample at least ``min_speed`` fast; IdPhi
    sums the absolute wrapped changes between consecutive such headings.
    """
    velocity = reference_velocity(times, positions, sigma)
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    heading = heading[(speed > 0) & (speed >= min_speed)]
    return float(np.sum(np.abs((np.diff(heading) + np.pi) % (2 * np.pi) - np.pi)))


def _idphi(headings):
    """Sum of |wrapped heading change|, computed with complex exponentials."""
    steps = np.angle(np.exp(1j * np.diff(headings)))
    return float(np.sum(np.abs(steps)))


def _sine_scan(n_samples):
    """East at 10 cm/s at 30 Hz, swinging 8 cm North and South once a second."""
    times = np.arange(n_samples) / 30
    return times, np.column_stack([10.0 * times, 8.0 * np.sin(2 * np.pi * times)])


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

    def test_degree_headings_warn(self):
        """Headings in degrees would be wrapped as radians into nonsense."""
        from neurospatial.behavior.vte import head_sweep_magnitude

        with pytest.warns(UserWarning, match="look like degrees"):
            head_sweep_magnitude(np.array([0.0, 45.0, 0.0, 45.0, 0.0]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            head_sweep_magnitude(np.radians([0.0, 45.0, 0.0, 45.0, 0.0]))

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

    @pytest.mark.parametrize("method", METHODS)
    def test_straight_line_trajectory(self, method):
        """Test that straight trajectory has low head sweep."""
        from neurospatial.behavior.vte import head_sweep_from_positions

        sweep = functools.partial(head_sweep_from_positions, method=method)

        # Straight line moving East (x increases, y constant)
        n_samples = 20
        times = np.linspace(0, 2, n_samples)  # 2 seconds
        positions = np.column_stack(
            [
                np.linspace(0, 100, n_samples),  # x: 0 to 100
                np.ones(n_samples) * 50,  # y: constant at 50
            ]
        )

        result = sweep(times, positions, min_speed=5.0)

        # Constant y: the fitted y-velocity is 0 up to round-off, so every
        # heading is East and IdPhi is 0 to well below 1e-12 rad.
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_zigzag_trajectory(self):
        """A zig-zag of 1 s legs: the lab's estimate, short of the corners' turn.

        Four 1 s legs at 30 Hz, each moving 20 cm/s East and 10 cm/s North
        or South in turn: headings of +-atan(1/2) with three corners, a turn
        of ``6 * atan(1/2)`` in all. IdPhi equals the loop form of the
        package's estimator (``reference_head_sweep``), which, like the
        lab's, spreads each corner over its backward windows and averages
        part of the turn away.
        """
        from neurospatial.behavior.vte import head_sweep_from_positions

        times = np.arange(4 * _FS + 1) / _FS
        leg = (np.arange(times.size - 1) // _FS) % 2  # 0 North, 1 South
        y = np.r_[0.0, np.cumsum(np.where(leg == 0, 10.0, -10.0) / _FS)]
        positions = np.column_stack([20.0 * times, y])

        result = head_sweep_from_positions(times, positions, min_speed=1.0)

        assert_allclose(result, reference_head_sweep(times, positions, 1.0), rtol=1e-12)
        assert 0.5 * 6 * np.arctan(0.5) < result < 6 * np.arctan(0.5)

    @pytest.mark.parametrize("method", METHODS)
    def test_stationary_trajectory(self, method):
        """A still animal has no movement heading: NaN, without a warning."""
        from neurospatial.behavior.vte import head_sweep_from_positions

        sweep = functools.partial(head_sweep_from_positions, method=method)

        n_samples = 20
        times = np.linspace(0, 2, n_samples)
        positions = np.full((n_samples, 2), 50.0)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = sweep(times, positions, min_speed=5.0)

        assert np.isnan(result)

    def test_one_sample_is_nan(self):
        from neurospatial.behavior.vte import head_sweep_from_positions

        result = head_sweep_from_positions(
            np.array([0.0]), np.array([[1.0, 2.0]]), min_speed=1.0
        )
        assert np.isnan(result)

    @pytest.mark.parametrize("method", METHODS)
    def test_nan_row_is_bridged_like_a_deleted_row(self, method):
        """A lost frame inside max_gap is skipped as if its row were deleted."""
        from neurospatial.behavior.vte import head_sweep_from_positions

        sweep = functools.partial(head_sweep_from_positions, method=method)

        times, positions = _sine_scan(90)
        positions[45] = np.nan
        kept = np.delete(np.arange(90), 45)
        expected = sweep(times[kept], positions[kept], min_speed=1.0)
        actual = sweep(times, positions, min_speed=1.0)
        assert expected > 0
        assert_allclose(actual, expected, rtol=1e-12)

    @pytest.mark.parametrize("method", METHODS)
    def test_nan_row_beyond_max_gap_splits_runs(self, method):
        """A lost frame longer than max_gap: no heading change spans it."""
        from neurospatial.behavior.vte import head_sweep_from_positions

        sweep = functools.partial(head_sweep_from_positions, method=method)

        times, positions = _sine_scan(90)
        positions[45] = np.nan
        expected = sweep(times[:45], positions[:45], min_speed=1.0) + sweep(
            times[46:], positions[46:], min_speed=1.0
        )
        actual = sweep(times, positions, min_speed=1.0, max_gap=0.05)
        assert expected > 0
        assert_allclose(actual, expected, rtol=1e-12)

    @pytest.mark.parametrize("method", METHODS)
    def test_break_before_equals_separate_recordings(self, method):
        """Two recordings joined on one clock are analyzed separately."""
        from neurospatial.behavior.vte import head_sweep_from_positions

        sweep = functools.partial(head_sweep_from_positions, method=method)

        times, positions = _sine_scan(90)
        positions[45:] += np.array([-60.0, 30.0])  # put back elsewhere
        break_before = np.arange(90) == 45
        expected = sweep(times[:45], positions[:45], min_speed=1.0) + sweep(
            times[45:], positions[45:], min_speed=1.0
        )
        actual = sweep(times, positions, min_speed=1.0, break_before=break_before)
        undeclared = sweep(times, positions, min_speed=1.0)
        assert_allclose(actual, expected, rtol=1e-12)
        assert not np.isclose(undeclared, expected)

    @pytest.mark.parametrize("method", METHODS)
    def test_slow_run_adds_nothing(self, method):
        """A run with no observed heading contributes nothing, not zero turns."""
        from neurospatial.behavior.vte import head_sweep_from_positions

        sweep = functools.partial(head_sweep_from_positions, method=method)

        times, positions = _sine_scan(90)
        positions[45:] = positions[45]  # still after a 2 s break
        times = times.copy()
        times[45:] += 2.0
        moving = sweep(times[:45], positions[:45], min_speed=1.0)
        both = sweep(times, positions, min_speed=1.0)
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
# Test vte_index()
# =============================================================================


class TestComputeVTEIndex:
    """Test vte_index() function."""

    def test_equal_weighting(self):
        """Test VTE index with equal weighting (head_sweep_weight=0.5)."""
        from neurospatial.behavior.vte import vte_index

        z_head_sweep = 1.0
        z_speed_inv = 1.0

        result = vte_index(z_head_sweep, z_speed_inv, head_sweep_weight=0.5)

        # 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        assert result == 1.0

    def test_head_sweep_only(self):
        """Test VTE index with head_sweep_weight=1.0 (head sweep only)."""
        from neurospatial.behavior.vte import vte_index

        z_head_sweep = 2.0
        z_speed_inv = 0.5

        result = vte_index(z_head_sweep, z_speed_inv, head_sweep_weight=1.0)

        # 1.0 * 2.0 + 0.0 * 0.5 = 2.0
        assert result == 2.0

    def test_speed_only(self):
        """Test VTE index with head_sweep_weight=0.0 (speed only)."""
        from neurospatial.behavior.vte import vte_index

        z_head_sweep = 2.0
        z_speed_inv = 0.5

        result = vte_index(z_head_sweep, z_speed_inv, head_sweep_weight=0.0)

        # 0.0 * 2.0 + 1.0 * 0.5 = 0.5
        assert result == 0.5

    def test_a_zero_weighted_score_does_not_enter(self):
        """A missing z-score with weight 0 leaves the index to the other one."""
        from neurospatial.behavior.vte import vte_index

        assert vte_index(np.nan, 0.5, head_sweep_weight=0.0) == 0.5
        assert vte_index(2.0, np.nan, head_sweep_weight=1.0) == 2.0
        assert np.isnan(vte_index(np.nan, 0.5, head_sweep_weight=0.5))


# =============================================================================
# Test classify_vte()
# =============================================================================


@pytest.mark.parametrize("alpha", [-0.1, 1.5, np.nan])
def test_vte_index_weight_must_be_a_weight(alpha):
    from neurospatial.behavior.vte import vte_index

    with pytest.raises(ValueError, match=r"head_sweep_weight in \[0, 1\]"):
        vte_index(1.0, 1.0, head_sweep_weight=alpha)


def test_session_head_sweep_weight_and_threshold_are_validated(t_maze_environment):
    from neurospatial.behavior.vte import compute_vte_session

    times, positions, trials = _t_maze_session([2.0, 4.0])
    kwargs = {**_SESSION_KWARGS, "trials": trials}
    with pytest.raises(ValueError, match=r"head_sweep_weight in \[0, 1\]"):
        compute_vte_session(
            t_maze_environment, times, positions, head_sweep_weight=2.0, **kwargs
        )
    with pytest.raises(ValueError, match="finite vte_threshold"):
        compute_vte_session(
            t_maze_environment,
            times,
            positions,
            **{**kwargs, "vte_threshold": np.nan},
        )


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
        """A threshold screen is inclusive: an index at the threshold is VTE."""
        from neurospatial.behavior.vte import classify_vte

        assert classify_vte(0.5, threshold=0.5) is True

    def test_unmeasured_index_is_unclassified(self):
        """A NaN index is missing evidence, not evidence against VTE."""
        from neurospatial.behavior.vte import classify_vte

        assert classify_vte(np.nan, threshold=0.5) is None

    @pytest.mark.parametrize("threshold", [np.nan, np.inf])
    def test_threshold_must_be_finite(self, threshold):
        from neurospatial.behavior.vte import classify_vte

        with pytest.raises(ValueError, match="finite threshold"):
            classify_vte(1.0, threshold=threshold)


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
            start_time=1.0,
            stop_time=2.0,
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
        assert result.window_start == 1.0  # start_time
        assert result.window_end == 2.0  # stop_time

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
            start_time=2.0,
            stop_time=3.0,
            min_speed=1.0,
        )

        # Straight line should have low head sweep
        assert result.head_sweep_magnitude < 0.5

        # Mean speed should be > 0 (we're moving)
        assert result.mean_speed > 0

    @pytest.mark.parametrize("method", METHODS)
    def test_one_sample_window_is_nan(self, method):
        """A window with one sample has no speed, so no gated turn: NaN, not 0.

        The speed gate uses the window's own speed, which one sample does not
        have, so even the adaptive window, whose dPhi exists there, measured
        no heading change.
        """
        from neurospatial.behavior.vte import compute_vte_trial

        times, positions = _sine_scan(90)
        # One sample, t = 1.5 s, where the path swings fastest.
        result = compute_vte_trial(
            times, positions, 1.49, 1.52, min_speed=1.0, method=method
        )
        assert np.isnan(result.mean_speed)
        assert np.isnan(result.min_observed_speed)
        assert np.isnan(result.head_sweep_magnitude)

    def test_head_sweep_adds_over_adjacent_windows(self):
        """Adaptive window: splitting a window splits IdPhi; each turn counts once.

        Its dPhi is estimated over the whole recording, so only the gate, by
        each window's own speed, could differ; the scan stays far above it.
        """
        from neurospatial.behavior.vte import compute_vte_trial

        times, positions = _sine_scan(90)

        def sweep(start, stop):
            return compute_vte_trial(
                times, positions, start, stop, min_speed=1.0
            ).head_sweep_magnitude

        assert sweep(0.5, 2.5) > 0
        assert_allclose(sweep(0.5, 2.5), sweep(0.5, 1.4) + sweep(1.4, 2.5))

    @pytest.mark.parametrize("method", METHODS)
    def test_still_window_has_zero_speed_and_no_head_sweep(self, method):
        """A still animal moves at an observed 0 cm/s but faces no heading."""
        from neurospatial.behavior.vte import compute_vte_trial

        trial = functools.partial(compute_vte_trial, method=method)

        times = np.arange(30) / 10
        positions = np.full((30, 2), 50.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = trial(times, positions, 1.0, 2.0, min_speed=1.0)
        assert result.mean_speed == 0.0
        assert result.min_observed_speed == 0.0
        assert np.isnan(result.head_sweep_magnitude)

    @pytest.mark.parametrize("method", METHODS)
    def test_break_before_equals_separate_recording(self, method):
        """The window never reaches across a declared recording break."""
        from neurospatial.behavior.vte import compute_vte_trial

        trial = functools.partial(compute_vte_trial, method=method)

        times, positions = _sine_scan(90)
        positions[45:] += np.array([-60.0, 30.0])
        break_before = np.arange(90) == 45
        kwargs = {"start_time": 0.5, "stop_time": 2.5, "min_speed": 1.0}
        declared = trial(times, positions, break_before=break_before, **kwargs)
        alone = trial(times[:45], positions[:45], **kwargs)
        assert declared == alone
        assert trial(times, positions, **kwargs) != alone

    @pytest.mark.parametrize("method", METHODS)
    def test_nan_row_in_window_is_skipped_like_a_deleted_row(self, method):
        """A lost frame inside the window is bridged as if its row were deleted."""
        from neurospatial.behavior.vte import compute_vte_trial

        trial = functools.partial(compute_vte_trial, method=method)

        times, positions = _sine_scan(90)
        positions[45] = np.nan
        result = trial(times, positions, 1.0, 2.0, min_speed=1.0)
        kept = np.delete(np.arange(90), 45)
        deleted = trial(times[kept], positions[kept], 1.0, 2.0, min_speed=1.0)
        assert result.head_sweep_magnitude > 0
        assert result.head_sweep_magnitude == deleted.head_sweep_magnitude
        assert np.isfinite(result.mean_speed)


# =============================================================================
# Test compute_vte_session()
# =============================================================================


class TestComputeVTESession:
    """Test compute_vte_session() function."""

    @pytest.mark.parametrize("method", METHODS)
    def test_scanning_trial_has_the_larger_head_sweep(self, t_maze_environment, method):
        """A zig-zag approach sweeps its heading; a straight one does not.

        Both rise 26 cm in 59 frames at 30 Hz, 5 s apart (a tracking gap
        longer than ``max_gap``, so no estimate spans the jump between
        them); the zig-zag also moves sideways, so it is the faster of the
        two. The straight approach has no head sweep and runs at exactly
        26 * 30 / 59 cm/s.
        """
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.behavior.vte import compute_vte_session

        env = t_maze_environment

        # Build continuous position data for entire session
        n_per_trial = 60
        sample_rate = 30  # 30 Hz
        dt = 1.0 / sample_rate

        # Trial 1: head-scanning trajectory approaching center
        t1_times = np.arange(n_per_trial) * dt
        t1_x = np.linspace(50, 50, n_per_trial)  # Stay at x=50
        t1_y = np.linspace(30, 56, n_per_trial)  # Move up into the center

        # Add zigzag to create head scanning
        t1_x = t1_x + 5 * np.sin(np.linspace(0, 8 * np.pi, n_per_trial))

        # Trial 2: straight trajectory approaching center
        t2_times = t1_times[-1] + 5.0 + np.arange(n_per_trial) * dt
        t2_x = np.linspace(50, 50, n_per_trial)
        t2_y = np.linspace(30, 56, n_per_trial)

        # Combine into session
        times = np.concatenate([t1_times, t2_times])
        positions = np.column_stack(
            [np.concatenate([t1_x, t2_x]), np.concatenate([t1_y, t2_y])]
        )

        # Define trials
        trials = Intervals(
            [
                Trial(
                    start_time=t1_times[0],
                    stop_time=t1_times[-1],
                    start_region="start",
                    end_region="center",
                    outcome="reached",
                ),
                Trial(
                    start_time=t2_times[0],
                    stop_time=t2_times[-1],
                    start_region="start",
                    end_region="center",
                    outcome="reached",
                ),
            ]
        )

        result = compute_vte_session(
            positions=positions,
            times=times,
            trials=trials,
            region="choice",
            env=env,
            min_speed=1.0,
            head_sweep_weight=0.5,
            vte_threshold=0.0,  # Low threshold to ensure some VTE classification
            method=method,
        )

        # Both trials reach the center
        assert len(result.trial_results) == 2
        scanning, straight = result.trial_results
        assert straight.head_sweep_magnitude == 0.0
        window = (t1_times >= scanning.window_start) & (t1_times < scanning.window_end)
        if method == "adaptive_window":
            # The reference turns of the pass samples, [window_start,
            # window_end), estimated over the scan's whole run.
            expected = reference_head_sweep(
                t1_times, positions[:n_per_trial], 1.0, samples=window
            )
        else:
            # The reference line-fit IdPhi of the pass samples alone.
            expected = _line_fit_head_sweep(
                t1_times[window], positions[:n_per_trial][window], 1.0
            )
        assert expected > np.pi  # the zig-zag turns back and forth
        assert_allclose(scanning.head_sweep_magnitude, expected, rtol=1e-10)
        assert_allclose(straight.mean_speed, 26 * 30 / 59, rtol=1e-12)
        assert scanning.mean_speed > straight.mean_speed
        assert_allclose(
            result.mean_head_sweep, scanning.head_sweep_magnitude / 2, rtol=1e-12
        )

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
                np.linspace(0, 20, 60),  # Never reaches the zone at y=32.5
            ]
        )

        trials = Intervals(
            [
                Trial(
                    start_time=0.0,
                    stop_time=2.0,
                    start_region="start",
                    end_region=None,
                    outcome="returned",
                ),
            ]
        )

        # A trial that never enters the decision region keeps its place with
        # a status saying why, not silently dropped; this is not a warning.
        result = compute_vte_session(
            positions=positions,
            times=times,
            trials=trials,
            region="choice",
            env=env,
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
                # High head sweep (zigzag), inside the 15 cm wide stem
                x = 50 + 7 * np.sin(np.linspace(0, 10 * np.pi, n_per_trial))
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
                    stop_time=trial_times[-1],
                    start_region="start",
                    end_region="center",
                    outcome="reached",
                )
            )

        times = np.concatenate(all_times)
        positions = np.column_stack(
            [np.concatenate(all_positions_x), np.concatenate(all_positions_y)]
        )

        result = compute_vte_session(
            positions=positions,
            times=times,
            trials=Intervals(trials),
            region="choice",
            env=env,
            min_speed=0.5,
            vte_threshold=0.0,
        )

        assert [t.is_vte for t in result.trial_results] == [
            True,
            False,
            True,
            False,
            True,
        ]
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

    # Build a slow head-scanning trajectory up through the choice-point zone.
    n = 60
    dt = 1.0 / 30.0
    times = np.arange(n) * dt
    x = 50.0 + 5.0 * np.sin(np.linspace(0, 8 * np.pi, n))
    y = np.linspace(30.0, 56.0, n)  # into the zone at y >= 32.5
    positions = np.column_stack([x, y])

    trials = Intervals(
        [
            Trial(
                start_time=times[0],
                stop_time=times[-1],
                start_region="start",
                end_region="center",
                outcome="reached",
            )
        ]
    )

    # Positional (env, times, positions) call analyzes the trial. One trial
    # has no spread to z-score against, and that is not worth a warning.
    result = compute_vte_session(
        env,
        times,
        positions,
        region="choice",
        trials=trials,
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


def test_session_window_starts_at_the_entry(zig_zag_then_run):
    """The window is the pass through the region, from the 5.267 s entry.

    The zig-zag before the trial and the straight run before the entry are
    not part of it, and it ends at the first sample past the region.
    """
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.behavior.vte import compute_vte_session

    env, times, positions = zig_zag_then_run
    result = compute_vte_session(
        env,
        times,
        positions,
        region="decision",
        trials=Intervals([Trial(4.0, 9.9, "start", "decision", "reached")]),
        min_speed=5.0,
    )

    in_region = np.isin(env.bin_at(positions), env.bins_in_region("decision"))
    entry = 158  # 5.0 s + 8 / 30 s, the first sample at x >= 46 cm
    exit_ = entry + int(np.flatnonzero(~in_region[entry:])[0])
    assert in_region[entry]
    assert not in_region[entry - 1]
    (trial,) = result.trial_results
    assert trial.status == "ok"
    assert trial.window_start == times[entry]
    assert trial.window_end == times[exit_]
    assert abs(trial.head_sweep_magnitude) < 1e-12
    assert_allclose(trial.mean_speed, 30.0, rtol=1e-9)


def test_session_marks_trial_window_too_short(zig_zag_then_run):
    """A trial that stops two samples after its entry is skipped with status
    "too_few_samples", not silently; its window ends at the trial stop.
    """
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.behavior.vte import compute_vte_session

    env, times, positions = zig_zag_then_run

    result = compute_vte_session(
        env,
        times,
        positions,
        region="decision",
        trials=Intervals([Trial(5.0, times[160], "start", "decision", "reached")]),
        min_speed=5.0,
    )

    (trial,) = result.trial_results
    assert trial.status == "too_few_samples"
    assert trial.window_start == times[158]
    assert trial.window_end == times[160]
    assert np.isnan(trial.mean_speed)
    assert trial.is_vte is None


def _t_maze_session(amplitudes, *, creep=None):
    """Approaches up the T-maze stem through the "choice" zone, one per trial.

    Each trial lasts 2 s at 30 Hz, starts 3 s after the previous one and
    rises from y = 30 to 56 cm (13 cm/s) with an east-west zig-zag of the
    given amplitude (at most 7.5 cm, the half-width of the stem). It enters
    the zone at y = 32.5 cm and stays in it to the trial's stop. With
    ``creep=i``, trial ``i`` instead creeps up from y = 50 cm at 1.5 cm/s,
    too slow for a movement heading at 5 cm/s.
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
        trials.append(Trial(t[0], t[-1], "start", "center", "reached"))
    positions = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
    return np.concatenate(times), positions, Intervals(trials)


def _first_choice_sample(env, times, positions, *, after):
    in_zone = np.isin(env.bin_at(positions), env.bins_in_region("choice"))
    return int(np.flatnonzero(in_zone & (times >= after))[0])


_SESSION_KWARGS = {
    "region": "choice",
    "min_speed": 5.0,
    "vte_threshold": 0.0,
}


def test_trials_table_rows_join_their_input_intervals(t_maze_environment):
    """With a subset of trials, each row joins the interval it describes."""
    from neurospatial.behavior.vte import compute_vte_session

    times, positions, trials = _t_maze_session([2.0, 4.0, 0.0, 7.0])
    subset = trials[[1, 3]]
    result = compute_vte_session(
        t_maze_environment, times, positions, trials=subset, **_SESSION_KWARGS
    )
    table = result.trials_table()

    assert table.index.name == "interval_id"
    assert table.index.tolist() == [1, 3]
    joined = trials.intervals_table().join(table, how="inner")
    assert joined.index.tolist() == [1, 3]
    # Each trial lasts 2 s and starts 3 s after the previous one, so its
    # pass window lies inside its own interval only.
    starts = 3.0 * np.array([1, 3])
    np.testing.assert_array_equal(joined["start_time"], starts)
    assert np.all(joined["window_start"] >= starts)
    assert np.all(joined["window_end"] < starts + 2.0)
    np.testing.assert_array_equal(
        joined["head_sweep_magnitude"],
        [t.head_sweep_magnitude for t in result.trial_results],
    )


def test_session_result_interval_ids_default_and_validation():
    from neurospatial.behavior import VTESessionResult

    trials = _two_trials()
    assert VTESessionResult(trials).trials_table().index.tolist() == [0, 1]
    assert VTESessionResult(trials, interval_ids=[7, 2]).interval_ids.tolist() == [
        7,
        2,
    ]
    with pytest.raises(ValueError, match="one interval_id per interval"):
        VTESessionResult(trials, interval_ids=[0])
    with pytest.raises(ValueError, match="repeated ids"):
        VTESessionResult(trials, interval_ids=[4, 4])


def test_session_trial_without_heading_is_unclassified(t_maze_environment):
    """A pause with no observed heading is not scored as "no head sweep".

    Its IdPhi is NaN, so it leaves the head-sweep z-scores of the other
    trials unchanged and is not classified; its slow speed is still measured.
    """
    from neurospatial.behavior.vte import compute_vte_session

    env = t_maze_environment
    times, positions, trials = _t_maze_session([2.0, 4.0, 0.0, 7.0], creep=2)
    result = compute_vte_session(
        env, times, positions, trials=trials, **_SESSION_KWARGS
    )
    moving = [0, 1, 3]
    without = compute_vte_session(
        env,
        times,
        positions,
        trials=trials[moving],
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
    """Missing rows in the window do not count toward the 3-sample minimum.

    With every gap bridged, the missing rows do not end the pass, so trial 1's
    window runs to its stop but holds only two observed samples.
    """
    from neurospatial.behavior.vte import compute_vte_session

    env = t_maze_environment
    times, positions, trials = _t_maze_session([0.0, 4.0, 7.0])
    entry = _first_choice_sample(env, times, positions, after=3.0)
    positions[entry + 2 : 120] = np.nan  # trial 1 holds samples 60-118

    result = compute_vte_session(
        env, times, positions, trials=trials, max_gap=None, **_SESSION_KWARGS
    )
    statuses = [t.status for t in result.trial_results]
    assert statuses == ["ok", "too_few_samples", "ok"]
    skipped = result.trial_results[1]
    assert skipped.is_vte is None
    assert skipped.window_start == times[entry]
    assert skipped.window_end == trials[1].stop_time


def test_session_break_before_ends_the_window(t_maze_environment):
    """A declared break in a pass ends the window, like the trial stopping there."""
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.behavior.vte import compute_vte_session, compute_vte_trial

    env = t_maze_environment
    times, positions, trials = _t_maze_session([0.0, 4.0, 7.0])
    entry = _first_choice_sample(env, times, positions, after=3.0)
    cut = entry + 20  # break 20 samples (0.67 s) into the pass of trial 1
    break_before = np.arange(times.size) == cut
    declared = compute_vte_session(
        env,
        times,
        positions,
        trials=trials,
        break_before=break_before,
        **_SESSION_KWARGS,
    )
    shortened = Intervals(
        [
            trials[0],
            Trial(trials[1].start_time, times[cut], "start", "choice", "reached"),
            trials[2],
        ]
    )
    stopped = compute_vte_session(
        env, times, positions, trials=shortened, **_SESSION_KWARGS
    )
    undeclared = compute_vte_session(
        env, times, positions, trials=trials, **_SESSION_KWARGS
    )

    def window(result):
        return [(t.window_start, t.window_end) for t in result.trial_results]

    assert window(declared) == window(stopped)
    assert declared.trial_results[1].window_end == times[cut]
    # The other trials are separate runs, untouched by the break.
    for i in (0, 2):
        assert (
            declared.trial_results[i].head_sweep_magnitude
            == stopped.trial_results[i].head_sweep_magnitude
        )
    # No estimate reaches across the break, so the pass is measured as if the
    # recording ended there; the shortened trial still sees the samples after.
    before = compute_vte_trial(
        times[:cut],
        positions[:cut],
        declared.trial_results[1].window_start,
        times[cut],
        min_speed=_SESSION_KWARGS["min_speed"],
    )
    assert declared.trial_results[1].head_sweep_magnitude == before.head_sweep_magnitude
    assert (
        declared.trial_results[1].head_sweep_magnitude
        != stopped.trial_results[1].head_sweep_magnitude
    )
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
            times, p, 0.0, 2.0, min_speed=1.0
        ),
        "compute_vte_session": lambda p: vte.compute_vte_session(
            t_maze_environment,
            times,
            p,
            trials=Intervals([], record_type=Trial),
            **_SESSION_KWARGS,
        ),
    }
    match = rf"got shape \({n_samples},\)(.|\n)*heading .*needs both x and y"
    with pytest.raises(ValueError, match=match):
        calls[name](10 * times)


def test_session_summary_counts_classified_trials(t_maze_environment):
    """The text and the summary say which trials the VTE fraction is over."""
    from neurospatial.behavior.vte import compute_vte_session

    times, positions, trials = _t_maze_session([2.0, 4.0, 0.0, 7.0], creep=2)
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
    trials = Intervals(
        [
            Trial(
                start_time=0.0,
                stop_time=10.0,
                start_region="start",
                end_region="left",
                outcome="reached",
            )
        ]
    )
    kwargs = {"region": "center", "min_speed": 5.0}
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
    assert_allclose(whole.trial_results[0].window_start, 2.8, rtol=1e-12)
    assert_allclose(late.trial_results[0].window_start, 7.9, rtol=1e-12)
    assert_allclose(late.trial_results[0].mean_speed, 20.0, rtol=1e-9)


def test_session_keeps_one_result_per_trial_in_input_order(t_maze_environment):
    """Unmeasured trials keep their place, with NaN metrics and a status."""
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.behavior.vte import compute_vte_session

    times, positions, trials = _t_maze_session([2.0, 4.0, 7.0])
    # Trial 1 never reaches the center: it stays low on the stem.
    in_trial_1 = (times >= trials[1].start_time) & (times <= trials[1].stop_time)
    positions[in_trial_1, 1] = np.linspace(10.0, 30.0, in_trial_1.sum())
    # A trial on a millisecond clock has no samples on the seconds clock.
    ms_trial = Trial(
        start_time=1000.0,
        stop_time=2000.0,
        start_region="start",
        end_region="center",
        outcome="reached",
    )
    trials = Intervals([trials[0], ms_trial, trials[1], trials[2]])
    with pytest.warns(
        UserWarning,
        match=r"has no samples in \[start_time, stop_time\)",
    ) as record:
        result = compute_vte_session(
            t_maze_environment, times, positions, trials=trials, **_SESSION_KWARGS
        )
    messages = [str(w.message) for w in record]
    # Rows are in time order, so the trial on the millisecond clock is last.
    assert any("Trial 3 " in m and "no samples" in m and "clock" in m for m in messages)

    assert [t.status for t in result.trial_results] == [
        "ok",
        "no_entry",
        "ok",
        "no_samples",
    ]
    for unmeasured in result.trial_results[1::2]:
        assert np.isnan(unmeasured.head_sweep_magnitude)
        assert np.isnan(unmeasured.window_end)
        assert np.isnan(unmeasured.vte_index)
        assert unmeasured.is_vte is None
    measured = [result.trial_results[i] for i in (0, 2)]
    assert measured[0].window_end < measured[1].window_end
    assert measured[0].head_sweep_magnitude < measured[1].head_sweep_magnitude
    assert result.n_classified == 2

    table = result.trials_table()
    assert table.index.name == "interval_id"
    # Rows are in time order; each keeps the id of its position in the list.
    assert table.index.tolist() == [0, 2, 3, 1]
    assert table["status"].tolist() == ["ok", "no_entry", "ok", "no_samples"]
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


# =============================================================================
# Known answers
# =============================================================================

_FS = 30  # Hz


def _stem_scan(t0, n_reversals, speed):
    """Up the stem at x = 50 cm, reversing direction ``n_reversals`` times.

    ``n_reversals`` is even: the animal goes 6 cm up and back down
    ``n_reversals / 2`` times, starting at y = 35 cm inside the "choice"
    zone, then runs up to the junction at y = 56 cm, at a
    constant ``speed`` (cm/s) between reversals, sampled at 30 Hz. Each
    reversal turns the direction of movement by exactly pi, and the animal
    is slower than any positive ``min_speed`` at the turning point, so IdPhi
    is ``n_reversals * pi``.
    """
    turns = [35.0]
    for k in range(n_reversals):
        turns.append(turns[-1] + (6.0 if k % 2 == 0 else -6.0))
    turns.append(56.0)
    y = [turns[0]]
    for start, stop in itertools.pairwise(turns):
        n = round(abs(stop - start) / speed * _FS)
        y.extend(np.linspace(start, stop, n + 1)[1:])
    y = np.asarray(y)
    times = t0 + np.arange(y.size) / _FS
    return times, np.column_stack([np.full(y.size, 50.0), y])


def _stem_run(t0, speed):
    """Straight up the stem from y = 35 to 56 cm at a constant speed (cm/s)."""
    y = np.arange(35.0, 56.0 + 1e-9, speed / _FS)
    times = t0 + np.arange(y.size) / _FS
    return times, np.column_stack([np.full(y.size, 50.0), y])


def test_head_sweep_counts_each_reversal_as_its_three_estimated_turns():
    """Each reversal of the stem scan adds pi * (2/3 + 1/n_max) at 30 Hz.

    The stem scan moves along y only, so the velocity's x is exactly 0 and
    Phi is +-pi/2, stepping by pi at the one sample ``j`` where the
    estimated y-velocity changes sign. At ``i > j + 2`` some 3-sample
    window lies after the step, scores 0 and gives a rate of 0. At
    ``i = j``, ``j + 1`` and ``j + 2`` every window spans the step: a window
    of ``w`` samples has slope ``pi / w`` per sample, and the lab's score,
    ``mean((d_k + d_w * k / w)**2)`` over the earlier samples' offsets
    ``d_k`` from ``x[i]``, is in units of pi**2
    ``mean((1 + k / w)**2)``, falling with ``w``, at ``i = j`` (so the
    longest window, ``n_max = ceil(0.5 * 30) = 15``, wins); and at ``i = j + 1``
    ``(1/9 + 25/9 + 4) / 3 = 62/27`` for ``w = 3`` against 2.34 for
    ``w = 4`` and more beyond, and at ``i = j + 2`` ``(1/9 + 4/9 + 4) / 3``
    for ``w = 3`` against 1.84 for ``w = 4``, so ``w = 3`` wins both. The
    turns sum to ``pi * (1/15 + 1/3 + 1/3)``, and the boxcar keeps the sum.
    The Gaussian speed stays above the 2 cm/s gate at those samples.
    """
    from neurospatial.behavior.vte import head_sweep_from_positions

    n_max = int(np.ceil(0.5 * _FS))
    assert n_max == 15
    per_reversal = np.pi * (2 / 3 + 1 / n_max)
    for n_reversals in (0, 2, 4, 6):
        times, positions = _stem_scan(0.0, n_reversals, speed=8.0)
        assert_allclose(
            head_sweep_from_positions(times, positions, min_speed=2.0),
            n_reversals * per_reversal,
            rtol=1e-12,
            atol=1e-12,
        )


def _north_pause_north():
    """Whole-pixel tracking at 30 Hz: 2 s north, 2 s still, 2 s north.

    x stays at 50 px; y climbs 2 px per frame, holds for 60 frames, then
    climbs again.
    """
    climb = 2.0 * np.arange(60)
    top = 100.0 + climb[-1]
    y = np.concatenate([100.0 + climb, np.full(60, top), top + 2.0 + climb])
    times = np.arange(y.size) / _FS
    return times, np.column_stack([np.full(y.size, 50.0), y])


def test_min_speed_zero_counts_the_turns_at_zero_speed():
    """With no speed gate IdPhi is the ungated sum, zero-speed samples included.

    x never changes, so its estimated velocity is exactly 0. The estimated
    y-velocity is positive while its window sees the climb and exactly 0
    in the pause, so Phi is atan2(positive, 0) = pi/2 while the animal moves
    and atan2(0, 0) = 0 in the pause: the lab's code has no rule for a zero
    velocity. Phi steps by pi/2 on stopping and again on starting. A step
    adds its size times (2/3 + 1/15) at 30 Hz (the three turns derived in
    ``test_head_sweep_counts_each_reversal_as_its_three_estimated_turns``,
    which scale with the step), so the ungated IdPhi is pi * (2/3 + 1/15).

    The turns of the stopping step lie inside the pause, where a
    frame-to-frame speed is exactly 0. With ``min_speed=0`` they count; a
    positive gate on that speed drops them, leaving at most the starting
    step's half.
    """
    from neurospatial.behavior.vte import head_sweep_from_positions

    times, positions = _north_pause_north()
    frame_speed = np.r_[0.0, np.abs(np.diff(positions[:, 1]))] * _FS
    assert (frame_speed[60:120] == 0).all()
    ungated = np.pi * (2 / 3 + 1 / 15)
    for speed in (None, frame_speed):
        assert_allclose(
            head_sweep_from_positions(times, positions, min_speed=0.0, speed=speed),
            ungated,
            rtol=1e-12,
        )
    gated = head_sweep_from_positions(
        times, positions, min_speed=1.0, speed=frame_speed
    )
    assert 0 < gated <= ungated / 2 + 1e-12


@pytest.mark.parametrize("method", METHODS)
def test_pass_with_no_movement_has_a_head_sweep_only_without_the_gate(method):
    """No movement: NaN under a speed gate; with none, 0 from the lab's sum.

    Whole-pixel tracking of a still animal has a speed of exactly 0. Under a
    positive gate no sample counts and the head sweep is NaN, not 0, with
    either method. With ``min_speed=0`` the adaptive window counts every
    sample: Phi is atan2(0, 0) = 0 throughout, so dPhi and IdPhi are 0. The
    line fit gives a sample with zero speed no direction of movement, so its
    IdPhi stays NaN. The same holds for a pass that lies inside a pause of a
    longer recording.
    """
    from neurospatial.behavior.vte import compute_vte_trial, head_sweep_from_positions

    def expect(head_sweep, *, gated):
        if gated or method == "line_fit":
            assert np.isnan(head_sweep)
        else:
            assert head_sweep == 0.0

    times = np.arange(90) / _FS
    still = np.tile([137.0, 88.0], (times.size, 1))
    times_pause, positions = _north_pause_north()
    for min_speed in (5.0, 0.0):
        expect(
            head_sweep_from_positions(times, still, min_speed=min_speed, method=method),
            gated=min_speed > 0,
        )
        in_pause = compute_vte_trial(
            times_pause,
            positions,
            times_pause[75],
            times_pause[105],
            min_speed=min_speed,
            method=method,
        )
        assert in_pause.mean_speed == 0.0
        expect(in_pause.head_sweep_magnitude, gated=min_speed > 0)


def _circle(n_revolutions, fs=30, radius=20.0, period=4.0):
    """Constant-speed counter-clockwise circling, ``period`` s per revolution."""
    times = np.arange(0, n_revolutions * period, 1 / fs)
    angle = 2 * np.pi * times / period
    return times, radius * np.column_stack([np.cos(angle), np.sin(angle)])


@pytest.mark.parametrize(
    ("path", "lab_idphi"),
    [
        # Values printed by GNU Octave 11 running the Redish lab's dxdt.m
        # (andrewpapale/odorTrails @ 91954210ec) as compute_IdPhi_AvgdPhi.m
        # calls it: dx, dy = dxdt(x), dxdt(y); dphi = dxdt(unwrap(atan2(dy,
        # dx))), window 0.5 s, smoothing 0.1 s; sum(abs(dphi)) * dt over the
        # samples with 1 <= t < 3 s of a 4 s path.
        ("scan", 5.2667604081609056),
        ("circle", 3.0014556323368304),
        ("zigzag", 1.6002749839840285),
    ],
)
def test_head_sweep_matches_the_redish_lab_code(path, lab_idphi):
    """IdPhi over a pass inside a recording equals the lab's zIdPhi code.

    These three values are the only ones in the tests that come from the
    lab's own code: synthetic paths, evenly sampled, summed over samples
    1 s or more from either end of the recording. Real tracking, pauses of
    whole-pixel tracking and the first samples of a run (where the package
    gives no rate and the lab's code gives 0) have no stored value from the
    lab's code; other tests compare the package with ``lab_rate``, which is
    the package's algorithm written as a loop.

    Away from the recording's ends the estimate is the lab's, so the pass's
    IdPhi equals the lab's sum of |dPhi| times the sample interval (the lab
    sums |dPhi| in rad/s; the constant interval cancels in zIdPhi). Video
    rates (29.97 and 59.94 Hz) keep the lab's window counts, ``ceil(0.5 /
    dt)`` and ``ceil(0.1 / dt)``, clear of float noise. The scan moves 10
    cm/s East and swings 8 cm at 1 Hz, the circle turns pi/2 rad/s, and the
    zig-zag moves 20 cm/s East and 10 cm/s North or South in 1 s legs. Over
    the 2 s the lab's estimate is about 48% of the scan's turn (four swings
    of 2 atan(1.6 pi), 11.0 rad) and 96% of the circle's (pi rad).
    """
    from neurospatial.behavior.vte import compute_vte_trial

    video = 30000 / 1001  # Hz
    times = np.arange(120) / video
    if path == "scan":
        positions = np.column_stack([10.0 * times, 8.0 * np.sin(2 * np.pi * times)])
    elif path == "circle":
        times = np.arange(240) / (2 * video)
        angle = 2 * np.pi * times / 4.0
        positions = 20.0 * np.column_stack([np.cos(angle), np.sin(angle)])
    else:
        y = 10.0 * (np.abs((times + 1) % 2 - 1) - 0.5)
        positions = np.column_stack([20.0 * times, y])
    result = compute_vte_trial(times, positions, 1.0, 3.0, min_speed=0.0)
    assert_allclose(result.head_sweep_magnitude, lab_idphi, rtol=1e-9)


def _tied_signal(kind, n_samples, rng):
    """Signals whose backward windows tie, exactly or to rounding.

    Whole-pixel tracking, pauses and steps give many windows the same score;
    scaling the pixels by 0.37 and shifting them leaves ties that differ
    only by rounding, where the order of the sum decides.
    """
    walk = np.cumsum(rng.integers(-1, 2, n_samples)).astype(float)
    if kind == "pixels":
        return walk
    if kind == "scaled pixels":
        return 0.37 * walk + 1234.567
    if kind == "pause":
        half = n_samples // 2
        return np.minimum(np.arange(n_samples), half) * 2.0
    if kind == "steps":
        return np.repeat(rng.integers(0, 4, n_samples // 5 + 1), 5)[:n_samples] * 0.1
    return np.cumsum(rng.normal(0.0, 1.0, n_samples))  # "jitter"


@pytest.mark.parametrize("clock", ["even", "uneven"])
@pytest.mark.parametrize("fs", [30.0, 30000 / 1001, 60.0])
@pytest.mark.parametrize(
    "kind", ["pixels", "scaled pixels", "pause", "steps", "jitter"]
)
def test_adaptive_window_rate_is_that_of_summing_every_window(kind, fs, clock):
    """The rate equals, bit for bit, the rate from every window's full sum.

    The package bounds each window's score and sums in full only the windows
    that can hold the smallest; the reference sums every window of every
    sample in order. The slopes agree exactly, so the same window was chosen
    at every sample, ties included (the shorter window wins a tie).
    """
    from neurospatial.behavior._adaptive_window import adaptive_window_rate

    rng = np.random.default_rng([int(fs * 100), len(kind), len(clock)])
    n_samples = int(3 * fs)
    times = np.arange(n_samples) / fs
    if clock == "uneven":
        times = np.sort(times + rng.uniform(-0.3, 0.3, n_samples) / fs)
    values = _tied_signal(kind, n_samples, rng)
    rate = adaptive_window_rate(times, values, smoothing=0)
    assert np.isfinite(rate[3:]).all()
    np.testing.assert_array_equal(rate, lab_rate(times, values, smoothing=0))


def test_adaptive_window_tie_decided_by_rounding_follows_the_sum_in_order():
    """Two windows that tie exactly on paper: the order of the sum decides.

    At the last of these 16 Hz samples the 6-sample window has slope 0 and
    misses 0, 0, 0.1, 0.1, 0.2, 0, so its score is 0.06 / 6 = 0.01. The
    7-sample window has slope 0.1 / 7 per sample and misses -1, -2, 4, 3, 9,
    -6 and -14 seventieths, so its score is (343 / 4900) / 7 = 0.01 too.
    Summed in order, k = 1 to the window length, the 6-sample score rounds
    to 0.010000000000000004 and the 7-sample score to 0.01, so the longer
    window is chosen and the rate is its slope, 1.6 / 7 per second, not 0.
    (The expanded square, summed another way, ranks them the other way.)
    """
    from neurospatial.behavior._adaptive_window import adaptive_window_rate

    values = 0.1 * np.array([1.0, 1, 2, 4, 3, 3, 2, 2, 2])
    times = np.arange(values.size) / 16.0
    rate = adaptive_window_rate(times, values, smoothing=0)
    assert rate[-1] == (values[8] - values[1]) / (times[8] - times[1])
    assert_allclose(rate[-1], 1.6 / 7, rtol=1e-12)
    np.testing.assert_array_equal(rate, lab_rate(times, values, smoothing=0))


def test_adaptive_window_keeps_the_shortest_of_exactly_tied_windows():
    """Windows with the same score, bit for bit: the shortest is chosen.

    The signal repeats 0, 1, 1 at 16 Hz, where every sum is exact. At a
    sample where it is 0, the windows of 3, 6 and 9 samples end on a 0, so
    their slope is 0 and their misses are 1, 1, 0 repeated: scores 2/3, 4/6
    and 6/9, the same number. The windows of 4, 5, 7 and 8 samples score
    67/32, 11/5, 93/49 and 255/128.
    """
    from neurospatial.behavior._adaptive_window import _best_windows

    values = np.tile([0.0, 1.0, 1.0], 6)[:16]
    times = np.arange(values.size) / 16.0
    assert _best_windows(times, values, 9)[15] == 3


# A serial run takes about 0.3 s; the budget leaves room for a loaded machine
# while still catching the full sum over every window this replaced (6 s
# serial, growing with the square of the sampling rate).
@pytest.mark.wallclock
def test_head_sweep_of_one_kilohertz_tracking_is_fast():
    """20 s of 1 kHz tracking: the cost grows with the window, not its square."""
    import time

    from neurospatial.behavior.vte import head_sweep_from_positions

    rng = np.random.default_rng(0)
    times = np.arange(20_000) / 1000.0
    positions = np.column_stack([10.0 * times, 8.0 * np.sin(2 * np.pi * times)])
    positions += rng.normal(0.0, 0.3, positions.shape)
    start = time.perf_counter()
    head_sweep = head_sweep_from_positions(times, positions, min_speed=5.0)
    assert time.perf_counter() - start <= 2.5
    assert head_sweep > 0


def test_passing_the_estimated_heading_reproduces_the_default():
    """The default Phi is the direction of the adaptive-window velocity.

    Passing that direction as ``headings`` (NaN at the first samples, which
    have no velocity) gives the same IdPhi: dPhi is estimated from either
    the same way.
    """
    from neurospatial.behavior.vte import head_sweep_from_positions

    times, positions = _sine_scan(90)
    phi = np.arctan2(lab_rate(times, positions[:, 1]), lab_rate(times, positions[:, 0]))
    assert np.isnan(phi[0])
    default = head_sweep_from_positions(times, positions, min_speed=1.0)
    assert default > 0
    assert_allclose(
        head_sweep_from_positions(times, positions, min_speed=1.0, headings=phi),
        default,
        rtol=1e-12,
    )


@pytest.mark.parametrize("method", METHODS)
def test_head_sweep_is_invariant_to_translation_clock_units_and_axis_swaps(method):
    """IdPhi is unchanged by moving, re-clocking or rescaling the path.

    Either estimator is unchanged by swapping or mirroring the axes; the
    line fit also by any rotation (see
    ``test_only_the_line_fit_is_rotation_invariant``).
    """
    from neurospatial.behavior.vte import head_sweep_from_positions as sweep

    head_sweep_from_positions = functools.partial(sweep, method=method)

    for t, p, min_speed in [
        (*_circle(1), 1.0),
        (*_sine_scan(90), 1.0),
        (*_stem_scan(0.0, 4, speed=8.0), 2.0),
    ]:
        expected = head_sweep_from_positions(t, p, min_speed=min_speed)
        assert expected > 0
        moved = p + np.array([37.0, -12.0])
        assert_allclose(
            head_sweep_from_positions(t + 1000.0, moved, min_speed=min_speed),
            expected,
            rtol=1e-9,
        )
        # The same path in meters, with the gate in meters per second.
        assert_allclose(
            head_sweep_from_positions(t, p / 100, min_speed=min_speed / 100),
            expected,
            rtol=1e-9,
        )
        for swapped in (p[:, ::-1], p * np.array([-1.0, 1.0])):
            assert_allclose(
                head_sweep_from_positions(t, swapped, min_speed=min_speed),
                expected,
                rtol=1e-9,
            )


def test_only_the_line_fit_is_rotation_invariant():
    """An oblique rotation of the camera changes the adaptive window's IdPhi.

    The lab's estimator fits x and y separately; the line fit's heading is
    the direction of one 2-D velocity, so it turns with the path.
    """
    from neurospatial.behavior.vte import head_sweep_from_positions

    times, positions = _sine_scan(120)
    angle = np.pi / 4  # the adaptive window's IdPhi moves 21% here
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated = positions @ rotation.T

    def sweep(p, method):
        return head_sweep_from_positions(times, p, min_speed=1.0, method=method)

    assert_allclose(sweep(rotated, "line_fit"), sweep(positions, "line_fit"), rtol=1e-9)
    adaptive, turned = (
        sweep(positions, "adaptive_window"),
        sweep(rotated, "adaptive_window"),
    )
    assert abs(turned / adaptive - 1) > 0.1


# =============================================================================
# method="line_fit": neurospatial's rotation-invariant heading
# =============================================================================


def test_line_fit_zigzag_turns_by_its_corners():
    """A zig-zag of 1 s legs turns by 2 atan(1/2) at each of its corners.

    Four 1 s legs at 30 Hz, each moving 20 cm/s East and 10 cm/s North
    or South in turn: headings of +-atan(1/2) with three corners. Away
    from a corner by more than the smoothing reach (4 sigma = 0.4 s) the
    line fit sees one straight leg, so the heading is exactly
    +-atan(1/2), also at the two ends of the recording; through a corner
    it turns monotonically from one to the other. IdPhi is therefore
    exactly ``3 * 2 * atan(1/2)``, to round-off.
    """
    from neurospatial.behavior.vte import head_sweep_from_positions

    times = np.arange(4 * _FS + 1) / _FS
    leg = (np.arange(times.size - 1) // _FS) % 2  # 0 North, 1 South
    y = np.r_[0.0, np.cumsum(np.where(leg == 0, 10.0, -10.0) / _FS)]
    positions = np.column_stack([20.0 * times, y])

    result = head_sweep_from_positions(
        times, positions, min_speed=1.0, method="line_fit"
    )

    assert_allclose(result, 6 * np.arctan(0.5), rtol=1e-12)
    assert_allclose(result, _line_fit_head_sweep(times, positions, 1.0), rtol=1e-12)


def test_line_fit_counts_each_reversal_as_pi():
    from neurospatial.behavior.vte import head_sweep_from_positions

    for n_reversals in (0, 2, 4, 6):
        times, positions = _stem_scan(0.0, n_reversals, speed=8.0)
        assert_allclose(
            head_sweep_from_positions(
                times, positions, min_speed=2.0, method="line_fit"
            ),
            n_reversals * np.pi,
            rtol=1e-12,
            atol=1e-12,
        )


def test_line_fit_full_circle_is_two_pi():
    """Each extra revolution adds exactly 2 pi.

    The first revolution falls short of 2 pi only by its edges: the last
    sample is one frame short of a full turn, and the velocity, the slope of
    a Gaussian-weighted line fit (sigma 0.1 s), cannot see past the ends of
    the recording. At each end the fit holds one half of the window, and on a
    path turning at a constant rate its slope points along the path as it was
    cov(tau, tau**2) / (2 var(tau)) seconds earlier, with tau half-normal:
    sigma * sqrt(2 / pi) / (2 * (1 - 2 / pi)), about 1.1 sigma. Each end
    costs at most the angle turned in that time.
    """
    from neurospatial.behavior.vte import head_sweep_from_positions

    for fs in (30, 120):
        one, two = (
            head_sweep_from_positions(
                *_circle(n, fs=fs), min_speed=1.0, method="line_fit"
            )
            for n in (1, 2)
        )
        assert_allclose(two - one, 2 * np.pi, rtol=1e-9)
        omega = 2 * np.pi / 4.0  # rad/s
        one_sided_lag = 0.1 * np.sqrt(2 / np.pi) / (2 * (1 - 2 / np.pi))
        assert 0 < 2 * np.pi - one < omega * (1 / fs + 2 * one_sided_lag)


def test_line_fit_window_is_measured_from_its_own_samples():
    """The line fit's pass IdPhi is that of the pass's samples alone."""
    from neurospatial.behavior.vte import compute_vte_trial, head_sweep_from_positions

    times, positions = _sine_scan(90)
    result = compute_vte_trial(
        times, positions, 0.5, 2.5, min_speed=1.0, method="line_fit"
    )
    window = (times >= 0.5) & (times < 2.5)
    alone = head_sweep_from_positions(
        times[window], positions[window], min_speed=1.0, method="line_fit"
    )
    assert alone > 0
    assert result.head_sweep_magnitude == alone


def test_line_fit_break_in_a_pass_equals_the_trial_stopping_there(t_maze_environment):
    """With the line fit, a declared break is exactly a trial stopping there."""
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.behavior.vte import compute_vte_session

    env = t_maze_environment
    times, positions, trials = _t_maze_session([0.0, 4.0, 7.0])
    cut = _first_choice_sample(env, times, positions, after=3.0) + 20
    kwargs = {**_SESSION_KWARGS, "method": "line_fit"}
    declared = compute_vte_session(
        env,
        times,
        positions,
        trials=trials,
        break_before=np.arange(times.size) == cut,
        **kwargs,
    )
    shortened = Intervals(
        [
            trials[0],
            Trial(trials[1].start_time, times[cut], "start", "choice", "reached"),
            trials[2],
        ]
    )
    stopped = compute_vte_session(env, times, positions, trials=shortened, **kwargs)
    assert declared.trial_results == stopped.trial_results


def test_method_defaults_to_the_adaptive_window():
    import inspect

    from neurospatial.behavior.vte import (
        compute_vte_session,
        compute_vte_trial,
        head_sweep_from_positions,
    )

    for function in (head_sweep_from_positions, compute_vte_trial, compute_vte_session):
        parameter = inspect.signature(function).parameters["method"]
        assert parameter.default == "adaptive_window"
    times, positions = _sine_scan(90)
    assert head_sweep_from_positions(
        times, positions, min_speed=1.0
    ) == head_sweep_from_positions(
        times, positions, min_speed=1.0, method="adaptive_window"
    )


@pytest.mark.parametrize("method", ["line", "adaptive", "gaussian", None])
def test_unknown_method_raises(method, t_maze_environment):
    from neurospatial.behavior.vte import (
        compute_vte_session,
        compute_vte_trial,
        head_sweep_from_positions,
    )

    times, positions = _sine_scan(90)
    trials = Intervals([Trial(0.0, 2.9, "start", "center", "reached")])
    calls = [
        lambda: head_sweep_from_positions(
            times, positions, min_speed=1.0, method=method
        ),
        lambda: compute_vte_trial(
            times, positions, 0.5, 2.5, min_speed=1.0, method=method
        ),
        lambda: compute_vte_session(
            t_maze_environment,
            times,
            positions,
            region="choice",
            trials=trials,
            min_speed=1.0,
            method=method,
        ),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="method"):
            call()


def test_vte_trial_straight_constant_speed():
    """A straight 20 cm/s run: no head sweep and a measured 20 cm/s."""
    from neurospatial.behavior.vte import compute_vte_trial

    times = np.arange(90) / _FS
    positions = np.column_stack([20.0 * times, np.full(90, 50.0)])
    result = compute_vte_trial(times, positions, 1.0, 2.0, min_speed=5.0)
    assert result.head_sweep_magnitude == 0.0
    assert_allclose(result.mean_speed, 20.0, rtol=1e-12)
    assert_allclose(result.min_observed_speed, 20.0, rtol=1e-12)
    assert (result.window_start, result.window_end) == (1.0, 2.0)
    assert result.status == "ok"
    assert result.is_vte is None


@pytest.mark.parametrize(
    ("method", "r"),
    [("adaptive_window", np.pi * (2 / 3 + 1 / 15)), ("line_fit", np.pi)],
)
def test_session_z_scores_vte_index_and_verdicts(t_maze_environment, method, r):
    """Two slow scanning trials and two fast straight controls.

    Every trial runs inside the "choice" zone from its start, so its pass is
    the whole trial. The scanning trials reverse 2 and 4 times, each reversal
    adding ``r``: ``pi * (2/3 + 1/15)`` at 30 Hz with the adaptive window
    (derived in
    ``test_head_sweep_counts_each_reversal_as_its_three_estimated_turns``),
    ``pi`` with the line fit. The controls run straight (IdPhi 0) at 20 and
    30 cm/s. Head-sweep z-scores over [2, 4, 0, 0] r are [1, 5, -3, -3] /
    sqrt(11) (population std), whatever r is.
    """
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.behavior.vte import compute_vte_session, compute_vte_trial

    pieces = [
        _stem_scan(0.0, 2, speed=8.0),
        _stem_scan(10.0, 4, speed=8.0),
        _stem_run(20.0, speed=20.0),
        _stem_run(25.0, speed=30.0),
    ]
    times = np.concatenate([t for t, _ in pieces])
    positions = np.vstack([p for _, p in pieces])
    trials = Intervals(
        [
            Trial(
                start_time=t[0],
                stop_time=t[-1],
                start_region="start",
                end_region="center",
                outcome="reached",
            )
            for t, _ in pieces
        ]
    )
    alpha, threshold = 0.3, 0.5
    result = compute_vte_session(
        t_maze_environment,
        times,
        positions,
        region="choice",
        trials=trials,
        min_speed=2.0,
        head_sweep_weight=alpha,
        vte_threshold=threshold,
        method=method,
    )
    table = result.trials_table()

    assert_allclose(table["head_sweep_magnitude"], [2 * r, 4 * r, 0, 0], atol=1e-12)
    assert_allclose(
        table["z_head_sweep"], np.array([1, 5, -3, -3]) / np.sqrt(11), rtol=1e-12
    )
    speeds = table["mean_speed"].to_numpy()
    assert_allclose(speeds[2:], [20.0, 30.0], rtol=1e-12)
    for i in (0, 1):  # the scanning speeds are those of the windows alone
        trial = result.trial_results[i]
        assert trial.window_start == trials[i].start_time
        assert trial.window_end == trials[i].stop_time
        alone = compute_vte_trial(
            times,
            positions,
            trial.window_start,
            trial.window_end,
            min_speed=2.0,
            method=method,
        )
        assert_allclose(speeds[i], alone.mean_speed, rtol=1e-12)
        assert_allclose(
            trial.head_sweep_magnitude, alone.head_sweep_magnitude, rtol=1e-12
        )
        assert speeds[i] < 8.0  # slowed by the reversals
    z_speed = -(speeds - speeds.mean()) / speeds.std()
    assert_allclose(table["z_speed_inverse"], z_speed, rtol=1e-12)
    vte_index = alpha * np.array([1, 5, -3, -3]) / np.sqrt(11) + (1 - alpha) * z_speed
    assert_allclose(table["vte_index"], vte_index, rtol=1e-12)
    assert table["is_vte"].tolist() == list(vte_index >= threshold)
    assert table["is_vte"].tolist() == [True, True, False, False]
    assert result.n_vte_trials == 2
    assert result.vte_fraction == 0.5
    assert_allclose(result.mean_head_sweep, 1.5 * r, rtol=1e-12)
    assert_allclose(result.std_head_sweep, r * np.sqrt(11) / 2, rtol=1e-12)


def test_session_result_refuses_entries_that_are_not_trial_results():
    from neurospatial.behavior.vte import VTESessionResult

    with pytest.raises(
        TypeError, match=r"trial_results entry must be a VTETrialResult"
    ):
        VTESessionResult(trial_results=(1.0,))


def test_session_trial_without_duration_has_no_samples_and_no_warning(
    t_maze_environment,
):
    """A trial with start == stop holds no sample; that is not a clock problem."""
    import warnings

    from neurospatial.behavior.vte import compute_vte_session

    times, positions, trials = _t_maze_session([2.0, 4.0])
    instant = Trial(trials[0].stop_time, trials[0].stop_time, "start", None, "censored")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compute_vte_session(
            t_maze_environment,
            times,
            positions,
            trials=Intervals([*trials, instant]),
            **_SESSION_KWARGS,
        )
    statuses = [r.status for r in result.trial_results]
    assert statuses.count("no_samples") == 1
    assert statuses.count("ok") == 2


def _straight_runs():
    """Six straight eastward approaches to "center" at 10-60 cm/s.

    Returns the positional arguments of ``compute_vte_session``, its keyword
    arguments, and each trial's sample count.
    """
    from shapely.geometry import box

    from neurospatial import Environment

    env = Environment.from_polygon(box(0, 0, 100, 10), bin_size=2.0, units="cm")
    regions = Regions()
    regions.add("center", polygon=box(70, 0, 80, 10))
    env = env.with_regions(regions)
    times, positions, trials, t0 = [], [], [], 0.0
    for speed in (10, 20, 30, 40, 50, 60):
        x = np.arange(0.0, 90.0, speed * 0.02)
        t = t0 + np.arange(x.size) * 0.02
        times.append(t)
        positions.append(np.c_[x, np.full_like(x, 5.0)])
        trials.append(Trial(t[0], t[-1] + 0.01, "start", "center", "reached"))
        t0 = t[-1] + 5.0
    args = (env, np.concatenate(times), np.vstack(positions))
    kwargs = {
        "region": "center",
        "trials": Intervals(trials),
        "min_speed": 5.0,
    }
    return args, kwargs, [t.size for t in times]


# Speed z-scores of the six straight runs: those of six evenly spaced values,
# (2.5 - k) / sqrt(35 / 12) for trial k, negated so the slowest scores highest.
_STRAIGHT_RUN_SPEED_Z = (2.5 - np.arange(6)) / np.sqrt(35 / 12)


def test_speed_only_index_classifies_straight_runs():
    """head_sweep_weight=0 asks for a speed-only index, so straight runs at
    different speeds are classified, with no head-sweep warning.
    """
    from neurospatial.behavior.vte import compute_vte_session

    args, kwargs, _ = _straight_runs()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compute_vte_session(*args, head_sweep_weight=0.0, **kwargs)
    expected = _STRAIGHT_RUN_SPEED_Z
    index = np.array([t.vte_index for t in result.trial_results])
    assert_allclose(index, expected, rtol=1e-6)
    assert [t.is_vte for t in result.trial_results] == list(expected >= 0.5)

    # Weighting head sweeps, which do not vary, still leaves them unclassified.
    with pytest.warns(UserWarning, match="No variation in head sweep"):
        weighted = compute_vte_session(*args, head_sweep_weight=0.5, **kwargs)
    assert all(t.is_vte is None for t in weighted.trial_results)


def test_speed_only_index_needs_no_headings():
    """A trial with no heading has no head sweep, which a speed-only index
    does not use: it is scored and classified by its speed alone."""
    from neurospatial.behavior.vte import compute_vte_session

    args, kwargs, n_samples = _straight_runs()
    headings = np.zeros(sum(n_samples))  # eastward, as the runs go
    headings[: n_samples[0]] = np.nan  # the slowest trial has no heading
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compute_vte_session(
            *args, headings=headings, head_sweep_weight=0.0, **kwargs
        )
    first = result.trial_results[0]
    assert first.status == "ok"
    assert np.isnan(first.head_sweep_magnitude)
    index = np.array([t.vte_index for t in result.trial_results])
    assert_allclose(index, _STRAIGHT_RUN_SPEED_Z, rtol=1e-6)
    assert first.is_vte is True  # the slowest trial, z = 1.46 >= 0.5
    assert [t.is_vte for t in result.trial_results] == list(
        _STRAIGHT_RUN_SPEED_Z >= 0.5
    )


# =============================================================================
# The pass window (Papale et al. 2012)
# =============================================================================


def _enter_sweep_exit_reenter():
    """East through a 40-60 cm zone, out, back in, at 1 cm per 30 Hz sample.

    Samples 0-9 approach (x = 30.5-39.5 cm), 10-29 cross the zone
    (x = 40.5-59.5) while zig-zagging in y, 30-34 leave it (x = 60.5-64.5),
    and 35-44 re-enter it heading west (x = 59.5-50.5), zig-zagging again.
    Returns ``(env, times, positions, headings)``; the headings are seeded
    random directions, so IdPhi over any span is known without the
    package's velocity code (``reference_head_sweep`` with ``headings``).
    """
    from shapely.geometry import box

    from neurospatial import Environment

    env = Environment.from_polygon(box(0, 0, 100, 20), bin_size=2.0, units="cm")
    regions = Regions()
    regions.add("choice", polygon=box(40, 0, 60, 20))
    env = env.with_regions(regions)
    x = np.r_[
        np.arange(30.5, 40.0),
        np.arange(40.5, 60.0),
        np.arange(60.5, 65.0),
        np.arange(59.5, 50.0, -1.0),
    ]
    zig = np.where(np.arange(x.size) % 2 == 0, 7.0, 13.0)
    y = np.where((x > 40) & (x < 60), zig, 10.0)
    times = np.arange(x.size) / 30
    headings = np.random.default_rng(0).uniform(-np.pi, np.pi, x.size)
    return env, times, np.column_stack([x, y]), headings


@pytest.mark.parametrize("method", METHODS)
def test_session_window_is_the_first_contiguous_pass(method):
    """IdPhi is integrated over the first pass only; the re-entry is not used.

    The adaptive window estimates dPhi from the headings over the whole
    recording; the line fit sums the changes of the pass's headings alone.
    """
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.behavior.vte import compute_vte_session

    env, times, positions, headings = _enter_sweep_exit_reenter()
    in_zone = np.isin(env.bin_at(positions), env.bins_in_region("choice"))
    np.testing.assert_array_equal(
        np.flatnonzero(in_zone), np.r_[np.arange(10, 30), np.arange(35, 45)]
    )
    trials = Intervals([Trial(0.0, times[-1] + 0.01, "start", "choice", "reached")])
    result = compute_vte_session(
        env,
        times,
        positions,
        region="choice",
        trials=trials,
        headings=headings,
        min_speed=1.0,
        method=method,
    )

    (trial,) = result.trial_results
    assert trial.status == "ok"
    assert trial.window_start == times[10]
    assert trial.window_end == times[30]  # the first sample past the zone

    def idphi(samples):
        if method == "line_fit":
            return _idphi(headings[samples])
        return reference_head_sweep(
            times, positions, 1.0, headings=headings, samples=samples
        )

    assert_allclose(trial.head_sweep_magnitude, idphi(slice(10, 30)), rtol=1e-12)
    # Every in-zone sample, re-entry included, would give another IdPhi.
    in_zone = idphi(slice(10, 30)) + idphi(slice(35, 45))
    assert not np.isclose(trial.head_sweep_magnitude, in_zone)


@pytest.mark.parametrize("method", METHODS)
def test_gap_longer_than_max_gap_ends_the_pass(method):
    """A tracking loss inside the pass ends the window at its last sample."""
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.behavior.vte import compute_vte_session

    env, times, positions, headings = _enter_sweep_exit_reenter()
    times = times.copy()
    times[21:] += 2.0  # a 2 s loss between samples 20 and 21, inside the pass
    trials = Intervals([Trial(0.0, times[-1] + 0.01, "start", "choice", "reached")])
    kwargs = {
        "region": "choice",
        "trials": trials,
        "headings": headings,
        "min_speed": 1.0,
        "method": method,
    }
    (cut,) = compute_vte_session(env, times, positions, **kwargs).trial_results
    (bridged,) = compute_vte_session(
        env, times, positions, max_gap=None, **kwargs
    ).trial_results

    assert (cut.window_start, cut.window_end) == (times[10], times[21])
    if method == "line_fit":
        # The pass's own headings, before and across the gap.
        assert_allclose(cut.head_sweep_magnitude, _idphi(headings[10:21]), rtol=1e-12)
        assert (bridged.window_start, bridged.window_end) == (times[10], times[30])
        assert_allclose(
            bridged.head_sweep_magnitude, _idphi(headings[10:30]), rtol=1e-12
        )
        return
    # Cut: the pass is measured within the run that ends at sample 20.
    before = reference_head_sweep(
        times[:21], positions[:21], 1.0, headings=headings[:21], samples=slice(10, 21)
    )
    assert_allclose(cut.head_sweep_magnitude, before, rtol=1e-12)
    # Bridged: one run, whose estimate spans the 2 s gap.
    assert (bridged.window_start, bridged.window_end) == (times[10], times[30])
    across = reference_head_sweep(
        times, positions, 1.0, headings=headings, samples=slice(10, 30)
    )
    assert_allclose(bridged.head_sweep_magnitude, across, rtol=1e-12)


def test_default_vte_index_is_z_idphi(t_maze_environment):
    """By default the VTE index is the head-sweep z-score alone (zIdPhi)."""
    import inspect

    from neurospatial.behavior.vte import compute_vte_session, vte_index

    assert (
        inspect.signature(compute_vte_session).parameters["head_sweep_weight"].default
        == 1.0
    )
    assert inspect.signature(vte_index).parameters["head_sweep_weight"].default == 1.0
    times, positions, trials = _t_maze_session([2.0, 4.0, 0.5, 7.0])
    result = compute_vte_session(
        t_maze_environment,
        times,
        positions,
        region="choice",
        trials=trials,
        min_speed=5.0,
    )
    table = result.trials_table()
    sweeps = table["head_sweep_magnitude"].to_numpy()
    z_idphi = (sweeps - sweeps.mean()) / sweeps.std()
    assert_allclose(table["z_head_sweep"], z_idphi, rtol=1e-12)
    np.testing.assert_array_equal(table["vte_index"], table["z_head_sweep"])
    assert table["is_vte"].tolist() == list(z_idphi >= 0.5)
    # The speed term is still computed, but it does not enter the index.
    assert table["z_speed_inverse"].notna().all()


@pytest.mark.parametrize(
    ("start_time", "stop_time", "match"),
    [
        (np.nan, 2.0, "finite start_time < stop_time"),
        (1.0, np.inf, "finite start_time < stop_time"),
        (2.0, 2.0, "finite start_time < stop_time"),
        (2.0, 1.0, "finite start_time < stop_time"),
        (-1.0, 1.0, "outside the recording"),
        (2000.0, 3000.0, "outside the recording"),
    ],
)
def test_vte_trial_window_bounds_are_validated(start_time, stop_time, match):
    """A NaN, empty or reversed span, or a start the recording misses, raises."""
    from neurospatial.behavior.vte import compute_vte_trial

    times, positions = _zigzag(30)
    with pytest.raises(ValueError, match=match):
        compute_vte_trial(times, positions, start_time, stop_time, min_speed=1.0)


@pytest.mark.parametrize("method", METHODS)
def test_vte_trial_measures_the_session_pass(method):
    """The session's window bounds, passed to compute_vte_trial, measure the
    same samples: the same head sweep and speed."""
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.behavior.vte import compute_vte_session, compute_vte_trial

    env, times, positions, headings = _enter_sweep_exit_reenter()
    trials = Intervals([Trial(0.0, times[-1] + 0.01, "start", "choice", "reached")])
    (pass_,) = compute_vte_session(
        env,
        times,
        positions,
        region="choice",
        trials=trials,
        headings=headings,
        min_speed=1.0,
        method=method,
    ).trial_results
    alone = compute_vte_trial(
        times,
        positions,
        pass_.window_start,
        pass_.window_end,
        headings=headings,
        min_speed=1.0,
        method=method,
    )
    assert alone.head_sweep_magnitude == pass_.head_sweep_magnitude
    assert alone.mean_speed == pass_.mean_speed
    assert alone.min_observed_speed == pass_.min_observed_speed


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("seed", range(5))
def test_jittered_pause_does_not_score_as_head_sweeping(seed, method):
    """Tracking jitter on a still or slowly moving animal is not VTE.

    Jitter sets the heading of a slow animal, with either estimator: without
    the speed gate, over seeds 0-9, a still animal scores 5-17 rad and a
    2 cm/s creep 5-12 rad against 3.5-4.9 rad for the scan with the adaptive
    window, and a still animal about 16 rad against 8 rad with the line fit.

    With the gate at most one sample of a still animal passes (the first,
    where the velocity fit is one-sided), and one sample has no turn, so no
    head sweep was measured: NaN, not 0. A straight creep at 1.5 times the
    gate is the slow movement the gate lets through: the jitter moves its
    estimated speed by up to 1.9 cm/s (seeds 0-9), so every sample stays
    above the gate and its head sweep is a number, all of it jitter. It is
    below the scan's: 0.2-2.6 rad against 3.5-4.9 rad with the adaptive
    window (6-53% of the scan) and 0.7-1.9 against 7.8-8.2 rad with the line
    fit (9-23%), over seeds 0-9. It is not always far below: with the
    adaptive window seed 3 reaches half of the scan.
    """
    from neurospatial.behavior import compute_speed, compute_vte_trial

    gate = 5.0  # cm/s
    rng = np.random.default_rng(seed)
    times = np.arange(45) / 30.0  # 1.5 s at 30 Hz
    jitter = rng.normal(0.0, 0.3, (times.size, 2))  # cm
    still = np.zeros((times.size, 2)) + jitter
    creep = np.column_stack([1.5 * gate * times, np.zeros_like(times)]) + jitter
    scan = np.column_stack([10.0 * times, 8.0 * np.sin(2 * np.pi * times)]) + jitter

    def sweep(xy):
        return compute_vte_trial(
            times, xy, 0.0, 1.5, min_speed=gate, method=method
        ).head_sweep_magnitude

    # Without jitter, the scan's IdPhi is that of the reference estimator.
    clean = scan - jitter
    clean_sweep = sweep(clean)
    reference = (
        reference_head_sweep(times, clean, gate)
        if method == "adaptive_window"
        else _line_fit_head_sweep(times, clean, gate)
    )
    assert_allclose(clean_sweep, reference, rtol=1e-10)
    scan_sweep = sweep(scan)
    # The clean scan sweeps 3.8 rad with the adaptive window (which averages
    # away about half of a 1 Hz scan's turn; jitter moves it to 3.5-4.9 rad
    # over seeds 0-9) and 7.9 rad with the line fit (whose velocity error
    # from the jitter, about 0.6 cm/s, is small against the scan's 10 cm/s or
    # more). Every scan sample stays above the gate, and 3/4 of the clean
    # value is still a strong sweep.
    assert scan_sweep > 0.75 * clean_sweep
    # At most the first sample of the still animal reaches the gate (seed 3,
    # where the fit is one-sided), and one sample alone has no turn.
    assert (compute_speed(times, still)[1:] < gate).all()
    assert np.isnan(sweep(still))
    # Every sample of the creep passes the gate, so its sweep is compared.
    assert (compute_speed(times, creep) >= gate).all()
    assert sweep(creep) < scan_sweep
