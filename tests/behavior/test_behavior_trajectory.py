"""Tests for behavior/trajectory.py module - TDD RED phase.

This test file verifies that all trajectory functions are importable from
the new location: neurospatial.behavior.trajectory

Following TDD: Tests written FIRST, then implementation.
"""

import numpy as np
from numpy.testing import assert_allclose


class TestImportsFromNewLocation:
    """Verify all functions are importable from behavior.trajectory."""

    def test_import_compute_turn_angles(self):
        """Test compute_turn_angles is importable from new location."""
        from neurospatial.behavior.trajectory import compute_turn_angles

        assert callable(compute_turn_angles)

    def test_import_compute_step_lengths(self):
        """Test compute_step_lengths is importable from new location."""
        from neurospatial.behavior.trajectory import compute_step_lengths

        assert callable(compute_step_lengths)

    def test_import_compute_home_range(self):
        """Test compute_home_range is importable from new location."""
        from neurospatial.behavior.trajectory import compute_home_range

        assert callable(compute_home_range)

    def test_import_mean_square_displacement(self):
        """Test mean_square_displacement is importable from new location."""
        from neurospatial.behavior.trajectory import mean_square_displacement

        assert callable(mean_square_displacement)

    def test_import_compute_trajectory_curvature(self):
        """Test compute_trajectory_curvature is importable from new location."""
        from neurospatial.behavior.trajectory import compute_trajectory_curvature

        assert callable(compute_trajectory_curvature)


class TestImportsFromBehaviorInit:
    """Verify all functions are re-exported from behavior/__init__.py."""

    def test_import_compute_turn_angles_from_behavior(self):
        """Test compute_turn_angles is importable from behavior package."""
        from neurospatial.behavior import compute_turn_angles

        assert callable(compute_turn_angles)

    def test_import_compute_step_lengths_from_behavior(self):
        """Test compute_step_lengths is importable from behavior package."""
        from neurospatial.behavior import compute_step_lengths

        assert callable(compute_step_lengths)

    def test_import_compute_home_range_from_behavior(self):
        """Test compute_home_range is importable from behavior package."""
        from neurospatial.behavior import compute_home_range

        assert callable(compute_home_range)

    def test_import_mean_square_displacement_from_behavior(self):
        """Test mean_square_displacement is importable from behavior package."""
        from neurospatial.behavior import mean_square_displacement

        assert callable(mean_square_displacement)

    def test_import_compute_trajectory_curvature_from_behavior(self):
        """Test compute_trajectory_curvature is importable from behavior package."""
        from neurospatial.behavior import compute_trajectory_curvature

        assert callable(compute_trajectory_curvature)


class TestComputeTurnAnglesBasic:
    """Basic functionality tests for compute_turn_angles from new location."""

    def test_straight_line_trajectory(self):
        """Test that straight line movement has near-zero turn angles."""
        from neurospatial.behavior.trajectory import compute_turn_angles

        positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])
        angles = compute_turn_angles(positions)

        assert angles.shape == (18,)
        assert_allclose(angles, 0.0, atol=0.1)

    def test_turn_angles_range(self):
        """Test that turn angles are in [-π, π] range."""
        from neurospatial.behavior.trajectory import compute_turn_angles

        t = np.linspace(0, 4 * np.pi, 200)
        x = t * 5 + 20 * np.sin(t)
        y = 20 * np.cos(t)
        positions = np.column_stack([x, y])

        angles = compute_turn_angles(positions)

        assert np.all(angles >= -np.pi)
        assert np.all(angles <= np.pi)


class TestComputeStepLengthsBasic:
    """Basic functionality tests for compute_step_lengths from new location."""

    def test_step_lengths_straight_line(self):
        """Test step lengths on a straight trajectory."""
        from neurospatial.behavior.trajectory import compute_step_lengths

        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
        step_lengths = compute_step_lengths(positions, distance_type="euclidean")

        assert step_lengths.shape == (20,)
        assert np.all(step_lengths >= 0)
        assert_allclose(step_lengths, step_lengths[0], rtol=0.01)


class TestComputeHomeRangeBasic:
    """Basic functionality tests for compute_home_range from new location."""

    def test_home_range_95_percentile(self):
        """Test 95% home range selection."""
        from neurospatial.behavior.trajectory import compute_home_range

        trajectory_bins = np.concatenate(
            [
                np.repeat(0, 50),
                np.repeat(1, 30),
                np.repeat(2, 15),
                np.repeat(3, 5),
            ]
        )

        home_range = compute_home_range(trajectory_bins, percentile=95.0)

        assert set(home_range) == {0, 1, 2}


class TestMeanSquareDisplacementBasic:
    """Basic functionality tests for mean_square_displacement from new location."""

    def test_msd_shape(self):
        """Test that MSD returns two arrays (tau, msd)."""
        from neurospatial.behavior.trajectory import mean_square_displacement

        positions = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        times = np.linspace(0, 10, 50)

        tau_values, msd_values = mean_square_displacement(
            positions, times, distance_type="euclidean", max_tau=5.0
        )

        assert tau_values.ndim == 1
        assert msd_values.ndim == 1
        assert len(tau_values) == len(msd_values)


class TestComputeTrajectoryCurvatureBasic:
    """Basic functionality tests for compute_trajectory_curvature from new location."""

    def test_curvature_straight_line(self):
        """Test curvature for straight trajectory (should be ~0)."""
        from neurospatial.behavior.trajectory import compute_trajectory_curvature

        positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])
        curvature = compute_trajectory_curvature(positions)

        assert curvature.shape == (20,)
        assert np.allclose(curvature, 0.0, atol=0.01)

    def test_curvature_left_turn(self):
        """Test curvature for left turn (positive)."""
        from neurospatial.behavior.trajectory import compute_trajectory_curvature

        positions = np.array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
            ]
        )

        curvature = compute_trajectory_curvature(positions)

        assert curvature.shape == (3,)
        assert curvature[1] > 0  # Left turn is positive
        assert np.isclose(curvature[1], np.pi / 2, atol=0.1)

    def test_curvature_output_length_matches_input(self):
        """Test curvature output length matches input (n_samples)."""
        from neurospatial.behavior.trajectory import compute_trajectory_curvature

        rng = np.random.default_rng(42)
        for n_samples in [5, 10, 20, 100]:
            positions = rng.standard_normal((n_samples, 2)) * 10
            curvature = compute_trajectory_curvature(positions)
            assert curvature.shape == (n_samples,)
