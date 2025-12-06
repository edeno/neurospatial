"""Tests for trajectory metrics with continuous position API."""

import numpy as np
import pytest

from neurospatial.behavior.trajectory import compute_turn_angles


class TestComputeTurnAngles:
    """Test compute_turn_angles function with continuous positions."""

    def test_straight_line_trajectory(self):
        """Test that straight line gives zero turn angles."""
        # Straight horizontal line
        positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])
        angles = compute_turn_angles(positions)

        # All angles should be near zero for straight movement
        assert np.allclose(angles, 0.0, atol=0.01)

    def test_right_angle_turn(self):
        """Test 90-degree right turn."""
        # Path: [0,0] → [10,0] → [10,10]
        positions = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        angles = compute_turn_angles(positions)

        # Should have one angle ≈ π/2 (90° left turn)
        assert len(angles) == 1
        assert np.isclose(angles[0], np.pi / 2, atol=0.01)

    def test_continuous_positions_required(self):
        """Test that function expects continuous positions, not bin indices."""
        rng = np.random.default_rng(42)
        # Create continuous trajectory
        positions = rng.standard_normal((100, 2)) * 10

        # This should work - continuous positions
        angles = compute_turn_angles(positions)
        assert len(angles) >= 0  # Valid result

        # Verify output is angles (not something else)
        assert np.all(np.abs(angles) <= np.pi)  # Angles in [-π, π]

    def test_position_shape_validation(self):
        """Test that positions must be 2D array (n_samples, n_dims)."""
        rng = np.random.default_rng(42)
        # 1D array should fail
        with pytest.raises(ValueError, match="positions must be 2D array"):
            compute_turn_angles(np.array([1, 2, 3]))

        # 3D array should fail
        with pytest.raises(ValueError, match="positions must be 2D array"):
            compute_turn_angles(rng.standard_normal((10, 5, 2)))

    def test_minimum_positions_required(self):
        """Test that function handles trajectories with <3 positions."""
        # 0 positions
        angles = compute_turn_angles(np.empty((0, 2)))
        assert len(angles) == 0

        # 1 position
        angles = compute_turn_angles(np.array([[0.0, 0.0]]))
        assert len(angles) == 0

        # 2 positions
        angles = compute_turn_angles(np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert len(angles) == 0

        # 3 positions should work
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        angles = compute_turn_angles(positions)
        assert len(angles) == 1

    def test_stationary_periods_filtered(self):
        """Test that consecutive duplicate positions are filtered out."""
        # Trajectory with stationary periods
        positions = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],  # Duplicate (stationary)
                [1.0, 0.0],  # Duplicate (stationary)
                [2.0, 1.0],
                [2.0, 1.0],  # Duplicate
                [3.0, 2.0],
            ]
        )

        angles = compute_turn_angles(positions)

        # Should have filtered duplicates: [0,0] → [1,0] → [2,1] → [3,2]
        # Two turn angles between 3 unique movement vectors
        assert len(angles) == 2

    def test_preserves_sub_bin_precision(self):
        """Test that continuous positions preserve directional precision."""
        # Gradual turn that would be missed with coarse binning
        positions = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.1],  # Slight upward movement
                [2.0, 0.3],  # More upward
                [3.0, 0.6],  # Even more upward
            ]
        )

        angles = compute_turn_angles(positions)

        # Should detect the gradual turning (non-zero angles)
        # All angles should be positive (gradual left turn)
        assert len(angles) == 2
        assert np.all(angles > 0)  # Gradual left turns

    def test_angle_range(self):
        """Test that output angles are in [-π, π]."""
        # Random trajectory
        rng = np.random.default_rng(42)
        positions = np.cumsum(rng.standard_normal((100, 2)), axis=0)

        angles = compute_turn_angles(positions)

        # All angles should be in valid range
        assert np.all(angles >= -np.pi)
        assert np.all(angles <= np.pi)

    def test_left_vs_right_turns(self):
        """Test that left and right turns have opposite signs."""
        # Left turn (counterclockwise): positive angle
        left_turn = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        left_angle = compute_turn_angles(left_turn)[0]

        # Right turn (clockwise): negative angle
        right_turn = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, -1.0]])
        right_angle = compute_turn_angles(right_turn)[0]

        assert left_angle > 0  # Left = positive
        assert right_angle < 0  # Right = negative
        assert np.isclose(abs(left_angle), abs(right_angle))  # Same magnitude

    def test_u_turn(self):
        """Test 180-degree U-turn."""
        # Forward then backward
        positions = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]])
        angles = compute_turn_angles(positions)

        # Should have one angle ≈ π (180°)
        assert len(angles) == 1
        assert np.isclose(abs(angles[0]), np.pi, atol=0.01)

    def test_1d_positions(self):
        """Test 1D trajectory (back-and-forth movement)."""
        # 1D positions
        positions = np.array([[0.0], [1.0], [2.0], [1.0], [0.0]])
        angles = compute_turn_angles(positions)

        # 1D angles should be 0 (same direction) or π (opposite direction)
        assert len(angles) == 3
        # Vectors: [1.0] (0→1), [1.0] (1→2), [-1.0] (2→1), [-1.0] (1→0)
        # Turn angles between consecutive vectors:
        assert np.isclose(angles[0], 0.0, atol=0.01)  # v1→v2: same direction
        assert np.isclose(abs(angles[1]), np.pi, atol=0.01)  # v2→v3: U-turn
        assert np.isclose(angles[2], 0.0, atol=0.01)  # v3→v4: same direction

    def test_realistic_trajectory(self):
        """Test with realistic animal trajectory."""
        # Simulate random walk
        rng = np.random.default_rng(42)
        n_steps = 200
        step_angles = rng.standard_normal(n_steps) * 0.3  # Small angle changes
        step_lengths = np.abs(rng.standard_normal(n_steps)) * 2

        # Build trajectory from steps
        cumulative_angle = np.cumsum(step_angles)
        dx = step_lengths * np.cos(cumulative_angle)
        dy = step_lengths * np.sin(cumulative_angle)
        positions = np.column_stack([np.cumsum(dx), np.cumsum(dy)])

        # Should compute without error
        angles = compute_turn_angles(positions)

        # Should have reasonable number of angles
        assert len(angles) > 0
        assert len(angles) < len(positions)

        # Angles should be reasonable (not all identical)
        assert np.std(angles) > 0.1  # Some variability in turning
