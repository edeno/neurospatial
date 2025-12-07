"""Tests for reward field generation functionality."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from neurospatial import Environment
from neurospatial.behavior.reward import goal_reward_field, region_reward_field


class TestRegionRewardField:
    """Test suite for region_reward_field function."""

    def test_region_reward_field_constant(self):
        """Test constant (binary) reward in region."""
        # Create a simple 2D environment
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)

        # Add a goal region at center (buffer to create polygon with area)
        env.regions.add("goal_point", point=np.array([50.0, 50.0]))
        env.regions.buffer(
            "goal_point", distance=15.0, new_name="goal"
        )  # Create polygon around point

        # Generate constant reward field
        reward = region_reward_field(env, "goal", reward_value=1.0, decay="constant")

        # Check shape
        assert reward.shape == (env.n_bins,)

        # Get region mask to identify goal bins
        from neurospatial.ops import regions_to_mask

        goal_mask = regions_to_mask(env, "goal")

        # Bins inside region should have reward_value
        assert np.all(reward[goal_mask] == 1.0)

        # Bins outside region should have 0
        assert np.all(reward[~goal_mask] == 0.0)

    def test_region_reward_field_constant_custom_value(self):
        """Test constant reward with custom reward value."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)
        env.regions.add("goal_point", point=np.array([50.0, 50.0]))
        env.regions.buffer(
            "goal_point", distance=15.0, new_name="goal"
        )  # Create polygon around point

        reward_value = 10.0
        reward = region_reward_field(
            env, "goal", reward_value=reward_value, decay="constant"
        )

        from neurospatial.ops import regions_to_mask

        goal_mask = regions_to_mask(env, "goal")

        # Bins inside region should have custom reward_value
        assert np.all(reward[goal_mask] == reward_value)

    def test_region_reward_field_linear(self):
        """Test linear decay from boundary."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)
        env.regions.add("goal_point", point=np.array([50.0, 50.0]))
        env.regions.buffer(
            "goal_point", distance=15.0, new_name="goal"
        )  # Create polygon around point

        # Generate linear decay reward field
        reward = region_reward_field(env, "goal", reward_value=1.0, decay="linear")

        # Check shape
        assert reward.shape == (env.n_bins,)

        # Check that reward decreases with distance from region boundary
        from neurospatial.ops import regions_to_mask

        goal_mask = regions_to_mask(env, "goal")

        # Bins inside region should have maximum reward (1.0)
        assert np.all(reward[goal_mask] == 1.0)

        # Bins outside region should have decreasing reward with distance
        outside_bins = np.where(~goal_mask)[0]
        if len(outside_bins) > 0:
            # Reward should be between 0 and 1 for outside bins
            assert np.all(reward[outside_bins] >= 0.0)
            assert np.all(reward[outside_bins] <= 1.0)

    def test_region_reward_field_gaussian(self):
        """Test Gaussian falloff with smooth decay."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)
        env.regions.add("goal_point", point=np.array([50.0, 50.0]))
        env.regions.buffer(
            "goal_point", distance=15.0, new_name="goal"
        )  # Create polygon around point

        # Generate Gaussian reward field
        bandwidth = 10.0
        reward = region_reward_field(
            env, "goal", reward_value=1.0, decay="gaussian", bandwidth=bandwidth
        )

        # Check shape
        assert reward.shape == (env.n_bins,)

        # Check that reward is smoothly varying
        from neurospatial.ops import regions_to_mask

        goal_mask = regions_to_mask(env, "goal")

        # Peak reward in region should be 1.0 (after rescaling)
        assert np.max(reward[goal_mask]) == pytest.approx(1.0, abs=1e-6)

        # Reward should decay smoothly outside region
        assert np.all(reward >= 0.0)
        assert np.all(reward <= 1.0)

        # Should have smooth transitions (no sharp edges)
        # Check by looking at gradient magnitude
        reward_variance = np.var(reward)
        assert reward_variance > 0  # Not constant

    def test_region_reward_field_gaussian_no_bandwidth_error(self):
        """Test that Gaussian decay without bandwidth raises ValueError."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)
        env.regions.add("goal_point", point=np.array([50.0, 50.0]))
        env.regions.buffer(
            "goal_point", distance=15.0, new_name="goal"
        )  # Create polygon around point

        # Should raise ValueError when bandwidth not provided for Gaussian
        with pytest.raises(ValueError, match=r"bandwidth.*required.*gaussian|Gaussian"):
            region_reward_field(env, "goal", decay="gaussian", bandwidth=None)

    def test_region_reward_field_invalid_region_error(self):
        """Test that non-existent region raises KeyError."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)

        # No regions added
        with pytest.raises(KeyError, match="nonexistent"):
            region_reward_field(env, "nonexistent", decay="constant")

    def test_region_reward_field_parameter_naming(self):
        """Test that parameter is named 'decay' (not 'falloff')."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)
        env.regions.add("goal_point", point=np.array([50.0, 50.0]))
        env.regions.buffer(
            "goal_point", distance=15.0, new_name="goal"
        )  # Create polygon around point

        # This should work (correct parameter name)
        reward = region_reward_field(env, "goal", decay="constant")
        assert reward.shape == (env.n_bins,)

        # This should fail (wrong parameter name)
        with pytest.raises(TypeError):
            region_reward_field(env, "goal", falloff="constant")


class TestGoalRewardField:
    """Test suite for goal_reward_field function."""

    def test_goal_reward_field_exponential(self):
        """Test exponential decay from goal bins."""
        # Create simple 2D environment
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)

        # Select a goal bin near center
        center_bin = env.bin_at(np.array([[50.0, 50.0]]))[0]

        # Generate exponential decay field
        scale = 20.0
        reward = goal_reward_field(
            env, goal_bins=center_bin, decay="exponential", scale=scale
        )

        # Check shape
        assert reward.shape == (env.n_bins,)

        # Goal bin should have maximum reward (scale)
        assert reward[center_bin] == pytest.approx(scale, abs=1e-6)

        # All rewards should be positive
        assert np.all(reward >= 0.0)

        # Reward should decrease exponentially with distance
        # Check bins at different distances
        from neurospatial.ops.distance import distance_field

        distances = distance_field(env.connectivity, sources=[center_bin])

        # Expected: reward = scale * exp(-distances / scale)
        expected = scale * np.exp(-distances / scale)
        assert_array_almost_equal(reward, expected, decimal=6)

    def test_goal_reward_field_linear(self):
        """Test linear decay reaching zero at max distance."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)

        # Select a goal bin
        center_bin = env.bin_at(np.array([[50.0, 50.0]]))[0]

        # Generate linear decay field
        scale = 1.0
        max_distance = 50.0
        reward = goal_reward_field(
            env,
            goal_bins=center_bin,
            decay="linear",
            scale=scale,
            max_distance=max_distance,
        )

        # Check shape
        assert reward.shape == (env.n_bins,)

        # Goal bin should have maximum reward (scale)
        assert reward[center_bin] == pytest.approx(scale, abs=1e-6)

        # All rewards should be non-negative
        assert np.all(reward >= 0.0)

        # Bins beyond max_distance should have zero reward
        from neurospatial.ops.distance import distance_field

        distances = distance_field(env.connectivity, sources=[center_bin])
        far_bins = distances > max_distance
        if far_bins.any():
            assert np.all(reward[far_bins] == 0.0)

        # Bins within range should have reward = scale * (1 - d/max_d)
        close_bins = distances <= max_distance
        expected_close = scale * np.maximum(0, 1 - distances[close_bins] / max_distance)
        assert_array_almost_equal(reward[close_bins], expected_close, decimal=6)

    def test_goal_reward_field_inverse(self):
        """Test inverse distance formula."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)

        center_bin = env.bin_at(np.array([[50.0, 50.0]]))[0]

        # Generate inverse decay field
        scale = 10.0
        reward = goal_reward_field(
            env, goal_bins=center_bin, decay="inverse", scale=scale
        )

        # Check shape
        assert reward.shape == (env.n_bins,)

        # All rewards should be positive
        assert np.all(reward >= 0.0)

        # Verify inverse distance formula: scale / (1 + distance)
        from neurospatial.ops.distance import distance_field

        distances = distance_field(env.connectivity, sources=[center_bin])
        expected = scale / (1 + distances)
        assert_array_almost_equal(reward, expected, decimal=6)

    def test_goal_reward_field_multiple_goals(self):
        """Test that nearest goal dominates."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)

        # Select multiple goal bins
        goal_bins = [
            env.bin_at(np.array([[25.0, 25.0]]))[0],
            env.bin_at(np.array([[75.0, 75.0]]))[0],
        ]

        # Generate exponential decay field
        scale = 20.0
        reward = goal_reward_field(
            env, goal_bins=goal_bins, decay="exponential", scale=scale
        )

        # Check shape
        assert reward.shape == (env.n_bins,)

        # Both goal bins should have maximum reward
        assert reward[goal_bins[0]] == pytest.approx(scale, abs=1e-6)
        assert reward[goal_bins[1]] == pytest.approx(scale, abs=1e-6)

        # All rewards should be positive
        assert np.all(reward >= 0.0)

        # Verify that distance field uses nearest goal
        from neurospatial.ops.distance import distance_field

        distances = distance_field(env.connectivity, sources=goal_bins)
        expected = scale * np.exp(-distances / scale)
        assert_array_almost_equal(reward, expected, decimal=6)

    def test_goal_reward_field_scalar_goal_bin(self):
        """Test that scalar goal_bins is converted to array."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)

        center_bin = env.bin_at(np.array([[50.0, 50.0]]))[0]

        # Pass scalar (not list)
        reward = goal_reward_field(
            env, goal_bins=center_bin, decay="exponential", scale=20.0
        )

        assert reward.shape == (env.n_bins,)
        assert reward[center_bin] > 0

    def test_goal_reward_field_invalid_goal_bins_error(self):
        """Test that invalid goal bins raise error."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)

        # Out of range goal bin
        with pytest.raises((ValueError, IndexError)):
            goal_reward_field(env, goal_bins=[9999], decay="exponential", scale=1.0)

    def test_goal_reward_field_exponential_scale_positive(self):
        """Test that exponential decay requires positive scale."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)
        center_bin = env.bin_at(np.array([[50.0, 50.0]]))[0]

        # Negative scale should raise error for exponential
        with pytest.raises(ValueError, match=r"scale.*positive"):
            goal_reward_field(
                env, goal_bins=center_bin, decay="exponential", scale=-1.0
            )

    def test_goal_reward_field_parameter_naming(self):
        """Test that parameter is named 'decay' (not 'kind')."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)
        center_bin = env.bin_at(np.array([[50.0, 50.0]]))[0]

        # This should work (correct parameter name)
        reward = goal_reward_field(
            env, goal_bins=center_bin, decay="exponential", scale=20.0
        )
        assert reward.shape == (env.n_bins,)

        # This should fail (wrong parameter name)
        with pytest.raises(TypeError):
            goal_reward_field(env, goal_bins=center_bin, kind="exponential", scale=20.0)
