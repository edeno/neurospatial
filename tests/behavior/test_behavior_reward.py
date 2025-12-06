"""Tests for behavior.reward module - reward field functions.

This module tests that reward functions are correctly importable from
the new behavior.reward location.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal


class TestBehaviorRewardImports:
    """Test that all reward functions are importable from behavior.reward."""

    def test_import_goal_reward_field(self):
        """Test that goal_reward_field is importable from behavior.reward."""
        from neurospatial.behavior.reward import goal_reward_field

        assert callable(goal_reward_field)

    def test_import_region_reward_field(self):
        """Test that region_reward_field is importable from behavior.reward."""
        from neurospatial.behavior.reward import region_reward_field

        assert callable(region_reward_field)

    def test_import_from_behavior_init(self):
        """Test that reward functions are exported from behavior/__init__.py."""
        from neurospatial.behavior import goal_reward_field, region_reward_field

        assert callable(goal_reward_field)
        assert callable(region_reward_field)


class TestRegionRewardFieldBehavior:
    """Tests for region_reward_field from behavior.reward location."""

    def test_region_reward_field_constant(self):
        """Test constant (binary) reward in region."""
        from neurospatial import Environment
        from neurospatial.behavior.reward import region_reward_field
        from neurospatial.ops.binning import regions_to_mask

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
        env.regions.buffer("goal_point", distance=15.0, new_name="goal")

        # Generate constant reward field
        reward = region_reward_field(env, "goal", reward_value=1.0, decay="constant")

        # Check shape
        assert reward.shape == (env.n_bins,)

        # Get region mask to identify goal bins
        goal_mask = regions_to_mask(env, "goal")

        # Bins inside region should have reward_value
        assert np.all(reward[goal_mask] == 1.0)

        # Bins outside region should have 0
        assert np.all(reward[~goal_mask] == 0.0)

    def test_region_reward_field_linear(self):
        """Test linear decay from boundary."""
        from neurospatial import Environment
        from neurospatial.behavior.reward import region_reward_field
        from neurospatial.ops.binning import regions_to_mask

        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)
        env.regions.add("goal_point", point=np.array([50.0, 50.0]))
        env.regions.buffer("goal_point", distance=15.0, new_name="goal")

        # Generate linear decay reward field
        reward = region_reward_field(env, "goal", reward_value=1.0, decay="linear")

        # Check shape
        assert reward.shape == (env.n_bins,)

        goal_mask = regions_to_mask(env, "goal")

        # Bins inside region should have maximum reward (1.0)
        assert np.all(reward[goal_mask] == 1.0)

        # Bins outside region should have decreasing reward with distance
        outside_bins = np.where(~goal_mask)[0]
        if len(outside_bins) > 0:
            assert np.all(reward[outside_bins] >= 0.0)
            assert np.all(reward[outside_bins] <= 1.0)

    def test_region_reward_field_gaussian(self):
        """Test Gaussian falloff with smooth decay."""
        from neurospatial import Environment
        from neurospatial.behavior.reward import region_reward_field
        from neurospatial.ops.binning import regions_to_mask

        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)
        env.regions.add("goal_point", point=np.array([50.0, 50.0]))
        env.regions.buffer("goal_point", distance=15.0, new_name="goal")

        # Generate Gaussian reward field
        bandwidth = 10.0
        reward = region_reward_field(
            env, "goal", reward_value=1.0, decay="gaussian", bandwidth=bandwidth
        )

        # Check shape
        assert reward.shape == (env.n_bins,)

        goal_mask = regions_to_mask(env, "goal")

        # Peak reward in region should be 1.0 (after rescaling)
        assert np.max(reward[goal_mask]) == pytest.approx(1.0, abs=1e-6)

        # Reward should decay smoothly outside region
        assert np.all(reward >= 0.0)
        assert np.all(reward <= 1.0)

    def test_region_reward_field_gaussian_no_bandwidth_error(self):
        """Test that Gaussian decay without bandwidth raises ValueError."""
        from neurospatial import Environment
        from neurospatial.behavior.reward import region_reward_field

        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)
        env.regions.add("goal_point", point=np.array([50.0, 50.0]))
        env.regions.buffer("goal_point", distance=15.0, new_name="goal")

        with pytest.raises(ValueError, match=r"bandwidth.*required.*gaussian|Gaussian"):
            region_reward_field(env, "goal", decay="gaussian", bandwidth=None)

    def test_region_reward_field_invalid_region_error(self):
        """Test that non-existent region raises KeyError."""
        from neurospatial import Environment
        from neurospatial.behavior.reward import region_reward_field

        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)

        with pytest.raises(KeyError, match="nonexistent"):
            region_reward_field(env, "nonexistent", decay="constant")


class TestGoalRewardFieldBehavior:
    """Tests for goal_reward_field from behavior.reward location."""

    def test_goal_reward_field_exponential(self):
        """Test exponential decay from goal bins."""
        from neurospatial import Environment
        from neurospatial.behavior.reward import goal_reward_field
        from neurospatial.ops.distance import distance_field

        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)

        center_bin = env.bin_at(np.array([[50.0, 50.0]]))[0]

        scale = 20.0
        reward = goal_reward_field(
            env, goal_bins=center_bin, decay="exponential", scale=scale
        )

        assert reward.shape == (env.n_bins,)
        assert reward[center_bin] == pytest.approx(scale, abs=1e-6)
        assert np.all(reward >= 0.0)

        # Verify exponential formula
        distances = distance_field(env.connectivity, sources=[center_bin])
        expected = scale * np.exp(-distances / scale)
        assert_array_almost_equal(reward, expected, decimal=6)

    def test_goal_reward_field_linear(self):
        """Test linear decay reaching zero at max distance."""
        from neurospatial import Environment
        from neurospatial.behavior.reward import goal_reward_field
        from neurospatial.ops.distance import distance_field

        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)

        center_bin = env.bin_at(np.array([[50.0, 50.0]]))[0]

        scale = 1.0
        max_distance = 50.0
        reward = goal_reward_field(
            env,
            goal_bins=center_bin,
            decay="linear",
            scale=scale,
            max_distance=max_distance,
        )

        assert reward.shape == (env.n_bins,)
        assert reward[center_bin] == pytest.approx(scale, abs=1e-6)
        assert np.all(reward >= 0.0)

        # Bins beyond max_distance should have zero reward
        distances = distance_field(env.connectivity, sources=[center_bin])
        far_bins = distances > max_distance
        if far_bins.any():
            assert np.all(reward[far_bins] == 0.0)

    def test_goal_reward_field_inverse(self):
        """Test inverse distance formula."""
        from neurospatial import Environment
        from neurospatial.behavior.reward import goal_reward_field
        from neurospatial.ops.distance import distance_field

        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)

        center_bin = env.bin_at(np.array([[50.0, 50.0]]))[0]

        scale = 10.0
        reward = goal_reward_field(
            env, goal_bins=center_bin, decay="inverse", scale=scale
        )

        assert reward.shape == (env.n_bins,)
        assert np.all(reward >= 0.0)

        # Verify inverse distance formula: scale / (1 + distance)
        distances = distance_field(env.connectivity, sources=[center_bin])
        expected = scale / (1 + distances)
        assert_array_almost_equal(reward, expected, decimal=6)

    def test_goal_reward_field_multiple_goals(self):
        """Test that nearest goal dominates."""
        from neurospatial import Environment
        from neurospatial.behavior.reward import goal_reward_field
        from neurospatial.ops.distance import distance_field

        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)

        goal_bins = [
            env.bin_at(np.array([[25.0, 25.0]]))[0],
            env.bin_at(np.array([[75.0, 75.0]]))[0],
        ]

        scale = 20.0
        reward = goal_reward_field(
            env, goal_bins=goal_bins, decay="exponential", scale=scale
        )

        assert reward.shape == (env.n_bins,)
        assert reward[goal_bins[0]] == pytest.approx(scale, abs=1e-6)
        assert reward[goal_bins[1]] == pytest.approx(scale, abs=1e-6)
        assert np.all(reward >= 0.0)

        # Verify that distance field uses nearest goal
        distances = distance_field(env.connectivity, sources=goal_bins)
        expected = scale * np.exp(-distances / scale)
        assert_array_almost_equal(reward, expected, decimal=6)

    def test_goal_reward_field_invalid_goal_bins_error(self):
        """Test that invalid goal bins raise error."""
        from neurospatial import Environment
        from neurospatial.behavior.reward import goal_reward_field

        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)

        with pytest.raises((ValueError, IndexError)):
            goal_reward_field(env, goal_bins=[9999], decay="exponential", scale=1.0)

    def test_goal_reward_field_exponential_scale_positive(self):
        """Test that exponential decay requires positive scale."""
        from neurospatial import Environment
        from neurospatial.behavior.reward import goal_reward_field

        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)
        center_bin = env.bin_at(np.array([[50.0, 50.0]]))[0]

        with pytest.raises(ValueError, match=r"scale.*positive"):
            goal_reward_field(
                env, goal_bins=center_bin, decay="exponential", scale=-1.0
            )
