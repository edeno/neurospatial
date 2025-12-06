"""Tests for behavior/navigation.py module (new location).

This test file verifies that all navigation functions are importable from
the new location and work correctly. These tests follow TDD - written before
the implementation is moved.

Functions being moved:
- From behavioral.py: path_progress, distance_to_region, cost_to_goal,
  time_to_goal, trials_to_region_arrays, graph_turn_sequence,
  goal_pair_direction_labels, heading_direction_labels,
  compute_trajectory_curvature (already in behavior/trajectory.py)
- From metrics/path_efficiency.py: PathEfficiencyResult, SubgoalEfficiencyResult,
  traveled_path_length, shortest_path_length, path_efficiency, time_efficiency,
  angular_efficiency, subgoal_efficiency, compute_path_efficiency
- From metrics/goal_directed.py: GoalDirectedMetrics, goal_vector, goal_direction,
  instantaneous_goal_alignment, goal_bias, approach_rate, compute_goal_directed_metrics
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestNavigationImports:
    """Test that all navigation functions are importable from new location."""

    def test_import_path_progress(self):
        """Test path_progress is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import path_progress

        assert callable(path_progress)

    def test_import_distance_to_region(self):
        """Test distance_to_region is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import distance_to_region

        assert callable(distance_to_region)

    def test_import_cost_to_goal(self):
        """Test cost_to_goal is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import cost_to_goal

        assert callable(cost_to_goal)

    def test_import_time_to_goal(self):
        """Test time_to_goal is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import time_to_goal

        assert callable(time_to_goal)

    def test_import_trials_to_region_arrays(self):
        """Test trials_to_region_arrays is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import trials_to_region_arrays

        assert callable(trials_to_region_arrays)

    def test_import_graph_turn_sequence(self):
        """Test graph_turn_sequence is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import graph_turn_sequence

        assert callable(graph_turn_sequence)

    def test_import_goal_pair_direction_labels(self):
        """Test goal_pair_direction_labels is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import goal_pair_direction_labels

        assert callable(goal_pair_direction_labels)

    def test_import_heading_direction_labels(self):
        """Test heading_direction_labels is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import heading_direction_labels

        assert callable(heading_direction_labels)

    def test_import_path_efficiency_result(self):
        """Test PathEfficiencyResult is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import PathEfficiencyResult

        assert PathEfficiencyResult is not None

    def test_import_subgoal_efficiency_result(self):
        """Test SubgoalEfficiencyResult is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import SubgoalEfficiencyResult

        assert SubgoalEfficiencyResult is not None

    def test_import_traveled_path_length(self):
        """Test traveled_path_length is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import traveled_path_length

        assert callable(traveled_path_length)

    def test_import_shortest_path_length(self):
        """Test shortest_path_length is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import shortest_path_length

        assert callable(shortest_path_length)

    def test_import_path_efficiency(self):
        """Test path_efficiency is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import path_efficiency

        assert callable(path_efficiency)

    def test_import_time_efficiency(self):
        """Test time_efficiency is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import time_efficiency

        assert callable(time_efficiency)

    def test_import_angular_efficiency(self):
        """Test angular_efficiency is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import angular_efficiency

        assert callable(angular_efficiency)

    def test_import_subgoal_efficiency(self):
        """Test subgoal_efficiency is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import subgoal_efficiency

        assert callable(subgoal_efficiency)

    def test_import_compute_path_efficiency(self):
        """Test compute_path_efficiency is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import compute_path_efficiency

        assert callable(compute_path_efficiency)

    def test_import_goal_directed_metrics(self):
        """Test GoalDirectedMetrics is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import GoalDirectedMetrics

        assert GoalDirectedMetrics is not None

    def test_import_goal_vector(self):
        """Test goal_vector is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import goal_vector

        assert callable(goal_vector)

    def test_import_goal_direction(self):
        """Test goal_direction is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import goal_direction

        assert callable(goal_direction)

    def test_import_instantaneous_goal_alignment(self):
        """Test instantaneous_goal_alignment is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import instantaneous_goal_alignment

        assert callable(instantaneous_goal_alignment)

    def test_import_goal_bias(self):
        """Test goal_bias is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import goal_bias

        assert callable(goal_bias)

    def test_import_approach_rate(self):
        """Test approach_rate is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import approach_rate

        assert callable(approach_rate)

    def test_import_compute_goal_directed_metrics(self):
        """Test compute_goal_directed_metrics is importable from behavior.navigation."""
        from neurospatial.behavior.navigation import compute_goal_directed_metrics

        assert callable(compute_goal_directed_metrics)


class TestBehaviorModuleReExports:
    """Test that navigation functions are re-exported from behavior/__init__.py."""

    def test_import_path_progress_from_behavior(self):
        """Test path_progress is re-exported from behavior module."""
        from neurospatial.behavior import path_progress

        assert callable(path_progress)

    def test_import_distance_to_region_from_behavior(self):
        """Test distance_to_region is re-exported from behavior module."""
        from neurospatial.behavior import distance_to_region

        assert callable(distance_to_region)

    def test_import_cost_to_goal_from_behavior(self):
        """Test cost_to_goal is re-exported from behavior module."""
        from neurospatial.behavior import cost_to_goal

        assert callable(cost_to_goal)

    def test_import_time_to_goal_from_behavior(self):
        """Test time_to_goal is re-exported from behavior module."""
        from neurospatial.behavior import time_to_goal

        assert callable(time_to_goal)

    def test_import_path_efficiency_from_behavior(self):
        """Test path_efficiency is re-exported from behavior module."""
        from neurospatial.behavior import path_efficiency

        assert callable(path_efficiency)

    def test_import_goal_bias_from_behavior(self):
        """Test goal_bias is re-exported from behavior module."""
        from neurospatial.behavior import goal_bias

        assert callable(goal_bias)


class TestNavigationFunctionality:
    """Test basic functionality of navigation functions."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        from neurospatial import Environment

        np.random.seed(42)
        positions = np.random.rand(100, 2) * 100
        return Environment.from_samples(positions, bin_size=10.0)

    def test_goal_vector_basic(self):
        """Test goal_vector computes correct vectors."""
        from neurospatial.behavior.navigation import goal_vector

        positions = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        goal = np.array([50.0, 50.0])

        vectors = goal_vector(positions, goal)

        assert vectors.shape == (3, 2)
        assert_allclose(vectors[0], [50.0, 50.0])
        assert_allclose(vectors[1], [40.0, 50.0])
        assert_allclose(vectors[2], [40.0, 40.0])

    def test_goal_direction_basic(self):
        """Test goal_direction computes correct angles."""
        from neurospatial.behavior.navigation import goal_direction

        positions = np.array([[0.0, 0.0]])
        goal_east = np.array([1.0, 0.0])
        goal_north = np.array([0.0, 1.0])

        direction_east = goal_direction(positions, goal_east)
        direction_north = goal_direction(positions, goal_north)

        assert_allclose(direction_east[0], 0.0, atol=1e-10)
        assert_allclose(direction_north[0], np.pi / 2, atol=1e-10)

    def test_traveled_path_length_basic(self):
        """Test traveled_path_length computes correct length."""
        from neurospatial.behavior.navigation import traveled_path_length

        # Straight line path
        positions = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        length = traveled_path_length(positions)

        assert_allclose(length, 20.0)

    def test_angular_efficiency_straight_path(self):
        """Test angular_efficiency is 1.0 for straight path."""
        from neurospatial.behavior.navigation import angular_efficiency

        positions = np.array(
            [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]]
        )
        goal = np.array([100.0, 0.0])

        eff = angular_efficiency(positions, goal)
        assert eff >= 0.95  # Should be close to 1.0

    def test_path_efficiency_result_summary(self):
        """Test PathEfficiencyResult has working summary method."""
        from neurospatial.behavior.navigation import PathEfficiencyResult

        result = PathEfficiencyResult(
            traveled_length=45.2,
            shortest_length=32.1,
            efficiency=0.71,
            time_efficiency=None,
            angular_efficiency=0.85,
            start_position=np.array([0.0, 0.0]),
            goal_position=np.array([30.0, 10.0]),
            metric="geodesic",
        )

        summary = result.summary()
        assert "45.2" in summary
        assert "32.1" in summary
        assert "71" in summary

    def test_goal_directed_metrics_summary(self):
        """Test GoalDirectedMetrics has working summary method."""
        from neurospatial.behavior.navigation import GoalDirectedMetrics

        result = GoalDirectedMetrics(
            goal_bias=0.65,
            mean_approach_rate=-8.5,
            time_to_goal=5.2,
            min_distance_to_goal=2.1,
            goal_distance_at_start=50.0,
            goal_distance_at_end=2.1,
            goal_position=np.array([50.0, 50.0]),
            metric="euclidean",
        )

        summary = result.summary()
        assert "0.65" in summary
        assert "-8.5" in summary

    def test_heading_direction_labels_from_positions(self):
        """Test heading_direction_labels generates labels from positions."""
        from neurospatial.behavior.navigation import heading_direction_labels

        # Create a simple trajectory moving east
        times = np.linspace(0, 10, 100)
        positions = np.column_stack(
            [np.linspace(0, 100, 100), np.zeros(100)]  # Moving east
        )

        labels = heading_direction_labels(
            positions=positions, times=times, min_speed=0.1
        )

        assert len(labels) == 100
        # Most labels should be in the 0-degree range (moving east)
        assert "stationary" in labels or any("0" in str(label) for label in labels[1:])

    def test_path_efficiency_with_env(self, simple_env):
        """Test path_efficiency works with environment."""
        from neurospatial.behavior.navigation import path_efficiency

        # Create a trajectory
        positions = np.array(
            [[10.0, 10.0], [20.0, 15.0], [30.0, 20.0], [40.0, 25.0], [50.0, 30.0]]
        )
        goal = np.array([50.0, 30.0])

        eff = path_efficiency(simple_env, positions, goal, metric="euclidean")

        # Efficiency should be between 0 and 1
        assert 0.0 < eff <= 1.0

    def test_compute_goal_directed_metrics_basic(self, simple_env):
        """Test compute_goal_directed_metrics returns valid metrics."""
        from neurospatial.behavior.navigation import compute_goal_directed_metrics

        positions = np.column_stack(
            [np.linspace(0, 50, 51), np.linspace(0, 50, 51)]  # Diagonal movement
        )
        times = np.linspace(0, 10, 51)
        goal = np.array([50.0, 50.0])

        result = compute_goal_directed_metrics(
            simple_env, positions, times, goal, min_speed=0.1
        )

        assert hasattr(result, "goal_bias")
        assert hasattr(result, "mean_approach_rate")
        assert hasattr(result, "min_distance_to_goal")
        assert -1.0 <= result.goal_bias <= 1.0


class TestPathEfficiencyFunctions:
    """Test path efficiency functions."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        from neurospatial import Environment

        np.random.seed(42)
        positions = np.random.rand(100, 2) * 100
        return Environment.from_samples(positions, bin_size=10.0)

    def test_shortest_path_length_euclidean(self, simple_env):
        """Test shortest_path_length with Euclidean metric."""
        from neurospatial.behavior.navigation import shortest_path_length

        start = np.array([0.0, 0.0])
        goal = np.array([30.0, 40.0])  # 3-4-5 triangle

        dist = shortest_path_length(simple_env, start, goal, metric="euclidean")
        assert_allclose(dist, 50.0)

    def test_time_efficiency_basic(self):
        """Test time_efficiency computation."""
        from neurospatial.behavior.navigation import time_efficiency

        # Moving at 10 units/s toward goal 50 units away
        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        times = np.linspace(0, 5, 11)  # 5 seconds
        goal = np.array([50.0, 0.0])

        eff = time_efficiency(positions, times, goal, reference_speed=10.0)

        # Optimal time = 50/10 = 5s, actual = 5s, efficiency = 1.0
        assert_allclose(eff, 1.0, atol=0.01)

    def test_compute_path_efficiency_comprehensive(self, simple_env):
        """Test compute_path_efficiency returns all metrics."""
        from neurospatial.behavior.navigation import compute_path_efficiency

        positions = np.column_stack([np.linspace(10, 90, 50), np.linspace(10, 90, 50)])
        times = np.linspace(0, 10, 50)
        goal = np.array([90.0, 90.0])

        result = compute_path_efficiency(
            simple_env, positions, times, goal, metric="euclidean", reference_speed=10.0
        )

        assert hasattr(result, "efficiency")
        assert hasattr(result, "time_efficiency")
        assert hasattr(result, "angular_efficiency")
        assert hasattr(result, "traveled_length")
        assert hasattr(result, "shortest_length")


class TestGoalDirectedFunctions:
    """Test goal-directed navigation functions."""

    def test_approach_rate_moving_toward_goal(self):
        """Test approach_rate is negative when moving toward goal."""
        from neurospatial.behavior.navigation import approach_rate

        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        times = np.linspace(0, 5, 11)
        goal = np.array([100.0, 0.0])

        rates = approach_rate(positions, times, goal)

        # Should be negative (distance decreasing)
        assert np.nanmean(rates) < 0

    def test_goal_bias_toward_goal(self):
        """Test goal_bias is positive when moving toward goal."""
        from neurospatial.behavior.navigation import goal_bias

        positions = np.column_stack([np.linspace(0, 100, 101), np.zeros(101)])
        times = np.linspace(0, 10, 101)
        goal = np.array([100.0, 0.0])

        bias = goal_bias(positions, times, goal, min_speed=0.1)

        # Should be close to 1.0 (moving directly toward goal)
        assert bias > 0.8

    def test_instantaneous_goal_alignment_direct(self):
        """Test instantaneous_goal_alignment for direct approach."""
        from neurospatial.behavior.navigation import instantaneous_goal_alignment

        positions = np.column_stack([np.linspace(0, 50, 51), np.zeros(51)])
        times = np.linspace(0, 10, 51)
        goal = np.array([100.0, 0.0])

        alignment = instantaneous_goal_alignment(positions, times, goal, min_speed=0.01)

        # Should be close to 1.0 for most samples
        assert np.nanmean(alignment) > 0.8


class TestBehavioralFunctions:
    """Test functions from behavioral.py."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        from neurospatial import Environment

        np.random.seed(42)
        positions = np.random.rand(100, 2) * 100
        env = Environment.from_samples(positions, bin_size=10.0)

        # Add test regions using the correct API
        env.regions.add("start", point=np.array([10.0, 10.0]))
        env.regions.add("goal", point=np.array([90.0, 90.0]))

        return env

    def test_path_progress_basic(self, simple_env):
        """Test path_progress computation."""
        from neurospatial.behavior.navigation import path_progress

        # Create trajectory bins
        n_samples = 10
        trajectory_bins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        start_bins = np.full(n_samples, 0)
        goal_bins = np.full(n_samples, 9)

        # Limit to valid bin range
        trajectory_bins = np.minimum(trajectory_bins, simple_env.n_bins - 1)
        start_bins = np.minimum(start_bins, simple_env.n_bins - 1)
        goal_bins = np.minimum(goal_bins, simple_env.n_bins - 1)

        progress = path_progress(
            simple_env, trajectory_bins, start_bins, goal_bins, metric="euclidean"
        )

        assert len(progress) == n_samples
        # First sample should be close to 0, last close to 1
        assert progress[0] <= 0.2 or np.isnan(progress[0])

    def test_distance_to_region_scalar(self, simple_env):
        """Test distance_to_region with scalar target."""
        from neurospatial.behavior.navigation import distance_to_region

        # Create trajectory
        trajectory_bins = np.array([0, 1, 2, 3, 4])
        trajectory_bins = np.minimum(trajectory_bins, simple_env.n_bins - 1)
        target_bin = min(10, simple_env.n_bins - 1)

        distances = distance_to_region(
            simple_env, trajectory_bins, target_bin, metric="euclidean"
        )

        assert len(distances) == len(trajectory_bins)
        assert all(d >= 0 or np.isnan(d) for d in distances)

    def test_distance_to_region_array(self, simple_env):
        """Test distance_to_region with array targets."""
        from neurospatial.behavior.navigation import distance_to_region

        trajectory_bins = np.array([0, 1, 2, 3, 4])
        trajectory_bins = np.minimum(trajectory_bins, simple_env.n_bins - 1)
        target_bins = np.array([10, 10, 10, 20, 20])
        target_bins = np.minimum(target_bins, simple_env.n_bins - 1)

        distances = distance_to_region(
            simple_env, trajectory_bins, target_bins, metric="euclidean"
        )

        assert len(distances) == len(trajectory_bins)
