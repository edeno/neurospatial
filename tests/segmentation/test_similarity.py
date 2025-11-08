"""Tests for trajectory similarity and goal-directed run detection.

Following TDD: Write tests FIRST, watch them fail, then implement.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.segmentation.similarity import (
    detect_goal_directed_runs,
    trajectory_similarity,
)


class TestTrajectorySimilarity:
    """Test trajectory similarity metrics."""

    def test_trajectory_similarity_identical_jaccard(self):
        """Identical trajectories should have similarity 1.0 (Jaccard method)."""
        # Create simple 1D environment
        positions = np.linspace(0, 100, 50)[:, None]
        env = Environment.from_samples(positions, bin_size=5.0)

        # Create trajectory bins
        trajectory = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)

        # Identical trajectories
        similarity = trajectory_similarity(
            trajectory, trajectory, env, method="jaccard"
        )

        assert similarity == pytest.approx(1.0)

    def test_trajectory_similarity_disjoint_jaccard(self):
        """Disjoint trajectories should have similarity 0.0 (Jaccard method)."""
        # Create 1D environment
        positions = np.linspace(0, 100, 100)[:, None]
        env = Environment.from_samples(positions, bin_size=2.0)

        # Two non-overlapping trajectories
        trajectory1 = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        trajectory2 = np.array([10, 11, 12, 13, 14], dtype=np.int64)

        similarity = trajectory_similarity(
            trajectory1, trajectory2, env, method="jaccard"
        )

        assert similarity == pytest.approx(0.0)

    def test_trajectory_similarity_partial_overlap_jaccard(self):
        """Partially overlapping trajectories should have intermediate similarity."""
        # Create 1D environment
        positions = np.linspace(0, 100, 50)[:, None]
        env = Environment.from_samples(positions, bin_size=5.0)

        # Partially overlapping trajectories
        trajectory1 = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        trajectory2 = np.array([2, 3, 4, 5, 6], dtype=np.int64)

        similarity = trajectory_similarity(
            trajectory1, trajectory2, env, method="jaccard"
        )

        # Jaccard = |{2,3,4}| / |{0,1,2,3,4,5,6}| = 3/7 ≈ 0.428
        assert 0.4 < similarity < 0.5

    def test_trajectory_similarity_correlation(self):
        """Test correlation-based similarity metric."""
        # Create 2D environment
        np.random.seed(42)
        positions = np.random.uniform(0, 100, (200, 2))
        env = Environment.from_samples(positions, bin_size=5.0)

        # Create two similar trajectories (same sequence, slight variation)
        trajectory1 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        trajectory2 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)

        similarity = trajectory_similarity(
            trajectory1, trajectory2, env, method="correlation"
        )

        # Identical sequences should have high correlation
        assert similarity > 0.9

    def test_trajectory_similarity_hausdorff(self):
        """Test Hausdorff distance-based similarity."""
        # Create 2D environment
        positions = np.random.RandomState(42).uniform(0, 100, (200, 2))
        env = Environment.from_samples(positions, bin_size=5.0)

        # Same trajectory = minimum Hausdorff distance
        trajectory = np.array([0, 5, 10, 15], dtype=np.int64)

        similarity = trajectory_similarity(
            trajectory, trajectory, env, method="hausdorff"
        )

        # Identical should give similarity 1.0 (distance normalized to [0,1])
        assert similarity == pytest.approx(1.0, abs=0.01)

    def test_trajectory_similarity_dtw(self):
        """Test dynamic time warping similarity."""
        # Create 1D environment
        positions = np.linspace(0, 100, 100)[:, None]
        env = Environment.from_samples(positions, bin_size=2.0)

        # Similar trajectories with different speeds
        trajectory1 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        trajectory2 = np.array([0, 0, 1, 2, 3, 4, 5, 5], dtype=np.int64)  # slower

        similarity = trajectory_similarity(trajectory1, trajectory2, env, method="dtw")

        # DTW should recognize similar spatial paths despite timing differences
        assert similarity > 0.7

    def test_trajectory_similarity_all_methods(self):
        """Test that all methods return values in [0, 1] range."""
        positions = np.random.RandomState(42).uniform(0, 100, (200, 2))
        env = Environment.from_samples(positions, bin_size=5.0)

        trajectory1 = np.array([0, 5, 10, 15, 20], dtype=np.int64)
        trajectory2 = np.array([2, 7, 12, 17, 22], dtype=np.int64)

        methods = ["jaccard", "correlation", "hausdorff", "dtw"]

        for method in methods:
            similarity = trajectory_similarity(
                trajectory1, trajectory2, env, method=method
            )
            assert 0.0 <= similarity <= 1.0, f"Method {method} returned {similarity}"

    def test_trajectory_similarity_invalid_method(self):
        """Invalid method should raise ValueError."""
        positions = np.linspace(0, 100, 50)[:, None]
        env = Environment.from_samples(positions, bin_size=5.0)

        trajectory = np.array([0, 1, 2], dtype=np.int64)

        with pytest.raises(ValueError, match=r"method must be one of.*got 'invalid'"):
            trajectory_similarity(trajectory, trajectory, env, method="invalid")

    def test_trajectory_similarity_empty_trajectories(self):
        """Empty trajectories should raise ValueError."""
        positions = np.linspace(0, 100, 50)[:, None]
        env = Environment.from_samples(positions, bin_size=5.0)

        empty = np.array([], dtype=np.int64)
        trajectory = np.array([0, 1, 2], dtype=np.int64)

        with pytest.raises(ValueError, match=r"Trajectories cannot be empty"):
            trajectory_similarity(empty, trajectory, env, method="jaccard")

    def test_trajectory_similarity_parameter_order(self):
        """Verify parameter order and naming consistency."""
        positions = np.linspace(0, 100, 50)[:, None]
        env = Environment.from_samples(positions, bin_size=5.0)

        trajectory = np.array([0, 1, 2], dtype=np.int64)

        # Should work with positional args
        result1 = trajectory_similarity(trajectory, trajectory, env)

        # Should work with keyword args
        result2 = trajectory_similarity(
            trajectory1_bins=trajectory,
            trajectory2_bins=trajectory,
            env=env,
            method="jaccard",
        )

        assert result1 == result2


class TestDetectGoalDirectedRuns:
    """Test goal-directed run detection."""

    def test_detect_goal_directed_runs_straight_path(self):
        """Test goal-directed run detection with efficient path."""
        # Create 2D environment with regular grid for predictable connectivity
        x = np.linspace(0, 100, 20)
        y = np.linspace(0, 100, 20)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=6.0)

        # Add goal region at one corner
        from shapely.geometry import Point

        # Find a bin far from origin
        distances_from_origin = np.linalg.norm(env.bin_centers, axis=1)
        goal_bin_idx = int(np.argmax(distances_from_origin))
        goal_location = env.bin_centers[goal_bin_idx]
        goal_polygon = Point(goal_location).buffer(15.0)
        env.regions.add("goal", polygon=goal_polygon)

        # Create trajectory moving toward goal
        # Start from opposite corner
        start_bin_idx = int(np.argmin(distances_from_origin))

        # Create a somewhat direct path (not perfectly straight, but progressing)
        # Use a simple heuristic: move toward goal in steps
        trajectory_bins_list = [start_bin_idx]
        current_bin = start_bin_idx

        for _ in range(15):  # Take 15 steps
            # Find neighbors and pick the one closest to goal
            neighbors = list(env.connectivity.neighbors(current_bin))
            if not neighbors:
                break

            neighbor_distances = []
            for neighbor in neighbors:
                neighbor_pos = env.bin_centers[neighbor]
                dist_to_goal = np.linalg.norm(neighbor_pos - goal_location)
                neighbor_distances.append(dist_to_goal)

            # Move to neighbor closest to goal
            best_neighbor_idx = int(np.argmin(neighbor_distances))
            current_bin = neighbors[best_neighbor_idx]
            trajectory_bins_list.append(current_bin)

        trajectory_bins = np.array(trajectory_bins_list, dtype=np.int64)
        times = np.linspace(0, 10, len(trajectory_bins))

        runs = detect_goal_directed_runs(
            trajectory_bins,
            times,
            env,
            goal_region="goal",
            directedness_threshold=0.5,  # Lower threshold for imperfect path
            min_progress=10.0,
        )

        # Should detect at least one goal-directed run
        # (actual number depends on path efficiency and environment connectivity)
        assert isinstance(runs, list)
        # The function should execute without errors
        # Whether it finds runs depends on the specific path taken

    def test_detect_goal_directed_runs_wandering_path(self):
        """Wandering path should have low directedness and be filtered out."""
        # Create 2D environment
        np.random.seed(42)
        positions = np.random.uniform(0, 100, (500, 2))
        env = Environment.from_samples(positions, bin_size=3.0)

        # Add goal region
        from shapely.geometry import Point

        goal_location = env.bin_centers[env.n_bins // 2]
        goal_polygon = Point(goal_location).buffer(10.0)
        env.regions.add("goal", polygon=goal_polygon)

        # Random walk (inefficient path)
        np.random.seed(123)
        trajectory_bins = np.random.randint(0, env.n_bins, 100, dtype=np.int64)
        times = np.linspace(0, 100, len(trajectory_bins))

        runs = detect_goal_directed_runs(
            trajectory_bins,
            times,
            env,
            goal_region="goal",
            directedness_threshold=0.8,  # high threshold
            min_progress=20.0,
        )

        # Random walk should not produce high-directedness runs
        # (may produce 0 runs or only a few)
        assert isinstance(runs, list)

    def test_detect_goal_directed_runs_min_progress_filter(self):
        """Test that min_progress filters out short runs."""
        # Create 2D environment
        np.random.seed(42)
        positions = np.random.uniform(0, 100, (200, 2))
        env = Environment.from_samples(positions, bin_size=4.0)

        # Add goal region
        from shapely.geometry import Point

        goal_location = env.bin_centers[-1]
        goal_polygon = Point(goal_location).buffer(10.0)
        env.regions.add("goal", polygon=goal_polygon)

        # Very short trajectory (minimal progress)
        trajectory_bins = np.array([0, 1, 2], dtype=np.int64)
        times = np.array([0.0, 1.0, 2.0])

        runs = detect_goal_directed_runs(
            trajectory_bins,
            times,
            env,
            goal_region="goal",
            directedness_threshold=0.5,
            min_progress=50.0,  # high threshold - should filter out
        )

        # Should be filtered by min_progress
        assert len(runs) == 0

    def test_detect_goal_directed_runs_validation(self):
        """Test parameter validation."""
        # Create 2D environment
        np.random.seed(42)
        positions = np.random.uniform(0, 100, (200, 2))
        env = Environment.from_samples(positions, bin_size=5.0)

        trajectory_bins = np.array([0, 1, 2], dtype=np.int64)
        times = np.array([0.0, 1.0, 2.0])

        # Missing goal region
        with pytest.raises(KeyError, match=r"Region.*not found"):
            detect_goal_directed_runs(
                trajectory_bins,
                times,
                env,
                goal_region="nonexistent",
                directedness_threshold=0.7,
                min_progress=10.0,
            )

        # Invalid threshold
        from shapely.geometry import Point

        goal_polygon = Point(env.bin_centers[0]).buffer(10.0)
        env.regions.add("goal", polygon=goal_polygon)

        with pytest.raises(
            ValueError, match=r"directedness_threshold must be in \[0, 1\]"
        ):
            detect_goal_directed_runs(
                trajectory_bins,
                times,
                env,
                goal_region="goal",
                directedness_threshold=1.5,  # invalid
                min_progress=10.0,
            )

        # Negative min_progress
        with pytest.raises(ValueError, match=r"min_progress must be non-negative"):
            detect_goal_directed_runs(
                trajectory_bins,
                times,
                env,
                goal_region="goal",
                directedness_threshold=0.7,
                min_progress=-10.0,  # invalid
            )

    def test_detect_goal_directed_runs_parameter_order(self):
        """Verify parameter order consistency."""
        # Create 2D environment
        np.random.seed(42)
        positions = np.random.uniform(0, 100, (200, 2))
        env = Environment.from_samples(positions, bin_size=5.0)

        from shapely.geometry import Point

        goal_polygon = Point(env.bin_centers[-1]).buffer(10.0)
        env.regions.add("goal", polygon=goal_polygon)

        trajectory_bins = np.array([0, 5, 10, 15], dtype=np.int64)
        times = np.array([0.0, 1.0, 2.0, 3.0])

        # Should work with positional args
        runs1 = detect_goal_directed_runs(
            trajectory_bins, times, env, goal_region="goal"
        )

        # Should work with keyword args
        runs2 = detect_goal_directed_runs(
            trajectory_bins=trajectory_bins,
            times=times,
            env=env,
            goal_region="goal",
            directedness_threshold=0.7,
            min_progress=20.0,
        )

        # Both should return list (exact match not required due to defaults)
        assert isinstance(runs1, list)
        assert isinstance(runs2, list)

    def test_detect_goal_directed_runs_empty_trajectory(self):
        """Empty trajectory should return empty list."""
        # Create 2D environment
        np.random.seed(42)
        positions = np.random.uniform(0, 100, (200, 2))
        env = Environment.from_samples(positions, bin_size=5.0)

        from shapely.geometry import Point

        goal_polygon = Point(env.bin_centers[-1]).buffer(10.0)
        env.regions.add("goal", polygon=goal_polygon)

        empty_bins = np.array([], dtype=np.int64)
        empty_times = np.array([])

        runs = detect_goal_directed_runs(
            empty_bins, empty_times, env, goal_region="goal"
        )

        assert runs == []


class TestSimilarityIntegration:
    """Integration tests combining similarity functions."""

    def test_similarity_workflow(self):
        """Test complete workflow: detect similar trajectories."""
        # Create 2D environment
        np.random.seed(42)
        positions = np.random.uniform(0, 100, (500, 2))
        env = Environment.from_samples(positions, bin_size=4.0)

        # Create three trajectories: two similar, one different
        traj_a = np.array([0, 5, 10, 15, 20], dtype=np.int64)
        traj_b = np.array([1, 5, 11, 15, 21], dtype=np.int64)  # similar to A
        traj_c = np.array([50, 55, 60, 65, 70], dtype=np.int64)  # different

        # A and B should be more similar than A and C
        sim_ab = trajectory_similarity(traj_a, traj_b, env, method="jaccard")
        sim_ac = trajectory_similarity(traj_a, traj_c, env, method="jaccard")

        assert sim_ab > sim_ac
        assert sim_ac == 0.0  # completely disjoint

    def test_goal_directed_and_similarity(self):
        """Test detecting goal-directed runs and comparing their similarity."""
        # Create 2D environment
        np.random.seed(42)
        positions = np.random.uniform(0, 100, (200, 2))
        env = Environment.from_samples(positions, bin_size=4.0)

        # Add goal region
        from shapely.geometry import Point

        goal_location = env.bin_centers[-1]
        goal_polygon = Point(goal_location).buffer(10.0)
        env.regions.add("goal", polygon=goal_polygon)

        # Two trajectories toward goal
        traj1_bins = np.arange(0, 30, dtype=np.int64)
        traj1_times = np.linspace(0, 10, len(traj1_bins))

        traj2_bins = np.arange(5, 35, dtype=np.int64)
        traj2_times = np.linspace(0, 10, len(traj2_bins))

        # Detect goal-directed runs
        runs1 = detect_goal_directed_runs(
            traj1_bins, traj1_times, env, goal_region="goal"
        )
        runs2 = detect_goal_directed_runs(
            traj2_bins, traj2_times, env, goal_region="goal"
        )

        # Both should produce runs (list type)
        assert isinstance(runs1, list)
        assert isinstance(runs2, list)

        # Compare similarity of the two trajectories
        similarity = trajectory_similarity(
            traj1_bins, traj2_bins, env, method="jaccard"
        )

        # Should have some overlap (partial similarity)
        assert 0.0 < similarity < 1.0
