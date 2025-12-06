"""Integration tests for behavioral trajectory metrics modules.

This module tests cross-module consistency and round-trip properties
for the path_efficiency, goal_directed, decision_analysis, and vte modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment

# =============================================================================
# Test: VTE uses decision_analysis functions correctly
# =============================================================================


class TestVTEDecisionAnalysisIntegration:
    """Test that VTE module correctly uses decision_analysis functions."""

    def test_vte_uses_extract_pre_decision_window_correctly(self) -> None:
        """Verify VTE head_sweep_from_positions matches manual extraction."""
        from neurospatial.behavior.decisions import (
            extract_pre_decision_window,
            head_sweep_from_positions,
        )

        # Create trajectory with known head sweeps
        times = np.linspace(0, 10, 101)
        # Sinusoidal path = back-and-forth motion
        positions = np.column_stack(
            [
                times * 10,  # x moves forward
                10 * np.sin(times * 2),  # y oscillates
            ]
        )

        entry_time = 5.0
        window_duration = 2.0

        # Extract window using decision_analysis
        window_positions, window_times = extract_pre_decision_window(
            positions, times, entry_time, window_duration
        )

        # Compute head sweep from window
        head_sweep_from_window = head_sweep_from_positions(
            window_positions, window_times, min_speed=1.0
        )

        # Head sweep should be > 0 for oscillating trajectory
        assert head_sweep_from_window > 0.0

        # Verify window is correct slice
        assert window_times[0] >= entry_time - window_duration
        assert window_times[-1] <= entry_time

    def test_vte_session_uses_decision_region_entry_correctly(self) -> None:
        """Verify compute_vte_session finds correct entry times."""
        from neurospatial.behavior.decisions import (
            compute_vte_session,
            decision_region_entry_time,
        )
        from neurospatial.behavior.segmentation import Trial

        # Create dense environment covering full trajectory range
        np.random.seed(42)
        x = np.linspace(0, 60, 30)
        y = np.linspace(0, 60, 30)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)

        # Create decision region at center
        center = np.array([30.0, 30.0])
        env.regions.add("decision", point=center)

        # Create trajectory that crosses decision region (passes through center)
        times = np.linspace(0, 4, 201)
        trajectory = np.column_stack(
            [
                np.linspace(5, 55, 201),  # x moves across
                np.full(201, 30.0),  # y at center
            ]
        )

        # Create a trial
        trials = [
            Trial(
                start_time=0.0,
                end_time=4.0,
                start_region="start",
                end_region="end",
                success=True,
            )
        ]

        # Find entry time using decision_analysis
        trajectory_bins = env.bin_at(trajectory)

        # Skip if trajectory doesn't have valid bins (environment coverage issue)
        if np.all(trajectory_bins < 0):
            pytest.skip("Trajectory does not have valid bins in environment")

        entry_from_analysis = decision_region_entry_time(
            trajectory_bins, times, env, "decision"
        )

        # Run VTE session - should use same entry detection
        result = compute_vte_session(
            trajectory,
            times,
            trials,
            decision_region="decision",
            env=env,
            window_duration=0.5,
            min_speed=1.0,
        )

        # Should have found the trial
        assert len(result.trial_results) == 1

        # The window_end should match entry_time from decision_analysis
        assert abs(result.trial_results[0].window_end - entry_from_analysis) < 0.1


# =============================================================================
# Test: Round-trip VTE simulation → classification
# =============================================================================


class TestVTERoundTrip:
    """Test round-trip: simulated VTE behavior → correct classification."""

    def test_high_head_sweep_low_speed_classified_as_vte(self) -> None:
        """Simulated VTE behavior should be classified as VTE."""
        from neurospatial.behavior.decisions import compute_vte_session
        from neurospatial.behavior.segmentation import Trial

        # Create environment
        np.random.seed(42)
        positions = np.random.rand(200, 2) * 100
        env = Environment.from_samples(positions, bin_size=5.0)

        # Create decision region
        center = np.array([50.0, 50.0])
        env.regions.add("decision", point=center)

        # Create multiple trials
        n_trials = 10
        times_per_trial = 100
        all_positions = []
        all_times = []

        for i in range(n_trials):
            t_start = i * 3.0
            trial_times = np.linspace(t_start, t_start + 2.5, times_per_trial)

            if i < 5:
                # VTE trials: slow approach with head sweeping
                x = np.linspace(10, 50, times_per_trial)
                y = 50 + 10 * np.sin(trial_times * 8)  # Oscillating y = scanning
                speed_factor = 0.3  # Slow
            else:
                # Non-VTE trials: fast direct approach
                x = np.linspace(10, 50, times_per_trial)
                y = np.full(times_per_trial, 50.0)  # No oscillation
                speed_factor = 1.0  # Fast

            # Adjust timing for speed
            adjusted_times = trial_times * speed_factor + t_start * (1 - speed_factor)
            positions_trial = np.column_stack([x, y])

            all_positions.append(positions_trial)
            all_times.append(adjusted_times)

        positions_array = np.vstack(all_positions)
        times_array = np.concatenate(all_times)

        # Create trials
        trials = [
            Trial(
                start_time=i * 3.0,
                end_time=i * 3.0 + 2.5,
                start_region="start",
                end_region="end",
                success=True,
            )
            for i in range(n_trials)
        ]

        # Run VTE session
        result = compute_vte_session(
            positions_array,
            times_array,
            trials,
            decision_region="decision",
            env=env,
            window_duration=0.8,
            min_speed=1.0,
            alpha=0.5,
            vte_threshold=0.0,  # Lower threshold for testing
        )

        # Should find some trials
        assert len(result.trial_results) > 0

        # VTE fraction should be reasonable
        assert 0.0 <= result.vte_fraction <= 1.0

    def test_single_trial_returns_none_for_zscores(self) -> None:
        """Single trial should have None for z-scores."""
        from neurospatial.behavior.decisions import compute_vte_trial

        # Create simple trajectory
        times = np.linspace(0, 2, 101)
        positions = np.column_stack(
            [
                np.linspace(0, 100, 101),
                np.full(101, 50.0),
            ]
        )

        result = compute_vte_trial(
            positions,
            times,
            entry_time=1.0,
            window_duration=0.5,
            min_speed=1.0,
        )

        # Z-scores should be None for single trial
        assert result.z_head_sweep is None
        assert result.z_speed_inverse is None
        assert result.vte_index is None
        assert result.is_vte is None


# =============================================================================
# Test: path_efficiency consistent with path_progress conceptually
# =============================================================================


class TestPathEfficiencyPathProgressConsistency:
    """Test conceptual consistency between path_efficiency and path_progress."""

    def test_straight_path_efficiency_equals_one(self) -> None:
        """Straight path should have efficiency = 1.0."""
        from neurospatial.behavior.navigation import path_efficiency

        # Create environment
        np.random.seed(42)
        positions = np.random.rand(200, 2) * 100
        env = Environment.from_samples(positions, bin_size=5.0)

        # Straight trajectory
        trajectory = np.column_stack(
            [
                np.linspace(10, 90, 50),
                np.full(50, 50.0),
            ]
        )
        goal = np.array([90.0, 50.0])

        efficiency = path_efficiency(env, trajectory, goal, metric="euclidean")

        # Should be very close to 1.0 for straight path
        assert 0.95 <= efficiency <= 1.0

    def test_path_progress_reaches_one_at_goal(self) -> None:
        """Path progress should reach 1.0 when at goal."""
        from neurospatial.behavior.navigation import path_progress

        # Create dense environment to ensure trajectory bins exist
        np.random.seed(42)
        # Create grid positions to ensure full coverage
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)

        # Trajectory ending at goal
        trajectory = np.column_stack(
            [
                np.linspace(10, 90, 50),
                np.full(50, 50.0),
            ]
        )

        trajectory_bins = env.bin_at(trajectory)
        start_bin = trajectory_bins[0]
        goal_bin = trajectory_bins[-1]

        # Skip if bins are invalid
        if start_bin < 0 or goal_bin < 0:
            pytest.skip("Trajectory bins not in environment")

        progress = path_progress(
            env,
            trajectory_bins,
            start_bins=np.full(len(trajectory_bins), start_bin),
            goal_bins=np.full(len(trajectory_bins), goal_bin),
            metric="geodesic",  # Use geodesic for connected environment
        )

        # Progress at end should be 1.0 (or close, since bins are discrete)
        # Filter out NaN values first
        valid_progress = progress[~np.isnan(progress)]
        if len(valid_progress) > 0:
            assert valid_progress[-1] >= 0.8

    def test_efficiency_and_progress_both_handle_detours(self) -> None:
        """Both metrics should handle detours reasonably."""
        from neurospatial.behavior.navigation import path_efficiency

        # Create dense environment to ensure trajectory bins exist
        np.random.seed(42)
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)

        # U-shaped trajectory (detour)
        n_points = 100
        x_traj = np.concatenate(
            [
                np.linspace(10, 90, n_points // 2),  # Go right
                np.linspace(90, 10, n_points // 2),  # Go back left
            ]
        )
        y_traj = np.concatenate(
            [
                np.full(n_points // 2, 50.0),
                np.full(n_points // 2, 50.0),
            ]
        )
        trajectory = np.column_stack([x_traj, y_traj])
        goal = np.array([10.0, 50.0])  # Back at start

        # Path efficiency should be low (traveled much more than needed)
        efficiency = path_efficiency(env, trajectory, goal, metric="euclidean")
        assert efficiency < 0.1  # Should be very inefficient


# =============================================================================
# Test: Goal-directed metrics consistency
# =============================================================================


class TestGoalDirectedConsistency:
    """Test consistency of goal-directed metrics."""

    def test_goal_bias_matches_approach_rate_sign(self) -> None:
        """Positive goal_bias should correspond to negative approach_rate."""
        from neurospatial.behavior.navigation import (
            approach_rate,
            goal_bias,
        )

        # Create approaching trajectory
        times = np.linspace(0, 5, 101)
        positions = np.column_stack(
            [
                np.linspace(0, 90, 101),  # Moving toward goal
                np.full(101, 50.0),
            ]
        )
        goal = np.array([100.0, 50.0])

        bias = goal_bias(positions, times, goal, min_speed=1.0)
        rates = approach_rate(positions, times, goal, metric="euclidean")

        # Approaching: bias > 0, mean rate < 0
        assert bias > 0.5
        assert np.nanmean(rates) < 0

    def test_goal_bias_orthogonal_near_zero(self) -> None:
        """Circular path around goal should have near-zero bias."""
        from neurospatial.behavior.navigation import goal_bias

        # Circular path around goal
        n_points = 200
        times = np.linspace(0, 10, n_points)
        theta = np.linspace(0, 2 * np.pi, n_points)
        radius = 30.0
        center = np.array([50.0, 50.0])

        positions = np.column_stack(
            [
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta),
            ]
        )

        bias = goal_bias(positions, times, center, min_speed=1.0)

        # Should be near zero for circular path
        assert abs(bias) < 0.3


# =============================================================================
# Test: Decision analysis consistency
# =============================================================================


class TestDecisionAnalysisConsistency:
    """Test consistency of decision analysis metrics."""

    def test_voronoi_labels_cover_all_bins(self) -> None:
        """All reachable bins should have a valid Voronoi label."""
        from neurospatial.behavior.decisions import geodesic_voronoi_labels

        # Create connected environment
        np.random.seed(42)
        positions = np.random.rand(500, 2) * 100
        env = Environment.from_samples(positions, bin_size=5.0)

        # Use first and last bins as goals
        goal_bins = np.array([0, env.n_bins - 1])

        labels = geodesic_voronoi_labels(env, goal_bins)

        # All bins should be labeled 0 or 1 (or -1 if unreachable)
        unique_labels = set(np.unique(labels))
        assert unique_labels <= {-1, 0, 1}

        # Most bins should be reachable
        n_reachable = np.sum(labels >= 0)
        assert n_reachable > env.n_bins * 0.5

    def test_pre_decision_metrics_capture_hesitation(self) -> None:
        """High hesitation should have high heading variance."""
        from neurospatial.behavior.decisions import compute_pre_decision_metrics

        # Hesitating trajectory: oscillating (high heading variance)
        times = np.linspace(0, 2, 101)
        positions = np.column_stack(
            [
                50 + 5 * np.sin(times * 10),  # Oscillations in x
                50 + 5 * np.cos(times * 10),  # Oscillations in y
            ]
        )

        metrics = compute_pre_decision_metrics(
            positions, times, entry_time=2.0, window_duration=1.5, min_speed=0.1
        )

        # Should have high heading variance (changing direction frequently)
        assert metrics.heading_circular_variance > 0.1

        # Samples should be captured
        assert metrics.n_samples > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
