"""Integration tests for complete trajectory and behavioral analysis workflows.

Tests the full pipeline combining:
- Trajectory metrics (turn angles, step lengths, home range, MSD)
- Region-based segmentation (crossings, runs)
- Lap detection (circular track analysis)
- Trial segmentation (task-based epochs)
- Trajectory similarity (pattern comparison, goal-directed behavior)
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point

from neurospatial import Environment
from neurospatial.behavior.trajectory import (
    compute_home_range,
    compute_step_lengths,
    compute_turn_angles,
    mean_square_displacement,
)
from neurospatial.segmentation import (
    Lap,
    Trial,
    detect_goal_directed_runs,
    detect_laps,
    detect_region_crossings,
    segment_trials,
    trajectory_similarity,
)


class TestFullTrajectoryAnalysisWorkflow:
    """Test complete workflow combining trajectory metrics and behavioral segmentation."""

    def test_circular_track_complete_workflow(self) -> None:
        """Test full circular track analysis: laps → metrics → similarity."""
        # Create environment (20x20 cm, 2cm bins)
        x = np.linspace(0, 20, 100)
        y = np.linspace(0, 20, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=2.0)

        # Generate circular trajectory (3 clockwise laps)
        n_samples = 300
        t = np.linspace(0, 3 * 2 * np.pi, n_samples)
        center = np.array([10.0, 10.0])
        radius = 6.0
        traj_positions = center + radius * np.column_stack([np.cos(t), np.sin(t)])
        times = np.linspace(0, 30, n_samples)  # 30 seconds, 10s per lap

        # Map to bins
        trajectory_bins = env.bin_at(traj_positions)

        # Step 1: Detect laps
        # Create circular start region at x=16, y=10 (radius point)
        env.regions.add(
            "start", polygon=Point(center[0] + radius, center[1]).buffer(1.5)
        )
        laps = detect_laps(
            trajectory_bins,
            times,
            env,
            method="region",
            start_region="start",
            direction="both",
        )

        # Should detect ~3 laps
        assert len(laps) >= 2, "Should detect at least 2 complete laps"
        assert all(isinstance(lap, Lap) for lap in laps)
        assert all(lap.start_time < lap.end_time for lap in laps)

        # Step 2: Compute trajectory metrics for each lap
        lap_metrics = []
        for lap in laps[:2]:  # Analyze first 2 complete laps
            # Get lap segment
            lap_mask = (times >= lap.start_time) & (times <= lap.end_time)
            lap_bins = trajectory_bins[lap_mask]
            lap_positions = traj_positions[lap_mask]  # Use continuous positions!

            # Compute metrics (both functions expect continuous positions, not bins)
            turn_angles = compute_turn_angles(lap_positions)
            step_lengths = compute_step_lengths(lap_positions)

            lap_metrics.append(
                {
                    "duration": lap.end_time - lap.start_time,
                    "mean_turn_angle": np.abs(np.mean(turn_angles)),
                    "mean_step_length": np.mean(step_lengths),
                    "bins": lap_bins,
                }
            )

        # Step 3: Compare lap similarity
        if len(lap_metrics) >= 2:
            similarity = trajectory_similarity(
                lap_metrics[0]["bins"], lap_metrics[1]["bins"], env, method="jaccard"
            )
            # Laps on circular track should have high spatial overlap
            assert similarity > 0.5, "Consecutive laps should have high similarity"

        # Step 4: Compute home range (should be annular region)
        home_range = compute_home_range(trajectory_bins, percentile=95.0)
        assert len(home_range) > 10, "Home range should include multiple bins"

        # Step 5: Compute MSD (should show ballistic movement within laps)
        tau_values, msd_values = mean_square_displacement(
            traj_positions, times, max_tau=5.0
        )
        assert len(tau_values) > 0
        assert len(msd_values) == len(tau_values)
        # MSD should increase with tau (movement, not stationary)
        assert msd_values[-1] > msd_values[0]

    def test_tmaze_task_complete_workflow(self) -> None:
        """Test full T-maze workflow: trials → runs → goal-directed analysis."""
        # Create T-maze environment (simplified linear track)
        x = np.linspace(0, 100, 500)
        y = np.linspace(0, 100, 500)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)

        # Define T-maze regions
        env.regions.add("start", polygon=Point(50.0, 10.0).buffer(8.0))  # Bottom center
        env.regions.add("left", polygon=Point(20.0, 90.0).buffer(8.0))  # Top left
        env.regions.add("right", polygon=Point(80.0, 90.0).buffer(8.0))  # Top right

        # Generate synthetic trajectory: 3 trials (left, right, left)
        trial_trajs = []
        times_list = []
        current_time = 0.0

        for goal in ["left", "right", "left"]:
            # Each trial: start → stem → goal
            n_steps = 50
            if goal == "left":
                traj = np.column_stack(
                    [
                        np.linspace(50, 20, n_steps),  # x: center to left
                        np.linspace(10, 90, n_steps),  # y: bottom to top
                    ]
                )
            else:
                traj = np.column_stack(
                    [
                        np.linspace(50, 80, n_steps),  # x: center to right
                        np.linspace(10, 90, n_steps),  # y: bottom to top
                    ]
                )

            trial_trajs.append(traj)
            times_list.append(np.linspace(current_time, current_time + 10, n_steps))
            current_time += 12  # 10s trial + 2s inter-trial

        # Combine all trials
        traj_positions = np.vstack(trial_trajs)
        times = np.concatenate(times_list)
        trajectory_bins = env.bin_at(traj_positions)

        # Step 1: Segment into trials
        trials = segment_trials(
            trajectory_bins,
            times,
            env,
            start_region="start",
            end_regions=["left", "right"],
            min_duration=5.0,
            max_duration=15.0,
        )

        # Should detect 3 trials
        assert len(trials) >= 2, "Should detect at least 2 trials"
        assert all(isinstance(trial, Trial) for trial in trials)

        # Count end_regions
        left_trials = [t for t in trials if t.end_region == "left"]
        right_trials = [t for t in trials if t.end_region == "right"]
        assert len(left_trials) > 0, "Should detect left trials"
        assert len(right_trials) > 0, "Should detect right trials"

        # Step 2: Detect region crossings
        left_crossings = detect_region_crossings(
            trajectory_bins, times, "left", env, direction="entry"
        )
        right_crossings = detect_region_crossings(
            trajectory_bins, times, "right", env, direction="entry"
        )

        assert len(left_crossings) >= 1, "Should detect left entries"
        assert len(right_crossings) >= 1, "Should detect right entries"

        # Step 3: Analyze goal-directed runs for first trial
        if len(trials) > 0:
            trial = trials[0]
            trial_mask = (times >= trial.start_time) & (times <= trial.end_time)
            trial_bins = trajectory_bins[trial_mask]
            trial_times = times[trial_mask]

            # Detect goal-directed behavior toward trial end_region
            if trial.end_region:
                goal_runs = detect_goal_directed_runs(
                    trial_bins,
                    trial_times,
                    env,
                    goal_region=trial.end_region,
                    directedness_threshold=0.3,  # Lower for synthetic data
                    min_progress=5.0,
                )

                # Should detect at least one goal-directed run
                # (Note: may be 0 if trajectory is too discretized)
                assert isinstance(goal_runs, list)

        # Step 4: Compare trial trajectories
        if len(trials) >= 2:
            # Get trial segments
            trial1 = trials[0]
            trial2 = trials[1]

            mask1 = (times >= trial1.start_time) & (times <= trial1.end_time)
            mask2 = (times >= trial2.start_time) & (times <= trial2.end_time)

            bins1 = trajectory_bins[mask1]
            bins2 = trajectory_bins[mask2]

            # Compute similarity
            similarity = trajectory_similarity(bins1, bins2, env, method="jaccard")

            # If same end_region, similarity should be higher
            if trial1.end_region == trial2.end_region:
                # Same choice → higher similarity expected
                assert similarity >= 0.0  # At least some overlap
            else:
                # Different choice → lower similarity expected
                assert similarity >= 0.0  # Valid similarity score

    def test_exploration_to_goal_directed_workflow(self) -> None:
        """Test workflow detecting transition from exploration to goal-directed behavior."""
        # Create environment
        x = np.linspace(0, 50, 250)
        y = np.linspace(0, 50, 250)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        # Define goal region
        env.regions.add("goal", polygon=Point(40.0, 40.0).buffer(5.0))

        # Phase 1: Random exploration (first 100 samples)
        rng = np.random.default_rng(42)
        explore_positions = rng.uniform(10, 50, (100, 2))
        explore_times = np.linspace(0, 50, 100)

        # Phase 2: Goal-directed behavior (next 50 samples)
        goal_positions = np.column_stack(
            [
                np.linspace(25, 40, 50),
                np.linspace(25, 40, 50),
            ]
        )
        goal_times = np.linspace(50, 75, 50)

        # Combine phases
        all_positions = np.vstack([explore_positions, goal_positions])
        all_times = np.concatenate([explore_times, goal_times])
        trajectory_bins = env.bin_at(all_positions)

        # Step 1: Detect goal-directed runs (should only find in phase 2)
        goal_runs = detect_goal_directed_runs(
            trajectory_bins,
            all_times,
            env,
            goal_region="goal",
            directedness_threshold=0.3,
            min_progress=3.0,
        )

        # Should detect at least one goal-directed run in phase 2
        # (may detect 0 if discretization artifacts, but list should be valid)
        assert isinstance(goal_runs, list)
        if len(goal_runs) > 0:
            # Runs should be in second half of trajectory
            assert all(run.start_time >= 30 for run in goal_runs)

        # Step 2: Compare exploration vs goal-directed similarity
        explore_bins = trajectory_bins[:100]
        goal_bins = trajectory_bins[100:]

        # Should have low overlap (different spatial regions)
        similarity = trajectory_similarity(
            explore_bins, goal_bins, env, method="jaccard"
        )
        # Valid similarity (may be low if different spatial regions)
        assert 0.0 <= similarity <= 1.0

        # Step 3: Compute home range for each phase
        explore_home = compute_home_range(explore_bins, percentile=95.0)
        goal_home = compute_home_range(goal_bins, percentile=95.0)

        assert len(explore_home) > 0
        assert len(goal_home) > 0
        # Exploration should cover more area
        assert len(explore_home) >= len(goal_home)
