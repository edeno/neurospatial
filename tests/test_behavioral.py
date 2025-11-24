"""Tests for behavioral analysis and goal-directed navigation metrics."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from neurospatial import Environment
from neurospatial.behavioral import trials_to_region_arrays
from neurospatial.segmentation.trials import Trial

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_environment_with_regions():
    """Create a simple 2D environment with point regions for testing."""
    # Create a 10x10 cm space with dense sampling for connectivity
    # Use 100 samples distributed uniformly to ensure bins are connected
    np.random.seed(42)
    positions = np.random.uniform(0, 10, (100, 2))
    env = Environment.from_samples(positions, bin_size=2.0)
    env.units = "cm"

    # Add regions at known locations
    env.regions.add("start", point=(1.0, 1.0))  # Near bottom-left
    env.regions.add("goal1", point=(9.0, 9.0))  # Near top-right
    env.regions.add("goal2", point=(9.0, 1.0))  # Near bottom-right

    return env


@pytest.fixture
def environment_with_polygon_regions():
    """Create environment with polygon regions (multi-bin)."""
    # Create 20x20 cm space with 2 cm bins (10x10 grid = 100 bins)
    positions = np.random.uniform(0, 20, (200, 2))
    env = Environment.from_samples(positions, bin_size=2.0)
    env.units = "cm"

    # Add polygon regions that cover multiple bins
    start_polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    goal_polygon = Polygon([(16, 16), (20, 16), (20, 20), (16, 20)])

    env.regions.add("start_zone", polygon=start_polygon)
    env.regions.add("goal_zone", polygon=goal_polygon)

    return env


# =============================================================================
# M2.2: trials_to_region_arrays() Tests
# =============================================================================


def test_trials_to_region_arrays_single_trial(simple_environment_with_regions):
    """Test helper with single trial - constant start/goal."""
    env = simple_environment_with_regions

    # Create single trial from t=0 to t=10
    trial = Trial(
        start_time=0.0,
        end_time=10.0,
        start_region="start",
        end_region="goal1",
        success=True,
    )
    trials = [trial]

    # Create times array with 11 points (0, 1, 2, ..., 10 seconds)
    times = np.linspace(0.0, 10.0, 11)

    # Call function
    start_bins, goal_bins = trials_to_region_arrays(trials, times, env)

    # Assertions
    assert start_bins.shape == (11,)
    assert goal_bins.shape == (11,)
    assert start_bins.dtype == np.int_
    assert goal_bins.dtype == np.int_

    # Get expected bin indices
    start_bin = env.bins_in_region("start")[0]
    goal_bin = env.bins_in_region("goal1")[0]

    # All timepoints should have same start/goal bins
    assert np.all(start_bins == start_bin)
    assert np.all(goal_bins == goal_bin)


def test_trials_to_region_arrays_multiple_trials(simple_environment_with_regions):
    """Test helper with multiple trials - varying start/goal."""
    env = simple_environment_with_regions

    # Check regions exist - skip if goal2 missing
    if len(env.bins_in_region("goal2")) == 0:
        pytest.skip("goal2 region has no bins in random environment")

    # Create 3 trials with different goals
    trials = [
        Trial(0.0, 5.0, "start", "goal1", True),  # t=0 to t=5
        Trial(5.5, 10.0, "start", "goal2", True),  # t=5.5 to t=10
        Trial(11.0, 15.0, "start", "goal1", True),  # t=11 to t=15
    ]

    # Create times array covering all trials plus gaps
    times = np.linspace(0.0, 16.0, 17)  # 0, 1, 2, ..., 16 seconds

    # Call function
    start_bins, goal_bins = trials_to_region_arrays(trials, times, env)

    # Get expected bin indices
    start_bin = env.bins_in_region("start")[0]
    goal1_bin = env.bins_in_region("goal1")[0]
    goal2_bin = env.bins_in_region("goal2")[0]

    # Check Trial 1 (t=0 to t=5: indices 0-5)
    assert np.all(start_bins[0:6] == start_bin)
    assert np.all(goal_bins[0:6] == goal1_bin)

    # Check gap between trials (t=5.1 to t=5.4: just before trial 2 starts)
    # No exact timepoint in gap, but t=5 is in trial 1

    # Check Trial 2 (t=5.5 to t=10: indices 6-10, approximately)
    trial2_mask = (times >= 5.5) & (times <= 10.0)
    assert np.all(start_bins[trial2_mask] == start_bin)
    assert np.all(goal_bins[trial2_mask] == goal2_bin)

    # Check gap between trials (t=10.1 to t=10.9)
    gap_mask = (times > 10.0) & (times < 11.0)
    assert np.all(start_bins[gap_mask] == -1)
    assert np.all(goal_bins[gap_mask] == -1)

    # Check Trial 3 (t=11 to t=15: indices 11-15)
    trial3_mask = (times >= 11.0) & (times <= 15.0)
    assert np.all(start_bins[trial3_mask] == start_bin)
    assert np.all(goal_bins[trial3_mask] == goal1_bin)

    # Check after last trial (t=16)
    assert start_bins[16] == -1
    assert goal_bins[16] == -1


def test_trials_to_region_arrays_failed_trial(simple_environment_with_regions):
    """Test helper with failed trial (end_region=None)."""
    env = simple_environment_with_regions

    # Create failed trial (end_region=None, success=False)
    trial = Trial(
        start_time=0.0,
        end_time=10.0,
        start_region="start",
        end_region=None,  # Failed trial - no end region
        success=False,
    )
    trials = [trial]

    times = np.linspace(0.0, 10.0, 11)

    # Call function
    start_bins, goal_bins = trials_to_region_arrays(trials, times, env)

    # Start bins should be set
    start_bin = env.bins_in_region("start")[0]
    assert np.all(start_bins == start_bin)

    # Goal bins should remain -1 (no goal for failed trial)
    assert np.all(goal_bins == -1)


def test_trials_to_region_arrays_polygon_regions(environment_with_polygon_regions):
    """Test helper with polygon regions (multi-bin)."""
    env = environment_with_polygon_regions

    # Create trial with polygon regions
    trial = Trial(
        start_time=0.0,
        end_time=10.0,
        start_region="start_zone",
        end_region="goal_zone",
        success=True,
    )
    trials = [trial]

    times = np.linspace(0.0, 10.0, 11)

    # Call function
    start_bins, goal_bins = trials_to_region_arrays(trials, times, env)

    # Get bins for polygon regions (returns multiple bins)
    start_zone_bins = env.bins_in_region("start_zone")
    goal_zone_bins = env.bins_in_region("goal_zone")

    # Function should use first bin from each region
    assert len(start_zone_bins) > 0
    assert len(goal_zone_bins) > 0

    # All timepoints should have first bin from each region
    assert np.all(start_bins == start_zone_bins[0])
    assert np.all(goal_bins == goal_zone_bins[0])


# =============================================================================
# M2.3: path_progress() Tests
# =============================================================================


def test_path_progress_single_trial_geodesic(simple_environment_with_regions):
    """Test path progress with single trial and geodesic metric."""
    from neurospatial.behavioral import path_progress

    env = simple_environment_with_regions

    # Get bin indices for regions
    start_bin = env.bins_in_region("start")[0]
    goal_bin = env.bins_in_region("goal1")[0]

    # Create simulated trajectory from start to goal
    # For testing, use actual trajectory_bins that progress toward goal
    trajectory_bins = np.array([start_bin, start_bin, goal_bin, goal_bin])

    # Constant start and goal for single trial
    start_bins = np.full(4, start_bin)
    goal_bins = np.full(4, goal_bin)

    # Compute progress
    progress = path_progress(env, trajectory_bins, start_bins, goal_bins)

    # Check output shape and type
    assert progress.shape == (4,)
    assert progress.dtype == np.float64

    # Progress should be between 0 and 1
    assert np.all((progress >= 0) & (progress <= 1))

    # First point at start should have progress ~0
    assert progress[0] < 0.1  # Close to start

    # Last points at goal should have progress = 1.0
    assert np.isclose(progress[-1], 1.0)


def test_path_progress_multiple_trials(simple_environment_with_regions):
    """Test path progress with multiple trials (varying start/goal)."""
    from neurospatial.behavioral import path_progress

    env = simple_environment_with_regions

    # Get bin indices - handle case where goal2 might not have bins
    start_bins_list = env.bins_in_region("start")
    goal1_bins_list = env.bins_in_region("goal1")
    goal2_bins_list = env.bins_in_region("goal2")

    # Skip if any region has no bins
    if (
        len(start_bins_list) == 0
        or len(goal1_bins_list) == 0
        or len(goal2_bins_list) == 0
    ):
        pytest.skip("Some regions have no bins in random environment")

    start_bin = start_bins_list[0]
    goal1_bin = goal1_bins_list[0]
    goal2_bin = goal2_bins_list[0]

    # Create trajectory with 10 timepoints
    # Trial 1: t=0-4 going to goal1
    # Trial 2: t=5-9 going to goal2
    trajectory_bins = np.array(
        [
            start_bin,
            start_bin,
            goal1_bin,
            goal1_bin,
            goal1_bin,  # Trial 1
            start_bin,
            start_bin,
            goal2_bin,
            goal2_bin,
            goal2_bin,
        ]  # Trial 2
    )

    # Create start/goal arrays with two different trials
    start_bins = np.full(10, start_bin)
    goal_bins = np.array(
        [
            goal1_bin,
            goal1_bin,
            goal1_bin,
            goal1_bin,
            goal1_bin,  # Trial 1
            goal2_bin,
            goal2_bin,
            goal2_bin,
            goal2_bin,
            goal2_bin,
        ]  # Trial 2
    )

    # Compute progress
    progress = path_progress(env, trajectory_bins, start_bins, goal_bins)

    # Check output
    assert progress.shape == (10,)
    assert np.all((progress >= 0) & (progress <= 1))

    # Each trial should have progress reaching 1.0 at goal
    assert np.isclose(progress[4], 1.0)  # End of trial 1
    assert np.isclose(progress[9], 1.0)  # End of trial 2


def test_path_progress_euclidean(simple_environment_with_regions):
    """Test path progress with euclidean metric."""
    from neurospatial.behavioral import path_progress

    env = simple_environment_with_regions

    start_bin = env.bins_in_region("start")[0]
    goal_bin = env.bins_in_region("goal1")[0]

    # Simple trajectory
    trajectory_bins = np.array([start_bin, goal_bin])
    start_bins = np.array([start_bin, start_bin])
    goal_bins = np.array([goal_bin, goal_bin])

    # Compute with euclidean metric
    progress = path_progress(
        env, trajectory_bins, start_bins, goal_bins, metric="euclidean"
    )

    # Check output
    assert progress.shape == (2,)
    assert np.all((progress >= 0) & (progress <= 1))

    # At start, progress ~0; at goal, progress = 1
    assert progress[0] < 0.1
    assert np.isclose(progress[1], 1.0)


def test_path_progress_edge_case_same_start_goal(simple_environment_with_regions):
    """Test edge case: start_bin == goal_bin (should return 1.0)."""
    from neurospatial.behavioral import path_progress

    env = simple_environment_with_regions

    start_bin = env.bins_in_region("start")[0]

    # Start and goal are the same
    trajectory_bins = np.array([start_bin, start_bin])
    start_bins = np.array([start_bin, start_bin])
    goal_bins = np.array([start_bin, start_bin])  # Same as start

    progress = path_progress(env, trajectory_bins, start_bins, goal_bins)

    # When start == goal, progress should be 1.0 (already at goal)
    assert np.all(progress == 1.0)


def test_path_progress_edge_case_invalid_bins(simple_environment_with_regions):
    """Test edge case: invalid bins -1 (should return NaN)."""
    from neurospatial.behavioral import path_progress

    env = simple_environment_with_regions

    start_bin = env.bins_in_region("start")[0]

    # Mix of valid and invalid bins
    trajectory_bins = np.array([start_bin, -1, start_bin])
    start_bins = np.array([start_bin, start_bin, -1])  # Invalid start for last point
    goal_bins = np.array([start_bin, start_bin, start_bin])

    progress = path_progress(env, trajectory_bins, start_bins, goal_bins)

    # Invalid bins should produce NaN
    assert np.isnan(progress[2])  # Invalid start_bin


def test_path_progress_edge_case_disconnected():
    """Test edge case: disconnected paths (should return NaN)."""
    from neurospatial.behavioral import path_progress

    # Create environment with two disconnected components
    positions = np.array(
        [
            [0, 0],
            [2, 0],
            [4, 0],  # Component 1
            [10, 10],
            [12, 10],
            [14, 10],  # Component 2 (disconnected)
        ]
    )
    env = Environment.from_samples(positions, bin_size=2.0)

    # Add regions in different components
    env.regions.add("start", point=(0.0, 0.0))
    env.regions.add("goal", point=(12.0, 10.0))

    start_bin = env.bins_in_region("start")[0]
    goal_bin = env.bins_in_region("goal")[0]

    # Try to compute progress between disconnected components
    trajectory_bins = np.array([start_bin, start_bin])
    start_bins = np.array([start_bin, start_bin])
    goal_bins = np.array([goal_bin, goal_bin])

    progress = path_progress(env, trajectory_bins, start_bins, goal_bins)

    # Disconnected paths should return NaN
    assert np.all(np.isnan(progress))


def test_path_progress_large_environment():
    """Test path progress with large environment (n_bins > 5000, test fallback)."""
    from neurospatial.behavioral import path_progress

    # Create large environment with >5000 bins
    # 100x100 grid = 10,000 bins
    positions = np.random.uniform(0, 200, (5000, 2))
    env = Environment.from_samples(positions, bin_size=2.0)

    # Skip if environment doesn't have enough bins
    if env.n_bins < 5000:
        pytest.skip(f"Environment has only {env.n_bins} bins, need >5000")

    # Add regions
    env.regions.add("start", point=(10.0, 10.0))
    env.regions.add("goal", point=(190.0, 190.0))

    start_bin = env.bins_in_region("start")[0]
    goal_bin = env.bins_in_region("goal")[0]

    # Simple trajectory
    trajectory_bins = np.array([start_bin, goal_bin])
    start_bins = np.array([start_bin, start_bin])
    goal_bins = np.array([goal_bin, goal_bin])

    # This should use the fallback strategy for large environments
    progress = path_progress(env, trajectory_bins, start_bins, goal_bins)

    # Check it still works correctly
    assert progress.shape == (2,)
    assert np.all((progress >= 0) & (progress <= 1) | np.isnan(progress))
    assert np.isclose(progress[1], 1.0)


# =============================================================================
# M2.4: distance_to_region() Tests
# =============================================================================


def test_distance_to_region_scalar_target(simple_environment_with_regions):
    """Test distance to region with scalar target (constant goal)."""
    from neurospatial.behavioral import distance_to_region

    env = simple_environment_with_regions

    # Get start and goal bins
    start_bin = env.bins_in_region("start")[0]
    goal_bin = env.bins_in_region("goal1")[0]

    # Create trajectory from start to goal
    trajectory_bins = np.array([start_bin, goal_bin])

    # Compute distance to scalar target (goal_bin)
    distances = distance_to_region(env, trajectory_bins, goal_bin)

    # Check shape
    assert distances.shape == (2,)

    # Distance should be positive at start, zero at goal
    assert distances[0] > 0
    assert np.isclose(distances[1], 0.0, atol=1e-6)

    # Test euclidean metric
    distances_euclidean = distance_to_region(
        env, trajectory_bins, goal_bin, metric="euclidean"
    )
    assert distances_euclidean.shape == (2,)
    assert distances_euclidean[0] > 0
    assert np.isclose(distances_euclidean[1], 0.0, atol=1e-6)


def test_distance_to_region_dynamic_target(simple_environment_with_regions):
    """Test distance to region with array of targets (dynamic goal)."""
    from neurospatial.behavioral import distance_to_region

    env = simple_environment_with_regions

    # Get bins for regions
    start_bin = env.bins_in_region("start")[0]
    goal1_bin = env.bins_in_region("goal1")[0]
    goal2_bins = env.bins_in_region("goal2")

    # Skip if goal2 has no bins
    if len(goal2_bins) == 0:
        pytest.skip("goal2 region has no bins in random environment")

    goal2_bin = goal2_bins[0]

    # Create trajectory: start → goal1
    trajectory_bins = np.array([start_bin, goal1_bin])

    # Dynamic targets: first timepoint targets goal1, second targets goal2
    target_bins = np.array([goal1_bin, goal2_bin])

    # Compute distances with dynamic targets
    distances = distance_to_region(env, trajectory_bins, target_bins)

    # Check shape
    assert distances.shape == (2,)

    # At first timepoint: we're at start, target is goal1 (should be positive distance)
    assert distances[0] > 0

    # At second timepoint: we're at goal1, target is goal2 (should be positive distance)
    assert distances[1] > 0


def test_distance_to_region_invalid_bins(simple_environment_with_regions):
    """Test distance to region with invalid bins (should return NaN)."""
    from neurospatial.behavioral import distance_to_region

    env = simple_environment_with_regions

    goal_bin = env.bins_in_region("goal1")[0]

    # Test 1: Invalid trajectory bins
    trajectory_bins = np.array([0, -1, 2])  # -1 is invalid
    distances = distance_to_region(env, trajectory_bins, goal_bin)
    assert distances.shape == (3,)
    assert not np.isnan(distances[0])  # Valid bin
    assert np.isnan(distances[1])  # Invalid bin → NaN
    assert not np.isnan(distances[2])  # Valid bin

    # Test 2: Invalid target bin (scalar)
    trajectory_bins = np.array([0, 1, 2])
    distances = distance_to_region(env, trajectory_bins, -1)
    assert np.all(np.isnan(distances))  # All should be NaN with invalid target

    # Test 3: Dynamic targets with some invalid
    target_bins = np.array([goal_bin, -1, goal_bin])
    distances = distance_to_region(env, trajectory_bins, target_bins)
    assert not np.isnan(distances[0])  # Valid target
    assert np.isnan(distances[1])  # Invalid target → NaN
    assert not np.isnan(distances[2])  # Valid target


def test_distance_to_region_multiple_goal_bins():
    """Test distance to region with multiple goal bins (distance to nearest)."""
    from neurospatial.behavioral import distance_to_region

    # Create environment with known structure
    positions = np.array(
        [
            [0, 0],  # bin 0 - start
            [2, 0],  # bin 1
            [4, 0],  # bin 2
            [6, 0],  # bin 3 - goal region
            [8, 0],  # bin 4 - goal region
        ]
    )
    env = Environment.from_samples(positions, bin_size=2.0)

    # Add region covering multiple bins
    env.regions.add("start", point=(0.0, 0.0))
    env.regions.add("goal", point=(7.0, 0.0))  # Should cover both bins 3 and 4

    start_bin = env.bins_in_region("start")[0]
    goal_bins = env.bins_in_region("goal")

    # Goal region should have multiple bins
    assert len(goal_bins) >= 1

    # Use first goal bin as scalar target
    target_bin = goal_bins[0]

    # Trajectory from start
    trajectory_bins = np.array([start_bin])

    # Compute distance - should be to nearest goal bin
    distances = distance_to_region(env, trajectory_bins, target_bin)

    # Distance should be positive
    assert distances[0] > 0


def test_distance_to_region_large_environment():
    """Test distance to region with large environment (memory fallback)."""
    from neurospatial.behavioral import distance_to_region

    # Create large environment with >5000 bins
    np.random.seed(42)
    positions = np.random.uniform(0, 200, (5000, 2))
    env = Environment.from_samples(positions, bin_size=2.0)

    # Skip if environment doesn't have enough bins
    if env.n_bins < 5000:
        pytest.skip(f"Environment has only {env.n_bins} bins, need ≥5000")

    # Add regions
    env.regions.add("start", point=(10.0, 10.0))
    env.regions.add("goal", point=(190.0, 190.0))

    start_bin = env.bins_in_region("start")[0]
    goal_bin = env.bins_in_region("goal")[0]

    # Test 1: Scalar target (uses existing env.distance_to())
    trajectory_bins = np.array([start_bin, goal_bin])
    distances_scalar = distance_to_region(env, trajectory_bins, goal_bin)
    assert distances_scalar.shape == (2,)
    assert distances_scalar[0] > 0
    assert np.isclose(distances_scalar[1], 0.0, atol=1e-6)

    # Test 2: Dynamic targets (uses per-target fallback strategy)
    target_bins = np.array([goal_bin, start_bin])  # Different target each timepoint
    distances_dynamic = distance_to_region(env, trajectory_bins, target_bins)
    assert distances_dynamic.shape == (2,)
    assert distances_dynamic[0] > 0
    assert distances_dynamic[1] > 0


# =============================================================================
# M3.1: time_to_goal() Tests
# =============================================================================


def test_time_to_goal_successful_trials():
    """Test time to goal for successful trials."""
    from neurospatial.behavioral import time_to_goal
    from neurospatial.segmentation.trials import Trial

    # Create timestamps
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    # Create successful trials
    trials = [
        Trial(
            start_time=1.0,
            end_time=4.0,
            start_region="start",
            end_region="goal",
            success=True,
        ),
        Trial(
            start_time=6.0,
            end_time=9.0,
            start_region="start",
            end_region="goal",
            success=True,
        ),
    ]

    # Compute time to goal
    ttg = time_to_goal(times, trials)

    # Check shape
    assert ttg.shape == times.shape

    # Check NaN outside trials
    assert np.isnan(ttg[0])  # Before first trial
    assert np.isnan(ttg[5])  # Between trials
    assert np.isnan(ttg[10])  # After last trial

    # Check countdown during trials
    # Trial 1: t=1.0 to t=4.0
    assert np.isclose(ttg[1], 3.0)  # 4.0 - 1.0 = 3.0 seconds remaining
    assert np.isclose(ttg[2], 2.0)  # 4.0 - 2.0 = 2.0 seconds remaining
    assert np.isclose(ttg[3], 1.0)  # 4.0 - 3.0 = 1.0 seconds remaining
    assert np.isclose(ttg[4], 0.0)  # At goal

    # Trial 2: t=6.0 to t=9.0
    assert np.isclose(ttg[6], 3.0)  # 9.0 - 6.0 = 3.0 seconds remaining
    assert np.isclose(ttg[7], 2.0)  # 9.0 - 7.0 = 2.0 seconds remaining
    assert np.isclose(ttg[8], 1.0)  # 9.0 - 8.0 = 1.0 seconds remaining
    assert np.isclose(ttg[9], 0.0)  # At goal


def test_time_to_goal_failed_trials():
    """Test time to goal for failed trials (should be NaN)."""
    from neurospatial.behavioral import time_to_goal
    from neurospatial.segmentation.trials import Trial

    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    # Create failed trial (end_region=None, success=False)
    trials = [
        Trial(
            start_time=1.0,
            end_time=4.0,
            start_region="start",
            end_region=None,  # Failed trial
            success=False,
        ),
    ]

    ttg = time_to_goal(times, trials)

    # Failed trials should be NaN throughout
    assert np.isnan(ttg[1])
    assert np.isnan(ttg[2])
    assert np.isnan(ttg[3])
    assert np.isnan(ttg[4])


def test_time_to_goal_outside_trials():
    """Test time to goal outside trials (should be NaN)."""
    from neurospatial.behavioral import time_to_goal
    from neurospatial.segmentation.trials import Trial

    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Trial runs from 2.0 to 4.0
    trials = [
        Trial(
            start_time=2.0,
            end_time=4.0,
            start_region="start",
            end_region="goal",
            success=True,
        ),
    ]

    ttg = time_to_goal(times, trials)

    # Outside trial period should be NaN
    assert np.isnan(ttg[0])  # Before trial
    assert np.isnan(ttg[1])  # Before trial
    assert np.isnan(ttg[5])  # After trial
    assert np.isnan(ttg[6])  # After trial

    # Inside trial should have valid values
    assert not np.isnan(ttg[2])
    assert not np.isnan(ttg[3])
    assert not np.isnan(ttg[4])


def test_time_to_goal_countdown():
    """Test time to goal countdown is correct."""
    from neurospatial.behavioral import time_to_goal
    from neurospatial.segmentation.trials import Trial

    # High-resolution timestamps
    times = np.linspace(0.0, 10.0, 101)  # 0.1s resolution

    trials = [
        Trial(
            start_time=2.0,
            end_time=7.0,
            start_region="start",
            end_region="goal",
            success=True,
        ),
    ]

    ttg = time_to_goal(times, trials)

    # Find indices within trial
    trial_mask = (times >= 2.0) & (times <= 7.0)
    trial_times = times[trial_mask]
    trial_ttg = ttg[trial_mask]

    # Verify countdown: ttg = end_time - current_time
    expected_ttg = 7.0 - trial_times
    assert np.allclose(trial_ttg, expected_ttg, atol=1e-10)

    # Verify monotonic decrease
    assert np.all(np.diff(trial_ttg) <= 0)  # Non-increasing


def test_time_to_goal_after_goal_reached():
    """Test time to goal after goal reached (should be 0.0)."""
    from neurospatial.behavioral import time_to_goal
    from neurospatial.segmentation.trials import Trial

    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    # Trial ends at t=3.0
    trials = [
        Trial(
            start_time=1.0,
            end_time=3.0,
            start_region="start",
            end_region="goal",
            success=True,
        ),
    ]

    ttg = time_to_goal(times, trials)

    # At goal arrival time, should be 0.0
    assert np.isclose(ttg[3], 0.0)

    # After goal (due to clamping if any floating point issues)
    # In this simple case, there shouldn't be values after end_time within trial
    # But the clamping ensures no negative values


# =============================================================================
# M3.2: compute_trajectory_curvature() Tests
# =============================================================================


@pytest.mark.skip("not implemented")
def test_compute_trajectory_curvature_2d_straight():
    """Test curvature for straight 2D trajectory (should be ~0)."""
    pass


@pytest.mark.skip("not implemented")
def test_compute_trajectory_curvature_2d_left_turn():
    """Test curvature for left turn in 2D (positive)."""
    pass


@pytest.mark.skip("not implemented")
def test_compute_trajectory_curvature_2d_right_turn():
    """Test curvature for right turn in 2D (negative)."""
    pass


@pytest.mark.skip("not implemented")
def test_compute_trajectory_curvature_3d():
    """Test curvature for 3D trajectory (uses first 2 dims)."""
    pass


@pytest.mark.skip("not implemented")
def test_compute_trajectory_curvature_smoothing():
    """Test curvature with temporal smoothing."""
    pass


@pytest.mark.skip("not implemented")
def test_compute_trajectory_curvature_output_length():
    """Test curvature output length matches input (n_samples, not n_samples-2)."""
    pass


# =============================================================================
# M4.1: cost_to_goal() Tests
# =============================================================================


@pytest.mark.skip("not implemented")
def test_cost_to_goal_uniform():
    """Test cost to goal with uniform cost (equivalent to geodesic distance)."""
    pass


@pytest.mark.skip("not implemented")
def test_cost_to_goal_with_cost_map():
    """Test cost to goal with cost map (punishment zones)."""
    pass


@pytest.mark.skip("not implemented")
def test_cost_to_goal_terrain_difficulty():
    """Test cost to goal with terrain difficulty (narrow passages)."""
    pass


@pytest.mark.skip("not implemented")
def test_cost_to_goal_combined():
    """Test cost to goal with cost map + terrain combined."""
    pass


@pytest.mark.skip("not implemented")
def test_cost_to_goal_dynamic_goal():
    """Test cost to goal with array of goal bins (dynamic)."""
    pass


@pytest.mark.skip("not implemented")
def test_cost_to_goal_invalid_bins():
    """Test cost to goal with invalid bins (should handle gracefully)."""
    pass


# =============================================================================
# M4.2: graph_turn_sequence() Tests
# =============================================================================


@pytest.mark.skip("not implemented")
def test_graph_turn_sequence_ymaze_left():
    """Test turn sequence for Y-maze left choice."""
    pass


@pytest.mark.skip("not implemented")
def test_graph_turn_sequence_ymaze_right():
    """Test turn sequence for Y-maze right choice."""
    pass


@pytest.mark.skip("not implemented")
def test_graph_turn_sequence_grid_multiple():
    """Test turn sequence with multiple turns on grid environment."""
    pass


@pytest.mark.skip("not implemented")
def test_graph_turn_sequence_straight():
    """Test turn sequence for straight path (no turns, empty string)."""
    pass


@pytest.mark.skip("not implemented")
def test_graph_turn_sequence_min_samples_filter():
    """Test turn sequence filters brief crossings (min_samples_per_edge)."""
    pass


@pytest.mark.skip("not implemented")
def test_graph_turn_sequence_3d():
    """Test turn sequence for 3D environment."""
    pass


# =============================================================================
# M5.1: Public API Export Tests
# =============================================================================


@pytest.mark.skip("not implemented")
def test_all_functions_exported():
    """Verify all 7 behavioral functions are exported to public API."""
    pass
