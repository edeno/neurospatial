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
    # Create a 10x10 cm space with 2 cm bins (5x5 grid = 25 bins)
    positions = np.array(
        [
            [0, 0],
            [0, 8],
            [8, 0],
            [8, 8],
            [4, 4],  # Corners and center
        ]
    )
    env = Environment.from_samples(positions, bin_size=2.0)
    env.units = "cm"

    # Add regions at known locations
    env.regions.add("start", point=(0.0, 0.0))  # Bottom-left corner
    env.regions.add("goal1", point=(8.0, 8.0))  # Top-right corner
    env.regions.add("goal2", point=(8.0, 0.0))  # Bottom-right corner

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


@pytest.mark.skip("not implemented")
def test_path_progress_single_trial_geodesic():
    """Test path progress with single trial and geodesic metric."""
    pass


@pytest.mark.skip("not implemented")
def test_path_progress_multiple_trials():
    """Test path progress with multiple trials (varying start/goal)."""
    pass


@pytest.mark.skip("not implemented")
def test_path_progress_euclidean():
    """Test path progress with euclidean metric."""
    pass


@pytest.mark.skip("not implemented")
def test_path_progress_edge_case_same_start_goal():
    """Test edge case: start_bin == goal_bin (should return 1.0)."""
    pass


@pytest.mark.skip("not implemented")
def test_path_progress_edge_case_disconnected():
    """Test edge case: disconnected paths (should return NaN)."""
    pass


@pytest.mark.skip("not implemented")
def test_path_progress_edge_case_invalid_bins():
    """Test edge case: invalid bins -1 (should return NaN)."""
    pass


@pytest.mark.skip("not implemented")
def test_path_progress_large_environment():
    """Test path progress with large environment (n_bins > 5000, test fallback)."""
    pass


# =============================================================================
# M2.4: distance_to_region() Tests
# =============================================================================


@pytest.mark.skip("not implemented")
def test_distance_to_region_scalar_target():
    """Test distance to region with scalar target (constant goal)."""
    pass


@pytest.mark.skip("not implemented")
def test_distance_to_region_dynamic_target():
    """Test distance to region with array of targets (dynamic goal)."""
    pass


@pytest.mark.skip("not implemented")
def test_distance_to_region_invalid_bins():
    """Test distance to region with invalid bins (should return NaN)."""
    pass


@pytest.mark.skip("not implemented")
def test_distance_to_region_multiple_goal_bins():
    """Test distance to region with multiple goal bins (distance to nearest)."""
    pass


@pytest.mark.skip("not implemented")
def test_distance_to_region_large_environment():
    """Test distance to region with large environment (memory fallback)."""
    pass


# =============================================================================
# M3.1: time_to_goal() Tests
# =============================================================================


@pytest.mark.skip("not implemented")
def test_time_to_goal_successful_trials():
    """Test time to goal for successful trials."""
    pass


@pytest.mark.skip("not implemented")
def test_time_to_goal_failed_trials():
    """Test time to goal for failed trials (should be NaN)."""
    pass


@pytest.mark.skip("not implemented")
def test_time_to_goal_outside_trials():
    """Test time to goal outside trials (should be NaN)."""
    pass


@pytest.mark.skip("not implemented")
def test_time_to_goal_countdown():
    """Test time to goal countdown is correct."""
    pass


@pytest.mark.skip("not implemented")
def test_time_to_goal_after_goal_reached():
    """Test time to goal after goal reached (should be 0.0)."""
    pass


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
