"""Tests for behavioral analysis and goal-directed navigation metrics."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import BinIndexOutOfRangeError, Environment, Intervals, Regions
from neurospatial.behavior.segmentation import Trial

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_environment_with_regions():
    """Create a simple 2D environment with point regions for testing.

    Uses deterministic grid - no randomness needed for this fixture.
    """
    # Create deterministic 10x10 cm grid with bin_size=2.0 (6x6 = 36 bins)
    x = np.linspace(0, 10, 6)
    y = np.linspace(0, 10, 6)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])

    env = Environment.from_samples(positions, bin_size=2.0)
    env = env.with_metadata(units="cm")

    # Add regions at bin centers (0, 2, 4, 6, 8, 10 in each dimension)
    region_builder = Regions(env.regions.values())
    region_builder.add("start", point=(0.0, 0.0))  # Bottom-left bin
    region_builder.add("goal1", point=(10.0, 10.0))  # Top-right bin
    region_builder.add("goal2", point=(10.0, 0.0))
    return env.with_regions(region_builder)  # Bottom-right bin


@pytest.fixture
def environment_with_polygon_regions():
    """Create environment with polygon regions (multi-bin).

    Uses deterministic grid - no randomness needed for this fixture.
    """
    # Create deterministic 20x20 cm grid with bin_size=2.0 (11x11 = 121 bins)
    x = np.linspace(0, 20, 11)
    y = np.linspace(0, 20, 11)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    env = Environment.from_samples(positions, bin_size=2.0)
    env = env.with_metadata(units="cm")

    # Polygon regions covering 2 x 2 bins in opposite corners; their edges
    # lie between bin centres, so the 180-degree turn about (10, 10) swaps them.
    region_builder = Regions(env.regions.values())
    region_builder.add("start_zone", polygon=box(-1, -1, 3, 3))
    region_builder.add("goal_zone", polygon=box(17, 17, 21, 21))
    return env.with_regions(region_builder)


# =============================================================================
# path_progress() Tests
# =============================================================================


def test_path_progress_runs_from_start_to_goal(environment_with_polygon_regions):
    """0 on the start zone, 1 on the goal zone, 1/2 where symmetry demands it.

    Turning the arena by 180 degrees about its centre swaps the start and goal
    zones, so progress p there becomes 1 - p: every place the turn maps onto
    itself, or onto its mirror across the anti-diagonal, reads 1/2.
    """
    from neurospatial.behavior.navigation import path_progress

    env = environment_with_polygon_regions
    positions = np.array(
        [[0.0, 0.0], [2.0, 2.0], [10.0, 10.0], [6.0, 14.0], [18.0, 18.0], [20.0, 20.0]]
    )
    progress = path_progress(
        env,
        np.arange(len(positions), dtype=float),
        positions,
        start="start_zone",
        goal="goal_zone",
    )
    assert progress.dtype == np.float64
    np.testing.assert_allclose(progress, [0.0, 0.0, 0.5, 0.5, 1.0, 1.0], atol=1e-12)


def test_path_progress_outside_the_environment_is_nan(
    environment_with_polygon_regions,
):
    """A NaN row or a position outside the environment gives NaN, not 0."""
    from neurospatial.behavior.navigation import path_progress

    env = environment_with_polygon_regions
    positions = np.array(
        [[0.0, 0.0], [np.nan, np.nan], [10.0, 10.0], [100.0, 100.0], [20.0, 20.0]]
    )
    progress = path_progress(
        env, np.arange(5.0), positions, start="start_zone", goal="goal_zone"
    )
    np.testing.assert_allclose(progress, [0.0, np.nan, 0.5, np.nan, 1.0], atol=1e-12)


def test_path_progress_start_and_goal_in_different_components():
    """No path joins start and goal, so progress is undefined everywhere."""
    from neurospatial.behavior.navigation import path_progress

    left = np.column_stack([np.arange(0.0, 20.0, 0.5), np.zeros(40)])
    env = Environment.from_samples(
        np.vstack([left, left + np.array([100.0, 0.0])]), bin_size=2.0
    )
    positions = np.array([[1.0, 0.0], [10.0, 0.0], [118.0, 0.0]])
    with pytest.warns(UserWarning, match=r"does not connect to both start and goal"):
        progress = path_progress(
            env,
            np.arange(3.0),
            positions,
            start=[1.0, 0.0],
            goal=[118.0, 0.0],
            point_radius=2.5,
        )
    assert np.all(np.isnan(progress))


# =============================================================================
# distance_to_goal() Tests
# =============================================================================


def test_distance_to_goal_point_region(simple_environment_with_regions):
    """The grid is convex, so the shortest path is the straight line."""
    from neurospatial.behavior.navigation import distance_to_goal

    env = simple_environment_with_regions
    goal_center = env.bin_centers[env.bins_in_region("goal1")[0]]
    positions = np.array([[0.0, 0.0], [4.0, 6.0], goal_center])
    expected = np.linalg.norm(positions - goal_center, axis=1)
    for metric in ("geodesic", "euclidean"):
        distances = distance_to_goal(
            env, np.arange(3.0), positions, goal="goal1", metric=metric
        )
        np.testing.assert_allclose(distances, expected, rtol=1e-12)
        assert distances[-1] == 0.0


def test_distance_to_goal_per_interval_goals(simple_environment_with_regions):
    """Each trial measures to its own goal; an unknown goal gives NaN."""
    from neurospatial.behavior.navigation import distance_to_goal

    env = simple_environment_with_regions
    goal1 = env.bin_centers[env.bins_in_region("goal1")[0]]
    goal2 = env.bin_centers[env.bins_in_region("goal2")[0]]
    positions = np.array([[0.0, 0.0], [4.0, 4.0], [0.0, 0.0], [4.0, 6.0], [2.0, 2.0]])
    trials = Intervals(
        [
            Trial(0.0, 2.0, "start", "goal1", "reached"),
            Trial(2.0, 4.0, "start", "goal2", "reached"),
            Trial(4.0, 5.0, "start", None, "timeout"),
        ]
    )
    distances = distance_to_goal(
        env, np.arange(5.0), positions, trials, goal=trials.end_region
    )
    expected = np.r_[
        np.linalg.norm(positions[:2] - goal1, axis=1),
        np.linalg.norm(positions[2:4] - goal2, axis=1),
        np.nan,
    ]
    np.testing.assert_allclose(distances, expected, rtol=1e-12)


def test_distance_to_goal_outside_the_environment_is_nan(
    simple_environment_with_regions,
):
    from neurospatial.behavior.navigation import distance_to_goal

    env = simple_environment_with_regions
    positions = np.array(
        [[0.0, 0.0], [np.nan, np.nan], [4.0, 4.0], [50.0, 50.0], [6.0, 6.0]]
    )
    distances = distance_to_goal(env, np.arange(5.0), positions, goal="goal1")
    assert np.isnan(distances[[1, 3]]).all()
    assert np.isfinite(distances[[0, 2, 4]]).all()


# =============================================================================
# time_to_goal() Tests
# =============================================================================


def test_time_to_goal_successful_trials():
    """Test time to goal for successful trials."""
    from neurospatial.behavior.navigation import time_to_goal
    from neurospatial.behavior.segmentation import Trial

    # Create timestamps
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    # Create successful trials
    trials = Intervals(
        [
            Trial(
                start_time=1.0,
                stop_time=4.0,
                start_region="start",
                end_region="goal",
                outcome="reached",
            ),
            Trial(
                start_time=6.0,
                stop_time=9.0,
                start_region="start",
                end_region="goal",
                outcome="reached",
            ),
        ]
    )

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
    assert np.isnan(ttg[4])  # The arrival sample is outside [start, stop)

    # Trial 2: t=6.0 to t=9.0
    assert np.isclose(ttg[6], 3.0)  # 9.0 - 6.0 = 3.0 seconds remaining
    assert np.isclose(ttg[7], 2.0)  # 9.0 - 7.0 = 2.0 seconds remaining
    assert np.isclose(ttg[8], 1.0)  # 9.0 - 8.0 = 1.0 seconds remaining
    assert np.isnan(ttg[9])  # Arrival


def test_time_to_goal_failed_trials():
    """Test time to goal for failed trials (should be NaN)."""
    from neurospatial.behavior.navigation import time_to_goal
    from neurospatial.behavior.segmentation import Trial

    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    # Create failed trial (end_region=None, success=False)
    trials = Intervals(
        [
            Trial(
                start_time=1.0,
                stop_time=4.0,
                start_region="start",
                end_region=None,  # Failed trial
                outcome="timeout",
            ),
        ]
    )

    ttg = time_to_goal(times, trials)

    # Failed trials should be NaN throughout
    assert np.isnan(ttg[1])
    assert np.isnan(ttg[2])
    assert np.isnan(ttg[3])
    assert np.isnan(ttg[4])


def test_time_to_goal_outside_trials():
    """Test time to goal outside trials (should be NaN)."""
    from neurospatial.behavior.navigation import time_to_goal
    from neurospatial.behavior.segmentation import Trial

    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Trial runs from 2.0 to 4.0
    trials = Intervals(
        [
            Trial(
                start_time=2.0,
                stop_time=4.0,
                start_region="start",
                end_region="goal",
                outcome="reached",
            ),
        ]
    )

    ttg = time_to_goal(times, trials)

    # Outside trial period should be NaN
    assert np.isnan(ttg[0])  # Before trial
    assert np.isnan(ttg[1])  # Before trial
    assert np.isnan(ttg[5])  # After trial
    assert np.isnan(ttg[6])  # After trial

    # Inside [2, 4) should have valid values; the arrival at 4 is outside
    assert not np.isnan(ttg[2])
    assert not np.isnan(ttg[3])
    assert np.isnan(ttg[4])


def test_time_to_goal_countdown():
    """Test time to goal countdown is correct."""
    from neurospatial.behavior.navigation import time_to_goal
    from neurospatial.behavior.segmentation import Trial

    # High-resolution timestamps
    times = np.linspace(0.0, 10.0, 101)  # 0.1s resolution

    trials = Intervals(
        [
            Trial(
                start_time=2.0,
                stop_time=7.0,
                start_region="start",
                end_region="goal",
                outcome="reached",
            ),
        ]
    )

    ttg = time_to_goal(times, trials)

    # Find indices within trial
    trial_mask = (times >= 2.0) & (times < 7.0)
    trial_times = times[trial_mask]
    trial_ttg = ttg[trial_mask]
    np.testing.assert_array_equal(np.isnan(ttg), ~trial_mask)

    # Verify countdown: ttg = stop_time - current_time
    expected_ttg = 7.0 - trial_times
    assert np.allclose(trial_ttg, expected_ttg, atol=1e-10)

    # Verify monotonic decrease
    assert np.all(np.diff(trial_ttg) <= 0)  # Non-increasing


def test_time_to_goal_arrival_is_outside_the_trial():
    """The arrival sample, at stop_time, is outside the half-open trial."""
    from neurospatial.behavior.navigation import time_to_goal
    from neurospatial.behavior.segmentation import Trial

    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    trials = Intervals([Trial(1.0, 3.0, "start", "goal", "reached")])

    ttg = time_to_goal(times, trials)

    np.testing.assert_array_equal(ttg, [np.nan, 2.0, 1.0, np.nan, np.nan, np.nan])


# =============================================================================
# cost_to_goal() Tests
# =============================================================================


def test_cost_to_goal_uniform(simple_environment_with_regions):
    """Test cost to goal with uniform cost (equivalent to geodesic distance)."""
    from neurospatial.behavior.navigation import cost_to_goal, distance_to_goal

    env = simple_environment_with_regions

    # Get bins
    start_bin = env.bins_in_region("start")[0]
    goal_bin = env.bins_in_region("goal1")[0]

    # Create simple trajectory: start → intermediate → goal
    position_bins = np.array([start_bin, 5, goal_bin])

    # Compute cost with uniform cost (no cost_map or terrain_difficulty)
    cost = cost_to_goal(position_bins, env, goal_bins=goal_bin)

    # Geodesic distance from the bin centres, for comparison
    distance = distance_to_goal(
        env, np.arange(3.0), env.bin_centers[position_bins], goal="goal1"
    )

    # With uniform cost, cost should equal geodesic distance
    np.testing.assert_array_almost_equal(cost, distance)

    # Cost should decrease as we approach goal
    assert cost[0] > cost[1] > cost[2]

    # Cost at goal should be 0
    assert cost[2] == 0.0


def test_cost_to_goal_with_cost_map(simple_environment_with_regions):
    """Test cost to goal with cost map (punishment zones)."""
    from neurospatial.behavior.navigation import cost_to_goal

    env = simple_environment_with_regions

    start_bin = env.bins_in_region("start")[0]
    goal_bin = env.bins_in_region("goal1")[0]

    # Create cost map with high cost in middle bins (punishment zone)
    cost_map = np.ones(env.n_bins)
    punishment_bins = [3, 4, 5]  # Middle bins have high cost
    cost_map[punishment_bins] = 10.0

    # Trajectory through punishment zone
    position_bins = np.array([start_bin, 4, goal_bin])

    # Compute cost with cost map
    cost_with_map = cost_to_goal(
        position_bins, env, goal_bins=goal_bin, cost_map=cost_map
    )

    # Compute cost without map
    cost_uniform = cost_to_goal(position_bins, env, goal_bins=goal_bin)

    # Cost with punishment zone should be higher than uniform cost
    assert cost_with_map[0] > cost_uniform[0]

    # Cost should still decrease as we approach goal
    assert cost_with_map[0] > cost_with_map[2]


def test_cost_to_goal_terrain_difficulty(simple_environment_with_regions):
    """Test cost to goal with terrain difficulty (narrow passages)."""
    from neurospatial.behavior.navigation import cost_to_goal

    env = simple_environment_with_regions

    start_bin = env.bins_in_region("start")[0]
    goal_bin = env.bins_in_region("goal1")[0]

    # Create terrain difficulty - make MOST bins difficult, not just a narrow passage
    # This ensures the cost is affected regardless of path
    terrain_difficulty = np.ones(env.n_bins) * 2.0  # Most bins are 2x harder
    easy_bins = [start_bin, goal_bin]  # Only start and goal are easy
    terrain_difficulty[easy_bins] = 1.0

    # Trajectory
    position_bins = np.array([start_bin, 5, goal_bin])

    # Compute cost with terrain difficulty
    cost_with_terrain = cost_to_goal(
        position_bins, env, goal_bins=goal_bin, terrain_difficulty=terrain_difficulty
    )

    # Compute cost without terrain
    cost_uniform = cost_to_goal(position_bins, env, goal_bins=goal_bin)

    # Cost with terrain difficulty should be higher (most bins are 2x harder)
    assert cost_with_terrain[0] > cost_uniform[0]

    # Cost should still decrease approaching goal
    assert cost_with_terrain[0] > cost_with_terrain[2]

    # Cost at goal should be 0
    assert cost_with_terrain[2] == 0.0


def test_cost_to_goal_combined(simple_environment_with_regions):
    """Test cost to goal with cost map + terrain combined."""
    from neurospatial.behavior.navigation import cost_to_goal

    env = simple_environment_with_regions

    start_bin = env.bins_in_region("start")[0]
    goal_bin = env.bins_in_region("goal1")[0]

    # Create both cost map and terrain difficulty
    cost_map = np.ones(env.n_bins)
    cost_map[3] = 5.0  # High cost at bin 3

    terrain_difficulty = np.ones(env.n_bins)
    terrain_difficulty[4] = 2.0  # 2x difficulty at bin 4

    # Trajectory
    position_bins = np.array([start_bin, 3, 4, goal_bin])

    # Compute with combined cost
    cost_combined = cost_to_goal(
        position_bins,
        env,
        goal_bins=goal_bin,
        cost_map=cost_map,
        terrain_difficulty=terrain_difficulty,
    )

    # Combined cost should account for both factors
    # All costs should be positive and decrease toward goal
    assert cost_combined[0] > 0

    # Cost should decrease as we approach goal
    assert cost_combined[0] > cost_combined[-1]

    # Cost at goal should be 0
    assert cost_combined[-1] == 0.0


def test_cost_to_goal_dynamic_goal(simple_environment_with_regions):
    """Test cost to goal with array of goal bins (dynamic)."""
    from neurospatial.behavior.navigation import cost_to_goal

    env = simple_environment_with_regions

    # Check goal2 exists
    goal2_bins = env.bins_in_region("goal2")
    if len(goal2_bins) == 0:
        pytest.skip("goal2 region has no bins in random environment")

    start_bin = env.bins_in_region("start")[0]
    goal1_bin = env.bins_in_region("goal1")[0]
    goal2_bin = goal2_bins[0]

    # Create trajectory with dynamic goals
    position_bins = np.array([start_bin, 5, goal1_bin])
    goal_bins_array = np.array(
        [goal1_bin, goal1_bin, goal2_bin]
    )  # Goal changes at last step

    # Compute cost with dynamic goals
    cost = cost_to_goal(position_bins, env, goal_bins=goal_bins_array)

    # Check shape
    assert cost.shape == (3,)

    # All costs should be non-negative
    assert np.all(cost >= 0)

    # At goal1 but targeting goal2, cost should be positive (distance from goal1 to goal2)
    assert cost[2] > 0.0


def test_cost_to_goal_invalid_bins(simple_environment_with_regions):
    """Test cost to goal with invalid bins (should handle gracefully)."""
    from neurospatial.behavior.navigation import cost_to_goal

    env = simple_environment_with_regions

    goal_bin = env.bins_in_region("goal1")[0]

    # Test 1: Invalid trajectory bins
    position_bins = np.array([0, -1, 2])  # -1 is invalid
    cost = cost_to_goal(position_bins, env, goal_bins=goal_bin)
    assert cost.shape == (3,)
    assert not np.isnan(cost[0])  # Valid bin
    assert np.isnan(cost[1])  # Invalid bin → NaN
    assert not np.isnan(cost[2])  # Valid bin

    # Test 2: -1 is the outside marker, never a single goal
    position_bins = np.array([0, 1, 2])
    with pytest.raises(BinIndexOutOfRangeError, match="-1 is the value bin_at"):
        cost_to_goal(position_bins, env, goal_bins=-1)

    # Test 3: Dynamic goals with some invalid
    goal_bins_array = np.array([goal_bin, -1, goal_bin])
    cost = cost_to_goal(position_bins, env, goal_bins=goal_bins_array)
    assert not np.isnan(cost[0])  # Valid goal
    assert np.isnan(cost[1])  # Invalid goal → NaN
    assert not np.isnan(cost[2])  # Valid goal


# =============================================================================
# graph_turn_sequence() Tests
# =============================================================================


def test_graph_turn_sequence_ymaze_north_arm_into_west_arm_is_right(ymaze_env):
    """Walking South down the North arm, the Northwest arm is on the right."""
    from neurospatial.behavior.navigation import graph_turn_sequence

    env = ymaze_env

    # Y-maze has bins along edges, not just at nodes

    bin_centers = env.bin_centers
    node_positions = {
        0: (0.0, 0.0),  # Center
        1: (0.0, 10.0),  # Straight
        2: (-7.0, 7.0),  # Left
    }

    def find_closest_bin(pos):
        distances = np.linalg.norm(bin_centers - pos, axis=1)
        return np.argmin(distances)

    # Find bins along the path: straight → center → left arm. Heading South,
    # the West side is the animal's right, so this is a right turn.
    straight_bin = find_closest_bin(node_positions[1])
    center_bin = find_closest_bin(node_positions[0])
    left_bin = find_closest_bin(node_positions[2])

    # Create trajectory with 3 segments, each with sufficient samples
    n_samples_per_segment = 40
    position_bins = np.concatenate(
        [
            np.full(n_samples_per_segment, straight_bin, dtype=np.int_),
            np.full(n_samples_per_segment, center_bin, dtype=np.int_),
            np.full(n_samples_per_segment, left_bin, dtype=np.int_),
        ]
    )

    # Call function
    turn_seq = graph_turn_sequence(
        env,
        position_bins,
    )

    assert turn_seq == "right"


def test_graph_turn_sequence_ymaze_north_arm_into_east_arm_is_left(ymaze_env):
    """Walking South down the North arm, the Northeast arm is on the left."""
    from neurospatial.behavior.navigation import graph_turn_sequence

    env = ymaze_env

    bin_centers = env.bin_centers
    node_positions = {
        0: (0.0, 0.0),  # Center
        1: (0.0, 10.0),  # Straight
        3: (7.0, 7.0),  # Right
    }

    def find_closest_bin(pos):
        distances = np.linalg.norm(bin_centers - pos, axis=1)
        return np.argmin(distances)

    # Find bins along the path: straight → center → right arm. Heading South,
    # the East side is the animal's left, so this is a left turn.
    straight_bin = find_closest_bin(node_positions[1])
    center_bin = find_closest_bin(node_positions[0])
    right_bin = find_closest_bin(node_positions[3])

    # Create trajectory with 3 segments, each with sufficient samples
    n_samples_per_segment = 40
    position_bins = np.concatenate(
        [
            np.full(n_samples_per_segment, straight_bin, dtype=np.int_),
            np.full(n_samples_per_segment, center_bin, dtype=np.int_),
            np.full(n_samples_per_segment, right_bin, dtype=np.int_),
        ]
    )

    # Call function
    turn_seq = graph_turn_sequence(
        env,
        position_bins,
    )

    assert turn_seq == "left"


def test_graph_turn_sequence_grid_multiple(small_2d_env):
    """Test turn sequence with multiple turns on grid environment."""
    from neurospatial.behavior.navigation import graph_turn_sequence

    env = small_2d_env

    # Create a trajectory with multiple turns in a grid
    # Simulate a path that makes several turns:
    # Start → Right → Up → Left

    # Find bins along a path
    bin_centers = env.bin_centers

    # Define a path that makes clear turns
    # Start at one corner and trace a rectangular path
    # Legs of four 2 cm bins: a one-bin leg is below the default tolerance
    # (one bin diagonal) and cannot be told apart from a lattice staircase.
    path_positions = [
        bin_centers[0],  # Start
        bin_centers[0] + [8, 0],  # Move right
        bin_centers[0] + [8, 8],  # Move up (North): a left turn
        bin_centers[0] + [0, 8],  # Move West: a left turn again
    ]

    # Find bins closest to each position
    def find_closest_bin(pos):
        distances = np.linalg.norm(bin_centers - pos, axis=1)
        return np.argmin(distances)

    path_bins = [find_closest_bin(pos) for pos in path_positions]

    # Create trajectory with sufficient samples per segment
    n_samples_per_segment = 60
    position_bins = []
    for bin_idx in path_bins:
        position_bins.extend([bin_idx] * n_samples_per_segment)
    position_bins = np.array(position_bins, dtype=np.int_)

    # Call function
    turn_seq = graph_turn_sequence(
        env,
        position_bins,
    )

    # East, then North, then West: two counterclockwise turns.
    assert turn_seq == "left-left"


def _turns_along(env, corners, samples_per_bin=60):
    """Turn labels for a walk that dwells in the bin at each corner, in order."""
    from neurospatial.behavior.navigation import graph_turn_sequence

    bins = env.bin_at(np.asarray(corners, dtype=float))
    assert np.all(bins >= 0)
    return graph_turn_sequence(env, np.repeat(bins, samples_per_bin))


@pytest.fixture(scope="module")
def turn_grid():
    """A 30 x 30 cm open field with 3 cm bins, x East and y North (y-up)."""
    x, y = np.meshgrid(np.arange(0.0, 30.0), np.arange(0.0, 30.0))
    return Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=3.0)


@pytest.mark.parametrize(
    ("corners", "expected"),
    [
        # Heading North, then West: counterclockwise, the animal's left.
        ([(16.5, 1.5), (16.5, 16.5), (1.5, 16.5)], "left"),
        # Heading North, then East: clockwise, the animal's right.
        ([(1.5, 1.5), (1.5, 16.5), (16.5, 16.5)], "right"),
        # Heading East, then North (left), then West (left).
        ([(1.5, 1.5), (16.5, 1.5), (16.5, 16.5), (1.5, 16.5)], "left-left"),
        # Heading East, then North (left), then East (right).
        ([(1.5, 1.5), (13.5, 1.5), (13.5, 13.5), (28.5, 13.5)], "left-right"),
    ],
)
def test_graph_turn_sequence_labels_turns_in_the_animal_frame(
    turn_grid, corners, expected
):
    """Left and right are egocentric: +pi/2 from the heading is left (y-up)."""
    assert _turns_along(turn_grid, corners) == expected


def test_graph_turn_sequence_straight(ymaze_env):
    """Test turn sequence for straight path (no turns, empty string)."""
    from neurospatial.behavior.navigation import graph_turn_sequence

    env = ymaze_env

    # Get bin centers to map nodes to bins
    bin_centers = env.bin_centers
    node_positions = {
        0: (0.0, 0.0),  # Center
        1: (0.0, 10.0),  # Straight
    }

    # Find bins closest to each node position
    def find_closest_bin(pos):
        distances = np.linalg.norm(bin_centers - pos, axis=1)
        return np.argmin(distances)

    center_bin = find_closest_bin(node_positions[0])
    straight_bin = find_closest_bin(node_positions[1])

    # Create trajectory: center → straight (no turn)
    n_samples = 100
    position_bins = np.full(n_samples, center_bin, dtype=np.int_)
    position_bins[50:] = straight_bin

    # Call function
    turn_seq = graph_turn_sequence(
        env,
        position_bins,
    )

    # Center → straight ahead is one direction, so no turn.
    assert turn_seq == ""


def test_graph_turn_sequence_one_step_is_not_a_turn(ymaze_env):
    """Two visited bins make one leg, which has no corner to turn at."""
    from neurospatial.behavior.navigation import graph_turn_sequence

    env = ymaze_env
    bin_centers = env.bin_centers

    def find_closest_bin(pos):
        return np.argmin(np.linalg.norm(bin_centers - pos, axis=1))

    center_bin = find_closest_bin((0.0, 0.0))
    left_bin = find_closest_bin((-7.0, 7.0))
    position_bins = np.r_[np.full(10, center_bin), np.full(5, left_bin)]

    assert graph_turn_sequence(env, position_bins) == ""


def test_graph_turn_sequence_3d(simple_3d_env):
    """Test turn sequence for 3D environment."""
    from neurospatial.behavior.navigation import graph_turn_sequence

    env = simple_3d_env

    # For 3D environment, the function should project to 2D
    # and classify turns based on the primary movement plane

    # Create a simple trajectory in 3D space
    # Find start and end bins in different locations
    start_bin = 0
    end_bin = min(10, env.n_bins - 1)  # Safe end bin

    # Create trajectory
    n_samples = 100
    position_bins = np.linspace(start_bin, end_bin, n_samples, dtype=np.int_)

    # Call function
    turn_seq = graph_turn_sequence(
        env,
        position_bins,
    )

    # Turns are read in the x-y plane. The walk climbs in z and steps along y
    # at constant x, so its x-y direction never changes: no turn to report.
    assert np.unique(env.bin_centers[np.unique(position_bins), 0]).size == 1
    assert turn_seq == ""


# =============================================================================
# Public API Export Tests
# =============================================================================


def test_path_progress_array_length_mismatch(environment_with_polygon_regions):
    """Test path_progress raises error for mismatched array lengths."""
    from neurospatial.behavior.navigation import path_progress

    with pytest.raises(ValueError, match=r"times has 3 samples, positions has 2"):
        path_progress(
            environment_with_polygon_regions,
            np.arange(3.0),
            np.array([[0.0, 0.0], [2.0, 2.0]]),
            start="start_zone",
            goal="goal_zone",
        )


def test_cost_to_goal_dynamic_goal_with_cost_map(simple_environment_with_regions):
    """Test cost_to_goal with dynamic goals AND cost_map."""
    from neurospatial.behavior.navigation import cost_to_goal

    env = simple_environment_with_regions

    # Create trajectory visiting multiple bins
    position_bins = np.array([0, 5, 10, 15, 20])

    # Dynamic goals (changes over time)
    goal_bins = np.array([20, 20, 20, 10, 10])  # Goal switches

    # Create cost map (make some bins costly)
    cost_map = np.ones(env.n_bins)
    cost_map[10:15] = 5.0  # High cost zone

    # Compute cost with dynamic goals
    costs = cost_to_goal(
        position_bins,
        env,
        goal_bins=goal_bins,
        cost_map=cost_map,
    )

    # Should return valid costs
    assert costs.shape == (5,)
    # All values should be either non-negative or NaN
    assert np.all((costs >= 0) | np.isnan(costs))

    # First position (at bin 0, goal at bin 20) should have > 0 cost
    assert costs[0] > 0


def test_distance_to_goal_refuses_a_goal_outside(simple_environment_with_regions):
    """A goal point outside the environment raises instead of giving NaN."""
    from neurospatial.behavior.navigation import distance_to_goal

    with pytest.raises(ValueError, match=r"goal= point \[500.0, 500.0\] is outside"):
        distance_to_goal(
            simple_environment_with_regions,
            np.arange(2.0),
            np.array([[0.0, 0.0], [2.0, 2.0]]),
            goal=[500.0, 500.0],
        )
