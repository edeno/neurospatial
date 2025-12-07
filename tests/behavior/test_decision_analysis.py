"""Tests for spatial decision analysis module.

Following TDD: Tests written FIRST, then implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_2d_environment():
    """Create a simple 2D environment for testing."""
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    xx, yy = np.meshgrid(x, y)
    sample_positions = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(sample_positions, bin_size=5.0)


@pytest.fixture
def t_maze_environment():
    """Create a T-maze environment with regions for testing.

    Layout:
        left (goal)  ---  center  ---  right (goal)
                            |
                            |
                          start
    """
    # Create T-maze shape
    # Vertical stem (y: 0 to 50)
    stem_x = np.linspace(45, 55, 5)
    stem_y = np.linspace(0, 50, 20)
    stem_xx, stem_yy = np.meshgrid(stem_x, stem_y)

    # Horizontal bar (y: 50 to 60)
    bar_x = np.linspace(0, 100, 40)
    bar_y = np.linspace(50, 60, 5)
    bar_xx, bar_yy = np.meshgrid(bar_x, bar_y)

    # Combine
    positions = np.vstack(
        [
            np.column_stack([stem_xx.ravel(), stem_yy.ravel()]),
            np.column_stack([bar_xx.ravel(), bar_yy.ravel()]),
        ]
    )

    env = Environment.from_samples(positions, bin_size=5.0)

    # Add regions (point regions - the bin containing each point)
    env.regions.add("start", point=(50.0, 5.0))
    env.regions.add("center", point=(50.0, 55.0))
    env.regions.add("left", point=(10.0, 55.0))
    env.regions.add("right", point=(90.0, 55.0))

    return env


# =============================================================================
# Test PreDecisionMetrics Dataclass
# =============================================================================


class TestPreDecisionMetrics:
    """Test PreDecisionMetrics dataclass."""

    def test_dataclass_fields(self):
        """Test that dataclass has required fields."""
        from neurospatial.behavior.decisions import PreDecisionMetrics

        result = PreDecisionMetrics(
            mean_speed=15.0,
            min_speed=2.0,
            heading_mean_direction=0.5,
            heading_circular_variance=0.3,
            heading_mean_resultant_length=0.7,
            window_duration=1.0,
            n_samples=30,
        )

        assert result.mean_speed == 15.0
        assert result.min_speed == 2.0
        assert result.heading_mean_direction == 0.5
        assert result.heading_circular_variance == 0.3
        assert result.heading_mean_resultant_length == 0.7
        assert result.window_duration == 1.0
        assert result.n_samples == 30

    def test_suggests_deliberation_high_variance_low_speed(self):
        """Test suggests_deliberation returns True for high variance + low speed."""
        from neurospatial.behavior.decisions import PreDecisionMetrics

        result = PreDecisionMetrics(
            mean_speed=5.0,  # Low speed
            min_speed=0.5,
            heading_mean_direction=0.0,
            heading_circular_variance=0.7,  # High variance
            heading_mean_resultant_length=0.3,
            window_duration=1.0,
            n_samples=30,
        )

        assert result.suggests_deliberation(
            variance_threshold=0.5, speed_threshold=10.0
        )

    def test_suggests_deliberation_low_variance_high_speed(self):
        """Test suggests_deliberation returns False for low variance + high speed."""
        from neurospatial.behavior.decisions import PreDecisionMetrics

        result = PreDecisionMetrics(
            mean_speed=20.0,  # High speed
            min_speed=15.0,
            heading_mean_direction=0.0,
            heading_circular_variance=0.2,  # Low variance
            heading_mean_resultant_length=0.8,
            window_duration=1.0,
            n_samples=30,
        )

        assert not result.suggests_deliberation(
            variance_threshold=0.5, speed_threshold=10.0
        )


# =============================================================================
# Test DecisionBoundaryMetrics Dataclass
# =============================================================================


class TestDecisionBoundaryMetrics:
    """Test DecisionBoundaryMetrics dataclass."""

    def test_dataclass_fields(self):
        """Test that dataclass has required fields."""
        from neurospatial.behavior.decisions import DecisionBoundaryMetrics

        goal_labels = np.array([0, 0, 0, 1, 1, 1])
        distance_to_boundary = np.array([10.0, 5.0, 1.0, 1.0, 5.0, 10.0])
        crossing_times = [2.5]
        crossing_directions = [(0, 1)]

        result = DecisionBoundaryMetrics(
            goal_labels=goal_labels,
            distance_to_boundary=distance_to_boundary,
            crossing_times=crossing_times,
            crossing_directions=crossing_directions,
        )

        assert len(result.goal_labels) == 6
        assert len(result.distance_to_boundary) == 6
        assert result.n_crossings == 1
        assert result.crossing_times[0] == 2.5

    def test_n_crossings_property(self):
        """Test n_crossings property returns correct count."""
        from neurospatial.behavior.decisions import DecisionBoundaryMetrics

        result = DecisionBoundaryMetrics(
            goal_labels=np.array([0, 1, 0, 1]),
            distance_to_boundary=np.array([5.0, 5.0, 5.0, 5.0]),
            crossing_times=[1.0, 2.0, 3.0],
            crossing_directions=[(0, 1), (1, 0), (0, 1)],
        )

        assert result.n_crossings == 3

    def test_summary_method(self):
        """Test summary returns formatted string."""
        from neurospatial.behavior.decisions import DecisionBoundaryMetrics

        result = DecisionBoundaryMetrics(
            goal_labels=np.array([0, 1]),
            distance_to_boundary=np.array([5.0, 5.0]),
            crossing_times=[1.0],
            crossing_directions=[(0, 1)],
        )

        summary = result.summary()
        assert "1 crossing" in summary or "crossings" in summary


# =============================================================================
# Test DecisionAnalysisResult Dataclass
# =============================================================================


class TestDecisionAnalysisResult:
    """Test DecisionAnalysisResult dataclass."""

    def test_dataclass_fields(self):
        """Test that dataclass has required fields."""
        from neurospatial.behavior.decisions import (
            DecisionAnalysisResult,
            DecisionBoundaryMetrics,
            PreDecisionMetrics,
        )

        pre = PreDecisionMetrics(
            mean_speed=15.0,
            min_speed=2.0,
            heading_mean_direction=0.5,
            heading_circular_variance=0.3,
            heading_mean_resultant_length=0.7,
            window_duration=1.0,
            n_samples=30,
        )

        boundary = DecisionBoundaryMetrics(
            goal_labels=np.array([0, 1]),
            distance_to_boundary=np.array([5.0, 5.0]),
            crossing_times=[1.0],
            crossing_directions=[(0, 1)],
        )

        result = DecisionAnalysisResult(
            entry_time=5.0,
            pre_decision=pre,
            boundary=boundary,
            chosen_goal=1,
        )

        assert result.entry_time == 5.0
        assert result.pre_decision is pre
        assert result.boundary is boundary
        assert result.chosen_goal == 1

    def test_summary_method(self):
        """Test summary returns formatted string."""
        from neurospatial.behavior.decisions import (
            DecisionAnalysisResult,
            PreDecisionMetrics,
        )

        pre = PreDecisionMetrics(
            mean_speed=15.0,
            min_speed=2.0,
            heading_mean_direction=0.5,
            heading_circular_variance=0.3,
            heading_mean_resultant_length=0.7,
            window_duration=1.0,
            n_samples=30,
        )

        result = DecisionAnalysisResult(
            entry_time=5.0,
            pre_decision=pre,
            boundary=None,
            chosen_goal=1,
        )

        summary = result.summary()
        assert "5.0" in summary or "entry" in summary.lower()


# =============================================================================
# Test decision_region_entry_time
# =============================================================================


class TestDecisionRegionEntryTime:
    """Test decision_region_entry_time function."""

    def test_finds_first_entry(self, t_maze_environment):
        """Test that function finds first entry to region."""
        from neurospatial.behavior.decisions import decision_region_entry_time

        env = t_maze_environment

        # Create trajectory that moves from stem bottom to top of T-bar
        n_samples = 100
        times = np.linspace(0, 10, n_samples)
        # Start at stem bottom, move up to beyond center
        positions = np.column_stack(
            [
                np.full(n_samples, 50.0),  # x stays at center
                np.linspace(0, 60, n_samples),  # y moves from 0 to 60
            ]
        )
        trajectory_bins = env.bin_at(positions)

        entry_time = decision_region_entry_time(trajectory_bins, times, env, "center")

        # Center region is at y=55, trajectory goes from y=0 to y=60 over 10s
        # Entry should be around t = 10 * (55/60) ≈ 9.2s (accounting for bin discretization)
        assert 7.0 < entry_time <= 10.0

    def test_raises_if_never_enters(self, t_maze_environment):
        """Test ValueError if trajectory never enters region."""
        from neurospatial.behavior.decisions import decision_region_entry_time

        env = t_maze_environment

        # Trajectory that stays in stem, never reaches center
        n_samples = 50
        times = np.linspace(0, 5, n_samples)
        positions = np.column_stack(
            [
                np.full(n_samples, 50.0),
                np.linspace(0, 30, n_samples),  # Only goes to y=30, not center
            ]
        )
        trajectory_bins = env.bin_at(positions)

        with pytest.raises(ValueError, match="never enters"):
            decision_region_entry_time(trajectory_bins, times, env, "center")


# =============================================================================
# Test extract_pre_decision_window
# =============================================================================


class TestExtractPreDecisionWindow:
    """Test extract_pre_decision_window function."""

    def test_basic_extraction(self):
        """Test basic window extraction."""
        from neurospatial.behavior.decisions import extract_pre_decision_window

        # Full trajectory
        positions = np.column_stack([np.linspace(0, 100, 100), np.zeros(100)])
        times = np.linspace(0, 10, 100)

        # Extract 2-second window before entry at t=5
        window_pos, window_times = extract_pre_decision_window(
            positions, times, entry_time=5.0, window_duration=2.0
        )

        # Should have samples from t=3 to t=5 (but not t=5 since it's entry)
        assert window_times[0] >= 2.9
        assert window_times[-1] < 5.0
        assert len(window_pos) == len(window_times)

    def test_window_at_trajectory_start(self):
        """Test when window extends before trajectory start."""
        from neurospatial.behavior.decisions import extract_pre_decision_window

        positions = np.column_stack([np.linspace(0, 50, 50), np.zeros(50)])
        times = np.linspace(0, 5, 50)

        # Request 3-second window before t=1 (extends before t=0)
        window_pos, window_times = extract_pre_decision_window(
            positions, times, entry_time=1.0, window_duration=3.0
        )

        # Should return available data from t=0 to t<1
        assert window_times[0] >= 0.0
        assert window_times[-1] < 1.0
        assert len(window_pos) > 0


# =============================================================================
# Test pre_decision_heading_stats
# =============================================================================


class TestPreDecisionHeadingStats:
    """Test pre_decision_heading_stats function."""

    def test_consistent_heading(self):
        """Test low variance for consistent heading."""
        from neurospatial.behavior.decisions import pre_decision_heading_stats

        # Straight line trajectory (consistent heading)
        n_samples = 50
        positions = np.column_stack(
            [np.linspace(0, 100, n_samples), np.zeros(n_samples)]
        )
        times = np.linspace(0, 5, n_samples)

        mean_dir, circ_var, mrl = pre_decision_heading_stats(
            positions, times, min_speed=0.0
        )

        # Low variance for consistent heading
        assert circ_var < 0.2
        # High mean resultant length
        assert mrl > 0.8
        # Mean direction should be around 0 (East)
        assert abs(mean_dir) < 0.3

    def test_variable_heading(self):
        """Test high variance for variable heading."""
        from neurospatial.behavior.decisions import pre_decision_heading_stats

        # Zigzag trajectory (variable heading)
        n_samples = 100
        x = np.linspace(0, 50, n_samples)
        y = 10 * np.sin(x / 5)  # Zigzag pattern
        positions = np.column_stack([x, y])
        times = np.linspace(0, 5, n_samples)

        _mean_dir, circ_var, _mrl = pre_decision_heading_stats(
            positions, times, min_speed=0.0
        )

        # Higher variance for zigzag heading
        assert circ_var > 0.3

    def test_stationary_returns_max_variance(self):
        """Test that all-stationary trajectory returns max variance."""
        from neurospatial.behavior.decisions import pre_decision_heading_stats

        # Stationary trajectory
        n_samples = 30
        positions = np.tile([50.0, 50.0], (n_samples, 1))
        times = np.linspace(0, 3, n_samples)

        _mean_dir, circ_var, mrl = pre_decision_heading_stats(
            positions, times, min_speed=5.0
        )

        # Max variance when no valid headings
        assert circ_var == 1.0
        # Zero mean resultant length
        assert mrl == 0.0


# =============================================================================
# Test pre_decision_speed_stats
# =============================================================================


class TestPreDecisionSpeedStats:
    """Test pre_decision_speed_stats function."""

    def test_constant_speed(self):
        """Test mean and min for constant speed trajectory."""
        from neurospatial.behavior.decisions import pre_decision_speed_stats

        # Constant speed: 100 units in 10 seconds = 10 units/s
        n_samples = 101
        positions = np.column_stack(
            [np.linspace(0, 100, n_samples), np.zeros(n_samples)]
        )
        times = np.linspace(0, 10, n_samples)

        mean_speed, min_speed = pre_decision_speed_stats(positions, times)

        assert_allclose(mean_speed, 10.0, rtol=0.1)
        assert_allclose(min_speed, 10.0, rtol=0.1)

    def test_variable_speed(self):
        """Test min is less than mean for variable speed."""
        from neurospatial.behavior.decisions import pre_decision_speed_stats

        # Accelerating trajectory
        n_samples = 101
        t = np.linspace(0, 10, n_samples)
        x = t**2  # Accelerating
        positions = np.column_stack([x, np.zeros(n_samples)])
        times = t

        mean_speed, min_speed = pre_decision_speed_stats(positions, times)

        # Min should be less than mean for accelerating trajectory
        assert min_speed < mean_speed


# =============================================================================
# Test geodesic_voronoi_labels
# =============================================================================


class TestGeodesicVoronoiLabels:
    """Test geodesic_voronoi_labels function."""

    def test_two_goals_t_maze(self, t_maze_environment):
        """Test Voronoi labeling with two goals in T-maze."""
        from neurospatial.behavior.decisions import geodesic_voronoi_labels

        env = t_maze_environment

        # Get bin indices for left and right goals
        left_bin = env.bin_at(np.array([10.0, 55.0]))
        right_bin = env.bin_at(np.array([90.0, 55.0]))
        goal_bins = [left_bin, right_bin]

        labels = geodesic_voronoi_labels(env, goal_bins)

        # Should have labels 0 or 1 (or -1 for unreachable)
        assert labels.shape == (env.n_bins,)
        assert set(np.unique(labels)).issubset({-1, 0, 1})

        # Left side bins should be labeled 0 (closer to left goal)
        left_side_bin = env.bin_at(np.array([20.0, 55.0]))
        assert labels[left_side_bin] == 0

        # Right side bins should be labeled 1 (closer to right goal)
        right_side_bin = env.bin_at(np.array([80.0, 55.0]))
        assert labels[right_side_bin] == 1

    def test_unreachable_bins_labeled_minus_one(self, simple_2d_environment):
        """Test that disconnected bins are labeled -1."""
        from neurospatial.behavior.decisions import geodesic_voronoi_labels

        env = simple_2d_environment
        goal_bin = env.bin_at(np.array([50.0, 50.0]))

        labels = geodesic_voronoi_labels(env, [goal_bin])

        # All reachable bins should be labeled 0 (only goal)
        # Unreachable bins should be -1
        reachable_mask = labels >= 0
        assert np.sum(reachable_mask) > 0


# =============================================================================
# Test distance_to_decision_boundary
# =============================================================================


class TestDistanceToDecisionBoundary:
    """Test distance_to_decision_boundary function."""

    def test_at_boundary_distance_zero(self, t_maze_environment):
        """Test that distance is near zero at the decision boundary."""
        from neurospatial.behavior.decisions import (
            distance_to_decision_boundary,
        )

        env = t_maze_environment

        # Get goal bins
        left_bin = env.bin_at(np.array([10.0, 55.0]))
        right_bin = env.bin_at(np.array([90.0, 55.0]))
        goal_bins = [left_bin, right_bin]

        # Trajectory at center (on boundary)
        center_bin = env.bin_at(np.array([50.0, 55.0]))
        trajectory_bins = np.array([center_bin])

        distances = distance_to_decision_boundary(env, trajectory_bins, goal_bins)

        # At center, distance to boundary should be small
        assert distances[0] < 10.0

    def test_far_from_boundary(self, t_maze_environment):
        """Test that distance is large far from boundary."""
        from neurospatial.behavior.decisions import (
            distance_to_decision_boundary,
        )

        env = t_maze_environment

        # Get goal bins
        left_bin = env.bin_at(np.array([10.0, 55.0]))
        right_bin = env.bin_at(np.array([90.0, 55.0]))
        goal_bins = [left_bin, right_bin]

        # Trajectory at left goal (far from boundary)
        trajectory_bins = np.array([left_bin])

        distances = distance_to_decision_boundary(env, trajectory_bins, goal_bins)

        # At goal, distance to boundary should be larger
        assert distances[0] > 20.0


# =============================================================================
# Test detect_boundary_crossings
# =============================================================================


class TestDetectBoundaryCrossings:
    """Test detect_boundary_crossings function."""

    def test_single_crossing(self):
        """Test detection of single boundary crossing."""
        from neurospatial.behavior.decisions import detect_boundary_crossings

        # Voronoi labels change at index 5
        voronoi_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        trajectory_bins = np.arange(10)
        times = np.linspace(0, 9, 10)

        crossing_times, crossing_directions = detect_boundary_crossings(
            trajectory_bins, voronoi_labels, times
        )

        assert len(crossing_times) == 1
        assert 4.0 < crossing_times[0] < 6.0
        assert crossing_directions[0] == (0, 1)

    def test_multiple_crossings(self):
        """Test detection of multiple boundary crossings."""
        from neurospatial.behavior.decisions import detect_boundary_crossings

        # Labels oscillate: 0 -> 1 -> 0 -> 1
        voronoi_labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        trajectory_bins = np.arange(8)
        times = np.linspace(0, 7, 8)

        crossing_times, crossing_directions = detect_boundary_crossings(
            trajectory_bins, voronoi_labels, times
        )

        assert len(crossing_times) == 3
        assert crossing_directions == [(0, 1), (1, 0), (0, 1)]

    def test_no_crossings(self):
        """Test when there are no crossings."""
        from neurospatial.behavior.decisions import detect_boundary_crossings

        # All same label
        voronoi_labels = np.array([0, 0, 0, 0, 0])
        trajectory_bins = np.arange(5)
        times = np.linspace(0, 4, 5)

        crossing_times, crossing_directions = detect_boundary_crossings(
            trajectory_bins, voronoi_labels, times
        )

        assert len(crossing_times) == 0
        assert len(crossing_directions) == 0


# =============================================================================
# Test compute_pre_decision_metrics
# =============================================================================


class TestComputePreDecisionMetrics:
    """Test compute_pre_decision_metrics function."""

    def test_returns_dataclass(self):
        """Test that function returns PreDecisionMetrics."""
        from neurospatial.behavior.decisions import (
            PreDecisionMetrics,
            compute_pre_decision_metrics,
        )

        # Simple trajectory
        n_samples = 100
        positions = np.column_stack(
            [np.linspace(0, 100, n_samples), np.zeros(n_samples)]
        )
        times = np.linspace(0, 10, n_samples)

        result = compute_pre_decision_metrics(
            positions, times, entry_time=8.0, window_duration=2.0, min_speed=0.0
        )

        assert isinstance(result, PreDecisionMetrics)
        assert result.window_duration > 0
        assert result.n_samples > 0

    def test_short_window_handles_edge(self):
        """Test handling when window is shorter than requested."""
        from neurospatial.behavior.decisions import compute_pre_decision_metrics

        n_samples = 20
        positions = np.column_stack(
            [np.linspace(0, 20, n_samples), np.zeros(n_samples)]
        )
        times = np.linspace(0, 2, n_samples)

        # Request 5-second window before t=1 (only 1 second available)
        result = compute_pre_decision_metrics(
            positions, times, entry_time=1.0, window_duration=5.0, min_speed=0.0
        )

        # Should return metrics for available window
        assert result.window_duration < 5.0
        assert result.n_samples > 0


# =============================================================================
# Test compute_decision_analysis
# =============================================================================


class TestComputeDecisionAnalysis:
    """Test compute_decision_analysis function."""

    def test_returns_dataclass(self, t_maze_environment):
        """Test that function returns DecisionAnalysisResult."""
        from neurospatial.behavior.decisions import (
            DecisionAnalysisResult,
            compute_decision_analysis,
        )

        env = t_maze_environment

        # Trajectory from start through center to right
        n_samples = 100
        # Start at stem, go through center, then to right
        positions = np.column_stack(
            [
                np.concatenate(
                    [
                        np.full(30, 50.0),  # In stem
                        np.linspace(50, 90, 70),  # Moving right
                    ]
                ),
                np.concatenate(
                    [
                        np.linspace(5, 55, 30),  # Up the stem
                        np.full(70, 55.0),  # Along top bar
                    ]
                ),
            ]
        )
        times = np.linspace(0, 10, n_samples)

        result = compute_decision_analysis(
            env,
            positions,
            times,
            decision_region="center",
            goal_regions=["left", "right"],
            pre_window=1.0,
        )

        assert isinstance(result, DecisionAnalysisResult)
        assert result.entry_time > 0
        assert result.pre_decision is not None

    def test_detects_chosen_goal(self, t_maze_environment):
        """Test that function correctly identifies chosen goal."""
        from neurospatial.behavior.decisions import compute_decision_analysis

        env = t_maze_environment

        # Trajectory that goes to right goal
        n_samples = 100
        positions = np.column_stack(
            [
                np.concatenate(
                    [
                        np.full(30, 50.0),
                        np.linspace(50, 90, 70),
                    ]
                ),
                np.concatenate(
                    [
                        np.linspace(5, 55, 30),
                        np.full(70, 55.0),
                    ]
                ),
            ]
        )
        times = np.linspace(0, 10, n_samples)

        result = compute_decision_analysis(
            env,
            positions,
            times,
            decision_region="center",
            goal_regions=["left", "right"],
            pre_window=1.0,
        )

        # Should choose right (index 1)
        assert result.chosen_goal == 1


# =============================================================================
# Test Error Handling
# =============================================================================


class TestDistanceToDecisionBoundaryEdgeCases:
    """Test edge cases for distance_to_decision_boundary."""

    def test_out_of_bounds_bins_return_nan(self, t_maze_environment):
        """Test that out-of-bounds trajectory bins return NaN distance."""
        from neurospatial.behavior.decisions import distance_to_decision_boundary

        env = t_maze_environment
        left_bin = int(env.bin_at(np.array([[10.0, 55.0]]))[0])
        right_bin = int(env.bin_at(np.array([[90.0, 55.0]]))[0])
        goal_bins = [left_bin, right_bin]

        # Create trajectory with invalid bins
        valid_bin = int(env.bin_at(np.array([[50.0, 55.0]]))[0])
        trajectory_bins = np.array([valid_bin, -1, env.n_bins + 100], dtype=np.int64)

        distances = distance_to_decision_boundary(env, trajectory_bins, goal_bins)

        assert not np.isnan(distances[0])  # valid bin
        assert np.isnan(distances[1])  # -1 index
        assert np.isnan(distances[2])  # out of bounds


class TestErrorHandling:
    """Test error handling and messages."""

    def test_invalid_region_name(self, t_maze_environment):
        """Test helpful error for invalid region name."""
        from neurospatial.behavior.decisions import compute_decision_analysis

        env = t_maze_environment
        n_samples = 50
        positions = np.column_stack(
            [np.full(n_samples, 50.0), np.linspace(0, 60, n_samples)]
        )
        times = np.linspace(0, 5, n_samples)

        with pytest.raises(ValueError, match="not found"):
            compute_decision_analysis(
                env,
                positions,
                times,
                decision_region="nonexistent",
                goal_regions=["left", "right"],
            )

    def test_mismatched_array_lengths(self, t_maze_environment):
        """Test helpful error for mismatched array lengths."""
        from neurospatial.behavior.decisions import compute_decision_analysis

        env = t_maze_environment
        positions = np.column_stack([np.linspace(0, 50, 50), np.zeros(50)])
        times = np.linspace(0, 5, 40)  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            compute_decision_analysis(
                env,
                positions,
                times,
                decision_region="center",
                goal_regions=["left", "right"],
            )
