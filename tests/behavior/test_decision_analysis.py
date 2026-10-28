"""Tests for spatial decision analysis module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment, Regions

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
    region_builder = Regions(env.regions.values())
    region_builder.add("start", point=(50.0, 5.0))
    region_builder.add("center", point=(50.0, 55.0))
    region_builder.add("left", point=(10.0, 55.0))
    region_builder.add("right", point=(90.0, 55.0))
    env = env.with_regions(region_builder)

    return env


# =============================================================================
# Test PreDecisionMetrics Dataclass
# =============================================================================


class TestPreDecisionMetrics:
    """Test PreDecisionMetrics dataclass."""

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

        assert result.suggests_deliberation(variance_threshold=0.5, min_speed=10.0)

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

        assert not result.suggests_deliberation(variance_threshold=0.5, min_speed=10.0)


# =============================================================================
# Test DecisionBoundaryMetrics Dataclass
# =============================================================================


class TestDecisionBoundaryMetrics:
    """Test DecisionBoundaryMetrics dataclass."""

    def test_n_crossings_property(self):
        """Test n_crossings property returns correct count."""
        from neurospatial.behavior.decisions import DecisionBoundaryMetrics

        result = DecisionBoundaryMetrics(
            goal_labels=np.array([0, 1, 0, 1]),
            distance_to_decision_boundary=np.array([5.0, 5.0, 5.0, 5.0]),
            crossing_times=[1.0, 2.0, 3.0],
            crossing_directions=[(0, 1), (1, 0), (0, 1)],
        )

        assert result.n_crossings == 3


class TestDecisionAnalysisResult:
    """Test DecisionAnalysisResult dataclass."""


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
        position_bins = env.bin_at(positions)

        entry_time = decision_region_entry_time(
            position_bins, times, env, region="center"
        )

        # The center bin spans y in [52.5, 57.5); y = 60 k / 99 first reaches
        # it at sample k = 87.
        assert entry_time == times[87]

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
        position_bins = env.bin_at(positions)

        with pytest.raises(ValueError, match="never enters"):
            decision_region_entry_time(position_bins, times, env, region="center")

    def test_validates_bins_and_times(self, t_maze_environment):
        from neurospatial.behavior.decisions import decision_region_entry_time

        env = t_maze_environment
        bins = env.bin_at(np.column_stack([np.full(5, 50.0), np.linspace(40, 56, 5)]))
        times = np.arange(5) / 10
        with pytest.raises(ValueError, match="one position bin per timestamp"):
            decision_region_entry_time(bins, times[:4], env, region="center")
        with pytest.raises(ValueError, match="non-decreasing"):
            decision_region_entry_time(bins, times[::-1], env, region="center")
        with pytest.raises(ValueError, match="strictly increasing"):
            decision_region_entry_time(
                bins, np.r_[0.0, times[:4]], env, region="center"
            )
        with pytest.raises(ValueError, match="finite"):
            decision_region_entry_time(
                bins, np.r_[times[:4], np.nan], env, region="center"
            )
        with pytest.raises(TypeError, match="integer position_bins"):
            decision_region_entry_time(bins.astype(float), times, env, region="center")


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
        window_times, window_pos = extract_pre_decision_window(
            times, positions, entry_time=5.0, window_duration=2.0
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
        window_times, window_pos = extract_pre_decision_window(
            times, positions, entry_time=1.0, window_duration=3.0
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
            times, positions, min_speed=0.0
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
            times, positions, min_speed=0.0
        )

        # Higher variance for zigzag heading
        assert circ_var > 0.3

    def test_stationary_returns_missing_heading_statistics(self):
        """Stationary tracking supplies no heading evidence."""
        from neurospatial.behavior.decisions import pre_decision_heading_stats

        # Stationary trajectory
        n_samples = 30
        positions = np.tile([50.0, 50.0], (n_samples, 1))
        times = np.linspace(0, 3, n_samples)

        _mean_dir, circ_var, mrl = pre_decision_heading_stats(
            times, positions, min_speed=5.0
        )

        assert np.isnan(circ_var)
        assert np.isnan(mrl)


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

        mean_speed, min_speed = pre_decision_speed_stats(times, positions)

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

        mean_speed, min_speed = pre_decision_speed_stats(times, positions)

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
        left_bin = env.bin_at(np.array([[10.0, 55.0]]))[0]
        right_bin = env.bin_at(np.array([[90.0, 55.0]]))[0]
        goal_bins = [left_bin, right_bin]

        labels = geodesic_voronoi_labels(env, goal_bins)

        # Should have labels 0 or 1 (or -1 for unreachable)
        assert labels.shape == (env.n_bins,)
        assert set(np.unique(labels)).issubset({-1, 0, 1})

        # Left side bins should be labeled 0 (closer to left goal)
        left_side_bin = env.bin_at(np.array([[20.0, 55.0]]))[0]
        assert labels[left_side_bin] == 0

        # Right side bins should be labeled 1 (closer to right goal)
        right_side_bin = env.bin_at(np.array([[80.0, 55.0]]))[0]
        assert labels[right_side_bin] == 1

    def test_equidistant_bins_are_undecided(self):
        """Goals at 0 and 100 cm: the bin at 50 cm belongs to neither goal."""
        from neurospatial.behavior.decisions import geodesic_voronoi_labels

        env = _linear_track()
        labels = geodesic_voronoi_labels(env, _bins_at(env, [0, 100]))
        assert labels.tolist() == [0, 0, 0, 0, 0, -1, 1, 1, 1, 1, 1]
        swapped = geodesic_voronoi_labels(env, _bins_at(env, [100, 0]))
        assert swapped.tolist() == [1, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0]

    def test_t_maze_stem_is_undecided(self, t_maze_environment):
        """The stem's middle column is equidistant from the two arm goals."""
        from neurospatial.behavior.decisions import geodesic_voronoi_labels

        env = t_maze_environment
        goals = env.bin_at(np.array([[10.0, 55.0], [90.0, 55.0]]))
        labels = geodesic_voronoi_labels(env, goals)
        centers = env.bin_centers
        middle = np.isclose(centers[:, 0], 50.0)
        assert np.all(labels[middle] == -1)
        assert np.all(labels[~middle & (centers[:, 0] < 50)] == 0)
        assert np.all(labels[~middle & (centers[:, 0] > 50)] == 1)

    def test_unreachable_second_goal_is_not_a_tie(self):
        """A bin whose second goal is unreachable is labeled by the reachable one."""
        from neurospatial.behavior.decisions import geodesic_voronoi_labels

        samples = np.r_[np.arange(0, 31.0), np.arange(70, 101.0)][:, None]
        env = Environment.from_samples(samples, bin_size=10.0)
        labels = geodesic_voronoi_labels(env, _bins_at(env, [0, 100]))
        assert labels.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]

    def test_unreachable_bins_labeled_minus_one(self, simple_2d_environment):
        """Test that disconnected bins are labeled -1."""
        from neurospatial.behavior.decisions import geodesic_voronoi_labels

        env = simple_2d_environment
        goal_bin = env.bin_at(np.array([[50.0, 50.0]]))[0]

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
        left_bin = env.bin_at(np.array([[10.0, 55.0]]))[0]
        right_bin = env.bin_at(np.array([[90.0, 55.0]]))[0]
        goal_bins = [left_bin, right_bin]

        # Trajectory at center (on boundary)
        center_bin = env.bin_at(np.array([[50.0, 55.0]]))[0]
        position_bins = np.array([center_bin])

        distances = distance_to_decision_boundary(env, position_bins, goal_bins)

        # The center bin is equidistant from both goals: on the boundary.
        assert distances.tolist() == [0.0]

    def test_far_from_boundary(self, t_maze_environment):
        """Test that distance is large far from boundary."""
        from neurospatial.behavior.decisions import (
            distance_to_decision_boundary,
        )

        env = t_maze_environment

        # Get goal bins
        left_bin = env.bin_at(np.array([[10.0, 55.0]]))[0]
        right_bin = env.bin_at(np.array([[90.0, 55.0]]))[0]
        goal_bins = [left_bin, right_bin]

        # Trajectory at left goal (far from boundary)
        position_bins = np.array([left_bin])

        distances = distance_to_decision_boundary(env, position_bins, goal_bins)

        # The goals are 80 cm apart along the crossbar: the boundary is
        # halfway, 40 cm from the left goal.
        assert_allclose(distances, [40.0], rtol=1e-12)


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
        position_bins = np.arange(10)
        times = np.linspace(0, 9, 10)

        crossing_times, crossing_directions = detect_boundary_crossings(
            position_bins, voronoi_labels, times, max_gap=None
        )

        assert len(crossing_times) == 1
        assert 4.0 < crossing_times[0] < 6.0
        assert crossing_directions[0] == (0, 1)

    def test_multiple_crossings(self):
        """Test detection of multiple boundary crossings."""
        from neurospatial.behavior.decisions import detect_boundary_crossings

        # Labels oscillate: 0 -> 1 -> 0 -> 1
        voronoi_labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        position_bins = np.arange(8)
        times = np.linspace(0, 7, 8)

        crossing_times, crossing_directions = detect_boundary_crossings(
            position_bins, voronoi_labels, times, max_gap=None
        )

        assert len(crossing_times) == 3
        assert crossing_directions == [(0, 1), (1, 0), (0, 1)]

    def test_no_crossings(self):
        """Test when there are no crossings."""
        from neurospatial.behavior.decisions import detect_boundary_crossings

        # All same label
        voronoi_labels = np.array([0, 0, 0, 0, 0])
        position_bins = np.arange(5)
        times = np.linspace(0, 4, 5)

        crossing_times, crossing_directions = detect_boundary_crossings(
            position_bins,
            voronoi_labels,
            times,
            max_gap=None,
        )

        assert len(crossing_times) == 0
        assert len(crossing_directions) == 0


# =============================================================================
# Test compute_pre_decision_metrics
# =============================================================================


class TestComputePreDecisionMetrics:
    """Test compute_pre_decision_metrics function."""

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
            times, positions, entry_time=1.0, window_duration=5.0, min_speed=0.0
        )

        # Should return metrics for available window
        assert result.window_duration < 5.0
        assert result.n_samples > 0


# =============================================================================
# Test compute_decision_analysis
# =============================================================================


def _stem_right_then_back_left():
    """Up the T-maze stem, east to the right goal, then west past the center.

    10 Hz. The stem (0-2.9 s) enters the center bin at 2.8 s; the east leg
    (3.0-5.9 s) reaches the right goal; the return leg (6.0-10.0 s) walks
    west 2 cm per sample from x = 90 cm and re-enters the center bin
    (x < 52.5 cm) at x = 52 cm, 7.9 s.
    """
    times = np.arange(101) / 10
    x = np.r_[np.full(30, 50.0), np.linspace(50, 90, 30), 90 - 2.0 * np.arange(41)]
    y = np.r_[np.linspace(5, 55, 30), np.full(71, 55.0)]
    return times, np.column_stack([x, y])


class TestComputeDecisionAnalysis:
    """Test compute_decision_analysis function."""

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
            times,
            positions,
            decision_region="center",
            goal_regions=["left", "right"],
            pre_window=1.0,
            min_speed=5.0,
        )

        # Should choose right (index 1)
        assert result.chosen_goal == 1

    @staticmethod
    def _chosen_goal_region(env, x, y, goal_regions):
        from neurospatial.behavior.decisions import compute_decision_analysis

        result = compute_decision_analysis(
            env,
            np.arange(len(x)) / 10,
            np.column_stack([x, y]),
            decision_region="center",
            goal_regions=goal_regions,
            min_speed=5.0,
        )
        return None if result.chosen_goal is None else goal_regions[result.chosen_goal]

    def test_goal_the_trial_started_from_is_not_chosen(self, t_maze_environment):
        """Starting at the left goal and running to the right chooses right."""
        x = np.r_[np.linspace(10, 50, 40), np.linspace(50, 90, 40)]
        y = np.full(80, 55.0)
        for goals in (["left", "right"], ["right", "left"]):
            assert self._chosen_goal_region(t_maze_environment, x, y, goals) == (
                "right"
            )

    def test_first_goal_entered_after_entry_is_chosen(self, t_maze_environment):
        """Up the stem, right goal, then back to the left goal: right was chosen."""
        x = np.r_[np.full(30, 50.0), np.linspace(50, 90, 30), np.linspace(90, 10, 60)]
        y = np.r_[np.linspace(5, 55, 30), np.full(90, 55.0)]
        for goals in (["left", "right"], ["right", "left"]):
            assert self._chosen_goal_region(t_maze_environment, x, y, goals) == (
                "right"
            )

    def test_entry_is_searched_inside_epochs(self, t_maze_environment):
        """Epochs (4, 10) s skip the 2.8 s entry and use the return at 7.9 s."""
        from neurospatial.behavior.decisions import compute_decision_analysis

        times, positions = _stem_right_then_back_left()
        kwargs = {
            "decision_region": "center",
            "goal_regions": ["left", "right"],
            "min_speed": 5.0,
        }
        whole = compute_decision_analysis(
            t_maze_environment, times, positions, **kwargs
        )
        assert_allclose(whole.entry_time, 2.8, rtol=1e-12)
        assert whole.chosen_goal == 1
        late = compute_decision_analysis(
            t_maze_environment, times, positions, epochs=(4.0, 10.0), **kwargs
        )
        assert_allclose(late.entry_time, 7.9, rtol=1e-12)
        assert late.chosen_goal == 0
        assert late.pre_decision.n_samples == 10

    def test_no_entry_inside_epochs_raises(self, t_maze_environment):
        from neurospatial.behavior.decisions import compute_decision_analysis

        times, positions = _stem_right_then_back_left()
        with pytest.raises(ValueError, match="never enters region 'center' inside"):
            compute_decision_analysis(
                t_maze_environment,
                times,
                positions,
                decision_region="center",
                goal_regions=["left", "right"],
                min_speed=5.0,
                epochs=(3.5, 6.0),
            )

    def test_single_goal_has_no_boundary(self, t_maze_environment):
        """One goal defines no decision boundary: boundary is None."""
        from neurospatial.behavior.decisions import compute_decision_analysis

        x = np.r_[np.full(30, 50.0), np.linspace(50, 90, 30)]
        y = np.r_[np.linspace(5, 55, 30), np.full(30, 55.0)]
        for goals in (["right"], "right"):
            result = compute_decision_analysis(
                t_maze_environment,
                np.arange(60) / 10,
                np.column_stack([x, y]),
                decision_region="center",
                goal_regions=goals,
                min_speed=5.0,
            )
            assert result.boundary is None
            assert result.chosen_goal == 0
            assert result.summary()["n_boundary_crossings"] is None

    def test_empty_goal_list_raises(self, t_maze_environment):
        from neurospatial.behavior.decisions import compute_decision_analysis

        with pytest.raises(ValueError, match="empty goal_regions"):
            compute_decision_analysis(
                t_maze_environment,
                np.arange(30) / 10,
                np.column_stack([np.full(30, 50.0), np.linspace(5, 55, 30)]),
                decision_region="center",
                goal_regions=[],
                min_speed=5.0,
            )

    def test_no_goal_entered_after_entry_is_none(self, t_maze_environment):
        """Start at the left goal, stop at the center: no goal was chosen."""
        x = np.linspace(10, 50, 40)
        y = np.full(40, 55.0)
        assert (
            self._chosen_goal_region(t_maze_environment, x, y, ["left", "right"])
            is None
        )


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
        position_bins = np.array([valid_bin, -1, env.n_bins + 100], dtype=np.int64)

        distances = distance_to_decision_boundary(env, position_bins, goal_bins)

        assert not np.isnan(distances[0])  # valid bin
        assert np.isnan(distances[1])  # -1 index
        assert np.isnan(distances[2])  # out of bounds


def _linear_track():
    """A 100 cm track with bin centers every 10 cm from 0 to 100."""
    return Environment.from_samples(np.arange(101.0)[:, None], bin_size=10.0)


def _bins_at(env, xs):
    return env.bin_at(np.asarray(xs, dtype=float)[:, None])


class TestDistanceToDecisionBoundaryKnownAnswers:
    """Distances to the equidistant point between goals, from track geometry."""

    def test_two_goals_on_a_linear_track(self):
        """Goals at 0 and 100 cm: the boundary is at 50 cm."""
        from neurospatial.behavior.decisions import distance_to_decision_boundary

        env = _linear_track()
        result = distance_to_decision_boundary(
            env, _bins_at(env, [0, 30, 50, 70, 90]), _bins_at(env, [0, 100])
        )
        assert_allclose(result, [50.0, 20.0, 0.0, 20.0, 40.0], rtol=1e-12)

    def test_three_goals_use_the_two_nearest(self):
        """Goals at 0, 50 and 100 cm: boundaries at 25 and 75 cm."""
        from neurospatial.behavior.decisions import distance_to_decision_boundary

        env = _linear_track()
        result = distance_to_decision_boundary(
            env, _bins_at(env, [10, 30, 50, 70]), _bins_at(env, [0, 50, 100])
        )
        assert_allclose(result, [15.0, 5.0, 25.0, 5.0], rtol=1e-12)

    def test_unreachable_second_goal_is_infinite(self):
        """A goal on a disconnected piece of track has no boundary with ours."""
        from neurospatial.behavior.decisions import distance_to_decision_boundary

        samples = np.r_[np.arange(0, 31.0), np.arange(70, 101.0)][:, None]
        env = Environment.from_samples(samples, bin_size=10.0)
        result = distance_to_decision_boundary(
            env, _bins_at(env, [10, 90]), _bins_at(env, [0, 100])
        )
        assert result.tolist() == [np.inf, np.inf]

    def test_single_goal_has_no_boundary(self):
        from neurospatial.behavior.decisions import distance_to_decision_boundary

        env = _linear_track()
        result = distance_to_decision_boundary(
            env, _bins_at(env, [10, 90]), _bins_at(env, [0])
        )
        assert result.tolist() == [np.inf, np.inf]


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
                times,
                positions,
                decision_region="nonexistent",
                goal_regions=["left", "right"],
                min_speed=5.0,
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
                times,
                positions,
                decision_region="center",
                goal_regions=["left", "right"],
                min_speed=5.0,
            )


def test_pre_decision_heading_stats_excludes_stationary(east_stop_north):
    from neurospatial.behavior import pre_decision_heading_stats

    positions, times = east_stop_north
    np.testing.assert_allclose(
        pre_decision_heading_stats(times, positions, min_speed=5.0),
        # 11 eastward samples (the stop's first sample still moves) and 10
        # northward ones
        [0.7378, 0.2921, 0.7079],
        atol=1e-4,
    )


def test_pre_decision_heading_stats_all_stationary():
    from neurospatial.behavior import pre_decision_heading_stats

    assert np.isnan(
        pre_decision_heading_stats(np.arange(10) / 10, np.ones((10, 2)), min_speed=5.0)
    ).all()


def test_goal_regions_are_represented_by_their_centroid_bin():
    """compute_decision_analysis and trials_to_region_arrays share one goal bin.

    The representative is the region bin nearest the centroid of the region's
    bins, not the lowest-index bin, so the decision boundary does not depend on
    how bins happen to be numbered.
    """
    from shapely.geometry import box

    from neurospatial.behavior import (
        compute_decision_analysis,
        distance_to_decision_boundary,
        trials_to_region_arrays,
    )
    from neurospatial.behavior.segmentation import Trial

    x, y = np.meshgrid(np.arange(0.0, 101.0, 2.0), np.arange(0.0, 11.0, 2.0))
    env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
    regions = Regions()
    regions.add("west", polygon=box(0, 0, 10, 10))
    regions.add("east", polygon=box(90, 0, 100, 10))
    regions.add("middle", polygon=box(46, 0, 54, 10))
    env = env.with_regions(regions)
    times = np.arange(200) / 10.0
    positions = np.c_[np.linspace(5.0, 95.0, times.size), np.full(times.size, 5.0)]

    result = compute_decision_analysis(
        env,
        times,
        positions,
        decision_region="middle",
        goal_regions=["west", "east"],
        min_speed=5.0,
    )

    trial = Trial(times[0], times[-1], "west", "east", True)
    start_bins, goal_bins = trials_to_region_arrays([trial], times, env)
    goals = [int(start_bins[0]), int(goal_bins[0])]
    for name, goal in zip(["west", "east"], goals, strict=True):
        assert goal != env.bins_in_region(name)[0]  # not the lowest-index bin
    np.testing.assert_array_equal(
        result.boundary.distance_to_decision_boundary,
        distance_to_decision_boundary(env, env.bin_at(positions), goals),
    )


# =============================================================================
# Missing samples, declared breaks and strict clocks
# =============================================================================


@pytest.fixture
def eastward_path():
    """3 s of a 10 cm/s eastward run sampled at 10 Hz."""
    times = np.arange(30) / 10
    positions = np.column_stack([10 * times, np.zeros(30)])
    return times, positions


def test_pre_decision_window_returns_times_first(eastward_path):
    from neurospatial.behavior import extract_pre_decision_window

    times, positions = eastward_path
    window_times, window_positions = extract_pre_decision_window(
        times, positions, entry_time=2.0, window_duration=1.0
    )
    np.testing.assert_array_equal(window_times, times[10:20])
    np.testing.assert_array_equal(window_positions, positions[10:20])


def test_pre_decision_window_starts_at_declared_break(eastward_path):
    from neurospatial.behavior import extract_pre_decision_window

    times, positions = eastward_path
    break_before = np.zeros(30, dtype=bool)
    break_before[18] = True
    window_times, window_positions = extract_pre_decision_window(
        times, positions, 2.0, window_duration=1.0, break_before=break_before
    )
    np.testing.assert_array_equal(window_times, times[18:20])
    np.testing.assert_array_equal(window_positions, positions[18:20])


def test_pre_decision_window_keeps_missing_row(eastward_path):
    """A lost frame stays in the window as a missing row instead of cutting it."""
    from neurospatial.behavior import extract_pre_decision_window

    times, positions = eastward_path
    positions = positions.copy()
    positions[15] = np.nan
    window_times, window_positions = extract_pre_decision_window(
        times, positions, 2.0, window_duration=1.0
    )
    np.testing.assert_array_equal(window_times, times[10:20])
    assert np.isnan(window_positions[5]).all()
    np.testing.assert_array_equal(
        np.delete(window_positions, 5, axis=0), np.delete(positions[10:20], 5, axis=0)
    )


def test_pre_decision_heading_stats_skip_missing_row(eastward_path):
    from neurospatial.behavior import pre_decision_heading_stats

    times, positions = eastward_path
    positions = positions.copy()
    positions[15] = np.nan
    np.testing.assert_allclose(
        pre_decision_heading_stats(times, positions, min_speed=1.0),
        (0.0, 0.0, 1.0),
        atol=1e-12,
    )


def test_pre_decision_speed_stats_do_not_span_declared_break():
    """Two 10 cm/s recordings 50 cm apart on one clock: the jump is not speed."""
    from neurospatial.behavior import pre_decision_speed_stats

    times = np.arange(40) / 10
    x = np.r_[10 * times[:20], 10 * times[20:] + 50.0]
    positions = np.column_stack([x, np.zeros(40)])
    break_before = np.zeros(40, dtype=bool)
    break_before[20] = True
    mean_speed, min_speed = pre_decision_speed_stats(
        times, positions, break_before=break_before
    )
    np.testing.assert_allclose([mean_speed, min_speed], [10.0, 10.0], atol=1e-9)
    undeclared_mean, _ = pre_decision_speed_stats(times, positions)
    assert undeclared_mean > 10.0


def test_pre_decision_metrics_count_observed_samples():
    from neurospatial.behavior import compute_pre_decision_metrics

    times = np.arange(50) / 10
    positions = np.column_stack([10 * times, np.zeros(50)])
    positions[[22, 27, 33]] = np.nan
    metrics = compute_pre_decision_metrics(
        times, positions, entry_time=4.0, window_duration=2.0, min_speed=1.0
    )
    # The window holds samples 20..39 (20 rows), 3 of them missing.
    assert metrics.n_samples == 17
    assert metrics.window_duration == pytest.approx(times[39] - times[20])
    np.testing.assert_allclose(metrics.mean_speed, 10.0, atol=1e-9)
    np.testing.assert_allclose(metrics.heading_mean_resultant_length, 1.0, atol=1e-12)


def test_pre_decision_metrics_window_duration_spans_observed_samples():
    from neurospatial.behavior import compute_pre_decision_metrics

    times = np.arange(50) / 10
    positions = np.column_stack([10 * times, np.zeros(50)])
    positions[[20, 39]] = np.nan  # first and last window rows lost
    metrics = compute_pre_decision_metrics(
        times, positions, entry_time=4.0, window_duration=2.0, min_speed=1.0
    )
    assert metrics.n_samples == 18
    assert metrics.window_duration == pytest.approx(times[38] - times[21])


def test_pre_decision_metrics_follow_declared_break():
    from neurospatial.behavior import compute_pre_decision_metrics

    times = np.arange(50) / 10
    positions = np.column_stack([10 * times, np.zeros(50)])
    break_before = np.zeros(50, dtype=bool)
    break_before[30] = True
    with_break = compute_pre_decision_metrics(
        times,
        positions,
        entry_time=4.0,
        window_duration=2.0,
        min_speed=1.0,
        break_before=break_before,
    )
    alone = compute_pre_decision_metrics(
        times[30:], positions[30:], entry_time=4.0, window_duration=2.0, min_speed=1.0
    )
    assert with_break == alone
    assert with_break.n_samples == 10


def test_boundary_crossing_at_interval_midpoint():
    from neurospatial.behavior import detect_boundary_crossings

    labels = np.array([0, 1])
    crossing_times, directions = detect_boundary_crossings(
        np.array([0, 0, 1, 1]), labels, np.arange(4) / 8
    )
    assert crossing_times == [0.1875]
    assert directions == [(0, 1)]


def test_boundary_crossing_not_across_declared_break():
    from neurospatial.behavior import detect_boundary_crossings

    crossing_times, directions = detect_boundary_crossings(
        np.array([0, 0, 1, 1]),
        np.array([0, 1]),
        np.arange(4) / 8,
        break_before=np.array([False, False, True, False]),
    )
    assert crossing_times == []
    assert directions == []


def test_decision_analysis_forwards_declared_break(t_maze_environment):
    from neurospatial.behavior import compute_decision_analysis

    env = t_maze_environment
    # Along the crossbar from the left arm, through the center, to the right.
    times = np.arange(100) / 10
    positions = np.column_stack([np.linspace(20, 90, 100), np.full(100, 55.0)])
    kwargs = {
        "decision_region": "center",
        "goal_regions": ["left", "right"],
        "pre_window": 1.0,
        "min_speed": 5.0,
    }
    result = compute_decision_analysis(env, times, positions, **kwargs)
    entry = int(np.flatnonzero(times == result.entry_time)[0])
    labels = result.boundary.goal_labels
    # The last sample nearer the left goal and the first nearer the right
    # goal, with the equidistant center bins (label -1) between them.
    last_left = int(np.flatnonzero(labels == 0)[-1])
    first_right = int(np.flatnonzero(labels == 1)[0])
    assert first_right > last_left + 1
    assert np.all(labels[last_left + 1 : first_right] == -1)
    break_before = np.zeros(100, dtype=bool)
    break_before[[entry - 3, first_right]] = True
    broken = compute_decision_analysis(
        env, times, positions, break_before=break_before, **kwargs
    )
    assert result.pre_decision.n_samples == 10
    assert broken.pre_decision.n_samples == 3
    assert result.boundary.crossing_times == [
        (times[last_left] + times[first_right]) / 2
    ]
    assert result.boundary.crossing_directions == [(0, 1)]
    assert broken.boundary.crossing_times == []
    assert broken.boundary.crossing_directions == []


def test_left_and_right_turns_cross_the_boundary_equally(t_maze_environment):
    """On a mirror-symmetric T-maze the stem is equidistant from both goals.

    Walking up the stem and turning into either arm leaves the boundary
    toward one goal without crossing from the other goal's side, so both
    turns have no crossing.
    """
    from neurospatial.behavior import compute_decision_analysis

    env = t_maze_environment
    times = np.arange(100) / 10
    results = {}
    for arm_end in (10.0, 90.0):
        positions = np.column_stack(
            [
                np.r_[np.full(30, 50.0), np.linspace(50, arm_end, 70)],
                np.r_[np.linspace(5, 55, 30), np.full(70, 55.0)],
            ]
        )
        results[arm_end] = compute_decision_analysis(
            env,
            times,
            positions,
            decision_region="center",
            goal_regions=["left", "right"],
            min_speed=5.0,
        )
    left, right = results[10.0], results[90.0]
    assert left.boundary.n_crossings == right.boundary.n_crossings == 0
    assert np.all(left.boundary.goal_labels[:30] == -1)
    assert np.all(right.boundary.goal_labels[:30] == -1)
    assert (left.chosen_goal, right.chosen_goal) == (0, 1)


def _repeated_clock_calls():
    from neurospatial.behavior import (
        compute_decision_analysis,
        compute_pre_decision_metrics,
        detect_boundary_crossings,
        extract_pre_decision_window,
        pre_decision_heading_stats,
        pre_decision_speed_stats,
    )

    return {
        "extract_pre_decision_window": lambda env, t, p: extract_pre_decision_window(
            t, p, 0.5, window_duration=1.0
        ),
        "pre_decision_speed_stats": lambda env, t, p: pre_decision_speed_stats(t, p),
        "pre_decision_heading_stats": lambda env, t, p: pre_decision_heading_stats(
            t, p, min_speed=1.0
        ),
        "compute_pre_decision_metrics": lambda env, t, p: compute_pre_decision_metrics(
            t, p, 0.5, window_duration=1.0, min_speed=1.0
        ),
        "detect_boundary_crossings": lambda env, t, p: detect_boundary_crossings(
            np.zeros(len(t), dtype=np.int64), np.array([0]), t
        ),
        "compute_decision_analysis": lambda env, t, p: compute_decision_analysis(
            env,
            t,
            p,
            decision_region="center",
            goal_regions=["left", "right"],
            min_speed=5.0,
        ),
    }


@pytest.mark.parametrize(
    "name",
    [
        "extract_pre_decision_window",
        "pre_decision_speed_stats",
        "pre_decision_heading_stats",
        "compute_pre_decision_metrics",
        "detect_boundary_crossings",
        "compute_decision_analysis",
    ],
)
def test_repeated_timestamp_raises(t_maze_environment, name):
    times = np.array([0.0, 0.1, 0.1, 0.2, 0.3, 0.4])
    positions = np.column_stack([np.full(6, 50.0), np.linspace(40, 58, 6)])
    with pytest.raises(ValueError, match="strictly increasing"):
        _repeated_clock_calls()[name](t_maze_environment, times, positions)


@pytest.mark.parametrize(
    "name",
    [
        "extract_pre_decision_window",
        "pre_decision_speed_stats",
        "pre_decision_heading_stats",
        "compute_pre_decision_metrics",
        "detect_boundary_crossings",
        "compute_decision_analysis",
    ],
)
def test_empty_input_raises(t_maze_environment, name):
    with pytest.raises(ValueError, match="no samples"):
        _repeated_clock_calls()[name](
            t_maze_environment, np.array([]), np.empty((0, 2))
        )


@pytest.mark.parametrize("n_samples", [1, 30])
def test_heading_stats_one_dimensional_positions_raise(n_samples):
    """A heading needs x and y, so 1-D positions raise, naming the passed shape."""
    from neurospatial.behavior import pre_decision_heading_stats

    times = np.arange(n_samples) / 10
    match = rf"got shape \({n_samples},\)(.|\n)*heading .*needs both x and y"
    with pytest.raises(ValueError, match=match):
        pre_decision_heading_stats(times, 10 * times, min_speed=1.0)


@pytest.mark.parametrize("n_samples", [1, 30])
@pytest.mark.parametrize(
    "name", ["compute_pre_decision_metrics", "compute_decision_analysis"]
)
def test_decision_functions_one_dimensional_positions_raise(
    t_maze_environment, name, n_samples
):
    """A heading needs x and y, so 1-D positions raise, naming the passed shape."""
    from neurospatial.behavior import (
        compute_decision_analysis,
        compute_pre_decision_metrics,
    )

    times = np.arange(n_samples) / 10
    calls = {
        "compute_pre_decision_metrics": lambda p: compute_pre_decision_metrics(
            times, p, 2.0, window_duration=1.0, min_speed=1.0
        ),
        "compute_decision_analysis": lambda p: compute_decision_analysis(
            t_maze_environment,
            times,
            p,
            decision_region="center",
            goal_regions=["left", "right"],
            min_speed=1.0,
        ),
    }
    match = rf"got shape \({n_samples},\)(.|\n)*heading .*needs both x and y"
    with pytest.raises(ValueError, match=match):
        calls[name](10 * times)
