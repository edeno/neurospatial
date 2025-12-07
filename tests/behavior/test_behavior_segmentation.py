"""Tests for behavior/segmentation.py module - TDD RED phase.

This test file verifies that all segmentation functions are importable from
the new location: neurospatial.behavior.segmentation

Following TDD: Tests written FIRST, then implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from shapely.geometry import Point


class TestImportsFromNewLocation:
    """Verify all classes and functions are importable from behavior.segmentation."""

    # Dataclasses
    def test_import_crossing(self):
        """Test Crossing is importable from new location."""
        from neurospatial.behavior.segmentation import Crossing

        assert Crossing is not None

    def test_import_lap(self):
        """Test Lap is importable from new location."""
        from neurospatial.behavior.segmentation import Lap

        assert Lap is not None

    def test_import_run(self):
        """Test Run is importable from new location."""
        from neurospatial.behavior.segmentation import Run

        assert Run is not None

    def test_import_trial(self):
        """Test Trial is importable from new location."""
        from neurospatial.behavior.segmentation import Trial

        assert Trial is not None

    # Functions
    def test_import_detect_region_crossings(self):
        """Test detect_region_crossings is importable from new location."""
        from neurospatial.behavior.segmentation import detect_region_crossings

        assert callable(detect_region_crossings)

    def test_import_detect_runs_between_regions(self):
        """Test detect_runs_between_regions is importable from new location."""
        from neurospatial.behavior.segmentation import detect_runs_between_regions

        assert callable(detect_runs_between_regions)

    def test_import_segment_by_velocity(self):
        """Test segment_by_velocity is importable from new location."""
        from neurospatial.behavior.segmentation import segment_by_velocity

        assert callable(segment_by_velocity)

    def test_import_detect_laps(self):
        """Test detect_laps is importable from new location."""
        from neurospatial.behavior.segmentation import detect_laps

        assert callable(detect_laps)

    def test_import_segment_trials(self):
        """Test segment_trials is importable from new location."""
        from neurospatial.behavior.segmentation import segment_trials

        assert callable(segment_trials)

    def test_import_trajectory_similarity(self):
        """Test trajectory_similarity is importable from new location."""
        from neurospatial.behavior.segmentation import trajectory_similarity

        assert callable(trajectory_similarity)

    def test_import_detect_goal_directed_runs(self):
        """Test detect_goal_directed_runs is importable from new location."""
        from neurospatial.behavior.segmentation import detect_goal_directed_runs

        assert callable(detect_goal_directed_runs)


class TestImportsFromBehaviorInit:
    """Verify all classes and functions are re-exported from behavior/__init__.py."""

    # Dataclasses
    def test_import_crossing_from_behavior(self):
        """Test Crossing is importable from behavior package."""
        from neurospatial.behavior import Crossing

        assert Crossing is not None

    def test_import_lap_from_behavior(self):
        """Test Lap is importable from behavior package."""
        from neurospatial.behavior import Lap

        assert Lap is not None

    def test_import_run_from_behavior(self):
        """Test Run is importable from behavior package."""
        from neurospatial.behavior import Run

        assert Run is not None

    def test_import_trial_from_behavior(self):
        """Test Trial is importable from behavior package."""
        from neurospatial.behavior import Trial

        assert Trial is not None

    # Functions
    def test_import_detect_region_crossings_from_behavior(self):
        """Test detect_region_crossings is importable from behavior package."""
        from neurospatial.behavior import detect_region_crossings

        assert callable(detect_region_crossings)

    def test_import_detect_runs_between_regions_from_behavior(self):
        """Test detect_runs_between_regions is importable from behavior package."""
        from neurospatial.behavior import detect_runs_between_regions

        assert callable(detect_runs_between_regions)

    def test_import_segment_by_velocity_from_behavior(self):
        """Test segment_by_velocity is importable from behavior package."""
        from neurospatial.behavior import segment_by_velocity

        assert callable(segment_by_velocity)

    def test_import_detect_laps_from_behavior(self):
        """Test detect_laps is importable from behavior package."""
        from neurospatial.behavior import detect_laps

        assert callable(detect_laps)

    def test_import_segment_trials_from_behavior(self):
        """Test segment_trials is importable from behavior package."""
        from neurospatial.behavior import segment_trials

        assert callable(segment_trials)

    def test_import_trajectory_similarity_from_behavior(self):
        """Test trajectory_similarity is importable from behavior package."""
        from neurospatial.behavior import trajectory_similarity

        assert callable(trajectory_similarity)

    def test_import_detect_goal_directed_runs_from_behavior(self):
        """Test detect_goal_directed_runs is importable from behavior package."""
        from neurospatial.behavior import detect_goal_directed_runs

        assert callable(detect_goal_directed_runs)


class TestCrossingDataclass:
    """Basic functionality tests for Crossing dataclass from new location."""

    def test_crossing_creation(self):
        """Test Crossing can be instantiated."""
        from neurospatial.behavior.segmentation import Crossing

        crossing = Crossing(time=1.5, direction="entry", bin_index=42)

        assert crossing.time == 1.5
        assert crossing.direction == "entry"
        assert crossing.bin_index == 42

    def test_crossing_is_frozen(self):
        """Test Crossing is immutable (frozen dataclass)."""
        from neurospatial.behavior.segmentation import Crossing

        crossing = Crossing(time=1.5, direction="entry", bin_index=42)

        with pytest.raises(AttributeError):  # FrozenInstanceError (or TypeError)
            crossing.time = 2.0


class TestLapDataclass:
    """Basic functionality tests for Lap dataclass from new location."""

    def test_lap_creation(self):
        """Test Lap can be instantiated."""
        from neurospatial.behavior.segmentation import Lap

        lap = Lap(
            start_time=0.0,
            end_time=10.0,
            direction="clockwise",
            overlap_score=0.95,
        )

        assert lap.start_time == 0.0
        assert lap.end_time == 10.0
        assert lap.direction == "clockwise"
        assert lap.overlap_score == 0.95


class TestRunDataclass:
    """Basic functionality tests for Run dataclass from new location."""

    def test_run_creation(self):
        """Test Run can be instantiated."""
        from neurospatial.behavior.segmentation import Run

        bins = np.array([0, 1, 2, 3], dtype=np.int64)
        run = Run(
            start_time=0.0,
            end_time=5.0,
            bins=bins,
            success=True,
        )

        assert run.start_time == 0.0
        assert run.end_time == 5.0
        assert np.array_equal(run.bins, bins)
        assert run.success is True


class TestTrialDataclass:
    """Basic functionality tests for Trial dataclass from new location."""

    def test_trial_creation(self):
        """Test Trial can be instantiated."""
        from neurospatial.behavior.segmentation import Trial

        trial = Trial(
            start_time=0.0,
            end_time=5.0,
            start_region="start",
            end_region="left",
            success=True,
        )

        assert trial.start_time == 0.0
        assert trial.end_time == 5.0
        assert trial.start_region == "start"
        assert trial.end_region == "left"
        assert trial.success is True

    def test_trial_with_none_end_region(self):
        """Test Trial with None end_region (timeout)."""
        from neurospatial.behavior.segmentation import Trial

        trial = Trial(
            start_time=0.0,
            end_time=15.0,
            start_region="start",
            end_region=None,
            success=False,
        )

        assert trial.end_region is None
        assert trial.success is False


class TestDetectRegionCrossingsBasic:
    """Basic functionality tests for detect_region_crossings from new location."""

    def test_empty_trajectory_returns_empty_list(self):
        """Test that empty trajectory returns empty list."""
        from neurospatial import Environment
        from neurospatial.behavior.segmentation import detect_region_crossings

        positions = np.linspace(0, 100, 50)[:, None]
        env = Environment.from_samples(positions, bin_size=2.0)
        env.regions.add("test", polygon=Point(50.0, 0.0).buffer(10.0))

        trajectory_bins = np.array([], dtype=np.int64)
        times = np.array([], dtype=np.float64)

        crossings = detect_region_crossings(trajectory_bins, times, "test", env)

        assert crossings == []

    def test_region_not_found_raises_error(self):
        """Test that missing region raises ValueError."""
        from neurospatial import Environment
        from neurospatial.behavior.segmentation import detect_region_crossings

        positions = np.linspace(0, 100, 50)[:, None]
        env = Environment.from_samples(positions, bin_size=2.0)

        trajectory_bins = np.array([0, 1, 2], dtype=np.int64)
        times = np.array([0.0, 1.0, 2.0])

        with pytest.raises(ValueError, match="not found"):
            detect_region_crossings(trajectory_bins, times, "nonexistent", env)


class TestSegmentByVelocityBasic:
    """Basic functionality tests for segment_by_velocity from new location."""

    def test_stationary_trajectory_returns_empty(self):
        """Test that stationary trajectory returns no movement segments."""
        from neurospatial.behavior.segmentation import segment_by_velocity

        positions = np.zeros((50, 2))  # Stationary
        times = np.linspace(0, 10, 50)

        segments = segment_by_velocity(
            positions,
            times,
            threshold=1.0,
            min_duration=0.5,
        )

        assert segments == []

    def test_threshold_validation(self):
        """Test that non-positive threshold raises error."""
        from neurospatial.behavior.segmentation import segment_by_velocity

        positions = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        times = np.linspace(0, 10, 50)

        with pytest.raises(ValueError, match="positive"):
            segment_by_velocity(positions, times, threshold=-1.0)


class TestDetectLapsBasic:
    """Basic functionality tests for detect_laps from new location."""

    @pytest.fixture
    def rng(self):
        """Create a seeded random number generator for reproducible tests."""
        return np.random.default_rng(42)

    def test_empty_trajectory_returns_empty(self, rng):
        """Test that empty trajectory returns empty list."""
        from neurospatial import Environment
        from neurospatial.behavior.segmentation import detect_laps

        positions = rng.random((100, 2)) * 100
        env = Environment.from_samples(positions, bin_size=3.0)

        trajectory_bins = np.array([], dtype=np.int64)
        times = np.array([], dtype=np.float64)

        laps = detect_laps(trajectory_bins, times, env)

        assert laps == []

    def test_invalid_method_raises_error(self, rng):
        """Test that invalid method raises ValueError."""
        from neurospatial import Environment
        from neurospatial.behavior.segmentation import detect_laps

        positions = rng.random((100, 2)) * 100
        env = Environment.from_samples(positions, bin_size=3.0)

        trajectory_bins = np.array([0, 1, 2, 3], dtype=np.int64)
        times = np.array([0.0, 1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="method"):
            detect_laps(trajectory_bins, times, env, method="invalid")


class TestSegmentTrialsBasic:
    """Basic functionality tests for segment_trials from new location."""

    @pytest.fixture
    def rng(self):
        """Create a seeded random number generator for reproducible tests."""
        return np.random.default_rng(42)

    def test_missing_start_region_raises_error(self, rng):
        """Test that missing start_region raises ValueError."""
        from neurospatial import Environment
        from neurospatial.behavior.segmentation import segment_trials

        positions = rng.random((100, 2)) * 100
        env = Environment.from_samples(positions, bin_size=3.0)

        trajectory_bins = np.array([0, 1, 2], dtype=np.int64)
        times = np.array([0.0, 1.0, 2.0])

        with pytest.raises(ValueError, match="start_region"):
            segment_trials(
                trajectory_bins,
                times,
                env,
                start_region=None,  # Missing
                end_regions=["goal"],
            )

    def test_missing_end_regions_raises_error(self, rng):
        """Test that missing end_regions raises ValueError."""
        from neurospatial import Environment
        from neurospatial.behavior.segmentation import segment_trials

        positions = rng.random((100, 2)) * 100
        env = Environment.from_samples(positions, bin_size=3.0)
        env.regions.add("start", polygon=Point(50.0, 50.0).buffer(10.0))

        trajectory_bins = np.array([0, 1, 2], dtype=np.int64)
        times = np.array([0.0, 1.0, 2.0])

        with pytest.raises(ValueError, match="end_regions"):
            segment_trials(
                trajectory_bins,
                times,
                env,
                start_region="start",
                end_regions=None,  # Missing
            )


class TestTrajectorySimilarityBasic:
    """Basic functionality tests for trajectory_similarity from new location."""

    @pytest.fixture
    def rng(self):
        """Create a seeded random number generator for reproducible tests."""
        return np.random.default_rng(42)

    def test_identical_trajectories_jaccard(self, rng):
        """Test that identical trajectories have similarity 1.0."""
        from neurospatial import Environment
        from neurospatial.behavior.segmentation import trajectory_similarity

        positions = rng.random((100, 2)) * 100
        env = Environment.from_samples(positions, bin_size=5.0)

        traj = np.array([0, 1, 2, 3, 4], dtype=np.int64)

        similarity = trajectory_similarity(traj, traj, env, method="jaccard")

        assert_allclose(similarity, 1.0)

    def test_empty_trajectory_raises_error(self, rng):
        """Test that empty trajectory raises ValueError."""
        from neurospatial import Environment
        from neurospatial.behavior.segmentation import trajectory_similarity

        positions = rng.random((100, 2)) * 100
        env = Environment.from_samples(positions, bin_size=5.0)

        traj1 = np.array([], dtype=np.int64)
        traj2 = np.array([0, 1, 2], dtype=np.int64)

        with pytest.raises(ValueError, match="empty"):
            trajectory_similarity(traj1, traj2, env)


class TestDetectGoalDirectedRunsBasic:
    """Basic functionality tests for detect_goal_directed_runs from new location."""

    def test_empty_trajectory_returns_empty(self):
        """Test that empty trajectory returns empty list."""
        from neurospatial import Environment
        from neurospatial.behavior.segmentation import detect_goal_directed_runs

        positions = np.linspace(0, 100, 50)[:, None]
        env = Environment.from_samples(positions, bin_size=2.0)
        env.regions.add("goal", polygon=Point(90.0, 0.0).buffer(5.0))

        trajectory_bins = np.array([], dtype=np.int64)
        times = np.array([], dtype=np.float64)

        runs = detect_goal_directed_runs(
            trajectory_bins, times, env, goal_region="goal"
        )

        assert runs == []

    def test_missing_goal_region_raises_error(self):
        """Test that missing goal_region raises KeyError."""
        from neurospatial import Environment
        from neurospatial.behavior.segmentation import detect_goal_directed_runs

        positions = np.linspace(0, 100, 50)[:, None]
        env = Environment.from_samples(positions, bin_size=2.0)

        trajectory_bins = np.array([0, 1, 2], dtype=np.int64)
        times = np.array([0.0, 1.0, 2.0])

        with pytest.raises(KeyError, match="not found"):
            detect_goal_directed_runs(
                trajectory_bins, times, env, goal_region="nonexistent"
            )
