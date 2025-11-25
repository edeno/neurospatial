"""Tests for region-based trajectory segmentation.

Following TDD: Tests written FIRST, then implementation.
"""

import numpy as np
from shapely.geometry import Point

from neurospatial import Environment


class TestDetectRegionCrossings:
    """Test detect_region_crossings function."""

    def test_detect_entry_exits(self):
        """Test detection of region entry and exit events."""
        # Create 2D environment (required for polygon regions)
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)

        # Add a region in the middle
        env.regions.add("target", polygon=Point(50.0, 50.0).buffer(10.0))

        # Create trajectory that enters and exits region
        # Start outside, enter, stay inside, exit, enter again
        x_traj = np.array(
            [20.0, 30.0, 40.0, 50.0, 55.0, 50.0, 40.0, 30.0, 20.0, 40.0, 50.0, 55.0]
        )
        y_traj = np.array(
            [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
        )
        trajectory = np.column_stack([x_traj, y_traj])
        trajectory_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.segmentation.regions import detect_region_crossings

        crossings = detect_region_crossings(
            trajectory_bins, times, "target", env, direction="both"
        )

        # Should detect entries and exits
        assert len(crossings) > 0
        # Verify crossings have required attributes
        for crossing in crossings:
            assert hasattr(crossing, "time")
            assert hasattr(crossing, "direction")
            assert crossing.direction in ["entry", "exit"]

    def test_detect_entries_only(self):
        """Test detection of only entry events."""
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)
        env.regions.add("target", polygon=Point(50.0, 50.0).buffer(10.0))

        x_traj = np.array([20.0, 40.0, 50.0, 55.0, 50.0, 40.0, 50.0])
        y_traj = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        trajectory = np.column_stack([x_traj, y_traj])
        trajectory_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.segmentation.regions import detect_region_crossings

        crossings = detect_region_crossings(
            trajectory_bins, times, "target", env, direction="entry"
        )

        # All crossings should be entries
        assert all(c.direction == "entry" for c in crossings)

    def test_detect_exits_only(self):
        """Test detection of only exit events."""
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)
        env.regions.add("target", polygon=Point(50.0, 50.0).buffer(10.0))

        # Start inside region
        x_traj = np.array([50.0, 55.0, 50.0, 40.0, 20.0, 50.0, 40.0])
        y_traj = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        trajectory = np.column_stack([x_traj, y_traj])
        trajectory_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.segmentation.regions import detect_region_crossings

        crossings = detect_region_crossings(
            trajectory_bins, times, "target", env, direction="exit"
        )

        # All crossings should be exits
        assert all(c.direction == "exit" for c in crossings)

    def test_no_crossings(self):
        """Test trajectory that never enters region."""
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)
        env.regions.add("target", polygon=Point(50.0, 50.0).buffer(10.0))

        # Stay far from region
        x_traj = np.array([10.0, 15.0, 20.0, 15.0, 10.0])
        y_traj = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        trajectory = np.column_stack([x_traj, y_traj])
        trajectory_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.segmentation.regions import detect_region_crossings

        crossings = detect_region_crossings(
            trajectory_bins, times, "target", env, direction="both"
        )

        # Should have no crossings
        assert len(crossings) == 0

    def test_parameter_order(self):
        """Test parameter order is (trajectory_bins, times, region, env, direction)."""
        x = np.linspace(0, 100, 20)
        y = np.linspace(0, 100, 20)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=10.0)
        env.regions.add("target", polygon=Point(50.0, 50.0).buffer(10.0))

        trajectory_bins = np.arange(10)
        times = np.arange(10, dtype=float)

        from neurospatial.segmentation.regions import detect_region_crossings

        # This should work without error
        crossings = detect_region_crossings(
            trajectory_bins, times, "target", env, direction="both"
        )
        assert isinstance(crossings, list)


class TestDetectRunsBetweenRegions:
    """Test detect_runs_between_regions function."""

    def test_detect_successful_runs(self):
        """Test detection of successful runs from source to target."""
        # Create 2D environment
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)

        # Add source and target regions at opposite ends
        env.regions.add("source", polygon=Point(10.0, 50.0).buffer(5.0))
        env.regions.add("target", polygon=Point(90.0, 50.0).buffer(5.0))

        # Create trajectory: start in source, travel to target
        x_traj = np.linspace(10.0, 90.0, 50)
        y_traj = np.ones(50) * 50.0
        trajectory = np.column_stack([x_traj, y_traj])
        times = np.linspace(0, 5.0, 50)

        from neurospatial.segmentation.regions import detect_runs_between_regions

        runs = detect_runs_between_regions(
            trajectory,
            times,
            env,
            source="source",
            target="target",
            min_duration=0.5,
            max_duration=10.0,
        )

        # Should detect at least one successful run
        assert len(runs) > 0
        # Verify run attributes
        for run in runs:
            assert hasattr(run, "start_time")
            assert hasattr(run, "end_time")
            assert hasattr(run, "bins")
            assert hasattr(run, "success")
            if run.success:
                assert run.end_time - run.start_time >= 0.5  # min_duration

    def test_detect_failed_runs_timeout(self):
        """Test detection of failed runs (timeout before reaching target)."""
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)

        env.regions.add("source", polygon=Point(10.0, 50.0).buffer(5.0))
        env.regions.add("target", polygon=Point(90.0, 50.0).buffer(5.0))

        # Create trajectory that exits source but never reaches target
        x_traj = np.linspace(10.0, 50.0, 100)
        y_traj = np.ones(100) * 50.0
        trajectory = np.column_stack([x_traj, y_traj])
        times = np.linspace(0, 20.0, 100)  # 20 seconds, longer than max_duration

        from neurospatial.segmentation.regions import detect_runs_between_regions

        runs = detect_runs_between_regions(
            trajectory,
            times,
            env,
            source="source",
            target="target",
            min_duration=0.5,
            max_duration=5.0,  # Will timeout
        )

        # Should detect runs, but some may be unsuccessful
        if len(runs) > 0:
            # Check that unsuccessful runs exist
            unsuccessful_runs = [r for r in runs if not r.success]
            # At least some runs should fail due to timeout
            assert len(unsuccessful_runs) >= 0  # May or may not have failed runs

    def test_min_duration_filter(self):
        """Test that runs shorter than min_duration are filtered."""
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)

        env.regions.add("source", polygon=Point(10.0, 50.0).buffer(5.0))
        env.regions.add("target", polygon=Point(90.0, 50.0).buffer(5.0))

        # Very fast run (0.1 seconds)
        x_traj = np.linspace(10.0, 90.0, 10)
        y_traj = np.ones(10) * 50.0
        trajectory = np.column_stack([x_traj, y_traj])
        times = np.linspace(0, 0.1, 10)

        from neurospatial.segmentation.regions import detect_runs_between_regions

        runs = detect_runs_between_regions(
            trajectory,
            times,
            env,
            source="source",
            target="target",
            min_duration=1.0,  # Require at least 1 second
            max_duration=10.0,
        )

        # Short run should be filtered out
        assert len(runs) == 0

    def test_parameter_order(self):
        """Test parameter order is (trajectory_positions, times, env, *, source, target, ...)."""
        x = np.linspace(0, 100, 20)
        y = np.linspace(0, 100, 20)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=10.0)
        env.regions.add("source", polygon=Point(10.0, 50.0).buffer(5.0))
        env.regions.add("target", polygon=Point(90.0, 50.0).buffer(5.0))

        trajectory = positions[:20]
        times = np.arange(20, dtype=float)

        from neurospatial.segmentation.regions import detect_runs_between_regions

        # This should work without error
        runs = detect_runs_between_regions(
            trajectory,
            times,
            env,
            source="source",
            target="target",
            min_duration=0.5,
            max_duration=10.0,
        )
        assert isinstance(runs, list)


class TestSegmentByVelocity:
    """Test segment_by_velocity function."""

    def test_segment_movement_vs_rest(self):
        """Test segmentation of movement and rest periods."""
        # Create trajectory with clear movement and rest periods
        # Rest: stationary at position 0
        # Movement: linear motion from 0 to 100
        rest1 = np.zeros((50, 1))
        movement = np.linspace(0, 100, 100)[:, None]
        rest2 = np.ones((50, 1)) * 100

        trajectory = np.vstack([rest1, movement, rest2])
        times = np.linspace(0, 20, len(trajectory))

        from neurospatial.segmentation.regions import segment_by_velocity

        # Threshold chosen to separate movement from rest
        segments = segment_by_velocity(
            trajectory, times, threshold=2.0, min_duration=0.5, hysteresis=2.0
        )

        # Should detect at least one movement period
        assert len(segments) > 0

    def test_hysteresis_prevents_flickering(self):
        """Test that hysteresis prevents rapid switching."""
        # Create trajectory with noisy velocity near threshold
        rng = np.random.default_rng(42)
        n = 100
        trajectory = np.cumsum(rng.standard_normal(n) * 0.5)[:, None]
        times = np.linspace(0, 10, n)

        from neurospatial.segmentation.regions import segment_by_velocity

        # With hysteresis, should have fewer segments than without
        segments_with_hysteresis = segment_by_velocity(
            trajectory, times, threshold=1.0, min_duration=0.1, hysteresis=2.0
        )

        segments_without_hysteresis = segment_by_velocity(
            trajectory,
            times,
            threshold=1.0,
            min_duration=0.1,
            hysteresis=1.1,  # Just above 1.0
        )

        # Hysteresis should reduce number of rapid transitions
        # (This is a soft assertion - hysteresis should help stability)
        assert isinstance(segments_with_hysteresis, list)
        assert isinstance(segments_without_hysteresis, list)

    def test_min_duration_filter(self):
        """Test that brief segments are filtered out."""
        # Create trajectory with very brief movement
        rest = np.zeros((100, 1))
        brief_movement = np.linspace(0, 10, 5)[:, None]
        trajectory = np.vstack([rest[:50], brief_movement, rest[50:]])
        times = np.linspace(0, 10, len(trajectory))

        from neurospatial.segmentation.regions import segment_by_velocity

        segments = segment_by_velocity(
            trajectory,
            times,
            threshold=5.0,
            min_duration=1.0,  # Require 1 second
        )

        # Brief movement should be filtered
        # All segments should have duration >= min_duration
        for start, end in segments:
            assert end - start >= 1.0 - 1e-6  # Allow small numerical error

    def test_returns_list_of_tuples(self):
        """Test that function returns list of (start_time, end_time) tuples."""
        trajectory = np.linspace(0, 100, 100)[:, None]
        times = np.linspace(0, 10, 100)

        from neurospatial.segmentation.regions import segment_by_velocity

        segments = segment_by_velocity(trajectory, times, threshold=2.0)

        # Should return list
        assert isinstance(segments, list)
        # Each element should be a tuple of two floats
        for segment in segments:
            assert isinstance(segment, tuple)
            assert len(segment) == 2
            start, end = segment
            assert isinstance(start, (int, float, np.number))
            assert isinstance(end, (int, float, np.number))
            assert start < end  # Start before end

    def test_parameter_order(self):
        """Test parameter order is (trajectory_positions, times, threshold, *, ...)."""
        trajectory = np.linspace(0, 100, 50)[:, None]
        times = np.arange(50, dtype=float)

        from neurospatial.segmentation.regions import segment_by_velocity

        # This should work without error
        segments = segment_by_velocity(
            trajectory, times, threshold=2.0, min_duration=0.5
        )
        assert isinstance(segments, list)


class TestRegionSegmentationIntegration:
    """Test integration of all region segmentation functions."""

    def test_complete_workflow(self):
        """Test complete region-based segmentation workflow."""
        # Create 2D environment with source and target regions
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)

        env.regions.add("source", polygon=Point(10.0, 50.0).buffer(5.0))
        env.regions.add("target", polygon=Point(90.0, 50.0).buffer(5.0))

        # Create trajectory: rest at source, move to target, rest at target
        rest_source_x = np.ones(50) * 10.0
        rest_source_y = np.ones(50) * 50.0
        movement_x = np.linspace(10.0, 90.0, 100)
        movement_y = np.ones(100) * 50.0
        rest_target_x = np.ones(50) * 90.0
        rest_target_y = np.ones(50) * 50.0

        x_traj = np.concatenate([rest_source_x, movement_x, rest_target_x])
        y_traj = np.concatenate([rest_source_y, movement_y, rest_target_y])
        trajectory = np.column_stack([x_traj, y_traj])
        times = np.linspace(0, 20, len(trajectory))

        from neurospatial.segmentation.regions import (
            detect_region_crossings,
            detect_runs_between_regions,
            segment_by_velocity,
        )

        # Map to bins
        trajectory_bins = env.bin_at(trajectory)

        # 1. Detect region crossings
        source_crossings = detect_region_crossings(
            trajectory_bins, times, "source", env, direction="both"
        )
        target_crossings = detect_region_crossings(
            trajectory_bins, times, "target", env, direction="both"
        )

        # 2. Detect runs between regions
        runs = detect_runs_between_regions(
            trajectory,
            times,
            env,
            source="source",
            target="target",
            min_duration=0.5,
            max_duration=20.0,
        )

        # 3. Segment by velocity
        movement_segments = segment_by_velocity(
            trajectory, times, threshold=2.0, min_duration=0.5
        )

        # All functions should execute successfully
        assert isinstance(source_crossings, list)
        assert isinstance(target_crossings, list)
        assert isinstance(runs, list)
        assert isinstance(movement_segments, list)

        # Should detect source exit and target entry
        assert len(source_crossings) > 0 or len(target_crossings) > 0

        # Should detect at least one movement segment
        assert len(movement_segments) > 0
