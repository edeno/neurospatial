"""Tests for region-based trajectory segmentation."""

import numpy as np
import pytest
from shapely.geometry import Point

from neurospatial import Environment, Regions


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
        region_builder = Regions(env.regions.values())
        region_builder.add("target", polygon=Point(50.0, 50.0).buffer(10.0))
        env = env.with_regions(region_builder)

        # Create trajectory that enters and exits region
        # Start outside, enter, stay inside, exit, enter again
        x_traj = np.array(
            [20.0, 30.0, 40.0, 50.0, 55.0, 50.0, 40.0, 30.0, 20.0, 40.0, 50.0, 55.0]
        )
        y_traj = np.array(
            [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
        )
        trajectory = np.column_stack([x_traj, y_traj])
        position_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import detect_region_crossings

        crossings = detect_region_crossings(
            position_bins,
            times,
            env,
            region_name="target",
            direction="both",
            max_gap=None,
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
        region_builder = Regions(env.regions.values())
        region_builder.add("target", polygon=Point(50.0, 50.0).buffer(10.0))
        env = env.with_regions(region_builder)

        x_traj = np.array([20.0, 40.0, 50.0, 55.0, 50.0, 40.0, 50.0])
        y_traj = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        trajectory = np.column_stack([x_traj, y_traj])
        position_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import detect_region_crossings

        crossings = detect_region_crossings(
            position_bins,
            times,
            env,
            region_name="target",
            direction="entry",
            max_gap=None,
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
        region_builder = Regions(env.regions.values())
        region_builder.add("target", polygon=Point(50.0, 50.0).buffer(10.0))
        env = env.with_regions(region_builder)

        # Start inside region
        x_traj = np.array([50.0, 55.0, 50.0, 40.0, 20.0, 50.0, 40.0])
        y_traj = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        trajectory = np.column_stack([x_traj, y_traj])
        position_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import detect_region_crossings

        crossings = detect_region_crossings(
            position_bins,
            times,
            env,
            region_name="target",
            direction="exit",
            max_gap=None,
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
        region_builder = Regions(env.regions.values())
        region_builder.add("target", polygon=Point(50.0, 50.0).buffer(10.0))
        env = env.with_regions(region_builder)

        # Stay far from region
        x_traj = np.array([10.0, 15.0, 20.0, 15.0, 10.0])
        y_traj = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        trajectory = np.column_stack([x_traj, y_traj])
        position_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import detect_region_crossings

        crossings = detect_region_crossings(
            position_bins,
            times,
            env,
            region_name="target",
            direction="both",
            max_gap=None,
        )

        # Should have no crossings
        assert len(crossings) == 0


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
        region_builder = Regions(env.regions.values())
        region_builder.add("source", polygon=Point(10.0, 50.0).buffer(5.0))
        region_builder.add("target", polygon=Point(90.0, 50.0).buffer(5.0))
        env = env.with_regions(region_builder)

        # Create trajectory: start in source, travel to target
        x_traj = np.linspace(10.0, 90.0, 50)
        y_traj = np.ones(50) * 50.0
        trajectory = np.column_stack([x_traj, y_traj])
        times = np.linspace(0, 5.0, 50)

        from neurospatial.behavior.segmentation import detect_runs_between_regions

        position_bins = env.bin_at(trajectory)
        runs = detect_runs_between_regions(
            position_bins,
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
        """A trajectory that never reaches the target must produce a
        failed run when ``max_duration`` is exceeded.

        Construct a trajectory that leaves the source region and then
        stalls in the middle of the arena — never touching the target.
        The detector should yield at least one ``Run`` whose
        ``success`` flag is ``False`` (timeout).
        """
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("source", polygon=Point(10.0, 50.0).buffer(5.0))
        region_builder.add("target", polygon=Point(90.0, 50.0).buffer(5.0))
        env = env.with_regions(region_builder)

        # Trajectory leaves source (10, 50), moves to mid-arena (50, 50)
        # over 1.5 s, then dwells there. ``max_duration=1.0`` < dwell
        # time ⇒ at least one timeout failure must be reported.
        x_traj = np.concatenate([np.linspace(10.0, 50.0, 30), np.full(70, 50.0)])
        y_traj = np.full(100, 50.0)
        trajectory = np.column_stack([x_traj, y_traj])
        times = np.linspace(0, 20.0, 100)

        from neurospatial.behavior.segmentation import detect_runs_between_regions

        position_bins = env.bin_at(trajectory)
        runs = detect_runs_between_regions(
            position_bins,
            times,
            env,
            source="source",
            target="target",
            min_duration=0.5,
            max_duration=1.0,  # Will timeout — trajectory dwells past this
        )

        assert len(runs) >= 1, "Expected at least one run to be reported."
        # At least one run must be a documented failure (not success).
        # No real run can succeed because the trajectory never reaches
        # ``target``.
        assert any(not r.success for r in runs), (
            "Trajectory never reached target; at least one failed run "
            f"should be reported, got runs={runs}."
        )
        assert not any(r.success for r in runs), (
            "No run can succeed since the trajectory never touches "
            f"`target`; got runs={runs}."
        )

    def test_min_duration_filter(self):
        """Test that runs shorter than min_duration are filtered."""
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("source", polygon=Point(10.0, 50.0).buffer(5.0))
        region_builder.add("target", polygon=Point(90.0, 50.0).buffer(5.0))
        env = env.with_regions(region_builder)

        # Very fast run (0.1 seconds)
        x_traj = np.linspace(10.0, 90.0, 10)
        y_traj = np.ones(10) * 50.0
        trajectory = np.column_stack([x_traj, y_traj])
        times = np.linspace(0, 0.1, 10)

        from neurospatial.behavior.segmentation import detect_runs_between_regions

        position_bins = env.bin_at(trajectory)
        runs = detect_runs_between_regions(
            position_bins,
            times,
            env,
            source="source",
            target="target",
            min_duration=1.0,  # Require at least 1 second
            max_duration=10.0,
        )

        # Short run should be filtered out
        assert len(runs) == 0

    def test_timeout_run_bins_consistent_with_velocity_filter(self):
        """Timeout run's stored bins span exactly the velocity-filtered slice.

        The stored ``run.bins`` and the slice used for the optional velocity
        filter must describe the same samples, including at the timeout
        boundary. This pins the unified slice so the path and filter cannot
        diverge by an off-by-one.
        """
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("source", polygon=Point(10.0, 50.0).buffer(5.0))
        region_builder.add("target", polygon=Point(90.0, 50.0).buffer(5.0))
        env = env.with_regions(region_builder)

        # Leave source, move steadily toward (but never reaching) target,
        # so the run times out partway with a non-trivial moving path.
        x_traj = np.concatenate([np.full(5, 10.0), np.linspace(10.0, 60.0, 45)])
        y_traj = np.full(50, 50.0)
        trajectory = np.column_stack([x_traj, y_traj])
        times = np.linspace(0, 10.0, 50)

        from neurospatial.behavior.segmentation import detect_runs_between_regions

        position_bins = env.bin_at(trajectory)
        runs = detect_runs_between_regions(
            position_bins,
            times,
            env,
            source="source",
            target="target",
            min_duration=0.5,
            max_duration=2.0,  # Times out before reaching target
            min_speed=1.0,  # Activate the velocity filter
        )

        assert len(runs) >= 1
        run = runs[0]
        assert run.success is False, "Run should time out without reaching target"

        # The stored bins must correspond to a contiguous trajectory slice
        # whose endpoints match start_time / end_time. Recover that slice from
        # the timestamps and confirm the stored bins equal it exactly.
        start_i = int(np.searchsorted(times, run.start_time))
        end_i = int(np.searchsorted(times, run.end_time))
        expected = position_bins[start_i : end_i + 1]
        np.testing.assert_array_equal(run.bins, expected)


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

        from neurospatial.behavior.segmentation import segment_by_velocity

        # Threshold chosen to separate movement from rest
        segments = segment_by_velocity(
            times, trajectory, min_speed=2.0, min_duration=0.5, hysteresis=2.0
        )

        # Should detect at least one movement period
        assert len(segments) > 0

    def test_hysteresis_prevents_flickering(self):
        """Higher hysteresis must produce no more segments than lower.

        Construct a velocity time series that oscillates just above and
        below the threshold. A wide hysteresis band suppresses crossings
        in the middle of the band; a narrow band lets every crossing
        through. The test asserts the monotone direction
        (wide ≤ narrow), which is the contract — not just that both
        calls returned a ``list``.
        """
        rng = np.random.default_rng(42)
        n = 200
        # Trajectory whose first-difference (≈ speed) wanders around the
        # threshold of 1.0, generating many crossings.
        velocity = 1.0 + 0.5 * rng.standard_normal(n)
        trajectory = np.cumsum(velocity)[:, None]
        times = np.linspace(0, 20, n)

        from neurospatial.behavior.segmentation import segment_by_velocity

        wide = segment_by_velocity(
            times, trajectory, min_speed=1.0, min_duration=0.0, hysteresis=2.0
        )
        narrow = segment_by_velocity(
            times, trajectory, min_speed=1.0, min_duration=0.0, hysteresis=1.01
        )

        # Wide hysteresis band ⇒ fewer (or equal) transitions, hence
        # fewer segments. A strict ``<`` would over-fit this noise; the
        # real contract is the monotone direction.
        assert len(wide) <= len(narrow), (
            f"Wider hysteresis produced more segments "
            f"({len(wide)} vs {len(narrow)}) — flickering not suppressed."
        )

    def test_min_duration_filter(self):
        """Test that brief segments are filtered out."""
        # Create trajectory with very brief movement
        rest = np.zeros((100, 1))
        brief_movement = np.linspace(0, 10, 5)[:, None]
        trajectory = np.vstack([rest[:50], brief_movement, rest[50:]])
        times = np.linspace(0, 10, len(trajectory))

        from neurospatial.behavior.segmentation import segment_by_velocity

        segments = segment_by_velocity(
            times,
            trajectory,
            min_speed=5.0,
            min_duration=1.0,  # Require 1 second
        )

        # Brief movement should be filtered
        # All segments should have duration >= min_duration
        for run in segments:
            assert run.end_time - run.start_time >= 1.0 - 1e-6

    def test_movement_epoch_extends_to_recording_boundary(self):
        """A constant-speed run touching the recording edges is not broken.

        Zero-padded ``mode="same"`` smoothing artificially suppresses velocity
        at the trajectory boundaries, which can drop edge samples below the
        movement threshold and truncate (or drop) an epoch that runs to the
        recording edge. With edge-aware smoothing, a trajectory that moves at a
        constant speed from the first sample to the last must be detected as a
        single movement epoch spanning essentially the whole recording.
        """
        # Constant speed of 10 units/s for the entire recording.
        n = 60
        times = np.linspace(0, 6.0, n)  # dt = ~0.1017 s
        speed = 10.0
        x = speed * times
        trajectory = x[:, None]

        from neurospatial.behavior.segmentation import segment_by_velocity

        # min_speed just below the true speed; with zero-padding the suppressed
        # edges would dip under this threshold and break the epoch.
        segments = segment_by_velocity(
            times,
            trajectory,
            min_speed=8.0,
            min_duration=0.5,
            hysteresis=2.0,
            smooth_window=0.5,  # multi-sample window -> noticeable edge effect
        )

        assert len(segments) == 1, (
            f"Constant-speed run should be one epoch, got {len(segments)}"
        )
        start_time, end_time = segments[0]
        # With correct edge handling the smoothed velocity stays at the true
        # constant speed across the whole recording, so the epoch begins at the
        # very first velocity sample (times[1]) and runs to the last sample.
        # Zero-padded "same" smoothing would instead delay the start by several
        # samples. Allow only a single sample of slack at each end.
        dt_med = float(np.median(np.diff(times)))
        assert start_time <= times[1] + 1.5 * dt_med, (
            "Movement epoch start was delayed by boundary smoothing; "
            f"start_time={start_time:.3f}, expected near {times[1]:.3f}."
        )
        assert end_time >= times[-1] - 1.5 * dt_med, (
            "Movement epoch end was truncated by boundary smoothing; "
            f"end_time={end_time:.3f}, expected near {times[-1]:.3f}."
        )

    def test_returns_epoch_array(self):
        """Movement epochs come back as a float (n_epochs, 2) [start, stop) array."""
        # Rest 0-5 s, move at 20 cm/s 5-10 s, rest 10-15 s, sampled at 100 Hz.
        times = np.arange(1501) / 100.0
        x = np.clip(20.0 * (times - 5.0), 0.0, 100.0)

        from neurospatial.behavior.segmentation import segment_by_velocity

        segments = segment_by_velocity(
            times, x[:, None], min_speed=5.0, smooth_window=0.0
        )

        assert isinstance(segments, np.ndarray)
        assert segments.dtype == np.float64
        assert segments.shape == (1, 2)
        # The centered-secant speed is 10 cm/s at 5.00 s (x goes 0 -> 0.2 cm
        # over 4.99-5.01 s) and stays 10-20 cm/s through 10.00 s; at 10.01 s
        # it is 0. The epoch is [first moving sample, first resting sample).
        np.testing.assert_allclose(segments[0], [5.00, 10.01], atol=1e-9)


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

        region_builder = Regions(env.regions.values())
        region_builder.add("source", polygon=Point(10.0, 50.0).buffer(5.0))
        region_builder.add("target", polygon=Point(90.0, 50.0).buffer(5.0))
        env = env.with_regions(region_builder)

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

        from neurospatial.behavior.segmentation import (
            detect_region_crossings,
            detect_runs_between_regions,
            segment_by_velocity,
        )

        # Map to bins
        position_bins = env.bin_at(trajectory)

        # 1. Detect region crossings
        source_crossings = detect_region_crossings(
            position_bins, times, env, region_name="source", direction="both"
        )
        target_crossings = detect_region_crossings(
            position_bins, times, env, region_name="target", direction="both"
        )

        # 2. Detect runs between regions
        position_bins = env.bin_at(trajectory)
        runs = detect_runs_between_regions(
            position_bins,
            times,
            env,
            source="source",
            target="target",
            min_duration=0.5,
            max_duration=20.0,
        )

        # 3. Segment by velocity
        movement_segments = segment_by_velocity(
            times, trajectory, min_speed=2.0, min_duration=0.5
        )

        # All functions should execute successfully
        assert isinstance(source_crossings, list)
        assert isinstance(target_crossings, list)
        assert isinstance(runs, list)
        assert movement_segments.shape[1:] == (2,)

        # Should detect source exit and target entry
        assert len(source_crossings) > 0 or len(target_crossings) > 0

        # Should detect at least one movement segment
        assert len(movement_segments) > 0


class TestRunRecord:
    """Run records source, target and outcome; success is derived."""

    def test_fields_and_derived_success(self):
        from neurospatial.behavior.segmentation import Run

        run = Run(1.0, 2.5, "home", "reward", "reached", np.array([3, 4, 5]))
        assert run.source_region == "home"
        assert run.target_region == "reward"
        assert run.success is True
        assert run.bins.dtype == np.int64
        assert not run.bins.flags.writeable
        for outcome in ("timeout", "returned", "censored"):
            assert Run(1.0, 2.5, "home", "reward", outcome, run.bins).success is False

    @pytest.mark.parametrize(
        ("args", "match"),
        [
            ((3.0, 2.0, "home", "reward", "reached", [1]), "start_time <= end_time"),
            ((0.0, 2.0, "home", "reward", "done", [1]), "outcome must be one of"),
            ((0.0, 2.0, "home", "reward", "reached", [1.5]), "integer"),
        ],
    )
    def test_invalid_records_raise(self, args, match):
        from neurospatial.behavior.segmentation import Run

        with pytest.raises(ValueError, match=match):
            Run(*args)

    def test_run_cut_off_by_recording_end_is_censored(self):
        """A run still under way when the recording ends has an unknown result."""
        from shapely.geometry import box

        from neurospatial.behavior.segmentation import detect_runs_between_regions

        xs = np.linspace(0.0, 100.0, 401)
        env = Environment.from_samples(np.c_[xs, np.zeros_like(xs)], bin_size=2.0)
        regions = Regions(env.regions.values())
        regions.add("home", polygon=box(-5.0, -5.0, 5.0, 5.0))
        regions.add("reward", polygon=box(95.0, -5.0, 105.0, 5.0))
        env = env.with_regions(regions)
        # Leave home at 10 cm/s; the recording stops at 60 cm, 5.9 s later.
        x = np.r_[np.zeros(5), np.linspace(1.0, 60.0, 60)]
        times = np.arange(x.size) / 10.0
        bins = env.bin_at(np.c_[x, np.zeros_like(x)])

        (run,) = detect_runs_between_regions(
            bins, times, env, source="home", target="reward", max_gap=None
        )

        assert run.outcome == "censored"
        assert run.end_time == times[-1]
        assert (run.source_region, run.target_region) == ("home", "reward")


def _w_track():
    """Center well at 50 cm, end wells at 0 and 100 cm of a 2 cm-bin corridor.

    One sample every 2 cm at 10 Hz, on bin centers, so region membership
    changes at known samples: center -> left -> center -> right -> center.
    """
    from shapely.geometry import box

    xs = np.linspace(0.0, 100.0, 401)
    env = Environment.from_samples(np.c_[xs, np.zeros_like(xs)], bin_size=2.0)
    regions = Regions(env.regions.values())
    regions.add("center", polygon=box(45.0, -5.0, 55.0, 5.0))  # bins 46-54
    regions.add("left", polygon=box(-5.0, -5.0, 5.0, 5.0))  # bins 0-4
    regions.add("right", polygon=box(95.0, -5.0, 105.0, 5.0))  # bins 96-100
    env = env.with_regions(regions)
    x = np.r_[
        [50.0] * 5,  # 0-4
        np.arange(48, -2, -2),  # 5-29: leave center at 7 (44 cm), left at 27
        [0.0] * 5,  # 30-34
        np.arange(2, 52, 2),  # 35-59: leave left at 37, center at 57
        [50.0] * 3,  # 60-62
        np.arange(52, 102, 2),  # 63-87: leave center at 65, right at 85
        [100.0] * 5,  # 88-92
        np.arange(98, 48, -2),  # 93-117: leave right at 95, center at 115
        [50.0] * 5,  # 118-122
    ]
    times = np.arange(x.size) / 10.0
    return env, env.bin_at(np.c_[x, np.zeros_like(x)]), times


class TestRunsEndOnSourceReentry:
    """A source re-entry ends the attempt; runs never overlap."""

    @pytest.mark.parametrize(
        ("source", "target", "expected"),
        [
            ("center", "left", [(0.7, 2.7, "reached"), (6.5, 11.5, "returned")]),
            ("center", "right", [(0.7, 5.7, "returned"), (6.5, 8.5, "reached")]),
            ("left", "center", [(3.7, 5.7, "reached")]),
            ("right", "center", [(9.5, 11.5, "reached")]),
        ],
    )
    def test_w_track_runs(self, source, target, expected):
        from neurospatial.behavior.segmentation import detect_runs_between_regions

        env, bins, times = _w_track()
        runs = detect_runs_between_regions(
            bins, times, env, source=source, target=target
        )
        got = [(run.start_time, run.end_time, run.outcome) for run in runs]
        assert len(got) == len(expected)
        for (s, e, o), (es, ee, eo) in zip(got, expected, strict=True):
            assert (s, e, o) == pytest.approx((es, ee, eo))

    @pytest.mark.parametrize("successful_only", [True, False])
    def test_w_track_running_direction_labels(self, successful_only):
        from neurospatial.behavior.segmentation import running_direction_labels

        env, bins, times = _w_track()
        labels = running_direction_labels(
            bins,
            times,
            env,
            start_region="center",
            end_regions=["left", "right"],
            successful_only=successful_only,
        )
        expected = np.full(times.size, None, dtype=object)
        expected[7:27] = "outbound"  # [0.7, 2.7) center -> left
        expected[37:57] = "inbound"  # [3.7, 5.7) left -> center
        expected[65:85] = "outbound"  # [6.5, 8.5) center -> right
        expected[95:115] = "inbound"  # [9.5, 11.5) right -> center
        assert labels.tolist() == expected.tolist()


class TestCrossingAndLapRecords:
    """Crossing names its region; records hold plain, checked scalars."""

    def test_crossing_records_region_and_plain_scalars(self):
        from shapely.geometry import box

        from neurospatial.behavior.segmentation import detect_region_crossings

        xs = np.linspace(0.0, 100.0, 401)
        env = Environment.from_samples(np.c_[xs, np.zeros_like(xs)], bin_size=2.0)
        regions = Regions(env.regions.values())
        regions.add("goal", polygon=box(95.0, -5.0, 105.0, 5.0))
        env = env.with_regions(regions)
        x = np.array([90.0, 92.0, 96.0, 98.0, 92.0])
        bins = env.bin_at(np.c_[x, np.zeros_like(x)])

        crossings = detect_region_crossings(
            bins, np.arange(5.0), env, region_name="goal", max_gap=None
        )

        assert [(c.time, c.direction, c.region) for c in crossings] == [
            (2.0, "entry", "goal"),
            (4.0, "exit", "goal"),
        ]
        assert all(type(c.time) is float for c in crossings)
        assert all(type(c.bin_index) is int for c in crossings)

    def test_invalid_crossing_raises(self):
        from neurospatial.behavior.segmentation import Crossing

        with pytest.raises(ValueError, match="direction must be one of"):
            Crossing(1.0, "enter", 3, "goal")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="time must be finite"):
            Crossing(np.nan, "entry", 3, "goal")

    @pytest.mark.parametrize(
        ("args", "match"),
        [
            ((4.0, 1.0, "clockwise", 0.9), "start_time <= end_time"),
            ((0.0, 1.0, "cw", 0.9), "direction must be one of"),
            ((0.0, 1.0, "clockwise", 1.5), r"overlap_score must be in \[0, 1\]"),
        ],
    )
    def test_invalid_lap_raises(self, args, match):
        from neurospatial.behavior.segmentation import Lap

        with pytest.raises(ValueError, match=match):
            Lap(*args)

    def test_lap_stores_floats_and_allows_nan_overlap(self):
        from neurospatial.behavior.segmentation import Lap

        lap = Lap(np.int64(0), np.float32(2.5), "unknown", np.nan)
        assert (type(lap.start_time), type(lap.end_time)) == (float, float)
        assert np.isnan(lap.overlap_score)
