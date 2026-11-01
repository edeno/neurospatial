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
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import detect_region_crossings

        crossings = detect_region_crossings(
            env,
            times,
            trajectory,
            region_name="target",
            direction="both",
            max_gap=None,
        )

        # Region bins have centers 40-60 cm on this row, so x = 40, 50, 55 are
        # inside and x = 20, 30 outside: in at 2 s, out at 7 s, in at 9 s.
        assert [(c.time, c.direction) for c in crossings] == [
            (2.0, "entry"),
            (7.0, "exit"),
            (9.0, "entry"),
        ]

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

        # Outside (20, 30 cm) and inside (40, 50 cm): entries at 1 s and 4 s.
        x_traj = np.array([20.0, 40.0, 50.0, 30.0, 50.0])
        y_traj = np.full(5, 50.0)
        trajectory = np.column_stack([x_traj, y_traj])
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import detect_region_crossings

        crossings = detect_region_crossings(
            env,
            times,
            trajectory,
            region_name="target",
            direction="entry",
            max_gap=None,
        )

        assert [(c.time, c.direction) for c in crossings] == [
            (1.0, "entry"),
            (4.0, "entry"),
        ]

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

        # Start inside the region; exits at 2 s and 4 s.
        x_traj = np.array([50.0, 55.0, 30.0, 50.0, 20.0])
        y_traj = np.full(5, 50.0)
        trajectory = np.column_stack([x_traj, y_traj])
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import detect_region_crossings

        crossings = detect_region_crossings(
            env,
            times,
            trajectory,
            region_name="target",
            direction="exit",
            max_gap=None,
        )

        assert [(c.time, c.direction) for c in crossings] == [
            (2.0, "exit"),
            (4.0, "exit"),
        ]

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
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import detect_region_crossings

        crossings = detect_region_crossings(
            env,
            times,
            trajectory,
            region_name="target",
            direction="both",
            max_gap=None,
        )

        # Should have no crossings
        assert len(crossings) == 0


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
        ).windows

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
        ).windows
        narrow = segment_by_velocity(
            times, trajectory, min_speed=1.0, min_duration=0.0, hysteresis=1.01
        ).windows

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
        ).windows

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
        ).windows

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

    def test_returns_epoch_intervals(self):
        """Movement epochs come back as Intervals of [start, stop) windows."""
        # Rest 0-5 s, move at 20 cm/s 5-10 s, rest 10-15 s, sampled at 100 Hz.
        times = np.arange(1501) / 100.0
        x = np.clip(20.0 * (times - 5.0), 0.0, 100.0)

        from neurospatial.behavior.segmentation import segment_by_velocity

        from neurospatial import Interval, Intervals

        epochs = segment_by_velocity(
            times, x[:, None], min_speed=5.0, smooth_window=0.0
        )
        assert isinstance(epochs, Intervals)
        assert epochs.record_type is Interval
        assert epochs.fields == ()
        assert epochs.interval_ids.tolist() == [0]
        assert epochs.analysis_settings["min_speed"] == 5.0
        segments = epochs.windows
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

        from neurospatial.behavior import detect_trials
        from neurospatial.behavior.segmentation import (
            detect_region_crossings,
            segment_by_velocity,
        )

        # Map to bins

        # 1. Detect region crossings
        source_crossings = detect_region_crossings(
            env, times, trajectory, region_name="source", direction="both"
        )
        target_crossings = detect_region_crossings(
            env, times, trajectory, region_name="target", direction="both"
        )

        # 2. Detect trials between the regions
        trials = detect_trials(env, times, trajectory, regions=["source", "target"])

        # 3. Segment by velocity
        movement_segments = segment_by_velocity(
            times, trajectory, min_speed=2.0, min_duration=0.5
        ).windows

        # All functions should execute successfully
        assert isinstance(source_crossings, list)
        assert isinstance(target_crossings, list)
        assert trials.end_region.tolist() == ["target"]
        assert movement_segments.shape[1:] == (2,)

        # Should detect source exit and target entry
        assert len(source_crossings) > 0 or len(target_crossings) > 0

        # Should detect at least one movement segment
        assert len(movement_segments) > 0


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


class TestTrialsOnWTrack:
    """Each trial runs from leaving one well to arriving at the next."""

    def test_w_track_trials(self):
        from neurospatial.behavior import detect_trials

        env, bins, times = _w_track()
        x = env.bin_centers[bins]
        trials = detect_trials(env, times, x, regions=["center", "left", "right"])
        got = list(
            zip(
                trials.start,
                trials.stop,
                trials.start_region,
                trials.end_region,
                strict=True,
            )
        )
        expected = [
            (0.7, 2.7, "center", "left"),
            (3.7, 5.7, "left", "center"),
            (6.5, 8.5, "center", "right"),
            (9.5, 11.5, "right", "center"),
        ]
        assert len(got) == len(expected)
        for (s, e, a, b), (es, ee, ea, eb) in zip(got, expected, strict=True):
            assert (s, e) == pytest.approx((es, ee))
            assert (a, b) == (ea, eb)
        assert set(trials.outcome) == {"reached"}

    def test_w_track_direction_labels_from_trials(self):
        """Inbound/outbound labels are the trials' start and end regions."""
        from neurospatial.behavior import detect_trials

        env, bins, times = _w_track()
        trials = detect_trials(
            env, times, env.bin_centers[bins], regions=["center", "left", "right"]
        )
        reached = trials[trials.outcome == "reached"]
        labels = reached.sample_values(times, ("start_region", "end_region"))
        expected = np.full(times.size, None, dtype=object)
        expected[7:27] = [("center", "left")] * 20
        expected[37:57] = [("left", "center")] * 20
        expected[65:85] = [("center", "right")] * 20
        expected[95:115] = [("right", "center")] * 20
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

        crossings = detect_region_crossings(
            env,
            np.arange(5.0),
            np.c_[x, np.zeros_like(x)],
            region_name="goal",
            max_gap=None,
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
            ((4.0, 1.0, "clockwise", 0.9), "start_time <= stop_time"),
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
        assert (type(lap.start_time), type(lap.stop_time)) == (float, float)
        assert np.isnan(lap.overlap_score)
