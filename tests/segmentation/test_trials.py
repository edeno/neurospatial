"""Tests for trial segmentation functions."""

import numpy as np
import pytest
from shapely.geometry import Point

from neurospatial import Environment, Regions


class TestSegmentTrials:
    """Test segment_trials function."""

    def test_segment_trials_tmaze_left_right(self):
        """Test trial segmentation on T-maze with left/right outcomes."""
        # Create T-maze trajectory (2D required for polygon regions)
        # Maze: start at bottom, go up stem, turn left or right
        #
        #   L--+--R
        #      |
        #      S
        #
        # Trial 1: Start → Left
        # Trial 2: Start → Right
        # Trial 3: Start → Left

        # Create environment with full coverage
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        # Define regions
        region_builder = Regions(env.regions.values())
        region_builder.add("start", polygon=Point(50.0, 10.0).buffer(8.0))
        region_builder.add("left", polygon=Point(20.0, 80.0).buffer(8.0))
        region_builder.add("right", polygon=Point(80.0, 80.0).buffer(8.0))
        env = env.with_regions(region_builder)

        # Create trajectory with 3 trials
        # Trial 1: Start (50, 10) → Stem (50, 50) → Left (20, 80)
        trial1_x = np.linspace(50, 50, 10).tolist() + np.linspace(50, 20, 10).tolist()
        trial1_y = np.linspace(10, 50, 10).tolist() + np.linspace(50, 80, 10).tolist()

        # Trial 2: Return to start, then go right
        # Start (50, 10) → Stem (50, 50) → Right (80, 80)
        trial2_x = (
            np.linspace(20, 50, 10).tolist()
            + np.linspace(50, 50, 10).tolist()
            + np.linspace(50, 80, 10).tolist()
        )
        trial2_y = (
            np.linspace(80, 10, 10).tolist()
            + np.linspace(10, 50, 10).tolist()
            + np.linspace(50, 80, 10).tolist()
        )

        # Trial 3: Return to start, then go left again
        trial3_x = (
            np.linspace(80, 50, 10).tolist()
            + np.linspace(50, 50, 10).tolist()
            + np.linspace(50, 20, 10).tolist()
        )
        trial3_y = (
            np.linspace(80, 10, 10).tolist()
            + np.linspace(10, 50, 10).tolist()
            + np.linspace(50, 80, 10).tolist()
        )

        x_traj = np.array(trial1_x + trial2_x + trial3_x)
        y_traj = np.array(trial1_y + trial2_y + trial3_y)
        trajectory = np.column_stack([x_traj, y_traj])

        position_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import segment_trials

        trials = segment_trials(
            position_bins,
            times,
            env,
            start_region="start",
            end_regions=["left", "right"],
            min_duration=5.0,
            max_duration=50.0,
            max_gap=None,
        )

        assert [trial.end_region for trial in trials] == ["left", "right", "left"]
        assert [trial.outcome for trial in trials] == ["reached"] * 3
        for trial in trials:
            assert trial.end_time > trial.start_time
            assert trial.start_region == "start"

    def test_segment_trials_duration_filter_min(self):
        """Test that trials shorter than min_duration are excluded."""
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        region_builder.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))
        env = env.with_regions(region_builder)

        # Create very short trial (only 5 timepoints, duration = 4 seconds)
        x_traj = np.linspace(20, 80, 5)
        y_traj = np.linspace(20, 80, 5)
        trajectory = np.column_stack([x_traj, y_traj])
        position_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import segment_trials

        # With min_duration=10.0, should exclude this short trial
        trials = segment_trials(
            position_bins,
            times,
            env,
            start_region="start",
            end_regions=["goal"],
            min_duration=10.0,
            max_duration=50.0,
            max_gap=None,
        )

        assert len(trials) == 0, "Short trial should be excluded"

    def test_segment_trials_duration_filter_max(self):
        """Test that trials longer than max_duration are marked as timeouts."""
        x = np.linspace(0, 100, 200)
        y = np.linspace(0, 100, 200)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        region_builder.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))
        env = env.with_regions(region_builder)

        # Create long meandering trial (100 timepoints, never reaches goal)
        # Start in start region, meander but don't reach goal
        x_traj = np.concatenate(
            [np.linspace(20, 50, 50), np.linspace(50, 40, 50)]  # Start region
        )  # Meander  # Back and forth
        y_traj = np.concatenate([np.linspace(20, 40, 50), np.linspace(40, 30, 50)])
        trajectory = np.column_stack([x_traj, y_traj])
        position_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import segment_trials

        # With max_duration=20.0, this 99-second trial should timeout
        trials = segment_trials(
            position_bins,
            times,
            env,
            start_region="start",
            end_regions=["goal"],
            min_duration=1.0,
            max_duration=20.0,
            max_gap=None,
        )

        # One trial that times out exactly max_duration after it started
        # (1 Hz samples, so a sample lands on the deadline).
        assert len(trials) == 1
        assert trials[0].outcome == "timeout"
        assert trials[0].end_region is None
        assert trials[0].start_region == "start"
        assert trials[0].end_time - trials[0].start_time == 20.0

    def test_segment_trials_successful_completion(self):
        """Test successful trial completion within duration bounds."""
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        region_builder.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))
        env = env.with_regions(region_builder)

        # Create successful trial: Start → Goal
        x_traj = np.linspace(20, 80, 30)
        y_traj = np.linspace(20, 80, 30)
        trajectory = np.column_stack([x_traj, y_traj])
        position_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import segment_trials

        trials = segment_trials(
            position_bins,
            times,
            env,
            start_region="start",
            end_regions=["goal"],
            min_duration=5.0,
            max_duration=50.0,
            max_gap=None,
        )

        # Should detect successful trial
        assert len(trials) >= 1, "Should detect at least one trial"
        assert trials[0].success, "Trial should be successful"
        assert trials[0].end_region == "goal"
        assert trials[0].start_region == "start", "start_region should be 'start'"
        assert 5.0 <= (trials[0].end_time - trials[0].start_time) <= 50.0

    def test_segment_trials_empty_trajectory(self):
        """Test handling of empty trajectory."""
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        region_builder.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))
        env = env.with_regions(region_builder)

        position_bins = np.array([], dtype=int)
        times = np.array([], dtype=float)

        from neurospatial.behavior.segmentation import segment_trials

        trials = segment_trials(
            position_bins,
            times,
            env,
            start_region="start",
            end_regions=["goal"],
        )

        assert len(trials) == 0, "Empty trajectory should produce no trials"

    def test_segment_trials_no_end_region_reached(self):
        """Test trajectory that never reaches any end region."""
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        region_builder.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))
        env = env.with_regions(region_builder)

        # Trajectory that stays in middle, never reaching goal
        x_traj = np.linspace(20, 50, 50)
        y_traj = np.linspace(20, 50, 50)
        trajectory = np.column_stack([x_traj, y_traj])
        position_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import segment_trials

        trials = segment_trials(
            position_bins,
            times,
            env,
            start_region="start",
            end_regions=["goal"],
            min_duration=1.0,
            max_duration=20.0,
            max_gap=None,
        )

        assert len(trials) == 1
        assert trials[0].outcome == "timeout"
        assert trials[0].end_time - trials[0].start_time == 20.0

    def test_segment_trials_parameter_validation(self):
        """Test parameter validation."""
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        region_builder.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))
        env = env.with_regions(region_builder)

        position_bins = np.arange(10)
        times = np.arange(10, dtype=float)

        from neurospatial.behavior.segmentation import segment_trials

        # Missing start_region - passes None which won't be found in regions
        with pytest.raises(ValueError, match=r"(?s)not found.*start_region="):
            segment_trials(
                position_bins,
                times,
                env,
                start_region=None,
                end_regions=["goal"],
            )

        # Missing end_regions - need to pass a string in list form that doesn't exist
        with pytest.raises(ValueError, match=r"(?s)not found.*end_regions="):
            segment_trials(
                position_bins,
                times,
                env,
                start_region="start",
                end_regions=["nonexistent"],
            )

        # Empty end_regions
        with pytest.raises(ValueError, match=r"end_regions.*cannot be empty"):
            segment_trials(
                position_bins,
                times,
                env,
                start_region="start",
                end_regions=[],
            )

        # Nonexistent start_region
        with pytest.raises(ValueError, match=r"(?s)not found.*start_region="):
            segment_trials(
                position_bins,
                times,
                env,
                start_region="nonexistent",
                end_regions=["goal"],
            )

        # Nonexistent end_region
        with pytest.raises(ValueError, match=r"(?s)not found.*end_regions="):
            segment_trials(
                position_bins,
                times,
                env,
                start_region="start",
                end_regions=["nonexistent"],
            )

        # start_region in end_regions (not allowed)
        with pytest.raises(ValueError, match=r"start_region.*cannot be in end_regions"):
            segment_trials(
                position_bins,
                times,
                env,
                start_region="start",
                end_regions=["start", "goal"],
            )

        # Negative min_duration
        with pytest.raises(ValueError, match=r"min_duration.*must be positive"):
            segment_trials(
                position_bins,
                times,
                env,
                start_region="start",
                end_regions=["goal"],
                min_duration=-1.0,
            )

        # max_duration < min_duration
        with pytest.raises(
            ValueError, match=r"max_duration.*must be greater than min_duration"
        ):
            segment_trials(
                position_bins,
                times,
                env,
                start_region="start",
                end_regions=["goal"],
                min_duration=10.0,
                max_duration=5.0,
            )

    def test_segment_trials_aborted_on_reentry_emitted(self):
        """Aborted trial (re-entry before max_duration) is emitted as failed.

        A trial that ran well under max_duration but was abandoned when the
        animal re-entered the start region must still be emitted with
        outcome="returned" and end_region=None (subject to min_duration), not
        silently dropped.
        """
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        region_builder.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))
        env = env.with_regions(region_builder)

        # Trial 1: enter start, leave toward middle (never reach goal),
        #          then re-enter start -> aborts trial 1.
        # Trial 2: from start, reach goal.
        x_traj = np.concatenate(
            [
                np.linspace(20, 20, 5),  # in start (trial 1 begins)
                np.linspace(20, 50, 8),  # leave start toward middle
                np.linspace(50, 20, 8),  # return to start (re-entry -> abort)
                np.linspace(20, 80, 20),  # trial 2: go to goal
            ]
        )
        y_traj = np.concatenate(
            [
                np.linspace(20, 20, 5),
                np.linspace(20, 50, 8),
                np.linspace(50, 20, 8),
                np.linspace(20, 80, 20),
            ]
        )
        trajectory = np.column_stack([x_traj, y_traj])
        position_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import segment_trials

        # max_duration large so trial 1 never times out; it ran ~20s < 50s.
        trials = segment_trials(
            position_bins,
            times,
            env,
            start_region="start",
            end_regions=["goal"],
            min_duration=2.0,
            max_duration=50.0,
            max_gap=None,
        )

        # Aborted trial 1 must be present as a failed trial, plus successful trial 2.
        assert len(trials) == 2, f"Expected aborted + successful trial, got {trials}"
        aborted = trials[0]
        assert aborted.outcome == "returned"
        assert aborted.success is False
        assert aborted.end_region is None
        assert aborted.start_region == "start"
        # Aborted trial ended at the re-entry sample, well under max_duration.
        assert (aborted.end_time - aborted.start_time) < 50.0
        # Trial 2 is the successful one.
        assert trials[1].outcome == "reached"
        assert trials[1].success is True
        assert trials[1].end_region == "goal"

    @staticmethod
    def _track_with_wells():
        """A 100 cm track with a start well at 0-10 cm and a goal at 90-100 cm."""
        from shapely.geometry import box

        env = Environment.linear_track(
            endpoints=[(0.0, 0.0), (100.0, 0.0)], bin_size=2.0, track_width=6.0
        )
        regions = Regions(env.regions.values())
        regions.add("start", polygon=box(-1.0, -3.0, 10.0, 3.0))
        regions.add("goal", polygon=box(90.0, -3.0, 101.0, 3.0))
        return env.with_regions(regions)

    @staticmethod
    def _dwell_then_run(dwell_x):
        """Dwell at ``dwell_x`` (30 Hz), then run to the goal and back, twice."""
        run = np.linspace(10.5, 95.0, 90)
        back = run[::-1]
        x = np.concatenate([dwell_x, run, np.full(30, 95.0), back, dwell_x, run])
        return np.column_stack([x, np.zeros_like(x)]), np.arange(len(x)) / 30.0

    def test_segment_trials_dwell_jitter_is_not_an_aborted_trial(self):
        """Waiting at the start well with tracking jitter across the region edge
        is one departure, not a failed trial per flicker."""
        from neurospatial.behavior.segmentation import segment_trials

        env = self._track_with_wells()
        # 2 s at the well, then jitter across its edge (10 cm) while leaving.
        dwell = np.concatenate([np.full(60, 5.0), np.repeat([10.5, 9.5, 10.5, 9.5], 5)])
        positions, times = self._dwell_then_run(dwell)
        trials = segment_trials(
            env.bin_sequence(times, positions, dedup=False),
            times,
            env,
            start_region="start",
            end_regions=["goal"],
        )
        assert [t.success for t in trials] == [True, True]

    def test_segment_trials_tracking_dropout_at_well_is_not_a_departure(self):
        """Lost tracking (bin -1) while the animal waits in the start region
        does not end the wait or start a trial."""
        from neurospatial.behavior.segmentation import segment_trials

        env = self._track_with_wells()
        positions, times = self._dwell_then_run(np.full(90, 5.0))
        bins = env.bin_sequence(times, positions, dedup=False)
        bins[30:75] = -1  # 1.5 s dropout during each 3 s wait
        second_wait = int(np.flatnonzero(np.diff(bins[200:] == bins[0]))[0]) + 201
        bins[second_wait + 30 : second_wait + 75] = -1
        trials = segment_trials(
            bins, times, env, start_region="start", end_regions=["goal"]
        )
        assert [t.success for t in trials] == [True, True]

    def test_segment_trials_starts_when_the_animal_leaves_the_start_region(self):
        """start_time is the last sample in the start region before departure,
        so trial duration is travel time, not time spent waiting at the well."""
        from neurospatial.behavior.segmentation import segment_trials

        env = self._track_with_wells()
        positions, times = self._dwell_then_run(np.full(90, 5.0))
        trials = segment_trials(
            env.bin_sequence(times, positions, dedup=False),
            times,
            env,
            start_region="start",
            end_regions=["goal"],
        )
        np.testing.assert_allclose(trials[0].start_time, 89 / 30.0)

    def test_segment_trials_aborted_on_reentry_respects_min_duration(self):
        """An aborted trial shorter than min_duration is still dropped."""
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        region_builder.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))
        env = env.with_regions(region_builder)

        # Very brief excursion (< min_duration) then re-entry, then a real trial.
        x_traj = np.concatenate(
            [
                np.linspace(20, 20, 2),  # in start (trial 1)
                np.linspace(20, 30, 2),  # brief leave
                np.linspace(30, 20, 2),  # re-enter (abort, but too short)
                np.linspace(20, 80, 20),  # trial 2 -> goal
            ]
        )
        y_traj = np.concatenate(
            [
                np.linspace(20, 20, 2),
                np.linspace(20, 30, 2),
                np.linspace(30, 20, 2),
                np.linspace(20, 80, 20),
            ]
        )
        trajectory = np.column_stack([x_traj, y_traj])
        position_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import segment_trials

        trials = segment_trials(
            position_bins,
            times,
            env,
            start_region="start",
            end_regions=["goal"],
            min_duration=10.0,
            max_duration=50.0,
            max_gap=None,
        )

        # Aborted trial too short to pass min_duration -> dropped.
        # Only the successful trial 2 survives (it spans >= 10s).
        assert len(trials) == 1
        assert trials[0].success is True
        assert trials[0].end_region == "goal"

    def test_segment_trials_in_progress_at_end_of_data_emitted(self):
        """In-progress trial at end of data is emitted as failed (not dropped).

        If recording ends while a trial is still in progress and it ran less
        than max_duration, the trial must still be emitted with success=False
        and end_region=None (subject to min_duration).
        """
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        region_builder.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))
        env = env.with_regions(region_builder)

        # Enter start, head toward goal but recording ends before reaching it.
        x_traj = np.concatenate(
            [
                np.linspace(20, 20, 5),  # in start
                np.linspace(20, 50, 15),  # heading out, never reach goal
            ]
        )
        y_traj = np.concatenate(
            [
                np.linspace(20, 20, 5),
                np.linspace(20, 50, 15),
            ]
        )
        trajectory = np.column_stack([x_traj, y_traj])
        position_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import segment_trials

        # Trial ran ~19s < max_duration 50s; must still be emitted as failed.
        trials = segment_trials(
            position_bins,
            times,
            env,
            start_region="start",
            end_regions=["goal"],
            min_duration=2.0,
            max_duration=50.0,
            max_gap=None,
        )

        assert len(trials) == 1
        # The recording ended first: the result is unknown, not a failure.
        assert trials[0].outcome == "censored"
        assert trials[0].success is False
        assert trials[0].end_region is None
        assert trials[0].start_region == "start"
        assert trials[0].end_time == times[-1]

    def test_segment_trials_in_progress_at_end_respects_min_duration(self):
        """In-progress trial shorter than min_duration at end is dropped."""
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        region_builder.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))
        env = env.with_regions(region_builder)

        # Wait in the start region, then leave it 3 s before the recording
        # ends: the trial in progress lasts 3 s from its start sample.
        x_traj = np.r_[np.full(10, 20.0), 40.0, 45.0, 50.0]
        y_traj = np.r_[np.full(10, 20.0), 40.0, 45.0, 50.0]
        trajectory = np.column_stack([x_traj, y_traj])
        position_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import segment_trials

        options = {
            "start_region": "start",
            "end_regions": ["goal"],
            "max_duration": 50.0,
            "max_gap": None,
        }
        long_enough = segment_trials(
            position_bins, times, env, min_duration=3.0, **options
        )
        too_short = segment_trials(
            position_bins, times, env, min_duration=3.5, **options
        )

        # The trial runs from the last sample in the start region (9 s) to
        # the end of the recording (12 s) and is censored there.
        assert [(t.start_time, t.end_time, t.outcome) for t in long_enough] == [
            (9.0, 12.0, "censored")
        ]
        assert too_short == []

    def test_segment_trials_multiple_starts(self):
        """Test handling of multiple trial starts without completion."""
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        region_builder.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))
        env = env.with_regions(region_builder)

        # Enter start region twice, reach goal after second entry
        # Start 1 → leave → Start 2 → Goal
        x_traj = np.concatenate(
            [
                np.linspace(20, 20, 5),  # In start region
                np.linspace(20, 40, 5),  # Leave start
                np.linspace(40, 20, 5),  # Return to start
                np.linspace(20, 80, 15),  # Go to goal
            ]
        )
        y_traj = np.concatenate(
            [
                np.linspace(20, 20, 5),
                np.linspace(20, 40, 5),
                np.linspace(40, 20, 5),
                np.linspace(20, 80, 15),
            ]
        )
        trajectory = np.column_stack([x_traj, y_traj])
        position_bins = env.bin_at(trajectory)
        times = np.arange(len(trajectory), dtype=float)

        from neurospatial.behavior.segmentation import segment_trials

        trials = segment_trials(
            position_bins,
            times,
            env,
            start_region="start",
            end_regions=["goal"],
            min_duration=1.0,
            max_duration=50.0,
            max_gap=None,
        )

        # The first departure comes back to the start (returned); the second
        # reaches the goal. The aborted trial ends at the re-entry sample,
        # where the next trial starts after a single sample inside.
        assert [t.outcome for t in trials] == ["returned", "reached"]
        assert trials[1].end_region == "goal"
        assert trials[0].end_time <= trials[1].start_time


class TestTrialRecord:
    """Trial stores how it ended; success is derived from the outcome."""

    def test_success_is_derived_from_outcome(self):
        from neurospatial.behavior.segmentation import Trial

        assert Trial(0.0, 1.0, "start", "left", "reached").success is True
        for outcome in ("timeout", "returned", "censored"):
            assert Trial(0.0, 1.0, "start", None, outcome).success is False
        with pytest.raises(TypeError):
            Trial(0.0, 1.0, "start", "left", success=True)  # type: ignore[call-arg]

    def test_scalars_are_plain_floats(self):
        from neurospatial.behavior.segmentation import Trial

        trial = Trial(np.float32(0.5), np.int64(2), "start", None, "timeout")
        assert type(trial.start_time) is float
        assert type(trial.end_time) is float

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"start_time": 2.0, "end_time": 1.0}, "start_time <= end_time"),
            ({"end_time": np.nan}, "start_time <= end_time"),
            ({"outcome": "success"}, "outcome must be one of"),
            ({"outcome": "timeout"}, "does not match"),
            ({"end_region": None}, "does not match"),
        ],
    )
    def test_invalid_records_raise(self, kwargs, match):
        from neurospatial.behavior.segmentation import Trial

        fields = {
            "start_time": 0.0,
            "end_time": 1.0,
            "start_region": "start",
            "end_region": "left",
            "outcome": "reached",
        }
        with pytest.raises(ValueError, match=match):
            Trial(**{**fields, **kwargs})


class TestTrialEnds:
    """Exact trial boundaries on a corridor sampled at bin centers."""

    @staticmethod
    def _corridor():
        from shapely.geometry import box

        xs = np.linspace(0.0, 100.0, 401)
        env = Environment.from_samples(np.c_[xs, np.zeros_like(xs)], bin_size=2.0)
        regions = Regions(env.regions.values())
        regions.add("start", polygon=box(-5.0, -5.0, 5.0, 5.0))  # bins 0-4 cm
        regions.add("goal", polygon=box(95.0, -5.0, 105.0, 5.0))  # bins 96-100 cm
        return env.with_regions(regions)

    def _trials(self, x, times, **options):
        from neurospatial.behavior.segmentation import segment_trials

        env = self._corridor()
        x = np.asarray(x, dtype=float)
        trials = segment_trials(
            env.bin_at(np.c_[x, np.zeros_like(x)]),
            np.asarray(times, dtype=float),
            env,
            start_region="start",
            end_regions=["goal"],
            max_gap=None,
            **options,
        )
        return [(t.start_time, t.end_time, t.end_region, t.outcome) for t in trials]

    def test_goal_entry_after_the_deadline_is_a_timeout(self):
        """Arrival 10.5 s after departure with max_duration=10 is a timeout.

        The trial starts at 1 s (last sample in the start region) and times
        out at 6 s, the last sample within the 11 s deadline. It used to be
        dropped because the goal-entry sample was past the deadline.
        """
        x = [0, 0, 30, 60, 98, 98]
        times = [0, 1, 2, 6, 11.5, 12]
        assert self._trials(x, times, min_duration=1.0, max_duration=10.0) == [
            (1.0, 6.0, None, "timeout")
        ]
        assert self._trials(x, times, min_duration=1.0, max_duration=11.0) == [
            (1.0, 11.5, "goal", "reached")
        ]

    def test_sample_on_the_deadline_ends_a_timeout(self):
        x = [0, 0, 20, 30, 40]
        times = [0, 1, 6, 11, 12]
        assert self._trials(x, times, min_duration=1.0, max_duration=10.0) == [
            (1.0, 11.0, None, "timeout")
        ]

    def test_returned_trial_ends_at_the_reentry_sample(self):
        x = [0, 0, 10, 20, 10, 4, 30, 60, 98]
        times = np.arange(len(x))
        assert self._trials(x, times, min_duration=1.0, max_duration=10.0) == [
            (1.0, 5.0, None, "returned"),
            (5.0, 8.0, "goal", "reached"),
        ]
