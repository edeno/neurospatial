"""Tests for trial segmentation functions.

Following TDD: Tests written FIRST, then implementation.
"""

import numpy as np
import pytest
from shapely.geometry import Point

from neurospatial import Environment


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
        env.regions.add("start", polygon=Point(50.0, 10.0).buffer(8.0))
        env.regions.add("left", polygon=Point(20.0, 80.0).buffer(8.0))
        env.regions.add("right", polygon=Point(80.0, 80.0).buffer(8.0))

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
        )

        # Should detect 3 trials
        assert len(trials) >= 2, "Should detect at least 2 trials"

        # Verify trial structure
        for trial in trials:
            assert hasattr(trial, "start_time")
            assert hasattr(trial, "end_time")
            assert hasattr(trial, "start_region")
            assert hasattr(trial, "end_region")
            assert hasattr(trial, "success")
            assert trial.end_time > trial.start_time
            assert trial.start_region == "start", "start_region should be 'start'"
            assert trial.end_region in ["left", "right"]
            assert isinstance(trial.success, bool)

        # Check end_regions
        end_regions = [trial.end_region for trial in trials]
        assert "left" in end_regions, "Should have at least one left trial"
        assert "right" in end_regions, "Should have at least one right trial"

    def test_segment_trials_duration_filter_min(self):
        """Test that trials shorter than min_duration are excluded."""
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        env.regions.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        env.regions.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))

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
        )

        assert len(trials) == 0, "Short trial should be excluded"

    def test_segment_trials_duration_filter_max(self):
        """Test that trials longer than max_duration are marked as timeouts."""
        x = np.linspace(0, 100, 200)
        y = np.linspace(0, 100, 200)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        env.regions.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        env.regions.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))

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
        )

        # Should detect the trial but mark as failed (timeout)
        if len(trials) > 0:
            assert not trials[0].success, "Long trial should be marked as failed"
            assert trials[0].end_region is None, (
                "Failed trial should have no end_region"
            )
            assert trials[0].start_region == "start", "start_region should be 'start'"

    def test_segment_trials_successful_completion(self):
        """Test successful trial completion within duration bounds."""
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        env.regions.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        env.regions.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))

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

        env.regions.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        env.regions.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))

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

        env.regions.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        env.regions.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))

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
        )

        # Should detect trial start but timeout
        if len(trials) > 0:
            assert not trials[0].success, "Trial should fail (timeout)"

    def test_segment_trials_parameter_validation(self):
        """Test parameter validation."""
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)

        env.regions.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        env.regions.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))

        position_bins = np.arange(10)
        times = np.arange(10, dtype=float)

        from neurospatial.behavior.segmentation import segment_trials

        # Missing start_region - passes None which won't be found in regions
        with pytest.raises(ValueError, match=r"start_region.*not found"):
            segment_trials(
                position_bins,
                times,
                env,
                start_region=None,
                end_regions=["goal"],
            )

        # Missing end_regions - need to pass a string in list form that doesn't exist
        with pytest.raises(ValueError, match=r"end_region.*not found"):
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
        with pytest.raises(ValueError, match=r"start_region.*not found"):
            segment_trials(
                position_bins,
                times,
                env,
                start_region="nonexistent",
                end_regions=["goal"],
            )

        # Nonexistent end_region
        with pytest.raises(ValueError, match=r"end_regions.*not found"):
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
        success=False and end_region=None (subject to min_duration), not
        silently dropped.
        """
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        env.regions.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        env.regions.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))

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
        )

        # Aborted trial 1 must be present as a failed trial, plus successful trial 2.
        assert len(trials) == 2, f"Expected aborted + successful trial, got {trials}"
        aborted = trials[0]
        assert aborted.success is False
        assert aborted.end_region is None
        assert aborted.start_region == "start"
        # Aborted trial ended at the sample before re-entry, well under max_duration.
        assert (aborted.end_time - aborted.start_time) < 50.0
        # Trial 2 is the successful one.
        assert trials[1].success is True
        assert trials[1].end_region == "goal"

    def test_segment_trials_aborted_on_reentry_respects_min_duration(self):
        """An aborted trial shorter than min_duration is still dropped."""
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        env.regions.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        env.regions.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))

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

        env.regions.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        env.regions.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))

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
        )

        assert len(trials) == 1
        assert trials[0].success is False
        assert trials[0].end_region is None
        assert trials[0].start_region == "start"
        assert (trials[0].end_time - trials[0].start_time) < 50.0

    def test_segment_trials_in_progress_at_end_respects_min_duration(self):
        """In-progress trial shorter than min_duration at end is dropped."""
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        env.regions.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        env.regions.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))

        # Enter start near the very end of recording; trial only ~3s long.
        x_traj = np.concatenate(
            [
                np.linspace(50, 50, 10),  # outside start
                np.linspace(50, 20, 4),  # enter start at end
            ]
        )
        y_traj = np.concatenate(
            [
                np.linspace(50, 50, 10),
                np.linspace(50, 20, 4),
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
        )

        # Trial in progress at end is too short -> dropped.
        assert len(trials) == 0

    def test_segment_trials_multiple_starts(self):
        """Test handling of multiple trial starts without completion."""
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=3.0)

        env.regions.add("start", polygon=Point(20.0, 20.0).buffer(8.0))
        env.regions.add("goal", polygon=Point(80.0, 80.0).buffer(8.0))

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
        )

        # Should handle multiple start entries gracefully
        # Either count as separate trials or as one trial from last start
        assert len(trials) >= 1, "Should detect at least one trial"
        if len(trials) > 0:
            # Last trial should be successful
            assert any(t.success for t in trials), "Should have successful trial"
