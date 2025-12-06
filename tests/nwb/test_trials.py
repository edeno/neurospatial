"""
Tests for NWB trial writing and reading functions.

Tests the write_trials() and read_trials() functions for persisting
Trial segmentation results to NWB files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# pynwb is required for all tests
pynwb = pytest.importorskip("pynwb")


class TestWriteTrialsFromTrialObjects:
    """Tests for write_trials() using list[Trial] input."""

    def test_basic_trial_writing(self, empty_nwb):
        """Test writing Trial objects to NWB file."""
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        trials = [
            Trial(
                start_time=0.0,
                end_time=5.0,
                start_region="home",
                end_region="goal",
                success=True,
            ),
            Trial(
                start_time=10.0,
                end_time=15.0,
                start_region="home",
                end_region=None,
                success=False,
            ),
        ]

        write_trials(nwbfile, trials)

        # Verify trials table was created
        assert nwbfile.trials is not None
        assert len(nwbfile.trials) == 2

    def test_trial_fields_written_correctly(self, empty_nwb):
        """Test that all Trial fields are written to NWB."""
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        trials = [
            Trial(
                start_time=1.5,
                end_time=6.2,
                start_region="start_zone",
                end_region="reward_left",
                success=True,
            ),
        ]

        write_trials(nwbfile, trials)

        # Check built-in columns
        assert nwbfile.trials["start_time"][0] == pytest.approx(1.5)
        assert nwbfile.trials["stop_time"][0] == pytest.approx(6.2)

        # Check custom columns
        assert "start_region" in nwbfile.trials.colnames
        assert "end_region" in nwbfile.trials.colnames
        assert "success" in nwbfile.trials.colnames

        assert nwbfile.trials["start_region"][0] == "start_zone"
        assert nwbfile.trials["end_region"][0] == "reward_left"
        assert nwbfile.trials["success"][0] is True

    def test_trial_with_none_end_region(self, empty_nwb):
        """Test that None end_region is handled correctly."""
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        trials = [
            Trial(
                start_time=0.0,
                end_time=15.0,
                start_region="home",
                end_region=None,  # Timeout trial
                success=False,
            ),
        ]

        write_trials(nwbfile, trials)

        # None should be stored as empty string or "None"
        end_region = nwbfile.trials["end_region"][0]
        assert end_region in ("", "None", None)

    def test_empty_trials_list(self, empty_nwb):
        """Test writing empty trials list.

        Note: NWB doesn't create trials table until at least one trial is added.
        Empty trials list results in no trials table.
        """
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb

        write_trials(nwbfile, trials=[])

        # NWB behavior: No trials table created when no trials added
        assert nwbfile.trials is None


class TestWriteTrialsFromArrays:
    """Tests for write_trials() using raw array input."""

    def test_basic_array_writing(self, empty_nwb):
        """Test writing trials from raw arrays."""
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        start_times = np.array([0.0, 10.0, 20.0])
        stop_times = np.array([5.0, 15.0, 25.0])

        write_trials(nwbfile, start_times=start_times, stop_times=stop_times)

        assert nwbfile.trials is not None
        assert len(nwbfile.trials) == 3

        np.testing.assert_array_almost_equal(
            nwbfile.trials["start_time"][:], start_times
        )
        np.testing.assert_array_almost_equal(nwbfile.trials["stop_time"][:], stop_times)

    def test_arrays_with_all_optional_columns(self, empty_nwb):
        """Test writing trials with all optional columns."""
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        start_times = np.array([0.0, 10.0])
        stop_times = np.array([5.0, 15.0])
        start_regions = ["home", "home"]
        end_regions = ["goal_left", "goal_right"]
        successes = [True, True]

        write_trials(
            nwbfile,
            start_times=start_times,
            stop_times=stop_times,
            start_regions=start_regions,
            end_regions=end_regions,
            successes=successes,
        )

        assert len(nwbfile.trials) == 2
        assert "start_region" in nwbfile.trials.colnames
        assert "end_region" in nwbfile.trials.colnames
        assert "success" in nwbfile.trials.colnames

    def test_arrays_with_partial_optional_columns(self, empty_nwb):
        """Test writing trials with only some optional columns."""
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        start_times = np.array([0.0, 10.0])
        stop_times = np.array([5.0, 15.0])
        start_regions = ["home", "home"]

        # Only start_regions provided, no end_regions or successes
        write_trials(
            nwbfile,
            start_times=start_times,
            stop_times=stop_times,
            start_regions=start_regions,
        )

        assert len(nwbfile.trials) == 2
        assert "start_region" in nwbfile.trials.colnames
        # These columns should NOT be present since not provided
        assert "end_region" not in nwbfile.trials.colnames
        assert "success" not in nwbfile.trials.colnames


class TestWriteTrialsValidation:
    """Tests for write_trials() input validation."""

    def test_mixed_args_error(self, empty_nwb):
        """Test error when both trials and arrays provided."""
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        trials = [
            Trial(
                start_time=0.0,
                end_time=5.0,
                start_region="home",
                end_region="goal",
                success=True,
            )
        ]
        start_times = np.array([0.0])
        stop_times = np.array([5.0])

        with pytest.raises(ValueError, match="Cannot specify both"):
            write_trials(
                nwbfile, trials=trials, start_times=start_times, stop_times=stop_times
            )

    def test_missing_required_arrays_error(self, empty_nwb):
        """Test error when required arrays are missing."""
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb

        # Missing stop_times
        with pytest.raises(ValueError, match="stop_times"):
            write_trials(nwbfile, start_times=np.array([0.0]))

        # Missing start_times
        with pytest.raises(ValueError, match="start_times"):
            write_trials(nwbfile, stop_times=np.array([5.0]))

    def test_length_mismatch_error(self, empty_nwb):
        """Test error when array lengths don't match."""
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        start_times = np.array([0.0, 10.0, 20.0])
        stop_times = np.array([5.0, 15.0])  # Wrong length

        with pytest.raises(ValueError, match="length"):
            write_trials(nwbfile, start_times=start_times, stop_times=stop_times)

    def test_optional_array_length_mismatch_error(self, empty_nwb):
        """Test error when optional arrays have wrong length."""
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        start_times = np.array([0.0, 10.0])
        stop_times = np.array([5.0, 15.0])
        start_regions = ["home"]  # Wrong length

        with pytest.raises(ValueError, match="length"):
            write_trials(
                nwbfile,
                start_times=start_times,
                stop_times=stop_times,
                start_regions=start_regions,
            )

    def test_stop_before_start_error(self, empty_nwb):
        """Test error when stop_time < start_time."""
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        start_times = np.array([10.0])
        stop_times = np.array([5.0])  # Before start

        with pytest.raises(ValueError, match=r"stop_time.*start_time"):
            write_trials(nwbfile, start_times=start_times, stop_times=stop_times)

    def test_nan_timestamps_error(self, empty_nwb):
        """Test error when timestamps contain NaN."""
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        start_times = np.array([0.0, np.nan])
        stop_times = np.array([5.0, 15.0])

        with pytest.raises(ValueError, match=r"non-finite|NaN"):
            write_trials(nwbfile, start_times=start_times, stop_times=stop_times)

    def test_negative_timestamps_error(self, empty_nwb):
        """Test error when timestamps are negative."""
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        start_times = np.array([-1.0, 10.0])
        stop_times = np.array([5.0, 15.0])

        with pytest.raises(ValueError, match="negative"):
            write_trials(nwbfile, start_times=start_times, stop_times=stop_times)

    def test_no_input_error(self, empty_nwb):
        """Test error when no input provided."""
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb

        with pytest.raises(ValueError, match="Must provide"):
            write_trials(nwbfile)


class TestWriteTrialsOverwrite:
    """Tests for write_trials() overwrite behavior."""

    def test_overwrite_false_error(self, empty_nwb):
        """Test error when trials exist and overwrite=False."""
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        trials = [
            Trial(
                start_time=0.0,
                end_time=5.0,
                start_region="home",
                end_region="goal",
                success=True,
            )
        ]

        # Write first time
        write_trials(nwbfile, trials)

        # Try to write again without overwrite
        with pytest.raises(ValueError, match=r"already exists|overwrite"):
            write_trials(nwbfile, trials)

    def test_overwrite_true_replaces(self, empty_nwb):
        """Test that overwrite=True replaces existing trials."""
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        original_trials = [
            Trial(
                start_time=0.0,
                end_time=5.0,
                start_region="home",
                end_region="goal",
                success=True,
            )
        ]
        new_trials = [
            Trial(
                start_time=10.0,
                end_time=15.0,
                start_region="start",
                end_region="reward",
                success=True,
            ),
            Trial(
                start_time=20.0,
                end_time=25.0,
                start_region="start",
                end_region="reward",
                success=True,
            ),
        ]

        # Write original
        write_trials(nwbfile, original_trials)
        assert len(nwbfile.trials) == 1

        # Overwrite with new
        write_trials(nwbfile, new_trials, overwrite=True)
        assert len(nwbfile.trials) == 2
        assert nwbfile.trials["start_time"][0] == pytest.approx(10.0)


class TestReadTrials:
    """Tests for read_trials() function."""

    def test_basic_reading(self, empty_nwb):
        """Test basic trial reading."""
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.io.nwb import read_trials, write_trials

        nwbfile = empty_nwb
        trials = [
            Trial(
                start_time=0.0,
                end_time=5.0,
                start_region="home",
                end_region="goal",
                success=True,
            ),
            Trial(
                start_time=10.0,
                end_time=15.0,
                start_region="home",
                end_region="goal",
                success=True,
            ),
        ]

        write_trials(nwbfile, trials)
        result = read_trials(nwbfile)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "start_time" in result.columns
        assert "stop_time" in result.columns

    def test_read_trials_not_found(self, empty_nwb):
        """Test KeyError when no trials table exists."""
        from neurospatial.io.nwb import read_trials

        nwbfile = empty_nwb

        with pytest.raises(KeyError, match="trials"):
            read_trials(nwbfile)

    def test_read_trials_with_custom_columns(self, empty_nwb):
        """Test reading trials with custom columns (start/end regions)."""
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.io.nwb import read_trials, write_trials

        nwbfile = empty_nwb
        trials = [
            Trial(
                start_time=0.0,
                end_time=5.0,
                start_region="home",
                end_region="goal_left",
                success=True,
            ),
            Trial(
                start_time=10.0,
                end_time=20.0,
                start_region="home",
                end_region=None,
                success=False,
            ),
        ]

        write_trials(nwbfile, trials)
        result = read_trials(nwbfile)

        # Check custom columns present
        assert "start_region" in result.columns
        assert "end_region" in result.columns
        assert "success" in result.columns

        # Check values
        assert result["start_region"].iloc[0] == "home"
        assert result["end_region"].iloc[0] == "goal_left"
        assert result["success"].iloc[0] == True  # noqa: E712

    def test_roundtrip_data_integrity(self, empty_nwb):
        """Test data integrity through write/read round-trip."""
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.io.nwb import read_trials, write_trials

        nwbfile = empty_nwb
        trials = [
            Trial(
                start_time=1.5,
                end_time=6.2,
                start_region="start_zone",
                end_region="reward_right",
                success=True,
            ),
            Trial(
                start_time=10.0,
                end_time=18.5,
                start_region="start_zone",
                end_region="reward_left",
                success=True,
            ),
            Trial(
                start_time=25.0,
                end_time=40.0,
                start_region="start_zone",
                end_region=None,
                success=False,
            ),
        ]

        write_trials(nwbfile, trials)
        result = read_trials(nwbfile)

        # Verify data matches
        assert len(result) == 3

        # Check timestamps
        np.testing.assert_array_almost_equal(
            result["start_time"].values, [1.5, 10.0, 25.0]
        )
        np.testing.assert_array_almost_equal(
            result["stop_time"].values, [6.2, 18.5, 40.0]
        )

        # Check start regions
        assert list(result["start_region"]) == ["start_zone"] * 3

        # Check end regions (None stored as empty string or "None")
        assert result["end_region"].iloc[0] == "reward_right"
        assert result["end_region"].iloc[1] == "reward_left"

        # Check success (use == for numpy bool comparison)
        assert result["success"].iloc[0] == True  # noqa: E712
        assert result["success"].iloc[1] == True  # noqa: E712
        assert result["success"].iloc[2] == False  # noqa: E712


class TestWriteTrialsDescription:
    """Tests for write_trials() description parameter.

    Note: NWB doesn't allow changing the description after the trials table
    is created. Custom descriptions only work with overwrite=True, where we
    create a fresh TimeIntervals with the custom description.
    """

    def test_custom_description_with_overwrite(self, empty_nwb):
        """Test custom description with overwrite mode.

        Custom descriptions only work when using overwrite=True because
        NWB creates the trials table with a fixed description on first add_trial.
        """
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        # First write some trials
        initial_trials = [
            Trial(
                start_time=0.0,
                end_time=5.0,
                start_region="home",
                end_region="goal",
                success=True,
            )
        ]
        write_trials(nwbfile, initial_trials)

        # Now overwrite with custom description
        new_trials = [
            Trial(
                start_time=10.0,
                end_time=15.0,
                start_region="home",
                end_region="goal",
                success=True,
            )
        ]
        write_trials(
            nwbfile, new_trials, description="T-maze behavioral trials", overwrite=True
        )

        assert nwbfile.trials.description == "T-maze behavioral trials"

    def test_default_description(self, empty_nwb):
        """Test default description is used."""
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        trials = [
            Trial(
                start_time=0.0,
                end_time=5.0,
                start_region="home",
                end_region="goal",
                success=True,
            )
        ]

        write_trials(nwbfile, trials)

        # NWB creates trials table with its own default description
        assert nwbfile.trials.description is not None
