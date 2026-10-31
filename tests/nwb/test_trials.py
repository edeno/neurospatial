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

from ._nwb_helpers import create_roundtrip_nwb  # noqa: E402


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
                outcome="reached",
            ),
            Trial(
                start_time=10.0,
                end_time=15.0,
                start_region="home",
                end_region=None,
                outcome="timeout",
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
                outcome="reached",
            ),
        ]

        write_trials(nwbfile, trials)

        # Check built-in columns
        assert nwbfile.trials["start_time"][0] == pytest.approx(1.5)
        assert nwbfile.trials["stop_time"][0] == pytest.approx(6.2)

        # Check custom columns
        assert "start_region" in nwbfile.trials.colnames
        assert "end_region" in nwbfile.trials.colnames
        assert "outcome" in nwbfile.trials.colnames

        assert nwbfile.trials["start_region"][0] == "start_zone"
        assert nwbfile.trials["end_region"][0] == "reward_left"
        assert nwbfile.trials["outcome"][0] == "reached"

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
                outcome="timeout",
            ),
        ]

        write_trials(nwbfile, trials)

        # None should be stored as empty string or "None"
        end_region = nwbfile.trials["end_region"][0]
        assert end_region in ("", "None", None)

    def test_zero_trials_write_an_empty_table(self, tmp_path):
        """A session without trials reads back as an empty table, not a KeyError."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_trials, write_trials

        nwbfile = create_roundtrip_nwb()
        write_trials(nwbfile, trials=[])
        with NWBHDF5IO(str(tmp_path / "trials.nwb"), "w") as io:
            io.write(nwbfile)

        with NWBHDF5IO(str(tmp_path / "trials.nwb"), "r") as io:
            recovered = read_trials(io.read())
        assert len(recovered) == 0
        assert set(recovered.columns) == {
            "start_time",
            "stop_time",
            "start_region",
            "end_region",
            "outcome",
        }

    def test_zero_trials_from_arrays_write_an_empty_table(self, empty_nwb):
        from neurospatial.io.nwb import read_trials, write_trials

        write_trials(empty_nwb, start_times=[], stop_times=[])

        recovered = read_trials(empty_nwb)
        assert len(recovered) == 0
        assert set(recovered.columns) == {"start_time", "stop_time"}

    @pytest.mark.parametrize(
        ("trials", "message"),
        [
            (
                np.array([0.0, 5.0]),
                "trials must be a list of Trial objects; got ndarray",
            ),
            (
                [{"start_time": 0.0, "end_time": 5.0}],
                r"trials\[0\] is a dict, not a Trial",
            ),
        ],
    )
    def test_non_trial_records_are_refused(self, empty_nwb, trials, message):
        from neurospatial.io.nwb import write_trials

        with pytest.raises(TypeError, match=message):
            write_trials(empty_nwb, trials)
        assert empty_nwb.trials is None


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
        outcomes = ["reached", "censored"]

        write_trials(
            nwbfile,
            start_times=start_times,
            stop_times=stop_times,
            start_regions=start_regions,
            end_regions=end_regions,
            outcomes=outcomes,
        )

        assert len(nwbfile.trials) == 2
        assert "start_region" in nwbfile.trials.colnames
        assert "end_region" in nwbfile.trials.colnames
        assert list(nwbfile.trials["outcome"][:]) == outcomes

    def test_arrays_reject_an_unknown_outcome(self, empty_nwb):
        """A boolean-era value is not a trial outcome."""
        from neurospatial.io.nwb import write_trials

        with pytest.raises(ValueError, match="not a trial outcome"):
            write_trials(
                empty_nwb,
                start_times=np.array([0.0]),
                stop_times=np.array([5.0]),
                outcomes=["success"],
            )

    def test_arrays_with_partial_optional_columns(self, empty_nwb):
        """Test writing trials with only some optional columns."""
        from neurospatial.io.nwb import write_trials

        nwbfile = empty_nwb
        start_times = np.array([0.0, 10.0])
        stop_times = np.array([5.0, 15.0])
        start_regions = ["home", "home"]

        # Only start_regions provided, no end_regions or outcomes
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
        assert "outcome" not in nwbfile.trials.colnames


class TestWriteTrialsValidation:
    """Tests for write_trials() input validation."""

    @pytest.mark.parametrize(
        ("arrays", "names"),
        [
            (
                {"start_times": np.array([0.0]), "stop_times": np.array([5.0])},
                "start_times, stop_times",
            ),
            ({"start_regions": ["home"]}, "start_regions"),
            ({"end_regions": ["goal"]}, "end_regions"),
            ({"outcomes": ["reached"]}, "outcomes"),
        ],
    )
    def test_trials_with_arrays_are_refused(self, empty_nwb, arrays, names):
        """Arrays passed with Trial objects would be ignored, so they raise."""
        from neurospatial.behavior.segmentation import Trial
        from neurospatial.io.nwb import write_trials

        trials = [
            Trial(
                start_time=0.0,
                end_time=5.0,
                start_region="home",
                end_region="goal",
                outcome="reached",
            )
        ]

        with pytest.raises(ValueError, match=f"Cannot combine 'trials' with {names}"):
            write_trials(empty_nwb, trials, **arrays)
        assert empty_nwb.trials is None

    @pytest.mark.parametrize(
        "arrays",
        [{}, {"start_times": np.array([0.0])}, {"stop_times": np.array([5.0])}],
    )
    def test_needs_trials_or_both_time_arrays(self, empty_nwb, arrays):
        from neurospatial.io.nwb import write_trials

        with pytest.raises(ValueError, match="needs either 'trials'"):
            write_trials(empty_nwb, **arrays)

    @pytest.mark.parametrize(
        ("column", "message"),
        [
            (
                {"stop_times": np.array([5.0, 15.0])},
                r"stop_times length \(2\) must match start_times length \(3\)",
            ),
            (
                {"start_regions": ["home"]},
                r"start_regions length \(1\) must match start_times length \(3\)",
            ),
            (
                {"end_regions": ["goal"]},
                r"end_regions length \(1\) must match start_times length \(3\)",
            ),
            (
                {"outcomes": ["reached"]},
                r"outcomes length \(1\) must match start_times length \(3\)",
            ),
        ],
    )
    def test_length_mismatch_names_the_argument(self, empty_nwb, column, message):
        from neurospatial.io.nwb import write_trials

        arrays = {
            "start_times": np.array([0.0, 10.0, 20.0]),
            "stop_times": np.array([5.0, 15.0, 25.0]),
            **column,
        }
        with pytest.raises(ValueError, match=message):
            write_trials(empty_nwb, **arrays)

    def test_stop_before_start_error(self, empty_nwb):
        """Test error when stop_time < start_time."""
        from neurospatial.io.nwb import write_trials

        with pytest.raises(
            ValueError,
            match=r"every trial. Trial 1: start_time=10.0, stop_time=5.0",
        ):
            write_trials(
                empty_nwb,
                start_times=np.array([0.0, 10.0]),
                stop_times=np.array([1.0, 5.0]),
            )

    @pytest.mark.parametrize(
        ("start", "stop", "message"),
        [
            ([0.0, np.nan], [5.0, 15.0], "start_times contains non-finite"),
            ([0.0, 10.0], [5.0, np.nan], "stop_times contains non-finite"),
            ([0.0, 10.0], [np.inf, 15.0], "stop_times contains non-finite"),
            ([-1.0, 10.0], [5.0, 15.0], "start_times contains negative"),
        ],
    )
    def test_invalid_times_name_the_argument(self, empty_nwb, start, stop, message):
        from neurospatial.io.nwb import write_trials

        with pytest.raises(ValueError, match=message):
            write_trials(empty_nwb, start_times=start, stop_times=stop)
        assert empty_nwb.trials is None

    def test_negative_stop_times_are_refused(self, empty_nwb):
        """A negative stop time is reported for stop_times, not start_times."""
        from neurospatial.io.nwb import write_trials

        with pytest.raises(ValueError, match="stop_times contains negative"):
            write_trials(empty_nwb, start_times=[0.0, 10.0], stop_times=[5.0, -1.0])

    def test_arrays_store_missing_end_region_as_empty_string(self, empty_nwb):
        from neurospatial.io.nwb import read_trials, write_trials

        write_trials(
            empty_nwb,
            start_times=[0.0, 10.0],
            stop_times=[5.0, 15.0],
            end_regions=["goal", None],
        )

        assert list(read_trials(empty_nwb)["end_region"]) == ["goal", ""]


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
                outcome="reached",
            )
        ]

        # Write first time
        write_trials(nwbfile, trials)

        # Try to write again without overwrite
        with pytest.raises(ValueError, match="Pass overwrite=True to replace it"):
            write_trials(nwbfile, trials)

    @staticmethod
    def _trials(start: float, n: int = 1):
        from neurospatial.behavior.segmentation import Trial

        return [
            Trial(
                start_time=start + 10.0 * i,
                end_time=start + 10.0 * i + 5.0,
                start_region="home",
                end_region="goal",
                outcome="reached",
            )
            for i in range(n)
        ]

    def test_overwrite_replaces_trials_in_memory(self, empty_nwb, tmp_path):
        """An in-memory file's trials are replaced, and only the new trials
        reach the file on disk."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_trials, write_trials

        write_trials(empty_nwb, self._trials(0.0, n=2))
        write_trials(
            empty_nwb, self._trials(100.0), description="redone", overwrite=True
        )

        np.testing.assert_array_equal(read_trials(empty_nwb)["start_time"], [100.0])
        path = tmp_path / "trials.nwb"
        with NWBHDF5IO(str(path), "w") as io:
            io.write(empty_nwb)
        with NWBHDF5IO(str(path), "r") as io:
            nwbfile = io.read()
            back = read_trials(nwbfile)
            assert nwbfile.trials.description == "redone"
        np.testing.assert_array_equal(back["start_time"], [100.0])
        np.testing.assert_array_equal(back["stop_time"], [105.0])
        assert list(back["outcome"]) == ["reached"]

    def test_invalid_replacement_keeps_the_original(self, empty_nwb):
        from neurospatial.io.nwb import read_trials, write_trials

        write_trials(empty_nwb, self._trials(0.0))
        with pytest.raises(ValueError, match="stop_times contains negative"):
            write_trials(
                empty_nwb,
                start_times=np.array([1.0]),
                stop_times=np.array([-1.0]),
                overwrite=True,
            )
        np.testing.assert_array_equal(read_trials(empty_nwb)["start_time"], [0.0])

    def test_replaced_table_is_detached_from_the_file(self, empty_nwb):
        from neurospatial.io.nwb import write_trials

        write_trials(empty_nwb, self._trials(0.0))
        old = empty_nwb.trials
        write_trials(empty_nwb, self._trials(100.0), overwrite=True)

        assert old.parent is None
        assert old not in empty_nwb.children
        for objects in (empty_nwb.all_children(), empty_nwb.objects.values()):
            assert [obj for obj in objects if obj.name == "trials"] == [
                empty_nwb.trials
            ]

    def test_trials_referenced_by_another_table_are_kept(self, empty_nwb):
        """Replacing referenced trials would leave a dangling reference that
        only fails at io.write, so it is refused up front."""
        from hdmf.common import DynamicTable, DynamicTableRegion

        from neurospatial.io.nwb import read_trials, write_trials

        write_trials(empty_nwb, self._trials(0.0, n=2))
        selection = DynamicTable(
            name="rewarded_trials",
            description="trials that were rewarded",
            columns=[
                DynamicTableRegion(
                    name="trial",
                    description="trial row",
                    data=[1],
                    table=empty_nwb.trials,
                )
            ],
        )
        empty_nwb.add_scratch(selection)

        with pytest.raises(
            ValueError, match="column 'trial' of 'rewarded_trials' refers to its rows"
        ):
            write_trials(empty_nwb, self._trials(100.0), overwrite=True)
        np.testing.assert_array_equal(read_trials(empty_nwb)["start_time"], [0.0, 10.0])

    @pytest.mark.parametrize("mode", ["r", "r+", "a"])
    def test_overwrite_in_a_file_read_from_disk_is_refused(
        self, empty_nwb, tmp_path, mode
    ):
        """Export would silently keep the written trials, so even mode 'r'
        is refused."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_trials, write_trials

        write_trials(empty_nwb, self._trials(0.0))
        path = tmp_path / "trials.nwb"
        with NWBHDF5IO(str(path), "w") as io:
            io.write(empty_nwb)

        with NWBHDF5IO(str(path), mode) as io:
            nwbfile = io.read()
            with pytest.raises(
                NotImplementedError,
                match="Cannot replace the trials table of a file read from disk",
            ):
                write_trials(nwbfile, self._trials(100.0), overwrite=True)
            np.testing.assert_array_equal(read_trials(nwbfile)["start_time"], [0.0])


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
                outcome="reached",
            ),
            Trial(
                start_time=10.0,
                end_time=15.0,
                start_region="home",
                end_region="goal",
                outcome="reached",
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

        with pytest.raises(KeyError, match="TimeIntervals 'trials' not found"):
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
                outcome="reached",
            ),
            Trial(
                start_time=10.0,
                end_time=20.0,
                start_region="home",
                end_region=None,
                outcome="timeout",
            ),
        ]

        write_trials(nwbfile, trials)
        result = read_trials(nwbfile)

        # Check custom columns present
        assert "start_region" in result.columns
        assert "end_region" in result.columns
        assert "outcome" in result.columns

        # Check values
        assert result["start_region"].iloc[0] == "home"
        assert result["end_region"].iloc[0] == "goal_left"
        assert result["outcome"].iloc[0] == "reached"

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
                outcome="reached",
            ),
            Trial(
                start_time=10.0,
                end_time=18.5,
                start_region="start_zone",
                end_region="reward_left",
                outcome="reached",
            ),
            Trial(
                start_time=25.0,
                end_time=40.0,
                start_region="start_zone",
                end_region=None,
                outcome="censored",
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

        # A censored trial survives the round trip as censored, not as failed.
        assert list(result["outcome"]) == ["reached", "reached", "censored"]


class TestWriteTrialsDescription:
    """The description argument describes the trials table."""

    def test_custom_description_survives_a_file(self, tmp_path):
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import write_trials

        nwbfile = create_roundtrip_nwb()
        write_trials(
            nwbfile,
            start_times=[0.0],
            stop_times=[5.0],
            description="T-maze behavioral trials",
        )
        with NWBHDF5IO(str(tmp_path / "trials.nwb"), "w") as io:
            io.write(nwbfile)

        with NWBHDF5IO(str(tmp_path / "trials.nwb"), "r") as io:
            assert io.read().trials.description == "T-maze behavioral trials"

    def test_default_description(self, empty_nwb):
        from neurospatial.io.nwb import write_trials

        write_trials(empty_nwb, start_times=[0.0], stop_times=[5.0])

        assert empty_nwb.trials.description == "Behavioral trials"


@pytest.mark.slow
class TestWriteTrialsDiskRoundTrip:
    """write_trials survives a real NWBHDF5IO write -> close -> reopen cycle.

    The existing write_trials tests check only the live in-memory NWBFile.
    These tests serialize the built-in TimeIntervals trials table to an HDF5
    file on disk and reload it via read_trials, catching pynwb serialization
    regressions the in-memory tests would miss.
    """

    def test_trials_survive_disk_roundtrip(self, tmp_path):
        """Trial intervals and all metadata columns survive disk round-trip."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_trials, write_trials

        start_times = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        stop_times = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
        start_regions = ["home", "home", "home", "home", "home"]
        # Every outcome, including censored, must survive the round trip.
        end_regions = ["goal_l", "", "goal_l", "", ""]
        outcomes = ["reached", "timeout", "reached", "censored", "returned"]

        nwbfile = create_roundtrip_nwb()
        write_trials(
            nwbfile,
            start_times=start_times,
            stop_times=stop_times,
            start_regions=start_regions,
            end_regions=end_regions,
            outcomes=outcomes,
        )

        nwb_path = tmp_path / "trials.nwb"
        with NWBHDF5IO(str(nwb_path), "w") as io:
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            reopened = io.read()
            recovered = read_trials(reopened)

        expected = pd.DataFrame(
            {
                "start_time": start_times,
                "stop_time": stop_times,
                "start_region": start_regions,
                "end_region": end_regions,
                "outcome": outcomes,
            }
        )
        pd.testing.assert_frame_equal(
            recovered.reset_index(drop=True)[list(expected.columns)],
            expected,
            check_dtype=False,
        )

    def test_trials_with_optional_columns_survives(self, tmp_path):
        """Trials written with only some optional columns reload as expected.

        Only start_region is supplied; end_region and outcome are never added
        as columns, so the recovered table must contain neither.
        """
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_trials, write_trials

        start_times = np.array([0.0, 10.0, 20.0])
        stop_times = np.array([5.0, 15.0, 25.0])
        start_regions = ["home", "home", "home"]

        nwbfile = create_roundtrip_nwb()
        write_trials(
            nwbfile,
            start_times=start_times,
            stop_times=stop_times,
            start_regions=start_regions,
        )

        nwb_path = tmp_path / "trials_partial.nwb"
        with NWBHDF5IO(str(nwb_path), "w") as io:
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            reopened = io.read()
            recovered = read_trials(reopened)

        # Populated columns survive.
        expected = pd.DataFrame(
            {
                "start_time": start_times,
                "stop_time": stop_times,
                "start_region": start_regions,
            }
        )
        pd.testing.assert_frame_equal(
            recovered.reset_index(drop=True)[list(expected.columns)],
            expected,
            check_dtype=False,
        )
        # Unsupplied optional columns are absent, not silently fabricated.
        assert "end_region" not in recovered.columns
        assert "outcome" not in recovered.columns
