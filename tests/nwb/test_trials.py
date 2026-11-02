"""Tests for write_trials() and read_trials(): Intervals to and from NWB trials."""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

# pynwb is required for all tests
pynwb = pytest.importorskip("pynwb")

from neurospatial import Interval, Intervals  # noqa: E402
from neurospatial.behavior.segmentation import Trial  # noqa: E402

from ._nwb_helpers import create_roundtrip_nwb  # noqa: E402


def _trials(start: float = 0.0, n: int = 1) -> Intervals[Trial]:
    return Intervals(
        [
            Trial(start + 10.0 * i, start + 10.0 * i + 5.0, "home", "goal", "reached")
            for i in range(n)
        ]
    )


@pytest.fixture
def detected() -> Intervals[Trial]:
    """Trials of every outcome, with non-contiguous ids out of time order."""
    return Intervals(
        [
            Trial(25.0, 40.0, "start_zone", None, "censored"),
            Trial(1.5, 6.2, "start_zone", "reward_right", "reached"),
            Trial(10.0, 18.5, "start_zone", "reward_left", "reached"),
            Trial(19.0, 24.0, "start_zone", None, "returned"),
        ],
        interval_ids=[8, 2, 5, 6],
    )


def _write_and_read(nwbfile, tmp_path, name: str = "trials.nwb"):
    from pynwb import NWBHDF5IO

    from neurospatial.io.nwb import read_trials

    path = tmp_path / name
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    with NWBHDF5IO(str(path), "r") as io:
        reopened = io.read()
        return read_trials(reopened), reopened.trials.description


class TestWriteTrials:
    def test_fields_are_columns_and_ids_an_interval_id_column(
        self, empty_nwb, detected
    ):
        from neurospatial.io.nwb import write_trials

        write_trials(empty_nwb, detected)

        table = empty_nwb.trials.to_dataframe()
        # The table id is the row number every NWB reader expects; the
        # stable ids are their own column.
        assert table.index.tolist() == [0, 1, 2, 3]
        assert table["interval_id"].tolist() == [2, 5, 6, 8]
        assert list(table.columns) == [
            "start_time",
            "stop_time",
            "interval_id",
            "start_region",
            "end_region",
            "outcome",
        ]
        np.testing.assert_array_equal(table["start_time"], [1.5, 10.0, 19.0, 25.0])
        # NWB text columns cannot hold None, so a missing region is "".
        assert table["end_region"].tolist() == ["reward_right", "reward_left", "", ""]

    def test_zero_trials_write_nothing(self, tmp_path, detected):
        """NWB files should not hold empty tables, so no trials table is added."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_trials, write_trials

        nwbfile = create_roundtrip_nwb()
        write_trials(nwbfile, detected[[]])
        assert nwbfile.trials is None
        with NWBHDF5IO(str(tmp_path / "trials.nwb"), "w") as io:
            io.write(nwbfile)
        with (
            NWBHDF5IO(str(tmp_path / "trials.nwb"), "r") as io,
            pytest.raises(KeyError, match="'trials' not found"),
        ):
            read_trials(io.read())

    def test_list_of_records_is_refused(self, empty_nwb, detected):
        from neurospatial.io.nwb import write_trials

        with pytest.raises(
            TypeError, match=r"wrap records with neurospatial\.Intervals"
        ):
            write_trials(empty_nwb, list(detected))
        assert empty_nwb.trials is None

    def test_field_without_an_nwb_column_type_is_refused(self, empty_nwb):
        from neurospatial.io.nwb import write_trials

        trials = Intervals([Interval(0.0, 1.0, {"path": ("a", "b")})])
        with pytest.raises(TypeError, match="field 'path' holds a tuple"):
            write_trials(empty_nwb, trials)
        assert empty_nwb.trials is None

    def test_negative_times_are_refused(self, empty_nwb):
        from neurospatial.io.nwb import write_trials

        with pytest.raises(ValueError, match="start_time contains 1 negative"):
            write_trials(empty_nwb, Intervals.from_arrays([-1.0], [2.0]))

    def test_description(self, empty_nwb, tmp_path):
        from neurospatial.io.nwb import write_trials

        write_trials(empty_nwb, _trials())
        assert empty_nwb.trials.description == "Behavioral trials"

        nwbfile = create_roundtrip_nwb()
        write_trials(nwbfile, _trials(), description="T-maze behavioral trials")
        _, description = _write_and_read(nwbfile, tmp_path)
        assert description == "T-maze behavioral trials"


class TestWriteTrialsOverwrite:
    def test_overwrite_false_error(self, empty_nwb):
        from neurospatial.io.nwb import write_trials

        write_trials(empty_nwb, _trials())
        with pytest.raises(ValueError, match="Pass overwrite=True to replace it"):
            write_trials(empty_nwb, _trials())

    def test_overwrite_replaces_trials_in_memory(self, empty_nwb, tmp_path):
        """An in-memory file's trials are replaced, and only the new trials
        reach the file on disk."""
        from neurospatial.io.nwb import read_trials, write_trials

        write_trials(empty_nwb, _trials(0.0, n=2))
        write_trials(empty_nwb, _trials(100.0), description="redone", overwrite=True)

        np.testing.assert_array_equal(read_trials(empty_nwb).start, [100.0])
        back, description = _write_and_read(empty_nwb, tmp_path)
        assert description == "redone"
        np.testing.assert_array_equal(back.start, [100.0])
        np.testing.assert_array_equal(back.stop, [105.0])
        assert back.outcome.tolist() == ["reached"]

    def test_overwrite_with_zero_trials_removes_the_table(self, empty_nwb):
        from neurospatial.io.nwb import write_trials

        write_trials(empty_nwb, _trials(n=2))
        write_trials(empty_nwb, _trials()[[]], overwrite=True)

        assert empty_nwb.trials is None

    def test_invalid_replacement_keeps_the_original(self, empty_nwb):
        from neurospatial.io.nwb import read_trials, write_trials

        write_trials(empty_nwb, _trials(0.0))
        with pytest.raises(ValueError, match="start_time contains 1 negative"):
            write_trials(
                empty_nwb, Intervals.from_arrays([-1.0], [1.0]), overwrite=True
            )
        np.testing.assert_array_equal(read_trials(empty_nwb).start, [0.0])

    def test_replaced_table_is_detached_from_the_file(self, empty_nwb):
        from neurospatial.io.nwb import write_trials

        write_trials(empty_nwb, _trials(0.0))
        old = empty_nwb.trials
        write_trials(empty_nwb, _trials(100.0), overwrite=True)

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

        write_trials(empty_nwb, _trials(0.0, n=2))
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
            write_trials(empty_nwb, _trials(100.0), overwrite=True)
        np.testing.assert_array_equal(read_trials(empty_nwb).start, [0.0, 10.0])

    @pytest.mark.parametrize("mode", ["r", "r+", "a"])
    def test_overwrite_in_a_file_read_from_disk_is_refused(
        self, empty_nwb, tmp_path, mode
    ):
        """Export would silently keep the written trials, so even mode 'r'
        is refused."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_trials, write_trials

        write_trials(empty_nwb, _trials(0.0))
        path = tmp_path / "trials.nwb"
        with NWBHDF5IO(str(path), "w") as io:
            io.write(empty_nwb)

        with NWBHDF5IO(str(path), mode) as io:
            nwbfile = io.read()
            with pytest.raises(
                NotImplementedError,
                match="Cannot replace the trials table of a file read from disk",
            ):
                write_trials(nwbfile, _trials(100.0), overwrite=True)
            np.testing.assert_array_equal(read_trials(nwbfile).start, [0.0])


class TestReadTrials:
    def test_read_trials_not_found(self, empty_nwb):
        from neurospatial.io.nwb import read_trials

        with pytest.raises(KeyError, match="TimeIntervals 'trials' not found"):
            read_trials(empty_nwb)

    def test_round_trip_keeps_ids_and_fields(self, tmp_path, detected):
        from neurospatial.io.nwb import write_trials

        nwbfile = create_roundtrip_nwb()
        write_trials(nwbfile, detected)
        back, _ = _write_and_read(nwbfile, tmp_path)

        assert back.record_type is Interval
        np.testing.assert_array_equal(back.start, detected.start)
        np.testing.assert_array_equal(back.stop, detected.stop)
        np.testing.assert_array_equal(back.interval_ids, detected.interval_ids)
        assert back.fields == detected.fields
        for name in detected.fields:
            assert getattr(back, name).tolist() == getattr(detected, name).tolist()
        # A trial that reached no end region reads back as missing, not "".
        assert back.sel(interval_id=8).end_region is None

    def test_task_log_round_trip(self, tmp_path):
        """A task log with its own fields and trial numbers round-trips."""
        from neurospatial.io.nwb import write_trials

        log = pd.DataFrame(
            {
                "trial": [3, 1, 2],
                "start_time": [20.0, 0.0, 10.0],
                "stop_time": [28.0, 7.5, 17.0],
                "instructed_goal": ["left", "right", None],
                "reward_ml": [0.1, 0.0, 0.1],
                "rewarded": [True, False, True],
            }
        )
        trials = Intervals.from_dataframe(log, id_column="trial")
        nwbfile = create_roundtrip_nwb()
        write_trials(nwbfile, trials)
        back, _ = _write_and_read(nwbfile, tmp_path)

        assert back.interval_ids.tolist() == [1, 2, 3]
        assert back.fields == ("instructed_goal", "reward_ml", "rewarded")
        assert back.instructed_goal.tolist() == ["right", None, "left"]
        np.testing.assert_array_equal(back.reward_ml, [0.0, 0.1, 0.1])
        assert back.rewarded.tolist() == [False, True, True]

    def test_detected_and_reloaded_schema_match(self, tmp_path):
        """Trials from detect_trials and from NWB support the same calls."""
        from shapely.geometry import box

        from neurospatial import Environment, Regions
        from neurospatial.behavior import detect_trials
        from neurospatial.io.nwb import write_trials

        x = np.r_[
            np.full(5, 5.0), np.linspace(5.0, 95.0, 90), np.linspace(95.0, 5.0, 90)
        ]
        times = np.arange(x.size) / 10.0
        positions = np.column_stack([x, np.zeros_like(x)])
        env = Environment.from_samples(positions, bin_size=5.0)
        regions = Regions(env.regions.values())
        regions.add("home", polygon=box(-1, -5, 10, 5))
        regions.add("goal", polygon=box(90, -5, 101, 5))
        env = env.with_regions(regions)
        detected = detect_trials(env, times, positions, regions=["home", "goal"])
        assert detected.outcome.tolist() == ["reached", "reached"]

        nwbfile = create_roundtrip_nwb()
        write_trials(nwbfile, detected)
        back, _ = _write_and_read(nwbfile, tmp_path)

        assert back.fields == detected.fields
        assert list(back.intervals_table().columns) == list(
            detected.intervals_table().columns
        )
        assert list(back.intervals_table().columns)[:2] == ["start_time", "stop_time"]
        reached = back[back.outcome == "reached"]
        np.testing.assert_array_equal(
            reached.sample_interval_ids(times),
            detected[detected.outcome == "reached"].sample_interval_ids(times),
        )

    def test_list_cells_become_tuples(self, empty_nwb):
        """NWB tags are ragged lists; fields are hashable, so they read as tuples."""
        from neurospatial.io.nwb import read_trials

        empty_nwb.add_trial(start_time=0.0, stop_time=1.0, tags=["a", "b"])
        empty_nwb.add_trial(start_time=2.0, stop_time=3.0, tags=["c"])

        trials = read_trials(empty_nwb)
        assert trials.tags.tolist() == [("a", "b"), ("c",)]


def test_missing_text_reads_as_none_with_pandas_string_dtype(empty_nwb):
    """pandas 3 reads NWB text as string dtype; "" must still become None."""
    from neurospatial.io.nwb import read_trials, write_trials

    trials = Intervals([Trial(0.0, 5.0, "home", None, "returned")])
    write_trials(empty_nwb, trials)
    # The string dtype is the default from pandas 3 and opt-in before it.
    string_dtype = (
        contextlib.nullcontext()
        if int(pd.__version__.split(".")[0]) >= 3
        else pd.option_context("future.infer_string", True)
    )
    with string_dtype:
        back = read_trials(empty_nwb)
    assert back.end_region.tolist() == [None]


def test_field_named_id_is_refused(empty_nwb):
    """NWB tables reserve id for the row ids, which hold the interval_ids."""
    from neurospatial.io.nwb import write_trials

    trials = Intervals.from_arrays([0.0, 1.0], [1.0, 2.0], id=[10, 20])
    with pytest.raises(ValueError, match="field 'id'"):
        write_trials(empty_nwb, trials)
    assert empty_nwb.trials is None


@pytest.mark.parametrize("writer", ["write_trials", "write_laps"])
def test_missing_numbers_mixed_with_none_name_the_real_cause(empty_nwb, writer):
    """None among numbers is the problem, not the numbers; the writer is named."""
    from neurospatial.io import nwb

    intervals = Intervals.from_arrays(
        [0.0, 2.0, 4.0], [1.0, 3.0, 5.0], goal_well=[1, None, 2]
    )
    with pytest.raises(TypeError, match=rf"^{writer}: field 'goal_well'") as exc:
        getattr(nwb, writer)(empty_nwb, intervals)
    message = str(exc.value)
    assert "None" in message
    assert "NaN" in message
    # A text field with a stray number names the number's type.
    mixed = Intervals.from_arrays([0.0], [1.0], label=[("a", 1)])
    with pytest.raises(TypeError, match=rf"^{writer}: field 'label' holds a tuple"):
        getattr(nwb, writer)(empty_nwb, mixed)


def test_detected_and_reloaded_trials_give_the_same_progress(empty_nwb):
    """Route functions read fields, so NWB-reloaded trials behave identically."""
    from neurospatial.behavior import distance_to_goal, path_progress
    from neurospatial.io.nwb import read_trials, write_trials
    from neurospatial.simulation.mazes import make_t_maze

    track = make_t_maze().env_track
    stem = np.column_stack([np.zeros(10), np.linspace(-40.0, 45.0, 10)])
    positions = np.vstack([stem, stem, stem])
    times = np.arange(30.0)
    detected = Intervals(
        [
            Trial(0.0, 10.0, "start", "well_left", "reached"),
            Trial(10.0, 20.0, "start", None, "returned"),
            Trial(20.0, 30.0, "start", "well_right", "reached"),
        ]
    )
    write_trials(empty_nwb, detected)
    reloaded = read_trials(empty_nwb)

    def route(trials):
        return (
            path_progress(
                track, times, positions, trials, start="start", goal=trials.end_region
            ),
            distance_to_goal(track, times, positions, trials, goal=trials.end_region),
        )

    progress, remaining = route(detected)
    reloaded_progress, reloaded_remaining = route(reloaded)
    np.testing.assert_array_equal(reloaded_progress, progress)
    np.testing.assert_array_equal(reloaded_remaining, remaining)
    assert np.isnan(progress[10:20]).all()  # the trial without a destination
    assert np.isfinite(progress[:10]).all()


def test_imported_and_reloaded_trials_give_the_same_tuning_curves(empty_nwb):
    """A task log's trials and the same trials read back from NWB select alike."""
    from neurospatial.encoding import compute_tuning_curves
    from neurospatial.io.nwb import read_trials, write_trials

    times = np.arange(400) / 10.0
    progress = (np.arange(400) % 40) / 40.0
    log = pd.DataFrame(
        {
            "start_time": np.arange(10) * 4.0,
            "stop_time": np.arange(1, 11) * 4.0,
            "instructed_goal": ["well_1", "well_3"] * 5,
            "outcome": ["correct"] * 7 + ["error"] * 3,
        }
    )
    imported = Intervals.from_dataframe(log)
    write_trials(empty_nwb, imported)
    reloaded = read_trials(empty_nwb)
    spikes = [times[progress >= 0.5] + 0.05, times[::3] + 0.02]

    def left_correct(trials):
        chosen = trials[
            (trials.instructed_goal == "well_1") & (trials.outcome == "correct")
        ]
        return compute_tuning_curves(
            spikes, times, progress, bin_edges=np.linspace(0, 1, 11), epochs=chosen
        )

    before, after = left_correct(imported), left_correct(reloaded)
    np.testing.assert_array_equal(after.firing_rates, before.firing_rates)
    np.testing.assert_array_equal(after.occupancy, before.occupancy)
    assert after.analysis_settings["epoch_interval_ids"] == [0, 2, 4, 6]
    assert before.occupancy.sum() == pytest.approx(16.0)


@pytest.mark.pynapple
def test_pynapple_loads_a_written_selection_with_its_ids(tmp_path):
    """A selection's ids are not 0..n-1; pynapple still builds the IntervalSet."""
    nap = pytest.importorskip("pynapple")
    from neurospatial.io import from_pynapple
    from neurospatial.io.nwb import write_trials

    trials = Intervals.from_arrays(
        [0.0, 10.0, 20.0, 30.0], [5.0, 15.0, 25.0, 35.0], goal=["a", "b", "a", "b"]
    )
    selected = trials[trials.goal == "b"]  # ids 1 and 3
    nwbfile = create_roundtrip_nwb()
    write_trials(nwbfile, selected)
    path = tmp_path / "selected.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)

    loaded = nap.load_file(str(path))["trials"]
    assert isinstance(loaded, nap.IntervalSet)
    back = from_pynapple(loaded)
    assert back.interval_ids.tolist() == [1, 3]
    assert back.goal.tolist() == ["b", "b"]
