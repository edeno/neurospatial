"""Tests for write_laps: laps stored as an NWB TimeIntervals table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pynwb = pytest.importorskip("pynwb")

from neurospatial.behavior.segmentation import Lap  # noqa: E402

from ._nwb_helpers import create_roundtrip_nwb  # noqa: E402

LAPS = [
    Lap(start_time=1.0, end_time=4.5, direction="clockwise", overlap_score=0.9),
    Lap(start_time=5.0, end_time=9.0, direction="counter-clockwise", overlap_score=0.8),
    Lap(start_time=10.0, end_time=12.0, direction="unknown", overlap_score=np.nan),
]


def _save(nwbfile, path):
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)


def _read_laps(path, name="laps"):
    from pynwb import NWBHDF5IO

    from neurospatial.io.nwb import read_intervals

    with NWBHDF5IO(str(path), "r") as io:
        return read_intervals(io.read(), interval_name=name)


class TestWriteLapsFromLaps:
    """Lap objects round-trip through a file with every field."""

    def test_lap_fields_survive_a_file(self, tmp_path):
        from neurospatial.io.nwb import write_laps

        nwbfile = create_roundtrip_nwb()
        write_laps(nwbfile, LAPS)
        _save(nwbfile, tmp_path / "laps.nwb")

        recovered = _read_laps(tmp_path / "laps.nwb").reset_index(drop=True)
        expected = pd.DataFrame(
            {
                "start_time": [lap.start_time for lap in LAPS],
                "stop_time": [lap.end_time for lap in LAPS],
                "direction": [lap.direction for lap in LAPS],
                "overlap_score": [lap.overlap_score for lap in LAPS],
            }
        )
        # The region method's NaN overlap score stays NaN, not zero.
        pd.testing.assert_frame_equal(recovered[list(expected.columns)], expected)

    def test_laps_are_a_time_intervals_table(self, empty_nwb):
        from pynwb.epoch import TimeIntervals

        from neurospatial.io.nwb import write_laps

        write_laps(empty_nwb, LAPS, name="track_laps", description="Laps on day 1")

        table = empty_nwb.intervals["track_laps"]
        assert isinstance(table, TimeIntervals)
        assert table.description == "Laps on day 1"
        assert "behavior" not in empty_nwb.processing

    @pytest.mark.pynapple
    def test_pynapple_loads_laps_as_an_interval_set(self, tmp_path):
        nap = pytest.importorskip("pynapple")

        from neurospatial.io.nwb import write_laps

        nwbfile = create_roundtrip_nwb()
        write_laps(nwbfile, LAPS)
        _save(nwbfile, tmp_path / "laps.nwb")

        laps = nap.load_file(str(tmp_path / "laps.nwb"))["laps"]
        assert isinstance(laps, nap.IntervalSet)
        np.testing.assert_array_equal(laps.start, [lap.start_time for lap in LAPS])
        np.testing.assert_array_equal(laps.end, [lap.end_time for lap in LAPS])
        assert list(laps["direction"]) == [lap.direction for lap in LAPS]

    @pytest.mark.parametrize(
        "kwargs",
        [{"laps": []}, {"start_times": [], "stop_times": []}],
        ids=["records", "arrays"],
    )
    def test_zero_laps_write_nothing_and_warn(self, tmp_path, kwargs):
        """NWB files should not hold empty tables, so no table is added."""
        from neurospatial.io.nwb import write_laps

        nwbfile = create_roundtrip_nwb()
        with pytest.warns(
            UserWarning, match="write_laps wrote nothing: there are no laps"
        ):
            write_laps(nwbfile, **kwargs)
        assert "laps" not in nwbfile.intervals
        _save(nwbfile, tmp_path / "laps.nwb")

        with pytest.raises(KeyError, match="'laps' not found"):
            _read_laps(tmp_path / "laps.nwb")

    def test_zero_laps_still_validate_the_name(self, empty_nwb):
        from neurospatial.io.nwb import write_laps

        with pytest.raises(ValueError, match="cannot use name='trials'"):
            write_laps(empty_nwb, [], name="trials")

    def test_region_method_laps_omit_the_all_nan_overlap_score(self, tmp_path):
        """Laps found without a template have no overlap score to store."""
        from neurospatial.io.nwb import write_laps

        region_laps = [
            Lap(
                start_time=1.0,
                end_time=4.0,
                direction="clockwise",
                overlap_score=np.nan,
            ),
            Lap(
                start_time=5.0, end_time=9.0, direction="unknown", overlap_score=np.nan
            ),
        ]
        nwbfile = create_roundtrip_nwb()
        write_laps(nwbfile, region_laps)
        _save(nwbfile, tmp_path / "laps.nwb")

        recovered = _read_laps(tmp_path / "laps.nwb")
        assert list(recovered.columns) == ["start_time", "stop_time", "direction"]
        np.testing.assert_array_equal(recovered["start_time"], [1.0, 5.0])

    @pytest.mark.parametrize(
        ("laps", "message"),
        [
            (np.array([1.0, 2.0]), "laps must be a list of Lap objects; got ndarray"),
            ([1.0, 2.0], r"laps\[0\] is a float, not a Lap"),
            ((LAPS[0], "lap"), r"laps\[1\] is a str, not a Lap"),
        ],
    )
    def test_non_lap_records_are_refused(self, empty_nwb, laps, message):
        from neurospatial.io.nwb import write_laps

        with pytest.raises(TypeError, match=message):
            write_laps(empty_nwb, laps)

    def test_laps_with_arrays_are_refused(self, empty_nwb):
        from neurospatial.io.nwb import write_laps

        with pytest.raises(ValueError, match="Cannot combine 'laps' with directions"):
            write_laps(empty_nwb, LAPS, directions=["clockwise"] * 3)


class TestWriteLapsFromArrays:
    """Explicit arrays use the Lap.direction vocabulary."""

    def test_arrays_survive_a_file(self, tmp_path):
        from neurospatial.io.nwb import write_laps

        nwbfile = create_roundtrip_nwb()
        write_laps(
            nwbfile,
            start_times=np.array([1.0, 5.0]),
            stop_times=np.array([4.0, 9.5]),
            directions=["clockwise", "counter-clockwise"],
        )
        _save(nwbfile, tmp_path / "laps.nwb")

        recovered = _read_laps(tmp_path / "laps.nwb").reset_index(drop=True)
        np.testing.assert_array_equal(recovered["start_time"], [1.0, 5.0])
        np.testing.assert_array_equal(recovered["stop_time"], [4.0, 9.5])
        assert list(recovered["direction"]) == ["clockwise", "counter-clockwise"]
        assert "overlap_score" not in recovered.columns

    def test_directions_are_optional(self, empty_nwb):
        from neurospatial.io.nwb import read_intervals, write_laps

        write_laps(empty_nwb, start_times=[1.0], stop_times=[2.0])

        table = read_intervals(empty_nwb, interval_name="laps")
        assert set(table.columns) == {"start_time", "stop_time"}

    @pytest.mark.parametrize(
        ("directions", "bad"),
        [(["clockwise", "outbound"], "'outbound'"), ([0, 1], "0")],
    )
    def test_direction_outside_the_lap_vocabulary_is_refused(
        self, empty_nwb, directions, bad
    ):
        from neurospatial.io.nwb import write_laps

        with pytest.raises(
            ValueError, match=f"directions contains {bad}, which is not a lap direction"
        ):
            write_laps(
                empty_nwb,
                start_times=[1.0, 5.0],
                stop_times=[4.0, 9.0],
                directions=directions,
            )

    def test_needs_laps_or_both_time_arrays(self, empty_nwb):
        from neurospatial.io.nwb import write_laps

        with pytest.raises(ValueError, match="needs either 'laps'"):
            write_laps(empty_nwb, start_times=[1.0])

    @pytest.mark.parametrize(
        ("start", "stop", "directions", "message"),
        [
            ([[1.0, 2.0]], [[3.0, 4.0]], None, "start_times must be a 1-D array"),
            ([1.0, 2.0, 3.0], [2.0, 3.0], None, r"stop_times length \(2\) must match"),
            (
                [1.0, 2.0],
                [2.0, 3.0],
                ["clockwise"],
                r"directions length \(1\) must match start_times length \(2\)",
            ),
            ([1.0, np.nan], [2.0, 3.0], None, r"start_times contains non-finite"),
            ([1.0, 2.0], [2.0, np.inf], None, r"stop_times contains non-finite"),
            ([-1.0, 2.0], [2.0, 3.0], None, "start_times contains negative"),
            ([1.0, 5.0], [2.0, 3.0], None, "every lap. Lap 1: start_time=5.0"),
        ],
    )
    def test_invalid_arrays_are_refused(
        self, empty_nwb, start, stop, directions, message
    ):
        from neurospatial.io.nwb import write_laps

        with pytest.raises(ValueError, match=message):
            write_laps(
                empty_nwb, start_times=start, stop_times=stop, directions=directions
            )
        assert "laps" not in empty_nwb.intervals


class TestWriteLapsNames:
    def test_predefined_interval_names_are_refused(self, empty_nwb):
        from neurospatial.io.nwb import write_laps

        with pytest.raises(ValueError, match="write_laps cannot use name='trials'"):
            write_laps(empty_nwb, LAPS, name="trials")
        assert empty_nwb.trials is None

    def test_duplicate_name_without_overwrite_is_refused(self, empty_nwb):
        from neurospatial.io.nwb import write_laps

        write_laps(empty_nwb, LAPS)
        with pytest.raises(ValueError, match="'laps' already exists"):
            write_laps(empty_nwb, LAPS[:1])


class TestWriteLapsOverwrite:
    """overwrite=True never loses the existing table to a failed write."""

    def test_overwrite_replaces_in_memory(self, empty_nwb):
        from neurospatial.io.nwb import read_intervals, write_laps

        write_laps(empty_nwb, LAPS)
        write_laps(empty_nwb, LAPS[:1], overwrite=True)

        table = read_intervals(empty_nwb, interval_name="laps")
        np.testing.assert_array_equal(table["start_time"], [1.0])

    def test_overwrite_with_zero_laps_removes_the_table(self, empty_nwb):
        from neurospatial.io.nwb import write_laps

        write_laps(empty_nwb, LAPS)
        with pytest.warns(UserWarning, match="existing table 'laps' was removed"):
            write_laps(empty_nwb, [], overwrite=True)

        assert "laps" not in empty_nwb.intervals

    def test_zero_laps_without_overwrite_keep_the_existing_table(self, empty_nwb):
        from neurospatial.io.nwb import write_laps

        write_laps(empty_nwb, LAPS)
        with pytest.raises(ValueError, match="'laps' already exists"):
            write_laps(empty_nwb, [])

        assert len(empty_nwb.intervals["laps"]) == len(LAPS)

    def test_invalid_replacement_keeps_the_original(self, empty_nwb):
        from neurospatial.io.nwb import read_intervals, write_laps

        write_laps(empty_nwb, LAPS)
        with pytest.raises(ValueError, match="directions contains 1"):
            write_laps(
                empty_nwb,
                start_times=[1.0],
                stop_times=[2.0],
                directions=[1],
                overwrite=True,
            )

        kept = read_intervals(empty_nwb, interval_name="laps")
        assert list(kept["direction"]) == [lap.direction for lap in LAPS]

    @pytest.mark.parametrize("mode", ["r+", "a"])
    def test_in_place_overwrite_is_refused(self, tmp_path, mode):
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_intervals, write_laps

        nwbfile = create_roundtrip_nwb()
        write_laps(nwbfile, LAPS)
        _save(nwbfile, tmp_path / "laps.nwb")

        with NWBHDF5IO(str(tmp_path / "laps.nwb"), mode) as io:
            opened = io.read()
            with pytest.raises(NotImplementedError, match=f"mode='{mode}'"):
                write_laps(opened, LAPS[:1], overwrite=True)
            kept = read_intervals(opened, interval_name="laps")
            assert len(kept) == len(LAPS)

    def test_overwrite_then_export_round_trips(self, tmp_path):
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import write_laps

        nwbfile = create_roundtrip_nwb()
        write_laps(nwbfile, LAPS)
        _save(nwbfile, tmp_path / "laps.nwb")

        with NWBHDF5IO(str(tmp_path / "laps.nwb"), "r") as io:
            opened = io.read()
            write_laps(opened, LAPS[1:], overwrite=True)
            with NWBHDF5IO(str(tmp_path / "copy.nwb"), "w") as export_io:
                export_io.export(src_io=io, nwbfile=opened)

        recovered = _read_laps(tmp_path / "copy.nwb")
        assert list(recovered["direction"]) == ["counter-clockwise", "unknown"]
