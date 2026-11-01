"""Tests for write_laps: laps stored as an NWB TimeIntervals table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pynwb = pytest.importorskip("pynwb")

from neurospatial import Intervals  # noqa: E402
from neurospatial.behavior.segmentation import Lap  # noqa: E402

from ._nwb_helpers import create_roundtrip_nwb  # noqa: E402

LAPS = Intervals(
    [
        Lap(start_time=1.0, stop_time=4.5, direction="clockwise", overlap_score=0.9),
        Lap(5.0, 9.0, "counter-clockwise", overlap_score=0.8),
        Lap(10.0, 12.0, "unknown", overlap_score=np.nan),
    ]
)
NO_LAPS = Intervals(record_type=Lap)


def _save(nwbfile, path):
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)


def _read_laps(path, name="laps"):
    from pynwb import NWBHDF5IO

    from neurospatial.io.nwb import read_intervals

    with NWBHDF5IO(str(path), "r") as io:
        return read_intervals(io.read(), interval_name=name).intervals_table()


class TestWriteLaps:
    """Laps round-trip through a file with their ids and every field."""

    def test_lap_fields_survive_a_file(self, tmp_path):
        from neurospatial.io.nwb import write_laps

        nwbfile = create_roundtrip_nwb()
        write_laps(nwbfile, LAPS)
        _save(nwbfile, tmp_path / "laps.nwb")

        recovered = _read_laps(tmp_path / "laps.nwb").reset_index(drop=True)
        expected = pd.DataFrame(
            {
                "start_time": [lap.start_time for lap in LAPS],
                "stop_time": [lap.stop_time for lap in LAPS],
                "direction": [lap.direction for lap in LAPS],
                "overlap_score": [lap.overlap_score for lap in LAPS],
            }
        )
        # The region method's NaN overlap score stays NaN, not zero.
        pd.testing.assert_frame_equal(recovered[list(expected.columns)], expected)

    def test_interval_ids_are_the_table_ids(self, empty_nwb):
        from neurospatial.io.nwb import read_intervals, write_laps

        selected = LAPS[LAPS.direction != "clockwise"]
        write_laps(empty_nwb, selected)

        laps = read_intervals(empty_nwb, interval_name="laps")
        assert laps.interval_ids.tolist() == [1, 2]
        assert laps.direction.tolist() == ["counter-clockwise", "unknown"]

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
        np.testing.assert_array_equal(laps.end, [lap.stop_time for lap in LAPS])
        assert list(laps["direction"]) == [lap.direction for lap in LAPS]

    def test_zero_laps_write_nothing(self, tmp_path):
        """NWB files should not hold empty tables, so no table is added."""
        from neurospatial.io.nwb import write_laps

        nwbfile = create_roundtrip_nwb()
        write_laps(nwbfile, NO_LAPS)
        assert "laps" not in nwbfile.intervals
        _save(nwbfile, tmp_path / "laps.nwb")

        with pytest.raises(KeyError, match="'laps' not found"):
            _read_laps(tmp_path / "laps.nwb")

    def test_zero_laps_still_validate_the_name(self, empty_nwb):
        from neurospatial.io.nwb import write_laps

        with pytest.raises(ValueError, match="cannot use name='trials'"):
            write_laps(empty_nwb, NO_LAPS, name="trials")

    def test_region_method_laps_omit_the_all_nan_overlap_score(self, tmp_path):
        """Laps found without a template have no overlap score to store."""
        from neurospatial.io.nwb import write_laps

        region_laps = Intervals(
            [Lap(1.0, 4.0, "clockwise", np.nan), Lap(5.0, 9.0, "unknown", np.nan)]
        )
        nwbfile = create_roundtrip_nwb()
        write_laps(nwbfile, region_laps)
        _save(nwbfile, tmp_path / "laps.nwb")

        recovered = _read_laps(tmp_path / "laps.nwb")
        assert list(recovered.columns) == ["start_time", "stop_time", "direction"]
        np.testing.assert_array_equal(recovered["start_time"], [1.0, 5.0])

    @pytest.mark.parametrize(
        "laps", [list(LAPS), np.array([1.0, 2.0])], ids=["records", "array"]
    )
    def test_anything_but_intervals_is_refused(self, empty_nwb, laps):
        from neurospatial.io.nwb import write_laps

        with pytest.raises(TypeError, match=r"laps must be an Intervals"):
            write_laps(empty_nwb, laps)

    def test_laps_read_back_write_again(self, tmp_path):
        """Laps read from NWB are Intervals of Interval and write the same table."""
        from neurospatial.io.nwb import write_laps

        nwbfile = create_roundtrip_nwb()
        write_laps(nwbfile, LAPS)
        _save(nwbfile, tmp_path / "laps.nwb")
        table = _read_laps(tmp_path / "laps.nwb")
        read_back = Intervals.from_dataframe(
            table.reset_index(), id_column="interval_id"
        )

        again = create_roundtrip_nwb()
        write_laps(again, read_back)
        _save(again, tmp_path / "again.nwb")
        pd.testing.assert_frame_equal(_read_laps(tmp_path / "again.nwb"), table)


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

        table = read_intervals(empty_nwb, interval_name="laps").intervals_table()
        np.testing.assert_array_equal(table["start_time"], [1.0])

    def test_overwrite_with_zero_laps_removes_the_table(self, empty_nwb):
        from neurospatial.io.nwb import write_laps

        write_laps(empty_nwb, LAPS)
        write_laps(empty_nwb, NO_LAPS, overwrite=True)

        assert "laps" not in empty_nwb.intervals

    def test_zero_laps_without_overwrite_keep_the_existing_table(self, empty_nwb):
        from neurospatial.io.nwb import write_laps

        write_laps(empty_nwb, LAPS)
        with pytest.raises(ValueError, match="'laps' already exists"):
            write_laps(empty_nwb, NO_LAPS)

        assert len(empty_nwb.intervals["laps"]) == len(LAPS)

    def test_invalid_replacement_keeps_the_original(self, empty_nwb):
        from neurospatial.io.nwb import read_intervals, write_laps

        write_laps(empty_nwb, LAPS)
        negative = Intervals.from_arrays([-1.0], [2.0], direction=["clockwise"])
        with pytest.raises(ValueError, match="negative timestamp"):
            write_laps(empty_nwb, negative, overwrite=True)

        kept = read_intervals(empty_nwb, interval_name="laps")
        assert list(kept.direction) == [lap.direction for lap in LAPS]

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
