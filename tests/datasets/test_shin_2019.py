"""Shin et al. (2019), DANDI 000978: synthetic NWB files and the real sessions."""

from __future__ import annotations

import datetime
import io
import itertools
import urllib.error
import warnings

import numpy as np
import pytest

from neurospatial.datasets._checks import run_checks

#: The synthetic W-maze's nodes, cm: arms at x = 30, 70 and 110, wells at y = 100.
NODES = {
    "well_1": (70.0, 100.0),
    "well_2": (30.0, 100.0),
    "well_3": (110.0, 100.0),
    "base_1": (70.0, 10.0),
    "base_2": (30.0, 10.0),
    "base_3": (110.0, 10.0),
}
#: Rest, run, rest, s.
EPOCHS = ((0.0, 100.0), (110.0, 410.0), (420.0, 520.0))
#: The run's well sequence: centre, side, centre, other side, ...
WELLS = (1, 2, 1, 3, 1, 2, 1, 3, 1)
DWELL = 6.0
SPEED = 40.0
RATE = 30.0
#: Tetrodes (number, location); units sit on tetrodes 1, 3 and 4.
TETRODES = ((1, "CA1"), (2, "CA1"), (3, "PFC"), (4, "PFC"))
UNIT_TETRODES = (1, 1, 3, 4)
LFP_RATE = 1500.0


def _route(a: int, b: int) -> np.ndarray:
    """Corner points from well ``a`` to well ``b`` along the W."""
    return np.array(
        [NODES[f"well_{a}"], NODES[f"base_{a}"], NODES[f"base_{b}"], NODES[f"well_{b}"]]
    )


def _run_path(start: float):
    """Times, positions and trial rows of the run: dwell at a well, then go."""
    times, xy, rows = [], [], []
    now = start
    leave_times = []
    for a, b in itertools.pairwise(WELLS):
        dwell = np.arange(now, now + DWELL, 1 / RATE)
        times.append(dwell)
        xy.append(np.tile(NODES[f"well_{a}"], (dwell.size, 1)))
        now += DWELL
        leave_times.append(now)
        corners = _route(a, b)
        lengths = np.linalg.norm(np.diff(corners, axis=0), axis=1)
        travel = np.arange(0.0, lengths.sum() / SPEED, 1 / RATE)
        along = travel * SPEED
        cumulative = np.r_[0, np.cumsum(lengths)]
        segment = np.clip(np.searchsorted(cumulative, along, side="right") - 1, 0, 2)
        share = (along - cumulative[segment]) / lengths[segment]
        points = corners[segment] + share[:, None] * (
            corners[segment + 1] - corners[segment]
        )
        times.append(now + travel)
        xy.append(points)
        now += travel[-1] + 1 / RATE
    last = np.arange(now, now + DWELL, 1 / RATE)
    times.append(last)
    xy.append(np.tile(NODES[f"well_{WELLS[-1]}"], (last.size, 1)))
    leave_times.append(now + DWELL)
    # A trial runs from leaving one well to leaving the next, 0.5 s late.
    stops = np.array(leave_times) + 0.5
    rows = [(start, stops[1])] + [
        (stops[k], stops[k + 1]) for k in range(1, len(stops) - 1)
    ]
    return np.concatenate(times), np.concatenate(xy), rows


def _write_session(path, *, comments=None):
    """An NWB file laid out as DANDI 000978's."""
    pynwb = pytest.importorskip("pynwb")
    from pynwb.behavior import Position, SpatialSeries
    from pynwb.ecephys import LFP, ElectricalSeries
    from pynwb.epoch import TimeIntervals
    from pynwb.file import Subject

    nwb = pynwb.NWBFile(
        "Single Day WTrack Learning Experiment",
        "SYN_01",
        datetime.datetime(2024, 5, 1, tzinfo=datetime.timezone.utc),
        subject=Subject(subject_id="JDS-SingleDay-JS14", species="Rattus norvegicus"),
    )
    run_times, run_xy, trial_rows = _run_path(EPOCHS[1][0])
    rest = [np.arange(a, b, 1 / RATE) for a, b in (EPOCHS[0], EPOCHS[2])]
    times = np.concatenate([rest[0], run_times, rest[1]])
    box = [
        np.column_stack([60 + 5 * np.sin(r / 7.0), 30 + 5 * np.cos(r / 5.0)])
        for r in rest
    ]
    xy = np.concatenate([box[0], run_xy, box[1]])
    speed = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1) * RATE]
    behavior = nwb.create_processing_module("behavior", "contains behavioral data")
    position = Position(name="Position")
    position.add_spatial_series(
        SpatialSeries(
            name="SpatialSeries",
            data=np.column_stack([xy, speed]),
            timestamps=times,
            reference_frame="center well",
            unit="centimeters; centimeters/second",
            comments=comments
            or "Video frames taken at 30fps - row1:xpos row2:ypos row3:velocity",
        )
    )
    behavior.add(position)
    device = nwb.create_device("array")
    nwb.add_electrode_column("label", "whether the channel is referenced")
    for number, location in TETRODES:
        group = nwb.create_electrode_group(
            f"Tetrode{number}", "tetrode", location=location, device=device
        )
        for label in ("Ref OFF", "Ref ON"):
            nwb.add_electrode(
                group=group,
                group_name=f"tetrode{number}",
                label=label,
                location=location,
            )
    region = nwb.create_electrode_table_region(list(range(2 * len(TETRODES))), "all")
    lfp_times = np.concatenate([np.arange(a, b, 1 / LFP_RATE) for a, b in EPOCHS])
    columns = np.arange(2 * len(TETRODES))
    voltage = (np.sin(lfp_times)[:, None] * 10 + columns[None, :]).astype(np.float64)
    nwb.add_acquisition(
        ElectricalSeries(
            name="ElectricalSeries",
            data=voltage,
            electrodes=region,
            timestamps=lfp_times,
        )
    )
    ecephys = nwb.create_processing_module("ecephys", "ecephys")
    lfp = LFP()
    ecephys.add(lfp)
    lfp.add_electrical_series(
        ElectricalSeries(
            name="ElectricalSeries",
            data=voltage,
            electrodes=nwb.create_electrode_table_region(
                list(range(2 * len(TETRODES))), "lfp"
            ),
            timestamps=lfp_times,
        )
    )
    epochs = TimeIntervals(
        name="epoch intervals", description="Start and end times of all epochs"
    )
    for start, stop in EPOCHS:
        epochs.add_interval(start_time=start, stop_time=stop)
    nwb.add_time_intervals(epochs)
    for column, text in (
        ("correct", "correct"),
        ("trajectory_type", "type"),
        ("start_well", "start"),
        ("end_well", "end"),
    ):
        nwb.add_trial_column(column, text)
    for (start, stop), a, b in zip(trial_rows, WELLS[:-1], WELLS[1:], strict=True):
        nwb.add_trial(
            start_time=start,
            stop_time=stop,
            correct=1.0,
            trajectory_type=0.0 if a == 1 else 1.0,
            start_well=float(a),
            end_well=float(b),
        )
    rng = np.random.default_rng(1)
    with warnings.catch_warnings():
        # pynwb warns while the units table is not yet in the file.
        warnings.filterwarnings("ignore", message="The linked table for")
        for tetrode in UNIT_TETRODES:
            # The file's 'electrodes' value is the tetrode number, not a table row.
            nwb.add_unit(
                spike_times=np.sort(rng.uniform(1.0, 515.0, 400)), electrodes=[tetrode]
            )
        with pynwb.NWBHDF5IO(str(path), "w") as nwb_io:
            nwb_io.write(nwb)
    import h5py

    # The files store one value per unit, not the ragged list NWB defines.
    with h5py.File(path, "r+") as h5:
        del h5["units/electrodes_index"]
    return times, xy, np.array(trial_rows), lfp_times, voltage


@pytest.fixture
def entry(session_entry):
    return session_entry("shin_2019")


@pytest.fixture
def synthetic(tmp_path):
    path = tmp_path / "session.nwb"
    return path, *_write_session(path)


def test_loads_positions_epochs_trials_and_units(synthetic, entry):
    from neurospatial.datasets import shin_2019

    path, times, xy, trial_rows, _, _ = synthetic
    rec = run_checks(shin_2019.load_session(entry, path), stated=entry.info.stated)
    assert rec.failed_checks == ()
    np.testing.assert_array_equal(rec.times, times)
    np.testing.assert_array_equal(rec.positions, xy)
    np.testing.assert_array_equal(rec.epochs["run"], [EPOCHS[1]])
    np.testing.assert_array_equal(rec.epochs["rest"], [EPOCHS[0], EPOCHS[2]])
    np.testing.assert_array_equal(rec.epochs["trials"], trial_rows)
    np.testing.assert_array_equal(rec.params["analysis_epochs"], [EPOCHS[1]])
    assert len(rec.trials) == len(WELLS) - 1
    # Brain areas by tetrode number: 1 is CA1, 3 and 4 are PFC.
    assert list(rec.units["brain_region"]) == ["CA1", "CA1", "PFC", "PFC"]
    assert list(rec.units["tetrode"]) == list(UNIT_TETRODES)
    assert rec.units["label"].isna().all()
    assert rec.animal == "rat JS14"


def test_trials_alternate_train_and_test_within_each_path(synthetic, entry):
    from neurospatial.datasets import shin_2019

    path, *_ = synthetic
    rec = shin_2019.load_session(entry, path)
    # Paths in order: 1-2, 2-1, 1-3, 3-1, 1-2, 2-1, 1-3, 3-1; the first of each
    # path trains, the second tests.
    train, test = rec.params["train_epochs"], rec.params["test_epochs"]
    trials = rec.epochs["trials"]
    np.testing.assert_array_equal(train, trials[:4])
    np.testing.assert_array_equal(test, trials[4:])


def test_w_maze_is_fitted_with_the_wells_at_the_arms_ends(synthetic, entry):
    from neurospatial.datasets import shin_2019

    path, *_ = synthetic
    rec = shin_2019.load_session(entry, path)
    for k in (1, 2, 3):
        well = rec.env.regions[f"well_{k}"].data
        assert np.linalg.norm(well - NODES[f"well_{k}"]) < 3.0, k
    on_track = rec.env.bin_at(np.array([NODES["base_1"], (50.0, 10.0), (110.0, 55.0)]))
    assert np.all(on_track >= 0)
    assert rec.env.bin_at(np.array([[50.0, 60.0]]))[0] == -1  # between two arms


def test_watch_outs_count_units_pynwb_would_place_in_the_wrong_area(synthetic, entry):
    from neurospatial.datasets import shin_2019

    path, *_ = synthetic
    rec = shin_2019.load_session(entry, path)
    # As rows, tetrode numbers 1, 1, 3, 4 name rows 1 (CA1), 1, 3 (CA1) and 4
    # (PFC): unit 3 lands in the wrong area.
    text = next(t for t in rec.watch_outs if "tetrode number" in t)
    assert "for 1 of the 4 units here the wrong brain area" in text
    region = next(c for c in rec.conversions if c.text.startswith("units:"))
    assert "(CA1 2, PFC 2)" in region.text
    assert "Read as rows here: CA1 3, PFC 1" in region.evidence
    assert any("not one of the paper's six rats" in t for t in rec.watch_outs) is False


def test_trial_stops_are_reported_as_departures(synthetic, entry):
    from neurospatial.datasets import shin_2019

    path, *_ = synthetic
    rec = shin_2019.load_session(entry, path)
    trials = next(c for c in rec.conversions if c.text.startswith("trials:"))
    # 0.5 s after leaving at 40 cm/s the rat is 20 cm down the arm.
    assert "the rat is a median 20.0 cm from the end of its end well's arm" in (
        trials.evidence
    )


def test_lfp_source_reads_columns_and_windows(synthetic, entry):
    from neurospatial.datasets import shin_2019

    path, _, _, _, lfp_times, voltage = synthetic
    rec = shin_2019.load_session(entry, path)
    assert list(rec.lfp.channels)[:2] == ["tetrode1 Ref OFF", "tetrode1 Ref ON"]
    assert rec.lfp.channels["tetrode3 Ref ON"].startswith("PFC, tetrode 3, referenced")
    samples, times = rec.lfp.reader((path,), ("tetrode2 Ref ON", "tetrode1 Ref OFF"))
    np.testing.assert_array_equal(times, lfp_times)
    np.testing.assert_array_equal(samples, voltage[:, [3, 0]])
    ((window, at),) = shin_2019._read_lfp_windows(
        path, ("tetrode4 Ref ON",), np.array([EPOCHS[1]])
    )
    inside = (lfp_times >= EPOCHS[1][0]) & (lfp_times < EPOCHS[1][1])
    np.testing.assert_array_equal(at, lfp_times[inside])
    np.testing.assert_array_equal(window[:, 0], voltage[inside, 7])


def test_byte_ranges_are_retried_and_reassembled(monkeypatch):
    """Rows read by ranged requests: a reset is retried, columns are picked."""
    from neurospatial.datasets import shin_2019

    array = np.arange(40 * 3, dtype="<f8").reshape(40, 3)
    blob = b"x" * 16 + array.tobytes()
    failures = {"left": 1}

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout):
        if failures["left"]:
            failures["left"] -= 1
            raise urllib.error.URLError(ConnectionResetError(54, "reset"))
        low, high = request.headers["Range"].removeprefix("bytes=").split("-")
        return _Response(blob[int(low) : int(high) + 1])

    monkeypatch.setattr(shin_2019, "_LFP_BLOCK_BYTES", 8 * 3 * 7)
    monkeypatch.setattr(shin_2019.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(shin_2019.urllib.request, "urlopen", urlopen)
    rows = shin_2019._ranged_rows("https://x", 16, 3, 5, 33, np.array([2, 0]))
    np.testing.assert_array_equal(rows, array[5:33][:, [2, 0]])


def test_refuses_a_position_series_without_its_column_names(tmp_path, entry):
    from neurospatial.datasets import shin_2019

    path = tmp_path / "session.nwb"
    _write_session(path, comments="x and y")
    with pytest.raises(ValueError, match="not the x, y and velocity columns"):
        shin_2019.load_session(entry, path)


def test_load_streams_a_stream_only_session_that_is_not_cached(
    monkeypatch, tmp_path, entry
):
    """load() reads such a session by byte range, never downloading it."""
    from neurospatial.datasets import _load, shin_2019

    assert entry.stream_only
    seen = {}
    monkeypatch.setattr(_load, "_stream_url", lambda e: "https://streamed")
    monkeypatch.setattr(
        _load,
        "fetch_session",
        lambda *a, **k: pytest.fail("a stream-only session was downloaded"),
    )

    def read(e, source):
        seen["source"] = source
        msg = "stop"
        raise RuntimeError(msg)

    monkeypatch.setattr(shin_2019, "load_session", read)
    with pytest.raises(RuntimeError, match="stop"):
        _load.load("shin_2019", cache=tmp_path)
    assert seen["source"] == "https://streamed"
