"""Peyrache lab RSC, DANDI 000987: synthetic NWB files and the real session."""

from __future__ import annotations

import datetime
import re

import numpy as np
import pytest

from neurospatial.datasets import load
from neurospatial.datasets._checks import run_checks

#: The motion-capture rate of most of the dandiset's files, Hz.
RATE = 119.76
SQUARE = "Freely behaving mouse RSC data. Order of epochs \u2013 sleep, square arena."
TWO = (
    "Freely behaving mouse RSC data. Order of epochs \u2013 sleep, square arena, "
    "sleep, circular arena, sleep"
)
#: The file's deprecated device field, written by the real files.
OLD_DEVICE = pytest.mark.filterwarnings(
    "ignore:The 'manufacturer' field is deprecated:DeprecationWarning"
)


def _path(times, side_m):
    """A smooth closed path in a square of ``side_m`` metres, East and North."""
    phase = 2 * np.pi * (times - times[0])
    east = side_m / 2 * np.sin(phase / 17.0) * np.cos(phase / 53.0)
    north = side_m / 2 * np.sin(phase / 23.0)
    return east, north


def _write_session(
    path,
    *,
    description=SQUARE,
    offset=0.0,
    side_m=0.5,
    unit="",
    identifier="A7803-210726",
):
    """An NWB file laid out as DANDI 000987's: x, y, z and rx, ry, rz series.

    Sleep from 0 to 100 s, then one tracked arena epoch per arena the
    description names, each 300 s, with sleep between them. The floor is
    (x, -z) in metres, y the head's height; ``ry`` is the direction of
    movement in that frame plus ``offset``, in [0, 2π).
    """
    pynwb = pytest.importorskip("pynwb")
    from pynwb.behavior import CompassDirection, Position, SpatialSeries
    from pynwb.epoch import TimeIntervals
    from pynwb.file import Subject

    nwb = pynwb.NWBFile(
        description,
        identifier,
        datetime.datetime(2024, 5, 9, 21, 31, tzinfo=datetime.timezone.utc),
        subject=Subject(subject_id="A7803", species="Mus musculus"),
    )
    n_arenas = description.count("arena")
    bounds = [(100.0 + 400.0 * k, 400.0 + 400.0 * k) for k in range(n_arenas)]
    times = np.concatenate(
        [np.arange(start + 0.5, stop - 0.5, 1 / RATE) for start, stop in bounds]
    )
    east, north = _path(times, side_m)
    heading = np.arctan2(np.gradient(north), np.gradient(east)) + offset
    rng = np.random.default_rng(1)
    height = 0.05 + 0.01 * rng.standard_normal(times.size)
    position = Position(name="Position")
    for name, values in (("x", east), ("y", height), ("z", -north)):
        position.add_spatial_series(
            SpatialSeries(
                name=name, data=values, timestamps=times, reference_frame="", unit=unit
            )
        )
    compass = CompassDirection(name="CompassDirection")
    for name, values in (
        ("rx", np.zeros_like(times)),
        ("ry", np.mod(heading, 2 * np.pi)),
        ("rz", np.zeros_like(times)),
    ):
        compass.add_spatial_series(
            SpatialSeries(
                name=name,
                data=values,
                timestamps=times,
                reference_frame="",
                unit="radian",
            )
        )
    nwb.add_acquisition(position)
    nwb.add_acquisition(compass)
    edges = [0.0]
    for start, stop in bounds:
        nwb.add_epoch(start_time=edges[-1], stop_time=start, tags=["sleep"])
        nwb.add_epoch(start_time=start, stop_time=stop, tags=["wake"])
        edges.append(stop)
    nwb.add_epoch(start_time=edges[-1], stop_time=edges[-1] + 100.0, tags=["sleep"])
    for name, rows in (
        ("SWS", [(10.0, 50.0)]),
        ("UP", [(10.0, 10.5), (11.0, 11.4)]),
        ("DOWN", [(10.5, 11.0)]),
        ("position_time_support", [(t[0], t[-1]) for t in _split(times, bounds)]),
    ):
        table = TimeIntervals(name=name, description=name)
        for start, stop in rows:
            table.add_interval(start_time=float(start), stop_time=float(stop))
        nwb.add_time_intervals(table)
    nwb.add_unit_column("location", "the anatomical location of this unit")
    for _ in range(3):
        nwb.add_unit(
            spike_times=np.sort(rng.uniform(1.0, edges[-1] + 99.0, 300)),
            location="RSC",
        )
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return times, east, north, heading


def _split(times, bounds):
    return [times[(times >= start) & (times < stop)] for start, stop in bounds]


@pytest.fixture
def entry(session_entry):
    return session_entry("peyrache_rsc")


def test_floor_is_x_and_minus_z_in_cm_and_ry_is_wrapped(tmp_path, entry):
    """Metres to cm, the vertical y left out, ry wrapped with its directions kept."""
    from neurospatial.datasets import peyrache_rsc

    path = tmp_path / "session.nwb"
    times, east, north, heading = _write_session(path)
    rec = run_checks(peyrache_rsc.load_session(entry, path), stated=entry.info.stated)
    assert rec.failed_checks == ()
    np.testing.assert_array_equal(rec.times, times)
    np.testing.assert_allclose(rec.positions, np.column_stack([east, north]) * 100.0)
    assert rec.headings.min() >= -np.pi
    assert rec.headings.max() <= np.pi
    np.testing.assert_allclose(
        np.exp(1j * rec.headings), np.exp(1j * heading), atol=1e-12
    )
    assert rec.env_kind == "inferred"
    assert rec.env.units == "cm"
    assert set(rec.epochs) == {
        "sleep",
        "wake",
        "sws",
        "up",
        "down",
        "tracking",
        "square",
    }
    np.testing.assert_array_equal(rec.epochs["square"], rec.epochs["tracking"])
    np.testing.assert_array_equal(rec.params["analysis_epochs"], rec.epochs["tracking"])
    assert list(rec.units["brain_region"]) == ["RSC"] * 3
    bases = {c.text.split(":")[0]: c.basis for c in rec.conversions}
    assert bases["positions"] == "inferred"
    assert bases["headings"] == "inferred"
    positions = next(c for c in rec.conversions if c.text.startswith("positions"))
    assert "mirroring the positions and the heading together" in positions.evidence
    assert "the identifier A7803-210726 reads as 2021-07-26" in rec.clock


def test_heading_that_follows_movement_is_reported_as_following(tmp_path, entry):
    """No offset: mean near 0 and a tight spread, and ry negated spreads wider."""
    from neurospatial.datasets import peyrache_rsc

    path = tmp_path / "session.nwb"
    _write_session(path, offset=0.0)
    rec = peyrache_rsc.load_session(entry, path)
    (line,) = [w for w in rec.watch_outs if "heading frame" in w]
    assert line.startswith("The heading frame follows movement")
    match = re.search(r"mean ([+-]\d\.\d\d) rad, spread (\d+) degrees", line)
    assert abs(float(match.group(1))) <= 0.05
    sign = next(c for c in rec.conversions if c.text.startswith("headings"))
    negated, stored = re.search(
        r"ry negated spreads (\d+) degrees per tracked epoch and ry as stored (\d+)",
        sign.evidence,
    ).groups()
    assert int(negated) > int(stored)


def test_offset_heading_is_not_verified(tmp_path, entry):
    """A constant 0.5 rad offset is measured and the frame is not verified."""
    from neurospatial.datasets import peyrache_rsc

    path = tmp_path / "session.nwb"
    _write_session(path, offset=0.5)
    rec = peyrache_rsc.load_session(entry, path)
    (line,) = [w for w in rec.watch_outs if "heading frame" in w]
    assert line.startswith("The heading frame is not verified by movement")
    mean = float(re.search(r"mean ([+-]\d\.\d\d) rad", line).group(1))
    assert mean == pytest.approx(0.5, abs=0.05)


def test_two_arenas_name_their_epochs_and_warn_one_environment(tmp_path, entry):
    from neurospatial.datasets import peyrache_rsc

    path = tmp_path / "session.nwb"
    _write_session(path, description=TWO)
    rec = peyrache_rsc.load_session(entry, path)
    tracking = rec.epochs["tracking"]
    np.testing.assert_array_equal(rec.epochs["square"], tracking[:1])
    np.testing.assert_array_equal(rec.epochs["circle"], tracking[1:])
    assert any("visits 2 arenas (square, circle)" in w for w in rec.watch_outs)
    (frame,) = [w for w in rec.watch_outs if "heading frame" in w]
    assert frame.count(" s, mean ") == 2  # one comparison per tracked epoch


def test_description_that_does_not_match_the_epochs_names_none():
    """Two arenas named but one tracked epoch: no epoch is named by arena."""
    from neurospatial.datasets import peyrache_rsc

    arenas, lines = peyrache_rsc._arena_epochs(TWO, np.array([[0.0, 1.0]]))
    assert arenas == {}
    assert lines == ()


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"unit": "meters"}, "position units"),
        ({"side_m": 50.0}, "spans"),  # cm stored as if metres
    ],
)
def test_refuses_a_file_that_no_longer_reads_as_blank_metres(
    tmp_path, entry, kwargs, why
):
    from neurospatial.datasets import peyrache_rsc

    path = tmp_path / "session.nwb"
    _write_session(path, **kwargs)
    with pytest.raises(ValueError, match="metres-with-y-vertical") as raised:
        peyrache_rsc.load_session(entry, path)
    assert why in str(raised.value)


@pytest.mark.realdata
@OLD_DEVICE
def test_loads_default_session(session_entry, assert_default_session):
    """The survey's session: 37 RSC units, 100 Hz, the square tracked 1203 s."""
    rec = load("peyrache_rsc")
    assert_default_session(rec, session_entry("peyrache_rsc"))
    assert rec.n_units == 37
    assert rec.times.size == 120_287
    assert rec.sampling_rate == pytest.approx(100.0, rel=1e-6)
    np.testing.assert_allclose(rec.epochs["square"], [[3880.15755, 5082.9515]])
    spans = np.ptp(rec.positions, axis=0)
    np.testing.assert_allclose(spans, [52.70, 53.16], atol=0.01)
    (frame,) = [w for w in rec.watch_outs if "heading frame" in w]
    assert "mean +0.51 rad, spread 49 degrees" in frame
    sign = next(c for c in rec.conversions if c.text.startswith("headings"))
    assert "ry negated spreads 87 degrees" in sign.evidence
    assert rec.headings.min() < -3.0
    assert rec.headings.max() > 3.0


@pytest.mark.realdata
@OLD_DEVICE
def test_whole_session_rates_warn_about_the_sleep_spikes():
    """Two thirds of the spikes are in the sleep before tracking: rates warn, epochs not."""
    import warnings

    from neurospatial import compute_spatial_rates

    rec = load("peyrache_rsc")
    with pytest.warns(UserWarning, match=r"\(66%\) .* fell outside the position time"):
        compute_spatial_rates(rec.env, rec.spike_times, rec.times, rec.positions)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_spatial_rates(
            rec.env,
            rec.spike_times,
            rec.times,
            rec.positions,
            epochs=rec.params["analysis_epochs"],
        )
