"""Peyrache et al. (2015): a synthetic NWB file in the OSF layout, and the real one."""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from neurospatial.datasets import _fetch, load
from neurospatial.datasets._catalog import sessions
from neurospatial.datasets._checks import run_checks

DEFAULT = "Mouse32-140822"
WAKE = (100.0, 400.0)


def _entry(session=DEFAULT):
    return next(
        entry for entry in sessions("peyrache_2015") if entry.session == session
    )


def _write_session(path, *, gap=None):
    """An NWB file laid out as the OSF tutorial file: ry in [0, 2π), tagged epochs."""
    pynwb = pytest.importorskip("pynwb")
    from pynwb.behavior import CompassDirection, SpatialSeries

    nwb = pynwb.NWBFile(
        "Mouse32-140822",
        "synthetic",
        datetime.datetime(2022, 3, 3, tzinfo=datetime.timezone.utc),
    )
    times = np.arange(WAKE[0] + 0.1, WAKE[1] - 0.1, 0.0256)
    if gap is not None:
        times = times[(times < gap[0]) | (times > gap[1])]
    rng = np.random.default_rng(0)
    ry = np.mod(np.cumsum(rng.normal(0, 0.05, times.size)), 2 * np.pi)
    compass = CompassDirection(name="CompassDirection")
    compass.add_spatial_series(
        SpatialSeries(
            name="ry", data=ry, timestamps=times, reference_frame="", unit="radian"
        )
    )
    nwb.add_acquisition(compass)
    for start, stop, tag in ((0.0, WAKE[0] - 1e-4, "sleep"), (*WAKE, "wake")):
        nwb.add_epoch(start_time=start, stop_time=stop, tags=[tag])
    nwb.add_epoch(start_time=WAKE[1], stop_time=900.0, tags=["sleep"])
    for name, rows in (
        ("sws", [(10.0, 50.0), (500.0, 700.0)]),
        ("rem", [(60.0, 80.0)]),
    ):
        table = pynwb.epoch.TimeIntervals(name=name, description=name)
        for start, stop in rows:
            table.add_interval(start_time=start, stop_time=stop)
        nwb.add_time_intervals(table)
    nwb.add_unit_column("location", "the anatomical location of this unit")
    for location in ("thalamus", "adn", "adn"):
        spikes = np.sort(rng.uniform(0.0, 900.0, 400))
        nwb.add_unit(spike_times=spikes, location=location)
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return times, ry


def test_loads_a_heading_only_session(tmp_path):
    from neurospatial.datasets import peyrache_2015

    path = tmp_path / "session.nwb"
    times, ry = _write_session(path)
    entry = _entry()
    rec = run_checks(peyrache_2015.load_session(entry, path), stated=entry.info.stated)
    assert rec.failed_checks == ()
    assert rec.env is None
    assert rec.positions is None
    np.testing.assert_array_equal(rec.times, times)
    # Wrapped to [-π, π], the same directions as stored.
    assert rec.headings.min() >= -np.pi
    assert rec.headings.max() <= np.pi
    np.testing.assert_allclose(np.exp(1j * rec.headings), np.exp(1j * ry), atol=1e-12)
    np.testing.assert_array_equal(rec.epochs["wake"], [WAKE])
    np.testing.assert_array_equal(
        rec.epochs["sleep"], [[0.0, WAKE[0] - 1e-4], [WAKE[1], 900.0]]
    )
    np.testing.assert_array_equal(rec.epochs["sws"], [[10.0, 50.0], [500.0, 700.0]])
    np.testing.assert_array_equal(rec.params["analysis_epochs"], [WAKE])
    assert list(rec.units["brain_region"]) == ["thalamus", "adn", "adn"]
    frame = next(c for c in rec.conversions if "0 = East" in c.text)
    assert frame.basis == "inferred"
    assert "reference_frame is empty" in frame.evidence
    assert rec.license == "not stated"
    assert rec.provenance.archive == "OSF jb2gd"


def test_tracking_gaps_are_counted(tmp_path):
    from neurospatial.datasets import peyrache_2015

    path = tmp_path / "session.nwb"
    times, _ = _write_session(path, gap=(200.0, 205.0))
    rec = peyrache_2015.load_session(_entry(), path)
    longest = np.diff(times).max()
    assert f"1 gap longer than 1 s ({longest:.1f} s in all, the longest " in "".join(
        rec.watch_outs
    )


def test_missing_extra_names_the_install(monkeypatch, tmp_path, without_pynwb):
    def no_download(*args, **kwargs):
        msg = "a file was downloaded"
        raise AssertionError(msg)

    monkeypatch.setattr(_fetch, "fetch", no_download)
    with pytest.raises(ImportError, match=r"pip install 'neurospatial\[datasets\]'"):
        load("peyrache_2015", cache=tmp_path)


@pytest.mark.realdata
# The file's devices carry the empty 'manufacturer' field pynwb now deprecates.
@pytest.mark.filterwarnings(
    "ignore:The 'manufacturer' field is deprecated:DeprecationWarning"
)
def test_loads_default_session():
    """The tutorials' Mouse32-140822: 49 units, 31 in ADn, wake and sleep epochs."""
    rec = load("peyrache_2015")
    assert rec.session == DEFAULT
    assert rec.verified
    assert rec.failed_checks == ()
    assert rec.n_units == _entry().n_units == 49
    adn = np.flatnonzero(rec.units["brain_region"].to_numpy() == "adn")
    np.testing.assert_array_equal(adn, np.arange(4, 35))
    assert {"wake", "sleep", "sws", "rem"} <= set(rec.epochs)
    assert rec.env is None
    assert rec.headings.min() < -3.0
    assert rec.headings.max() > 3.0
    assert rec.sampling_rate == pytest.approx(39.0625, rel=1e-6)
    np.testing.assert_allclose(rec.epochs["wake"], [[8812.3, 10771.2999]])
    assert rec.provenance.sha256 == _entry().sha256
