from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from _record import Recording
from operations import nwb_readers

pynwb = pytest.importorskip("pynwb")


def _write(path, *, heading_unit: str, generic_position: bool = False) -> Recording:
    """A small NWB file with position, head direction (radians) and two units."""
    from pynwb.behavior import CompassDirection, Position

    rng = np.random.default_rng(0)
    times = np.arange(600) / 30.0
    positions = np.cumsum(rng.normal(0, 0.5, (600, 2)), axis=0) + 50.0
    headings = np.angle(np.exp(1j * np.cumsum(rng.normal(0, 0.1, 600))))
    spikes = [np.sort(rng.uniform(0, times[-1], 40)) for _ in range(2)]
    nwbfile = pynwb.NWBFile(
        session_description="reader test",
        identifier="reader-test",
        session_start_time=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
    )
    behavior = nwbfile.create_processing_module("behavior", "behavior")
    if generic_position:
        nwbfile.add_acquisition(
            pynwb.TimeSeries(
                name="track_data",
                data=positions,
                unit="cm",
                timestamps=times,
                description="columns=x,y",
            )
        )
    else:
        position = Position(name="Position")
        position.create_spatial_series(
            name="position",
            data=positions,
            unit="cm",
            timestamps=times,
            reference_frame="corner",
        )
        behavior.add(position)
    compass = CompassDirection(name="CompassDirection")
    compass.create_spatial_series(
        name="head_direction",
        data=headings,
        unit=heading_unit,
        timestamps=times,
        reference_frame="0 is east",
    )
    behavior.add(compass)
    for train in spikes:
        nwbfile.add_unit(spike_times=train)
    with pynwb.NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)
    return Recording(
        dataset="reader_test",
        session="0",
        source=str(path),
        license="n/a",
        times=times,
        positions=positions,
        position_units="cm",
        headings=headings,
        spike_times=tuple(spikes),
        unit_ids=np.arange(2),
        params={"bin_size": 2.0, "dt": 0.25, "nwb_path": str(path)},
    )


def test_readers_flag_mislabelled_heading_unit(tmp_path):
    rec = _write(tmp_path / "degrees.nwb", heading_unit="degrees")
    outcome = nwb_readers(rec, {})
    assert outcome.status == "workaround"
    assert outcome.summary["head_direction"].startswith("mismatch")
    assert outcome.summary["position"] == "ok"
    assert outcome.summary["units"] == "ok"


def test_readers_ok_on_well_formed_file(tmp_path):
    rec = _write(tmp_path / "radians.nwb", heading_unit="radians")
    outcome = nwb_readers(rec, {})
    assert outcome.status == "ok", outcome.summary
    assert set(outcome.summary.values()) == {"ok"}


def test_readers_find_position_in_generic_timeseries(tmp_path):
    # Documents current behavior: a generic TimeSeries is not found as position.
    rec = _write(
        tmp_path / "generic.nwb", heading_unit="radians", generic_position=True
    )
    outcome = nwb_readers(rec, {})
    assert outcome.summary["position"] == "not found, but the loader found it"
    assert outcome.status == "workaround"
