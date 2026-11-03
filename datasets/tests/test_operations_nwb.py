from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from _record import Recording
from operations import nwb_readers

pynwb = pytest.importorskip("pynwb")


def _write(
    path,
    *,
    heading_unit: str,
    generic_position: bool = False,
    position_unit: str = "cm",
) -> Recording:
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
            unit=position_unit,
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
        position_units="pixel" if position_unit == "pixels" else "cm",
        headings=headings,
        spike_times=tuple(spikes),
        unit_ids=np.arange(2),
        params={"bin_size": 2.0, "dt": 0.25, "nwb_source": str(path)},
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


def test_readers_compare_times(tmp_path):
    from dataclasses import replace

    rec = _write(tmp_path / "clock.nwb", heading_unit="radians")
    shifted = replace(rec, times=rec.times + 1.0)
    outcome = nwb_readers(shifted, {})
    assert outcome.summary["position"] == "mismatch: times differ"
    assert outcome.summary["head_direction"] == "mismatch: times differ"
    assert outcome.status == "workaround"


def test_self_comparison_is_labelled(tmp_path):
    from dataclasses import replace

    rec = _write(tmp_path / "used.nwb", heading_unit="radians")
    used = replace(rec, params={**rec.params, "readers_used": ("units",)})
    outcome = nwb_readers(used, {})
    assert outcome.summary["units"] == "ok (loader used this reader)"
    assert outcome.summary["position"] == "ok"
    assert outcome.status == "ok"


def test_readers_compare_pixels_with_pixels(tmp_path):
    rec = _write(
        tmp_path / "pixels.nwb", heading_unit="radians", position_unit="pixels"
    )
    outcome = nwb_readers(rec, {})
    assert outcome.summary["position"] == "ok"


def test_same_times_tolerance_is_absolute():
    # 18,000 s into a 30 Hz session, a 0.18 s offset is five frames: the
    # relative tolerance of np.allclose would call the clocks the same.
    from operations import _same_times

    times = 18_000.0 + np.arange(1000) / 30.0
    assert _same_times(times, times + 1e-6)
    assert not _same_times(times, times + 0.18)
    assert not _same_times(times, times + 0.5 / 30.0)
