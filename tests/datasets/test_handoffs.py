"""The calls the dataset guide shows for moving a recording to pynapple and NWB."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest


def test_recording_to_pynapple(square_recording):
    nap = pytest.importorskip("pynapple")
    from neurospatial.io import to_pynapple

    rec = square_recording
    position = to_pynapple(rec.times, rec.positions)
    heading = to_pynapple(rec.times, rec.headings)
    spikes = nap.TsGroup(
        {int(u): nap.Ts(t) for u, t in zip(rec.units.index, rec.spike_times)}
    )
    np.testing.assert_array_equal(position.values, rec.positions)
    np.testing.assert_array_equal(heading.values, rec.headings)
    assert len(spikes) == rec.n_units
    for unit, train in zip(rec.units.index, rec.spike_times):
        np.testing.assert_array_equal(spikes[int(unit)].t, train)


def test_recording_environment_to_nwb(square_recording, tmp_path):
    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    rec = square_recording
    nwbfile = pynwb.NWBFile(
        "Session", "session", datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    write_environment(nwbfile, rec.env, name=rec.session)
    path = tmp_path / "session.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        env = read_environment(io.read(), name=rec.session)
    assert env.n_bins == rec.env.n_bins
    np.testing.assert_allclose(env.bin_centers, rec.env.bin_centers)
