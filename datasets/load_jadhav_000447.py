"""Shin & Jadhav rat CA1 and PFC units on a W-track (DANDI 000447).

What was recorded: rat JDS-NFN-AM2, tetrodes in dorsal CA1 and prefrontal
cortex (57 units), running a W-track between three reward wells over three
run epochs (104 trajectories, each labelled with its start and end well,
inbound or outbound, and correct or not). Position at about 33 Hz.

Where it lives: DANDI 000447 version 0.250724.2224, CC-BY-4.0. The file is
5.1 GB, almost all raw voltage, so it is read by HTTP byte range
(``_fetch.open_nwb``) after checking DANDI's published SHA-256.

What the loader converts: ``Position/SpatialSeries`` has three columns under
the unit ``"centimeters; centimeters/second"``: x and y in cm, then speed in
cm/s (its values match the speed computed from x and y; see
``conversions``). The loader keeps x and y. Trajectories become epochs
(``trials``, ``inbound``, ``outbound``) and their wells ``events``.

Analysis choices (``params``): ``bin_size`` 2.5 cm; ``min_speed`` 4 cm/s, as
for the other tracks; ``dt`` 0.25 s; trajectories alternate between training
and testing the decoder.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import dandi_stream_url, open_nwb
from _record import (
    Range,
    Recording,
    check_expectations,
    checked_span,
    validate_recording,
)

DATASET = {
    "key": "jadhav_000447",
    "title": "Shin & Jadhav, rat JDS-NFN-AM2, CA1 and PFC units on a W-track",
    "source": "https://dandiarchive.org/dandiset/000447/0.250724.2224",
    "license": "CC-BY-4.0",
    "download_size": "5.05 GB (streamed)",
}

# Pre-registered from the dataset catalog (57 units, about 32.9 Hz, x and y in cm plus a velocity column) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 57,
    "n_dims": 2,
    "sampling_rate_hz": Range(31.0, 35.0),
}


_ASSET_ID = "b048bd0a-48a4-4d19-ad96-fd879666bca5"
_SHA256 = "aac7a7c12e7d129f6c6733b462c1e4fa6ee1d4e6d0893af347c78baf3f7ec97d"
_PATH = "sub-JDS-NFN-AM2/sub-JDS-NFN-AM2_behavior+ecephys.nwb"


def load(cache: Path | None = None) -> Recording:
    """Stream the session; x, y in cm, spikes, trajectories."""
    from neurospatial.io.nwb import read_units

    del cache  # read by byte range: nothing is cached
    source = dandi_stream_url(_ASSET_ID, sha256=_SHA256)
    with open_nwb(source) as nwbfile:
        series = nwbfile.processing["behavior"]["Position"]["SpatialSeries"]
        data = np.asarray(series.data[:], dtype=float)
        times = np.asarray(series.timestamps[:], dtype=float)
        unit_label = series.unit
        units = read_units(nwbfile)
        trials = nwbfile.trials.to_dataframe()
        run_epochs = nwbfile.intervals["epoch intervals"].to_dataframe()

    if unit_label != "centimeters; centimeters/second" or data.shape[1] != 3:
        msg = f"expected x, y, speed under 'centimeters; centimeters/second', got {unit_label!r}"
        raise ValueError(msg)
    positions = data[:, :2]
    span = checked_span(positions, 80.0, 200.0)
    step_speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
    finite = np.isfinite(step_speed) & np.isfinite(data[1:, 2])
    agreement = float(np.corrcoef(step_speed[finite], data[1:, 2][finite])[0, 1])
    # A scale-free correlation cannot confirm cm/s; the magnitudes must agree too.
    moving = finite & (step_speed > 5.0)
    scale = float(np.median(data[1:, 2][moving]) / np.median(step_speed[moving]))
    if not 0.5 <= scale <= 2.0 or agreement < 0.5:
        msg = (
            f"column 2 is not the speed of x, y (scale {scale:.2f}, r {agreement:.2f})"
        )
        raise ValueError(msg)
    windows = trials[["start_time", "stop_time"]].to_numpy(dtype=float)
    inbound = trials["trajectory_type"].to_numpy() == 1
    return Recording(
        dataset=DATASET["key"],
        session="JDS-NFN-AM2",
        source=f"{DATASET['source']} {_PATH} (asset {_ASSET_ID})",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        spike_times=tuple(np.asarray(s, dtype=float) for s in units.spike_times),
        unit_ids=np.asarray(units.unit_ids),
        epochs={
            "run": run_epochs[["start_time", "stop_time"]].to_numpy(dtype=float),
            "trials": windows,
            "inbound": windows[inbound],
            "outbound": windows[~inbound],
        },
        events={
            "start_well": trials["start_well"].to_numpy(dtype=float),
            "end_well": trials["end_well"].to_numpy(dtype=float),
            "correct": trials["correct"].to_numpy(dtype=float),
        },
        params={
            "bin_size": 2.5,
            "min_speed": 4.0,
            "dt": 0.25,
            "train_epochs": windows[::2],
            "test_epochs": windows[1::2],
            "nwb_source": source,
            "readers_used": ("units",),
        },
        conversions=(
            f"positions: Position/SpatialSeries columns 0, 1 (cm); {span}; column 2"
            f" is speed in cm/s (median ratio {scale:.2f} to the speed from x, y while"
            f" moving, r = {agreement:.2f}), dropped",
            "epochs: trials table rows as trials, split by trajectory_type"
            " (1 inbound, 0 outbound); 'epoch intervals' as run",
        ),
    )


def _speed_read_as_a_dimension(rec: Recording) -> Recording:
    # A reader taking every column of the series as a position axis.
    assert rec.positions is not None
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.linalg.norm(np.diff(rec.positions, axis=0), axis=1) / np.diff(
            rec.times
        )
    return replace(
        rec, positions=np.column_stack([rec.positions, np.append(speed, np.nan)])
    )


HAZARDS = {"speed_read_as_a_dimension": _speed_read_as_a_dimension}


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    print(
        f"{rec.dataset}: {len(rec.spike_times or ())} units, {rec.times[-1] - rec.times[0]:.0f} s"
    )
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
