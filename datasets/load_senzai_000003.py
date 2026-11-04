"""Senzai & Buzsaki 2017 mouse dentate gyrus and CA3 units in open exploration and a theta maze (DANDI 000003).

What was recorded: mouse YutaMouse41 (POMC-Cre::Arch), 29 Aug 2015, a
silicon probe in the dentate gyrus and CA3 (23 units: 10 pyramidal, 3
granule, 1 mossy cell and others), 17,050 s of exploration, maze running and
sleep, with optogenetic pulses and dentate spikes as annotations. Two position
sensors at 39.0625 Hz for the whole recording.

Where it lives: DANDI 000003, version 0.260218.2052, CC-BY-4.0; the smallest
file of the set is 4.66 GB, almost all LFP and per-spike waveforms, so it is
read by HTTP byte range (``_fetch.open_nwb``) after checking DANDI's
published SHA-256: the loader reads the position series with pynwb
(``read_position`` searches only ``Position`` containers, and these are bare
``SpatialSeries`` in ``acquisition``), the spike times and the sleep-state
table, about 30 MB in all.

What the loader converts: ``acquisition/position_sensor0`` and ``1`` are
labelled ``unit="meters"`` with reference frame "unknown", but their values
run 117-499 (spans about 160 by 110): read as metres the arena would be
160 m wide, so they are video pixels. The file and the paper give no
pixel-to-cm calibration, so positions stay in pixels. The loader uses sensor 0
(0.04 % NaN; sensor 1 has 0.75 %). The two sensors sit about 14 px apart,
but the direction from one to the other does not follow the direction of
movement (``_record.heading_vs_movement`` spread 66-83 degrees), so no head
direction is derived from them. The series store a sampling rate, not
timestamps; the loader trusts the rate only because the duration it implies
(666,029 / 39.0625 Hz = 17,050.3 s) matches the LFP's (21,312,875 samples at
1,250 Hz) within 1 s and the last spike (17,050.2 s) falls within its last
10 s. The file has no epochs table and does not
say when the mouse was in the open field or on the maze; the sleep-state
table labels "awake" periods.

Analysis choices (``params``):

- ``bin_size`` 4 px (about 40 by 28 bins over the tracked area).
- ``analysis_epochs``: the "awake" rows of the sleep-state table
  (3,309 s); the rest is sleep, unlabelled stimulation blocks or transitions.
  The file does not separate the open field from the theta maze, so the
  awake periods of both are pooled into one environment and one set of rate
  maps; place maps here mix two apparatus seen in the same camera image.
- ``dt`` 0.25 s, alternating 60 s ``train_epochs``/``test_epochs`` inside the
  awake periods.

Environment: built from the samples (inferred). Positions are pixels with no
stated scale, and the open field and the theta maze share one image with no
epochs separating them, so no stated outline can be placed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    validate_recording,
)

from neurospatial.datasets._fetch import dandi_stream_url, open_nwb

DATASET = {
    "key": "senzai_000003",
    "title": "Senzai & Buzsaki (2017), YutaMouse41 150829, hippocampal units, open exploration and theta maze",
    "source": "https://dandiarchive.org/dandiset/000003/0.260218.2052",
    "license": "CC-BY-4.0",
    "download_size": "4.66 GB (streamed)",
}

# Pre-registered from the dataset catalog (position labelled "meters" with
# pixel-like values 117-500) and the file's metadata (23 units; two
# position_sensor series of 666,029 samples at 39.0625 Hz labelled meters,
# reference frame "unknown"; sleep states; no epochs table) before the first
# run of this loader. Positions stay in pixels unless the file or paper gives
# a calibration.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 23,
    "n_dims": 2,
    "sampling_rate_hz": Range(39.0, 39.2),
    "pixels": lambda rec: rec.position_units == "pixel",
    "values_are_not_metres": lambda rec: bool(
        np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0)) > 50
    ),
}


_PATH = "sub-YutaMouse41/sub-YutaMouse41_ses-YutaMouse41-150829_behavior+ecephys.nwb"
_ASSET_ID = "70aaf3be-9f95-40af-8abc-a8316cbe50ca"
_SHA256 = "bd02d1c6c50391d9e7dbfc3be5e743c556dda9537ef3d3b365fc4a1596655bc9"
_SENSOR = "acquisition/position_sensor0"
# Pixel coordinates of a camera image; a mouse arena in metres would span
# under 3 m.
_PIXEL_SPAN = (50.0, 1000.0)
# The duration implied by the sensor's rate must match the LFP's within this,
# and the last spike must fall in the last this-many seconds of it.
_DURATION_TOLERANCE_S = (1.0, 10.0)


def load(cache: Path | None = None) -> Recording:
    """Stream the session; sensor-0 positions in pixels, spikes, sleep states."""
    from neurospatial.io.nwb import read_units

    del cache  # read by byte range: nothing is cached
    source = dandi_stream_url(_ASSET_ID, sha256=_SHA256)
    with open_nwb(source) as nwbfile:
        series = nwbfile.acquisition[_SENSOR.split("/")[1]]
        positions = np.asarray(series.data[:], dtype=float)
        label, rate = series.unit, float(series.rate)
        start = float(series.starting_time)
        lfp = nwbfile.processing["ecephys"]["LFP"]["LFP"]
        lfp_duration = lfp.data.shape[0] / float(lfp.rate)
        units = read_units(nwbfile)
        states = nwbfile.processing["behavior"]["states"].to_dataframe()

    span = np.nanmax(positions, axis=0) - np.nanmin(positions, axis=0)
    if label != "meters" or not all(
        _PIXEL_SPAN[0] <= v <= _PIXEL_SPAN[1] for v in span
    ):
        msg = f"{_SENSOR}: unit {label!r}, span {span.round(1)}; recheck"
        raise ValueError(msg)
    duration = len(positions) / rate
    last_spike = max(float(np.max(t)) for t in units.spike_times if len(t))
    if (
        abs(duration - lfp_duration) > _DURATION_TOLERANCE_S[0]
        or not duration - _DURATION_TOLERANCE_S[1] <= last_spike - start <= duration
    ):
        msg = (
            f"{_SENSOR}: {len(positions)} samples at {rate} Hz last {duration:.1f} s,"
            f" but the LFP lasts {lfp_duration:.1f} s and the last spike is at"
            f" {last_spike:.1f} s; the stored rate cannot be trusted"
        )
        raise ValueError(msg)
    times = start + np.arange(len(positions)) / rate
    labels = states["label"].to_numpy().astype(str)
    windows = states[["start_time", "stop_time"]].to_numpy(dtype=float)
    awake = windows[labels == "awake"]
    blocks = [alternating_blocks(a, b, 60.0) for a, b in awake]
    return Recording(
        geometry={"tracked_point": "not stated (video pixels)"},
        dataset=DATASET["key"],
        session="YutaMouse41-150829",
        source=f"{DATASET['source']} {_PATH} (asset {_ASSET_ID})",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="pixel",
        spike_times=tuple(np.asarray(s, dtype=float) for s in units.spike_times),
        unit_ids=np.asarray(units.unit_ids),
        epochs={label: windows[labels == label] for label in dict.fromkeys(labels)},
        params={
            "bin_size": 4.0,
            "dt": 0.25,
            "analysis_epochs": awake,
            "train_epochs": np.vstack([train for train, _ in blocks]),
            "test_epochs": np.vstack([test for _, test in blocks]),
            "nwb_source": source,
            "readers_used": ("units",),
        },
        conversions=(
            f"positions: {_SENSOR} (a bare SpatialSeries, read with pynwb), labelled"
            " metres;"
            f" range checked: {np.nanmin(positions[:, 0]):.0f}-"
            f"{np.nanmax(positions[:, 0]):.0f} by {np.nanmin(positions[:, 1]):.0f}-"
            f"{np.nanmax(positions[:, 1]):.0f}: pixels, kept as pixels (no calibration);"
            " reference frame 'unknown', so whether pixel y runs up or down the image"
            " is not stated and y is kept as stored",
            f"times: starting_time {start} s + index / {rate} Hz; the implied"
            f" {duration:.1f} s matches the LFP's {lfp_duration:.1f} s and the last"
            f" spike at {last_spike:.1f} s",
            "headings: none; the direction between the two sensors does not follow"
            " movement",
            "epochs: sleep-state rows by label; params['analysis_epochs'] the awake rows",
        ),
    )


# No loader hazard: trusting the "meters" label (pixels x 100 as cm, bin_size
# 4) builds a 4,181 x 2,494 grid (97,993 active bins); the default
# diffusion_kde rate map on it passed 7 GB for one unit, so the probe cannot
# run in the survey (see GAPS.md).


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
