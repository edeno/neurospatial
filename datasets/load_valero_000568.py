"""Valero et al. 2022 mouse CA1 units on a 110 cm linear track with ripples (DANDI 000568).

What was recorded: mouse fCamk2, 14 Oct 2020, session 3: 35 CA1 units across
five epochs (14,794 s); the third (7,674-9,941 s) is a linear track (a PVC
track "110 cm long, 6.35 cm wide", per the series description), tracked at
30 Hz by an overhead camera. The file also holds 49 ripples, sleep states
and up/down states.

Where it lives: DANDI 000568, version 0.230705.1633, CC-BY-4.0; a
"no-raw-data" file of 5.8 MB.

What the loader converts: nothing changes units (cm). Positions are read with
``read_position``. The video ``ImageSeries`` timestamps run 22.55-23.12 s earlier
than the position timestamps for the same 67,131 frames; the position
timestamps are the ones aligned with the spikes: the mean spatial information
of the 35 units peaks at that alignment (1.09 bits/spike) and drops to 0.81
with the video timestamps and to at most 0.85 at any shift of 5-40 s in either
direction (checked once while writing this loader). The track runs along y
(0.5-116.5 cm); x spans 4-21 cm, wider than the 6.35 cm track, as the LED
moves with the head. Lost tracking is set to NaN
(``_record.mask_lost_tracking``): 46 runs of 10 or more identical frames
(the longest held 6 s), which the tracker filled with its last fix, and
samples reached faster than 100 cm/s (a mouse does not run that fast; the
fast steps are the straight-line bridges after a held run). The counts are
in ``conversions``.

Analysis choices (``params``):

- ``bin_size`` 2.5 cm; ``min_speed`` 4 cm/s.
- ``dt`` 0.25 s, alternating 60 s ``train_epochs``/``test_epochs`` over the
  track epoch (the file has no trial table).
- ``replay_dt`` 0.02 s; ``events["ripples"]``: the 24 ripples inside the
  tracked track epoch.
- Declared environment (``geometry["environment"]``): the stated 110 cm by
  6.35 cm track as a one-edge graph along the samples' principal axis,
  centred on them (``_tracks.straight_track``); the fit is in ``conversions``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import dandi_asset_url, fetch, open_nwb
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    mask_lost_tracking,
    validate_recording,
)
from _tracks import declared_geometry, straight_placement, straight_track

DATASET = {
    "key": "valero_000568",
    "title": "Valero et al. (2022), mouse fCamk2 session 3, CA1 units on a linear track",
    "source": "https://dandiarchive.org/dandiset/000568/0.230705.1633",
    "license": "CC-BY-4.0",
    "download_size": "5.8 MB",
}

# Pre-registered from the dataset catalog and the file's description
# ("PVC linear track (110 cm long, 6.35 cm wide)"), and its metadata (35 units,
# position in cm at 30 Hz, a ripple table) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 35,
    "n_dims": 2,
    "sampling_rate_hz": Range(29.0, 31.0),
    "track_length_cm": lambda rec: Range(100, 125).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "has_ripples": lambda rec: "ripples" in rec.events,
}


_PATH = (
    "sub-fCamk2/"
    "sub-fCamk2_ses-fCamk2-201014-sess3-no-raw-data_behavior+ecephys+image.nwb"
)
_ASSET_ID = "48f2daa0-752b-4de5-932a-6e4682478493"
_SHA256 = "17bc28bed791fad03b3a87702a77f2f8c74ca5daac0a8e8dbc742df3506b1790"
# The track is 110 cm; the tracked span along it must lie here (cm).
_TRACK_CM = (100.0, 130.0)
_LENGTH_CM, _WIDTH_CM = 110.0, 6.35


def _windows(table: object) -> np.ndarray:
    frame = table.to_dataframe()  # type: ignore[attr-defined]
    return frame[["start_time", "stop_time"]].to_numpy(dtype=float)


def load(cache: Path | None = None) -> Recording:
    """Fetch the session; positions in cm, spikes, epochs, ripples."""
    from neurospatial.io.nwb import read_position, read_units

    path = fetch(
        dandi_asset_url(_ASSET_ID),
        Path(_PATH).name,
        sha256=_SHA256,
        subdir=DATASET["key"],
        cache=cache,
    )
    with open_nwb(str(path)) as nwbfile:
        position = read_position(nwbfile)
        units = read_units(nwbfile)
        video_times = np.asarray(
            nwbfile.acquisition["ImageSeriesTrackingVideo1"].timestamps[:],
            dtype=float,
        )
        epochs = nwbfile.epochs.to_dataframe()
        ripples = _windows(nwbfile.processing["ecephys"]["RippleTimeIntervals"])
        states = nwbfile.processing["behavior"]["SleepStates"].to_dataframe()

    positions = np.asarray(position.positions, dtype=float)
    times = np.asarray(position.times, dtype=float)
    span = np.ptp(positions, axis=0)
    offset = times - video_times
    if (
        position.units != "cm"
        or not _TRACK_CM[0] <= span.max() <= _TRACK_CM[1]
        or not 22.5 <= offset.min() <= offset.max() <= 23.2
    ):
        msg = (
            f"position unit {position.units!r}, span {span.round(1)}, position minus"
            f" video timestamps {offset.min():.3f}-{offset.max():.3f} s; recheck"
        )
        raise ValueError(msg)
    # A mouse does not run 100 cm/s on a 110 cm track (99.9th percentile of
    # its steps in this file: 218 cm/s, from held-then-bridged frames).
    positions, tracking_line = mask_lost_tracking(
        times, positions, min_frozen=10, max_speed=100.0
    )
    windows = epochs[["start_time", "stop_time"]].to_numpy(dtype=float)
    track = windows[(windows[:, 0] <= times[0]) & (windows[:, 1] >= times[-1])]
    on_track = (ripples[:, 0] >= times[0]) & (ripples[:, 1] <= times[-1])
    train, test = alternating_blocks(times[0], times[-1], 60.0)
    graph, fit = straight_track(positions, _LENGTH_CM)
    declared, declared_line = declared_geometry(
        graph,
        positions,
        times=times,
        kind=f"straight track {_LENGTH_CM:g} cm",
        source="NWB series description '110 cm long, 6.35 cm wide'",
        bin_size=2.5,
        physical_width=_WIDTH_CM,
        units="cm",
        name=DATASET["key"],
        placement=straight_placement(fit, "cm"),
        placed_by="samples' principal axis",
        stated_lengths={"length": (fit["samples_span"], _LENGTH_CM)},
        tracked_point="not stated",
        speed_thresholds=(4.0, 30.0),
    )
    return Recording(
        dataset=DATASET["key"],
        session="fCamk2_201014_sess3",
        source=f"{DATASET['source']} {_PATH} (asset {_ASSET_ID})",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        spike_times=tuple(np.asarray(s, dtype=float) for s in units.spike_times),
        unit_ids=np.asarray(units.unit_ids),
        epochs={
            "track": track,
            **{
                str(label).lower(): states.loc[
                    states["label"] == label, ["start_time", "stop_time"]
                ].to_numpy(dtype=float)
                for label in sorted(states["label"].unique())
            },
        },
        events={"ripples": ripples[on_track], "all_ripples": ripples},
        geometry={
            "track_length_cm": _LENGTH_CM,
            "track_width_cm": _WIDTH_CM,
            **declared,
        },
        params={
            "bin_size": 2.5,
            "min_speed": 4.0,
            "dt": 0.25,
            "replay_dt": 0.02,
            "train_epochs": train,
            "test_epochs": test,
            "nwb_source": str(path),
            "readers_used": ("position", "units"),
        },
        conversions=(
            f"positions: read_position, cm; range checked: {np.nanmin(positions[:, 0]):.1f}-"
            f"{np.nanmax(positions[:, 0]):.1f} by {np.nanmin(positions[:, 1]):.1f}-"
            f"{np.nanmax(positions[:, 1]):.1f} cm (track 110 cm along y)",
            tracking_line,
            "times: position timestamps, not the video timestamps"
            f" ({offset.min():.2f}-{offset.max():.2f} s earlier for the same"
            " frames); spatial information peaks at the position timestamps",
            f"events['ripples']: the {int(on_track.sum())} of {len(ripples)}"
            " RippleTimeIntervals rows inside the track tracking",
            "epochs: the epoch holding the tracking as track; SleepStates by label",
            declared_line,
        ),
    )


def _video_timestamps(rec: Recording) -> Recording:
    # A user taking the frame times from the video ImageSeries instead.
    with open_nwb(rec.params["nwb_source"]) as nwbfile:
        video_times = np.asarray(
            nwbfile.acquisition["ImageSeriesTrackingVideo1"].timestamps[:],
            dtype=float,
        )
    return replace(rec, times=video_times)


HAZARDS = {"video_timestamps": _video_timestamps}


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
