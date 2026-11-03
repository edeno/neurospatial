"""Neurolab STS-90 rat CA1 place cells recorded in spaceflight, Escher staircase and baseline track (DANDI 001754).

What was recorded: rat 2 on flight day 4 of the Space Shuttle Neurolab
mission (April 1998), tetrodes in CA1 (72 units), five task sessions: three
baseline sessions on a rectangular track (BL), the "Escher staircase", a
three-dimensional track with 90 degree yaw and pitch turns (ES), and the flat
"Magic Carpet" (MC). The file also holds the authors' 64 x 64 rate and
occupancy maps from the original analysis.

Where it lives: DANDI 001754, version 0.260728.1352, CC-BY-4.0; a 3.2 MB
behavior+ecephys file.

What the loader converts: position is overhead-video pixel coordinates
(0-255; "Video tracker pixel coordinates") of a three-dimensional track seen
from one camera, and it is "sampled at spike occurrence times": the 25,320
position samples sit at exactly the 25,320 distinct spike times of the 72
units, so there is no position between spikes (median gap 20-70 ms within a
session, 90th percentile 0.3-0.5 s). The file gives no pixel-to-cm
calibration, so positions stay in pixels. 2,218 samples are exactly
(0, 255), a corner of the image that no tracked sample otherwise reaches
(x is exactly 0 on one other sample only): the tracker's code for a lost LED. The loader sets
them to NaN. Sessions are epochs by type.

Analysis choices (``params``):

- ``bin_size`` 4 pixels (64 bins across the image, the authors' own map
  resolution).
- ``analysis_epochs``: the Escher staircase session (3,850-4,800 s); the five
  sessions are different apparatus at different places in the image, so one
  environment from all of them would mix tracks.
- ``dt`` 0.25 s, alternating 60 s ``train_epochs``/``test_epochs`` within
  the staircase session.
- Occupancy from samples at spike times weights each position by the time to
  the next spike of any unit, not by time spent; nothing here can recover
  time-based occupancy, so rate maps are as good as the sampling allows.
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
    validate_recording,
)

DATASET = {
    "key": "neurolab_001754",
    "title": "Knierim, McNaughton et al., Neurolab flight day 4, rat 2, CA1 units",
    "source": "https://dandiarchive.org/dandiset/001754/0.260728.1352",
    "license": "CC-BY-4.0",
    "download_size": "3.2 MB",
}

# Pre-registered from the dataset catalog (pixel coordinates 0-255, Escher
# staircase, position sampled at spike times) and the file's metadata (72
# units; 25,320 position samples labelled pixels; five epochs BL, ES, BL, BL,
# MC) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 72,
    "n_dims": 2,
    "position_min": Range([0, 0], [255, 255]),
    "position_max": Range([0, 0], [255, 255]),
    "pixels": lambda rec: rec.position_units == "pixel",
    "five_task_epochs": lambda rec: sum(len(v) for v in rec.epochs.values()) == 5,
    "position_sampled_at_spikes": lambda rec: bool(
        np.all(np.isin(rec.times, np.concatenate(rec.spike_times)))
    ),
}


_PATH = "sub-Rat2/sub-Rat2_ses-19980420T095700_behavior+ecephys.nwb"
_ASSET_ID = "697481a1-9460-4f66-a8d2-ab41310ec0ce"
_SHA256 = "773b3221d507fb19a7f4f5af2dabb648cc0b6c8e41acbe7846b2ac048a289775"
_LOST = (0.0, 255.0)


def load(cache: Path | None = None) -> Recording:
    """Fetch the session; positions in pixels at spike times, spikes, sessions."""
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
        sessions = nwbfile.epochs.to_dataframe()

    raw = np.asarray(position.positions, dtype=float)
    times = np.asarray(position.times, dtype=float)
    spikes = np.unique(np.concatenate([np.asarray(s) for s in units.spike_times]))
    lost = (raw == _LOST).all(axis=1)
    if (
        position.units != "px"
        or raw.min() < 0
        or raw.max() > 255
        or not np.array_equal(times, spikes)
        or np.sum(raw[~lost, 0] == 0) > 0.001 * len(raw)
    ):
        msg = (
            "position no longer matches pixels 0-255 sampled at spike times with"
            " (0, 255) for a lost LED; recheck"
        )
        raise ValueError(msg)
    positions = np.where(lost[:, None], np.nan, raw)
    windows = sessions[["start_time", "stop_time"]].to_numpy(dtype=float)
    kinds = sessions["session_type"].to_numpy().astype(str)
    staircase = windows[kinds == "ES"]
    train, test = alternating_blocks(staircase[0, 0], staircase[0, 1], 60.0)
    return Recording(
        dataset=DATASET["key"],
        session="Rat2-FD4-19980420",
        source=f"{DATASET['source']} {_PATH} (asset {_ASSET_ID})",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="pixel",
        spike_times=tuple(np.asarray(s, dtype=float) for s in units.spike_times),
        unit_ids=np.asarray(units.unit_ids),
        epochs={kind: windows[kinds == kind] for kind in dict.fromkeys(kinds)},
        geometry={"image_pixels": 256, "sampled_at_spike_times": True},
        params={
            "bin_size": 4.0,
            "dt": 0.25,
            "analysis_epochs": staircase,
            "train_epochs": train,
            "test_epochs": test,
            "nwb_source": str(path),
            "readers_used": ("units",),
        },
        conversions=(
            "positions: read_position, pixels, no calibration in the file; range"
            f" checked: {np.nanmin(positions[:, 0]):.0f}-{np.nanmax(positions[:, 0]):.0f}"
            f" by {np.nanmin(positions[:, 1]):.0f}-{np.nanmax(positions[:, 1]):.0f} px",
            f"positions: {int(lost.sum())} samples at exactly (0, 255), the lost-LED"
            " code, set to NaN; other samples with x exactly 0:"
            f" {int(np.sum(raw[~lost, 0] == 0))}, kept",
            f"times: the {len(times)} distinct spike times of all units (position is"
            " sampled at spikes)",
            "epochs: sessions by type (BL, ES, MC)",
        ),
    )


def _lost_led_kept(rec: Recording) -> Recording:
    # What read_position returns: the lost-LED code (0, 255) as a position.
    assert rec.positions is not None
    lost = np.isnan(rec.positions).any(axis=1)
    return replace(
        rec, positions=np.where(lost[:, None], np.asarray(_LOST), rec.positions)
    )


HAZARDS = {"lost_led_code_kept": _lost_led_kept}


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
