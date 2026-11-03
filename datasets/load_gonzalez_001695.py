"""Buzsaki lab mouse CA1, CA3 and RSC units on a novel maze with sleep before and after (DANDI 001695).

What was recorded: mouse M02, 12 March 2024, a silicon probe in CA1, CA3 and
retrosplenial cortex (110 units: 80 CA1, 21 CA3, 9 RSC), sleep, then a novel
linear maze for 31 min (4,583-6,474 s), then sleep. Position at 25 Hz. The
sleep-state table also lists 2,798 ripples.

Where it lives: DANDI 001695, version 0.260319.2023, CC-BY-4.0; the smallest
file with units and position (61.3 MB). The smaller ``_ecephys`` files of the
set are head-fixed sessions with no position.

What the loader converts: ``AnimalPosition/Position`` is labelled
``unit="centimeters"`` with ``conversion=0.01``. NWB defines the value as
stored x conversion in the stated unit, which here gives a 1.1 cm "track".
The stored values span about 108 by 6, a linear track in cm, and the file's
own ``Speed`` series (cm/s) matches the speed computed from the stored values
(median ratio near 1), not one 100 times smaller. So the loader reads the
stored values with h5py (``.data[:]`` without the conversion) and keeps them
as cm. The reference frame says "(0,0) is center of arena", but every value is
positive; the frame is not used. Ripples are the ``state == "Ripple"`` rows of
``intervals/SleepStates``; the WAKE, NREM and REM rows become epochs.

Analysis choices (``params``):

- ``bin_size`` 2.5 cm; ``min_speed`` 4 cm/s, as for the other tracks.
- ``dt`` 0.25 s and alternating 60 s ``train_epochs``/``test_epochs`` over the
  maze session (the file has no trial table).
- ``replay_dt`` 0.02 s; ``events["ripples"]``: the first 200 ripples after the
  maze (post-maze sleep), decoded with the maze's rate maps. Only 6 ripples
  fall inside the tracked maze session.
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
    "key": "gonzalez_001695",
    "title": "Buzsaki lab, mouse M02, CA1/CA3/RSC units on a novel maze with pre and post sleep",
    "source": "https://dandiarchive.org/dandiset/001695/0.260319.2023",
    "license": "CC-BY-4.0",
    "download_size": "61.3 MB",
}

# Pre-registered from the dataset catalog (position stored with conversion
# 0.01) and the file's metadata (110 units; position labelled "centimeters"
# with conversion 0.01, stored values 6.6-114 by 13-19, 25 Hz; a Speed series
# in cm/s; ripples listed among the sleep states) before the first run of this
# loader. The extent expectation says the stored values, not the converted
# ones, are centimetres.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 110,
    "n_dims": 2,
    "sampling_rate_hz": Range(24.0, 26.0),
    "track_length_cm": lambda rec: Range(90, 130).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "has_ripples": lambda rec: "ripples" in rec.events,
}


_PATH = "sub-M02/sub-M02_ses-20240312T100000_behavior+ecephys.nwb"
_ASSET_ID = "1e4d5403-a8cc-4814-a904-7aff57f8cc4d"
_SHA256 = "323e99a1ec1b028a35da8cd955f2671a1f9e1853b46622c19863424943c962db"
_SERIES = "processing/behavior/AnimalPosition/Position"
# A plausible linear maze for a mouse, in cm: the longest span must lie here.
_TRACK_CM = (80.0, 150.0)
_N_REPLAYED = 200


def load(cache: Path | None = None) -> Recording:
    """Fetch the session; positions in cm (stored values), spikes, ripples."""
    import h5py

    from neurospatial.io.nwb import read_units

    path = fetch(
        dandi_asset_url(_ASSET_ID),
        Path(_PATH).name,
        sha256=_SHA256,
        subdir=DATASET["key"],
        cache=cache,
    )
    with h5py.File(path, "r") as h5:
        series = h5[_SERIES]
        stored = np.asarray(series["data"][:], dtype=float)
        times = np.asarray(series["timestamps"][:], dtype=float)
        label = series["data"].attrs["unit"]
        conversion = float(series["data"].attrs["conversion"])
        file_speed = np.asarray(h5["processing/behavior/Speed/data"][:], dtype=float)
        table = h5["intervals/SleepStates"]
        states = np.asarray(table["state"][:]).astype(str)
        windows = np.column_stack(
            [table["start_time"][:], table["stop_time"][:]]
        ).astype(float)
    with open_nwb(str(path)) as nwbfile:
        units = read_units(nwbfile)

    span = np.ptp(stored, axis=0)
    step = np.linalg.norm(np.diff(stored, axis=0), axis=1) / np.diff(times)
    moving = step > 2.0
    ratio = float(np.median(file_speed[1:][moving] / step[moving]))
    if (
        label != "centimeters"
        or conversion != 0.01
        or not _TRACK_CM[0] <= span.max() <= _TRACK_CM[1]
        or not 0.5 <= ratio <= 2.0
    ):
        msg = (
            f"{_SERIES} no longer matches the cm-stored-with-conversion-0.01 quirk"
            f" (unit {label!r}, conversion {conversion}, span {span.round(1)},"
            f" file speed / stored-value speed {ratio:.2f}); recheck"
        )
        raise ValueError(msg)
    ripples = windows[states == "Ripple"]
    after_maze = ripples[ripples[:, 0] > times[-1]][:_N_REPLAYED]
    train, test = alternating_blocks(times[0], times[-1], 60.0)
    return Recording(
        dataset=DATASET["key"],
        session="M02-20240312",
        source=f"{DATASET['source']} {_PATH} (asset {_ASSET_ID})",
        license=DATASET["license"],
        times=times,
        positions=stored,
        position_units="cm",
        spike_times=tuple(np.asarray(s, dtype=float) for s in units.spike_times),
        unit_ids=np.asarray(units.unit_ids),
        epochs={
            "maze": np.array([[times[0], times[-1]]]),
            **{
                label.lower(): windows[states == label]
                for label in ("WAKE", "NREM", "REM")
            },
        },
        events={"ripples": after_maze, "all_ripples": ripples},
        params={
            "bin_size": 2.5,
            "min_speed": 4.0,
            "dt": 0.25,
            "replay_dt": 0.02,
            "train_epochs": train,
            "test_epochs": test,
            "nwb_source": str(path),
            "readers_used": ("units",),
        },
        conversions=(
            f"positions: {_SERIES} stored values (h5py, conversion {conversion} not"
            f" applied), cm; range checked: {stored[:, 0].min():.1f}-"
            f"{stored[:, 0].max():.1f} by {stored[:, 1].min():.1f}-"
            f"{stored[:, 1].max():.1f} cm; the file's Speed (cm/s) / speed from the"
            f" stored values: median {ratio:.2f}",
            "events['ripples']: the first 200 SleepStates rows with state 'Ripple'"
            f" after the maze ({len(after_maze)} of {len(ripples)}; "
            f"{int(np.sum((ripples[:, 0] >= times[0]) & (ripples[:, 1] <= times[-1])))}"
            " fall inside the maze session)",
            "epochs: SleepStates rows WAKE, NREM, REM by label",
        ),
    )


def _conversion_applied(rec: Recording) -> Recording:
    # What read_position and pynwb return: stored x 0.01, still labelled cm.
    assert rec.positions is not None
    return replace(rec, positions=rec.positions * 0.01)


HAZARDS = {"conversion_applied_labelled_cm": _conversion_applied}


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
