"""Aery Jones & Giocomo mouse MEC and CA1 units on an X maze (DANDI 001701).

What was recorded: mouse BaggySweatpants, day 15, running an alternation task
on an X maze with a Neuropixels probe in medial entorhinal cortex and CA1
(320 units). Position (cm) and head direction are tracked at 60 Hz for
967 s.

Where it lives: DANDI 001701, version 0.260120.0303, CC-BY-4.0; the smallest
file with units and position, 13.0 MB.

What the loader converts: head direction (``CompassDirection/head
direction``) is labelled ``unit="degrees"``, but its values lie in −3.14 to
3.14, so they are radians; the loader reads the values with pynwb and keeps
them as radians (``read_head_direction`` would convert them as degrees). The
frame is stated: "0 is east". Positions are cm, "(0,0) is bottom left corner".

Analysis choices (``params``):

- ``bin_size`` 2.5 cm.
- ``dt`` 0.25 s and alternating 60 s ``train_epochs``/``test_epochs``, as for
  the other open recordings.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import dandi_asset_url, fetch
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    validate_recording,
)

DATASET = {
    "key": "giocomo_001701",
    "title": "Giocomo lab, mouse MEC and CA1 Neuropixels units on an X maze",
    "source": "https://dandiarchive.org/dandiset/001701/0.260120.0303",
    "license": "CC-BY-4.0",
    "download_size": "13.0 MB",
}

# Pre-registered from the survey appendix before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 320,
    "n_dims": 2,
    "sampling_rate_hz": Range(59.0, 61.0),
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}

_PATH = (
    "sub-BaggySweatpants/"
    "sub-BaggySweatpants_ses-BaggySweatpants-DY15-g1_behavior+ecephys.nwb"
)
_ASSET_ID = "3f3d0b16-9b3e-42ac-a5e6-327829df1116"
_SHA256 = "5a2246041e421cd5b321adf9ccc40ba6f11379b40b08794c1b214590c50921f3"


def load(cache: Path | None = None) -> Recording:
    """Fetch the session; positions in cm, headings in radians, spikes."""
    import pynwb

    from neurospatial.io.nwb import read_position, read_units

    path = fetch(
        dandi_asset_url(_ASSET_ID),
        Path(_PATH).name,
        sha256=_SHA256,
        subdir=DATASET["key"],
        cache=cache,
    )
    with pynwb.NWBHDF5IO(path, "r", load_namespaces=True) as io:
        nwbfile = io.read()
        position = read_position(nwbfile)
        units = read_units(nwbfile)
        compass = nwbfile.processing["behavior"]["CompassDirection"]["head direction"]
        headings = np.asarray(compass.data[:], dtype=float)
        label = compass.unit

    finite = headings[np.isfinite(headings)]
    if (
        label != "degrees"
        or finite.max() > np.pi + 1e-6
        or finite.min() < -np.pi - 1e-6
    ):
        msg = "head direction no longer matches the radians-labelled-degrees quirk; recheck"
        raise ValueError(msg)
    times = np.asarray(position.times, dtype=float)
    train, test = alternating_blocks(times[0], times[-1], 60.0)
    return Recording(
        dataset=DATASET["key"],
        session="BaggySweatpantsDY15-g1",
        source=f"{DATASET['source']} {_PATH}",
        license=DATASET["license"],
        times=times,
        positions=np.asarray(position.positions, dtype=float),
        position_units="cm",
        headings=headings,
        spike_times=tuple(np.asarray(s, dtype=float) for s in units.spike_times),
        unit_ids=np.asarray(units.unit_ids),
        params={
            "bin_size": 2.5,
            "dt": 0.25,
            "train_epochs": train,
            "test_epochs": test,
            "nwb_source": str(path),
            "readers_used": ("position", "units"),
        },
        conversions=(
            "positions: read_position, cm; range checked: 3.6-79 by 3.9-94 cm",
            "headings: CompassDirection/head direction labelled degrees, values in"
            " [-pi, pi]; kept as radians (read with pynwb)",
        ),
    )


def _headings_read_as_degrees(rec: Recording) -> Recording:
    # What a reader trusting the "degrees" label returns: np.radians(radians).
    assert rec.headings is not None
    return replace(rec, headings=np.radians(rec.headings))


HAZARDS = {"headings_read_as_degrees": _headings_read_as_degrees}


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
