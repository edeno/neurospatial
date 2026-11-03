"""Grosmark & Buzsaki 2016 rat CA1 on a 1.6 m linear track with sleep before and after (DANDI 000044).

What was recorded: rat Achilles, 25 Oct 2013 (the session also published as
CRCNS hc-11 ``Achilles_10252013``), 137 CA1 units, PRE sleep, running on a
1.6 m linear track (MAZE) and POST sleep, position at 39.06 Hz.

Where it lives: DANDI 000044 version 0.261005.2101, CC-BY-4.0, an 8.7 GB file
read by HTTP byte range. Reading, conversions, candidate events and analysis
choices are those of ``load_grosmark_000044_circular`` (shared code there).
On a linear track the linearized coordinate's 0 is a real place, the start of
the track. Only 16 % of the tracked maze-epoch samples are on the track (the
linearized coordinate is defined); the rest lie off it, so the 2-D positions
span about 2 m in each direction while the track is 1.6 m long.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording, check_expectations, validate_recording
from load_grosmark_000044_circular import load_session

DATASET = {
    "key": "grosmark_000044_linear",
    "title": "Grosmark & Buzsaki (2016), Achilles 10252013, rat CA1, 1.6 m linear track",
    "source": "https://dandiarchive.org/dandiset/000044/0.261005.2101",
    "license": "CC-BY-4.0",
    "download_size": "8.74 GB (streamed)",
}

# Pre-registered from the dataset catalog (137 units, 39.06 Hz, 1.6 m linear track, PRE/MAZE/POST epochs) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 137,
    "n_dims": 2,
    "sampling_rate_hz": Range(38.0, 40.0),
    "position_extent_cm": lambda rec: Range(150, 175).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "epoch_names": ["maze", "post", "pre"],
}


_ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"
_SHA256 = "ff10511d44d53f634c82331f345f6839574b8926ba037bde94988adfd70c0ef3"
_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"


def load(cache: Path | None = None) -> Recording:
    """Stream the session; positions in cm, spikes, epochs and candidate events."""
    del cache  # read by byte range: nothing is cached
    return load_session(
        dataset=DATASET,
        asset_id=_ASSET_ID,
        sha256=_SHA256,
        path=_PATH,
        track="1.6mLinearMaze",
        span_cm=(150.0, 260.0),
        track_cm=(150.0, 170.0),
    )


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
