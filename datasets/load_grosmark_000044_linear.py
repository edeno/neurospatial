"""Grosmark & Buzsaki 2016 rat CA1 on a 1.6 m linear track with sleep before and after (DANDI 000044).

What was recorded: rat Achilles, 25 Oct 2013 (the session also published as
CRCNS hc-11 ``Achilles_10252013``, and on OSF as ``grosmark_2016``'s default
session), 137 CA1 units, PRE sleep, running on a 1.6 m linear track (MAZE)
and POST sleep, position at 39.06 Hz.

The session is read by ``neurospatial.datasets.load("grosmark_2016",
"sub-Achilles_ses-Achilles-10252013", stream=True)``, the 8.7 GB file read
by HTTP byte range, through ``load_grosmark_000044_circular.dandi_session``
(the survey-only candidate events, replay bin and NWB source are described
there). The public module (``neurospatial/datasets/grosmark_2016.py``) and
its docs page give the conversions, the declared environment (the paper's
1.9 m maze, a 1.6 m stem with a 15 cm reward area at each end, as a
straight graph along the line the file's linearized coordinate traces) and
the watch-outs, among them the rat's first minutes off the track.

Analysis choices are the public loader's ``params``: ``bin_size`` 2 cm,
``min_speed`` 5 cm/s, ``dt`` 0.25 s and alternating 60 s train/test blocks
of the tracked span.

Survey-only, beside those of ``dandi_session``: the declared track's facts
the environment fidelity study reads (``_tracks.public_track_facts``),
declared from the file's linearized coordinate, no stated width.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording, check_expectations, validate_recording
from _tracks import public_track_facts
from load_grosmark_000044_circular import dandi_session

DATASET = {
    "key": "grosmark_000044_linear",
    "public": "grosmark_2016",
    "session": "sub-Achilles_ses-Achilles-10252013",
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


def load(cache: Path | None = None) -> Recording:
    """Stream the session; positions in cm, spikes, epochs and candidate events."""
    del cache  # read by byte range: nothing is cached
    return dandi_session(
        DATASET["key"],
        DATASET["session"],
        geometry=lambda rec: public_track_facts(
            rec,
            declared_from="file's linearized coordinate",
            physical_width=None,
            speed_thresholds=(rec.params["min_speed"], 30.0),
        ),
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
