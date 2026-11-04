"""Valero et al. 2022 mouse CA1 units on a 110 cm linear track with ripples (DANDI 000568).

What was recorded: mouse fCamk2, 14 Oct 2020, session 3: 35 CA1 units across
five epochs (14,794 s); the third (7,674-9,941 s) is a linear track (a PVC
track "110 cm long, 6.35 cm wide", per the series description), tracked at
30 Hz by an overhead camera. The file also holds 49 ripples, sleep states
and up/down states.

The session is read by ``neurospatial.datasets.load("valero_2022",
"sub-fCamk2_ses-fCamk2-201014-sess3")``, whose module
(``neurospatial/datasets/valero_2022.py``) and docs page give the file, the
conversions (the position timestamps, not the video's, which run 22.55-23.12
s earlier), the declared track (the stated 110 cm as a one-edge graph along
the samples' principal axis) and the watch-outs, among them the held runs
that may be lost tracking. ``_record.from_public`` converts it to the
survey's record.

Analysis choices are the public loader's ``params``: ``bin_size`` 2.5 cm,
``dt`` 0.25 s and alternating 60 s train/test blocks of the tracked span.

Survey-only: ``params["min_speed"]`` 4 cm/s (in place of the public None),
which ``behavior.segment_by_velocity`` needs; ``params["replay_dt"]`` 0.02
s; ``events["ripples"]``, the ripples inside the tracked track epoch, which
replay is decoded on (the public ``ripples``, every ripple of the session,
are ``events["all_ripples"]``); ``params["nwb_source"]`` and
``params["readers_used"]``, for the ``io.nwb_readers`` operation and the
``video_timestamps`` hazard.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _record import (
    Range,
    Recording,
    SurveyOnly,
    check_expectations,
    from_public,
    public_file,
    validate_recording,
)

from neurospatial.datasets._fetch import open_nwb

DATASET = {
    "key": "valero_000568",
    "public": "valero_2022",
    "session": "sub-fCamk2_ses-fCamk2-201014-sess3",
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


def load(cache: Path | None = None) -> Recording:
    """The session from ``neurospatial.datasets``; positions in cm, ripples."""
    from neurospatial.datasets import load as load_public

    rec = load_public(DATASET["public"], DATASET["session"], cache=cache)
    source = public_file(DATASET["public"], DATASET["session"], cache=cache)
    ripples = rec.events["ripples"]
    on_track = (ripples[:, 0] >= rec.times[0]) & (ripples[:, 1] <= rec.times[-1])
    return from_public(
        rec,
        key=DATASET["key"],
        params={
            "replay_dt": SurveyOnly(0.02, "the replay decoding bin"),
            "min_speed": SurveyOnly(
                4.0,
                "the survey's running threshold, 4 cm/s, which"
                " behavior.segment_by_velocity needs; no source read states one",
            ),
            "nwb_source": SurveyOnly(
                source, "the cached NWB file the session was read from"
            ),
            "readers_used": SurveyOnly(
                ("position", "units"), "the io.nwb readers the public loader calls"
            ),
        },
        events={
            "ripples": SurveyOnly(
                ripples[on_track],
                f"the {int(on_track.sum())} of {len(ripples)} ripples inside the"
                " tracking, so replay is decoded where the rate maps were fitted",
            ),
            "all_ripples": SurveyOnly(np.array(ripples), "every ripple of the session"),
        },
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
