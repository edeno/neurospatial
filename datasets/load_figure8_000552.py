"""Huszár, Zhang, Blockus & Buzsáki mouse CA1 on a figure-8 maze (DANDI 000552).

What was recorded: 113 CA1 units in mouse ``e14_2m2`` across a sleep, maze,
sleep session (18,557 s). The file holds 21 rewarded alternation trials, position
at 30 Hz only on the maze, 4,297 ripples, and Awake, Non-REM and REM states.

The session is read by ``neurospatial.datasets.load("huszar_2022")``, whose
module (``neurospatial/datasets/huszar_2022.py``) and docs page give the file,
the conversions (camera positions in cm, the linearized position, a loop, in
``rec.linearized``), the declared figure eight with both loops closed, and the
watch-outs. ``_record.from_public`` converts it to the survey's record.

Analysis choices are the public loader's ``params``: ``bin_size`` 2 cm,
``min_speed`` 4 cm/s, ``dt`` 0.25 s, ``replay_dt`` 0.02 s, trials alternating
within each visited arm as the train and test epochs.

Survey-only: ``events["ripples"]``, the 71 ripples inside the maze epoch (awake
ripples, decoded with the maze's own rate maps; the 4,226 sleep ripples are not
replayed, to keep the shuffle tests to minutes), every row in
``events["all_ripples"]``; ``geometry["track_length_cm"]`` and
``geometry["lap_start_region"]``, a cut across the centre bar at its middle, which
every trial crosses once; the declared track's facts the environment fidelity
study reads; ``params["nwb_source"]`` and ``params["readers_used"]``.
"""

from __future__ import annotations

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
from _tracks import cut_region, public_track_facts

DATASET = {
    "key": "figure8_000552",
    "public": "huszar_2022",
    "session": "sub-e14-2m2_ses-e14-2m2-201019",
}

# Pre-registered from the dandiset description and a metadata probe of this
# file, before the first run of this loader.
# The positions became the camera (x, y) when the loop was declared; the
# registered range checks now apply to the linearized coordinate.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 113,
    "n_dims": 2,
    "linearized_min": lambda rec: Range(0.0, 222.0).contains(
        float(np.nanmin(rec.params["linearized"]))
    ),
    "linearized_max": lambda rec: Range(0.0, 222.0).contains(
        float(np.nanmax(rec.params["linearized"]))
    ),
    "ripples_present": lambda rec: (
        "ripples" in rec.events and len(rec.events["ripples"]) > 0
    ),
    "sleep_states_present": lambda rec: len(rec.epochs) > 0,
}


def load(cache: Path | None = None) -> Recording:
    """The session from ``neurospatial.datasets``; camera (x, y) in cm."""
    from neurospatial.datasets import load as load_public

    rec = load_public(DATASET["public"], DATASET["session"], cache=cache)
    ripples = rec.events["ripples"]
    ((start, stop),) = rec.epochs["maze"]
    in_maze = (ripples[:, 0] >= start) & (ripples[:, 1] <= stop)
    graph = rec.env.binning.linearization.graph
    bar = [np.asarray(graph.nodes[n]["pos"]) for n in ("bar_0", "bar_1")]
    why_events = "awake ripples only, to keep the shuffle tests to minutes"
    return from_public(
        rec,
        key=DATASET["key"],
        events={
            "ripples": SurveyOnly(ripples[in_maze], why_events),
            "all_ripples": SurveyOnly(ripples, why_events),
        },
        geometry={
            "track_length_cm": SurveyOnly(222.0, "the linearized coordinate's span"),
            "lap_start_region": SurveyOnly(
                cut_region(
                    "centre_bar",
                    0.5 * (bar[0] + bar[1]),
                    bar[1] - bar[0],
                    half_length=2.0,
                    half_width=10.0,
                ),
                "the lap operation's start, across the centre bar at its middle",
            ),
            **public_track_facts(
                rec,
                declared_from="fitted to the samples",
                physical_width=None,
                speed_thresholds=(4.0, 30.0),
            ),
        },
        params={
            "nwb_source": SurveyOnly(
                public_file(DATASET["public"], DATASET["session"], cache=cache),
                "the cached NWB file the session was read from",
            ),
            "readers_used": SurveyOnly(
                ("units",), "the io.nwb readers the public loader calls"
            ),
        },
    )


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(
        f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s tracked"
    )
    print(f"  ripples on the maze: {len(rec.events['ripples'])}")
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
