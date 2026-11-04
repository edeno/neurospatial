"""Peyrache lab mouse retrosplenial cortex units in a square arena with tracked head direction (DANDI 000987).

What was recorded: mouse A7803, 37 retrosplenial units on silicon probes,
sleep (0-3,878 s) then a square arena (3,878-5,085 s), with motion-capture
tracking of a rigid body on the head at 100 Hz (3,880-5,083 s). The file also
holds slow-wave sleep and UP/DOWN state intervals.

The session is read by ``neurospatial.datasets.load("peyrache_rsc",
"sub-A7803_ses-20240509T213129")``, whose module
(``neurospatial/datasets/peyrache_rsc.py``) and docs page give the file, the
conversions (blank-unit metres with y vertical, the floor (x, -z) in cm;
``ry`` wrapped to [-pi, pi] as heading, its frame not verified by movement:
mean offset +0.51 rad, spread 49 degrees, against 87 degrees with ``ry``
negated), the environment inferred from the samples (the preprint's
80 cm-wide wall panels do not give the square's side; the samples' square is
52.7 x 53.2 cm, G-130) and the watch-outs, among them the held runs that may
be lost tracking. ``_record.from_public`` converts it to the survey's record.

Analysis choices are the public loader's ``params``: ``bin_size`` 2.5 cm,
``analysis_epochs`` the tracked span (``position_time_support``), ``dt``
0.25 s and alternating 60 s train/test blocks.

Survey-only: ``params["nwb_source"]`` (the cached file, which the
``vertical_axis_as_floor`` hazard reads) and ``params["readers_used"]``, for
the ``io.nwb_readers`` operation.
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
    heading_vs_movement,
    public_file,
    validate_recording,
)

from neurospatial.datasets._fetch import open_nwb

DATASET = {
    "key": "rsc_000987",
    "public": "peyrache_rsc",
    "session": "sub-A7803_ses-20240509T213129",
}

# Pre-registered from the dataset catalog (head direction plus 2-D, blank position
# units) and the file's metadata (37 units; x, y, z and rx, ry, rz series of
# 120,287 samples over 1203 s) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 37,
    "n_dims": 2,
    "sampling_rate_hz": Range(99.0, 101.0),
    "arena_extent_cm": lambda rec: Range(40, 100).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
    "headings_follow_movement": lambda rec: bool(
        abs(
            heading_vs_movement(rec.times, rec.positions, rec.headings, min_speed=5.0)[
                0
            ]
        )
        < np.pi / 4
    ),
}


def load(cache: Path | None = None) -> Recording:
    """The session from ``neurospatial.datasets``; floor positions in cm, yaw."""
    from neurospatial.datasets import load as load_public

    rec = load_public(DATASET["public"], DATASET["session"], cache=cache)
    source = public_file(DATASET["public"], DATASET["session"], cache=cache)
    return from_public(
        rec,
        key=DATASET["key"],
        params={
            "nwb_source": SurveyOnly(
                source, "the cached NWB file the session was read from"
            ),
            "readers_used": SurveyOnly(
                ("units",), "the io.nwb readers the public loader calls"
            ),
        },
    )


def _vertical_axis_as_floor(rec: Recording) -> Recording:
    # A user taking the first two series (x, y) as the floor plane: y is the
    # head's height, so the arena collapses to a strip a few cm wide.
    assert rec.positions is not None
    with open_nwb(rec.params["nwb_source"]) as nwbfile:
        height = np.asarray(nwbfile.acquisition["Position"]["y"].data[:], dtype=float)
    return replace(
        rec, positions=np.column_stack([rec.positions[:, 0], height * 100.0])
    )


# The blank unit read as cm (metres 100 times too small) is the generic
# length_unit_off_by_100 probe.
HAZARDS = {"vertical_axis_as_floor": _vertical_axis_as_floor}


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
