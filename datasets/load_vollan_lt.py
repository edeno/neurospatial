"""Vollan et al. 2025 rat MEC units on a linear track (EBRAINS 10.25493/R5FR-EDG).

What was recorded: rat 27764, Neuropixels in medial entorhinal cortex (389
units, 84 of them in the authors' one grid module), running a 200 cm linear
track (the head positions span about 2.1 m). Head position and head
direction come from an OptiTrack motion-capture system at 100 Hz.

The session is read by ``neurospatial.datasets.load("vollan_2025",
"lt_27764_1")``, whose module (``neurospatial/datasets/vollan_2025.py``) and
docs page give the file (``sharing_v4/navigation/lt/27764_1.mat``, a MATLAB
v7 ``Dsession`` read with ``mat-io``), the conversions (positions m to cm,
head direction kept as radians after a frame check against the direction of
movement, the authors' 5 cm/s speed filter kept as gaps in ``times``, the
listed session clipped to the samples), the declared 200 cm track along the
samples' principal axis and the watch-outs (55% of the samples after the
listed session). ``_record.from_public`` converts it to the survey's record.

Analysis choices are the public loader's ``params``: ``bin_size`` 2.5 cm;
``min_speed`` None (the data are already speed-filtered); ``max_gap`` 0.015
s; ``dt`` 0.25 s and alternating 60 s train/test blocks; the authors' grid
modules (``module``, ``is_grid``, per-module grid scores).

Survey-only: the declared track's facts the environment fidelity study reads
(``_tracks.public_track_facts``: placed on the samples' principal axis, no
stated width, the samples' span along the axis against the stated 200 cm).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _record import (
    Range,
    Recording,
    check_expectations,
    from_public,
    validate_recording,
)
from _tracks import public_track_facts

DATASET = {"key": "vollan_lt", "public": "vollan_2025", "session": "lt_27764_1"}

# Pre-registered from the dataset documentation (100 Hz, speed-filtered at 5 cm/s) and the file's unit table (389 MEC units) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 389,
    "n_dims": 2,
    "sampling_rate_hz": Range(99.0, 101.0),
    # No extent expectation: the public loader's units check refuses a span
    # outside 120-240 cm, so one here could never fail.
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
    "speed_filter_leaves_gaps": lambda rec: bool(np.max(np.diff(rec.times)) > 0.1),
}


def load(cache: Path | None = None) -> Recording:
    """The session from ``neurospatial.datasets``; positions in cm, headings in radians."""
    from neurospatial.datasets import load as load_public
    from neurospatial.datasets import vollan_2025
    from neurospatial.datasets._recording import tracked_mask

    rec = load_public(DATASET["public"], DATASET["session"], cache=cache)
    samples = rec.positions[tracked_mask(rec.positions)]
    _, _, _, (low, high) = vollan_2025.principal_axis(samples)
    return from_public(
        rec,
        key=DATASET["key"],
        geometry=public_track_facts(
            rec,
            declared_from="samples' principal axis",
            physical_width=None,
            speed_thresholds=(vollan_2025.SPEED_FILTER, 30.0),
            stated_lengths={"length": (high - low, vollan_2025.TRACK)},
        ),
    )


def times_rebuilt_from_dt(rec: Recording) -> Recording:
    # A reader trusting dt and nt: evenly spaced times that close the
    # speed-filter gaps, pulling every later sample earlier than its spikes.
    assert rec.times is not None
    dt = float(np.median(np.diff(rec.times)))
    return replace(rec, times=rec.times[0] + dt * np.arange(len(rec.times)))


HAZARDS = {"times_rebuilt_from_dt": times_rebuilt_from_dt}


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
