"""Shipley, Barry et al. 2026 mouse CA1 on an 8-arm radial maze (UCL RDR 30902972).

What was recorded: mouse m4005 on 2020-09-26, 16 tetrodes in CA1 sorted
with Kilosort 2 and curated in Phy (71 clusters labelled ``good``), over 11
trials on an 8-arm radial maze (``0`` to ``10``) and a 45-minute rest
recording (``S2``), on the clock of the day's recordings combined. One
tracking LED.

The session is read by ``neurospatial.datasets.load("shipley_2025")``,
whose module (``neurospatial/datasets/shipley_2025.py``) and docs page give
the four zip members read by byte range, the conversions (spike samples /
30 kHz; pixels / 274 pixels per metre; position times rebuilt per trial from
the spike-to-sample map, the camera at 30.05 Hz, not the header's 30 Hz,
with the residuals reported), the declared star (the stated 80 cm arms from
a 24.7 cm platform, placed on the maze trials' samples) and the watch-outs
(``pos.dir`` equals ``dir_disp``, so no headings). ``_record.from_public``
converts it to the survey's record.

Analysis choices are the public loader's ``params``: ``bin_size`` 2 cm and
``min_speed`` 2 cm/s (the paper's; this survey used 3 cm and 3 cm/s before
the paper was read, and no expectation depends on them);
``analysis_epochs`` the 11 maze trials; even and odd trials train and test
the decoder at ``dt`` 0.25 s.

Survey-only: the declared track's facts the environment fidelity study
reads (``_tracks.public_track_facts``: placed on the samples, the stated 10
cm arm width, each arm's reach against the stated 92.35 cm).
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

DATASET = {
    "key": "shipley_30902972",
    "public": "shipley_2025",
    "session": "m4005_2020-09-26",
}

# Pre-registered from the zip listing (11 maze trials and a rest recording
# S2) and trialData's header (30 Hz, 274 pixels per metre, one LED) and
# cluster_info.tsv (71 clusters labelled good) before the first run of this
# loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 71,
    "n_dims": 2,
    "sampling_rate_hz": Range(29.9, 30.1),
    "maze_extent_cm": lambda rec: Range(140.0, 200.0).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "eleven_maze_trials": lambda rec: len(rec.epochs["maze"]) == 11,
    "one_led_no_heading": lambda rec: rec.headings is None,
}


def load(cache: Path | None = None) -> Recording:
    """The session from ``neurospatial.datasets``; positions in cm, good units' spikes."""
    from neurospatial.datasets import load as load_public
    from neurospatial.datasets import shipley_2025

    rec = load_public(DATASET["public"], DATASET["session"], cache=cache)
    graph = rec.env.binning.linearization.graph
    centre = np.asarray(graph.nodes["centre"]["pos"])
    reach = shipley_2025.PLATFORM / 2 + shipley_2025.ARM_LENGTH
    maze = np.zeros(len(rec.times), dtype=bool)
    for start, stop in rec.params["analysis_epochs"]:
        maze |= (rec.times >= start) & (rec.times <= stop)
    stated_lengths = {}
    for k in range(shipley_2025.N_ARMS):
        end = np.asarray(graph.nodes[f"arm_{k}"]["pos"])
        u = (end - centre) / np.linalg.norm(end - centre)
        rel = rec.positions[maze] - centre
        along = rel @ u
        on_arm = (np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0]) < 10.0) & (
            along > shipley_2025.PLATFORM / 2
        )
        if on_arm.any():
            stated_lengths[f"arm {k}"] = (
                float(np.quantile(along[on_arm], 0.99)),
                reach,
            )
    return from_public(
        rec,
        key=DATASET["key"],
        geometry=public_track_facts(
            rec,
            declared_from="fitted to the samples",
            physical_width=shipley_2025.ARM_WIDTH,
            speed_thresholds=(rec.params["min_speed"], 30.0),
            stated_lengths=stated_lengths,
        ),
    )


def clock_from_header_rate(rec: Recording) -> Recording:
    # A reader trusting the header's 30 Hz: sample index / 30, which runs
    # 8 s past the spikes by the end of the session.
    assert rec.times is not None
    return replace(rec, times=rec.times[0] + np.arange(len(rec.times)) / 30.0)


def pixels_labelled_cm(rec: Recording) -> Recording:
    # The stored xy are pixels; a reader that missed pixels_per_metre would
    # call them cm (2.74 x too large).
    assert rec.positions is not None
    return replace(rec, positions=rec.positions * 2.74)


HAZARDS = {
    "clock_from_header_rate": clock_from_header_rate,
    "pixels_labelled_cm": pixels_labelled_cm,
}


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s")
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
