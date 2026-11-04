"""Forli, Fan, Qi & Yartsev (2025) bat hippocampus during 3-D flight (Zenodo demo session).

What was recorded: an Egyptian fruit bat flying freely in a room for about
90 minutes (38 flights) while three Neuropixels probes recorded hippocampal
units. Position is 3-D (metres) at 120 Hz from the room's tracking system.

The session is read by ``neurospatial.datasets.load("forli_2025")``, whose
module (``neurospatial/datasets/forli_2025.py``) and docs page give the
files (members of Zenodo 15738988's ``Demo_Session.zip``, read by byte
range), the conversions (spike times in s, putative duplicates pooled, the
1 ms rule, flights clipped to the neural recording), the declared room
(every voxel of a grid tiling ``r_lim``) and the watch-outs.
``_record.from_public`` converts it to the survey's record.

Analysis choices are the public loader's ``params``: ``bin_size`` 0.2 m,
``dt`` 0.25 s, ``min_speed`` the file's ``v_th`` (0.5 m/s), and the flights
during the neural recording as ``analysis_epochs``, with even and odd
flights as ``train_epochs`` and ``test_epochs``.
"""

from __future__ import annotations

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

DATASET = {
    "key": "yartsev_bat_demo",
    "public": "forli_2025",
    "session": "Dataset_3_14543_240419",
}

# Pre-registered from the record's demo script and a metadata probe of the
# behavior file, before the first run of this loader. y is not constrained:
# some samples lie outside the room bounds the file states.
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 3,
    "sampling_rate_hz": Range(119.0, 121.0),
    "position_min": Range((-3.1, -np.inf, -0.2), (np.inf, np.inf, np.inf)),
    "position_max": Range((-np.inf, -np.inf, -np.inf), (3.1, np.inf, 2.5)),
    "probe1_at_most_24_units": lambda rec: (
        sum(str(uid).startswith("probe1-") for uid in rec.unit_ids) <= 24
    ),
    "flights_present": lambda rec: len(rec.epochs["flight"]) == 38,
}


def load(cache: Path | None = None) -> Recording:
    """The session from ``neurospatial.datasets``; metres, 3-D."""
    from neurospatial.datasets import load as load_public

    rec = load_public(DATASET["public"], DATASET["session"], cache=cache)
    return from_public(rec, key=DATASET["key"])


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s")
    print(
        f"  positions {np.nanmin(rec.positions, 0)} to {np.nanmax(rec.positions, 0)} m"
    )
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
