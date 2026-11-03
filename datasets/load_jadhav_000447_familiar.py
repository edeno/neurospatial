"""Shin & Jadhav rat CA1 and PFC units on the familiar W-track (DANDI 000447, run 2).

The same file and session as ``load_jadhav_000447``, which documents the
recording, the reading and the analysis choices; this loader returns the
file's second run epoch, on the familiar W-track. The familiar and novel
W-tracks lie rotated 90 degrees in one camera frame, so each is its own
recording with its own declared environment (``load_jadhav_000447.load_maze``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _record import Range, Recording, check_expectations, validate_recording
from load_jadhav_000447 import HAZARDS, load_maze

DATASET = {
    "key": "jadhav_000447_familiar",
    "title": "Shin & Jadhav, rat JDS-NFN-AM2, CA1 and PFC units on the familiar W-track",
    "source": "https://dandiarchive.org/dandiset/000447/0.250724.2224",
    "license": "CC-BY-4.0",
    "download_size": "5.05 GB (streamed)",
}

# As registered for load_jadhav_000447 (the same file), before the split.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 57,
    "n_dims": 2,
    "sampling_rate_hz": Range(31.0, 35.0),
}

__all__ = ["DATASET", "EXPECTATIONS", "HAZARDS", "load"]


def load(cache: Path | None = None) -> Recording:
    """Stream the session; the familiar W-track's run, x, y in cm, spikes, trajectories."""
    del cache  # read by byte range: nothing is cached
    return load_maze("familiar", DATASET)


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
