"""Shin & Jadhav rat CA1 and PFC on a W-track (DANDI 000447)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _record import Range, Recording

DATASET = {
    "key": "jadhav_000447",
    "title": "Shin & Jadhav, rat JDS-NFN-AM2, CA1 and PFC units on a W-track",
    "source": "https://dandiarchive.org/dandiset/000447/0.250724.2224",
    "license": "CC-BY-4.0",
    "download_size": "5.05 GB (streamed)",
}

# Pre-registered from the dataset catalog (57 units, about 32.9 Hz, x and y in cm plus a velocity column) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 57,
    "n_dims": 2,
    "sampling_rate_hz": Range(31.0, 35.0),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
