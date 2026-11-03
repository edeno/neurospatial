"""Tirole, Huelin Gorriz, Takigawa, Kukovska & Bendor rat CA1 on two linear tracks (Zenodo 7114733)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "takigawa_7114733",
    "title": "Tirole et al. (2022), rat 5 session 1, CA1 units on two novel 2 m linear tracks",
    "source": "https://doi.org/10.5281/zenodo.7114733",
    "license": "CC0",
    "download_size": "20.0 MB",
}

# Pre-registered from the record's README (25 Hz camera, two 2 m tracks,
# rest before, between and after) and the files' headers (109 clusters,
# track lengths 2 m) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 109,
    "n_dims": 2,
    "sampling_rate_hz": Range(24.9, 25.1),
    "two_tracks": lambda rec: len(rec.epochs["track_1"]) == 1
    and len(rec.epochs["track_2"]) == 1,
    "track_extent_cm": lambda rec: Range(180.0, 240.0).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
