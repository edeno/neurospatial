"""Huszar & Buzsaki mouse CA1 and CA3 units on a cheeseboard maze with ripples (DANDI 001880)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "ca1ca3_001880",
    "title": "Huszar & Buzsaki, mouse iuvE14-3, CA1 and CA3 units, cheeseboard place-reward task",
    "source": "https://dandiarchive.org/dandiset/001880/0.261007.1849",
    "license": "CC0-1.0",
    "download_size": "65.7 MB",
}

# Pre-registered from the dataset catalog (place-reward task, about 5 % NaN
# position, ripples) and the file's metadata (49 units, position in cm at
# 30 Hz, PRE / Learning / POST sub-sessions, a Ripples table) before the first
# run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 49,
    "n_dims": 2,
    "sampling_rate_hz": Range(29.0, 31.0),
    "position_nan_fraction": Range(0.03, 0.08),
    "maze_extent_cm": lambda rec: Range(80, 120).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "has_ripples": lambda rec: "ripples" in rec.events,
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
