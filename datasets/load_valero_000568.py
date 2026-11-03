"""Valero et al. 2022 mouse CA1 units on a 110 cm linear track with ripples (DANDI 000568)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "valero_000568",
    "title": "Valero et al. (2022), mouse fCamk2 session 3, CA1 units on a linear track",
    "source": "https://dandiarchive.org/dandiset/000568/0.230705.1633",
    "license": "CC-BY-4.0",
    "download_size": "5.8 MB",
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
    raise NotImplementedError
