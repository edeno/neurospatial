"""Vollan et al. 2025 rat MEC units on a linear track (EBRAINS 10.25493/R5FR-EDG)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "vollan_lt",
    "title": "Vollan et al. (2025), rat 27764, MEC Neuropixels units on a linear track",
    "source": "https://doi.org/10.25493/R5FR-EDG",
    "license": "CC-BY-SA-4.0",
    "download_size": "110.6 MB",
}

# Pre-registered from the dataset documentation (100 Hz, speed-filtered at 5 cm/s) and the file's unit table (389 MEC units) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 389,
    "n_dims": 2,
    "sampling_rate_hz": Range(99.0, 101.0),
    "position_extent_cm": lambda rec: Range(190, 240).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
    "speed_filter_leaves_gaps": lambda rec: bool(np.max(np.diff(rec.times)) > 0.1),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
