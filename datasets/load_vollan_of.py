"""Vollan et al. 2025 rat MEC grid units in an open field (EBRAINS 10.25493/R5FR-EDG)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "vollan_of",
    "title": "Vollan et al. (2025), rat 27764, MEC Neuropixels units in an open field",
    "source": "https://doi.org/10.25493/R5FR-EDG",
    "license": "CC-BY-SA-4.0",
    "download_size": "162.7 MB",
}

# Pre-registered from the dataset documentation (100 Hz, 1.5 m box) and the file's unit table (384 MEC units, one grid module) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 384,
    "n_dims": 2,
    "sampling_rate_hz": Range(99.0, 101.0),
    "position_extent_cm": lambda rec: Range(130, 160).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
