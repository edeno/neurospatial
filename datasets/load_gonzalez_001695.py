"""Buzsaki lab mouse CA1, CA3 and RSC units on a novel maze with sleep before and after (DANDI 001695)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "gonzalez_001695",
    "title": "Buzsaki lab, mouse M02, CA1/CA3/RSC units on a novel maze with pre and post sleep",
    "source": "https://dandiarchive.org/dandiset/001695/0.260319.2023",
    "license": "CC-BY-4.0",
    "download_size": "61.3 MB",
}

# Pre-registered from the dataset catalog (position stored with conversion
# 0.01) and the file's metadata (110 units; position labelled "centimeters"
# with conversion 0.01, stored values 6.6-114 by 13-19, 25 Hz; a Speed series
# in cm/s; ripples listed among the sleep states) before the first run of this
# loader. The extent expectation says the stored values, not the converted
# ones, are centimetres.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 110,
    "n_dims": 2,
    "sampling_rate_hz": Range(24.0, 26.0),
    "track_length_cm": lambda rec: Range(90, 130).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "has_ripples": lambda rec: "ripples" in rec.events,
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
