"""Kanter et al. rat LEC and CA1 Neuropixels units, open field with novel objects (EBRAINS 10.25493/GTVB-FQR)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording, heading_vs_movement

DATASET = {
    "key": "kanter_2025",
    "title": "Kanter et al. (2025), rat 29629, LEC and CA1 units, 1 m open field with novel objects",
    "source": "https://doi.org/10.25493/GTVB-FQR",
    "license": "CC-BY-SA-4.0",
    "download_size": "29.6 MB",
}

# Pre-registered from the data descriptor (OptiTrack head tracking projected on
# the floor; foraging in an empty 1 m arena, then three novel objects for
# 7.5 min each; no object positions listed among the data) and the file's
# metadata (380 unit rows, position and head direction at 120 Hz, four epochs)
# before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 380,
    "n_dims": 2,
    "sampling_rate_hz": Range(119.0, 121.0),
    "arena_extent_cm": lambda rec: Range(90, 105).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "epoch_names": ["empty", "object_1", "object_2", "object_3"],
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
    "headings_follow_movement": lambda rec: bool(
        abs(
            heading_vs_movement(rec.times, rec.positions, rec.headings, min_speed=5.0)[
                0
            ]
        )
        < np.pi / 4
    ),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
