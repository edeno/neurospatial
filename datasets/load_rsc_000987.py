"""Peyrache lab mouse retrosplenial cortex units in a square arena with tracked head direction (DANDI 000987)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording, heading_vs_movement

DATASET = {
    "key": "rsc_000987",
    "title": "Peyrache lab, mouse A7803, retrosplenial cortex units, sleep then square arena",
    "source": "https://dandiarchive.org/dandiset/000987/0.261003.2342",
    "license": "CC-BY-4.0",
    "download_size": "20.2 MB",
}

# Pre-registered from the dataset catalog (head direction plus 2-D, blank position
# units) and the file's metadata (37 units; x, y, z and rx, ry, rz series of
# 120,287 samples over 1203 s) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 37,
    "n_dims": 2,
    "sampling_rate_hz": Range(99.0, 101.0),
    "arena_extent_cm": lambda rec: Range(40, 100).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
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
