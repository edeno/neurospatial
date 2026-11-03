"""Mizuseki et al. rat ec013 CA1 and entorhinal units on a 250 cm linear track (CRCNS hc-3, session ec013.669).

Not implemented yet: ``DATASET`` and ``EXPECTATIONS`` are registered before
the first load.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from _record import Range

DATASET = {
    "key": "hc3_ec013_669",
    "title": "Mizuseki et al., rat ec013 session ec013.669, CA1 and EC units on a 250 cm linear track (CRCNS hc-3)",
    "source": "https://crcns.org/data-sets/hc/hc-3",
    "license": "CRCNS terms of use (account required; cite the data set)",
    "download_size": "about 5 MB (members of a 233.8 MB archive)",
}

# Pre-registered from the hc-3 metadata tables (topdir ec013.38: 125 sorted
# cells; session ec013.669, behavior "linear", 798.3 s), the original docs'
# README ("linear: linear maze, 250 cm") and the hc-2 description of the .whl
# file (39.06 Hz, two LEDs, -1 where untracked, no unit stated), before the
# first run of this loader. The .whl values are expected to be video pixels.
EXPECTATIONS: dict[str, Any] = {
    "n_units": Range(100, 125),
    "n_dims": 2,
    "sampling_rate_hz": Range(38.5, 39.5),
    "duration_s": Range(780.0, 800.0),
    "track_long_axis_px": lambda rec: Range(100, 720).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "has_headings": lambda rec: rec.headings is not None
    and bool(np.mean(np.isfinite(rec.headings)) > 0.5),
}
