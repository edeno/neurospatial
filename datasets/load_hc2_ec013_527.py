"""Mizuseki et al. rat ec013 CA1 units foraging in a 180 cm square (CRCNS hc-2, session ec013.527).

Not implemented yet: ``DATASET`` and ``EXPECTATIONS`` are registered before
the first load.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from _record import Range

DATASET = {
    "key": "hc2_ec013_527",
    "title": "Mizuseki et al., rat ec013 session ec013.527, CA1 units in a 180 cm open field (CRCNS hc-2)",
    "source": "https://crcns.org/data-sets/hc/hc-2",
    "license": "CRCNS terms of use (account required; cite the data set)",
    "download_size": "about 6 MB (members of a 211.8 MB archive)",
}

# Pre-registered from the hc-2 data description (ec013.527: 180 cm square,
# 4 shanks in CA1, 17.7 min; .whl at 39.06 Hz with two LEDs, -1 where
# untracked, no unit stated) and hc-3's cell table for the same recording
# (topdir ec013.32: 41 CA1 cells), before the first run of this loader.
# The .whl values are expected to be video pixels.
EXPECTATIONS: dict[str, Any] = {
    "n_units": Range(30, 80),
    "n_dims": 2,
    "sampling_rate_hz": Range(38.5, 39.5),
    "duration_s": Range(1000.0, 1070.0),
    "arena_aspect_square": lambda rec: Range(0.8, 1.25).contains(
        float(
            np.divide(*np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}
