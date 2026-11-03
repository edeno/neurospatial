"""Grosmark & Buzsaki 2016 rat CA1 on a 1.6 m linear track with sleep before and after (DANDI 000044)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "grosmark_000044_linear",
    "title": "Grosmark & Buzsaki (2016), Achilles 10252013, rat CA1, 1.6 m linear track",
    "source": "https://dandiarchive.org/dandiset/000044/0.261005.2101",
    "license": "CC-BY-4.0",
    "download_size": "8.74 GB (streamed)",
}

# Pre-registered from the survey appendix (137 units, 39.06 Hz, 1.6 m linear track, PRE/MAZE/POST epochs) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 137,
    "n_dims": 2,
    "sampling_rate_hz": Range(38.0, 40.0),
    "position_extent_cm": lambda rec: Range(150, 175).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "epoch_names": ["maze", "post", "pre"],
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
