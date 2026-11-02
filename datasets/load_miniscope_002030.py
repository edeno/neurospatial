"""Mouse CA1 one-photon calcium imaging in a circular open field (DANDI 002030)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "miniscope_002030",
    "title": "Mouse dorsal CA1 miniscope imaging, 96 cm circular open field",
    "source": "https://dandiarchive.org/dandiset/002030/0.261016.1321",
    "license": "CC-BY-4.0",
    "download_size": "4.9 MB",
}

# Pre-registered from the survey appendix before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 2,
    "sampling_rate_hz": Range(29.0, 31.0),
    "position_extent_at_most_100_cm": lambda rec: bool(
        np.all(np.nanmax(rec.positions, 0) - np.nanmin(rec.positions, 0) <= 100.0)
    ),
    "activity_is_dff": lambda rec: rec.activity_kind == "dff",
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
