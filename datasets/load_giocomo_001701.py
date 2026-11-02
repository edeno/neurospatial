"""Aery Jones & Giocomo mouse MEC and CA1 units on an X maze (DANDI 001701)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "giocomo_001701",
    "title": "Giocomo lab, mouse MEC and CA1 Neuropixels units on an X maze",
    "source": "https://dandiarchive.org/dandiset/001701/0.260120.0303",
    "license": "CC-BY-4.0",
    "download_size": "13.0 MB",
}

# Pre-registered from the survey appendix before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 320,
    "n_dims": 2,
    "sampling_rate_hz": Range(59.0, 61.0),
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
