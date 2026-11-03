"""Gardner et al. 2022 rat MEC grid modules in an open field (Figshare 16764508)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "gardner_2022_grid",
    "title": "Gardner et al. (2022), rat R day 1, MEC grid modules 1-3, open field",
    "source": "https://figshare.com/articles/dataset/16764508",
    "license": "CC0",
    "download_size": "168.9 MB",
}

# Pre-registered from the paper (1.5 m box, three modules of roughly 100-150 cells) and the file's array headers (100 Hz) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": Range(250, 450),
    "n_dims": 2,
    "sampling_rate_hz": Range(99.0, 101.0),
    "position_extent_cm": lambda rec: Range(140, 160).contains(
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
