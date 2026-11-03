"""Agarwal, Pastalkova et al. rat CA1 and mPFC on a 3-arm maze with a wheel delay (Figshare 24757638)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "agarwal_24757638",
    "title": "Rat 3-arm alternation with wheel running, session rec02, CA1 and mPFC units",
    "source": "https://doi.org/10.6084/m9.figshare.24757638.v4",
    "license": "CC-BY-4.0",
    "download_size": "226 MB",
}

# Pre-registered from the Figshare description (spikes as 1250 Hz sample
# indices, xMM/yMM in mm, laps) and the file's variable list (439 clusters,
# 95 trials, tracking at 1250 Hz kept every 25th sample) before the first
# run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 439,
    "n_dims": 2,
    "sampling_rate_hz": Range(49.9, 50.1),
    "maze_extent_cm": lambda rec: Range(80.0, 220.0).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "95_trials": lambda rec: len(rec.epochs["trials"]) == 95,
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
