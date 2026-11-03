"""Shipley, Barry et al. 2025 mouse CA1 on an 8-arm radial maze (UCL RDR 30902972)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "shipley_30902972",
    "title": "Shipley et al. (2025), mouse m4005, 2020-09-26, CA1 units on an 8-arm radial maze and sleep",
    "source": "https://doi.org/10.5522/04/30902972.v1",
    "license": "CC-BY-4.0",
    "download_size": "67 MB of a 4.4 GB zip (byte-range reads)",
}

# Pre-registered from the zip listing (11 maze trials and a sleep session
# S2) and trialData's header (30 Hz, 274 pixels per metre, one LED) and
# cluster_info.tsv (71 clusters labelled good) before the first run of this
# loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 71,
    "n_dims": 2,
    "sampling_rate_hz": Range(29.9, 30.1),
    "maze_extent_cm": lambda rec: Range(140.0, 200.0).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "eleven_maze_trials": lambda rec: len(rec.epochs["maze_trials"]) == 11,
    "one_led_no_heading": lambda rec: rec.headings is None,
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
