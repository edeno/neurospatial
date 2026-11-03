"""Senzai & Buzsaki 2017 mouse dentate gyrus and CA3 units in open exploration and a theta maze (DANDI 000003)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "senzai_000003",
    "title": "Senzai & Buzsaki (2017), YutaMouse41 150829, hippocampal units, open exploration and theta maze",
    "source": "https://dandiarchive.org/dandiset/000003/0.260218.2052",
    "license": "CC-BY-4.0",
    "download_size": "4.66 GB (streamed)",
}

# Pre-registered from the dataset catalog (position labelled "meters" with
# pixel-like values 117-500) and the file's metadata (23 units; two
# position_sensor series of 666,029 samples at 39.0625 Hz labelled meters,
# reference frame "unknown"; sleep states; no epochs table) before the first
# run of this loader. Positions stay in pixels unless the file or paper gives
# a calibration.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 23,
    "n_dims": 2,
    "sampling_rate_hz": Range(39.0, 39.2),
    "pixels": lambda rec: rec.position_units == "pixel",
    "values_are_not_metres": lambda rec: bool(
        np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0)) > 50
    ),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
