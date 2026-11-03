"""Grieves et al. 2020 rat CA1 place cells in a 3-D climbing lattice (UCL RDR 10.5522/04/9926477)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "grieves_2020_lattice",
    "title": "Grieves et al. (2020), rat 926, 2018-04-06, CA1 units in a cubic climbing lattice",
    "source": "https://doi.org/10.5522/04/9926477.v1",
    "license": "CC0",
    "download_size": "431 MB of an 11.2 GB Deflate64 zip (byte-range read)",
}

# Pre-registered from the readme, the zip listing and the session file's
# HDF5 layout (lattice_3d pox/poy/poz at 50 Hz; 30 clusters, 6 of them
# cluster 0) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 24,
    "n_dims": 3,
    "sampling_rate_hz": Range(49.0, 51.0),
    "duration_s": Range(3000.0, 4000.0),
    "lattice_extent_cm_per_axis": lambda rec: Range(100.0, 160.0).contains(
        np.nanpercentile(rec.positions, 99, axis=0)
        - np.nanpercentile(rec.positions, 1, axis=0)
    ),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
