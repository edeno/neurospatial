"""Forli, Fan, Qi & Yartsev (2025) bat hippocampus during 3-D flight (Zenodo demo session)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "yartsev_bat_demo",
    "title": "Forli et al. 2025, Egyptian fruit bat CA1 during free flight in a room",
    "source": "https://doi.org/10.5281/zenodo.15738988",
    "license": "CC-BY-4.0",
    "download_size": "335 MB of a 3.3 GB zip (byte-range reads)",
}

# Pre-registered from the record's demo script and a metadata probe of the
# behavior file, before the first run of this loader. y is not constrained:
# some samples lie outside the room bounds the file states.
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 3,
    "sampling_rate_hz": Range(119.0, 121.0),
    "position_min": Range((-3.1, -np.inf, -0.2), (np.inf, np.inf, np.inf)),
    "position_max": Range((-np.inf, -np.inf, -np.inf), (3.1, np.inf, 2.5)),
    "probe1_at_most_24_units": lambda rec: sum(
        str(uid).startswith("probe1-") for uid in rec.unit_ids
    )
    <= 24,
    "flights_present": lambda rec: len(rec.epochs["flight"]) == 38,
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
