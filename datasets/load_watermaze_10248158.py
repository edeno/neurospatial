"""Overall & Kempermann mouse Morris water maze swim paths (Figshare 10248158), behavior only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "watermaze_10248158",
    "title": "Overall & Kempermann, C57BL/6 mouse B6_6, 30 water-maze trials (behavior only)",
    "source": "https://doi.org/10.6084/m9.figshare.10248158.v3",
    "license": "CC-BY-4.0",
    "download_size": "9.9 MB",
}

# Pre-registered from the Figshare description (3 training + 2 reversal days,
# 6 trials a day, last trial of days 3 and 5 a probe) and the JSON's first
# track (raw.t every 0.08 s; arena circle radius 95 in tracking units)
# before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 2,
    "sampling_rate_hz": Range(12.4, 12.6),
    "behavior_only": lambda rec: rec.spike_times is None,
    "thirty_trials": lambda rec: len(rec.epochs["trials"]) == 30,
    "two_probe_trials": lambda rec: len(rec.epochs["probe_trials"]) == 2,
    "inside_the_pool": lambda rec: bool(
        np.nanmax(np.hypot(*(rec.positions - rec.geometry["pool_centre"]).T))
        <= 1.1 * rec.geometry["pool_radius"]
    ),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
