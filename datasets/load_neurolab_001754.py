"""Neurolab STS-90 rat CA1 place cells recorded in spaceflight, Escher staircase and baseline track (DANDI 001754)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "neurolab_001754",
    "title": "Knierim, McNaughton et al., Neurolab flight day 4, rat 2, CA1 units",
    "source": "https://dandiarchive.org/dandiset/001754/0.260728.1352",
    "license": "CC-BY-4.0",
    "download_size": "3.2 MB",
}

# Pre-registered from the dataset catalog (pixel coordinates 0-255, Escher
# staircase, position sampled at spike times) and the file's metadata (72
# units; 25,320 position samples labelled pixels; five epochs BL, ES, BL, BL,
# MC) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 72,
    "n_dims": 2,
    "position_min": Range([0, 0], [255, 255]),
    "position_max": Range([0, 0], [255, 255]),
    "pixels": lambda rec: rec.position_units == "pixel",
    "five_task_epochs": lambda rec: sum(len(v) for v in rec.epochs.values()) == 5,
    "position_sampled_at_spikes": lambda rec: bool(
        np.all(np.isin(rec.times, np.concatenate(rec.spike_times)))
    ),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
