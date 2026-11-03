"""Comrie, Frank (curated by Denovellis) rat j16 hippocampus in a spatial bandit maze (Zenodo 23020757)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "j16_23020757",
    "title": "Rat j16, 2021-07-10, run epoch 02_r1, hippocampal units in a spatial bandit maze",
    "source": "https://doi.org/10.5281/zenodo.23020757",
    "license": "CC-BY-4.0",
    "download_size": "75.1 MB",
}

# Pre-registered from the catalog (203 units, 870,018 spikes, 709,321
# samples, track graph) and the .npz array headers (10 track nodes, 9
# edges) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 203,
    "n_dims": 2,
    "sampling_rate_hz": Range(499.0, 501.0),
    "total_spikes": lambda rec: sum(len(s) for s in rec.spike_times) == 870018,
    "graph_10_nodes_9_edges": lambda rec: (
        rec.geometry["graph"].number_of_nodes() == 10
        and rec.geometry["graph"].number_of_edges() == 9
    ),
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
