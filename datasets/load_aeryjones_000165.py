"""Aery Jones et al. mouse hippocampal multi-unit activity on a linear track after CNO (DANDI 000165)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "aeryjones_000165",
    "title": "Aery Jones et al., mouse Sumac day 6, hippocampal multi-unit activity on a linear track",
    "source": "https://dandiarchive.org/dandiset/000165/0.211118.1526",
    "license": "CC-BY-4.0",
    "download_size": "23.3 MB",
}

# Pre-registered from the dataset catalog (linear track) and the file's
# metadata (a 32-row "MUA table", position labelled centimeters) before the
# first run of this loader. The last two are consistency requirements of any
# session: tracking at a video rate, and spikes over the tracked session.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 32,
    "n_dims": 2,
    "sampling_rate_hz": Range(25.0, 35.0),
    "spikes_cover_tracking": lambda rec: bool(
        max(float(np.max(t)) for t in rec.spike_times if t.size)
        >= rec.times[0] + 0.9 * (rec.times[-1] - rec.times[0])
    ),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
