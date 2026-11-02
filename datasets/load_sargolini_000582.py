"""Sargolini et al. (2006) rat medial entorhinal cortex, open field (DANDI 000582)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _record import Recording

DATASET = {
    "key": "sargolini_000582",
    "title": "Sargolini et al. 2006, rat MEC grid and conjunctive cells, 1 m box",
    "source": "https://dandiarchive.org/dandiset/000582/0.251111.2151",
    "license": "CC-BY-4.0",
    "download_size": "15.9 MB",
}

# Written from the survey appendix before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 6,
    "n_dims": 2,
    "sampling_rate_hz": (49.0, 51.0),
    "position_min": ((-55.0, -55.0), (0.0, 0.0)),
    "position_max": ((0.0, 0.0), (55.0, 55.0)),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
