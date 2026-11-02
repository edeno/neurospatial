"""Huszár, Zhang, Blockus & Buzsáki mouse CA1 on a figure-8 maze (DANDI 000552)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _record import Range, Recording

DATASET = {
    "key": "figure8_000552",
    "title": "Huszár et al., mouse CA1 on a figure-8 maze with sleep and ripples",
    "source": "https://dandiarchive.org/dandiset/000552/0.230630.2304",
    "license": "CC-BY-4.0",
    "download_size": "96.0 MB",
}

# Pre-registered from the dandiset description and a metadata probe of this
# file, before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 113,
    "n_dims": 1,
    "position_min": Range(0.0, 222.0),
    "position_max": Range(0.0, 222.0),
    "ripples_present": lambda rec: "ripples" in rec.events
    and len(rec.events["ripples"]) > 0,
    "sleep_states_present": lambda rec: len(rec.epochs) > 0,
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
