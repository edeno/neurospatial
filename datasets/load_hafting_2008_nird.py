"""Hafting et al. 2008 rat MEC grid cells on a linear track (NIRD 10.11582/2017.00020)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _record import Range, Recording

DATASET = {
    "key": "hafting_2008_nird",
    "title": "Hafting et al. (2008), rat 11265, sessions 16030611+12, MEC units on a 320 cm linear track",
    "source": "https://doi.org/10.11582/2017.00020",
    "license": "CC-BY-4.0",
    "download_size": "1.0 MB of a 20.9 MB zip (byte-range reads)",
}

# Pre-registered from the readme (mostly 320 cm linear tracks, 10 min
# sessions joined with "+") and the zip listing (three cells t4c1, t4c2,
# t4c4) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 3,
    "n_dims": 2,
    "sampling_rate_hz": Range(49.0, 51.0),
    "duration_s": Range(1190.0, 1210.0),
    "track_length_cm": lambda rec: Range(300.0, 340.0).contains(
        float(rec.geometry["position_span_cm"][0])
    ),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
