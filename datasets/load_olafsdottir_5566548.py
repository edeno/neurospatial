"""Olafsdottir, Carpenter & Barry 2016 rat hippocampus and deep MEC on a Z-track (Zenodo 5566548)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _record import Range, Recording

DATASET = {
    "key": "olafsdottir_5566548",
    "title": "Olafsdottir et al. (2016), rat R2142, 2014-08-06, Z-track, 16 tetrodes",
    "source": "https://doi.org/10.5281/zenodo.5566548",
    "license": "CC-BY-4.0",
    "download_size": "105 MB of a 9.9 GB zip (byte-range reads)",
}

# Pre-registered from the README (.pos at 50 Hz, .cut per tetrode) and the
# .pos/.1 headers (duration 2276 s, 390 pixels per metre, window 717 x 535
# pixels, one tracked LED) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 2,
    "n_units": Range(10, 150),
    "sampling_rate_hz": Range(49.9, 50.1),
    "duration_s": Range(2270.0, 2280.0),
    "inside_the_tracking_window_cm": lambda rec: Range(0.0, 190.0).contains(
        rec.geometry["position_span_cm"]
    ),
    "one_led_no_heading": lambda rec: rec.headings is None,
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
