"""Gonzalez & Giocomo rat open-field session with binned firing rates (DANDI 000405)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Range, Recording

DATASET = {
    "key": "giocomo_000405",
    "title": "Gonzalez & Giocomo (2022), rat parahippocampal units in an open field",
    "source": "https://dandiarchive.org/dandiset/000405/draft",
    "license": "unstated (draft dandiset)",
    "download_size": "27.1 MB",
}

# Pre-registered from the survey appendix before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_cells": 46,
    "n_dims": 2,
    "sampling_rate_hz": Range(49.0, 51.0),
    "activity_is_binned_rate": lambda rec: rec.activity_kind == "binned_rate",
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
