"""Peyrache et al. mouse anterodorsal thalamus head-direction session Mouse32-140822."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import Recording

DATASET = {
    "key": "peyrache_mouse32",
    "title": "Peyrache et al., mouse ADn/thalamus head-direction cells, wake and sleep",
    "source": "https://osf.io/jb2gd",
    "license": "unstated on OSF; the source dandiset 000056 is CC-BY-4.0",
    "download_size": "36.6 MB",
}

# Written from the survey appendix before the first run of this loader. The
# appendix's "31 units in ADn" needs per-unit regions, which a Recording does
# not hold; the head-direction reference check verifies it instead.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 49,
    "epochs_include_wake_and_sleep": lambda rec: {"wake", "sleep"} <= set(rec.epochs),
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
