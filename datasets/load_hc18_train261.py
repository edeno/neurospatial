"""Drieu, Todorova & Zugaro rat CA1 units while carried on a model train (CRCNS hc-18, session Train-261-20140617).

Not implemented yet: ``DATASET`` and ``EXPECTATIONS`` are registered before
the first load.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from _record import Range

DATASET = {
    "key": "hc18_train261",
    "title": "Drieu, Todorova & Zugaro (2018), rat 261, CA1 units on a treadmill carried by a model train (CRCNS hc-18)",
    "source": "https://doi.org/10.6080/K0Z899MM",
    "license": "CRCNS terms of use (account required; cite Drieu et al. 2018 and the data set)",
    "download_size": "about 175 MB (members of a 6.5 GB archive)",
}

# Pre-registered from the hc-18 description (three travel sessions, passive,
# active, passive, 2 x ~15 laps each, interleaved with ~90 min of sleep;
# Neuroscope format) and the session's .xml and .cat.evt (video 768 x 576 at
# 39.0625 Hz; 12 subsessions), before the first run of this loader. The
# .pos values are expected to be video pixels of one LED.
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 2,
    "n_units": Range(20, 300),
    "sampling_rate_hz": Range(39.0, 39.1),
    "positions_inside_the_video": lambda rec: bool(
        np.nanmin(rec.positions) >= 0
        and np.nanmax(rec.positions[:, 0]) <= 768
        and np.nanmax(rec.positions[:, 1]) <= 576
    ),
    "travel_epochs": lambda rec: len(rec.epochs.get("travel", ())) == 7,
}
