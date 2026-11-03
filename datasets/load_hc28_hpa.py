"""Jadhav & Frank rat HPa CA1 and PFC units on a W-track, day 1 run epoch 2 (CRCNS hc-28).

Not implemented yet: ``DATASET`` and ``EXPECTATIONS`` are registered before
the first load.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from _record import Range

DATASET = {
    "key": "hc28_hpa",
    "title": "Jadhav & Frank, rat HPa day 1 W-track run, CA1 and PFC units (CRCNS hc-28)",
    "source": "https://doi.org/10.6080/K02N50G9",
    "license": "CRCNS terms of use (account required; lab asks for authorship, cite the data set)",
    "download_size": "about 25 MB (members of a 2.8 GB archive)",
}

# Pre-registered from the hc-28 data description (rawpos: non-interpolated
# video positions, zeros where untracked; pos: interpolated at 30 Hz with
# cmperpix; W-track alternation), the task file (day 1 epoch 2 is a run on
# 'wtr1') and the published W-track size (about 80 x 80 cm), before the first
# run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 2,
    "n_units": Range(10, 120),
    "sampling_rate_hz": Range(29.0, 31.0),
    "position_extent_cm": lambda rec: Range(60, 120).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "run_lasts_10_to_30_min": lambda rec: Range(600, 1800).contains(
        float(rec.times[-1] - rec.times[0])
    ),
    "untracked_frames_present": lambda rec: bool(np.isnan(rec.positions).any()),
}
