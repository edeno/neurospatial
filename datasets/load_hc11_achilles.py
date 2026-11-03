"""Grosmark & Buzsaki 2016 rat CA1 on a 1.6 m linear track (CRCNS hc-11), the session also in DANDI 000044.

Not implemented yet: ``DATASET`` and ``EXPECTATIONS`` are registered before
the first load.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from _record import Range

DATASET = {
    "key": "hc11_achilles",
    "title": "Grosmark & Buzsaki (2016), Achilles 10252013, rat CA1, 1.6 m linear track (CRCNS hc-11)",
    "source": "https://doi.org/10.6080/K0862DC5",
    "license": "CRCNS terms of use (account required; cite Grosmark & Buzsaki 2016 and the data set)",
    "download_size": "38.5 MB (one member of a 277.5 MB archive)",
}


def _epoch_s(name: str) -> Any:
    return lambda rec: float(np.sum(np.diff(rec.epochs[name], axis=1)))


# Pre-registered from the hc-11 documentation (Sessions_Recordings_Summary.pdf:
# 120 putative pyramidal cells and 17 interneurons; PRE 18,080 s, MAZE 2,068 s,
# POST 14,714 s; position at about 39 Hz, metres) and, for the extent, from the
# same session's DANDI 000044 file (G-039: the 2-D positions span about 2 m),
# before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 137,
    "n_dims": 2,
    "sampling_rate_hz": Range(38.0, 40.0),
    "position_extent_cm": lambda rec: Range(150, 260).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "has_pre_maze_post": lambda rec: {"pre", "maze", "post"} <= set(rec.epochs),
    "pre_s": lambda rec: Range(18070, 18090).contains(_epoch_s("pre")(rec)),
    "maze_s": lambda rec: Range(2058, 2078).contains(_epoch_s("maze")(rec)),
    "post_s": lambda rec: Range(14704, 14724).contains(_epoch_s("post")(rec)),
}
