"""Vollan et al. 2025 rat MEC grid units in an open field (EBRAINS 10.25493/R5FR-EDG).

What was recorded: rat 26018, recording 2, Neuropixels in medial entorhinal
cortex (644 units; the authors' three grid modules hold 48, 90 and 95 units,
median unit grid scores 1.11, 0.99 and 0.68), foraging in an open field about
1.5 m across, tracked at 100 Hz. Chosen as the open-field file with the
strongest and most grid modules among the smaller files of each MEC rat
(rat 27764, the linear-track rat, has one weak module: median 0.26).

Where it lives: ``sharing_v4/navigation/of/26018_2.mat`` (205.7 MB) in the
same public EBRAINS bucket. File layout, reading and conversions are those of
``load_vollan_lt`` (shared code there): positions m → cm, head direction kept
as radians after a frame check against the direction of movement, the
authors' 5 cm/s speed filter kept as gaps in ``times``.

The authors' grid module is recorded in ``params``: per-unit membership
(``module``, ``is_grid``) and, per module (``module_ids``), its median unit
grid score (``module_median_grid_score``) and the grid score of its median
autocorrelogram (``module_grid_score``). These are the reference for the grid
score comparison.

Analysis choices (``params``): as in ``load_vollan_lt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch
from _record import Range, Recording, check_expectations, validate_recording
from load_vollan_lt import (
    BUCKET,
    SUBDIR,
    read_session,
    times_rebuilt_from_dt,
    to_recording,
)

DATASET = {
    "key": "vollan_of",
    "title": "Vollan et al. (2025), rat 26018, MEC Neuropixels grid modules in an open field",
    "source": "https://doi.org/10.25493/R5FR-EDG",
    "license": "CC-BY-SA-4.0",
    "download_size": "205.7 MB",
}

# Pre-registered from the dataset documentation (100 Hz, 1.5 m box) and the file's
# unit table (644 MEC units, three grid modules) before the first run of this
# loader; replaces the registration for rat 27764, whose one module is weak.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 644,
    "n_dims": 2,
    "sampling_rate_hz": Range(99.0, 101.0),
    "position_extent_cm": lambda rec: Range(130, 160).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}

_FILE = "sharing_v4/navigation/of/26018_2.mat"
_SHA256 = "24fcf65faa3c04ccc4188fda31d32d3728535a0967452dbbbb61fb72c9ad776d"


def load(cache: Path | None = None) -> Recording:
    """Fetch the open-field file; positions in cm, headings in radians, spikes."""
    path = fetch(
        f"{BUCKET}/{_FILE}?redirect=true",
        "of_26018_2.mat",
        sha256=_SHA256,
        subdir=SUBDIR,
        cache=cache,
    )
    return to_recording(
        read_session(path),
        dataset=DATASET,
        file=_FILE,
        session="26018_2 open_field",
        span_cm=(130.0, 160.0),
    )


HAZARDS = {"times_rebuilt_from_dt": times_rebuilt_from_dt}


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    print(
        f"{rec.dataset}: {len(rec.spike_times or ())} units, {rec.times[-1] - rec.times[0]:.0f} s"
    )
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
