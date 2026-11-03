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

Environment (``geometry["environment"]``): a circular arena 150 cm in
diameter, centred at the origin (the dataset's ``mat_files.md``: x, y are
"relative to arena center"). Vollan et al. 2025 (Methods) state a 150 x 150 cm
box for the open-field rats, with one exception: "a rat foraged during
recording in a dark room encountered for the first time ... a circular arena
150 cm in diameter was used", and decoding there used "tuning curves from a
succeeding recording session with room lights on (same arena)" (Extended Data
Fig. 9k). This file's one session is of type ``open_field_light``. Of the
18 other open-field files downloaded for this survey, 16 are ``open_field``,
whose samples fill a square (convex hull 0.98-1.00 of its minimum rectangle,
one at 0.92), and 2 are ``open_field_darkness``, whose samples are round
(hull 0.79 of its minimum rectangle, 0.97 of its minimum circle). This
session's samples are round as well (0.81 and 0.94), so it is taken to be the lit
session in the circular arena (the other exception, a 1 m square with a
transparent floor, is ruled out by the samples' 1.4 m span). The paper names neither the rat nor the file:
the identification is an inference from the session type and the samples'
shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch
from _geometry import declared_circle, with_environment
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
    # No extent expectation: the loader refuses a span outside 130-160 cm
    # (checked_span), so one here could never fail; the span is in
    # ``conversions``.
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
    rec = to_recording(
        read_session(path),
        dataset=DATASET,
        file=_FILE,
        session="26018_2 open_field_light",
        span_cm=(130.0, 160.0),
    )
    return with_environment(
        rec,
        *declared_circle(
            rec.positions,
            150.0,
            bin_size=rec.params["bin_size"],
            units="cm",
            name=DATASET["key"],
            source="Vollan et al. 2025 Methods: 'a circular arena 150 cm in diameter'"
            " (dark-room session; the lit session after it in the same arena);"
            " file session type 'open_field_light'",
            centre=(0.0, 0.0),
        ),
        tracked_point="the head, OptiTrack motion capture (per the loader, from the dataset's documentation)",
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
