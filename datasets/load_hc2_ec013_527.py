"""Mizuseki et al. rat ec013 CA1 units foraging in a 180 cm square (CRCNS hc-2, session ec013.527).

What was recorded: rat ec013, a 4-shank silicon probe in dorsal CA1,
chasing food in a 180 cm x 180 cm elevated open field for 17.7 min. Two
head-mounted LEDs tracked by video at 39.06 Hz.

Where it lives: CRCNS hc-2, which needs a free crcns.org account
(``CRCNS_USERNAME``/``CRCNS_PASSWORD``). The session is
``ec013.527/ec013.527.tar.gz`` (211.8 MB) in Neuroscope format; the loader
streams it once and keeps ``.res.N``, ``.clu.N``, ``.whl`` and ``.xml``. The
same recording is in hc-3 (topdir ``ec013.32``) with eight shanks; hc-2 keeps
the four CA1 shanks.

What the loader converts: as ``load_hc3_ec013_669`` (shared code there),
except that hc-2 has no cell table, so every cluster from 2 up is a unit, as
the hc-2 description defines them (0 artefact, 1 unsortable). The pixel
scale is inferred from the 180 cm side over the larger per-axis span of the
tracked positions.

Environment (``geometry["environment"]``): the platform the hc-2 data
description (``crcns-hc2-data-description.pdf``, read) states, "an elevated
square platform (180 cm x 180 cm, or 120 cm x 120 cm)", 180 cm for ec013.527
in its session table. It is declared in pixels at the inferred scale (its
side is the larger per-axis span of the samples, so the residual of the
longer fitted side checks only the trimming; the shorter side is the check),
with a drop edge (``OpenArena(..., outline="drop")``), built with
``Environment.from_polygon`` at ``bin_size``. Centre and rotation are fitted
to the samples (``_geometry.declared_rectangle``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _geometry import declared_rectangle, with_environment
from _record import Range, Recording, check_expectations, validate_recording
from load_hc3_ec013_669 import (
    coincident_leds_as_east,
    fetch_session,
    leds_swapped,
    neuroscope_recording,
    whl_minus_one_kept,
)

DATASET = {
    "key": "hc2_ec013_527",
    "title": "Mizuseki et al., rat ec013 session ec013.527, CA1 units in a 180 cm open field (CRCNS hc-2)",
    "source": "https://crcns.org/data-sets/hc/hc-2",
    "license": "CRCNS terms of use (account required; cite the data set)",
    "download_size": "about 6 MB (members of a 211.8 MB archive)",
}

# Pre-registered from the hc-2 data description (ec013.527: 180 cm square,
# 4 shanks in CA1, 17.7 min; .whl at 39.06 Hz with two LEDs, -1 where
# untracked, no unit stated) and hc-3's cell table for the same recording
# (topdir ec013.32: 41 CA1 cells), before the first run of this loader.
# The .whl values are expected to be video pixels.
EXPECTATIONS: dict[str, Any] = {
    "n_units": Range(30, 80),
    "n_dims": 2,
    "sampling_rate_hz": Range(38.5, 39.5),
    "duration_s": Range(1000.0, 1070.0),
    "arena_aspect_square": lambda rec: Range(0.8, 1.25).contains(
        float(
            np.divide(*np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}


SUBDIR = "crcns_hc2"
ARCHIVE = "hc-2/ec013.527/ec013.527.tar.gz"
MEMBER_DIR = "crcns/hc2/ec013.527"
BASE = "ec013.527"
N_SHANKS = 4
ARENA_CM = 180.0
_SHA256 = {
    "clu.1": "52df2bcd6eebd92240479c089488a2f7074bcd283f42690f0544460ba4323609",
    "clu.2": "ea306a0e3ce05fd835e6719c0df1e3cd84b9f45ae7db728402251bd402a3726a",
    "clu.3": "df002700798154f1103fe0bf5c4c929d16f27cd0872b17f27146339ed5c1f9dd",
    "clu.4": "58e7bae6467525f099edc8e922f4f994f53bcc823e89cc52e786fe43df1a422a",
    "res.1": "1ea2b727ea556a0f87e402eb57a6801e5e3037b6d6f4adea3dde1b8198f6a70a",
    "res.2": "6835ecd3a94990d39661fa8d6242d698ab5e38226baf88bf9205ca2dc22c557e",
    "res.3": "533b64fd0eb19bdd39f97217f0f6e239af8064840b08107aaedd758eece30a8a",
    "res.4": "b8ab60365f5c54a456817dbb85ebea512da810dcc6c3c9db8aef384b9a1c832c",
    "whl": "697a526cacf37da2e31f677306f39e20abc47cecc4c680827d17ba9f015310e2",
    "xml": "cba6a756d83474366c53b2d0d80780c6a196dfd006165620c3c5c607a2bce724",
}


def _largest_axis_span(positions: np.ndarray) -> float:
    tracked = positions[np.isfinite(positions).all(axis=1)]
    return float(np.ptp(tracked, axis=0).max())


def load(cache: Path | None = None) -> Recording:
    """Fetch the session; positions in pixels, two-LED headings, every sorted cluster."""
    files = fetch_session(
        ARCHIVE, MEMBER_DIR, BASE, _SHA256, subdir=SUBDIR, cache=cache
    )
    rec = neuroscope_recording(
        files,
        dataset=DATASET,
        session=BASE,
        source=f"{DATASET['source']} {ARCHIVE}",
        n_shanks=N_SHANKS,
        units=None,
        apparatus_cm=ARENA_CM,
        apparatus_span_px=_largest_axis_span,
        span_px=(120.0, 300.0),
        geometry={"shape": "square", "size_cm": ARENA_CM},
    )
    side_px = ARENA_CM / rec.geometry["inferred_cm_per_pixel"]
    return with_environment(
        rec,
        *declared_rectangle(
            rec.positions,
            side_px,
            side_px,
            bin_size=rec.params["bin_size"],
            units="pixel",
            name=DATASET["key"],
            source="crcns-hc2-data-description.pdf: 'an elevated square platform"
            " (180 cm x 180 cm ...)', ec013.527 180 cm; side in pixels from the"
            " inferred scale",
            outline="drop",
        ),
    )


HAZARDS = {
    "coincident_leds_as_east": coincident_leds_as_east,
    "leds_swapped": leds_swapped,
    "whl_minus_one_kept": whl_minus_one_kept,
}


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
