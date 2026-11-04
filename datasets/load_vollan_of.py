"""Vollan et al. 2025 rat MEC grid units in an open field (EBRAINS 10.25493/R5FR-EDG).

What was recorded: rat 26018, recording 2, Neuropixels in medial entorhinal
cortex (644 units; the authors' three grid modules hold 48, 90 and 95 units,
median unit grid scores 1.11, 0.99 and 0.68), foraging in an open field about
1.5 m across, tracked at 100 Hz. Chosen as the open-field file with the
strongest and most grid modules among the smaller files of each MEC rat
(rat 27764, the linear-track rat, has one weak module: median 0.26).

The session is read by ``neurospatial.datasets.load("vollan_2025")`` (its
default, ``of_26018_2``), whose module
(``neurospatial/datasets/vollan_2025.py``) and docs page give the file
(``sharing_v4/navigation/of/26018_2.mat``), the conversions (as for
``load_vollan_lt``), the authors' grid modules in ``params`` (per-unit
``module`` and ``is_grid``; per module, ``module_ids``, its median unit grid
score ``module_median_grid_score``, the grid score of its median
autocorrelogram ``module_grid_score`` and ``module_spacing_cm``, the
reference for the grid score comparison), and the declared arena: a circle
150 cm in diameter placed on the samples. Vollan et al. 2025 (Methods)
state a 150 x 150 cm box for the open-field rats, with one exception: "a
rat foraged during recording in a dark room encountered for the first time
... a circular arena 150 cm in diameter was used", decoded with "tuning
curves from a succeeding recording session with room lights on (same
arena)" (Extended Data Fig. 9k). This file's one session is of type
``open_field_light`` and its samples are round (convex hull 0.81 of its
minimum rectangle, 0.94 of its minimum circle; plain ``open_field`` files
fill a square, 0.92-1.00 of their rectangle), so the identification is an
inference from the session type and the samples' shape; the paper names
neither the rat nor the file. ``_record.from_public`` converts the public
recording to the survey's record.

Survey-only: the declared circle's facts the environment fidelity study
reads (placed on the samples' minimum bounding circle, the stated diameter,
the samples' own diameter against it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import (
    Range,
    Recording,
    SurveyOnly,
    check_expectations,
    from_public,
    validate_recording,
)
from load_vollan_lt import times_rebuilt_from_dt

DATASET = {"key": "vollan_of", "public": "vollan_2025", "session": "of_26018_2"}

# Pre-registered from the dataset documentation (100 Hz, 1.5 m box) and the file's
# unit table (644 MEC units, three grid modules) before the first run of this
# loader; replaces the registration for rat 27764, whose one module is weak.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 644,
    "n_dims": 2,
    "sampling_rate_hz": Range(99.0, 101.0),
    # No extent expectation: the public loader's units check refuses a span
    # outside 120-240 cm, so one here could never fail.
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}


def load(cache: Path | None = None) -> Recording:
    """The session from ``neurospatial.datasets``; positions in cm, headings in radians."""
    from neurospatial.datasets import load as load_public
    from neurospatial.datasets import vollan_2025
    from neurospatial.datasets._arenas import fitted_circle

    rec = load_public(DATASET["public"], DATASET["session"], cache=cache)
    centre, diameter = fitted_circle(rec.positions)
    why = "the declared circle's facts the environment fidelity study reads"
    return from_public(
        rec,
        key=DATASET["key"],
        geometry={
            "declared_from": SurveyOnly(
                "fitted to the samples",
                f"{why}: placed on the samples' minimum bounding circle",
            ),
            "shape": SurveyOnly("circle", f"{why}: the stated shape"),
            "size_cm": SurveyOnly(vollan_2025.DIAMETER, f"{why}: the stated diameter"),
            "environment_fit": SurveyOnly(
                {
                    "centre_x": float(centre[0]),
                    "centre_y": float(centre[1]),
                    "samples_diameter": diameter,
                    "size_residual": diameter / vollan_2025.DIAMETER - 1,
                },
                f"{why}: the placement and the samples' diameter against the stated one",
            ),
        },
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
