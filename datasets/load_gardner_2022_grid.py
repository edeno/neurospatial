"""Gardner et al. 2022 rat MEC grid modules in an open field (Figshare 16764508).

What was recorded: rat R, day 1, Neuropixels in the medial entorhinal
cortex-parasubiculum region, with the 483 units the paper assigned to grid
modules 1, 2 and 3, in the recording's open-field session (7457-16045 s) of
a 1.5 m box. Head position and azimuth are tracked at 100 Hz.

The session is read by ``neurospatial.datasets.load("gardner_2022",
"rat_r_day1")``, whose module (``neurospatial/datasets/gardner_2022.py``)
and docs page give the files (two members of the Figshare zip, read by byte
range), the restricted unpickler for the pickled spike trains
(``neurospatial.datasets._pickled``), the conversions (positions in cm,
azimuth wrapped to [-π, π] after a check against the direction of movement,
the open-field session's samples and spikes, its valid times as analysis
epochs), the declared 150 cm square placed on the samples and the
watch-outs. ``_record.from_public`` converts it to the survey's record.

Analysis choices are the public loader's ``params``: the Methods' ``bin_size``
3 cm and ``min_speed`` 2.5 cm/s, ``dt`` 0.25 s, alternating 60 s train/test
blocks inside the valid times, each unit's grid module in ``params["module"]``.

Survey-only: the declared square's facts the environment fidelity study reads
(placed on the samples' minimum rotated rectangle, the stated square, each
fitted side against 150 cm).
"""

from __future__ import annotations

from dataclasses import replace
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

# The restricted reader the public loader uses, for the survey's own tests.
from neurospatial.datasets._pickled import SpikeUnpickler as _SpikeUnpickler
from neurospatial.datasets._pickled import pickle_imports, read_npz

__all__ = ["_SpikeUnpickler", "load", "pickle_imports", "read_npz"]

DATASET = {
    "key": "gardner_2022_grid",
    "public": "gardner_2022",
    "session": "rat_r_day1",
}

# Pre-registered from the paper (1.5 m box, three modules of roughly 100-150
# cells) and the file's array headers (100 Hz) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": Range(250, 450),
    "n_dims": 2,
    "sampling_rate_hz": Range(99.0, 101.0),
    # No extent expectation: the public loader's units check refuses a span
    # outside 120-180 cm, so one here could never fail.
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}


def load(cache: Path | None = None) -> Recording:
    """The session from ``neurospatial.datasets``; positions in cm, headings in radians."""
    from neurospatial.datasets import gardner_2022
    from neurospatial.datasets import load as load_public
    from neurospatial.datasets._arenas import fitted_rectangle

    rec = load_public(DATASET["public"], DATASET["session"], cache=cache)
    centre, angle, sides = fitted_rectangle(rec.positions)
    side = gardner_2022.SIDE
    why = "the declared square's facts the environment fidelity study reads"
    return from_public(
        rec,
        key=DATASET["key"],
        geometry={
            "declared_from": SurveyOnly(
                "fitted to the samples",
                f"{why}: placed on the samples' minimum rotated rectangle",
            ),
            "shape": SurveyOnly("square", f"{why}: the stated shape"),
            "size_cm": SurveyOnly(side, f"{why}: the stated side"),
            "environment_fit": SurveyOnly(
                {
                    "centre_x": float(centre[0]),
                    "centre_y": float(centre[1]),
                    "rotation_rad": angle,
                    "size_residual_a": float(max(sides) / side - 1),
                    "size_residual_b": float(min(sides) / side - 1),
                },
                f"{why}: the placement and each fitted side against the stated side",
            ),
        },
    )


def _headings_as_stored(rec: Recording) -> Recording:
    # The stored azimuth: radians in [0, 2pi) instead of [-pi, pi].
    assert rec.headings is not None
    return replace(rec, headings=np.mod(rec.headings, 2 * np.pi))


HAZARDS = {"headings_as_stored": _headings_as_stored}


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
