"""Sargolini et al. (2006) rat medial entorhinal cortex, open field (DANDI 000582).

What was recorded: six units from layer III of medial entorhinal cortex in a
Long Evans rat running for 600 s in a 1 m × 1 m enclosure (session
``sub-10884_ses-03080402``, the session pynapple's null-distribution tutorial
uses), one tracking LED at 50 Hz.

The session is read by ``neurospatial.datasets.load("sargolini_2006",
"sub-10884_ses-03080402")``, whose module
(``neurospatial/datasets/sargolini_2006.py``) and docs page give the file,
the conversions (positions in cm), the declared 100 cm square (placed at the
samples' midrange, the file's axes kept) and the watch-outs.
``_record.from_public`` converts it to the survey's record.

Analysis choices are the public loader's ``params``: ``bin_size`` 2.5 cm,
``dt`` 0.25 s, no speed filter, alternating 60 s train/test blocks.

Survey-only: ``params["nwb_source"]`` (the cached file) and
``params["readers_used"]``, for the ``io.nwb_readers`` operation; the
declared box's facts the environment fidelity study reads (placed at the
samples' midrange, the stated square, the file's ``session_description``,
each axis's range against the stated side).
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
    public_file,
    validate_recording,
)

DATASET = {
    "key": "sargolini_000582",
    "public": "sargolini_2006",
    "session": "sub-10884_ses-03080402",
}

# Pre-registered from the dataset description and pynapple's tutorial, before
# the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 6,
    "n_dims": 2,
    "sampling_rate_hz": Range(49.0, 51.0),
    "position_min": Range((-55.0, -55.0), (0.0, 0.0)),
    "position_max": Range((0.0, 0.0), (55.0, 55.0)),
}


def load(cache: Path | None = None) -> Recording:
    """The session from ``neurospatial.datasets``; positions in cm."""
    from neurospatial.datasets import load as load_public
    from neurospatial.datasets import sargolini_2006
    from neurospatial.datasets._fetch import open_nwb

    rec = load_public(DATASET["public"], DATASET["session"], cache=cache)
    source = public_file(DATASET["public"], DATASET["session"], cache=cache)
    with open_nwb(source) as nwbfile:
        description = nwbfile.session_description
    tracked = rec.positions[np.isfinite(rec.positions).all(axis=1)]
    spans = np.ptp(tracked, axis=0)
    why = "the declared box's facts the environment fidelity study reads"
    return from_public(
        rec,
        key=DATASET["key"],
        geometry={
            "declared_from": SurveyOnly(
                "fitted to the samples", f"{why}: placed at the samples' midrange"
            ),
            "shape": SurveyOnly("square", f"{why}: the stated shape"),
            "size_cm": SurveyOnly(sargolini_2006.SIDE, f"{why}: the stated side"),
            "description": SurveyOnly(
                description, f"{why}: the file's session_description"
            ),
            "environment_fit": SurveyOnly(
                {
                    f"size_residual_{axis}": float(span / sargolini_2006.SIDE - 1)
                    for axis, span in zip("xy", spans, strict=True)
                },
                f"{why}: the samples' range on each axis against the stated side",
            ),
        },
        params={
            "nwb_source": SurveyOnly(
                source, "the cached NWB file the session was read from"
            ),
            "readers_used": SurveyOnly(
                ("position", "units"), "the io.nwb readers the public loader calls"
            ),
        },
    )


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s")
    print(
        f"  positions {np.nanmin(rec.positions, 0)} to {np.nanmax(rec.positions, 0)} cm"
    )
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
