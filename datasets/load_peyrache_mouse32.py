"""Peyrache et al. mouse anterodorsal thalamus head-direction session Mouse32-140822.

What was recorded: 49 units (31 labelled ``adn``, 18 ``thalamus``) across a
sleep, wake, sleep session; head direction tracked at about 39 Hz during the
wake epoch only. The file has no x/y position.

The session is read by ``neurospatial.datasets.load("peyrache_2015")``,
whose module (``neurospatial/datasets/peyrache_2015.py``) and docs page give
the file (OSF ``jb2gd``, the file pynapple's and NeMoS's head-direction
tutorials download), the conversions (headings wrapped to [-π, π], frame
assumed) and the watch-outs. ``_record.from_public`` converts it to the
survey's record.

Analysis choices are the public loader's ``params``: ``analysis_epochs``,
the wake epoch, and ``dt`` 0.1 s.

Survey-only: ``params["nwb_source"]`` (the cached file) and
``params["readers_used"]``, for the ``io.nwb_readers`` operation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import (
    Recording,
    SurveyOnly,
    check_expectations,
    from_public,
    public_file,
    validate_recording,
)

DATASET = {
    "key": "peyrache_mouse32",
    "public": "peyrache_2015",
    "session": "Mouse32-140822",
}

# Pre-registered from pynapple's head-direction tutorial, before the first run
# of this loader. Its "31 units in ADn" needs per-unit regions, which a
# Recording does not hold; the head-direction reference check verifies it.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 49,
    "epochs_include_wake_and_sleep": lambda rec: {"wake", "sleep"} <= set(rec.epochs),
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}


def load(cache: Path | None = None) -> Recording:
    """The session from ``neurospatial.datasets``; headings only."""
    from neurospatial.datasets import load as load_public

    rec = load_public(DATASET["public"], DATASET["session"], cache=cache)
    source = public_file(DATASET["public"], DATASET["session"], cache=cache)
    return from_public(
        rec,
        key=DATASET["key"],
        params={
            "nwb_source": SurveyOnly(
                source, "the cached NWB file the session was read from"
            ),
            "readers_used": SurveyOnly(
                ("head_direction", "units"),
                "the io.nwb readers the public loader calls",
            ),
        },
    )


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(
        f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s tracked"
    )
    print(f"  epochs: {', '.join(f'{k} {len(v)}' for k, v in rec.epochs.items())}")
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
