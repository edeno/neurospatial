"""Kanter et al. rat LEC and CA1 Neuropixels units, open field with novel objects (EBRAINS 10.25493/GTVB-FQR).

What was recorded: rat 29629 ("Karri"), 29 Oct 2024, Neuropixels 2.0 probes in
lateral entorhinal cortex (156 units) and CA1 (224 units; 125 of the 380 are
"good", the rest "mua"), foraging in a 1 m square arena: 605 s empty, then
three novel objects placed one at a time for 450, 523 and 452 s.

The session is read by ``neurospatial.datasets.load("kanter_2025")``, whose
module (``neurospatial/datasets/kanter_2025.py``) and docs page give the
file, the conversions (positions in cm, x and y only; headings wrapped to
[-π, π], the frame compared with movement; units named ``<area>-<id>``
because the table repeats ids across the probes; the object epochs named
from the data descriptor), the declared square (the stated 100 cm arena,
the file's axes kept, placed where it holds the most samples) and the
watch-outs (no object positions in the archive; the stated "Bottom left of
the arena" origin contradicted by the values; the heading offset drifts).
``_record.from_public`` converts it to the survey's record.

Analysis choices are the public loader's ``params``: ``bin_size`` 2.5 cm,
``min_speed`` None, ``dt`` 0.25 s, alternating 60 s ``train_epochs`` /
``test_epochs``; every unit is kept, "mua" included.

Survey-only: ``params["nwb_source"]`` (the cached file),
``params["readers_used"]`` and ``params["unit_quality"]``; the declared box's
facts the environment fidelity study reads (placed on the samples, the
stated square, each axis's range against the stated side).
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
    heading_vs_movement,
    public_file,
    validate_recording,
)

DATASET = {
    "key": "kanter_2025",
    "public": "kanter_2025",
    "session": "29629_2024-Oct-29_15-05-13",
}

# Pre-registered from the data descriptor (OptiTrack head tracking projected on
# the floor; foraging in an empty 1 m arena, then three novel objects for
# 7.5 min each; no object positions listed among the data) and the file's
# metadata (380 unit rows, position and head direction at 120 Hz, four epochs)
# before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 380,
    "n_dims": 2,
    "sampling_rate_hz": Range(119.0, 121.0),
    # No extent expectation: the units check refuses a span outside the
    # stated arenas, so one here could never fail; the span is in
    # ``conversions``.
    "epoch_names": ["empty", "object_1", "object_2", "object_3"],
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
    "headings_follow_movement": lambda rec: bool(
        abs(
            heading_vs_movement(rec.times, rec.positions, rec.headings, min_speed=5.0)[
                0
            ]
        )
        < np.pi / 4
    ),
}


def load(cache: Path | None = None) -> Recording:
    """The session from ``neurospatial.datasets``; positions in cm, object epochs."""
    from neurospatial.datasets import load as load_public

    rec = load_public(DATASET["public"], DATASET["session"], cache=cache)
    source = public_file(DATASET["public"], DATASET["session"], cache=cache)
    side = float(np.diff(rec.env.dimension_ranges[0])[0])
    tracked = rec.positions[np.isfinite(rec.positions).all(axis=1)]
    spans = np.ptp(tracked, axis=0)
    why = "the declared box's facts the environment fidelity study reads"
    return from_public(
        rec,
        key=DATASET["key"],
        geometry={
            "declared_from": SurveyOnly(
                "fitted to the samples",
                f"{why}: placed where it holds the most samples",
            ),
            "shape": SurveyOnly("square", f"{why}: the stated shape"),
            "size_cm": SurveyOnly(side, f"{why}: the stated side"),
            "environment_fit": SurveyOnly(
                {
                    f"size_residual_{axis}": float(span / side - 1)
                    for axis, span in zip("xy", spans, strict=True)
                },
                f"{why}: the samples' range on each axis against the stated side",
            ),
            "object_positions": SurveyOnly(
                None,
                "the archive states no object positions (the public loader's "
                "watch-outs say why)",
            ),
        },
        params={
            "nwb_source": SurveyOnly(
                source, "the cached NWB file the session was read from"
            ),
            "readers_used": SurveyOnly(
                ("position", "head_direction"),
                "the io.nwb readers the public loader calls (read_units refuses "
                "the repeated unit ids)",
            ),
            "unit_quality": SurveyOnly(
                rec.units["cluster_quality"].to_numpy(),
                "the spike sorting's quality label per unit",
            ),
        },
    )


def _units_merged_by_id(rec: Recording) -> Recording:
    # A user keying units by the table's id: the two probes' units that share
    # an id become one train.
    assert rec.spike_times is not None
    assert rec.unit_ids is not None
    plain = np.array([u.split("-", 1)[1] for u in rec.unit_ids])
    merged = {
        uid: np.sort(
            np.concatenate(
                [t for t, p in zip(rec.spike_times, plain, strict=True) if p == uid]
            )
        )
        for uid in dict.fromkeys(plain)
    }
    return replace(
        rec, spike_times=tuple(merged.values()), unit_ids=np.array(list(merged))
    )


HAZARDS = {"units_merged_by_id": _units_merged_by_id}


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
