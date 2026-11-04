"""Huszar & Buzsaki mouse CA1 and CA3 units on a cheeseboard maze with ripples (DANDI 001880).

What was recorded: mouse iuvE14-cagCre-ef1aDIOchr2-3, 8 Dec 2022, 49 units
in CA1 and CA3 (opto-tagged), during a PRE, a Learning and a POST
sub-session on a 90 cm cheeseboard maze with two reward locations,
separated by rest in the home cage. Position (one LED, cm) at 30 Hz across
all three sub-sessions (64-17,301 s), 5.3 % NaN, most of it off the maze.
The file holds 2,118 ripples, the maze and box intervals, 45 learning trials
and the reward coordinates.

The session is read by ``neurospatial.datasets.load("huszar_cheeseboard",
"sub-iuvE14-cagCre-ef1aDIOchr2-3_ses-iuvE14-cagCre-ef1aDIOchr2-3-221208")``,
whose module (``neurospatial/datasets/huszar_cheeseboard.py``) and docs page
give the file, the conversions (positions as stored in cm; ripples, rewards,
trials and epochs from the file's tables), the declared 90 cm disc of the
paper (Huszár et al. 2026, Neuron, STAR Methods), placed at the on-maze
samples' midrange, with the reward wells as regions, and the watch-outs,
among them the held runs (29 % of the frames, most of them the tracker's
last fix while the mouse is in the holding box). ``_record.from_public``
converts it to the survey's record.

Analysis choices are the public loader's ``params``: ``bin_size`` 2.5 cm,
``min_speed`` 5 cm/s (the paper's), ``analysis_epochs`` the 52 ``onMaze``
intervals, ``dt`` 0.25 s, and the even and odd ``onMaze`` intervals as
train/test epochs.

Survey-only: ``params["replay_dt"]`` 0.02 s; ``events["ripples"]``, the
ripples inside an ``onMaze`` interval (awake ripples on the maze), which
replay is decoded on (the public ``ripples``, every ripple of the session,
are ``events["all_ripples"]``); ``params["nwb_source"]`` and
``params["readers_used"]``, for the ``io.nwb_readers`` operation; the
declared disc's facts the environment fidelity study reads (placed at the
samples' midrange, the stated circle and diameter, each axis's range
against the stated diameter); ``geometry["reward_locations_cm"]``.
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
    "key": "ca1ca3_001880",
    "public": "huszar_cheeseboard",
    "session": "sub-iuvE14-cagCre-ef1aDIOchr2-3_ses-iuvE14-cagCre-ef1aDIOchr2-3-221208",
}

# Pre-registered from the dataset catalog (place-reward task, about 5 % NaN
# position, ripples) and the file's metadata (49 units, position in cm at
# 30 Hz, PRE / Learning / POST sub-sessions, a Ripples table) before the first
# run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 49,
    "n_dims": 2,
    "sampling_rate_hz": Range(29.0, 31.0),
    "file_nan_share": lambda rec: Range(0.03, 0.08).contains(
        float(np.mean(~np.isfinite(rec.positions).all(axis=1)))
    ),
    "maze_extent_cm": lambda rec: Range(80, 120).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "has_ripples": lambda rec: "ripples" in rec.events,
}


def load(cache: Path | None = None) -> Recording:
    """The session from ``neurospatial.datasets``; positions in cm, ripples."""
    from neurospatial.datasets import huszar_cheeseboard
    from neurospatial.datasets import load as load_public

    rec = load_public(DATASET["public"], DATASET["session"], cache=cache)
    source = public_file(DATASET["public"], DATASET["session"], cache=cache)
    ripples = rec.events["ripples"]
    on_maze = rec.epochs["on_maze"]
    inside = (ripples[:, :1] >= on_maze[:, 0]) & (ripples[:, 1:] <= on_maze[:, 1])
    on = inside.any(axis=1)
    tracked = rec.positions[np.isfinite(rec.positions).all(axis=1)]
    spans = np.ptp(tracked, axis=0)
    regions = rec.env.regions
    why = "the declared disc's facts the environment fidelity study reads"
    return from_public(
        rec,
        key=DATASET["key"],
        params={
            "replay_dt": SurveyOnly(0.02, "the replay decoding bin"),
            "nwb_source": SurveyOnly(
                source, "the cached NWB file the session was read from"
            ),
            "readers_used": SurveyOnly(
                ("position", "units"), "the io.nwb readers the public loader calls"
            ),
        },
        events={
            "ripples": SurveyOnly(
                ripples[on],
                f"the {int(on.sum())} of {len(ripples)} ripples inside an onMaze"
                " interval (awake ripples on the maze), which replay is decoded on",
            ),
            "all_ripples": SurveyOnly(np.array(ripples), "every ripple of the session"),
        },
        geometry={
            "declared_from": SurveyOnly(
                "fitted to the samples",
                f"{why}: placed at the on-maze samples' midrange",
            ),
            "shape": SurveyOnly("circle", f"{why}: the stated shape"),
            "size_cm": SurveyOnly(
                huszar_cheeseboard.DIAMETER, f"{why}: the stated diameter"
            ),
            "environment_fit": SurveyOnly(
                {
                    f"size_residual_{axis}": float(
                        span / huszar_cheeseboard.DIAMETER - 1
                    )
                    for axis, span in zip("xy", spans, strict=True)
                },
                f"{why}: the samples' range on each axis against the stated diameter",
            ),
            "reward_locations_cm": SurveyOnly(
                np.array([regions[name].data for name in ("reward_1", "reward_2")]),
                "the file's rewardCoordinates (the public env's reward regions)",
            ),
        },
    )


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
