"""Overall & Kempermann mouse Morris water maze swim paths (Figshare 10248158), behavior only.

What was recorded: C57BL/6 and DBA/2 mice (5 per strain) in a Morris water
maze: 3 days of training and 2 days of reversal, 6 trials a day; the last
trial of days 3 and 5 is a probe trial with no platform. Swim paths were
tracked with Noldus EthoVision 3 at 12.5 samples/s and exported with Rtrack
0.9.3 into one JSON file with each trial's arena description.

Where it lives: Figshare 10248158 v3 (CC BY 4.0), ``Experiment.json``
(9.9 MB). The file is read as plain JSON (no code runs from it).

What the loader keeps: one mouse, ``B6_6`` (C57BL/6), all 30 of its
trials. All trials share one pool (``arena.bounds`` circle centre
(133.655, 103.538), radius 95); the goal is the training platform
(121.893, 154.683) on days 1-3 and the reversal platform (140.427, 51.324)
on days 4-5, both radius 10.

What the loader converts:

- ``raw.x``, ``raw.y`` and ``raw.t`` are used; Rtrack's ``x``, ``y``, ``t``
  are the same samples rescaled to the pool radius and trial length. The
  raw coordinates are EthoVision's tracking units. Neither the file, the
  Figshare record nor the Rtrack export states their unit or a pixels-per-cm
  calibration, so ``position_units`` is ``"arbitrary"`` (one unit = 1/95 of
  the pool radius). The values are range checked against the stated pool.
- Empty strings in ``raw.x``/``raw.y`` (untracked samples) become NaN.
- Each trial's clock starts at 0. The file has no time between trials, so
  trials are laid on one clock 200 s apart, in day and trial order, and a
  NaN sample 0.08 s after each trial's last sample marks the end of tracking
  before the gap.

Analysis choices (``params``): ``bin_size`` 5 units (about 1/19 of the pool
radius); no ``min_speed``: a speed threshold needs a length unit, which the
file does not give.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch
from _record import Range, Recording, check_expectations, validate_recording

DATASET = {
    "key": "watermaze_10248158",
    "title": "Overall & Kempermann, C57BL/6 mouse B6_6, 30 water-maze trials (behavior only)",
    "source": "https://doi.org/10.6084/m9.figshare.10248158.v3",
    "license": "CC-BY-4.0",
    "download_size": "9.9 MB",
}

# Pre-registered from the Figshare description (3 training + 2 reversal days,
# 6 trials a day, last trial of days 3 and 5 a probe) and the JSON's first
# track (raw.t every 0.08 s; arena circle radius 95 in tracking units)
# before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 2,
    "sampling_rate_hz": Range(12.4, 12.6),
    "behavior_only": lambda rec: rec.spike_times is None,
    "thirty_trials": lambda rec: len(rec.epochs["trials"]) == 30,
    "two_probe_trials": lambda rec: len(rec.epochs["probe_trials"]) == 2,
    "inside_the_pool": lambda rec: bool(
        np.nanmax(np.hypot(*(rec.positions - rec.geometry["pool_centre"]).T))
        <= 1.1 * rec.geometry["pool_radius"]
    ),
}

_URL = "https://ndownloader.figshare.com/files/21738963"
_SHA256 = "b31d48cd4fde29ebdd883f704f45e75937456f4aa68eb37153d5238ccd0fe531"
_MOUSE = "B6_6"
_TRIAL_SPACING_S = 200.0
_SAMPLE_S = 0.08


def _numbers(text: str) -> np.ndarray:
    return np.array([float(v) if v else np.nan for v in text.split(",")])


def _circle(text: str) -> tuple[np.ndarray, float]:
    kind, x, y, radius = text.split()
    if kind != "circle":
        msg = f"expected a circle arena description, got {text!r}"
        raise ValueError(msg)
    return np.array([float(x), float(y)]), float(radius)


def load(cache: Path | None = None) -> Recording:
    """Fetch the experiment and return mouse B6_6's 30 trials on one clock."""
    path = fetch(
        _URL, "Experiment.json", sha256=_SHA256, subdir=DATASET["key"], cache=cache
    )
    tracks = json.loads(path.read_text())["data"]
    trials = sorted(
        (t for t in tracks if t["target"] == _MOUSE),
        key=lambda t: (int(t["day"]), int(t["trial"])),
    )
    pools = {t["arena"]["arena.bounds"] for t in trials}
    if len(pools) != 1:
        msg = f"{_MOUSE}'s trials use more than one pool: {sorted(pools)}"
        raise ValueError(msg)
    centre, radius = _circle(pools.pop())
    times, positions, windows, probes, goals = [], [], [], [], []
    for index, trial in enumerate(trials):
        t = _numbers(trial["raw.t"])
        xy = np.column_stack([_numbers(trial["raw.x"]), _numbers(trial["raw.y"])])
        start = index * _TRIAL_SPACING_S
        times.append(np.append(start + t, start + t[-1] + _SAMPLE_S))
        positions.append(np.vstack([xy, [np.nan, np.nan]]))
        windows.append([start + t[0], start + t[-1]])
        if trial["factor_Probe"] == "TRUE":
            probes.append(windows[-1])
        goals.append(_circle(trial["arena"]["goal"])[0])
    xy = np.vstack(positions)
    distance = np.hypot(*(xy - centre).T)
    if np.nanmax(distance) > 1.1 * radius:
        msg = (
            f"samples reach {np.nanmax(distance):.1f} units from the pool centre,"
            f" beyond 1.1 x the stated radius {radius:g}"
        )
        raise ValueError(msg)
    n_missing = int(np.isnan(xy).any(axis=1).sum()) - len(trials)
    return Recording(
        dataset=DATASET["key"],
        session=_MOUSE,
        source=_URL,
        license=DATASET["license"],
        times=np.concatenate(times),
        positions=xy,
        position_units="arbitrary",
        epochs={
            "trials": np.asarray(windows),
            "probe_trials": np.asarray(probes).reshape(-1, 2),
        },
        geometry={
            "pool_centre": centre,
            "pool_radius": radius,
            "goal_per_trial": np.asarray(goals),
            "units": "EthoVision tracking units (not stated)",
        },
        params={"bin_size": 5.0},
        conversions=(
            "positions: raw.x, raw.y as stored; unit not stated (arbitrary)",
            f"positions: range checked: farthest sample {np.nanmax(distance):.1f} units"
            f" from the pool centre (stated radius {radius:g}, limit 1.1 x radius)",
            f"positions: {n_missing} empty-string samples -> NaN",
            f"times: each trial's raw.t (from 0) + {_TRIAL_SPACING_S:g} s x trial"
            " index, trials in day/trial order",
            f"times: one NaN sample {_SAMPLE_S:g} s after each trial's last sample",
        ),
    )


def trials_back_to_back(rec: Recording) -> Recording:
    # A reader that strings the trials' tracked samples on one running clock:
    # the jump from the platform to the next release point looks like a
    # 0.08 s swim across the pool.
    assert rec.positions is not None
    tracked = rec.positions[np.isfinite(rec.positions).all(axis=1)]
    return replace(rec, times=_SAMPLE_S * np.arange(len(tracked)), positions=tracked)


HAZARDS = {"trials_back_to_back": trials_back_to_back}


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    n_trials = len(rec.epochs["trials"])
    print(f"{rec.dataset}: {len(rec.times)} samples in {n_trials} trials")
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
