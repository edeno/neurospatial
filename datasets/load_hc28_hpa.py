"""Jadhav & Frank rat HPa CA1 and PFC units on a W-track, day 1, first W-track run (CRCNS hc-28).

What was recorded: rat HPa, tetrodes in dorsal CA1, intermediate CA1 and
prefrontal cortex, learning a continuous alternation task on a W-track
(about 80 x 80 cm). Day 1 has seven epochs: sleep, a linear-track run,
rest, a W-track run, rest, a W-track run and sleep. This loader takes the
first W-track run (epoch 4, 20 min). Two head-mounted LEDs tracked by video
at 30 Hz.

Where it lives: CRCNS hc-28 (doi:10.6080/K02N50G9), which needs a free
crcns.org account (``CRCNS_USERNAME``/``CRCNS_PASSWORD``). Each animal is
one ``data/<animal>_data.tar.gz`` (HPa: 2.8 GB, almost all LFP). The loader
streams it once and keeps the day's ``HParawpos01.mat``, ``HPapos01.mat``,
``HPaspikes01.mat``, ``HPatask01.mat`` and ``HPatetinfo.mat`` (MATLAB v5,
nested cell arrays day > epoch > tetrode > unit).

What the loader converts:

- ``rawpos`` (time, x1, y1, x2, y2; video pixels, not interpolated) is the
  position source. Frames where the LEDs were not tracked hold 0 for every
  coordinate (the documentation says so); they become NaN. Position is the
  midpoint of the LEDs, times ``cmperpixel`` (0.65) from ``pos`` (the
  documentation calls it ``cmperpix``), so cm.
- Head direction is not kept. The angle from the back LED to the front LED
  (x1, y1 in front) minus the direction of movement while running has a
  circular mean of -25.5 deg and a median spread of 51 deg with one-frame
  steps, but ``rawpos`` is whole pixels, so one-frame directions are
  quantised; over 15 frames (0.5 s) it is -14.4 deg and 18.8 deg, over 30
  frames -12.7 and 17.1. The authors' ``pos`` ``dir`` against their smoothed
  positions gives -21.9 and 28.0. What remains looks like a mounting offset
  of about -15 to -20 deg plus a direction-dependent residual (sector means
  -30 to +3 deg at 15 frames). The mean stays outside the frame check's
  0.2 rad at every lag, so the heading is not kept rather than corrected by
  a rotation fitted to the data. Both lags are recorded in ``conversions``.
- The untracked zeros sit at the pixel origin, which is outside the track:
  the driver's generic ``missing_positions_as_zero`` hazard (NaN -> 0) puts
  exactly those frames back, so it is this file's own defect.
- Spike times are column 0 of each unit's ``data`` (seconds, on the same
  clock as ``rawpos``); every clustered unit of the epoch is kept, with its
  tetrode's area from ``tetinfo``.

Analysis choices (``params``): ``bin_size`` 2.5 cm; ``min_speed`` 4 cm/s, as
for the other tracks; ``dt`` 0.25 s with alternating 60 s train/test blocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch_crcns_tar_members
from _formats import two_led_heading
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    checked_span,
    heading_vs_movement,
    validate_recording,
)

DATASET = {
    "key": "hc28_hpa",
    "title": "Jadhav & Frank, rat HPa day 1 W-track run, CA1 and PFC units (CRCNS hc-28)",
    "source": "https://doi.org/10.6080/K02N50G9",
    "license": "CRCNS terms of use (account required; lab asks for authorship, cite the data set)",
    "download_size": "21.3 MB (members of a 2.8 GB archive)",
}

# Pre-registered from the hc-28 data description (rawpos: non-interpolated
# video positions, zeros where untracked; pos: interpolated at 30 Hz with
# cmperpix; W-track alternation), the task file (day 1 epoch 2 is a run on
# 'wtr1') and the published W-track size (about 80 x 80 cm), before the first
# run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 2,
    "n_units": Range(10, 120),
    "sampling_rate_hz": Range(29.0, 31.0),
    "position_extent_cm": lambda rec: Range(60, 120).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "run_lasts_10_to_30_min": lambda rec: Range(600, 1800).contains(
        float(rec.times[-1] - rec.times[0])
    ),
    "untracked_frames_present": lambda rec: bool(np.isnan(rec.positions).any()),
}


ARCHIVE = "hc-28/data/HPa_data.tar.gz"
SUBDIR = "crcns_hc28"
EPOCH = 4  # day 1: sleep, lin, rest, wtr1, rest, wtr1, sleep
RUNNING_CM_S = 10.0
LONG_LAG = 15  # frames, 0.5 s
_SHA256 = {
    "HPa_data/HParawpos01.mat": "e8b19d39d2ebfddc6930c3bd0badcceb474034b061bea58a05184b5e837d5fc6",
    "HPa_data/HPapos01.mat": "a8986576d39311cbb29bd22c95a3ab95270d78881db36a8bb7ad293e2c4286f2",
    "HPa_data/HPaspikes01.mat": "8ff174df29fd5bcbdbd392532cbb0a457beb43dfa0eaa440b3e37ea26d52e4e8",
    "HPa_data/HPatask01.mat": "b8ad8699930ab80c1713d3880cb9b89c7672f01be0e91088c3ae9d9c9e0c963c",
    "HPa_data/HPatetinfo.mat": "61dc6a25c5f1131d7edd7258487df29e18275c7b4163b247331aa0cc6c352929",
}


def _cell(path: Path, name: str) -> Any:
    import scipy.io

    return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)[name]


def load(cache: Path | None = None) -> Recording:
    """Fetch day 1; the first W-track run in cm with two-LED headings and units."""
    files = fetch_crcns_tar_members(ARCHIVE, _SHA256, subdir=SUBDIR, cache=cache)
    path = {Path(k).stem: v for k, v in files.items()}
    task = _cell(path["HPatask01"], "task")[EPOCH - 1]
    if (task.type, task.environment) != ("run", "wtr1"):
        msg = f"day 1 epoch {EPOCH} is {task.type!r} on {task.environment!r}, not wtr1"
        raise ValueError(msg)
    raw = _cell(path["HParawpos01"], "rawpos")[EPOCH - 1]
    pos = _cell(path["HPapos01"], "pos")[EPOCH - 1]
    if str(raw.fields).split() != ["timestamp", "x1", "y1", "x2", "y2"]:
        msg = f"rawpos fields are {raw.fields!r}"
        raise ValueError(msg)
    data = np.asarray(raw.data, dtype=float)
    times = data[:, 0]
    leds = data[:, 1:5].copy()
    untracked = np.all(leds == 0, axis=1)
    leds[untracked] = np.nan
    cm_per_px = float(pos.cmperpixel)
    first, second = leds[:, :2] * cm_per_px, leds[:, 2:] * cm_per_px
    positions = (first + second) / 2.0
    span = checked_span(positions, 60.0, 120.0)
    # rawpos is whole pixels: one-frame steps snap the direction of movement
    # to a few angles, so the check is also made over LONG_LAG frames.
    offsets = {
        (order, lag): heading_vs_movement(
            times,
            positions,
            two_led_heading(front, back),
            min_speed=RUNNING_CM_S,
            lag=lag,
        )
        for order, (front, back) in {
            "x1, y1": (first, second),
            "x2, y2": (second, first),
        }.items()
        for lag in (1, LONG_LAG)
    }
    pos_data = np.asarray(pos.data, dtype=float)
    if not np.array_equal(pos_data[:, 0], times):
        msg = "pos and rawpos have different sample times"
        raise ValueError(msg)
    their_offset = heading_vs_movement(
        times, pos_data[:, 5:7], pos_data[:, 3], min_speed=RUNNING_CM_S
    )
    pos_diff = np.linalg.norm(positions - pos_data[:, 1:3], axis=1)

    spikes = _cell(path["HPaspikes01"], "spikes")[EPOCH - 1]
    tetinfo = _cell(path["HPatetinfo"], "tetinfo")[0][EPOCH - 1]
    trains, ids, areas, n_empty = [], [], [], 0
    for t, tetrode in enumerate(np.atleast_1d(spikes), start=1):
        for u, unit in enumerate(np.atleast_1d(tetrode), start=1):
            if not hasattr(unit, "data"):
                continue
            # One row per spike; a unit with one spike is squeezed to 1-D.
            unit_data = np.atleast_2d(np.asarray(unit.data, dtype=float))
            if unit_data.size == 0:
                n_empty += 1
                continue
            trains.append(np.sort(unit_data[:, 0]))
            ids.append(f"{t}.{u}")
            areas.append(str(getattr(np.atleast_1d(tetinfo)[t - 1], "area", "")))
    train, test = alternating_blocks(times[0], times[-1], 60.0)
    return Recording(
        dataset=DATASET["key"],
        session="HPa day 1 epoch 4 (wtr1)",
        source=f"{DATASET['source']} {ARCHIVE}",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        spike_times=tuple(trains),
        unit_ids=np.asarray(ids),
        epochs={"run": np.array([[times[0], times[-1]]])},
        geometry={"shape": "W-track", "cm_per_pixel": cm_per_px},
        params={
            "bin_size": 2.5,
            "min_speed": 4.0,
            "dt": 0.25,
            "train_epochs": train,
            "test_epochs": test,
            "area": np.asarray(areas),
        },
        conversions=(
            f"positions: rawpos LED midpoint times cmperpixel {cm_per_px} -> cm; frames"
            f" with every coordinate 0 ({untracked.mean():.1%}) -> NaN; {span}; median"
            f" {np.nanmedian(pos_diff):.1f} cm from the authors' interpolated pos x, y",
            "headings: not kept; heading minus movement direction above"
            f" {RUNNING_CM_S:g} cm/s (circular mean, median spread, deg):"
            + "; ".join(
                f" {order} LED in front, lag {lag} {np.degrees(m):.1f},"
                f" {np.degrees(w):.1f}"
                for (order, lag), (m, w) in offsets.items()
            )
            + f"; the authors' pos dir against their smoothed x, y"
            f" {np.degrees(their_offset[0]):.1f}, {np.degrees(their_offset[1]):.1f};"
            " |mean| above 0.2 rad at every lag",
            f"units: {len(trains)} clustered units of day 1 epoch {EPOCH} ({n_empty} with"
            " no spikes dropped), spike times"
            f" column 0 (s); areas {sorted(set(areas))}",
        ),
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
