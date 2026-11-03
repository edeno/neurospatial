"""Rich, Liaw & Lee rat CA1 units on a 48 m linear track, repeat exposure (CRCNS hc-31, animal ANM00190422).

What was recorded: rat ANM00190422, tetrodes in dorsal CA1, 63 units. The
rat first explored a novel track that was lengthened in four steps to 48 m
(run 1), then ran the whole track again (run 2, 21 min, this loader), then
explored a small open field (run 3). The track is folded inside a room about
14 x 10 m. Position from video at 29.97 Hz.

Where it lives: CRCNS hc-31 (doi:10.6080/K0BV7DT2), which needs a free
crcns.org account (``CRCNS_USERNAME``/``CRCNS_PASSWORD``). Everything but
the LFP is one MATLAB v5 file, ``data/resSave.mat`` (166.8 MB), an array of
nine structs of which elements 5-9 are the paper's animals
(``resSave_readme.txt``); this is element 5. Read with ``scipy.io``.

What the loader converts:

- Times (``posMazeLin.data`` column 0 and every ``timeStamps``) are
  microseconds (``unitsPerSecond`` 1e6); divided by 1e6.
- ``posMazeLin(2).data`` column 1, the position along the track, is in
  inches: ``unitsPerCm`` is 0.3937 and ``px`` 2.54. Divided by
  ``unitsPerCm`` it is cm, 0-4815 cm, the stated 48 m; that magnitude is the
  check. The 2-D ``posMaze`` x, y (inches too) go to
  ``geometry["positions_2d"]`` in cm; their steps are checked against the
  linear steps. The velocity column is inches per microsecond (up to about
  6e-5); converted, it is checked against the speed of the track coordinate.
- Units are run 2's spike trains (``spEpochSep(:, 3)``, epoch ``run2``);
  units with no spike in run 2 are dropped. The readme says interneurons
  and poor units have NaN in ``ID`` and excludes them with
  ``all(ID <= 2, 2)``; ``ID`` is uint8, where MATLAB stores NaN as 0, so
  the marker, if it was set, is indistinguishable from "silent" (7 units are
  0 in every column). ``width`` is empty and the stored ``rate`` is at most
  1.29 Hz, so no unit looks like a fast-spiking interneuron; every unit is
  kept.

Analysis choices (``params``): the track coordinate is the 1-D position;
``bin_size`` 2.5 cm (1928 bins); ``min_speed`` 4 cm/s; ``dt`` 0.25 s with
alternating 60 s train/test blocks.

Declared environment (``geometry["environment"]``): the 48 m track (the
hc-31 data description) as one 1-D edge along the authors' track
coordinate, from its smallest to its largest value; the width and the folded
layout are not stated in any source read here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    checked_span,
    validate_recording,
)
from _tracks import declared_geometry, graph_from_points

from neurospatial.datasets._fetch import fetch_crcns

DATASET = {
    "key": "hc31_anm190422",
    "title": "Rich, Liaw & Lee (2014), rat ANM00190422, CA1 units on a 48 m track, second run (CRCNS hc-31)",
    "source": "https://doi.org/10.6080/K0BV7DT2",
    "license": "CRCNS terms of use (account required; cite Rich et al. 2014 or the data set)",
    "download_size": "166.8 MB",
}

# Pre-registered from the hc-31 description and resSave_readme.txt (48 m
# track; posMazeLin is the position along it; run2 is the repeat exposure;
# interneurons and poor units have ID > 2) and the field listing of
# resSave(5) (63 units; posMaze 'run2' 'linear'), before the first run of this
# loader. The sampling rate is a guess (Neuralynx video at 25-30 Hz).
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 1,
    "n_units": Range(20, 63),
    "sampling_rate_hz": Range(24.0, 31.0),
    "position_min": Range(-50.0, 100.0),
    "position_max": Range(4000.0, 5000.0),
}


PATH = "hc-31/data/resSave.mat"
_SHA256 = "753f12752a9f0c52eaca731e947fe9183c994b2b4859e00caa1ec99ccbbe382a"
SUBDIR = "crcns_hc31"
ANIMAL_INDEX = 4  # resSave(5)
RUN = 1  # posMaze(2): 'run2'
SPIKE_EPOCH = 2  # spEpochSep(:, 3): 'run2'


def load(cache: Path | None = None) -> Recording:
    """Fetch resSave.mat; run 2 on the 48 m track as a 1-D position in cm."""
    import scipy.io

    path = fetch_crcns(PATH, sha256=_SHA256, subdir=SUBDIR, cache=cache)
    animal = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)["resSave"][
        ANIMAL_INDEX
    ]
    linear, planar = animal.posMazeLin[RUN], animal.posMaze[RUN]
    if (linear.epoch, linear.topol, planar.epoch) != ("run2", "linear", "run2"):
        msg = f"posMazeLin({RUN + 1}) is {linear.epoch!r} {linear.topol!r}, not run2"
        raise ValueError(msg)
    per_s, per_cm = float(linear.unitsPerSecond), float(linear.unitsPerCm)
    if (per_s, round(per_cm, 4), planar.unitsPerCm) != (1e6, 0.3937, linear.unitsPerCm):
        msg = f"unexpected units: {per_s} per s, {per_cm} per cm"
        raise ValueError(msg)
    times = linear.data[:, 0] / per_s
    track = linear.data[:, 1:2] / per_cm
    span = checked_span(track, 4500.0, 5000.0)
    xy = planar.data[:, 1:3] / per_cm
    if not np.array_equal(planar.data[:, 0], linear.data[:, 0]):
        msg = "posMaze and posMazeLin have different sample times"
        raise ValueError(msg)
    step_track = np.abs(np.diff(track[:, 0]))
    step_xy = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    moving = (step_track > 1.0) & (step_xy > 1.0)
    step_ratio = float(np.median(step_track[moving] / step_xy[moving]))
    if not 0.8 <= step_ratio <= 1.25:
        msg = f"linear and 2-D steps differ by a factor {step_ratio:.2f}"
        raise ValueError(msg)
    trains, ids, silent = [], [], 0
    for index, unit in enumerate(animal.spEpochSep[:, SPIKE_EPOCH]):
        if unit.clustInf.epoch != "run2" or float(unit.unitsPerSecond) != per_s:
            msg = f"unit {index} column {SPIKE_EPOCH + 1} is not run2 in microseconds"
            raise ValueError(msg)
        train = np.atleast_1d(np.asarray(unit.timeStamps, dtype=float)) / per_s
        if train.size == 0:
            silent += 1
            continue
        trains.append(np.sort(train))
        ids.append(f"TT{unit.clustInf.TT}.{unit.clustInf.cellNum}")
    if len(set(ids)) != len(ids):
        ids = [f"{i}:{index}" for index, i in enumerate(ids)]
    run = np.array([[times[0], times[-1]]])
    train, test = alternating_blocks(times[0], times[-1], 60.0)
    stored_speed = linear.data[1:, 3] / per_cm * per_s
    step_speed = step_track / np.diff(times)
    fast = step_speed > 10.0
    speed_ratio = float(np.median(stored_speed[fast] / step_speed[fast]))
    ends = np.nanmin(track[:, 0]), np.nanmax(track[:, 0])
    declared, declared_line = declared_geometry(
        graph_from_points({"start": ends[:1], "end": ends[1:]}, [("start", "end")]),
        track,
        times=times,
        kind=f"1-D track along the authors' coordinate, {ends[1] - ends[0]:.0f} cm",
        source="hc-31 data description: 'a novel 48-m long linear track'; width"
        " not stated",
        bin_size=2.5,
        physical_width=None,
        units="cm",
        name=DATASET["key"],
        placement="the file's track coordinate, from its smallest to its largest value",
        placed_by="file's linearized coordinate",
        stated_lengths={"length": (float(ends[1] - ends[0]), 4800.0)},
        tracked_point="not stated (video, the authors' track coordinate)",
        speed_thresholds=(4.0, 30.0),
    )
    return Recording(
        dataset=DATASET["key"],
        session=f"{linear.animal} {linear.session} run2",
        source=f"{DATASET['source']} {PATH} resSave({ANIMAL_INDEX + 1})",
        license=DATASET["license"],
        times=times,
        positions=track,
        position_units="cm",
        spike_times=tuple(trains),
        unit_ids=np.asarray(ids),
        epochs={"run2": run},
        geometry={
            "shape": "48 m linear track, folded",
            "length_cm": 4800.0,
            "positions_2d": xy,
            **declared,
        },
        params={
            "bin_size": 2.5,
            "min_speed": 4.0,
            "dt": 0.25,
            "train_epochs": train,
            "test_epochs": test,
        },
        conversions=(
            "times: posMazeLin data column 0 in microseconds / 1e6 -> s",
            f"positions: posMazeLin(2) column 1 in inches / unitsPerCm {per_cm:.4f} -> cm;"
            f" {span}",
            f"geometry['positions_2d']: posMaze(2) x, y in inches -> cm; median ratio of"
            f" linear to 2-D step while moving {step_ratio:.3f}",
            f"velocity column: inches per microsecond -> cm/s; median ratio to the"
            f" speed of the track coordinate above 10 cm/s {speed_ratio:.2f} (not used)",
            f"units: spEpochSep(:, 3) run2 trains, microseconds -> s; {len(trains)} kept,"
            f" {silent} without a spike in run 2 dropped; ID is uint8 (no NaN for"
            " interneurons), so none excluded",
            declared_line,
        ),
    )


def inches_as_cm(rec: Recording) -> Recording:
    # A reader taking the stored track coordinate as cm (unitsPerCm ignored).
    assert rec.positions is not None
    return replace(rec, positions=rec.positions * 0.3937007874015748)


def times_in_microseconds(rec: Recording) -> Recording:
    # The file's clock: tracking and spikes both in microseconds, consistently.
    assert rec.times is not None
    assert rec.spike_times is not None
    params = dict(rec.params)
    for key in ("train_epochs", "test_epochs"):
        params[key] = params[key] * 1e6
    return replace(
        rec,
        times=rec.times * 1e6,
        spike_times=tuple(t * 1e6 for t in rec.spike_times),
        epochs={k: v * 1e6 for k, v in rec.epochs.items()},
        params=params,
    )


HAZARDS = {"inches_as_cm": inches_as_cm, "times_in_microseconds": times_in_microseconds}


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
