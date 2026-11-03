"""Grosmark & Buzsaki 2016 rat CA1 on a 1.6 m linear track (CRCNS hc-11), the session also in DANDI 000044.

What was recorded: rat Achilles, 25 Oct 2013, 137 CA1 units (120 putative
pyramidal cells, 17 interneurons) from bilateral silicon probes: about 5 h of
sleep in the home cage (PRE), 35 min running a 1.6 m linear track (MAZE) and
4 h of sleep (POST). One LED tracked at about 39 Hz during MAZE only.

Where it lives: CRCNS hc-11 (doi:10.6080/K0862DC5), which needs a free
crcns.org account (``CRCNS_USERNAME``/``CRCNS_PASSWORD``). The spike times,
cell types, position and epochs of each session are in
``<session>_sessInfo.mat`` (MATLAB v7.3), bundled in
``data/NoveltySessInfoMatFiles.tar.gz`` (277.5 MB); the loader streams that
archive once and keeps the one 38.5 MB member. The same session is in DANDI
000044 as NWB (``load_grosmark_000044_linear``);
``studies/hc11_vs_000044.py`` compares the two.

What the loader converts:

- h5py returns MATLAB arrays transposed; positions become (n_time, 2) and
  intervals (n, 2).
- ``TwoDLocation`` (m) becomes cm; ``OneDLocation`` (m, the linearized track
  coordinate) goes to ``params["linearized"]`` in cm. Untracked samples are
  NaN in the file and stay NaN.
- ``SpikeIDs`` are shank * 100 + cluster. Units are the clusters listed in
  ``PyrIDs`` and ``IntIDs``; the cell type goes to ``params["cell_type"]``.
- Epochs ``PREEpoch``/``MazeEpoch``/``POSTEpoch`` become ``pre``/``maze``/
  ``post``; the state-scoring epochs ``Wake``, ``Drowsy``, ``NREM``,
  ``Intermediate``, ``REM`` become ``wake``, ``drowsy``, ``non_rem``,
  ``intermediate``, ``rem``.
- Replay candidates are population bursts in POST non-REM sleep, found by the
  rule of ``load_grosmark_000044_circular.population_bursts``, as for the
  000044 loader (the file has no ripple times).

Analysis choices (``params``) are those of ``load_grosmark_000044_linear``:
``bin_size`` 2.5 cm, ``min_speed`` 4 cm/s, ``dt`` 0.25 s with alternating
60 s train/test blocks inside MAZE, ``replay_dt`` 0.02 s.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch_crcns_tar_members
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    checked_span,
    validate_recording,
)
from load_grosmark_000044_circular import N_CANDIDATES, population_bursts

DATASET = {
    "key": "hc11_achilles",
    "title": "Grosmark & Buzsaki (2016), Achilles 10252013, rat CA1, 1.6 m linear track (CRCNS hc-11)",
    "source": "https://doi.org/10.6080/K0862DC5",
    "license": "CRCNS terms of use (account required; cite Grosmark & Buzsaki 2016 and the data set)",
    "download_size": "38.5 MB (one member of a 277.5 MB archive)",
}


def _epoch_s(name: str) -> Any:
    return lambda rec: float(np.sum(np.diff(rec.epochs[name], axis=1)))


# Pre-registered from the hc-11 documentation (Sessions_Recordings_Summary.pdf:
# 120 putative pyramidal cells and 17 interneurons; PRE 18,080 s, MAZE 2,068 s,
# POST 14,714 s; position at about 39 Hz, metres) and, for the extent, from the
# same session's DANDI 000044 file (G-039: the 2-D positions span about 2 m),
# before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 137,
    "n_dims": 2,
    "sampling_rate_hz": Range(38.0, 40.0),
    "position_extent_cm": lambda rec: Range(150, 260).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "has_pre_maze_post": lambda rec: {"pre", "maze", "post"} <= set(rec.epochs),
    "pre_s": lambda rec: Range(18070, 18090).contains(_epoch_s("pre")(rec)),
    "maze_s": lambda rec: Range(2058, 2078).contains(_epoch_s("maze")(rec)),
    "post_s": lambda rec: Range(14704, 14724).contains(_epoch_s("post")(rec)),
}


ARCHIVE = "hc-11/data/NoveltySessInfoMatFiles.tar.gz"
MEMBER = "NoveltySessInfoMatFiles/Achilles_10252013_sessInfo.mat"
_SHA256 = "94ea9a03b77600d9015b902c1241ecaf8ccc360b70d9a9ea13db4aaeeedac18b"
SUBDIR = "crcns_hc11"
_SESSION_EPOCHS = {"PREEpoch": "pre", "MazeEpoch": "maze", "POSTEpoch": "post"}
_STATES = {
    "Wake": "wake",
    "Drowsy": "drowsy",
    "NREM": "non_rem",
    "Intermediate": "intermediate",
    "REM": "rem",
}


def read_sess_info(path: Path) -> dict[str, Any]:
    """Arrays of a ``*_sessInfo.mat`` as stored, with MATLAB's transposition undone."""
    import h5py

    with h5py.File(path, "r") as f:
        s = f["sessInfo"]
        return {
            "spike_times": s["Spikes/SpikeTimes"][:].ravel(),
            "spike_ids": s["Spikes/SpikeIDs"][:].ravel().astype(np.int64),
            "pyr_ids": s["Spikes/PyrIDs"][:].ravel().astype(np.int64),
            "int_ids": s["Spikes/IntIDs"][:].ravel().astype(np.int64),
            "times": s["Position/TimeStamps"][:].ravel(),
            "xy_m": s["Position/TwoDLocation"][:].T,
            "linear_m": s["Position/OneDLocation"][:].ravel(),
            "maze_type": "".join(map(chr, s["Position/MazeType"][:].ravel())),
            "epochs": {
                name: s[f"Epochs/{name}"][:].T for name in (*_SESSION_EPOCHS, *_STATES)
            },
            "duration_s": float(s["Epochs/sessDuration"][0, 0]),
        }


def load(cache: Path | None = None) -> Recording:
    """Fetch the session's sessInfo file; positions in cm, spikes, epochs, candidates."""
    path = fetch_crcns_tar_members(
        ARCHIVE, {MEMBER: _SHA256}, subdir=SUBDIR, cache=cache
    )[MEMBER]
    raw = read_sess_info(path)
    if raw["maze_type"].strip() != "1.6m Linear Maze":
        msg = f"MazeType is {raw['maze_type']!r}, not the 1.6 m linear maze"
        raise ValueError(msg)
    times = raw["times"]
    positions = raw["xy_m"] * 100.0
    span = checked_span(positions, 150.0, 260.0)
    linearized = raw["linear_m"] * 100.0
    track_length = float(np.nanmax(linearized) - np.nanmin(linearized))
    if not 150.0 <= track_length <= 170.0:
        msg = f"OneDLocation spans {track_length:.0f} cm, not the 1.6 m track"
        raise ValueError(msg)
    epochs = {key: raw["epochs"][name] for name, key in _SESSION_EPOCHS.items()}
    epochs |= {key: raw["epochs"][name] for name, key in _STATES.items()}
    cell_type = {int(i): "pyramidal" for i in raw["pyr_ids"]} | {
        int(i): "interneuron" for i in raw["int_ids"]
    }
    unit_ids = np.array(sorted(cell_type))
    order = np.argsort(raw["spike_ids"], kind="stable")
    ids, spike_times = raw["spike_ids"][order], raw["spike_times"][order]
    bounds = np.searchsorted(ids, unit_ids), np.searchsorted(ids, unit_ids, "right")
    trains = tuple(np.sort(spike_times[a:b]) for a, b in zip(*bounds, strict=True))
    other = np.setdiff1d(np.unique(ids), unit_ids)
    n_other = int(np.isin(ids, other).sum())
    maze, post = epochs["maze"], epochs["post"]
    post_nrem = epochs["non_rem"][
        (epochs["non_rem"][:, 0] >= post[0, 0])
        & (epochs["non_rem"][:, 1] <= post[0, 1])
    ]
    bursts = population_bursts(trains, post_nrem)
    train, test = alternating_blocks(maze[0, 0], maze[0, 1], 60.0)
    finite = np.isfinite(linearized)
    return Recording(
        geometry={
            "tracked_point": "one LED (per the loader: one LED tracked at about 39 Hz)"
        },
        dataset=DATASET["key"],
        session="Achilles_10252013",
        source=f"{DATASET['source']} {ARCHIVE}:{MEMBER}",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        spike_times=trains,
        unit_ids=unit_ids,
        epochs=epochs,
        events={"candidate_events": bursts[:N_CANDIDATES], "all_bursts": bursts},
        params={
            "bin_size": 2.5,
            "min_speed": 4.0,
            "dt": 0.25,
            "replay_dt": 0.02,
            "analysis_epochs": maze,
            "train_epochs": train,
            "test_epochs": test,
            "linearized": linearized,
            "cell_type": np.array([cell_type[i] for i in unit_ids]),
        },
        conversions=(
            f"positions: Position.TwoDLocation transposed, m times 100 -> cm; {span}",
            f"params['linearized']: Position.OneDLocation, m times 100 -> cm,"
            f" {np.nanmin(linearized):.0f} to {np.nanmax(linearized):.0f} cm, defined"
            f" on {finite.mean():.1%} of samples;"
            f" {np.mean(linearized[finite] == 0):.2%} of finite samples are exactly 0",
            f"units: SpikeIDs (shank * 100 + cluster) listed in PyrIDs ({raw['pyr_ids'].size})"
            f" or IntIDs ({raw['int_ids'].size}); {other.size} other ids with"
            f" {n_other} spikes dropped",
            "epochs: PRE/Maze/POSTEpoch and the five scored states, transposed to"
            " (n, 2)",
            f"events['candidate_events']: first {N_CANDIDATES} of {len(bursts)}"
            " population bursts in POST non-REM (rule of the 000044 loader)",
        ),
    )


def _epochs_left_transposed(rec: Recording) -> Recording:
    # A reader that keeps h5py's (2, n) layout of MATLAB intervals.
    params = dict(rec.params)
    for key in ("analysis_epochs", "train_epochs", "test_epochs"):
        params[key] = np.asarray(params[key]).T
    return replace(rec, params=params)


HAZARDS = {"epochs_left_transposed": _epochs_left_transposed}


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
