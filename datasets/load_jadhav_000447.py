"""Shin & Jadhav rat CA1 and PFC units on a W-track (DANDI 000447).

What was recorded: rat JDS-NFN-AM2, tetrodes in dorsal CA1 and prefrontal
cortex (57 units), running a W-track between three reward wells over three
run epochs (104 trajectories, each labelled with its start and end well,
inbound or outbound, and correct or not). Position at about 33 Hz.

Where it lives: DANDI 000447 version 0.250724.2224, CC-BY-4.0. The file is
5.1 GB, almost all raw voltage, so it is read by HTTP byte range
(``_fetch.open_nwb``) after checking DANDI's published SHA-256.

What the loader converts: ``Position/SpatialSeries`` has three columns under
the unit ``"centimeters; centimeters/second"``: x and y in cm, then speed in
cm/s (its values match the speed computed from x and y; see
``conversions``). The loader keeps x and y. Trajectories become epochs
(``trials``, ``inbound``, ``outbound``) and their wells ``events``.

Analysis choices (``params``): ``bin_size`` 2.5 cm; ``min_speed`` 4 cm/s, as
for the other tracks; ``dt`` 0.25 s; trajectories alternate between training
and testing the decoder.

Two apparatus, two recordings. The file's run epochs are on two W-tracks
(the dandiset description: a novel W-track in runs 1 and 3, a familiar one
in run 2) that lie rotated 90 degrees in one camera frame, so one
environment cannot hold both: their arms cross. This loader returns the
novel W-track (runs 1 and 3); ``load_jadhav_000447_familiar`` returns run 2
(``load_maze``). Positions outside the maze's runs are NaN, trajectories are
the maze's own, and ``analysis_epochs`` are its runs.

Declared environment (``geometry["environment"]``): the W-track as a graph
of three parallel arms joined by a base, its node positions fitted to the
maze's samples, with the 7 cm track width Tang, Shin & Jadhav 2023 (the
paper the dandiset is described by, STAR Methods) state for both mazes
("~80 x 80 cm with ~7-cm wide tracks").
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import dandi_stream_url, open_nwb
from _record import (
    Range,
    Recording,
    check_expectations,
    checked_span,
    validate_recording,
)
from _tracks import (
    arms_text,
    declared_geometry,
    edge_lengths,
    fitted_maze,
    parallel_arm_candidates,
    parallel_arm_lengths,
)

DATASET = {
    "key": "jadhav_000447",
    "title": "Shin & Jadhav, rat JDS-NFN-AM2, CA1 and PFC units on the novel W-track",
    "source": "https://dandiarchive.org/dandiset/000447/0.250724.2224",
    "license": "CC-BY-4.0",
    "download_size": "5.05 GB (streamed)",
}

# Pre-registered from the dataset catalog (57 units, about 32.9 Hz, x and y in cm plus a velocity column) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 57,
    "n_dims": 2,
    "sampling_rate_hz": Range(31.0, 35.0),
}


_ASSET_ID = "b048bd0a-48a4-4d19-ad96-fd879666bca5"
_SHA256 = "aac7a7c12e7d129f6c6733b462c1e4fa6ee1d4e6d0893af347c78baf3f7ec97d"
_PATH = "sub-JDS-NFN-AM2/sub-JDS-NFN-AM2_behavior+ecephys.nwb"


# Rows of the file's run epochs ('epoch intervals') on each maze.
MAZE_RUNS = {"novel": (0, 2), "familiar": (1,)}
TRACK_WIDTH_CM = 7.0
FIT_CAP_CM = 10.0


def load(cache: Path | None = None) -> Recording:
    """Stream the session; the novel W-track's runs, x, y in cm, spikes, trajectories."""
    del cache  # read by byte range: nothing is cached
    return load_maze("novel", DATASET)


def load_maze(maze: str, dataset: dict[str, str]) -> Recording:
    """One maze's runs: positions outside them NaN, its trajectories, its declared W."""
    from neurospatial.io.nwb import read_units

    source = dandi_stream_url(_ASSET_ID, sha256=_SHA256)
    with open_nwb(source) as nwbfile:
        series = nwbfile.processing["behavior"]["Position"]["SpatialSeries"]
        data = np.asarray(series.data[:], dtype=float)
        times = np.asarray(series.timestamps[:], dtype=float)
        unit_label = series.unit
        units = read_units(nwbfile)
        trials = nwbfile.trials.to_dataframe()
        run_epochs = nwbfile.intervals["epoch intervals"].to_dataframe()

    if unit_label != "centimeters; centimeters/second" or data.shape[1] != 3:
        msg = f"expected x, y, speed under 'centimeters; centimeters/second', got {unit_label!r}"
        raise ValueError(msg)
    positions = data[:, :2]
    span = checked_span(positions, 80.0, 200.0)
    step_speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
    finite = np.isfinite(step_speed) & np.isfinite(data[1:, 2])
    agreement = float(np.corrcoef(step_speed[finite], data[1:, 2][finite])[0, 1])
    # A scale-free correlation cannot confirm cm/s; the magnitudes must agree too.
    moving = finite & (step_speed > 5.0)
    scale = float(np.median(data[1:, 2][moving]) / np.median(step_speed[moving]))
    if not 0.5 <= scale <= 2.0 or agreement < 0.5:
        msg = (
            f"column 2 is not the speed of x, y (scale {scale:.2f}, r {agreement:.2f})"
        )
        raise ValueError(msg)
    runs = run_epochs[["start_time", "stop_time"]].to_numpy(dtype=float)
    if len(runs) != 3:
        msg = f"expected three run epochs (novel, familiar, novel), got {len(runs)}"
        raise ValueError(msg)
    maze_runs = runs[list(MAZE_RUNS[maze])]
    in_maze = np.zeros(len(times), dtype=bool)
    for start, stop in maze_runs:
        in_maze |= (times >= start) & (times < stop)
    positions = np.where(in_maze[:, None], positions, np.nan)
    all_windows = trials[["start_time", "stop_time"]].to_numpy(dtype=float)
    mine = np.zeros(len(all_windows), dtype=bool)
    for start, stop in maze_runs:
        mine |= (all_windows[:, 0] >= start) & (all_windows[:, 1] <= stop)
    trials = trials[mine]
    windows = all_windows[mine]
    inbound = trials["trajectory_type"].to_numpy() == 1
    graph = fitted_maze(parallel_arm_candidates(positions), positions, cap=FIT_CAP_CM)
    declared, declared_line = declared_geometry(
        graph,
        positions,
        times=times,
        kind="W-track: three parallel arms joined by a base",
        source="Tang, Shin & Jadhav 2023 STAR Methods: 'the W-maze (~80 x 80 cm with"
        " ~7-cm wide tracks)'",
        bin_size=2.5,
        physical_width=TRACK_WIDTH_CM,
        units="cm",
        name=dataset["key"],
        placed_by="fitted to the samples",
        stated_lengths=dict(
            zip(
                ("base", "arms"),
                ((x, 80.0) for x in parallel_arm_lengths(graph)),
                strict=True,
            )
        ),
        placement=f"node positions fitted to this maze's samples ({arms_text(graph, 'cm')};"
        f" stated about 80 x 80 cm); edges (cm): {edge_lengths(graph)}",
        tracked_point="not stated",
        speed_thresholds=(4.0, 30.0),
    )
    return Recording(
        dataset=dataset["key"],
        session=f"JDS-NFN-AM2 {maze} W-track",
        source=f"{dataset['source']} {_PATH} (asset {_ASSET_ID})",
        license=dataset["license"],
        times=times,
        positions=positions,
        position_units="cm",
        spike_times=tuple(np.asarray(s, dtype=float) for s in units.spike_times),
        unit_ids=np.asarray(units.unit_ids),
        epochs={
            "run": runs,
            "maze": maze_runs,
            "trials": windows,
            "inbound": windows[inbound],
            "outbound": windows[~inbound],
        },
        events={
            "start_well": trials["start_well"].to_numpy(dtype=float),
            "end_well": trials["end_well"].to_numpy(dtype=float),
            "correct": trials["correct"].to_numpy(dtype=float),
        },
        geometry=declared,
        params={
            "bin_size": 2.5,
            "min_speed": 4.0,
            "dt": 0.25,
            "analysis_epochs": maze_runs,
            "train_epochs": windows[::2],
            "test_epochs": windows[1::2],
            "nwb_source": source,
            "readers_used": ("units",),
        },
        conversions=(
            f"positions: Position/SpatialSeries columns 0, 1 (cm); {span}; column 2"
            f" is speed in cm/s (median ratio {scale:.2f} to the speed from x, y while"
            f" moving, r = {agreement:.2f}), dropped",
            "epochs: trials table rows as trials, split by trajectory_type"
            " (1 inbound, 0 outbound); 'epoch intervals' as run",
            f"maze: the {maze} W-track, run epochs"
            f" {[i + 1 for i in MAZE_RUNS[maze]]}; positions outside them -> NaN"
            f" ({in_maze.mean():.1%} of samples kept); {int(mine.sum())} of"
            f" {len(all_windows)} trajectories inside them",
            declared_line,
        ),
    )


def _speed_read_as_a_dimension(rec: Recording) -> Recording:
    # A reader taking every column of the series as a position axis.
    assert rec.positions is not None
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.linalg.norm(np.diff(rec.positions, axis=0), axis=1) / np.diff(
            rec.times
        )
    return replace(
        rec, positions=np.column_stack([rec.positions, np.append(speed, np.nan)])
    )


HAZARDS = {"speed_read_as_a_dimension": _speed_read_as_a_dimension}


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
