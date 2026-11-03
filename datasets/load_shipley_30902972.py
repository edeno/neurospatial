"""Shipley, Barry et al. 2025 mouse CA1 on an 8-arm radial maze (UCL RDR 30902972).

What was recorded: mouse m4005 on 2020-09-26, 16 tetrodes in CA1 sorted
with Kilosort 2 and curated in Phy (71 clusters labelled ``good``), over 11 trials on an 8-arm radial maze (``0`` to ``10``) and a
45-minute sleep session (``S2``), recorded back to back. One tracking LED.

Where it lives: UCL Research Data Repository 30902972 (CC BY 4.0),
``Shipley2025_Data.zip`` (4.4 GB); four members of the session's
``PhyOutput/OldData`` folder are read by HTTP byte range:
``spike_times.npy``, ``spike_clusters.npy`` (Kilosort, read with
``np.load(..., allow_pickle=False)``: uint64 and int32, no objects),
``cluster_info.tsv`` (Phy labels) and ``trialData_newMUA.mat`` (MATLAB v5:
the session's positions, and per tetrode the curated spikes' times and
position samples).

What the loader converts:

- Spike times: ``spike_times.npy`` sample indices / 30 kHz, for the clusters
  ``cluster_info.tsv`` labels ``good`` (Phy's ``group``). They are exactly
  the 1,201,058 spike times ``trialData.tetrode(k).timestamp`` holds.
- Positions: ``trialData.pos.xy`` (pixels) / 2.74, from the header's
  ``pixels_per_metre 274``; range checked.
- Position times. ``trialData.pos`` is the trials' tracking concatenated,
  and its header says 30 Hz, but samples / 30 Hz runs 8 s past the last
  spike by the end. The file maps every curated spike to its position sample
  (``tetrode(k).pos_sample``, 1-based). Within each trial
  (``trialFinalInd``) the loader fits a line from position sample to spike
  time and times the samples with it: the camera ran at 30.05 Hz, trials
  start within 0.4 s of where 30 Hz puts them, and the largest residual is
  41 ms (about one frame).
- ``pos.dir`` is not a head direction: with one LED it equals
  ``dir_disp``, the direction of displacement.

Analysis choices (``params``): ``bin_size`` 3 cm; ``min_speed`` 3 cm/s
(mouse running); ``analysis_epochs`` the 11 maze trials; even and odd trials
train and test the decoder at ``dt`` 0.25 s.

Declared environment (``geometry["environment"]``): the radial maze (Shipley
et al. 2026, Methods: eight 80 x 10 cm arms on an octagonal centre platform
24.7 cm across) as a star of eight arms from one centre node, fitted to the
maze trials' samples and each arm then refitted to its own samples (an arm
leaves the platform's edge, so it need not point exactly at the centre),
with the stated 10 cm arm width; the platform is represented by the short
segments joining the arms to one centre node. The fit is in ``conversions``.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch_zip_member
from _record import (
    Range,
    Recording,
    check_expectations,
    checked_span,
    validate_recording,
)
from _tracks import (
    declared_geometry,
    fitted_maze,
    radial_arms,
    star_candidates,
    with_declared,
)

DATASET = {
    "key": "shipley_30902972",
    "title": "Shipley et al. (2025), mouse m4005, 2020-09-26, CA1 units on an 8-arm radial maze and sleep",
    "source": "https://doi.org/10.5522/04/30902972.v1",
    "license": "CC-BY-4.0",
    "download_size": "67 MB of a 4.4 GB zip (byte-range reads)",
}

# Pre-registered from the zip listing (11 maze trials and a sleep session
# S2) and trialData's header (30 Hz, 274 pixels per metre, one LED) and
# cluster_info.tsv (71 clusters labelled good) before the first run of this
# loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 71,
    "n_dims": 2,
    "sampling_rate_hz": Range(29.9, 30.1),
    "maze_extent_cm": lambda rec: Range(140.0, 200.0).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "eleven_maze_trials": lambda rec: len(rec.epochs["maze_trials"]) == 11,
    "one_led_no_heading": lambda rec: rec.headings is None,
}

_ZIP = "https://ndownloader.figshare.com/files/60449825"
_FOLDER = "Shipley2025_Data/m4005/m4005_2020-09-26/PhyOutput/OldData"
_FILES = {
    "spike_times.npy": "494f8affcdda4d91f8a9045626abdc0c9a533ee35359687d7d352a512c24c7a3",
    "spike_clusters.npy": "20231ecea49ac65dda7b9bd9929712dbd468ab65f74060cfdcc6ecabf4936cbc",
    "cluster_info.tsv": "4611e8a79f99ad9fed508a89d2abc49d066c5628432741ad80ebdf8131d8722b",
    "trialData_newMUA.mat": "83ad15210b1e975d9d4f3082fcfcca3537a45999a4dc77d82956f36b10e190bd",
}
_SAMPLING_HZ = 30000.0
_SPAN_CM = (140.0, 220.0)
_ARM_WIDTH_CM = 10.0
# Centre to arm end: an 80 cm arm from the edge of a 24.7 cm platform.
_ARM_REACH_CM = 80.0 + 24.7 / 2
_PLATFORM_CM = 20.0
_FIT_CAP_CM = 10.0
_MAX_RESIDUAL_S = 0.05


def _fetch(name: str, cache: Path | None) -> Path:
    return fetch_zip_member(
        _ZIP,
        f"{_FOLDER}/{name}",
        sha256=_FILES[name],
        subdir=DATASET["key"],
        cache=cache,
    )


def position_times(
    pos_sample: np.ndarray, spike_time: np.ndarray, trial_ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Time each position sample from the spikes the file maps to it.

    Parameters
    ----------
    pos_sample : ndarray, shape (n_spikes,)
        1-based position sample of each spike.
    spike_time : ndarray, shape (n_spikes,)
        The spikes' times (s).
    trial_ends : ndarray, shape (n_trials,)
        1-based index of each trial's last position sample.

    Returns
    -------
    times : ndarray, shape (trial_ends[-1],)
        Per-trial least-squares line from sample index to spike time.
    rates : ndarray, shape (n_trials,)
        Fitted sampling rate of each trial (Hz).
    worst : float
        Largest absolute residual of any spike (s).
    """
    index = pos_sample.astype(float) - 1
    times = np.empty(int(trial_ends[-1]))
    rates, worst = [], 0.0
    starts = np.r_[0, trial_ends[:-1]]
    for start, stop in zip(starts, trial_ends, strict=True):
        inside = (index >= start) & (index < stop)
        slope, intercept = np.polyfit(index[inside], spike_time[inside], 1)
        worst = max(
            worst,
            float(
                np.abs(spike_time[inside] - (intercept + slope * index[inside])).max()
            ),
        )
        times[start:stop] = intercept + slope * np.arange(start, stop)
        rates.append(1.0 / slope)
    return times, np.asarray(rates), worst


def _extent_centre(positions: np.ndarray) -> np.ndarray:
    """Middle of the samples' extent (0.5th to 99.5th percentile per axis).

    Eight arms of equal length put the maze's centre there; the densest
    point is a reward well, where the mouse waits longest.
    """
    low, high = np.nanpercentile(positions, [0.5, 99.5], axis=0)
    return 0.5 * (low + high)


def load(cache: Path | None = None) -> Recording:
    """Fetch m4005's 2020-09-26 session; positions in cm, good units' spikes."""
    from scipy.io import loadmat

    samples = np.load(_fetch("spike_times.npy", cache), allow_pickle=False).ravel()
    clusters = np.load(_fetch("spike_clusters.npy", cache), allow_pickle=False).ravel()
    with _fetch("cluster_info.tsv", cache).open() as table:
        rows = list(csv.DictReader(table, delimiter="\t"))
    good = np.array([int(r["id"]) for r in rows if r["group"].strip() == "good"])
    trial = loadmat(
        _fetch("trialData_newMUA.mat", cache), squeeze_me=True, struct_as_record=False
    )["trialData"]
    header = dict(np.atleast_2d(trial.pos.header))
    ppm = float(header["pixels_per_metre"])
    spike_seconds = samples.astype(float) / _SAMPLING_HZ
    order = np.argsort(spike_seconds, kind="stable")
    spike_seconds, clusters = spike_seconds[order], clusters[order]
    spikes = tuple(spike_seconds[clusters == c] for c in good)
    curated = np.sort(
        np.concatenate([np.atleast_1d(t.timestamp) for t in trial.tetrode])
    )
    if not np.array_equal(np.sort(np.concatenate(spikes)), curated):
        msg = "good clusters' spike times differ from trialData's tetrode timestamps"
        raise ValueError(msg)
    pos_sample = np.concatenate([np.atleast_1d(t.pos_sample) for t in trial.tetrode])
    stamp = np.concatenate([np.atleast_1d(t.timestamp) for t in trial.tetrode])
    ends = np.asarray(trial.pos.trialFinalInd, dtype=int)
    times, rates, worst = position_times(pos_sample, stamp, ends)
    if worst > _MAX_RESIDUAL_S or not np.all(np.diff(times) > 0):
        msg = f"position clock fit fails: worst residual {worst:.3f} s or times not increasing"
        raise ValueError(msg)
    positions = np.asarray(trial.pos.xy, dtype=float) / ppm * 100.0
    span = checked_span(positions, *_SPAN_CM)
    names = [str(n) for n in trial.pos.trialNames]
    starts = np.r_[0, ends[:-1]]
    windows = np.column_stack([times[starts], times[ends - 1]])
    maze = np.array([not n.startswith("S") for n in names])
    maze_trials = windows[maze]
    rec = Recording(
        dataset=DATASET["key"],
        session="m4005_2020-09-26",
        source=f"{DATASET['source']} {_FOLDER}",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        spike_times=spikes,
        unit_ids=good,
        epochs={"maze_trials": maze_trials, "sleep": windows[~maze]},
        params={
            "bin_size": 3.0,
            "min_speed": 3.0,
            "dt": 0.25,
            "analysis_epochs": maze_trials,
            "train_epochs": maze_trials[::2],
            "test_epochs": maze_trials[1::2],
        },
        conversions=(
            f"positions: pos.xy (pixels) / {ppm:g} pixels per metre x 100 -> cm",
            span,
            "times: per-trial line from pos_sample to the curated spikes' times;"
            f" fitted rates {rates.min():.3f}-{rates.max():.3f} Hz (header 30 Hz),"
            f" worst residual {worst * 1e3:.0f} ms",
            f"spike_times: spike_times.npy / {_SAMPLING_HZ:g} Hz for the {len(good)}"
            " clusters labelled good; identical to trialData tetrode timestamps",
            f"epochs: trials {names} from trialFinalInd; S* is sleep",
            "headings: none (pos.dir equals dir_disp, the displacement direction)",
        ),
    )
    in_maze = np.zeros(len(times), dtype=bool)
    for start, stop in maze_trials:
        in_maze |= (times >= start) & (times <= stop)
    maze_xy = np.where(in_maze[:, None], positions, np.nan)
    star = fitted_maze(
        star_candidates(maze_xy, 8, _extent_centre(maze_xy)), maze_xy, cap=_FIT_CAP_CM
    )
    graph = radial_arms(star, maze_xy, min_radius=_PLATFORM_CM, window=_FIT_CAP_CM)
    centre = np.asarray(graph.nodes["centre"]["pos"])
    reach = {
        f"arm {k}": float(
            np.linalg.norm(np.asarray(graph.nodes[f"arm_{k}"]["pos"]) - centre)
        )
        for k in range(8)
    }
    entries, line = declared_geometry(
        graph,
        maze_xy,
        times=times,
        kind="8-arm radial maze: eight arms joined at a centre",
        source="Shipley et al. 2026 Methods: '80cm x 10cm arms connected by an"
        " octagonal centre platform (24.7cm diameter, 10cm sides)'",
        bin_size=3.0,
        physical_width=_ARM_WIDTH_CM,
        units="cm",
        name=DATASET["key"],
        placed_by="fitted to the samples",
        stated_lengths={k: (v, _ARM_REACH_CM) for k, v in reach.items()},
        placement="a star fitted to the maze trials' samples, then each arm the"
        " principal axis of its own samples beyond the platform, joined to the"
        f" centre; centre to arm end {min(reach.values()):.1f}-"
        f"{max(reach.values()):.1f} cm (stated 80 cm arms from a 24.7 cm platform:"
        f" {_ARM_REACH_CM:g} cm)",
        tracked_point="one tracking LED (per the loader; trialData.pos)",
        speed_thresholds=(3.0, 30.0),
    )
    return with_declared(rec, entries, line)


def clock_from_header_rate(rec: Recording) -> Recording:
    # A reader trusting the header's 30 Hz: sample index / 30, which runs
    # 8 s past the spikes by the end of the session.
    assert rec.times is not None
    return replace(rec, times=rec.times[0] + np.arange(len(rec.times)) / 30.0)


def pixels_labelled_cm(rec: Recording) -> Recording:
    # The stored xy are pixels; a reader that missed pixels_per_metre would
    # call them cm (2.74 x too large).
    assert rec.positions is not None
    return replace(rec, positions=rec.positions * 2.74)


HAZARDS = {
    "clock_from_header_rate": clock_from_header_rate,
    "pixels_labelled_cm": pixels_labelled_cm,
}


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s")
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
