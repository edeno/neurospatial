"""Shin, Tang & Jadhav (2019): rat CA1 and prefrontal cortex learning a W-maze in a day (DANDI 000978).

What was recorded: tetrodes in dorsal CA1 and prefrontal cortex (mostly
prelimbic) of rats that learned a W-maze alternation task in one day: "eight
15-20 min sessions on a W-maze interleaved with 20-30 min rest sessions in a
sleep box". The W-maze "had dimensions of 80 x 80 cm with 7-cm-wide track
sections", with "Three reward food wells (i.e., right, center, and left wells)
... located at the end of three arms"; a reward came for returning to the
centre well after either side well, and for choosing the side well not visited
last after leaving the centre. "The animal's position and running speed were
recorded with an overhead color CCD camera (30 fps) and tracked by color LEDs
affixed to the headstage"; LFP was "sampled at 1.5 kHz and bandpass filtered
between 0.5 Hz and 400 Hz" (the STAR Methods of Shin et al. 2019).

Sessions are DANDI 000978's nine files, version 0.240511.0307, CC-BY-4.0: one
per rat for ER1, JS14, JS15, JS17, JS21, JS34 and KL8, and two for ZT2, whose day
is split across them (9 and 8 epochs). The dandiset names no paper; the link to
Shin et al. 2019 is inferred from its six rats, ER1, KL8, JS14, JS15, JS17 and
JS21 (Table S1 and Figure S1), all in the dandiset, the task, and the
correct-trial count: the six files hold 1,537 correct trials, the paper's per
visit figures 1,533. JS34 and ZT2 are not in the paper. The default session,
``sub-JDS-SingleDay-JS14``, is the smallest file and the paper's example rat
in Figure S1Fii.

Each file is 28 to 47 GB, and about half of it is ``acquisition/ElectricalSeries``,
which holds the same samples as the LFP. The files are never downloaded: a
load reads units, position, epochs and trials by byte range (47 MB in 20 s for JS14), and
``rec.lfp.fetch(channels, stream=True)`` reads the LFP the same way.

Sources read: DANDI 000978's metadata, the files, and Shin et al. 2019 with its
Supplemental Information (Neuron 104:1110).

What the loader converts:

- positions: columns 0 and 1 (x, y) of ``processing/behavior/Position/SpatialSeries``,
  in cm, the frame origin at the centre well; its column 2, the authors'
  velocity in cm/s, is not loaded (see ``params["speed_smooth_window"]``).
- epochs: ``run``, the rows of the file's ``epoch intervals`` that hold trials
  (W-maze sessions), ``rest`` the others (the sleep box), and ``trials``.
- trials: the file's table: ``start_well``, ``end_well`` (1 is the centre
  well), ``trajectory_type`` (0 outbound, 1 inbound) and ``correct``. A trial
  runs from leaving one well to leaving the next, so its end holds the stay
  at its end well.
- ``units``: ``brain_region`` CA1 or PFC from the electrodes table's
  ``location``, by each unit's tetrode; ``tetrode`` is that number. The units
  table's ``electrodes`` column holds tetrode numbers, not rows of the
  electrodes table, which pynwb's own reading assumes. ``label`` is missing:
  the file marks no interneurons.
- environment: the W-maze as a track graph (three arms joined by a base), its
  six nodes fitted to the tracked samples of the run epochs, in 2 cm bins; the
  three wells, the arms' ends, as the point regions ``well_1`` (centre),
  ``well_2`` and ``well_3``.
- headings: none.
- ``rec.lfp``: the file's LFP, 1,500 Hz, in µV, two channels per tetrode:
  ``Ref ON``, referenced to the local reference tetrode as the paper's
  ripple-band LFPs were, and ``Ref OFF``, not referenced.

Analysis choices (``params``):

- ``bin_size`` 2 cm, the paper's linearized bins;
- ``min_speed`` 5 cm/s, the paper's movement threshold; ``immobility_speed``
  4 cm/s, its immobility threshold ("speed <= 4 cm/s");
- ``speed_smooth_window`` 0.5 s: ``compute_speed(times, positions,
  smooth_window=0.5)`` gives the speed of the file's velocity column (each
  session's conversions give the agreement);
- ``analysis_epochs``: the run epochs;
- ``train_epochs`` and ``test_epochs``: the trials alternating within each
  ``(start_well, end_well)`` path;
- ``dt`` 0.5 s and ``replay_dt`` 0.01 s, the paper's decoding bins during
  running and during SWRs.
"""

from __future__ import annotations

import http.client
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._lfp import LFPFile, LFPSource
from neurospatial.datasets._recording import (
    Conversion,
    Recording,
    tracked_mask,
)
from neurospatial.datasets._tracks import nearest_segment, placement_tolerance

if TYPE_CHECKING:
    import networkx as nx

    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = [
    "EXPECTED_WATCH_OUTS",
    "PAPER_PER_VISIT",
    "PAPER_RATS",
    "PER_VISIT",
    "REQUIRES",
    "WATCH_OUTS",
    "WORKED_RESULT",
    "load_session",
]

#: The paper's per-visit counts of SWRs, candidate events and replay events,
#: attempted on its six rats; five of the six conditions fixed before the run
#: hold. The numbers are ``PER_VISIT``, recomputed by
#: ``tests/datasets/test_shin_2019.py``.
WORKED_RESULT: dict[str, Any] = {
    "status": "not possible yet",
    "level": None,
    "finding": None,
    "quote": None,
    "source": None,
    "paper": None,
    "ours": None,
    "criterion": None,
    "parameters": None,
    "unmatched": (),
    "attempted": {
        "value": 1.024,
        "criterion": (
            "fixed after the well-visit rule was written and JS14's visits counted, "
            "before any SWR was detected: over the six rats' correct trials, each "
            "per-visit mean within half the paper's SD of its mean (immobility "
            "10.3 +/- 2.85 s, SWRs 7.3 +/- 2.75, candidates 2.3 +/- 1.2, replay "
            "0.9 +/- 0.6) and each share of visits with two or more events within "
            "10 percentage points (SWRs 85.7%, candidates 53.4%). Five hold: "
            "immobility 9.46 s, SWRs 6.68, candidates 2.98 and replay 1.02 per "
            "visit (the value above), and 62.2% of visits with two or more "
            "candidates; 75.5% with two or more SWRs is 10.2 points short of "
            "85.7%, so the criterion is not met"
        ),
        "parameters": (
            "1,488 visits of the 1,537 correct trials (the others: the rat never "
            "within 15 cm of the start well before the trial); a visit is the stay "
            "at a correct trial's start well, arrival and departure the first and "
            "last sample of the last stay within 15 cm of the well, entry the "
            "first sample from arrival at or below 4 cm/s, exit the start of the "
            "last stretch above 4 cm/s reaching departure, speed by "
            "compute_speed(smooth_window=0.5); SWRs by ripple_detection's "
            "shin_2019 at 0eddae64 per run session on the Ref ON channel of every "
            "CA1 tetrode holding a unit, counted by start; candidates by its "
            "decoding_candidates stage, place cells the CA1 units with 100 spikes "
            "in every run session and a 3 Hz peak in that session's maps; "
            "templates per session and trajectory from its trials above 5 cm/s, "
            "2 cm bins, a 4 cm SD Gaussian, 20 ms occupancy, cut 15 cm from each "
            "well; SHIN_2019 unchanged on decode_jointly over the trajectories run "
            "in that session, 10 ms bins, seeds [2019, session, event]"
        ),
        "differs": (
            "the paper's well entry and exit come from an infrared beam not in the "
            "file, replaced by a 15 cm radius around the well",
            "the paper excluded putative interneurons by spike width and rate; the "
            "file holds no waveforms, so the place cells may include interneurons",
            "the ripple tetrodes are not named; we use every CA1 tetrode holding a "
            "unit",
            "the paper counts per rat over 1,533 trials; ours are 1,488 visits, "
            "and the six rats differ widely (SWRs per visit 3.6 to 10.2, replay "
            "0.28 to 1.53)",
            "the detector's 15 ms minimum above 3 SD is Karlsson & Frank's; the "
            "paper states none. Without it, SWRs per visit are 19.2 pooled (7.1 "
            "on JS14, where the minimum explains the gap, and 16 to 29 on the "
            "other rats), so the minimum is not the one cause of the short share",
        ),
    },
    "why": (
        "the criterion fixed before the run is not met: 75.5% of visits hold two "
        "or more SWRs against the paper's 85.7%, 10.2 points apart against a "
        "mark of 10, while the four per-visit means and the share with two or "
        "more candidates are within their marks. The beam times of well entry and "
        "exit, the interneuron labels and the ripple tetrodes the paper used are "
        "not in the files"
    ),
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("pynwb", "fsspec")

#: The paper's six rats (Table S1, Figure S1), as the dandiset names them.
PAPER_RATS = ("ER1", "KL8", "JS14", "JS15", "JS17", "JS21")

#: The paper's per-visit numbers over its six rats' correct trials, (mean, SD):
#: "immobility time: 10.3 ± 5.7 s; SWRs: 7.3 ± 5.5 events; candidate events:
#: 2.3 ± 2.4 events; replay events: 0.9 ± 1.2 events; data are presented in mean
#: ± SD" (Figure 2G-I), and "immobility periods with ≥2 events, 85.7% or
#: 1,313/1,533 trials for SWRs; 53.4% or 818/1,533 trials for replay candidate
#: events" (Results).
PAPER_PER_VISIT: dict[str, Any] = {
    "n_visits": 1533,
    "immobility_s": (10.3, 5.7),
    "swrs": (7.3, 5.5),
    "candidates": (2.3, 2.4),
    "replay": (0.9, 1.2),
    "share_two_swrs": 0.857,
    "share_two_candidates": 0.534,
}

#: Our per-visit numbers, recomputed by ``tests/datasets/test_shin_2019.py``:
#: (mean, SD) per visit and the shares of visits with two or more events; per
#: rat, the place cells of each run epoch, the session's SWRs and candidates,
#: the candidates judged (those in visits), the verdicts the Jadhav lab's
#: line-fit p-value changes each way, and the units' rate in SWRs over their
#: rate in the run epochs (median by area).
PER_VISIT: dict[str, Any] = {
    "ER1": {
        "summary": {
            "n_visits": 301,
            "immobility_s": (7.655, 3.761),
            "swrs": (8.392, 5.58),
            "candidates": (3.927, 2.877),
            "replay": (1.296, 1.22),
            "replay_lab_code": (1.249, 1.186),
            "swrs_no_minimum": (22.286, 11.759),
            "share_two_swrs": 0.85,
            "share_two_candidates": 0.774,
        },
        "criterion": {
            "immobility_s": True,
            "swrs": True,
            "candidates": False,
            "replay": True,
            "share_two_swrs": True,
            "share_two_candidates": False,
        },
        "n_place_cells": [44, 43, 42, 43, 41, 43, 40, 40],
        "n_swrs": 3782,
        "n_candidates": 1750,
        "n_judged": 1182,
        "ours_only": 55,
        "lab_code_only": 41,
        "swr_modulation": {"CA1": 1.4018, "PFC": 0.8444},
    },
    "KL8": {
        "summary": {
            "n_visits": 189,
            "immobility_s": (6.872, 4.44),
            "swrs": (3.577, 3.671),
            "candidates": (1.503, 1.69),
            "replay": (0.28, 0.611),
            "replay_lab_code": (0.28, 0.575),
            "swrs_no_minimum": (16.196, 17.22),
            "share_two_swrs": 0.608,
            "share_two_candidates": 0.386,
        },
        "criterion": {
            "immobility_s": False,
            "swrs": False,
            "candidates": True,
            "replay": False,
            "share_two_swrs": False,
            "share_two_candidates": False,
        },
        "n_place_cells": [44, 47, 46, 47, 46, 49, 46, 49],
        "n_swrs": 2322,
        "n_candidates": 1281,
        "n_judged": 284,
        "ours_only": 13,
        "lab_code_only": 13,
        "swr_modulation": {"CA1": 1.7744, "PFC": 0.8135},
    },
    "JS14": {
        "summary": {
            "n_visits": 250,
            "immobility_s": (10.155, 7.504),
            "swrs": (3.82, 3.726),
            "candidates": (2.94, 2.998),
            "replay": (0.72, 1.05),
            "replay_lab_code": (0.648, 1.085),
            "swrs_no_minimum": (7.144, 6.529),
            "share_two_swrs": 0.716,
            "share_two_candidates": 0.628,
        },
        "criterion": {
            "immobility_s": True,
            "swrs": False,
            "candidates": True,
            "replay": True,
            "share_two_swrs": False,
            "share_two_candidates": True,
        },
        "n_place_cells": [38, 39, 35, 37, 39, 38, 36, 38],
        "n_swrs": 1466,
        "n_candidates": 1118,
        "n_judged": 735,
        "ours_only": 36,
        "lab_code_only": 18,
        "swr_modulation": {"CA1": 2.1894, "PFC": 0.7483},
    },
    "JS15": {
        "summary": {
            "n_visits": 217,
            "immobility_s": (10.849, 7.9),
            "swrs": (5.811, 5.351),
            "candidates": (3.461, 3.556),
            "replay": (1.203, 1.492),
            "replay_lab_code": (1.24, 1.557),
            "swrs_no_minimum": (17.728, 14.565),
            "share_two_swrs": 0.705,
            "share_two_candidates": 0.631,
        },
        "criterion": {
            "immobility_s": True,
            "swrs": True,
            "candidates": True,
            "replay": True,
            "share_two_swrs": False,
            "share_two_candidates": True,
        },
        "n_place_cells": [40, 39, 37, 41, 41, 41, 41, 41],
        "n_swrs": 2602,
        "n_candidates": 1425,
        "n_judged": 751,
        "ours_only": 29,
        "lab_code_only": 37,
        "swr_modulation": {"CA1": 1.9177, "PFC": 0.7399},
    },
    "JS17": {
        "summary": {
            "n_visits": 288,
            "immobility_s": (9.387, 5.339),
            "swrs": (10.24, 7.146),
            "candidates": (1.965, 1.729),
            "replay": (0.931, 1.093),
            "replay_lab_code": (0.955, 1.079),
            "swrs_no_minimum": (29.236, 18.064),
            "share_two_swrs": 0.792,
            "share_two_candidates": 0.542,
        },
        "criterion": {
            "immobility_s": True,
            "swrs": False,
            "candidates": True,
            "replay": True,
            "share_two_swrs": True,
            "share_two_candidates": True,
        },
        "n_place_cells": [26, 24, 21, 24, 21, 23, 23, 23],
        "n_swrs": 4816,
        "n_candidates": 912,
        "n_judged": 566,
        "ours_only": 15,
        "lab_code_only": 22,
        "swr_modulation": {"CA1": 1.9473, "PFC": 0.8163},
    },
    "JS21": {
        "summary": {
            "n_visits": 243,
            "immobility_s": (11.835, 7.76),
            "swrs": (6.494, 5.68),
            "candidates": (3.753, 3.672),
            "replay": (1.531, 1.551),
            "replay_lab_code": (1.502, 1.596),
            "swrs_no_minimum": (19.733, 15.21),
            "share_two_swrs": 0.79,
            "share_two_candidates": 0.695,
        },
        "criterion": {
            "immobility_s": True,
            "swrs": True,
            "candidates": False,
            "replay": False,
            "share_two_swrs": True,
            "share_two_candidates": False,
        },
        "n_place_cells": [27, 27, 28, 28, 27, 28, 28, 28],
        "n_swrs": 2757,
        "n_candidates": 1739,
        "n_judged": 912,
        "ours_only": 44,
        "lab_code_only": 37,
        "swr_modulation": {"CA1": 2.5566, "PFC": 0.9772},
    },
    "pooled": {
        "summary": {
            "n_visits": 1488,
            "immobility_s": (9.459, 6.476),
            "swrs": (6.683, 5.945),
            "candidates": (2.977, 2.989),
            "replay": (1.024, 1.275),
            "replay_lab_code": (1.008, 1.29),
            "swrs_no_minimum": (19.232, 15.854),
            "share_two_swrs": 0.755,
            "share_two_candidates": 0.622,
        },
        "criterion": {
            "immobility_s": True,
            "swrs": True,
            "candidates": True,
            "replay": True,
            "share_two_swrs": False,
            "share_two_candidates": True,
        },
    },
}

BIN_SIZE = 2.0
MIN_SPEED = 5.0
IMMOBILITY_SPEED = 4.0
SPEED_SMOOTH_WINDOW = 0.5
DT = 0.5
REPLAY_DT = 0.01
#: The stated W-maze, cm: outer size and track width (STAR Methods).
MAZE_SIDE = 80.0
TRACK_WIDTH = 7.0
#: The LFP's stated rate, Hz.
LFP_RATE = 1500.0
#: The centre well's number in the trials table.
CENTRE_WELL = 1
#: How far a fitted node may be pulled by one sample, cm; samples per fit;
#: points along each edge; and the seed of the sample subset.
_FIT_CAP = 10.0
_FIT_SAMPLES = 6000
_EDGE_POINTS = 25
_FIT_SEED = 0
#: The share of samples the placement tolerance holds.
_TOLERANCE_QUANTILE = 0.99
#: Rows read per byte-range request of the LFP, and requests at once.
_LFP_BLOCK_BYTES = 32 << 20
_LFP_WORKERS = 4
#: Tries of each byte range: a connection reset is retried after 1, 2, 4 ... s.
_LFP_ATTEMPTS = 6

_POSITION = "processing/behavior/Position/SpatialSeries"
_LFP = ("ecephys", "LFP", "ElectricalSeries")
_PAPER = "the STAR Methods of Shin et al. 2019"
_EDGES = (
    ("well_2", "base_2"),
    ("base_2", "base_1"),
    ("base_1", "well_1"),
    ("base_1", "base_3"),
    ("base_3", "well_3"),
)

#: Findings computed from the data (by the checks or this loader) that each
#: verified session must report, as patterns searched for in its watch_outs.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "sub-JDS-SingleDay-JS14": (
        r"for 35 of the 72 units here the wrong brain area",
        r"75% of the tracked samples of the rest epochs fall in a bin",
        r"the sampling rate is 32\.57\d* Hz",
        r"runs of 11 or more identical position samples",
        r"lie outside the declared environment",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "sub-JDS-SingleDay-JS14": (
        "The CA1 tetrodes are referenced to tetrode 14 and the PFC tetrodes to "
        "tetrode 1: on every CA1 tetrode, Ref OFF minus Ref ON is tetrode 14's "
        "signal (correlation 1.0), and on every PFC tetrode tetrode 1's. Tetrodes "
        "14 and 1 are labelled CA1 and PFC but are the references, and no unit is "
        "on them.",
    ),
}

_LINK = (
    "The dandiset names no paper. Its link to Shin, Tang & Jadhav 2019 is "
    "inferred: the paper's six rats (ER1, KL8, JS14, JS15, JS17, JS21; Table S1) "
    "are subjects here, the task and day match, and those six files hold 1,537 "
    "correct trials against the paper's 1,533. JS34 and ZT2 are not in the "
    "paper."
)
_RAW = (
    "The file is mostly voltage: acquisition/ElectricalSeries holds the same "
    "samples, on the same clock, as processing/ecephys/LFP (equal wherever "
    "compared), and the two are each about half the file. Never download it "
    "for the spikes: load() reads them by byte range. rec.lfp.fetch(channels, stream=True) "
    "reads the LFP's rows, about 14 to 24 GB whatever the channels, since the "
    "file stores each sample's channels together; ask for all the channels you "
    "need at once."
)
_NO_LABELS = (
    "The file marks no interneurons and no place cells: units hold every sorted "
    "unit with its tetrode. The paper excluded putative interneurons by spike "
    "width and firing rate (not in the file) and kept cells recorded stably "
    "across all 8 sessions with at least 100 spikes in each, 216 CA1 place "
    "cells and 154 PFC cells over its six rats (Figure S1)."
)
_NO_RIPPLES = (
    "The file has no ripple table and states no ripple channel: detect SWRs "
    "yourself on the CA1 tetrodes' Ref ON channels (the paper's 'at least one "
    "tetrode' rule is ripple_detection.literature_methods.shin_2019)."
)


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one DANDI 000978 file into a :class:`Recording`.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The cached NWB file, or its URL to read by byte range.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.

    Raises
    ------
    ValueError
        If the position series' comments no longer name x, y and velocity, a
        unit's tetrode is not in the electrodes table, or a tetrode's two
        electrodes give different locations.
    """
    from neurospatial import Intervals
    from neurospatial.datasets._fetch import open_nwb
    from neurospatial.io.nwb import read_intervals, read_position, read_units

    source = _direct(os.fspath(path_or_stream))
    with open_nwb(source) as nwb:
        series = nwb.processing["behavior"]["Position"]["SpatialSeries"]
        comments, stated_unit = str(series.comments), str(series.unit)
        position = read_position(nwb, position_name=_POSITION, columns=[0, 1])
        velocity = np.asarray(series.data[:, 2], dtype=np.float64)
        units = read_units(nwb)
        unit_tetrodes = np.asarray(nwb.units.electrodes.data[:], dtype=int)
        electrodes = nwb.electrodes.to_dataframe()[["group_name", "label", "location"]]
        epoch_rows = (
            read_intervals(nwb, interval_name="epoch intervals")
            .intervals_table()[["start_time", "stop_time"]]
            .to_numpy(dtype=np.float64)
        )
        table = nwb.trials.to_dataframe().reset_index()
        lfp = nwb.processing[_LFP[0]][_LFP[1]][_LFP[2]]
        lfp_rows = [int(row) for row in lfp.electrodes.data[:]]
        lfp_span = (float(lfp.timestamps[0]), float(lfp.timestamps[-1]))
        lfp_unit = str(lfp.unit)
        subject = str(nwb.subject.subject_id)
        started = nwb.session_start_time
    if unit_tetrodes.shape != (len(units.spike_times),):
        msg = (
            f"{entry.session}: the units table's electrodes column holds "
            f"{unit_tetrodes.shape} values for {len(units.spike_times)} units, not "
            "one tetrode per unit"
        )
        raise ValueError(msg)
    if "row1:xpos row2:ypos row3:velocity" not in comments:
        msg = (
            f"{entry.session}: the position series' comments are {comments!r}, "
            "not the x, y and velocity columns the loader reads"
        )
        raise ValueError(msg)
    times = np.asarray(position.times, dtype=np.float64)
    positions = np.asarray(position.positions, dtype=np.float64)
    order = np.argsort(epoch_rows[:, 0], kind="stable")
    epoch_rows = epoch_rows[order]
    table = table.sort_values("start_time", kind="stable")
    trial_rows = table[["start_time", "stop_time"]].to_numpy(dtype=np.float64)
    holds = np.array(
        [
            ((trial_rows[:, 0] >= a) & (trial_rows[:, 0] < b)).any()
            for a, b in epoch_rows
        ]
    )
    run, rest = epoch_rows[holds], epoch_rows[~holds]
    in_run = _within(times, run)
    env, placement, wells = _declared_w_maze(
        positions[in_run & tracked_mask(positions)], entry.session
    )
    unit_table = _unit_table(units, unit_tetrodes, electrodes, entry.session)
    paths = list(zip(table["start_well"], table["end_well"], strict=True))
    train = _alternate_within(paths)
    animal = subject.removeprefix("JDS-SingleDay-")
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"rat {animal}",
        apparatus=entry.info.apparatus,
        brain_regions=entry.info.brain_regions,
        clock=(
            "the file's clock, seconds from its session start time "
            f"({started:%Y-%m-%d}, when the file was written, not the recording's "
            "date); spikes, position, LFP, epochs and trials share it"
        ),
        times=times,
        spike_times=tuple(units.spike_times),
        units=unit_table,
        env=env,
        env_kind="declared",
        env_source=(
            f"the W-maze ({MAZE_SIDE:g} x {MAZE_SIDE:g} cm, {TRACK_WIDTH:g} cm wide "
            f"track sections, {_PAPER}) as a track graph, its nodes fitted to the "
            f"samples of the run epochs, {BIN_SIZE:g} cm bins; the wells as regions "
            "well_1 (centre), well_2 and well_3"
        ),
        track_width_quantile=placement.quantile,
        positions=positions,
        position_units="cm",
        tracking=(
            "color LEDs on the headstage, filmed from above at a stated 30 frames/s "
            f"({_PAPER})"
        ),
        epochs={"run": run, "rest": rest, "trials": trial_rows},
        trials=Intervals.from_dataframe(table, id_column="id"),
        params={
            "bin_size": BIN_SIZE,
            "min_speed": MIN_SPEED,
            "immobility_speed": IMMOBILITY_SPEED,
            "speed_smooth_window": SPEED_SMOOTH_WINDOW,
            "dt": DT,
            "replay_dt": REPLAY_DT,
            "analysis_epochs": run,
            "train_epochs": trial_rows[train],
            "test_epochs": trial_rows[~train],
        },
        conversions=(
            Conversion(
                f"positions: columns 0 and 1 of {_POSITION}, cm, as stored",
                "stated",
                f"the series' comments {comments!r} and unit {stated_unit!r}; its "
                "reference frame 'center well'. The camera frame's handedness is not "
                "stated, so which of wells 2 and 3 is the paper's left is not known",
            ),
            _speed_conversion(times, positions, velocity, in_run),
            Conversion(
                "epochs: 'run', the rows of the file's 'epoch intervals' holding "
                "trials; 'rest', the other rows; 'trials', the trials' rows",
                "inferred",
                "the table has start and stop times only; the paper alternates 8 "
                f"W-maze sessions with 9 rest sessions in a sleep box ({_PAPER}). "
                f"Here {len(run)} rows hold the {len(trial_rows)} trials and "
                f"{len(rest)} hold none",
            ),
            _trial_conversion(times, positions, velocity, table, wells),
            _region_conversion(unit_tetrodes, electrodes),
            placement.conversion,
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            _LINK,
            *_not_in_paper(animal, entry.session),
            _RAW,
            _electrode_watch_out(unit_tetrodes, electrodes),
            _reference_watch_out(electrodes),
            _sleep_box_watch_out(times, positions, rest, env),
            _NO_LABELS,
            _NO_RIPPLES,
        ),
        lfp=_lfp_source(
            entry, path_or_stream, electrodes, lfp_rows, lfp_span, lfp_unit
        ),
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Raise a redirect as an ``HTTPError`` instead of following it."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        """Follow no redirect: ``urllib`` then raises it, with its Location."""
        return


def _direct(source: str) -> str:
    """The URL a DANDI asset redirects to, so each byte range skips the redirect.

    A DANDI download URL answers every request with a redirect to the file on
    S3; following it once makes each later byte-range read one request instead
    of two. A local path is returned unchanged.
    """
    if not source.startswith(("https://", "http://")):
        return source

    request = urllib.request.Request(source, method="HEAD")
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=60):
            return source
    except urllib.error.HTTPError as redirect:
        location = redirect.headers.get("Location")
        if redirect.code in (301, 302, 303, 307, 308) and location:
            return str(location)
        raise


def _within(times: NDArray[np.float64], rows: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Whether each sorted time lies in a [start, stop) row, shape (n_time,)."""
    starts = np.searchsorted(times, rows[:, 0], side="left")
    stops = np.searchsorted(times, rows[:, 1], side="left")
    counts = np.zeros(times.size + 1, dtype=np.int64)
    np.add.at(counts, starts, 1)
    np.add.at(counts, stops, -1)
    inside: NDArray[np.bool_] = np.cumsum(counts[:-1]) > 0
    return inside


def _alternate_within(paths: list[tuple[Any, Any]]) -> NDArray[np.bool_]:
    """Trials alternating between train and test within each path, in order.

    Parameters
    ----------
    paths : list of (start_well, end_well)
        Each trial's path, in time order.

    Returns
    -------
    ndarray of bool, shape (n_trials,)
        True for the 1st, 3rd, ... trial of each path.
    """
    train = np.zeros(len(paths), dtype=bool)
    seen: dict[tuple[Any, Any], int] = {}
    for k, path in enumerate(paths):
        train[k] = seen.get(path, 0) % 2 == 0
        seen[path] = seen.get(path, 0) + 1
    return train


def _tetrode_numbers(electrodes: Any) -> NDArray[np.int_]:
    """Each electrode row's tetrode number, from ``group_name`` (``tetrode6``)."""
    return np.array(
        [int(str(name).removeprefix("tetrode")) for name in electrodes["group_name"]]
    )


def _tetrode_locations(electrodes: Any, session: str) -> dict[int, str]:
    """Each tetrode's location; its two electrodes must agree."""
    found: dict[int, set[str]] = {}
    for number, location in zip(
        _tetrode_numbers(electrodes), electrodes["location"], strict=True
    ):
        found.setdefault(int(number), set()).add(str(location))
    if split := {k: v for k, v in found.items() if len(v) > 1}:
        msg = f"{session}: tetrodes with two locations: {split}"
        raise ValueError(msg)
    return {k: next(iter(v)) for k, v in found.items()}


def _unit_table(
    units: Any, tetrodes: NDArray[np.int_], electrodes: Any, session: str
) -> Any:
    """``units``: brain region by each unit's tetrode number, which ``tetrode`` keeps."""
    from neurospatial.datasets._recording import nwb_unit_table

    locations = _tetrode_locations(electrodes, session)
    if missing := sorted(set(tetrodes.tolist()) - set(locations)):
        msg = (
            f"{session}: units name tetrodes {missing}, which the electrodes "
            f"table does not hold (it holds {sorted(locations)})"
        )
        raise ValueError(msg)
    table, _ = nwb_unit_table(units)
    table["brain_region"] = [locations[int(k)] for k in tetrodes]
    table["tetrode"] = tetrodes
    return table


def _region_conversion(tetrodes: NDArray[np.int_], electrodes: Any) -> Conversion:
    """Why a unit's ``electrodes`` value is a tetrode number, with the counts."""
    locations = _tetrode_locations(electrodes, "")
    by_tetrode = np.array([locations[int(k)] for k in tetrodes])
    rows = electrodes["location"].to_numpy(dtype=str)
    in_range = tetrodes < len(rows)
    by_row = np.where(in_range, rows[np.clip(tetrodes, 0, len(rows) - 1)], "none")
    counts = ", ".join(
        f"{area} {int(np.sum(by_tetrode == area))}" for area in sorted(set(by_tetrode))
    )
    as_rows = ", ".join(
        f"{area} {int(np.sum(by_row == area))}" for area in sorted(set(by_row))
    )
    return Conversion(
        f"units: brain_region from the electrodes table's location of each unit's "
        f"tetrode, the units table's 'electrodes' value read as a tetrode number "
        f"({counts})",
        "inferred",
        "the file does not say what the values are. Read as tetrode numbers "
        "(group_name 'tetrodeN'), every value names a tetrode the table holds and "
        "each of the paper's six rats has at least as many CA1 and PFC units as "
        "its continuously tracked CA1 place cells and PFC cells (Figure S1F-G: "
        "ER1 44 and 20, KL8 47 and 25, JS14 37 and 12, JS15 40 and 16, JS17 24 and "
        "43, JS21 24 and 38). Read as rows of the table, as pynwb does, JS14 would "
        "have 31 CA1 units, KL8 22 PFC units and JS17 42 PFC units, fewer than the "
        f"paper's cells. Read as rows here: {as_rows}",
    )


def _electrode_watch_out(tetrodes: NDArray[np.int_], electrodes: Any) -> str:
    """How many units pynwb's reading of ``electrodes`` puts in the wrong area."""
    locations = _tetrode_locations(electrodes, "")
    rows = electrodes["location"].to_numpy(dtype=str)
    by_tetrode = np.array([locations[int(k)] for k in tetrodes])
    by_row = np.where(
        tetrodes < len(rows), rows[np.clip(tetrodes, 0, len(rows) - 1)], "none"
    )
    wrong = int(np.sum(by_row != by_tetrode))
    return (
        "The units table's 'electrodes' column holds each unit's tetrode number, "
        "not a row of the electrodes table. pynwb resolves it as rows, so "
        "nwbfile.units.to_dataframe() and pynapple give the wrong electrode, and "
        f"for {wrong} of the {len(tetrodes)} units here the wrong brain area. "
        "rec.units['brain_region'] and rec.units['tetrode'] read it as tetrodes."
    )


def _reference_watch_out(electrodes: Any) -> str:
    """Which tetrodes the others are referenced to, as their labels say."""
    numbers = _tetrode_numbers(electrodes)
    labels = electrodes["label"].to_numpy(dtype=str)
    locations = electrodes["location"].to_numpy(dtype=str)
    return (
        f"Each tetrode has two LFP channels: '{labels[1]}' (rec.lfp channel "
        f"'tetrodeN {labels[1]}'), referenced to the region's reference tetrode, "
        f"and '{labels[0]}', not referenced. The paper's ripple-band LFPs were "
        "'recorded relative to the local reference tetrode', a tetrode in corpus "
        "callosum for CA1 and one in overlying cortex for PFC, but the file does "
        f"not say which tetrodes those are; they are labelled with their region "
        f"({len(set(numbers.tolist()))} tetrodes: "
        f"{int(np.sum(locations[labels == labels[0]] == 'CA1'))} CA1, "
        f"{int(np.sum(locations[labels == labels[0]] == 'PFC'))} PFC). A reference "
        "tetrode's two channels are equal, so a referenced channel that equals "
        "its unreferenced one is a reference, not a CA1 or PFC signal."
    )


def _sleep_box_watch_out(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    rest: NDArray[np.float64],
    env: Environment,
) -> str:
    """How many sleep-box samples land in the W-maze's bins."""
    resting = _within(times, rest) & tracked_mask(positions)
    inside = (
        float(np.mean(env.bin_at(positions[resting]) >= 0)) if resting.any() else 0.0
    )
    return (
        "The sleep box is filmed in the same camera frame as the W-maze, and its "
        f"positions overlap the maze's: {inside:.0%} of the tracked samples of the "
        "rest epochs fall in a bin of rec.env. Use params['analysis_epochs'] (the "
        "run epochs) for rate maps and occupancy; the positions are kept."
    )


def _not_in_paper(animal: str, session: str) -> tuple[str, ...]:
    """A watch-out for the rats and files outside Shin et al. 2019."""
    if animal in PAPER_RATS:
        return ()
    split = (
        " ZT2's day is split across two files (9 and 8 epochs, sorted separately: "
        "the unit tables differ), so neither holds all 8 W-maze sessions."
        if animal == "ZT2"
        else ""
    )
    return (
        f"{animal} ({session}) is not one of the paper's six rats, so no number of "
        f"the paper describes it.{split}",
    )


def _speed_conversion(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    velocity: NDArray[np.float64],
    in_run: NDArray[np.bool_],
) -> Conversion:
    """The file's velocity column against ``compute_speed`` at the paper's thresholds."""
    from neurospatial.behavior import compute_speed

    speed = np.asarray(
        compute_speed(times, positions, smooth_window=SPEED_SMOOTH_WINDOW),
        dtype=np.float64,
    )
    both = in_run & np.isfinite(speed) & np.isfinite(velocity)
    agree = {
        limit: float(np.mean((speed[both] <= limit) == (velocity[both] <= limit)))
        for limit in (IMMOBILITY_SPEED, MIN_SPEED)
    }
    moving = both & (speed > 10.0)
    ratio = float(np.median(velocity[moving] / speed[moving]))
    return Conversion(
        "speed: the file's velocity column is not loaded; compute_speed(times, "
        f"positions, smooth_window={SPEED_SMOOTH_WINDOW:g}) stands in for it "
        "(params['speed_smooth_window'])",
        "inferred",
        "the file does not say how its velocity was computed. In the run epochs "
        f"its samples above 10 cm/s are a median {ratio:.3f} times that speed, "
        f"and the two agree on {agree[IMMOBILITY_SPEED]:.1%} of samples at the "
        f"paper's {IMMOBILITY_SPEED:g} cm/s immobility threshold and on "
        f"{agree[MIN_SPEED]:.1%} at its {MIN_SPEED:g} cm/s movement threshold",
    )


def _trial_conversion(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    velocity: NDArray[np.float64],
    table: Any,
    wells: Mapping[int, NDArray[np.float64]],
) -> Conversion:
    """Where a trial's stop falls: as the rat leaves its end well, with numbers."""
    stops = table["stop_time"].to_numpy(dtype=np.float64)
    ends = table["end_well"].to_numpy(dtype=int)
    index = np.clip(np.searchsorted(times, stops), 0, times.size - 1)
    distance = np.array(
        [
            np.linalg.norm(positions[i] - wells[int(w)])
            for i, w in zip(index, ends, strict=True)
        ]
    )
    later = np.clip(index + 30, 0, times.size - 1)
    return Conversion(
        "trials: the file's table as stored (start_well, end_well, "
        "trajectory_type, correct); well 1 is the centre well, trajectory_type 0 "
        "outbound (from well 1) and 1 inbound; a trial ends as the rat leaves its "
        "end well",
        "inferred",
        "the file says only 'trial data and properties'. Every trial with "
        "trajectory_type 0 starts at well 1, whose arm is the middle one. Each "
        "trial starts where the one before stops; at a trial's stop the authors' "
        f"velocity is a median {np.nanmedian(velocity[index]):.1f} cm/s, and a "
        f"median {np.nanmedian(velocity[later]):.1f} cm/s 30 frames later, and "
        f"the rat is a median {np.nanmedian(distance):.1f} cm from the end of its "
        "end well's arm: it has just left the well, so the stay at the well is the "
        "end of the trial before. The infrared beam times of arrival and "
        "departure the paper used are not in the file",
    )


def _w_maze_graph(nodes: Mapping[str, NDArray[np.float64]]) -> nx.Graph:
    """The W-maze graph with node positions and edge lengths."""
    import networkx as nx

    graph = nx.Graph()
    for name, position in nodes.items():
        graph.add_node(name, pos=tuple(float(value) for value in position))
    for a, b in _EDGES:
        graph.add_edge(
            a, b, distance=float(np.linalg.norm(nodes[a] - nodes[b])), edge_id=None
        )
    return graph


_NAMES = ("well_1", "well_2", "well_3", "base_1", "base_2", "base_3")


def _fit_w_maze(samples: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    """The W-maze's six nodes fitted to tracked samples.

    Two truncated squared terms, as for Huszár et al. 2022's figure eight: the
    samples' distance to the track, and the distance from points along each
    edge to the nearest sample. The start puts the arms at the quartiles of
    the samples' x range, the wells at their 95th percentile of y and the base
    at their 5th; the centre arm is the middle one.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, 2)
        Tracked positions of the run epochs, cm.

    Returns
    -------
    dict of str to ndarray, shape (2,)
        ``well_1`` to ``well_3`` (the arms' ends: centre, low x, high x) and
        ``base_1`` to ``base_3`` (the arms' feet).
    """
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(_FIT_SEED)
    if len(samples) > _FIT_SAMPLES:
        samples = samples[rng.choice(len(samples), _FIT_SAMPLES, replace=False)]
    tree = cKDTree(samples)
    index = {name: k for k, name in enumerate(_NAMES)}
    a = np.array([index[first] for first, _ in _EDGES])
    b = np.array([index[second] for _, second in _EDGES])
    steps = np.linspace(0.0, 1.0, _EDGE_POINTS)[None, :, None]

    def loss(flat: NDArray[np.float64]) -> float:
        nodes = flat.reshape(-1, 2)
        to_track = nearest_segment(samples, nodes[a], nodes[b]).distance
        along = nodes[a][:, None] + steps * (nodes[b] - nodes[a])[:, None]
        to_samples = tree.query(along.reshape(-1, 2))[0]
        lengths = np.linalg.norm(nodes[b] - nodes[a], axis=1)
        weight = np.repeat(lengths / max(lengths.sum(), 1e-12), _EDGE_POINTS)
        return float(
            np.mean(np.minimum(to_track, _FIT_CAP) ** 2)
            + np.sum(weight * np.minimum(to_samples, _FIT_CAP) ** 2) / _EDGE_POINTS
        )

    x_low, x_high = np.percentile(samples[:, 0], [2.5, 97.5])
    y_low, y_high = np.percentile(samples[:, 1], [5.0, 95.0])
    arm_x = {1: (x_low + x_high) / 2, 2: x_low, 3: x_high}
    start = np.array(
        [[arm_x[k], y_high] for k in (1, 2, 3)] + [[arm_x[k], y_low] for k in (1, 2, 3)]
    )
    fit = minimize(loss, start.ravel(), method="L-BFGS-B")
    return dict(zip(_NAMES, fit.x.reshape(-1, 2), strict=True))


class _Placement(NamedTuple):
    """How the declared track is placed: its conversion and width's quantile."""

    conversion: Conversion
    quantile: float | None


def _declared_w_maze(
    samples: NDArray[np.float64], session: str
) -> tuple[Environment, _Placement, dict[int, NDArray[np.float64]]]:
    """The W-maze on the run samples, its width the stated 7 cm or the tolerance.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, 2)
        Tracked positions of the run epochs, cm.
    session : str

    Returns
    -------
    env : Environment
    placement : _Placement
    wells : dict of int to ndarray, shape (2,)
        Each well's position (the end of its arm), by the trials' well number.
    """
    from neurospatial import Environment
    from neurospatial.regions import Region

    nodes = _fit_w_maze(samples)
    graph = _w_maze_graph(nodes)
    distance = nearest_segment(
        samples,
        np.array([nodes[a] for a, _ in _EDGES]),
        np.array([nodes[b] for _, b in _EDGES]),
    ).distance
    width, quantile, evidence = placement_tolerance(
        distance,
        _TOLERANCE_QUANTILE,
        floor=TRACK_WIDTH,
        measured="the tracked LEDs' distance from the graph",
        reason=(
            f"The track sections are {TRACK_WIDTH:g} cm wide ({_PAPER}), but LEDs on "
            "the headstage swing past them, most of all as the rat turns at a well."
        ),
    )
    wells = {k: nodes[f"well_{k}"] for k in (1, 2, 3)}
    env = Environment.from_graph(
        graph,
        list(_EDGES),
        bin_size=BIN_SIZE,
        units="cm",
        name=f"shin_2019 {session}",
        track_width=width,
        regions={
            f"well_{k}": Region(name=f"well_{k}", kind="point", data=well)
            for k, well in wells.items()
        },
    )
    lengths = ", ".join(
        f"{a}-{b} {graph.edges[a, b]['distance']:.1f}" for a, b in _EDGES
    )
    spans = np.ptp(samples, axis=0)
    conversion = Conversion(
        "environment: the W-maze's six nodes fitted to the samples of the run "
        f"epochs, five edges (cm: {lengths}), track_width {width:.3g} cm; the arms' "
        "ends as the wells, well_1 the middle arm's",
        "inferred",
        f"no source read gives the nodes; the stated maze is {MAZE_SIDE:g} x "
        f"{MAZE_SIDE:g} cm and these samples span {spans[0]:.1f} x {spans[1]:.1f} cm. "
        f"{evidence}",
    )
    return env, _Placement(conversion, quantile), wells


def _lfp_source(
    entry: SessionEntry,
    source: str | os.PathLike[str],
    electrodes: Any,
    rows: list[int],
    span: tuple[float, float],
    unit: str,
) -> LFPSource:
    """The file's LFP, read by byte range on request; nothing is read here."""
    streamed = isinstance(source, str) and source.startswith(("https://", "http://"))
    names = _channel_names(electrodes)
    numbers = _tetrode_numbers(electrodes)
    channels = {
        names[row]: (
            f"{electrodes['location'].iloc[row]}, tetrode {numbers[row]}, "
            + (
                "referenced to the region's reference tetrode"
                if electrodes["label"].iloc[row] == "Ref ON"
                else "not referenced"
            )
        )
        for row in rows
    }
    assert entry.asset_id is not None  # every session is a DANDI asset
    return LFPSource(
        dataset=entry.info.dataset,
        files=(
            LFPFile(
                path=entry.file_name,
                sha256=entry.sha256,
                size_bytes=entry.size_bytes,
                asset_id=entry.asset_id,
            ),
        ),
        sampling_rate=LFP_RATE,
        channels=channels,
        unit="uV",
        unit_basis=Conversion(
            "LFP: the stored values, in µV",
            "stated",
            f"the file's processing/ecephys/LFP series: unit {unit!r}, conversion 1",
        ),
        times_basis=Conversion(
            "LFP times: the series' own timestamps, on the recording's clock",
            "stated",
            f"the series stores one timestamp per sample, {span[0]:.6g} to "
            f"{span[1]:.6g} s, stepping 1/1,500 s within each epoch, with gaps "
            "between epochs",
        ),
        reader=_read_lfp,
        cache=None if streamed else Path(source).parents[1],
    )


def _channel_names(electrodes: Any) -> list[str]:
    """``"tetrode6 Ref ON"``: each electrode row's channel name."""
    return [
        f"{group} {label}"
        for group, label in zip(
            electrodes["group_name"], electrodes["label"], strict=True
        )
    ]


def _read_lfp(
    sources: tuple[str | Path, ...], channels: tuple[str, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read channels of the file's LFP, every sample, in µV on the file's clock.

    Parameters
    ----------
    sources : tuple of (str or path)
        The one NWB file, or its URL.
    channels : tuple of str
        Channel names, ``"tetrode6 Ref ON"``.

    Returns
    -------
    samples : ndarray, shape (n_time, n_channels)
    times : ndarray, shape (n_time,)
    """
    ((samples, times),) = _read_lfp_windows(sources[0], channels, None)
    return samples, times


def _read_lfp_windows(
    source: str | Path,
    channels: tuple[str, ...],
    windows: NDArray[np.float64] | None,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Read channels of the file's LFP within time windows, in µV.

    The file stores the LFP as one contiguous float64 array, ``(n_time,
    n_channels)``, so a time window is one byte range. From a URL the rows
    are read by parallel byte ranges straight from that array; from a local
    file, or for any other layout, with h5py.

    Parameters
    ----------
    source : str or path
        The NWB file, or its URL.
    channels : tuple of str
        Channel names, ``"tetrode6 Ref ON"``.
    windows : ndarray, shape (n_windows, 2), or None
        [start, stop) seconds on the file's clock; None reads every sample.

    Returns
    -------
    list of (samples, times)
        One per window: samples, shape (n_time, n_channels), columns in the
        order of ``channels``; times, shape (n_time,), the series' timestamps.
    """
    from neurospatial.datasets._fetch import open_nwb

    location = _direct(os.fspath(source))
    with open_nwb(location) as nwb:
        series = nwb.processing[_LFP[0]][_LFP[1]][_LFP[2]]
        names = _channel_names(nwb.electrodes.to_dataframe())
        column = {names[row]: k for k, row in enumerate(series.electrodes.data[:])}
        wanted = np.array([column[name] for name in channels])
        stamps = np.asarray(series.timestamps[:], dtype=np.float64)
        n_time, n_columns = series.data.shape
        bounds = (
            [(0, n_time)]
            if windows is None
            else [
                (
                    int(np.searchsorted(stamps, start, side="left")),
                    int(np.searchsorted(stamps, stop, side="left")),
                )
                for start, stop in np.asarray(windows, dtype=np.float64)
            ]
        )
        data = series.data
        contiguous = (
            data.chunks is None
            and data.compression is None
            and data.dtype == np.dtype("<f8")
        )
        ranged = location.startswith(("https://", "http://")) and contiguous
        offset = int(data.id.get_offset()) if ranged else 0
        found = []
        for first, stop in bounds:
            if ranged:
                samples = _ranged_rows(location, offset, n_columns, first, stop, wanted)
            else:
                samples = np.empty((stop - first, len(wanted)))
                step = max(_LFP_BLOCK_BYTES // (8 * n_columns), 1)
                for start in range(first, stop, step):
                    block = np.asarray(data[start : min(start + step, stop)])
                    samples[start - first : start - first + len(block)] = block[
                        :, wanted
                    ]
            found.append((samples, stamps[first:stop].copy()))
    return found


def _ranged_rows(
    url: str,
    offset: int,
    n_columns: int,
    first: int,
    stop: int,
    wanted: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Rows ``first`` to ``stop`` of a contiguous little-endian float64 array.

    Parameters
    ----------
    url : str
        The file's URL, past any redirect.
    offset : int
        The array's byte offset in the file.
    n_columns : int
    first, stop : int
        The rows to read, [first, stop).
    wanted : ndarray of int
        The columns to keep, in order.

    Returns
    -------
    ndarray, shape (stop - first, len(wanted))
    """
    row_bytes = 8 * n_columns
    step = max(_LFP_BLOCK_BYTES // row_bytes, 1)
    blocks = [(start, min(start + step, stop)) for start in range(first, stop, step)]
    out = np.empty((stop - first, len(wanted)))

    def read(block: tuple[int, int]) -> None:
        start, end = block
        size = (end - start) * row_bytes
        low = offset + start * row_bytes
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={low}-{low + size - 1}"}
        )
        for attempt in range(_LFP_ATTEMPTS):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    raw = response.read()
                if len(raw) == size:
                    break
                failure: str = f"returned {len(raw)} bytes, not {size}"
            except (OSError, http.client.HTTPException) as error:
                failure = f"failed: {type(error).__name__}: {error}"
            time.sleep(2.0**attempt)
        else:
            msg = (
                f"byte range of LFP rows {start} to {end} {failure}, "
                f"{_LFP_ATTEMPTS} times"
            )
            raise OSError(msg)
        rows = np.frombuffer(raw, dtype="<f8").reshape(end - start, n_columns)
        out[start - first : end - first] = rows[:, wanted]

    with ThreadPoolExecutor(_LFP_WORKERS) as pool:
        list(pool.map(read, blocks))
    return out
