"""Huszár et al. (2022): mouse CA1 on a figure-eight maze with sleep (DANDI 000552).

What was recorded: CA1 units from silicon probes in adult mice whose pyramidal
cells were birthdated in utero. Nine mice ran "a spatial alternation task in a
figure-eight maze", "a customized 79 x 79 cm2 figure-eight maze ... raised 61
cm above the ground": visiting alternate arms between trials "to receive a
water reward in the first corner reached after making a correct left/right
turn", with "a 5-s delay in the start area" between trials (the Methods of
Huszár et al. 2022, read on PubMed Central, PMC10807234). Yang et al. 2024,
Science, reused the data: "animals spent ~120-min in the homecage before running
on the maze and another ~120-min in the homecage afterward", tracked by
"head-mounted red LEDs" filmed "at a frame rate of 30-Hz" (Supplementary
Materials, Behavior, dataset 1). Some sessions add a second recording on another
maze, which the processing module's description names (``alternation,
linearMaze``, ``alternation, alternation``).

Sessions are DANDI 000552's files that hold units and position, version
0.230630.2304, CC-BY-4.0: 65 files of 41 MB to 2.5 GB. The other 39
``behavior+ecephys`` files hold ripples and sleep states only, and are not
offered. The default session, ``sub-e14-2m2_ses-e14-2m2-201019``, is the survey
of public datasets' session: 113 units, 21 trials, 4,297 ripples in 18,557 s.

Sources read: DANDI 000552's metadata, the files, Huszár et al. 2022 (PMC10807234)
and Yang et al. 2024 with its Supplementary Materials.

What the loader converts:

- positions: the file's ``SubjectPosition`` as stored, camera (x, y) in cm.
- ``rec.linearized``: the file's ``LinearizedPosition`` as stored, 0 to 222 cm.
  It runs around the maze as a loop, so 0 and 222 cm are one place, and both
  loops of the figure eight share it: the top and bottom arms have the same
  values. The line cannot hold a loop, so the environment is declared on the
  camera positions, not on this.
- environment: the figure eight as a track graph with both loops closed, a
  rectangle with a bar across its middle (six nodes, seven edges), in 2 cm bins.
  The nodes are fitted to the samples of the first alternation recording, since
  no source gives them; the stated 79 x 79 cm is the maze's outer size. Distances
  on it run through the junctions, so a path through the seam of the linear
  position is continuous.
- epochs: ``maze`` (where the file's ``epochs`` rows hold position samples, else
  the tracked span), ``alternation`` and any ``linear_maze`` by the recordings
  the file names, ``off_maze`` (the rest of the recording), ``trials``, and ``awake``, ``nrem`` and ``rem`` (``SleepStates``
  by label).
- trials: the file's trials table; rows without a time are left out and counted.
- events: ``ripples``, the ``Ripples`` table's [start, stop) rows;
  ``ripple_peaks``, its ``peaks``; ``rewards``, ``left_rewards`` and
  ``right_rewards``, from ``RewardEventsEightMazeTrack``.
- ``units``: ``brain_region`` is CA1 for every unit (the papers' recording site;
  the file says ``unknown``); ``label`` is missing, since the file has no
  cell-type column. The file's ``unit_name``, ``clu_id`` and ``group_id`` are kept.
- headings: none. One LED position is stored.
- ``rec.lfp``: for the 12 sessions with a sibling ``-raw_ecephys`` file holding
  the 1,250 Hz LFP, in µV. It is None elsewhere: the ``behavior+ecephys`` files
  hold no usable LFP.

Analysis choices (``params``):

- ``bin_size`` 2 cm, the loader's choice: the bin of the ``GROSMARK_2016`` replay
  recipe. Huszár et al. 2022 use 1 cm bins smoothed with a 9 cm (FWHM) Gaussian.
- ``min_speed`` 4 cm/s, the loader's choice, which leaves out stops at the reward
  corners, where the animal is still and ripples happen. Huszár et al. 2022 use
  1.5 cm/s after a Kalman filter; Yang et al. 2024 state "2-5-cm/s".
- ``analysis_epochs``: the first alternation recording.
- ``train_epochs`` and ``test_epochs``: its trials alternating within each visited
  arm, so both halves hold both arms. Consecutive trials alternate between the
  arms, so even and odd trials would split them by arm.
- ``dt`` 0.25 s and ``replay_dt`` 0.02 s, the decoding bins.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._lfp import LFPSource
from neurospatial.datasets._recording import (
    Conversion,
    Recording,
    interval_rows,
    nwb_unit_table,
    tracked_mask,
)
from neurospatial.datasets._tracks import nearest_segment, placement_tolerance

if TYPE_CHECKING:
    import networkx as nx

    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = [
    "EXPECTED_WATCH_OUTS",
    "REPLAY_ANALYSIS",
    "REQUIRES",
    "WATCH_OUTS",
    "WORKED_RESULT",
    "YANG_VALIDATION",
    "load_session",
]

#: A sanity property of our own awake-replay analysis, not a finding of either
#: paper: Yang et al.'s replay is a UMAP manifold test, which is no preset here.
#: The test is in ``tests/datasets/test_huszar_2022.py``.
WORKED_RESULT: dict[str, Any] = {
    "status": "sanity property",
    "level": "B",
    "finding": (
        "a sanity property of our own analysis, not a finding of either paper: "
        "awake ripple candidates on the maze are judged trajectories more often "
        "than the same windows after circularly shifting the spikes"
    ),
    "quote": (
        '"Ripple events were conditioned on the coincidence of both population '
        'synchrony events, and LFP detected ripples" (Supplementary Materials, '
        '"Ripple detection"), the candidates used here; "Because spike sequences '
        "within SPW-Rs are known to replay place field sequences (27, 28), we "
        "asked whether trial block identity could also be decoded in replay "
        'events." (Results); the replay test there is "distance to manifold and '
        'trajectory length", on a UMAP embedding'
    ),
    "source": (
        "Yang et al. 2024, Science 383:1478-1483, Results and Supplementary "
        "Materials, read in full"
    ),
    "paper": "no comparable number",
    "ours": 0.25,
    "criterion": (
        "the count of candidates judged trajectories (5 of 20) exceeds the 95th "
        "percentile of the counts of each of two nulls, 20 draws each (seed "
        "2022): every unit shifted by its own offset, and all units by one common "
        "offset, circularly over the first alternation recording, by at least "
        "10 s (every draw of both: 0); fixed before any event was decoded, and "
        "met"
    ),
    "parameters": (
        "session sub-e15-13f1_ses-e15-13f1-220117, the one with the most units "
        "among those meeting Yang et al.'s inclusion with one figure-eight "
        "recording; candidates by ripple_detection's yang_2024 on all sorted units "
        "(the file has no cell-type labels), the Non-REM rows, the Ripples table's "
        "peaks, eligible on the maze while the file says Awake; 325 of 376 units "
        "with has_place_field over the first recording at min_speed; "
        "compute_spatial_rates on rec.env (default diffusion KDE) likewise; "
        "decoded in 20 ms bins stepped by 5 ms with the default prior; judged by "
        "PFEIFFER_FOSTER_2013 unchanged, a MAP step under 20 cm over at least 11 "
        "bins and at least 40 cm from start to end, geodesic on the closed loops"
    ),
    "unmatched": (
        "this is neurospatial's method on Yang et al.'s candidates, not Yang et "
        "al.'s analysis: their replay is a UMAP distance to the manifold and "
        "trajectory length against shuffles, with no stated cutoff or shuffle "
        "count, and gives 'about 33% of the awake SPW-Rs'; that number measures "
        "something else and is not compared",
        "the candidates pool every sorted unit where the paper uses pyramidal "
        "cells, since the file does not label them",
        "the awake candidates are those on the maze while the file says Awake; "
        "the paper's 'quite awake immobility' has no stated speed",
        "20 candidates in one session is a small count; the shifted-spike draws "
        "judge none, so this shows the method finds sequences here, not their "
        "rate",
    ),
    "attempted": None,
    "why": None,
}

#: Yang et al. (2024)'s events on a session of the dandiset against our detection.
#: The event file ``ripple_detection``'s note names,
#: ``e13_26m1_210913.rippleHSE.events.mat``, is from a session not in DANDI 000552
#: (the animal's four sessions there are 211019, 211116, 211119, 211124), and it
#: is in the paper's GitHub repository, not its Zenodo archive. The repository's
#: analysis file for ``e15_13f1_220118`` is on a session of the dandiset, so the
#: comparison is made there.
YANG_VALIDATION: dict[str, Any] = {
    "session": "sub-e15-13f1_ses-e15-13f1-220118",
    "file": (
        "e15_13f1_220118.data_RSE_all_trial_5_pos_5_speed_0_smooth_3_z_1_together"
        "_equal.mat"
    ),
    "url": (
        "https://raw.githubusercontent.com/winnieyangwannan/Selection-of-experience"
        "-for-memory-by-hippocampal-sharp-wave-ripples/"
        "f1bc2fb34de4a5b68ff0aad6849c2ea22e9c5cf3/Data/e15_13f1_220118."
        "data_RSE_all_trial_5_pos_5_speed_0_smooth_3_z_1_together_equal.mat"
    ),
    "sha256": "917622c7c9c6d0b833ef02ce0bf34f62ee9533b4b64c2d1b687a60e882717757",
    "n_theirs": 4088,
    "n_ours": 7536,
    "largest_id": 8758,
    "share_of_theirs_we_overlap": 0.943,
    "share_of_ours_they_overlap": 0.501,
    "why_ours_exceed_theirs": (
        "the analysis file holds a selection of the authors' own events: 4,088 "
        "ids, the largest 8,758, none under four 20 ms bins, and no source read "
        "states that selection. Their ids follow the rank of our events "
        "(correlation 0.9999, 0.87 of ours per id), so the list they were cut from "
        "is about ours, and none of our events under 100 ms is in the file"
    ),
    "why_theirs_are_missed": (
        "232 of their events: 145 peak below our trigger of 3 SD of the NREM "
        "rate (99.9% of all their events reach 2.5 SD), and the other 87 lie in an "
        "excursion above the NREM mean longer than the paper's 500 ms limit, so "
        "the paper's rule as written drops them whole. So their bounds and "
        "trigger sit tighter and lower than the text. Neither normalizing over "
        "all bins, as the find_HSE code does (88.4%), nor leaving out the 33 units "
        "over 5 Hz (93.6%) explains it"
    ),
}

#: The numbers behind ``WORKED_RESULT``, recomputed by its test.
REPLAY_ANALYSIS: dict[str, Any] = {
    "session": "sub-e15-13f1_ses-e15-13f1-220117",
    "n_events": 20,
    "n_judged": 5,
    "n_units": 325,
    "null_own_offsets": (0,) * 20,
    "null_common_offset": (0,) * 20,
    "held_out_laps": {"median_error_cm": 1.98, "shuffled_error_cm": 69.61},
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("pynwb",)

BIN_SIZE = 2.0
MIN_SPEED = 4.0
DT = 0.25
REPLAY_DT = 0.02
#: The stated maze's side, cm.
MAZE_SIDE = 79.0
#: The file's linear position runs from 0 to about this, cm.
LOOP_LENGTH = 222.0
#: A step of the linear position larger than this is its seam, cm.
SEAM_STEP = 100.0
#: The LFP's rate, as the raw files state it, Hz.
LFP_RATE = 1250.0
#: Tracked samples more than this apart are different recordings, s. The longest
#: gap inside one recording of the 65 sessions is under this, and the shortest
#: between two recordings is 3,052 s.
RECORDING_GAP = 60.0
#: Yang et al. 2024's inclusion: "more than 150 cells and >20 trials".
YANG_MIN_CELLS = 150
YANG_MIN_TRIALS = 20
#: How far a fitted node may be pulled by one sample, cm; samples per fit; points
#: along each edge; and the seed of the sample subset.
_FIT_CAP = 10.0
_FIT_SAMPLES = 6000
_EDGE_POINTS = 25
_FIT_SEED = 0
#: The share of samples the placement tolerance holds.
_TOLERANCE_QUANTILE = 0.99

_BEHAVIOR = "processing/behavior"
_CAMERA = f"{_BEHAVIOR}/SubjectPosition/SpatialSeries"
_LINEAR = f"{_BEHAVIOR}/LinearizedPosition/LinearizedSpatialSeries"
_STATES = {"Awake": "awake", "Non-REM": "nrem", "REM": "rem"}
_PAPER = "the Methods of Huszár et al. 2022, PMC10807234"
_YANG = "Yang et al. 2024, Supplementary Materials"
_EDGES = (
    ("corner_0", "corner_1"),
    ("corner_1", "bar_1"),
    ("bar_1", "corner_2"),
    ("corner_2", "corner_3"),
    ("corner_3", "bar_0"),
    ("bar_0", "corner_0"),
    ("bar_0", "bar_1"),
)

#: Findings computed from the data (by the checks or this loader) that each
#: verified session must report, as patterns searched for in its watch_outs.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "sub-e14-2m2_ses-e14-2m2-201019": (
        r"rec\.linearized jumps by more than 100 cm 62 times",
        r"This session has 113 units and 21 trials",
        r"runs of 11 or more identical position samples",
        r"lie outside the declared environment",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "sub-e14-2m2_ses-e14-2m2-201019": (
        "Position covers 1,591 s of this 18,557 s session, from 32 s into the "
        "maze epoch to 9 s before its end; the home-cage sleep around it has none.",
        "The trials' is_familiar_maze is True for the first 2 of the 21 trials and "
        "False after, all in one recording on one maze, so it does not describe "
        "the maze here; the loader does not use it.",
    ),
}

_NO_LABELS = (
    "The file does not say which units are pyramidal cells: its units table holds "
    "only unit_name, clu_id, group_id and the spike times. Huszár et al. 2022 "
    "classified 7,411 putative pyramidal cells and 2,060 putative interneurons "
    "with CellExplorer, from waveforms and autocorrelograms; the waveforms are not "
    "in these files. Anything the papers compute on pyramidal cells can only be "
    "computed on all sorted units here; do not read a firing rate cut as the "
    "classification."
)
_UNLOADED = (
    "The Ripples table also holds each ripple's power, frequency and waveform "
    "(ripple_raw, ripple_amplitudes, ripple_frequencies, ripple_phases), which are "
    "not loaded; read them with pynwb."
)


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one DANDI 000552 session into a :class:`Recording`.

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
        If a position series is not in cm, or the two series have different
        timestamps.
    """
    from neurospatial import Intervals
    from neurospatial.datasets._fetch import open_nwb
    from neurospatial.io.nwb import read_position, read_units

    with open_nwb(os.fspath(path_or_stream)) as nwb:
        camera = read_position(nwb, position_name=_CAMERA)
        linear = read_position(nwb, position_name=_LINEAR)
        units = read_units(nwb)
        table = nwb.trials.to_dataframe().reset_index()
        session_rows = (
            np.empty((0, 2))
            if nwb.epochs is None
            else nwb.epochs.to_dataframe()[["start_time", "stop_time"]].to_numpy(
                dtype=np.float64
            )
        )
        states = nwb.processing["behavior"]["SleepStates"].to_dataframe()
        ripples = nwb.processing["ecephys"]["Ripples"]
        ripple_rows = np.column_stack(
            [ripples["start_time"][:], ripples["stop_time"][:]]
        ).astype(np.float64)
        ripple_peaks = np.asarray(ripples["peaks"][:], dtype=np.float64)
        reward_times, reward_arms = _rewards(nwb)
        description = str(nwb.processing["behavior"].description)
        electrodes = nwb.electrodes.to_dataframe().drop(columns="group")
        subject = nwb.subject.subject_id
        started = nwb.session_start_time
    for series, name in ((camera, _CAMERA), (linear, _LINEAR)):
        if series.units != "cm":
            msg = f"{entry.session}: {name} is in {series.units!r}, not cm"
            raise ValueError(msg)
    times = np.asarray(camera.times, dtype=np.float64)
    if not np.array_equal(times, np.asarray(linear.times, dtype=np.float64)):
        msg = f"{entry.session}: the two position series have different timestamps"
        raise ValueError(msg)
    positions = np.asarray(camera.positions, dtype=np.float64)
    linearized = np.asarray(linear.positions, dtype=np.float64).ravel()

    mazes = _recordings_named(description)
    runs = _maze_rows(times, session_rows)
    named = mazes if len(mazes) == len(runs) else ["alternation"] * len(runs)
    alternation = runs[[name == "alternation" for name in named]]
    if not len(alternation):
        alternation = runs[:1]
    first = alternation[:1]
    in_first = _within(times, first)
    env, placement = _declared_figure_eight(
        positions[in_first & tracked_mask(positions)], entry.session
    )

    known = table[table["start_time"].notna() & table["stop_time"].notna()]
    trial_rows = interval_rows(known)
    trials = Intervals.from_dataframe(known, id_column="id")
    arms = known.sort_values("start_time", kind="stable")["visited_arm"].to_numpy(
        dtype=str
    )
    first_rows = (trial_rows[:, 0] >= first[0, 0]) & (trial_rows[:, 1] <= first[0, 1])
    train = _alternate_within(arms[first_rows])
    kept = trial_rows[first_rows]

    zero = states["start_time"] >= states["stop_time"]
    state_rows = states[~zero]
    order = ripple_rows[:, 0].argsort(kind="stable")
    unit_table, _ = nwb_unit_table(units, region="CA1", keep_columns=True)
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"mouse {subject}",
        apparatus=entry.info.apparatus,
        brain_regions=entry.info.brain_regions,
        clock=(
            "the file's clock, seconds from its session start time "
            f"({started:%Y-%m-%d}, the recording's date with no time of day)"
        ),
        times=times,
        spike_times=tuple(units.spike_times),
        units=unit_table,
        env=env,
        env_kind="declared",
        env_source=(
            f"the figure-eight maze ({MAZE_SIDE:g} x {MAZE_SIDE:g} cm, {_PAPER}) as a "
            f"track graph with both loops closed, its nodes fitted to the samples "
            f"of the first alternation recording, {BIN_SIZE:g} cm bins"
        ),
        track_width_quantile=placement.quantile,
        positions=positions,
        position_units="cm",
        tracking=f"head-mounted red LEDs, filmed from above at 30 frames/s ({_YANG})",
        linearized=linearized,
        epochs={
            "maze": runs,
            "alternation": alternation,
            **(
                {"linear_maze": rows}
                if len(rows := runs[[name == "linearMaze" for name in named]])
                else {}
            ),
            "off_maze": _off_maze(
                _tracked_stretches(times), _recording_stop(units.spike_times, states)
            ),
            "trials": trial_rows,
            **{
                key: interval_rows(state_rows[state_rows["label"] == label])
                for label, key in _STATES.items()
                if (state_rows["label"] == label).any()
            },
        },
        events={
            "ripples": ripple_rows[order],
            "ripple_peaks": np.sort(ripple_peaks),
            "rewards": reward_times,
            "left_rewards": reward_times[reward_arms == "left_reward"],
            "right_rewards": reward_times[reward_arms == "right_reward"],
        },
        trials=trials,
        params={
            "bin_size": BIN_SIZE,
            "min_speed": MIN_SPEED,
            "dt": DT,
            "replay_dt": REPLAY_DT,
            "analysis_epochs": first,
            "train_epochs": kept[train],
            "test_epochs": kept[~train],
        },
        conversions=(
            Conversion(
                f"positions: {_CAMERA} as stored, cm",
                "stated",
                "the file's series: unit 'cm', conversion 1, '(x,y) coordinates "
                "tracking subject movement.'; reference frame 'Arbitrary, camera', "
                "so the axes' handedness is not stated",
            ),
            _seam_conversion(positions, linearized),
            Conversion(
                "epochs: 'maze', the rows of the file's epochs table that hold "
                "position samples, or the tracked span where no row does; "
                "'alternation' and 'linear_maze', those of the recordings the "
                f"behavior module's description names ({', '.join(mazes)}); 'trials'; "
                "SleepStates Awake, Non-REM and REM as 'awake', 'nrem' and 'rem'",
                "stated",
                "the file's tables and descriptions ('experimental epochs', "
                "'experimental trials', 'Sleep state of the animal.'); position "
                "was tracked only on a maze, in stretches apart by more than "
                f"{RECORDING_GAP:g} s where a session has two recordings",
            ),
            Conversion(
                "epochs: 'off_maze', the recording from 0 s to the last spike or "
                "sleep-state stop outside the tracked stretches",
                "inferred",
                "position was tracked only on a maze, and 'animals spent ~120-min "
                f"in the homecage before running on the maze and another ~120-min "
                f"in the homecage afterward' ({_YANG}), so this is mostly the home "
                "cage; no table says so",
            ),
            Conversion(
                f"events['ripples']: the {len(ripple_rows)} Ripples rows as "
                "[start, stop) in s; events['ripple_peaks']: their peaks; "
                "events['rewards'], 'left_rewards' and 'right_rewards': "
                "RewardEventsEightMazeTrack by its labels",
                "stated",
                "the file's tables, 'Ripples and their metrics' and 'Rewards in a "
                f"figure-eight maze'; detection as {_PAPER} state it (bz_FindRipples "
                "on manually selected channels, 130-200 Hz, 5 SD peak, 2 SD edges, "
                "20-200 ms, a noise channel's events excluded)",
            ),
            _arm_conversion(times, positions, placement.bar, kept, arms[first_rows]),
            Conversion(
                "units: brain_region CA1 for every unit; label missing",
                "stated",
                f"both papers record CA1 ({_PAPER}; {_YANG}); the file's electrodes "
                "say 'unknown' and its units table has no cell-type column",
            ),
            placement.conversion,
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            *_seam_watch_out(linearized, in_first),
            *_yang_inclusion(unit_table, kept),
            *_second_recording(times, positions, env, runs, named),
            *_missing_rows(len(table) - len(known), len(table), int(zero.sum())),
            *(
                ()
                if len(session_rows)
                else (
                    "The file has no epochs table: epochs['maze'] is the tracked "
                    "span of each recording.",
                )
            ),
            _NO_LABELS,
            _UNLOADED,
            _lfp_watch_out(entry),
        ),
        lfp=_lfp_source(entry, path_or_stream, electrodes),
    )


def _rewards(nwb: Any) -> tuple[NDArray[np.float64], NDArray[Any]]:
    """The times of ``RewardEventsEightMazeTrack`` and each one's label."""
    rewards = nwb.processing["behavior"]["RewardEventsEightMazeTrack"]
    labels = np.asarray(rewards.fields["data__labels"])
    return (
        np.asarray(rewards.timestamps[:], dtype=np.float64),
        labels[np.asarray(rewards.data[:], dtype=int)],
    )


def _recordings_named(description: str) -> list[str]:
    """The recordings a ``behavior`` module's description names, in file order.

    The files write them after a colon, comma-separated; one recording's name
    is spelled out letter by letter (``a, l, t, e, r, n, a, t, i, o, n``).

    Parameters
    ----------
    description : str
        ``processing/behavior``'s description.

    Returns
    -------
    list of str
    """
    parts = [part.strip() for part in description.rsplit(":", 1)[-1].split(",")]
    return ["".join(parts)] if all(len(part) == 1 for part in parts) else parts


def _tracked_stretches(times: NDArray[np.float64]) -> NDArray[np.float64]:
    """[first, last] sample of each stretch without a gap over ``RECORDING_GAP``.

    Parameters
    ----------
    times : ndarray, shape (n_time,)
        The position timestamps, sorted.

    Returns
    -------
    ndarray, shape (n_stretches, 2)
    """
    breaks = np.flatnonzero(np.diff(times) > RECORDING_GAP)
    return np.column_stack(
        [times[np.r_[0, breaks + 1]], times[np.r_[breaks, times.size - 1]]]
    )


def _maze_rows(
    times: NDArray[np.float64], session_rows: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The [start, stop) of each recording on a maze.

    Parameters
    ----------
    times : ndarray, shape (n_time,)
        The position timestamps, sorted.
    session_rows : ndarray, shape (n_rows, 2)
        The file's ``epochs`` table, perhaps empty.

    Returns
    -------
    ndarray, shape (n_recordings, 2)
        For each stretch of samples without a gap over ``RECORDING_GAP``, the
        ``epochs`` row that holds all of it, else the stretch itself.
    """
    rows = []
    for start, stop in _tracked_stretches(times):
        holding = session_rows[
            (session_rows[:, 0] <= start) & (session_rows[:, 1] >= stop)
        ]
        rows.append(holding[0] if len(holding) else (start, stop))
    return np.asarray(rows, dtype=np.float64)


def _off_maze(rows: NDArray[np.float64], stop: float) -> NDArray[np.float64]:
    """The recording around the tracked stretches, from 0 s to ``stop``.

    Parameters
    ----------
    rows : ndarray, shape (n_stretches, 2)
        The tracked stretches, sorted.
    stop : float
        The recording's end, s.

    Returns
    -------
    ndarray, shape (n, 2)
        The non-empty stretches before, between and after them.
    """
    edges = np.r_[0.0, rows.ravel(), stop]
    gaps = edges.reshape(-1, 2)
    kept: NDArray[np.float64] = gaps[gaps[:, 1] > gaps[:, 0]]
    return kept


def _recording_stop(spike_times: Any, states: Any) -> float:
    """The last spike or sleep-state stop, s."""
    last = max(float(train[-1]) for train in spike_times if len(train))
    return max(last, float(states["stop_time"].max()))


def _alternate_within(labels: NDArray[Any]) -> NDArray[np.bool_]:
    """Trials alternating between train and test within each label, in order.

    Parameters
    ----------
    labels : ndarray, shape (n_trials,)
        Each trial's visited arm.

    Returns
    -------
    ndarray of bool, shape (n_trials,)
        True for the 1st, 3rd, ... trial of each label.
    """
    train = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        train[np.flatnonzero(labels == label)[::2]] = True
    return train


def _read_lfp(
    sources: tuple[str | Path, ...], channels: tuple[str, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read channels of a ``-raw_ecephys`` file's LFP, in µV on the file's clock.

    Parameters
    ----------
    sources : tuple of (str or path)
        The one NWB file, or its URL.
    channels : tuple of str
        Channel names, ``ch<channel_name>``.

    Returns
    -------
    samples : ndarray, shape (n_time, n_channels)
        Microvolts, by the series' own conversion.
    times : ndarray, shape (n_time,)
        Seconds, from the series' starting time and rate.
    """
    from neurospatial.datasets._fetch import open_nwb

    with open_nwb(str(sources[0])) as nwb:
        series = nwb.processing["ecephys"]["LFP"]["ElectricalSeriesLFP"]
        table = series.electrodes.table
        names = [f"ch{name}" for name in table["channel_name"][:]]
        column = {names[row]: k for k, row in enumerate(series.electrodes.data[:])}
        wanted = np.array([column[name] for name in channels])
        order = np.argsort(wanted)
        stored = np.asarray(series.data[:, wanted[order].tolist()], dtype=np.float64)
        samples = stored[:, np.argsort(order)] * series.conversion * 1e6
        times = series.starting_time + np.arange(samples.shape[0]) / series.rate
    return samples, times


def _lfp_source(
    entry: SessionEntry, source: str | os.PathLike[str], electrodes: Any
) -> LFPSource | None:
    """The session's LFP, where a sibling file holds it; nothing is read here."""
    if not entry.lfp_files:
        return None
    streamed = isinstance(source, str) and source.startswith(("https://", "http://"))
    return LFPSource(
        dataset=entry.info.dataset,
        files=entry.lfp_files,
        sampling_rate=LFP_RATE,
        channels={
            f"ch{row.channel_name}": (
                f"{row.group_name}, probe position ({row.rel_x:g}, {row.rel_y:g}) µm"
                if hasattr(row, "rel_x")
                else str(row.group_name)
            )
            for row in electrodes.itertuples()
        },
        unit="uV",
        unit_basis=Conversion(
            "LFP: the stored integers times the series' conversion, in µV",
            "stated",
            "the raw file's ElectricalSeriesLFP: unit 'volts', conversion 1.95e-07; "
            "its electrodes' gain_to_uV is 0.195",
        ),
        times_basis=Conversion(
            "LFP times: the series' starting_time (0 s) plus sample index / 1,250 Hz",
            "stated",
            "the raw file's ElectricalSeriesLFP; its length matches the session's "
            "last epoch stop, and the 130-200 Hz power peaks at the Ripples "
            "table's peaks on the session checked",
        ),
        reader=_read_lfp,
        cache=None if streamed else Path(source).parents[1],
    )


def _lfp_watch_out(entry: SessionEntry) -> str:
    """Where this session's LFP is, and the size of asking for all of it."""
    if not entry.lfp_files:
        return (
            "This session has no LFP in the dandiset: rec.lfp is None. Twelve "
            "sessions have one, in a sibling -raw_ecephys file."
        )
    gigabytes = entry.lfp_files[0].size_bytes / 1e9
    return (
        f"The LFP sits in a {gigabytes:.0f} GB file beside the raw voltage, so read "
        "a channel by byte range, rec.lfp.fetch(['ch1'], stream=True). No source "
        "names the ripple channels ('manually selected channels', "
        f"{_PAPER}), so rec.lfp.ripple_channels is empty and fetch() asks for "
        "every channel."
    )


def _within(times: NDArray[np.float64], rows: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Whether each sorted time lies in a [start, stop) row, shape (n_time,)."""
    inside = np.searchsorted(rows[:, 0], times, side="right") - 1
    held: NDArray[np.bool_] = (inside >= 0) & (
        times < rows[np.clip(inside, 0, None), 1]
    )
    return held


def _figure_eight_graph(corners: NDArray[np.float64]) -> nx.Graph:
    """A rectangle with a bar across its middle, from its four corners in order.

    Sides ``corners[0]-corners[3]`` and ``corners[1]-corners[2]`` are cut in half
    by the bar.
    """
    import networkx as nx

    c0, c1, c2, c3 = corners
    nodes = {
        "corner_0": c0,
        "corner_1": c1,
        "corner_2": c2,
        "corner_3": c3,
        "bar_0": 0.5 * (c0 + c3),
        "bar_1": 0.5 * (c1 + c2),
    }
    return _with_nodes(nx.Graph(), nodes)


def _with_nodes(graph: nx.Graph, nodes: dict[str, NDArray[np.float64]]) -> nx.Graph:
    """``graph`` with the node positions, and the edges at their lengths."""
    for name, position in nodes.items():
        graph.add_node(name, pos=tuple(float(value) for value in position))
    for a, b in _EDGES:
        graph.add_edge(
            a, b, distance=float(np.linalg.norm(nodes[a] - nodes[b])), edge_id=None
        )
    return graph


def _fit_figure_eight(samples: NDArray[np.float64]) -> nx.Graph:
    """The figure-eight graph fitted to tracked samples.

    Two truncated squared terms: the samples' distance to the track (the track
    must pass where the samples are), and the distance from points along each
    edge to the nearest sample (the track must not run where none is). Each of
    the two starting layouts, the bar parallel to either pair of sides of the
    samples' smallest rotated rectangle, is fitted; the lower loss is kept.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, 2)
        Tracked positions, cm.

    Returns
    -------
    networkx.Graph
        Nodes ``corner_0`` to ``corner_3``, ``bar_0`` and ``bar_1`` with ``pos``,
        edges with ``distance``.
    """
    import networkx as nx
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
    from shapely.geometry import MultiPoint

    rng = np.random.default_rng(_FIT_SEED)
    if len(samples) > _FIT_SAMPLES:
        samples = samples[rng.choice(len(samples), _FIT_SAMPLES, replace=False)]
    tree = cKDTree(samples)
    names = list(_figure_eight_graph(np.zeros((4, 2))).nodes)
    index = {name: k for k, name in enumerate(names)}
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

    rectangle = MultiPoint(samples).minimum_rotated_rectangle
    corners = np.asarray(rectangle.exterior.coords)[:4]
    fits = []
    for turn in range(2):
        start = _figure_eight_graph(np.roll(corners, -turn, axis=0))
        nodes = np.array([start.nodes[name]["pos"] for name in names])
        fits.append(minimize(loss, nodes.ravel(), method="L-BFGS-B"))
    best = min(fits, key=lambda fit: fit.fun).x.reshape(-1, 2)
    return _with_nodes(nx.Graph(), dict(zip(names, best, strict=True)))


class _Placement(NamedTuple):
    """How a declared track is placed: its conversion, width's quantile and bar."""

    conversion: Conversion
    quantile: float | None
    bar: tuple[NDArray[np.float64], NDArray[np.float64]]


def _declared_figure_eight(
    samples: NDArray[np.float64], session: str
) -> tuple[Environment, _Placement]:
    """The figure eight on the samples, its width a placement tolerance.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, 2)
        Tracked positions of the first alternation recording, cm.
    session : str

    Returns
    -------
    env : Environment
    placement : _Placement
    """
    from neurospatial import Environment

    graph = _fit_figure_eight(samples)
    pos = {name: np.asarray(graph.nodes[name]["pos"]) for name in graph.nodes}
    distance = nearest_segment(
        samples,
        np.array([pos[a] for a, _ in _EDGES]),
        np.array([pos[b] for _, b in _EDGES]),
    ).distance
    width, quantile, evidence = placement_tolerance(
        distance,
        _TOLERANCE_QUANTILE,
        floor=BIN_SIZE,
        measured="the tracked LEDs' distance from the graph",
        reason=(
            "A head-mounted LED swings past the edge of an elevated track with no "
            "walls, and no source read states the track's width."
        ),
    )
    env = Environment.from_graph(
        graph,
        list(_EDGES),
        bin_size=BIN_SIZE,
        units="cm",
        name=f"huszar_2022 {session}",
        track_width=width,
    )
    lengths = ", ".join(
        f"{a}-{b} {graph.edges[a, b]['distance']:.1f}" for a, b in _EDGES
    )
    spans = np.ptp(samples, axis=0)
    conversion = Conversion(
        "environment: the figure eight's six nodes fitted to the samples of the "
        f"first alternation recording, seven edges (cm: {lengths}), track_width "
        f"{width:.3g} cm",
        "inferred",
        f"no source read gives the nodes; the stated maze is {MAZE_SIDE:g} x "
        f"{MAZE_SIDE:g} cm and these samples span {spans[0]:.1f} x {spans[1]:.1f} cm. "
        f"{evidence}",
    )
    return env, _Placement(conversion, quantile, (pos["bar_0"], pos["bar_1"]))


def _seam_conversion(
    positions: NDArray[np.float64],
    linearized: NDArray[np.float64],
) -> Conversion:
    """Why 0 and 222 cm of the linear position are one place, from the samples."""
    near_zero = positions[linearized < 5.0]
    near_end = positions[linearized > LOOP_LENGTH - 5.0]
    apart = float(
        np.linalg.norm(np.median(near_zero, axis=0) - np.median(near_end, axis=0))
    )
    return Conversion(
        f"linearized: {_LINEAR} as stored, cm; 0 and {LOOP_LENGTH:g} cm are one place",
        "inferred",
        "the file says only 'Linearization of the (x,y) coordinates tracking "
        f"subject movement.'; the samples within 5 cm of either end lie a median "
        f"{apart:.1f} cm apart on the camera, and they span "
        f"{linearized.min():.1f} to {linearized.max():.1f} cm",
    )


def _seam_watch_out(
    linearized: NDArray[np.float64], in_first: NDArray[np.bool_]
) -> tuple[str, ...]:
    """How often the linear position jumps at its seam."""
    jumps = int(np.sum(np.abs(np.diff(linearized[in_first])) > SEAM_STEP))
    if not jumps:
        return ()
    return (
        f"rec.linearized jumps by more than {SEAM_STEP:g} cm {jumps} times in the "
        f"first alternation recording, while the camera position moves smoothly: "
        f"it is a loop cut at one place, so 0 and {LOOP_LENGTH:g} cm are the same "
        "spot, and both loops of the figure eight share its values. A linear "
        "environment on it would put neighbouring places far apart: use rec.env "
        "and rec.positions, whose distances run through the junctions.",
    )


def _yang_inclusion(units: Any, trials: NDArray[np.float64]) -> tuple[str, ...]:
    """A watch-out for a session outside Yang et al. 2024's inclusion rule."""
    if len(units) > YANG_MIN_CELLS and len(trials) > YANG_MIN_TRIALS:
        return ()
    return (
        f"This session has {len(units)} units and {len(trials)} trials in its first "
        f"alternation recording; Yang et al. 2024 kept 'sessions with more than "
        f"{YANG_MIN_CELLS} cells and >{YANG_MIN_TRIALS} trials' ({_YANG}), so its "
        "analyses did not use it.",
    )


def _second_recording(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    env: Environment,
    runs: NDArray[np.float64],
    named: Sequence[str],
) -> tuple[str, ...]:
    """How far a second recording's samples fall from the declared figure eight."""
    if len(runs) < 2:
        return ()
    later = _within(times, runs[1:]) & tracked_mask(positions)
    share = float(np.mean(env.bin_at(positions[later]) < 0))
    return (
        f"This file holds {len(runs)} recordings ({', '.join(named)}); rec.env is "
        f"the figure eight of the first, and {share:.0%} of the samples of the "
        "later ones lie outside it. A later maze can stand elsewhere in the room "
        "or be another shape, and no source read gives its layout: the positions "
        "are kept, and params['analysis_epochs'] holds the first alternation "
        "recording.",
    )


def _missing_rows(n_missing: int, n_trials: int, n_zero: int) -> tuple[str, ...]:
    """Table rows the epochs leave out, with their counts."""
    found = []
    if n_missing:
        found.append(
            f"{n_missing} of the file's {n_trials} trials have no start or stop "
            "time (NaN) and are left out of epochs['trials'] and the train and test "
            "epochs."
        )
    if n_zero:
        found.append(
            f"{n_zero} SleepStates rows start where they stop and are left out of "
            "the epochs, since such a row holds no time."
        )
    return tuple(found)


def _arm_conversion(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    bar: tuple[NDArray[np.float64], NDArray[np.float64]],
    trials: NDArray[np.float64],
    arms: NDArray[Any],
) -> Conversion:
    """Why the trials' ``visited_arm`` names a side of the centre bar."""
    bar_0, bar_1 = bar
    along = (bar_1 - bar_0) / np.linalg.norm(bar_1 - bar_0)
    sides = []
    for start, stop in trials:
        relative = positions[(times >= start) & (times < stop)] - bar_0
        sides.append(
            float(np.nanmedian(along[0] * relative[:, 1] - along[1] * relative[:, 0]))
        )
    side = np.sign(sides)
    agree = [
        float(max(np.mean(side[arms == arm] > 0), np.mean(side[arms == arm] < 0)))
        for arm in np.unique(arms)
    ]
    return Conversion(
        "trials: the file's table; train and test alternate within visited_arm "
        f"({', '.join(f'{arm} {int(np.sum(arms == arm))}' for arm in np.unique(arms))})",
        "stated",
        "the file's table, 'experimental trials'; each label's trials lie on one "
        "side of the centre bar by their median offset from it ("
        + ", ".join(f"{share:.0%}" for share in agree)
        + " of each label's trials on its side)",
    )
