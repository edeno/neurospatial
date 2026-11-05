"""Bhattarai, Lee & Jung (2020): rat CA1 in a spatial sequence memory task (Figshare 10032866).

What was recorded: three rats ran a loop with a figure 8-shaped section, one
of four spatial sequences (RR, LL, RL, LR) leading to one reward site, while
15 tetrodes recorded dorsal CA1. Each day the rats ran two consecutive
sessions, each of four blocks, one per correct sequence, in random order; a
block is 5 forced-choice trials followed by 15 free-choice trials, with a
3 s delay in each trial and 30 uL of water for a forced or correct trial.
The two sessions of a day are a *session set*, and the archive holds the 25
sets the paper analysed (three of the 28 recorded were excluded for
decoding errors above 20 cm). The head position comes from a red and a green
LED on the headstage. The paper's question is how reward and navigation
history change awake forward and reverse replay.

Where it lives: Figshare 10032866, version 2 (CC0), ``DataSet.zip``, 659 MB.
Each of ``spikes.mat``, ``VT.mat``, ``events.mat`` and ``beh.mat`` (MATLAB
v5, read with ``scipy.io.loadmat``) and ``lfp.mat`` (MATLAB v7.3, HDF5)
holds 25 cells, one per set, so every session of this dataset reads the same
members. The zip's ``MetaData.docx`` describes the columns.

Sources read: the paper (Bhattarai, Lee & Jung 2020, PNAS 117:689-697), its SI
Appendix (``pnas.1912533117.sapp.pdf``, Materials and Methods and Table S1),
``MetaData.docx``, the files.

Which cell is which set: the files name no rat or set. Table S1 lists the 25
sets by rat and set, and the cells are in its order: the share of rewarded
free-choice trials in each cell tracks the table's "Performance" row by row
(largest difference 1.8 points, in the one cell with an aborted block; the
others 0.8 at most). The session ids (``rat2_set1``) are the table's.

What the loader converts:

- positions: ``VT.mat`` columns 2 and 3 are camera pixels, which nothing read
  states. cm = pixels / 3.6, and the y axis is negated. Neither is stated.
  The scale comes from two lengths the paper gives: the linearized maze runs
  from 0 to 300 cm (Fig. 3A) and is 306 cm on the graph below at this scale,
  and a trial starts 30 cm from the nozzle (SI Appendix), where the head is a
  median 29 cm from where it was when the reward started. The sign comes from
  three agreements: read with y up, the stored path runs the opposite way
  round from the paper's own drawing (Fig. S1A), the "L" sequences go through
  the arms on the rat's right, and the stored head direction is a compass
  bearing only if y points down. Samples exactly at (0, 0) are kept as stored.
- headings: ``VT.mat`` column 4, degrees, is a clockwise bearing from the
  camera's up, so radians = pi/2 - degrees * pi/180 in the negated frame. The
  loader checks it against movement and reports the offset. A stored 0 means
  no head direction: 45% to 83% of a set's samples hold exactly 0, the next
  value held is 1 or 3 degrees, and they are NaN here.
- spike_times: the cells as stored, sorted. Five units of ``rat3_set3`` hold
  exact zeros between their spikes. They are kept, so that session fails the
  clock check and loads only with ``accept_failed_checks=("clock",)``.
- trials: ``events.mat`` rows are trials on the recording's clock.
  ``beh.mat``, on the task computer's clock, adds the block, sequence, choice
  and reward. Its rows match the events' at a fixed row shift (0, or 1 and 8
  in two sets), found by matching ``beh.mat``'s last sensor to the start of
  reward. ``MetaData.docx`` is one column off from column 7: column 7 is the
  session's running count of correct trials, 8 to 11 count the four error
  sequences, and 12 and 13 hold the forced trials and the delay.
- units: ``label`` applies the paper's rule, mean rate below 5 Hz and burst
  index above 40%, which gives the paper's 648 of 698 units as complex-spike
  cells. The rest are ``"theta"``.

Nothing in the files records the rats' rest: tracking starts at most 34 s
before the first trial and ends at most 101 s after the last, so there are no
sleep or rest epochs.

The environment is inferred. The paper's text and figures give the zones'
lengths and a schematic but not the rings' dimensions, so the graph's nodes
were read from the tracked samples of the 25 sets, which share one camera
view. It holds the return arm, the delay zone, the two rings with two arms
each, and the reward site. Bins are 2 cm along the track.

Analysis choices (``params``):

- ``bin_size`` 2 cm along the track and ``bandwidth`` 6 cm, the paper's rate
  maps (SI Appendix);
- ``min_speed`` 4 cm/s, the paper's cutoff for its rate maps;
- ``dt`` 0.02 s and ``time_step`` 0.01 s, the paper's replay decoding bins;
- ``analysis_epochs``, the first trial's start to the last trial's end;
- ``sequence_tracks``, each sequence's 1-D environment from :func:`sequence_track`,
  its linear path from the start of a trial to the reward site, the track the
  paper decodes each block on.
"""

from __future__ import annotations

import functools
import os
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._lfp import LFPSource
from neurospatial.datasets._recording import (
    Conversion,
    Recording,
    clip_epochs_to_recording,
    heading_vs_movement,
)
from neurospatial.datasets._tracks import nearest_segment, placement_tolerance

if TYPE_CHECKING:
    import networkx as nx

    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = [
    "EXPECTED_WATCH_OUTS",
    "REQUIRES",
    "WATCH_OUTS",
    "WORKED_RESULT",
    "load_session",
    "read_lfp",
    "sequence_track",
]

#: The paper's replay finding, and our attempt on the default session.
WORKED_RESULT: dict[str, Any] = {
    "status": "not possible yet",
    "level": None,
    "finding": (
        "about half of the candidate replay events in the reward zone are "
        "significant replays of the current trajectory"
    ),
    "quote": (
        '"We found 2,573 candidate replay events that are related to the current '
        'trajectory (correct path in the current block)" (Results, "Replay '
        'Events"); "An event was considered as a significant replay if P < 0.05 '
        '(n = 1342; 52.2% of candidate replay events" (SI Appendix, "Replay '
        'events"); Table S1 lists each set, 278 candidates and 153 replays for '
        "rat 2, set 2"
    ),
    "source": (
        "Bhattarai, Lee & Jung 2020, PNAS 117:689-697, Results, and its SI "
        'Appendix, Materials and Methods ("SWR events", "Replay events") and '
        "Table S1"
    ),
    "paper": 0.522,
    "ours": None,
    "criterion": None,
    "parameters": None,
    "unmatched": (
        "one LFP channel, where the paper averaged the power of two tetrodes",
        "the channel sits at its rails in 21% of samples on this set",
        "place cells are those detect_place_fields finds a field in, not the "
        "paper's three criteria; 550 over the 25 sets' session unions against "
        "Table S1's 497",
        "an SWR's five place cells are the block's, not any block's, since "
        "bhattarai_2020 takes one selection",
        "how an event ends is not stated: the last spike of the 300 ms window",
    ),
    "attempted": {
        "value": 0.243,
        "criterion": (
            "the share of candidates judged replay exceeds the 95th percentile of "
            "the same share with cell identities permuted within each block (20 "
            "draws, seed 2026: 2.7% to 27.0%, 95th percentile 27.0%); fixed before "
            "any null ran, and not met"
        ),
        "parameters": (
            "literature_methods.bhattarai_2020 at ripple_detection 0eddae64, the "
            "file's one channel in both LFP columns, power_measure "
            "'squared_signal', window_end_rule 'last_spike', each block's place "
            "cells, the reward period of its free-choice trials; rate maps of "
            "complex-spike units on sequence_track (binned, 2 cm, bandwidth 6 cm, "
            "above 4 cm/s, forced and correct free-choice trials); 20 ms windows "
            "in 10 ms steps; BHATTARAI_2020"
        ),
        "differs": (
            "9 of 37 candidates are judged replay against Table S1's 153 of 278. "
            "The loss is at the SWR gate: 793 threshold crossings on this channel "
            "and 24,028 over the 25 sets, against the paper's 132,665 SWRs. "
            "Without the gate the same bursts number 534 here (5,019 over the sets, "
            "2,573 in the paper), of which 96 are judged replay (18.0%, above that "
            "null's 95th percentile of 16.0%), and over the sets 783 (15.6%). "
            "Events last 232 ms against the paper's 152 ms. The null itself calls "
            "about 14% of events replay"
        ),
    },
    "why": (
        "the paper's candidates need an SWR detected on two tetrodes, and the "
        "archive holds one channel, at its rails for 5% to 56% of each set, whose "
        "ripple-band power rises little at population bursts. On it the paper's "
        "detector passes 187 candidates over the 25 sets, not 2,573, and 35 "
        "replays, not 1,342. Nothing was tuned to close this"
    ),
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES: tuple[str, ...] = ("fsspec",)

#: Camera pixels per cm, from the two lengths in the module's docstring.
PIXELS_PER_CM = 3.6
#: The paper's linearized maze, cm (the axes of Fig. 3A-E).
PAPER_MAZE_LENGTH = 300.0
#: How far from the nozzle a trial starts, cm (SI Appendix, "Behavior task").
TRIAL_START_DISTANCE = 30.0
BIN_SIZE = 2.0
BANDWIDTH = 6.0
MIN_SPEED = 4.0
DT = 0.02
TIME_STEP = 0.01
LFP_RATE = 2000.0
#: The paper's rule for complex-spike cells: below this mean rate, in Hz...
MAX_RATE = 5.0
#: ... and above this burst index, the percentage of inter-spike intervals
#: shorter than a quarter of the mean interval.
MIN_BURST_INDEX = 40.0
#: Above this speed, cm/s, heading is compared with the direction of movement.
RUNNING = 20.0
#: Smoothing of positions before that comparison, in samples: tracking is
#: quantized to pixels, so a step of one frame has only a few directions.
HEADING_SMOOTHING = 3
#: A heading in the positions' frame lies within this of movement while
#: running, and spreads less than this; radians (20 and 30 degrees).
MAX_HEADING_OFFSET = np.deg2rad(20.0)
MAX_HEADING_SPREAD = np.deg2rad(30.0)
#: The largest distance between ``beh.mat``'s last sensor and the start of
#: reward that counts as one trial, s. Most agree within 25 ms; about 1 in 50
#: differs by 1 to 1.3 s.
MAX_SENSOR_DIFFERENCE = 2.0
#: The share of samples the placement tolerance leaves outside the track.
TOLERANCE_QUANTILE = 0.99
#: Samples farther than this from the track are off it, cm: 1.5 ring widths.
OFF_TRACK = 30.0

#: The four sequences, in the order of ``beh.mat``'s count columns and of the
#: values of its "correct choice" column.
SEQUENCES = ("RR", "LL", "RL", "LR")
_N_SETS_BY_RAT = {1: 6, 2: 9, 3: 10}

#: Waypoints of the maze in ``VT.mat`` pixels, read from the tracked samples of
#: all 25 sets. A trial runs along the return arm to the delay zone, through
#: the first ring (the higher x) on one arm, the crossing, the second ring on
#: one arm, and into the reward site.
_RETURN = (
    (225.0, 103.0),
    (290.0, 68.0),
    (340.0, 52.0),
    (450.0, 57.0),
    (600.0, 72.0),
    (645.0, 105.0),
    (668.0, 165.0),
    (650.0, 200.0),
    (585.0, 208.0),
)
#: From the reward site's stem to where a trial starts.
_EXIT = ((168.0, 168.0), (175.0, 145.0), (225.0, 103.0))
#: The first ring's arms from its fork to the crossing, by the rat's turn: in
#: pixels, the right arm has the lower y.
_FIRST_RING = {
    "L": (
        (585.0, 208.0),
        (550.0, 218.0),
        (520.0, 230.0),
        (490.0, 210.0),
        (440.0, 176.0),
    ),
    "R": (
        (585.0, 208.0),
        (560.0, 190.0),
        (545.0, 165.0),
        (512.0, 148.0),
        (475.0, 162.0),
        (440.0, 176.0),
    ),
}
#: The second ring's arms from the crossing to its fork.
_SECOND_RING = {
    "L": (
        (440.0, 176.0),
        (400.0, 190.0),
        (350.0, 202.0),
        (305.0, 206.0),
        (280.0, 185.0),
        (250.0, 162.0),
    ),
    "R": (
        (440.0, 176.0),
        (390.0, 155.0),
        (320.0, 130.0),
        (275.0, 148.0),
        (250.0, 162.0),
    ),
}
_ENTRY = ((250.0, 162.0), (215.0, 170.0), (168.0, 168.0), (178.0, 196.0))

_PAPER = "Bhattarai, Lee & Jung 2020"
_SI = f"the SI Appendix of {_PAPER}"
_METADATA = "the archive's MetaData.docx"
_FLIP = (
    "read with y up, the stored path runs the opposite way round from the "
    f"animal's path in Fig. S1A of {_PAPER} under any rotation; the blocks the "
    "file calls LL take the arms on the rat's right; and the stored head "
    "direction is a clockwise bearing from the camera's up only if y points "
    "down the image"
)

#: Findings computed from the data that each verified session must report.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "rat2_set2": (
        r"Head direction is missing at \d+\.\d% of samples",
        r"exactly at \(0, 0\)",
        r"runs of 11 or more identical position samples",
        r"lie outside the inferred environment",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "rat2_set2": (
        "21.4% of this set's LFP samples sit at the rails of +/-327.67 (5% to 56% "
        "over the 25 sets, the reason this set of the three with 40 or more units "
        "is the default). At the population rate's scale the channel marks events "
        "weakly: around peaks of its 100-250 Hz power the rate of all units rises "
        "to 1.52 times its level 120 to 200 ms away.",
    ),
}

_SCALE_NOT_SPEED = (
    "The cm scale is inferred from lengths, not speeds. The paper's mean speeds "
    "(track 15.8, return 17.5, delay 12.2, reward 7.3 cm/s) do not fit it: at "
    f"{PIXELS_PER_CM:g} pixels per cm the rat's head averages about 40 cm/s "
    "between the end of the delay and the reward and 5 cm/s in the reward "
    "period, so no one scale gives the paper's four speeds. The maze lengths "
    "would have to be about 2.5 times shorter to fit its running speeds, "
    "which would make the 713 pixels of x about 80 cm. x and y share one "
    "scale; nothing read says the pixels are square."
)
_ONE_TETRODE = (
    "lfp.mat holds one channel per set. The paper's sharp-wave ripple detection "
    "averaged the power of one channel each from two tetrodes, and its theta "
    "power came from a tetrode above the cell layer; which of these this "
    "channel is, the files do not say. A ripple analysis here has one channel, "
    "not the paper's two."
)
_NO_REST = (
    "The files hold no rest or sleep: tracking, spikes and LFP start at most "
    "34 s before the first trial and end at most 101 s after the last, and the "
    "two sessions of a set follow each other in the trial table. So nothing "
    "here can serve as a pre-task baseline."
)
_RAILS = (
    "lfp.mat's values stop at +/-327.67, in 0.01 steps, with no unit stated: 5% "
    "to 56% of a set's samples sit at those rails. On a rail the 100-250 Hz band "
    "is flat, and each step off one adds broadband power, so ripple power is "
    "missing there and inflated at the edges. A threshold on the session's mean "
    "and SD sees both."
)
_UNITS = (
    "Each set's units are all those recorded that day, not its place cells. "
    "units['label'] applies the paper's rule for complex-spike cells; the paper "
    "chose place cells per block and session, which the files do not record."
)
_METADATA_COLUMNS = (
    "MetaData.docx is one column off in beh.mat from column 7: column 7 is the "
    "session's count of correct trials, 8 to 11 count the error sequences, 12 "
    "and 13 hold the forced trials and the delay. The trials table reads them "
    "that way. Columns 16 to 22 are the seven sensors on the task computer's "
    "clock, in ms; a sensor the trial did not cross keeps its earlier value."
)


class _Session(NamedTuple):
    """One set's cells of the four small ``.mat`` members.

    Attributes
    ----------
    tracking : ndarray, shape (n_time, 4)
        ``VT.mat``: time (s), x, y (pixels), head direction (degrees).
    trains : tuple of ndarray
        ``spikes.mat``: one array of times per unit, as stored.
    events : ndarray, shape (n_trials, 4)
        ``events.mat``: trial start, end of delay, start of reward, trial end.
    behavior : ndarray, shape (n_rows, 23)
        ``beh.mat``, as float64.
    """

    tracking: NDArray[np.float64]
    trains: tuple[NDArray[np.float64], ...]
    events: NDArray[np.float64]
    behavior: NDArray[np.float64]


def set_index(session: str) -> int:
    """The 0-based cell of a session id, in the order of Table S1.

    Parameters
    ----------
    session : str
        ``rat<r>_set<s>``.

    Returns
    -------
    int

    Examples
    --------
    >>> set_index("rat1_set1"), set_index("rat2_set1"), set_index("rat3_set10")
    (0, 6, 24)
    """
    rat, number = session.removeprefix("rat").split("_set")
    return sum(n for r, n in _N_SETS_BY_RAT.items() if r < int(rat)) + int(number) - 1


def _cm(points: Any) -> NDArray[np.float64]:
    """Pixels to the positions' frame: cm, y negated.

    Parameters
    ----------
    points : array_like, shape (..., 2)
    """
    cm: NDArray[np.float64] = (
        np.asarray(points, dtype=np.float64) * np.array([1.0, -1.0]) / PIXELS_PER_CM
    )
    return cm


def read_session(folder: Path, index: int) -> _Session:
    """One set's cells of the four small members.

    Parameters
    ----------
    folder : Path
        The directory holding ``VT.mat``, ``spikes.mat``, ``events.mat`` and
        ``beh.mat``.
    index : int
        The 0-based cell.

    Returns
    -------
    _Session
    """
    from scipy.io import loadmat

    def cell(name: str, key: str) -> Any:
        # MATLAB cells are object arrays of numeric arrays; loadmat makes no
        # objects from them and runs nothing.
        return loadmat(folder / name, variable_names=[key])[key].ravel()[index]

    return _Session(
        tracking=np.asarray(cell("VT.mat", "VT"), dtype=np.float64),
        trains=tuple(
            np.asarray(train, dtype=np.float64).ravel()
            for train in cell("spikes.mat", "spikes").ravel()
        ),
        events=np.asarray(cell("events.mat", "events"), dtype=np.float64),
        behavior=np.asarray(cell("beh.mat", "beh"), dtype=np.float64),
    )


def _graph() -> nx.Graph:
    """The maze: integer nodes with ``pos`` in cm, edge ``distance`` in cm."""
    import networkx as nx

    paths = (
        _RETURN,
        _EXIT,
        _ENTRY,
        *_FIRST_RING.values(),
        *_SECOND_RING.values(),
    )
    nodes: dict[tuple[float, float], int] = {}
    graph = nx.Graph()
    for path in paths:
        for a, b in pairwise(path):
            for point in (a, b):
                if point not in nodes:
                    nodes[point] = len(nodes)
                    graph.add_node(nodes[point], pos=tuple(_cm(point).tolist()))
            length = np.linalg.norm(_cm(a) - _cm(b))
            graph.add_edge(nodes[a], nodes[b], distance=float(length))
    return graph


def sequence_path(sequence: str) -> NDArray[np.float64]:
    """The waypoints of one sequence, from the start of a trial to the reward.

    Parameters
    ----------
    sequence : {"RR", "LL", "RL", "LR"}

    Returns
    -------
    ndarray, shape (n_points, 2)
        In cm, in the positions' frame.

    Raises
    ------
    KeyError
        If ``sequence`` is not one of the four.
    """
    return _cm(
        [
            *_RETURN[:-1],
            *_FIRST_RING[sequence[0]],
            *_SECOND_RING[sequence[1]][1:],
            *_ENTRY[1:],
        ]
    )


def sequence_track(sequence: str, *, bin_size: float = BIN_SIZE) -> Environment:
    """A 1-D environment along one sequence's path, the track a block is decoded on.

    The paper projects each position onto a linear path from the start of a
    trial to the reward; ``env.to_linear`` does that here. Distance along the
    track runs from the start of the trial, 0, to the reward site. Samples up
    to ``OFF_TRACK`` from the path are placed on it, so a head on the other
    arm of a ring projects onto this sequence's arm, as in the paper.

    Parameters
    ----------
    sequence : {"RR", "LL", "RL", "LR"}
    bin_size : float, default 2.0
        Bin length, cm; the paper's rate maps use 2 cm.

    Returns
    -------
    Environment

    Raises
    ------
    KeyError
        If ``sequence`` is not one of the four.

    Examples
    --------
    >>> track = sequence_track("RR")  # doctest: +SKIP
    >>> track.n_bins  # doctest: +SKIP
    154
    """
    import networkx as nx

    from neurospatial import Environment

    points = sequence_path(sequence)
    graph = nx.Graph()
    for node, point in enumerate(points):
        graph.add_node(node, pos=tuple(point.tolist()))
    edges = [(node, node + 1) for node in range(len(points) - 1)]
    for a, b in edges:
        graph.add_edge(a, b, distance=float(np.linalg.norm(points[a] - points[b])))
    return Environment.from_graph(
        graph,
        edges,
        bin_size=bin_size,
        edge_spacing=0.0,
        units="cm",
        name=f"bhattarai_2020 {sequence}",
        track_width=2 * OFF_TRACK,
    )


def _row_shift(events: NDArray[np.float64], behavior: NDArray[np.float64]) -> int:
    """How many ``beh.mat`` rows come before the first ``events.mat`` row.

    The two tables are on different clocks and neither numbers its rows
    against the other. ``beh.mat``'s last sensor is the start of reward, so
    at the right shift, the differences between the two are one constant,
    the difference of the clocks.

    Raises
    ------
    ValueError
        If no shift makes most rows agree.
    """
    reward = events[:, 2]
    sensor = behavior[:, 21] / 1e3
    best, agreeing = 0, -1
    for shift in range(behavior.shape[0] - events.shape[0] + 2):
        n = min(reward.size, sensor.size - shift)
        difference = reward[:n] - sensor[shift : shift + n]
        count = int(
            np.sum(np.abs(difference - np.median(difference)) < MAX_SENSOR_DIFFERENCE)
        )
        if count > agreeing:
            best, agreeing = shift, count
    if agreeing < 0.9 * reward.size:
        msg = (
            f"beh.mat and events.mat agree on only {agreeing} of {reward.size} "
            "trials at any row shift, so trials cannot be labelled by block"
        )
        raise ValueError(msg)
    return best


def _trial_labels(
    behavior: NDArray[np.float64], shift: int, n_trials: int
) -> dict[str, NDArray[Any]]:
    """Session, block, sequence, choice and reward of each ``events.mat`` row.

    A new block starts where the trial-in-block column falls, and a new session
    where the session's trial count falls; a set's last two sessions are 1 and
    2, so an aborted start before them (rat3_set7) labels no trial.

    Parameters
    ----------
    behavior : ndarray, shape (n_rows, 23)
    shift : int
        ``beh.mat`` rows before the first ``events.mat`` row.
    n_trials : int

    Returns
    -------
    dict of str to ndarray, each shape (n_trials,)
    """
    in_block = behavior[:, 0]
    count = behavior[:, 1]
    new_block = np.r_[True, np.diff(in_block) < 0]
    new_session = np.r_[True, np.diff(count) < 0]
    session = np.cumsum(new_session)
    session = session - session[-1] + 2
    block_ids = np.cumsum(new_block)
    first_block = np.array(
        [block_ids[session == value].min() for value in session], dtype=int
    )
    block = block_ids - first_block + 1
    sequence = np.array(SEQUENCES, dtype=object)[behavior[:, 14].astype(int)]
    correct = np.diff(behavior[:, 2:6], axis=0, prepend=np.zeros((1, 4)))
    errors = np.diff(behavior[:, 7:11], axis=0, prepend=np.zeros((1, 4)))
    # The counters restart with a session.
    correct[new_session] = behavior[new_session, 2:6]
    errors[new_session] = behavior[new_session, 7:11]
    choice = sequence.copy()
    wrong = errors.sum(axis=1) == 1
    choice[wrong] = np.array(SEQUENCES, dtype=object)[errors[wrong].argmax(axis=1)]
    rows = np.arange(n_trials) + shift
    rows = rows[rows < behavior.shape[0]]
    pad = n_trials - rows.size

    def labelled(values: NDArray[Any], fill: Any) -> NDArray[Any]:
        return np.concatenate([values[rows], np.full(pad, fill, dtype=values.dtype)])

    return {
        "session": labelled(session.astype(int), -1),
        "block": labelled(block.astype(int), -1),
        "trial_in_block": labelled(in_block.astype(int), -1),
        "sequence": labelled(sequence, ""),
        "choice": labelled(choice, ""),
        "forced": labelled(in_block <= behavior[:, 11], False),
        "rewarded": labelled(behavior[:, 22] == 1, False),
    }


def _burst_index(train: NDArray[np.float64]) -> float:
    """The percentage of inter-spike intervals under a quarter of the mean."""
    intervals = np.diff(train)
    if intervals.size == 0:
        return 0.0
    return float(100.0 * np.mean(intervals < intervals.mean() / 4.0))


def _labels(
    trains: tuple[NDArray[np.float64], ...], duration: float
) -> NDArray[np.object_]:
    """``"complex-spike"`` or ``"theta"``, the paper's rule, for each unit."""
    labels = [
        "complex-spike"
        if train.size / duration < MAX_RATE and _burst_index(train) > MIN_BURST_INDEX
        else "theta"
        for train in trains
    ]
    return np.array(labels, dtype=object)


def _declared_track(
    positions: NDArray[np.float64], session: str
) -> tuple[Environment, Conversion, float | None, NDArray[np.float64]]:
    """The maze as an environment, with a placement tolerance from the samples.

    Returns
    -------
    env : Environment
    placement : Conversion
    width_quantile : float or None
    distance : ndarray, shape (n_on_camera,)
        Each sample's distance from the track, cm, without the (0, 0) samples.
    """
    from neurospatial import Environment
    from neurospatial.regions import Region

    graph = _graph()
    starts = np.array([graph.nodes[a]["pos"] for a, _ in graph.edges])
    ends = np.array([graph.nodes[b]["pos"] for _, b in graph.edges])
    on_camera = np.any(positions != 0, axis=1)
    distance = nearest_segment(positions[on_camera], starts, ends).distance
    width, width_quantile, evidence = placement_tolerance(
        distance,
        TOLERANCE_QUANTILE,
        floor=BIN_SIZE,
        measured="the tracked head's distance from the graph, without the samples "
        "at (0, 0)",
        reason="The head swings at the reward site and in the delay zone and the "
        "tracking leaves the maze at times, so the tolerance is taken from the "
        "samples; no track width is stated.",
    )
    reward = _cm(_ENTRY[-1])
    delay = _cm(_RETURN[6])
    order = list(graph.edges)
    env = Environment.from_graph(
        graph,
        order,
        bin_size=BIN_SIZE,
        units="cm",
        name=f"bhattarai_2020 {session}",
        track_width=width,
        regions={
            "reward_site": Region("reward_site", "point", reward),
            "delay_zone": Region("delay_zone", "point", delay),
        },
    )
    placement = Conversion(
        f"environment: track_width {width:.3g} cm, a placement tolerance",
        "inferred",
        evidence,
    )
    return env, placement, width_quantile, distance


def _heading_evidence(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
) -> tuple[float, float, int]:
    """Heading against movement on smoothed positions, tracked samples only.

    Returns
    -------
    offset, spread : float
        Radians, as :func:`heading_vs_movement`.
    n_compared : int
        The samples compared.
    """
    from scipy.ndimage import gaussian_filter1d

    on_camera = np.any(positions != 0, axis=1)
    smooth = gaussian_filter1d(positions[on_camera], HEADING_SMOOTHING, axis=0)
    offset, spread = heading_vs_movement(
        times[on_camera], smooth, headings[on_camera], min_speed=RUNNING
    )
    step = np.diff(smooth, axis=0)
    speed = np.linalg.norm(step, axis=1) / np.diff(times[on_camera])
    compared = (speed > RUNNING) & np.isfinite(headings[on_camera][:-1])
    return offset, spread, int(compared.sum())


def _heading_watch_out(offset: float, spread: float, n_missing: float) -> str:
    """How the headings compare with movement, and how many are missing."""
    verified = abs(offset) <= MAX_HEADING_OFFSET and spread <= MAX_HEADING_SPREAD
    frame = (
        "so the frame is verified by movement"
        if verified
        else "so the frame is not verified by movement"
    )
    return (
        f"Head direction is missing at {n_missing:.1%} of samples (a stored 0, "
        f"which the loader reads as no direction). Elsewhere, heading minus the "
        f"direction of movement above {RUNNING:g} cm/s has a circular mean of "
        f"{np.degrees(offset):+.1f} deg and a median spread of "
        f"{np.degrees(spread):.1f} deg, {frame}."
    )


def _off_track(distance: NDArray[np.float64]) -> str:
    """How far from the maze the tracking goes."""
    return (
        f"{np.mean(distance > OFF_TRACK):.2%} of samples away from (0, 0) lie more "
        f"than {OFF_TRACK:g} cm from the track, up to {distance.max():.0f} cm: a "
        "head LED on no part of the maze, or its reflection. They are kept."
    )


def _zero_spikes(trains: tuple[NDArray[np.float64], ...]) -> tuple[str, ...]:
    """A watch-out for spike times of exactly 0, which the clock check refuses."""
    counts = {
        unit: int(np.sum(train == 0.0))
        for unit, train in enumerate(trains)
        if np.any(train == 0.0)
    }
    if not counts:
        return ()
    listed = ", ".join(f"unit {unit}: {n}" for unit, n in counts.items())
    return (
        f"spikes.mat holds exact zeros among the spike times of {len(counts)} "
        f"units ({listed}), far from the recording. They are kept as stored, "
        "sorted to the start of each train, so this session fails the clock check: "
        'load it with accept_failed_checks=("clock",) and leave them out with '
        "train[train > 0].",
    )


def _lost_boundaries(end_known: NDArray[np.bool_]) -> tuple[str, ...]:
    """A watch-out for the trials whose start events.mat lost."""
    n_lost = int(np.sum(~end_known))
    if n_lost == 0:
        return ()
    return (
        f"events.mat lost the start of {n_lost} of {end_known.size} trials: those "
        "rows start after their own end of delay. Each such trial starts at its "
        "end of delay here, the trial before ends there "
        "(rec.trials.end_known is False for it), and that trial is left out of "
        "epochs['reward'], since its true end is not known.",
    )


def _lost_boundary_conversion(end_known: NDArray[np.bool_]) -> tuple[Conversion, ...]:
    """The Conversion of the trial boundaries events.mat lost, where it lost any."""
    n_lost = int(np.sum(~end_known))
    if n_lost == 0:
        return ()
    return (
        Conversion(
            f"trials: {n_lost} starts events.mat lost set to the trial's end of "
            "delay, and the previous trial's end to match",
            "inferred",
            "those rows start after their own end of delay, at their own end, and "
            "the row before ends at the same time, while ends of delay and starts "
            "of reward stay in order",
        ),
    )


def _unmatched(
    behavior: NDArray[np.float64], events: NDArray[np.float64], shift: int
) -> tuple[str, ...]:
    """A watch-out when ``beh.mat`` and ``events.mat`` differ by more than a row."""
    extra = behavior.shape[0] - events.shape[0]
    if shift == 0 and extra <= 1:
        return ()
    return (
        f"beh.mat has {behavior.shape[0]} rows and events.mat "
        f"{events.shape[0]}: the events start at beh.mat's row {shift + 1}, so "
        f"{extra} rows of beh.mat have no trial here. Trials per block can differ "
        "from 20.",
    )


def _lfp_source(entry: SessionEntry, index: int, folder: Path) -> LFPSource:
    """The set's cell of ``lfp.mat``, one channel, one tetrode."""
    return LFPSource(
        dataset=entry.info.dataset,
        files=entry.lfp_files,
        sampling_rate=LFP_RATE,
        channels={
            "tetrode": "the one channel lfp.mat holds for this set, from one CA1 "
            "tetrode; the file does not say which",
        },
        unit="arbitrary",
        unit_basis=Conversion(
            "LFP: lfp.mat column 2 as stored; no unit is stated",
            "stated",
            f"{_METADATA}: 'the local field potential signal from a single tetrode'",
        ),
        times_basis=Conversion(
            "LFP times: lfp.mat column 1 as stored, s",
            "stated",
            f"{_METADATA}: 'Column 1 is the time vector (s)'",
        ),
        reader=functools.partial(read_lfp, index=index),
        ripple_channels=("tetrode",),
        cache=folder.parents[1],
    )


def read_lfp(
    sources: tuple[str | Path, ...], channels: tuple[str, ...], *, index: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read one set's cell of ``lfp.mat``, a MATLAB v7.3 (HDF5) file.

    Parameters
    ----------
    sources : tuple of path
        ``lfp.mat``.
    channels : tuple of str
        ``("tetrode",)``, the one channel.
    index : int
        The 0-based cell.

    Returns
    -------
    samples : ndarray, shape (n_time, 1)
    times : ndarray, shape (n_time,)
        Seconds.
    """
    import h5py

    with h5py.File(sources[0], "r") as file:
        # A cell array is a dataset of references; MATLAB's columns are rows.
        cell = file[file["lfp"][index, 0]]
        times, samples = np.asarray(cell, dtype=np.float64)
    return np.repeat(samples[:, None], len(channels), axis=1), times


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one set's cells into a :class:`Recording`.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The directory holding the fetched zip members.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.

    Raises
    ------
    ValueError
        If ``beh.mat`` and ``events.mat`` cannot be matched row by row.
    """
    import pandas as pd

    from neurospatial import Intervals

    folder = Path(path_or_stream)
    index = set_index(entry.session)
    raw = read_session(folder, index)
    times = raw.tracking[:, 0]
    positions = _cm(raw.tracking[:, 1:3])
    stored_heading = raw.tracking[:, 3]
    missing = stored_heading == 0
    headings = np.where(missing, np.nan, np.pi / 2 - np.deg2rad(stored_heading))
    from neurospatial._angles import wrap_angle

    headings = np.asarray(wrap_angle(headings), dtype=np.float64)
    offset, spread, n_compared = _heading_evidence(times, positions, headings)
    spike_times = tuple(np.sort(train) for train in raw.trains)
    n_unsorted = sum(
        not np.array_equal(a, b) for a, b in zip(raw.trains, spike_times, strict=True)
    )
    shift = _row_shift(raw.events, raw.behavior)
    labels = _trial_labels(raw.behavior, shift, raw.events.shape[0])
    starts, delay_ends, reward_starts, stops, known_end = _trial_bounds(raw.events)
    env, placement, width_quantile, distance = _declared_track(positions, entry.session)
    duration = float(times[-1] - times[0])
    unit_labels = _labels(spike_times, duration)
    table = pd.DataFrame(
        {"brain_region": "dorsal CA1", "label": unit_labels},
        index=pd.Index(np.arange(len(spike_times)), name="unit_id"),
    )
    forced = labels["forced"]
    rows = np.column_stack([starts, stops])
    epochs = {
        "session_1": _span(rows, labels["session"] == 1),
        "session_2": _span(rows, labels["session"] == 2),
        "forced_choice": rows[forced],
        "free_choice": rows[~forced],
        "reward": _reward_rows(reward_starts, stops, known_end),
    }
    clipped = clip_epochs_to_recording(epochs, times, spike_times, source="events.mat")
    task = np.array([[starts.min(), stops.max()]])
    clipped_task = clip_epochs_to_recording(
        {"task": task}, times, spike_times, source="events.mat"
    )
    lfp = _lfp_source(entry, index, folder)
    rat, _ = entry.session.split("_")
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"rat {rat.removeprefix('rat')} (male Sprague Dawley)",
        apparatus=entry.info.apparatus,
        brain_regions=entry.info.brain_regions,
        clock=(
            "the acquisition system's clock, seconds, as stored; tracking starts "
            f"at {times[0]:.1f} s"
        ),
        times=times,
        spike_times=spike_times,
        units=table,
        env=env,
        env_kind="inferred",
        track_width_quantile=width_quantile,
        env_source=(
            "a track graph read from the tracked samples of the 25 sets: the "
            "return arm, the delay zone, two rings of two arms each and the "
            f"reward site, {BIN_SIZE:g} cm bins along the track; {_PAPER} state "
            f"no ring dimensions (a linearized maze of {PAPER_MAZE_LENGTH:g} cm "
            f"in Fig. 3, and a trial starting {TRIAL_START_DISTANCE:g} cm from "
            "the nozzle)"
        ),
        positions=positions,
        position_units="cm",
        tracking=(
            f"the head: a set of red and green LEDs on the headstage, 30 Hz ({_SI})"
        ),
        headings=headings,
        heading_source=(
            "VT.mat's head direction, a clockwise bearing from the camera's up "
            "in degrees, converted to the positions' frame"
        ),
        epochs=clipped.epochs,
        events={"delay_end": delay_ends, "reward_start": reward_starts},
        trials=Intervals.from_arrays(
            starts,
            stops,
            delay_end=delay_ends,
            reward_start=reward_starts,
            end_known=known_end,
            **labels,
        ),
        params={
            "bin_size": BIN_SIZE,
            "bandwidth": BANDWIDTH,
            "min_speed": MIN_SPEED,
            "dt": DT,
            "time_step": TIME_STEP,
            "analysis_epochs": clipped_task.epochs["task"],
            "sequence_tracks": {name: sequence_track(name) for name in SEQUENCES},
        },
        conversions=(
            Conversion(
                f"positions: cm = (x, -y) in VT.mat pixels / {PIXELS_PER_CM:g}",
                "inferred",
                f"no source read states the unit or the scale. {_scale_evidence(raw)} "
                f"The y axis is negated because {_FLIP}",
            ),
            Conversion(
                "headings: pi/2 - radians(VT.mat column 4), in the positions' frame",
                "inferred",
                f"{_METADATA} states degrees and no frame. Against movement above "
                f"{RUNNING:g} cm/s, on {n_compared} samples, the converted heading "
                f"has a circular mean difference of {np.degrees(offset):+.1f} deg "
                f"and a median spread of {np.degrees(spread):.1f} deg",
            ),
            Conversion(
                f"headings: NaN at the {int(missing.sum())} samples holding exactly 0",
                "inferred",
                f"no source read says what a 0 means; {missing.mean():.1%} of samples "
                "hold exactly 0, which no direction would, and their neighbouring "
                f"values, 1 to 3 degrees, hold {np.mean(np.isin(stored_heading, (1, 2, 3))):.2%}",
            ),
            Conversion(
                f"spike_times: spikes.mat's {len(spike_times)} structures as stored, "
                f"s, sorted ({n_unsorted} were not)",
                "stated",
                f"{_METADATA}: 'time stamps (s) from individual neurons'",
            ),
            Conversion(
                "units['label']: 'complex-spike' for a mean rate below "
                f"{MAX_RATE:g} Hz over the tracked span and a burst index above "
                f"{MIN_BURST_INDEX:g}%, else 'theta'",
                "stated",
                f"{_SI}, and the rule gives the paper's 648 of 698 units over the "
                "archive",
            ),
            Conversion(
                f"trials: events.mat rows with beh.mat's rows from row {shift + 1}",
                "inferred",
                "the two tables are on different clocks and neither states the "
                "match: at that shift the start of reward equals beh.mat's last "
                "sensor plus one constant, within 25 ms in most trials",
            ),
            Conversion(
                "trials: rat and set from the order of the cells",
                "inferred",
                "the files name neither; the share of rewarded free-choice trials "
                "follows Table S1's Performance column in order, row by row",
            ),
            *_lost_boundary_conversion(known_end),
            placement,
            lfp.unit_basis,
            lfp.times_basis,
            *(() if clipped.conversion is None else (clipped.conversion,)),
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            _heading_watch_out(offset, spread, float(missing.mean())),
            _off_track(distance),
            *_zero_spikes(raw.trains),
            *_unmatched(raw.behavior, raw.events, shift),
            *_lost_boundaries(known_end),
            *(() if clipped.watch_out is None else (clipped.watch_out,)),
            _SCALE_NOT_SPEED,
            _ONE_TETRODE,
            _RAILS,
            _NO_REST,
            _UNITS,
            _METADATA_COLUMNS,
        ),
        lfp=lfp,
    )


def _span(rows: NDArray[np.float64], chosen: NDArray[np.bool_]) -> NDArray[np.float64]:
    """The first chosen trial's start to the last one's end, shape (1, 2)."""
    return np.array([[rows[chosen, 0].min(), rows[chosen, 1].max()]])


def _trial_bounds(
    events: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
]:
    """Each trial's stored times, with the boundaries the file lost.

    In seven sets, some rows start after their own end of delay, at their own
    end: the boundary between that trial and the one before was not found,
    and the row before ends at the same time. The ends of delay and starts of
    reward stay in order, so such a trial starts at its end of delay, the
    earliest time the file gives for it, and the trial before ends there.

    Parameters
    ----------
    events : ndarray, shape (n_trials, 4)

    Returns
    -------
    starts, delay_ends, reward_starts, stops : ndarray, shape (n_trials,)
    end_known : ndarray of bool, shape (n_trials,)
        False for a trial whose stored end was replaced.
    """
    starts, delay_ends, reward_starts, stops = events.T
    lost = starts >= delay_ends
    end_known = ~np.r_[lost[1:], False]
    return (
        np.where(lost, delay_ends, starts),
        delay_ends,
        reward_starts,
        np.where(end_known, stops, np.r_[delay_ends[1:], stops[-1]]),
        end_known,
    )


def _reward_rows(
    reward_starts: NDArray[np.float64],
    stops: NDArray[np.float64],
    end_known: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """From the start of reward to the trial's stored end, positive lengths only."""
    rows = np.column_stack([reward_starts, stops])
    kept: NDArray[np.float64] = rows[end_known & (stops > reward_starts)]
    return kept


def _scale_evidence(raw: _Session) -> str:
    """This set's two lengths, in pixels, against the paper's in cm.

    The head's distance between the start of reward and the next trial's
    start, against the 30 cm from the nozzle at which a trial starts; and the
    path of a trial on the maze graph, against the 300 cm of the linearized
    maze.
    """
    on_camera = np.any(raw.tracking[:, 1:3] != 0, axis=1)
    samples = raw.tracking[on_camera]

    def at(moments: NDArray[np.float64]) -> NDArray[np.float64]:
        rows = np.clip(np.searchsorted(samples[:, 0], moments), 0, len(samples) - 1)
        return samples[rows, 1:3]

    reward = at(raw.events[:-1, 2])
    start = at(raw.events[1:, 0])
    exits = float(np.median(np.linalg.norm(reward - start, axis=1)))
    lengths = [
        float(np.sum(np.linalg.norm(np.diff(sequence_path(name), axis=0), axis=1)))
        for name in SEQUENCES
    ]
    return (
        f"The head moves a median {exits:.0f} pixels from the start of reward to "
        f"the next trial's start, which {_SI} puts {TRIAL_START_DISTANCE:g} cm from "
        f"the nozzle ({exits / TRIAL_START_DISTANCE:.2f} pixels per cm); a trial's "
        f"path on the maze graph is {min(lengths):.0f} to {max(lengths):.0f} cm at "
        f"this scale against the {PAPER_MAZE_LENGTH:g} cm linearized maze of Fig. "
        f"3A of {_PAPER}; and the arms of a ring lie about 10 cm either side of "
        "their middle, as in Fig. S1B."
    )
