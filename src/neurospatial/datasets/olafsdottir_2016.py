"""Ólafsdóttir, Carpenter & Barry (2016): CA1 and deep MEC on a Z-track, then rest (Zenodo 5566548).

What was recorded: tetrodes in dorsal CA1 and in layers V/VI of the medial
entorhinal cortex of six male Lister Hooded rats, on the Axona dacqUSB system.
On a recording day a rat shuttled back and forth on an elevated Z-shaped track
(two parallel 190 cm runways joined by a 220 cm diagonal, 10 cm wide) for 20
laps, rested for about an hour and a half in a cylindrical enclosure (18 cm
across, 61 cm high) outside the track's curtains, and then foraged in a 1 m
open field, where the MEC cells were classified. The paper asks whether grid
cells replay with place cells during that rest; its replay events are all from
the rest enclosure.

Where it lives: Zenodo 5566548 (CC-BY-4.0), one stored (uncompressed) zip of
9.91 GB, a folder per rat and day. The paper itself says its data are
available on request; the deposit, titled for it, came in 2021. A load reads
the members it needs by HTTP byte range: each trial's tetrode files, Tint
cuts, ``.pos`` and ``.set`` (0.3 to 1.2 GB a day); the ``.egf`` LFP comes
with ``rec.lfp.fetch``.

Sources read: the paper and its Online Methods (Nature Neuroscience
19:792-794), its Supplementary Information (Supplementary Table 1) and
Supplementary Software (``decode_calcPosterior.m``, ``lineTraj_decode.m``,
``lineTraj_get_res.m``, ``lineTraj_reget_lnBnd.m``), Ólafsdóttir, Carpenter &
Barry 2017 (Neuron 96:925-935, the same rats and track), the deposit's
``README.txt`` and record description, the files, and the format sources
:mod:`neurospatial.datasets._axona` names.

Sessions: one per recording day, ``<rat>-<date>`` from the deposit's folder,
with every trial of that day on one clock. The track trial (``track1``) holds
the positions; the rest (``sleepPOST``) and open-field (``screening``) trials
are untracked epochs here, their positions in ``params`` (see below), since
they are in other frames.

What the loader converts:

- clock: the track trial's, from its first sample. The rest and open-field
  trials start at their ``.set`` ``trial_time`` minus the track's (1 s
  resolution, inferred; each file's own clock starts at 0).
- spike times: each tetrode file's 4-byte timestamps / its ``timebase``
  (96 kHz), plus the trial's start.
- units: every cluster from 1 up in a trial's Tint ``.cut`` (cluster 0 is
  Tint's unassigned spikes), named ``T<tetrode>_c<cluster>``, with the same
  name across the day's trials, as the paper's decoding of rest spikes with
  track rate maps implies. The watch-outs give how often a rest cluster's mean
  waveform is nearest the same cluster on the track. ``.clu`` files are not
  read: the README says the ``.cut`` files supersede them.
- brain regions: tetrodes 1 to 8 in MEC and 9 to 16 in CA1, the other way
  round for R2142 (the README).
- marks: every detected spike of every tetrode, sorted or not, with its peak
  on each channel in microvolts: the stored sample's maximum over the 50
  samples x ``1000 ADC_fullscale_mv / (gain_ch 128)`` from that trial's
  ``.set``. ``rec.marks.waveforms("T10")`` reads the waveforms.
- positions: the big LED (the ``.pos`` file's first spot), pixels x 100 /
  ``pixels_per_metre``, in cm on the camera's axes: y runs down the image. 1023,
  the format's mark for a spot not found, is NaN. Frame ``k`` (from 1) is at
  ``k / 50`` s.
- headings: none. The little LED is found in few track frames, and where both
  are, the direction from it to the big one does not follow movement (the
  watch-outs give the numbers). ``params["little_led_positions"]`` keeps the
  little LED's positions, in cm, for a reader who wants to try.
- ``params["rest_times"]``, ``["rest_positions"]`` (and ``open_field_...``):
  that trial's frames and big-LED positions, cm in its own tracking window,
  which is not the track's frame.

The environment is a Z of three straight pieces fitted to the track trial's
running samples (inferred; the paper states 190, 220 and 190 cm, and at the
header's pixels_per_metre the fit is about 12% longer), in 2 cm bins (the
paper's), its first node the end where the session started.
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Mapping
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._axona import (
    UNTRACKED,
    PositionTrack,
    microvolts_per_unit,
    read_continuous,
    read_cut,
    read_position,
    read_set,
    read_tetrode,
    tetrode_channels,
)
from neurospatial.datasets._lfp import LFPSource
from neurospatial.datasets._marks import UNSORTED, SpikeMarks
from neurospatial.datasets._recording import (
    Conversion,
    Recording,
    heading_vs_movement,
    session_speed,
    tracked_mask,
)
from neurospatial.datasets._tracks import nearest_segment, placement_tolerance

if TYPE_CHECKING:
    import networkx as nx
    import pandas as pd

    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = [
    "EXPECTED_WATCH_OUTS",
    "PAPER_SESSIONS",
    "REQUIRES",
    "WATCH_OUTS",
    "WORKED_RESULT",
    "corner_stops",
    "engagement_sections",
    "load_session",
    "read_session_lfp",
    "run_intervals",
    "session_members",
    "z_track_graph",
]

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES: tuple[str, ...] = ("fsspec",)

#: The paper's linearized bins, cm ("dwell time and spikes binned into 2 cm bins").
BIN_SIZE = 2.0
#: The paper's run gate for rate maps, cm/s ("less than 3 cm/s were also excluded").
MIN_SPEED = 3.0
#: The paper's rate-map smoothing, a Gaussian of sigma 5 bins, cm.
BANDWIDTH = 5 * BIN_SIZE
#: The paper's decoding check on the track: 500 ms bins, s.
DT = 0.5
#: The paper's replay decoding bins, s.
REPLAY_DT = 0.01
#: The stated Z: two parallel runways and the diagonal joining them, cm.
ARM_LENGTH = 190.0
DIAGONAL_LENGTH = 220.0
STATED_LENGTH = 2 * ARM_LENGTH + DIAGONAL_LENGTH
#: The stated runway width, cm: the narrowest ``track_width``.
RUNWAY_WIDTH = 10.0
#: Left out of the paper's rate maps: the last 10 cm at each end, 5 cm round a corner.
END_EXCLUSION = 10.0
CORNER_EXCLUSION = 5.0
#: A run joins one end zone to the other; the zone is the end's excluded 10 cm.
END_ZONE = END_EXCLUSION
#: The track is fitted to samples faster than this, cm/s: running, not the
#: turning and eating at the ends and corners, where the head sweeps off the runway.
FIT_MIN_SPEED = 10.0
#: The share of track samples the placement tolerance leaves outside the track.
TOLERANCE_QUANTILE = 0.99
#: Ólafsdóttir et al. 2017's rate maps leave out speeds below 10 cm/s, cm/s.
MIN_SPEED_2017 = 10.0
#: Its replay events are at the corners with the animal below 3 cm/s, cm/s.
IMMOBILE_SPEED = 3.0
#: A corner stop starts when the animal comes within CORNER_ZONE of a corner,
#: along the track, and ends when it was last that near before going
#: CORNER_LEAVE away, cm. Our choice: the paper names no distance, and on the
#: default day the samples below 3 cm/s near a corner lie within 20 cm of it;
#: the head LED swings past 20 cm and back while the rat eats, which the
#: second, wider bound does not count as leaving.
CORNER_ZONE = 20.0
CORNER_LEAVE = 40.0
#: The paper's engaged periods: the first and last 5 s of a corner stop, s. A
#: stop shorter than 10 s is engaged throughout, and a section shorter than
#: 2 s is left out.
ENGAGED_SECONDS = 5.0
MIN_SECTION = 2.0
#: Above this speed, cm/s, the LED heading is compared with movement.
HEADING_MIN_SPEED = 10.0
#: The ``.egf`` rate, Hz (each header's ``sample_rate 4800 Hz``).
LFP_RATE = 4800.0
#: The trials of a day, in their order, and the epoch each becomes.
EPOCHS = {
    "track1": "track",
    "sleepPOST": "rest",
    "screening": "open_field",
    "screening2": "open_field_2",
}
#: The rat whose tetrodes 1 to 8 are in CA1 and 9 to 16 in MEC (the README).
REVERSED_RAT = "R2142"
#: Truncation of the Z fit's squared distances, cm, and the samples it uses.
#: Directions of travel are binned in degrees; the diagonal's peak lies at
#: least this far from the runways'; a sample within the tolerance of a peak
#: belongs to it.
_ANGLE_BIN = 5.0
_PEAK_APART = 30.0
_ANGLE_TOLERANCE = 10.0
#: A sample's direction of travel is its displacement over this long, s: a
#: frame's step is a pixel or less while walking.
_STEP_SECONDS = 0.5
#: The share of a runway's samples beyond the end the fit places.
_END_QUANTILE = 0.005
#: Supplementary Table 1 of the paper, by rat: place cells, grid cells and the
#: "Number of events" of each of its sessions, in its order.
PAPER_TABLE = {
    "R2192": ((72, 4, 192), (70, 3, 240), (68, 6, 155)),
    "R2217": ((40, 6, 93), (34, 3, 14)),
    "R2142": ((39, 3, 11),),
    "R2335": ((34, 6, 5),),
    "R2336": ((71, 3, 200), (48, 3, 76)),
    "R2337": ((70, 3, 15), (46, 3, 4)),
}
#: Each deposit day as the paper's session (rat, its number in Supplementary
#: Table 1, the evidence), or None where the files cannot tell. Inferred: the
#: paper numbers sessions without dates. A session's place cells are among its
#: CA1 units, so a day with fewer CA1 units than a session's place cells is not
#: that session; date order is no evidence, since it contradicts this for R2336.
PAPER_SESSIONS: dict[str, tuple[str, int, str] | None] = {
    "R2142-2014-08-06": ("R2142", 1, "the paper's only R2142 session"),
    "R2192-2014-09-17": None,
    "R2192-2014-10-01": None,
    "R2217-2014-12-13": None,
    "R2217-2014-12-18": None,
    "R2335-2015-10-26": ("R2335", 1, "the paper's only R2335 session"),
    "R2336-2015-11-01": (
        "R2336",
        2,
        "56 CA1 units, fewer than the 71 place cells of session 1",
    ),
    "R2336-2015-11-04": (
        "R2336",
        1,
        "71 CA1 units, as many as session 1's 71 place cells, where the other "
        "day has 56",
    ),
    "R2337-2015-11-27": (
        "R2337",
        1,
        "72 CA1 units against session 1's 70 place cells, where the other day has 46",
    ),
    "R2337-2015-12-01": (
        "R2337",
        2,
        "46 CA1 units, fewer than the 70 place cells of session 1",
    ),
}

#: The paper's finding this dataset reproduces, or why there is none yet.
WORKED_RESULT: dict[str, Any] = {
    "status": "not possible yet",
    "level": None,
    "finding": (
        "the share of the rest's candidate events judged replay: 1,826 of the "
        "paper's 4,382, pooled over its 11 sessions"
    ),
    "quote": (
        '"Robust events, with an individual P-value of less than 0.2, were '
        "accepted as replay events and submitted to further analyses (1,826 out "
        'of 4,382 putative events)." (Online Methods)'
    ),
    "source": (
        "Ólafsdóttir, Carpenter & Barry 2016, Nature Neuroscience 19:792-794, "
        "Online Methods"
    ),
    "paper": "41.7% over 11 sessions pooled; no share per session",
    "ours": None,
    "criterion": None,
    "parameters": None,
    "unmatched": (
        "the paper's place cells: interneurons were 'narrow waveforms and high "
        "firing rates' with no numbers; here a mean track rate above 5 Hz, fixed "
        "before the run, which removes 2 of 56 CA1 units",
        "the paper's sessions and candidates: it gives neither its candidates "
        "per session nor which day each session is; here 1,118 on one day, "
        "against a mean of 398 per session in the paper",
        "the shuffle: rate maps rotated by 0 to n_bins - 1 bins here, 1 to "
        "n_bins - 1 in the Methods",
    ),
    "attempted": {
        "value": 0.609,
        "criterion": (
            "not met. 681 of the 1,118 candidates (60.9%) are judged replay, and "
            "the criterion, fixed in writing before any analysis of these data, "
            "asks for 31.7% to 51.7%, within 10 points of the paper's pooled "
            "41.7%"
        ),
        "parameters": (
            "default session R2336-2015-11-01. Place cells: CA1 units with a "
            "field of 20 cm or more above their mean track rate and a peak above "
            "1 Hz in the outbound or inbound map (2 cm bins, 3 cm/s, Gaussian "
            "sigma 10 cm, ends and corners left out), mean rate 5 Hz or less: 52. "
            "Candidates: literature_methods.olafsdottir_2016 at ripple_detection "
            "0eddae64 on the rest alone at 1 kHz: 1,118. Judged by "
            "OLAFSDOTTIR_2016 as packaged (100 place-field shuffles, p < 0.2, the "
            "smaller p of the outbound and inbound maps), 10 ms bins, rng 2026 + "
            "the event's index"
        ),
        "differs": (
            "the paper's own line-fit code, ported and run with the Methods' "
            "shuffle on 150 of these events, judges 91 replay where the preset "
            "judges 93 (88 by both): the gap to the paper is not in the line fit "
            "or the decoder. The candidates' CA1 ripple power is 10.6 times that "
            "of the rest outside them, so they are ripple-like events"
        ),
    },
    "why": (
        "the share of 60.9% lies above the band fixed beforehand, and the "
        "paper's own code gives about the same on these events. What the "
        "deposit cannot settle is which events and cells the paper judged: it "
        "gives no share or candidate count per session, its place-cell rule "
        "leaves the interneuron cut unstated, and this day holds 56 CA1 units "
        "where Supplementary Table 1 lists 71 or 48 place cells for R2336. So "
        "no share here is presented as the paper's"
    ),
}

#: Findings computed from the data that each verified session must report, as
#: patterns searched for in its watch_outs.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "R2336-2015-11-01": (
        r"the fitted Z is 674 cm from end to end, 1\.12 times",
        r"The big LED is not found in 20\.6% of the track frames",
        r"circular mean of \+111 degrees",
        r"48 of 56 clusters sorted in both the rest and the track trial",
        r"tetrodes \[1, 3, 4, 5, 7, 8\]",
        r"68 runs from end to end \(34 outbound, 34 inbound",
        r"session 2 of 2 \(inferred: 56 CA1 units",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "R2336-2015-11-01": (
        "Though the earlier of R2336's two days, this is the paper's session 2 "
        "as the deposit sorts it: it holds 56 CA1 units, fewer than the 71 place "
        "cells Supplementary Table 1 lists for session 1, and R2336-2015-11-04 "
        "holds 71. By the paper's place-cell rule (with an interneuron cut of 5 Hz "
        "chosen here) this day has 52 place cells and R2336-2015-11-04 59, where "
        "the table lists 48 and 71.",
    ),
}

_STEM = re.compile(r"^(\d{8})_(R\d{4})_(track1|sleepPOST|screening2?)$")
_TETRODE = re.compile(r"^(?P<stem>.+)\.(?P<tetrode>\d{1,2})$")
_CUT = re.compile(r"^(?P<stem>.+)_(?P<tetrode>\d{1,2})\.cut$")


class _Trial(NamedTuple):
    """One recording of the day: its files, settings and place on the clock."""

    kind: str
    stem: str
    settings: dict[str, str]
    start: float
    duration: float
    tetrodes: dict[int, Path]
    cuts: dict[int, Path]
    position: Path


def _trials(folder: Path, names: set[str]) -> list[_Trial]:
    """The day's trials found among ``names``, in time order, each with its start.

    Raises
    ------
    ValueError
        If a trial lacks its ``.set`` or ``.pos`` file, the day has no track
        trial, or two trials overlap in time.
    """
    stems: dict[str, str] = {}
    for name in names:
        stem = name.rsplit(".", 1)[0] if not name.endswith(".cut") else ""
        if (match := _STEM.match(stem)) is not None:
            stems[stem] = match[3]
    starts = {}
    trials = []
    for stem, kind in stems.items():
        if f"{stem}.set" not in names or f"{stem}.pos" not in names:
            msg = f"{stem}: its .set or .pos file is not among the session's files"
            raise ValueError(msg)
        settings = read_set(folder / f"{stem}.set")
        starts[stem] = datetime.strptime(
            f"{settings['trial_date'].split(', ')[-1]} {settings['trial_time']}",
            "%d %b %Y %H:%M:%S",
        )
        tetrodes = {
            int(m["tetrode"]): folder / name
            for name in names
            if (m := _TETRODE.match(name)) and m["stem"] == stem
        }
        cuts = {
            int(m["tetrode"]): folder / name
            for name in names
            if (m := _CUT.match(name)) and m["stem"] == stem
        }
        trials.append(
            (kind, stem, settings, float(settings["duration"]), tetrodes, cuts)
        )
    track = [stem for stem, kind in stems.items() if kind == "track1"]
    if len(track) != 1:
        msg = f"{len(track)} track trials among the session's files, not one"
        raise ValueError(msg)
    zero = starts[track[0]]
    built = sorted(
        (
            _Trial(
                kind, stem, settings, (starts[stem] - zero).total_seconds(), duration,
                tetrodes, cuts, folder / f"{stem}.pos",
            )
            for kind, stem, settings, duration, tetrodes, cuts in trials
        ),
        key=lambda trial: trial.start,
    )  # fmt: skip
    for first, second in pairwise(built):
        if first.start + first.duration > second.start:
            msg = (
                f"{first.stem} runs to {first.start + first.duration:g} s, past the "
                f"start of {second.stem} at {second.start:g} s"
            )
            raise ValueError(msg)
    return built


def session_members(names: list[str]) -> dict[str, tuple[list[str], list[str]]]:
    """Each recording day's members in the deposit's zip: those a load reads, and the LFP.

    A day is a ``r<rat>/<date>`` folder, whose session id is ``R<rat>-<date>``.
    A load reads each trial's tetrode files, ``_<tetrode>.cut`` files, ``.pos``
    and ``.set``; the LFP is each trial's ``.egf`` files. Left out: ``.clu``
    (the README: superseded by the ``.cut`` files), ``.eeg`` (the ``.egf`` at
    250 Hz), ``.inp``, the ``*_cell_info.mat`` and ``*_tet_info.mat`` files, and
    the cut files named for no tetrode (``<trial>.cut``, ``_6.ut``).
    ``scripts/build_dataset_registry.py`` calls this to list the sessions.

    Parameters
    ----------
    names : list of str
        Every member path of the zip.

    Returns
    -------
    dict of str to (list of str, list of str)
        Each session id, its load members and its ``.egf`` members, sorted.
    """
    days: dict[str, tuple[list[str], list[str]]] = {}
    for name in names:
        parts = name.split("/")
        if len(parts) != 3 or not parts[2]:
            continue
        folder, date, file = parts
        stem, _, suffix = file.partition(".")
        cut = _CUT.match(file)
        if cut is not None:
            stem = cut["stem"]
        if _STEM.match(stem) is None:
            continue
        session = f"{folder.upper()}-{date}"
        load, lfp = days.setdefault(session, ([], []))
        if cut is not None or suffix in ("pos", "set") or suffix.isdigit():
            load.append(name)
        elif re.fullmatch(r"egf\d*", suffix):
            lfp.append(name)
    return {
        day: (sorted(load), sorted(lfp)) for day, (load, lfp) in sorted(days.items())
    }


def _region(rat: str, tetrode: int) -> str:
    """The brain region of a tetrode: 1 to 8 MEC, 9 to 16 CA1, reversed for R2142."""
    first_half = tetrode <= 8
    if rat == REVERSED_RAT:
        first_half = not first_half
    return "MEC" if first_half else "CA1"


def _positions_cm(track: PositionTrack, spot: int) -> NDArray[np.float64]:
    """One spot's tracked pixels in cm on the camera's axes; NaN where not found.

    ``spot`` 0 is the big LED (words 0 and 1), 1 the little one (words 2, 3).
    """
    xy = track.words[:, 2 * spot : 2 * spot + 2].astype(np.float64)
    xy[np.any(track.words[:, 2 * spot : 2 * spot + 2] == UNTRACKED, axis=1)] = np.nan
    return xy * 100.0 / track.pixels_per_metre


def _headings(
    big: NDArray[np.float64], little: NDArray[np.float64], bearing: float
) -> NDArray[np.float64]:
    """From the little LED to the big one, radians in the positions' frame.

    The lab's code (``postprocess_pos_data.m`` as ported) takes the big light as
    the front and subtracts its ``lightBearing_1`` in a y-up frame; with y
    down, as these positions are, the bearing is added.
    """
    step = big - little
    angle = np.arctan2(step[:, 1], step[:, 0]) + np.radians(bearing)
    return np.asarray(np.angle(np.exp(1j * angle)), dtype=np.float64)


def z_track_graph(nodes: NDArray[np.float64]) -> nx.Graph:
    """The Z-track as a graph of its four ends and corners.

    Parameters
    ----------
    nodes : ndarray, shape (4, 2)
        The start of the first runway, the two corners, and the end of the
        last runway, in order along the track, cm.

    Returns
    -------
    networkx.Graph
        Nodes ``start``, ``corner_1``, ``corner_2`` and ``end`` with ``pos``;
        edges along the track with ``distance``, cm.
    """
    import networkx as nx

    names = ("start", "corner_1", "corner_2", "end")
    graph = nx.Graph()
    for name, point in zip(names, nodes, strict=True):
        graph.add_node(name, pos=(float(point[0]), float(point[1])))
    for (a, pa), (b, pb) in zip(
        zip(names, nodes, strict=True),
        zip(names[1:], nodes[1:], strict=True),
        strict=False,
    ):
        graph.add_edge(a, b, distance=float(np.linalg.norm(pb - pa)))
    return graph


def _axial_peaks(angle: NDArray[np.float64]) -> tuple[float, float]:
    """The two most common directions of travel, mod 180 degrees, radians.

    A histogram of 5 degree bins over [0, pi); the second peak is the most
    common direction at least 30 degrees from the first.
    """
    edges = np.radians(np.arange(0.0, 180.0 + _ANGLE_BIN, _ANGLE_BIN))
    counts, _ = np.histogram(np.mod(angle, np.pi), bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    first = int(np.argmax(counts))
    apart = np.abs(np.angle(np.exp(2j * (centres - centres[first])))) / 2
    second = int(np.argmax(np.where(apart >= np.radians(_PEAK_APART), counts, -1)))
    return float(centres[first]), float(centres[second])


def _line(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The total-least-squares line through ``points``: a point and a unit direction."""
    centre = points.mean(axis=0)
    return centre, np.linalg.svd(points - centre, full_matrices=False)[2][0]


def _fit_z(
    points: NDArray[np.float64], steps: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The Z's ends and corners from running samples and their directions of travel.

    The two runways are parallel and the diagonal is not, so running samples
    fall into two directions of travel. Samples moving within 10 degrees of
    the commoner direction are the runways, split into the two by their side
    of the line between them; those within 10 degrees of the other are the
    diagonal. Each runway is a total-least-squares line through its samples,
    from where they reach at one end to the other (their 0.5% and 99.5%
    quantiles along it); its tip nearer the diagonal's line is a corner, and
    the diagonal runs from corner to corner.

    Parameters
    ----------
    points : ndarray, shape (n, 2)
        Positions of running samples, cm.
    steps : ndarray, shape (n, 2)
        Each sample's displacement over the next half second, any length
        unit.

    Returns
    -------
    ndarray, shape (4, 2)
        An end, the corners, and the other end, in order along the track, cm.
    """
    angle = np.arctan2(steps[:, 1], steps[:, 0])
    arm_angle, diagonal_angle = _axial_peaks(angle)

    def near(target: float) -> NDArray[np.bool_]:
        apart = np.abs(np.angle(np.exp(2j * (angle - target)))) / 2
        return np.asarray(apart <= np.radians(_ANGLE_TOLERANCE), dtype=bool)

    arms, diagonal = points[near(arm_angle)], points[near(diagonal_angle)]
    normal = np.array([-np.sin(arm_angle), np.cos(arm_angle)])
    side = arms @ normal
    split = 0.5 * (np.quantile(side, 0.1) + np.quantile(side, 0.9))
    centre, direction = _line(diagonal)
    nodes = []
    for mine in (arms[side < split], arms[side >= split]):
        point, along_runway = _line(mine)
        along = (mine - point) @ along_runway
        reach = np.quantile(along, [_END_QUANTILE, 1.0 - _END_QUANTILE])
        tips = point + reach[:, None] * along_runway
        # The corner is the tip nearer the diagonal's line.
        relative = tips - centre
        off = np.abs(direction[0] * relative[:, 1] - direction[1] * relative[:, 0])
        corner, end = (tips[0], tips[1]) if off[0] < off[1] else (tips[1], tips[0])
        nodes.append((end, corner))
    return np.array([nodes[0][0], nodes[0][1], nodes[1][1], nodes[1][0]])


def _oriented(
    nodes: NDArray[np.float64], first: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The nodes in track order from the end nearer the first tracked sample."""
    if np.linalg.norm(nodes[-1] - first) < np.linalg.norm(nodes[0] - first):
        return np.array(nodes[::-1], dtype=np.float64)
    return nodes


def _linear(
    positions: NDArray[np.float64], nodes: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Each sample's distance along the track from its start, and from the track.

    NaN for an untracked sample. Distances are cm.
    """
    lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    before = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    tracked = tracked_mask(positions)
    linear = np.full(len(positions), np.nan)
    distance = np.full(len(positions), np.nan)
    nearest = nearest_segment(positions[tracked], nodes[:-1], nodes[1:])
    linear[tracked] = before[nearest.segment] + nearest.along * lengths[nearest.segment]
    distance[tracked] = nearest.distance
    return linear, distance


def run_intervals(
    times: NDArray[np.float64],
    linear: NDArray[np.float64],
    length: float,
    *,
    end_zone: float = END_ZONE,
) -> tuple[NDArray[np.float64], NDArray[np.str_]]:
    """Runs from one end of the track to the other, and their direction.

    A run starts at the last sample in one end zone (within ``end_zone`` of
    that end) before the animal reaches the other end zone, and stops at the
    first sample there. Running out of a zone and back into it is no run.

    Parameters
    ----------
    times : ndarray, shape (n_time,)
        Seconds.
    linear : ndarray, shape (n_time,)
        Distance along the track from its start, cm; NaN where untracked.
    length : float
        The track's length, cm.
    end_zone : float, default 10.0
        cm.

    Returns
    -------
    rows : ndarray, shape (n_runs, 2)
        ``[start, stop]`` seconds.
    direction : ndarray of str, shape (n_runs,)
        ``"outbound"`` from the start end to the far one, ``"inbound"`` back.

    Examples
    --------
    >>> times = np.arange(7.0)
    >>> linear = np.array([2.0, 5.0, 50.0, 95.0, 60.0, 98.0, 3.0])
    >>> rows, direction = run_intervals(times, linear, 100.0)
    >>> rows.tolist(), direction.tolist()
    ([[1.0, 3.0], [5.0, 6.0]], ['outbound', 'inbound'])
    """
    zone = np.full(linear.shape, -1)
    zone[linear <= end_zone] = 0
    zone[linear >= length - end_zone] = 1
    at = np.flatnonzero(zone >= 0)
    rows, direction = [], []
    for before, after in pairwise(at):
        if zone[before] != zone[after]:
            rows.append([times[before], times[after]])
            direction.append("outbound" if zone[before] == 0 else "inbound")
    return np.array(rows, dtype=np.float64).reshape(-1, 2), np.array(
        direction, dtype=str
    )


def corner_stops(
    times: NDArray[np.float64],
    linear: NDArray[np.float64],
    corners: NDArray[np.float64],
    *,
    zone: float = CORNER_ZONE,
    leave: float = CORNER_LEAVE,
) -> tuple[NDArray[np.float64], NDArray[np.str_], NDArray[np.int64]]:
    """Stays at the track's corners: from arriving to leaving, with the run's direction.

    A stop at a corner starts at the first tracked sample within ``zone`` of
    it, along the track, after the animal was more than ``leave`` away, and
    ends at the last sample within ``zone`` before it is more than ``leave``
    away again. Samples between the two distances, and untracked ones, neither
    start nor end a stop. A stay with no sample beyond ``leave`` before it
    (the trial starting at the corner) has no direction and is left out.

    Parameters
    ----------
    times : ndarray, shape (n_time,)
        Seconds.
    linear : ndarray, shape (n_time,)
        Distance along the track from its start, cm; NaN where untracked.
    corners : ndarray, shape (n_corners,)
        Each corner's distance along the track, cm.
    zone, leave : float
        cm.

    Returns
    -------
    rows : ndarray, shape (n_stops, 2)
        ``[arrival, departure]`` seconds, sorted by arrival.
    direction : ndarray of str, shape (n_stops,)
        The run the animal arrived on: ``"outbound"`` from the start's side,
        ``"inbound"`` from the end's.
    corner : ndarray of int, shape (n_stops,)
        Which corner, from 1 at the start's end.

    Examples
    --------
    >>> times = np.arange(9.0)
    >>> linear = np.array([0.0, 70.0, 95.0, 125.0, 102.0, 99.0, 150.0, 104.0, 30.0])
    >>> rows, direction, corner = corner_stops(times, linear, np.array([100.0]))
    >>> rows.tolist(), direction.tolist(), corner.tolist()
    ([[2.0, 5.0], [7.0, 7.0]], ['outbound', 'inbound'], [1, 1])
    """
    tracked = np.flatnonzero(np.isfinite(linear))
    rows, direction, which = [], [], []
    for number, corner in enumerate(corners, 1):
        offset = linear[tracked] - corner
        state = np.where(np.abs(offset) <= zone, 1, np.where(np.abs(offset) > leave, -1, 0))
        decided = np.flatnonzero(state != 0)
        near = state[decided] == 1
        edges = np.diff(np.concatenate([[0], near.astype(np.int8), [0]]))
        for first, after in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
        ):
            if first == 0:
                continue
            came_from = offset[decided[first - 1]]
            rows.append([times[tracked[decided[first]]], times[tracked[decided[after - 1]]]])
            direction.append("outbound" if came_from < 0 else "inbound")
            which.append(number)
    order = np.argsort([row[0] for row in rows], kind="stable")
    return (
        np.array(rows, dtype=np.float64).reshape(-1, 2)[order],
        np.array(direction, dtype=str)[order],
        np.array(which, dtype=np.int64)[order],
    )


def engagement_sections(
    stops: NDArray[np.float64],
    *,
    engaged: float = ENGAGED_SECONDS,
    min_section: float = MIN_SECTION,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The engaged and disengaged sections of corner stops (Ólafsdóttir et al. 2017).

    The first and last ``engaged`` seconds of a stop are engaged and the time
    between them disengaged; a stop no longer than ``2 * engaged`` is engaged
    throughout. Sections shorter than ``min_section`` are left out.

    Parameters
    ----------
    stops : ndarray, shape (n_stops, 2)
        ``[arrival, departure]`` seconds.
    engaged, min_section : float
        Seconds.

    Returns
    -------
    engaged_rows, disengaged_rows : ndarray, shape (n, 2)
        ``[start, stop]`` seconds, in time order.

    Examples
    --------
    >>> stops = np.array([[0.0, 1.0], [10.0, 18.0], [30.0, 41.0], [50.0, 70.0]])
    >>> engaged, disengaged = engagement_sections(stops)
    >>> engaged.tolist()
    [[10.0, 18.0], [30.0, 35.0], [36.0, 41.0], [50.0, 55.0], [65.0, 70.0]]
    >>> disengaged.tolist()
    [[55.0, 65.0]]
    """
    start, stop = stops[:, 0], stops[:, 1]
    whole = stop - start <= 2 * engaged
    split = ~whole
    engaged_rows = np.concatenate(
        [
            stops[whole],
            np.column_stack([start[split], start[split] + engaged]),
            np.column_stack([stop[split] - engaged, stop[split]]),
        ]
    )
    disengaged_rows = np.column_stack([start[split] + engaged, stop[split] - engaged])

    def kept(rows: NDArray[np.float64]) -> NDArray[np.float64]:
        rows = rows[rows[:, 1] - rows[:, 0] >= min_section]
        return rows[np.argsort(rows[:, 0], kind="stable")]

    return kept(engaged_rows), kept(disengaged_rows)


def _declared_track(
    times: NDArray[np.float64], positions: NDArray[np.float64], session: str
) -> tuple[Environment, NDArray[np.float64], float, Conversion]:
    """The stated Z on the track trial's samples, its width a placement tolerance.

    Returns
    -------
    env : Environment
    nodes : ndarray, shape (4, 2)
    width : float
        ``track_width``, cm.
    placement : Conversion
    """
    from neurospatial import Environment

    tracked = tracked_mask(positions)
    samples = positions[tracked]
    speed = session_speed(times, positions)
    fast = tracked & (
        np.nan_to_num(speed if speed is not None else 0.0) > FIT_MIN_SPEED
    )
    lag = round(_STEP_SECONDS / float(np.median(np.diff(times))))
    running = fast[:-lag] & fast[lag:]
    steps = positions[lag:] - positions[:-lag]
    nodes = _oriented(_fit_z(positions[:-lag][running], steps[running]), samples[0])
    _, distance = _linear(positions, nodes)
    width, _, evidence = placement_tolerance(
        distance[tracked],
        TOLERANCE_QUANTILE,
        floor=RUNWAY_WIDTH,
        measured="the big LED's distance from the fitted track",
        reason=(
            f"The runways are {RUNWAY_WIDTH:g} cm wide with no walls, and a "
            "head-mounted LED swings beyond them."
        ),
    )
    graph = z_track_graph(nodes)
    env = Environment.from_graph(
        graph,
        [("start", "corner_1"), ("corner_1", "corner_2"), ("corner_2", "end")],
        bin_size=BIN_SIZE,
        units="cm",
        name=f"olafsdottir_2016 {session}",
        track_width=width,
    )
    lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    placement = Conversion(
        "environment: a Z of three straight pieces, its four ends and corners "
        f"fitted to the track trial's samples ({lengths[0]:.0f}, {lengths[1]:.0f} "
        f"and {lengths[2]:.0f} cm); track_width {width:.3g} cm",
        "inferred",
        "no file gives the track's place in the camera's frame. The paper states "
        f"runways of {ARM_LENGTH:g} cm joined by a {DIAGONAL_LENGTH:g} cm diagonal; "
        f"the fit's pieces are {lengths[0]:.0f}, {lengths[1]:.0f} and "
        f"{lengths[2]:.0f} cm at the .pos file's pixels_per_metre. {evidence}",
    )
    return env, nodes, width, placement


def _read_cut_for(
    trial: _Trial, tetrode: int, n_spikes: int
) -> NDArray[np.int64] | None:
    """The trial's clusters for a tetrode, checked against its spikes; None without a cut."""
    path = trial.cuts.get(tetrode)
    if path is None:
        return None
    clusters, cut_for = read_cut(path)
    if cut_for != trial.stem or clusters.size != n_spikes:
        msg = (
            f"{path.name}: cut for {cut_for} with {clusters.size} spikes, not for "
            f"{trial.stem} with its {n_spikes}"
        )
        raise ValueError(msg)
    return clusters


class _Spikes(NamedTuple):
    """One tetrode's spikes over the day."""

    times: NDArray[np.float64]
    amplitudes: NDArray[np.float32]
    clusters: NDArray[np.int64]
    trial: NDArray[np.int64]
    means: dict[tuple[int, int], NDArray[np.float64]]


def _tetrode_spikes(trials: list[_Trial], tetrode: int) -> _Spikes:
    """A tetrode's spikes in every trial: times on the day's clock, peaks in uV.

    ``means`` holds each (trial, cluster)'s mean waveform in uV, for checking
    that a cluster id names one cell across trials.
    """
    times, amplitudes, clusters, which = [], [], [], []
    means = {}
    for index, trial in enumerate(trials):
        path = trial.tetrodes.get(tetrode)
        if path is None:
            continue
        spikes = read_tetrode(path)
        scale = np.array(
            [
                microvolts_per_unit(trial.settings, ch, 1)
                for ch in tetrode_channels(tetrode)
            ]
        )
        n = spikes.times.size
        cut = _read_cut_for(trial, tetrode, n)
        ids = np.full(n, UNSORTED, dtype=np.int64) if cut is None else cut
        times.append(spikes.times + trial.start)
        amplitudes.append((spikes.waveforms.max(axis=2) * scale).astype(np.float32))
        clusters.append(ids)
        which.append(np.full(n, index))
        if cut is not None:
            for cluster in np.unique(cut[cut > 0]):
                waves = spikes.waveforms[cut == cluster].astype(np.float64)
                means[index, int(cluster)] = waves.mean(axis=0) * scale[:, None]
    if not times:
        empty = np.empty(0)
        return _Spikes(empty, np.empty((0, 4), np.float32), empty.astype(np.int64),
                       empty.astype(np.int64), means)  # fmt: skip
    return _Spikes(
        np.concatenate(times), np.concatenate(amplitudes), np.concatenate(clusters),
        np.concatenate(which), means,
    )  # fmt: skip


def read_waveforms(
    folder: Path, electrode: str, *, files: Mapping[str, tuple[tuple[str, str], ...]]
) -> NDArray[np.float32]:
    """One tetrode's waveforms over the day, in microvolts.

    Parameters
    ----------
    folder : path
        The session's fetched files.
    electrode : str
        ``"T<tetrode>"``.
    files : Mapping[str, tuple of (str, str)]
        For each electrode, its tetrode file and that trial's ``.set`` file,
        in the order of the day's trials.

    Returns
    -------
    ndarray of float32, shape (n_spikes, 4, 50)
    """
    tetrode = int(electrode[1:])
    parts = []
    for tetrode_file, set_file in files[electrode]:
        settings = read_set(folder / set_file)
        scale = np.array(
            [microvolts_per_unit(settings, ch, 1) for ch in tetrode_channels(tetrode)],
            dtype=np.float32,
        )
        waves = read_tetrode(folder / tetrode_file).waveforms
        parts.append(waves.astype(np.float32) * scale[None, :, None])
    return np.concatenate(parts)


def _same_cluster_share(
    means: Mapping[tuple[int, int], NDArray[np.float64]], reference: int, other: int
) -> tuple[int, int]:
    """Of the clusters sorted in both trials, how many are nearest the same id.

    Distance is the root mean square difference of the mean waveforms over the
    four channels and fifty samples, uV.
    """
    ids_a = sorted(c for t, c in means if t == reference)
    ids_b = sorted(c for t, c in means if t == other)
    shared = sorted(set(ids_a) & set(ids_b))
    if not shared or len(ids_a) < 2:
        return 0, len(shared)
    a = np.array([means[reference, c] for c in ids_a]).reshape(len(ids_a), -1)
    same = 0
    for cluster in shared:
        b = means[other, cluster].ravel()
        nearest = ids_a[int(np.argmin(np.sqrt(np.mean((a - b) ** 2, axis=1))))]
        same += nearest == cluster
    return same, len(shared)


def read_session_lfp(
    sources: tuple[str | Path, ...],
    channels: tuple[str, ...],
    *,
    layout: Mapping[str, tuple[str, float, int]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read ``.egf`` channels of the day's trials, in microvolts on the day's clock.

    Parameters
    ----------
    sources : tuple of path
        The fetched ``.egf`` files holding ``channels``.
    channels : tuple of str
        ``"T<tetrode>.<channel>"`` names, in the order of the returned columns.
    layout : Mapping[str, tuple of (str, float, int)]
        For each ``.egf`` file name: the channel it holds, its trial's start
        on the day's clock (s), and its recording channel from 0.

    Returns
    -------
    samples : ndarray, shape (n_time, n_channels)
        Microvolts: stored values x ``microvolts_per_unit`` of the trial's
        ``.set`` file.
    times : ndarray, shape (n_time,)
        Seconds on the day's clock: sample ``k`` (from 1) of a trial at its
        start plus ``k / 4800``.

    Raises
    ------
    ValueError
        If a trial lacks one of ``channels``, or two channels of a trial differ
        in length.
    """
    by_trial: dict[float, dict[str, Path]] = {}
    for source in sources:
        path = Path(source)
        channel, start, _ = layout[path.name]
        by_trial.setdefault(start, {})[channel] = path
    columns, clocks = [], []
    for start in sorted(by_trial):
        held = by_trial[start]
        if missing := [name for name in channels if name not in held]:
            msg = f"the trial starting at {start:g} s has no .egf for {missing}"
            raise ValueError(msg)
        block = []
        for name in channels:
            path = held[name]
            samples, rate, _ = read_continuous(path)
            stem = path.name.split(".egf")[0]
            settings = read_set(path.parent / f"{stem}.set")
            block.append(
                samples * microvolts_per_unit(settings, layout[path.name][2], 2)
            )
        if len({len(column) for column in block}) != 1:
            msg = f"the channels of the trial at {start:g} s differ in length"
            raise ValueError(msg)
        columns.append(np.column_stack(block))
        clocks.append(start + np.arange(1, len(block[0]) + 1) / rate)
    return np.concatenate(columns), np.concatenate(clocks)


def _egf_channel(settings: Mapping[str, str], slot: int) -> int | None:
    """The recording channel (from 0) of EEG slot ``slot``; None if not saved."""
    if settings.get(f"saveEEG_ch_{slot}", "0") != "1":
        return None
    channel = int(settings.get(f"EEG_ch_{slot}", "0"))
    return channel - 1 if channel > 0 else None


def _channel_name(channel: int) -> str:
    """``"T9.1"`` for recording channel 32: tetrode 9, its first wire."""
    return f"T{channel // 4 + 1}.{channel % 4 + 1}"


def _lfp_source(
    entry: SessionEntry, trials: list[_Trial], rat: str, folder: Path
) -> LFPSource | None:
    """The day's ``.egf`` LFP, one channel per file; nothing fetched."""
    from neurospatial.datasets._lfp import LFPFile

    if not entry.lfp_files:
        return None
    layout: dict[str, tuple[str, float, int]] = {}
    files = []
    described: dict[str, str] = {}
    for file in entry.lfp_files:
        name = Path(file.path).name
        stem, _, suffix = name.partition(".egf")
        trial = next(t for t in trials if t.stem == stem)
        channel = _egf_channel(trial.settings, int(suffix or 1))
        if channel is None:
            continue
        label = _channel_name(channel)
        layout[name] = (label, trial.start, channel)
        files.append(
            LFPFile(
                path=file.path, sha256=file.sha256, size_bytes=file.size_bytes,
                url=file.url, member=file.member, channels=(label,),
            )
        )  # fmt: skip
        tetrode = channel // 4 + 1
        described.setdefault(
            label,
            f"tetrode {tetrode} wire {channel % 4 + 1} ({_region(rat, tetrode)}), "
            f"recording channel {channel + 1}",
        )
    held = {label for label, *_ in layout.values()}
    return LFPSource(
        dataset=entry.info.dataset,
        files=tuple(files),
        sampling_rate=LFP_RATE,
        channels={label: described[label] for label in sorted(held)},
        unit="uV",
        unit_basis=Conversion(
            "LFP: stored little-endian int16 x 1000 ADC_fullscale_mv / (gain 32768), "
            "the gain of the slot's recording channel in its trial's .set",
            "stated",
            "the .set's ADC_fullscale_mv and gain_ch_<EEG_ch_n - 1> (the Barry "
            "lab's eeg_convert_to_volts, as ported; Neo's axonarawio gives the "
            "same scale)",
        ),
        times_basis=Conversion(
            "LFP times: sample k (from 1) of a trial at its start + k / 4800 s",
            "inferred",
            "the dacqUSB document times the k-th position sample at k / 50 s and "
            "gives no rule for .egf; the same rule is used, which is one sample "
            "(0.21 ms) from the other reading. Each .egf holds duration x 4800 "
            "samples",
        ),
        reader=functools.partial(read_session_lfp, layout=layout),
        ripple_channels=(),
        cache=folder.parents[1],
    )


def _units(
    spikes: Mapping[int, _Spikes], trials: list[_Trial], rat: str
) -> tuple[tuple[NDArray[np.float64], ...], pd.DataFrame]:
    """Every cluster from 1 up on any trial, its spikes over the day, and its table."""
    import pandas as pd

    rows, trains = [], []
    for tetrode in sorted(spikes):
        day = spikes[tetrode]
        for cluster in np.unique(day.clusters[day.clusters > 0]):
            mine = day.clusters == cluster
            trains.append(day.times[mine])
            row = {
                "unit_id": f"T{tetrode}_c{cluster}",
                "brain_region": _region(rat, tetrode),
                "label": None,
                "electrode": f"T{tetrode}",
                "cluster": int(cluster),
            }
            for index, trial in enumerate(trials):
                row[f"n_spikes_{EPOCHS[trial.kind]}"] = int(
                    np.sum(mine & (day.trial == index))
                )
            rows.append(row)
    table = pd.DataFrame(rows).set_index("unit_id")
    return tuple(trains), table


def _cluster_watch_out(
    spikes: Mapping[int, _Spikes], trials: list[_Trial]
) -> str | None:
    """How often a cluster's mean waveform off the track is nearest its own id there."""
    track = next(i for i, t in enumerate(trials) if t.kind == "track1")
    parts = []
    for index, trial in enumerate(trials):
        if index == track:
            continue
        same = total = 0
        for day in spikes.values():
            a, b = _same_cluster_share(day.means, track, index)
            same, total = same + a, total + b
        if total:
            parts.append(
                f"{same} of {total} clusters sorted in both the {EPOCHS[trial.kind]} "
                "and the track trial have a mean waveform nearest that of the same "
                "id on the track"
            )
    if not parts:
        return None
    return (
        "Each trial was cut on its own (every .cut is 'Exact_cut_for' one trial), "
        "and a unit joins the clusters of one id across the day's trials, as the "
        f"paper's decoding of rest spikes with track rate maps implies: {'; '.join(parts)} "
        "(root mean square difference over the four channels). A cluster whose "
        "waveform is nearer another id may not be the same cell in both trials."
    )


def _sorting_watch_out(
    spikes: Mapping[int, _Spikes], trials: list[_Trial]
) -> str | None:
    """Which trials leave which tetrodes unsorted."""
    parts = []
    for trial in trials:
        unsorted = sorted(set(trial.tetrodes) - set(trial.cuts))
        if unsorted:
            parts.append(f"{EPOCHS[trial.kind]}: tetrodes {unsorted}")
    if not parts:
        return None
    n_spikes = sum(int(np.sum(day.clusters == UNSORTED)) for day in spikes.values())
    return (
        f"Some tetrodes have no .cut in a trial ({'; '.join(parts)}): their "
        f"{n_spikes:,} spikes there are in rec.marks with cluster -1 and in no "
        "unit, so a unit's silence in that trial is not an absence of firing."
    )


def _tracking_watch_outs(
    track: PositionTrack,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    distance: NDArray[np.float64],
    width: float,
) -> list[str]:
    """The findings of the track trial's tracking, each with its numbers."""
    notes = []
    big = np.mean(np.any(track.words[:, :2] == UNTRACKED, axis=1))
    little = np.mean(np.any(track.words[:, 2:4] == UNTRACKED, axis=1))
    notes.append(
        f"The big LED is not found in {big:.1%} of the track frames (1023 in the "
        f".pos file; positions are NaN there) and the little one in {little:.1%}: "
        f"both are found in {np.mean(np.isfinite(headings)):.1%} of frames."
    )
    tracked = np.isfinite(distance)
    off = float(np.mean(distance[tracked] > width / 2))
    notes.append(
        f"{off:.1%} of tracked samples lie beyond the {width:.3g} cm track_width of "
        "the fitted Z and fall in no bin of rec.env."
    )
    repeats = int(np.sum(np.diff(track.frame_counter.astype(np.int64)) == 0))
    if repeats:
        notes.append(
            f"The .pos frame counter repeats {repeats} times on the track: the "
            "dacqUSB document says to ignore it and time frame k at k / 50 s, as "
            "here."
        )
    if np.any(np.isfinite(headings)):
        offset, spread = heading_vs_movement(
            times, positions, headings, min_speed=HEADING_MIN_SPEED
        )
        notes.append(
            "No headings: where both LEDs are found, the direction from the little "
            "LED to the big one (the lab's front light, lightBearing_1 subtracted), "
            f"minus the direction of movement above {HEADING_MIN_SPEED:g} cm/s, has "
            f"a circular mean of {np.degrees(offset):+.0f} degrees and a median "
            f"spread of {np.degrees(spread):.0f} degrees, so it does not give the "
            "head's direction. params['little_led_positions'] keeps that LED."
        )
    return notes


def _paper_watch_out(session: str, n_ca1: int) -> str:
    """What the paper's Supplementary Table 1 says of this day, and how it is matched."""
    rat = session.split("-")[0]
    match = PAPER_SESSIONS.get(session)
    rows = PAPER_TABLE[rat]
    listed = "; ".join(
        f"session {k}: {p} place cells, {g} grid cells"
        for k, (p, g, _) in enumerate(rows, 1)
    )
    if match is None:
        return (
            f"Which of the paper's {len(rows)} {rat} sessions this day is cannot be "
            f"told: Supplementary Table 1 lists {listed}, and the day has {n_ca1} "
            "CA1 units, which rules none of them out."
        )
    _, number, evidence = match
    place, grid, events = rows[number - 1]
    return (
        f"Taken as the paper's {rat} session {number} of {len(rows)} (inferred: "
        f"{evidence}). Supplementary Table 1 lists {place} place cells, {grid} grid "
        f"cells and {events} under 'Number of events' for it."
    )


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one recording day's files into a :class:`Recording`.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The directory holding the fetched members.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.

    Raises
    ------
    ValueError
        If a trial lacks its ``.set`` or ``.pos``, two trials overlap, a cut
        file is for another trial or another number of spikes, or a file
        breaks its format.
    """
    from neurospatial import Intervals

    folder = Path(path_or_stream)
    names = {Path(member.path).name for member in entry.members}
    trials = _trials(folder, names)
    rat = entry.session.split("-")[0]
    track_trial = next(t for t in trials if t.kind == "track1")
    track = read_position(track_trial.position)
    times = track.times
    positions = _positions_cm(track, 0)
    little = _positions_cm(track, 1)
    bearing = float(track_trial.settings.get("lightBearing_1", "0"))
    headings = _headings(positions, little, bearing)
    env, nodes, width, placement = _declared_track(times, positions, entry.session)
    linear, distance = _linear(positions, nodes)
    length = float(np.sum(np.linalg.norm(np.diff(nodes, axis=0), axis=1)))
    runs, direction = run_intervals(times, linear, length)

    tetrodes = sorted({t for trial in trials for t in trial.tetrodes})
    spikes = {t: _tetrode_spikes(trials, t) for t in tetrodes}
    spike_times, units = _units(spikes, trials, rat)

    epochs: dict[str, NDArray[np.float64]] = {
        EPOCHS[t.kind]: np.array([[t.start, t.start + t.duration]]) for t in trials
    }
    epochs["outbound"] = runs[direction == "outbound"]
    epochs["inbound"] = runs[direction == "inbound"]
    corners = np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))[:-1]
    stops, stop_direction, stop_corner = corner_stops(times, linear, corners)
    epochs["corner_stops"] = stops
    epochs["engaged"], epochs["disengaged"] = engagement_sections(stops)
    params: dict[str, Any] = {
        "bin_size": BIN_SIZE,
        "min_speed": MIN_SPEED,
        "bandwidth": BANDWIDTH,
        "dt": DT,
        "replay_dt": REPLAY_DT,
        "analysis_epochs": runs,
        "train_epochs": runs[::2],
        "test_epochs": runs[1::2],
        "track_nodes": nodes,
        "track_length": length,
        "track_width": width,
        "end_exclusion": END_EXCLUSION,
        "corner_exclusion": CORNER_EXCLUSION,
        "min_speed_2017": MIN_SPEED_2017,
        "immobile_speed": IMMOBILE_SPEED,
        "corner_stop_direction": stop_direction,
        "corner_stop_corner": stop_corner,
        "paper_session":(PAPER_SESSIONS.get(entry.session) or (None, None, ""))[:2],
        "little_led_positions": little,
    }
    for trial in trials:
        if trial.kind == "track1":
            continue
        other = read_position(trial.position)
        name = EPOCHS[trial.kind]
        params[f"{name}_times"] = other.times + trial.start
        params[f"{name}_positions"] = _positions_cm(other, 0)

    files = {
        f"T{t}": tuple(
            (trial.tetrodes[t].name, f"{trial.stem}.set")
            for trial in trials
            if t in trial.tetrodes
        )
        for t in tetrodes
    }
    marks = SpikeMarks(
        electrodes=tuple(f"T{t}" for t in tetrodes),
        times=tuple(spikes[t].times for t in tetrodes),
        amplitudes=tuple(spikes[t].amplitudes for t in tetrodes),
        clusters=tuple(spikes[t].clusters for t in tetrodes),
        unit="uV",
        basis=Conversion(
            "marks: each channel's largest stored sample of the spike's 50 x "
            "1000 ADC_fullscale_mv / (gain_ch 128), in microvolts",
            "stated",
            "the dacqUSB document's tetrode layout (50 8-bit samples per "
            "channel); ephysiopy's axonaIO scales tetrode waveforms by "
            "ADC_fullscale_mv / 1000 / gain / 128 volts with the gains of "
            "channels 4 (tetrode - 1) to 4 (tetrode - 1) + 3",
        ),
        reader=functools.partial(read_waveforms, files=files),
        folder=folder,
        regions={f"T{t}": _region(rat, t) for t in tetrodes},
    )
    gaps = [
        f"{EPOCHS[b.kind]} starts {b.start - (a.start + a.duration):.0f} s after "
        f"{EPOCHS[a.kind]} ends"
        for a, b in pairwise(trials)
    ]
    notes = [
        _paper_watch_out(entry.session, int(np.sum(units["brain_region"] == "CA1"))),
        "The rest and open-field trials are in other tracking windows than the "
        "track, so their positions are not in rec.positions: they are "
        "params['rest_times'] and params['rest_positions'] (and open_field_...), "
        "cm in that trial's own window. The trials are placed on one clock by "
        f"their .set trial_time, to the second ({'; '.join(gaps)}).",
        "Positions are on the camera's axes, whose y runs down the image: with "
        "y up, a plot is mirrored.",
        f"Lengths are longer than the stated apparatus: at the .pos header's "
        f"pixels_per_metre ({track.pixels_per_metre:g}, the .set's "
        f"tracker_pixels_per_metre), the fitted Z is {length:.0f} cm from end to "
        f"end, {length / STATED_LENGTH:.2f} times the stated "
        f"{ARM_LENGTH:g} + {DIAGONAL_LENGTH:g} + {ARM_LENGTH:g} cm. The lab's code "
        "converts pixels with the same header value, so the paper's 2 cm bins "
        "were most likely in these units (inferred).",
        *_tracking_watch_outs(track, times, positions, headings, distance, width),
    ]
    findings = (
        _cluster_watch_out(spikes, trials),
        _sorting_watch_out(spikes, trials),
    )
    notes.extend(finding for finding in findings if finding is not None)
    n_runs = len(runs)
    notes.append(
        f"{n_runs} runs from end to end ({np.sum(direction == 'outbound')} outbound, "
        f"{np.sum(direction == 'inbound')} inbound; the paper: 'rats completed 20 "
        "full laps'), each from leaving one end's 10 cm to reaching the other's."
    )
    durations = stops[:, 1] - stops[:, 0]
    if durations.size:
        notes.append(
            f"{durations.size} corner stops (rec.epochs['corner_stops']: within "
            f"{CORNER_ZONE:g} cm of a corner until {CORNER_LEAVE:g} cm away, our "
            f"choice), {np.mean(durations):.1f} s long on average (SD "
            f"{np.std(durations):.1f} s; Ólafsdóttir et al. 2017: 'mean pause "
            "duration = 10.71 s [SD = 16.49 s]', over their 24 sessions). "
            f"{len(epochs['engaged'])} engaged and {len(epochs['disengaged'])} "
            "disengaged sections, the paper's first and last 5 s of a stop and the "
            "time between."
        )
        speed = session_speed(times, positions)
        if speed is not None:
            inside = np.zeros(times.size, dtype=bool)
            for start, stop in stops:
                inside |= (times >= start) & (times <= stop)
            slow = float(np.nanmean(speed[inside] < IMMOBILE_SPEED))
            notes.append(
                f"At the corners the speed is below {IMMOBILE_SPEED:g} cm/s in "
                f"{slow:.1%} of the stops' samples (compute_speed's defaults): "
                "the head LED moves while the rat eats, and the 2017 paper's "
                "events must stay below that speed."
            )
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"rat {rat}",
        apparatus=entry.info.apparatus,
        brain_regions=entry.info.brain_regions,
        clock=(
            "the track trial's Axona clock, from its first sample; the rest and "
            "open-field trials placed by their .set start times"
        ),
        times=times,
        spike_times=spike_times,
        units=units,
        env=env,
        env_kind="inferred",
        env_source=(
            f"the stated Z-track (two {ARM_LENGTH:g} cm runways joined by a "
            f"{DIAGONAL_LENGTH:g} cm diagonal, {RUNWAY_WIDTH:g} cm wide; the "
            f"paper's Online Methods) fitted to the track samples, {BIN_SIZE:g} cm "
            "bins"
        ),
        track_width_quantile=TOLERANCE_QUANTILE if width > RUNWAY_WIDTH else None,
        positions=positions,
        position_units="cm",
        tracking="the head: the big of the two headstage LEDs, by the overhead camera",
        epochs=epochs,
        trials=Intervals.from_arrays(runs[:, 0], runs[:, 1], direction=direction),
        params=params,
        conversions=(
            Conversion(
                "spike_times: each tetrode file's big-endian 4-byte timestamps / "
                "its timebase (96 kHz), plus the trial's start",
                "stated",
                "the dacqUSB document: 'divided by 96000 to get a time in seconds'; "
                "each tetrode header's timebase",
            ),
            Conversion(
                "trial starts: the .set trial_time of the rest and open-field "
                "trials minus the track trial's, to the second",
                "inferred",
                "each Axona file's clock starts at 0 s; the .set files give each "
                "trial's start to the second and its duration, and the trials do "
                "not overlap",
            ),
            Conversion(
                "positions: the big LED's pixels x 100 / pixels_per_metre "
                f"({track.pixels_per_metre:g}), cm; 1023 is NaN",
                "stated",
                "the .pos header's pixels_per_metre; the dacqUSB document: "
                "'If a position wasn't tracked ... the values for x and y will "
                "both be 0x3ff (= 1023)'",
            ),
            Conversion(
                "times: frame k (from 1) at k / 50 s, not the frame counter",
                "stated",
                "the dacqUSB document: 'the Xth sample in the file was taken at "
                "time X/50 seconds', the counter 'should be ignored'",
            ),
            placement,
        ),
        watch_outs=(*WATCH_OUTS.get(entry.session, ()), *notes),
        lfp=_lfp_source(entry, trials, rat, folder),
        marks=marks,
    )
