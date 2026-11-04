"""Grosmark & Buzsáki (2016): rat CA1 on a novel maze, with sleep before and after.

What was recorded: units in dorsal CA1 of four rats, implanted bilaterally
with silicon probes, over about ten hours: sleep in the home cage (PRE),
running for water on a maze in a room the rat had never entered (MAZE), and
sleep again (POST). The Supplementary Materials' Methods give the mazes: a
1.9 m wooden linear maze, a 1.6 m stem with a 15 cm reward area at each end
(4 sessions); a 2.3 m metal linear maze, a 2 m stem with the same reward
areas (1 session); and a 1 m diameter wooden circular maze, run clockwise,
with a 30 cm reward area (3 sessions). Two LEDs 5 cm apart on the
head-mounted Faraday cage were tracked "at 39.69 Hz"; the files sample every
0.0256 s (39.06 Hz).

Sessions are the archives' own files, so a session is one recording on one
maze; ``rec.apparatus`` names the maze the file holds. The default session,
``Achilles_10252013``, is the NWB file NeMoS's place-cell tutorial downloads:
OSF ``hu5ma`` in the "NeMoS Tutorial" project (765 MB), which states no
license. It holds 137 units (120 putative pyramidal ``pE``, 17
interneurons ``pI``), 84 trials, the theta phase at 1250 Hz, and a 1-D
position along the 1.6 m linear maze's stem. The other sessions are the
eight files of DANDI 000044 (CC-BY-4.0, 5 to 9 GB each, most of it LFP),
one per maze session, among them ``sub-Achilles_ses-Achilles-10252013``,
DANDI's copy of the default session. Their files hold 2-D positions, the
authors' linearized position, the PRE/MAZE/POST epochs and a sleep scoring.
Load them with ``stream=True`` to read them by byte range instead of
downloading them.

Sources read: the paper and its Supplementary Materials (Science 351,
1440-1443), NeMoS's place-cell tutorial, the OSF file's metadata, DANDI
000044's metadata, and the nine files.

What the loader converts:

- DANDI positions and linearized positions: metres (the files' unit) times
  100, cm. Untracked samples are NaN in the files and stay NaN.
- the OSF position: as stored, in cm (the file's unit), as one column.
- DANDI epochs: ``PREEpoch``, ``MazeEpoch`` and ``POSTEpoch`` become
  ``pre``, ``maze`` and ``post``; the ``states`` table in
  ``processing/behavior`` becomes ``awake``, ``non_rem`` and ``rem``.
- OSF epochs: the file has none. ``pre``, ``maze`` and ``post`` are those of
  DANDI's copy of the session, on the same clock (an inferred
  ``Conversion``); ``maze_before_track`` is the start of the maze epoch
  before the file's first position sample, when DANDI's copy shows the rat
  off the track.
- headings: none. The files hold one position per sample, not the two LEDs.

The declared environment:

- the OSF session: the stated 1.6 m stem as a 1-D track, placed at the
  samples' midrange, in 2 cm bins. Its positions are the head's distance
  along the track and cover the stem only.
- a DANDI linear-maze session: the stated maze, a 15 cm reward area, the
  stem and another 15 cm reward area, as a straight track graph along the
  line the file's linearized position traces in 2-D (a least-squares fit
  from the linearized position to the positions, the stem centred on it).
- a DANDI circular-maze session: the stated 1 m diameter circle as a closed
  loop of 158 segments of 1.99 cm, one bin each, centred by a robust circle
  fit to the samples the file's linearized position covers.
  ``env.to_linear`` runs counter-clockwise from east of the centre, in the
  file's coordinates. ``rec.linearized`` is the authors' coordinate: it
  leaves out the reward area, and in the three files it increases
  counter-clockwise while the rat runs clockwise, so it decreases along a
  run (a watch-out gives each session's numbers).

No source read states the tracks' width. A DANDI track is built with a
placement tolerance measured from the samples (``track_width``, an inferred
``Conversion``): twice a high quantile of the tracked samples' distance from
the track, and at least one bin, so that a known share of the samples falls
in no bin by construction. On a circle the quantile is the 99th. On a linear
maze it is the 95th: in some sessions the rat spends the first minutes off
the track, a metre or more from it (Achilles 10252013, about 110 s, 3.5% of
the samples), and the 99th would make the track 3 m wide. The OSF session's
1-D stem has no tolerance: its positions lie along the track by
construction.

Analysis choices (``params``), from the Supplementary Materials' place-field
analysis: ``bin_size`` 2 cm, and ``min_speed`` 5 cm/s ("restricted to epochs
during which the animal's velocity was at least 5 cm/s"); it also left out
the reward areas and analysed each running direction on a linear maze
separately, which ``params`` does not do. ``dt`` 0.25 s, the decoding bin,
and ``train_epochs`` / ``test_epochs``, alternating 60 s blocks of the
tracked span, so decoding is tested on held-out time.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._params import alternating_blocks
from neurospatial.datasets._recording import Conversion, Recording

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = ["REQUIRES", "WATCH_OUTS", "WORKED_RESULT", "load_session"]

#: The paper's finding this dataset reproduces: level B, a property the
#: paper states, tested against a shifted-spike null
#: (``tests/datasets/test_worked_results.py``).
WORKED_RESULT: dict[str, Any] = {
    "status": "reproduced",
    "level": "B",
    "finding": (
        "the rate-map peaks of each running direction are enriched near the "
        "stem's two ends, beyond a shifted-spike null"
    ),
    "quote": (
        '"Note that the place fields of rigid (blue) and plastic cells (red) '
        'are similarly enriched near the start and goal ends of the track." '
        "(Fig. S2 legend); place-field analysis: 2 cm bins, spikes and "
        "occupancy each smoothed with a 5 cm Gaussian, velocity at least "
        '5 cm/s, and "For linear track maze runs all place field analysis '
        "was carried out independently for left and right directions of "
        'movement" ("Place Field Analysis")'
    ),
    "source": (
        "Grosmark & Buzsaki 2016, Supplementary Materials: Methods, "
        '"Place Field Analysis" (p. 4), and the Fig. S2 legend (p. 14)'
    ),
    "paper": "enriched near the ends (no share stated); 491 place fields over 8 sessions",
    "ours": 0.503,
    "criterion": (
        "the share of directional rate-map peaks within 20 cm of an end "
        "exceeds the 95th percentile of the same share with each unit's spikes "
        "shifted circularly on the tracked time (20 draws, seed 2016); fixed "
        "before the run. A fixed 25% bar is not used: shifted spikes alone "
        "give 22-36%, since occupancy, speed and smoothing crowd peaks at the "
        "ends"
    ),
    "parameters": (
        "the 120 pyramidal ('pE') units; per direction (the sign of each "
        "step's displacement) compute_spatial_rates with method 'binned', "
        "bandwidth 5 cm, min_speed 5 cm/s, on rec.env's 2 cm bins; the maps "
        "peaking at 1 Hz or more, both directions pooled: 50.3% of their "
        "peaks lie within 20 cm of an end; the null "
        "(stats.shuffle_spike_times_circular on runs without a gap over 1 s, "
        "shifts of at least 20 s) spans 22.2% to 36.5%, 95th percentile 36.3%"
    ),
    "unmatched": (
        "the place-cell test: the paper keeps place cells by its occupancy "
        "resampling null (5,000 draws, 5 consecutive bins above the 99th "
        "percentile); ours keeps every pyramidal map peaking at 1 Hz or more, "
        "so the real and null shares come from the same selection",
        "the reward areas: the paper leaves them out; the OSF file's "
        "positions cover the stem only, so nothing is left out",
        "the 5 cm Gaussian is read as its standard deviation",
    ),
    "attempted": None,
    "why": None,
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("pynwb",)

#: The place-field bin of the Supplementary Materials, cm.
BIN_SIZE = 2.0
#: The Supplementary Materials' running threshold, cm/s.
MIN_SPEED = 5.0
DT = 0.25
BLOCK = 60.0
#: The stated stem of each linear maze, cm, by the track name in the DANDI
#: files' processing/behavior containers.
STEMS = {"1.6mLinearMaze": 160.0, "2mLinearMaze": 200.0}
#: The stated reward area at each end of a linear maze, cm.
REWARD_AREA = 15.0
CIRCULAR = "CircularMaze"
#: The stated diameter of the circular maze, cm.
DIAMETER = 100.0
#: The stated reward area of the circular maze, cm.
CIRCLE_REWARD_AREA = 30.0
#: Vertices of the circle's polygon: enough that each segment is one bin, a
#: little under BIN_SIZE (158 segments of 1.99 cm for 2 cm bins); a longer
#: segment would be split into bins shorter still.
N_VERTICES = int(np.ceil(np.pi * DIAMETER / BIN_SIZE))
#: The DANDI tracks' placement tolerance is twice this quantile of the
#: tracked samples' distance from the track: the 99th on the circle; the 95th
#: on a linear maze, whose sessions can start with minutes off the track.
CIRCLE_TOLERANCE_QUANTILE = 0.99
LINEAR_TOLERANCE_QUANTILE = 0.95
_LINEAR_QUANTILE_REASON = (
    "The 95th, not the 99th, percentile, because in some linear-maze sessions "
    "the rat spends the first minutes off the track, a metre or more from it "
    "(Achilles 10252013: about 110 s, 3.5% of the samples, 140 cm and more "
    "away), and "
    "the 99th would make the track 3 m wide."
)
_CIRCLE_QUANTILE_REASON = (
    "The 99th percentile, so that 1% of the samples fall outside: a lower one "
    "leaves out head samples just beside the ring and breaks laps there "
    "(Achilles 11012013: 67 laps at the 99th, 37 at the 95th)."
)
#: Soft-L1 scale of the circle fit, cm: samples farther than this from the
#: ring count less.
CIRCLE_FIT_SCALE = 3.0
#: A head this far from the track, cm, is off it.
OFF_TRACK = 20.0
#: A maze session whose head first reaches the track later than this, s,
#: gets a watch-out.
LATE_START = 10.0
#: A linearized position whose slope against the 2-D track differs from 1 cm
#: per cm by more than this share is reported as rescaled.
RESCALED = 0.01
#: The stated LED tracking rate, Hz, which no file matches.
PAPER_RATE = 39.69

_SUPPLEMENT = "the Supplementary Materials' Methods of Grosmark & Buzsáki 2016"
_DANDI_ASSET = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"

#: The epochs of DANDI 000044's copy of each OSF session (its
#: intervals/epochs: PREEpoch, MazeEpoch, POSTEpoch), [start, stop) in s.
DANDI_EPOCHS: dict[str, dict[str, tuple[float, float]]] = {
    "Achilles_10252013": {
        "pre": (0.0, 18079.5),
        "maze": (18079.5, 20147.0),
        "post": (20147.0, 34861.1032),
    },
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "Achilles_10252013": (
        "Positions are 1-D: the file holds only the head's distance along the "
        "linear maze's axis, in cm, for the samples on or near the track. Against "
        "DANDI 000044's copy of the session it is 33.10 + 0.99975 x - 0.02299 y "
        "of the 2-D positions at the same times (residual SD 0.04 cm, at most "
        "0.07 cm, over 12,817 samples): a projection onto the track's axis, 1 cm "
        "per cm. The rat's time elsewhere (the start of the maze epoch, the "
        "reward ends beyond the samples' range, stretches far beside the stem) "
        "is left out, not stored as NaN.",
        "This position is not DANDI's linearized position of the same session. "
        "Where both are defined (10,336 samples) it is 8.17 + 0.9626 times "
        "DANDI's (residual SD 0.11 cm): DANDI's 0-160 cm covers 154.2 cm of this "
        "axis, and the two differ by up to 8.4 cm. This file also keeps 2511 "
        "samples where DANDI's is NaN: 694 below and 641 above DANDI's ends "
        "(5.4-8.1 cm and 162.3-165.3 cm here) and 1176 between them, farther "
        "beside the axis. Place-field positions from the two archives are not "
        "interchangeable.",
        "30 position samples, in 5 runs of 2 to 19 samples (the longest from "
        "19,136.42 s), lie on a straight line between their neighbours: DANDI's "
        "copy has no 2-D position there, so this file's values there are "
        "interpolated, not tracked.",
        "The file has no epochs and no sleep scoring. epochs['pre'], ['maze'] "
        "and ['post'] are DANDI 000044's for this session (see conversions); "
        "load sub-Achilles_ses-Achilles-10252013 for its sleep states.",
        "The file also holds the theta phase (acquisition/theta_phase, 1250 Hz, "
        "the whole session), which is not loaded; read it with pynwb or "
        "pynapple.",
    ),
}


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one session, the OSF tutorial file or a DANDI 000044 file.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The cached NWB file, or a DANDI file's URL to read by byte range.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.

    Raises
    ------
    ValueError
        If the file's positions are not in the unit its layout states (cm for
        the OSF file, metres for DANDI's), a DANDI file holds no maze or
        more than one, or its maze is not one the Supplementary Materials
        describe; or the OSF session has no DANDI copy to take epochs from.
    """
    from neurospatial.datasets._fetch import open_nwb

    with open_nwb(os.fspath(path_or_stream)) as nwb:
        if "behavior" in nwb.processing:
            return _dandi_session(entry, path_or_stream, nwb)
        return _osf_session(entry, path_or_stream, nwb)


def _osf_session(
    entry: SessionEntry, source: str | os.PathLike[str], nwb: Any
) -> Recording:
    """The OSF tutorial layout: a 1-D position TimeSeries, units and trials."""
    from neurospatial.io.nwb import read_position, read_trials, read_units

    if entry.session not in DANDI_EPOCHS:
        msg = (
            f"{entry.session}: no DANDI 000044 copy is known to take its epochs "
            f"from; known: {sorted(DANDI_EPOCHS)}"
        )
        raise ValueError(msg)
    position = read_position(nwb, position_name="acquisition/position")
    series = nwb.acquisition["position"]
    if position.units != "cm":
        msg = f"{entry.session}: acquisition/position is in {position.units!r}, not cm"
        raise ValueError(msg)
    units = read_units(nwb)
    trials = read_trials(nwb)
    started = nwb.session_start_time
    times = np.asarray(position.times, dtype=np.float64)
    positions = np.asarray(position.positions, dtype=np.float64).reshape(-1, 1)
    epochs = {
        name: np.array([row], dtype=np.float64)
        for name, row in DANDI_EPOCHS[entry.session].items()
    }
    maze_start = epochs["maze"][0, 0]
    if times[0] > maze_start:
        epochs["maze_before_track"] = np.array([[maze_start, times[0]]])
    env, placement = _declared_stem(positions, entry.session)
    train, test = alternating_blocks(times[0], times[-1], BLOCK)
    copy = f"DANDI 000044's copy of this session (asset {_DANDI_ASSET})"
    return Recording(
        **entry.identity(source),
        animal="rat Achilles",
        apparatus="1.9 m wooden linear maze: a 1.6 m stem between two 15 cm "
        "reward areas",
        brain_regions=_regions(units),
        clock=(
            "the acquisition clock, seconds from the start of the PRE sleep "
            "(the file's session_start_time, "
            f"{started:%Y-%m-%d}, is the date the file was written, not the "
            "recording's 2013-10-25)"
        ),
        times=times,
        spike_times=tuple(np.asarray(t, dtype=np.float64) for t in units.spike_times),
        units=_unit_table(units),
        env=env,
        env_kind="declared",
        env_source=(
            f"the stated {STEMS['1.6mLinearMaze']:g} cm stem of the 1.9 m linear "
            f"maze ({_SUPPLEMENT}) as a 1-D track, {BIN_SIZE:g} cm bins"
        ),
        positions=positions,
        position_units="cm",
        tracking=(
            "the head, from two LEDs 5 cm apart on the head-mounted Faraday cage "
            f"({_SUPPLEMENT}): its distance along the linear maze's axis, on the "
            "track only"
        ),
        epochs=epochs,
        trials=trials,
        params={
            "bin_size": BIN_SIZE,
            "min_speed": MIN_SPEED,
            "dt": DT,
            "train_epochs": train,
            "test_epochs": test,
        },
        conversions=(
            Conversion(
                "positions: acquisition/position as stored, in cm, as one column",
                "stated",
                f"the file's acquisition/position: unit {series.unit!r}, "
                f"description {series.description!r}",
            ),
            Conversion(
                "epochs: 'pre', 'maze' and 'post' are the PREEpoch, MazeEpoch and "
                f"POSTEpoch of {copy}",
                "inferred",
                "the file has no epochs. Checked against DANDI's copy on "
                "2026-11-03: the two files' 8,414,607 spike times of 137 units "
                "are identical, unit by unit, and every position time here is a "
                "DANDI sample time to 5e-8 s, so both are on one clock",
            ),
            Conversion(
                "epochs: 'maze_before_track' runs from the start of the maze epoch "
                f"to the file's first position sample ({times[0]:.6g} s)",
                "inferred",
                f"the file has no position there; in {copy}, 2214 of the 2292 "
                "tracked samples there lie more than 20 cm from the track's axis "
                "(median 156 cm): the rat had not reached the track",
            ),
            placement,
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            *_trials_watch_out(trials, times, positions),
            *_gaps(times),
            _rate_watch_out(),
            _NO_HEADINGS,
        ),
    )


def _dandi_session(
    entry: SessionEntry, source: str | os.PathLike[str], nwb: Any
) -> Recording:
    """A DANDI 000044 file: 2-D and linearized positions, epochs and states."""
    from neurospatial.io.nwb import read_intervals, read_position, read_units

    track = _track(nwb, entry.session)
    behavior = nwb.processing["behavior"]
    paths, stated = {}, {}
    for kind in ("Position", "LinearizedPosition"):
        container = behavior[f"{track}{kind}"]
        (name,) = container.spatial_series
        paths[kind] = f"processing/behavior/{track}{kind}/{name}"
        series = container.spatial_series[name]
        stated[kind] = f"unit {series.unit!r}, conversion {series.conversion:g}"
    position = read_position(nwb, position_name=paths["Position"])
    linear = read_position(nwb, position_name=paths["LinearizedPosition"])
    for read, path in (
        (position, paths["Position"]),
        (linear, paths["LinearizedPosition"]),
    ):
        if read.units != "m":
            msg = f"{entry.session}: {path} is in {read.units!r}, not metres"
            raise ValueError(msg)
    units = read_units(nwb)
    stored = read_intervals(nwb, interval_name="epochs").intervals_table()
    states = read_intervals(
        nwb, interval_name="processing/behavior/states"
    ).intervals_table()
    started = nwb.session_start_time
    subject = nwb.subject.subject_id
    times = np.asarray(position.times, dtype=np.float64)
    positions = np.asarray(position.positions, dtype=np.float64) * 100.0
    linearized = np.asarray(linear.positions, dtype=np.float64).reshape(-1) * 100.0
    epochs = {
        str(label).removesuffix("Epoch").lower(): _rows(
            stored[stored["label"] == label]
        )
        for label in stored["label"].unique()
    }
    for label in states["label"].unique():
        key = str(label).lower().replace("-", "_")
        epochs[key] = _rows(states[states["label"] == label])
    if track == CIRCULAR:
        env, environment, distance = _declared_circle(
            positions, linearized, entry.session
        )
        apparatus = "1 m diameter wooden circular maze"
        maze_watch_outs = (
            _uncovered_loop(positions, linearized, distance),
            _running_direction(times, positions, linearized, environment.centre),
        )
    else:
        stem = STEMS[track]
        env, environment, distance = _declared_linear(
            positions, linearized, stem, entry.session
        )
        apparatus = (
            f"{stem / 100 + 2 * REWARD_AREA / 100:.1f} m "
            f"{'wooden' if stem == 160.0 else 'metal'} linear maze: a "
            f"{stem / 100:g} m stem between two {REWARD_AREA:g} cm reward areas"
        )
        maze_watch_outs = (
            _rescaled_linearization(environment.slope, stem),
            _reward_ends(positions, distance, environment),
        )
    train, test = alternating_blocks(times[0], times[-1], BLOCK)
    return Recording(
        **entry.identity(source),
        animal=f"rat {subject}",
        apparatus=apparatus,
        brain_regions=_regions(units),
        clock=(
            "the acquisition clock, seconds from the file's session start time "
            f"({started:%Y-%m-%d}), the start of the PRE sleep"
        ),
        times=times,
        spike_times=tuple(np.asarray(t, dtype=np.float64) for t in units.spike_times),
        units=_unit_table(units),
        env=env,
        env_kind="declared",
        env_source=environment.source,
        track_width_quantile=environment.width_quantile,
        positions=positions,
        position_units="cm",
        tracking=(
            "the head, from two LEDs 5 cm apart on the head-mounted Faraday cage "
            f"({_SUPPLEMENT}); the files do not say which point between the "
            "LEDs the position is"
        ),
        epochs=epochs,
        linearized=linearized,
        params={
            "bin_size": BIN_SIZE,
            "min_speed": MIN_SPEED,
            "dt": DT,
            "train_epochs": train,
            "test_epochs": test,
        },
        conversions=(
            Conversion(
                f"positions: {paths['Position']} in m x 100, cm",
                "stated",
                f"the file's series: {stated['Position']} (read_position applies "
                "the conversion)",
            ),
            Conversion(
                f"linearized: {paths['LinearizedPosition']} in m x 100, cm",
                "stated",
                f"the file's series: {stated['LinearizedPosition']}, described "
                "as starting at the edge of the reward area and ending at the "
                "opposing edge",
            ),
            Conversion(
                "epochs: PREEpoch, MazeEpoch and POSTEpoch of intervals/epochs as "
                "'pre', 'maze' and 'post'; the states table of processing/behavior "
                "as 'awake', 'non_rem' and 'rem'",
                "stated",
                "the file's intervals/epochs labels and processing/behavior/states",
            ),
            *environment.conversions,
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            *maze_watch_outs,
            *_before_track(times, positions, distance, environment.tolerance),
            *_post_ends_early(epochs, units.spike_times),
            _SLEEP_SCORING,
            _rate_watch_out(),
            _NO_HEADINGS,
        ),
    )


_NO_HEADINGS = (
    "No headings: two LEDs were tracked, but the files hold one position per "
    "sample, so head direction cannot be computed."
)

_SLEEP_SCORING = (
    "The sleep epochs ('awake', 'non_rem', 'rem') are DANDI 000044's "
    "three-state scoring. CRCNS hc-11 publishes the same recordings with its "
    "own scoring: for Achilles 10252013, five states (Wake, Drowsy, NREM, "
    "Intermediate, REM) that do not merge into these three (REM: 28 intervals, "
    "2653 s in hc-11; 22 intervals, 2071 s here). Results that select sleep "
    "states depend on which archive's scoring is used."
)


@dataclass(frozen=True)
class _Track:
    """A declared DANDI track: its text, conversions and placement.

    Attributes
    ----------
    source : str
        ``env_source``: the stated dimensions and where they are stated.
    conversions : tuple of Conversion
        The track as stated, its placement and its tolerance.
    tolerance : float
        ``track_width``, cm.
    width_quantile : float or None
        The quantile of the samples' distance from the track that half of
        ``tolerance`` is; None where the one-bin floor set it instead.
    centre : ndarray, shape (2,)
        The circle's centre, or the stem's midpoint, cm.
    axis : ndarray, shape (2,)
        A linear track's direction, from its start; zeros for the circle.
    slope : float
        A linear track's 2-D cm per linearized cm; NaN for the circle.
    half_stem : float
        Half the stated stem, cm; NaN for the circle.
    """

    source: str
    conversions: tuple[Conversion, ...]
    tolerance: float
    width_quantile: float | None
    centre: NDArray[np.float64]
    axis: NDArray[np.float64]
    slope: float
    half_stem: float


def _track(nwb: Any, session: str) -> str:
    """The maze name of a DANDI file, from its ``<maze>Position`` container."""
    names: list[str] = [
        str(name).removesuffix("Position")
        for name in nwb.processing["behavior"].data_interfaces
        if name.endswith("Position") and not name.endswith("LinearizedPosition")
    ]
    if len(names) != 1:
        msg = (
            f"{session}: processing/behavior holds {len(names)} maze position "
            f"containers ({names}), not one"
        )
        raise ValueError(msg)
    (track,) = names
    if track != CIRCULAR and track not in STEMS:
        msg = (
            f"{session}: maze {track!r} is none of the mazes the Supplementary "
            f"Materials describe ({[*STEMS, CIRCULAR]})"
        )
        raise ValueError(msg)
    return track


def _regions(units: Any) -> tuple[str, ...]:
    """The units' locations (``lCA1``, ``rCA1``), in order of first appearance."""
    table = units.unit_table
    assert table is not None  # read_units always reads the table
    return tuple(dict.fromkeys(table["location"].dropna().astype(str)))


def _unit_table(units: Any) -> Any:
    """``brain_region`` from ``location``, ``label`` from ``cell_type``."""
    table = units.unit_table
    assert table is not None  # read_units always reads the table
    out = table[[]].copy()
    out["brain_region"] = table["location"]
    out["label"] = table["cell_type"]
    return out


def _rows(table: Any) -> NDArray[np.float64]:
    """The ``(n, 2)`` [start, stop) rows of an interval table, sorted by start."""
    rows: NDArray[np.float64] = table[["start_time", "stop_time"]].to_numpy(
        dtype=np.float64
    )
    ordered: NDArray[np.float64] = rows[np.argsort(rows[:, 0], kind="stable")]
    return ordered


def _tracked(positions: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Whether each sample is tracked (every coordinate finite), shape (n_time,)."""
    return np.asarray(np.all(np.isfinite(positions), axis=1))


def _declared_stem(
    positions: NDArray[np.float64], session: str
) -> tuple[Environment, Conversion]:
    """The stated 1.6 m stem as a 1-D track at the samples' midrange."""
    import networkx as nx

    from neurospatial import Environment

    stem = STEMS["1.6mLinearMaze"]
    values = positions[_tracked(positions), 0]
    low, high = float(values.min()), float(values.max())
    centre = (low + high) / 2
    graph = nx.Graph()
    graph.add_node("stem_start", pos=(centre - stem / 2,))
    graph.add_node("stem_end", pos=(centre + stem / 2,))
    graph.add_edge("stem_start", "stem_end", distance=stem)
    env = Environment.from_graph(
        graph,
        [("stem_start", "stem_end")],
        bin_size=BIN_SIZE,
        units="cm",
        name=f"grosmark_2016 {session}",
    )
    placement = Conversion(
        f"environment: the stated {stem:g} cm stem from {centre - stem / 2:.4g} to "
        f"{centre + stem / 2:.4g} cm of the file's position, centred on the "
        f"samples' range; the samples span {high - low:.4g} cm, "
        f"{(high - low) / stem - 1:+.1%} of the stem",
        "inferred",
        "the file states no origin for its position; the stem is centred on the "
        f"samples' range, {low:.4g} to {high:.4g} cm. The reward areas are left "
        "out, as the file's samples are",
    )
    return env, placement


def _segment_distance(
    points: NDArray[np.float64], nodes: NDArray[np.float64], *, closed: bool
) -> NDArray[np.float64]:
    """Each point's distance to a polyline through ``nodes``, shape (n_points,)."""
    starts = nodes if closed else nodes[:-1]
    ends = np.roll(nodes, -1, axis=0) if closed else nodes[1:]
    distance = np.full(len(points), np.inf)
    for start, end in zip(starts, ends, strict=True):  # at most 158 segments
        along = end - start
        t = np.clip((points - start) @ along / (along @ along), 0.0, 1.0)
        distance = np.minimum(
            distance, np.linalg.norm(points - start - t[:, None] * along, axis=1)
        )
    return distance


def _tolerance(
    distance: NDArray[np.float64], quantile: float, reason: str
) -> tuple[float, str, float | None]:
    """``track_width`` from the tracked samples' distance to the track, and why.

    Twice ``quantile`` of ``distance`` (cm), and at least one bin; no source
    states a physical width to floor it with. ``reason`` says why that
    quantile. The third value is ``quantile`` when it set the width, None
    when the one-bin floor did.
    """
    reach = float(np.quantile(distance, quantile))
    width = max(2 * reach, BIN_SIZE)
    evidence = (
        "no source read states the track's width. The tracked head's distance "
        f"from the track: median {np.median(distance):.2f} cm, "
        f"{quantile:.0%} quantile {reach:.3g} cm; "
        f"{np.mean(distance > width / 2):.1%} of tracked samples lie beyond the "
        f"tolerance, by construction, and fall in no bin. {reason} track_width "
        "decides only which samples are placed, not the bins, their lengths or "
        "connectivity"
    )
    return width, evidence, quantile if width == 2 * reach else None


def _declared_linear(
    positions: NDArray[np.float64],
    linearized: NDArray[np.float64],
    stem: float,
    session: str,
) -> tuple[Environment, _Track, NDArray[np.float64]]:
    """The stated linear maze along the line the linearized position traces.

    A least-squares line from the file's linearized position to the 2-D
    positions, over the samples where both are defined, gives the track's
    axis; the stated stem is centred on the stretch the linearized position
    covers, and the reward areas continue it at both ends.

    Returns
    -------
    env : Environment
    track : _Track
    distance : ndarray, shape (n_tracked,)
        Each tracked sample's distance from the track, cm.
    """
    import networkx as nx

    from neurospatial import Environment

    tracked = _tracked(positions)
    both = tracked & np.isfinite(linearized)
    design = np.column_stack([np.ones(both.sum()), linearized[both]])
    (origin, slope), *_ = np.linalg.lstsq(design, positions[both], rcond=None)
    scale = float(np.linalg.norm(slope))
    axis = slope / scale
    middle = origin + slope * 0.5 * float(np.nanmax(linearized))
    off_line = np.abs((positions[both] - origin) @ np.array([-axis[1], axis[0]]))
    half = stem / 2
    names = ("reward_start", "stem_start", "stem_end", "reward_end")
    offsets = (-half - REWARD_AREA, -half, half, half + REWARD_AREA)
    points = middle + np.outer(offsets, axis)
    graph = nx.Graph()
    for name, point in zip(names, points, strict=True):
        graph.add_node(name, pos=(float(point[0]), float(point[1])))
    order = list(itertools.pairwise(names))
    for (a, b), length in zip(order, np.diff(offsets), strict=True):
        graph.add_edge(a, b, distance=float(length))
    distance = _segment_distance(positions[tracked], points, closed=False)
    width, evidence, width_quantile = _tolerance(
        distance, LINEAR_TOLERANCE_QUANTILE, _LINEAR_QUANTILE_REASON
    )
    env = Environment.from_graph(
        graph,
        order,
        bin_size=BIN_SIZE,
        edge_spacing=0.0,
        units="cm",
        name=f"grosmark_2016 {session}",
        track_width=width,
    )
    length = stem + 2 * REWARD_AREA
    track = _Track(
        source=(
            f"the stated {length / 100:.1f} m linear maze, a {REWARD_AREA:g} cm "
            f"reward area, a {stem / 100:g} m stem and another {REWARD_AREA:g} cm "
            f"reward area ({_SUPPLEMENT}), as a straight track graph, "
            f"{BIN_SIZE:g} cm bins"
        ),
        conversions=(
            Conversion(
                f"environment: a straight track of three edges, {REWARD_AREA:g}, "
                f"{stem:g} and {REWARD_AREA:g} cm, the stem centred on the middle "
                "of the file's linearized position, along a least-squares line "
                "from the linearized position to the positions",
                "inferred",
                f"{_SUPPLEMENT} state the lengths; the file states no placement. "
                f"The line fits the {both.sum()} samples with a linearized "
                f"position: {scale:.4f} cm of 2-D track per linearized cm, the "
                f"samples a median {np.median(off_line):.2f} cm from it",
            ),
            Conversion(
                f"environment: track_width {width:.3g} cm, a placement tolerance, "
                "not the track's width: twice the tracked samples' "
                f"{LINEAR_TOLERANCE_QUANTILE:.0%} quantile distance from the track",
                "inferred",
                evidence,
            ),
        ),
        tolerance=width,
        width_quantile=width_quantile,
        centre=middle,
        axis=axis,
        slope=scale,
        half_stem=half,
    )
    return env, track, distance


def _declared_circle(
    positions: NDArray[np.float64], linearized: NDArray[np.float64], session: str
) -> tuple[Environment, _Track, NDArray[np.float64]]:
    """The stated 1 m circle, centred by a robust fit to the covered samples.

    Returns
    -------
    env : Environment
    track : _Track
    distance : ndarray, shape (n_tracked,)
        Each tracked sample's distance from the circle's polygon, cm.
    """
    import networkx as nx
    from scipy.optimize import least_squares

    from neurospatial import Environment

    tracked = _tracked(positions)
    covered = positions[tracked & np.isfinite(linearized)]
    start = 0.5 * (covered.min(axis=0) + covered.max(axis=0))
    fit = least_squares(
        lambda x: np.linalg.norm(covered - x[:2], axis=1) - x[2],
        [*start, np.median(np.linalg.norm(covered - start, axis=1))],
        loss="soft_l1",
        f_scale=CIRCLE_FIT_SCALE,
    )
    centre, radius = np.asarray(fit.x[:2]), float(fit.x[2])
    angle = np.linspace(0.0, 2 * np.pi, N_VERTICES, endpoint=False)
    points = centre + DIAMETER / 2 * np.column_stack([np.cos(angle), np.sin(angle)])
    names = [f"p{k}" for k in range(N_VERTICES)]
    graph = nx.Graph()
    for name, point in zip(names, points, strict=True):
        graph.add_node(name, pos=(float(point[0]), float(point[1])))
    order = list(zip(names, names[1:] + names[:1], strict=True))
    for k, (a, b) in enumerate(order):
        graph.add_edge(
            a,
            b,
            distance=float(np.linalg.norm(points[(k + 1) % N_VERTICES] - points[k])),
        )
    distance = _segment_distance(positions[tracked], points, closed=True)
    width, evidence, width_quantile = _tolerance(
        distance, CIRCLE_TOLERANCE_QUANTILE, _CIRCLE_QUANTILE_REASON
    )
    env = Environment.from_graph(
        graph,
        order,
        bin_size=BIN_SIZE,
        units="cm",
        name=f"grosmark_2016 {session}",
        track_width=width,
    )
    track = _Track(
        source=(
            f"the stated {DIAMETER / 100:g} m diameter circular maze ({_SUPPLEMENT}) "
            f"as a closed loop of {N_VERTICES} segments of {env.bin_sizes[0]:.2f} cm, "
            "one bin each"
        ),
        conversions=(
            Conversion(
                f"environment: a circle {DIAMETER:g} cm across centred at "
                f"({centre[0]:.1f}, {centre[1]:.1f}) cm, as {N_VERTICES} segments "
                "counter-clockwise from east of the centre",
                "inferred",
                f"{_SUPPLEMENT} state the diameter; the file states no placement. "
                "The centre is a robust (soft-L1, 3 cm) circle fit to the "
                f"{len(covered)} samples with a linearized position, whose own "
                f"radius is {radius:.1f} cm against the stated {DIAMETER / 2:g} cm",
            ),
            Conversion(
                f"environment: track_width {width:.3g} cm, a placement tolerance, "
                "not the track's width: twice the tracked samples' "
                f"{CIRCLE_TOLERANCE_QUANTILE:.0%} quantile distance from the track",
                "inferred",
                evidence,
            ),
        ),
        tolerance=width,
        width_quantile=width_quantile,
        centre=centre,
        axis=np.zeros(2),
        slope=float("nan"),
        half_stem=float("nan"),
    )
    return env, track, distance


def _rate_watch_out() -> str:
    return (
        f"The paper states LED tracking at {PAPER_RATE:g} Hz ({_SUPPLEMENT}); "
        "the files sample every 0.0256 s, 39.06 Hz, 1.6% slower. The files' "
        "rate is used."
    )


def _rescaled_linearization(slope: float, stem: float) -> str:
    """How the file's linearized position maps onto the 2-D track."""
    rescaled = (
        ", so it is rescaled, not distance along the track"
        if abs(slope - 1) > RESCALED
        else ""
    )
    return (
        "rec.linearized is the file's linearized position: defined only on the "
        f"stem (NaN at the reward ends and off the track) and running 0 to "
        f"{stem:g} cm over {slope * stem:.1f} cm of the 2-D track ({slope:.4f} cm "
        f"of track per linearized cm){rescaled}. env.to_linear gives distance "
        "along the declared track, the reward areas included "
        f"({REWARD_AREA:g} cm before the stem)."
    )


def _reward_ends(
    positions: NDArray[np.float64], distance: NDArray[np.float64], track: _Track
) -> str:
    """The share of the samples placed on the track beyond each end of the stem."""
    placed = positions[_tracked(positions)][distance <= track.tolerance / 2]
    along = (placed - track.centre) @ track.axis
    start = float(np.mean(along < -track.half_stem))
    end = float(np.mean(along > track.half_stem))
    return (
        f"{start:.1%} of the samples placed on the track lie beyond the stem's "
        f"start and {end:.1%} beyond its end: the reward ends, where the rat waits "
        f"for water and its head covers more than the {REWARD_AREA:g} cm reward "
        "area (see the environment's conversions for the samples' spread). "
        "The file's linearized position is NaN there."
    )


def _uncovered_loop(
    positions: NDArray[np.float64],
    linearized: NDArray[np.float64],
    distance: NDArray[np.float64],
) -> str:
    """The share of on-track samples the linearized position leaves out."""
    tracked = _tracked(positions)
    uncovered = ~np.isfinite(linearized[tracked])
    on_track = distance <= OFF_TRACK
    share = float(np.mean(uncovered & on_track))
    covered = linearized[np.isfinite(linearized)]
    return (
        f"rec.linearized, the file's linearized position, runs {covered.min():.0f} "
        f"to {covered.max():.0f} cm, and {share:.1%} of tracked samples lie within "
        f"{OFF_TRACK:g} cm of the circle but have none: it leaves out the "
        f"{CIRCLE_REWARD_AREA:g} cm reward area ({_SUPPLEMENT}: 'starting at the "
        "edge of reward area ... terminating at the opposing edge'), where the "
        "rat spends much of its time. The declared environment is the whole "
        "loop."
    )


def _running_direction(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    linearized: NDArray[np.float64],
    centre: NDArray[np.float64],
) -> str:
    """Which way round the circle the rat runs, and its linearized position grows.

    Steps faster than ``MIN_SPEED`` with the head within ``OFF_TRACK`` of the
    ring count as running; the linearized position's direction is the sign
    of its steps against the angle's, over steps within the covered arc.
    """
    angle = np.arctan2(*(positions - centre).T[::-1])
    turn = np.angle(np.exp(1j * np.diff(angle)))
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
    ring = np.abs(np.linalg.norm(positions - centre, axis=1) - DIAMETER / 2)
    running = np.isfinite(turn) & (speed > MIN_SPEED) & (ring[:-1] < OFF_TRACK)
    clockwise = float(np.mean(turn[running] < 0))
    grows = np.diff(linearized)
    within = np.isfinite(grows) & np.isfinite(turn) & (np.abs(grows) < OFF_TRACK)
    along_turn = float(np.mean(np.sign(grows[within]) == np.sign(turn[within])))
    runs = "clockwise" if clockwise >= 0.5 else "counter-clockwise"
    rises = "counter-clockwise" if along_turn >= 0.5 else "clockwise"
    with_run = runs == rises
    return (
        f"In the file's x-y coordinates the rat runs {runs} in "
        f"{max(clockwise, 1 - clockwise):.0%} of running steps (above "
        f"{MIN_SPEED:g} cm/s), and rec.linearized increases {rises} in "
        f"{max(along_turn, 1 - along_turn):.1%} of its steps, so it "
        f"{'increases' if with_run else 'decreases'} along a run. "
        "The Supplementary Materials' Methods say the rats were made to run "
        "clockwise and that the linearized position increases clockwise. The "
        "file states no reference frame for its coordinates."
    )


def _before_track(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    distance: NDArray[np.float64],
    tolerance: float,
) -> tuple[str, ...]:
    """How long the head stays off the track at the start of the maze epoch.

    Reported when the head first comes within the placement tolerance after
    ``LATE_START`` seconds or more, a median ``OFF_TRACK`` cm or more away.
    """
    on = np.flatnonzero(distance <= tolerance / 2)
    if not on.size:
        return ()
    waited = float(times[_tracked(positions)][on[0]] - times[0])
    away = float(np.median(distance[: on[0]])) if on[0] else 0.0
    if waited < LATE_START or away < OFF_TRACK:
        return ()
    return (
        f"For the first {waited:.0f} s of tracking the rat is not on the track "
        f"(the head a median {away:.0f} cm from it): it starts the maze epoch "
        "elsewhere in the room. Those samples are kept and fall in no bin.",
    )


def _post_ends_early(
    epochs: dict[str, NDArray[np.float64]], spike_times: Any
) -> tuple[str, ...]:
    """When spikes run on past the end of the POST epoch, by how much."""
    trains = [np.asarray(train) for train in spike_times if len(train)]
    if "post" not in epochs or not trains:
        return ()
    end = float(epochs["post"][:, 1].max())
    last = max(float(train[-1]) for train in trains)
    if last - end <= LATE_START:
        return ()
    scored = max(
        float(epochs[k][:, 1].max()) for k in ("awake", "non_rem", "rem") if k in epochs
    )
    return (
        f"The file's POST epoch ends at {end:.6g} s, but spikes run to {last:.6g} s "
        f"and the sleep states to {scored:.6g} s: {last - end:.0f} s of recording "
        "after POST belong to no pre/maze/post epoch.",
    )


def _trials_watch_out(
    trials: Any, times: NDArray[np.float64], positions: NDArray[np.float64]
) -> tuple[str, ...]:
    """What the file's trials are, with their numbers."""
    starts, stops = np.asarray(trials.start), np.asarray(trials.stop)
    durations = stops - starts
    values = positions[:, 0]
    first = values[np.searchsorted(times, starts)]
    last = values[np.searchsorted(times, stops, side="right") - 1]
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    reach = 0.1 * (high - low)
    whole = ((first < low + reach) & (last > high - reach)) | (
        (first > high - reach) & (last < low + reach)
    )
    text = (
        f"rec.trials holds the file's {len(durations)} trials (described only as "
        "'experimental trials'; NeMoS's tutorial uses them as the times the rat "
        f"traversed the track): {int(whole.sum())} start within {reach:.0f} cm of "
        "one end of the samples' range and stop within it of the other, "
        f"{int((durations < 0.1).sum())} last under 0.1 s, {durations.sum():.0f} s "
        "in all."
    )
    if np.all(np.isin(starts, times)) and np.all(np.isin(stops, times)):
        text += (
            " Each starts and stops at a position sample; read as [start, stop), "
            "a trial leaves out its stop sample, which pynapple's inclusive "
            "intervals keep."
        )
    return (text,)


def _gaps(times: NDArray[np.float64]) -> tuple[str, ...]:
    """The stretches with no position sample, longer than a tenth of a second."""
    from neurospatial.datasets._checks import _count

    steps = np.diff(times)
    long = steps > 0.1
    if not long.any():
        return ()
    return (
        f"times has {_count(int(long.sum()), 'gap')} longer than 0.1 s "
        f"({steps[long].sum():.0f} s of the {times[-1] - times[0]:.0f} s tracked "
        f"span, the longest {steps.max():.0f} s): the rat off the track. Spikes "
        "there have no position; nothing is filled in.",
    )
