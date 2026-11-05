"""Grieves et al. (2020): rat CA1 place cells in a 3-D climbing lattice (UCL RDR 9926477).

What was recorded: CA1 tetrodes in 13 rats, recorded wirelessly while each
rat foraged in a square arena, then climbed a cubic lattice maze, then (in 41
of the 57 sessions) foraged in the arena again (the paper's Supplementary
Methods). The lattice is a 6 x 6 x 6 maze of cubes, 0.97 m a side, built of
150 mm tubes 10 mm across; nine rats climbed it "aligned" (bars horizontal
and vertical) and four climbed it "tilted", standing on one vertex (the
paper's Methods and Supplementary Methods). Position is 3-D: five cameras
tracked four infrared LEDs on the headstage, and the head's path was
triangulated from every pair of cameras (Supplementary Methods). Each
session is one rat and date, one ``<rat>_<ddmmyyyy>.mat`` file; this loader
reads its lattice part.

Where it lives: UCL Research Data Repository 9926477 v1 (CC0),
``Summary data.zip`` (12.0 GB). Its members are compressed with Deflate64,
which Python's ``zipfile`` cannot inflate; ``_fetch`` reads the session's
member by HTTP byte range, inflates it with ``inflate64`` and checks its
CRC-32 and SHA-256. Each member is MATLAB v7.3 (HDF5), holding one
``sdata`` struct that the record's ``Summary data readme.m`` describes.

Sources read: the paper (Grieves, Jedidi-Ayoub, Mishchanchuk, Liu,
Renaudineau & Jeffery 2020, Nature Communications 11:789,
doi:10.1038/s41467-020-14611-7, main text on Europe PMC, PMC7005894) and its
Supplementary Information (Supplementary Methods and Table 1), the record's
``Summary data readme.m``, and the files.

What the loader converts:

- positions: ``sdata.lattice_3d.pox/poy/poz``, in mm, divided by 10 to cm.
  No source read states the unit. It is mm because the speed of the stored
  positions is 10 times the stored velocity ``pov`` (median ratio, checked
  on every load where ``pov`` is stored, refused outside 9 to 11), whose
  median is a climbing rat's speed in cm/s; ``config3.lat_size`` is
  ``[970 970 970]`` and every bar is 970 long, against the Supplementary
  Methods' 0.97 m lattice.
- times: ``lattice_3d.pot``, seconds on the session's clock: the session's
  recordings (``part_config``'s ``square1``, ``lattice`` and ``square2``
  rows, named by ``session_names``) end to end from 0 s. The file's rate,
  ``position_srate_hz``, is 50 Hz, and ``pot`` steps 0.02 s.
- units: every ``r<rat>_<date>_t<tetrode>_c<cluster>`` field with spike
  times (s, the same clock), named ``t<tetrode>c<cluster>``, labelled with
  the field's ``automatic_cell_name``. Cluster 0 of each tetrode is left
  out: those fields hold only spike times, with no label and none of the
  per-cell analyses every other cluster has. Without them the sessions of
  each rat hold the cells Supplementary Table 1 counts for it (1064 in all).
- headings: ``sdata.lattice.poh``, degrees to radians, turned into the
  positions' frame (an inferred ``Conversion``: no source read gives its
  unit or frame). ``poh`` runs from 0 to 360, so degrees, and is taken to be
  an azimuth in the frame of the 2-D positions ``lattice.pox/poy``, which
  hold the same path as the 3-D x and y in another frame. The similarity map from the 2-D to the 3-D x, y (a
  rotation or reflection, a scale and a shift) is fitted over the samples
  the two series share, and its rotation or reflection turns the azimuth.
  Where the 2-D positions are the 3-D x, y mirrored in y and shifted,
  exactly (the default session), this is ``-poh``. A sample whose ``poh``
  and 2-D speed ``lattice.pov`` are both exactly 0 has no azimuth: its
  heading is NaN (an inferred ``Conversion``: no source read says what a 0
  means). In every session ``pov`` is exactly 0 only where ``poh`` is,
  while the head keeps moving in no preferred direction; a 0 with ``pov``
  not 0 is kept. The remaining headings are then compared with the direction
  of movement; a session where they disagree keeps the turned azimuth and
  says the frame is not verified.
- epochs: ``square1``, ``lattice`` and ``square2`` from
  ``part_config.<part>.times``, a row per recording. Only the lattice part
  is tracked here: the square parts are a different apparatus, so they are
  epochs without positions, holding the spikes recorded there. A part that
  runs past the recording (its first to last sample or spike: in two
  sessions the sorted units stop firing before ``square2`` ends) is clipped
  to it (``clip_epochs_to_recording``, an inferred ``Conversion`` and a
  watch-out), so a per-epoch rate over it covers the data, not the listed
  part. ``params["analysis_epochs"]`` and the decoding blocks come from the
  clipped rows.
- ``params["lattice_bars"]``: ``lattice_3d.lx/ly/lz``, the lattice's 147
  bars in the positions' frame (start, stop and a NaN separator per bar),
  divided by 10 to cm, shape (147, 2, 3), for drawing the lattice; the
  loader checks there are 147 bars (7 x 7 along each of three axes) of
  97 cm.

The environment is every voxel of the authors' 3-D rate-map grid, 5 cm
voxels (the paper's "50 x 50 x 50 mm") in the positions' frame:
``lattice_3d.dwellmap`` has the grid's shape (MATLAB order y, x, z), and
``lattice_3d.lxrm/lyrm/lzrm`` are the bars in its voxel units, ``lx / 50 +
c`` exactly, which places it. Voxel ``i`` (1-based) spans ``[i - 1, i]`` in
those units: in every session the authors' kernel dwell time recomputed on
that grid matches the stored ``dwellmap`` better than on grids half a voxel
either way. The grid spans the tracked samples padded by 1.5 to 2.5 voxels,
equally at both ends of each axis, in every session: it is the authors'
analysis volume, sized from the samples, not the apparatus, so an inferred
environment. It holds the air around the bars a
climbing rat cannot reach, so most voxels are never visited and have NaN
rates, not zero.

Analysis choices (``params``):

- ``bin_size`` 5 cm, the paper's voxel;
- ``min_speed`` None: the paper's rate maps state no speed filter;
- ``analysis_epochs`` the lattice part; ``train_epochs`` / ``test_epochs``
  alternating 60 s blocks of each of its recordings, for held-out decoding;
- ``dt`` 0.25 s, as for the other datasets.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._checks import MIN_HELD_REPEATS
from neurospatial.datasets._params import alternating_blocks
from neurospatial.datasets._recording import (
    Conversion,
    Recording,
    clip_epochs_to_recording,
    heading_vs_movement,
    tracked_mask,
)

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = [
    "EXPECTED_WATCH_OUTS",
    "REQUIRES",
    "WATCH_OUTS",
    "WORKED_RESULT",
    "held_runs",
    "load_session",
]

#: The paper's finding this dataset reproduces, or why there is none yet.
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
    "attempted": None,
    "why": (
        "Every result read in the paper pools sessions: the main text over "
        "the 57 sessions of 13 rats, and Supplementary Table 1 per rat over "
        "all of its sessions (rat 926: 9 sessions, 295 cells, 232 place "
        "cells in the tilted lattice). A cluster is a place cell by its "
        "waveform's peak-to-trough width, its mean rate and its spatial "
        "information in the session with the most spikes (Methods, 'Place "
        "cell criteria'); the loader gives one session at a time and reads "
        "no waveforms, so no published result can be reproduced from one "
        "session yet. The cell counts per rat do match Supplementary Table 1 "
        "when every session of the rat is loaded; that checks the units "
        "the loader keeps, not a finding."
    ),
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("h5py", "fsspec", "inflate64")

#: The analysis bin, cm: the paper's 50 mm voxel.
BIN_SIZE = 5.0
DT = 0.25
BLOCK = 60.0
#: Stored length units per cm (the file's mm).
MM_PER_CM = 10.0
#: The lattice's side, cm (Supplementary Methods: 0.97 x 0.97 x 0.97 m).
LATTICE_SIDE = 97.0
#: 6 x 6 x 6 cubes: 7 x 7 bars along each of the three axes.
N_BARS = 147
#: The authors' voxel in the file's mm (the paper: 50 x 50 x 50 mm).
VOXEL_MM = 50.0
#: The file's ``config3.lattice_type`` against the paper's names.
CONFIGURATIONS = {"normal": "aligned", "diagonal": "tilted"}
#: Steps faster than this, cm/s, count as moving when heading is compared
#: with the direction of movement.
HEADING_SPEED = 10.0
#: A heading in the positions' frame: heading minus the direction of
#: movement has a circular mean within this of 0 (rad) ...
HEADING_MEAN = 0.2
#: ... and a median absolute deviation from it under this (rad).
HEADING_SPREAD = float(np.radians(45.0))
#: Below this RMS, mm, with a scale of 1 to this many decimals, the 2-D
#: positions are the 3-D x, y themselves in another frame, not a fit.
EXACT_RMS = 0.01
EXACT_SCALE_DECIMALS = 6
#: Position speed over the stored velocity ``pov``: about 10 for mm against
#: cm/s.
SPEED_RATIO = (9.0, 11.0)
#: A bar's length may differ from the lattice's side by this, cm.
BAR_TOLERANCE = 0.1
#: Largest residual of ``lxrm`` against ``lx / 50 + c``, in voxels.
GRID_RESIDUAL = 1e-6

_CELL = re.compile(r"r\d+_\d+_t(\d+)_c(\d+)$")
_PARTS = ("square1", "lattice", "square2")
_SUPPLEMENT = "the Supplementary Methods of Grieves et al. 2020"

#: Findings computed from the data (by the checks or this loader) that each
#: verified session must report, as patterns searched for in its watch_outs.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "926_06042018": (
        r"have lattice\.poh and lattice\.pov both exactly 0",
        r"samples with a heading repeat the sample before",
        r"lie outside the lattice's cube",
        r"stored in single precision",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "926_06042018": (
        "Three units stop firing partway through the lattice part (1816 to "
        "5458 s) and never fire again: t14c4 after 3259.0 s, t14c5 after "
        "2762.4 s and t14c6 after 3419.7 s; t15c1 first fires at 1571.8 s, late "
        "in square1. A rate map over the whole lattice part averages in time "
        "they did not fire; the file does not say whether the clusters were "
        "lost or the cells stopped.",
        "units['label'] is the file's automatic_cell_name (Place cell 15, "
        "Pyramidal 9 here), and the file does not say which criteria set it. "
        "The paper classifies a cluster as a place cell by its waveform width, "
        "mean rate and spatial information in the session with the most "
        "spikes (Methods, 'Place cell criteria').",
    ),
}

_PROCESSED = (
    "Positions are the authors' 3-D reconstruction, not raw tracking: "
    f"{_SUPPLEMENT} say the cameras tracked the LEDs at 25 Hz, the paths "
    "from every pair of cameras were combined into one, and 'for segments "
    "of missing tracking data, we simultaneously interpolated and smoothed "
    "the existing data'. The file holds them at 50 Hz, so the tracked share "
    "and speeds describe that processing as much as the tracking."
)
_SQUARES = (
    "Only the lattice part is loaded. The square arena parts (epochs "
    "'square1' and 'square2') are a different apparatus, so their positions "
    "are not in rec.positions; their spikes are in rec.spike_times, so "
    "restrict analyses to params['analysis_epochs'] (the lattice)."
)
_CLOCK = (
    "The clock joins the session's recordings end to end: the rests of "
    f"about 10 minutes between the parts ({_SUPPLEMENT}) are not on it, so a "
    "time difference across a part boundary is not elapsed time."
)


@dataclass(frozen=True)
class _Session:
    """What the loader reads from one session file, in the file's units."""

    rat: str
    lattice_type: str
    times: NDArray[np.float64]
    stored: NDArray[np.float64]
    velocity: NDArray[np.float64]
    bars: NDArray[np.float64]
    lat_size: NDArray[np.float64]
    grid_shape: tuple[int, int, int]
    voxel_axes: NDArray[np.float64]
    flat_times: NDArray[np.float64]
    flat: NDArray[np.float64]
    azimuth: NDArray[np.float64]
    #: ``lattice.pov``, the 2-D speed, at each 2-D sample.
    flat_speed: NDArray[np.float64]
    parts: dict[str, NDArray[np.float64]]
    recordings: tuple[str, ...]
    sampling_rate: float
    units: list[tuple[int, int, NDArray[np.float64], str | None]]
    n_unsorted: int
    #: The HDF5 dtypes of ``lattice_3d.pot`` and of each unit's spike times.
    time_dtype: str
    spike_dtypes: tuple[str, ...]


@dataclass(frozen=True)
class _FrameMap:
    """The similarity map from the 2-D positions to the 3-D x, y.

    ``x3 = scale * rotation @ x2 + shift``, fitted over ``n_paired`` samples
    whose times the two series share.
    """

    rotation: NDArray[np.float64]
    scale: float
    shift: NDArray[np.float64]
    rms: float
    r_squared: float
    n_paired: int
    held: float

    @property
    def exact(self) -> bool:
        """Whether the 2-D positions are the 3-D x, y in another frame, not a fit."""
        return self.rms < EXACT_RMS and round(self.scale, EXACT_SCALE_DECIMALS) == 1

    def describe(self) -> str:
        """``"a mirror in y"``, ``"a rotation by 90 deg"``, ...: the map's turn."""
        angle = float(np.degrees(np.arctan2(self.rotation[1, 0], self.rotation[0, 0])))
        if np.linalg.det(self.rotation) > 0:
            return f"a rotation by {angle:.1f} deg"
        axis = angle / 2
        if abs(axis) < 0.05:
            return "a mirror in y (a reflection across the x axis)"
        return f"a reflection across the line at {axis:.1f} deg to the x axis"


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one session file's lattice part into a 3-D :class:`Recording`.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The directory holding the fetched zip member.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.

    Raises
    ------
    ValueError
        If the file contradicts what the conversions rest on: the position
        speed is not about 10 times ``pov``, the lattice is not 147 bars of
        97 cm, or the bars' voxel coordinates are not ``lx / 50 + c``.
    """
    (member,) = entry.members
    raw = _read(Path(path_or_stream) / Path(member.path).name)
    times = raw.times
    ratio = _speed_ratio(raw)
    positions = raw.stored / MM_PER_CM
    bars = _bars_cm(raw.bars)
    frame = _frame_map(raw)
    lost = _lost(raw)
    headings = _turned_headings(raw, frame, lost)
    offset, spread = heading_vs_movement(
        times, positions[:, :2], headings, min_speed=HEADING_SPEED
    )
    verified = abs(offset) <= HEADING_MEAN and spread < HEADING_SPREAD
    env, grid, in_grid = _authors_grid(raw, positions, entry.session)
    configuration = CONFIGURATIONS.get(raw.lattice_type, raw.lattice_type)
    spike_times = tuple(spikes for _, _, spikes, _ in raw.units)
    clipped = clip_epochs_to_recording(
        raw.parts, times, spike_times, source="the file's part_config"
    )
    lattice = clipped.epochs["lattice"]
    blocks = [alternating_blocks(start, stop, BLOCK) for start, stop in lattice]
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"rat {raw.rat}",
        apparatus=(
            f"{entry.info.apparatus}; this session the {configuration} lattice "
            f"(config3.lattice_type {raw.lattice_type!r})"
        ),
        brain_regions=entry.info.brain_regions,
        clock=(
            f"the session's clock: its recordings ({', '.join(raw.recordings)}) "
            "end to end from 0 s, as the file's part_config places them"
        ),
        times=times,
        spike_times=spike_times,
        units=_unit_table(raw),
        env=env,
        env_kind="inferred",
        env_source=(
            f"the authors' 3-D rate-map grid, every voxel: {grid}; it spans "
            "the tracked samples padded by about two voxels"
        ),
        positions=positions,
        position_units="cm",
        tracking=(
            "the head: four infrared LEDs on the headstage, seen by five "
            f"cameras and triangulated in 3-D ({_SUPPLEMENT})"
        ),
        headings=headings,
        heading_source=(
            "lattice.poh, the file's head azimuth, turned into the positions' "
            "frame, NaN where poh and pov are both 0"
            + ("" if verified else "; the frame is not verified by movement")
        ),
        epochs=clipped.epochs,
        params={
            "bin_size": BIN_SIZE,
            "dt": DT,
            "min_speed": None,
            "analysis_epochs": lattice,
            "train_epochs": np.concatenate([train for train, _ in blocks]),
            "test_epochs": np.concatenate([test for _, test in blocks]),
            "lattice_bars": bars,
        },
        conversions=(
            Conversion(
                "positions: lattice_3d pox, poy, poz / 10 (mm to cm)",
                "inferred",
                "no source read states the unit; "
                + (
                    "pov holds no value here, so it cannot be compared"
                    if ratio is None
                    else f"the stored positions' speed is {ratio:.2f} times the "
                    "stored velocity pov (median ratio), whose median "
                    f"{np.nanmedian(raw.velocity):.3g} is a climbing rat's speed "
                    "in cm/s"
                )
                + "; config3.lat_size is "
                f"{' x '.join(f'{v:g}' for v in raw.lat_size)} and every bar is "
                f"{np.median(_bar_lengths(bars)) * MM_PER_CM:.0f} long, against "
                f"the 0.97 m lattice of {_SUPPLEMENT}",
            ),
            Conversion(
                "times: lattice_3d pot as stored, seconds on the session's clock",
                "stated",
                f"the file: position_srate_hz {raw.sampling_rate:g}, and pot steps "
                f"{np.median(np.diff(times)):.4g}",
            ),
            Conversion(
                f"spike_times: the {len(raw.units)} tetrode/cluster fields' "
                "spike_times as stored (s); cluster 0 of each tetrode "
                f"({raw.n_unsorted} spikes) left out",
                "inferred",
                "the cluster-0 fields hold only spike_times, with no "
                "automatic_cell_name and none of the per-cell analyses every "
                "other cluster has; without them every rat's sessions hold the "
                "cells Supplementary Table 1 of Grieves et al. 2020 counts for "
                "it (1064 in all, every session read when the loader was written)",
            ),
            _heading_conversion(raw, frame, offset, spread, verified),
            *_lost_conversion(raw, lost, times, positions, headings),
            Conversion(
                "epochs: part_config.<part>.times as stored (s), a row per recording",
                "stated",
                "the file's part_config places the session's recordings "
                f"({', '.join(raw.recordings)}, its session_names) on its clock",
            ),
            *(() if clipped.conversion is None else (clipped.conversion,)),
            Conversion(
                f"params['lattice_bars']: lattice_3d lx, ly, lz / 10, {len(bars)} "
                f"bars of {LATTICE_SIDE:g} cm, shape {bars.shape}",
                "stated",
                "the readme: 'Maze limits/boundaries/outlines are consistently "
                f"stored as lx, ly & lz'; {_SUPPLEMENT}: a 6 x 6 x 6 cubic maze, "
                f"0.97 m a side; config3.lattice_type {raw.lattice_type!r}",
            ),
            Conversion(
                f"environment: every voxel of the authors' rate-map grid, {grid}",
                "inferred",
                "lattice_3d.dwellmap's shape gives the grid; lxrm, lyrm and lzrm "
                f"are lx, ly and lz / {VOXEL_MM:g} + c exactly, the paper's "
                f"{VOXEL_MM:g} mm voxels, which places it; voxel i (1-based) is "
                "taken to span [i - 1, i] in those units, where in every session "
                "the authors' kernel dwell time matches the stored dwellmap best "
                "(checked when the loader was written); the grid's extent comes "
                "from the authors' processing of the samples, not from the "
                "apparatus",
            ),
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            *(() if clipped.watch_out is None else (clipped.watch_out,)),
            *_parts_past_the_spikes(raw.parts, clipped.epochs, times, spike_times),
            *_unknown_configuration(raw),
            *_unverified_heading(frame, offset, spread, verified),
            *_lost_and_zero_azimuths(raw, frame, lost),
            *_held_headings(times, positions, headings),
            *_outside_cube(positions, bars, in_grid),
            *_recording_days(raw),
            *_single_precision(times, raw),
            _SQUARES,
            _CLOCK,
            _PROCESSED,
        ),
    )


def _text(dataset: Any) -> str:
    """A MATLAB v7.3 char array as a string."""
    return "".join(map(chr, np.asarray(dataset[()]).ravel()))


def _vector(group: Any, name: str) -> NDArray[np.float64]:
    """A stored array as a flat float64 vector."""
    return np.asarray(group[name][()], dtype=np.float64).ravel()


def _read(path: Path) -> _Session:
    """The fields the loader needs from one session file."""
    import h5py

    with h5py.File(path, "r") as mat:
        sdata = mat["sdata"]
        part = sdata["lattice_3d"]
        flat = sdata["lattice"]
        config = sdata["part_config"]
        # A MATLAB (n, 2) array of [start stop] rows reads transposed.
        parts = {
            name: np.asarray(config[name]["times"][()], dtype=np.float64).T
            for name in _PARTS
            if name in config
        }
        recordings = tuple(
            _text(mat[ref]) for ref in sdata["session_names"][()].ravel()
        )
        units = []
        spike_dtypes = []
        n_unsorted = 0
        fields = sorted(
            (int(m[1]), int(m[2]), name) for name in sdata if (m := _CELL.match(name))
        )
        for tetrode, cluster, name in fields:
            cell = sdata[name]
            spikes = _vector(cell, "spike_times")
            if cluster == 0:
                n_unsorted += spikes.size
                continue
            label = (
                _text(cell["automatic_cell_name"])
                if "automatic_cell_name" in cell
                else None
            )
            units.append((tetrode, cluster, spikes, label))
            spike_dtypes.append(str(cell["spike_times"].dtype))
        # h5py reads MATLAB arrays with their axes reversed: (z, x, y).
        n_z, n_x, n_y = part["dwellmap"].shape
        return _Session(
            rat=_text(sdata["rat_num"]),
            lattice_type=_text(sdata["config3"]["lattice_type"]),
            times=_vector(part, "pot"),
            stored=np.column_stack([_vector(part, k) for k in ("pox", "poy", "poz")]),
            velocity=_vector(part, "pov"),
            bars=np.column_stack([_vector(part, k) for k in ("lx", "ly", "lz")]),
            lat_size=_vector(sdata["config3"], "lat_size"),
            grid_shape=(int(n_x), int(n_y), int(n_z)),
            voxel_axes=np.stack(
                [
                    np.column_stack([_vector(part, f"l{a}"), _vector(part, f"l{a}rm")])
                    for a in "xyz"
                ]
            ),
            flat_times=_vector(flat, "pot"),
            flat=np.column_stack([_vector(flat, "pox"), _vector(flat, "poy")]),
            azimuth=np.radians(_vector(flat, "poh")),
            flat_speed=_vector(flat, "pov"),
            parts=dict(sorted(parts.items(), key=lambda item: item[1][0, 0])),
            recordings=recordings,
            sampling_rate=float(_vector(sdata, "position_srate_hz")[0]),
            units=units,
            n_unsorted=n_unsorted,
            time_dtype=str(part["pot"].dtype),
            spike_dtypes=tuple(spike_dtypes),
        )


def _speed_ratio(raw: _Session) -> float | None:
    """The stored positions' speed over ``pov``; None where pov holds no value.

    Refused unless about 10.
    """
    step = np.linalg.norm(np.diff(raw.stored, axis=0), axis=1) / np.diff(raw.times)
    with np.errstate(divide="ignore", invalid="ignore"):  # pov NaN or 0
        ratios = step / raw.velocity[1:]
    if not np.isfinite(ratios).any():
        return None
    ratio = float(np.nanmedian(ratios))
    if not SPEED_RATIO[0] < ratio < SPEED_RATIO[1]:
        msg = (
            f"lattice_3d: the positions' speed is {ratio:.2f} times pov, not about "
            "10, so the positions are not mm against a velocity in cm/s; recheck "
            "the unit before converting"
        )
        raise ValueError(msg)
    return ratio


def _bar_lengths(bars: NDArray[np.float64]) -> NDArray[np.float64]:
    """Length of each bar, shape (n_bars,), from bars (n_bars, 2, 3)."""
    lengths: NDArray[np.float64] = np.linalg.norm(bars[:, 1] - bars[:, 0], axis=1)
    return lengths


def _bars_cm(stored: NDArray[np.float64]) -> NDArray[np.float64]:
    """The lattice's bars in cm, (n_bars, 2, 3), refused unless 147 of 97 cm.

    ``stored`` is (3 n_bars, 3): each bar's start, stop and a separator row
    holding a NaN (all three coordinates, or only some in a few files).
    """
    if len(stored) % 3:
        msg = (
            f"lattice_3d lx/ly/lz: {len(stored)} rows, not (start, stop, NaN) triplets"
        )
        raise ValueError(msg)
    triplets = stored.reshape(-1, 3, 3)
    bars = triplets[:, :2] / MM_PER_CM
    lengths = _bar_lengths(bars)
    if (
        len(bars) != N_BARS
        or not np.isnan(triplets[:, 2]).any(axis=1).all()
        or np.isnan(bars).any()
        or not np.allclose(lengths, LATTICE_SIDE, atol=BAR_TOLERANCE)
    ):
        msg = (
            f"lattice_3d lx/ly/lz: expected {N_BARS} (start, stop, NaN) bars of "
            f"{LATTICE_SIDE:g} cm, got {len(bars)} with lengths "
            f"{np.nanmin(lengths):.1f}-{np.nanmax(lengths):.1f} cm"
        )
        raise ValueError(msg)
    return bars


def _paired(raw: _Session) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Indices into the 3-D and the 2-D samples whose stored times are equal."""
    _, in_3d, in_2d = np.intersect1d(raw.times, raw.flat_times, return_indices=True)
    return in_3d, in_2d


def similarity_map(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """The rotation or reflection, scale and shift taking ``source`` onto ``target``.

    The least-squares fit of ``target ≈ scale * source @ rotation.T + shift``
    with ``rotation`` orthogonal, a reflection allowed (Umeyama's solution
    from the singular value decomposition of the cross-covariance).

    Parameters
    ----------
    source, target : ndarray, shape (n, 2)
        Paired points, finite.

    Returns
    -------
    rotation : ndarray, shape (2, 2)
    scale : float
    shift : ndarray, shape (2,)

    Examples
    --------
    >>> source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    >>> target = source * [1.0, -1.0] * 3 + [5.0, 1.0]
    >>> rotation, scale, shift = similarity_map(source, target)
    >>> # + 0.0 turns -0.0, which some platforms give, into 0.0.
    >>> (
    ...     (np.round(rotation, 6) + 0.0).tolist(),
    ...     round(scale, 6),
    ...     np.round(shift, 6).tolist(),
    ... )
    ([[1.0, 0.0], [0.0, -1.0]], 3.0, [5.0, 1.0])
    """
    centre_s, centre_t = source.mean(axis=0), target.mean(axis=0)
    a, b = source - centre_s, target - centre_t
    left, singular, right = np.linalg.svd(b.T @ a)
    rotation: NDArray[np.float64] = left @ right
    scale = float(singular.sum() / np.sum(a**2))
    shift: NDArray[np.float64] = centre_t - scale * rotation @ centre_s
    return rotation, scale, shift


def _frame_map(raw: _Session) -> _FrameMap:
    """The similarity map from ``lattice.pox/poy`` to ``lattice_3d.pox/poy``."""
    in_3d, in_2d = _paired(raw)
    target, source = raw.stored[in_3d, :2], raw.flat[in_2d]
    both = np.isfinite(target).all(axis=1) & np.isfinite(source).all(axis=1)
    rotation, scale, shift = similarity_map(source[both], target[both])
    residual = target[both] - (scale * source[both] @ rotation.T + shift)
    spread = target[both] - target[both].mean(axis=0)
    held = np.all(np.diff(raw.flat, axis=0) == 0, axis=1)
    rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    if rms < EXACT_RMS and round(scale, EXACT_SCALE_DECIMALS) == 1:
        # One path in two frames: drop the fit's float noise (1e-11), so a
        # mirror in y turns poh into exactly -poh.
        rotation = np.round(rotation, EXACT_SCALE_DECIMALS) + 0.0
    return _FrameMap(
        rotation=rotation,
        scale=scale,
        shift=shift,
        rms=rms,
        r_squared=float(1 - np.sum(residual**2) / np.sum(spread**2)),
        n_paired=int(both.sum()),
        held=float(held.mean()),
    )


def _lost(raw: _Session) -> NDArray[np.bool_]:
    """The 2-D samples without an azimuth: ``poh`` and ``pov`` both exactly 0."""
    lost: NDArray[np.bool_] = (raw.azimuth == 0) & (raw.flat_speed == 0)
    return lost


def _turned_headings(
    raw: _Session, frame: _FrameMap, lost: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """``poh`` turned by the map's rotation or reflection, at each 3-D sample.

    A 3-D sample with no 2-D sample at its time, or whose 2-D sample is
    ``lost`` (shape (n_2d,)), gets NaN.
    """
    in_3d, in_2d = _paired(raw)
    azimuth = np.where(lost, np.nan, raw.azimuth)[in_2d]
    unit = np.column_stack([np.cos(azimuth), np.sin(azimuth)])
    turned = unit @ frame.rotation.T
    headings = np.full(raw.times.size, np.nan)
    headings[in_3d] = np.arctan2(turned[:, 1], turned[:, 0])
    return headings


def _heading_conversion(
    raw: _Session, frame: _FrameMap, offset: float, spread: float, verified: bool
) -> Conversion:
    """The heading's Conversion, inferred: no source read gives poh's unit or frame.

    That ``poh`` is in degrees and in the frame of ``lattice.pox/poy`` is
    inferred from its range and from the turned azimuth following movement;
    where the 2-D positions are the 3-D x, y in another frame exactly, the
    exact map is the evidence for the turn.
    """
    shift = ", ".join(f"{value:.5g}" for value in frame.shift)
    relation = (
        f"lattice_3d.pox, poy are lattice.pox, poy under {frame.describe()} "
        f"plus a shift of ({shift}) mm, sample by sample (RMS {frame.rms:.2g} mm)"
    )
    if not frame.exact:
        relation = (
            f"the fit of lattice_3d.pox, poy onto lattice.pox, poy over "
            f"{frame.n_paired} paired samples is {frame.describe()}, a scale of "
            f"{frame.scale:.4g} mm per 2-D unit and a shift of ({shift}) mm, with "
            f"RMS {frame.rms:.3g} mm (R^2 {frame.r_squared:.3f}): the same path "
            "seen in the 2-D frame, which no source read describes"
        )
    finite = raw.azimuth[np.isfinite(raw.azimuth)]
    return Conversion(
        "headings: lattice.poh, degrees to radians, turned by the 2-D to 3-D "
        "map's rotation or reflection into the positions' frame"
        + (" (here -poh)" if frame.exact and np.linalg.det(frame.rotation) < 0 else ""),
        "inferred",
        "no source read gives poh's unit or frame; it is taken to be an "
        f"azimuth in the frame of lattice.pox, poy, and {relation}; poh runs "
        f"from {np.degrees(finite.min()):.3g} to {np.degrees(finite.max()):.4g}, "
        "so degrees; against the direction of "
        f"movement above {HEADING_SPEED:g} cm/s, heading minus movement has "
        f"circular mean {np.degrees(offset):.1f} deg and median absolute "
        f"deviation {np.degrees(spread):.1f} deg"
        + (
            ", a heading in the positions' frame"
            if verified
            else ", outside what a heading in the positions' frame meets: the "
            "frame is not verified"
        ),
    )


def _authors_grid(
    raw: _Session, positions: NDArray[np.float64], session: str
) -> tuple[Environment, str, NDArray[np.bool_]]:
    """The authors' rate-map grid in cm, every voxel active, its description, and which tracked samples it holds.

    ``l<a>rm = l<a> / 50 + c`` along each axis; voxel ``i`` (1-based) spans
    ``[i - 1, i]``, so the edges are ``(k - c) * 50`` mm for ``k = 0 .. n``.
    The mask, shape (n_tracked,), is over the tracked samples in order.
    """
    from neurospatial import Environment

    edges: list[NDArray[np.float64]] = []
    for index, (axis, n_voxels) in enumerate(zip("xyz", raw.grid_shape, strict=True)):
        pairs = raw.voxel_axes[index]
        pairs = pairs[np.isfinite(pairs).all(axis=1)]
        offset = float(np.median(pairs[:, 1] - pairs[:, 0] / VOXEL_MM))
        residual = float(np.max(np.abs(pairs[:, 0] / VOXEL_MM + offset - pairs[:, 1])))
        if residual > GRID_RESIDUAL:
            msg = (
                f"lattice_3d l{axis}rm is not l{axis} / {VOXEL_MM:g} + c (largest "
                f"residual {residual:.3g} voxels), so the authors' grid cannot be "
                "placed from it"
            )
            raise ValueError(msg)
        voxel_edges = np.arange(n_voxels + 1, dtype=np.float64) - offset
        edges.append(voxel_edges * (VOXEL_MM / MM_PER_CM))
    env = Environment.from_grid_mask(
        np.ones(raw.grid_shape, dtype=bool),
        grid_edges=tuple(edges),
        units="cm",
        name=f"grieves_2020 {session}",
    )
    span = ", ".join(
        f"{axis} {e[0]:.4g} to {e[-1]:.4g}"
        for axis, e in zip("xyz", edges, strict=True)
    )
    tracked = positions[tracked_mask(positions)]
    low, high = [e[0] for e in edges], [e[-1] for e in edges]
    inside = np.all((tracked >= low) & (tracked <= high), axis=1)
    description = (
        f"{' x '.join(map(str, raw.grid_shape))} voxels of {BIN_SIZE:g} cm, "
        f"{span} cm, holding {inside.mean():.2%} of the tracked samples"
    )
    return env, description, inside


def _unit_table(raw: _Session) -> pd.DataFrame:
    """One row per sorted cluster: CA1, the file's automatic_cell_name as label."""
    import pandas as pd

    return pd.DataFrame(
        {
            "brain_region": "CA1",
            "label": [label for *_, label in raw.units],
            "tetrode": [tetrode for tetrode, *_ in raw.units],
        },
        index=pd.Index(
            [f"t{tetrode}c{cluster}" for tetrode, cluster, *_ in raw.units],
            name="unit_id",
        ),
    )


def _parts_past_the_spikes(
    parts: dict[str, NDArray[np.float64]],
    clipped: dict[str, NDArray[np.float64]],
    times: NDArray[np.float64],
    spike_times: tuple[NDArray[np.float64], ...],
) -> tuple[str, ...]:
    """A watch-out for each part the file lists as longer than its data.

    Only the lattice is tracked, so a square part's data end with its last
    spike; a part clipped to the recording keeps the data's span, not the
    listed one.
    """
    spikes = np.concatenate([np.empty(0), *spike_times])
    first = float(min(times[0], spikes.min())) if spikes.size else float(times[0])
    last = float(max(times[-1], spikes.max())) if spikes.size else float(times[-1])
    found = []
    for name, rows in parts.items():
        kept = clipped.get(name)
        if kept is not None and np.array_equal(kept, rows):
            continue
        listed = ", ".join(f"[{a:.6g}, {b:.6g}]" for a, b in rows)
        edges = []
        if rows.min() < first:
            edges.append(f"start firing (first spike {first:.6g} s) after it begins")
        if rows.max() > last:
            edges.append(f"stop firing (last spike {last:.6g} s) before it ends")
        found.append(
            f"part_config lists {name} as {listed} s, longer than the data: the "
            "square parts hold no position sample, and the sorted units "
            f"{' and '.join(edges)}. epochs[{name!r}] is clipped to the data, so a "
            "per-epoch rate over it covers the data, not the listed part."
        )
    return tuple(found)


def _unknown_configuration(raw: _Session) -> tuple[str, ...]:
    """A watch-out when ``config3.lattice_type`` is not a configuration the paper names."""
    if raw.lattice_type in CONFIGURATIONS:
        return ()
    return (
        f"config3.lattice_type is {raw.lattice_type!r}, which no source read "
        "describes (the paper's configurations are aligned, 'normal' in the "
        "files, and tilted, 'diagonal'); its bars are those of the lattice "
        "the summary draws.",
    )


def _unverified_heading(
    frame: _FrameMap, offset: float, spread: float, verified: bool
) -> tuple[str, ...]:
    """A watch-out when heading minus movement is not that of a heading."""
    if verified:
        return ()
    return (
        "The heading frame is not verified by movement: the turned azimuth "
        f"minus the direction of movement above {HEADING_SPEED:g} cm/s has "
        f"circular mean {np.degrees(offset):.1f} deg and median absolute "
        f"deviation {np.degrees(spread):.1f} deg (a heading in the positions' "
        f"frame is within {np.degrees(HEADING_MEAN):.1f} deg and under "
        f"{np.degrees(HEADING_SPREAD):.0f} deg). The 2-D positions it is turned "
        f"by fit the 3-D path with R^2 {frame.r_squared:.3f}, and "
        f"{frame.held:.1%} of their steps hold still. The values are kept; "
        "preferred directions may be offset, mirrored or meaningless.",
    )


def _lost_conversion(
    raw: _Session,
    lost: NDArray[np.bool_],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
) -> tuple[Conversion, ...]:
    """The inferred Conversion of the samples without an azimuth to NaN, with this session's evidence.

    ``lost``, shape (n_2d,), marks the 2-D samples whose ``poh`` and ``pov``
    are both exactly 0; ``headings`` are the turned headings, NaN there.
    """
    if not lost.any():
        return ()
    zero = raw.azimuth == 0
    still = (raw.flat_speed == 0) & ~zero
    in_3d, in_2d = _paired(raw)
    lost_3d = np.zeros(times.size, dtype=bool)
    lost_3d[in_3d] = lost[in_2d]
    step = np.diff(positions, axis=0)
    speed = np.linalg.norm(step, axis=1) / np.diff(times)
    moving = np.arctan2(step[:, 1], step[:, 0])
    running = np.linalg.norm(step[:, :2], axis=1) / np.diff(times) > HEADING_SPEED
    resultant_lost = np.abs(np.mean(np.exp(1j * moving[running & lost_3d[:-1]])))
    other = running & np.isfinite(headings[:-1])
    resultant_other = np.abs(
        np.mean(np.exp(1j * (headings[:-1][other] - moving[other])))
    )
    edges = np.diff(np.r_[0, lost.astype(np.int8), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    long = stops - starts > MIN_HELD_REPEATS
    longest = float(np.max(raw.flat_times[stops - 1] - raw.flat_times[starts]))
    height = positions[:, 2]
    low, high = np.nanpercentile(height, [25, 75])
    return (
        Conversion(
            f"headings: NaN at the {int(lost.sum())} samples whose lattice.poh and "
            "lattice.pov are both exactly 0, taken to have no azimuth",
            "inferred",
            "no source read says what a 0 in poh means; lattice.poh is exactly 0 "
            f"in {int(zero.sum())} of {zero.size} samples ({zero.mean():.1%}), "
            f"{int(lost.sum())} of them ({lost.sum() / zero.sum():.1%}) with "
            "lattice.pov, the 2-D speed, also exactly 0, and pov is "
            + (
                "never exactly 0 where poh is not"
                if not still.any()
                else f"exactly 0 in {int(still.sum())} samples where poh is not"
            )
            + "; over those samples the "
            "head keeps moving (3-D speed median "
            f"{np.nanmedian(speed[lost_3d[:-1]]):.3g} cm/s, against "
            f"{np.nanmedian(speed[~lost_3d[:-1]]):.3g} cm/s elsewhere) in no "
            f"preferred direction (above {HEADING_SPEED:g} cm/s the direction of "
            f"movement has mean resultant length {resultant_lost:.3f}, where "
            "heading minus movement has "
            f"{resultant_other:.3f} over the other headings); "
            f"{(stops - starts)[long].sum() / lost.sum():.0%} of them lie in runs "
            f"of {MIN_HELD_REPEATS + 1} or more, the longest {longest:.0f} s; and "
            f"they are {lost_3d[height <= low].mean():.1%} of the samples in the "
            f"lowest quarter of heights and {lost_3d[height >= high].mean():.1%} "
            "in the highest",
        ),
    )


def _lost_and_zero_azimuths(
    raw: _Session, frame: _FrameMap, lost: NDArray[np.bool_]
) -> tuple[str, ...]:
    """A watch-out for the samples without an azimuth (NaN) and the 0 deg azimuths kept.

    ``lost``, shape (n_2d,), marks the 2-D samples whose ``poh`` and ``pov``
    are both exactly 0.
    """
    kept = (raw.azimuth == 0) & ~lost
    if not lost.any() and not kept.any():
        return ()
    text = []
    if lost.any():
        text.append(
            f"{lost.mean():.1%} of heading samples ({int(lost.sum())}) have "
            "lattice.poh and lattice.pov both exactly 0, which the loader takes to "
            "mark a sample without an azimuth (the headings Conversion gives the "
            "evidence): they are NaN in rec.headings, so head-direction tuning "
            "leaves them out."
        )
    if kept.any():
        turned = frame.rotation @ np.array([1.0, 0.0])
        value = float(np.arctan2(turned[1], turned[0]))
        axes = ", ".join(
            str(int(np.sum(raw.azimuth == np.radians(deg)))) for deg in (90, 180, 270)
        )
        text.append(
            f"{kept.sum() / (~lost).sum():.2%} of the other samples "
            f"({int(kept.sum())}) are exactly 0 deg in lattice.poh with pov not 0, "
            f"against {axes} at exactly 90, 180 and 270 deg, so they may be "
            "azimuths along an axis; they are kept as stored, at "
            f"{value!r} rad in rec.headings."
        )
    return (" ".join(text),)


def held_runs(values: NDArray[np.float64]) -> NDArray[np.intp]:
    """First and last index of each run of identical values, ``MIN_HELD_REPEATS`` repeats or more.

    Parameters
    ----------
    values : ndarray, shape (n,)

    Returns
    -------
    ndarray, shape (n_runs, 2)
        ``[first, last]`` sample index of each run of at least
        ``MIN_HELD_REPEATS + 1`` equal values (NaN never repeats).

    Examples
    --------
    >>> held_runs(np.r_[1.0, np.full(11, 2.0), 3.0, np.full(5, 4.0)]).tolist()
    [[1, 11]]
    """
    repeat = np.r_[False, values[1:] == values[:-1]]
    edges = np.diff(np.r_[0, repeat.astype(np.int8), 0])
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    long = stops - starts >= MIN_HELD_REPEATS
    runs: NDArray[np.intp] = np.column_stack([starts[long] - 1, stops[long] - 1])
    return runs


def _held_headings(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
) -> tuple[str, ...]:
    """Heading samples equal to the one before, and the long runs, while the head moves."""
    runs = held_runs(headings)
    if not len(runs):
        return ()
    repeat = (headings[1:] == headings[:-1])[np.isfinite(headings[1:])]
    seconds = times[runs[:, 1]] - times[runs[:, 0]]
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    path = np.r_[0.0, np.cumsum(np.nan_to_num(steps))]
    moved = path[runs[:, 1]] - path[runs[:, 0]]
    n_held = int(np.sum(runs[:, 1] - runs[:, 0]))
    return (
        f"{repeat.mean():.1%} of the samples with a heading repeat the sample "
        "before; "
        f"{len(runs)} runs of {MIN_HELD_REPEATS + 1} or more identical headings "
        f"({n_held} repeated samples, {seconds.sum():.0f} s in all, the longest "
        f"{seconds.max():.1f} s), over which the head moves a median "
        f"{np.median(moved):.3g} cm along its path: the file holds the azimuth "
        "where it was not updated, and does not say why. The values are kept; "
        "head-direction tuning counts those samples at the held direction.",
    )


def _outside_cube(
    positions: NDArray[np.float64],
    bars: NDArray[np.float64],
    in_grid: NDArray[np.bool_],
) -> tuple[str, ...]:
    """The share of tracked samples outside the convex hull of the bars.

    ``in_grid``, shape (n_tracked,), says which tracked samples the authors'
    grid holds (from :func:`_authors_grid`).
    """
    from scipy.spatial import ConvexHull

    hull = ConvexHull(bars.reshape(-1, 3))
    tracked = positions[tracked_mask(positions)]
    beyond = (tracked @ hull.equations[:, :3].T + hull.equations[:, 3]).max(axis=1)
    outside = beyond > 0
    if not outside.any():
        return ()
    return (
        f"{outside.mean():.1%} of tracked samples lie outside the lattice's cube "
        f"(the convex hull of its bars, {hull.volume / 1e6:.3g} m^3), up to "
        f"{beyond.max():.3g} cm beyond a face (median "
        f"{np.median(beyond[outside]):.3g} cm): where the tracked head of a rat "
        f"on the outer bars is. The environment, the authors' grid, holds "
        f"{in_grid[outside].mean():.1%} of them ({in_grid.mean():.2%} of all "
        "tracked samples).",
    )


def _recording_days(raw: _Session) -> tuple[str, ...]:
    """A watch-out when the session's recordings were made on different days.

    ``session_names`` start with the recording's date, ``yymmdd``.
    """
    days = {name[:6] for name in raw.recordings if name[:6].isdigit()}
    if len(days) < 2:
        return ()
    return (
        f"The session's recordings ({', '.join(raw.recordings)}) were made on "
        f"{len(days)} days, by the dates their names start with, yet one clock "
        "joins them end to end.",
    )


def _single_precision(times: NDArray[np.float64], raw: _Session) -> tuple[str, ...]:
    """How finely times stored as float32 are resolved at the session's end.

    Only for ``lattice_3d.pot`` or spike times whose HDF5 dtype is float32.
    """
    single_times = raw.time_dtype == "float32"
    single = [
        spikes
        for (_, _, spikes, _), dtype in zip(raw.units, raw.spike_dtypes, strict=True)
        if dtype == "float32"
    ]
    if not single_times and not single:
        return ()
    ends = [float(s[-1]) for s in single if s.size]
    if single_times:
        ends.append(float(times[-1]))
    last = max(ends)
    step = float(np.spacing(np.float32(last)))
    spikes = (
        "spike times"
        if len(single) == len(raw.units)
        else f"the spike times of {len(single)} of {len(raw.units)} units"
    )
    what = (
        f"Times (lattice_3d.pot) and {spikes} are"
        if single_times and single
        else "Times (lattice_3d.pot) are"
        if single_times
        else f"{spikes[0].upper()}{spikes[1:]} are"
    )
    text = (
        f"{what} stored in single precision (float32): at {last:.0f} s, one "
        f"float32 step is {step * 1e3:.2g} ms, so each time is rounded to that"
    )
    if single_times:
        intervals = np.diff(times)
        text += (
            f", and the sample intervals run from {intervals.min():.5g} to "
            f"{intervals.max():.5g} s around {np.median(intervals):.4g} s"
        )
    return (text + ".",)
