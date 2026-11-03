"""Grieves et al. 2020 rat CA1 place cells in a 3-D climbing lattice (UCL RDR 10.5522/04/9926477).

What was recorded: rat 926 on 2018-04-06, CA1 tetrodes recorded wirelessly
in three parts: a square arena, a cubic lattice climbing frame, and the
square again. Position comes from a multi-camera 3-D reconstruction at
50 Hz. This loader keeps the lattice part (about 1 h).

Where it lives: UCL Research Data Repository 9926477 v1 (CC0),
``Summary data.zip`` (11.2 GB, members compressed with Deflate64, which
Python's ``zipfile`` cannot inflate; ``_fetch`` inflates it with
``inflate64`` and checks the CRC-32). One member is read by HTTP byte range:
``926_06042018.mat`` (431 MB, MATLAB v7.3 / HDF5), holding one ``sdata``
struct described in the record's ``Summary data readme.m``.

What the loader converts:

- Positions: ``sdata.lattice_3d.pox/poy/poz``. Their unit is not documented.
  They are mm: the speed of the stored positions is exactly 10 x the stored
  velocity ``pov`` (median ratio 10.03), and ``pov``'s median of 11.7 is a
  climbing rat's speed in cm/s; the 2-D part of the same file states
  ``ppm`` = 1000 (pixels, here mm, per metre). Divided by 10 -> cm; range
  checked.
- Times: ``sdata.lattice_3d.pot`` (s, the session clock); strictly
  increasing.
- Units: every ``r926_06042018_t<tetrode>_c<cluster>`` field with
  ``spike_times`` (s, the same clock), except cluster 0 of each tetrode:
  those six fields hold only ``spike_times`` (0.68-1.5 million each, no cell
  analyses), the unsorted remainder. 24 units remain.
- Lattice: ``sdata.lattice_3d.lx/ly/lz`` hold the lattice's 147 bars in
  the positions' frame (mm, start, stop and a NaN separator per bar; each
  bar 970 mm, the cube's side) and ``sdata.config3.lattice_type`` reads
  "diagonal": this rat climbed the tilted lattice. Divided by 10 -> cm,
  kept in ``geometry["lattice_bars_cm"]``, shape (147, 2, 3); the loader
  checks there are 147 bars (7 x 7 per axis), each 97.0 cm (the paper's
  Supplementary Methods: 0.97 m a side).
- Headings: ``-sdata.lattice.poh`` in radians. ``poh`` (degrees) is the
  azimuth in the overhead camera's 2-D frame (``sdata.lattice.pox/poy``):
  against those positions heading minus movement direction has mean -0.8°
  and spread 42°. The 3-D frame's x, y are the camera's mirrored in y and
  shifted (``pox3 = pox + 165.6``, ``poy3 = 1039.6 - poy`` mm, sample by
  sample; the loader checks it), so in the positions' frame the azimuth turns
  the other way and ``-poh`` is the heading (mean and spread in
  ``conversions``; the loader refuses |mean| over 0.2 rad or a spread over
  45°).

Analysis choices (``params``): ``bin_size`` 4 cm, close to the authors' 3-D
rate maps (33 x 35 x 39 bins over the volume); ``min_speed`` 3 cm/s;
``analysis_epochs`` the lattice part; ``dt`` 0.25 s and alternating 60 s
train/test blocks.

Environment (``geometry["environment"]``): the lattice the rat climbed, as its
bars. The file's 147 bars are cut at their crossings into a graph of
segments (``_geometry.lattice_graph``) and built with
``_geometry.graph_environment``, which keeps every segment
(``Environment.from_graph`` keeps only the segments its edge order lists).
The Supplementary Methods (read) state the tubes' diameter, 10 mm, but the
tracked point is the head, which a rat on a bar holds a median 6 cm from it,
so a 10 mm track would leave almost every sample off it. No source states how
far the head reaches from a bar, so ``track_width`` is twice the 99th
percentile of the head's distance from the nearest bar, measured here (about
31 cm): it places 99 % of the samples. The voxel environment built from the
samples stays the driver's ``env.from_samples`` and is used whole with
``--inferred-env``.
"""

from __future__ import annotations

import re
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch_zip_member
from _geometry import distance_to_segments, graph_environment, lattice_graph
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    checked_span,
    heading_vs_movement,
    validate_recording,
)

DATASET = {
    "key": "grieves_2020_lattice",
    "title": "Grieves et al. (2020), rat 926, 2018-04-06, CA1 units in a cubic climbing lattice",
    "source": "https://doi.org/10.5522/04/9926477.v1",
    "license": "CC0",
    "download_size": "431 MB of an 11.2 GB Deflate64 zip (byte-range read)",
}

# Pre-registered from the readme, the zip listing and the session file's
# HDF5 layout (lattice_3d pox/poy/poz at 50 Hz; 30 clusters, 6 of them
# cluster 0) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 24,
    "n_dims": 3,
    "sampling_rate_hz": Range(49.0, 51.0),
    "duration_s": Range(3000.0, 4000.0),
    "lattice_extent_cm_per_axis": lambda rec: Range(100.0, 160.0).contains(
        np.nanpercentile(rec.positions, 99, axis=0)
        - np.nanpercentile(rec.positions, 1, axis=0)
    ),
}

_ZIP = "https://ndownloader.figshare.com/files/17989991"
_MEMBER = "926_06042018.mat"
_SHA256 = "fd0f62bc04c588d56b8810b98e250588e22c37684d39c4db3809ad491a8c8aa5"
_CELL = re.compile(r"r926_06042018_t(\d+)_c(\d+)$")
_SPAN_CM = (120.0, 200.0)
_MM_PER_CM = 10.0
# Supplementary Methods: a 6 x 6 x 6 lattice of cubes, 0.97 m a side, so 7 x 7
# bars along each of the three axes.
_N_BARS = 147
_LATTICE_SIDE_CM = 97.0
# Grieves et al. 2020 Supplementary Methods: tubes 10 mm across.
_TUBE_CM = 1.0
_N_CUBES = 6
_BIN_SIZE = 4.0
RUNNING_CM_S = 10.0
# The survey's rule for a heading in the positions' frame: heading minus the
# direction of movement has |mean| at most 0.2 rad and a spread under 45 deg.
_HEADING_MEAN_RAD = 0.2
_HEADING_SPREAD_RAD = np.radians(45.0)
_MIRROR_RMS_MM = 0.01


def _vector(group: Any, name: str) -> np.ndarray:
    return np.asarray(group[name][()], dtype=float).ravel()


def load(cache: Path | None = None) -> Recording:
    """Fetch the session file; the lattice part's 3-D positions in cm and units."""
    import h5py

    path = fetch_zip_member(
        _ZIP, _MEMBER, sha256=_SHA256, subdir=DATASET["key"], cache=cache
    )
    with h5py.File(path, "r") as mat:
        sdata = mat["sdata"]
        part = sdata["lattice_3d"]
        times = _vector(part, "pot")
        stored = np.column_stack([_vector(part, k) for k in ("pox", "poy", "poz")])
        velocity = _vector(part, "pov")
        bars = np.column_stack([_vector(part, k) for k in ("lx", "ly", "lz")])
        lattice_type = "".join(map(chr, sdata["config3"]["lattice_type"][()].ravel()))
        flat = sdata["lattice"]
        azimuth = np.angle(np.exp(1j * np.radians(_vector(flat, "poh"))))
        camera = np.column_stack([_vector(flat, "pox"), _vector(flat, "poy")])
        cells = sorted(
            (int(m[1]), int(m[2]), name) for name in sdata if (m := _CELL.match(name))
        )
        spikes, ids, n_unsorted = [], [], 0
        for tetrode, cluster, name in cells:
            train = np.sort(_vector(sdata[name], "spike_times"))
            if cluster == 0:
                n_unsorted += train.size
                continue
            spikes.append(train)
            ids.append(f"t{tetrode}c{cluster}")
    step = np.linalg.norm(np.diff(stored, axis=0), axis=1) / np.diff(times)
    ratio = float(np.nanmedian(step / velocity[1:]))
    if not 9.0 < ratio < 11.0:
        msg = f"position speed / pov = {ratio:.2f}; expected 10 (mm vs cm/s)"
        raise ValueError(msg)
    positions = stored / _MM_PER_CM
    triplets = bars.reshape(-1, 3, 3)
    bars_cm = triplets[:, :2] / _MM_PER_CM
    bar_lengths = np.linalg.norm(bars_cm[:, 1] - bars_cm[:, 0], axis=1)
    if (
        len(bars_cm) != _N_BARS
        or not np.isnan(triplets[:, 2]).all()
        or np.isnan(bars_cm).any()
        or not np.allclose(bar_lengths, _LATTICE_SIDE_CM, atol=0.1)
    ):
        msg = (
            f"lattice_3d lx/ly/lz: expected {_N_BARS} (start, stop, NaN) bars of"
            f" {_LATTICE_SIDE_CM:g} cm, got {len(bars_cm)} with lengths"
            f" {bar_lengths.min():.1f}-{bar_lengths.max():.1f} cm"
        )
        raise ValueError(msg)
    span = checked_span(positions, *_SPAN_CM)
    # The 3-D frame's x, y are the camera's mirrored in y: x3 = x + a and
    # y3 = b - y, sample by sample. So the camera azimuth, negated, is the
    # heading in the positions' frame.
    shift = stored[:, :2] - camera * np.array([1.0, -1.0])
    both = np.isfinite(shift).all(axis=1)
    mirror_rms = float(np.sqrt(np.mean((shift[both] - np.median(shift[both], 0)) ** 2)))
    if mirror_rms > _MIRROR_RMS_MM:
        msg = (
            f"lattice_3d pox/poy against lattice pox/poy: RMS {mirror_rms:.3g} mm"
            " from a y-mirror plus a shift; recheck the frames"
        )
        raise ValueError(msg)
    headings = np.angle(np.exp(-1j * azimuth))
    in_camera = heading_vs_movement(
        times, camera / _MM_PER_CM, azimuth, min_speed=RUNNING_CM_S
    )
    in_lattice = heading_vs_movement(
        times, positions[:, :2], headings, min_speed=RUNNING_CM_S
    )
    if abs(in_lattice[0]) > _HEADING_MEAN_RAD or in_lattice[1] > _HEADING_SPREAD_RAD:
        msg = (
            f"-lattice.poh against movement: mean {np.degrees(in_lattice[0]):.1f},"
            f" spread {np.degrees(in_lattice[1]):.1f} deg; not a heading, recheck"
        )
        raise ValueError(msg)
    lattice = np.array([[times[0], times[-1]]])
    train, test = alternating_blocks(times[0], times[-1], 60.0)
    graph = lattice_graph(bars_cm, _N_CUBES)
    reach = distance_to_segments(positions[np.isfinite(positions).all(axis=1)], bars_cm)
    reach_pct = np.percentile(reach, [50, 95, 99])
    width = 2 * float(reach_pct[2])
    environment = partial(
        graph_environment,
        graph,
        bin_size=_BIN_SIZE,
        track_width=width,
        units="cm",
        name=DATASET["key"],
    )
    return Recording(
        dataset=DATASET["key"],
        session="926_06042018 lattice",
        source=f"{DATASET['source']} Summary data.zip/{_MEMBER}",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        headings=headings,
        spike_times=tuple(spikes),
        unit_ids=np.asarray(ids),
        epochs={"lattice": lattice},
        geometry={
            "tracked_point": "the head, per the loader (lattice_3d positions with the head azimuth poh); how it was tracked is not stated here",
            "lattice_bars_cm": bars_cm,
            "lattice_type": lattice_type,
            "environment": environment,
            "declared_from": "authors' bars (file)",
            # A placement tolerance chosen from the samples, not a stated
            # width: the tubes are 10 mm across (Supplementary Methods).
            "declared_track_width": width,
            "declared_track_width_kind": "placement tolerance (2 x the head's 99th"
            " percentile distance from a bar)",
            "declared_physical_width": _TUBE_CM,
            "declared_physical_width_kind": "stated",
            "declared_speed_thresholds": (3.0, 30.0),
            # The file's bars against the paper's 0.97 m side.
            "environment_fit": {
                "size_residual_lattice_side": float(
                    np.median(bar_lengths) / _LATTICE_SIDE_CM - 1
                )
            },
        },
        params={
            "bin_size": _BIN_SIZE,
            "min_speed": 3.0,
            "dt": 0.25,
            "analysis_epochs": lattice,
            "train_epochs": train,
            "test_epochs": test,
        },
        conversions=(
            "positions: lattice_3d pox, poy, poz / 10 (mm -> cm; unit inferred:"
            f" position speed / pov median {ratio:.2f})",
            span,
            "times: lattice_3d pot as stored (s)",
            f"geometry['lattice_bars_cm']: lattice_3d lx, ly, lz / 10, {len(bars_cm)}"
            f" bars; config3.lattice_type {lattice_type!r}",
            f"units: {len(spikes)} tetrode/cluster fields; cluster 0 of each tetrode"
            f" ({n_unsorted} spikes, unsorted) left out",
            f"environment: declared lattice, its {len(bars_cm)} bars (Grieves et al."
            " 2020 Supplementary Methods: 6 x 6 x 6 cubes, 0.97 m a side) as a graph of"
            f" {graph.number_of_nodes()} crossings and {graph.number_of_edges()}"
            " segments, every segment kept (_geometry.graph_environment); placement:"
            " the file's bars, in the positions' frame (not fitted); track_width"
            f" {width:.1f} cm = 2 x the 99th percentile of the head's distance from"
            f" the nearest bar (median {reach_pct[0]:.1f}, 95th percentile"
            f" {reach_pct[1]:.1f} cm), not the stated 10 mm tube diameter",
            "headings: -lattice.poh (deg -> rad, wrapped): the 3-D frame's x, y"
            " are the camera's (lattice.pox, poy) mirrored in y plus a shift (RMS"
            f" {mirror_rms:.1g} mm), so the camera azimuth turns the other way;"
            f" heading minus the direction of movement above {RUNNING_CM_S:g} cm/s:"
            f" mean {np.degrees(in_lattice[0]):.1f} deg, spread"
            f" {np.degrees(in_lattice[1]):.1f} deg (poh in the camera frame: mean"
            f" {np.degrees(in_camera[0]):.1f}, spread {np.degrees(in_camera[1]):.1f})",
        ),
    )


def millimetres_labelled_cm(rec: Recording) -> Recording:
    """The stored mm taken as cm: the lattice 10 x too large, bins 10 x too fine.

    Not in ``HAZARDS``: probed by hand, ``Environment.from_samples`` warns
    about the grid (388 x 429 x 371 cells, 109,862 visited bins), and the
    rate maps then ran for over an hour at 18 GB before the run was stopped.
    """
    assert rec.positions is not None
    return replace(rec, positions=rec.positions * _MM_PER_CM)


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s")
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
