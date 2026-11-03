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
  kept in ``geometry["lattice_bars_cm"]``, shape (147, 2, 3).
- Head direction is not used. ``sdata.lattice.poh`` (degrees) is the azimuth
  in the overhead camera's 2-D frame: against the 2-D positions
  (``sdata.lattice.pox/poy``) heading minus movement direction has mean
  -0.8° and spread 42°, but against the 3-D frame's x, y the mean is -113°:
  the lattice frame is rotated from the camera's, so the azimuth is not a
  heading in the positions' frame.

Analysis choices (``params``): ``bin_size`` 4 cm, close to the authors' 3-D
rate maps (33 x 35 x 39 bins over the volume); ``min_speed`` 3 cm/s;
``analysis_epochs`` the lattice part; ``dt`` 0.25 s and alternating 60 s
train/test blocks.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch_zip_member
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
RUNNING_CM_S = 10.0


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
    bars_cm = bars.reshape(-1, 3, 3)[:, :2] / _MM_PER_CM
    if np.isnan(bars_cm).any():
        msg = "lattice_3d lx/ly/lz are not (start, stop, NaN) triplets"
        raise ValueError(msg)
    span = checked_span(positions, *_SPAN_CM)
    # The azimuth against the camera's 2-D frame and against the 3-D frame.
    in_camera = heading_vs_movement(
        times, camera / _MM_PER_CM, azimuth, min_speed=RUNNING_CM_S
    )
    in_lattice = heading_vs_movement(
        times, positions[:, :2], azimuth, min_speed=RUNNING_CM_S
    )
    lattice = np.array([[times[0], times[-1]]])
    train, test = alternating_blocks(times[0], times[-1], 60.0)
    return Recording(
        dataset=DATASET["key"],
        session="926_06042018 lattice",
        source=f"{DATASET['source']} Summary data.zip/{_MEMBER}",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        spike_times=tuple(spikes),
        unit_ids=np.asarray(ids),
        epochs={"lattice": lattice},
        geometry={"lattice_bars_cm": bars_cm, "lattice_type": lattice_type},
        params={
            "bin_size": 4.0,
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
            "headings: none; lattice.poh minus the direction of movement above"
            f" {RUNNING_CM_S:g} cm/s: in the camera's 2-D frame (lattice.pox, poy)"
            f" mean {np.degrees(in_camera[0]):.1f} deg, spread"
            f" {np.degrees(in_camera[1]):.1f} deg; in the 3-D frame's x, y mean"
            f" {np.degrees(in_lattice[0]):.1f} deg, spread"
            f" {np.degrees(in_lattice[1]):.1f} deg: not a heading in the positions'"
            " frame",
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
