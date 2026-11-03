"""Vollan et al. 2025 rat MEC units on a linear track (EBRAINS 10.25493/R5FR-EDG).

What was recorded: rat 27764, Neuropixels in medial entorhinal cortex (389
units, 84 of them in the authors' one grid module), running a 200 cm linear
track (the head positions span about 2.1 m). Head position and head direction come from an OptiTrack
motion-capture system at 100 Hz.

Where it lives: EBRAINS dataset "Left-right-alternating theta sweeps in the
entorhinal-hippocampal spatial map" (DOI 10.25493/R5FR-EDG), CC BY-SA 4.0,
public bucket, no login. One MATLAB v7 file per session type and recording:
``sharing_v4/navigation/lt/27764_1.mat`` (110.6 MB). The file holds one struct,
``Dsession``, described in the dataset's ``documentation/mat_files.md``. It is
read with ``mat-io``, which reads its MATLAB ``string`` fields (unit ids, cell
types); ``scipy.io.loadmat`` returns them as opaque objects.

What the loader converts:

- ``x``, ``y`` are metres relative to the arena centre; converted to cm.
- ``hd`` ("2D head direction (azimuth)", unit and zero direction not stated)
  lies in [−π, π]; the loader checks it against the direction of movement
  while running (``heading_vs_movement``) and keeps it as radians.
- The authors removed every sample slower than 5 cm/s, so ``t`` has gaps
  (most under a second; the longest 332 s). They are kept as gaps: rebuilding
  times from ``dt`` would pull every later sample earlier.
- ``sessions`` lists one linear-track session, 7227–7486 s, but samples run
  to 8412 s; all samples are kept, and the mismatch is recorded.
- Units from every recorded area (``units.mec``; ``units.hc`` is empty for
  this rat), Kilosort labels ``good`` and ``mua`` both kept as the authors
  shared them. Grid-module membership (``acorrCluId``, ``isGrid``) goes to
  ``params``: a unit's ``acorrCluId`` k (0 for none) is the k-th entry of
  ``unitAcorrClus.grid``, defined on the open-field session, so a module
  counts at most its ``nUnits`` units (checked; 84 of 85 here); that entry's
  ``cluId`` is a different number and is not used.

Analysis choices (``params``): ``bin_size`` 2.5 cm; ``min_speed`` None (the
data are already speed-filtered); ``dt`` 0.25 s and alternating 60 s
train/test blocks for decoding.

Declared environment (``geometry["environment"]``): the 200 cm linear track
of Vollan et al. 2025 (Methods) as a one-edge graph along the samples'
principal axis, centred on them; the paper states no width, so the track is
built with a placement tolerance measured from the samples. The fit is in
``conversions``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    checked_span,
    heading_vs_movement,
    validate_recording,
)
from _tracks import (
    declared_geometry,
    straight_placement,
    straight_track,
    with_declared,
)

DATASET = {
    "key": "vollan_lt",
    "title": "Vollan et al. (2025), rat 27764, MEC Neuropixels units on a linear track",
    "source": "https://doi.org/10.25493/R5FR-EDG",
    "license": "CC-BY-SA-4.0",
    "download_size": "110.6 MB",
}

# Pre-registered from the dataset documentation (100 Hz, speed-filtered at 5 cm/s) and the file's unit table (389 MEC units) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 389,
    "n_dims": 2,
    "sampling_rate_hz": Range(99.0, 101.0),
    # No extent expectation: the loader refuses a span outside 190-240 cm
    # (checked_span), so one here could never fail; the span is in
    # ``conversions``.
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
    "speed_filter_leaves_gaps": lambda rec: bool(np.max(np.diff(rec.times)) > 0.1),
}

BUCKET = "https://data-proxy.ebrains.eu/api/v1/buckets/d-4080b78d-edc5-4ae4-8144-7f6de79930ea"
SUBDIR = "vollan_2025"
_FILE = "sharing_v4/navigation/lt/27764_1.mat"
_SHA256 = "f76dfd313d796402e06b192bb8666bbfa9765a8e3e84fde01e5a59def4436750"
RUNNING_CM_S = 10.0


def _value(field: Any) -> Any:
    """Unwrap mat-io's nested 1×1 cells to the array or scalar inside."""
    while isinstance(field, np.ndarray) and field.dtype == object and field.size == 1:
        field = field.ravel()[0]
    return field


def _text(field: Any) -> str:
    return str(np.asarray(_value(field)).ravel()[0])


def read_session(path: Path) -> dict[str, Any]:
    """Arrays of one ``Dsession`` file: tracking in m and rad, units, sessions, modules."""
    import matio

    d = matio.load_from_mat(str(path))["Dsession"]

    def series(name: str) -> np.ndarray:
        return np.asarray(_value(d[name]), dtype=float).ravel()

    units = _value(d["units"])
    trains, ids, areas, labels, modules, is_grid = [], [], [], [], [], []
    for area in units.dtype.names:
        table = _value(units[area])
        if not isinstance(table, np.ndarray) or table.size == 0:
            continue
        for unit in table.ravel():
            trains.append(
                np.sort(np.asarray(_value(unit["spikeTimes"]), float).ravel())
            )
            ids.append(_text(unit["id"]))
            areas.append(area)
            labels.append(_text(unit["ks2Label"]))
            modules.append(int(np.asarray(_value(unit["acorrCluId"])).ravel()[0]))
            is_grid.append(bool(np.asarray(_value(unit["isGrid"])).ravel()[0]))
    sessions = _value(d["sessions"])
    ranges = np.vstack(
        [
            np.asarray(_value(r), float).reshape(-1, 2)
            for r in sessions["validTimeRanges"].ravel()
        ]
    )
    grid = _value(_value(d["unitAcorrClus"])["grid"])

    def per_module(name: str) -> np.ndarray:
        return np.array(
            [float(np.asarray(_value(m[name])).ravel()[0]) for m in grid.ravel()]
        )

    sizes = per_module("nUnits").astype(int)
    found = np.array([modules.count(k) for k in range(1, grid.size + 1)])
    # Modules are defined on the open-field session; another session's file
    # can hold fewer of their units, never more.
    if np.any(found > sizes):
        msg = f"unit acorrCluId counts {found} exceed the module sizes {sizes}"
        raise ValueError(msg)
    return {
        "t": series("t"),
        "x": series("x"),
        "y": series("y"),
        "hd": series("hd"),
        "dt": float(np.asarray(_value(d["dt"])).ravel()[0]),
        "spike_times": tuple(trains),
        "unit_ids": np.asarray(ids),
        "areas": np.asarray(areas),
        "ks2_labels": np.asarray(labels),
        "module": np.asarray(modules),
        "is_grid": np.asarray(is_grid),
        "session_names": [_text(n) for n in sessions["name"].ravel()],
        "session_ranges": ranges,
        "module_ids": np.arange(1, grid.size + 1),
        "module_median_grid_score": per_module("medianGridScore"),
        "module_grid_score": per_module("gridScore"),
        "module_spacing_cm": per_module("gridSpacingMeters") * 100.0,
    }


def to_recording(
    raw: dict[str, Any],
    *,
    dataset: dict[str, str],
    file: str,
    session: str,
    span_cm: tuple[float, float],
) -> Recording:
    """Recording in cm and radians, with a frame check of the head direction."""
    times = raw["t"]
    positions = np.column_stack([raw["x"], raw["y"]]) * 100.0
    span = checked_span(positions, *span_cm)
    headings = raw["hd"]
    finite = headings[np.isfinite(headings)]
    if finite.min() < -np.pi - 1e-6 or finite.max() > np.pi + 1e-6:
        msg = "hd is no longer within [-pi, pi]; recheck its unit"
        raise ValueError(msg)
    offset, spread = heading_vs_movement(
        times, positions, headings, min_speed=RUNNING_CM_S
    )
    if abs(offset) > 0.2 or spread > np.radians(45):
        msg = (
            f"hd minus the direction of movement: mean {offset:.2f} rad, spread"
            f" {spread:.2f} rad; recheck its frame"
        )
        raise ValueError(msg)
    gaps = np.diff(times)
    train, test = alternating_blocks(times[0], times[-1], 60.0)
    ranges = raw["session_ranges"]
    outside = float(np.mean((times < ranges[0, 0]) | (times > ranges[-1, 1])))
    return Recording(
        dataset=dataset["key"],
        session=session,
        source=f"{dataset['source']} {file}",
        license=dataset["license"],
        times=times,
        positions=positions,
        position_units="cm",
        headings=headings,
        spike_times=raw["spike_times"],
        unit_ids=raw["unit_ids"],
        epochs={"sessions": ranges},
        params={
            "bin_size": 2.5,
            "min_speed": None,
            "dt": 0.25,
            "train_epochs": train,
            "test_epochs": test,
            "module": raw["module"],
            "is_grid": raw["is_grid"],
            "ks2_labels": raw["ks2_labels"],
            "module_ids": raw["module_ids"],
            "module_median_grid_score": raw["module_median_grid_score"],
            "module_grid_score": raw["module_grid_score"],
            "module_spacing_cm": raw["module_spacing_cm"],
        },
        conversions=(
            f"positions: x, y in m relative to the arena centre, times 100 -> cm; {span}",
            "headings: hd in [-pi, pi] rad, kept; minus the direction of movement"
            f" above {RUNNING_CM_S:g} cm/s: circular mean {np.degrees(offset):.1f} deg,"
            f" median spread {np.degrees(spread):.1f} deg",
            f"times: t as stored; speed-filtered at 5 cm/s by the authors, so"
            f" {int(np.sum(gaps > 2 * raw['dt']))} gaps longer than 2 dt (longest"
            f" {gaps.max():.1f} s)",
            f"sessions: {raw['session_names']} {ranges.tolist()}; {outside:.1%} of"
            " samples lie outside the listed sessions and are kept",
            f"units: {len(raw['unit_ids'])} from areas {sorted(set(raw['areas'].tolist()))},"
            f" Kilosort labels {sorted(set(raw['ks2_labels'].tolist()))} kept",
        ),
    )


TRACK_CM = 200.0


def load(cache: Path | None = None) -> Recording:
    """Fetch the linear-track file; positions in cm, headings in radians, spikes."""
    path = fetch(
        f"{BUCKET}/{_FILE}?redirect=true",
        "lt_27764_1.mat",
        sha256=_SHA256,
        subdir=SUBDIR,
        cache=cache,
    )
    rec = to_recording(
        read_session(path),
        dataset=DATASET,
        file=_FILE,
        session="27764_1 linear_track",
        span_cm=(190.0, 240.0),
    )
    graph, fit = straight_track(rec.positions, TRACK_CM)
    entries, line = declared_geometry(
        graph,
        rec.positions,
        times=rec.times,
        kind=f"straight track {TRACK_CM:g} cm",
        source="Vollan et al. 2025 Methods: 'a 200 cm linear track'; width not stated",
        bin_size=rec.params["bin_size"],
        physical_width=None,
        units="cm",
        name=DATASET["key"],
        placement=straight_placement(fit, "cm"),
        placed_by="samples' principal axis",
        stated_lengths={"length": (fit["samples_span"], TRACK_CM)},
        tracked_point="the head, OptiTrack motion capture (per the loader, from the dataset's documentation)",
        speed_thresholds=(5.0, 30.0),
    )
    return with_declared(rec, entries, line)


def times_rebuilt_from_dt(rec: Recording) -> Recording:
    # A reader trusting dt and nt: evenly spaced times that close the
    # speed-filter gaps, pulling every later sample earlier than its spikes.
    assert rec.times is not None
    dt = float(np.median(np.diff(rec.times)))
    return replace(rec, times=rec.times[0] + dt * np.arange(len(rec.times)))


HAZARDS = {"times_rebuilt_from_dt": times_rebuilt_from_dt}


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
