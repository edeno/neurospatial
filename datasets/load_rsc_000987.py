"""Peyrache lab mouse retrosplenial cortex units in a square arena with tracked head direction (DANDI 000987).

What was recorded: mouse A7803, 37 retrosplenial units on silicon probes,
sleep (0-3,878 s) then a square arena (3,878-5,085 s), with motion-capture
tracking of a rigid body on the head at 100 Hz (3,880-5,083 s). The file also
holds slow-wave sleep and UP/DOWN state intervals.

Where it lives: DANDI 000987, version 0.261003.2342, CC-BY-4.0; the smallest
file with units and position (20.2 MB).

What the loader converts: position is three separate ``SpatialSeries``
(``Position/x``, ``y``, ``z``) with a blank unit and a blank reference frame,
and head orientation is three (``CompassDirection/rx``, ``ry``, ``rz``) in
radians from 0 to 2 pi. Nothing in the file says which axis is vertical, so
the value ranges decide: x and z span about 0.5 m each and y 0.07 m (head
height), so y is vertical, the floor is (x, z), and the values are metres
(a mouse at a median 6 cm/s, 90th percentile 14 cm/s, read as metres). The
loader converts to cm. Lost tracking is set to NaN, positions and heading
together (``_record.mask_lost_tracking``): runs of 10 or more identical
frames (0.4 % of the frames, in 22 runs of up to 1 s), which the
motion-capture system filled with its last fix, and samples reached faster
than 100 cm/s, which a mouse does not run. The positions are not quantized
(about 100,000 distinct values per axis on the file's 1 um resolution,
``_record.position_grid``), as the held-run rule needs. ``ry``, rotation about the vertical axis, is the yaw;
of the three angles it alone tracks the direction of movement. Motion-capture
systems with y up are right-handed, so seen from above x points East and -z
North; the loader returns (x, -z) and ``ry`` wrapped to [-pi, pi]. Under that
frame ``_record.heading_vs_movement`` (10 Hz samples, faster than 10 cm/s)
gives a mean offset of about +0.5 rad and a spread of about 49 degrees: the
frame is not verified by movement (the survey's rule needs |mean| <= 0.2 rad
and spread < 45 degrees). Possible causes, not tested: a mouse that walks
slowly (median 6 cm/s) while turning its head, and a rigid-body zero set by
hand. The sign of the angle is tested: ``ry`` negated against the same
movement gives a spread of 87 degrees (against 49), so ``ry`` turns the same
way as the (x, -z) frame. Mirroring positions and heading together cannot be
told apart this way, because it leaves heading minus movement unchanged.
Measures that do not depend on the frame (mean vector length) are
unaffected; preferred directions are not interpreted.

Analysis choices (``params``):

- ``bin_size`` 2.5 cm; ``analysis_epochs`` the tracked span
  (``position_time_support``).
- ``dt`` 0.25 s and alternating 60 s ``train_epochs``/``test_epochs``.

Environment: built from the samples (inferred). The file says only "square
arena". The study's bioRxiv preprint (Mehrotra, Levenstein et al. 2023,
10.1101/2023.09.12.557442, Methods, read) describes "removable walls (width:
80 cm, height: 50 cm)" arranged into a square, and for the RSC dataset a
70 cm circle as well; the samples' fitted square is 52.6 x 52.1 cm, 34 % short
of 80 cm, so the stated size is refused (``_geometry.MAX_SIZE_RESIDUAL``)
until a source explains the difference (G-130).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import dandi_asset_url, fetch, open_nwb
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    heading_vs_movement,
    mask_lost_tracking,
    validate_recording,
)

DATASET = {
    "key": "rsc_000987",
    "title": "Peyrache lab, mouse A7803, retrosplenial cortex units, sleep then square arena",
    "source": "https://dandiarchive.org/dandiset/000987/0.261003.2342",
    "license": "CC-BY-4.0",
    "download_size": "20.2 MB",
}

# Pre-registered from the dataset catalog (head direction plus 2-D, blank position
# units) and the file's metadata (37 units; x, y, z and rx, ry, rz series of
# 120,287 samples over 1203 s) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 37,
    "n_dims": 2,
    "sampling_rate_hz": Range(99.0, 101.0),
    "arena_extent_cm": lambda rec: Range(40, 100).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
    "headings_follow_movement": lambda rec: bool(
        abs(
            heading_vs_movement(rec.times, rec.positions, rec.headings, min_speed=5.0)[
                0
            ]
        )
        < np.pi / 4
    ),
}


_PATH = "sub-A7803/sub-A7803_ses-20240509T213129_behavior+ecephys.nwb"
_ASSET_ID = "fccdeeeb-6f18-4c39-b385-d68932b2e665"
_SHA256 = "3b9083c7b37882e8d4d1396f9b751449a948bd1f183afa53ab406cea05d8e876"
# Read as metres: a mouse arena's floor spans 0.3-1.5 m; head height varies
# by under 0.2 m.
_FLOOR_M = (0.3, 1.5)
_VERTICAL_M = 0.2


def _windows(table: object) -> np.ndarray:
    frame = table.to_dataframe()  # type: ignore[attr-defined]
    return frame[["start_time", "stop_time"]].to_numpy(dtype=float)


def load(cache: Path | None = None) -> Recording:
    """Fetch the session; floor positions in cm, yaw in radians, spikes."""
    from neurospatial.io.nwb import read_units

    path = fetch(
        dandi_asset_url(_ASSET_ID),
        Path(_PATH).name,
        sha256=_SHA256,
        subdir=DATASET["key"],
        cache=cache,
    )
    with open_nwb(str(path)) as nwbfile:
        position = nwbfile.acquisition["Position"]
        compass = nwbfile.acquisition["CompassDirection"]
        axes = {k: position[k] for k in "xyz"}
        labels = {k: series.unit for k, series in axes.items()}
        xyz = {k: np.asarray(series.data[:], dtype=float) for k, series in axes.items()}
        times = np.asarray(axes["x"].timestamps[:], dtype=float)
        same_clock = all(
            np.array_equal(np.asarray(series.timestamps[:]), times)
            for series in [*axes.values(), *compass.spatial_series.values()]
        )
        yaw = np.asarray(compass["ry"].data[:], dtype=float)
        yaw_unit = compass["ry"].unit
        units = read_units(nwbfile)
        epochs = nwbfile.epochs.to_dataframe()
        support = _windows(nwbfile.intervals["position_time_support"])
        intervals = {
            name.lower(): _windows(nwbfile.intervals[name])
            for name in ("SWS", "UP", "DOWN")
        }

    floor = np.ptp(xyz["x"]), np.ptp(xyz["z"])
    if (
        set(labels.values()) != {""}
        or yaw_unit != "radian"
        or not same_clock
        or not all(_FLOOR_M[0] <= v <= _FLOOR_M[1] for v in floor)
        or np.ptp(xyz["y"]) >= _VERTICAL_M
    ):
        msg = (
            f"position units {labels}, yaw unit {yaw_unit!r}, same clock {same_clock},"
            f" spans x {floor[0]:.2f}, y {np.ptp(xyz['y']):.2f}, z {floor[1]:.2f};"
            " the blank-unit metres-with-y-up reading no longer holds, recheck"
        )
        raise ValueError(msg)
    positions, tracking_line = mask_lost_tracking(
        times,
        np.column_stack([xyz["x"], -xyz["z"]]) * 100.0,
        min_frozen=10,
        max_speed=100.0,
    )
    # The rigid body's yaw is held with its position.
    headings = np.where(
        np.isfinite(positions).all(axis=1), np.angle(np.exp(1j * yaw)), np.nan
    )
    mean, spread = heading_vs_movement(
        times[::10], positions[::10], headings[::10], min_speed=10.0
    )
    if spread > np.radians(70):
        msg = f"ry no longer follows movement (spread {np.degrees(spread):.0f} deg)"
        raise ValueError(msg)
    tags = epochs["tags"].map(lambda t: t[0]).to_numpy()
    windows = epochs[["start_time", "stop_time"]].to_numpy(dtype=float)
    train, test = alternating_blocks(times[0], times[-1], 60.0)
    return Recording(
        geometry={
            "tracked_point": "a motion-capture rigid body on the head (per the loader)"
        },
        dataset=DATASET["key"],
        session="A7803-210726",
        source=f"{DATASET['source']} {_PATH} (asset {_ASSET_ID})",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        headings=headings,
        spike_times=tuple(np.asarray(s, dtype=float) for s in units.spike_times),
        unit_ids=np.asarray(units.unit_ids),
        epochs={
            **{str(tag): windows[tags == tag] for tag in tags},
            **intervals,
            "tracking": support,
        },
        params={
            "bin_size": 2.5,
            "dt": 0.25,
            "analysis_epochs": support,
            "train_epochs": train,
            "test_epochs": test,
            "nwb_source": str(path),
            "readers_used": ("units",),
        },
        conversions=(
            "positions: Position/x and -Position/z (blank unit) x 100, cm; y is"
            f" vertical (span {np.ptp(xyz['y']) * 100:.1f} cm); range checked:"
            f" {np.nanmin(positions[:, 0]):.1f}-{np.nanmax(positions[:, 0]):.1f} by"
            f" {np.nanmin(positions[:, 1]):.1f}-{np.nanmax(positions[:, 1]):.1f} cm",
            tracking_line + "; headings NaN at the same samples",
            "headings: CompassDirection/ry (radian, 0-2 pi) wrapped to [-pi, pi];"
            " frame not verified: heading_vs_movement mean"
            f" {mean:+.2f} rad, spread {np.degrees(spread):.0f} deg (10 Hz, > 10 cm/s)",
            "epochs: epochs table by tag (sleep, wake); SWS, UP, DOWN intervals;"
            " position_time_support as tracking",
        ),
    )


def _vertical_axis_as_floor(rec: Recording) -> Recording:
    # A user taking the first two series (x, y) as the floor plane: y is the
    # head's height, so the arena collapses to a strip a few cm wide.
    assert rec.positions is not None
    with open_nwb(rec.params["nwb_source"]) as nwbfile:
        height = np.asarray(nwbfile.acquisition["Position"]["y"].data[:], dtype=float)
    return replace(
        rec, positions=np.column_stack([rec.positions[:, 0], height * 100.0])
    )


# The blank unit read as cm (metres 100 times too small) is the generic
# length_unit_off_by_100 probe.
HAZARDS = {"vertical_axis_as_floor": _vertical_axis_as_floor}


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
