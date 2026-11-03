"""Huszar & Buzsaki mouse CA1 and CA3 units on a cheeseboard maze with ripples (DANDI 001880).

What was recorded: mouse iuvE14-cagCre-ef1aDIOchr2-3, 8 Dec 2022, 49 units
in CA1 and CA3 (opto-tagged), during a PRE, a Learning and a POST
sub-session on a cheeseboard maze with two reward locations, separated by
sleep in a start box. Position (one LED, cm) at 30 Hz across all three
sub-sessions (64-17,301 s), 5.3 % NaN, most of it off the maze. The file holds
2,118 ripples, the maze and box intervals, 45 learning trials and the reward
coordinates.

Where it lives: DANDI 001880, version 0.261007.1849, CC0-1.0; the smallest
file with units and position (65.7 MB).

What the loader converts: nothing changes units. ``SubjectPosition`` holds two
series, ``PositionSeries`` (x, y in cm) and ``VelocitySeries`` (vx, vy, speed
in cm/s); the loader reads the first with ``read_position(position_name=...)``
because the default call refuses to choose between them. Ripples are read with
pynwb from ``processing/ecephys/Ripples``; ``onMaze`` intervals are the maze
epochs.

Analysis choices (``params``):

- ``bin_size`` 2.5 cm; ``min_speed`` 4 cm/s.
- ``analysis_epochs``: the 52 ``onMaze`` intervals (5,412 s), so the box is
  left out of rate maps and occupancy.
- ``dt`` 0.25 s; ``train_epochs``/``test_epochs``: even and odd ``onMaze``
  intervals.
- ``replay_dt`` 0.02 s; ``events["ripples"]``: the 195 ripples inside an
  ``onMaze`` interval (awake ripples on the maze).

Environment: built from the samples (inferred). Neither the file nor the
DANDI description gives the cheeseboard's size; the paper (Neuron 2026,
10.1016/j.neuron.2026.07.023) could not be read here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _fetch import dandi_asset_url, fetch, open_nwb
from _record import Range, Recording, check_expectations, validate_recording

DATASET = {
    "key": "ca1ca3_001880",
    "title": "Huszar & Buzsaki, mouse iuvE14-3, CA1 and CA3 units, cheeseboard place-reward task",
    "source": "https://dandiarchive.org/dandiset/001880/0.261007.1849",
    "license": "CC0-1.0",
    "download_size": "65.7 MB",
}

# Pre-registered from the dataset catalog (place-reward task, about 5 % NaN
# position, ripples) and the file's metadata (49 units, position in cm at
# 30 Hz, PRE / Learning / POST sub-sessions, a Ripples table) before the first
# run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 49,
    "n_dims": 2,
    "sampling_rate_hz": Range(29.0, 31.0),
    "position_nan_fraction": Range(0.03, 0.08),
    "maze_extent_cm": lambda rec: Range(80, 120).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "has_ripples": lambda rec: "ripples" in rec.events,
}


_PATH = (
    "sub-iuvE14-cagCre-ef1aDIOchr2-3/"
    "sub-iuvE14-cagCre-ef1aDIOchr2-3_ses-iuvE14-cagCre-ef1aDIOchr2-3-221208"
    "_behavior+ecephys.nwb"
)
_ASSET_ID = "67a90fac-4acd-40f6-9199-9a7668686121"
_SHA256 = "293741812fba73117dc02f590a7182b1a2c7eaa4f1bebdb09ab0ad92770582cb"
_SERIES = "processing/behavior/SubjectPosition/PositionSeries"
# A mouse cheeseboard maze, in cm: the longest span must lie here.
_MAZE_CM = (60.0, 150.0)


def _windows(table: object) -> np.ndarray:
    frame = table.to_dataframe()  # type: ignore[attr-defined]
    return frame[["start_time", "stop_time"]].to_numpy(dtype=float)


def load(cache: Path | None = None) -> Recording:
    """Fetch the session; positions in cm, spikes, maze epochs and ripples."""
    from neurospatial.io.nwb import read_position, read_units

    path = fetch(
        dandi_asset_url(_ASSET_ID),
        Path(_PATH).name,
        sha256=_SHA256,
        subdir=DATASET["key"],
        cache=cache,
    )
    with open_nwb(str(path)) as nwbfile:
        position = read_position(nwbfile, position_name=_SERIES)
        units = read_units(nwbfile)
        behavior = nwbfile.processing["behavior"]
        on_maze = _windows(behavior["onMaze"])
        sub_sessions = behavior["subSessions"].to_dataframe()
        ripples = _windows(nwbfile.processing["ecephys"]["Ripples"])
        rewards = behavior["rewardCoordinates"].to_dataframe()[["x", "y"]]

    positions = np.asarray(position.positions, dtype=float)
    span = np.nanmax(positions, axis=0) - np.nanmin(positions, axis=0)
    if position.units != "cm" or not _MAZE_CM[0] <= span.max() <= _MAZE_CM[1]:
        msg = f"{_SERIES}: unit {position.units!r}, span {span.round(1)}; recheck"
        raise ValueError(msg)
    on = (ripples[:, :1] >= on_maze[:, 0]) & (ripples[:, 1:] <= on_maze[:, 1])
    times = np.asarray(position.times, dtype=float)
    return Recording(
        dataset=DATASET["key"],
        session="iuvE14_cagCre_ef1aDIOchr2_3_221208",
        source=f"{DATASET['source']} {_PATH} (asset {_ASSET_ID})",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        spike_times=tuple(np.asarray(s, dtype=float) for s in units.spike_times),
        unit_ids=np.asarray(units.unit_ids),
        epochs={
            "on_maze": on_maze,
            **{
                str(row.label).lower(): np.array([[row.start_time, row.stop_time]])
                for row in sub_sessions.itertuples()
            },
        },
        events={"ripples": ripples[on.any(axis=1)], "all_ripples": ripples},
        geometry={"reward_locations_cm": rewards.to_numpy(dtype=float)},
        params={
            "bin_size": 2.5,
            "min_speed": 4.0,
            "dt": 0.25,
            "replay_dt": 0.02,
            "analysis_epochs": on_maze,
            "train_epochs": on_maze[::2],
            "test_epochs": on_maze[1::2],
            "nwb_source": str(path),
            "readers_used": ("position", "units"),
        },
        conversions=(
            f"positions: read_position(position_name={_SERIES!r}), cm; range checked:"
            f" {np.nanmin(positions[:, 0]):.1f}-{np.nanmax(positions[:, 0]):.1f} by"
            f" {np.nanmin(positions[:, 1]):.1f}-{np.nanmax(positions[:, 1]):.1f} cm",
            "epochs: onMaze intervals as on_maze; subSessions rows by label",
            f"events['ripples']: the {int(on.any(axis=1).sum())} Ripples rows inside"
            f" an onMaze interval, of {len(ripples)}",
        ),
    )


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
