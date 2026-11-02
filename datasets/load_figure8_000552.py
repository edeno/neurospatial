"""Huszár, Zhang, Blockus & Buzsáki mouse CA1 on a figure-8 maze (DANDI 000552).

What was recorded: 113 CA1 units in mouse ``e14_2m2`` across a sleep, maze,
sleep session (18,557 s). On the figure-8 maze the animal ran 21 rewarded
alternation trials; position was tracked at 30 Hz only on the maze (7,891–9,483
s, inside the maze epoch 7,859–9,492 s), as camera (x, y) in cm and as a
linearized position along the track, 0–222 cm. The linearization runs around
the maze as a loop: 0 and 222 cm are the same place (both ends sit at camera
(85, 43) cm), so the linear position jumps by about 216 cm 62 times. The file also holds 4,297 detected ripples and Awake, Non-REM
and REM sleep states.

Where it lives: DANDI 000552, version 0.230630.2304, one 96 MB NWB file,
CC-BY-4.0.

What the loader converts: nothing changes units. ``positions`` is the
linearized track position (cm); the camera (x, y) is kept in
``geometry["positions_2d"]``. Ripples and sleep states are read with pynwb
because they live in ``processing/`` tables that ``read_intervals`` does not
search.

Analysis choices (``params``):

- ``bin_size`` 2 cm: 112 bins along the 222 cm track, the position bin of the
  ``GROSMARK_2016`` replay recipe.
- ``min_speed`` 4 cm/s: excludes stops at the reward wells, where the animal
  is still and ripples happen, from the running rate maps.
- ``train_epochs``/``test_epochs``: even and odd trials, so position decoding
  is tested on held-out laps.
- ``dt`` 0.25 s: running-decoding bin, as for the open-field datasets.
- ``replay_dt`` 0.02 s: the decoding bin of both ``DAVIDSON_2009`` and
  ``GROSMARK_2016``.
- ``events["ripples"]``: the 71 ripples inside the maze epoch (awake ripples,
  decoded with the maze's own rate maps). The 4,226 sleep ripples are not
  replayed, to keep the shuffle tests to minutes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _fetch import dandi_asset_url, fetch
from _record import Range, Recording, check_expectations, validate_recording

DATASET = {
    "key": "figure8_000552",
    "title": "Huszár et al., mouse CA1 on a figure-8 maze with sleep and ripples",
    "source": "https://dandiarchive.org/dandiset/000552/0.230630.2304",
    "license": "CC-BY-4.0",
    "download_size": "96.0 MB",
}

# Pre-registered from the dandiset description and a metadata probe of this
# file, before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 113,
    "n_dims": 1,
    "position_min": Range(0.0, 222.0),
    "position_max": Range(0.0, 222.0),
    "ripples_present": lambda rec: (
        "ripples" in rec.events and len(rec.events["ripples"]) > 0
    ),
    "sleep_states_present": lambda rec: len(rec.epochs) > 0,
}

_PATH = "sub-e14-2m2/sub-e14-2m2_ses-e14-2m2-201019_behavior+ecephys.nwb"
_ASSET_ID = "61ea529d-46db-42e6-b742-17588b2f70a2"
_SHA256 = "3ab5f99419a9d0070ef6b5bc468841b75467ac95f538e9735312fa0d29aa3d4a"
_LINEAR = "processing/behavior/LinearizedPosition/LinearizedSpatialSeries"
_CAMERA = "processing/behavior/SubjectPosition/SpatialSeries"


def load(cache: Path | None = None) -> Recording:
    """Fetch the session and return it as a ``Recording`` (linearized cm)."""
    import pynwb

    from neurospatial.io.nwb import read_intervals, read_position, read_units

    path = fetch(
        dandi_asset_url(_ASSET_ID),
        Path(_PATH).name,
        sha256=_SHA256,
        subdir=DATASET["key"],
        cache=cache,
    )
    with pynwb.NWBHDF5IO(path, "r", load_namespaces=True) as io:
        nwbfile = io.read()
        linear = read_position(nwbfile, position_name=_LINEAR)
        camera = read_position(nwbfile, position_name=_CAMERA)
        units = read_units(nwbfile)
        trials = read_intervals(nwbfile, interval_name="trials")
        session_epochs = read_intervals(nwbfile, interval_name="epochs")
        states = nwbfile.processing["behavior"]["SleepStates"].to_dataframe()
        ripple_table = nwbfile.processing["ecephys"]["Ripples"].to_dataframe()
        ripples = ripple_table[["start_time", "stop_time"]].to_numpy(dtype=float)

    if linear.units != "cm" or camera.units != "cm":
        msg = f"expected cm positions, got {linear.units!r} and {camera.units!r}"
        raise ValueError(msg)
    times = np.asarray(linear.times, dtype=float)
    trial_windows = np.column_stack([np.asarray(trials.start), np.asarray(trials.stop)])
    epoch_windows = np.column_stack(
        [np.asarray(session_epochs.start), np.asarray(session_epochs.stop)]
    )
    maze = epoch_windows[
        (epoch_windows[:, 0] <= times[0]) & (epoch_windows[:, 1] >= times[-1])
    ]
    in_maze = (ripples[:, 0] >= maze[0, 0]) & (ripples[:, 1] <= maze[0, 1])
    epochs = {
        "maze": maze,
        "trials": trial_windows,
        **{
            str(label).lower().replace("-", "_"): states.loc[
                states["label"] == label, ["start_time", "stop_time"]
            ].to_numpy(dtype=float)
            for label in sorted(states["label"].unique())
        },
    }
    return Recording(
        dataset=DATASET["key"],
        session="e14_2m2_201019",
        source=f"{DATASET['source']} {_PATH}",
        license=DATASET["license"],
        times=times,
        positions=np.asarray(linear.positions, dtype=float).reshape(-1, 1),
        position_units="cm",
        spike_times=tuple(
            np.asarray(spikes, dtype=float) for spikes in units.spike_times
        ),
        unit_ids=np.asarray(units.unit_ids),
        epochs=epochs,
        events={"ripples": ripples[in_maze], "all_ripples": ripples},
        geometry={
            "track_length_cm": 222.0,
            "positions_2d": np.asarray(camera.positions, dtype=float),
            "positions_2d_units": "cm",
        },
        params={
            "bin_size": 2.0,
            "min_speed": 4.0,
            "dt": 0.25,
            "replay_dt": 0.02,
            "analysis_epochs": maze,
            "train_epochs": trial_windows[::2],
            "test_epochs": trial_windows[1::2],
            "nwb_path": str(path),
        },
        conversions=(
            f"positions: {_LINEAR} (cm), reshaped to (n, 1)",
            f"geometry['positions_2d']: {_CAMERA} (cm)",
            "events['ripples']: Ripples table rows inside the maze epoch;"
            " every row in events['all_ripples']",
            "epochs: sleep states by label from processing/behavior/SleepStates (pynwb)",
        ),
    )


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(
        f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s tracked"
    )
    print(f"  ripples on the maze: {len(rec.events['ripples'])}")
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
