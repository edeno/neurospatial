"""Forli, Fan, Qi & Yartsev (2025) bat hippocampus during 3-D flight (Zenodo demo session).

What was recorded: an Egyptian fruit bat flying freely in a room for about
90 minutes (38 flights) while three Neuropixels probes recorded hippocampal
units. Position is 3-D (metres) at 120 Hz from the room's tracking system.

Where it lives: Zenodo record 15738988 (CC-BY-4.0), ``Demo_Session.zip``
(3.3 GB). Only the needed members are read, by HTTP byte range: the behavior
file, the three Kilosort 4 single-unit tables and the merge list.

What the loader converts:

- Spike times from the global clock in µs (``spikeTimes_usec``) to seconds.
  ``localSpikeTimes_usec`` is the probe's own clock and is not used.
- Putative duplicate units in ``merges_Dataset_*.mat`` are pooled under the
  first cluster's id, as the record's ``NP_Bat_Demo.m`` does; the pooled train
  is sorted (the script concatenates without sorting). In every unit a spike
  within 1 ms of its predecessor is dropped, the script's rule.
- Flight epochs from the 1-based sample indices ``f_smp``. The behavior starts
  before the neural recording (the first flight, −15.4 to −14.1 s, ends before
  the first spike at −12.7 s), so the flights used for analysis are clipped to
  the recording, taken as the first to last spike of any unit.
- Positions are kept as recorded. 3.6 % of samples lie outside the room bounds
  ``r_lim`` the file states, all on y and almost all outside flights.

Analysis choices (``params``):

- ``bin_size`` 0.2 m: 3-D bins about the size of a bat's body (1,352 bins
  over the volume the bat visited).
- ``min_speed``: the file's ``v_th``, the velocity threshold the authors used
  to segment flights (0.5 m/s).
- ``analysis_epochs``: flights during the neural recording; rate maps
  describe the bat in flight.
- ``train_epochs``/``test_epochs``: even and odd of those flights, for held-out
  decoding.
- ``dt`` 0.25 s, as for the 2-D datasets.

Environment (``geometry["environment"]``): the room, every voxel of a grid
that tiles ``r_lim`` exactly (``Environment.from_samples`` with
``dimension_ranges`` set to it and ``infer_active_bins=False``), so the
environment holds the air the bat could fly through, not only the voxels it
visited. Each axis has a whole number of cells close to ``bin_size`` (29 x
26 x 11 cells of 0.2 x 0.2 x 0.209 m): with 0.2 m cells on every axis the 2.3 m
height is 11.5 cells and the grid would reach 5 cm past the floor and the
ceiling. Forli et al. 2025 (Methods, read
from Zotero's full-text cache) flew four bats in "an indoor flight room
(... 5.6 m x 5.2 m x 2.5 m)" and two in an outdoor enclosure 10.1 m long;
``r_lim`` (5.8 x 5.2 x 2.3 m; "boundaries of the room" in ``NP_Bat_Demo.m``)
and the samples' 5.7 m span place this session in the indoor room. ``r_lim``
is used because it is stated in the positions' frame: the paper gives only a
size. Centring the paper's size on ``r_lim`` would cut the samples at the x
walls (they reach x = +-2.85 m, the paper's half-length is 2.8 m). The share
of samples outside each is in ``conversions``; the resting samples beyond the
-y wall (G-025) stay outside.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch_zip_member
from _record import Range, Recording, check_expectations, validate_recording

from neurospatial import Environment

DATASET = {
    "key": "yartsev_bat_demo",
    "title": "Forli et al. 2025, Egyptian fruit bat CA1 during free flight in a room",
    "source": "https://doi.org/10.5281/zenodo.15738988",
    "license": "CC-BY-4.0",
    "download_size": "260 MB of a 3.3 GB zip (byte-range reads)",
}

# Pre-registered from the record's demo script and a metadata probe of the
# behavior file, before the first run of this loader. y is not constrained:
# some samples lie outside the room bounds the file states.
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 3,
    "sampling_rate_hz": Range(119.0, 121.0),
    "position_min": Range((-3.1, -np.inf, -0.2), (np.inf, np.inf, np.inf)),
    "position_max": Range((-np.inf, -np.inf, -np.inf), (3.1, np.inf, 2.5)),
    "probe1_at_most_24_units": lambda rec: (
        sum(str(uid).startswith("probe1-") for uid in rec.unit_ids) <= 24
    ),
    "flights_present": lambda rec: len(rec.epochs["flight"]) == 38,
}

_ZIP = "https://zenodo.org/records/15738988/files/Demo_Session.zip?download=1"
_FILES = {
    "Extracted_Behavior_240419.mat": "88bd2164121088a789077b72a69d26e616a6e0c84a91ef200e09e89018427ddc",
    "merges_Dataset_3_14543_240419.mat": "7c05b2a9b3127ab5b8e3ebf540355f82419a7328c0d6b0e9a60b4fcb7806da51",
    "SU_kilosort4_outdir_probe1.mat": "ce0fc9442538415cdb85fd4d38dda66032ea12e59af6897690a4f47285d2ed05",
    "SU_kilosort4_outdir_probe2.mat": "1efbec9057625bb92967da00045e5746b3c07aaeff37815fafcba28bf12a727d",
    "SU_kilosort4_outdir_probe3.mat": "7a1b3254f176b192ec0bfa44366946fff104b3cdb5f038daa25900c5eeea0882",
}
_CLOSE_S = 0.001
_BIN_SIZE = 0.2
# Forli et al. 2025 (Methods): "an indoor flight room (n = 4 bats, 5.6 m x 5.2 m
# x 2.5 m) or ... an outdoor flight enclosure (n = 2 bats, 10.1 m x 4.1 m x 2.8 m)".
_PAPER_ROOM_M = np.array([5.6, 5.2, 2.5])


def _outside(positions: np.ndarray, bounds: np.ndarray) -> float:
    """Share of tracked samples outside the box ``bounds``, shape (n_dims, 2)."""
    tracked = positions[np.isfinite(positions).all(axis=1)]
    inside = np.all((tracked >= bounds[:, 0]) & (tracked <= bounds[:, 1]), axis=1)
    return float(1 - inside.mean())


def _fetch(name: str, cache: Path | None) -> Path:
    return fetch_zip_member(
        _ZIP,
        f"Demo_Session/{name}",
        sha256=_FILES[name],
        subdir=DATASET["key"],
        cache=cache,
    )


def _good_units(path: Path) -> Any:
    import matio

    value: Any = matio.load_from_mat(str(path))["out"]["good_units"]
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def _merge_groups(path: Path) -> list[list[list[int]]]:
    """Per probe, the groups of cluster ids to pool (``merges{probe}{group}``)."""
    from scipy.io import loadmat

    merges = loadmat(path)["merges"]
    return [
        [
            np.asarray(cell[j, 0]).ravel().astype(int).tolist()
            for j in range(cell.shape[0])
        ]
        for cell in merges[:, 0]
    ]


def _drop_close_spikes(spikes: np.ndarray) -> np.ndarray:
    """Drop each spike within 1 ms of the spike before it, as the record's script does.

    ``NP_Bat_Demo.m`` removes ``s(find(diff(s) < 0.001) + 1)``: each spike is
    compared with its predecessor in the train, whether or not that one is
    kept.
    """
    close = np.flatnonzero(np.diff(spikes) < _CLOSE_S) + 1
    return np.delete(spikes, close)


def _pool(
    trains: dict[int, np.ndarray], groups: list[list[int]]
) -> dict[int, np.ndarray]:
    """Pool each group of putative duplicates under its first cluster id, sorted.

    The record's script concatenates the trains without sorting; sorting
    first lets the 1 ms rule remove spikes the two clusters share.
    """
    pooled = dict(trains)
    for group in groups:
        pooled[group[0]] = np.sort(np.concatenate([pooled.pop(c) for c in group]))
    return {
        cluster: _drop_close_spikes(np.sort(pooled[cluster]))
        for cluster in sorted(pooled)
    }


def _probe_units(
    path: Path, probe: int, groups: list[list[int]]
) -> dict[str, np.ndarray]:
    table = _good_units(path)
    trains = {
        int(cluster): np.asarray(spikes, dtype=float).ravel() / 1e6
        for cluster, spikes in zip(
            table["cluster_id"], table["spikeTimes_usec"], strict=True
        )
    }
    return {
        f"probe{probe}-{cluster}": spikes
        for cluster, spikes in _pool(trains, groups).items()
    }


def _clip(windows: np.ndarray, span: tuple[float, float]) -> np.ndarray:
    clipped = np.column_stack(
        [np.maximum(windows[:, 0], span[0]), np.minimum(windows[:, 1], span[1])]
    )
    return clipped[clipped[:, 1] > clipped[:, 0]]


def load(cache: Path | None = None) -> Recording:
    """Fetch the needed members and return the session (metres, 3-D)."""
    from scipy.io import loadmat

    behavior = loadmat(_fetch("Extracted_Behavior_240419.mat", cache), squeeze_me=True)
    groups = _merge_groups(_fetch("merges_Dataset_3_14543_240419.mat", cache))
    units: dict[str, np.ndarray] = {}
    for probe in (1, 2, 3):
        path = _fetch(f"SU_kilosort4_outdir_probe{probe}.mat", cache)
        units.update(_probe_units(path, probe, groups[probe - 1]))

    times = np.asarray(behavior["t"], dtype=float)
    flights = np.asarray(behavior["f_smp"], dtype=int) - 1
    flight_windows = np.column_stack([times[flights[:, 0]], times[flights[:, 1]]])
    room = np.asarray(behavior["r_lim"], dtype=float)
    # The ephys start and stop are not in these files; the first and last
    # spike of the 289 units (population rate about 800 spikes/s) bound them.
    recorded = (
        min(float(s[0]) for s in units.values() if s.size),
        max(float(s[-1]) for s in units.values() if s.size),
    )
    neural_flights = _clip(flight_windows, recorded)
    positions = np.asarray(behavior["r"], dtype=float)
    # A whole number of cells close to _BIN_SIZE on each axis, so the grid
    # tiles r_lim exactly.
    n_cells = np.floor(np.ptp(room, axis=1) / _BIN_SIZE + 0.5)
    room_bins = np.ptp(room, axis=1) / n_cells
    environment = partial(
        Environment.from_samples,
        positions,
        bin_size=tuple(float(b) for b in room_bins),
        dimension_ranges=[tuple(bounds) for bounds in room],
        infer_active_bins=False,
        units="m",
        name=DATASET["key"],
    )
    # The paper's room size, centred where r_lim is, for comparison.
    paper = np.mean(room, axis=1)[:, None] + np.outer(_PAPER_ROOM_M, [-0.5, 0.5])
    paper[2] = [room[2, 0], room[2, 0] + _PAPER_ROOM_M[2]]
    return Recording(
        dataset=DATASET["key"],
        session="Dataset_3_14543_240419",
        source=_ZIP,
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="m",
        spike_times=tuple(units.values()),
        unit_ids=np.asarray(list(units)),
        epochs={"flight": flight_windows, "neural_recording": np.array([recorded])},
        geometry={"room_bounds_m": room, "environment": environment},
        params={
            "bin_size": _BIN_SIZE,
            "dt": 0.25,
            "min_speed": float(behavior["v_th"]),
            "analysis_epochs": neural_flights,
            "train_epochs": neural_flights[::2],
            "test_epochs": neural_flights[1::2],
        },
        conversions=(
            "spike_times: spikeTimes_usec (global clock) / 1e6",
            "spike_times: merges_Dataset_*.mat groups pooled and sorted under the first id",
            "spike_times: in every unit, a spike within 1 ms of its predecessor dropped",
            "epochs['flight']: times at the 1-based sample indices f_smp (all 38)",
            "epochs['neural_recording']: first to last spike of any unit",
            "params flights: flights clipped to the neural recording",
            "unit_ids: probe<n>-<kilosort cluster id>",
            "environment: declared room, every voxel of a grid tiling r_lim"
            f" {np.ptp(room, axis=1).round(2).tolist()} m (NP_Bat_Demo.m: 'boundaries"
            f" of the room') exactly: {n_cells.astype(int).tolist()} cells of"
            f" {room_bins.round(3).tolist()} m; {_outside(positions, room):.2%} of tracked samples lie"
            " outside it; Forli et al. 2025 Methods state 5.6 x 5.2 x 2.5 m: that"
            " size centred on r_lim in x and y, from r_lim's floor in z, leaves"
            f" {_outside(positions, paper):.2%} outside; placement: the file's frame"
            " (not fitted)",
        ),
    )


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s")
    print(
        f"  positions {np.nanmin(rec.positions, 0)} to {np.nanmax(rec.positions, 0)} m"
    )
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
