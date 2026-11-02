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
- Putative duplicate units in ``merges_Dataset_*.mat`` are pooled, keeping
  the first cluster's id, and spikes closer than 1 ms are dropped, as the
  record's ``NP_Bat_Demo.m`` does.
- Flight epochs from the 1-based sample indices ``f_smp``.
- Positions are kept as recorded. 3.6 % of samples lie outside the room bounds
  ``r_lim`` the file states, all on y and almost all outside flights.

Analysis choices (``params``):

- ``bin_size`` 0.2 m: 3-D bins about the size of a bat's body.
- ``analysis_epochs``: flights; rate maps describe the bat in flight.
- ``train_epochs``/``test_epochs``: even and odd flights, for held-out decoding.
- ``dt`` 0.25 s, as for the 2-D datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch_zip_member
from _record import Range, Recording, check_expectations, validate_recording

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
_REFRACTORY_S = 0.001


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


def _drop_refractory(spikes: np.ndarray) -> np.ndarray:
    spikes = np.sort(spikes)
    keep = np.ones(spikes.size, dtype=bool)
    last = -np.inf
    for i, time in enumerate(spikes):
        if time - last < _REFRACTORY_S:
            keep[i] = False
        else:
            last = time
    return spikes[keep]


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
    for group in groups:
        pooled = np.concatenate([trains.pop(cluster) for cluster in group])
        trains[group[0]] = _drop_refractory(pooled)
    return {
        f"probe{probe}-{cluster}": np.sort(trains[cluster])
        for cluster in sorted(trains)
    }


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
    return Recording(
        dataset=DATASET["key"],
        session="Dataset_3_14543_240419",
        source=_ZIP,
        license=DATASET["license"],
        times=times,
        positions=np.asarray(behavior["r"], dtype=float),
        position_units="m",
        spike_times=tuple(units.values()),
        unit_ids=np.asarray(list(units)),
        epochs={"flight": flight_windows},
        geometry={"room_bounds_m": room},
        params={
            "bin_size": 0.2,
            "dt": 0.25,
            "analysis_epochs": flight_windows,
            "train_epochs": flight_windows[::2],
            "test_epochs": flight_windows[1::2],
        },
        conversions=(
            "spike_times: spikeTimes_usec (global clock) / 1e6",
            "spike_times: merges_Dataset_*.mat groups pooled; spikes < 1 ms apart dropped",
            "epochs['flight']: times at the 1-based sample indices f_smp",
            "unit_ids: probe<n>-<kilosort cluster id>",
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
