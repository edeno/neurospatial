"""Sargolini et al. (2006) rat medial entorhinal cortex, open field (DANDI 000582).

What was recorded: six units from layer III of medial entorhinal cortex in a
Long Evans rat running for 600 s in a 1 m × 1 m enclosure (session
``sub-10884_ses-03080402``, the session pynapple's null-distribution tutorial
uses). Position comes from one tracking LED at 50 Hz.

Where it lives: DANDI 000582, published version 0.251111.2151, one 15.9 MB NWB
file, CC-BY-4.0.

What the loader converts: ``read_position`` returns metres (the file stores cm
with ``conversion=0.01``); positions are multiplied by 100 to give cm. The file
has a single LED, so head direction cannot be computed and ``headings`` is None.

Analysis choices (``params``):

- ``bin_size`` 2.5 cm: 40 bins across the 100 cm box, the binning of the
  pynapple reference table.
- ``dt`` 0.25 s: open-field decoding bin. On the harness's simulated open field
  the default 25 ms bin gives a median error near 30 cm and 250 ms about 9 cm.
- ``min_speed`` None: no speed filter, as in the pynapple reference.
- ``train_epochs``/``test_epochs``: alternating 60 s blocks of the session, so
  decoding is always tested on held-out time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _fetch import dandi_asset_url, fetch
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    validate_recording,
)

DATASET = {
    "key": "sargolini_000582",
    "title": "Sargolini et al. 2006, rat MEC grid and conjunctive cells, 1 m box",
    "source": "https://dandiarchive.org/dandiset/000582/0.251111.2151",
    "license": "CC-BY-4.0",
    "download_size": "15.9 MB",
}

# Pre-registered from the dataset description and pynapple's tutorial, before
# the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 6,
    "n_dims": 2,
    "sampling_rate_hz": Range(49.0, 51.0),
    "position_min": Range((-55.0, -55.0), (0.0, 0.0)),
    "position_max": Range((0.0, 0.0), (55.0, 55.0)),
}

_PATH = "sub-10884/sub-10884_ses-03080402_behavior+ecephys.nwb"
_ASSET_ID = "49d80982-99f4-4e17-a553-498fc9a4688f"
_SHA256 = "89e1a1030565030d69bc061058403da8161ccaf103f20bdfeca56c37f4881501"


def load(cache: Path | None = None) -> Recording:
    """Fetch the session and return it as a ``Recording`` (positions in cm)."""
    import pynwb

    from neurospatial.io.nwb import read_position, read_units

    path = fetch(
        dandi_asset_url(_ASSET_ID),
        Path(_PATH).name,
        sha256=_SHA256,
        subdir=DATASET["key"],
        cache=cache,
    )
    with pynwb.NWBHDF5IO(path, "r", load_namespaces=True) as io:
        nwbfile = io.read()
        position = read_position(nwbfile)
        units = read_units(nwbfile)
        description = nwbfile.session_description

    if position.units != "m":
        msg = f"expected read_position to return metres, got {position.units!r}"
        raise ValueError(msg)
    times = np.asarray(position.times, dtype=float)
    train, test = alternating_blocks(times[0], times[-1], 60.0)
    return Recording(
        dataset=DATASET["key"],
        session="sub-10884_ses-03080402",
        source=f"{DATASET['source']} {_PATH}",
        license=DATASET["license"],
        times=times,
        positions=np.asarray(position.positions, dtype=float) * 100.0,
        position_units="cm",
        spike_times=tuple(
            np.asarray(spikes, dtype=float) for spikes in units.spike_times
        ),
        unit_ids=np.asarray(units.unit_ids),
        geometry={"shape": "square", "size_cm": 100.0, "description": description},
        params={
            "bin_size": 2.5,
            "dt": 0.25,
            "min_speed": None,
            "train_epochs": train,
            "test_epochs": test,
            "nwb_path": str(path),
        },
        conversions=(
            "positions: cm = read_position metres x 100 (file stores cm, conversion=0.01)",
            "headings: none; the file has one LED (SpatialSeriesLED1)",
        ),
    )


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s")
    print(
        f"  positions {np.nanmin(rec.positions, 0)} to {np.nanmax(rec.positions, 0)} cm"
    )
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
