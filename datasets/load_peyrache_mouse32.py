"""Peyrache et al. mouse anterodorsal thalamus head-direction session Mouse32-140822.

What was recorded: 49 units (31 labelled ``adn``, 18 ``thalamus``) across a
sleep, wake, sleep session; head direction (``ry``, radians) was tracked at
about 39 Hz during the wake epoch only. The file has no x/y position.

Where it lives: OSF ``jb2gd`` (the file pynapple's and NeMoS's head-direction
tutorials download), 36.6 MB. The OSF project states no license; the source
dandiset 000056 is CC-BY-4.0.

What the loader converts: headings are wrapped from [0, 2π) to [−π, π] with
``neurospatial.stats.wrap_angle``. The file does not state the heading's
reference frame (where 0 points, which way angles increase), so headings are
not rotated; neurospatial's allocentric convention (0 = East, counter-clockwise)
is assumed but unverified.

Analysis choices (``params``):

- ``analysis_epochs``: the wake epoch, the only time with tracked head
  direction (spikes during sleep have no heading to pair with).
- ``dt`` 0.1 s: the decoding bin of pynapple's head-direction tutorial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch
from _record import Recording, check_expectations, validate_recording

DATASET = {
    "key": "peyrache_mouse32",
    "title": "Peyrache et al., mouse ADn/thalamus head-direction cells, wake and sleep",
    "source": "https://osf.io/jb2gd",
    "license": "unstated on OSF; the source dandiset 000056 is CC-BY-4.0",
    "download_size": "36.6 MB",
}

# Pre-registered from pynapple's head-direction tutorial, before the first run
# of this loader. Its "31 units in ADn" needs per-unit regions, which a
# Recording does not hold; the head-direction reference check verifies it.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 49,
    "epochs_include_wake_and_sleep": lambda rec: {"wake", "sleep"} <= set(rec.epochs),
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}

_URL = "https://osf.io/jb2gd/download"
_NAME = "Mouse32-140822.nwb"
_SHA256 = "1a919a033305b8f58b5c3e217577256183b14ed5b436d9c70989dee6dafe0f35"


def _intervals(table: Any) -> np.ndarray:
    return np.column_stack([np.asarray(table.start), np.asarray(table.stop)])


def load(cache: Path | None = None) -> Recording:
    """Fetch the session and return it as a ``Recording`` (headings only)."""
    import pynwb

    from neurospatial.io.nwb import read_head_direction, read_intervals, read_units
    from neurospatial.stats import wrap_angle

    path = fetch(_URL, _NAME, sha256=_SHA256, subdir=DATASET["key"], cache=cache)
    with pynwb.NWBHDF5IO(path, "r", load_namespaces=True) as io:
        nwbfile = io.read()
        heading = read_head_direction(nwbfile)
        units = read_units(nwbfile)
        tagged = read_intervals(nwbfile, interval_name="epochs").intervals_table()
        sleep_states = {
            name: _intervals(read_intervals(nwbfile, interval_name=name))
            for name in ("sws", "rem")
        }

    epochs: dict[str, np.ndarray] = {}
    for tag in ("wake", "sleep"):
        rows = tagged[[tag in tags for tags in tagged["tags"]]]
        epochs[tag] = rows[["start_time", "stop_time"]].to_numpy(dtype=float)
    epochs.update(sleep_states)
    return Recording(
        dataset=DATASET["key"],
        session="Mouse32-140822",
        source=DATASET["source"],
        license=DATASET["license"],
        times=np.asarray(heading.times, dtype=float),
        headings=np.asarray(wrap_angle(np.asarray(heading.headings, dtype=float))),
        spike_times=tuple(
            np.asarray(spikes, dtype=float) for spikes in units.spike_times
        ),
        unit_ids=np.asarray(units.unit_ids),
        epochs=epochs,
        params={"analysis_epochs": epochs["wake"], "dt": 0.1, "nwb_path": str(path)},
        conversions=(
            "headings: wrapped from [0, 2pi) to [-pi, pi] with neurospatial.stats.wrap_angle",
            "headings: reference frame unstated in the file; not rotated",
            "epochs: wake/sleep from the epochs table tags; sws/rem from their tables",
        ),
    )


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(
        f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s tracked"
    )
    print(f"  epochs: {', '.join(f'{k} {len(v)}' for k, v in rec.epochs.items())}")
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
