"""Gonzalez & Giocomo rat open-field session with binned firing rates (DANDI 000405).

What was recorded: rat ``s2`` exploring an open field (session
``Cl_OF_120618``), with 46 parahippocampal units. The file holds no spike
times and no ``Position``: tracking is a generic ``TimeSeries``,
``acquisition/track_data``, whose columns are named only in its description
(``columns=x,y,hd,sp,ha``), and the units' activity is
``acquisition/firing_rates``, binned rates at 50 Hz (65,904 × 46).

Where it lives: DANDI 000405, draft only (no published version, no license
stated), one 27 MB file pinned by asset id and SHA-256.

What the loader converts, trusting value ranges over labels:

- ``track_data`` is labelled ``unit="mm"``, but x spans −62 to 60 and y −5 to
  134; in millimetres a rat's open field would be 12 cm across. The ``sp``
  column agrees with the speed computed from x, y read as cm at 0.02 s per
  sample (r = 0.91). The values are read as cm, unchanged.
- ``track_data``'s ``rate`` attribute is 0.02, which is the sampling period:
  ``firing_rates`` has the same length with ``rate=50``. Times are
  ``0.02 * arange(n)`` s for both.
- ``hd`` is not head direction: it equals the direction of movement computed
  from x and y (median difference 0.00°, 90th percentile at most 1.5° in every
  speed band), so ``headings`` is left empty. What ``ha`` holds is not stated.
- ``firing_rates`` is labelled ``unit="uv"``; it is kept as rates (Hz) per
  20 ms bin, ``activity_kind="binned_rate"``.

Analysis choices (``params``): ``bin_size`` 2.5 cm.

Environment (``geometry["environment"]``): the open field Gonzalez &
Giocomo (eLife 2024, 13:RP85646, Methods "Open-Field"; open-access PDF read)
state, "The dimensions of field are 1.3m x 1.5m", floor panels laid on the
Tree-Maze frame. Which side is which is not stated: the 1.5 m side is matched
to the longer side of the samples' fitted rectangle, and centre and rotation
are fitted (``_geometry.declared_rectangle``); the residuals are in
``conversions``. Whether the field has walls or open edges is not stated; the
outline is declared as a wall (border scores treat both alike).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _geometry import declared_rectangle, with_environment
from _record import Range, Recording, check_expectations, validate_recording

from neurospatial.datasets._fetch import dandi_asset_url, fetch

DATASET = {
    "key": "giocomo_000405",
    "title": "Gonzalez & Giocomo (2022), rat parahippocampal units in an open field",
    "source": "https://dandiarchive.org/dandiset/000405/draft",
    "license": "unstated (draft dandiset)",
    "download_size": "27.1 MB",
}

# Pre-registered from the survey appendix before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_cells": 46,
    "n_dims": 2,
    "sampling_rate_hz": Range(49.0, 51.0),
    "activity_is_binned_rate": lambda rec: rec.activity_kind == "binned_rate",
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}

_PATH = "sub-s2/sub-s2_ses-Cl-OF-120618.nwb"
_ASSET_ID = "627b3e66-79c1-481a-9a74-afea5c9f641c"
_SHA256 = "34fee9d5676366d4cffd484ffa39383898aaa8c09f1fd94e8c1bcfd5dc20b87e"
_PERIOD_S = 0.02


def _columns(description: str) -> list[str]:
    if not description.startswith("columns="):
        msg = f"track_data description {description!r} does not name its columns"
        raise ValueError(msg)
    return description.removeprefix("columns=").split(",")


def load(cache: Path | None = None) -> Recording:
    """Fetch the session; positions in cm, headings in radians, binned rates."""
    import pynwb

    path = fetch(
        dandi_asset_url(_ASSET_ID),
        Path(_PATH).name,
        sha256=_SHA256,
        subdir=DATASET["key"],
        cache=cache,
    )
    with pynwb.NWBHDF5IO(path, "r", load_namespaces=True) as io:
        nwbfile = io.read()
        track = nwbfile.acquisition["track_data"]
        columns = _columns(track.description)
        data = np.asarray(track.data[:], dtype=float)
        rates = np.asarray(nwbfile.acquisition["firing_rates"].data[:], dtype=float)
        unit_names = np.asarray(nwbfile.acquisition["unit_table"]["cl_name"][:])

    col = {name: data[:, i] for i, name in enumerate(columns)}
    times = _PERIOD_S * np.arange(len(data))
    rec = Recording(
        dataset=DATASET["key"],
        session="Cl_OF_120618",
        source=f"{DATASET['source']} {_PATH} (asset {_ASSET_ID})",
        license=DATASET["license"],
        times=times,
        positions=np.column_stack([col["x"], col["y"]]),
        position_units="cm",
        unit_ids=unit_names,
        activity=rates,
        activity_times=times.copy(),
        activity_kind="binned_rate",
        params={"bin_size": 2.5, "nwb_source": str(path)},
        conversions=(
            "positions: track_data columns x, y by the description's column names;"
            " labelled mm, read as cm (ranges -62..60, -5..134)",
            "times: 0.02 * arange(n) s; track_data's rate attribute (0.02) is the period",
            "headings: none; track_data column hd is the direction of movement from x, y",
            "activity: firing_rates (labelled uv) as binned rates in Hz",
        ),
    )
    return with_environment(
        rec,
        *declared_rectangle(
            rec.positions,
            150.0,
            130.0,
            bin_size=rec.params["bin_size"],
            units="cm",
            name=DATASET["key"],
            source="Gonzalez & Giocomo 2024 eLife Methods: 'The dimensions of field are 1.3m x 1.5m'",
        ),
        tracked_point="not stated (the file's track_data columns x, y)",
    )


def _positions_read_as_mm(rec: Recording) -> Recording:
    # Trusts the "mm" label: a user converting to cm would divide by 10.
    assert rec.positions is not None
    return replace(rec, positions=rec.positions / 10.0)


HAZARDS = {"positions_read_as_mm": _positions_read_as_mm}


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    print(
        f"{rec.dataset}: {rec.activity.shape[1]} cells, {rec.times[-1] - rec.times[0]:.0f} s"
    )
    print(
        f"  positions {np.nanmin(rec.positions, 0)} to {np.nanmax(rec.positions, 0)} cm"
    )
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
