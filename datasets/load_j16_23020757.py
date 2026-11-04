"""Comrie, Frank (curated by Denovellis) rat j16 hippocampus in a spatial bandit maze (Zenodo 23020757).

What was recorded: rat j16 on 2021-07-10, run epoch ``02_r1`` of the
spatial-bandit task of Comrie et al. (2026), with 203 spike-sorted
hippocampal units, the head position at 500 Hz, the authors' linearized
position and their track graph (10 nodes, 9 corridors).

The session is read by ``neurospatial.datasets.load("comrie_2026")``, whose
module (``neurospatial/datasets/comrie_2026.py``) and docs page give the
file, the conversions, the declared track and the watch-outs, for example
why ``head_orientation`` is not used as the heading. ``_record.from_public``
converts it to the survey's record.

Analysis choices are the public loader's ``params``: ``bin_size`` 4 cm,
``graph_bin_size`` 2 cm, ``min_speed`` 10 cm/s (the preprint's locomotion
threshold; this survey used 4 cm/s before the preprint was read, and no
expectation depends on it), ``dt`` 0.25 s and alternating 60 s train/test
blocks. ``params["linearized"]`` is the file's ``linear_position``.

Survey-only: ``geometry["graph"]``, the authors' track graph (the declared
environment's linearization graph), which the expectations count, and
``geometry["graph_root"]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import (
    Range,
    Recording,
    SurveyOnly,
    check_expectations,
    from_public,
    validate_recording,
)

DATASET = {
    "key": "j16_23020757",
    "public": "comrie_2026",
    "session": "j1620210710_02_r1_figure04_inputs",
}

# Pre-registered from the catalog (203 units, 870,018 spikes, 709,321
# samples, track graph) and the .npz array headers (10 track nodes, 9
# edges) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 203,
    "n_dims": 2,
    "sampling_rate_hz": Range(499.0, 501.0),
    "total_spikes": lambda rec: sum(len(s) for s in rec.spike_times) == 870018,
    "graph_10_nodes_9_edges": lambda rec: (
        rec.geometry["graph"].number_of_nodes() == 10
        and rec.geometry["graph"].number_of_edges() == 9
    ),
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}


def load(cache: Path | None = None) -> Recording:
    """The session from ``neurospatial.datasets``; positions in cm, the track graph."""
    from neurospatial.datasets import load as load_public

    rec = load_public(DATASET["public"], DATASET["session"], cache=cache)
    return from_public(
        rec,
        key=DATASET["key"],
        geometry={
            "graph": SurveyOnly(
                rec.env.binning.linearization.graph,
                "the authors' track graph from the file (track_nodes,"
                " track_edges), the declared environment's linearization graph",
            ),
            "graph_root": SurveyOnly(
                9, "the node a graph environment of geometry['graph'] is walked from"
            ),
        },
    )


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s")
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
