"""Grieves et al. 2020 rat CA1 place cells in a 3-D climbing lattice (UCL RDR 10.5522/04/9926477).

What was recorded: rat 926 on 2018-04-06, CA1 tetrodes recorded wirelessly
in three parts: a square arena, a cubic lattice climbing frame (tilted, on
one vertex), and the square again. Position comes from a multi-camera 3-D
reconstruction at 50 Hz. This loader keeps the lattice part (about 1 h).

The session is read by ``neurospatial.datasets.load("grieves_2020",
"926_06042018")``, whose module (``neurospatial/datasets/grieves_2020.py``)
and docs page give the file (one Deflate64 member of the record's 12 GB
``Summary data.zip``, read by byte range), the conversions (positions mm to
cm from the stored velocity, cluster 0 left out, the head azimuth turned
into the positions' frame by the file's own 2-D to 3-D map), the
environment (the authors' 5 cm voxel grid, inferred) and the watch-outs.
``_record.from_public`` converts it to the survey's record.

Analysis choices are the public loader's ``params``: ``bin_size`` 5 cm (the
paper's voxel), no speed filter, the lattice part as ``analysis_epochs``,
``dt`` 0.25 s and alternating 60 s train/test blocks.

Survey-only: the declared lattice the environment fidelity study and the
driver use, as before: ``geometry["environment"]``, the file's 147 bars cut
at their crossings into a graph of segments (``_geometry.lattice_graph``)
and built with ``_geometry.graph_environment``, which keeps every segment.
The Supplementary Methods (read) state the tubes' diameter, 10 mm, but the
tracked point is the head, which a rat on a bar holds a median 6 cm from it,
so ``track_width`` is twice the 99th percentile of the head's distance from
the nearest bar (about 31 cm), a placement tolerance that places 99 % of the
samples. Also ``lattice_bars_cm`` (the public ``params["lattice_bars"]``),
the file's ``config3.lattice_type`` and the bars' side against the paper's
0.97 m.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from _geometry import distance_to_segments, graph_environment, lattice_graph
from _record import (
    Range,
    Recording,
    SurveyOnly,
    check_expectations,
    from_public,
    public_file,
    validate_recording,
)

DATASET = {
    "key": "grieves_2020_lattice",
    "public": "grieves_2020",
    "session": "926_06042018",
}

# Pre-registered from the readme, the zip listing and the session file's
# HDF5 layout (lattice_3d pox/poy/poz at 50 Hz; 30 clusters, 6 of them
# cluster 0) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 24,
    "n_dims": 3,
    "sampling_rate_hz": Range(49.0, 51.0),
    "duration_s": Range(3000.0, 4000.0),
    "lattice_extent_cm_per_axis": lambda rec: Range(100.0, 160.0).contains(
        np.nanpercentile(rec.positions, 99, axis=0)
        - np.nanpercentile(rec.positions, 1, axis=0)
    ),
}

_MM_PER_CM = 10.0
_N_CUBES = 6
# Grieves et al. 2020 Supplementary Methods: tubes 10 mm across.
_TUBE_CM = 1.0


def _lattice_type(folder: str) -> str:
    """The session file's ``config3.lattice_type``."""
    import h5py

    path = Path(folder) / f"{DATASET['session']}.mat"
    with h5py.File(path, "r") as mat:
        stored = mat["sdata"]["config3"]["lattice_type"][()]
    return "".join(map(chr, np.asarray(stored).ravel()))


def load(cache: Path | None = None) -> Recording:
    """The session from ``neurospatial.datasets``; 3-D positions in cm."""
    from neurospatial.datasets import grieves_2020
    from neurospatial.datasets import load as load_public

    rec = load_public(DATASET["public"], DATASET["session"], cache=cache)
    lattice_type = _lattice_type(
        public_file(DATASET["public"], DATASET["session"], cache=cache)
    )
    bars_cm = np.array(rec.params["lattice_bars"])
    graph = lattice_graph(bars_cm, _N_CUBES)
    tracked = rec.positions[np.isfinite(rec.positions).all(axis=1)]
    reach_pct = np.percentile(distance_to_segments(tracked, bars_cm), [50, 95, 99])
    width = 2 * float(reach_pct[2])
    bar_lengths = np.linalg.norm(bars_cm[:, 1] - bars_cm[:, 0], axis=1)
    why = "the declared lattice the environment fidelity study and the driver use"
    return from_public(
        rec,
        key=DATASET["key"],
        geometry={
            "environment": SurveyOnly(
                partial(
                    graph_environment,
                    graph,
                    bin_size=rec.params["bin_size"],
                    track_width=width,
                    units="cm",
                    name=DATASET["key"],
                ),
                f"{why}: the file's {len(bars_cm)} bars as a graph of "
                f"{graph.number_of_nodes()} crossings and "
                f"{graph.number_of_edges()} segments, every segment kept; "
                f"track_width {width:.1f} cm = 2 x the 99th percentile of the "
                f"head's distance from the nearest bar (median {reach_pct[0]:.1f}, "
                f"95th percentile {reach_pct[1]:.1f} cm), not the stated 10 mm "
                "tube diameter",
            ),
            "declared_from": SurveyOnly("authors' bars (file)", why),
            "lattice_bars_cm": SurveyOnly(bars_cm, f"{why}: params['lattice_bars']"),
            "lattice_type": SurveyOnly(lattice_type, f"{why}: config3.lattice_type"),
            # A placement tolerance chosen from the samples, not a stated
            # width: the tubes are 10 mm across (Supplementary Methods).
            "declared_track_width": SurveyOnly(width, why),
            "declared_track_width_kind": SurveyOnly(
                "placement tolerance (2 x the head's 99th percentile distance "
                "from a bar)",
                why,
            ),
            "declared_physical_width": SurveyOnly(_TUBE_CM, f"{why}: stated tubes"),
            "declared_physical_width_kind": SurveyOnly("stated", why),
            "declared_speed_thresholds": SurveyOnly((3.0, 30.0), why),
            # The file's bars against the paper's 0.97 m side.
            "environment_fit": SurveyOnly(
                {
                    "size_residual_lattice_side": float(
                        np.median(bar_lengths) / grieves_2020.LATTICE_SIDE - 1
                    )
                },
                f"{why}: the bars' length against the paper's 0.97 m side",
            ),
        },
    )


def millimetres_labelled_cm(rec: Recording) -> Recording:
    """The stored mm taken as cm: the lattice 10 x too large, bins 10 x too fine.

    Not in ``HAZARDS``: probed by hand, ``Environment.from_samples`` warns
    about the grid (388 x 429 x 371 cells, 109,862 visited bins), and the
    rate maps then ran for over an hour at 18 GB before the run was stopped.
    """
    assert rec.positions is not None
    return replace(rec, positions=rec.positions * _MM_PER_CM)


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s")
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
