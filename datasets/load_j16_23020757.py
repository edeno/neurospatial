"""Comrie, Frank (curated by Denovellis) rat j16 hippocampus in a spatial bandit maze (Zenodo 23020757).

What was recorded: rat j16 on 2021-07-10, run epoch ``02_r1`` of the
spatial-bandit task of Comrie et al. (2026), with 203 spike-sorted
hippocampal units. Head position, velocity and orientation are sampled at
500 Hz; the file also holds the linearized position and the track graph the
authors used to linearize it (10 nodes, 9 corridors: three arms of two
branches each, joined at a centre).

Where it lives: Zenodo record 23020757 (CC BY 4.0), one ``.npz`` of numeric
and string arrays, 75.1 MB. The record states it has no pickled objects; it
is read with ``np.load(..., allow_pickle=False)``, which refuses any object
array instead of unpickling it.

What the loader converts:

- Nothing in scale. The record does not state the position unit. It is cm:
  the Frank lab's convention, the maze spans about 2 m, and the file's
  ``head_speed`` matches the speed of the stored positions (|step| / dt) in
  the same units per second. The span is range checked.
- Times stay as stored, in seconds of the Unix epoch (about 1.6e9 s);
  float64 resolves them to about 0.2 µs.
- Spike trains are cut from ``spike_times`` at ``spike_offsets``. 21 of the
  203 units have no spikes in this epoch; they are kept as empty trains, as
  the file lists them.
- ``head_orientation`` (radians, [-π, π]) is not used as head direction:
  heading minus the direction of movement while running has a circular mean
  of +13° (0.23 rad), with the same sign on each of the nine corridors,
  beyond the 0.2 rad frame check. The spread (20°) says it is the head's
  direction; the constant offset is either the headstage mounting or the
  rat's head posture, and the file cannot tell which.
  ``params["head_orientation"]`` keeps it for comparison.
- The track graph: nodes at ``track_node_positions``, edges
  ``track_edges`` with ``track_edge_distance`` as each edge's ``distance``.

Analysis choices (``params``): ``bin_size`` 4 cm; ``graph_bin_size`` 2 cm
along the track; ``min_speed`` 4 cm/s (running); ``dt`` 0.25 s and
alternating 60 s train/test blocks for decoding; ``linearized`` is the
file's ``linear_position`` (cm along the authors' edge order, with 15 cm
gaps between edges).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    checked_span,
    heading_vs_movement,
    validate_recording,
)

DATASET = {
    "key": "j16_23020757",
    "title": "Rat j16, 2021-07-10, run epoch 02_r1, hippocampal units in a spatial bandit maze",
    "source": "https://doi.org/10.5281/zenodo.23020757",
    "license": "CC-BY-4.0",
    "download_size": "75.1 MB",
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

_NAME = "j1620210710_02_r1_figure04_inputs.npz"
_URL = f"https://zenodo.org/api/records/23020757/files/{_NAME}/content"
_SHA256 = "60383b394b597e2900545548ecac7c53a8601038ace9dbeb42d7f9a5fe1c93b3"
_SPAN_CM = (150.0, 260.0)
RUNNING_CM_S = 10.0


def track_graph(arrays: dict[str, np.ndarray]) -> Any:
    """The authors' track graph: node ``pos`` in cm, edge ``distance`` in cm."""
    import networkx as nx

    graph = nx.Graph()
    for node, xy in zip(
        arrays["track_nodes"], arrays["track_node_positions"], strict=True
    ):
        graph.add_node(int(node), pos=(float(xy[0]), float(xy[1])))
    for (a, b), distance in zip(
        arrays["track_edges"], arrays["track_edge_distance"], strict=True
    ):
        graph.add_edge(int(a), int(b), distance=float(distance))
    return graph


def load(cache: Path | None = None) -> Recording:
    """Fetch the epoch's arrays; positions in cm, spikes in s, the track graph."""
    path = fetch(_URL, _NAME, sha256=_SHA256, subdir=DATASET["key"], cache=cache)
    with np.load(path, allow_pickle=False) as arrays:
        z = {name: arrays[name] for name in arrays.files}
    times = z["position_time"]
    positions = np.column_stack(
        [z["position/head_position_x"], z["position/head_position_y"]]
    )
    span = checked_span(positions, *_SPAN_CM)
    step_speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
    stored_speed = z["position/head_speed"][1:]
    agreement = float(np.corrcoef(step_speed, stored_speed)[0, 1])
    if agreement < 0.8:
        msg = f"head_speed no longer matches the positions' speed (r = {agreement:.2f})"
        raise ValueError(msg)
    orientation = z["position/head_orientation"]
    offset, spread = heading_vs_movement(
        times, positions, orientation, min_speed=RUNNING_CM_S
    )
    offsets = z["spike_offsets"]
    spikes = tuple(
        np.asarray(z["spike_times"][offsets[i] : offsets[i + 1]], dtype=float)
        for i in range(len(offsets) - 1)
    )
    train, test = alternating_blocks(times[0], times[-1], 60.0)
    return Recording(
        dataset=DATASET["key"],
        session="j16 20210710 02_r1",
        source=_URL,
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        spike_times=spikes,
        unit_ids=np.arange(len(spikes)),
        geometry={"graph": track_graph(z), "graph_root": 9},
        params={
            "bin_size": 4.0,
            "graph_bin_size": 2.0,
            "min_speed": 4.0,
            "dt": 0.25,
            "train_epochs": train,
            "test_epochs": test,
            "linearized": z["position/linear_position"],
            "head_orientation": orientation,
        },
        conversions=(
            "positions: head_position_x, head_position_y as stored (cm, inferred)",
            span,
            "positions: unit checked: head_speed vs |step| / dt correlation"
            f" {agreement:.2f}, medians {np.median(stored_speed):.2f} and"
            f" {np.median(step_speed):.2f} cm/s",
            "times: position_time as stored (Unix epoch seconds)",
            f"spike_times: spike_times split at spike_offsets; {len(spikes)} units,"
            f" {sum(len(s) == 0 for s in spikes)} with no spikes",
            "headings: head_orientation not used; minus the direction of movement"
            f" above {RUNNING_CM_S:g} cm/s: circular mean {np.degrees(offset):.1f} deg,"
            f" median spread {np.degrees(spread):.1f} deg (frame check: |mean| <="
            " 0.2 rad = 11.5 deg)",
            "geometry['graph']: track_nodes, track_node_positions, track_edges,"
            " track_edge_distance",
        ),
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
