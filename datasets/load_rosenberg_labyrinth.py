"""Rosenberg, Zhang, Perona & Meister (2021) mice exploring a binary-tree labyrinth.

What was recorded: one mouse (``B1``, rewarded) exploring a 6-level binary-tree
labyrinth in the dark for several hours, entering it from its home cage in
bouts. Each video frame (30 Hz) has the nose keypoint (DeepLabCut). No neurons.

Where it lives: the article's repository,
github.com/markusmeister/Rosenberg-2021-Repository at commit 9bac88c (MIT),
file ``outdata - tf files only/B1-tf`` (6.3 MB). The maze geometry is rebuilt
from ``code/MM_Maze_Utils.py:NewMaze`` of the same commit (MIT, © 2021 Markus
Meister), which lays the maze out as square cells one corridor wide, x and y in
0–14, y running top to bottom.

Reading the file: ``-tf`` files are Python pickles of a ``Traj`` dataclass of
NumPy arrays. They are read with a restricted unpickler that resolves only the
four globals those files contain: the ``Traj`` class (mapped to a local
dataclass), NumPy's array reconstruction, and ``dtype`` limited to the
built-in int64 and float64 types the files use. Any other global or dtype is
refused, which also blocks a dtype crafted to read raw bytes as object
pointers.

What the loader converts:

- Nose keypoints to maze-cell units with the repository's alignment,
  ``cell = 15 * ke - 0.5`` (``PlotTraj``; its comment calls it approximate).
- y flipped to point up (``y = 14 - y``), so left and right turns are those
  seen from above in a right-handed frame.
- Frames to seconds at 30 frames/s (``TimeInMaze`` in ``MM_Traj_Utils.py``);
  bouts are concatenated, with the time spent outside the maze left out.

The track graph has a node at each run end (127, the article's count) and one
at the entrance, joined along the corridors (127 edges).

Analysis choices (``params``):

- ``bin_size`` 0.25 cells for 2-D environments (4 bins per corridor width);
  ``graph_bin_size`` 0.5 cells along the track graph.
- ``min_speed`` 1 cell/s: about one corridor width per second.
"""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from _fetch import fetch
from _record import Range, Recording, check_expectations, validate_recording

DATASET = {
    "key": "rosenberg_labyrinth",
    "title": "Rosenberg et al. 2021, mouse in a 6-level binary-tree labyrinth (behavior only)",
    "source": "https://github.com/markusmeister/Rosenberg-2021-Repository",
    "license": "MIT",
    "download_size": "6.3 MB",
}

# Pre-registered from the article and the repository's maze code, before the
# first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 2,
    "sampling_rate_hz": Range(29.9, 30.1),
    "maze_has_127_nodes": lambda rec: rec.geometry["graph"].number_of_nodes() == 127,
    "maze_has_63_junctions": lambda rec: (
        sum(1 for _, degree in rec.geometry["graph"].degree() if degree == 3) == 63
    ),
}

_COMMIT = "9bac88c85a79be683d7a78ca02fdd929b200899b"
_MOUSE = "B1"
_URL = (
    "https://raw.githubusercontent.com/markusmeister/Rosenberg-2021-Repository/"
    f"{_COMMIT}/outdata%20-%20tf%20files%20only/{_MOUSE}-tf"
)
_SHA256 = "705321bc59abed1342b87184d52170f9cf494f340489f53bd0c3f95a717211df"
_LEVELS = 6
_FRAME_RATE = 30.0
_MAZE_MAX = 2 ** (_LEVELS // 2 + 1) - 2  # 14: cells run 0..14 on each axis


@dataclass
class _Traj:
    """Fields of the repository's ``Traj``: bouts' frames, cells, nose, nodes, rewards."""

    fr: Any = None
    ce: Any = None
    ke: Any = None
    no: Any = None
    re: Any = None


def _reconstruct() -> Any:
    try:
        from numpy._core.multiarray import _reconstruct as reconstruct
    except ImportError:  # NumPy 1.x
        from numpy.core.multiarray import _reconstruct as reconstruct
    return reconstruct


# The files' arrays are int64 and float64 only. Returning NumPy's built-in
# dtype objects (whose pickled state cannot be changed) instead of calling
# np.dtype keeps a crafted stream from building a dtype whose flags let raw
# bytes be read back as object pointers.
_DTYPES = {"i8": np.dtype("i8"), "f8": np.dtype("f8")}


def _dtype(spec: Any, align: bool = False, _copy: bool = False) -> np.dtype:
    if align or not isinstance(spec, str) or spec not in _DTYPES:
        msg = f"refusing dtype {spec!r} in a trajectory file (only i8 and f8 are expected)"
        raise pickle.UnpicklingError(msg)
    return _DTYPES[spec]


class _TrajUnpickler(pickle.Unpickler):
    """Unpickler that resolves only the globals a ``-tf`` file holds."""

    _ALLOWED: ClassVar[dict[tuple[str, str], Any]] = {
        ("MM_Traj_Utils", "Traj"): _Traj,
        ("numpy.core.multiarray", "_reconstruct"): _reconstruct(),
        ("numpy._core.multiarray", "_reconstruct"): _reconstruct(),
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): _dtype,
    }

    def find_class(self, module: str, name: str) -> Any:
        try:
            return self._ALLOWED[module, name]
        except KeyError:
            msg = f"refusing to load {module}.{name} from a trajectory file"
            raise pickle.UnpicklingError(msg) from None


def maze_graph(levels: int = _LEVELS) -> Any:
    """Track graph of the binary-tree labyrinth, in maze-cell units with y up.

    Follows ``NewMaze`` in the repository's ``MM_Maze_Utils.py``: run 0 enters
    from x = 0 to the centre; each run at level i ends 2**((levels - i) // 2)
    cells from its parent's end, along x at even levels and y at odd ones.
    Node ``r`` is the end of run ``r``; node ``"entrance"`` is where run 0
    starts. Edge ``distance`` is the corridor length in cells.
    """
    import networkx as nx

    centre = 2 ** (levels // 2) - 1
    ends: list[tuple[int, int]] = []
    parents: list[int] = []
    for level in range(levels + 1):
        along_x, along_y = (level + 1) % 2, level % 2
        length = 2 ** ((levels - level) // 2)
        for j in range(2**level):
            if level == 0:
                parents.append(-1)
                ends.append((centre, centre))
                continue
            parent = 2 ** (level - 1) - 1 + j // 2
            sign = 2 * (j % 2) - 1
            x0, y0 = ends[parent]
            parents.append(parent)
            ends.append((x0 + along_x * sign * length, y0 + along_y * sign * length))
    top = 2 ** (levels // 2 + 1) - 2
    graph = nx.Graph()

    def pos(x: float, y: float) -> tuple[float, float]:
        return (float(x), float(top - y))

    graph.add_node("entrance", pos=pos(0, centre))
    for run, (x, y) in enumerate(ends):
        graph.add_node(run, pos=pos(x, y))
    for run, parent in enumerate(parents):
        start = "entrance" if parent < 0 else parent
        a, b = (
            np.asarray(graph.nodes[start]["pos"]),
            np.asarray(graph.nodes[run]["pos"]),
        )
        graph.add_edge(start, run, distance=float(np.linalg.norm(a - b)))
    return graph


def load(cache: Path | None = None) -> Recording:
    """Fetch mouse B1's trajectory and return it with the maze graph."""
    path = fetch(
        _URL, f"{_MOUSE}-tf", sha256=_SHA256, subdir=DATASET["key"], cache=cache
    )
    traj = _TrajUnpickler(io.BytesIO(path.read_bytes())).load()

    times, nose = [], []
    for (start, _stop), keypoints in zip(traj.fr, traj.ke, strict=True):
        keypoints = np.asarray(keypoints, dtype=float)
        times.append((start + np.arange(len(keypoints))) / _FRAME_RATE)
        nose.append(keypoints)
    t = np.concatenate(times)
    cells = -0.5 + 15.0 * np.concatenate(nose)
    cells[:, 1] = _MAZE_MAX - cells[:, 1]
    bouts = np.column_stack(
        [
            np.asarray(traj.fr)[:, 0] / _FRAME_RATE,
            np.asarray(traj.fr)[:, 1] / _FRAME_RATE,
        ]
    )
    return Recording(
        dataset=DATASET["key"],
        session=_MOUSE,
        source=_URL,
        license=DATASET["license"],
        times=t,
        positions=cells,
        position_units="arbitrary",
        epochs={"bouts": bouts},
        geometry={
            "graph": maze_graph(),
            "graph_root": "entrance",
            "track_width": 1.0,  # NewMaze: cells are one corridor wide
            "units": "maze cells (one corridor width), y up",
        },
        params={"bin_size": 0.25, "graph_bin_size": 0.5, "min_speed": 1.0},
        conversions=(
            "positions: maze cells = 15 * nose keypoint - 0.5 (the repository's alignment)",
            f"positions: y flipped to point up (y = {_MAZE_MAX} - y)",
            f"times: frames / {_FRAME_RATE:g} s; bouts concatenated",
            "geometry['graph']: rebuilt from NewMaze (MM_Maze_Utils.py) plus an entrance node",
        ),
    )


def _image_y_down(rec: Recording) -> Recording:
    # Undoes the y flip: maze cells in image orientation, as stored.
    from dataclasses import replace

    assert rec.positions is not None
    positions = rec.positions.copy()
    positions[:, 1] = _MAZE_MAX - positions[:, 1]
    return replace(rec, positions=positions)


HAZARDS = {"image_y_down": _image_y_down}


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    print(f"{rec.dataset}: {len(rec.times)} frames in {len(rec.epochs['bouts'])} bouts")
    print(
        f"  positions {np.nanmin(rec.positions, 0)} to {np.nanmax(rec.positions, 0)} cells"
    )
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
