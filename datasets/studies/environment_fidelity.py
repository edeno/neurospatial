"""Does each recording's Environment capture the apparatus: its shape and connectivity?

For each loader in ``run.LOADERS`` with positions, the environment is built
from the samples as the driver builds it without a declared apparatus
(``operations._samples_env``: ``Environment.from_samples`` from every tracked
sample at the loader's ``bin_size``) and measured against the tracked samples
and, where a source states it, the apparatus. Where the loader gives the
authors' track graph, it is also built with ``operations._graph_env``. Where
the loader declares the apparatus (``geometry["environment"]``, the
environment the driver analyses), that environment is measured the same way
against the same samples and reference, plus the share of samples it cannot
place (``measure_declared``).

Measures of the environment against the samples:

- **components**: connected parts of the bin graph.
- **holes** (2-D): groups of unvisited grid cells, 4-connected, that do not
  reach the edge of the grid, so visited bins enclose them on every side. An
  open arena should have none; a loop track has one per loop (its inside).
- **sparse bins**: share of bins with no tracked sample within half a bin of
  the bin centre; **thin bins**: share holding fewer than ``THIN_BIN_SAMPLES``
  samples (tracking jumps and reflections make such bins).
- **edge support**: for each neighbour pair, the distance from the midpoint of
  the two bin centres (the face, edge or corner the animal crosses between
  them) to the nearest tracked sample. A pair is *unsupported* beyond half a
  bin and *far* beyond one bin: no sample came near the place the link
  crosses (a wall, a gap between arms, or air).
- **corner links**: neighbour pairs that share only a corner (2-D) or an edge
  or corner (3-D) with every other cell of their shared 2 x 2 (x 2) block
  unvisited: the environment connects two places through a point.
- **geodesic / straight-line**: for ``N_PAIRS`` random pairs of tracked
  samples (seeded) at least ``MIN_PAIR_BINS`` bins apart in one component,
  the environment's path length between their bins over the straight-line
  distance. Near 1 in a convex arena; above 1 between arms of a maze.

Measures against the apparatus, where a source gives it:

- **reference distance**: the track as a graph of segments (the authors'
  graph, the lattice's bars, the authors' centreline, a centreline from the
  file's linearized coordinate, or a straight track's principal axis).
  Samples are projected onto the nearest segment and the distance between two
  of them is the shortest path along the segments, computed here with NumPy
  and SciPy, independently of neurospatial. The ratio of the environment's
  geodesic to it is near 1 when the environment follows the track; below 1
  the environment takes a short cut the track does not offer.
- **on-track share**: on a track of full width ``w`` (the stated width,
  else ``ASSUMED_TRACK_WIDTH_CM``), the share of samples within ``w / 2`` of
  the reference and of bin centres within ``w / 2`` plus half a bin. None for
  the lattice, where every point is within 11.4 cm of a bar.
- **arena outline**: for a declared arena, its outline as the loader placed
  it; else, for a stated square or circle, the stated shape is
  placed on the samples (a square along the samples' minimum rotated
  rectangle, a circle at the centre of the samples' extent) and the share of
  its area covered by bins, the share of bin centres more than half a bin
  outside it and the gap between the wall and the outermost bins are
  measured, with the samples' convex hull over its minimum rotated rectangle and over its minimum
  enclosing circle (about 1 and 0.64 for a square that is explored to the
  corners; about 0.79 and 1 for a circle).

Each row gets a verdict against the stated apparatus: ``matches``,
``mismatch`` (with the reason) or ``not verified`` when no source read for
this study states the geometry (``APPARATUS``).

Figures go to ``$NEUROSPATIAL_DATA_DIR/figures/environment_fidelity/`` (not
tracked): bins, neighbour links (thinned when there are many; unsupported
ones in red) and samples, with the reference overlaid; three projections and
height slices for 3-D recordings.

Writes ``results/studies/environment_fidelity.{md,json}``; given dataset
keys, it replaces only their rows (``merged``). Run:
``MPLBACKEND=Agg uv run --group datasets --extra nwb --extra pynapple python
datasets/studies/environment_fidelity.py [KEY ...]``.
"""

from __future__ import annotations

import importlib
import json
import sys
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _fetch import cache_root  # noqa: E402
from _geometry import MAX_SIZE_RESIDUAL, judge_declared  # noqa: E402
from _record import Recording  # noqa: E402
from operations import _graph_env, _samples_env, env_kind  # noqa: E402
from run import LOADERS  # noqa: E402

RESULTS = HERE.parent / "results" / "studies"
SEED = 0
N_PAIRS = 3000
MIN_PAIR_BINS = 5
ASSUMED_TRACK_WIDTH_CM = 10.0
MAX_DRAWN_EDGES = 15000
MAX_DRAWN_SAMPLES = 30000
COVER_STEP_BINS = 0.25
MAJOR_COMPONENT_SHARE = 0.01
THIN_BIN_SAMPLES = 5
ALIGN_STEP_SAMPLES = 10
LOOP_CLOSURE_SHARE = 0.1
ALIGN_MIN_SPEED = 10.0
ALIGN_DEGREES = 20.0
# Layouts whose bins are the active cells of a regular grid (``grid_edges`` and
# ``active_mask``): from samples, and a grid masked by a declared polygon.
GRID_LAYOUTS = ("RegularGrid", "ShapelyPolygon")
# Declared from the authors' geometry, used as stated: a fact a verdict can
# rest on. A layout drawn in the file's coordinate frame ("file's
# coordinates") is not one: only a stated size checks it.
AUTHORS_GEOMETRY = (
    "authors' graph",
    "authors' centrelines",
    "file's linearized coordinate",
    "authors' bars (file)",
)
FILE_FRAME_TRACK = "track drawn in the file's frame (scaled to the samples)"
# The placements each reference is the same geometry as: a declared track is
# not judged against its own geometry. The loader's ``geometry["graph"]`` is
# the authors' graph or, where the loader draws the track in the file's
# frame (Hafting's straight line), that line.
REFERENCE_SOURCE = {
    "authors' track graph": ("authors' graph",),
    FILE_FRAME_TRACK: ("file's coordinates",),
    "authors' track centrelines (file)": ("authors' centrelines",),
    "centreline from the file's linearized position": ("file's linearized coordinate",),
    "principal axis of the samples (straight track)": ("samples' principal axis",),
    "lattice bars (file lattice_3d lx/ly/lz)": ("authors' bars (file)",),
}


# --- Apparatus stated by a source read for this study ---------------------------


@dataclass(frozen=True)
class Apparatus:
    """What a source read for this study states about the apparatus.

    ``kind`` is "open" (an arena), "track" (a track or maze) or "3d".
    ``shape`` is "square" or "circle" for an arena with a stated size
    ``size`` (side or diameter, in the recording's units); ``track_width`` is
    the stated full width; ``n_parts`` the number of separate apparatus the
    samples cover; ``n_loops`` the loops of a track (each is a hole in a 2-D
    environment built from the samples); ``linearized_reference`` False when the file's linearized
    coordinate cannot serve as a 2-D centreline (it gives two places one
    value). ``source`` names what was read. ``verified`` is True when
    every value given here (the number of parts, and any shape, size, length
    or width) comes from a source read for this study; a size not given is
    not compared.
    """

    kind: str
    source: str
    verified: bool
    shape: str | None = None
    size: float | None = None
    track_width: float | None = None
    track_length: float | None = None
    n_parts: int = 1
    notes: str = ""
    linearized_reference: bool = True
    n_loops: int = 0


APPARATUS: dict[str, Apparatus] = {
    "sargolini_000582": Apparatus(
        "open",
        "NWB session_description: 'a 1 x 1 meter enclosure'",
        True,
        shape="square",
        size=100.0,
    ),
    "miniscope_002030": Apparatus(
        "open",
        "DANDI 002030 description: 'circular open-field arena (96 cm in diameter)'",
        True,
        shape="circle",
        size=96.0,
    ),
    "giocomo_000405": Apparatus(
        "open",
        "Gonzalez & Giocomo 2024 eLife Methods (open-access PDF): 'The dimensions"
        " of field are 1.3m x 1.5m'",
        True,
        shape="rectangle",
        notes="150 x 130 cm; placed by the loader",
    ),
    "gardner_2022_grid": Apparatus(
        "open",
        "Gardner et al. 2022 Methods (open-access PDF): 'square open-field (OF)"
        " arena of size 1.5 x 1.5 m', 50 cm walls",
        True,
        shape="square",
        size=150.0,
    ),
    "vollan_of": Apparatus(
        "open",
        "Vollan et al. 2025 Methods: 150 x 150 cm box, except one rat's dark-room"
        " session and the lit session after it, in 'a circular arena 150 cm in"
        " diameter'; this file's session type is 'open_field_light' and its"
        " samples are round (G-114)",
        True,
        shape="circle",
        size=150.0,
    ),
    "rsc_000987": Apparatus(
        "open",
        "NWB session_description: 'square arena'; Mehrotra, Levenstein et al."
        " 2023 bioRxiv Methods: square of 'removable walls (width: 80 cm, height:"
        " 50 cm)'",
        True,
        shape="square",
        size=80.0,
    ),
    "ca1ca3_001880": Apparatus(
        "open",
        "NWB: cheeseboard maze and a start box (no size)",
        False,
        n_parts=2,
    ),
    "kanter_2025": Apparatus(
        "open",
        "NWB epochs table: arena_dimensions [100, 100, 50] cm, square",
        True,
        shape="square",
        size=100.0,
    ),
    "watermaze_10248158": Apparatus(
        "open",
        "Experiment.json arena.bounds: circle of radius 95 units",
        True,
        shape="circle",
        size=190.0,
    ),
    "hc2_ec013_527": Apparatus(
        "open",
        "crcns-hc2-data-description.pdf: 'an elevated square platform (180 cm x"
        " 180 cm ...)', ec013.527 180 cm (pixels; the scale is inferred from that"
        " side)",
        True,
        shape="square",
        notes="placed by the loader in pixels",
    ),
    "senzai_000003": Apparatus(
        "open",
        "loader: open field and theta maze in one camera image, pixels (no size)",
        False,
        n_parts=2,
    ),
    "giocomo_001701": Apparatus(
        "track",
        "Aery Jones et al. 2026 Methods: '130-cm X-maze with 20-cm walls' (what"
        " the 130 cm spans is not stated; the samples span 112 cm tip to tip)",
        True,
        notes="X maze: four arms from a centre",
    ),
    "figure8_000552": Apparatus(
        "track",
        "NWB: camera (x, y) cm and the linearized position around the maze as a loop",
        True,
        notes="figure-8: two loops sharing the centre arm",
        linearized_reference=False,
        n_loops=2,
    ),
    "figure8_000552_linearized": Apparatus(
        "track",
        "NWB: linearized position 0-222 cm, both ends at camera (85, 43) cm",
        True,
        track_length=222.0,
        notes="1-D loop",
    ),
    "neurolab_001754_escher": Apparatus(
        "track",
        "DANDI 001754: Escher staircase, three 90 deg yaw turns interleaved with"
        " three 90 deg pitch turns, a closed loop; seen from one overhead camera",
        False,
    ),
    "grosmark_000044_circular_linearized": Apparatus(
        "track",
        "NWB: CircularMazeLinearizedPosition, 0 to 2.9 m around the loop",
        True,
        notes="1-D loop",
    ),
    "grosmark_000044_linear_linearized": Apparatus(
        "track",
        "NWB: LinearMazeLinearizedPosition along the 1.6 m stem",
        True,
    ),
    "j16_23020757_linearized": Apparatus(
        "track",
        "file: linear_position along the authors' edge order, 15 cm gaps between"
        " edges; the maze is one connected track",
        True,
    ),
    "rosenberg_labyrinth": Apparatus(
        "track",
        "Rosenberg et al. 2021 repository NewMaze (rebuilt by the loader);"
        " corridors one cell wide",
        True,
        track_width=1.0,
    ),
    "grosmark_000044_circular": Apparatus(
        "track",
        "Grosmark & Buzsaki 2016 and DANDI 000044: 1 m diameter circular maze",
        True,
        size=100.0,
        notes="loop",
        n_loops=1,
    ),
    "grosmark_000044_linear": Apparatus(
        "track",
        "Grosmark & Buzsaki 2016: 1.6 m stem with 15 cm reward areas at each"
        " end (1.9 m); DANDI 000044: 1.6 m linear platform. The file's"
        " linearized position covers the stem",
        True,
        track_length=160.0,
    ),
    "jadhav_000447": Apparatus(
        "track",
        "DANDI 000447: the novel W-track (run epochs 1 and 3; the loader keeps"
        " only these). Tang, Shin & Jadhav 2023 (the paper the dandiset is"
        " described by): '~80 x 80 cm with ~7-cm wide tracks'",
        True,
        track_width=7.0,
        notes="W-track: three arms joined at their base",
    ),
    "jadhav_000447_familiar": Apparatus(
        "track",
        "DANDI 000447: the familiar W-track (run epoch 2). Tang, Shin & Jadhav"
        " 2023: '~80 x 80 cm with ~7-cm wide tracks'",
        True,
        track_width=7.0,
        notes="W-track: three arms joined at their base",
    ),
    "vollan_lt": Apparatus(
        "track",
        "Vollan et al. 2025 Methods: 200 cm linear track",
        True,
        track_length=200.0,
    ),
    "valero_000568": Apparatus(
        "track",
        "NWB series description: PVC linear track '110 cm long, 6.35 cm wide'",
        True,
        track_length=110.0,
        track_width=6.35,
    ),
    "aeryjones_000165": Apparatus(
        "track",
        "Aery Jones et al. 2021 (Cell Reports) Methods: 'a 36 cm linear track'",
        True,
        track_length=36.0,
    ),
    "gonzalez_001695": Apparatus(
        "track",
        "Gonzalez et al. 2026 Methods: a linear track with a water well at either"
        " end (its size, Supplementary Table 1, not read)",
        True,
    ),
    "hafting_2008_nird": Apparatus(
        "track",
        "archive readme_hafting2008.txt: 'a linear track (mostly 320 cm)'; no"
        " width. The file's frame is scaled to that length (posx spans exactly"
        " -160 to 160) and the unit is read from it, so it is not compared",
        True,
    ),
    "takigawa_7114733": Apparatus(
        "track",
        "Tirole et al. 2022: two novel 2 m linear tracks, view-obstructing"
        " dividers; file: each track's centreline (199.8, 199.6 cm; not an"
        " independent check, cm_per_pixel comes from the 2 m length)",
        True,
        track_length=200.0,
        n_parts=2,
    ),
    "olafsdottir_5566548": Apparatus(
        "track",
        "Olafsdottir et al. 2016 Online Methods: Z-track of '10 cm wide runways',"
        " parallel sections 190 cm, diagonal 220 cm",
        True,
        track_width=10.0,
    ),
    "j16_23020757": Apparatus(
        "track",
        "file: the authors' track graph (10 nodes, 9 corridors); Comrie et al."
        " 2026 STAR Methods: 'Hallways were 6.5 cm wide'",
        True,
        track_width=6.5,
    ),
    "agarwal_24757638": Apparatus(
        "track",
        "Figshare description: 3-arm maze and a wheel (no dimensions)",
        True,
    ),
    "shipley_30902972": Apparatus(
        "track",
        "Shipley et al. 2026 Methods: '80cm x 10cm arms connected by an octagonal"
        " centre platform (24.7cm diameter, 10cm sides)'",
        True,
        track_width=10.0,
    ),
    "dyads_001845": Apparatus(
        "track",
        "DANDI 001845: paired W-mazes, this file the left rat's; Shukla et al."
        " 2026 Methods: 'W-shaped mazes (80 x 80 cm) with ~7-cm-wide track"
        " sections'",
        True,
        track_width=7.0,
    ),
    "hc3_ec013_669": Apparatus(
        "track",
        "Mizuseki et al. 2012 Behavioral Testing: 'the linear track (250 x 7"
        " cm2)'; the file is in pixels and its scale is inferred from the same"
        " 250 cm, so neither size is compared",
        True,
    ),
    "hc28_hpa": Apparatus(
        "track",
        "Jadhav et al. 2016 Supplemental Experimental Procedures: 'The W-track had"
        " dimensions of ~80 x 80 cm', '7 cm wide metal sections'",
        True,
        track_width=7.0,
        notes="W-track: three arms joined at their base",
    ),
    "hc31_anm190422": Apparatus(
        "track",
        "hc-31 data description: 'a novel 48-m long linear track' (1-D coordinate)",
        True,
    ),
    "hc18_train261": Apparatus(
        "track",
        "hc-18 and Drieu et al. 2018: a model train's circuit (no shape or size"
        " stated)",
        False,
        notes="loop",
        n_loops=1,
    ),
    "neurolab_001754": Apparatus(
        "track",
        "DANDI 001754: Escher staircase, three 90 deg yaw turns interleaved with"
        " three 90 deg pitch turns, a closed loop; five sessions of three"
        " apparatus in one image",
        False,
        n_parts=3,
    ),
    "yartsev_bat_demo": Apparatus(
        "3d",
        "Forli et al. 2025 Methods (Zotero full-text cache): indoor flight room"
        " 5.6 x 5.2 x 2.5 m, two feeders on the wall at one end; file r_lim"
        " ('boundaries of the room', NP_Bat_Demo.m) 5.8 x 5.2 x 2.3 m and the"
        " feeder coordinates r_fd in NP_Bat_Demo.m",
        True,
    ),
    "grieves_2020_lattice": Apparatus(
        "3d",
        "Grieves et al. 2020 Supplementary Methods: 6 x 6 x 6 cubic lattice"
        " 0.97 m a side, 150 mm tubes and 10 mm connectors, tilted; file"
        " lattice_3d lx/ly/lz bars and config3.lattice_type 'diagonal'",
        True,
        size=97.0,
    ),
}

# Feeder positions (m) for this demo session, from NP_Bat_Demo.m (``r_fd``).
BAT_FEEDERS_M = np.array([[2.77, 0.82, 1.75], [2.79, -0.99, 1.64]])


# --- Geometry of the bin grid ---------------------------------------------------


def grid_index(env: Any) -> NDArray[np.intp]:
    """Grid cell index of each bin of a regular-grid environment, (n_bins, n_dims)."""
    centers = np.asarray(env.bin_centers, dtype=float)
    return np.column_stack(
        [
            np.searchsorted(np.asarray(edges), centers[:, d]) - 1
            for d, edges in enumerate(env.grid_edges)
        ]
    )


def hole_sizes(active: NDArray[np.bool_]) -> NDArray[np.intp]:
    """Sizes (cells) of unvisited regions that visited cells enclose.

    Parameters
    ----------
    active : ndarray of bool, shape grid_shape
        Visited (active) grid cells.

    Returns
    -------
    ndarray, shape (n_holes,)
        Cell count of each 4-connected (2-D; 6-connected in 3-D) group of
        inactive cells that does not touch the edge of the grid, largest first.
    """
    padded = np.pad(~np.asarray(active, dtype=bool), 1, constant_values=True)
    labels, n = ndimage.label(
        padded, structure=ndimage.generate_binary_structure(padded.ndim, 1)
    )
    outside = labels[(0,) * padded.ndim]
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = [k for k in range(1, n + 1) if k != outside]
    return np.sort(sizes[keep])[::-1].astype(np.intp)


def corner_links(
    index: NDArray[np.intp], pairs: NDArray[np.intp], active: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """Neighbour pairs that meet only at a corner or edge of the grid.

    Parameters
    ----------
    index : ndarray, shape (n_bins, n_dims)
        Grid cell of each bin.
    pairs : ndarray, shape (n_edges, 2)
        Bin ids of each neighbour pair.
    active : ndarray of bool, shape grid_shape

    Returns
    -------
    ndarray of bool, shape (n_edges,)
        True for a pair that differs in two or more grid axes while every
        other cell of the block spanned by the two cells is inactive.
    """
    a, b = index[pairs[:, 0]], index[pairs[:, 1]]
    step = b - a
    diagonal = np.count_nonzero(step, axis=1) >= 2
    result = np.zeros(len(pairs), dtype=bool)
    n_dims = index.shape[1]
    for k in np.flatnonzero(diagonal):
        moving = np.flatnonzero(step[k])
        through = False
        for corner in np.ndindex(*(2,) * len(moving)):
            if all(c == 0 for c in corner) or all(c == 1 for c in corner):
                continue
            cell = a[k].copy()
            cell[moving] += np.asarray(corner) * step[k][moving]
            if (
                all(0 <= cell[d] < active.shape[d] for d in range(n_dims))
                and active[tuple(cell)]
            ):
                through = True
                break
        result[k] = not through
    return result


# --- Measures against samples ----------------------------------------------------


def nearest_sample_distance(
    points: NDArray[np.float64], tree: cKDTree
) -> NDArray[np.float64]:
    """Distance from each point to the nearest tracked sample."""
    return tree.query(points)[0]


def edge_midpoint_gaps(
    centers: NDArray[np.float64], pairs: NDArray[np.intp], tree: cKDTree
) -> NDArray[np.float64]:
    """Distance from the midpoint of each neighbour pair to the nearest sample.

    Parameters
    ----------
    centers : ndarray, shape (n_bins, n_dims)
    pairs : ndarray, shape (n_edges, 2)
    tree : cKDTree
        Of the tracked samples, shape (n_samples, n_dims).

    Returns
    -------
    ndarray, shape (n_edges,)
        The midpoint is where a path between the two bins crosses from one to
        the other (the shared face, edge or corner); a distance much larger
        than the sampling spacing means the animal was never there.
    """
    midpoints = 0.5 * (centers[pairs[:, 0]] + centers[pairs[:, 1]])
    return nearest_sample_distance(midpoints, tree)


def bin_graph(env: Any) -> csr_array:
    """Sparse symmetric matrix of the environment's open links, weighted by distance."""
    pairs = np.asarray(env.edge_pairs)
    centers = np.asarray(env.bin_centers, dtype=float)
    lengths = np.linalg.norm(centers[pairs[:, 0]] - centers[pairs[:, 1]], axis=1)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return csr_array(
        (np.concatenate([lengths, lengths]), (rows, cols)),
        shape=(env.n_bins, env.n_bins),
    )


def sample_pairs(
    n: int, rng: np.random.Generator
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """``N_PAIRS`` index pairs into ``n`` samples: 100 sources, the rest targets."""
    sources = rng.choice(n, size=min(100, n), replace=False)
    first = np.repeat(sources, -(-N_PAIRS // len(sources)))[:N_PAIRS]
    second = rng.choice(n, size=len(first), replace=True)
    return first, second


def geodesics(
    graph: csr_array, bins_a: NDArray[np.intp], bins_b: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Shortest-path length through ``graph`` between paired bins (inf if apart)."""
    unique, inverse = np.unique(bins_a, return_inverse=True)
    table = dijkstra(graph, directed=False, indices=unique)
    return table[inverse, bins_b]


# --- Reference track -------------------------------------------------------------


@dataclass
class Reference:
    """A track as straight segments between nodes.

    Attributes
    ----------
    nodes : ndarray, shape (n_nodes, n_dims)
    edges : ndarray, shape (n_edges, 2)
        Node ids at the two ends of each segment.
    label : str
        Where it comes from.
    width : float or None
        Full track width used for the on-track share, and whether it was stated.
    """

    nodes: NDArray[np.float64]
    edges: NDArray[np.intp]
    label: str
    width: float | None = None
    width_stated: bool = False
    _node_distances: NDArray[np.float64] | None = field(default=None, repr=False)

    @property
    def node_distances(self) -> NDArray[np.float64]:
        if self._node_distances is None:
            n = len(self.nodes)
            lengths = self.lengths
            graph = csr_array(
                (
                    np.concatenate([lengths, lengths]),
                    (
                        np.concatenate([self.edges[:, 0], self.edges[:, 1]]),
                        np.concatenate([self.edges[:, 1], self.edges[:, 0]]),
                    ),
                ),
                shape=(n, n),
            )
            self._node_distances = dijkstra(graph, directed=False)
        return self._node_distances

    @property
    def lengths(self) -> NDArray[np.float64]:
        return np.linalg.norm(
            self.nodes[self.edges[:, 1]] - self.nodes[self.edges[:, 0]], axis=1
        )

    @property
    def total_length(self) -> float:
        return float(self.lengths.sum())


def project(
    points: NDArray[np.float64], ref: Reference
) -> tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.float64]]:
    """Nearest point of the reference to each point.

    Returns
    -------
    distance : ndarray, shape (n_points,)
        Distance to the nearest segment.
    segment : ndarray, shape (n_points,)
        Index of that segment.
    along : ndarray, shape (n_points,)
        Position along it from its first node, in length units.
    """
    a = ref.nodes[ref.edges[:, 0]]
    ab = ref.nodes[ref.edges[:, 1]] - a
    length2 = np.einsum("ij,ij->i", ab, ab)
    best = np.full(len(points), np.inf)
    segment = np.zeros(len(points), dtype=np.intp)
    along = np.zeros(len(points))
    for k in range(len(a)):  # loop over segments (<= ~1000), vectorized over points
        t = np.clip((points - a[k]) @ ab[k] / max(length2[k], 1e-12), 0.0, 1.0)
        dist = np.linalg.norm(points - (a[k] + t[:, None] * ab[k]), axis=1)
        closer = dist < best
        best[closer] = dist[closer]
        segment[closer] = k
        along[closer] = t[closer] * np.sqrt(length2[k])
    return best, segment, along


def along_track_distance(
    ref: Reference,
    seg_a: NDArray[np.intp],
    along_a: NDArray[np.float64],
    seg_b: NDArray[np.intp],
    along_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Shortest distance along the reference between paired projected points.

    Parameters
    ----------
    ref : Reference
    seg_a, seg_b : ndarray, shape (n,)
        Segment of each point (from ``project``).
    along_a, along_b : ndarray, shape (n,)
        Position along that segment from its first node.

    Returns
    -------
    ndarray, shape (n,)
        Within one segment the separation along it; otherwise the shortest of
        the four routes through either end of each segment.
    """
    lengths = ref.lengths
    ends_a = (along_a, lengths[seg_a] - along_a)
    ends_b = (along_b, lengths[seg_b] - along_b)
    nodes_a = (ref.edges[seg_a, 0], ref.edges[seg_a, 1])
    nodes_b = (ref.edges[seg_b, 0], ref.edges[seg_b, 1])
    best = np.full(len(seg_a), np.inf)
    for i in range(2):
        for j in range(2):
            route = ends_a[i] + ref.node_distances[nodes_a[i], nodes_b[j]] + ends_b[j]
            best = np.minimum(best, route)
    same = seg_a == seg_b
    best[same] = np.minimum(best[same], np.abs(along_a[same] - along_b[same]))
    return best


def polyline_reference(
    points: NDArray[np.float64], *, closed: bool, label: str
) -> Reference:
    """A reference through ordered points; ``closed`` joins the last to the first."""
    n = len(points)
    edges = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    if closed:
        edges = np.vstack([edges, [n - 1, 0]])
    return Reference(np.asarray(points, dtype=float), edges.astype(np.intp), label)


def graph_reference(graph: Any, label: str) -> Reference:
    """A reference from a networkx graph whose nodes carry ``pos``."""
    names = list(graph.nodes)
    lookup = {name: k for k, name in enumerate(names)}
    nodes = np.array([graph.nodes[name]["pos"] for name in names], dtype=float)
    edges = np.array([[lookup[u], lookup[v]] for u, v in graph.edges], dtype=np.intp)
    return Reference(nodes, edges, label)


def lattice_reference(bars: NDArray[np.float64], n_cubes: int = 6) -> Reference:
    """The lattice's bars as segments between their crossings.

    Parameters
    ----------
    bars : ndarray, shape (n_bars, 2, 3)
        Start and end of each full-length bar.
    n_cubes : int
        Cubes along each bar: a bar has ``n_cubes + 1`` crossings.
    """
    steps = np.linspace(0.0, 1.0, n_cubes + 1)
    points = (bars[:, :1] + steps[None, :, None] * (bars[:, 1:] - bars[:, :1])).reshape(
        -1, 3
    )
    spacing = float(np.linalg.norm(bars[0, 1] - bars[0, 0])) / n_cubes
    groups = cKDTree(points).query_ball_point(points, 0.05 * spacing)
    label = -np.ones(len(points), dtype=np.intp)
    nodes = []
    for k, group in enumerate(groups):
        if label[k] < 0:
            members = [g for g in group if label[g] < 0]
            label[members] = len(nodes)
            nodes.append(points[members].mean(axis=0))
    per_bar = label.reshape(len(bars), n_cubes + 1)
    edges = np.column_stack([per_bar[:, :-1].ravel(), per_bar[:, 1:].ravel()])
    return Reference(np.array(nodes), edges, "lattice bars (file lattice_3d lx/ly/lz)")


def centreline_from_linearized(
    positions: NDArray[np.float64],
    linear: NDArray[np.float64],
    *,
    step: float,
) -> NDArray[np.float64]:
    """Mean position in each ``step`` of the linearized coordinate, in order.

    Parameters
    ----------
    positions : ndarray, shape (n_time, n_dims)
    linear : ndarray, shape (n_time,)
        The file's linearized coordinate, NaN off the track.
    step : float
        Bin of the linearized coordinate, same units.

    Returns
    -------
    ndarray, shape (n_points, n_dims)
        One point per occupied step, ordered along the coordinate.
    """
    keep = np.isfinite(linear) & np.isfinite(positions).all(axis=1)
    edges = np.arange(np.nanmin(linear[keep]), np.nanmax(linear[keep]) + step, step)
    which = np.digitize(linear[keep], edges) - 1
    points = [
        positions[keep][which == k].mean(axis=0)
        for k in range(len(edges) - 1)
        if np.any(which == k)
    ]
    return np.array(points)


def straight_reference(positions: NDArray[np.float64], label: str) -> Reference:
    """A straight track: the samples' principal axis from the 0.5th to 99.5th percentile."""
    tracked = positions[np.isfinite(positions).all(axis=1)]
    centre = tracked.mean(axis=0)
    axis = np.linalg.svd(tracked - centre, full_matrices=False)[2][0]
    along = (tracked - centre) @ axis
    low, high = np.percentile(along, [0.5, 99.5])
    nodes = np.vstack([centre + low * axis, centre + high * axis])
    return Reference(nodes, np.array([[0, 1]], dtype=np.intp), label)


def reference_for(rec: Recording, app: Apparatus | None) -> Reference | None:
    """The track the source gives for ``rec``, or None."""
    geometry, params = rec.geometry, rec.params
    stated_width = None if app is None else app.track_width
    ref: Reference | None = None
    if "lattice_bars_cm" in geometry:
        ref = lattice_reference(np.asarray(geometry["lattice_bars_cm"], dtype=float))
    elif "graph" in geometry:
        # A loader that draws its track in the file's frame (Hafting's line,
        # in a frame the authors scaled to the stated length) gives no
        # authors' graph.
        drawn = geometry.get("declared_from") == "file's coordinates"
        ref = graph_reference(
            geometry["graph"],
            FILE_FRAME_TRACK if drawn else "authors' track graph",
        )
    elif "track_centrelines_cm" in geometry:
        parts = [np.asarray(c, dtype=float) for c in geometry["track_centrelines_cm"]]
        nodes = np.vstack(parts)
        offsets = np.cumsum([0] + [len(p) for p in parts[:-1]])
        edges = np.vstack(
            [
                np.column_stack([np.arange(len(p) - 1), np.arange(1, len(p))]) + o
                for p, o in zip(parts, offsets, strict=True)
            ]
        )
        ref = Reference(
            nodes, edges.astype(np.intp), "authors' track centrelines (file)"
        )
    elif (
        "linearized" in params
        and rec.positions.shape[1] == 2
        and (app is None or app.linearized_reference)
    ):
        closed = app is not None and "loop" in app.notes
        points = centreline_from_linearized(
            rec.positions, np.asarray(params["linearized"], dtype=float), step=5.0
        )
        ref = polyline_reference(
            points,
            closed=closed,
            label="centreline from the file's linearized position",
        )
    elif (
        app is not None and app.track_length is not None and rec.positions.shape[1] == 2
    ):
        ref = straight_reference(
            rec.positions, "principal axis of the samples (straight track)"
        )
    if ref is None:
        return None
    if stated_width is not None:
        ref.width, ref.width_stated = stated_width, True
    elif rec.position_units == "cm" and "lattice_bars_cm" not in geometry:
        # No on-track share for the lattice: every point inside it is within
        # 11.4 cm (a cube's centre) of a bar, so a band says nothing.
        ref.width = ASSUMED_TRACK_WIDTH_CM
    return ref


# --- Arena outline ---------------------------------------------------------------


def stated_shape(samples: NDArray[np.float64], app: Apparatus) -> Any:
    """The stated square or circle placed on the samples, as a shapely polygon."""
    import shapely
    from shapely import affinity

    if app.shape == "circle":
        centre = 0.5 * (samples.min(axis=0) + samples.max(axis=0))
        return shapely.Point(centre).buffer(app.size / 2, quad_segs=64)
    hull = shapely.MultiPoint(samples).convex_hull
    with np.errstate(divide="ignore", invalid="ignore"):  # GEOS on axis-aligned hulls
        rectangle = shapely.minimum_rotated_rectangle(hull)
    corners = np.asarray(rectangle.exterior.coords)[:4]
    side = corners[1] - corners[0]
    angle = float(np.degrees(np.arctan2(side[1], side[0])))
    centre = np.asarray(rectangle.centroid.coords[0])
    half = app.size / 2
    square = shapely.box(-half, -half, half, half)
    return affinity.translate(affinity.rotate(square, angle, origin=(0, 0)), *centre)


def rectangle_of(samples: NDArray[np.float64]) -> tuple[float, float, float]:
    """Sides (long, short) and orientation (deg, in [0, 90)) of the samples' minimum rotated rectangle."""
    import shapely

    with np.errstate(divide="ignore", invalid="ignore"):  # GEOS on axis-aligned hulls
        rectangle = shapely.minimum_rotated_rectangle(
            shapely.MultiPoint(samples).convex_hull
        )
    corners = np.asarray(rectangle.exterior.coords)[:4]
    sides = np.linalg.norm(np.diff(np.vstack([corners, corners[:1]]), axis=0), axis=1)[
        :2
    ]
    edge = corners[1] - corners[0]
    angle = float(np.degrees(np.arctan2(edge[1], edge[0])) % 90.0)
    return float(sides.max()), float(sides.min()), min(angle, 90.0 - angle)


def outline_measures(
    env: Any, samples: NDArray[np.float64], app: Apparatus | None, shape: Any = None
) -> dict[str, Any]:
    """Coverage of the stated shape by bins, bin area outside it, and wall gaps.

    ``shape`` is the stated outline as placed by the loader (a declared
    environment's polygon); without it the stated square or circle is placed
    here (``stated_shape``).
    """
    import shapely

    if shape is None:
        shape = stated_shape(samples, app)
    bin_size = float(np.min(env.bin_sizes) ** (1 / env.n_dims))
    step = COVER_STEP_BINS * float(np.mean(np.diff(np.asarray(env.grid_edges[0]))))
    xmin, ymin, xmax, ymax = shape.bounds
    pad = 2 * bin_size
    xs = np.arange(
        min(xmin, samples[:, 0].min()) - pad, max(xmax, samples[:, 0].max()) + pad, step
    )
    ys = np.arange(
        min(ymin, samples[:, 1].min()) - pad, max(ymax, samples[:, 1].max()) + pad, step
    )
    grid = np.column_stack([np.repeat(xs, len(ys)), np.tile(ys, len(xs))])
    inside_shape = shapely.contains_xy(shape, grid[:, 0], grid[:, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        in_bins = np.asarray(env.contains(grid))
    covered = float(np.mean(in_bins[inside_shape]))
    # A bin whose centre is more than half a bin beyond the wall lies outside
    # the arena; bins at the wall overhang the samples by up to a bin, which is
    # not counted.
    centers = np.asarray(env.bin_centers, dtype=float)
    beyond = shapely.distance(shape, shapely.points(centers))
    outside = float(np.mean(beyond > bin_size / 2))
    # Gap between the wall and the outermost bin, measured inward from the wall
    # at points along it: the distance from the wall point to the nearest point
    # of a bin.
    boundary = shape.exterior
    wall_points = np.array(
        [
            boundary.interpolate(f, normalized=True).coords[0]
            for f in np.linspace(0, 1, 361)[:-1]
        ]
    )
    bin_points = grid[in_bins]
    gaps = (
        cKDTree(bin_points).query(wall_points)[0]
        if len(bin_points)
        else np.full(360, np.nan)
    )
    long, short, angle = rectangle_of(samples)
    hull = shapely.MultiPoint(samples).convex_hull
    return {
        "hull_over_min_rectangle": float(hull.area / (long * short)),
        "hull_over_min_circle": float(
            hull.area / shapely.minimum_bounding_circle(hull).area
        ),
        "shape": None if app is None else app.shape,
        "stated_size": None if app is None else app.size,
        "sample_rectangle": [round(long, 1), round(short, 1)],
        "sample_rectangle_angle_deg": round(angle, 1),
        "stated_area_covered": covered,
        "bin_centres_outside_stated": outside,
        "wall_gap_median": float(np.median(gaps)),
        "wall_gap_p90": float(np.percentile(gaps, 90)),
        "wall_gap_bins_median": float(np.median(gaps)) / bin_size,
    }


# --- One recording ---------------------------------------------------------------


def tracked(rec: Recording) -> NDArray[np.float64]:
    positions = np.asarray(rec.positions, dtype=float)
    return positions[np.isfinite(positions).all(axis=1)]


def measure_env(
    env: Any, samples: NDArray[np.float64], rng: np.random.Generator
) -> dict[str, Any]:
    """Every sample-based measure of one environment."""
    centers = np.asarray(env.bin_centers, dtype=float)
    pairs = np.asarray(env.edge_pairs)
    bin_size = (
        float(np.mean(np.diff(np.asarray(env.grid_edges[0]))))
        if env.layout_type in GRID_LAYOUTS
        else float(np.median(np.asarray(env.bin_sizes)))
    )
    tree = cKDTree(samples)
    gaps = edge_midpoint_gaps(centers, pairs, tree) if len(pairs) else np.empty(0)
    graph = bin_graph(env)
    n_parts, part = connected_components(graph, directed=False)
    sizes = np.sort(np.bincount(part))[::-1]
    out: dict[str, Any] = {
        "n_bins": int(env.n_bins),
        "n_dims": int(env.n_dims),
        "n_edges": len(pairs),
        "bin_size": bin_size,
        "components": int(n_parts),
        "major_components": int(np.sum(sizes >= MAJOR_COMPONENT_SHARE * env.n_bins)),
        "component_sizes": sizes[:5].tolist(),
        "sparse_bins": float(
            np.mean(nearest_sample_distance(centers, tree) > bin_size / 2)
        ),
        "unsupported_edges": float(np.mean(gaps > bin_size / 2)) if len(gaps) else 0.0,
        "far_edges": float(np.mean(gaps > bin_size)) if len(gaps) else 0.0,
        "edge_gap_max_bins": float(gaps.max() / bin_size) if len(gaps) else 0.0,
    }
    if env.layout_type in GRID_LAYOUTS:
        active = np.asarray(env.active_mask)
        index = grid_index(env)
        corners = corner_links(index, pairs, active)
        out["corner_links"] = int(corners.sum())
        if env.n_dims == 2:
            holes = hole_sizes(active)
            out["holes"] = len(holes)
            # Holes of 1 % of the bins or more: the inside of a loop, not a
            # patch the animal happened to miss.
            out["major_holes"] = int(
                np.sum(holes >= MAJOR_COMPONENT_SHARE * env.n_bins)
            )
            out["hole_bins"] = int(holes.sum())
            out["largest_hole_bins"] = int(holes[0]) if len(holes) else 0
        out["_corners"] = corners
    out["_gaps"] = gaps
    out["_part"] = part
    # Geodesic over straight line for random sample pairs in one component.
    bins = np.asarray(env.bin_at(samples))
    counts = np.bincount(bins[bins >= 0], minlength=env.n_bins)
    out["thin_bins"] = float(np.mean(counts < THIN_BIN_SAMPLES))
    first, second = sample_pairs(len(samples), rng)
    straight = np.linalg.norm(samples[first] - samples[second], axis=1)
    # A declared environment leaves some samples outside (bin -1): skip them.
    inside = (bins[first] >= 0) & (bins[second] >= 0)
    keep = (
        inside
        & (straight >= MIN_PAIR_BINS * bin_size)
        & (part[bins[first]] == part[bins[second]])
    )
    first, second, straight = first[keep], second[keep], straight[keep]
    geo = geodesics(graph, bins[first], bins[second])
    ratio = geo / straight
    out["geodesic_ratio_median"] = (
        float(np.median(ratio)) if len(ratio) else float("nan")
    )
    out["geodesic_ratio_p95"] = (
        float(np.percentile(ratio, 95)) if len(ratio) else float("nan")
    )
    out["n_pairs"] = len(ratio)
    out["_pairs"] = (first, second, geo, straight)
    return out


def on_track_shares(
    sample_distance: NDArray[np.float64],
    bin_distance: NDArray[np.float64],
    width: float,
    bin_size: float,
) -> tuple[float, float]:
    """Shares of samples and of bins on a track of full width ``width``.

    A sample is on the track within ``width / 2`` of its centreline; a bin is
    when its cell can reach the track, its centre within ``width / 2 +
    bin_size / 2``.
    """
    on_samples = float(np.mean(sample_distance <= width / 2))
    on_bins = float(np.mean(bin_distance <= width / 2 + bin_size / 2))
    return on_samples, on_bins


JUMP_FACTOR = 3.0


def jump_links(
    env: Any,
    ref: Reference,
    segment: NDArray[np.intp],
    along: NDArray[np.float64],
    offset: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Links between bins that the stated layout puts far apart along the track.

    Each bin centre is projected onto the reference (``segment``, ``along``
    and its distance ``offset`` from ``project``); a link is a jump when the
    distance along the reference between its two bins is more than
    ``JUMP_FACTOR`` times the link's own length plus both bins' offsets (a
    bin beside an inner corner projects onto the other segment, round the
    corner): it joins corridors through a wall or a gap the layout keeps
    apart, whatever the samples near its midpoint. Unlike a gap in the
    samples, this does not need the wall to be wider than a bin.
    """
    pairs = np.asarray(env.edge_pairs)
    if not len(pairs):
        return np.zeros(0, dtype=bool)
    centers = np.asarray(env.bin_centers, dtype=float)
    length = np.linalg.norm(centers[pairs[:, 0]] - centers[pairs[:, 1]], axis=1)
    distance = along_track_distance(
        ref,
        segment[pairs[:, 0]],
        along[pairs[:, 0]],
        segment[pairs[:, 1]],
        along[pairs[:, 1]],
    )
    slack = offset[pairs[:, 0]] + offset[pairs[:, 1]]
    return distance > JUMP_FACTOR * length + slack


def against_reference(
    env_out: dict[str, Any], env: Any, samples: NDArray[np.float64], ref: Reference
) -> dict[str, Any]:
    """On-track shares and the environment's geodesic over the distance along the track."""
    dist_s, seg_s, along_s = project(samples, ref)
    centers = np.asarray(env.bin_centers, dtype=float)
    dist_b, seg_b, along_b = project(centers, ref)
    jumps = jump_links(env, ref, seg_b, along_b, dist_b)
    first, second, geo, _ = env_out["_pairs"]
    along = along_track_distance(
        ref, seg_s[first], along_s[first], seg_s[second], along_s[second]
    )
    usable = (
        np.isfinite(along)
        & np.isfinite(geo)
        & (along >= MIN_PAIR_BINS * env_out["bin_size"])
    )
    ratio = geo[usable] / along[usable]
    out: dict[str, Any] = {
        "reference": ref.label,
        "reference_length": ref.total_length,
        "sample_to_reference_median": float(np.median(dist_s)),
        "sample_to_reference_p95": float(np.percentile(dist_s, 95)),
        "bin_to_reference_p95": float(np.percentile(dist_b, 95)),
        "env_over_track_median": float(np.median(ratio))
        if len(ratio)
        else float("nan"),
        "env_over_track_p05": float(np.percentile(ratio, 5))
        if len(ratio)
        else float("nan"),
        "share_pairs_shortcut": float(np.mean(ratio < 0.9))
        if len(ratio)
        else float("nan"),
        "jump_links": int(jumps.sum()),
        "jump_link_share": float(np.mean(jumps)) if len(jumps) else 0.0,
    }
    if ref.width is not None:
        out["track_width"] = ref.width
        out["track_width_stated"] = ref.width_stated
        out["samples_on_track"], out["bins_on_track"] = on_track_shares(
            dist_s, dist_b, ref.width, env_out["bin_size"]
        )
    return out


def measure_graph_env(rec: Recording) -> dict[str, Any] | None:
    """The driver's track-graph environment, when the loader gives a graph."""
    import networkx as nx

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = _graph_env(rec, {})
    if env is None:
        return None
    graph = rec.geometry["graph"]
    return {
        "graph_edges": graph.number_of_edges(),
        "graph_is_tree": bool(nx.is_tree(graph)),
        "n_bins": int(env.n_bins),
        "components": int(connected_components(bin_graph(env), directed=False)[0]),
    }


def one_d_measures(rec: Recording, env: Any) -> dict[str, Any]:
    """Coverage, gaps and loop closure of a 1-D environment built from a linearized loop."""
    centers = np.asarray(env.bin_centers, dtype=float)[:, 0]
    bin_size = float(np.mean(np.diff(np.asarray(env.grid_edges[0]))))
    lo, hi = float(np.nanmin(rec.positions)), float(np.nanmax(rec.positions))
    first, last = int(np.argmin(centers)), int(np.argmax(centers))
    end_to_end = float(
        geodesics(bin_graph(env), np.array([first]), np.array([last]))[0]
    )
    spacing = np.diff(np.sort(centers))
    out = {
        "range": [lo, hi],
        "bins_over_range": float(env.n_bins * bin_size / (hi - lo)),
        "n_gaps": int(np.sum(spacing > 1.5 * bin_size)),
        "largest_gap": float(spacing.max() - bin_size) if len(spacing) else 0.0,
        "end_to_end_geodesic": end_to_end,
    }
    camera = rec.geometry.get("positions_2d")
    if camera is not None:
        lin = np.asarray(rec.positions, dtype=float)[:, 0]
        cam = np.asarray(camera, dtype=float)
        ok = np.isfinite(lin) & np.isfinite(cam).all(axis=1)
        near_start = cam[ok][lin[ok] < lo + bin_size].mean(axis=0)
        near_end = cam[ok][lin[ok] > hi - bin_size].mean(axis=0)
        out["camera_distance_between_ends"] = float(
            np.linalg.norm(near_start - near_end)
        )
    return out


def bat_room(
    env: Any, samples: NDArray[np.float64], room: NDArray[np.float64], rec: Recording
) -> dict[str, Any]:
    """The bat's environment against the room bounds and feeders the file states."""
    centers = np.asarray(env.bin_centers, dtype=float)
    half = 0.5 * float(np.mean(np.diff(np.asarray(env.grid_edges[0]))))
    outside_bins = ~np.all((centers >= room[:, 0]) & (centers <= room[:, 1]), axis=1)
    outside_samples = ~np.all((samples >= room[:, 0]) & (samples <= room[:, 1]), axis=1)
    low_gap = (centers.min(axis=0) - half) - room[:, 0]
    high_gap = room[:, 1] - (centers.max(axis=0) + half)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feeders_inside = np.asarray(env.contains(BAT_FEEDERS_M)).tolist()
    feeder_to_bin = np.min(
        np.linalg.norm(centers[None] - BAT_FEEDERS_M[:, None], axis=2), axis=1
    )
    in_flight = np.zeros(len(rec.times), dtype=bool)
    for start, stop in np.asarray(rec.epochs["flight"]):
        in_flight |= (rec.times >= start) & (rec.times <= stop)
    positions = np.asarray(rec.positions, dtype=float)
    finite = np.isfinite(positions).all(axis=1)
    rest = positions[finite & ~in_flight]
    near_feeder = (
        np.min(np.linalg.norm(rest[:, None] - BAT_FEEDERS_M[None], axis=2), axis=1)
        < 0.3
    )
    return {
        "room_m": room.tolist(),
        "bins_outside_room": float(outside_bins.mean()),
        "samples_outside_room": float(outside_samples.mean()),
        "gap_to_low_walls_m": low_gap.round(2).tolist(),
        "gap_to_high_walls_m": high_gap.round(2).tolist(),
        "feeders_in_env": feeders_inside,
        "feeder_to_nearest_bin_centre_m": feeder_to_bin.round(2).tolist(),
        "rest_share_of_samples": float(len(rest) / max(int(finite.sum()), 1)),
        "rest_samples_within_0.3m_of_feeder": float(near_feeder.mean()),
    }


def lattice_measures(
    env: Any, samples: NDArray[np.float64], ref: Reference, rec: Recording
) -> dict[str, Any]:
    """The 3-D lattice: voxels against the bars, and a graph built from the bars."""
    import networkx as nx

    from neurospatial import Environment

    centers = np.asarray(env.bin_centers, dtype=float)
    spacing = float(np.median(ref.lengths))
    face_centre = spacing / 2  # distance from a face's centre to its bars
    cube_centre = spacing / np.sqrt(2)  # distance from a cube's centre to its bars
    d_samples = project(samples, ref)[0]
    d_bins = project(centers, ref)[0]
    graph = nx.Graph()
    for k, node in enumerate(ref.nodes):
        graph.add_node(k, pos=tuple(node))
    for (u, v), length in zip(ref.edges, ref.lengths, strict=True):
        graph.add_edge(int(u), int(v), distance=float(length))
    tree_order = list(nx.dfs_edges(graph, source=0))
    width = 2 * float(np.percentile(d_samples, 99))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tree_env = Environment.from_graph(
            graph, tree_order, bin_size=rec.params["bin_size"], units="cm"
        )
    listed = {frozenset(e) for e in tree_order}
    full_order = tree_order + [e for e in graph.edges if frozenset(e) not in listed]
    bar_env = Environment.from_graph(
        graph,
        full_order,
        bin_size=rec.params["bin_size"],
        units="cm",
        track_width=width,
    )
    occupancy = np.asarray(bar_env.occupancy(rec.times, rec.positions))
    placed = np.asarray(bar_env.bin_at(samples))
    return {
        "n_bars": len(np.asarray(rec.geometry["lattice_bars_cm"])),
        "bar_spacing": spacing,
        "lattice_side": float(
            np.linalg.norm(
                np.diff(np.asarray(rec.geometry["lattice_bars_cm"])[0], axis=0)
            )
        ),
        "lattice_type": rec.geometry.get("lattice_type"),
        "sample_to_bar_pct": np.percentile(d_samples, [5, 25, 50, 75, 95, 99])
        .round(1)
        .tolist(),
        "samples_beyond_face_centre": float(np.mean(d_samples > face_centre)),
        "voxels_beyond_face_centre": float(np.mean(d_bins > face_centre)),
        "voxels_beyond_cube_centre": float(np.mean(d_bins > cube_centre)),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "from_graph_dfs_order_bins": int(tree_env.n_bins),
        "from_graph_dfs_order_edges_kept": len(tree_order),
        "from_graph_dfs_order_warnings": len(caught),
        "from_graph_all_edges_bins": int(bar_env.n_bins),
        "from_graph_all_edges_components": int(
            connected_components(bin_graph(bar_env), directed=False)[0]
        ),
        "from_graph_track_width": width,
        "from_graph_occupancy_share_of_bins": float(np.mean(occupancy > 0)),
        "from_graph_samples_placed": float(np.mean(placed >= 0)),
        "_bar_env": bar_env,
    }


def face_adjacent(
    foot: NDArray[np.float64], direction: NDArray[np.float64], spacing: float
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Pairs of points on parallel bars one spacing apart across a cube face.

    Parameters
    ----------
    foot : ndarray, shape (n, 3)
        Each point's nearest point on its bar.
    direction : ndarray, shape (n, 3)
        Unit direction of that bar.
    spacing : float
        Distance between neighbouring parallel bars.

    Returns
    -------
    i, j : ndarray, shape (n_pairs,)
        Pairs (i < j) on parallel bars whose feet are one spacing apart (within
        20 %) and offset across the bars, not along them: the offset makes an
        angle of more than about 84 degrees with the bar (|cos| < 0.1). Two
        consecutive segments of one bar are also parallel and one spacing
        apart, but offset along the bar, so they are left out.
    """
    parallel = np.abs(np.abs(direction @ direction.T) - 1) < 1e-3
    offset = foot[None, :, :] - foot[:, None, :]
    separation = np.linalg.norm(offset, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        along = np.abs(np.einsum("ijk,ik->ij", offset, direction)) / separation
    across = np.nan_to_num(along, nan=1.0) < 0.1
    near = np.abs(separation - spacing) < 0.2 * spacing
    i, j = np.nonzero(np.triu(parallel & near & across, 1))
    return i, j


def adjacent_bar_pairs(
    env_graph: csr_array, env: Any, samples: NDArray[np.float64], ref: Reference
) -> dict[str, Any]:
    """Distances between samples at the middle of adjacent parallel bars.

    A sample counts when it is within a third of the spacing of a bar and in
    the middle fifth of a segment between crossings. Two such samples on
    parallel bars that bound one cube face (``face_adjacent``) are one spacing
    apart straight across the face; along the bars the route is half a
    segment to a crossing, the crossing segment and half a segment back:
    about two spacings.
    """
    dist, seg, along = project(samples, ref)
    lengths = ref.lengths
    spacing = float(np.median(lengths))
    middle = (np.abs(along / lengths[seg] - 0.5) < 0.1) & (dist < spacing / 3)
    idx = np.flatnonzero(middle)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(idx, size=min(len(idx), 1500), replace=False)
    a = ref.nodes[ref.edges[:, 0]]
    direction = (ref.nodes[ref.edges[:, 1]] - a) / lengths[:, None]
    foot = a[seg[idx]] + along[idx, None] * direction[seg[idx]]
    i, j = face_adjacent(foot, direction[seg[idx]], spacing)
    bins = np.asarray(env.bin_at(samples[idx]))
    geo = geodesics(env_graph, bins[i], bins[j])
    track = along_track_distance(
        ref, seg[idx][i], along[idx][i], seg[idx][j], along[idx][j]
    )
    straight = np.linalg.norm(samples[idx][i] - samples[idx][j], axis=1)
    return {
        "n_pairs": len(i),
        "straight_median": float(np.median(straight)),
        "env_geodesic_median": float(np.median(geo)),
        "env_geodesic_p05_p95": np.percentile(geo, [5, 95]).round(1).tolist(),
        "along_bars_median": float(np.median(track)),
        "share_env_below_075_bars": float(np.mean(geo < 0.75 * track)),
    }


def bar_alignment(rec: Recording, ref: Reference) -> dict[str, Any]:
    """How closely head movement follows the bars' three directions.

    Steps over ``ALIGN_STEP_SAMPLES`` samples faster than ``ALIGN_MIN_SPEED``
    (cm/s): the share within ``ALIGN_DEGREES`` of the nearest bar direction,
    against the same share for isotropic random directions (seeded).
    """
    positions = np.asarray(rec.positions, dtype=float)
    k = ALIGN_STEP_SAMPLES
    step = positions[k:] - positions[:-k]
    length = np.linalg.norm(step, axis=1)
    speed = length / (rec.times[k:] - rec.times[:-k])
    moving = np.isfinite(speed) & (speed > ALIGN_MIN_SPEED)
    directions = step[moving] / length[moving, None]
    lengths = ref.lengths
    axes = (ref.nodes[ref.edges[:, 1]] - ref.nodes[ref.edges[:, 0]]) / lengths[:, None]
    first = np.argmax(np.abs(axes) > 1e-6, axis=1)
    sign = np.sign(axes[np.arange(len(axes)), first])
    axes = np.unique(np.round(axes * sign[:, None], 6), axis=0)

    def aligned(v: NDArray[np.float64]) -> float:
        cosine = np.clip(np.abs(v @ axes.T).max(axis=1), 0.0, 1.0)
        return float(np.mean(np.degrees(np.arccos(cosine)) < ALIGN_DEGREES))

    random = np.random.default_rng(SEED).normal(size=(100_000, 3))
    random /= np.linalg.norm(random, axis=1)[:, None]
    return {
        "n_axes": len(axes),
        "n_steps": int(moving.sum()),
        "share_within_degrees": aligned(directions),
        "share_isotropic": aligned(random),
    }


# References built from the recording's own samples. A track's shape judged
# against one of them is not judged against anything the source states.
SAMPLE_DERIVED_REFERENCES = (
    "centreline from the file's linearized position",
    "principal axis of the samples (straight track)",
    "track drawn in the file's frame (scaled to the samples)",
)


def verdict(row: dict[str, Any], app: Apparatus | None) -> tuple[str, str]:
    """Verdict against the stated apparatus, and the reasons for a mismatch.

    "matches" or "mismatch" rests only on facts a source states
    independently of the samples; with none, the verdict is "not verified"
    and the measurements are reported. Facts judged, each within
    ``_geometry.MAX_SIZE_RESIDUAL`` where it is a size: a track's stated
    loops (holes of ``MAJOR_COMPONENT_SHARE`` of the bins or more), a
    square's or circle's coverage, a track's stated length and a circular
    track's circumference (pi x its diameter) against the length the samples
    give, a lattice's side, the room's bounds, a 1-D loop whose ends meet,
    and, against an independent reference (the authors' graph, centrelines
    or bars, not a reference built from the samples:
    ``SAMPLE_DERIVED_REFERENCES``), links that join places far apart along
    it (``jump_links``) and the environment's geodesic over the distance
    along it (median 0.9 or more). The number of separate parts and, on a
    tree or straight track, loop-sized holes can make a mismatch but are not
    enough for a match. The share of bins beyond a stated width is a
    tracking measure (the tracked head swings past an unwalled track), not
    judged. A declared environment is judged by ``measure_declared``.
    """
    if app is None or not app.verified:
        return "not verified", "no source read here states the geometry"
    reasons: list[str] = []
    facts: list[str] = []
    env = row["env"]
    if env["major_components"] != app.n_parts:
        n = env["major_components"]
        reasons.append(
            f"{n} part{'s' if n != 1 else ''} with 1% of the bins or more, apparatus"
            f" has {app.n_parts}"
        )
    outline = row.get("outline")
    if outline:
        facts.append("stated outline")
        if outline["stated_area_covered"] < 0.9:
            filled = row["open_field_preset"]["stated_area_covered"]
            reasons.append(
                f"bins cover {outline['stated_area_covered']:.0%} of the stated"
                f" {app.shape} ({filled:.0%} with holes filled)"
            )
        if outline["bin_centres_outside_stated"] > 0.05:
            reasons.append(
                f"{outline['bin_centres_outside_stated']:.0%} of bins outside it"
            )
    if app.kind == "track" and "major_holes" in env:
        if app.n_loops > 0:
            facts.append("stated loops")
        if env["major_holes"] != app.n_loops:
            reasons.append(
                f"{env['major_holes']} loop-sized hole{'s' if env['major_holes'] != 1 else ''}"
                f" (1% of the bins or more), apparatus has {app.n_loops}"
                f" loop{'s' if app.n_loops != 1 else ''}"
            )
    track = row.get("reference")
    if track:
        if app.size is not None and "loop" in app.notes and app.kind == "track":
            facts.append("stated diameter")
            stated = np.pi * app.size
            if abs(track["reference_length"] / stated - 1) > MAX_SIZE_RESIDUAL:
                reasons.append(
                    f"loop {track['reference_length']:.0f} round, stated diameter"
                    f" {app.size:g} ({stated:.0f} round)"
                )
        if app.track_length is not None:
            facts.append("stated length")
            stated = app.track_length * app.n_parts
            if abs(track["reference_length"] / stated - 1) > MAX_SIZE_RESIDUAL:
                reasons.append(
                    f"track {track['reference_length']:.0f} long, stated {stated:g}"
                )
        if track.get("reference") not in SAMPLE_DERIVED_REFERENCES:
            facts.append(track.get("reference") or "an independent reference")
            if track.get("jump_links"):
                reasons.append(
                    f"{track['jump_links']} links join places far apart along the"
                    " track (through a wall or gap)"
                )
            if track["env_over_track_median"] < 0.9:
                reasons.append(
                    f"geodesic {track['env_over_track_median']:.2f} x the distance"
                    f" along the {'bars' if app.kind == '3d' else 'track'}"
                )
    room = row.get("room")
    if room:
        facts.append("stated room")
        if room["bins_outside_room"] > 0.0:
            reasons.append(
                f"{room['bins_outside_room']:.1%} of bins outside the room bounds"
            )
    one_d = row.get("one_d")
    if one_d and app.track_length is not None:
        facts.append("stated length")
        span = one_d["range"][1] - one_d["range"][0]
        if abs(span / app.track_length - 1) > MAX_SIZE_RESIDUAL:
            reasons.append(f"track {span:.0f} long, stated {app.track_length:g}")
    lattice = row.get("lattice")
    if lattice and app.size is not None and "lattice_side" in lattice:
        facts.append("stated lattice")
        if abs(lattice["lattice_side"] / app.size - 1) > MAX_SIZE_RESIDUAL:
            reasons.append(
                f"lattice {lattice['lattice_side']:.0f} a side, stated {app.size:g}"
            )
    if one_d and "loop" in app.notes and "camera_distance_between_ends" in one_d:
        facts.append("the file's loop")
        camera = one_d["camera_distance_between_ends"]
        length = one_d["range"][1] - one_d["range"][0]
        if camera > LOOP_CLOSURE_SHARE * length:
            reasons.append(
                f"the loop's ends are {camera:.0f} apart in the camera: the"
                " coordinate does not close"
            )
        elif one_d["end_to_end_geodesic"] > camera + LOOP_CLOSURE_SHARE * length:
            reasons.append(
                f"loop ends {one_d['end_to_end_geodesic']:.0f} apart along the bins,"
                f" {camera:.0f} in the camera"
            )
    if reasons:
        return "mismatch", "; ".join(reasons)
    if facts:
        return "matches", ""
    return "not verified", "no stated fact independent of the samples checks it"


# --- Figures ---------------------------------------------------------------------


def figure_dir() -> Path:
    path = cache_root() / "figures" / "environment_fidelity"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _draw(
    ax: Any,
    env: Any,
    samples: NDArray[np.float64],
    measures: dict[str, Any],
    dims: tuple[int, int],
    ref: Reference | None,
    rng: np.random.Generator,
) -> None:
    from matplotlib.collections import LineCollection

    centers = np.asarray(env.bin_centers, dtype=float)
    pairs = np.asarray(env.edge_pairs)
    gaps = measures["_gaps"]
    bin_size = measures["bin_size"]
    if env.n_dims == 2 and env.layout_type in GRID_LAYOUTS:
        active = np.asarray(env.active_mask)
        padded = np.pad(~active, 1, constant_values=True)
        labels = ndimage.label(
            padded, structure=ndimage.generate_binary_structure(2, 1)
        )[0]
        enclosed = (labels[1:-1, 1:-1] > 0) & (labels[1:-1, 1:-1] != labels[0, 0])
        image = np.where(active, 1.0, np.where(enclosed, 2.0, np.nan))
        x_edges, y_edges = (np.asarray(e) for e in env.grid_edges)
        from matplotlib.colors import ListedColormap

        ax.pcolormesh(
            x_edges,
            y_edges,
            image.T,
            cmap=ListedColormap(["#c6dbef", "#fdd49e"]),
            vmin=1,
            vmax=2,
            shading="flat",
            rasterized=True,
        )
    shown = samples[
        rng.choice(
            len(samples), size=min(len(samples), MAX_DRAWN_SAMPLES), replace=False
        )
    ]
    ax.scatter(
        *shown[:, dims].T, s=0.3, c="0.35", alpha=0.5, linewidths=0, rasterized=True
    )
    if env.n_dims != 2 or env.layout_type not in GRID_LAYOUTS:
        ax.scatter(
            *centers[:, dims].T,
            s=2,
            c="tab:blue",
            alpha=0.4,
            linewidths=0,
            rasterized=True,
        )
    good = np.flatnonzero(gaps <= bin_size)
    if len(good) > MAX_DRAWN_EDGES:
        good = rng.choice(good, MAX_DRAWN_EDGES, replace=False)
    if env.n_dims != 2 or env.layout_type not in GRID_LAYOUTS:
        ax.add_collection(
            LineCollection(
                centers[pairs[good]][:, :, dims],
                colors="tab:blue",
                linewidths=0.2,
                alpha=0.3,
            )
        )
    far = np.flatnonzero(gaps > bin_size)
    ax.add_collection(
        LineCollection(centers[pairs[far]][:, :, dims], colors="red", linewidths=0.9)
    )
    if "_corners" in measures:
        corner = np.flatnonzero(measures["_corners"])
        ax.add_collection(
            LineCollection(
                centers[pairs[corner]][:, :, dims], colors="darkorange", linewidths=1.2
            )
        )
    if ref is not None:
        segments = np.stack(
            [ref.nodes[ref.edges[:, 0]], ref.nodes[ref.edges[:, 1]]], axis=1
        )
        ax.add_collection(
            LineCollection(segments[:, :, dims], colors="black", linewidths=0.7)
        )
    ax.set_aspect("equal")
    ax.autoscale_view()


def save_figure(
    rec: Recording,
    env: Any,
    samples: NDArray[np.float64],
    measures: dict[str, Any],
    ref: Reference | None,
    outline: Any | None,
    room: NDArray[np.float64] | None,
) -> str:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)
    n_dims = env.n_dims
    title = (
        f"{rec.dataset}: {measures['n_bins']} bins, {measures['components']} component(s)\n"
        "blue = bins (3-D: centres and links), beige = enclosed holes, grey = samples,"
        " red = links with no sample within one bin of their midpoint,\n"
        "orange = links through a corner only, black = reference track,"
        " green dashed = stated arena or room"
    )
    if n_dims == 1:
        fig, ax = plt.subplots(figsize=(12, 2.5))
        centers = np.asarray(env.bin_centers, dtype=float)[:, 0]
        ax.hist(samples[:, 0], bins=200, color="0.6")
        ax.plot(centers, np.zeros_like(centers), "|", color="tab:blue", markersize=12)
        ax.set_xlabel(f"linearized position ({rec.position_units})")
    elif n_dims == 2:
        fig, ax = plt.subplots(figsize=(9, 9))
        _draw(ax, env, samples, measures, (0, 1), ref, rng)
        if outline is not None:
            ax.plot(*np.asarray(outline.exterior.coords).T, "--", color="green", lw=1)
        ax.set_xlabel(f"x ({rec.position_units})")
        ax.set_ylabel(f"y ({rec.position_units})")
    else:
        slabs = 4 if ref is not None else 0
        fig, axes = plt.subplots(
            1 + (slabs > 0), 3, figsize=(18, 6 * (1 + (slabs > 0))), squeeze=False
        )
        for ax, dims in zip(axes[0], [(0, 1), (0, 2), (1, 2)], strict=True):
            _draw(ax, env, samples, measures, dims, ref, rng)
            names = "xyz"
            ax.set_xlabel(f"{names[dims[0]]} ({rec.position_units})")
            ax.set_ylabel(f"{names[dims[1]]} ({rec.position_units})")
            if room is not None:
                lo, hi = room[list(dims), 0], room[list(dims), 1]
                ax.plot(
                    [lo[0], hi[0], hi[0], lo[0], lo[0]],
                    [lo[1], lo[1], hi[1], hi[1], lo[1]],
                    "--",
                    color="green",
                )
        if slabs:
            centers = np.asarray(env.bin_centers, dtype=float)
            edges = np.quantile(centers[:, 2], np.linspace(0.2, 0.8, 3))
            width = measures["bin_size"]
            for ax, height in zip(axes[1], edges, strict=True):
                in_slab = np.abs(centers[:, 2] - height) <= width / 2
                near = np.abs(samples[:, 2] - height) <= width / 2
                ax.scatter(
                    *samples[near][:, :2].T,
                    s=0.3,
                    c="0.6",
                    linewidths=0,
                    rasterized=True,
                )
                ax.scatter(
                    *centers[in_slab][:, :2].T,
                    s=6,
                    c="tab:blue",
                    marker="s",
                    linewidths=0,
                )
                a, b = ref.nodes[ref.edges[:, 0]], ref.nodes[ref.edges[:, 1]]
                for p, q in zip(a, b, strict=True):
                    if (p[2] - height) * (q[2] - height) <= 0 and p[2] != q[2]:
                        t = (height - p[2]) / (q[2] - p[2])
                        cross = p + t * (q - p)
                        ax.plot(cross[0], cross[1], "k+", markersize=8)
                ax.set_title(
                    f"slice z = {height:.0f} +/- {width / 2:.0f} {rec.position_units}: voxels (blue), samples (grey), bars crossing it (+)",
                    fontsize=8,
                )
                ax.set_aspect("equal")
    fig.suptitle(title, fontsize=9)
    path = figure_dir() / f"{rec.dataset}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


# --- Driver ----------------------------------------------------------------------


def _public(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _public(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def declared_outline(env: Any) -> Any:
    """The polygon a declared 2-D environment was built from, or None."""
    domain = getattr(env.arena, "polygonal_domain", None)
    return domain() if env.n_dims == 2 and domain is not None else None


def analyse(rec: Recording, *, figures: bool = True) -> dict[str, Any]:
    """Every measure for one recording, and the figure.

    ``row["env"]`` is the environment built from the samples
    (``operations._samples_env``). Where the loader declares the apparatus
    (``geometry["environment"]``, the environment the driver analyses), it is
    measured and judged in ``row["declared"]`` (``measure_declared``),
    against the same samples, reference, outline and room. The outline is the
    one the loader placed when it declares one, else the stated square or
    circle placed here (``stated_shape``).
    """
    from neurospatial import Environment

    rng = np.random.default_rng(SEED)
    app = APPARATUS.get(rec.dataset)
    samples = tracked(rec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = _samples_env(rec, {})
        declared_env = (
            rec.geometry["environment"]() if env_kind(rec, {}) == "declared" else None
        )
    row: dict[str, Any] = {
        "dataset": rec.dataset,
        "units": rec.position_units,
        "kind": None if app is None else app.kind,
        "source": None if app is None else app.source,
    }
    measures = measure_env(env, samples, rng)
    row["env"] = measures
    ref = reference_for(rec, app) if env.n_dims > 1 else None
    if ref is not None:
        row["reference"] = against_reference(measures, env, samples, ref)
    outline = None if declared_env is None else declared_outline(declared_env)
    if (
        outline is None
        and app is not None
        and app.shape is not None
        and app.size is not None
        and env.n_dims == 2
    ):
        outline = stated_shape(samples, app)
    if outline is not None:
        row["outline"] = outline_measures(env, samples, app, outline)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filled = Environment.open_field(
                rec.positions, bin_size=rec.params["bin_size"], units=_units(rec)
            )
        row["open_field_preset"] = {
            "n_bins": int(filled.n_bins),
            "holes": len(hole_sizes(np.asarray(filled.active_mask))),
            "stated_area_covered": outline_measures(filled, samples, app, outline)[
                "stated_area_covered"
            ],
        }
    if "graph" in rec.geometry:
        row["graph_env"] = measure_graph_env(rec)
    if env.n_dims == 1:
        row["one_d"] = one_d_measures(rec, env)
    room = None
    if "room_bounds_m" in rec.geometry:
        room = np.asarray(rec.geometry["room_bounds_m"], dtype=float)
        row["room"] = bat_room(env, samples, room, rec)
    if "lattice_bars_cm" in rec.geometry and ref is not None:
        lattice = lattice_measures(env, samples, ref, rec)
        lattice["adjacent_bars"] = adjacent_bar_pairs(bin_graph(env), env, samples, ref)
        lattice["movement_along_bars"] = bar_alignment(rec, ref)
        row["lattice"] = lattice
    row["verdict"], row["verdict_reason"] = verdict(row, app)
    if declared_env is not None:
        row["declared"] = measure_declared(
            rec, app, samples, ref, env=declared_env, outline=outline, room=room
        )
    if figures:
        row["figure"] = save_figure(rec, env, samples, measures, ref, outline, room)
        if declared_env is not None:
            row["declared_figure"] = save_figure(
                replace(rec, dataset=f"{rec.dataset}_declared"),
                declared_env,
                samples,
                row["declared"]["env"],
                ref,
                outline,
                room,
            )
    return row


def measure_declared(
    rec: Recording,
    app: Apparatus | None,
    samples: NDArray[np.float64],
    ref: Reference | None,
    *,
    env: Any = None,
    outline: Any = None,
    room: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """The declared environment, measured as the samples-built one is, and judged.

    A "mismatch" is misfit evidence (``_geometry.judge_declared``, on the
    samples in ``params["analysis_epochs"]`` where the loader sets them: a
    stretch of track beside its samples, a segment with no samples near it,
    samples reaching past a track's free end or falling short of a wall, a
    stated length off by more than ``MAX_SIZE_RESIDUAL``) or a stated fact
    that fails (the parts, an independent reference). A "matches" needs a
    stated fact independent of the samples: a stated length, diameter or
    arm length compared with the samples' own, the authors' graph,
    centreline or linearization used as stated (``AUTHORS_GEOMETRY``), or a
    reference that is neither built from the samples
    (``SAMPLE_DERIVED_REFERENCES``) nor the geometry the track was declared
    from (``REFERENCE_SOURCE``). Otherwise the verdict is "not verified" and
    the measurements stand. The share of samples beyond the physical width
    is a tracking measure, reported with its split by speed, not judged: a
    head-mounted LED above an unwalled track swings past its edge.
    """
    if env is None:
        env = rec.geometry["environment"]()
    measures = measure_env(env, samples, np.random.default_rng(SEED))
    fit = dict(rec.geometry.get("environment_fit", {}))
    # Judged on the samples the analyses use (``analysis_epochs``, else all):
    # hc-18's sleep-box sessions are off the circuit by design.
    positions = np.asarray(rec.positions, dtype=float)
    epochs = rec.params.get("analysis_epochs")
    if epochs is not None:
        analysed = np.zeros(len(rec.times), dtype=bool)
        for start, stop in np.asarray(epochs, dtype=float).reshape(-1, 2):
            analysed |= (rec.times >= start) & (rec.times < stop)
        positions = np.where(analysed[:, None], positions, np.nan)
    judged = judge_declared(
        env,
        rec.times,
        positions,
        physical_width=rec.geometry.get("declared_physical_width"),
        speed_thresholds=speed_thresholds(rec),
        # A circle's one residual is its diameter's (``declared_circle``).
        stated_residuals={
            (
                "diameter" if k == "size_residual" else k.removeprefix("size_residual_")
            ): v
            for k, v in fit.items()
            if k.startswith("size_residual")
        },
        centre_offset=centre_offset_share(fit),
        bin_size=rec.params.get("graph_bin_size", rec.params["bin_size"]),
        lattice="lattice_bars_cm" in rec.geometry,
    )
    declared_from = rec.geometry.get("declared_from")
    independent = (
        ref is not None
        and ref.label not in SAMPLE_DERIVED_REFERENCES
        and declared_from not in REFERENCE_SOURCE.get(ref.label, ())
    )
    out: dict[str, Any] = {
        "layout": env.layout_type,
        "tracked_point": rec.geometry.get("tracked_point", "not stated"),
        "frac_samples_outside": float(np.mean(np.asarray(env.bin_at(samples)) < 0)),
        "track_width": rec.geometry.get("declared_track_width"),
        "track_width_kind": rec.geometry.get("declared_track_width_kind"),
        "physical_width": rec.geometry.get("declared_physical_width"),
        "physical_width_kind": rec.geometry.get("declared_physical_width_kind"),
        "outside_physical_width": judged["outside"],
        "outside_physical_width_by_speed": judged["outside_by_speed"],
        "declared_from": rec.geometry.get("declared_from"),
        "fit": judged["fit"],
        "misfit": judged["misfit"],
        "reference_independent": independent,
        "env": measures,
    }
    if ref is not None:
        # Against its own geometry this measures the binning, not the placement.
        out["reference"] = against_reference(measures, env, samples, ref)
    if outline is not None and env.layout_type in GRID_LAYOUTS:
        # Covered by construction when ``outline`` is the declared one.
        out["outline"] = outline_measures(env, samples, app, outline)
    if room is not None:
        out["room"] = bat_room(env, samples, room, rec)
    if "lattice_bars_cm" in rec.geometry and ref is not None:
        out["adjacent_bars"] = adjacent_bar_pairs(bin_graph(env), env, samples, ref)
    view: dict[str, Any] = {"env": measures}
    if independent:
        view["reference"] = out["reference"]
    status, reason = verdict(view, app)
    # Stated facts the declared track can be checked against, independent of
    # the samples it was placed on.
    # The samples' extent across the declared outline compares the samples
    # with the declaration itself, so it is misfit evidence but not a fact.
    facts = [
        f"{key.removeprefix('size_residual_')} {value:+.0%} of the stated"
        for key, value in judged["fit"].items()
        if key.startswith("size_residual") and key != "size_residual_extent"
    ]
    if declared_from in AUTHORS_GEOMETRY:
        facts.append(f"the {declared_from} as stated")
    out["facts"] = facts
    if status == "not verified" and (app is None or not app.verified):
        pass
    elif status == "mismatch" or judged["misfit"]:
        reasons = ([reason] if status == "mismatch" else []) + judged["misfit"]
        status, reason = "mismatch", "; ".join(reasons)
    elif facts or (status == "matches" and independent):
        # The declared environment's own loops and outline are what was
        # declared, so they confirm nothing; only the facts above and an
        # independent reference do.
        status = "matches"
        reason = "on " + "; ".join(facts + ([ref.label] if independent else []))
    else:
        status = "not verified"
        reason = "no stated fact independent of the samples checks the layout" + (
            f" ({ref.label} is built from the samples or is what it was declared from)"
            if ref
            else ""
        )
    out["verdict"], out["verdict_reason"] = status, reason
    return out


def speed_thresholds(rec: Recording) -> tuple[float, ...]:
    """Speed bands for the share of samples outside a declared environment.

    The loader's (``geometry["declared_speed_thresholds"]``), else its
    ``min_speed`` and, in cm, 30 cm/s, the bands the track loaders use.
    """
    if (stated := rec.geometry.get("declared_speed_thresholds")) is not None:
        return tuple(stated)
    bands = [] if rec.params.get("min_speed") is None else [rec.params["min_speed"]]
    if rec.position_units == "cm":
        bands.append(30.0)
    return tuple(sorted(set(bands)))


def centre_offset_share(fit: dict[str, Any]) -> float | None:
    """A stated centre's distance from the samples' centre over the stated radius."""
    if "centre_offset" not in fit:
        return None
    radius = fit["samples_diameter"] / (1 + fit["size_residual"]) / 2
    return float(fit["centre_offset"] / radius)


def _units(rec: Recording) -> str | None:
    return None if rec.position_units == "arbitrary" else rec.position_units


def _fmt(value: Any, spec: str) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    return format(value, spec)


def compare_table(rows: list[dict[str, Any]]) -> list[str]:
    """Declared environments beside the samples-built ones (declared / samples)."""
    declared = [row for row in rows if "declared" in row]
    if not declared:
        return []
    lines = [
        "",
        "## Declared against samples-built environments",
        "",
        "Each cell: declared / samples-built, measured against the same samples,"
        " reference and outline. Outside: share of tracked samples the declared"
        " environment cannot place (`env.bin_at` -1), a tracking measure, not"
        " judged. Covered and wall gap: against the outline the loader placed; a"
        " declared environment covers its own outline by construction, so only"
        " the samples-built value is evidence.",
        "",
        "| Dataset | Bins | Components (1 %) | Holes (bins) | Sparse bins | Far links |"
        " Geodesic / straight | Env / track | Covered | Wall gap | Outside |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def pair(a: Any, b: Any, spec: str) -> str:
        return f"{_fmt(a, spec)} / {_fmt(b, spec)}"

    for row in declared:
        dec = row["declared"]
        env, base = dec["env"], row["env"]
        holes = (
            f"{env.get('holes', '–')} ({env.get('hole_bins', '–')}) /"
            f" {base.get('holes', '–')} ({base.get('hole_bins', '–')})"
        )
        ref, base_ref = dec.get("reference", {}), row.get("reference", {})
        outline, base_outline = dec.get("outline", {}), row.get("outline", {})
        lines.append(
            f"| {row['dataset']} | {env['n_bins']} / {base['n_bins']} |"
            f" {env['components']} ({env['major_components']}) /"
            f" {base['components']} ({base['major_components']}) | {holes} |"
            f" {pair(env['sparse_bins'], base['sparse_bins'], '.0%')} |"
            f" {pair(env['far_edges'], base['far_edges'], '.2%')} |"
            f" {pair(env['geodesic_ratio_median'], base['geodesic_ratio_median'], '.2f')} |"
            f" {pair(ref.get('env_over_track_median'), base_ref.get('env_over_track_median'), '.2f')} |"
            f" {pair(outline.get('stated_area_covered'), base_outline.get('stated_area_covered'), '.0%')} |"
            f" {pair(outline.get('wall_gap_median'), base_outline.get('wall_gap_median'), '.1f')} |"
            f" {dec['frac_samples_outside']:.2%} |"
        )
    return lines


def render(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Environment fidelity",
        "",
        "Generated by `datasets/studies/environment_fidelity.py`; see its docstring"
        " for each measure. Environments are the driver's (`from_samples` on every"
        " tracked sample at the loader's `bin_size`). Unsupported: share of"
        " neighbour links with no sample within half a bin of the link's midpoint"
        " (far: within one bin). Env / track: median geodesic over the distance"
        " along the reference for sample pairs.",
        "",
        "| Dataset | Kind | Bins | Dims | Components (1 % of bins or more) | Holes (bins) | Sparse / thin bins | Unsupported / far links | Corner links | Geodesic / straight (median, p95) | On-track samples / bins | Env / track (median, p5) | Outline: covered / centres outside / wall gap / hull over rectangle, circle | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        env, ref, outline = row["env"], row.get("reference", {}), row.get("outline")
        holes = f"{env['holes']} ({env['hole_bins']})" if "holes" in env else "–"
        on_track = (
            f"{ref['samples_on_track']:.0%} / {ref['bins_on_track']:.0%}"
            + ("" if ref.get("track_width_stated") else " (assumed width)")
            if "samples_on_track" in ref
            else "–"
        )
        track = (
            f"{_fmt(ref.get('env_over_track_median'), '.2f')}, {_fmt(ref.get('env_over_track_p05'), '.2f')}"
            if ref
            else "–"
        )
        outline_cell = (
            f"{outline['stated_area_covered']:.0%} / {outline['bin_centres_outside_stated']:.0%} /"
            f" {outline['wall_gap_median']:.1f} / {outline['hull_over_min_rectangle']:.2f},"
            f" {outline['hull_over_min_circle']:.2f}"
            if outline
            else "–"
        )
        verdict_cell = row["verdict"] + (
            f": {row['verdict_reason']}" if row["verdict_reason"] else ""
        )
        lines.append(
            f"| {row['dataset']} | {row['kind']} | {env['n_bins']} | {env['n_dims']} |"
            f" {env['components']} ({env['major_components']}) | {holes} | {env['sparse_bins']:.0%} / {env['thin_bins']:.0%} |"
            f" {env['unsupported_edges']:.1%} / {env['far_edges']:.2%} |"
            f" {env.get('corner_links', '–')} |"
            f" {_fmt(env['geodesic_ratio_median'], '.2f')}, {_fmt(env['geodesic_ratio_p95'], '.2f')} |"
            f" {on_track} | {track} | {outline_cell} | {verdict_cell} |"
        )
    declared = [row for row in rows if "declared" in row]
    if declared:
        lines += [
            "",
            "## Declared environments",
            "",
            "The environment the driver analyses where the loader declares the"
            " apparatus (`geometry['environment']`), measured as above against the"
            " same samples. Verdict: misfit evidence (`_geometry.judge_declared`: a"
            " stretch of track whose samples lie to one side by more than a bin, a"
            " track segment with no sample near it, a stated length off by more than"
            " 15 %; on an outline or the lattice the share with no sample near it is"
            " reported, not judged: the tracked head keeps off walls and bars),"
            " the stated parts and an independent reference. Beyond physical: the"
            " share of tracked samples farther than half the physical width from the"
            " track, a tracking measure (the tracked head swings past an unwalled"
            " track), not judged; in brackets the same share by speed, slowest band"
            " first (below the loader's running speed, up to 30 cm/s, above; bands"
            " in the details). Tolerance: the"
            " placement tolerance the track is built with (`track_width`), measured"
            " from the samples so about 1 % are dropped. Lateral: the median"
            " |lateral offset| and the worst stretch's median signed offset. Env /"
            " track: `own` when the reference is the geometry the track was declared"
            " from, which then measures only its binning.",
            "",
            "| Dataset | Tracked point | Declared from | Bins | Components (1 % of bins or more) | Physical width | Beyond physical | Tolerance (outside) | Lateral: median, worst stretch | Far links | Env / track (median, p5) | Verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in declared:
            dec = row["declared"]
            env, ref, fit = dec["env"], dec.get("reference", {}), dec.get("fit", {})
            physical = (
                "not stated"
                if dec.get("physical_width") is None
                else f"{dec['physical_width']:.3g} {dec['physical_width_kind']}"
            )
            by_speed = {
                k: v
                for k, v in (dec.get("outside_physical_width_by_speed") or {}).items()
                if k != "all"
            }
            beyond = (
                "–"
                if dec.get("outside_physical_width") is None
                else f"{dec['outside_physical_width']:.1%}"
                + (
                    " (" + " / ".join(f"{v:.0%}" for v in by_speed.values()) + ")"
                    if by_speed
                    else ""
                )
            )
            tolerance = (
                f"{dec['track_width']:.3g} ({dec['frac_samples_outside']:.1%})"
                if dec.get("track_width") is not None
                else "–"
            )
            lateral = (
                f"{fit['median_abs_lateral']:.2f}, {fit['max_edge_bias']:.2f}"
                if "max_edge_bias" in fit
                else "–"
            )
            track = (
                f"{_fmt(ref.get('env_over_track_median'), '.2f')}, {_fmt(ref.get('env_over_track_p05'), '.2f')}"
                + ("" if dec.get("reference_independent") else " (own)")
                if ref
                else "–"
            )
            verdict_cell = dec["verdict"] + (
                f": {dec['verdict_reason']}" if dec["verdict_reason"] else ""
            )
            lines.append(
                f"| {row['dataset']} | {dec.get('tracked_point', '–')} |"
                f" {dec.get('declared_from') or '–'} | {env['n_bins']} |"
                f" {env['components']} ({env['major_components']}) | {physical} |"
                f" {beyond} | {tolerance} | {lateral} | {env['far_edges']:.2%} |"
                f" {track} | {verdict_cell} |"
            )
    lines += compare_table(rows)
    lines += ["", "## Sources and details", ""]
    for row in rows:
        lines.append(f"### {row['dataset']}")
        lines.append("")
        lines.append(f"- Source: {row['source']}")
        for key in (
            "reference",
            "outline",
            "open_field_preset",
            "graph_env",
            "one_d",
            "room",
            "lattice",
            "declared",
        ):
            if key in row:
                lines.append(f"- {key}: `{json.dumps(_public(row[key]), default=str)}`")
        lines.append("")
    return "\n".join(lines)


def expand(rec: Recording) -> list[Recording]:
    """The recording, plus derived views that answer a question about its apparatus.

    - a recording with a 2-D track and the file's linearized position
      (``params["linearized"]``): the 1-D environment the replay operation
      builds from it without a declared track, ``Environment.from_samples``
      on that coordinate (for the figure-8, its linearized loop).
    - neurolab_001754: the driver builds one environment from all five
      sessions (three apparatus in one image); the Escher staircase session
      alone is also built.
    """
    views = [rec]
    if "linearized" in rec.params and np.asarray(rec.positions).shape[1] == 2:
        views.append(
            replace(
                rec,
                dataset=f"{rec.dataset}_linearized",
                positions=np.asarray(rec.params["linearized"], dtype=float)[:, None],
                geometry={"positions_2d": rec.positions},
                params={k: v for k, v in rec.params.items() if k != "linearized"},
            )
        )
    if rec.dataset == "neurolab_001754":
        session = np.zeros(len(rec.times), dtype=bool)
        for start, stop in np.asarray(rec.params["analysis_epochs"]):
            session |= (rec.times >= start) & (rec.times < stop)
        views.append(
            replace(
                rec,
                dataset="neurolab_001754_escher",
                times=rec.times[session],
                positions=np.asarray(rec.positions)[session],
            )
        )
    return views


def run(keys: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for key in keys:
        rec = importlib.import_module(LOADERS[key]).load()
        if rec.positions is None:
            continue
        for view in expand(rec):
            rows.append(analyse(view))
            declared = rows[-1].get("declared")
            print(
                view.dataset,
                rows[-1]["verdict"],
                rows[-1]["verdict_reason"],
                *(
                    ("| declared:", declared["verdict"], declared["verdict_reason"])
                    if declared
                    else ()
                ),
                flush=True,
            )
    return rows


def merged(
    previous: list[dict[str, Any]], new: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    """``previous`` rows with those of ``keys`` (and their views) replaced by ``new``.

    A row belongs to the longest key that its dataset equals or starts with
    followed by an underscore (a derived view). Rows keep their earlier order;
    rows of datasets not seen before are appended.
    """

    def owner(row: dict[str, Any]) -> str | None:
        name = row["dataset"]
        owners = [k for k in keys if name == k or name.startswith(f"{k}_")]
        return max(owners, key=len) if owners else None

    fresh: dict[str, list[dict[str, Any]]] = {}
    for row in new:
        fresh.setdefault(owner(row) or row["dataset"], []).append(row)
    out: list[dict[str, Any]] = []
    for row in previous:
        key = owner(row)
        if key is None:
            out.append(row)
        elif key in fresh:
            out += fresh.pop(key)
    for rows in fresh.values():
        out += rows
    return out


if __name__ == "__main__":
    keys = tuple(sys.argv[1:]) or tuple(LOADERS)
    rows = run(keys)
    RESULTS.mkdir(parents=True, exist_ok=True)
    public = [_public(row) for row in rows]
    previous_path = RESULTS / "environment_fidelity.json"
    if sys.argv[1:] and previous_path.exists():
        # Rerun on some datasets: keep the other rows of the last full run.
        public = merged(json.loads(previous_path.read_text()), public, keys)
    (RESULTS / "environment_fidelity.json").write_text(
        json.dumps(public, indent=2, default=str) + "\n"
    )
    (RESULTS / "environment_fidelity.md").write_text(render(public))
