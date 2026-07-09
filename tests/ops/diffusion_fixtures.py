"""Shared deterministic fixtures for the diffusion apply-path equivalence tests.

Provides:

- Deterministic environment builders for every supported geometry (1D grid,
  linear track, 2D open field, 2D masked wall/split, hexagonal, egocentric
  polar, triangular mesh).
- A canonical field set per environment (unit point source, seeded non-negative,
  seeded signed, and a partially-masked field) used for both the dense-operator
  baseline capture and the ``env.diffuse`` equivalence checks.
- The (geometry, sigma, mode) case registry the baseline is captured over.

The baseline (the *shipped dense operator's* outputs) is captured by
``capture_baseline`` in ``generate_diffusion_baseline.py`` and pickled to
``data/diffusion_perf_baseline.pkl``. The equivalence tests compare the new
matrix-free ``env.diffuse`` apply-path against that frozen dense oracle.

Every builder is explicit (no random sampling of positions), so the geometry --
and therefore the dense operator -- is reproducible run to run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from neurospatial import Environment
from neurospatial.environment.polar import EgocentricPolarEnvironment
from neurospatial.layout.engines.graph import GraphLayout

BASELINE_PATH = Path(__file__).parent / "data" / "diffusion_perf_baseline.pkl"

# Sigma labels: "small" is a fraction of the bin size, so the heat kernel is
# highly localized and needs (near-)all modes -> the adaptive resolver falls
# back to the full/dense basis, exercising the FULL-RANK equivalence. "large"
# is several bins, so the resolved rank is well below n and the truncated apply
# is exercised.
MODES: tuple[str, ...] = ("transition", "density", "average")


# ---------------------------------------------------------------------------
# Deterministic environment builders. Each returns (env, source_coordinate).
# ---------------------------------------------------------------------------
def build_grid_1d(h: float = 2.0, domain=(0.0, 60.0)) -> tuple[Environment, list[float]]:
    """Full 1D regular grid over ``domain`` at spacing ``h``."""
    n = round((domain[1] - domain[0]) / h)
    edges = np.linspace(domain[0], domain[1], n + 1)
    env = Environment.from_grid_mask(active_mask=np.ones(n, dtype=bool), grid_edges=(edges,))
    return env, [30.0]


def build_track_1d(bin_size: float = 2.0) -> tuple[Environment, None]:
    """Straight linear track (GraphLayout, single segment)."""
    tg = nx.Graph()
    tg.add_node(0, pos=(0.0, 0.0))
    tg.add_node(1, pos=(80.0, 0.0))
    tg.add_edge(0, 1, distance=80.0)
    params = {
        "graph_definition": tg,
        "edge_order": [(0, 1)],
        "edge_spacing": 0.0,
        "bin_size": bin_size,
    }
    layout = GraphLayout()
    layout.build(**params)
    env = Environment(
        name="track",
        layout=layout,
        layout_type_used="Graph",
        layout_params_used=params,
    )
    return env, None  # source chosen by index in the field builder


def build_grid_2d(h: float = 2.0, domain=(0.0, 40.0)) -> tuple[Environment, list[float]]:
    """Full 2D open-field regular grid (large enough that sigma_large truncates)."""
    n = round((domain[1] - domain[0]) / h)
    edges = np.linspace(domain[0], domain[1], n + 1)
    env = Environment.from_grid_mask(
        active_mask=np.ones((n, n), dtype=bool),
        grid_edges=(edges, edges),
        connect_diagonal_neighbors=True,
    )
    return env, [20.0, 20.0]


def build_grid_2d_masked(h: float = 2.0, domain=(0.0, 40.0)) -> tuple[Environment, list[float]]:
    """2D grid with a corner hole (still one connected component)."""
    n = round((domain[1] - domain[0]) / h)
    edges = np.linspace(domain[0], domain[1], n + 1)
    mask = np.ones((n, n), dtype=bool)
    hole = max(1, n // 6)
    mask[:hole, :hole] = False
    env = Environment.from_grid_mask(
        active_mask=mask, grid_edges=(edges, edges), connect_diagonal_neighbors=True
    )
    return env, [32.0, 32.0]


def build_grid_2d_split(h: float = 2.0, domain=(0.0, 40.0)) -> tuple[Environment, list[float]]:
    """2D grid with a full-height wall (inactive column) splitting it into TWO
    disconnected components (left / right). Exercises per-component null modes
    and the no-leakage-across-a-wall property under truncation."""
    n = round((domain[1] - domain[0]) / h)
    edges = np.linspace(domain[0], domain[1], n + 1)
    mask = np.ones((n, n), dtype=bool)
    mask[:, n // 2] = False  # vertical wall -> two disconnected halves
    env = Environment.from_grid_mask(
        active_mask=mask, grid_edges=(edges, edges), connect_diagonal_neighbors=False
    )
    # Source in the LEFT half, near the wall (to probe leakage across it).
    return env, [float(domain[1] * (n // 2 - 1.5) / n), 18.0]


def build_hex(bin_size: float = 3.0) -> tuple[Environment, list[float]]:
    """Hexagonal lattice over a square region."""
    positions = np.array(
        [[x, y] for x in np.arange(0, 48, 1.0) for y in np.arange(0, 48, 1.0)],
        dtype=float,
    )
    env = Environment.from_samples(
        positions, layout="Hexagonal", bin_size=bin_size, bin_count_threshold=0
    )
    return env, [24.0, 24.0]


def build_polar() -> tuple[EgocentricPolarEnvironment, list[float]]:
    """Full-circle egocentric polar env (non-uniform mass M = annular-sector area)."""
    env = EgocentricPolarEnvironment.create(
        distance_range=(0.0, 40.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=2.5,
        angle_bin_size=np.pi / 10,
    )
    # Source in the interior (r, theta) -- away from r=0 and the +-pi seam.
    return env, [20.0, 0.0]


def build_mesh(point_spacing: float = 5.0) -> tuple[Environment, list[float]]:
    """Well-shaped flat triangular mesh over a square."""
    boundary = Polygon([(-30, -30), (30, -30), (30, 30), (-30, 30)])
    env = Environment.from_layout(
        kind="TriangularMesh",
        layout_params={"boundary_polygon": boundary, "point_spacing": point_spacing},
    )
    return env, [0.0, 0.0]


# ---------------------------------------------------------------------------
# Geometry registry: name -> (builder, sigma_small, sigma_large).
# sigma_small forces (near-)full rank; sigma_large forces truncation.
# ---------------------------------------------------------------------------
GEOMETRIES: dict[str, tuple[Callable[[], tuple], float, float]] = {
    "grid_1d": (build_grid_1d, 1.0, 8.0),
    "track_1d": (build_track_1d, 1.0, 8.0),
    "grid_2d": (build_grid_2d, 1.5, 8.0),
    "grid_2d_masked": (build_grid_2d_masked, 1.5, 8.0),
    "grid_2d_split": (build_grid_2d_split, 1.5, 8.0),
    "hex": (build_hex, 1.5, 8.0),
    "polar": (build_polar, 2.0, 15.0),
    "mesh": (build_mesh, 2.5, 10.0),
}


def build_perf_grid(n: int = 60, domain: float = 120.0) -> Environment:
    """Large 2D grid for the perf baseline (dense-``expm`` reference).

    Default 60x60 = 3600 bins -- above the 3000-bin dense-kernel warn threshold,
    yet the dense ``expm`` still completes in ~1 minute so a pre-refactor
    baseline can be captured. (A true 10k-bin dense ``expm`` is ~O(n_bins^3):
    extrapolating the measured 2500-bin cost puts it near ~25 min and several GB
    -- impractical to capture or gate on. ``build_scaling_grid`` exercises the
    apply-path at that 10k scale, where the dense path is infeasible.)"""
    edges = np.linspace(0.0, domain, n + 1)
    return Environment.from_grid_mask(
        active_mask=np.ones((n, n), dtype=bool),
        grid_edges=(edges, edges),
        connect_diagonal_neighbors=True,
    )


def build_scaling_grid(n: int = 100, domain: float = 200.0) -> Environment:
    """~10k-bin grid to prove the apply-path SCALES where dense ``expm`` cannot.

    Default 100x100 = 10000 bins. The dense ``expm`` here is impractical
    (~25 min / multi-GB extrapolated); ``env.diffuse``'s truncated apply builds
    only a bandwidth-sized eigenbasis, so it completes in seconds."""
    edges = np.linspace(0.0, domain, n + 1)
    return Environment.from_grid_mask(
        active_mask=np.ones((n, n), dtype=bool),
        grid_edges=(edges, edges),
        connect_diagonal_neighbors=True,
    )


PERF_SIGMA = 10.0  # bandwidth for the perf grids (well-truncated: rank << n_bins)


def source_bin(env: Environment, source_coord: list[float] | None) -> int:
    """Nearest bin index to ``source_coord`` (in the env's own bin-center space).

    Polar bin centers are ``(r, theta)`` and the source coordinate is given in
    that same space; mesh/grid use Cartesian centers. ``None`` picks the middle
    bin by index (used for the linear track, whose centers are along-track).
    """
    if source_coord is None:
        return env.n_bins // 2
    centers = env.bin_centers
    return int(np.argmin(np.linalg.norm(centers - np.asarray(source_coord), axis=1)))


def make_fields(env: Environment, src: int) -> dict[str, NDArray[np.float64]]:
    """Canonical deterministic field set for equivalence + gate tests.

    - ``point``  : unit point source at ``src`` (leakage / localization).
    - ``nonneg`` : seeded non-negative field (occupancy / counts / KDE).
    - ``signed`` : seeded signed field (linearity + signed-smooth contract).
    - ``masked`` : the ``nonneg`` field with a contiguous block set to ``NaN``
      (support-gate tests); its finite mask rides alongside as ``masked_valid``.
    """
    n = env.n_bins
    rng = np.random.default_rng(1234)
    point = np.zeros(n, dtype=np.float64)
    point[src] = 1.0
    nonneg = rng.random(n).astype(np.float64)
    signed = rng.standard_normal(n).astype(np.float64)
    masked = nonneg.copy()
    # Mask out a chunk of bins (~20%) to create an interior invalid region.
    n_mask = max(1, n // 5)
    masked[rng.choice(n, size=n_mask, replace=False)] = np.nan
    return {
        "point": point,
        "nonneg": nonneg,
        "signed": signed,
        "masked": masked,
        "masked_valid": np.isfinite(masked).astype(np.float64),
    }
