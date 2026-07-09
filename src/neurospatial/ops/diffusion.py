"""Finite-volume diffusion operator for boundary-aware smoothing.

This module builds a single finite-volume (two-point flux) heat operator that
makes the smoothing ``bandwidth`` the true physical standard deviation (sigma)
on every environment layout, independent of bin size or resolution.

The operator is

.. math::

    H(\\sigma) = \\exp(-t\\,L), \\quad t = \\sigma^2 / 2, \\quad
    L = M^{-1}(D - W)

with ``W[i, j] = A[i, j] / d[i, j]`` (``A`` the measure of the face shared by
bins ``i`` and ``j``; ``d`` the center-to-center distance), ``D = diag(W @ 1)``
the degree matrix, and ``M = diag(volumes)`` the per-bin cell volumes. On any
K-orthogonal discretization (every regular lattice used here) ``L`` has the
continuum limit :math:`-\\nabla^2`, so ``H`` diffuses by exactly ``sigma``.

Three views of the same operator serve the three consumers (see
:func:`heat_kernel_from_W`):

- ``mode="transition"`` returns ``Hᵀ`` (column-stochastic) for mass-conserving
  smoothing of **extensive** quantities (occupancy, counts).
- ``mode="density"`` returns ``H·M⁻¹`` (M-weighted columns integrate to 1) for
  count→density KDE.
- ``mode="average"`` returns ``H`` (row-stochastic) for averaging an
  **intensive** field (rate maps). This low-level view exists here but is not
  yet exposed publicly on ``Environment.smooth`` / ``Environment.compute_kernel``.

Public entry points are ``Environment.compute_kernel`` / ``Environment.smooth``;
:func:`diffusion_kernel` is the internal dispatcher they route through.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment._protocols import EnvironmentProtocol

__all__ = [
    "diffusion_kernel",
    "heat_kernel_from_W",
]

# Non-orthogonality skew guard for triangle-centroid meshes: two-point flux is
# exact only as the dual approaches K-orthogonality. If more than this fraction
# of interior edges exceed the angle threshold, sigma is only approximate and
# the builder warns.
_MESH_SKEW_ANGLE_DEG = 30.0
_MESH_SKEW_FRACTION = 0.05


# ---------------------------------------------------------------------------
# Core operator (W -> H) -- the seam a future spectral engine will replace.
# ---------------------------------------------------------------------------
def _raw_heat_operator(
    W: scipy.sparse.spmatrix,
    volumes: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]:
    """Dense heat operator ``exp(-t L)`` with ``L = M^-1 (D - W)``, PRE clip/normalize.

    This is the M-self-adjoint operator whose raw invariants (``H @ 1 == 1`` and
    ``M_i H_ij == M_j H_ji``) hold to numerical tolerance at full rank. It is
    exposed as a seam so those invariants stay directly testable (the mode
    outputs of :func:`heat_kernel_from_W` are normalized and no longer expose
    them), and so a future spectral engine can replace only this function.

    Parameters
    ----------
    W : scipy.sparse matrix, shape (n_bins, n_bins)
        Symmetric finite-volume weight matrix ``W[i, j] = A[i, j] / d[i, j]``.
    volumes : NDArray[np.float64], shape (n_bins,)
        Per-bin cell volumes ``M`` (strictly positive).
    sigma : float
        Physical smoothing standard deviation (finite, > 0). ``t = sigma**2 / 2``.

    Returns
    -------
    NDArray[np.float64], shape (n_bins, n_bins)
        Dense raw heat operator ``H``.
    """
    degree = np.asarray(W.sum(axis=1)).ravel()
    inv_mass = scipy.sparse.diags(1.0 / volumes)
    # CSC so scipy.sparse.linalg.expm's internal solves stay in their preferred
    # format (avoids a SparseEfficiencyWarning).
    laplacian = (inv_mass @ (scipy.sparse.diags(degree) - W)).tocsc()  # M^-1 (D - W)
    t = sigma**2 / 2.0
    kernel = scipy.sparse.linalg.expm(-t * laplacian)
    if hasattr(kernel, "toarray"):
        kernel = kernel.toarray()
    return np.asarray(kernel, dtype=np.float64)


def heat_kernel_from_W(
    W: scipy.sparse.spmatrix,
    volumes: NDArray[np.float64],
    sigma: float,
    *,
    mode: Literal["transition", "density", "average"],
) -> NDArray[np.float64]:
    """Normalize the raw heat operator into one of the three mode kernels.

    Each mode is normalized to its OWN contract; the kernels are NOT rescalings
    of one shared normalization (row-normalizing once and reusing would break
    ``density``'s integrate-to-1 after clipping):

    - ``"average"`` -> ``H`` row-stochastic (``sum_j K[i, j] = 1``); averages an
      intensive field (``K @ rate``).
    - ``"transition"`` -> ``Hᵀ`` column-stochastic (``sum_i K[i, j] = 1``);
      mass-conserving smoothing of an extensive field (``K @ counts``).
    - ``"density"`` -> ``H·M⁻¹`` (``sum_i M_i K[i, j] = 1``); count -> density.

    Parameters
    ----------
    W : scipy.sparse matrix, shape (n_bins, n_bins)
        Finite-volume weight matrix (see :func:`_raw_heat_operator`).
    volumes : NDArray[np.float64], shape (n_bins,)
        Per-bin cell volumes ``M``.
    sigma : float
        Physical smoothing standard deviation.
    mode : {"transition", "density", "average"}
        Which normalized view to return.

    Returns
    -------
    NDArray[np.float64], shape (n_bins, n_bins)
        The requested mode kernel.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the three supported values.

    Notes
    -----
    At full rank the raw operator is block-diagonal across the connected
    components of ``W`` (a masked wall or corner-only ``A = 0`` splits it), and
    clipping round-off adds no cross-block entries, so the per-mode
    normalization is inherently within-component; no explicit component loop is
    needed here.
    """
    # Validate mode before the dense matrix exponential so a direct caller with
    # a bad mode fails fast rather than after paying the O(n^3) expm cost.
    if mode not in ("transition", "density", "average"):
        raise ValueError(
            f"Invalid mode {mode!r}. Choose 'transition', 'density', or 'average'."
        )

    # Clip round-off negatives (real cross-block lobes only appear under the
    # future truncated engine, not at full rank).
    H = np.clip(_raw_heat_operator(W, volumes, sigma), 0.0, None)

    kernel: NDArray[np.float64]
    if mode == "average":
        row_sums = H.sum(axis=1, keepdims=True)
        kernel = H / np.where(row_sums > 0.0, row_sums, 1.0)
    elif mode == "transition":
        row_sums = H.sum(axis=1, keepdims=True)
        kernel = (H / np.where(row_sums > 0.0, row_sums, 1.0)).T
    else:  # mode == "density"
        col_mass = volumes @ H  # col_mass[j] = sum_i M_i H[i, j]
        kernel = H / np.where(col_mass > 0.0, col_mass, 1.0)[np.newaxis, :]
    return kernel


def _components_from_W(W: scipy.sparse.spmatrix) -> tuple[int, NDArray[np.int_]]:
    """Connected components of the diffusion-weight matrix ``W`` (nonzero ``A``).

    Component structure for any component-aware step is derived from ``W``, NOT
    ``env.connectivity``: corner-touching 8-connected bins have ``A = 0`` and so
    are separate diffusion components. Used by tests and load-bearing under the
    future truncated engine.

    Parameters
    ----------
    W : scipy.sparse matrix, shape (n_bins, n_bins)
        Finite-volume weight matrix.

    Returns
    -------
    n_components : int
        Number of connected components.
    labels : NDArray[np.int_], shape (n_bins,)
        Component label per node.
    """
    n_components, labels = scipy.sparse.csgraph.connected_components(W, directed=False)
    return int(n_components), labels


# ---------------------------------------------------------------------------
# Environment entry point + geometry dispatch.
# ---------------------------------------------------------------------------
def diffusion_kernel(
    env: EnvironmentProtocol,
    sigma: float,
    *,
    mode: Literal["transition", "density", "average"] = "density",
) -> NDArray[np.float64]:
    """Finite-volume diffusion kernel for an environment.

    Resolves the layout geometry, builds a working graph carrying the
    finite-volume face measure ``"A"`` on every edge plus a node-ordered
    ``volumes`` array, then assembles ``H = exp(-t L)`` and returns the
    requested mode view.

    Parameters
    ----------
    env : Environment
        Fitted environment (any supported layout).
    sigma : float
        Physical smoothing standard deviation (finite, > 0).
    mode : {"transition", "density", "average"}, default="density"
        Kernel orientation (see :func:`heat_kernel_from_W`). ``"average"`` is a
        valid low-level request but is not yet exposed publicly on
        ``Environment.compute_kernel`` / ``Environment.smooth``.

    Returns
    -------
    NDArray[np.float64], shape (n_bins, n_bins)
        The diffusion kernel.

    Raises
    ------
    NotImplementedError
        If the environment's layout has no finite-volume builder.
    ValueError
        On invalid ``sigma``, ``volumes``, node labels, or edge attributes
        (see :func:`~neurospatial.ops.smoothing.compute_diffusion_kernels`).
    """
    from neurospatial.ops.smoothing import compute_diffusion_kernels

    graph_with_A, volumes = _finite_volume_geometry(env)
    return compute_diffusion_kernels(
        graph_with_A, volumes=volumes, sigma=sigma, mode=mode
    )


def _finite_volume_geometry(
    env: EnvironmentProtocol,
) -> tuple[nx.Graph, NDArray[np.float64]]:
    """Dispatch to the per-geometry finite-volume builder.

    Returns a COPY of ``env.connectivity`` with an ``"A"`` face-measure edge
    attribute stamped on every edge, plus the node-ordered ``volumes`` array.
    The env's own graph is never mutated.

    Polar dispatches on the environment type (``_POLAR``), NOT the layout
    engine, because ``EgocentricPolarEnvironment`` is built on a masked-grid
    layout and would otherwise be misclassified Cartesian.
    """
    if getattr(env, "_POLAR", False):
        return _polar_fv(env)

    engine = type(env.layout).__name__
    builders = {
        "RegularGridLayout": _cartesian_fv,
        "MaskedGridLayout": _cartesian_fv,
        "ImageMaskLayout": _cartesian_fv,
        "ShapelyPolygonLayout": _cartesian_fv,
        "HexagonalLayout": _hex_fv,
        "GraphLayout": _graph_fv,
        "TriangularMeshLayout": _mesh_fv,
    }
    try:
        builder = builders[engine]
    except KeyError:
        raise NotImplementedError(
            f"diffusion kernel unsupported for layout {engine!r}. Supported "
            f"layouts: {sorted(builders)} (and egocentric polar)."
        ) from None
    return builder(env)


# ---------------------------------------------------------------------------
# Per-geometry finite-volume builders. Each returns (graph_copy, volumes).
# ---------------------------------------------------------------------------
def _per_axis_bin_widths(env: EnvironmentProtocol) -> NDArray[np.float64]:
    """Per-axis Cartesian bin width from the layout's ``grid_edges``.

    Uses the first spacing per axis, matching ``_GridMixin.bin_sizes``' uniform
    cell assumption, so the face measure ``A`` and the mass ``M`` stay mutually
    consistent. Custom nonuniform ``grid_edges`` therefore inherit this uniform
    approximation and are outside the physical-sigma guarantee (a tracked
    follow-up), rather than silently mixing a nonuniform ``M`` with a uniform ``A``.
    """
    grid_edges = cast("Any", env.layout).grid_edges
    return np.array([float(np.diff(edges)[0]) for edges in grid_edges])


def _cartesian_fv(env: EnvironmentProtocol) -> tuple[nx.Graph, NDArray[np.float64]]:
    """Cartesian face measure: product of the OTHER axes' widths for an
    axis-aligned neighbor; corner-only (diagonal) neighbors get ``A = 0``.

    On a uniform grid this is ``A = h**(n_dims - 1)`` for face-adjacent edges
    (and ``A = 1`` in 1D). Masks/holes only drop nodes; the geometry of the
    remaining edges is unchanged.
    """
    g = env.connectivity.copy()
    centers = env.bin_centers
    n_dims = centers.shape[1]
    widths = _per_axis_bin_widths(env)
    for u, v, data in g.edges(data=True):
        offset = centers[u] - centers[v]
        moved = np.abs(offset) > 1e-9
        if moved.sum() != 1:  # diagonal / Moore edge -> shares only a corner
            data["A"] = 0.0
            continue
        axis = int(np.flatnonzero(moved)[0])
        data["A"] = float(np.prod([widths[d] for d in range(n_dims) if d != axis]))
    return g, env.bin_sizes


def _hex_fv(env: EnvironmentProtocol) -> tuple[nx.Graph, NDArray[np.float64]]:
    """Hex face measure: the shared hexagon side length ``s`` on every edge.

    For a pointy-top regular hexagonal lattice adjacent centers are ``sqrt(3)*s``
    apart, so ``s`` is derived per edge as ``distance / sqrt(3)`` from the
    edge's center-to-center ``"distance"`` (equivalently the layout's stored
    side length). Hex lattices are K-orthogonal, so no hand-tuned constant is
    needed.
    """
    g = env.connectivity.copy()
    for _u, _v, data in g.edges(data=True):
        data["A"] = float(data["distance"]) / np.sqrt(3.0)
    return g, env.bin_sizes


def _angular_delta(theta_a: float, theta_b: float) -> float:
    """Minimal absolute angular difference in radians, wrapped to [0, pi].

    Wrapping by ``2*pi`` collapses the polar seam (an angular step spanning the
    full ring "the wrong way") back to a single angular step, so seam edges are
    correctly classified as pure-angular.
    """
    d = (theta_b - theta_a + np.pi) % (2.0 * np.pi) - np.pi
    return float(abs(d))


def _polar_fv(env: EnvironmentProtocol) -> tuple[nx.Graph, NDArray[np.float64]]:
    """Polar face measure from whether two bins are radial or angular neighbors.

    ``bin_centers[:, 0]`` is radius, ``[:, 1]`` is angle. A pure radial neighbor
    shares the arc ``r_face * dtheta`` at the boundary radius; a pure angular
    (or seam) neighbor shares the radial segment ``dr`` of that ring; a diagonal
    neighbor (both differ) shares only a corner (``A = 0``), mirroring Cartesian
    diagonals. Polar sectors are K-orthogonal, so this is analytically exact.
    """
    g = env.connectivity.copy()
    r = env.bin_centers[:, 0]
    theta = env.bin_centers[:, 1]
    distance_edges, angle_edges = cast("Any", env.layout).grid_edges
    dtheta_bin = float(np.diff(angle_edges).mean())
    radial_edges = np.asarray(distance_edges, dtype=float)

    for u, v, data in g.edges(data=True):
        dr = abs(float(r[u]) - float(r[v])) > 1e-9
        dth = _angular_delta(float(theta[u]), float(theta[v])) > 1e-9
        if dr and not dth:  # pure radial: face = arc at the shared boundary radius
            r_face = 0.5 * (float(r[u]) + float(r[v]))
            data["A"] = float(r_face * dtheta_bin)
        elif dth and not dr:  # pure angular / seam: face = radial extent of the ring
            data["A"] = _radial_bin_width(radial_edges, float(r[u]))
        else:  # diagonal (both differ) or degenerate: corner touch, no face
            data["A"] = 0.0
    return g, env.bin_sizes


def _radial_bin_width(radial_edges: NDArray[np.float64], r_value: float) -> float:
    """Radial extent ``dr`` of the ring whose center radius is ``r_value``."""
    idx = int(np.searchsorted(radial_edges, r_value) - 1)
    idx = int(np.clip(idx, 0, len(radial_edges) - 2))
    return float(radial_edges[idx + 1] - radial_edges[idx])


def _graph_fv(env: EnvironmentProtocol) -> tuple[nx.Graph, NDArray[np.float64]]:
    """Linear-track face measure: unit cross-section ``A = 1`` on every edge,
    with junction (inter-segment) edge distances corrected to the true
    along-track length.

    An intra-segment edge already carries the correct along-track spacing (its
    ``"distance"`` is the straight-line length along that segment). An
    inter-segment edge currently carries the Euclidean CHORD between the two
    junction bins, which understates the along-track distance around a bend and
    oversmooths. Its ``"distance"`` is replaced by the along-track length
    ``||bin_u - J|| + ||bin_v - J||`` through the shared junction node ``J`` (the
    geodesic over the straight-segment substrate). A straight track has no
    inter-segment edges, so this is a no-op there.
    """
    g = env.connectivity.copy()
    layout = cast("Any", env.layout)
    build_params = layout._build_params_used
    track_graph = build_params["graph_definition"]
    track_edges = list(track_graph.edges())
    node_pos = nx.get_node_attributes(track_graph, "pos")
    centers = env.bin_centers

    for u, v, data in g.edges(data=True):
        seg_u = g.nodes[u].get("source_edge_id")
        seg_v = g.nodes[v].get("source_edge_id")
        data["A"] = 1.0
        if seg_u is None or seg_v is None or seg_u == seg_v:
            continue  # intra-segment edge: existing along-track distance is correct
        junction = _shared_track_node(track_edges, int(seg_u), int(seg_v))
        if junction is None:
            continue  # no identifiable junction: leave the chord distance as-is
        j_pos = np.asarray(node_pos[junction], dtype=float)
        d = float(np.linalg.norm(centers[u] - j_pos)) + float(
            np.linalg.norm(centers[v] - j_pos)
        )
        data["distance"] = d
    return g, env.bin_sizes


def _shared_track_node(
    track_edges: list[tuple[Any, Any]], seg_a: int, seg_b: int
) -> Any | None:
    """Track-graph node shared by two track segments, or None if not exactly one."""
    if not (0 <= seg_a < len(track_edges) and 0 <= seg_b < len(track_edges)):
        return None
    shared = set(track_edges[seg_a]) & set(track_edges[seg_b])
    if len(shared) != 1:
        return None
    return next(iter(shared))


def _mesh_fv(env: EnvironmentProtocol) -> tuple[nx.Graph, NDArray[np.float64]]:
    """Triangle-mesh face measure: the shared triangle-edge length on every edge.

    Bins are triangle centroids; the shared face between two adjacent triangles
    is their common edge (two shared vertices), whose length is ``A``. Centroid
    two-point flux is exact only as the dual approaches K-orthogonality, so a
    skew guard warns when too many interior edges are strongly non-orthogonal.
    """
    g = env.connectivity.copy()
    layout = cast("Any", env.layout)
    tri = layout._full_delaunay_tri
    active_to_original = np.asarray(layout._active_original_simplex_indices)
    points = tri.points
    simplices = tri.simplices
    centers = env.bin_centers

    skew_angle_rad = np.deg2rad(_MESH_SKEW_ANGLE_DEG)
    n_interior = 0
    n_skewed = 0

    for u, v, data in g.edges(data=True):
        verts_u = set(simplices[active_to_original[u]].tolist())
        verts_v = set(simplices[active_to_original[v]].tolist())
        shared = sorted(verts_u & verts_v)
        if len(shared) != 2:
            # Adjacent triangles must share an edge; if not, no diffusion face.
            data["A"] = 0.0
            continue
        p1 = points[shared[0]]
        p2 = points[shared[1]]
        edge_vec = np.asarray(p2 - p1, dtype=float)
        data["A"] = float(np.linalg.norm(edge_vec))

        # Skew: angle between the centroid-connection line and the shared-edge
        # normal (0 = K-orthogonal).
        n_interior += 1
        centroid_line = np.asarray(centers[v] - centers[u], dtype=float)
        edge_len = np.linalg.norm(edge_vec)
        line_len = np.linalg.norm(centroid_line)
        if edge_len > 0.0 and line_len > 0.0:
            normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
            cos_angle = abs(float(np.dot(centroid_line / line_len, normal)))
            angle = float(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
            if angle > skew_angle_rad:
                n_skewed += 1

    if n_interior > 0 and (n_skewed / n_interior) > _MESH_SKEW_FRACTION:
        warnings.warn(
            f"Triangular mesh is non-K-orthogonal: {n_skewed}/{n_interior} "
            f"({100.0 * n_skewed / n_interior:.1f}%) of interior edges exceed "
            f"{_MESH_SKEW_ANGLE_DEG:.0f} degrees of non-orthogonality "
            f"(threshold {100.0 * _MESH_SKEW_FRACTION:.0f}%). The diffusion "
            f"bandwidth (sigma) is only approximate on this mesh; refine toward "
            f"well-shaped (near-equilateral) triangles for an exact physical sigma.",
            UserWarning,
            stacklevel=2,
        )
    return g, env.bin_sizes
