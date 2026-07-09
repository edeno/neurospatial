"""Tests for the finite-volume diffusion operator (``neurospatial.ops.diffusion``).

Covers grid-independence of the smoothing bandwidth (measured physical sigma ==
``bandwidth`` regardless of bin size), per-geometry sigma recovery, the
mode/orientation contract (transition / density / average), the raw-operator
invariants, ``W``-component structure, and low-level input validation.

Deterministic, explicit construction only -- no random point sampling. The
sigma-measurement helper follows the spec protocol: smooth a unit point source
in the interior (away from boundaries / the polar seam / ``r = 0``) and measure
the physical second moment of the resulting distribution.

A-priori sigma-recovery tolerances are set from each geometry's discretization
error model, NOT tuned to pass. A geometry that misses its tolerance is an
operator/spec correction, not a tolerance relaxation.
"""

import warnings

import networkx as nx
import numpy as np
import pytest
import scipy.sparse
from shapely.geometry import Polygon

from neurospatial import Environment
from neurospatial.environment.polar import EgocentricPolarEnvironment
from neurospatial.layout.engines.graph import GraphLayout
from neurospatial.ops.diffusion import (
    _components_from_W,
    _finite_volume_geometry,
    _raw_heat_operator,
    diffusion_kernel,
)
from neurospatial.ops.smoothing import apply_kernel, compute_diffusion_kernels

# --- a-priori sigma-recovery tolerances (relative) -------------------------
# Cartesian/hex/graph are K-orthogonal regular lattices whose finite-volume
# operator recovers the discrete Gaussian to ~machine precision; polar is
# curvilinear (small Cartesian-anisotropy from measuring an annular-sector
# distribution); mesh is centroid TPFA on a well-shaped (non-equilateral)
# triangulation with an O(1) skew-dependent error.
RTOL_CARTESIAN = 0.02
RTOL_HEX = 0.02
RTOL_POLAR = 0.03
RTOL_GRAPH = 0.02
RTOL_MESH = 0.03


# ===========================================================================
# Deterministic environment builders + sigma-measurement helper
# ===========================================================================
def build_grid_1d(h: float, domain=(0.0, 100.0)) -> Environment:
    """Full 1D regular grid over ``domain`` at spacing ``h`` (explicit edges)."""
    n = round((domain[1] - domain[0]) / h)
    edges = np.linspace(domain[0], domain[1], n + 1)
    return Environment.from_grid_mask(
        active_mask=np.ones(n, dtype=bool), grid_edges=(edges,)
    )


def build_grid_2d(h: float, domain=(0.0, 36.0), *, diagonal=True) -> Environment:
    """Full 2D open-field regular grid over ``domain x domain`` at spacing ``h``."""
    n = round((domain[1] - domain[0]) / h)
    edges = np.linspace(domain[0], domain[1], n + 1)
    return Environment.from_grid_mask(
        active_mask=np.ones((n, n), dtype=bool),
        grid_edges=(edges, edges),
        connect_diagonal_neighbors=diagonal,
    )


def build_grid_2d_masked(h: float, domain=(0.0, 36.0)) -> Environment:
    """2D grid with a corner hole (masked), source region kept away from the hole."""
    n = round((domain[1] - domain[0]) / h)
    edges = np.linspace(domain[0], domain[1], n + 1)
    mask = np.ones((n, n), dtype=bool)
    hole = max(1, n // 6)
    mask[:hole, :hole] = False  # corner hole, far from the domain center
    return Environment.from_grid_mask(
        active_mask=mask, grid_edges=(edges, edges), connect_diagonal_neighbors=True
    )


def build_track(bin_size: float, *, straight: bool) -> Environment:
    """Linear track. ``straight`` -> single segment; else an L (right-angle bend).

    The bend makes the junction (inter-segment) edge's Euclidean chord shorter
    than the true along-track distance, exercising the junction contraction.
    """
    tg = nx.Graph()
    if straight:
        tg.add_node(0, pos=(0.0, 0.0))
        tg.add_node(1, pos=(80.0, 0.0))
        tg.add_edge(0, 1, distance=80.0)
        edge_order = [(0, 1)]
    else:
        tg.add_node(0, pos=(0.0, 0.0))
        tg.add_node(1, pos=(40.0, 0.0))
        tg.add_node(2, pos=(40.0, 40.0))
        tg.add_edge(0, 1, distance=40.0)
        tg.add_edge(1, 2, distance=40.0)
        edge_order = [(0, 1), (1, 2)]
    params = {
        "graph_definition": tg,
        "edge_order": edge_order,
        "edge_spacing": 0.0,
        "bin_size": bin_size,
    }
    layout = GraphLayout()
    layout.build(**params)
    return Environment(
        name="track",
        layout=layout,
        layout_type_used="Graph",
        layout_params_used=params,
    )


def build_W_from_graph(graph: nx.Graph) -> scipy.sparse.csr_matrix:
    """Assemble ``W[i, j] = A / d`` from a face-measure graph (test-side helper)."""
    n = graph.number_of_nodes()
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for u, v, data in graph.edges(data=True):
        a = float(data["A"])
        if a == 0.0:
            continue
        w = a / float(data["distance"])
        rows += [u, v]
        cols += [v, u]
        vals += [w, w]
    return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def measure_sigma(kernel, coords, src, n_measure_dims) -> float:
    """Physical std of the smoothed point source in column ``src`` of ``kernel``.

    ``kernel[:, src]`` is the diffused unit point mass at ``src`` (for the
    transition kernel ``Hᵀ`` this column integrates to 1). Normalizing the
    column to a probability (a per-column scalar) does not change its shape, so
    the second moment is well-defined; for an isotropic Gaussian of std sigma
    the per-axis variances sum to ``n_measure_dims * sigma**2``.
    """
    col = np.clip(np.asarray(kernel[:, src], dtype=float), 0.0, None)
    p = col / col.sum()
    var = 0.0
    for axis in range(coords.shape[1]):
        x = coords[:, axis]
        mean = np.sum(p * x)
        var += np.sum(p * (x - mean) ** 2)
    return float(np.sqrt(var / n_measure_dims))


def polar_cartesian(env) -> np.ndarray:
    """Cartesian ``(x, y)`` coordinates of a polar env's ``(r, theta)`` bins."""
    r = env.bin_centers[:, 0]
    theta = env.bin_centers[:, 1]
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def track_coords_1d(env) -> np.ndarray:
    """Along-track 1D coordinate (``pos_1D``) per bin, shape (n_bins, 1)."""
    pos = nx.get_node_attributes(env.connectivity, "pos_1D")
    return np.array([[pos[i]] for i in range(env.n_bins)])


def nearest_bin(coords: np.ndarray, target) -> int:
    return int(np.argmin(np.linalg.norm(coords - np.asarray(target), axis=1)))


# --- shared non-Cartesian environments -------------------------------------
@pytest.fixture(scope="module")
def polar_env() -> EgocentricPolarEnvironment:
    """Full-circle polar env (non-uniform mass matrix M = annular-sector area)."""
    return EgocentricPolarEnvironment.create(
        distance_range=(0.0, 40.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=2.0,
        angle_bin_size=np.pi / 12,
    )


@pytest.fixture(scope="module")
def mesh_env() -> Environment:
    """Well-shaped flat triangular mesh over a square (uniform point spacing)."""
    boundary = Polygon([(-30, -30), (30, -30), (30, 30), (-30, 30)])
    return Environment.from_layout(
        kind="TriangularMesh",
        layout_params={"boundary_polygon": boundary, "point_spacing": 4.0},
    )


# ===========================================================================
# Grid-independence (Cartesian): measured sigma == bandwidth at every bin size
# ===========================================================================
def test_bandwidth_grid_independent_1d():
    """1D cartesian: measured sigma ~ bandwidth across bin sizes {0.5,1,2,4}."""
    bandwidth = 5.0
    for h in (0.5, 1.0, 2.0, 4.0):
        env = build_grid_1d(h)
        kernel = diffusion_kernel(env, bandwidth, mode="transition")
        src = nearest_bin(env.bin_centers, [50.0])
        measured = measure_sigma(kernel, env.bin_centers, src, n_measure_dims=1)
        assert measured == pytest.approx(bandwidth, rel=RTOL_CARTESIAN), (
            f"1D h={h}: measured sigma {measured} != bandwidth {bandwidth}"
        )


def test_bandwidth_grid_independent_2d():
    """2D open field + masked: measured sigma ~ bandwidth across bin sizes."""
    bandwidth = 5.0
    for h in (2.0, 3.0, 4.0):
        for build in (build_grid_2d, build_grid_2d_masked):
            env = build(h)
            kernel = diffusion_kernel(env, bandwidth, mode="transition")
            src = nearest_bin(env.bin_centers, [18.0, 18.0])
            measured = measure_sigma(kernel, env.bin_centers, src, n_measure_dims=2)
            assert measured == pytest.approx(bandwidth, rel=RTOL_CARTESIAN), (
                f"2D {build.__name__} h={h}: sigma {measured} != {bandwidth}"
            )


# ===========================================================================
# Per-geometry sigma recovery
# ===========================================================================
def test_sigma_recovery_hex():
    """Hex: sigma recovered across >=2 hex sizes (a-priori tolerance)."""
    bandwidth = 6.0
    positions = np.array(
        [[x, y] for x in np.arange(0, 60, 1.0) for y in np.arange(0, 60, 1.0)],
        dtype=float,
    )
    for bin_size in (2.0, 3.0):
        env = Environment.from_samples(
            positions, layout="Hexagonal", bin_size=bin_size, bin_count_threshold=0
        )
        kernel = diffusion_kernel(env, bandwidth, mode="transition")
        src = nearest_bin(env.bin_centers, [30.0, 30.0])
        measured = measure_sigma(kernel, env.bin_centers, src, n_measure_dims=2)
        assert measured == pytest.approx(bandwidth, rel=RTOL_HEX), (
            f"hex bin_size={bin_size}: sigma {measured} != {bandwidth}"
        )


def test_sigma_recovery_polar():
    """Polar: sigma recovered across >=2 resolutions, measured in Cartesian coords.

    Source placed in the interior, away from r=0 and the +-pi seam (spec protocol).
    """
    bandwidth = 5.0
    resolutions = [(2.0, np.pi / 12), (1.5, np.pi / 16)]
    for distance_bin, angle_bin in resolutions:
        env = EgocentricPolarEnvironment.create(
            distance_range=(0.0, 40.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=distance_bin,
            angle_bin_size=angle_bin,
        )
        xy = polar_cartesian(env)
        kernel = diffusion_kernel(env, bandwidth, mode="transition")
        src = nearest_bin(xy, [20.0, 0.0])  # interior, ahead, away from seam/r=0
        measured = measure_sigma(kernel, xy, src, n_measure_dims=2)
        assert measured == pytest.approx(bandwidth, rel=RTOL_POLAR), (
            f"polar ({distance_bin},{angle_bin}): sigma {measured} != {bandwidth}"
        )


def test_sigma_recovery_graph():
    """Linear track with a junction: sigma recovered across bin sizes; the
    junction edge is corrected to the true along-track distance (not the chord),
    so a source near the junction is not oversmoothed."""
    bandwidth = 5.0
    for bin_size in (1.0, 2.0):
        env = build_track(bin_size, straight=False)
        graph_fv, _ = _finite_volume_geometry(env)
        # Junction (inter-segment) edges: the FV along-track distance must exceed
        # the raw Euclidean chord (which cuts the corner and would oversmooth).
        junction_edges = [
            (u, v)
            for u, v in graph_fv.edges()
            if graph_fv.nodes[u].get("source_edge_id")
            != graph_fv.nodes[v].get("source_edge_id")
        ]
        assert junction_edges, "L-track must have at least one junction edge"
        for u, v in junction_edges:
            assert (
                graph_fv.edges[u, v]["distance"]
                > env.connectivity.edges[u, v]["distance"]
            ), "junction edge distance must be corrected upward from the chord"

        coords = track_coords_1d(env)
        kernel = diffusion_kernel(env, bandwidth, mode="transition")
        # Source near the junction (pos_1D ~ 40, the shared node), so smoothing
        # crosses the bend; oversmoothing would inflate the measured sigma.
        src = nearest_bin(coords, [40.0])
        measured = measure_sigma(kernel, coords, src, n_measure_dims=1)
        assert measured == pytest.approx(bandwidth, rel=RTOL_GRAPH), (
            f"graph bin_size={bin_size}: sigma {measured} != {bandwidth}"
        )


def test_sigma_recovery_mesh(mesh_env):
    """Well-shaped flat triangulation: sigma recovered at 2 resolutions."""
    bandwidth = 8.0
    boundary = Polygon([(-30, -30), (30, -30), (30, 30), (-30, 30)])
    for spacing in (4.0, 5.0):
        env = (
            mesh_env
            if spacing == 4.0
            else Environment.from_layout(
                kind="TriangularMesh",
                layout_params={"boundary_polygon": boundary, "point_spacing": spacing},
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # well-shaped; ignore any skew note here
            kernel = diffusion_kernel(env, bandwidth, mode="transition")
        src = nearest_bin(env.bin_centers, [0.0, 0.0])
        measured = measure_sigma(kernel, env.bin_centers, src, n_measure_dims=2)
        assert measured == pytest.approx(bandwidth, rel=RTOL_MESH), (
            f"mesh spacing={spacing}: sigma {measured} != {bandwidth}"
        )


# ===========================================================================
# Mode / orientation contract (C2)
# ===========================================================================
def test_transition_is_column_stochastic_polar(polar_env):
    """transition kernel columns sum to 1; kernel @ counts conserves total mass
    on a non-uniform mass matrix M (polar)."""
    kernel = diffusion_kernel(polar_env, 5.0, mode="transition")
    col_sums = kernel.sum(axis=0)
    np.testing.assert_allclose(col_sums, 1.0, atol=1e-9)

    rng = np.random.default_rng(0)
    counts = rng.random(polar_env.n_bins)
    smoothed = kernel @ counts
    np.testing.assert_allclose(smoothed.sum(), counts.sum(), rtol=1e-9)


def test_transitions_row_stochastic_polar(polar_env):
    """transitions(method='diffusion') rows sum to 1 on a non-uniform-M env."""
    T = polar_env.transitions(method="diffusion", bandwidth=5.0)
    row_sums = np.asarray(T.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)


def test_density_integrates_to_one(polar_env):
    """density kernel: sum_i M_i K[i, j] = 1 (M-weighted columns integrate to 1)."""
    kernel = diffusion_kernel(polar_env, 5.0, mode="density")
    M = polar_env.bin_sizes
    weighted_col_mass = M @ kernel  # (n,): sum_i M_i K[i, j]
    np.testing.assert_allclose(weighted_col_mass, 1.0, atol=1e-9)


def test_low_level_average_row_stochastic(polar_env):
    """compute_diffusion_kernels(..., mode='average') returns row-stochastic H."""
    graph_fv, volumes = _finite_volume_geometry(polar_env)
    kernel = compute_diffusion_kernels(
        graph_fv, volumes=volumes, sigma=5.0, mode="average"
    )
    row_sums = kernel.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)


def test_raw_heat_operator_m_self_adjoint(polar_env):
    """Raw operator (pre clip/normalize): H @ 1 = 1 and M_i H_ij = M_j H_ji on a
    non-uniform-M (polar) env -- the C1 invariants the mode outputs no longer expose."""
    graph_fv, volumes = _finite_volume_geometry(polar_env)
    W = build_W_from_graph(graph_fv)
    H = _raw_heat_operator(W, volumes, sigma=5.0)

    np.testing.assert_allclose(H @ np.ones(len(volumes)), 1.0, atol=1e-9)
    # M-self-adjoint: M_i H_ij == M_j H_ji  <=>  diag(M) H is symmetric.
    mh = volumes[:, None] * H
    np.testing.assert_allclose(mh, mh.T, atol=1e-9)


# ===========================================================================
# W-component structure (C5) + no leakage
# ===========================================================================
def test_components_from_W_corner_split():
    """Corner-only 8-connected pair -> 2 W-components; no mass crosses the corner."""
    # Two active bins touching only at a corner; connect_diagonal_neighbors links
    # them in env.connectivity, but the diagonal shares only a corner => A = 0.
    mask = np.array([[True, False], [False, True]], dtype=bool)
    edges = np.array([0.0, 1.0, 2.0])
    env = Environment.from_grid_mask(
        active_mask=mask, grid_edges=(edges, edges), connect_diagonal_neighbors=True
    )
    assert env.n_bins == 2
    assert env.connectivity.has_edge(0, 1), "corner bins must be 8-connected"

    graph_fv, _ = _finite_volume_geometry(env)
    assert graph_fv.edges[0, 1]["A"] == 0.0, "corner-touching edge must have A == 0"

    W = build_W_from_graph(graph_fv)
    n_components, _labels = _components_from_W(W)
    assert n_components == 2, "corner-only pair must be two W-components"

    kernel = diffusion_kernel(env, 3.0, mode="transition")
    assert kernel[1, 0] == pytest.approx(0.0, abs=1e-12), "no mass across the corner"


def test_no_leakage_across_masked_wall():
    """Point source beside a full masked wall -> ~0 mass across it."""
    # A full inactive column (x-bin index 3) walls the grid into a left half
    # (x-bins 0-2) and a right half (x-bins 4-6), disconnected in connectivity.
    n_x, n_y = 7, 5
    mask = np.ones((n_x, n_y), dtype=bool)
    mask[3, :] = False
    edges_x = np.arange(n_x + 1, dtype=float)
    edges_y = np.arange(n_y + 1, dtype=float)
    env = Environment.from_grid_mask(
        active_mask=mask,
        grid_edges=(edges_x, edges_y),
        connect_diagonal_neighbors=False,
    )
    x = env.bin_centers[:, 0]
    left = x < 3.0
    right = x > 4.0
    assert left.any() and right.any()

    kernel = diffusion_kernel(env, 2.0, mode="transition")
    src = nearest_bin(env.bin_centers, [2.5, 2.5])  # left, beside the wall
    smoothed = kernel[:, src]
    assert smoothed[right].sum() == pytest.approx(0.0, abs=1e-12), (
        "mass must not cross the masked wall"
    )
    assert smoothed[left].sum() == pytest.approx(1.0, rel=1e-9), (
        "mass is conserved within the source's component"
    )


# ===========================================================================
# Adjoint contract on non-uniform M (regression, spec §4)
# ===========================================================================
def test_apply_kernel_adjoint_nonuniform_M(polar_env):
    """apply_kernel(mode='adjoint', bin_sizes=M) on the density kernel obeys the
    M-weighted inner-product contract on a non-uniform-M env: <K x, y>_M == <x, K* y>_M."""
    kernel = diffusion_kernel(polar_env, 5.0, mode="density")
    M = polar_env.bin_sizes
    rng = np.random.default_rng(1)
    x = rng.random(polar_env.n_bins)
    y = rng.random(polar_env.n_bins)

    kx = apply_kernel(x, kernel, mode="forward")
    kstar_y = apply_kernel(y, kernel, mode="adjoint", bin_sizes=M)

    lhs = np.sum(kx * M * y)  # <K x, y>_M
    rhs = np.sum(x * M * kstar_y)  # <x, K* y>_M
    assert lhs == pytest.approx(rhs, rel=1e-9)


# ===========================================================================
# Public mode surface
# ===========================================================================
def test_env_compute_kernel_rejects_unknown_mode():
    """env.compute_kernel / env.smooth reject an unknown mode; the three valid
    modes (transition, density, average) are accepted."""
    env = build_grid_2d(4.0)
    with pytest.raises(ValueError, match=r"mode must be one of"):
        env.compute_kernel(bandwidth=5.0, mode="bogus")
    with pytest.raises(ValueError, match=r"mode must be one of"):
        env.smooth(np.ones(env.n_bins), bandwidth=5.0, mode="bogus")

    for mode in ("transition", "density", "average"):
        kernel = env.compute_kernel(bandwidth=5.0, mode=mode)
        assert kernel.shape == (env.n_bins, env.n_bins)


# ===========================================================================
# Low-level input validation (C6)
# ===========================================================================
def _valid_graph(n=3):
    g = nx.path_graph(n)
    for u, v in g.edges():
        g.edges[u, v]["distance"] = 1.0
        g.edges[u, v]["A"] = 1.0
    return g


def test_compute_diffusion_kernels_missing_A_raises():
    """An edge without 'A' raises ValueError."""
    g = _valid_graph()
    del g.edges[0, 1]["A"]
    with pytest.raises(ValueError, match=r"missing 'A'"):
        compute_diffusion_kernels(g, volumes=np.ones(3), sigma=1.0, mode="transition")


def test_compute_diffusion_kernels_A_zero_no_edge():
    """An A=0 edge carries no diffusion weight (no raise; edge simply dropped)."""
    g = _valid_graph()
    g.edges[0, 1]["A"] = 0.0
    kernel = compute_diffusion_kernels(
        g, volumes=np.ones(3), sigma=1.0, mode="transition"
    )
    # Bin 0 is now only linked (via A) through bin 1's other edge? No: 0-1 has
    # A=0, so bin 0 is isolated -> its column is a delta (no smoothing out).
    assert kernel[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert kernel[1, 0] == pytest.approx(0.0, abs=1e-12)


def test_compute_diffusion_kernels_rejects_bad_inputs():
    """sigma / volumes / node-label / distance / A validation (C6)."""
    base_volumes = np.ones(3)

    # --- sigma: non-finite or <= 0 ---
    for bad_sigma in (np.nan, np.inf, 0.0, -1.0):
        with pytest.raises(ValueError, match=r"sigma must be finite and > 0"):
            compute_diffusion_kernels(
                _valid_graph(), volumes=base_volumes, sigma=bad_sigma, mode="transition"
            )

    # --- volumes: wrong shape ---
    with pytest.raises(ValueError, match=r"volumes must have shape"):
        compute_diffusion_kernels(
            _valid_graph(), volumes=np.ones(4), sigma=1.0, mode="transition"
        )
    # --- volumes: non-finite / non-positive ---
    for bad_volumes in (
        np.array([1.0, np.nan, 1.0]),
        np.array([1.0, np.inf, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, -1.0, 1.0]),
    ):
        with pytest.raises(ValueError, match=r"volumes must be finite"):
            compute_diffusion_kernels(
                _valid_graph(), volumes=bad_volumes, sigma=1.0, mode="transition"
            )

    # --- node labels: non-contiguous ---
    g_gappy = _valid_graph()
    g_gappy = nx.relabel_nodes(g_gappy, {2: 5})
    with pytest.raises(ValueError, match=r"contiguous integer labels"):
        compute_diffusion_kernels(
            g_gappy, volumes=base_volumes, sigma=1.0, mode="transition"
        )
    # --- node labels: float ---
    g_float = nx.relabel_nodes(_valid_graph(), {i: float(i) for i in range(3)})
    with pytest.raises(ValueError, match=r"contiguous integer labels"):
        compute_diffusion_kernels(
            g_float, volumes=base_volumes, sigma=1.0, mode="transition"
        )
    # --- node labels: bool (2-node graph so labels are {False, True}) ---
    g_bool = _valid_graph(2)
    g_bool = nx.relabel_nodes(g_bool, {0: False, 1: True})
    with pytest.raises(ValueError, match=r"contiguous integer labels"):
        compute_diffusion_kernels(
            g_bool, volumes=np.ones(2), sigma=1.0, mode="transition"
        )

    # --- distance: missing / non-finite / non-positive ---
    g_no_d = _valid_graph()
    del g_no_d.edges[0, 1]["distance"]
    with pytest.raises(ValueError, match=r"missing 'A' and/or 'distance'"):
        compute_diffusion_kernels(
            g_no_d, volumes=base_volumes, sigma=1.0, mode="transition"
        )
    for bad_d in (np.nan, np.inf, 0.0, -1.0):
        g = _valid_graph()
        g.edges[0, 1]["distance"] = bad_d
        with pytest.raises(ValueError, match=r"invalid distance"):
            compute_diffusion_kernels(
                g, volumes=base_volumes, sigma=1.0, mode="transition"
            )

    # --- A: negative / NaN / inf ---
    for bad_a in (-1.0, np.nan, np.inf):
        g = _valid_graph()
        g.edges[0, 1]["A"] = bad_a
        with pytest.raises(ValueError, match=r"invalid face measure"):
            compute_diffusion_kernels(
                g, volumes=base_volumes, sigma=1.0, mode="transition"
            )


# ===========================================================================
# Mesh skew guard (D3)
# ===========================================================================
def test_mesh_skew_guard_warns():
    """A strongly sheared (non-K-orthogonal) mesh warns; a well-shaped one does not."""
    # Well-shaped: square, uniform spacing -> right-isoceles triangles, no warning.
    square = Polygon([(-30, -30), (30, -30), (30, 30), (-30, 30)])
    env_ok = Environment.from_layout(
        kind="TriangularMesh",
        layout_params={"boundary_polygon": square, "point_spacing": 4.0},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning here fails the test
        _finite_volume_geometry(env_ok)

    # Skewed: a heavily sheared thin parallelogram -> elongated, obtuse triangles.
    shear = 3.0
    base = [(-30, -8), (30, -8), (30, 8), (-30, 8)]
    sheared = [(x + shear * y, y) for (x, y) in base]
    env_skew = Environment.from_layout(
        kind="TriangularMesh",
        layout_params={"boundary_polygon": Polygon(sheared), "point_spacing": 4.0},
    )
    with pytest.warns(UserWarning, match=r"non-K-orthogonal"):
        _finite_volume_geometry(env_skew)
