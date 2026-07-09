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
  **intensive** field (rate maps), volume-unbiased on non-uniform ``M``.

Public entry points are ``Environment.compute_kernel`` / ``Environment.smooth``;
:func:`diffusion_kernel` is the internal dispatcher they route through.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import networkx as nx
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment._protocols import EnvironmentProtocol

__all__ = [
    "apply_heat_operator",
    "diffusion_apply",
    "diffusion_component_labels",
    "diffusion_kernel",
    "heat_kernel_from_W",
    "heat_kernel_rank",
]

# Defaults for the truncated eigenbasis apply-path (``env.diffuse`` only; the
# dense ``compute_kernel`` path is untouched). Drop heat-kernel modes whose
# weight ``exp(-t*lambda)`` is below ``_HEAT_KERNEL_RANK_TOL``; once the resolved
# rank would exceed ``_HEAT_KERNEL_DENSE_FRACTION * n_bins`` a truncated ``eigsh``
# no longer beats a dense ``eigh``, so fall back to a transient dense basis.
# ``_HEAT_KERNEL_RANK_START`` is the first probe size for the adaptive search.
_HEAT_KERNEL_RANK_TOL = 1e-6
_HEAT_KERNEL_DENSE_FRACTION = 0.5
_HEAT_KERNEL_RANK_START = 32

# Tiny floor for masked/support-gated denominators in the consumers (the raw
# linear apply can leave a truncation-noise-tiny value where the dense
# denominator was tiny-positive; the W-component support gate keeps the bin,
# and this floor keeps the division finite -- see design-performance.md Section 5).
_DIFFUSE_DENOM_EPS = 1e-12

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
# Matrix-free apply-path: cached truncated symmetric eigenbasis (env.diffuse).
#
# This is the SOLE eigenbasis surface. It replaces the dense ``expm`` on the hot
# smoothing/apply consumers with a cached, per-component, bandwidth-aware
# truncated eigenbasis of the symmetric conjugate S = M^{-1/2}(D-W)M^{-1/2}, and
# a matrix-free application of H = M^{-1/2} exp(-tS) M^{1/2} that never
# materializes an (n, n) kernel. ``compute_kernel`` (which returns an actual
# matrix) keeps the dense ``expm`` path above and does NOT touch this cache.
# ---------------------------------------------------------------------------
def _assemble_W(graph: nx.Graph, n_bins: int) -> scipy.sparse.csr_matrix:
    """Finite-volume weight matrix ``W[i, j] = A[i, j] / d[i, j]`` from a graph.

    ``A`` (shared-face measure) and ``distance`` (center-to-center) are per-edge
    contract fields; a missing or invalid value raises rather than silently
    degrading the operator. An explicit ``A == 0`` means no diffusion across that
    edge (e.g. corner-touching Cartesian bins) and is skipped. This is the single
    W-assembly used by both the dense ``compute_diffusion_kernels`` path and the
    ``env.diffuse`` eigenbasis, so the two operate on an identical ``W``.

    Parameters
    ----------
    graph : nx.Graph
        Nodes are contiguous integers ``0..n_bins-1``; each edge carries ``"A"``
        (finite, >= 0) and ``"distance"`` (finite, > 0).
    n_bins : int
        Number of nodes (matrix dimension).

    Returns
    -------
    W : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        Symmetric finite-volume weight matrix.

    Raises
    ------
    ValueError
        If an edge is missing ``"A"``/``"distance"`` or carries a non-finite /
        negative ``"A"`` or a non-finite / non-positive ``"distance"``.
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for u, v, data in graph.edges(data=True):
        if "A" not in data or "distance" not in data:
            raise ValueError(
                f"edge ({u},{v}) is missing 'A' and/or 'distance' attribute."
            )
        A = float(data["A"])
        d = float(data["distance"])
        if not (np.isfinite(A) and A >= 0.0):
            raise ValueError(f"edge ({u},{v}) has invalid face measure A={A}.")
        if not (np.isfinite(d) and d > 0.0):
            raise ValueError(f"edge ({u},{v}) has invalid distance d={d}.")
        if A == 0.0:
            continue  # explicit A == 0 => no diffusion across this edge
        w = A / d
        rows += [int(u), int(v)]
        cols += [int(v), int(u)]
        vals += [w, w]
    return scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_bins, n_bins), dtype=np.float64
    )


def _symmetric_conjugate(
    W: scipy.sparse.spmatrix, volumes: NDArray[np.float64]
) -> scipy.sparse.csr_matrix:
    """Symmetric conjugate ``S = M^{-1/2}(D - W)M^{-1/2}`` (symmetric PSD, sparse).

    ``L = M^{-1}(D - W)`` is non-symmetric on non-uniform ``M``; conjugating by
    ``M^{1/2}`` gives the symmetric ``S`` with the same (real, non-negative)
    spectrum, so ``H = exp(-tL) = M^{-1/2} exp(-tS) M^{1/2}``. Its connected
    components equal ``W``'s, and its constant (``lambda = 0``) null mode per
    component is ``M^{1/2} 1`` -- always retained under truncation for exact mass
    conservation.
    """
    d = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt_m = scipy.sparse.diags(1.0 / np.sqrt(volumes))
    return (inv_sqrt_m @ (scipy.sparse.diags(d) - W) @ inv_sqrt_m).tocsr()


def heat_kernel_rank(
    eigvals_ascending: NDArray[np.float64], sigma: float, tol: float
) -> int:
    """Smallest rank keeping every heat-kernel mode with ``exp(-t*lambda) >= tol``.

    ``diffuse`` weights mode ``k`` by ``exp(-t*lambda_k)``, ``t = sigma**2 / 2``,
    so a mode below ``tol`` changes the smoothed field by at most ``tol`` in the
    M-weighted norm. With ``eigvals_ascending`` sorted ascending, the weights are
    descending, so the count of retained modes is a single ``searchsorted``. The
    ``lambda = 0`` null mode always survives (``exp(0) = 1 >= tol``), so a
    per-component call keeps at least that component's null mode.

    Parameters
    ----------
    eigvals_ascending : NDArray[np.float64], shape (m,)
        Eigenvalues of ``S`` in ascending order.
    sigma : float
        Smoothing bandwidth (physical sigma), > 0.
    tol : float
        Heat-kernel weight cutoff, in ``(0, 1)``.

    Returns
    -------
    rank : int
        Number of modes with weight ``>= tol`` (at least 1).
    """
    t = sigma**2 / 2.0
    keep = int(np.searchsorted(-np.exp(-t * eigvals_ascending), -tol, side="right"))
    return max(keep, 1)


def _block_eigenbasis(
    block: scipy.sparse.spmatrix, rank: int | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigendecomposition of a single **connected** block of ``S``.

    ``rank is None`` (or ``rank >= n``) uses dense ``scipy.linalg.eigh``; a
    smaller ``rank`` uses truncated ``scipy.sparse.linalg.eigsh`` with a small
    negative shift-invert, falling back to the no-shift-invert ``which="SM"``
    solver (with a ``UserWarning``) if the factorization fails. Eigenvalues are
    returned ascending and clipped to be non-negative (``S`` is PSD).
    """
    n = block.shape[0]
    if rank is None or rank >= n:
        eigvals, eigvecs = scipy.linalg.eigh(block.toarray())
        return np.clip(eigvals, 0.0, None), eigvecs

    # Deterministic ARPACK start vector so the truncated basis is reproducible
    # (eigsh otherwise draws a random v0, rotating eigenvectors within
    # near-degenerate eigenspaces run-to-run; the diffusion operator is invariant
    # to that rotation, but a fixed v0 makes the cached basis reproducible). A
    # generic direction avoids stagnating on v0 == the constant null mode.
    v0 = np.random.default_rng(0).standard_normal(n)
    try:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(
            block, k=rank, sigma=-1e-8, which="LM", v0=v0
        )
    except RuntimeError:
        # Shift-invert can fail on the near-singular S in some builds (a plain
        # RuntimeError from the sparse LU, or an ArpackError -- itself a
        # RuntimeError subclass). Fall back to the no-shift-invert solver, which
        # is less reliable, so warn.
        warnings.warn(
            "Shift-invert eigsh failed on the diffusion operator; falling back "
            "to the no-shift-invert which='SM' solver, which may return a "
            "lower-quality eigenbasis. Consider a larger bandwidth.",
            UserWarning,
            stacklevel=2,
        )
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(block, k=rank, which="SM", v0=v0)

    order = np.argsort(eigvals)
    return np.clip(eigvals[order], 0.0, None), eigvecs[:, order]


def _symmetric_eigenbasis(
    S: scipy.sparse.spmatrix, rank: int | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-component eigenbasis of ``S = Q Lambda Q^T`` (component-local modes).

    Each ``W``-connected component's block is decomposed separately, so every
    eigenvector is localized to a single component (zero elsewhere). This makes
    truncation (and slicing a cached larger basis) **leak-free by construction**:
    a single global ``eigsh`` could rotate eigenvectors *across* components within
    a degenerate eigenspace, and cutting through it would smear a point source
    across a wall. The globally smallest ``rank`` modes are then kept -- and since
    each component's null mode (``lambda = 0``) is the smallest, all
    ``n_components`` null modes are retained (mass conservation).

    Parameters
    ----------
    S : scipy.sparse.spmatrix, shape (n_bins, n_bins)
        Symmetric conjugate from :func:`_symmetric_conjugate`.
    rank : int or None
        Number of smallest modes to return; ``None`` returns the full basis.

    Returns
    -------
    eigvals : NDArray[np.float64], shape (m,)
        Eigenvalues ascending, non-negative. ``m == rank`` (truncated) or
        ``n_bins`` (full).
    eigvecs : NDArray[np.float64], shape (n_bins, m)
        Component-local orthonormal eigenvectors as columns.

    Raises
    ------
    ValueError
        If ``rank`` is below the number of connected components (which would drop
        a component's null mode and break mass conservation / linearity).
    """
    n_bins = S.shape[0]
    n_components, labels = scipy.sparse.csgraph.connected_components(S, directed=False)
    if rank is not None and rank < n_components:
        raise ValueError(
            f"rank={rank} is below n_components={n_components}: a truncation this "
            "small would drop a connected component's null mode and break "
            "mass conservation. Raise the bandwidth or the rank."
        )

    if n_components == 1:
        return _block_eigenbasis(S, rank)

    S = S.tocsr()
    per_component_rank = n_bins if rank is None else rank
    eigval_parts: list[NDArray[np.float64]] = []
    eigvec_parts: list[NDArray[np.float64]] = []
    for component in range(n_components):
        idx = np.flatnonzero(labels == component)
        block = S[idx][:, idx]
        block_rank = None if per_component_rank >= idx.size else per_component_rank
        block_vals, block_vecs = _block_eigenbasis(block, block_rank)
        padded = np.zeros((n_bins, block_vecs.shape[1]))
        padded[idx] = block_vecs
        eigval_parts.append(block_vals)
        eigvec_parts.append(padded)

    all_eigvals = np.concatenate(eigval_parts)
    all_eigvecs = np.concatenate(eigvec_parts, axis=1)
    keep = all_eigvals.size if rank is None else rank
    order = np.argsort(all_eigvals, kind="stable")[:keep]
    return all_eigvals[order], all_eigvecs[:, order]


def _adaptive_symmetric_basis(
    S: scipy.sparse.spmatrix,
    sigma: float,
    *,
    tol: float,
    dense_fraction: float,
    n_components: int,
) -> tuple[int | None, NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Resolve the truncation rank for ``sigma`` and return the basis at that rank.

    Grows a probe ``k`` (Weyl's law: the eigenvalue-counting function is ~linear
    in 2D, so jump toward the estimated cutoff index) until the largest computed
    eigenvalue brackets the heat-kernel cutoff ``lambda_cut = -ln(tol) / t``, then
    keeps every mode at or below it (``>= n_components``, so no null mode drops).
    Returns ``(None, None, None)`` when the resolved rank would exceed
    ``dense_fraction * n_bins`` -- signalling the caller to use a **transient**
    dense full basis (a light bandwidth on a large grid genuinely needs most
    modes; a truncated ``eigsh`` no longer beats a dense ``eigh`` there).
    """
    n = S.shape[0]
    t = sigma**2 / 2.0
    lambda_cut = -np.log(tol) / t
    max_trunc = int(dense_fraction * n)
    if max_trunc <= n_components:
        return None, None, None

    k = min(max(n_components + 1, _HEAT_KERNEL_RANK_START), max_trunc)
    while True:
        eigvals, eigvecs = _symmetric_eigenbasis(S, rank=k)
        if eigvals[-1] >= lambda_cut:
            keep = max(heat_kernel_rank(eigvals, sigma, tol), n_components)
            return keep, eigvals[:keep], eigvecs[:, :keep]
        if k >= max_trunc:
            return None, None, None  # cutoff not reached below the dense threshold
        estimated = int(np.ceil(1.1 * k * lambda_cut / max(float(eigvals[-1]), 1e-30)))
        if estimated > max_trunc:
            return None, None, None
        k = min(max(estimated, 2 * k), max_trunc)


def apply_heat_operator(
    Q: Any,
    Lam: Any,
    volumes: Any,
    sigma: float,
    F: Any,
    *,
    transpose: bool = False,
    xp: Any = np,
) -> Any:
    """Apply ``H @ F`` (or ``H^T @ F``) via the eigenbasis -- PURE LINEAR.

    ``H = M^{-1/2} Q diag(exp(-t*Lambda)) Q^T M^{1/2}``, ``t = sigma**2 / 2``,
    computed as ``M^{-1/2} Q (coeff ⊙ (Q^T (M^{1/2} F)))``; ``H^T`` swaps the
    ``M^{±1/2}`` powers. There is **no clip and no renormalization** -- positivity
    is a per-consumer concern. ``F`` MUST be 2-D ``(n_bins, n_fields)`` (a 1-D
    ``F`` would make ``sqrt_m * F`` broadcast to ``(n, n)``).

    ``xp`` is the array module (``numpy`` default, or ``jax.numpy`` to run the
    apply on-device); ``Q``, ``Lam``, ``volumes``, ``F`` must already be that
    module's arrays.
    """
    t = sigma**2 / 2.0
    coeff = xp.exp(-t * Lam)  # (rank,)
    sqrt_m = xp.sqrt(volumes).reshape(-1, 1)  # (n, 1)
    if not transpose:  # H @ F
        return (Q @ (coeff[:, None] * (Q.T @ (sqrt_m * F)))) / sqrt_m
    return sqrt_m * (Q @ (coeff[:, None] * (Q.T @ (F / sqrt_m))))  # H^T @ F


def _apply_modes(
    Q: Any,
    Lam: Any,
    volumes: Any,
    sigma: float,
    F: Any,
    mode: Literal["transition", "density", "average"],
    xp: Any,
) -> Any:
    """Apply one of the three mode operators to a 2-D field batch ``F``.

    Reproduces :func:`heat_kernel_from_W`'s three-mode contract via the row-sum
    vector ``r = H @ 1`` and the M-weighted column-mass ``m = H^T @ M`` (each
    computed once per ``sigma`` and a fixed vector, so every mode stays **linear
    in F**): ``average = (H @ F) / r``, ``transition = H^T @ (F / r)``,
    ``density = H @ (F / m)``. All reduce to ``H`` / ``H^T`` applied to ``F`` and
    to ``{1, M}`` -- no ``(n, n)``.
    """

    def _h(x: Any) -> Any:
        return apply_heat_operator(Q, Lam, volumes, sigma, x, transpose=False, xp=xp)

    def _ht(x: Any) -> Any:
        return apply_heat_operator(Q, Lam, volumes, sigma, x, transpose=True, xp=xp)

    if mode == "average":
        r = _safe_denominator(_h(xp.ones_like(F[:, :1])), xp)
        return _h(F) / r
    if mode == "transition":
        r = _safe_denominator(_h(xp.ones_like(F[:, :1])), xp)
        return _ht(F / r)
    if mode == "density":
        m = _safe_denominator(_ht(volumes.reshape(-1, 1)), xp)
        return _h(F / m)
    raise ValueError(
        f"Invalid mode {mode!r}. Choose 'transition', 'density', or 'average'."
    )


def _safe_denominator(d: Any, xp: Any) -> Any:
    """Replace non-positive entries by 1.0 (mirrors the dense kernel's guard).

    ``r`` and ``m`` are ``H @ 1`` / ``H^T @ M`` -- entrywise ~1 and ~volumes
    respectively because the null mode is retained -- so this only guards a
    degenerate all-zero row, matching :func:`heat_kernel_from_W`.
    """
    return xp.where(d > 0.0, d, 1.0)


def _warn_large_dense_basis(n: int) -> None:
    """Warn (never raise) before a near-full-rank ``env.diffuse`` builds a dense
    ``(n, n)`` eigenvector basis -- the inherent cost of a light bandwidth on a
    large grid. Mirrors the dense ``compute_kernel`` GB-estimate warning."""
    from neurospatial.ops.smoothing import _LARGE_KERNEL_THRESHOLD

    if n > _LARGE_KERNEL_THRESHOLD:
        estimated_gb = n * n * 8 / 1e9
        warnings.warn(
            f"env.diffuse resolved a near-full-rank basis for {n} bins (a light "
            f"bandwidth on a large grid needs most modes), so a dense {n} x {n} "
            f"float64 eigenvector matrix (~{estimated_gb:.1f} GB) is built "
            f"transiently, applied, and dropped (never cached). Proceeding; this "
            f"may be slow and memory-intensive. To reduce the cost, increase the "
            f"bandwidth or the bin_size (fewer bins).",
            UserWarning,
            stacklevel=2,
        )


def _resolve_basis(
    holder: dict[str, Any],
    W: scipy.sparse.spmatrix,
    volumes: NDArray[np.float64],
    n_components: int,
    labels: NDArray[np.int_],
    sigma: float,
    *,
    tol: float,
    dense_fraction: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(Q, Lam)`` for ``sigma`` from the growable single-basis cache.

    ``holder`` is a per-geometry mutable dict (owned by a
    ``versioned_cached_property`` on the Environment, so it is dropped wholesale
    on any geometry change). It holds ONE truncated basis -- the max rank
    requested so far, strictly below ``dense_fraction * n`` -- grown by
    **replace**, plus a memo of the resolved rank per ``(sigma, tol)`` so repeat
    calls at a bandwidth (every neuron) skip the adaptive probe.

    A request whose rank would reach ``dense_fraction * n`` (a near-full-rank
    ``env.diffuse``) builds a **transient** dense basis -- applied then dropped,
    never stored -- so it can never grow the cache to ``(n, n)``.
    """
    n = int(volumes.shape[0])
    resolved: dict[tuple[float, float], int | str] = holder.setdefault("resolved", {})
    key = (float(sigma), float(tol))
    t = sigma**2 / 2.0

    # Fast path: rank already resolved for this bandwidth.
    if key in resolved:
        r = resolved[key]
        if isinstance(r, str):  # "dense": transient full basis, never cached
            return _transient_dense_basis(W, volumes, n)
        Q, Lam, _lab = holder["basis"]
        return Q[:, :r], Lam[:r]

    # If the cached basis already brackets this bandwidth's cutoff (its largest
    # retained mode is negligible), slice it -- no new eigensolve.
    cached = holder.get("basis")
    if cached is not None:
        Qc, Lamc, _lab = cached
        if float(np.exp(-t * Lamc[-1])) < tol:
            r = min(
                max(heat_kernel_rank(Lamc, sigma, tol), n_components), Lamc.shape[0]
            )
            resolved[key] = r
            return Qc[:, :r], Lamc[:r]

    # Otherwise probe (and possibly grow the single cached basis).
    S = _symmetric_conjugate(W, volumes)
    rank, eigvals, eigvecs = _adaptive_symmetric_basis(
        S, sigma, tol=tol, dense_fraction=dense_fraction, n_components=n_components
    )
    if rank is None:
        resolved[key] = "dense"
        _warn_large_dense_basis(n)
        vals_full, vecs_full = _symmetric_eigenbasis(S, rank=None)
        return vecs_full, vals_full  # transient, NOT cached
    assert eigvals is not None and eigvecs is not None  # narrowed by rank is not None
    resolved[key] = rank
    cached_rank = holder["basis"][1].shape[0] if "basis" in holder else 0
    if rank >= cached_rank:
        holder["basis"] = (eigvecs, eigvals, labels)
        holder["rank"] = rank
        return eigvecs, eigvals
    Qc, Lamc, _lab = holder["basis"]
    return Qc[:, :rank], Lamc[:rank]


def _transient_dense_basis(
    W: scipy.sparse.spmatrix, volumes: NDArray[np.float64], n: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a full dense eigenbasis of ``S`` transiently (never cached).

    Returns ``(Q, Lam)`` -- eigenvectors first -- matching :func:`_resolve_basis`
    (``_symmetric_eigenbasis`` returns ``(eigvals, eigvecs)``, so swap).
    """
    S = _symmetric_conjugate(W, volumes)
    _warn_large_dense_basis(n)
    eigvals, eigvecs = _symmetric_eigenbasis(S, rank=None)
    return eigvecs, eigvals


def diffusion_apply(
    holder: dict[str, Any],
    W: scipy.sparse.spmatrix,
    volumes: NDArray[np.float64],
    n_components: int,
    labels: NDArray[np.int_],
    fields: NDArray[np.float64],
    sigma: float,
    mode: Literal["transition", "density", "average"],
    *,
    backend: Literal["numpy", "jax"] = "numpy",
    tol: float = _HEAT_KERNEL_RANK_TOL,
    dense_fraction: float = _HEAT_KERNEL_DENSE_FRACTION,
) -> Any:
    """Matrix-free application of the mode operator to a 2-D field batch.

    Resolves ``(Q, Lam)`` from the growable cache (:func:`_resolve_basis`) and
    applies the requested ``mode`` (:func:`_apply_modes`), never materializing an
    ``(n, n)`` kernel. For ``backend="jax"`` the apply runs in ``jax.numpy`` (the
    cached NumPy eigenbasis is cast to ``jnp``), so ``jit`` / ``grad`` / GPU still
    work; the eigenbasis **build** stays NumPy (``scipy``).

    Parameters
    ----------
    holder : dict
        The per-geometry eigenbasis cache (from the Environment).
    W, volumes, n_components, labels
        The finite-volume geometry (from the Environment).
    fields : NDArray, shape (n_bins, n_fields)
        Field batch (must be 2-D; ``env.diffuse`` coerces a 1-D field).
    sigma : float
        Bandwidth (physical sigma), > 0.
    mode : {"transition", "density", "average"}
        Kernel orientation.
    backend : {"numpy", "jax"}, default="numpy"
        Where the apply runs.
    tol, dense_fraction : float
        Truncation controls (see :func:`_adaptive_symmetric_basis`).

    Returns
    -------
    smoothed : NDArray, shape (n_bins, n_fields)
        ``numpy.ndarray`` for ``backend="numpy"``, ``jax.Array`` for
        ``backend="jax"``.
    """
    Q, Lam = _resolve_basis(
        holder,
        W,
        volumes,
        n_components,
        labels,
        sigma,
        tol=tol,
        dense_fraction=dense_fraction,
    )
    if backend == "jax":
        import jax.numpy as jnp

        return _apply_modes(
            jnp.asarray(Q, dtype=jnp.float64),
            jnp.asarray(Lam, dtype=jnp.float64),
            jnp.asarray(volumes, dtype=jnp.float64),
            sigma,
            jnp.asarray(fields, dtype=jnp.float64),
            mode,
            jnp,
        )
    return _apply_modes(
        Q, Lam, volumes, sigma, np.asarray(fields, dtype=np.float64), mode, np
    )


def diffusion_component_labels(
    env: EnvironmentProtocol,
) -> tuple[int, NDArray[np.int_]]:
    """``(n_components, labels)`` of the finite-volume ``W`` for ``env``.

    Reads the Environment's cached finite-volume geometry so the smoothing
    consumers can derive **W-component support** for their strict ``> 0`` gates
    (support that is exact and truncation-proof, unlike the smoothed
    denominator's sign). See design-performance.md Section 5.
    """
    _W, _volumes, n_components, labels = cast("Any", env)._diffusion_geometry
    return int(n_components), labels


def component_support_mask(
    labels: NDArray[np.int_], n_components: int, valid: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """Bins whose ``W``-component contains at least one valid input bin.

    Within a connected ``W``-component the heat kernel is entrywise positive, so
    the dense denominator ``den[i] > 0`` **iff** ``i``'s component holds any valid
    input mass -- a boolean that is exact and truncation-proof. This reproduces
    that support without relying on the (truncation-noisy) smoothed sign.

    Parameters
    ----------
    labels : NDArray[np.int_], shape (n_bins,)
        Per-bin ``W``-component id.
    n_components : int
        Number of components.
    valid : NDArray[np.bool_], shape (..., n_bins)
        Valid-input mask; a leading batch axis (e.g. neurons) is supported.

    Returns
    -------
    support : NDArray[np.bool_], shape (..., n_bins)
        True where the bin's component contains a valid bin.
    """
    valid = np.asarray(valid, dtype=bool)
    lead = valid.shape[:-1]
    flat = valid.reshape(-1, valid.shape[-1])  # (B, n_bins)
    n_batch = flat.shape[0]
    comp_has_valid = np.zeros((n_batch, n_components), dtype=bool)
    rows = np.broadcast_to(np.arange(n_batch)[:, None], flat.shape)
    cols = np.broadcast_to(labels[None, :], flat.shape)
    np.logical_or.at(comp_has_valid, (rows, cols), flat)
    support = comp_has_valid[:, labels]  # (B, n_bins)
    return support.reshape(*lead, valid.shape[-1])


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
        Kernel orientation (see :func:`heat_kernel_from_W`). ``"average"`` is
        the row-stochastic intensive-field smoother.

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
