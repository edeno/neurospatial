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

Public entry points are ``Environment.compute_kernel``, which builds the dense
kernel with :func:`heat_kernel_from_W`, and ``Environment.smooth`` /
``Environment.diffuse``, which apply the operator matrix-free through a cached
truncated eigenbasis and never build the kernel. :func:`diffusion_kernel`
builds the same dense kernel from any object that provides the diffusion
geometry.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, get_args

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
from numpy.typing import NDArray

from neurospatial import _constants
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.ops._types import KernelMode

if TYPE_CHECKING:
    from collections.abc import Callable

    from neurospatial.ops._protocols import DiffusionGeometryConsumer
    from neurospatial.ops._types import ResolvedBackend

# Module exports. The eigenbasis/apply-path functions below are internal:
# ``_diffusion_apply`` serves ``Environment.diffuse`` (environment/_fields.py),
# ``_component_support_mask`` the encoding rate-map smoothing, and
# ``apply_heat_operator`` / ``heat_kernel_rank`` are used only in this module.
__all__ = [
    "MRFBasis",
    "diffusion_kernel",
    "heat_kernel_from_W",
]

# Defaults for the truncated eigenbasis apply-path (``env.diffuse`` only; the
# dense ``compute_kernel`` path is untouched). Drop heat-kernel modes whose
# weight ``exp(-t*lambda)`` is below ``_HEAT_KERNEL_RANK_TOL``; once the resolved
# rank would exceed ``_HEAT_KERNEL_DENSE_FRACTION * n_bins`` a truncated ``eigsh``
# no longer beats a dense ``eigh``, so solve the full dense basis once and keep
# it: it serves every later bandwidth without another eigensolve.
# ``_HEAT_KERNEL_RANK_START`` is the first probe size for the adaptive search.
_HEAT_KERNEL_RANK_TOL = 1e-6
_HEAT_KERNEL_DENSE_FRACTION = 0.5
_HEAT_KERNEL_RANK_START = 32
# Relative accuracy of an untruncated (full-basis) apply: float64 roundoff in
# the eigenbasis product, measured near 1e-14 of the largest output.
_FULL_BASIS_APPLY_RTOL = 1e-13

# MRF penalty-basis resolver constant (consumed by ``_target_live_rank`` and
# ``_select_live_basis`` in this module). ``_DEFAULT_MAX_RANK`` is the
# requested-rank cap ``R`` used when ``rank is None``. It lives here (not in the
# encoding tier) because the pure selector below returns ``MRFBasis`` and must
# not import from a higher tier; no other module uses it. The null-vs-fill split
# is STRUCTURAL (see :func:`_null_mode_mask`), never an absolute eigenvalue
# cutoff, so no null-tolerance constant is needed.
_DEFAULT_MAX_RANK = 250


class MRFBasis(NamedTuple):
    """Reduced-rank penalty basis for the penalized-Poisson GAM.

    Produced by :func:`_select_live_basis` (and ``Environment.get_mrf_basis``);
    consumed by the penalized-Poisson fit. ``B`` and ``d`` are float64 in
    **live-bin order**; ``live_bins`` (np.intp) maps that order back to
    ``env.n_bins`` order. The column layout is fixed: ``B[:, :n_live_components]``
    are the mass-normalized per-component **intercepts** (each an exact constant
    on its live component, weight ``d == 0`` bit-exact), and
    ``B[:, n_live_components:]`` are the **fill** smoothness modes (``M^{-1/2}Q``)
    -- the ``n_fill`` smallest-eigenvalue modes that are not a component's
    constant null. The structural penalty rank ``r_eff - n_live_components`` is
    exact by construction: exactly one constant null is excluded per live
    component (identified structurally, see :func:`_null_mode_mask`), never by an
    absolute eigenvalue cutoff.

    Fields
    ------
    B : NDArray[np.float64], shape (n_live_bins, r_eff)
        ``[ intercepts | M^{-1/2}Q fill ]`` restricted to live bins.
    d : NDArray[np.float64], shape (r_eff,)
        ``[ n_live_components zeros | fill-mode eigenvalues ]``.
    live_bins : NDArray[np.intp], shape (n_live_bins,)
        Indices into the active-bin array (``env.n_bins`` order).
    n_live_components : int
        Count of W-components with ``sum(occupancy) > 0``.
    """

    B: NDArray[np.float64]
    d: NDArray[np.float64]
    live_bins: NDArray[np.intp]
    n_live_components: int


def _frozen(array: NDArray[Any]) -> NDArray[Any]:
    """Mark a freshly computed array read-only before a holder keeps it."""
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class _DiffusionEigenbasis:
    """Truncated heat-kernel eigenbasis of one geometry, replaced as it grows.

    A holder is never edited. Resolving a new bandwidth returns a new holder,
    which the analysis cache stores in place of the old one under its lock,
    so a reader always sees a complete holder.

    Attributes
    ----------
    basis : tuple of ndarray or None
        ``(Q, Lam)``: read-only eigenvectors, shape ``(n_bins, rank)``, and
        ascending eigenvalues, shape ``(rank,)``. The largest basis requested so
        far: a truncated basis below ``dense_fraction * n_bins`` modes, or the
        full ``(n_bins, n_bins)`` basis once any bandwidth needed it. None before
        the first solve.
    resolved : mapping of float to int or "dense"
        Rank resolved for each ``sigma`` at ``_HEAT_KERNEL_RANK_TOL``;
        ``"dense"`` marks a bandwidth applied with the full basis.
    """

    basis: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    resolved: Mapping[float, int | Literal["dense"]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def with_rank(
        self,
        key: float,
        rank: int | Literal["dense"],
        basis: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    ) -> _DiffusionEigenbasis:
        """Return a holder that also records ``rank`` for ``key``."""
        return _DiffusionEigenbasis(
            self.basis if basis is None else basis,
            MappingProxyType({**self.resolved, key: rank}),
        )


@dataclass(frozen=True, eq=False)
class _MRFEigenbasis:
    """Global eigenmodes of one geometry for the MRF basis, replaced as it grows.

    Attributes
    ----------
    Q : ndarray, shape (n_bins, G)
        Read-only component-local eigenvectors, ascending eigenvalue.
    Lam : ndarray, shape (G,)
        Read-only eigenvalues.
    mode_comp : ndarray of intp, shape (G,)
        Read-only source W-component of each mode.
    G : int
        Number of modes held; 0 before the first solve and ``n_bins`` once a
        request needed the dense basis.
    """

    Q: NDArray[np.float64]
    Lam: NDArray[np.float64]
    mode_comp: NDArray[np.intp]
    G: int

    @classmethod
    def empty(cls, n_bins: int) -> _MRFEigenbasis:
        """Return a holder with no modes for ``n_bins`` bins."""
        return cls(
            _frozen(np.zeros((n_bins, 0))),
            _frozen(np.zeros(0)),
            _frozen(np.zeros(0, dtype=np.intp)),
            0,
        )


# Tiny floor for the support-gated denominator of
# :func:`_masked_diffusion_average` (the raw linear apply can leave a
# truncation-noise-tiny value where the dense denominator was tiny-positive; the
# W-component support gate keeps the bin, and this floor keeps the division
# finite -- a near-boundary bin stays finite but is not value-equal to the dense
# path, which is part of the truncation contract).
_DIFFUSE_DENOM_EPS = 1e-12

# A smoothed weight is resolved when it exceeds this many times the diffusion
# apply's noise floor (``env.diffusion_accuracy``) relative to the largest value
# in its W-component. A ratio of two smoothed values at a resolved bin then has
# an error of at most about 1/1000 = 0.1% of its value.
_RESOLUTION_MARGIN = 1e3


# ---------------------------------------------------------------------------
# Core operator (W -> H) for the DENSE path (compute_kernel). The matrix-free
# spectral apply-path (env.diffuse) is a separate surface further below.
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
    them). It backs the dense ``compute_kernel`` path and is the equivalence
    oracle for the matrix-free ``env.diffuse`` apply-path.

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
    t = sigma**2 / 2.0
    # The heat kernel fills in completely, so the dense expm is far cheaper than
    # the sparse one, whose intermediate products become dense anyway.
    kernel = scipy.linalg.expm(-t * _generator(W, volumes).toarray())
    return np.asarray(kernel, dtype=np.float64)


def _generator(
    W: scipy.sparse.spmatrix, volumes: NDArray[np.float64]
) -> scipy.sparse.csr_matrix:
    """Finite-volume diffusion generator ``L = M^-1 (D - W)`` (sparse).

    Parameters
    ----------
    W : scipy.sparse matrix, shape (n_bins, n_bins)
        Symmetric finite-volume weight matrix.
    volumes : NDArray[np.float64], shape (n_bins,)
        Per-bin cell volumes ``M`` (strictly positive).

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        ``M^-1 (D - W)``, where ``D`` holds the row sums of ``W``.
    """
    degree = scipy.sparse.diags(np.asarray(W.sum(axis=1)).ravel())
    return (scipy.sparse.diags(1.0 / volumes) @ (degree - W)).tocsr()


def _warn_dense_kernel(n_bins: int) -> None:
    """Warn (never raise) before a dense ``(n_bins, n_bins)`` heat kernel is built."""
    if n_bins > _constants.LARGE_KERNEL_THRESHOLD:
        estimated_gb = n_bins * n_bins * 8 / 1e9
        warnings.warn(
            f"Computing a dense diffusion kernel for {n_bins} bins. The heat "
            f"kernel exp(-tL) is dense by construction (every entry > 0), so it "
            f"requires an {n_bins} x {n_bins} float64 matrix "
            f"(~{estimated_gb:.1f} GB) -- O(n^2) memory (and O(n^3) time for the "
            f"matrix exponential). Proceeding anyway; this may be slow and "
            f"memory-intensive. To avoid this dense kernel entirely, smooth "
            f"matrix-free via env.diffuse (or the 'diffusion_kde' / 'binned' "
            f"encoding methods, which route through it) -- both scale as "
            f"O(n * rank), not O(n^2). Only compute_kernel (this call) and the "
            f"'gaussian_kde' method build a dense kernel. You can also increase "
            f"bin_size to reduce the bin count.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )


def heat_kernel_from_W(
    W: scipy.sparse.spmatrix,
    volumes: NDArray[np.float64],
    bandwidth: float,
    *,
    mode: KernelMode,
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
    bandwidth : float
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
        If ``mode`` is not one of the three supported values, or ``bandwidth``
        is not finite and positive.

    Warns
    -----
    UserWarning
        When ``n_bins`` exceeds the large-kernel threshold: the kernel is a
        dense ``(n_bins, n_bins)`` float64 matrix whatever the mode.

    Notes
    -----
    At full rank the raw operator is block-diagonal across the connected
    components of ``W`` (a masked wall or corner-only ``A = 0`` splits it), and
    clipping round-off adds no cross-block entries, so the per-mode
    normalization is inherently within-component; no explicit component loop is
    needed here.
    """
    # Validate before the high-bin warning and the dense matrix exponential so a
    # bad argument fails fast rather than after paying the O(n^3) expm cost.
    if mode not in get_args(KernelMode):
        raise ValueError(
            f"Invalid mode {mode!r}. Choose 'transition', 'density', or 'average'."
        )
    if not (np.isfinite(bandwidth) and bandwidth > 0.0):
        raise ValueError(f"bandwidth must be finite and > 0, got {bandwidth}.")
    _warn_dense_kernel(volumes.shape[0])

    # Clip round-off negatives (real cross-block lobes only appear under the
    # truncated env.diffuse apply-path, not on this full-rank dense path).
    H = np.clip(_raw_heat_operator(W, volumes, bandwidth), 0.0, None)

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
            stacklevel=_external_stacklevel(),
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


# ---------------------------------------------------------------------------
# MRF penalty-basis resolver. Reuses the symmetric eigensolver above
# without modifying it. A cached GLOBAL resolver (``_ensure_global_modes`` and
# the environment's cached ``_MRFEigenbasis``) owns the eigensolve and grows on demand; a
# PURE selector (``_select_live_basis``) consumes the cached modes and only masks
# by occupancy. The split keeps the eigensolve behind the cache so a repeat call
# at the same-or-smaller rank / same live support never re-solves, sparse or
# dense.
# ---------------------------------------------------------------------------
def _ensure_global_modes(
    holder: _MRFEigenbasis,
    S: scipy.sparse.spmatrix,
    labels: NDArray[np.int_],
    needed_G: int,
) -> tuple[
    _MRFEigenbasis, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]
]:
    """Return the holder to keep and ``(Q, Lam, mode_comp)`` with ``needed_G`` modes.

    ``Q`` are the ``G`` globally-smallest modes of ``S = M^{-1/2}(D - W)M^{-1/2}``
    and ``mode_comp`` each mode's **source W-component**. In the truncated
    regime a short ``holder`` is replaced by a new one holding the new modes;
    ``holder`` itself is never changed. Each eigenvector is component-local -- nonzero on
    exactly one W-component, bit-exact 0 elsewhere (see
    :func:`_symmetric_eigenbasis`) -- so a mode's source component is the label of
    any nonzero row, recovering the mode -> component map WITHOUT touching the
    eigensolver.

    Past ``_HEAT_KERNEL_DENSE_FRACTION * n`` the resolved basis is the full
    ``n x n`` matrix.
    It is kept like a truncated one, so repeated fits reuse a single dense
    solve; ``env.clear_cache()`` releases it.

    Parameters
    ----------
    holder : _MRFEigenbasis
        The environment's cached modes (see ``environment._fields``).
    S : scipy.sparse.spmatrix, shape (n, n)
        Symmetric conjugate from :func:`_symmetric_conjugate`.
    labels : NDArray[np.int_], shape (n,)
        W-component label per bin.
    needed_G : int
        Minimum number of global modes required.

    Returns
    -------
    holder : _MRFEigenbasis
        The holder to keep: ``holder`` itself, or a grown replacement.
    modes : tuple of ndarray
        ``Q`` (shape ``(n, G)``, component-local eigenvectors as columns in
        ascending eigenvalue order), ``Lam`` (shape ``(G,)``, ascending,
        non-negative) and ``mode_comp`` (shape ``(G,)``, source W-component of
        each mode).
    """
    n = S.shape[0]
    if needed_G <= holder.G:  # cache hit -- no eigensolve
        return holder, (holder.Q, holder.Lam, holder.mode_comp)
    G = min(needed_G, n)
    dense = _HEAT_KERNEL_DENSE_FRACTION * n <= G
    Lam, Q = _symmetric_eigenbasis(S, None if dense else G)
    # Each eigenvector is nonzero on exactly one W-component (bit-exact 0
    # elsewhere from _symmetric_eigenbasis's per-component solve + zero padding),
    # so the label of any nonzero row is the mode's source component. This relies
    # on that per-component solve -- a refactor to a single global eigsh would
    # leak cross-component entries and silently corrupt this map.
    mode_comp = np.array([labels[np.flatnonzero(col)[0]] for col in Q.T], dtype=np.intp)
    grown = _MRFEigenbasis(_frozen(Q), _frozen(Lam), _frozen(mode_comp), Q.shape[1])
    return grown, (grown.Q, grown.Lam, grown.mode_comp)


def _validate_occupancy(occupancy: NDArray[np.float64]) -> None:
    """Reject non-finite or negative occupancy before the live/dead split.

    Occupancy is time (or samples) per active bin, so it must be finite and
    ``>= 0``. A non-finite bin would make its whole W-component's summed
    occupancy non-finite (and ``NaN > 0`` is ``False``), silently dropping the
    component as dead -- or, if it were the only live component, returning the
    empty basis, indistinguishable from genuine zero occupancy. A negative entry
    could likewise cancel a visited component's occupancy below zero. Both are
    caller errors, so fail loudly rather than emit a plausible-but-wrong basis.
    """
    occ = np.asarray(occupancy)
    if not np.all(np.isfinite(occ)):
        bad = np.flatnonzero(~np.isfinite(occ))
        shown = bad[:10].tolist()
        raise ValueError(
            f"occupancy must be finite; {bad.size} non-finite bin(s) at "
            f"indices {shown}{'...' if bad.size > 10 else ''}."
        )
    if np.any(occ < 0.0):
        bad = np.flatnonzero(occ < 0.0)
        shown = bad[:10].tolist()
        raise ValueError(
            f"occupancy must be non-negative (time/samples per bin); "
            f"{bad.size} negative bin(s) at indices "
            f"{shown}{'...' if bad.size > 10 else ''}."
        )


def _live_components(
    labels: NDArray[np.int_], occupancy: NDArray[np.float64]
) -> NDArray[np.intp]:
    """W-components with strictly-positive total occupancy (``sum(o) > 0``)."""
    n_components = int(labels.max()) + 1
    comp_occ = np.bincount(labels, weights=occupancy, minlength=n_components)
    return np.flatnonzero(comp_occ > 0.0).astype(np.intp)


def _target_live_rank(
    n_live_components: int, n_live_bins: int, rank: int | None
) -> int:
    """Effective rank ``r_eff = max(n_live_components, min(n_live_bins, R))``.

    ``R`` is ``rank``, or ``_DEFAULT_MAX_RANK`` when ``rank`` is None.
    """
    R = _DEFAULT_MAX_RANK if rank is None else int(rank)
    return max(n_live_components, min(n_live_bins, R))


def _null_mode_indices(
    Q: NDArray[np.float64],
    mode_comp: NDArray[np.intp],
    volumes: NDArray[np.float64],
    labels: NDArray[np.int_],
    live_comp: NDArray[np.intp],
) -> tuple[NDArray[np.bool_], NDArray[np.intp]]:
    """Locate each live component's structural null; report any that are absent.

    For each live component the constant mode ``q0_c`` proportional to
    ``sqrt(vol) . 1_c`` is the exact null of ``S``; the eigensolver returns it as
    one of that component's modes. It is identified by **maximal overlap** with
    the component's mass-weighted constant (overlap ~1 for the constant, ~0 for
    every orthogonal smoothness mode). This is invariant to the physical
    coordinate scale -- unlike an absolute eigenvalue cutoff, which would
    misclassify a genuine smoothness mode whose Laplacian eigenvalue is
    physically tiny (eigenvalues carry units of 1/length**2).

    Non-raising: a max overlap ``< 0.5`` (or a component with no modes in ``Q``)
    means that component's null is not present in this truncated eigenbasis --
    reported in ``missing`` so the caller (growth loop) can grow, or (selector)
    raise.

    Parameters
    ----------
    Q : NDArray[np.float64], shape (n_bins, G)
        Component-local eigenvectors as columns.
    mode_comp : NDArray[np.intp], shape (G,)
        Source W-component of each mode.
    volumes : NDArray[np.float64], shape (n_bins,)
        Finite-volume cell volumes ``M``.
    labels : NDArray[np.int_], shape (n_bins,)
        W-component label per bin.
    live_comp : NDArray[np.intp]
        Indices of the live components.

    Returns
    -------
    is_null : NDArray[np.bool_], shape (G,)
        ``True`` at the one null mode of each live component found in ``Q``.
    missing : NDArray[np.intp]
        Live components whose null is absent from ``Q``.
    """
    sqrt_vol = np.sqrt(volumes)
    is_null = np.zeros(mode_comp.shape[0], dtype=bool)
    missing: list[int] = []
    for c in live_comp:
        comp_modes = np.flatnonzero(mode_comp == c)
        if comp_modes.size == 0:
            missing.append(int(c))
            continue
        q0 = np.where(labels == c, sqrt_vol, 0.0)
        q0 /= np.linalg.norm(q0)
        overlaps = np.abs(q0 @ Q[:, comp_modes])
        best = int(np.argmax(overlaps))
        if overlaps[best] < 0.5:  # the component's null is not in this basis
            missing.append(int(c))
            continue
        is_null[comp_modes[best]] = True
    return is_null, np.asarray(missing, dtype=np.intp)


def _null_mode_mask(
    Q: NDArray[np.float64],
    mode_comp: NDArray[np.intp],
    volumes: NDArray[np.float64],
    labels: NDArray[np.int_],
    live_comp: NDArray[np.intp],
) -> NDArray[np.bool_]:
    """Mark each live component's structural null; raise if any is absent.

    Thin raising wrapper over :func:`_null_mode_indices` for the selector, which
    (unlike the growth loop) has no opportunity to grow the eigenbasis.
    """
    is_null, missing = _null_mode_indices(Q, mode_comp, volumes, labels, live_comp)
    if missing.size:
        raise ValueError(
            f"component(s) {missing.tolist()}: constant null mode not present in "
            "the supplied eigenbasis; grow it to include every component null "
            "before selecting."
        )
    return is_null


def _select_live_basis(
    Q: NDArray[np.float64],
    Lam: NDArray[np.float64],
    mode_comp: NDArray[np.intp],
    volumes: NDArray[np.float64],
    labels: NDArray[np.int_],
    occupancy: NDArray[np.float64],
    *,
    rank: int | None,
) -> MRFBasis:
    """Mask cached global modes into an :class:`MRFBasis` (pure; no eigensolve).

    Live components are those with ``sum(occupancy) > 0``; dead components
    contribute no rows and no modes. Intercepts are the **exact mass-normalized
    constant** per live component (``1/sqrt(sum_c volumes)`` on component ``c``,
    ``= M^{-1/2}`` of the S-null ``sqrt(vol) . 1_c``), constructed structurally
    with weight ``d == 0``. Fills are the ``n_fill`` smallest-``Lam`` live modes
    that are **not** a component's constant null, carrying ``B = M^{-1/2}Q``. The
    null is excluded **structurally** (:func:`_null_mode_mask`), so
    ``penalty_rank = n_fill`` is exact and no mis-ordered / clipped near-null can
    leak in as a penalized near-constant duplicate. Raises if the supplied
    eigenbasis has fewer than ``n_fill`` non-null live modes (the caller must
    grow it to full rank first).

    Parameters
    ----------
    Q : NDArray[np.float64], shape (n_bins, G)
        Component-local eigenvectors (from :func:`_ensure_global_modes`).
    Lam : NDArray[np.float64], shape (G,)
        Eigenvalues ascending, non-negative.
    mode_comp : NDArray[np.intp], shape (G,)
        Source W-component of each mode.
    volumes : NDArray[np.float64], shape (n_bins,)
        Finite-volume cell volumes ``M``.
    labels : NDArray[np.int_], shape (n_bins,)
        W-component label per bin.
    occupancy : NDArray[np.float64], shape (n_bins,)
        Time (or samples) spent per active bin, active-bin order.
    rank : int or None
        Requested rank cap ``R``; ``None`` resolves to ``_DEFAULT_MAX_RANK``.

    Returns
    -------
    MRFBasis
        The reduced-rank penalty basis in live-bin order.
    """
    _validate_occupancy(occupancy)
    live_comp = _live_components(labels, occupancy)  # live = sum(occupancy) > 0
    n_live_components = int(live_comp.size)
    live_bins = np.flatnonzero(np.isin(labels, live_comp)).astype(np.intp)
    if live_bins.size == 0:  # zero total occupancy -> the empty basis
        return MRFBasis(np.zeros((0, 0)), np.zeros((0,)), live_bins, 0)

    r_eff = _target_live_rank(n_live_components, live_bins.size, rank)
    n_fill = r_eff - n_live_components
    inv_sqrt_vol = 1.0 / np.sqrt(volumes)  # M^{-1/2}; fills are B = M^{-1/2}Q

    # (1) Intercepts: the exact mass-normalized constant per live component.
    #     B0_c = M^{-1/2} q0_c = 1_c / sqrt(sum_c vol) -- constant on component c,
    #     NOT 1_c/||1_c||_2 (which only matches on uniform volumes). d = 0.
    B_int = np.zeros((live_bins.size, n_live_components))
    lbl_live = labels[live_bins]
    for j, c in enumerate(live_comp):
        B_int[lbl_live == c, j] = 1.0 / np.sqrt(volumes[labels == c].sum())

    # (2) Fill: the n_fill smallest-Lam live modes that are NOT a component's
    #     constant null. The null is identified STRUCTURALLY (overlap with the
    #     mass-weighted constant), not by an absolute eigenvalue cutoff -- a
    #     cutoff would misread a genuine smoothness mode with a physically-tiny
    #     eigenvalue as a null (scale-dependent). Lam is ascending, so the
    #     surviving live modes stay Lam-sorted; the first n_fill are the smallest.
    is_null = _null_mode_mask(Q, mode_comp, volumes, labels, live_comp)
    fill_idx = np.flatnonzero(np.isin(mode_comp, live_comp) & ~is_null)[:n_fill]
    if fill_idx.size != n_fill:
        raise ValueError(
            f"_select_live_basis needs {n_fill} fill mode(s) but only "
            f"{fill_idx.size} non-null live mode(s) are present in the "
            f"{Q.shape[1]}-mode eigenbasis. Grow the eigenbasis to full rank "
            "before selecting (Environment.get_mrf_basis does this)."
        )
    d_fill = Lam[fill_idx]
    # A structurally non-null fill must carry a positive penalty weight. A
    # nonpositive weight means the mode's eigenvalue was clipped to <= 0 (a
    # numerically (near-)disconnected component) -- it would be an unpenalized
    # nonconstant direction contradicting the reported penalty_rank. Fail loudly.
    if d_fill.size and np.any(d_fill <= 0.0):
        raise ValueError(
            "a selected non-null fill mode has nonpositive penalty weight "
            f"(min {float(d_fill.min()):.3e}); its eigenvalue was clipped to "
            "<= 0, i.e. a numerically (near-)disconnected component. Check the "
            "environment connectivity / bin resolution."
        )
    B_fill = (inv_sqrt_vol[:, None] * Q[:, fill_idx])[live_bins, :]

    B = np.concatenate([B_int, B_fill], axis=1)  # (n_live_bins, r_eff)
    d = np.concatenate([np.zeros(n_live_components), d_fill])
    return MRFBasis(B, d, live_bins, n_live_components)


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
    ``dense_fraction * n_bins`` -- signalling the caller to use the dense full
    basis (a light bandwidth on a large grid genuinely needs most modes; a
    truncated ``eigsh`` no longer beats a dense ``eigh`` there).
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
    is a per-consumer concern.

    Parameters
    ----------
    Q : array, shape (n_bins, rank)
        Component-local eigenvectors of ``S`` (columns).
    Lam : array, shape (rank,)
        Corresponding eigenvalues.
    volumes : array, shape (n_bins,)
        Per-bin cell volumes ``M`` (strictly positive).
    sigma : float
        Physical smoothing standard deviation (``t = sigma**2 / 2``).
    F : array, shape (n_bins, n_fields)
        Field batch. MUST be 2-D -- a 1-D ``F`` would make ``sqrt_m * F``
        (``sqrt_m`` of shape ``(n, 1)``) broadcast to ``(n, n)`` -- so this is
        checked and raises rather than silently returning a wrong ``(n, n)``.
    transpose : bool, default=False
        Apply ``H^T`` instead of ``H``.
    xp : module, default=numpy
        Array module (``numpy`` or ``jax.numpy`` to run the apply on-device);
        ``Q``, ``Lam``, ``volumes``, ``F`` must already be that module's arrays.

    Returns
    -------
    array, shape (n_bins, n_fields)
        ``H @ F`` (or ``H^T @ F``), in ``xp``'s array type.

    Raises
    ------
    ValueError
        If ``F`` is not 2-D.
    """
    if F.ndim != 2:
        raise ValueError(
            f"F must be 2-D (n_bins, n_fields), got {F.ndim}-D. A 1-D field "
            "would silently broadcast to (n, n); reshape to (n_bins, 1)."
        )
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
    mode: KernelMode,
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
    if n > _constants.LARGE_KERNEL_THRESHOLD:
        estimated_gb = n * n * 8 / 1e9
        warnings.warn(
            f"env.diffuse resolved a near-full-rank basis for {n} bins (a light "
            f"bandwidth on a large grid needs most modes), so a dense {n} x {n} "
            f"float64 eigenvector matrix (~{estimated_gb:.1f} GB) is built once "
            f"and kept in this environment's cache for later bandwidths; "
            f"env.clear_cache() releases it. Proceeding; this may be slow and "
            f"memory-intensive. To reduce the cost, increase the bandwidth or the "
            f"bin_size (fewer bins).",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )


def _resolve_basis(
    holder: _DiffusionEigenbasis,
    W: scipy.sparse.spmatrix,
    volumes: NDArray[np.float64],
    n_components: int,
    sigma: float,
) -> tuple[_DiffusionEigenbasis, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Return the holder to keep and ``(Q, Lam)`` for ``sigma``.

    ``holder`` is the environment's cached :class:`_DiffusionEigenbasis`; it is
    never changed. It holds ONE basis -- the max rank requested so far -- plus
    the resolved rank per ``sigma`` so repeat calls at a bandwidth (every
    neuron) skip the adaptive probe. A request that needs more is answered with
    a replacement holder, which the caller stores in place of ``holder``.

    A request whose rank would reach ``_HEAT_KERNEL_DENSE_FRACTION * n`` (a
    near-full-rank ``env.diffuse``) solves the full dense basis once and keeps
    it, so shuffles, folds and every later bandwidth reuse that one ``(n, n)``
    solve.
    """
    n = int(volumes.shape[0])
    key = float(sigma)
    tol = _HEAT_KERNEL_RANK_TOL
    t = sigma**2 / 2.0

    # Fast path: rank already resolved for this bandwidth.
    r = holder.resolved.get(key)
    if r == "dense":
        # No re-warn here: the large-matrix warning fired once, when the full
        # basis was first solved (the probe path below).
        assert holder.basis is not None  # "dense" is resolved on the full basis
        return holder, holder.basis
    if isinstance(r, int):
        assert holder.basis is not None  # an int rank is resolved on a basis
        Q, Lam = holder.basis
        return holder, (Q[:, :r], Lam[:r])

    # If the cached basis already brackets this bandwidth's cutoff (its largest
    # retained mode is negligible) or is the full basis, slice it -- no new
    # eigensolve. A rank the truncated path would not serve uses all of it.
    if holder.basis is not None:
        Qc, Lamc = holder.basis
        full = Lamc.shape[0] == n
        if full or float(np.exp(-t * Lamc[-1])) < tol:
            r = int(
                min(
                    max(heat_kernel_rank(Lamc, sigma, tol), n_components), Lamc.shape[0]
                )
            )
            if full and r >= _HEAT_KERNEL_DENSE_FRACTION * n:
                return holder.with_rank(key, "dense"), holder.basis
            return holder.with_rank(key, r), (Qc[:, :r], Lamc[:r])

    # Otherwise probe (and possibly grow the single cached basis).
    S = _symmetric_conjugate(W, volumes)
    rank, eigvals, eigvecs = _adaptive_symmetric_basis(
        S,
        sigma,
        tol=tol,
        dense_fraction=_HEAT_KERNEL_DENSE_FRACTION,
        n_components=n_components,
    )
    if rank is None:
        _warn_large_dense_basis(n)
        vals_full, vecs_full = _symmetric_eigenbasis(S, rank=None)
        basis = (_frozen(vecs_full), _frozen(vals_full))
        return holder.with_rank(key, "dense", basis), basis
    assert eigvals is not None and eigvecs is not None  # narrowed by rank is not None
    # The single cached basis is replaced by a larger one (it never shrinks),
    # so a request needing more modes than are cached recomputes it.
    cached_rank = 0 if holder.basis is None else holder.basis[1].shape[0]
    if rank >= cached_rank:
        basis = (_frozen(eigvecs), _frozen(eigvals))
        return holder.with_rank(key, rank, basis), basis
    assert holder.basis is not None  # cached_rank > rank >= 0
    Qc, Lamc = holder.basis
    return holder.with_rank(key, rank), (Qc[:, :rank], Lamc[:rank])


def _diffusion_apply(
    basis: tuple[NDArray[np.float64], NDArray[np.float64]],
    volumes: NDArray[np.float64],
    fields: NDArray[np.float64],
    sigma: float,
    mode: KernelMode,
    *,
    backend: ResolvedBackend = "numpy",
) -> Any:
    """Matrix-free application of the mode operator to a 2-D field batch.

    Applies the requested ``mode`` (:func:`_apply_modes`) with a basis resolved
    by :func:`_resolve_basis`, never materializing an ``(n, n)`` kernel. For
    ``backend="jax"`` the apply runs in ``jax.numpy`` (the cached NumPy
    eigenbasis is cast to ``jnp``), so ``jit`` / ``grad`` / GPU still work; the
    eigenbasis **build** stays NumPy (``scipy``).

    Parameters
    ----------
    basis : tuple of ndarray
        ``(Q, Lam)`` for ``sigma``: eigenvectors, shape ``(n_bins, rank)``,
        and eigenvalues, shape ``(rank,)``.
    volumes : NDArray[np.float64], shape (n_bins,)
        Per-bin cell volumes ``M``.
    fields : NDArray, shape (n_bins, n_fields)
        Field batch (must be 2-D; ``env.diffuse`` coerces a 1-D field).
    sigma : float
        Bandwidth (physical sigma), > 0.
    mode : {"transition", "density", "average"}
        Kernel orientation.
    backend : {"numpy", "jax"}, default="numpy"
        Where the apply runs.

    Returns
    -------
    smoothed : NDArray, shape (n_bins, n_fields)
        ``numpy.ndarray`` for ``backend="numpy"``, ``jax.Array`` for
        ``backend="jax"``.
    """
    Q, Lam = basis
    if backend == "jax":
        import jax.numpy as jnp

        # Cast the cached NumPy eigenbasis to the JAX default float (float64 when
        # x64 is enabled, else float32 -- JAX narrows silently at the boundary, so
        # no explicit dtype is requested here). The apply then runs entirely in
        # jax.numpy, preserving jit / grad / GPU.
        return _apply_modes(
            jnp.asarray(Q),
            jnp.asarray(Lam),
            jnp.asarray(volumes),
            sigma,
            jnp.asarray(fields),
            mode,
            jnp,
        )
    return _apply_modes(
        Q, Lam, volumes, sigma, np.asarray(fields, dtype=np.float64), mode, np
    )


def _diffusion_apply_rtol(
    holder: _DiffusionEigenbasis, n_bins: int, sigma: float
) -> float:
    """Relative accuracy of an apply at ``sigma``, from its resolved rank."""
    rank = holder.resolved.get(float(sigma))
    if rank == "dense" or (isinstance(rank, int) and rank >= n_bins):
        return _FULL_BASIS_APPLY_RTOL
    return _HEAT_KERNEL_RANK_TOL


def _component_support_mask(
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


def _above_component_floor(
    labels: NDArray[np.int_],
    n_components: int,
    values: NDArray[np.float64],
    floor: float,
) -> NDArray[np.bool_]:
    """``values > floor`` times the maximum of ``values`` in each bin's component.

    Parameters
    ----------
    labels : NDArray[np.int_], shape (n_bins,)
        Per-bin ``W``-component id.
    n_components : int
        Number of components.
    values : NDArray[np.float64], shape (..., n_bins)
        Smoothed weights; a leading batch axis is handled row by row.
    floor : float
        Relative floor, a fraction of each component's largest value.

    Returns
    -------
    NDArray[np.bool_], shape (..., n_bins)
        True where the value is resolved above the apply's noise.
    """
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(-1, values.shape[-1])  # (B, n_bins)
    component_max = np.zeros((flat.shape[0], n_components))
    np.maximum.at(component_max, (np.arange(flat.shape[0])[:, None], labels), flat)
    return np.asarray(flat > floor * component_max[:, labels]).reshape(values.shape)


def _masked_diffusion_average(
    apply_average: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    values: NDArray[np.float64],
    valid: NDArray[np.bool_],
    labels: NDArray[np.int_],
    n_components: int,
    *,
    floor: float | Callable[[], float],
) -> NDArray[np.float64]:
    """Average intensive fields over their valid bins with the diffusion operator.

    Nadaraya-Watson with the row-stochastic ``mode="average"`` operator ``A``:
    ``A(values * valid) / A(valid)``. Invalid bins contribute no weight, so they
    neither pull valid neighbors toward zero nor spread NaN, and an invalid bin
    near valid ones is filled from them. The filled values and the validity
    weights are smoothed together in one batch, ``[values | weights]``.

    A bin is NaN when its connected ``W``-component holds no valid bin (exact
    and truncation-proof; see :func:`_component_support_mask`) or when its
    smoothed weight is at most ``floor`` times the largest weight in its
    component (see :func:`_above_component_floor`). Far from every valid bin
    both smoothed sums approach the apply's truncation noise, so their ratio
    would be noise too: one observation of 5 averaged to -53,000 there. The numerator is never floored,
    so signed fields keep their sign; the denominator is floored at
    ``_DIFFUSE_DENOM_EPS`` to keep the division finite.

    Parameters
    ----------
    apply_average : callable
        The environment's average-mode diffusion, ``(n_bins, k) -> (n_bins, k)``.
    values : NDArray[np.float64], shape (n_bins, n_fields)
        Fields to average; entries at invalid bins are ignored.
    valid : NDArray[np.bool_], shape (n_bins, n_fields)
        Which bins of each field are observed.
    labels : NDArray[np.int_], shape (n_bins,)
        Per-bin ``W``-component id.
    n_components : int
        Number of components.
    floor : float or callable
        The far-field gate: a relative floor on the smoothed weight. Each
        caller states its own. A callable is evaluated after smoothing, for a
        floor that depends on the operator the smoothing resolved
        (``env.diffusion_accuracy``).

    Returns
    -------
    NDArray[np.float64], shape (n_bins, n_fields)
        The averaged fields, NaN where unsupported.
    """
    n_fields = values.shape[1]
    filled = np.where(valid, values, 0.0)
    smoothed = np.asarray(
        apply_average(np.concatenate([filled, valid.astype(np.float64)], axis=1))
    )
    numerator = smoothed[:, :n_fields].T
    weights = smoothed[:, n_fields:].T
    support = _component_support_mask(labels, n_components, valid.T)
    relative = floor() if callable(floor) else floor
    support &= _above_component_floor(labels, n_components, weights, relative)
    denominator = np.maximum(weights, _DIFFUSE_DENOM_EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        averaged = np.where(support, numerator / denominator, np.nan)
    return np.asarray(averaged.T, dtype=np.float64)


# ---------------------------------------------------------------------------
# Environment adapters (no layout-specific geometry assembly).
# ---------------------------------------------------------------------------
def diffusion_kernel(
    env: DiffusionGeometryConsumer,
    bandwidth: float,
    *,
    mode: KernelMode = "density",
) -> NDArray[np.float64]:
    """Finite-volume diffusion kernel for an environment.

    Reads the same cached conductances and cell measures used by spectral
    diffusion and smoothness bases, then assembles ``H = exp(-t L)`` and
    returns the requested mode view. Geometry assembly belongs to providers.

    Parameters
    ----------
    env : Environment
        Environment (any supported layout).
    bandwidth : float
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
        If the environment's layout supplies no diffusion geometry.
    ValueError
        On invalid ``bandwidth`` or ``mode``, or on provider geometry that violates
        the diffusion contract.
    """
    geometry = env.get_diffusion_geometry()
    return heat_kernel_from_W(geometry.W, geometry.volumes, bandwidth, mode=mode)
