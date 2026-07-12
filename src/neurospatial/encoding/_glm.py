"""Penalized-Poisson GAM estimator: constants, ``MRFFit``, orchestrator.

The ``method="glm"`` estimator fits a batched penalized-Poisson GAM with the
finite-volume eigenbasis (:class:`neurospatial.ops.diffusion.MRFBasis`) as its
smoothness-penalty basis: occupancy enters as a log-offset (never a
denominator), ``lambda`` is chosen by REML, and the fit yields finite rates
where a ratio estimator divides by tiny occupancy.

This module hosts the module constants, the :class:`MRFFit` return contract, and
the orchestrator :func:`fit_mrf_gam` (degenerate-case dispatch + deviance). The
float64 numerics live in :mod:`neurospatial.encoding._glm_numpy`; an optional
float32 JAX mirror is a sibling module. It is internal: the public
``compute_spatial_rate(s)`` estimator wires to :func:`fit_mrf_gam`.

The fit is **bin-major**: ``counts`` arrive ``(n_live_bins, n_units)`` already
restricted to ``basis.live_bins`` and ``MRFFit.log_rate`` is
``(n_live_bins, n_units)``. The unit-major <-> bin-major transpose and the
restriction to live bins are the caller's responsibility; ``fit_mrf_gam``
validates the bin count and never re-slices.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial.ops.diffusion import MRFBasis

# ---------------------------------------------------------------------------
# Module constants. Fixed numerics, not public params -- the glm API surface is
# only ``penalty`` / ``rank``. The resolver's ``_DEFAULT_MAX_RANK`` is canonical
# in ``ops.diffusion`` (imported by the public estimator layer), NOT redefined
# here.
# ---------------------------------------------------------------------------
_RATE_FLOOR = 1e-10  # matches the decoding min_rate (likelihood.py:47)
_MAX_ITER = 100  # Newton iterations
_FIT_TOL = 1e-10  # float64 relative penalized-objective decrease
_FIT_TOL_FLOOR = 1e-6  # float32 floor: max(tol, this) -- float32 JAX path only
_DESCENT_TOL = 1e-5  # float32 step-halving "objective increased" slack (JAX path)
_ETA_CLIP = 30.0  # numerical safety limit on the linear predictor (before exp)
_MAX_STEP_HALVINGS = 30  # per Newton iteration
_HESSIAN_JITTER = 1e-10  # ridge on the Hessian diagonal
_LOG_PENALTY_BOUNDS = (-8.0, 20.0)  # REML search bounds in log lambda
_REML_XATOL = 1e-3  # scipy.optimize.minimize_scalar xatol


class MRFFit(NamedTuple):
    """Penalized-Poisson GAM fit result (bin-major, always NumPy arrays).

    Produced by :func:`fit_mrf_gam`; consumed by the public estimator layer,
    which transposes the bin-major ``log_rate`` back to the unit-major public
    ``firing_rates`` and scatters it into the full active-bin array. The public
    return-type conversion (dtype / JAX) is the caller's job -- these arrays are
    always NumPy.

    Fields
    ------
    coefficients : NDArray[np.float64], shape (r_eff, n_units)
        Coefficients ``gamma`` on the live basis.
    log_rate : NDArray[np.float64], shape (n_live_bins, n_units)
        The (clipped) linear predictor ``eta = B gamma``; the reported rate is
        ``max(exp(log_rate), _RATE_FLOOR)``.
    penalty : float or NDArray or None
        The ``lambda`` actually applied -- a scalar for shared / fixed
        ``lambda``, or ``None`` for the REML-skip (``penalty_rank == 0``) and
        no-data cases. A supplied fixed ``penalty`` is echoed, never discarded.
        (Per-unit ``lambda`` vectors are not yet supported.)
    penalty_weights : NDArray[np.float64], shape (r_eff,)
        The basis penalty weights ``d`` (``0`` on intercepts, ``> 0`` on fills).
    rank : int
        Effective rank ``r_eff == basis.B.shape[1]`` -- the single source of truth.
    penalty_rank : int
        Structural penalty rank ``r_eff - n_live_components``.
    deviance : NDArray[np.float64], shape (n_units,)
        Unpenalized per-unit Poisson deviance from the reported (floored) rate.
    converged : bool
        Batch-level convergence flag (one shared stopping criterion), never per-unit.
    n_iter : int
        Batch-level Newton iteration count, never per-unit.
    reml_objective : float or NDArray or None
        The minimized REML objective, or ``None`` when REML did not run
        (fixed penalty, REML-skip, or no data). (Per-unit REML objectives are
        not yet supported.)
    penalty_selected_by_reml : NDArray or None
        Per-unit REML-selection flags -- for the per-unit (``pooled=False``)
        path only, which is not yet supported; always ``None`` here.
    pooled : bool
        The input ``pooled`` flag. Stored because for fixed-penalty / ``r == 0``
        / no-data cases the scalar outputs are identical under ``pooled=True``
        and ``pooled=False``, so it cannot be reconstructed from the values.
    """

    coefficients: NDArray[np.float64]
    log_rate: NDArray[np.float64]
    penalty: float | NDArray[np.float64] | None
    penalty_weights: NDArray[np.float64]
    rank: int
    penalty_rank: int
    deviance: NDArray[np.float64]
    converged: bool
    n_iter: int
    reml_objective: float | NDArray[np.float64] | None
    penalty_selected_by_reml: NDArray[np.bool_] | None
    pooled: bool


def fit_mrf_gam(
    basis: MRFBasis,
    counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    penalty: float | None,
    pooled: bool = True,
    backend: str = "numpy",
) -> MRFFit:
    """Fit the batched penalized-Poisson GAM on a live-bin basis.

    Orchestrates the float64 Newton fit + REML ``lambda`` selection, handling the
    degenerate short-circuits first. ``counts`` / ``occupancy`` arrive **already
    restricted to** ``basis.live_bins`` (the caller owns the unit-major ->
    bin-major transpose and the restriction); this function validates the bin
    count and never re-slices.

    Parameters
    ----------
    basis : MRFBasis
        The reduced-rank penalty basis (live-bin order). The single source of
        truth for rank: ``MRFFit.rank == basis.B.shape[1]``.
    counts : NDArray, shape (n_live_bins, n_units)
        Per-unit binned spike counts, restricted to ``basis.live_bins``.
    occupancy : NDArray[np.float64], shape (n_live_bins,)
        Dwell time per live bin (the Poisson log-offset), same restriction.
    penalty : float or None, keyword-only
        Fixed ``lambda`` (echoed on the result), or ``None`` to select by REML.
    pooled : bool, keyword-only, default True
        Stored on the result. Only ``pooled=True`` (a single shared lambda) is
        supported; ``pooled=False`` (per-unit lambda) raises ``NotImplementedError``.
    backend : str, keyword-only, default ``"numpy"``
        Compute backend. ``"jax"`` routes the fit compute (the batched Newton
        fit and REML ``lambda`` selection) to the optional float32 JAX mirror
        when JAX is installed, else falls back to the NumPy float64 core;
        anything else runs the NumPy core. Either way the correctness reference
        is the float64 core and ``MRFFit`` arrays are always NumPy float64.

    Returns
    -------
    MRFFit
        The fit result; see :class:`MRFFit` for the field contract.

    Raises
    ------
    ValueError
        If ``counts`` is not 2-D or its bin count does not match the basis
        (i.e. it was not restricted to ``basis.live_bins``).
    NotImplementedError
        If ``pooled=False`` (per-unit lambda selection is not yet supported).
    """
    from ._backend import is_jax_available
    from ._glm_numpy import (
        _newton_fit_numpy,
        _poisson_deviance,
        select_penalty_by_reml,
    )

    # Backend-driven dispatch: route the fit compute (Newton fit + REML) to the
    # optional float32 JAX mirror when backend="jax" and the extra is installed,
    # else the NumPy float64 core. The mirror casts back to NumPy float64, so the
    # deviance and MRFFit assembly below are backend-agnostic. There is no user
    # flag -- backend arrives already resolved from the estimator layer.
    using_jax = backend == "jax" and is_jax_available()
    if using_jax:
        from ._glm_jax import _newton_fit_jax, select_penalty_by_reml_jax

        newton_fit = _newton_fit_jax
        select_penalty = select_penalty_by_reml_jax
    else:
        newton_fit = _newton_fit_numpy
        select_penalty = select_penalty_by_reml

    if not pooled:
        # Per-unit lambda selection is not implemented: the fit only ever selects
        # a single shared lambda. Reject rather than run pooled REML and mislabel
        # the result pooled=False (per-unit).
        raise NotImplementedError(
            "pooled=False (per-unit lambda selection) is not yet supported; the "
            "fit uses a single shared lambda. Pass pooled=True."
        )

    B = np.asarray(basis.B, dtype=np.float64)
    d = np.asarray(basis.d, dtype=np.float64)
    counts = np.asarray(counts)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    r_eff = int(B.shape[1])
    n_live_bins = int(B.shape[0])
    penalty_rank = int(d.size - basis.n_live_components)  # structural

    # --- shape validation: inputs must already be restricted to live_bins ---
    if counts.ndim != 2:
        raise ValueError(
            f"counts must be 2-D (n_live_bins, n_units); got shape {counts.shape}."
        )
    if counts.shape[0] != n_live_bins:
        raise ValueError(
            f"counts has {counts.shape[0]} bin rows but the basis has "
            f"{n_live_bins} live bins. counts and occupancy must be restricted to "
            "basis.live_bins before calling fit_mrf_gam; it validates the bin "
            "count and never re-slices."
        )
    if occupancy.shape != (n_live_bins,):
        raise ValueError(
            f"occupancy must have shape ({n_live_bins},) matching the basis live "
            f"bins; got {occupancy.shape}."
        )
    n_units = int(counts.shape[1])

    # --- degenerate: no neurons (skip the fit) ---
    if n_units == 0:
        return MRFFit(
            coefficients=np.zeros((r_eff, 0)),
            log_rate=np.zeros((n_live_bins, 0)),
            penalty=None,
            penalty_weights=d,
            rank=r_eff,
            penalty_rank=penalty_rank,
            deviance=np.zeros(0),
            converged=True,
            n_iter=0,
            reml_objective=None,
            penalty_selected_by_reml=None,
            pooled=pooled,
        )

    # --- degenerate: zero total occupancy / empty basis (skip the fit) ---
    if n_live_bins == 0:
        warnings.warn(
            "MRF-GAM fit: zero total occupancy (no live bins); every rate floors "
            f"to _RATE_FLOOR ({_RATE_FLOOR:.0e}).",
            UserWarning,
            stacklevel=2,
        )
        return MRFFit(
            coefficients=np.zeros((r_eff, n_units)),
            log_rate=np.zeros((0, n_units)),
            penalty=None,
            penalty_weights=d,
            rank=r_eff,
            penalty_rank=penalty_rank,
            deviance=np.zeros(n_units),
            converged=True,
            n_iter=0,
            reml_objective=None,
            penalty_selected_by_reml=None,
            pooled=pooled,
        )

    # --- rank-deficiency float64 fallback (decided BEFORE lambda selection) ---
    # A design that is rank-deficient on the exposed (visited) bins has a Hessian
    # whose null directions are regularized only by ``penalty * d`` (0 on the
    # intercept columns), so a small -- not only zero -- penalty leaves them
    # (near-)singular. A float32 (JAX) solve of a (near-)singular system is
    # unreliable: it can wander the null space to a physically-impossible
    # saturated rate while still reporting convergence (observed up to ~1e12 Hz
    # for penalties through ~1e-6), and float32 REML can select a wildly different
    # lambda in the flat over-smoothed regime. So a rank-deficient fit runs
    # ENTIRELY on the float64 core -- both the REML lambda search and the final
    # fit -- regardless of backend. Full-rank designs (the norm) are
    # well-conditioned at any penalty (including 0) and keep the fast JAX path;
    # REML never selects a blow-up penalty. The (SVD) rank check runs only when it
    # can change behavior: the JAX path (to decide the fallback) or an explicit
    # penalty=0 (to warn) -- never on a NumPy fit at a nonzero penalty.
    # Fast paths avoid the O(n_exposed * r_eff^2) SVD in the common cases: the
    # eigenbasis ``B`` has full column rank ``r_eff`` by construction, so when
    # every live bin is exposed the exposed design is full-rank (no SVD needed);
    # and fewer exposed bins than ``r_eff`` is rank-deficient outright (rank <=
    # n_exposed < r_eff). Only a partially-exposed design with enough rows needs
    # the SVD.
    rank_deficient = False
    if penalty_rank > 0 and (using_jax or penalty == 0.0):
        exposed = occupancy > 0
        n_exposed = int(exposed.sum())
        if n_exposed < r_eff:
            rank_deficient = True
        elif n_exposed < n_live_bins:
            rank_deficient = bool(np.linalg.matrix_rank(B[exposed]) < r_eff)
        # else n_exposed == n_live_bins -> full column rank by construction.
    if rank_deficient:
        newton_fit = _newton_fit_numpy
        select_penalty = select_penalty_by_reml
    # penalty=0 on a rank-deficient design is additionally UNIDENTIFIABLE (the
    # fill-mode coefficients are not pinned down at all) -- warn regardless of the
    # numerically-stable float64 fallback.
    if rank_deficient and penalty == 0.0:
        warnings.warn(
            "MRF-GAM fit: penalty=0 with a design rank-deficient on the "
            "exposed (visited) bins; the fill-mode coefficients are not "
            "identifiable. Increase penalty or reduce rank.",
            UserWarning,
            stacklevel=2,
        )

    # --- lambda selection (all-zero-spike population handled first) ---
    applied_penalty: float | None
    reml_objective: float | None = None
    if counts.sum() == 0:
        # No unit has a spike: lambda is unidentified, so skip REML *selection*.
        # Still respect the fixed-penalty contract -- echo a supplied fixed
        # penalty, else None (auto). The fit runs; every rate floors to the floor.
        warnings.warn(
            "MRF-GAM fit: no unit has any spikes; lambda is unidentified so REML "
            "selection is skipped and every rate floors to _RATE_FLOOR.",
            UserWarning,
            stacklevel=2,
        )
        applied_penalty = None if penalty is None else float(penalty)
    elif penalty is None:
        applied_penalty, reml_objective = select_penalty(
            counts, occupancy, B, d, penalty_rank, max_iter=_MAX_ITER, tol=_FIT_TOL
        )
    else:
        applied_penalty = float(penalty)

    # --- final fit at the chosen lambda ---
    # penalty is None (REML-skip at penalty_rank == 0, or the no-data auto case)
    # means an UNPENALIZED fit: use penalty_diag = zeros_like(d), never None
    # (correct -- penalty_rank == 0 means every weight is a structural null).
    penalty_diag = np.zeros_like(d) if applied_penalty is None else applied_penalty * d
    coeffs, eta, _mu, n_iter, max_step, converged = newton_fit(
        counts, occupancy, B, penalty_diag, _MAX_ITER, _FIT_TOL
    )

    # Floored rate from the FINAL eta so the deviance describes what is reported.
    firing_rate = np.maximum(np.exp(eta), _RATE_FLOOR)  # (n_live_bins, n_units)
    deviance = _poisson_deviance(counts, occupancy, firing_rate)

    # --- nonconvergence warning, keyed on the FLAG (not n_iter == max_iter) ---
    if not converged:
        warnings.warn(
            f"MRF-GAM fit did not converge in {n_iter} Newton iterations "
            f"(max coefficient step {max_step:.3e}); consider reducing rank or "
            "increasing penalty.",
            UserWarning,
            stacklevel=2,
        )

    return MRFFit(
        coefficients=coeffs,
        log_rate=eta,
        penalty=applied_penalty,
        penalty_weights=d,
        rank=r_eff,
        penalty_rank=penalty_rank,
        deviance=deviance,
        converged=converged,
        n_iter=n_iter,
        reml_objective=reml_objective,
        penalty_selected_by_reml=None,
        pooled=pooled,
    )
