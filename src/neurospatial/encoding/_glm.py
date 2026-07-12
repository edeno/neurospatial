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
from typing import Any, NamedTuple

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
_REML_BOUNDARY_TOL = 5 * _REML_XATOL  # within 5*xatol of either log-lambda bound


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
        The ``lambda`` actually applied. A **scalar** for shared / fixed
        ``lambda``, or ``None`` for the REML-skip (``penalty_rank == 0``) and
        no-data cases. Under ``pooled=False`` on the automatic-REML path
        (``penalty is None``, ``penalty_rank > 0``, at least one informative
        unit) it widens to a ``(n_units,)`` vector -- informative units carry
        their own ``lambda_k``, zero-spike units the pooled-``lambda`` fallback.
        A supplied fixed ``penalty`` is echoed (scalar), never discarded.
    penalty_weights : NDArray[np.float64], shape (r_eff,)
        The basis penalty weights ``d`` (``0`` on intercepts, ``> 0`` on fills).
    rank : int
        Effective rank ``r_eff == basis.B.shape[1]`` -- the single source of truth.
    penalty_rank : int
        Structural penalty rank ``r_eff - n_live_components``.
    deviance : NDArray[np.float64], shape (n_units,)
        Unpenalized per-unit Poisson deviance from the reported (floored) rate.
    converged : bool
        Batch-level convergence flag (one shared stopping criterion), never
        per-unit. For a looped per-unit (``pooled=False``) fit it is
        ``all(per-unit converged)``.
    n_iter : int
        Batch-level Newton iteration count, never per-unit. For a looped
        per-unit fit it is ``max(per-unit n_iter)``.
    reml_objective : float or NDArray or None
        The minimized REML objective, or ``None`` when REML did not run (fixed
        penalty, REML-skip, or no data). Under ``pooled=False`` it widens to a
        ``(n_units,)`` vector on the automatic-REML path; a zero-spike fallback
        unit stores ``np.nan`` (its ``lambda`` is not a per-unit REML minimum).
    reml_at_boundary : bool or NDArray or None
        Whether the selected ``log(lambda)`` lies within ``_REML_BOUNDARY_TOL``
        of either fixed search bound: the applied ``lambda`` is finite and the
        fit is valid, but ``lambda`` itself is weakly identified (the optimum may
        lie beyond the interval). A diagnostic/warning, never an automatic
        fallback. ``None`` when REML did not run; a scalar bool for shared
        (``pooled=True``) REML; a ``(n_units,)`` bool vector for per-unit REML (a
        zero-spike fallback unit inherits the pooled-informative search's flag).
    penalty_selected_by_reml : NDArray or None
        Per-unit REML-selection provenance -- ``None`` unless ``pooled=False``
        produced a per-unit vector, then ``(n_units,)`` bool: ``True`` for an
        informative unit whose ``lambda_k`` is its own REML minimum, ``False``
        for a zero-spike unit carrying the pooled-``lambda`` fallback.
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
    reml_at_boundary: bool | NDArray[np.bool_] | None
    penalty_selected_by_reml: NDArray[np.bool_] | None
    pooled: bool


def _structural_constant_base(
    B: NDArray[np.float64], n_live_components: int
) -> NDArray[np.float64]:
    """Return coefficients for the all-ones predictor from exact intercepts.

    The leading ``n_live_components`` columns of an :class:`MRFBasis` are
    disjoint, positive, component-wise constants.  Their levels therefore give
    the all-ones coefficients directly; a full least-squares/SVD solve against
    ``B`` is unnecessary.  Fill-mode coefficients remain exactly zero.
    """
    B = np.asarray(B, dtype=np.float64)
    n_components = int(n_live_components)
    if B.ndim != 2:
        raise ValueError(f"B must be 2-D; got shape {B.shape}.")
    if not 1 <= n_components <= B.shape[1]:
        raise ValueError(
            "n_live_components must be between 1 and the basis rank; got "
            f"{n_components} for B shape {B.shape}."
        )

    intercepts = B[:, :n_components]
    peak_rows = np.argmax(np.abs(intercepts), axis=0)
    levels = intercepts[peak_rows, np.arange(n_components)]
    if np.any(levels == 0.0):
        raise ValueError("a structural component-intercept column is identically zero.")

    base = np.zeros(B.shape[1], dtype=np.float64)
    base[:n_components] = 1.0 / levels
    if not np.allclose(B @ base, 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError(
            "the leading component-intercept columns do not span the all-ones "
            "predictor required by MRFBasis."
        )
    return base


def _is_rank_deficient_on_exposed(
    B: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    r_eff: int,
    n_live_bins: int,
) -> bool:
    """Whether ``B`` restricted to the exposed (``occupancy > 0``) rows has rank
    below ``r_eff``.

    Fast paths avoid the ``O(n_exposed * r_eff^2)`` SVD in the common cases: the
    finite-volume eigenbasis ``B`` has full column rank ``r_eff`` by
    construction, so an all-exposed design is full-rank (no SVD); and fewer
    exposed rows than ``r_eff`` is rank-deficient outright (``rank <= n_exposed <
    r_eff``). Only a partially-exposed design with enough rows needs the SVD.

    Parameters
    ----------
    B : NDArray[np.float64], shape (n_live_bins, r_eff)
        The reduced-rank penalty basis (live-bin order).
    occupancy : NDArray[np.float64], shape (n_live_bins,)
        Dwell time per live bin; exposed bins are ``occupancy > 0``.
    r_eff : int
        Effective rank (``B.shape[1]``).
    n_live_bins : int
        Number of live bins (``B.shape[0]``).

    Returns
    -------
    bool
    """
    n_exposed = int((occupancy > 0).sum())
    if n_exposed == n_live_bins:  # every live bin exposed -> full column rank
        return False
    if n_exposed < r_eff:  # too few rows to reach full rank
        return True
    return bool(np.linalg.matrix_rank(B[occupancy > 0]) < r_eff)


def _reml_boundary_side(penalty: float) -> str:
    """Which fixed log-lambda bound a selected ``penalty`` sits against.

    ``"lower"`` (under-smoothing) when ``log lambda`` is closer to the lower
    bound, else ``"upper"`` (over-smoothing).
    """
    log_lam = float(np.log(penalty))
    lower, upper = _LOG_PENALTY_BOUNDS
    return "lower" if (log_lam - lower) <= (upper - log_lam) else "upper"


def _format_affected_units(indices: Any, unit_ids: Any) -> str:
    """Name the affected units in a per-unit warning.

    When the caller supplies ``unit_ids`` (the public estimator passes its
    resolved labels), name them as ``"unit(s) [<labels>]"``. Otherwise -- a direct
    low-level ``fit_mrf_gam`` call, which is unit-agnostic -- name the positions
    along the unit axis as ``"unit index/indices [<i>]"``.
    """
    idx = list(indices)
    if unit_ids is not None:
        labels = np.asarray(unit_ids)[idx].tolist()
        return f"unit(s) {labels}"
    label = "index" if len(idx) == 1 else "indices"
    return f"unit {label} {idx}"


def _warn_reml_boundary(side: str, who: str, *, stacklevel: int = 3) -> None:
    """Emit the one weakly-identified-lambda warning for a boundary REML optimum.

    ``side`` is ``"lower"`` / ``"upper"`` (or ``"lower/upper"`` when a per-unit
    batch hits both); ``who`` names the affected fit -- ``"the pooled fit"``, or
    ``"unit(s) [<labels>]"`` when the caller supplied ``unit_ids``, else the
    positional ``"unit index/indices [...]"`` (see :func:`_format_affected_units`).
    The applied ``lambda`` is the finite value the search returned near the bound;
    the interval is never expanded. Only weak identification of ``lambda`` is
    established -- the fitted field is finite but its sensitivity to ``lambda`` is
    not measured here.
    """
    warnings.warn(
        f"MRF-GAM REML: the selected smoothing penalty lambda is near the {side} "
        f"search bound for {who}; lambda is weakly identified there (its optimum "
        "may lie at or beyond the bound). The applied lambda is the finite value "
        "the search returned near the bound (the interval is not expanded); the "
        "fitted field remains finite, but its sensitivity to lambda should be "
        "checked.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _fit_mrf_gam_per_unit(
    B: NDArray[np.float64],
    d: NDArray[np.float64],
    counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    r_eff: int,
    n_live_bins: int,
    penalty_rank: int,
    constant_base: NDArray[np.float64],
    newton_fit: Any,
    select_penalty: Any,
    pooled: bool,
    unit_ids: Any = None,
) -> MRFFit:
    """Per-unit REML fit (``pooled=False``, automatic REML, ``penalty_rank > 0``,
    at least one informative unit).

    Runs the shared-lambda REML objective **once per informative unit** to get a
    per-unit ``lambda_k``, assigns zero-spike units the pooled-``lambda`` fallback
    over the informative subset (their ``lambda`` is statistically unidentified),
    then runs the existing batched Newton fit once per distinct ``lambda`` value
    (reused, not forked) and assembles the per-unit vectors. Convergence /
    iteration diagnostics aggregate as ``all`` / ``max`` over the per-unit fits.

    Parameters
    ----------
    B, d, counts, occupancy : NDArray
        The shared basis, penalty weights, per-unit counts ``(n_live_bins,
        n_units)``, and occupancy ``(n_live_bins,)`` (already restricted).
    r_eff, n_live_bins, penalty_rank : int, keyword-only
        Effective rank, live-bin count, and structural penalty rank.
    constant_base : NDArray, keyword-only
        The exact all-ones warm-start direction, reused by every fit.
    newton_fit, select_penalty : callable, keyword-only
        The backend-resolved Newton fit and REML selector (NumPy float64 or the
        float32 JAX mirror), shared with the pooled path.
    pooled : bool, keyword-only
        Echoed onto the result (``False`` here).
    unit_ids : sequence or None, keyword-only, default None
        Per-unit identity labels (aligned with the unit axis) used **only** to
        name the affected units in the convergence / boundary warnings; ``None``
        falls back to the positional unit-axis index. Purely diagnostic.

    Returns
    -------
    MRFFit
        With ``penalty`` / ``reml_objective`` / ``reml_at_boundary`` as
        ``(n_units,)`` vectors and ``penalty_selected_by_reml`` a ``(n_units,)``
        bool mask (``True`` informative, ``False`` zero-spike fallback).
    """
    from ._glm_numpy import _poisson_deviance

    n_units = int(counts.shape[1])
    spike_totals = counts.sum(axis=0)
    informative = spike_totals > 0  # (n_units,) bool
    zero_spike = ~informative

    lam_per_unit = np.empty(n_units, dtype=np.float64)
    reml_obj_per_unit = np.full(n_units, np.nan, dtype=np.float64)
    boundary_per_unit = np.zeros(n_units, dtype=bool)
    # Provenance mask: informative units are REML-selected, zero-spike are fallback.
    selected_by_reml = informative.copy()

    # Per-unit REML over the informative units. penalty_rank > 0 by the guard, so
    # every returned lambda is a real float (never the None REML-skip).
    for k in np.flatnonzero(informative):
        lam_k, obj_k, bnd_k = select_penalty(
            counts[:, k : k + 1],
            occupancy,
            B,
            d,
            penalty_rank,
            constant_base=constant_base,
            max_iter=_MAX_ITER,
            tol=_FIT_TOL,
        )
        lam_per_unit[k] = lam_k
        reml_obj_per_unit[k] = obj_k
        boundary_per_unit[k] = bool(bnd_k)

    # Zero-spike units are statistically unidentified for a per-unit lambda: they
    # take the pooled lambda over the informative subset (computed ONCE, not the
    # optimizer's arbitrary per-unit point), keep reml_objective = nan (sentinel)
    # and inherit that pooled search's boundary flag.
    if zero_spike.any():
        pooled_lam, _pooled_obj, pooled_bnd = select_penalty(
            counts[:, informative],
            occupancy,
            B,
            d,
            penalty_rank,
            constant_base=constant_base,
            max_iter=_MAX_ITER,
            tol=_FIT_TOL,
        )
        lam_per_unit[zero_spike] = pooled_lam
        boundary_per_unit[zero_spike] = bool(pooled_bnd)

    # Final fits: reuse the batched Newton fit once per DISTINCT lambda (zero-spike
    # units share the fallback lambda, so they batch together). converged / n_iter
    # aggregate as all / max over the per-unit fits.
    coeffs = np.empty((r_eff, n_units), dtype=np.float64)
    eta = np.empty((n_live_bins, n_units), dtype=np.float64)
    per_unit_converged = np.empty(n_units, dtype=bool)
    per_unit_n_iter = np.empty(n_units, dtype=np.int64)
    for lam in np.unique(lam_per_unit):
        cols = np.flatnonzero(lam_per_unit == lam)
        c_g, e_g, _mu_g, n_iter_g, _max_step_g, conv_g = newton_fit(
            counts[:, cols], occupancy, B, lam * d, constant_base, _MAX_ITER, _FIT_TOL
        )
        coeffs[:, cols] = c_g
        eta[:, cols] = e_g
        per_unit_converged[cols] = conv_g
        per_unit_n_iter[cols] = n_iter_g

    converged = bool(per_unit_converged.all())
    n_iter = int(per_unit_n_iter.max())

    firing_rate = np.maximum(np.exp(eta), _RATE_FLOOR)
    deviance = _poisson_deviance(counts, occupancy, firing_rate)

    # One aggregate nonconvergence warning naming the affected units (batch-level,
    # not one per unit). Names the caller's ``unit_ids`` when supplied, else the
    # unit-axis indices (the fit itself is unit-agnostic).
    if not converged:
        failed_ids = np.flatnonzero(~per_unit_converged).tolist()
        warnings.warn(
            "MRF-GAM per-unit fit did not converge for "
            f"{_format_affected_units(failed_ids, unit_ids)} (line-search failure, "
            "iteration cap, or out-of-domain data); consider reducing rank or "
            "supplying a fixed penalty.",
            UserWarning,
            stacklevel=2,
        )

    # One boundary warning naming the affected units + side(s). A boundary lambda
    # stays the finite applied penalty (never expanded or replaced).
    if boundary_per_unit.any():
        at_ids = np.flatnonzero(boundary_per_unit)
        log_lams = np.log(lam_per_unit[at_ids])
        lower, upper = _LOG_PENALTY_BOUNDS
        sides = np.where((log_lams - lower) <= (upper - log_lams), "lower", "upper")
        side_desc = "/".join(sorted(set(sides.tolist())))
        _warn_reml_boundary(
            side_desc, _format_affected_units(at_ids.tolist(), unit_ids), stacklevel=2
        )

    return MRFFit(
        coefficients=coeffs,
        log_rate=eta,
        penalty=lam_per_unit,
        penalty_weights=d,
        rank=r_eff,
        penalty_rank=penalty_rank,
        deviance=deviance,
        converged=converged,
        n_iter=n_iter,
        reml_objective=reml_obj_per_unit,
        reml_at_boundary=boundary_per_unit,
        penalty_selected_by_reml=selected_by_reml,
        pooled=pooled,
    )


def fit_mrf_gam(
    basis: MRFBasis,
    counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    penalty: float | None,
    pooled: bool = True,
    backend: str = "numpy",
    unit_ids: Any = None,
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
        Stored on the result. ``pooled=True`` selects a single shared ``lambda``
        by REML over the whole population. ``pooled=False`` selects an
        independent ``lambda_k`` per informative unit (a per-unit REML search),
        widening ``penalty`` / ``reml_objective`` / ``reml_at_boundary`` to
        ``(n_units,)`` vectors on the automatic-REML path; zero-spike units take
        the pooled-``lambda`` fallback over the informative units. It has no
        effect when a fixed ``penalty`` is supplied, at ``penalty_rank == 0``, or
        when no unit has any spikes (the scalar outputs are then identical).
    backend : str, keyword-only, default ``"numpy"``
        Compute backend. ``"jax"`` routes the fit compute (the batched Newton
        fit and REML ``lambda`` selection) to the optional float32 JAX mirror
        when JAX is installed, else falls back to the NumPy float64 core;
        anything else runs the NumPy core. Either way the correctness reference
        is the float64 core and ``MRFFit`` arrays are always NumPy float64.
    unit_ids : sequence, keyword-only, optional
        Per-unit identity labels, aligned to the columns of ``counts``, used
        **only** to name the affected units in the per-unit (``pooled=False``)
        convergence / boundary warnings. Purely diagnostic -- it does not touch
        the numerics. ``None`` (the default, and every direct low-level call)
        names units by their unit-axis index instead; the public estimator passes
        its resolved ``unit_ids`` so the warnings carry the caller's labels.

    Returns
    -------
    MRFFit
        The fit result; see :class:`MRFFit` for the field contract.

    Raises
    ------
    ValueError
        If ``counts`` is not 2-D or its bin count does not match the basis
        (i.e. it was not restricted to ``basis.live_bins``); or if ``unit_ids``
        is supplied but is not a 1-D sequence aligned with the unit axis
        (length ``n_units``).
    """
    from ._backend import is_jax_available
    from ._glm_numpy import (
        _newton_fit_numpy,
        _poisson_deviance,
        select_penalty_by_reml,
    )

    # ``backend`` arrives already resolved from the estimator layer (no user
    # flag); the optional float32 JAX mirror is used only when requested AND the
    # extra is installed. The kernel pair (Newton fit + REML) is selected once,
    # below, after the rank-deficiency check may veto JAX.
    using_jax = backend == "jax" and is_jax_available()

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

    # ``unit_ids`` (diagnostic-only) must align with the unit axis, else a later
    # per-unit warning would index it out of bounds with a raw IndexError.
    if unit_ids is not None:
        ids_arr = np.asarray(unit_ids)
        if ids_arr.ndim != 1 or ids_arr.shape[0] != n_units:
            raise ValueError(
                f"unit_ids must be a 1-D sequence aligned with the unit axis "
                f"(length n_units={n_units}); got shape {ids_arr.shape}."
            )

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
            reml_at_boundary=None,
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
            reml_at_boundary=None,
            penalty_selected_by_reml=None,
            pooled=pooled,
        )

    # --- backend selection for the fit compute, decided BEFORE lambda selection ---
    # Route the Newton fit + REML to the float32 JAX mirror when requested and
    # available, EXCEPT for a design rank-deficient on the exposed (visited) bins:
    # its Hessian's null directions are regularized only by ``penalty * d`` (0 on
    # the intercept columns), so a small -- not only zero -- penalty leaves them
    # (near-)singular, and a float32 solve can wander the null space to a
    # physically-impossible saturated rate while still reporting convergence
    # (observed up to ~1e12 Hz for penalties through ~1e-6); float32 REML can
    # likewise pick a wildly different lambda in the flat over-smoothed regime. A
    # rank-deficient fit therefore runs ENTIRELY on the float64 core (both the REML
    # search and the final fit). Full-rank designs (the norm) keep the fast JAX
    # path. The rank check runs only when it can change behavior -- the JAX path
    # (to veto it) or an explicit penalty=0 (to warn) -- never on a NumPy fit at a
    # nonzero penalty.
    rank_deficient = (
        penalty_rank > 0
        and (using_jax or penalty == 0.0)
        and _is_rank_deficient_on_exposed(B, occupancy, r_eff, n_live_bins)
    )
    if using_jax and not rank_deficient:
        from ._glm_jax import _newton_fit_jax, select_penalty_by_reml_jax

        newton_fit = _newton_fit_jax
        select_penalty = select_penalty_by_reml_jax
    else:
        newton_fit = _newton_fit_numpy
        select_penalty = select_penalty_by_reml
    # penalty=0 on a rank-deficient design is additionally UNIDENTIFIABLE (the
    # fill-mode coefficients are not pinned down at all) -- warn.
    if rank_deficient and penalty == 0.0:
        warnings.warn(
            "MRF-GAM fit: penalty=0 with a design rank-deficient on the "
            "exposed (visited) bins; the fill-mode coefficients are not "
            "identifiable. Increase penalty or reduce rank.",
            UserWarning,
            stacklevel=2,
        )

    # Construct the all-ones warm-start direction exactly from the leading
    # component intercepts once per population fit. Both backends and every REML
    # evaluation reuse it; no basis-wide least-squares factorization is needed.
    constant_base = _structural_constant_base(B, basis.n_live_components)

    # --- per-unit lambda (pooled=False) branch --------------------------------
    # A per-unit REML is meaningful only on the automatic-REML path with a
    # penalized basis and at least one informative unit. Every other pooled=False
    # case (fixed penalty, penalty_rank == 0, all-zero-spike population) has scalar
    # outputs identical to pooled=True, so it falls through to the shared path and
    # only the stored ``pooled`` flag differs.
    if not pooled and penalty is None and penalty_rank > 0 and counts.sum() > 0:
        return _fit_mrf_gam_per_unit(
            B,
            d,
            counts,
            occupancy,
            r_eff=r_eff,
            n_live_bins=n_live_bins,
            penalty_rank=penalty_rank,
            constant_base=constant_base,
            newton_fit=newton_fit,
            select_penalty=select_penalty,
            pooled=pooled,
            unit_ids=unit_ids,
        )

    # --- shared / scalar lambda selection (all-zero-spike handled first) ------
    applied_penalty: float | None
    reml_objective: float | None = None
    reml_at_boundary: bool | None = None
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
        applied_penalty, reml_objective, reml_at_boundary = select_penalty(
            counts,
            occupancy,
            B,
            d,
            penalty_rank,
            constant_base=constant_base,
            max_iter=_MAX_ITER,
            tol=_FIT_TOL,
        )
    else:
        applied_penalty = float(penalty)

    # --- final fit at the chosen lambda ---
    # penalty is None (REML-skip at penalty_rank == 0, or the no-data auto case)
    # means an UNPENALIZED fit: use penalty_diag = zeros_like(d), never None
    # (correct -- penalty_rank == 0 means every weight is a structural null).
    penalty_diag = np.zeros_like(d) if applied_penalty is None else applied_penalty * d
    coeffs, eta, _mu, n_iter, max_step, converged = newton_fit(
        counts, occupancy, B, penalty_diag, constant_base, _MAX_ITER, _FIT_TOL
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

    # --- boundary diagnostic: the shared REML optimum sat on a search bound ---
    if reml_at_boundary and applied_penalty is not None:
        _warn_reml_boundary(_reml_boundary_side(applied_penalty), "the pooled fit")

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
        reml_at_boundary=reml_at_boundary,
        penalty_selected_by_reml=None,
        pooled=pooled,
    )
