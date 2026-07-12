"""Float64 NumPy/SciPy core of the penalized-Poisson GAM fit + REML.

The batched penalized-Poisson Newton/IRLS fit and REML ``lambda`` selection,
plus the deviance -- the correctness reference for the ``method="glm"``
estimator. Batched over the neuron axis: the smoothness basis ``B`` and
occupancy ``o`` are shared; the spike counts ``n_k`` are per neuron.

All ``MRFFit`` assembly, degenerate-case dispatch, and module constants live in
:mod:`neurospatial.encoding._glm`; this module is the pure float64 numerics it
orchestrates. An optional float32 JAX mirror lives in a sibling module.

Convergence is on the **relative penalized-objective decrease**, never the
coefficient step (weakly-identified null-mode directions drift forever). The
``converged`` / ``n_iter`` diagnostics are batch-level scalars -- one shared
stopping criterion (the max relative decrease across units), not per-unit.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.optimize
from numpy.typing import NDArray

from ._glm import (
    _ETA_CLIP,
    _HESSIAN_JITTER,
    _LOG_PENALTY_BOUNDS,
    _MAX_STEP_HALVINGS,
    _REML_BOUNDARY_TOL,
    _REML_XATOL,
)


# ---------------------------------------------------------------------------
# Objective, warm start, Hessian
# ---------------------------------------------------------------------------
def _constant_warm_start(
    B: NDArray[np.float64],
    constant_base: NDArray[np.float64],
    log_rate0: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Warm-start coefficients giving a constant log-rate per unit.

    ``constant_base`` is constructed exactly from the leading component
    intercepts by :func:`neurospatial.encoding._glm._structural_constant_base`.
    Scaling it by each unit's log rate gives a flat field without a
    least-squares factorization.

    Parameters
    ----------
    B : NDArray[np.float64], shape (n_bins, r)
        Reduced-rank penalty basis (live-bin order).
    constant_base : NDArray[np.float64], shape (r,)
        Exact coefficients for ``B @ constant_base == 1``.
    log_rate0 : NDArray[np.float64], shape (n_units,)
        Constant log-rate warm start per unit.

    Returns
    -------
    NDArray[np.float64], shape (r, n_units)
        Warm-start coefficients.
    """
    base = np.asarray(constant_base, dtype=np.float64)
    if base.shape != (B.shape[1],):
        raise ValueError(
            f"constant_base must have shape ({B.shape[1]},); got {base.shape}."
        )
    return base[:, None] * log_rate0[None, :]


def _penalized_obj(
    coeffs: NDArray[np.float64],
    B: NDArray[np.float64],
    counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    penalty_diag: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-unit negative penalized Poisson log-likelihood, evaluated with the
    **clipped** predictor ``eta_c = clip(B @ coeffs, +/-_ETA_CLIP)`` in BOTH the
    linear ``n*eta_c`` term and ``mu = occupancy * exp(eta_c)``:
    ``-sum(n*eta_c - mu) + 0.5*sum(penalty_diag * coeffs^2)``.

    The clip appears in both terms so the value stays **bounded** even for an
    out-of-box trial the line search is about to reject (with raw ``eta`` in
    ``n*eta`` but clipped ``mu``, the value runs to ``-inf`` as ``eta -> +inf``).
    The fit keeps every *accepted* iterate inside the box
    (:func:`_step_halve` treats ``|B @ coeffs| <= _ETA_CLIP`` as a constraint),
    so on accepted points ``eta_c == B @ coeffs`` and this is exactly the
    standard Poisson objective whose gradient the Newton step uses.

    Parameters
    ----------
    coeffs : NDArray[np.float64], shape (r, n_units)
    B : NDArray[np.float64], shape (n_bins, r)
    counts : NDArray[np.float64], shape (n_bins, n_units)
    occupancy : NDArray[np.float64], shape (n_bins,)
    penalty_diag : NDArray[np.float64], shape (r,)
        ``penalty * d``.

    Returns
    -------
    NDArray[np.float64], shape (n_units,)
    """
    eta = np.clip(B @ coeffs, -_ETA_CLIP, _ETA_CLIP)  # (n_bins, n_units)
    mu = occupancy[:, None] * np.exp(eta)
    neg_loglik = -np.sum(counts * eta - mu, axis=0)  # (n_units,)
    pen = 0.5 * np.sum(penalty_diag[:, None] * coeffs**2, axis=0)  # (n_units,)
    return np.asarray(neg_loglik + pen, dtype=np.float64)


def _rel_decrease(prev_obj: NDArray[np.float64], obj: NDArray[np.float64]) -> float:
    """Batch stopping scalar: the **max** across units of the penalized-objective
    decrease relative to ``1 + |objective|``. Convergence is declared when this
    falls below ``tol`` -- so every unit decreased by less than ``tol`` this
    iteration (a stuck unit's non-positive decrease never raises the max).

    The ``1 + |obj|`` denominator (mgcv-style) is a **hybrid** measure: relative
    for large objectives, absolute for small ones. This is what lets a unit whose
    objective shrinks toward 0 -- a zero-spike unit whose unpenalized intercept is
    driven toward ``-inf`` -- converge; a pure ``|objective|`` relative measure
    would report a fixed fractional decrease forever and never converge.

    Parameters
    ----------
    prev_obj, obj : NDArray[np.float64], shape (n_units,)
        Per-unit objective before / after the accepted step.

    Returns
    -------
    float
    """
    rel = (prev_obj - obj) / (1.0 + np.abs(obj))
    return float(np.max(rel))


def _penalized_hessian(
    B: NDArray[np.float64],
    mu: NDArray[np.float64],
    penalty_diag: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Batched penalized Hessian ``(n_units, r, r)``.

    ``H[k] = B^T diag(mu[:, k]) B + diag(penalty_diag) + _HESSIAN_JITTER I``.
    The einsum index order ``"ir,ik,is->krs"`` (``i`` = bins, ``k`` = unit,
    ``r`` / ``s`` = basis modes) yields the ``(n_units, r, r)`` orientation the
    per-unit diagonal add and batched solve require; the transposed
    ``"ik,ij,il->kjl"`` would give ``(r, n_units, r)`` and silently break both.

    Parameters
    ----------
    B : NDArray[np.float64], shape (n_bins, r)
    mu : NDArray[np.float64], shape (n_bins, n_units)
        Expected counts ``occupancy * exp(eta)``.
    penalty_diag : NDArray[np.float64], shape (r,)

    Returns
    -------
    NDArray[np.float64], shape (n_units, r, r)
    """
    r = B.shape[1]
    H = np.einsum("ir,ik,is->krs", B, mu, B, optimize=True)  # (n_units, r, r)
    H = H + (penalty_diag + _HESSIAN_JITTER) * np.eye(r)[None]
    return np.asarray(H, dtype=np.float64)


def _step_halve(
    coeffs: NDArray[np.float64],
    newton_step: NDArray[np.float64],
    B: NDArray[np.float64],
    counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    penalty_diag: NDArray[np.float64],
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], bool]:
    """Per-unit backtracking line search along ``newton_step``.

    Halves, per unit, until the step neither raises that unit's penalized
    objective nor leaves the feasible box (``|B @ coeffs| <= _ETA_CLIP``), or
    ``_MAX_STEP_HALVINGS`` halvings are spent; then returns the **accepted**
    (possibly halved) step -- so the caller reports the true convergence
    diagnostic, not the raw Newton step.

    **The clip is a predictor constraint, not a masked gradient.** Enforcing
    ``|B @ coeffs| <= _ETA_CLIP`` per accepted step keeps the fit in the region
    where ``exp(clip(eta)) == exp(eta)`` and the standard Poisson gradient is
    exact, so an over-shooting bin is halved back rather than pushed into a flat
    region it could never leave (a zeroed clip-Jacobian gradient would freeze
    it there). The current ``coeffs`` are feasible by invariant, so halving
    toward zero always recovers feasibility.

    A unit whose best trial still rises **above the previous objective by more
    than the convergence tolerance** (a genuine ascent), is **non-finite** (a
    diverged / NaN solve), or is **still infeasible** after all halvings has a
    *failed* line search: that unit is reverted to its previous coefficients and
    ``line_search_ok`` is returned ``False`` so the caller does not mistake the
    resulting non-positive objective change for convergence. A trial that only
    rises within the tolerance is float noise at a stationary point and is kept.

    Parameters
    ----------
    coeffs : NDArray[np.float64], shape (r, n_units)
        Current coefficients; feasible (``|B @ coeffs| <= _ETA_CLIP``) by invariant.
    newton_step : NDArray[np.float64], shape (r, n_units)
    B : NDArray[np.float64], shape (n_bins, r)
    counts : NDArray[np.float64], shape (n_bins, n_units)
    occupancy : NDArray[np.float64], shape (n_bins,)
    penalty_diag : NDArray[np.float64], shape (r,)
    tol : float
        Relative slack (``obj > prev + tol * (1 + |prev|)`` is a genuine ascent).

    Returns
    -------
    new_coeffs : NDArray[np.float64], shape (r, n_units)
    accepted_step : NDArray[np.float64], shape (r, n_units)
    obj : NDArray[np.float64], shape (n_units,)
    line_search_ok : bool
        ``False`` if any unit's line search failed (reverted). Batch scalar.
    """

    def _infeasible(c: NDArray[np.float64]) -> NDArray[np.bool_]:
        # Strict: |eta| must stay below the safety limit. _ETA_CLIP is a numerical
        # overflow guard, NOT a valid statistical boundary; a fit the data pushes
        # against it is out of domain (rate > exp(_ETA_CLIP)), and if no feasible
        # descending step remains the caller reports a line-search failure
        # (converged=False) -- it does not pretend to converge at the limit.
        return np.asarray((np.abs(B @ c) >= _ETA_CLIP).any(axis=0), dtype=np.bool_)

    prev = _penalized_obj(coeffs, B, counts, occupancy, penalty_diag)  # (n_units,)
    alpha = np.ones(coeffs.shape[1])
    accepted_step = newton_step
    new_coeffs = coeffs + accepted_step
    obj = _penalized_obj(new_coeffs, B, counts, occupancy, penalty_diag)
    for _ in range(_MAX_STEP_HALVINGS):
        # Halve away from an increase, a non-finite trial (NaN > prev is False,
        # so nonfinite must be flagged explicitly), OR a trial that left the box.
        reject = (obj > prev) | ~np.isfinite(obj) | _infeasible(new_coeffs)
        if not np.any(reject):
            break
        alpha = np.where(reject, 0.5 * alpha, alpha)
        accepted_step = alpha[None, :] * newton_step
        new_coeffs = coeffs + accepted_step
        obj = _penalized_obj(new_coeffs, B, counts, occupancy, penalty_diag)
    # Revert any unit whose best trial is a genuine ascent (beyond float noise),
    # non-finite, or still outside the box; flag the failure.
    failed = (
        (obj > prev + tol * (1.0 + np.abs(prev)))
        | ~np.isfinite(obj)
        | _infeasible(new_coeffs)
    )
    if np.any(failed):
        accepted_step = np.where(failed[None, :], 0.0, accepted_step)
        new_coeffs = coeffs + accepted_step
        obj = np.where(failed, prev, obj)
    return new_coeffs, accepted_step, obj, not bool(np.any(failed))


# ---------------------------------------------------------------------------
# Batched Newton/IRLS fit
# ---------------------------------------------------------------------------
def _newton_fit_numpy(
    counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    B: NDArray[np.float64],
    penalty_diag: NDArray[np.float64],
    constant_base: NDArray[np.float64],
    max_iter: int,
    tol: float,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int, float, bool
]:
    """Batched penalized-Poisson Newton/IRLS fit (**all args positional**).

    Positional-only so REML's ``minimize_scalar(args=...)`` can supply
    ``max_iter`` / ``tol`` -- keyword-only controls would raise ``TypeError``
    through the positional ``args=``. Warm-started from a constant log-rate
    represented exactly by ``constant_base``; ``eta`` is clipped before
    ``exp``; each iteration step-halves for monotone descent. Convergence is the **relative
    penalized-objective decrease** (batch-scalar), never the coefficient step.
    The final ``(eta, mu)`` is recomputed from the **updated** coefficients so
    the returned triple is consistent (never the pre-update arrays).

    Parameters
    ----------
    counts : NDArray[np.float64], shape (n_bins, n_units)
    occupancy : NDArray[np.float64], shape (n_bins,)
    B : NDArray[np.float64], shape (n_bins, r)
    penalty_diag : NDArray[np.float64], shape (r,)
        ``penalty * d``.
    constant_base : NDArray[np.float64], shape (r,)
        Exact coefficients for ``B @ constant_base == 1``, constructed from the
        structural component intercepts and shared across every fit.
    max_iter : int
    tol : float
        Relative penalized-objective decrease threshold.

    Returns
    -------
    coeffs : NDArray[np.float64], shape (r, n_units)
    eta : NDArray[np.float64], shape (n_bins, n_units)
        ``clip(B @ coeffs)`` from the final coefficients.
    mu : NDArray[np.float64], shape (n_bins, n_units)
        ``occupancy * exp(eta)`` from the final ``eta``.
    n_iter : int
        Batch-level iteration count.
    max_step : float
        Max absolute **accepted** coefficient step at the last iteration.
    converged : bool
        Batch-level convergence flag.
    """
    total_occ = float(occupancy.sum())
    # Warm start a constant log-rate per unit.
    rate0 = np.clip(
        counts.sum(0) / (total_occ if total_occ > 1e-9 else 1e-9),
        1e-6,
        np.exp(_ETA_CLIP),
    )  # (n_units,)
    coeffs = _constant_warm_start(B, constant_base, np.log(rate0))
    # Scale each unit's warm start strictly inside the box: rate0 is clipped at
    # exp(_ETA_CLIP), so log(rate0) can equal _ETA_CLIP exactly, putting eta0 on
    # (or a few ulp past) the boundary; the line search requires a strictly
    # feasible start (|eta| < _ETA_CLIP) to halve back toward.
    eta0 = B @ coeffs
    max_abs = np.max(np.abs(eta0), axis=0)  # (n_units,)
    limit = _ETA_CLIP - 1e-6
    scale = np.minimum(1.0, limit / np.maximum(max_abs, limit))  # (n_units,)
    coeffs = coeffs * scale[None, :]
    prev_obj = _penalized_obj(coeffs, B, counts, occupancy, penalty_diag)
    converged = False
    max_step = 0.0
    n_iter = 0
    for _ in range(max_iter):
        n_iter += 1
        eta = np.clip(B @ coeffs, -_ETA_CLIP, _ETA_CLIP)
        mu = occupancy[:, None] * np.exp(eta)
        # Standard Poisson score / Fisher info. They are exact inside the box
        # (where the line search keeps every accepted iterate, so eta is
        # unclipped there); near the clip the residual (counts - mu) provides the
        # inward-recovery direction -- masking it would freeze a saturated bin.
        grad = B.T @ (counts - mu) - penalty_diag[:, None] * coeffs  # (r, n_units)
        H = _penalized_hessian(B, mu, penalty_diag)  # (n_units, r, r)
        newton_step = np.linalg.solve(H, grad.T[..., None])[..., 0].T  # (r, n_units)
        coeffs, accepted_step, obj, line_search_ok = _step_halve(
            coeffs, newton_step, B, counts, occupancy, penalty_diag, tol
        )
        max_step = float(np.max(np.abs(accepted_step)))
        if not line_search_ok:
            # The line search could not descend (ascent direction / diverged /
            # NaN solve): stop with converged=False rather than read the reverted
            # step's non-positive objective change as convergence.
            converged = False
            break
        if _rel_decrease(prev_obj, obj) < tol:
            converged = True
            break
        prev_obj = obj
    # Recompute the final (eta, mu) from the UPDATED coefficients so REML /
    # deviance see a consistent (coeffs, eta, mu) triple, not a stale array.
    eta = np.clip(B @ coeffs, -_ETA_CLIP, _ETA_CLIP)
    mu = occupancy[:, None] * np.exp(eta)
    # Out-of-domain guard. _ETA_CLIP is a numerical safety limit, not a valid
    # statistical boundary: if any bin's empirical rate (counts / occupancy)
    # exceeds exp(_ETA_CLIP), the model cannot represent it -- the predictor is
    # pinned at the limit and a free bin sharing the basis can be throttled far
    # from its optimum (the objective plateaus below tol without an interior
    # optimum). Report converged=False regardless of the objective plateau; this
    # is the honest result for out-of-domain data (a rate above ~exp(30) is
    # physically impossible for neural firing) and, unlike a per-bin clip test,
    # it does not fire for the lower clip (a zero-spike unit whose rate floors is
    # a legitimate convergence).
    if np.any(counts > occupancy[:, None] * np.exp(_ETA_CLIP)):
        converged = False
    return coeffs, eta, mu, n_iter, max_step, converged


# ---------------------------------------------------------------------------
# REML lambda selection
# ---------------------------------------------------------------------------
def _batched_chol_logdet(
    B: NDArray[np.float64],
    mu: NDArray[np.float64],
    penalty_diag: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-unit ``log|H_k|`` via Cholesky; a non-PD ``H_k`` -> ``+inf`` (so the
    REML objective rejects that ``lambda``).

    Parameters
    ----------
    B : NDArray[np.float64], shape (n_bins, r)
    mu : NDArray[np.float64], shape (n_bins, n_units)
    penalty_diag : NDArray[np.float64], shape (r,)

    Returns
    -------
    NDArray[np.float64], shape (n_units,)
    """
    H = _penalized_hessian(B, mu, penalty_diag)  # (n_units, r, r)
    logdet = np.empty(H.shape[0])
    for k in range(H.shape[0]):
        try:
            chol = scipy.linalg.cholesky(H[k], lower=True)
        except np.linalg.LinAlgError:
            logdet[k] = np.inf
        else:
            logdet[k] = 2.0 * np.sum(np.log(np.diag(chol)))
    return logdet


def _reml_objective_numpy(
    log_penalty: float,
    counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    B: NDArray[np.float64],
    d: NDArray[np.float64],
    penalty_rank: int,
    constant_base: NDArray[np.float64],
    max_iter: int,
    tol: float,
) -> float:
    """Pooled REML objective at ``log_penalty`` (**all args positional**).

    Minimized over ``log lambda`` by :func:`select_penalty_by_reml`. Per unit:
    ``-loglik + 0.5*lambda*sum(d*gamma^2) - 0.5*penalty_rank*log(lambda) +
    0.5*log|H|``, then summed over units. Summing carries the penalty-rank df
    term (``-0.5*penalty_rank*log(lambda)``) an ``n_units`` number of times, so
    the objective (and its argmin) is pooled-invariant: duplicating the
    population scales the objective by the unit count and leaves ``lambda``
    unchanged. A non-PD ``H_k`` anywhere, or an inner fit that did not converge,
    -> ``+inf`` (reject this ``lambda``, so the search never selects it from a
    partial / diverged fit).

    All arguments are positional so ``scipy.optimize.minimize_scalar`` can pass
    the extras through its positional ``args=`` tuple.

    Parameters
    ----------
    log_penalty : float
        ``log lambda``.
    counts : NDArray[np.float64], shape (n_bins, n_units)
    occupancy : NDArray[np.float64], shape (n_bins,)
    B : NDArray[np.float64], shape (n_bins, r)
    d : NDArray[np.float64], shape (r,)
        Penalty weights (``0`` on intercept columns, ``> 0`` on fills).
    penalty_rank : int
        ``r_eff - n_live_components``.
    constant_base : NDArray[np.float64], shape (r,)
        Exact all-ones predictor coefficients reused by the inner fit.
    max_iter : int
    tol : float

    Returns
    -------
    float
        Pooled REML objective, or ``+inf`` if the inner fit did not converge or
        the Hessian is non-PD.
    """
    penalty = float(np.exp(log_penalty))
    coeffs, eta, mu, _n_iter, _max_step, converged = _newton_fit_numpy(
        counts, occupancy, B, penalty * d, constant_base, max_iter, tol
    )
    # A non-converged inner fit (line-search failure or iteration cap) gives an
    # unreliable score computed from partial coefficients; reject this lambda so
    # the search never selects it. Gating on convergence is safe here because the
    # fit reports failure exactly (unlike the original ported non_local_detector
    # reference, whose clamped converged flag was not a reliable failure signal;
    # note the neurospatial float32 mirror in _glm_jax.py DOES report failure
    # faithfully and its REML gates on converged too).
    if not converged:
        return float(np.inf)
    # Laplace-approx Hessian is the fit's Fisher information (standard Poisson);
    # the converged fit is feasible so ``eta`` is unclipped there and
    # ``counts * eta`` is the loglik at the constrained MLE.
    logdet = _batched_chol_logdet(B, mu, penalty * d)  # (n_units,)
    if not np.all(np.isfinite(logdet)):
        return float(np.inf)
    loglik = np.sum(counts * eta - mu, axis=0)  # (n_units,)
    pen = 0.5 * penalty * np.sum(d[:, None] * coeffs**2, axis=0)  # (n_units,)
    reml = -loglik + pen - 0.5 * penalty_rank * log_penalty + 0.5 * logdet
    return float(np.sum(reml))


def select_penalty_by_reml(
    counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    B: NDArray[np.float64],
    d: NDArray[np.float64],
    penalty_rank: int,
    *,
    constant_base: NDArray[np.float64] | None = None,
    max_iter: int,
    tol: float,
) -> tuple[float | None, float | None, bool | None]:
    """Select ``lambda`` by minimizing the pooled REML objective over
    ``log lambda`` with a bounded scalar minimizer.

    At ``penalty_rank == 0`` (an all-null basis) the objective is flat in
    ``lambda`` -- REML is **skipped** and ``(None, None, None)`` is returned.
    Raises ``ValueError`` if no finite objective exists across the interval (no
    ``lambda`` yields a converged inner fit with a PD Hessian).

    Parameters
    ----------
    counts : NDArray[np.float64], shape (n_bins, n_units)
    occupancy : NDArray[np.float64], shape (n_bins,)
    B : NDArray[np.float64], shape (n_bins, r)
    d : NDArray[np.float64], shape (r,)
    penalty_rank : int
    constant_base : NDArray[np.float64] or None, keyword-only
        Precomputed structural all-ones coefficients. Direct internal callers
        may omit it; the selector derives it once from ``d`` and
        ``penalty_rank``.
    max_iter : int, keyword-only
    tol : float, keyword-only

    Returns
    -------
    penalty : float or None
        Selected ``lambda`` (``None`` when REML is skipped at ``penalty_rank == 0``).
    reml_objective : float or None
        Minimized objective (``None`` when skipped).
    at_boundary : bool or None
        Whether the selected ``log(lambda)`` is within ``_REML_BOUNDARY_TOL`` of
        either fixed search bound (``lambda`` weakly identified); ``None`` when
        REML is skipped.
    """
    if penalty_rank == 0:  # flat in lambda -- skip
        return None, None, None
    if constant_base is None:
        # Direct internal callers still use the exact structural start; the
        # orchestrator normally supplies this precomputed vector so both
        # backends share it.
        from ._glm import _structural_constant_base

        constant_base = _structural_constant_base(B, d.size - penalty_rank)
    result = scipy.optimize.minimize_scalar(
        _reml_objective_numpy,
        bounds=_LOG_PENALTY_BOUNDS,
        method="bounded",
        args=(
            counts,
            occupancy,
            B,
            d,
            penalty_rank,
            constant_base,
            max_iter,
            tol,
        ),
        options={"xatol": _REML_XATOL},
    )
    if not np.isfinite(result.fun):
        raise ValueError(
            "REML found no finite objective in the log-penalty interval "
            f"{_LOG_PENALTY_BOUNDS}: no lambda yielded a converged inner fit with "
            "a positive-definite Hessian. Reduce the basis rank, improve "
            "occupancy coverage, or supply a fixed penalty."
        )
    lower, upper = _LOG_PENALTY_BOUNDS
    at_boundary = bool(min(result.x - lower, upper - result.x) <= _REML_BOUNDARY_TOL)
    return float(np.exp(result.x)), float(result.fun), at_boundary


# ---------------------------------------------------------------------------
# Deviance
# ---------------------------------------------------------------------------
def _poisson_deviance(
    counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    firing_rate: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Unpenalized per-unit Poisson deviance from the STORED floored rate.

    ``2 * sum_exposed[ n*log(n/mu) - (n - mu) ]`` with ``mu = occupancy *
    firing_rate``, ``0*log 0 == 0``, summed over exposed (``occupancy > 0``)
    bins. Computed from the floored rate ``max(exp(eta), _RATE_FLOOR)`` so the
    deviance describes exactly the field that is reported.

    Parameters
    ----------
    counts : NDArray[np.float64], shape (n_bins, n_units)
    occupancy : NDArray[np.float64], shape (n_bins,)
    firing_rate : NDArray[np.float64], shape (n_bins, n_units)
        The floored rate that will be reported.

    Returns
    -------
    NDArray[np.float64], shape (n_units,)
    """
    mu = occupancy[:, None] * firing_rate  # (n_bins, n_units)
    exposed = occupancy > 0
    n = counts[exposed].astype(np.float64)  # (n_exposed, n_units)
    m = mu[exposed]
    term = np.where(n > 0, n * np.log(np.where(n > 0, n, 1.0) / m), 0.0) - (n - m)
    return np.asarray(2.0 * term.sum(0), dtype=np.float64)
