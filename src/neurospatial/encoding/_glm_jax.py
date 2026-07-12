"""Float32 JAX mirror of the penalized-Poisson GAM fit + REML.

An optional acceleration path for ``method="glm"``: a faithful float32 mirror of
the float64 NumPy/SciPy core (:mod:`neurospatial.encoding._glm_numpy`), batched
over the neuron axis and JIT-compiled (GPU-ready). The NumPy core is the
correctness reference; this module reproduces its numerics -- **including the box
constraint** ``|B @ coeffs| <= _ETA_CLIP`` (a predictor constraint, not a masked
gradient), the line-search feasibility revert, and the out-of-domain rate guard
-- at ``_FIT_DTYPE = jnp.float32``. Where the two disagree, the NumPy core wins.

Two float32-specific deltas from the float64 core: the relative-objective
convergence tolerance is floored at ``_FIT_TOL_FLOOR`` (``1e-10`` is below float32
objective noise ``~1e-7``), and the step-halving "objective increased" test uses
the ``_DESCENT_TOL`` slack.

Inputs are cast to float32 at entry and every returned array back to float64 at
the boundary, so the ``MRFFit`` the orchestrator assembles stays NumPy float64 --
this module only accelerates the fit compute; the public return contract
(dtype / array type) is owned by the ``compute_spatial_rate(s)`` layer.

Selected by :func:`neurospatial.encoding._glm.fit_mrf_gam` when
``backend="jax"`` and JAX is installed; the base install runs the NumPy core.
Requires JAX (Linux/macOS only).
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import numpy as np
import scipy.optimize
from numpy.typing import NDArray

# Enable JAX float64 mode globally (the standard pattern for float64-capable JAX
# libraries; idempotent, no effect on processes that never import this module).
# The MRF-GAM fit itself runs in float32 (``_FIT_DTYPE``) via explicit casts, so
# this toggle only governs the *default* dtype of un-annotated literals -- it does
# not upcast the float32 fit. It matches ``_core_jax``'s placement so both modules
# agree on global config.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  -- must follow the x64 toggle above
from jax import lax  # noqa: E402
from jax.scipy.linalg import cho_solve  # noqa: E402

from ._glm import (  # noqa: E402
    _DESCENT_TOL,
    _ETA_CLIP,
    _FIT_TOL_FLOOR,
    _HESSIAN_JITTER,
    _LOG_PENALTY_BOUNDS,
    _MAX_STEP_HALVINGS,
    _REML_XATOL,
)

if TYPE_CHECKING:
    from jax import Array

# float32 fit dtype. The NumPy core is float64.
_FIT_DTYPE = jnp.float32


# ---------------------------------------------------------------------------
# Objective + Hessian (mirror _glm_numpy._penalized_obj / _penalized_hessian)
# ---------------------------------------------------------------------------
def _penalized_obj_jax(
    coeffs: Array,
    basis: Array,
    counts: Array,
    occupancy: Array,
    penalty_diag: Array,
) -> Array:
    """Per-unit negative penalized Poisson log-likelihood with the **clipped**
    predictor in BOTH the linear ``n*eta`` term and ``mu`` (so the value stays
    bounded for an out-of-box trial the line search is about to reject).

    Mirrors :func:`neurospatial.encoding._glm_numpy._penalized_obj`.
    """
    eta = jnp.clip(basis @ coeffs, -_ETA_CLIP, _ETA_CLIP)
    mu = occupancy[:, None] * jnp.exp(eta)
    neg_loglik = -jnp.sum(counts * eta - mu, axis=0)
    pen = 0.5 * jnp.sum(penalty_diag[:, None] * coeffs**2, axis=0)
    return neg_loglik + pen


def _penalized_hessian_jax(basis: Array, mu: Array, penalty_diag: Array) -> Array:
    """Batched penalized Hessian ``(n_units, r, r)``:
    ``B^T diag(mu[:, k]) B + diag(penalty_diag) + _HESSIAN_JITTER I``.

    Mirrors :func:`neurospatial.encoding._glm_numpy._penalized_hessian` (same
    ``"ir,ik,is->krs"`` index order -> ``(n_units, r, r)``).
    """
    r = basis.shape[1]
    hessian = jnp.einsum("ir,ik,is->krs", basis, mu, basis)
    hessian = hessian + (penalty_diag + _HESSIAN_JITTER) * jnp.eye(r, dtype=basis.dtype)
    return hessian


# ---------------------------------------------------------------------------
# Step-halving line search (mirror _glm_numpy._step_halve)
# ---------------------------------------------------------------------------
def _step_halve_jax(
    coeffs: Array,
    newton_step: Array,
    basis: Array,
    counts: Array,
    occupancy: Array,
    penalty_diag: Array,
) -> tuple[Array, Array, Array, Array]:
    """Per-unit backtracking line search along ``newton_step``.

    Halves, per unit, until the step neither raises that unit's penalized
    objective **beyond the float32 descent slack** ``_DESCENT_TOL`` nor leaves
    the feasible box (``|B @ coeffs| <= _ETA_CLIP``), or ``_MAX_STEP_HALVINGS``
    halvings are spent; then reverts any unit whose best trial is still a genuine
    ascent (beyond ``_DESCENT_TOL``), non-finite, or infeasible, and reports
    ``line_search_ok=False`` so the caller does not read the reverted step as
    convergence.

    Mirrors :func:`neurospatial.encoding._glm_numpy._step_halve` with the one
    documented float32 delta: the "objective increased" test uses the
    ``_DESCENT_TOL`` slack (``1e-5``, above the float32 objective noise
    ``~1e-7``) rather than the float64 core's exact/``tol`` test, so a
    within-noise trial is **accepted at the full Newton step** instead of being
    halved to zero. The ``~(obj <= prev + slack)`` idiom also flags NaN/inf.

    **The clip is a predictor constraint, not a masked gradient** -- an
    over-shooting bin is halved back rather than pushed into a flat region.
    Returns the accepted (possibly halved / reverted) step so the caller reports
    the true diagnostic.
    """
    n_units = coeffs.shape[1]

    def infeasible(candidate: Array) -> Array:
        # Strict: |eta| must stay below the safety limit. _ETA_CLIP is a
        # numerical overflow guard, not a valid statistical boundary.
        return (jnp.abs(basis @ candidate) >= _ETA_CLIP).any(axis=0)

    prev = _penalized_obj_jax(coeffs, basis, counts, occupancy, penalty_diag)

    def trial(alpha: Array) -> tuple[Array, Array]:
        candidate = coeffs + alpha[None, :] * newton_step
        obj = _penalized_obj_jax(candidate, basis, counts, occupancy, penalty_diag)
        return candidate, obj

    def worse(candidate: Array, obj: Array) -> Array:
        # A trial is "worse" if it raises the objective beyond the _DESCENT_TOL
        # float32 slack; ~(obj <= ...) also flags NaN/inf (NaN <= x is False).
        # Plus the box: an infeasible trial is always rejected.
        ascent = jnp.logical_not(obj <= prev + _DESCENT_TOL * (1.0 + jnp.abs(prev)))
        return ascent | infeasible(candidate)

    def cond(state: tuple[Array, Array, Array, Array]) -> Array:
        _alpha, candidate, obj, halvings = state
        return (halvings < _MAX_STEP_HALVINGS) & jnp.any(worse(candidate, obj))

    def body(
        state: tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        alpha, candidate, obj, halvings = state
        reject = worse(candidate, obj)
        alpha = jnp.where(reject, 0.5 * alpha, alpha)
        candidate, obj = trial(alpha)
        return alpha, candidate, obj, halvings + 1

    alpha0 = jnp.ones(n_units, coeffs.dtype)
    candidate0, obj0 = trial(alpha0)
    alpha, candidate, obj, _ = lax.while_loop(
        cond, body, (alpha0, candidate0, obj0, jnp.array(0))
    )

    accepted_step = alpha[None, :] * newton_step
    # Revert any unit whose best trial is still worse (genuine ascent beyond the
    # _DESCENT_TOL slack, non-finite, or infeasible); flag the failure.
    failed = worse(candidate, obj)
    accepted_step = jnp.where(failed[None, :], 0.0, accepted_step)
    new_coeffs = coeffs + accepted_step
    obj = jnp.where(failed, prev, obj)
    line_search_ok = ~jnp.any(failed)
    return new_coeffs, accepted_step, obj, line_search_ok


# ---------------------------------------------------------------------------
# Batched Newton/IRLS loop (mirror _glm_numpy._newton_fit_numpy)
# ---------------------------------------------------------------------------
@jax.jit
def _warm_start_jax(
    counts: Array, occupancy: Array, basis: Array, constant_base: Array
) -> Array:
    """Constant-log-rate warm start ``(r, n_units)``, scaled strictly inside the
    box (mirrors the NumPy core's warm start).

    Independent of the penalty, so it is computed **once** and reused across a
    REML lambda search. ``constant_base`` comes from the exact structural
    component intercepts, so no least-squares SVD appears in this kernel.
    Mirrors :func:`_glm_numpy._constant_warm_start` + the box scaling in
    :func:`_glm_numpy._newton_fit_numpy`.
    """
    dtype = basis.dtype
    total_occ = jnp.maximum(occupancy.sum(), jnp.asarray(1e-9, dtype))
    rate0 = jnp.clip(
        counts.sum(0) / total_occ, 1e-6, jnp.exp(jnp.asarray(_ETA_CLIP, dtype))
    )
    constant_base = jnp.asarray(constant_base, dtype)
    coeffs = constant_base[:, None] * jnp.log(rate0)[None, :]
    # Keep a clamped constant strictly inside the box so the line search can
    # always halve back toward a feasible start.
    max_abs = jnp.max(jnp.abs(basis @ coeffs), axis=0)  # (n_units,)
    limit = _ETA_CLIP - 1e-6
    scale = jnp.minimum(1.0, limit / jnp.maximum(max_abs, limit))
    return coeffs * scale[None, :]


@partial(jax.jit, static_argnums=(4,))
def _newton_loop_jax(
    counts: Array,
    occupancy: Array,
    basis: Array,
    penalty_diag: Array,
    max_iter: int,
    tol: Array,
    warm_coeffs: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """JIT-compiled float32 Newton/IRLS loop (``max_iter`` static).

    The raw loop -- ``tol`` is applied as given (**no floor**); the floor lives
    in :func:`_newton_fit_jax`. Started from ``warm_coeffs`` (a penalty-independent
    constant-log-rate warm start from :func:`_warm_start_jax`, passed in so a REML
    search computes it once); convergence is the **relative penalized-objective
    decrease** (batch-scalar), gated on a successful line search; the final
    ``(eta, mu)`` is recomputed from the updated coefficients. Mirrors
    :func:`neurospatial.encoding._glm_numpy._newton_fit_numpy`.

    Returns ``(coeffs, eta, mu, n_iter, max_step, converged)`` (all float32 /
    JAX scalars).
    """
    dtype = basis.dtype
    coeffs = warm_coeffs
    prev_obj = _penalized_obj_jax(coeffs, basis, counts, occupancy, penalty_diag)

    def cond(state):
        _coeffs, _prev, iteration, _max_step, _converged, done = state
        return (iteration < max_iter) & jnp.logical_not(done)

    def body(state):
        coeffs, prev_obj, iteration, _max_step, _converged, _done = state
        eta = jnp.clip(basis @ coeffs, -_ETA_CLIP, _ETA_CLIP)
        mu = occupancy[:, None] * jnp.exp(eta)
        # Standard Poisson score / Fisher info -- exact inside the box, where the
        # line search keeps every accepted iterate.
        grad = basis.T @ (counts - mu) - penalty_diag[:, None] * coeffs
        hessian = _penalized_hessian_jax(basis, mu, penalty_diag)
        # Cholesky solve (float32 delta from the NumPy core's LU ``np.linalg.solve``):
        # the penalized Hessian is SPD after ``penalty * d`` + jitter, so a Cholesky
        # + triangular solves is ~2x the LU solve and numerically preferable. A
        # non-PD Hessian (should not occur -- rank-deficient designs fall back to
        # the float64 core) gives a NaN factor, which the step-halving ``worse``
        # guard flags -> ``converged=False``; never a silent wrong result.
        chol = jnp.linalg.cholesky(hessian)
        newton_step = cho_solve((chol, True), grad.T[..., None])[..., 0].T
        new_coeffs, accepted_step, obj, line_search_ok = _step_halve_jax(
            coeffs, newton_step, basis, counts, occupancy, penalty_diag
        )
        max_step = jnp.max(jnp.abs(accepted_step))
        rel_decrease = jnp.max((prev_obj - obj) / (1.0 + jnp.abs(obj)))
        # Mirror the NumPy core: a failed line search stops with converged=False;
        # otherwise convergence is rel_decrease < tol.
        converged = line_search_ok & (rel_decrease < tol)
        done = jnp.logical_not(line_search_ok) | (rel_decrease < tol)
        return new_coeffs, obj, iteration + 1, max_step, converged, done

    init = (
        coeffs,
        prev_obj,
        jnp.array(0),
        jnp.asarray(0.0, dtype),
        jnp.asarray(False),
        jnp.asarray(False),
    )
    coeffs, _prev_obj, n_iter, max_step, converged, _done = lax.while_loop(
        cond, body, init
    )

    # Recompute (eta, mu) from the UPDATED coefficients so the returned triple is
    # consistent (never the pre-update arrays).
    eta = jnp.clip(basis @ coeffs, -_ETA_CLIP, _ETA_CLIP)
    mu = occupancy[:, None] * jnp.exp(eta)
    # Out-of-domain guard: _ETA_CLIP is a numerical limit, not a statistical
    # boundary. If any bin's empirical rate exceeds exp(_ETA_CLIP), the model
    # cannot represent it -- report converged=False regardless of the plateau.
    out_of_domain = jnp.any(
        counts > occupancy[:, None] * jnp.exp(jnp.asarray(_ETA_CLIP, dtype))
    )
    converged = converged & jnp.logical_not(out_of_domain)
    return coeffs, eta, mu, n_iter, max_step, converged


def _newton_fit_jax(
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
    """Float32 JAX mirror of :func:`_glm_numpy._newton_fit_numpy` (same signature).

    Casts inputs and the structural ``constant_base`` warm-start direction to
    float32, applies the float32 convergence-tolerance floor
    ``tol = max(tol, _FIT_TOL_FLOOR)``, runs the JIT-compiled loop, and casts
    every returned array back to float64 -- so the caller sees the same
    ``(coeffs, eta, mu, n_iter, max_step, converged)`` contract as the NumPy core
    with **NumPy float64 arrays**. The ``B`` name mirrors ``_newton_fit_numpy``
    so the two are drop-in interchangeable at the ``fit_mrf_gam`` dispatch.
    """
    tol_eff = max(float(tol), _FIT_TOL_FLOOR)
    counts_j = jnp.asarray(counts, _FIT_DTYPE)
    occ_j = jnp.asarray(occupancy, _FIT_DTYPE)
    basis_j = jnp.asarray(B, _FIT_DTYPE)
    warm = _warm_start_jax(
        counts_j, occ_j, basis_j, jnp.asarray(constant_base, _FIT_DTYPE)
    )
    coeffs, eta, mu, n_iter, max_step, converged = _newton_loop_jax(
        counts_j,
        occ_j,
        basis_j,
        jnp.asarray(penalty_diag, _FIT_DTYPE),
        int(max_iter),
        jnp.asarray(tol_eff, _FIT_DTYPE),
        warm,
    )
    return (
        np.asarray(coeffs, dtype=np.float64),
        np.asarray(eta, dtype=np.float64),
        np.asarray(mu, dtype=np.float64),
        int(n_iter),
        float(max_step),
        bool(converged),
    )


# ---------------------------------------------------------------------------
# REML lambda selection (mirror _glm_numpy._reml_objective_numpy / select_*)
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(5, 6))
def _reml_score_jax(
    log_penalty: Array,
    counts: Array,
    occupancy: Array,
    basis: Array,
    penalty_weights: Array,
    penalty_rank: int,
    max_iter: int,
    tol: Array,
    warm_coeffs: Array,
) -> Array:
    """Pooled REML objective at ``log_penalty`` (float32, on-device scalar).

    ``-loglik + 0.5*lambda*sum(d*gamma^2) - 0.5*penalty_rank*log(lambda) +
    0.5*log|H|`` summed over units, with ``+inf`` for any ``lambda`` whose inner
    fit did not converge or whose per-unit Hessian is not positive-definite (a
    NaN Cholesky), so the search never selects it. Mirrors
    :func:`neurospatial.encoding._glm_numpy._reml_objective_numpy`; the JAX fit's
    ``converged`` flag is a faithful mirror of the NumPy core's, so gating on it
    here matches the NumPy REML. ``warm_coeffs`` (penalty-independent) is passed in
    so the search computes the warm start once rather than per lambda.
    """
    penalty = jnp.exp(log_penalty)
    penalty_diag = penalty * penalty_weights
    coeffs, eta, mu, _n_iter, _max_step, converged = _newton_loop_jax(
        counts, occupancy, basis, penalty_diag, max_iter, tol, warm_coeffs
    )
    loglik = jnp.sum(counts * eta - mu, axis=0)
    pen = 0.5 * penalty * jnp.sum(penalty_weights[:, None] * coeffs**2, axis=0)
    chol = jnp.linalg.cholesky(_penalized_hessian_jax(basis, mu, penalty_diag))
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    reml = -loglik + pen - 0.5 * penalty_rank * log_penalty + 0.5 * logdet
    finite = converged & jnp.all(jnp.isfinite(logdet))
    return jnp.where(finite, jnp.sum(reml), jnp.asarray(jnp.inf, counts.dtype))


def select_penalty_by_reml_jax(
    counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    B: NDArray[np.float64],
    d: NDArray[np.float64],
    penalty_rank: int,
    *,
    constant_base: NDArray[np.float64] | None = None,
    max_iter: int,
    tol: float,
) -> tuple[float | None, float | None]:
    """Float32 JAX mirror of :func:`_glm_numpy.select_penalty_by_reml`.

    Minimizes the pooled float32 REML objective over ``log lambda`` with the same
    bounded scalar minimizer (``scipy.optimize.minimize_scalar``); ``lambda`` is
    returned as a Python float. Skips REML at ``penalty_rank == 0`` and raises the
    same ``ValueError`` when no finite objective exists across the interval. The
    ``B`` name mirrors ``select_penalty_by_reml`` so the two are drop-in
    interchangeable at the ``fit_mrf_gam`` dispatch. ``constant_base`` is
    normally supplied by the orchestrator; direct internal callers may omit it
    and let the selector derive it from the structural penalty rank.
    """
    if penalty_rank == 0:  # flat in lambda -- skip
        return None, None
    if constant_base is None:
        from ._glm import _structural_constant_base

        constant_base = _structural_constant_base(B, d.size - penalty_rank)
    counts_j = jnp.asarray(counts, _FIT_DTYPE)
    occ_j = jnp.asarray(occupancy, _FIT_DTYPE)
    basis_j = jnp.asarray(B, _FIT_DTYPE)
    d_j = jnp.asarray(d, _FIT_DTYPE)
    tol_eff = jnp.asarray(max(float(tol), _FIT_TOL_FLOOR), _FIT_DTYPE)
    # Penalty-independent structural warm start: compute once, not on every one
    # of the ~18 objective evaluations the search makes.
    warm = _warm_start_jax(
        counts_j,
        occ_j,
        basis_j,
        jnp.asarray(constant_base, _FIT_DTYPE),
    )

    def objective(log_penalty: float) -> float:
        value = _reml_score_jax(
            jnp.asarray(log_penalty, _FIT_DTYPE),
            counts_j,
            occ_j,
            basis_j,
            d_j,
            int(penalty_rank),
            int(max_iter),
            tol_eff,
            warm,
        )
        return float(value)

    result = scipy.optimize.minimize_scalar(
        objective,
        bounds=_LOG_PENALTY_BOUNDS,
        method="bounded",
        options={"xatol": _REML_XATOL},
    )
    if not np.isfinite(result.fun):
        raise ValueError(
            "REML found no finite objective in the log-penalty interval "
            f"{_LOG_PENALTY_BOUNDS}: no lambda yielded a converged inner fit with "
            "a positive-definite Hessian. Reduce the basis rank, improve "
            "occupancy coverage, or supply a fixed penalty."
        )
    return float(np.exp(result.x)), float(result.fun)
