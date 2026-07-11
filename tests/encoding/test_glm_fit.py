"""Penalized-Poisson GAM fit + REML lambda selection (the statistical core).

These tests pin the ``fit_mrf_gam`` / ``_newton_fit_numpy`` / REML contract on
simulated Poisson spike counts drawn from known place fields. Every quantity
that could be faked by "a value exists" -- deviance, penalty rank, the REML
objective's pooling factor, the batched-Hessian orientation -- is recomputed
independently, not read back from the fit's own internals.

The fit is bin-major: ``counts`` arrive ``(n_live_bins, n_units)`` already
restricted to ``basis.live_bins`` (the caller owns the unit-major<->bin-major
transpose and the restriction); ``MRFFit.log_rate`` is ``(n_live_bins,
n_units)``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from neurospatial.encoding._glm import (
    _FIT_TOL,
    _LOG_PENALTY_BOUNDS,
    _MAX_ITER,
    _RATE_FLOOR,
    fit_mrf_gam,
)
from neurospatial.encoding._glm_numpy import (
    _newton_fit_numpy,
    _penalized_hessian,
    _reml_objective_numpy,
    _step_halve,
    select_penalty_by_reml,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _restrict(counts, occupancy, basis):
    """Restrict active-bin-order ``counts``/``occupancy`` to ``basis.live_bins``."""
    return counts[basis.live_bins], occupancy[basis.live_bins]


def _firing_rate(fit):
    """The floored rate ``max(exp(log_rate), _RATE_FLOOR)`` the result reports."""
    return np.maximum(np.exp(fit.log_rate), _RATE_FLOOR)


# ---------------------------------------------------------------------------
# Statistical recovery
# ---------------------------------------------------------------------------
def test_recovers_place_fields(open_field_env, simulate_place_fields):
    """A population fit recovers each unit's peak near its simulated center.

    This is the headline: finite, sensible rate maps where a ratio estimator
    would divide by tiny occupancy. Centers sit on exact bin centers of the
    8x8 (2 cm) grid; recovery must land within ~1 bin.
    """
    env = open_field_env
    centers = [(5.0, 5.0), (11.0, 11.0), (5.0, 11.0)]
    counts_full, occ_full = simulate_place_fields(env, centers, seed=1)
    basis = env._mrf_basis(occ_full, rank=30)
    counts, occ = _restrict(counts_full, occ_full, basis)

    fit = fit_mrf_gam(basis, counts, occ, penalty=None)

    fr = _firing_rate(fit)  # (n_live_bins, n_units)
    for u, center in enumerate(centers):
        peak_bin = basis.live_bins[int(np.argmax(fr[:, u]))]
        peak_xy = np.asarray(env.bin_centers[peak_bin])
        assert np.linalg.norm(peak_xy - np.asarray(center)) <= 3.0  # < 1.5 bins


def test_finite_rates_where_ratio_would_nan(open_field_env, simulate_place_fields):
    """The GAM returns all-finite rates even with unvisited (zero-occupancy)
    live bins where a ratio estimator (counts / occupancy) would be NaN."""
    env = open_field_env
    occ = np.full(env.n_bins, 3.0)
    occ[: env.n_bins // 2] = 0.0  # half the arena never visited -> ratio NaNs there
    counts_full, occ_full = simulate_place_fields(
        env, [(8.0, 8.0)], occupancy=occ, seed=2
    )
    basis = env._mrf_basis(occ_full, rank=20)
    counts, occupancy = _restrict(counts_full, occ_full, basis)

    fit = fit_mrf_gam(basis, counts, occupancy, penalty=None)

    assert np.all(np.isfinite(fit.log_rate))
    assert np.all(np.isfinite(_firing_rate(fit)))


# ---------------------------------------------------------------------------
# REML lambda selection
# ---------------------------------------------------------------------------
def test_reml_selects_sensible_lambda(open_field_env, simulate_place_fields):
    """REML lambda is finite and inside exp(_LOG_PENALTY_BOUNDS); increasing a
    fixed lambda monotonically smooths the field (lower spatial variance)."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=3)
    basis = env._mrf_basis(occ_full, rank=25)
    counts, occ = _restrict(counts_full, occ_full, basis)

    fit = fit_mrf_gam(basis, counts, occ, penalty=None)
    lo, hi = np.exp(_LOG_PENALTY_BOUNDS[0]), np.exp(_LOG_PENALTY_BOUNDS[1])
    assert np.isfinite(fit.penalty)
    assert lo <= fit.penalty <= hi

    variances = []
    for lam in [1e-2, 1e0, 1e2, 1e4, 1e6]:
        f = fit_mrf_gam(basis, counts, occ, penalty=lam)
        variances.append(float(np.var(_firing_rate(f)[:, 0])))
    # Monotonically non-increasing: heavier penalty -> smoother -> flatter map.
    assert np.all(np.diff(variances) <= 1e-9)
    assert variances[0] > variances[-1]  # and it actually smooths


def test_reml_pooled_scaling(open_field_env, simulate_place_fields):
    """The REML objective carries the per-unit df factor, so duplicating the
    population scales the objective exactly by the unit count and leaves the
    selected lambda invariant. Omitting the factor would shift lambda."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(
        env, [(6.0, 6.0), (10.0, 10.0)], seed=4
    )
    basis = env._mrf_basis(occ_full, rank=20)
    counts, occ = _restrict(counts_full, occ_full, basis)
    penalty_rank = basis.d.size - basis.n_live_components

    # Exact 2x scaling of the objective under population duplication proves the
    # df term (-0.5 * penalty_rank * log_lambda) is summed PER UNIT: if it were
    # added once, the doubled objective would not be exactly 2x the original.
    counts2 = np.concatenate([counts, counts], axis=1)
    for log_lambda in [-2.0, 0.0, 3.0, 8.0]:
        one = _reml_objective_numpy(
            log_lambda, counts, occ, basis.B, basis.d, penalty_rank, _MAX_ITER, _FIT_TOL
        )
        two = _reml_objective_numpy(
            log_lambda,
            counts2,
            occ,
            basis.B,
            basis.d,
            penalty_rank,
            _MAX_ITER,
            _FIT_TOL,
        )
        np.testing.assert_allclose(two, 2.0 * one, rtol=1e-8)

    # And the argmin (selected lambda) is invariant to duplication.
    lam1, _ = select_penalty_by_reml(
        counts, occ, basis.B, basis.d, penalty_rank, max_iter=_MAX_ITER, tol=_FIT_TOL
    )
    lam2, _ = select_penalty_by_reml(
        counts2, occ, basis.B, basis.d, penalty_rank, max_iter=_MAX_ITER, tol=_FIT_TOL
    )
    np.testing.assert_allclose(lam1, lam2, rtol=1e-2)


def test_reml_objective_callable(open_field_env, simulate_place_fields):
    """select_penalty_by_reml runs end-to-end: minimize_scalar passes the extra
    args positionally with no TypeError, returning a finite (lambda, objective)."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=5)
    basis = env._mrf_basis(occ_full, rank=15)
    counts, occ = _restrict(counts_full, occ_full, basis)
    penalty_rank = basis.d.size - basis.n_live_components

    lam, obj = select_penalty_by_reml(
        counts, occ, basis.B, basis.d, penalty_rank, max_iter=_MAX_ITER, tol=_FIT_TOL
    )
    assert np.isfinite(lam) and np.isfinite(obj)
    assert lam > 0.0


def test_reml_rejects_nonconverged_inner_fit(
    open_field_env, simulate_place_fields, monkeypatch
):
    """A lambda whose inner Newton fit did not converge yields an unreliable
    score: _reml_objective_numpy must return +inf (reject it), never a finite
    score from a partial fit that the optimizer could select."""
    import neurospatial.encoding._glm_numpy as glmnp

    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=25)
    basis = env._mrf_basis(occ_full, rank=12)
    counts, occ = _restrict(counts_full, occ_full, basis)
    penalty_rank = basis.d.size - basis.n_live_components

    real_solve = np.linalg.solve
    # Force the inner line search to fail (ascent solve) -> inner converged=False.
    monkeypatch.setattr(
        glmnp.np.linalg, "solve", lambda H, rhs: -1e6 * real_solve(H, rhs)
    )

    score = _reml_objective_numpy(
        0.0, counts, occ, basis.B, basis.d, penalty_rank, _MAX_ITER, _FIT_TOL
    )
    assert not np.isfinite(score)  # rejected, not a finite selectable score


def test_reml_raises_when_all_inner_fits_fail(
    open_field_env, simulate_place_fields, monkeypatch
):
    """If no lambda in the interval yields a converged inner fit, REML has no
    finite objective and select_penalty_by_reml raises -- rather than returning
    a lambda selected from non-converged fits."""
    import neurospatial.encoding._glm_numpy as glmnp

    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=26)
    basis = env._mrf_basis(occ_full, rank=12)
    counts, occ = _restrict(counts_full, occ_full, basis)
    penalty_rank = basis.d.size - basis.n_live_components

    # Force every inner fit to report non-convergence (so no lambda is selectable).
    real_fit = glmnp._newton_fit_numpy

    def nonconverged_fit(*args):
        coeffs, eta, mu, n_iter, max_step, _conv = real_fit(*args)
        return coeffs, eta, mu, n_iter, max_step, False

    monkeypatch.setattr(glmnp, "_newton_fit_numpy", nonconverged_fit)

    # The error must name the convergence cause, not only "non-PD Hessian".
    with pytest.raises(ValueError, match="converged inner fit"):
        select_penalty_by_reml(
            counts,
            occ,
            basis.B,
            basis.d,
            penalty_rank,
            max_iter=_MAX_ITER,
            tol=_FIT_TOL,
        )


# ---------------------------------------------------------------------------
# r == 0 (all-null basis) -- the decisive REML-skip case
# ---------------------------------------------------------------------------
def test_r0_skips_reml(two_path_env):
    """Two disjoint 3-node paths at rank=2 -> penalty_rank == 0. REML is skipped
    (flat in lambda), penalty and reml_objective are None, rates finite -- NOT
    an arbitrary lambda."""
    env = two_path_env
    occ_full = np.ones(env.n_bins)
    basis = env._mrf_basis(occ_full, rank=2)
    assert basis.d.size - basis.n_live_components == 0  # penalty_rank == 0
    rng = np.random.default_rng(0)
    counts_full = rng.poisson(3.0, size=(env.n_bins, 2)).astype(np.int64)
    counts, occ = _restrict(counts_full, occ_full, basis)

    fit = fit_mrf_gam(basis, counts, occ, penalty=None)

    assert fit.penalty is None
    assert fit.reml_objective is None
    assert np.all(np.isfinite(_firing_rate(fit)))


def test_r0_final_fit_unpenalized(two_path_env):
    """At penalty_rank == 0 the final Newton fit runs with penalty_diag == 0
    (never None) and returns finite rates; MRFFit.penalty stays None."""
    env = two_path_env
    occ_full = np.ones(env.n_bins)
    basis = env._mrf_basis(occ_full, rank=2)
    rng = np.random.default_rng(1)
    counts_full = rng.poisson(5.0, size=(env.n_bins, 3)).astype(np.int64)
    counts, occ = _restrict(counts_full, occ_full, basis)

    # An explicit unpenalized fit (penalty_diag == 0) must equal the fit the
    # orchestrator runs when penalty is None at r == 0.
    coeffs, eta, _mu, _n, _s, conv = _newton_fit_numpy(
        counts, occ, basis.B, np.zeros_like(basis.d), _MAX_ITER, _FIT_TOL
    )
    fit = fit_mrf_gam(basis, counts, occ, penalty=None)

    assert fit.penalty is None
    assert conv
    np.testing.assert_allclose(fit.coefficients, coeffs, atol=1e-8)
    np.testing.assert_allclose(fit.log_rate, eta, atol=1e-8)
    assert np.all(np.isfinite(_firing_rate(fit)))


# ---------------------------------------------------------------------------
# Fixed-penalty contract
# ---------------------------------------------------------------------------
def test_fixed_penalty_recorded(open_field_env, simulate_place_fields):
    """A supplied fixed penalty is echoed on MRFFit (not discarded); REML did
    not run so reml_objective is None."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=6)
    basis = env._mrf_basis(occ_full, rank=15)
    counts, occ = _restrict(counts_full, occ_full, basis)

    fit = fit_mrf_gam(basis, counts, occ, penalty=2.5)

    assert fit.penalty == 2.5
    assert fit.reml_objective is None


# ---------------------------------------------------------------------------
# Structural rank / no rank arg
# ---------------------------------------------------------------------------
def test_penalty_rank_structural(open_field_env, simulate_place_fields):
    """penalty_rank == r_eff - n_live_components, independent of any relative
    eigenvalue-threshold recount."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=7)
    basis = env._mrf_basis(occ_full, rank=18)
    counts, occ = _restrict(counts_full, occ_full, basis)

    fit = fit_mrf_gam(basis, counts, occ, penalty=1.0)

    assert fit.rank == basis.B.shape[1]
    assert fit.penalty_rank == basis.B.shape[1] - basis.n_live_components
    assert fit.penalty_rank == basis.d.size - basis.n_live_components


def test_fit_no_rank_arg(open_field_env, simulate_place_fields):
    """fit_mrf_gam has no rank parameter; MRFFit.rank derives from the basis
    (the single source of truth), even when the basis rank was clamped."""
    assert "rank" not in inspect.signature(fit_mrf_gam).parameters

    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=8)
    basis = env._mrf_basis(occ_full, rank=10**9)  # clamped to n_live_bins
    counts, occ = _restrict(counts_full, occ_full, basis)

    fit = fit_mrf_gam(basis, counts, occ, penalty=1.0)
    assert fit.rank == basis.B.shape[1] == basis.live_bins.size


def test_fit_rejects_unrestricted_counts(two_component_env, simulate_place_fields):
    """Passing full (n_bins, n_units) counts (not restricted to live_bins)
    raises a clear shape error -- the fit validates, never re-slices."""
    env = two_component_env
    occ_full = np.zeros(env.n_bins)
    occ_full[0] = 5.0  # visit one bin of one component only
    basis = env._mrf_basis(occ_full, rank=None)
    assert basis.live_bins.size < env.n_bins  # dead component excluded

    counts_full, _ = simulate_place_fields(env, [(3.0, 3.0)], seed=9)
    assert counts_full.shape[0] == env.n_bins != basis.B.shape[0]
    with pytest.raises(ValueError, match="live bins"):
        fit_mrf_gam(basis, counts_full, occ_full[basis.live_bins], penalty=1.0)


# ---------------------------------------------------------------------------
# Deviance
# ---------------------------------------------------------------------------
def test_deviance_formula(open_field_env, simulate_place_fields):
    """MRFFit.deviance == 2 * sum[n log(n/mu) - (n - mu)] over exposed bins,
    with mu = occupancy * floored-rate and 0 log 0 == 0 -- recomputed here from
    the stored log_rate, independently of the fit's own deviance path."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(
        env, [(6.0, 6.0), (10.0, 10.0)], seed=10
    )
    basis = env._mrf_basis(occ_full, rank=20)
    counts, occ = _restrict(counts_full, occ_full, basis)

    fit = fit_mrf_gam(basis, counts, occ, penalty=1.0)

    fr = _firing_rate(fit)  # (n_live_bins, n_units)
    mu = occ[:, None] * fr
    exposed = occ > 0
    n = counts[exposed].astype(np.float64)
    m = mu[exposed]
    term = np.where(n > 0, n * np.log(np.where(n > 0, n, 1.0) / m), 0.0) - (n - m)
    expected = 2.0 * term.sum(0)

    np.testing.assert_allclose(fit.deviance, expected, rtol=1e-10, atol=1e-10)
    assert fit.deviance.shape == (counts.shape[1],)
    assert np.all(np.isfinite(fit.deviance))


# ---------------------------------------------------------------------------
# Convergence on the objective, not the coefficient step
# ---------------------------------------------------------------------------
def test_convergence_on_deviance(open_field_env, simulate_place_fields):
    """An undersampled arena (weakly-identified null-mode directions drift)
    still converges on the relative penalized-objective decrease. A coefficient-
    step criterion at the same tolerance would NOT: the accepted step is far
    above _FIT_TOL when the objective has already plateaued."""
    env = open_field_env
    occ = np.full(env.n_bins, 3.0)
    occ[env.n_bins // 3 :] = 0.0  # only a third of the arena is visited
    counts_full, occ_full = simulate_place_fields(
        env, [(4.0, 4.0)], occupancy=occ, seed=11
    )
    basis = env._mrf_basis(occ_full, rank=None)  # full rank -> null-mode drift
    counts, occupancy = _restrict(counts_full, occ_full, basis)

    coeffs, _eta, _mu, n_iter, max_step, converged = _newton_fit_numpy(
        counts, occupancy, basis.B, 1e-2 * basis.d, _MAX_ITER, _FIT_TOL
    )

    assert converged  # stopped on the objective
    assert n_iter < _MAX_ITER  # not by exhausting iterations
    # A coeff-step criterion at tol == _FIT_TOL would still be iterating here.
    assert max_step > _FIT_TOL
    assert np.all(np.isfinite(coeffs))


# ---------------------------------------------------------------------------
# Batched Hessian orientation
# ---------------------------------------------------------------------------
def test_hessian_batched_equals_per_unit():
    """The batched Hessian einsum('ir,ik,is->krs', B, mu, B) equals
    B^T diag(mu[:, k]) B (+ jitter) for every unit k, shape (n_units, r, r).

    The transposed order 'ik,ij,il->kjl' would give (r, n_units, r) and silently
    break the per-unit diagonal add and solve -- this guards against it."""
    rng = np.random.default_rng(0)
    n_bins, r, n_units = 7, 4, 3
    B = rng.standard_normal((n_bins, r))
    mu = rng.uniform(0.1, 2.0, size=(n_bins, n_units))
    penalty_diag = np.array([0.0, 2.0, 5.0, 0.0])

    H = _penalized_hessian(B, mu, penalty_diag)

    assert H.shape == (n_units, r, r)
    for k in range(n_units):
        expected = B.T @ np.diag(mu[:, k]) @ B + np.diag(penalty_diag)
        # Both carry the same jitter; compare the data + penalty structure.
        np.testing.assert_allclose(
            H[k] - np.eye(r) * _hessian_jitter(), expected, rtol=1e-10, atol=1e-10
        )


def _hessian_jitter():
    from neurospatial.encoding._glm import _HESSIAN_JITTER

    return _HESSIAN_JITTER


# ---------------------------------------------------------------------------
# eta clip as a predictor constraint (bounded fit, no frozen bins)
# ---------------------------------------------------------------------------
def test_fit_bounded_at_clip_no_overshoot():
    """A near-clip optimum converges to the true MLE, not an overshoot into the
    saturated region. The clip is a constraint on the predictor: the line search
    rejects steps that leave +/-clip, so a flat-curvature Newton step is halved
    back to the interior MLE instead of walking the (bounded) objective off.

    Reviewer example: count=1, occupancy=1e-12 -> MLE exp(eta)=count/occ=1e12,
    i.e. eta=log(1e12)~=27.63 (inside +/-clip) and mu=1 -- NOT the clip (eta=30,
    mu~=10.7)."""
    B = np.array([[1.0]])
    counts = np.array([[1.0]])
    occ = np.array([1e-12])

    coeffs, eta, mu, _n_iter, _max_step, converged = _newton_fit_numpy(
        counts, occ, B, np.zeros(1), _MAX_ITER, _FIT_TOL
    )

    assert converged
    assert coeffs[0, 0] < 30.0  # did NOT overshoot to / past the clip
    np.testing.assert_allclose(eta[0, 0], np.log(1e12), atol=0.05)
    np.testing.assert_allclose(mu[0, 0], 1.0, atol=0.02)


def test_multibin_saturated_bin_recovers():
    """A bin driven toward the clip by a shared step must recover to its interior
    MLE, not freeze at the boundary. This is the multi-bin case a one-bin test
    cannot expose: with a constant-plus-contrast basis the line search accepts on
    the summed objective, so a masked/frozen gradient at the clip would trap the
    under-occupied bin. Treating the clip as a predictor constraint keeps the
    standard (inward-recovering) gradient, so both bins reach their MLE.

    counts=[1, 1000], occupancy=[1e-12, 1] -> MLE eta=[log(1e12), log(1000)]
    ~=[27.63, 6.91], mu=[1, 1000] -- bin 0 must NOT stick at eta=30, mu~=10.7."""
    B = np.array([[1.0, 1.0], [1.0, -1.0]])  # eta = [a + b, a - b]
    counts = np.array([[1.0], [1000.0]])
    occ = np.array([1e-12, 1.0])

    _coeffs, eta, mu, _n_iter, _max_step, converged = _newton_fit_numpy(
        counts, occ, B, np.zeros(2), _MAX_ITER, _FIT_TOL
    )

    assert converged
    np.testing.assert_allclose(eta[:, 0], [np.log(1e12), np.log(1000.0)], atol=0.05)
    np.testing.assert_allclose(mu[:, 0], [1.0, 1000.0], rtol=1e-3)
    assert eta[0, 0] < 30.0  # bin 0 recovered from the boundary, not frozen


def test_fit_predictor_stays_in_box(open_field_env, simulate_place_fields):
    """Every accepted iterate keeps the raw predictor within the safety limit
    (``|B@coeffs| <= _ETA_CLIP``): the clip is enforced strictly during the line
    search, so a saturation-prone fit never leaves the box (where the reported
    rate would otherwise overshoot)."""
    from neurospatial.encoding._glm import _ETA_CLIP

    env = open_field_env
    # A few very under-occupied bins push the fit toward the (lower) clip.
    occ = np.full(env.n_bins, 3.0)
    occ[:5] = 1e-10
    counts_full, occ_full = simulate_place_fields(
        env, [(8.0, 8.0)], occupancy=occ, peak_rate=200.0, seed=28
    )
    basis = env._mrf_basis(occ_full, rank=15)
    counts, occupancy = _restrict(counts_full, occ_full, basis)

    coeffs, _eta, _mu, _n, _s, _c = _newton_fit_numpy(
        counts, occupancy, basis.B, 1.0 * basis.d, _MAX_ITER, _FIT_TOL
    )

    assert np.all(np.abs(basis.B @ coeffs) <= _ETA_CLIP)  # predictor in the box


def test_out_of_domain_not_converged():
    """A fit whose data wants a rate above the clip is out of domain: the clip is
    a numerical safety limit, not a valid statistical boundary, so the fit reports
    converged=False rather than pretending to converge with the predictor pinned
    at the limit. Both the single-bin case and the multi-bin case (where one
    boundary bin throttles a free bin's shared step) must report converged=False.

    count=2e13, occupancy=1 -> empirical rate 2e13 > exp(30)~=1.07e13."""
    # Single bin: unconstrained MLE eta=log(2e13)~=30.63 > clip.
    _c, _e, _m, _n, _s, converged = _newton_fit_numpy(
        np.array([[2e13]]),
        np.array([1.0]),
        np.array([[1.0]]),
        np.zeros(1),
        _MAX_ITER,
        _FIT_TOL,
    )
    assert converged is False

    # Multi bin: bin 0 is out of domain and throttles bin 1 through the shared
    # constant-plus-contrast basis; the fit must NOT report a (false) convergence.
    _c, _e, _m, _n, _s, converged = _newton_fit_numpy(
        np.array([[2e13], [1.0]]),
        np.array([1.0, 1.0]),
        np.array([[1.0, 1.0], [1.0, -1.0]]),
        np.zeros(2),
        _MAX_ITER,
        _FIT_TOL,
    )
    assert converged is False


def test_sparse_occupancy_in_domain_converges(open_field_env, simulate_place_fields):
    """A realistic sparse-occupancy / low-count fit (empirical rates well below the
    clip) is in domain and converges normally -- the out-of-domain guard does not
    false-trip on ordinary under-sampling or zero-count bins."""
    env = open_field_env
    # Sparse, low occupancy with ordinary (finite, small) counts -- rates ~ O(10) Hz.
    occ = np.full(env.n_bins, 0.05)
    occ[::3] = 0.0  # a third of the arena unvisited (zero-count, zero-occupancy)
    counts_full, occ_full = simulate_place_fields(
        env, [(8.0, 8.0)], occupancy=occ, peak_rate=15.0, seed=29
    )
    basis = env._mrf_basis(occ_full, rank=15)
    counts, occupancy = _restrict(counts_full, occ_full, basis)

    _c, _eta, _mu, _n, _s, converged = _newton_fit_numpy(
        counts, occupancy, basis.B, 1.0 * basis.d, _MAX_ITER, _FIT_TOL
    )

    assert converged is True  # in-domain: rates << exp(clip)


# ---------------------------------------------------------------------------
# Consistent (coeffs, eta, mu) triple + accepted step
# ---------------------------------------------------------------------------
def test_fit_returns_consistent_triple(open_field_env, simulate_place_fields):
    """The returned (coeffs, eta, mu) is internally consistent and recomputed
    from the FINAL coefficients: eta == clip(B @ coeffs), mu == occ * exp(eta)."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=12)
    basis = env._mrf_basis(occ_full, rank=15)
    counts, occ = _restrict(counts_full, occ_full, basis)

    from neurospatial.encoding._glm import _ETA_CLIP

    coeffs, eta, mu, _n, _s, _c = _newton_fit_numpy(
        counts, occ, basis.B, 1.0 * basis.d, _MAX_ITER, _FIT_TOL
    )

    np.testing.assert_array_equal(eta, np.clip(basis.B @ coeffs, -_ETA_CLIP, _ETA_CLIP))
    np.testing.assert_allclose(mu, occ[:, None] * np.exp(eta), rtol=0, atol=0)


def test_step_halve_returns_accepted_step(open_field_env, simulate_place_fields):
    """_step_halve returns the ACCEPTED (possibly halved) step and its new
    objective, not the raw step: a genuine Newton direction inflated so alpha=1
    overshoots (increases the objective) is halved back until it descends, and
    the reported step is that accepted (halved) one -- guarding the diagnostic."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=13)
    basis = env._mrf_basis(occ_full, rank=10)
    counts, occ = _restrict(counts_full, occ_full, basis)
    B, penalty_diag = basis.B, 1.0 * basis.d

    from neurospatial.encoding._glm_numpy import (
        _lstsq_constant,
        _penalized_hessian,
        _penalized_obj,
    )

    rate0 = np.clip(counts.sum(0) / max(occ.sum(), 1e-9), 1e-6, None)
    coeffs = _lstsq_constant(B, np.log(rate0))
    prev = _penalized_obj(coeffs, B, counts, occ, penalty_diag)
    # The true Newton step (a descent direction), then inflate it 20x so alpha=1
    # overshoots the minimum and increases the objective, forcing step-halving.
    eta = np.clip(B @ coeffs, -30.0, 30.0)
    mu = occ[:, None] * np.exp(eta)
    grad = B.T @ (counts - mu) - penalty_diag[:, None] * coeffs
    H = _penalized_hessian(B, mu, penalty_diag)
    newton = np.linalg.solve(H, grad.T[..., None])[..., 0].T
    inflated = 20.0 * newton

    new_coeffs, accepted_step, obj, ok = _step_halve(
        coeffs, inflated, B, counts, occ, penalty_diag, _FIT_TOL
    )

    assert ok is True  # a genuine descent direction: line search succeeds
    # The accepted step is a strict fraction of the inflated step (it was halved).
    ratio = accepted_step / inflated
    assert np.all(ratio <= 1.0) and np.any(ratio < 1.0)
    # By the halving criterion the accepted objective does not increase.
    assert np.all(obj <= prev)
    # new_coeffs == coeffs + accepted_step exactly.
    np.testing.assert_allclose(new_coeffs, coeffs + accepted_step, rtol=0, atol=0)


def test_step_halve_rejects_failed_descent(open_field_env, simulate_place_fields):
    """A genuine ascent direction cannot descend at any step size: _step_halve
    must NOT commit the uphill trial (which the caller would read as convergence
    via a negative relative decrease). It reverts to the previous coefficients
    and signals line_search_ok=False."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=21)
    basis = env._mrf_basis(occ_full, rank=10)
    counts, occ = _restrict(counts_full, occ_full, basis)
    B, penalty_diag = basis.B, 1.0 * basis.d

    from neurospatial.encoding._glm_numpy import _lstsq_constant, _penalized_obj

    rate0 = np.clip(counts.sum(0) / max(occ.sum(), 1e-9), 1e-6, None)
    coeffs = _lstsq_constant(B, np.log(rate0))
    prev = _penalized_obj(coeffs, B, counts, occ, penalty_diag)
    eta = np.clip(B @ coeffs, -30.0, 30.0)
    mu = occ[:, None] * np.exp(eta)
    grad = B.T @ (counts - mu) - penalty_diag[:, None] * coeffs
    ascent = -grad  # ascent for the objective-to-minimize; no positive step descends

    new_coeffs, _accepted_step, obj, ok = _step_halve(
        coeffs, 1e6 * ascent, B, counts, occ, penalty_diag, _FIT_TOL
    )

    assert ok is False  # failure signaled -> caller must not report convergence
    # The failed unit is reverted: coefficients unchanged, objective not worsened.
    np.testing.assert_allclose(new_coeffs, coeffs, rtol=0, atol=0)
    assert np.all(obj <= prev + _FIT_TOL * (1.0 + np.abs(prev)))


def test_step_halve_rejects_nonfinite_trial(open_field_env, simulate_place_fields):
    """A NaN Newton step must not be committed (NaN > prev is False, so a naive
    guard would accept it): _step_halve reverts to the previous coefficients and
    signals failure, keeping the returned fit finite."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=22)
    basis = env._mrf_basis(occ_full, rank=10)
    counts, occ = _restrict(counts_full, occ_full, basis)
    B, penalty_diag = basis.B, 1.0 * basis.d

    from neurospatial.encoding._glm_numpy import _lstsq_constant

    rate0 = np.clip(counts.sum(0) / max(occ.sum(), 1e-9), 1e-6, None)
    coeffs = _lstsq_constant(B, np.log(rate0))

    new_coeffs, _accepted_step, obj, ok = _step_halve(
        coeffs, np.full_like(coeffs, np.nan), B, counts, occ, penalty_diag, _FIT_TOL
    )

    assert ok is False
    assert np.all(np.isfinite(new_coeffs)) and np.all(np.isfinite(obj))
    np.testing.assert_allclose(new_coeffs, coeffs, rtol=0, atol=0)  # fully reverted


def test_newton_failed_line_search_not_reported_converged(
    open_field_env, simulate_place_fields, monkeypatch
):
    """When the line search fails, _newton_fit_numpy must return converged=False
    and finite fields -- never report convergence on a non-descending step.

    Force the failure by patching the batched solve to return a strong ascent
    direction (a non-PD/ill-conditioned Hessian would produce such a step in the
    wild); the fit must detect it and refuse to claim convergence."""
    import neurospatial.encoding._glm_numpy as glmnp

    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=23)
    basis = env._mrf_basis(occ_full, rank=10)
    counts, occ = _restrict(counts_full, occ_full, basis)

    real_solve = np.linalg.solve

    def ascent_solve(H, rhs):
        return -1e6 * real_solve(H, rhs)  # flip + inflate -> a strong ascent step

    monkeypatch.setattr(glmnp.np.linalg, "solve", ascent_solve)

    coeffs, eta, _mu, _n_iter, _max_step, converged = _newton_fit_numpy(
        counts, occ, basis.B, 1.0 * basis.d, _MAX_ITER, _FIT_TOL
    )

    assert converged is False  # a failed line search is not convergence
    assert np.all(np.isfinite(eta)) and np.all(np.isfinite(coeffs))


# ---------------------------------------------------------------------------
# pooled=False (per-unit lambda) is not yet supported -- reject, don't mislabel
# ---------------------------------------------------------------------------
def test_pooled_false_rejected(open_field_env, simulate_place_fields):
    """pooled=False requests per-unit lambda selection, which is not implemented:
    the fit only ever selects a single shared lambda. Reject it loudly rather
    than run pooled REML and mislabel the result pooled=False (per-unit)."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=24)
    basis = env._mrf_basis(occ_full, rank=15)
    counts, occ = _restrict(counts_full, occ_full, basis)

    with pytest.raises(NotImplementedError, match="pooled=False"):
        fit_mrf_gam(basis, counts, occ, penalty=None, pooled=False)

    # The default (pooled=True) is unaffected and records pooled=True.
    fit = fit_mrf_gam(basis, counts, occ, penalty=None)
    assert fit.pooled is True


# ---------------------------------------------------------------------------
# All-zero-spike population (fixed-penalty contract respected with no data)
# ---------------------------------------------------------------------------
def test_all_zero_spike_population(open_field_env):
    """counts.sum() == 0, penalty=None -> REML skipped, penalty is None,
    reml_objective is None, fields floor to _RATE_FLOOR, deviance finite, a
    warning is emitted, no raise."""
    env = open_field_env
    occ_full = np.full(env.n_bins, 3.0)
    basis = env._mrf_basis(occ_full, rank=15)
    counts = np.zeros((basis.live_bins.size, 4), dtype=np.int64)
    occ = occ_full[basis.live_bins]

    with pytest.warns(UserWarning):
        fit = fit_mrf_gam(basis, counts, occ, penalty=None)

    assert fit.penalty is None
    assert fit.reml_objective is None
    np.testing.assert_allclose(_firing_rate(fit), _RATE_FLOOR)
    assert np.all(np.isfinite(fit.deviance))


def test_all_zero_spike_fixed_penalty(open_field_env):
    """counts.sum() == 0, penalty=3.0 -> REML still skipped but the fixed penalty
    is the model actually applied and IS recorded; reml_objective is None; fields
    still floor. The fixed-penalty contract holds even with no data."""
    env = open_field_env
    occ_full = np.full(env.n_bins, 3.0)
    basis = env._mrf_basis(occ_full, rank=15)
    counts = np.zeros((basis.live_bins.size, 2), dtype=np.int64)
    occ = occ_full[basis.live_bins]

    with pytest.warns(UserWarning):
        fit = fit_mrf_gam(basis, counts, occ, penalty=3.0)

    assert fit.penalty == 3.0
    assert fit.reml_objective is None
    np.testing.assert_allclose(_firing_rate(fit), _RATE_FLOOR)


def test_zero_spike_neuron(open_field_env, simulate_place_fields):
    """A single zero-spike unit inside an otherwise-informative population fits
    normally under the shared lambda: finite near-floor rate, finite deviance,
    scalar (shared) penalty for all units."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(
        env, [(6.0, 6.0), (10.0, 10.0)], seed=14
    )
    basis = env._mrf_basis(occ_full, rank=18)
    counts, occ = _restrict(counts_full, occ_full, basis)
    counts = np.concatenate(
        [counts, np.zeros((counts.shape[0], 1), dtype=counts.dtype)], axis=1
    )  # append a silent unit

    fit = fit_mrf_gam(basis, counts, occ, penalty=1.0)

    fr = _firing_rate(fit)
    assert np.all(np.isfinite(fr))
    assert np.all(np.isfinite(fit.deviance))
    # The silent unit floors near _RATE_FLOOR everywhere.
    np.testing.assert_allclose(fr[:, -1], _RATE_FLOOR, rtol=0, atol=1e-6)
    assert np.isscalar(fit.penalty) or np.ndim(fit.penalty) == 0  # shared lambda
    # The silent unit's objective shrinks toward 0 as its intercept sinks; the
    # 1+|obj| convergence measure still converges (a pure-relative one would not).
    assert fit.converged is True


# ---------------------------------------------------------------------------
# Nonconvergence warning keyed on the flag
# ---------------------------------------------------------------------------
def test_nonconvergence_warns(open_field_env, simulate_place_fields, monkeypatch):
    """A fit forced non-converged (max_iter clamped to 1 via the module constant)
    emits a UserWarning keyed on `not converged`, still returns fields, no raise."""
    import neurospatial.encoding._glm as glm

    monkeypatch.setattr(glm, "_MAX_ITER", 1)  # one Newton step can't converge

    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=15)
    basis = env._mrf_basis(occ_full, rank=25)
    counts, occ = _restrict(counts_full, occ_full, basis)

    with pytest.warns(UserWarning, match="converge"):
        fit = fit_mrf_gam(basis, counts, occ, penalty=1.0)  # fixed lambda: skip REML

    assert fit.converged is False
    assert fit.n_iter == 1
    assert np.all(np.isfinite(_firing_rate(fit)))


# ---------------------------------------------------------------------------
# Degenerate short-circuits: no neurons, empty basis, dead component,
# penalty=0 rank-deficient
# ---------------------------------------------------------------------------
def test_no_neurons(open_field_env):
    """counts.shape[1] == 0 -> empty-unit MRFFit; no fit runs, converged True."""
    env = open_field_env
    occ_full = np.full(env.n_bins, 3.0)
    basis = env._mrf_basis(occ_full, rank=10)
    counts = np.zeros((basis.live_bins.size, 0), dtype=np.int64)
    occ = occ_full[basis.live_bins]

    fit = fit_mrf_gam(basis, counts, occ, penalty=None)

    assert fit.coefficients.shape == (basis.B.shape[1], 0)
    assert fit.log_rate.shape == (basis.live_bins.size, 0)
    assert fit.deviance.shape == (0,)
    assert fit.penalty is None and fit.reml_objective is None
    assert fit.converged is True
    assert fit.rank == basis.B.shape[1]


def test_zero_occupancy_empty_basis(open_field_env):
    """A (0, 0) basis (zero total occupancy) -> coefficients (0, n_units),
    log_rate (0, n_units), zero deviance, penalty/reml None, converged True,
    warns; no fit runs and nothing raises."""
    env = open_field_env
    occ_full = np.zeros(env.n_bins)
    basis = env._mrf_basis(occ_full, rank=None)
    assert basis.B.shape == (0, 0)
    counts = np.zeros((0, 3), dtype=np.int64)
    occ = np.zeros(0)

    with pytest.warns(UserWarning):
        fit = fit_mrf_gam(basis, counts, occ, penalty=None)

    assert fit.coefficients.shape == (0, 3)
    assert fit.log_rate.shape == (0, 3)
    np.testing.assert_array_equal(fit.deviance, np.zeros(3))
    assert fit.penalty is None and fit.reml_objective is None
    assert fit.converged is True


def test_dead_component_fits_live_bins(two_component_env, simulate_place_fields):
    """When a component is dead (never visited), the basis excludes its bins and
    the fit runs on live bins only, returning finite rates shaped to live_bins."""
    env = two_component_env
    from neurospatial.ops.diffusion import (
        _assemble_W,
        _components_from_W,
        _finite_volume_geometry,
    )

    graph, volumes = _finite_volume_geometry(env)
    W = _assemble_W(graph, len(volumes))
    _nc, labels = _components_from_W(W)
    occ_full = np.zeros(env.n_bins)
    occ_full[labels == 0] = 3.0  # visit only component 0
    basis = env._mrf_basis(occ_full, rank=None)
    assert basis.live_bins.size < env.n_bins  # component 1 excluded

    counts_full, _ = simulate_place_fields(
        env, [(3.0, 3.0)], occupancy=occ_full, seed=16
    )
    counts, occ = _restrict(counts_full, occ_full, basis)

    fit = fit_mrf_gam(basis, counts, occ, penalty=1.0)

    assert fit.log_rate.shape == (basis.live_bins.size, 1)
    assert np.all(np.isfinite(_firing_rate(fit)))


def test_penalty_zero_rank_deficient_warns(open_field_env, simulate_place_fields):
    """penalty == 0 with a design rank-deficient on the exposed (visited) bins
    warns about identifiability but still fits and returns finite rates."""
    env = open_field_env
    occ = np.full(env.n_bins, 3.0)
    occ[30:] = 0.0  # visit < r_eff bins so B[exposed] is rank-deficient at full rank
    counts_full, occ_full = simulate_place_fields(
        env, [(4.0, 4.0)], occupancy=occ, seed=17
    )
    basis = env._mrf_basis(occ_full, rank=None)  # r_eff == n_live_bins > n_exposed
    counts, occupancy = _restrict(counts_full, occ_full, basis)
    n_exposed = int((occupancy > 0).sum())
    assert n_exposed < basis.B.shape[1]  # rank-deficient on exposed bins

    with pytest.warns(UserWarning, match="identifi"):
        fit = fit_mrf_gam(basis, counts, occupancy, penalty=0.0)

    assert fit.penalty == 0.0
    assert np.all(np.isfinite(_firing_rate(fit)))
