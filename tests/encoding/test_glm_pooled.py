"""Per-neuron ``lambda`` (``pooled=False``) for the penalized-Poisson GAM fit.

``pooled=False`` selects an independent smoothing penalty ``lambda_k`` per unit
by running the REML objective once per informative unit, instead of a single
shared ``lambda`` over the whole population. These tests pin the additive
``fit_mrf_gam`` contract: ``pooled=True`` (the default) is byte-identical to the
shared-``lambda`` fit apart from the new scalar ``reml_at_boundary`` diagnostic,
and ``pooled=False`` widens ``penalty`` / ``reml_objective`` / ``reml_at_boundary``
to per-unit vectors on the automatic-REML path.

The fit is bin-major: ``counts`` arrive ``(n_live_bins, n_units)`` restricted to
``basis.live_bins``.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.encoding._glm import (
    _FIT_TOL,
    _LOG_PENALTY_BOUNDS,
    _MAX_ITER,
    _RATE_FLOOR,
    _REML_BOUNDARY_TOL,
    _REML_XATOL,
    _structural_constant_base,
    fit_mrf_gam,
)
from neurospatial.encoding._glm_numpy import select_penalty_by_reml


def _restrict(counts, occupancy, basis):
    return counts[basis.live_bins], occupancy[basis.live_bins]


def _firing_rate(fit):
    return np.maximum(np.exp(fit.log_rate), _RATE_FLOOR)


def _pooled_lambda(counts, occ, basis):
    """Independently recompute the shared-lambda REML optimum over ``counts``."""
    penalty_rank = basis.d.size - basis.n_live_components
    base = _structural_constant_base(basis.B, basis.n_live_components)
    lam, _obj, _bnd = select_penalty_by_reml(
        counts,
        occ,
        basis.B,
        basis.d,
        penalty_rank,
        constant_base=base,
        max_iter=_MAX_ITER,
        tol=_FIT_TOL,
    )
    return lam


# ---------------------------------------------------------------------------
# Constant contract
# ---------------------------------------------------------------------------
def test_reml_boundary_tol_is_five_xatol():
    """The boundary tolerance is exactly 5 * the REML xatol (shared-contracts)."""
    assert pytest.approx(5.0 * _REML_XATOL) == _REML_BOUNDARY_TOL
    assert pytest.approx(5e-3) == _REML_BOUNDARY_TOL


# ---------------------------------------------------------------------------
# The selectors return (penalty, objective, at_boundary)
# ---------------------------------------------------------------------------
def test_selector_returns_boundary_triple(open_field_env, simulate_place_fields):
    """select_penalty_by_reml returns a 3-tuple; an interior optimum is not at a
    boundary; REML-skip (penalty_rank == 0) returns (None, None, None)."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=5)
    basis = env._mrf_basis(occ_full, rank=15)
    counts, occ = _restrict(counts_full, occ_full, basis)
    penalty_rank = basis.d.size - basis.n_live_components

    lam, obj, at_boundary = select_penalty_by_reml(
        counts, occ, basis.B, basis.d, penalty_rank, max_iter=_MAX_ITER, tol=_FIT_TOL
    )
    assert np.isfinite(lam) and np.isfinite(obj)
    assert at_boundary is False
    # A well-identified place field sits inside the search interval.
    log_lam = np.log(lam)
    lower, upper = _LOG_PENALTY_BOUNDS
    assert min(log_lam - lower, upper - log_lam) > _REML_BOUNDARY_TOL

    # penalty_rank == 0 -> skip, all-None triple.
    assert select_penalty_by_reml(
        counts, occ, basis.B, basis.d, 0, max_iter=_MAX_ITER, tol=_FIT_TOL
    ) == (None, None, None)


# ---------------------------------------------------------------------------
# pooled=True unchanged (additive reml_at_boundary only)
# ---------------------------------------------------------------------------
def test_pooled_true_unchanged(open_field_env, simulate_place_fields):
    """pooled=True (default) keeps scalar penalty/reml_objective; the only new
    output is a scalar reml_at_boundary. Values match an explicit pooled=True."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(
        env, [(5.0, 5.0), (11.0, 11.0)], seed=2
    )
    basis = env._mrf_basis(occ_full, rank=20)
    counts, occ = _restrict(counts_full, occ_full, basis)

    fit = fit_mrf_gam(basis, counts, occ, penalty=None)  # default pooled=True

    assert fit.pooled is True
    assert np.isscalar(fit.penalty) or isinstance(fit.penalty, float)
    assert isinstance(fit.reml_objective, float)
    # reml_at_boundary is an additive scalar bool for pooled auto-REML.
    assert isinstance(fit.reml_at_boundary, bool)
    assert fit.penalty_selected_by_reml is None
    # Matches the shared-lambda optimum recomputed independently.
    assert fit.penalty == pytest.approx(_pooled_lambda(counts, occ, basis))


# ---------------------------------------------------------------------------
# pooled=False per-unit lambda
# ---------------------------------------------------------------------------
def test_pooled_false_per_unit_lambda(open_field_env, simulate_varied_smoothness):
    """A population with sharp + broad fields -> pooled=False recovers DISTINCT
    finite lambda_k (variance > 0), each at its own per-unit optimum, and a
    broader field is smoothed more than a sharp one."""
    env = open_field_env
    centers = [(5.0, 5.0), (11.0, 11.0), (8.0, 8.0)]
    sigmas = [1.5, 8.0, 4.0]  # sharp, broad, medium
    counts_full, occ_full = simulate_varied_smoothness(env, centers, sigmas, seed=3)
    basis = env._mrf_basis(occ_full, rank=30)
    counts, occ = _restrict(counts_full, occ_full, basis)

    fit = fit_mrf_gam(basis, counts, occ, penalty=None, pooled=False)

    assert fit.pooled is False
    penalty = np.asarray(fit.penalty)
    assert penalty.shape == (3,)
    assert np.all(np.isfinite(penalty))
    assert np.var(penalty) > 0.0  # genuinely per-unit, not one shared value
    assert np.all(np.asarray(fit.penalty_selected_by_reml))  # all informative

    # Each lambda_k is the unit's own REML optimum (not the shared one).
    for k in range(3):
        lam_k = _pooled_lambda(counts[:, k : k + 1], occ, basis)
        assert penalty[k] == pytest.approx(lam_k, rel=1e-6)

    # A single shared lambda is one number, distinct from the per-unit spread.
    fit_pooled = fit_mrf_gam(basis, counts, occ, penalty=None, pooled=True)
    assert np.isscalar(fit_pooled.penalty) or isinstance(fit_pooled.penalty, float)
    assert not np.allclose(penalty, fit_pooled.penalty)


def test_pooled_false_shapes(open_field_env, simulate_place_fields):
    """r>0, all units informative: penalty / reml_objective are (n_units,) finite,
    reml_at_boundary is (n_units,) bool, penalty_selected_by_reml all True."""
    env = open_field_env
    centers = [(5.0, 5.0), (11.0, 11.0), (5.0, 11.0), (11.0, 5.0)]
    counts_full, occ_full = simulate_place_fields(env, centers, seed=4)
    basis = env._mrf_basis(occ_full, rank=25)
    counts, occ = _restrict(counts_full, occ_full, basis)

    fit = fit_mrf_gam(basis, counts, occ, penalty=None, pooled=False)

    n = len(centers)
    assert np.asarray(fit.penalty).shape == (n,)
    assert np.all(np.isfinite(fit.penalty))
    assert np.asarray(fit.reml_objective).shape == (n,)
    assert np.all(np.isfinite(fit.reml_objective))
    boundary = np.asarray(fit.reml_at_boundary)
    assert boundary.shape == (n,)
    assert boundary.dtype == np.bool_
    selected = np.asarray(fit.penalty_selected_by_reml)
    assert selected.shape == (n,)
    assert selected.dtype == np.bool_
    assert np.all(selected)


def test_pooled_false_r0_scalar_none(two_path_env):
    """r == 0 population (two disjoint 3-node paths) with pooled=False keeps a
    scalar penalty=None (population-level), NOT a per-unit vector; rates finite."""
    env = two_path_env
    occ_full = np.ones(env.n_bins)
    basis = env._mrf_basis(occ_full, rank=2)
    assert basis.d.size - basis.n_live_components == 0  # penalty_rank == 0
    rng = np.random.default_rng(0)
    counts_full = rng.poisson(3.0, size=(env.n_bins, 3)).astype(np.int64)
    counts, occ = _restrict(counts_full, occ_full, basis)

    fit = fit_mrf_gam(basis, counts, occ, penalty=None, pooled=False)

    assert fit.penalty is None  # scalar None, not a vector
    assert fit.reml_objective is None
    assert fit.reml_at_boundary is None
    assert fit.penalty_selected_by_reml is None
    assert fit.pooled is False
    assert np.all(np.isfinite(_firing_rate(fit)))


def test_pooled_false_zero_spike_fallback(open_field_env, simulate_place_fields):
    """A zero-spike unit inside a pooled=False, r>0 population takes the fallback
    lambda = the pooled lambda over the informative units (NOT the optimizer's
    per-unit point); penalty_selected_by_reml is False and reml_objective is nan
    for it, while informative units keep their own lambda_k and True."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(
        env, [(5.0, 5.0), (11.0, 11.0)], seed=7
    )
    basis = env._mrf_basis(occ_full, rank=20)
    counts, occ = _restrict(counts_full, occ_full, basis)
    # Append a genuine zero-spike unit as the last column.
    counts = np.column_stack([counts, np.zeros(counts.shape[0], dtype=counts.dtype)])
    zs = counts.shape[1] - 1
    informative = np.array([True, True, False])

    fit = fit_mrf_gam(basis, counts, occ, penalty=None, pooled=False)

    penalty = np.asarray(fit.penalty)
    selected = np.asarray(fit.penalty_selected_by_reml)
    reml_obj = np.asarray(fit.reml_objective)

    # The fallback lambda equals the pooled lambda over the informative subset.
    pooled_lam = _pooled_lambda(counts[:, informative], occ, basis)
    assert penalty[zs] == pytest.approx(pooled_lam, rel=1e-6)
    assert selected[zs] is np.False_ or selected[zs] == False  # noqa: E712
    assert np.isnan(reml_obj[zs])  # documented sentinel
    # Informative units keep their own optimum + provenance.
    assert np.all(selected[informative])
    assert np.all(np.isfinite(reml_obj[informative]))
    for k in np.flatnonzero(informative):
        assert penalty[k] == pytest.approx(
            _pooled_lambda(counts[:, k : k + 1], occ, basis), rel=1e-6
        )
    # The zero-spike unit's field floors.
    fr = _firing_rate(fit)
    assert np.max(fr[:, zs]) == pytest.approx(_RATE_FLOOR, abs=1e-6)


def test_pooled_false_all_zero_spike(open_field_env):
    """No informative unit at all -> reuse the all-zero-spike degenerate path:
    scalar penalty=None, reml_objective=None, floor fields, NO pooled REML run."""
    env = open_field_env
    occ_full = np.full(env.n_bins, 3.0)
    basis = env._mrf_basis(occ_full, rank=15)
    counts = np.zeros((basis.live_bins.size, 4), dtype=np.int64)
    occ = occ_full[basis.live_bins]

    with pytest.warns(UserWarning, match="no unit has any spikes"):
        fit = fit_mrf_gam(basis, counts, occ, penalty=None, pooled=False)

    assert fit.penalty is None  # scalar, not a vector
    assert fit.reml_objective is None
    assert fit.reml_at_boundary is None
    assert fit.penalty_selected_by_reml is None
    assert fit.pooled is False
    assert np.allclose(_firing_rate(fit), _RATE_FLOOR)


def test_fixed_penalty_precedence(open_field_env, simulate_place_fields):
    """pooled=False with a fixed penalty -> REML skipped, scalar penalty echoed,
    reml_objective None (fixed penalty beats pooled)."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(
        env, [(5.0, 5.0), (11.0, 11.0)], seed=8
    )
    basis = env._mrf_basis(occ_full, rank=15)
    counts, occ = _restrict(counts_full, occ_full, basis)

    fit = fit_mrf_gam(basis, counts, occ, penalty=2.5, pooled=False)

    assert fit.penalty == 2.5  # scalar, not a vector
    assert np.isscalar(fit.penalty) or isinstance(fit.penalty, float)
    assert fit.reml_objective is None
    assert fit.reml_at_boundary is None
    assert fit.penalty_selected_by_reml is None
    assert fit.pooled is False


# ---------------------------------------------------------------------------
# Boundary diagnostic
# ---------------------------------------------------------------------------
def test_reml_boundary_diagnostic_pooled(open_field_env, simulate_flat_weak_signal):
    """A spatially flat / weak signal drives the pooled lambda to the UPPER search
    bound: reml_at_boundary is True (scalar) with one warning naming the upper
    boundary; a well-structured field is interior (False)."""
    env = open_field_env
    counts_full, occ_full = simulate_flat_weak_signal(env, 2, seed=11)
    basis = env._mrf_basis(occ_full, rank=20)
    counts, occ = _restrict(counts_full, occ_full, basis)

    with pytest.warns(UserWarning, match="upper"):
        fit = fit_mrf_gam(basis, counts, occ, penalty=None, pooled=True)
    assert fit.reml_at_boundary is True
    assert np.isfinite(fit.penalty)  # the applied lambda is still the finite bound
    log_lam = float(np.log(float(fit.penalty)))
    upper = _LOG_PENALTY_BOUNDS[1]
    assert upper - log_lam <= _REML_BOUNDARY_TOL


def test_reml_boundary_diagnostic_interior(open_field_env, simulate_place_fields):
    """A well-identified field is interior: reml_at_boundary is False."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=12)
    basis = env._mrf_basis(occ_full, rank=15)
    counts, occ = _restrict(counts_full, occ_full, basis)

    fit = fit_mrf_gam(basis, counts, occ, penalty=None, pooled=True)
    assert fit.reml_at_boundary is False


def test_reml_boundary_diagnostic_per_unit(open_field_env, simulate_flat_weak_signal):
    """pooled=False on flat data: reml_at_boundary is a (n_units,) bool vector, all
    at the boundary, with one warning."""
    env = open_field_env
    counts_full, occ_full = simulate_flat_weak_signal(env, 3, seed=13)
    basis = env._mrf_basis(occ_full, rank=20)
    counts, occ = _restrict(counts_full, occ_full, basis)

    with pytest.warns(UserWarning, match="upper"):
        fit = fit_mrf_gam(basis, counts, occ, penalty=None, pooled=False)
    boundary = np.asarray(fit.reml_at_boundary)
    assert boundary.shape == (3,)
    assert boundary.dtype == np.bool_
    assert np.all(boundary)


def test_per_unit_warning_names_unit_ids(open_field_env, simulate_flat_weak_signal):
    """The per-unit boundary warning names the caller's unit_ids when supplied,
    and falls back to positional 'unit index/indices' when they are not."""
    env = open_field_env
    counts_full, occ_full = simulate_flat_weak_signal(env, 2, seed=21)
    basis = env._mrf_basis(occ_full, rank=20)
    counts, occ = _restrict(counts_full, occ_full, basis)

    # Labels supplied -> the warning carries them, not bare indices.
    with pytest.warns(UserWarning) as rec:
        fit_mrf_gam(
            basis,
            counts,
            occ,
            penalty=None,
            pooled=False,
            unit_ids=["cell-A", "cell-B"],
        )
    msg = "\n".join(str(w.message) for w in rec)
    assert "cell-A" in msg and "cell-B" in msg
    assert "unit index" not in msg  # labels replace the positional fallback

    # No labels -> positional "unit indices".
    with pytest.warns(UserWarning) as rec2:
        fit_mrf_gam(basis, counts, occ, penalty=None, pooled=False)
    msg2 = "\n".join(str(w.message) for w in rec2)
    assert "unit indices" in msg2


def test_reml_boundary_none_when_reml_skipped(
    open_field_env, two_path_env, simulate_place_fields
):
    """Fixed penalty / r==0 / no-data -> reml_at_boundary is None (REML did not
    run), for both pooled settings."""
    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=14)
    basis = env._mrf_basis(occ_full, rank=15)
    counts, occ = _restrict(counts_full, occ_full, basis)

    # Fixed penalty.
    for pooled in (True, False):
        fit = fit_mrf_gam(basis, counts, occ, penalty=1.0, pooled=pooled)
        assert fit.reml_at_boundary is None

    # r == 0.
    r0_env = two_path_env
    occ0 = np.ones(r0_env.n_bins)
    basis0 = r0_env._mrf_basis(occ0, rank=2)
    rng = np.random.default_rng(0)
    counts0_full = rng.poisson(3.0, size=(r0_env.n_bins, 2)).astype(np.int64)
    counts0, occ0r = _restrict(counts0_full, occ0, basis0)
    for pooled in (True, False):
        fit0 = fit_mrf_gam(basis0, counts0, occ0r, penalty=None, pooled=pooled)
        assert fit0.reml_at_boundary is None


# ---------------------------------------------------------------------------
# Aggregate batch diagnostics for the looped per-unit fits
# ---------------------------------------------------------------------------
def test_aggregate_diagnostics(open_field_env, simulate_place_fields, monkeypatch):
    """When one per-unit final fit fails to converge: converged is False, n_iter
    is the max over the per-unit fits, and ONE warning names the failed unit id
    (not one per unit)."""
    import neurospatial.encoding._glm_numpy as glmnp

    env = open_field_env
    counts_full, occ_full = simulate_place_fields(
        env, [(5.0, 5.0), (11.0, 11.0)], seed=15
    )
    basis = env._mrf_basis(occ_full, rank=18)
    counts, occ = _restrict(counts_full, occ_full, basis)
    # Add a lone zero-spike unit; its final fit runs on an all-zero column and is
    # the only fit uniquely identifiable without disturbing the informative REML.
    counts = np.column_stack([counts, np.zeros(counts.shape[0], dtype=counts.dtype)])
    zs = counts.shape[1] - 1

    real_fit = glmnp._newton_fit_numpy

    def failing_fit(*args):
        counts_arg = args[0]
        coeffs, eta, mu, n_iter, max_step, converged = real_fit(*args)
        # Force non-convergence ONLY for the lone zero-spike unit's final fit
        # (its entire counts column is zero); every REML/informative fit is
        # untouched, so REML still selects a lambda.
        if np.all(np.asarray(counts_arg) == 0):
            return coeffs, eta, mu, 99, max_step, False
        return coeffs, eta, mu, n_iter, max_step, converged

    monkeypatch.setattr(glmnp, "_newton_fit_numpy", failing_fit)

    with pytest.warns(UserWarning) as record:
        fit = fit_mrf_gam(basis, counts, occ, penalty=None, pooled=False)

    assert fit.converged is False
    assert fit.n_iter == 99  # the max across per-unit fits
    # Exactly one non-convergence warning, naming the failed unit id.
    nonconv = [w for w in record if "converge" in str(w.message)]
    assert len(nonconv) == 1
    assert str(zs) in str(nonconv[0].message)
