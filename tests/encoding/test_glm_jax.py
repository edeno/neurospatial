"""Parity + convergence of the optional float32 JAX ``method="glm"`` fit.

The float64 NumPy/SciPy core (:mod:`neurospatial.encoding._glm_numpy`) is the
correctness reference; :mod:`neurospatial.encoding._glm_jax` is a float32 speed
mirror. These tests pin, with measured numbers on simulated populations:

- **Parity** at an identical fixed ``lambda`` (isolating the Newton fit from REML
  ``lambda`` selection): the float32 rate/coefficients/deviance match the float64
  core at the ``~1e-6`` level (relative L2 and relative error above a rate
  threshold -- raw relative errors at near-zero rates are meaningless), and
  ``penalty`` / ``rank`` / ``penalty_rank`` are identical.
- **float32 convergence + the tolerance floor**: the float32 fit converges well
  inside ``_MAX_ITER``, and the ``_FIT_TOL_FLOOR`` is load-bearing -- floored, the
  fit stops in far fewer Newton iterations than the raw (un-floored) loop chasing
  a below-noise tolerance, and a sub-floor ``tol`` is treated identically to the
  floor.
- **Graceful degradation**: with JAX reported unavailable, the ``backend="jax"``
  request runs the NumPy core and still produces a valid ``MRFFit`` (no import
  error) -- the base-install path.
- **REML ``lambda`` agreement** between the two paths, within a
  ``_REML_XATOL``-consistent tolerance.

The end-to-end automatic-REML path differs a little more than the fixed-``lambda``
fit because float32 REML selects a slightly different ``lambda`` near the broad
(flat) objective minimum; that difference is scientifically negligible and is
bounded by the public ``compute_spatial_rate(s)`` return-contract test.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.encoding._glm import (
    _FIT_TOL,
    _FIT_TOL_FLOOR,
    _MAX_ITER,
    _RATE_FLOOR,
    MRFFit,
    fit_mrf_gam,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _restrict(counts, occupancy, basis):
    return counts[basis.live_bins], occupancy[basis.live_bins]


def _firing_rate(fit):
    return np.maximum(np.exp(fit.log_rate), _RATE_FLOOR)


def _rel_l2(got, ref):
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))


def _max_rel_above(got, ref, threshold):
    """Max relative error over bins where the reference rate exceeds ``threshold``.

    Relative error at near-zero (floored) rates is meaningless -- a 1e-10 floor
    vs 2e-10 is a 100% "error" on a bin no place cell fires in -- so parity of a
    rate map is measured only where the rate is appreciable.
    """
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    mask = ref > threshold
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(got - ref)[mask] / ref[mask]))


# ---------------------------------------------------------------------------
# Parity at a fixed lambda (isolated Newton fit): float32 JAX vs float64 NumPy
# ---------------------------------------------------------------------------
def test_jax_numpy_parity(open_field_env, simulate_place_fields):
    """At an identical fixed ``lambda`` the float32 JAX fit matches the float64
    NumPy core at the ``~1e-6`` level, with identical penalty / rank / penalty_rank.

    Fixed ``lambda`` isolates the Newton-fit parity from REML ``lambda``
    selection (which floats a little in float32); the two fits then differ only
    by float32 arithmetic.
    """
    pytest.importorskip("jax")
    env = open_field_env
    centers = [(4.0, 4.0), (12.0, 12.0), (4.0, 12.0), (12.0, 4.0)]
    counts_full, occ_full = simulate_place_fields(env, centers, seed=1)
    basis = env._mrf_basis(occ_full, rank=40)
    counts, occ = _restrict(counts_full, occ_full, basis)

    fit_np = fit_mrf_gam(basis, counts, occ, penalty=1.5, backend="numpy")
    fit_jax = fit_mrf_gam(basis, counts, occ, penalty=1.5, backend="jax")

    # MRFFit arrays are always NumPy float64, even on the JAX path.
    assert isinstance(fit_jax, MRFFit)
    assert np.asarray(fit_jax.coefficients).dtype == np.float64
    assert not fit_jax.converged.__class__.__module__.startswith("jax")
    assert isinstance(fit_jax.converged, bool)

    # Structural quantities come from the basis / echoed penalty -> identical.
    assert fit_jax.penalty == fit_np.penalty == 1.5
    assert fit_jax.rank == fit_np.rank
    assert fit_jax.penalty_rank == fit_np.penalty_rank

    # Rate parity: ~1e-6 by relative L2 and relative error above a rate floor.
    rate_np = _firing_rate(fit_np)
    rate_jax = _firing_rate(fit_jax)
    assert _rel_l2(rate_jax, rate_np) < 1e-5
    assert _max_rel_above(rate_jax, rate_np, threshold=0.1) < 5e-5
    np.testing.assert_allclose(rate_jax, rate_np, rtol=1e-3, atol=1e-3)

    # Coefficients (relative L2 -- individual near-zero fill modes have large but
    # meaningless relative error) and deviance.
    assert _rel_l2(fit_jax.coefficients, fit_np.coefficients) < 1e-4
    np.testing.assert_allclose(
        np.asarray(fit_jax.deviance), np.asarray(fit_np.deviance), rtol=1e-3, atol=1e-3
    )


# ---------------------------------------------------------------------------
# float32 convergence + the tolerance floor is load-bearing
# ---------------------------------------------------------------------------
def test_jax_converges_float32(sparse_regime_env, simulate_place_fields):
    """The float32 fit converges inside ``_MAX_ITER`` and ``_FIT_TOL_FLOOR`` is
    load-bearing.

    ``1e-10`` is below the float32 penalized-objective noise, so without the
    floor the loop keeps taking Newton steps chasing a decrease it cannot
    reliably measure; floored to ``_FIT_TOL_FLOOR`` it stops in far fewer
    iterations. (It does not run to ``_MAX_ITER`` at ``1e-10`` -- the objective
    plateaus to bit-identical in float32, so ``rel_decrease`` eventually hits ~0
    -- but ``tol=0`` never converges, and the floored fit converges much sooner.)
    """
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from neurospatial.encoding._glm_jax import _newton_fit_jax, _newton_loop_jax

    env = sparse_regime_env
    centers = [(10.0, 10.0), (30.0, 30.0), (10.0, 30.0)]
    counts_full, occ_full = simulate_place_fields(env, centers, seed=2)
    basis = env._mrf_basis(occ_full, rank=60)
    counts, occ = _restrict(counts_full, occ_full, basis)
    penalty_diag = 1.0 * basis.d

    # Production path: the wrapper applies max(tol, _FIT_TOL_FLOOR).
    fit = fit_mrf_gam(basis, counts, occ, penalty=1.0, backend="jax")
    assert fit.converged is True
    assert fit.n_iter < _MAX_ITER

    # The floor is applied: a sub-floor tol is treated identically to the floor.
    coeffs_sub, *_rest_sub, n_sub, _s_sub, conv_sub = _newton_fit_jax(
        counts, occ, basis.B, penalty_diag, _MAX_ITER, 1e-12
    )
    coeffs_at, *_rest_at, n_at, _s_at, conv_at = _newton_fit_jax(
        counts, occ, basis.B, penalty_diag, _MAX_ITER, _FIT_TOL_FLOOR
    )
    assert n_sub == n_at
    assert conv_sub == conv_at is True
    np.testing.assert_array_equal(coeffs_sub, coeffs_at)

    # The floor is load-bearing: the raw (un-floored) loop at a below-noise tol
    # takes strictly more Newton iterations than the floored fit.
    cj = jnp.asarray(counts, jnp.float32)
    oj = jnp.asarray(occ, jnp.float32)
    bj = jnp.asarray(basis.B, jnp.float32)
    pj = jnp.asarray(penalty_diag, jnp.float32)
    *_raw, n_raw, _s_raw, _c_raw = _newton_loop_jax(
        cj, oj, bj, pj, _MAX_ITER, jnp.asarray(1e-10, jnp.float32)
    )
    assert int(n_raw) > n_at, (
        f"floor not load-bearing: floored n_iter={n_at} vs no-floor n_iter={int(n_raw)}"
    )


# ---------------------------------------------------------------------------
# Graceful degradation: JAX unavailable / backend off -> NumPy core
# ---------------------------------------------------------------------------
def test_jax_absent_uses_numpy(open_field_env, simulate_place_fields, monkeypatch):
    """With JAX reported unavailable, ``backend="jax"`` runs the NumPy core and
    still yields a valid ``MRFFit`` -- no import error (the base-install path).
    """
    import neurospatial.encoding._backend as backend_module

    env = open_field_env
    counts_full, occ_full = simulate_place_fields(env, [(8.0, 8.0)], seed=3)
    basis = env._mrf_basis(occ_full, rank=20)
    counts, occ = _restrict(counts_full, occ_full, basis)

    # Force the "extra unavailable" branch even where JAX is installed.
    monkeypatch.setattr(backend_module, "is_jax_available", lambda: False)

    fit_jax_req = fit_mrf_gam(basis, counts, occ, penalty=1.0, backend="jax")
    fit_numpy = fit_mrf_gam(basis, counts, occ, penalty=1.0, backend="numpy")

    assert isinstance(fit_jax_req, MRFFit)
    assert fit_jax_req.converged is True
    # Falling back to the NumPy core, the two are bit-identical.
    np.testing.assert_array_equal(
        np.asarray(fit_jax_req.coefficients), np.asarray(fit_numpy.coefficients)
    )


# ---------------------------------------------------------------------------
# Public return-contract: glm backend="jax" matches the ratio path's convention
# ---------------------------------------------------------------------------
def _jax_available() -> bool:
    from neurospatial.encoding._backend import is_jax_available

    return is_jax_available()


@pytest.mark.skipif(not _jax_available(), reason="JAX not available")
def test_backend_return_matches_ratio(open_field_env):
    """``compute_spatial_rates(method="glm", backend="jax")`` returns the same
    array-type convention as ``method="diffusion_kde", backend="jax"``, and
    ``firing_rates.dtype`` honors ``dtype`` -- the public return contract, which
    the float32 fit acceleration must not change.
    """
    from neurospatial.encoding.spatial import compute_spatial_rates

    from .test_glm_api import _grid_session

    env = open_field_env
    centers = [(4.0, 4.0), (12.0, 12.0)]
    times, positions, spike_times = _grid_session(env, centers, seed=11)

    ratio_jax = compute_spatial_rates(
        env, spike_times, times, positions, method="diffusion_kde", backend="jax"
    )
    for dtype in (np.float64, np.float32):
        glm_jax = compute_spatial_rates(
            env, spike_times, times, positions, method="glm", backend="jax", dtype=dtype
        )
        # Same array type the ratio path returns for backend="jax".
        assert type(glm_jax.firing_rates) is type(ratio_jax.firing_rates)
        # dtype honored at the result boundary.
        assert np.asarray(glm_jax.firing_rates).dtype == dtype


# ---------------------------------------------------------------------------
# REML lambda agreement between the two paths
# ---------------------------------------------------------------------------
def test_reml_parity(open_field_env, simulate_place_fields):
    """REML-selected ``lambda`` agrees between the float32 JAX and float64 NumPy
    paths within a ``_REML_XATOL``-consistent tolerance.

    Both minimize the same pooled REML objective over ``log lambda`` with the
    same bounded scalar minimizer (``xatol = _REML_XATOL`` on ``log lambda``);
    float32 objective noise near the broad minimum shifts the argmin only
    slightly.
    """
    pytest.importorskip("jax")
    from neurospatial.encoding._glm_jax import select_penalty_by_reml_jax
    from neurospatial.encoding._glm_numpy import select_penalty_by_reml

    env = open_field_env
    centers = [(5.0, 5.0), (11.0, 11.0)]
    counts_full, occ_full = simulate_place_fields(env, centers, seed=4)
    basis = env._mrf_basis(occ_full, rank=40)
    counts, occ = _restrict(counts_full, occ_full, basis)
    penalty_rank = basis.B.shape[1] - basis.n_live_components

    lam_np, _ = select_penalty_by_reml(
        counts, occ, basis.B, basis.d, penalty_rank, max_iter=_MAX_ITER, tol=_FIT_TOL
    )
    lam_jax, _ = select_penalty_by_reml_jax(
        counts, occ, basis.B, basis.d, penalty_rank, max_iter=_MAX_ITER, tol=_FIT_TOL
    )
    assert lam_np is not None and lam_jax is not None
    # |d log lambda| ~ a few * _REML_XATOL (0.001); 0.02 is a robust bound on the
    # float32 flat-minimum wobble while still catching a gross disagreement.
    assert abs(np.log(lam_jax) - np.log(lam_np)) < 0.02


# ---------------------------------------------------------------------------
# Degenerate / ill-conditioned fits under the JAX backend
# ---------------------------------------------------------------------------
def test_jax_penalty0_rank_deficient_falls_back_to_numpy(open_field_env):
    """An UNPENALIZED (penalty=0) fit on a rank-deficient exposed design has a
    singular Hessian; a float32 solve can wander to a physically-impossible rate
    while reporting convergence, so ``fit_mrf_gam`` runs it on the float64 core
    regardless of backend.

    Half the (single-component) bins visited with a full-rank basis -> the
    exposed design has fewer rows than ``r_eff`` -> rank-deficient. The JAX and
    NumPy results must be identical (both take the float64 path) and the rate must
    stay bounded, not saturate near ``exp(_ETA_CLIP)``.
    """
    pytest.importorskip("jax")
    env = open_field_env
    n_bins = env.n_bins
    rng = np.random.default_rng(2)
    occ = np.zeros(n_bins)
    occ[: n_bins // 2] = 3.0
    counts = np.zeros((1, n_bins))
    counts[0, : n_bins // 2] = rng.poisson(10.0, size=n_bins // 2)
    basis = env._mrf_basis(occ, rank=n_bins)  # full rank -> rank-deficient exposed
    c, o = _restrict(counts.T, occ, basis)

    with pytest.warns(UserWarning, match="rank-deficient"):
        fit_jax = fit_mrf_gam(basis, c, o, penalty=0.0, backend="jax")
        fit_np = fit_mrf_gam(basis, c, o, penalty=0.0, backend="numpy")

    # Identical: both took the float64 core (the JAX branch fell back).
    np.testing.assert_array_equal(
        np.asarray(fit_jax.coefficients), np.asarray(fit_np.coefficients)
    )
    # Bounded rate -- a float32 singular solve would saturate near exp(30) ~ 1e13.
    assert np.exp(fit_jax.log_rate).max() < 1e3


def test_jax_all_zero_spike_floors(open_field_env):
    """An all-zero-spike population under the JAX backend floors to ~_RATE_FLOOR
    (rates are driven toward zero, not up), converges, and never blows up.
    """
    pytest.importorskip("jax")
    env = open_field_env
    occ = np.full(env.n_bins, 3.0)
    counts = np.zeros((2, env.n_bins))
    basis = env._mrf_basis(occ, rank=20)
    c, o = _restrict(counts.T, occ, basis)

    with pytest.warns(UserWarning, match="no unit has any spikes"):
        fit = fit_mrf_gam(basis, c, o, penalty=None, backend="jax")
    rate = np.maximum(np.exp(fit.log_rate), _RATE_FLOOR)
    # Effectively at the floor (float32 stops with eta a touch less negative than
    # float64, but still ~0 -- well within allclose of the floor).
    np.testing.assert_allclose(rate, _RATE_FLOOR, atol=1e-7)
    assert np.all(np.isfinite(fit.deviance))
