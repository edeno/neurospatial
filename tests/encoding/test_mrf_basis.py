"""Live-component eigenbasis resolver for the penalized-Poisson GAM.

The resolver reuses the cached finite-volume geometry and symmetric
eigensolver (untouched) to build a reduced-rank penalty basis. These tests pin
the ``MRFBasis`` contract directly -- exact structural nulls, mass-normalized
intercepts, positivity-selected fills, dead-component exclusion, and the
cached-eigenbasis reuse / iterative-growth behavior -- against independent
recomputation, not the resolver's own internals.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.ops import diffusion as diff
from neurospatial.ops.diffusion import (
    _DEFAULT_MAX_RANK,
    MRFBasis,
    _assemble_W,
    _components_from_W,
    _ensure_global_modes,
    _finite_volume_geometry,
    _null_mode_mask,
    _symmetric_conjugate,
    _symmetric_eigenbasis,
    select_live_basis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _geometry(env):
    """(W, volumes, n_components, labels) from an env's finite-volume geometry."""
    graph, volumes = _finite_volume_geometry(env)
    volumes = np.asarray(volumes, dtype=np.float64)
    W = _assemble_W(graph, len(volumes))
    n_components, labels = _components_from_W(W)
    return W, volumes, n_components, labels


def _full_eigs(env):
    """Full (Q, Lam, mode_comp) for an env -- the selector's cached inputs."""
    W, volumes, _n_components, labels = _geometry(env)
    S = _symmetric_conjugate(W, volumes)
    Lam, Q = _symmetric_eigenbasis(S, None)
    mode_comp = np.array([labels[np.flatnonzero(col)[0]] for col in Q.T], dtype=np.intp)
    return Q, Lam, mode_comp, volumes, labels


def _non_null_eigs_sorted(Lam, mode_comp, live_comp):
    """Sorted live-mode eigenvalues with one structural null (the smallest per
    component) dropped -- an independent oracle for the fill weights.

    This drops the smallest eigenvalue of each live component (its constant
    null), a different computation from the resolver's overlap-based
    identification, so comparing against it is not a tautology.
    """
    keep = []
    for c in live_comp:
        idx = np.flatnonzero(mode_comp == c)  # ascending Lam
        keep.extend(Lam[idx[1:]])  # drop the component's smallest mode (its null)
    return np.sort(keep)


def _fresh_holder(env):
    """A pristine empty resolver holder (mirrors the cached-property initial)."""
    n = env.n_bins
    return {
        "Q": np.zeros((n, 0)),
        "Lam": np.zeros(0),
        "mode_comp": np.zeros(0, dtype=np.intp),
        "G": 0,
    }


# ---------------------------------------------------------------------------
# Shape / structural contract
# ---------------------------------------------------------------------------
def test_mrfbasis_shapes(open_field_env):
    """B is (n_live_bins, r_eff), d is (r_eff,), r_eff matches the formula."""
    occ = np.ones(open_field_env.n_bins)
    basis = open_field_env._mrf_basis(occ, rank=None)

    n_live_bins = basis.live_bins.size
    r_eff = max(basis.n_live_components, min(n_live_bins, _DEFAULT_MAX_RANK))
    assert basis.B.shape == (n_live_bins, r_eff)
    assert basis.d.shape == (r_eff,)
    assert basis.B.shape[1] == r_eff == basis.d.shape[0]
    assert basis.n_live_components == 1  # single connected component


def test_structural_nulls_zeroed(open_field_env):
    """d[:n_live] is bit-exact 0.0; exactly n_live entries are zero; fills are
    genuine (positive-eigenvalue) modes; penalty_rank is exact."""
    occ = np.ones(open_field_env.n_bins)
    basis = open_field_env._mrf_basis(occ, rank=20)

    # Single component -> 1 intercept (d == 0) + 19 fill modes.
    assert basis.n_live_components == 1
    assert basis.B.shape[1] == 20
    assert basis.d[0] == 0.0  # bit-exact structural null, not thresholded
    # Exactly the n_live structural nulls are zero; every fill on this
    # well-connected env is a genuine nonconstant mode (positive eigenvalue).
    assert int(np.count_nonzero(basis.d == 0.0)) == basis.n_live_components
    assert np.all(basis.d[1:] > 0.0)

    penalty_rank = basis.B.shape[1] - basis.n_live_components
    assert penalty_rank == 19  # r_eff - n_live_components, exact by construction


def test_live_component_budget(two_component_env):
    """Every B column indexes a live bin; the budget buys r_eff LIVE modes."""
    _W, _vol, _nc, labels = _geometry(two_component_env)
    live_comp = 0  # visit only the first component
    occ = np.zeros(two_component_env.n_bins)
    occ[labels == live_comp] = 1.0

    basis = two_component_env._mrf_basis(occ, rank=None)

    assert basis.n_live_components == 1
    # Every live bin belongs to the live component (dead component excluded).
    assert np.all(labels[basis.live_bins] == live_comp)
    n_live_bins = int((labels == live_comp).sum())
    assert basis.B.shape[1] == max(1, min(n_live_bins, _DEFAULT_MAX_RANK))


def test_live_bins_cover_whole_component_not_just_visited_bins(two_component_env):
    """live_bins is component-based: a component with ANY occupancy contributes
    ALL its bins, including bins whose own occupancy is 0 (the real-data case)."""
    _W, _vol, _nc, labels = _geometry(two_component_env)
    comp_bins = np.flatnonzero(labels == 0)
    occ = np.zeros(two_component_env.n_bins)
    occ[comp_bins[0]] = 5.0  # visit ONE bin; leave the rest of the component at 0

    basis = two_component_env._mrf_basis(occ, rank=None)

    assert basis.n_live_components == 1
    # All bins of the visited component are live, not just the visited one -- a
    # bin-based (occupancy > 0) selection would return only comp_bins[0].
    np.testing.assert_array_equal(basis.live_bins, comp_bins)
    # The intercept is constant and nonzero even on the unvisited bins.
    intercept = basis.B[:, 0]
    assert np.all(intercept > 0.0)
    np.testing.assert_allclose(intercept, intercept[0])


def test_two_3node_paths_rank2(two_path_env):
    """Two visited disjoint paths at rank=2 -> the all-null (r==0) basis."""
    occ = np.ones(two_path_env.n_bins)
    basis = two_path_env._mrf_basis(occ, rank=2)

    assert basis.n_live_components == 2
    assert basis.B.shape[1] == 2  # r_eff == n_live_components
    assert np.all(basis.d == 0.0)  # every column is a structural null


def test_zero_occupancy_empty_basis(open_field_env):
    """All-zero occupancy -> the empty (0, 0) basis (degenerate case)."""
    occ = np.zeros(open_field_env.n_bins)
    basis = open_field_env._mrf_basis(occ, rank=None)

    assert basis.B.shape == (0, 0)
    assert basis.live_bins.size == 0
    assert basis.n_live_components == 0
    assert basis.d.shape == (0,)


def test_rank_clamped_both_ways(open_field_env):
    """rank below n_live_components floors up; huge rank clamps to n_live_bins."""
    occ = np.ones(open_field_env.n_bins)
    n_live_bins = int((occ > 0).sum())

    low = open_field_env._mrf_basis(occ, rank=1)
    assert low.B.shape[1] == low.n_live_components  # floored to n_live_components

    high = open_field_env._mrf_basis(occ, rank=10**9)
    assert high.B.shape[1] == n_live_bins  # clamped down to n_live_bins


# ---------------------------------------------------------------------------
# Intercepts: exact mass-normalized constant per live component
# ---------------------------------------------------------------------------
def test_intercept_is_constant_per_component(polar_env):
    """Each intercept column is constant on its component == 1/sqrt(sum vol_c).

    Asserted on a non-uniform-volume env where the plain L2-normalized
    indicator 1_c/||1_c|| would give a different (wrong) constant.
    """
    _W, volumes, _nc, labels = _geometry(polar_env)
    occ = np.ones(polar_env.n_bins)
    basis = polar_env._mrf_basis(occ, rank=8)

    lbl_live = labels[basis.live_bins]
    for j in range(basis.n_live_components):
        col = basis.B[:, j]
        comp = np.unique(lbl_live[col != 0.0])
        assert comp.size == 1  # nonzero on exactly one component
        c = int(comp[0])
        on = col[lbl_live == c]
        expected = 1.0 / np.sqrt(volumes[labels == c].sum())
        np.testing.assert_allclose(on, expected)  # constant + mass-normalized
        assert np.all(col[lbl_live != c] == 0.0)  # zero elsewhere
        # The L2-normalized indicator would differ on non-uniform volumes.
        l2_const = 1.0 / np.sqrt((labels == c).sum())
        assert not np.isclose(expected, l2_const)


# ---------------------------------------------------------------------------
# Fills: positivity-selected smallest-lambda live modes, M^{-1/2}Q applied
# ---------------------------------------------------------------------------
def test_fill_applies_inv_sqrt_volume(polar_env):
    """Fill columns equal (M^{-1/2}Q)[live_bins, fill_idx], not Q itself.

    Compared against the resolver's OWN cached eigenbasis, not an independent
    solve: the polar env has degenerate eigenspaces (rotational symmetry), so
    two eigensolves pick different (equally valid) bases within them -- the
    well-defined invariant is that ``B_fill`` is ``M^{-1/2}`` applied to the
    modes the resolver actually selected.
    """
    _W, volumes, _nc, labels = _geometry(polar_env)
    occ = np.ones(polar_env.n_bins)
    basis = polar_env._mrf_basis(occ, rank=8)

    n_live = basis.n_live_components
    n_fill = basis.B.shape[1] - n_live
    # Recompute the fill block from the resolver's own persisted eigenbasis.
    holder = polar_env._mrf_eigenbasis
    Q, _Lam, mode_comp = holder["Q"], holder["Lam"], holder["mode_comp"]
    live_comp = np.flatnonzero(
        np.bincount(labels, weights=occ, minlength=int(labels.max()) + 1) > 0.0
    )
    is_null = _null_mode_mask(Q, mode_comp, volumes, labels, live_comp)
    fill_idx = np.flatnonzero(np.isin(mode_comp, live_comp) & ~is_null)[:n_fill]
    inv_sqrt = 1.0 / np.sqrt(volumes)
    expected = (inv_sqrt[:, None] * Q[:, fill_idx])[basis.live_bins, :]

    np.testing.assert_allclose(basis.B[:, n_live:], expected, atol=1e-10)
    # And it is NOT the raw Q (they differ where volumes vary).
    raw = Q[basis.live_bins][:, fill_idx]
    assert not np.allclose(basis.B[:, n_live:], raw)


def test_fill_exercised_rank_gt_nlive(asymmetric_two_component_env):
    """rank = n_live + 2 selects the 2 smallest non-null live modes."""
    env = asymmetric_two_component_env
    _Q, Lam, mode_comp, _vol, labels = _full_eigs(env)
    occ = np.ones(env.n_bins)
    n_live_components = 2
    basis = env._mrf_basis(occ, rank=n_live_components + 2)

    assert basis.n_live_components == n_live_components
    n_fill = basis.B.shape[1] - n_live_components
    assert n_fill == 2

    # Fill weights are positive (genuine nonconstant modes) and are the 2
    # smallest non-null live eigenvalues -- checked against an independent oracle
    # that drops one structural null (smallest per component).
    fill_d = basis.d[n_live_components:]
    assert np.all(fill_d > 0.0)
    non_null = _non_null_eigs_sorted(Lam, mode_comp, np.unique(labels))
    np.testing.assert_allclose(np.sort(fill_d), non_null[:2])

    # No fill column is a near-constant duplicate of any intercept.
    for f in range(n_live_components, basis.B.shape[1]):
        for i in range(n_live_components):
            fi, ic = basis.B[:, f], basis.B[:, i]
            denom = np.linalg.norm(fi) * np.linalg.norm(ic)
            corr = abs(float(fi @ ic) / denom) if denom > 0 else 0.0
            assert corr < 0.99


def test_both_components_present_asymmetric(asymmetric_two_component_env):
    """At rank == n_live_components both components get exactly one intercept."""
    env = asymmetric_two_component_env
    _W, _vol, _nc, labels = _geometry(env)
    occ = np.ones(env.n_bins)
    basis = env._mrf_basis(occ, rank=2)

    assert basis.n_live_components == 2
    assert basis.B.shape[1] == 2
    assert np.all(basis.d == 0.0)
    lbl_live = labels[basis.live_bins]
    represented = {int(np.unique(lbl_live[basis.B[:, j] != 0.0])[0]) for j in range(2)}
    assert represented == {0, 1}  # both live components represented


def test_multiple_live_and_dead_components(four_component_env):
    """Two live + two dead components jointly: two intercepts, live_bins span
    exactly the live components, fills drawn only from live modes."""
    env = four_component_env
    _W, _vol, n_components, labels = _geometry(env)
    assert n_components == 4
    live = [0, 2]
    occ = np.zeros(env.n_bins)
    occ[np.isin(labels, live)] = 1.0

    basis = env._mrf_basis(occ, rank=None)

    assert basis.n_live_components == 2
    np.testing.assert_array_equal(
        basis.live_bins, np.flatnonzero(np.isin(labels, live))
    )
    # Exactly two structural nulls (one intercept per live component); no dead
    # bins leak into live_bins.
    assert int(np.count_nonzero(basis.d == 0.0)) == 2
    assert np.all(np.isin(labels[basis.live_bins], live))


def test_fill_selection_is_scale_invariant():
    """A genuine nonconstant mode with a physically-tiny eigenvalue is kept as a
    fill, not misread as a null.

    Laplacian eigenvalues carry units of 1/length**2, so on a large/coarse env a
    real Fiedler mode can be positive yet below any absolute cutoff. The null is
    identified structurally (overlap with the mass-weighted constant), so the
    tiny-eigenvalue mode is still selected as a fill.
    """
    volumes = np.ones(2)
    labels = np.zeros(2, dtype=np.intp)
    occ = np.ones(2)
    # 2-bin single component: column 0 = mass-weighted constant (the null),
    # column 1 = the nonconstant Fiedler, with a physically-tiny eigenvalue.
    Q = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    Lam = np.array([0.0, 1e-13])  # Fiedler eigenvalue below any absolute cutoff
    mode_comp = np.array([0, 0], dtype=np.intp)

    basis = select_live_basis(Q, Lam, mode_comp, volumes, labels, occ, rank=2)

    assert basis.n_live_components == 1
    assert basis.B.shape == (2, 2)  # intercept + the tiny-eigenvalue fill
    assert basis.d[0] == 0.0
    assert basis.d[1] == 1e-13  # the fill weight is the true (tiny) eigenvalue
    np.testing.assert_allclose(basis.B[:, 1], Q[:, 1])  # M^{-1/2}Q, volumes == 1


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_occupancy_raises(two_component_env, bad_value):
    """A non-finite occupancy bin is rejected loudly, not silently swallowed.

    Without validation a NaN bin makes its whole component's summed occupancy
    NaN (``NaN > 0`` is False), silently dropping the component -- or, if it is
    the only live component, returning the empty basis, indistinguishable from
    genuine zero occupancy.
    """
    occ = np.ones(two_component_env.n_bins)
    occ[0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        two_component_env._mrf_basis(occ, rank=None)


def test_negative_occupancy_raises(two_component_env):
    """Negative occupancy is rejected: bincount sums with sign, so a component
    summing below zero would be silently misclassified as dead."""
    occ = np.ones(two_component_env.n_bins)
    occ[0] = -5.0
    with pytest.raises(ValueError, match="non-negative"):
        two_component_env._mrf_basis(occ, rank=None)


def test_null_absent_from_eigenbasis_raises():
    """If a live component's constant null is missing from Q, the selector fails
    loudly rather than silently promoting a smoothness mode to 'null'."""
    volumes = np.ones(2)
    labels = np.zeros(2, dtype=np.intp)
    occ = np.ones(2)
    # Only the nonconstant (Fiedler) mode is present -- the null is absent.
    Q = np.array([[1.0], [-1.0]]) / np.sqrt(2.0)
    Lam = np.array([1.0])
    mode_comp = np.array([0], dtype=np.intp)
    with pytest.raises(ValueError, match="null"):
        select_live_basis(Q, Lam, mode_comp, volumes, labels, occ, rank=1)


def test_clipped_fill_eigenvalue_raises():
    """A structurally non-null fill whose eigenvalue was clipped to <= 0 must not
    silently become an unpenalized direction (d == 0) -- the selector raises."""
    volumes = np.ones(2)
    labels = np.zeros(2, dtype=np.intp)
    occ = np.ones(2)
    # column 0 = constant null, column 1 = nonconstant Fiedler whose eigenvalue
    # was clipped to 0 (a numerically (near-)disconnected component).
    Q = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    Lam = np.array([0.0, 0.0])
    mode_comp = np.array([0, 0], dtype=np.intp)
    with pytest.raises(ValueError, match="nonpositive"):
        select_live_basis(Q, Lam, mode_comp, volumes, labels, occ, rank=2)


def test_growth_continues_until_null_covered(open_field_env, monkeypatch):
    """If a component's null is absent from a truncated eigenbasis (a positive
    mode sorted before it), _mrf_basis keeps growing rather than handing the
    selector a null-less basis and dying on its raise."""
    env = open_field_env  # single component
    occ = np.ones(env.n_bins)
    q_full, lam_full, mc_full, _vol, _lab = _full_eigs(env)  # col 0 == the null

    calls = {"n": 0}

    def fake_ensure(holder, S, labels, needed_G, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # First solve OMITS the null (column 0) -- the pathology.
            sel = slice(1, needed_G + 1)
            return q_full[:, sel], lam_full[sel], mc_full[sel]
        return q_full, lam_full, mc_full  # growth exposes the full basis

    monkeypatch.setattr(diff, "_ensure_global_modes", fake_ensure)

    basis = env._mrf_basis(occ, rank=5)  # must NOT raise

    assert calls["n"] >= 2  # it grew past the null-less first solve
    assert basis.B.shape[1] == 5
    assert basis.d[0] == 0.0
    assert np.all(basis.d[1:] > 0.0)


def test_insufficient_fill_modes_raises():
    """If the eigenbasis lacks enough non-null live modes for n_fill, the
    selector raises rather than silently returning fewer than r_eff columns."""
    volumes = np.ones(2)
    labels = np.zeros(2, dtype=np.intp)
    occ = np.ones(2)
    # Only the null mode is present, but rank=2 demands one fill.
    Q = np.ones((2, 1)) / np.sqrt(2.0)
    Lam = np.array([0.0])
    mode_comp = np.array([0], dtype=np.intp)

    with pytest.raises(ValueError, match="fill"):
        select_live_basis(Q, Lam, mode_comp, volumes, labels, occ, rank=2)


# ---------------------------------------------------------------------------
# First-solve rank floor + iterative growth
# ---------------------------------------------------------------------------
def test_g_floors_at_n_components(four_component_env):
    """One live + several dead components with r_eff < n_components: no raise."""
    env = four_component_env
    _W, _vol, n_components, labels = _geometry(env)
    assert n_components == 4
    occ = np.zeros(env.n_bins)
    occ[labels == 0] = 1.0  # only the smallest (2-bin) component is live
    r_eff_live = int((labels == 0).sum())
    assert r_eff_live < n_components  # would raise without the G floor

    basis = env._mrf_basis(occ, rank=None)  # must NOT raise
    assert basis.n_live_components == 1
    assert basis.B.shape[1] == r_eff_live


def test_growth_past_dead_modes(dead_dominant_env):
    """A large dead component forces iterative G growth to expose live fills."""
    env = dead_dominant_env
    _W, _vol, _nc, labels = _geometry(env)
    live_comp = int(np.argmin(np.bincount(labels)))  # the small live block
    occ = np.zeros(env.n_bins)
    occ[labels == live_comp] = 1.0
    n_live_bins = int((labels == live_comp).sum())

    basis = env._mrf_basis(occ, rank=None)

    assert basis.n_live_components == 1
    assert basis.B.shape[1] == n_live_bins  # all live modes exposed
    # The exposed fills are genuine nonconstant modes (positive eigenvalues).
    assert np.all(basis.d[1:] > 0.0)
    assert np.all(labels[basis.live_bins] == live_comp)


# ---------------------------------------------------------------------------
# Cached-eigenbasis resolver behavior (reuse / purity / mode recovery)
# ---------------------------------------------------------------------------
def test_reuses_cached_diffusion_geometry(open_field_env):
    """_mrf_basis reuses the cached _diffusion_geometry (no rebuild)."""
    env = open_field_env
    occ = np.ones(env.n_bins)
    geom_before = env._diffusion_geometry
    version_before = env._state_version

    env._mrf_basis(occ, rank=10)
    env._mrf_basis(occ, rank=10)

    assert env._state_version == version_before
    assert env._diffusion_geometry is geom_before  # same cached object


def test_reuses_eigensolve(sparse_regime_env, monkeypatch):
    """A second call at the same-or-smaller rank does not re-eigensolve."""
    env = sparse_regime_env
    occ = np.ones(env.n_bins)

    calls = {"n": 0}
    real = diff._symmetric_eigenbasis

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(diff, "_symmetric_eigenbasis", counting)

    env._mrf_basis(occ, rank=50)  # sparse regime -> persisted holder
    g_after_first = env._mrf_eigenbasis["G"]
    assert g_after_first == 50
    first_calls = calls["n"]
    assert first_calls > 0

    env._mrf_basis(occ, rank=50)  # same rank -> cache hit
    env._mrf_basis(occ, rank=30)  # smaller rank -> still a cache hit

    assert calls["n"] == first_calls  # no additional eigensolve
    assert env._mrf_eigenbasis["G"] == g_after_first  # holder G unchanged


def test_holder_resets_on_clear_cache(sparse_regime_env):
    """The grown eigenbasis holder is dropped wholesale on clear_cache (the
    versioned-cache invalidation promise), so it never outlives a geometry."""
    env = sparse_regime_env
    occ = np.ones(env.n_bins)
    env._mrf_basis(occ, rank=50)
    assert env._mrf_eigenbasis["G"] == 50  # grown

    env.clear_cache(cached_properties=True)

    holder = env._mrf_eigenbasis
    assert holder["G"] == 0  # freshly re-initialized to the empty state
    assert holder["Q"].shape == (env.n_bins, 0)


def test_selector_is_pure(open_field_env, monkeypatch):
    """select_live_basis never eigensolves -- it only masks cached modes."""
    Q, Lam, mode_comp, volumes, labels = _full_eigs(open_field_env)
    occ = np.ones(open_field_env.n_bins)

    calls = {"n": 0}
    real = diff._symmetric_eigenbasis

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(diff, "_symmetric_eigenbasis", counting)

    basis = select_live_basis(Q, Lam, mode_comp, volumes, labels, occ, rank=10)
    assert calls["n"] == 0
    assert isinstance(basis, MRFBasis)


def test_mode_component_recovery(two_component_env):
    """mode_comp[m] equals the W-component of every nonzero row of Q[:, m]."""
    _W, volumes, _nc, labels = _geometry(two_component_env)
    S = _symmetric_conjugate(volumes=volumes, W=_W)
    holder = _fresh_holder(two_component_env)

    Q, _Lam, mode_comp = _ensure_global_modes(holder, S, labels, 6)

    for m in range(Q.shape[1]):
        nonzero_labels = np.unique(labels[Q[:, m] != 0.0])
        assert nonzero_labels.size == 1
        assert int(nonzero_labels[0]) == int(mode_comp[m])


def test_dense_basis_not_persisted(open_field_env):
    """A request past dense_fraction*n returns a valid basis but does not
    grow the holder (the n x n dense basis is call-local, never stored)."""
    env = open_field_env
    occ = np.ones(env.n_bins)
    n = env.n_bins

    holder = env._mrf_eigenbasis
    assert holder["G"] == 0  # initial empty state
    assert holder["Q"].shape == (n, 0)

    # rank=40 on a 64-bin env: G=40 >= 0.5*64=32 -> dense (transient).
    basis = env._mrf_basis(occ, rank=40)

    assert basis.B.shape == (n, 40)
    assert env._mrf_eigenbasis["G"] == 0  # holder stayed at its sparse value
