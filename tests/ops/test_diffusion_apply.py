"""Behavior-preservation gate for the matrix-free diffusion apply-path.

``env.diffuse`` (the cached truncated-eigenbasis apply) must reproduce the shipped
dense diffusion operator to within a stated truncation tolerance. These tests
compare against the frozen pre-refactor dense baseline
(``data/diffusion_perf_baseline.pkl``, captured by ``generate_diffusion_baseline``)
-- not against a live recompute -- so equivalence is asserted against an
independent oracle, not a tautology.

The load-bearing invariants (linearity, exact mass conservation / null-mode
retention under truncation, W-component support gates, single growable cache, no
cache poisoning, JAX grad) are checked directly.
"""

from __future__ import annotations

import pickle
import time
import tracemalloc
import warnings

import numpy as np
import pytest
from diffusion_fixtures import (
    BASELINE_PATH,
    GEOMETRIES,
    MODES,
    PERF_SIGMA,
    build_grid_2d_split,
    build_perf_grid,
    build_scaling_grid,
    make_fields,
    source_bin,
)

from neurospatial.ops.diffusion import (
    _HEAT_KERNEL_DENSE_FRACTION,
    _HEAT_KERNEL_RANK_TOL,
    _adaptive_symmetric_basis,
    _assemble_W,
    _components_from_W,
    _finite_volume_geometry,
    _symmetric_conjugate,
    component_support_mask,
)

# Truncation tolerance for the M-weighted / dense-relative equivalence. The
# theory bounds dropped-mode energy by tol (1e-6); the aggregate over modes just
# above the cutoff can be a small multiple, so assert a comfortable margin.
_TRUNC_MNORM_TOL = 2e-5
_ORACLE_FIELDS = ("point", "nonneg", "signed")


# ---------------------------------------------------------------------------
# Baseline + fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def baseline() -> dict:
    with BASELINE_PATH.open("rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def cases_by_key(baseline) -> dict:
    return {(c["geom"], c["sigma_label"], c["mode"]): c for c in baseline["cases"]}


def _build_env(geom_name: str):
    """Fresh environment for ``geom_name`` (cold caches)."""
    env, source_coord = GEOMETRIES[geom_name][0]()
    return env, source_coord


def _m_norm_rel(a, ref, volumes) -> float:
    """M-weighted relative error ``||a - ref||_M / ||ref||_M``."""
    num = float(np.sqrt(np.sum(volumes * (a - ref) ** 2)))
    den = float(np.sqrt(np.sum(volumes * ref**2)))
    return num / den if den > 0 else num


def _geometry(env):
    graph, volumes = _finite_volume_geometry(env)
    volumes = np.asarray(volumes, dtype=np.float64)
    W = _assemble_W(graph, len(volumes))
    return W, volumes


# ---------------------------------------------------------------------------
# Full-rank + truncated equivalence to the dense baseline
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("geom", list(GEOMETRIES))
@pytest.mark.parametrize("mode", MODES)
def test_apply_matches_dense_full_rank(geom, mode, cases_by_key):
    """env.diffuse == dense kernel @ F within rtol=1e-8 at (near-)full rank.

    The 'small' sigma forces (near-)full rank on every fixture, so the apply is
    exact up to numerical precision -- verified on polar/mesh (worst kappa(M)).
    """
    env, _ = _build_env(geom)
    case = cases_by_key[(geom, "small", mode)]
    sigma = case["sigma"]
    for name in _ORACLE_FIELDS:
        field = case["fields"][name]
        out = np.asarray(env.diffuse(field, sigma, mode=mode))
        ref = case["kernel_at_field"][name]
        np.testing.assert_allclose(
            out, ref, rtol=1e-8, atol=1e-10, err_msg=f"{geom}/{mode}/{name}"
        )


@pytest.mark.parametrize("geom", list(GEOMETRIES))
@pytest.mark.parametrize("mode", MODES)
def test_apply_matches_dense_truncated(geom, mode, cases_by_key):
    """Truncated env.diffuse == dense within tol in the M-weighted norm.

    The 'large' sigma truncates (rank << n) on every fixture; the deviation is
    dense-relative in the M-weighted norm (absorbs kappa(M)), verified on
    polar/mesh, not a raw per-bin bound.
    """
    env, _ = _build_env(geom)
    _W, volumes = _geometry(env)
    case = cases_by_key[(geom, "large", mode)]
    sigma = case["sigma"]
    for name in _ORACLE_FIELDS:
        field = case["fields"][name]
        out = np.asarray(env.diffuse(field, sigma, mode=mode))
        ref = case["kernel_at_field"][name]
        rel = _m_norm_rel(out, ref, volumes)
        assert rel < _TRUNC_MNORM_TOL, f"{geom}/{mode}/{name}: M-norm {rel:.2e}"


def test_env_diffuse_is_linear():
    """diffuse(a*F1 + b*F2) == a*diffuse(F1) + b*diffuse(F2) on SIGNED fields.

    Guards the linearity contract: no positivity projection anywhere in the
    apply-path (checked at both truncated and full rank, all modes).
    """
    a, b = 1.7, -0.4
    for geom in ("grid_2d", "polar", "mesh", "grid_2d_split"):
        env, source_coord = _build_env(geom)
        src = source_bin(env, source_coord)
        fields = make_fields(env, src)
        f1, f2 = fields["signed"], fields["nonneg"]
        for mode in MODES:
            for sigma in (2.0, 12.0):  # near-full-rank and truncated
                lhs = np.asarray(env.diffuse(a * f1 + b * f2, sigma, mode=mode))
                rhs = a * np.asarray(
                    env.diffuse(f1, sigma, mode=mode)
                ) + b * np.asarray(env.diffuse(f2, sigma, mode=mode))
                np.testing.assert_allclose(
                    lhs, rhs, rtol=1e-9, atol=1e-11, err_msg=f"{geom}/{mode}/{sigma}"
                )


# ---------------------------------------------------------------------------
# compute_kernel unchanged (byte-identical dense-expm path)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("geom", list(GEOMETRIES))
def test_compute_kernel_unchanged(geom, cases_by_key):
    """compute_kernel's ACTION matches the frozen pre-refactor dense baseline.

    compute_kernel is left on its dense-expm path; its action on the field set
    (a linear operator is determined by its action; the set includes a full
    kernel column via the unit point source, plus non-negative and signed
    fields) must match the pre-refactor output to numerical precision (rtol
    1e-12 absorbs cross-platform BLAS ULP differences).
    """
    env, _ = _build_env(geom)
    for sigma_label in ("small", "large"):
        for mode in MODES:
            case = cases_by_key[(geom, sigma_label, mode)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kernel = env.compute_kernel(case["sigma"], mode=mode, cache=False)
            for name in _ORACLE_FIELDS:
                live = kernel @ case["fields"][name]
                np.testing.assert_allclose(
                    live,
                    case["kernel_at_field"][name],
                    rtol=1e-12,
                    atol=1e-14,
                    err_msg=f"{geom}/{sigma_label}/{mode}/{name}",
                )


# ---------------------------------------------------------------------------
# Null-mode retention: exact mass conservation under truncation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("geom", list(GEOMETRIES))
def test_null_mode_retained(geom):
    """Resolved rank >= n_components; H_trunc @ 1 == 1 and volumes @ H_trunc ==
    volumes EXACTLY under truncation (mass conservation, linearity foundation)."""
    env, _ = _build_env(geom)
    W, volumes = _geometry(env)
    S = _symmetric_conjugate(W, volumes)
    n_components, _labels = _components_from_W(W)
    sigma = GEOMETRIES[geom][2]  # sigma_large (truncated)
    rank, *_ = _adaptive_symmetric_basis(
        S,
        sigma,
        tol=_HEAT_KERNEL_RANK_TOL,
        dense_fraction=_HEAT_KERNEL_DENSE_FRACTION,
        n_components=n_components,
    )
    assert rank is None or rank >= n_components

    ones = np.ones(env.n_bins)
    # average(1) == 1 (row-stochastic, null mode) and transition conserves mass.
    avg_one = np.asarray(env.diffuse(ones, sigma, mode="average"))
    np.testing.assert_allclose(avg_one, 1.0, rtol=1e-9, atol=1e-9)

    # Σ_i M_i H_trunc[i, j] == M_j : a density kernel column integrates to 1 under
    # M, i.e. diffuse(delta_j, density) integrates (under M) to 1 exactly.
    rng = np.random.default_rng(0)
    counts = rng.random(env.n_bins)
    dens = np.asarray(env.diffuse(counts, sigma, mode="density"))
    # count -> density conserves the M-integral of the input total.
    np.testing.assert_allclose(
        np.sum(volumes * dens), np.sum(counts), rtol=1e-9, atol=1e-9
    )
    # transition conserves the plain sum exactly.
    trans = np.asarray(env.diffuse(counts, sigma, mode="transition"))
    np.testing.assert_allclose(np.sum(trans), np.sum(counts), rtol=1e-9, atol=1e-9)


def test_grid_independence_preserved():
    """Measured physical sigma == bandwidth still holds via env.diffuse
    (regression: the apply-path must not reintroduce grid dependence)."""
    bandwidth = 5.0
    for h in (1.0, 2.0, 4.0):
        n = round(60 / h)
        edges = np.linspace(0.0, 60.0, n + 1)
        from neurospatial import Environment

        env = Environment.from_grid_mask(
            active_mask=np.ones(n, dtype=bool), grid_edges=(edges,)
        )
        src = int(np.argmin(np.abs(env.bin_centers[:, 0] - 30.0)))
        point = np.zeros(env.n_bins)
        point[src] = 1.0
        # transition mode: mass-conserving, the diffused point is a probability.
        col = np.asarray(env.diffuse(point, bandwidth, mode="transition"))
        col = np.clip(col, 0.0, None)
        p = col / col.sum()
        x = env.bin_centers[:, 0]
        mean = np.sum(p * x)
        measured = float(np.sqrt(np.sum(p * (x - mean) ** 2)))
        assert measured == pytest.approx(bandwidth, rel=0.02), f"h={h}: {measured}"


def test_no_leakage_truncated():
    """Point source in one component of a split grid: 0 mass crosses the wall
    under truncation (component-local eigenbasis modes)."""
    env, source_coord = build_grid_2d_split()
    W, _volumes = _geometry(env)
    n_components, labels = _components_from_W(W)
    assert n_components == 2, "split grid must have two disconnected halves"
    src = source_bin(env, source_coord)
    left = labels[src]
    point = np.zeros(env.n_bins)
    point[src] = 1.0
    for mode in MODES:
        out = np.asarray(env.diffuse(point, 8.0, mode=mode))  # truncated
        other = out[labels != left]
        assert np.max(np.abs(other)) < 1e-12, f"{mode}: mass leaked across wall"


# ---------------------------------------------------------------------------
# Denominator / support policy (binned + resample gates, KDE magnitude gate)
# ---------------------------------------------------------------------------
def test_denominator_support_no_spurious_nan():
    """Strict support gates use W-component support (exact, truncation-proof):
    no spurious NaN where the dense denominator was tiny-positive; value matches
    dense where den >> tol; the numerator is not floored (signed-safe)."""
    from neurospatial.encoding._smoothing import _binned_gate

    env, source_coord = _build_env("grid_2d")
    src = source_bin(env, source_coord)
    fields = make_fields(env, src)
    W, _volumes = _geometry(env)
    n_components, labels = _components_from_W(W)

    # A partially-masked intensive rate (NaN block); one connected component here,
    # so with any valid bin, EVERY bin is supported -> no NaN at all.
    rate = fields["masked"].copy()  # some NaN
    valid = np.isfinite(rate)
    gated = _binned_gate(env, rate, bandwidth=6.0)
    support = component_support_mask(labels, n_components, valid)
    # No spurious NaN: finite exactly on the supported bins.
    assert np.all(np.isfinite(gated[support]))
    assert np.all(np.isnan(gated[~support]))

    # Value matches the dense masked-average where the dense denominator >> tol.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        avg_kernel = env.compute_kernel(6.0, mode="average", cache=False)
    rate_filled = np.where(valid, rate, 0.0)
    dense_num = avg_kernel @ rate_filled
    dense_den = avg_kernel @ valid.astype(float)
    healthy = dense_den > 1e-3  # comfortably above the truncation floor
    with np.errstate(divide="ignore", invalid="ignore"):
        dense_rate = np.where(healthy, dense_num / dense_den, np.nan)
    np.testing.assert_allclose(
        gated[healthy], dense_rate[healthy], rtol=1e-4, atol=1e-6
    )

    # Numerator NOT floored: a signed field survives the masked average signed.
    signed = fields["signed"].copy()
    signed_masked = signed.copy()
    signed_masked[~valid] = np.nan
    gated_signed = _binned_gate(env, signed_masked, bandwidth=6.0)
    assert np.any(gated_signed[support] < 0.0), "signed numerator must not be floored"


def test_diffusion_kde_nonnegative():
    """diffusion_kde clips its output >= 0 and matches the shipped KDE within tol."""
    from neurospatial.encoding._smoothing import smooth_rate_map

    env, _source_coord = _build_env("grid_2d")
    rng = np.random.default_rng(3)
    spike_counts = rng.poisson(3, env.n_bins).astype(float)
    occupancy = rng.random(env.n_bins) + 0.5
    rate = np.asarray(
        smooth_rate_map(
            env,
            spike_counts,
            occupancy,
            method="diffusion_kde",
            bandwidth=8.0,
        )
    )
    finite = np.isfinite(rate)
    assert np.all(rate[finite] >= 0.0), "diffusion_kde output must be nonnegative"

    # Matches the dense-kernel KDE within the truncation tolerance.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel = env.compute_kernel(8.0, mode="density", cache=False)
    dense_rate = np.clip((kernel @ spike_counts) / (kernel @ occupancy), 0.0, None)
    np.testing.assert_allclose(rate[finite], dense_rate[finite], rtol=1e-4, atol=1e-4)


def test_env_smooth_nonneg_within_tol():
    """env.smooth on a nonnegative field: negatives bounded relative to the dense
    output (M-weighted), verified on polar/mesh (worst kappa(M)); linear on signed."""
    for geom in ("polar", "mesh"):
        env, source_coord = _build_env(geom)
        src = source_bin(env, source_coord)
        fields = make_fields(env, src)
        _W, volumes = _geometry(env)
        nonneg = fields["nonneg"]
        for mode in MODES:
            sigma = GEOMETRIES[geom][2]  # truncated
            out = np.asarray(env.smooth(nonneg, sigma, mode=mode))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dense = env.compute_kernel(sigma, mode=mode, cache=False) @ nonneg
            # Any negatives are bounded relative to the dense output (M-weighted).
            rel = _m_norm_rel(out, dense, volumes)
            assert rel < _TRUNC_MNORM_TOL, f"{geom}/{mode}: M-norm {rel:.2e}"
        # Linear on a signed field.
        signed = fields["signed"]
        a, b = 0.5, -1.3
        lhs = np.asarray(env.smooth(a * signed + b * nonneg, 10.0, mode="average"))
        rhs = a * np.asarray(env.smooth(signed, 10.0, mode="average")) + b * np.asarray(
            env.smooth(nonneg, 10.0, mode="average")
        )
        np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-11)


# ---------------------------------------------------------------------------
# Cache behavior: single growable basis, invalidation, no poisoning
# ---------------------------------------------------------------------------
def test_cache_grows_with_smaller_sigma():
    """Large-sigma then small-sigma: the single truncated basis is recomputed +
    REPLACED at the larger rank (the smaller one evicted); result within tol."""
    env, source_coord = _build_env("hex")
    src = source_bin(env, source_coord)
    fields = make_fields(env, src)
    holder = env._diffusion_eigenbasis
    env.diffuse(fields["nonneg"], 10.0, mode="density")
    rank_large_sigma = holder["rank"]
    env.diffuse(fields["nonneg"], 8.0, mode="density")  # smaller sigma -> more modes
    rank_small_sigma = holder["rank"]
    assert rank_small_sigma > rank_large_sigma, "smaller sigma must grow the basis"
    # A single basis (grown by replace), still equivalent to dense at the small sigma.
    assert isinstance(holder["basis"], tuple)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dense = env.compute_kernel(8.0, mode="density", cache=False) @ fields["nonneg"]
    out = np.asarray(env.diffuse(fields["nonneg"], 8.0, mode="density"))
    assert _m_norm_rel(out, dense, env.bin_sizes) < _TRUNC_MNORM_TOL


def test_eigenbasis_single_basis_and_invalidated():
    """One max-rank basis is reused/sliced across sigma/mode (not an accumulating
    dict); replaced on growth; dropped wholesale on a _state_version bump."""
    env, source_coord = _build_env("hex")
    src = source_bin(env, source_coord)
    fields = make_fields(env, src)
    holder = env._diffusion_eigenbasis

    env.diffuse(fields["nonneg"], 10.0, mode="density")
    env.diffuse(fields["nonneg"], 10.0, mode="average")  # same sigma, different mode
    env.diffuse(fields["nonneg"], 12.0, mode="transition")  # larger sigma -> slice
    # Exactly one persistent basis (a tuple), not a {rank -> basis} dict.
    assert isinstance(holder["basis"], tuple) and len(holder["basis"]) == 3
    basis_id = id(holder["basis"][0])

    # A geometry change (state bump) drops the holder wholesale.
    env._state_version += 1
    holder2 = env._diffusion_eigenbasis
    assert holder2 == {} and id(holder2) != id(holder)
    env.diffuse(fields["nonneg"], 10.0, mode="density")
    assert id(env._diffusion_eigenbasis["basis"][0]) != basis_id


def test_compute_kernel_does_not_poison_apply_cache():
    """A compute_kernel call (dense-expm) does not create or grow the truncated
    env.diffuse eigenbasis cache; a later env.diffuse still uses it bounded."""
    env, source_coord = _build_env("hex")
    src = source_bin(env, source_coord)
    fields = make_fields(env, src)
    holder = env._diffusion_eigenbasis
    env.diffuse(fields["nonneg"], 10.0, mode="density")
    basis_shape = holder["basis"][0].shape
    assert basis_shape[1] < env.n_bins  # truncated, not (n, n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env.compute_kernel(1.0, mode="density", cache=False)  # tiny sigma, dense expm
    # The apply cache is untouched by compute_kernel.
    assert holder["basis"][0].shape == basis_shape
    env.diffuse(fields["nonneg"], 10.0, mode="average")
    assert holder["basis"][0].shape[1] < env.n_bins


def test_near_full_rank_diffuse_no_poison():
    """A near-full-rank env.diffuse (rank >= dense_fraction*n) uses a TRANSIENT
    dense basis (applied, dropped) and never caches an (n, n) basis; a later
    normal env.diffuse still uses the bounded truncated basis."""
    env, source_coord = _build_env("grid_2d")
    src = source_bin(env, source_coord)
    fields = make_fields(env, src)
    holder = env._diffusion_eigenbasis

    # Tiny sigma -> near-full rank -> transient dense (with the large-matrix note
    # suppressed on this small env, which is below the warn threshold anyway).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = np.asarray(env.diffuse(fields["nonneg"], 0.8, mode="density"))
    assert np.all(np.isfinite(out))
    # The transient dense basis is NOT cached.
    assert "basis" not in holder
    assert holder["resolved"][(0.8, _HEAT_KERNEL_RANK_TOL)] == "dense"

    # A subsequent truncated call caches a bounded (< n) basis.
    env.diffuse(fields["nonneg"], 10.0, mode="density")
    assert holder["basis"][0].shape[1] < env.n_bins


# ---------------------------------------------------------------------------
# JAX backend: return type, in-JAX apply, jit + grad
# ---------------------------------------------------------------------------
def test_jax_backend_return_type_and_grad():
    """backend='jax' returns a jax.Array; the apply runs IN JAX so jit and grad
    through the diffusion smoothing still work (documented contract preserved)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    env, source_coord = _build_env("grid_2d")
    src = source_bin(env, source_coord)
    fields = make_fields(env, src)
    field = jnp.asarray(fields["nonneg"])

    out = env.diffuse(field, 8.0, mode="density", backend="jax")
    assert isinstance(out, jax.Array)
    # Matches the NumPy backend.
    out_np = np.asarray(env.diffuse(fields["nonneg"], 8.0, mode="density"))
    np.testing.assert_allclose(np.asarray(out), out_np, rtol=1e-6, atol=1e-8)

    # jit + grad through the smoothing (the eigenbasis is a captured constant).
    def loss(x):
        return jnp.sum(env.diffuse(x, 8.0, mode="density", backend="jax") ** 2)

    jitted = jax.jit(loss)(field)
    assert bool(jnp.isfinite(jitted))
    grad = jax.grad(loss)(field)
    assert grad.shape == field.shape
    assert bool(jnp.all(jnp.isfinite(grad)))

    # smooth_rate_map(backend="jax") returns a jax.Array with the requested dtype.
    from neurospatial.encoding._smoothing import smooth_rate_map

    rate = smooth_rate_map(
        env,
        np.asarray(fields["nonneg"]),
        np.ones(env.n_bins),
        method="diffusion_kde",
        bandwidth=8.0,
        backend="jax",
    )
    assert isinstance(rate, jax.Array)

    # jit + grad through the PUBLIC wrapper (not just env.diffuse): the public
    # validation must not coerce a traced array to NumPy, or tracing would raise.
    occ = jnp.ones(env.n_bins)

    def rate_loss(x):
        return jnp.sum(
            smooth_rate_map(
                env, x, occ, method="diffusion_kde", bandwidth=8.0, backend="jax"
            )
            ** 2
        )

    assert bool(jnp.isfinite(jax.jit(rate_loss)(field)))
    wrapper_grad = jax.grad(rate_loss)(field)
    assert wrapper_grad.shape == field.shape
    assert bool(jnp.all(jnp.isfinite(wrapper_grad)))


# ---------------------------------------------------------------------------
# Performance: apply-path vs the captured dense baseline
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_perf_large_grid(baseline):
    """The matrix-free apply-path is dramatically faster and lower-memory than the
    baseline dense expm on a large grid, and scales to ~10k bins where the dense
    path is impractical."""
    perf = baseline["perf"]
    dense_time = perf["dense_expm_time_s"]
    dense_mem = perf["dense_peak_mem_bytes"]

    # Same 3600-bin grid the dense baseline was captured on.
    env = build_perf_grid()
    field = np.ones(env.n_bins)
    tracemalloc.start()
    t0 = time.perf_counter()
    out = np.asarray(env.diffuse(field, PERF_SIGMA, mode="density"))
    apply_time = time.perf_counter() - t0
    _current, apply_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert np.all(np.isfinite(out))
    # A real reduction vs the dense baseline (expect far more than these bounds).
    assert apply_time < 0.5 * dense_time, (
        f"apply {apply_time:.1f}s not < half of dense {dense_time:.1f}s"
    )
    assert apply_peak < 0.7 * dense_mem, (
        f"apply peak {apply_peak / 1e6:.0f}MB not < 70% of dense {dense_mem / 1e6:.0f}MB"
    )

    # Scales to ~10k bins, where a dense expm is impractical (~O(n^3)).
    big = build_scaling_grid()
    assert big.n_bins >= 9000
    t0 = time.perf_counter()
    out_big = np.asarray(big.diffuse(np.ones(big.n_bins), PERF_SIGMA, mode="density"))
    big_time = time.perf_counter() - t0
    assert np.all(np.isfinite(out_big))
    assert big_time < 120.0, f"10k-bin apply took {big_time:.1f}s"
