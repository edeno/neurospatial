# Designs — MRF-GAM implementation

[← back to PLAN.md](PLAN.md)

Concrete implementation detail beyond what fits in a phase's Tasks block. The *why* is in the
spec ([../design-mrf-gam.md](../design-mrf-gam.md)); this file is the *how* (signatures,
NumPy skeletons). Reference implementation line refs are in [appendix.md](appendix.md).

- [Module layout](#module-layout)
- [Resolver: live-component eigenbasis](#resolver)
- [Batched penalized-Poisson Newton fit](#fit)
- [REML λ selection](#reml)
- [Deviance](#deviance)
- [Degenerate-case dispatch](#degenerate)
- [JAX mirror](#jax)

---

## <a id="module-layout"></a>Module layout

New files under `src/neurospatial/encoding/`:

| file | phase | contents |
| --- | --- | --- |
| `_glm.py` | 2 | module constants ([shared-contracts](shared-contracts.md#constants)); `MRFFit`; the orchestrator **`fit_mrf_gam(basis, counts, occupancy, *, penalty, pooled=True, backend="numpy") -> MRFFit`** (**phased final signature — Finding 2**: `pooled` is wired in phase-6, `backend` in phase-5; both exist from phase-2 so phase-3 forwards them unchanged). **No `rank` arg** — effective rank is `basis.B.shape[1]`, the single source of truth; `MRFFit.rank` is derived from it. Includes degenerate-case dispatch + deviance. `counts`/`occupancy` arrive **already restricted to `basis.live_bins`** (phase-3 owns the restriction); `fit_mrf_gam` validates `counts.shape[0] == basis.B.shape[0]`, never re-slices. `MRFFit` arrays are **always NumPy** (the public return-type conversion is phase-3's job). Pure NumPy/SciPy core. |
| `_glm_numpy.py` | 2 | `_newton_fit_numpy(counts, occupancy, B, penalty_diag, constant_base, max_iter, tol) -> (coeffs, eta, mu, n_iter, max_step, converged)`; `constant_base` is the exact all-ones direction constructed once from `MRFBasis`'s leading component intercepts and reused throughout REML. `_reml_objective_numpy(...)` remains positional for `minimize_scalar(args=…)`. The float64 core. |
| `_glm_jax.py` | 5 | float32 JAX mirror of `_glm_numpy` (`jit`, batched); selected via `encoding/_backend.py`. |

The resolver lives on `Environment` (`environment/fields.py`) + a helper in `ops/diffusion.py`
(phase-1), because it reuses PR2's cached geometry/eigensolver. `compute_spatial_rate(s)` dispatch
to `fit_mrf_gam` is wired in `spatial.py` (phase-3).

`MRFBasis` / `MRFFit` NamedTuples: define `MRFBasis` next to `DiffusionGeometry`
(`ops/diffusion.py` or `environment/_types`); `MRFFit` in `_glm.py`.

### Boundary orientation, dtype, backend (phase-3 wiring — Finding 5)

The statistical core is **bin-major**; the encoding API is **unit-major**. The phase-3
orchestrator must transpose at **both** boundaries:

- **Counts in:** the per-unit binned counts on the encoding side are `(n_units, n_bins)` (as
  `SpatialRatesResult.firing_rates` and the spike-binning helpers use). The fit needs
  `(n_live_bins, n_units)` → `counts.T[live_bins, :]`. Occupancy is `(n_bins,)` → `[live_bins]`.
- **Rates out:** `MRFFit.log_rate` is `(n_live_bins, n_units)` bin-major. Build the full
  `firing_rates` `(n_units, n_bins)` = `_RATE_FLOOR`-filled, then scatter
  `max(exp(η), _RATE_FLOOR).T` into columns `live_bins` → i.e. rows are units. The **singular**
  `SpatialRateResult.firing_rate` is `(n_bins,)`.
- **dtype:** `compute_spatial_rates` has `dtype: {np.float32, np.float64} = np.float64`. The glm
  **core stays float64** (correctness); cast the assembled `firing_rates` (and stored
  coefficients/diagnostics as appropriate) to the requested `dtype` at the result boundary, matching
  the ratio path.
- **backend — two separable concerns (Finding 2):**
  1. **Compute backend** (which fit runs): `fit_mrf_gam(..., backend=)` selects the fit engine —
     NumPy float64 (phase-2) vs JAX float32 (phase-5). **Until phase-5 lands, `backend="jax"` falls
     back to the NumPy compute** (documented), so phase-3 can forward the resolved backend
     unconditionally. `MRFFit` arrays are **always cast back to NumPy** at the fit boundary.
  2. **Return-array type** (public `firing_rates`): phase-3 resolves `get_backend_name(backend)`
     ([_backend.py:165](../../../src/neurospatial/encoding/_backend.py)) — **not** a raw
     `backend != "numpy"` check — and, when the resolved backend is `"jax"`, converts the assembled
     `firing_rates` to a JAX array to match the ratio path. This fixes the public return contract in
     phase-3; phase-5 changes only concern (1), never the return type.

---

## <a id="resolver"></a>Resolver: live-component eigenbasis (phase-1)

Spec §6.1. Reuses `env._diffusion_geometry` (`fields.py:512` → `W, volumes, n_components,
labels`) and `_symmetric_conjugate` / `_symmetric_eigenbasis` (`diffusion.py:290,382`). Does
**not** modify the operator or the eigensolver.

**Clean split (Finding 3):** a **cached global resolver** (geometry-only, owns the eigensolve +
growth) and a **pure selector** (occupancy-only, no eigensolve). `_mrf_basis` calls the first
then the second — the selector never re-solves, so the cache is never bypassed.

### (a) Cached global resolver — owns the eigensolve

`Environment._mrf_eigenbasis` is a `versioned_cached_property` returning a **mutable holder**
`{"Q": (n_bins, G), "Lam": (G,), "mode_comp": (G,), "G": int}`: the `G` globally-smallest modes of
`S = _symmetric_conjugate(W, volumes)`, each mode's **source W-component**, and the built rank `G`.
**Initial (empty) state: `{"Q": zeros((n_bins, 0)), "Lam": zeros(0), "mode_comp": zeros(0, intp),
"G": 0}`.** Grow-by-replace, same pattern as `_diffusion_eigenbasis` (`fields.py:535`); dropped on
any `_state_version` bump; **keyed by geometry only** (no `(sigma, tol)` — the MRF penalty basis is
bandwidth-independent).

**Transient-dense policy (Finding 2):** the dense (`rank=None`) eigenbasis is an `n×n` float64
matrix — the design forbids persisting it ([design §11 / dense-fraction policy](../design-mrf-gam.md)).
So `_ensure_global_modes` **returns** `(Q, Lam, mode_comp)` and **only updates the holder in the
truncated (sparse) regime**; past `dense_fraction·n` it computes a **call-local** dense basis and
leaves the holder at its last sparse `G`:

```python
def _ensure_global_modes(holder, S, labels, needed_G):
    """Return (Q, Lam, mode_comp) with >= needed_G modes. Persist ONLY sparse bases."""
    n = S.shape[0]
    if holder["G"] >= needed_G:                          # cache hit — NO eigensolve
        return holder["Q"], holder["Lam"], holder["mode_comp"]
    G = min(needed_G, n)
    dense = G >= dense_fraction * n
    Lam, Q = _symmetric_eigenbasis(S, None if dense else G)   # Lam ascending, clipped ≥ 0
    # Each column of Q is component-local (nonzero on exactly one W-component, exact 0 elsewhere),
    # so the source component is the label of any nonzero row (recovers the mode→component map
    # WITHOUT modifying _symmetric_eigenbasis — replaces the undefined `labels_of_modes`).
    mode_comp = np.array([labels[np.flatnonzero(col)[0]] for col in Q.T], dtype=np.intp)
    if not dense:                                        # persist truncated bases only (transient-dense)
        holder.update(Q=Q, Lam=Lam, mode_comp=mode_comp, G=Q.shape[1])
    return Q, Lam, mode_comp
```

### (b) Pure selector — occupancy-only, no eigensolve

```python
# ops/diffusion.py — pure, array-only, unit-testable
def select_live_basis(Q, Lam, mode_comp, volumes, labels, occupancy, *, rank) -> MRFBasis:
    _validate_occupancy(occupancy)                                  # finite + non-negative (else raise)
    live_comp = _live_components(labels, occupancy)                 # live = Σ occupancy > 0
    n_live_components = int(live_comp.size)
    live_bins = np.flatnonzero(np.isin(labels, live_comp)).astype(np.intp)
    if live_bins.size == 0:                                          # zero total occupancy (spec §7)
        return MRFBasis(np.zeros((0, 0)), np.zeros((0,)), live_bins, 0)
    R = _DEFAULT_MAX_RANK if rank is None else int(rank)
    r_eff = max(n_live_components, min(live_bins.size, R))
    n_fill = r_eff - n_live_components
    inv_sqrt_vol = 1.0 / np.sqrt(volumes)                          # M^{-1/2}; B = M^{-1/2}Q (spec §3)

    # (1) INTERCEPTS — the EXACT MASS-NORMALIZED constant per live component. The null of S is
    #     q0_c = sqrt(vol)·1_c / sqrt(Σ_c vol), so the documented B = M^{-1/2}Q null column is
    #     B0_c = M^{-1/2} q0_c = 1_c / sqrt(Σ_c vol) — a constant on component c. (NOT 1_c/‖1_c‖₂,
    #     which would only match on uniform volumes and fail the M^{-1/2}Q basis test.) d = 0.
    B_int = np.zeros((live_bins.size, n_live_components))
    lbl_live = labels[live_bins]
    for j, c in enumerate(live_comp):
        B_int[lbl_live == c, j] = 1.0 / np.sqrt(volumes[labels == c].sum())

    # (2) FILL — the n_fill smallest-λ live modes that are NOT a component's constant null. The null
    #     is excluded STRUCTURALLY: _null_mode_mask marks, per live component, the mode with maximal
    #     overlap against that component's mass-weighted constant sqrt(vol)·1_c (overlap ~1 for the
    #     constant, ~0 for every orthogonal smoothness mode). This is SCALE-INVARIANT — an absolute
    #     λ cutoff would misread a genuine smoothness mode whose Laplacian eigenvalue is physically
    #     tiny (eigenvalues carry units 1/length²) as a null and drop it. It also can't admit a
    #     near-constant duplicate of the intercept. There are always ≥ n_fill non-null live modes
    #     (n_live_bins − n_live_components ≥ n_fill, since r_eff ≤ n_live_bins) once Q is full-grown;
    #     select_live_basis RAISES if fewer are present rather than returning a narrow B.
    is_null = _null_mode_mask(Q, mode_comp, volumes, labels, live_comp)  # raises if a null is absent
    fill_idx = np.flatnonzero(np.isin(mode_comp, live_comp) & ~is_null)[:n_fill]  # λ-sorted
    if fill_idx.size != n_fill:
        raise ValueError(...)                                      # grow the eigenbasis to full rank
    d_fill = Lam[fill_idx]                                         # eigenvalues, clipped ≥ 0 by the eigensolver
    if d_fill.size and np.any(d_fill <= 0.0):
        raise ValueError(...)   # a non-null fill with ≤0 weight = numerically (near-)disconnected component
    B_fill = (inv_sqrt_vol[:, None] * Q[:, fill_idx])[live_bins, :]

    B = np.concatenate([B_int, B_fill], axis=1)                    # (n_live_bins, r_eff); intercepts first
    d = np.concatenate([np.zeros(n_live_components), d_fill])      # nulls exactly 0; fills strictly > 0
    return MRFBasis(B, d, live_bins, n_live_components)
```

`B[:, :n_live_components]` are the constructed **mass-normalized** intercepts — each exactly
constant on its live component (unit-tested) — and `B[:, n_live_components:]` are the fill
(smoothness) modes with exactly one constant null excluded per live component. Because that
exclusion is **structural** (overlap, not an eigenvalue cutoff), `penalty_rank = n_fill = r_eff −
n_live_components` is exact by construction, independent of the physical coordinate scale and of any
eigenvalue-ordering fragility.

### (c) `Environment._mrf_basis(occupancy, *, rank)` — glue + iterative growth

Because dead components also contribute low modes, **the global `G` needed to expose `r_eff` live
modes is not knowable up front** (Finding 3) — grow iteratively until enough live modes appear:

```python
def _mrf_basis(self, occupancy, *, rank):
    _validate_occupancy(occupancy)                       # fail fast, before the eigensolve
    W, volumes, n_components, labels = self._diffusion_geometry
    S = _symmetric_conjugate(W, volumes)                 # cheap; eigensolve is the cost
    holder = self._mrf_eigenbasis                        # cached mutable holder (a)
    live = _live_components(labels, occupancy)
    n_live_components = int(live.size)
    r_eff = _target_live_rank(labels, occupancy, rank)   # max(n_live_comp, min(n_live_bins, R))
    n_fill = r_eff - n_live_components
    # G must be >= n_components: _symmetric_eigenbasis RAISES on rank < n_components (total geometry
    # components, diffusion.py:419). r_eff counts only LIVE components, so 1 live + several dead
    # would start G < n_components and raise on the first solve (Finding 1).
    G = max(holder["G"], r_eff, n_components)
    while True:                                          # grow until every live null present + enough fills
        Q, Lam, mode_comp = _ensure_global_modes(holder, S, labels, G)   # may be transient (dense)
        # VERIFY the null the selector requires is present for every live component (a component's
        # numerical null can sort AFTER one of its positive modes when both are ~machine-ε), and count
        # the non-null fill-eligible live modes — using the SAME overlap criterion select_live_basis
        # uses, so the loop's stop condition and the selector never disagree.
        is_null, missing = _null_mode_indices(Q, mode_comp, volumes, labels, live)
        n_fill_available = int((np.isin(mode_comp, live) & ~is_null).sum())
        if (missing.size == 0 and n_fill_available >= n_fill) or Q.shape[1] >= S.shape[0]:
            break
        G = min(2 * Q.shape[1], S.shape[0])              # double, capped at full rank
    return select_live_basis(Q, Lam, mode_comp, volumes, labels, occupancy, rank=rank)
```

Only `_ensure_global_modes` eigensolves, and only when growth is needed; it consumes/returns the
`(Q, Lam, mode_comp)` triple (never persisting a dense basis — Finding 2), and the selector is pure
masking on that triple. A repeated `_mrf_basis` at the same-or-smaller rank / same live support that
stayed in the sparse regime is a full cache hit (no eigensolve) — the phase-1 test asserts this by
spying on `_symmetric_eigenbasis` and the
holder `G`, distinct from the `_diffusion_geometry`-reuse assertion. `_target_live_rank` /
`_live_comp` are small pure helpers shared with the selector.

---

## <a id="fit"></a>Batched penalized-Poisson Newton fit (phase-2)

Spec §4. Batched over the neuron axis (shared `B`, `o`; per-neuron `n_k`, `γ_k`). Converge on the
**relative penalized-objective decrease**, NOT the coefficient step (spec §4 — null modes drift
forever). `penalty_diag = penalty * d` (length `r_eff`).

```python
def _newton_fit_numpy(counts, occupancy, B, penalty_diag, constant_base, max_iter, tol):
    # counts (n_live_bins, n_units); occupancy (n_live_bins,); B (n_live_bins, r_eff)
    n_bins, r = B.shape
    n_units = counts.shape[1]
    # warm start: constant log-rate per unit from exact structural intercepts
    rate0 = np.clip(counts.sum(0) / max(occupancy.sum(), 1e-9), 1e-6, None)  # (n_units,)
    coeffs = constant_base[:, None] * np.log(rate0)[None, :] # (r, n_units); B @ constant_base == 1
    prev_obj = _penalized_obj(coeffs, B, counts, occupancy, penalty_diag)  # warm-start objective
    converged = False
    max_accepted_step = 0.0
    for it in range(1, max_iter + 1):
        eta = np.clip(B @ coeffs, -_ETA_CLIP, _ETA_CLIP)    # (n_bins, n_units)
        mu = occupancy[:, None] * np.exp(eta)
        grad = B.T @ (counts - mu) - penalty_diag[:, None] * coeffs           # (r, n_units)
        # Hessian per unit: Bᵀ diag(mu_k) B + diag(penalty_diag) + jitter I.
        # Index carefully: i=bins, k=unit, r/s=basis modes → output (n_units, r, r).
        # ("ik,ij,il->kjl" would give (r, n_units, r) — WRONG for the diag-add + solve.)
        H = np.einsum("ir,ik,is->krs", B, mu, B, optimize=True)  # (n_units, r, r)
        H += (penalty_diag + _HESSIAN_JITTER) * np.eye(r)[None]  # broadcast add to each unit's block
        newton_step = np.linalg.solve(H, grad.T[..., None])[..., 0].T   # (r, n_units)
        # step-halving returns the ACCEPTED (possibly halved) step + new coeffs + new objective
        coeffs, accepted_step, obj = _step_halve(
            coeffs, newton_step, B, counts, occupancy, penalty_diag
        )
        max_accepted_step = float(np.max(np.abs(accepted_step)))
        if _rel_decrease(prev_obj, obj) < tol:              # relative penalized-objective decrease
            converged = True
            break
        prev_obj = obj
    # Recompute final eta/mu from the UPDATED coeffs so REML/deviance see a consistent
    # (coeffs, eta, mu) triple — never the pre-update arrays (Finding 3).
    eta = np.clip(B @ coeffs, -_ETA_CLIP, _ETA_CLIP)
    mu = occupancy[:, None] * np.exp(eta)
    return coeffs, eta, mu, it, max_accepted_step, converged
```

- **`_step_halve(coeffs, newton_step, …) -> (new_coeffs, accepted_step, obj)`**: up to
  `_MAX_STEP_HALVINGS` halvings; accept the first `α·newton_step` (α = 1, ½, ¼, …) that decreases
  the penalized objective (float32 uses `_DESCENT_TOL` slack — phase-5). **Returns the accepted
  step `α·newton_step`** (so the caller reports the real convergence diagnostic, not the unhalved
  Newton step) and the new objective. Penalized objective per unit: `−Σ(n·η − μ) +
  ½·Σ(penalty_diag·γ²)`; the batch stopping criterion is the **max** relative decrease across units
  (matches the reference; batch scalar).
- **Structural constant warm start:** the first `n_live_components` columns are exact disjoint
  intercepts, so construct `constant_base` with `B @ constant_base == 1` and zero fill
  coefficients, then scale it by each unit's `log_rate0`. Compute once per population fit and
  reuse throughout REML; no least-squares/SVD is needed.
- The `einsum` Hessian is O(n·r²·n_units); for large `r` loop over units instead. Correctness
  first; phase-5 JAX handles scale.

---

## <a id="reml"></a>REML λ selection (phase-2)

Spec §5. Pooled objective over `log λ`, minimized by `scipy.optimize.minimize_scalar`:

```python
def _reml_objective_numpy(
    log_penalty, counts, occupancy, B, d, penalty_rank, constant_base, max_iter, tol
):
    # ALL args positional — minimize_scalar supplies extras via positional `args=` (Finding 4).
    penalty = np.exp(log_penalty)
    coeffs, eta, mu, *_ = _newton_fit_numpy(counts, occupancy, B, penalty * d, max_iter, tol)
    loglik = np.sum(counts * eta - mu, axis=0)                    # (n_units,)
    pen = 0.5 * penalty * np.sum(d[:, None] * coeffs**2, axis=0)  # (n_units,)
    # log|H_k| via Cholesky; +inf if any H_k not PD (reject that λ)
    logdet = _batched_chol_logdet(B, mu, penalty * d)            # (n_units,); non-PD → inf
    reml = -loglik + pen - 0.5 * penalty_rank * log_penalty + 0.5 * logdet
    return float(np.sum(reml)) if np.all(np.isfinite(logdet)) else np.inf

def select_penalty_by_reml(counts, occupancy, B, d, penalty_rank, *, max_iter, tol):
    if penalty_rank == 0:                        # spec §5 — flat in λ, skip
        return None, None                        # (penalty, reml_objective)
    res = scipy.optimize.minimize_scalar(
        _reml_objective_numpy, bounds=_LOG_PENALTY_BOUNDS, method="bounded",
        args=(counts, occupancy, B, d, penalty_rank, max_iter, tol),   # positional args
        options={"xatol": _REML_XATOL},
    )
    if not np.isfinite(res.fun):
        raise ValueError("REML found no finite objective in the log-penalty interval ...")
    return float(np.exp(res.x)), float(res.fun)
```

**`_newton_fit_numpy` takes `max_iter`/`tol` positionally** (or bind them with
`functools.partial` at the `minimize_scalar` call). Keyword-only controls cannot be filled
through `minimize_scalar`'s positional `args=` — that raises `TypeError` (Finding 4).

**Final-fit `penalty_diag` when `penalty is None`.** REML-skip (`r==0`) and the no-data cases
return `penalty=None`, but the final `_newton_fit_numpy` still needs a numeric `penalty_diag`.
Use `penalty_diag = np.zeros_like(d)` (an unpenalized fit — correct, since `r==0` means every
weight is a structural null anyway), while **`MRFFit.penalty` stays `None`**. Never pass `None`
into the fit.

Note the `−½ · penalty_rank · log λ` term uses `penalty_rank = r_eff − n_live_components`
summed over units ⇒ the `n_units` factor is implicit in `np.sum(reml)` (spec §5). Cholesky:
`scipy.linalg.cholesky`; a non-PD `H_k` yields a `LinAlgError` → treat as `+inf`.

---

## <a id="deviance"></a>Deviance (phase-2/3)

Spec §6.3. Unpenalized Poisson deviance per unit, using the **stored** floored rate so deviance
describes what is reported:

```python
def _poisson_deviance(counts, occupancy, firing_rate):
    # firing_rate (n_live_bins, n_units) = max(exp(eta), _RATE_FLOOR); exposed bins only
    mu = occupancy[:, None] * firing_rate                    # (n_live_bins, n_units)
    exposed = occupancy > 0
    n, m = counts[exposed], mu[exposed]
    term = np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / m), 0.0) - (n - m)
    return 2.0 * term.sum(0)                                 # (n_units,)
```

---

## <a id="degenerate"></a>Degenerate-case dispatch (phase-3)

Spec §7 table. Handle before/around the fit so outputs stay model-consistent:

| case | detection | output |
| --- | --- | --- |
| no neurons | `counts.shape[1] == 0` | `coefficients (r_eff, 0)`, `firing_rate (0, n_bins)`, `deviance (0,)`, `penalty/reml None`, `converged True`. Skip fit. |
| zero total occupancy | `MRFBasis.live_bins.size == 0` | `coefficients (0, n_units)`, `firing_rate` all `_RATE_FLOOR`, `deviance` zeros, `penalty/reml None`, `converged True`, warn. Skip fit. |
| **all-zero-spike population** | `counts.sum() == 0` (no unit has a spike) | **λ is unidentified**, so **skip REML *selection*** — but respect the fixed-penalty contract (Finding 2): `reml_objective=None` always; **`penalty` = the supplied fixed float if given, else `None`** (auto). Fit with `penalty_diag = (fixed_penalty or 0)·d`; fields floor to `_RATE_FLOOR` either way; `deviance` finite (≈0), warn. Shared home for `pooled=True` and `pooled=False`-with-no-informative-unit. |
| dead component | `n_live_components < n_components` | fit on live bins; dead bins → `_RATE_FLOOR`; warn. |
| zero-spike neuron (population has ≥1 informative unit) | `counts[:, k].sum() == 0` | fit normally (low intercept); rate floors near `_RATE_FLOOR`. Shared-λ: no special path. Per-unit (`pooled=False`): pooled-λ fallback ([phase-6](phase-6-per-neuron-lambda.md)). |
| `penalty=0` rank-deficient | `np.linalg.matrix_rank(B[exposed_live_bins]) < r_eff` | warn (identifiability). Fit still runs. |

`firing_rate` is assembled into the **full active-bin array** `(n_bins,)` / `(n_bins, n_units)`:
`_RATE_FLOOR` everywhere, then `max(exp(η), _RATE_FLOOR)` scattered into `live_bins`.

---

## <a id="jax"></a>JAX mirror (phase-5)

`_glm_jax.py` mirrors `_glm_numpy.py` with `jnp`, `jax.jit` (static `max_iter`), batched. float32
(`_FIT_DTYPE = jnp.float32`). Differences from the NumPy core:

- `tol = max(tol, _FIT_TOL_FLOOR)` (float32 noise floor); step-halving uses `_DESCENT_TOL`.
- Dispatched by the existing `encoding/_backend.py` awareness (same pattern as `_core_numpy` /
  `_core_jax` elsewhere in `encoding/`).
- The invariant all-ones warm-start direction is constructed structurally once and passed into
  both backends; no SVD occurs in the JAX warm start, Newton loop, or REML score. The SPD Newton
  Hessian uses Cholesky + triangular solves on JAX rather than generic LU.
- Measured parity: at an identical fixed λ, rate error is approximately `~1e-6` vs the float64
  NumPy core; automatic REML is approximately `~1e-3` because float32 can select a slightly
  different point along a broad minimum. The parity test also asserts the float32 path converges
  (`converged is True`, `n_iter < _MAX_ITER`).
- **Return-type contract is owned by phase-3, not this phase (Finding 4).** Phase-3 already fixes
  what `compute_spatial_rates(method="glm", backend="jax")` returns (NumPy core + convert output
  to JAX arrays to match the ratio path, resolved via `get_backend_name`). Phase-5 accelerates the
  **fit compute only** and must keep that contract — the phase-3 `test_glm_backend_jax_return`
  stays green. The parity test asserts values match regardless of array type (`np.asarray(...)`
  both sides).
