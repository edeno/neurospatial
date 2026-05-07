# Encoding Cleanup — Task Breakdown

**Committed**: 2026-05-07

Reference: see [PLAN.md](PLAN.md) for design rationale, milestone ordering,
and gating criteria. This document tracks per-task progress with file paths
and concrete change descriptions.

## Status legend

- `[ ]` not started
- `[~]` in progress
- `[x]` complete

---

## M0 — Already applied (low-risk wins from /simplify review)

- [x] **0.1** Remove 8 stale `"to be implemented in Tasks X.Y"` docstring
      references in `spatial.py`, `directional.py`, `egocentric.py`,
      `view.py`. Pure deletions, no behavioral change.
- [x] **0.2** Cache `is_jax_available()` with `@functools.lru_cache(maxsize=1)`
      in [_backend.py:53](../../../src/neurospatial/encoding/_backend.py#L53).
      Removes per-call `importlib.util.find_spec` from every
      `compute_*_rate(s)` invocation.
- [x] **0.3** Vectorize `SpatialRatesResult.classify` Python loop in
      [spatial.py:1093](../../../src/neurospatial/encoding/spatial.py#L1093)
      to boolean-mask priority overlay.

---

## M1 — Validation and API consistency

**Goal**: every public `compute_*` entry validates inputs the same way and
fails at the boundary on stringly-typed-kwarg typos.

- [x] **1.1** Create `encoding/_validation.py` with shared helpers:
      - `_validate_trajectory(times, positions=None, headings=None)`
        — checks lengths match, non-empty, no NaN runs that would deadlock
        downstream
      - `_validate_smoothing_method(name)` — checks against the canonical
        set defined in `_smoothing.py:362`
      - `_validate_backend_arrays(*, dtype, **arrays)` — `np.asarray(..., float64)`
        each input, returns dict
      Move both byte-identical `_validate_times` from
      [`_view_binning.py:47`](../../../src/neurospatial/encoding/_view_binning.py#L47)
      and
      [`_egocentric_binning.py:161`](../../../src/neurospatial/encoding/_egocentric_binning.py#L161)
      here and update both call sites.

- [x] **1.2** Wire shared validators into every public `compute_*` entry
      (line numbers verified 2026-05-07; re-grep `^def compute_` if drifted):
      - [`compute_spatial_rate`](../../../src/neurospatial/encoding/spatial.py#L1210)
        / [`compute_spatial_rates`](../../../src/neurospatial/encoding/spatial.py#L1391)
        currently do no length validation — silent wrong answers if
        `len(times) != len(positions)`. Add `_validate_trajectory` and
        `_validate_smoothing_method`.
      - [`compute_directional_rate`](../../../src/neurospatial/encoding/directional.py#L1313)
        / [`compute_directional_rates`](../../../src/neurospatial/encoding/directional.py#L1512)
        — drop the inconsistent `.ravel()` call; require 1D inputs
        explicitly.
      - [`compute_view_rate`](../../../src/neurospatial/encoding/view.py#L875)
        / [`compute_view_rates`](../../../src/neurospatial/encoding/view.py#L1132)
        — replace inline length checks with `_validate_trajectory`.
      - [`compute_egocentric_rate`](../../../src/neurospatial/encoding/egocentric.py#L950)
        / [`compute_egocentric_rates`](../../../src/neurospatial/encoding/egocentric.py#L1215)
        — add `_validate_trajectory` and `_validate_smoothing_method`.

- [x] **1.3** **API change** — fix `compute_egocentric_rate(s)` argument
      order. CLAUDE.md prescribes
      `(env, spike_times, times, positions, headings, object_positions, ...)`.
      Currently `env` is an optional kwarg-only at the end of
      [`compute_egocentric_rate`](../../../src/neurospatial/encoding/egocentric.py#L950)
      and
      [`compute_egocentric_rates`](../../../src/neurospatial/encoding/egocentric.py#L1215).
      Make `env` the first positional arg (allow `None` only when
      `distance_metric != "geodesic"`; raise at the boundary if `None` is
      passed with `geodesic`). Update tests in
      `tests/encoding/test_compute_egocentric_rate*.py` and call sites in
      `docs/examples/` in the same PR. **Note**: this is the only task in
      M1 that is not behavior-preserving — see PLAN.md §M1.

**Verification**: `uv run pytest tests/encoding/ -q`. New error paths must
have at least one regression test each.

---

## M2 — Result-class reuse

**Goal**: one mixin and one frozen-dataclass pattern across all four
encoding domains.

- [x] **2.1** Generalize `SpatialResultMixin` in
      [`_base.py:158`](../../../src/neurospatial/encoding/_base.py#L158)
      to accept a `_bin_centers_source` class attribute (default
      `"env.bin_centers"`). Subclasses override to `"ego_env.bin_centers"`,
      `"bin_centers"` (directional), or env-less. Replace
      `hasattr(self, "firing_rates")` dispatch with explicit overrides
      in batch result classes.

- [x] **2.2** Migrate `DirectionalRateResult` / `DirectionalRatesResult`
      to inherit `SpatialResultMixin`. Remove inline `peak_*` reimplementations
      starting at
      [`directional.py:349 (peak_firing_rate, single)`](../../../src/neurospatial/encoding/directional.py#L349)
      and
      [`directional.py:1139 (peak_firing_rates, batch)`](../../../src/neurospatial/encoding/directional.py#L1139).

- [x] **2.3** Migrate `EgocentricRateResult(s)` and `ViewRateResult(s)` to
      inherit `SpatialResultMixin`. Remove inline `peak_firing_rates` block
      in
      [`egocentric.py:825`](../../../src/neurospatial/encoding/egocentric.py#L825)
      and the equivalent inline blocks in `view.py` (grep
      `def peak_firing_rate` to locate; line numbers verified at write time
      may have shifted).

- [x] **2.4 — SKIPPED.** Investigation showed these fields preserve
      user-provided construction parameters (the original `distance_range`,
      `n_distance_bins`, `n_direction_bins` passed to
      `Environment.from_polar_egocentric`) and are *not* exactly recoverable
      from `ego_env.bin_centers`, which holds bin centers, not the original
      ranges. Keeping them stored is correct.

- [x] **2.5** Vectorized `preferred_directions` and `mean_vector_lengths`
      in `DirectionalRatesResult`. NaN bins are masked out before the
      circular-stats reduction, matching the single-neuron path; a
      regression test covers the NaN case.
      `tuning_widths` and `detect_hd_cells` were left as single-neuron
      loops — they delegate to HWHM search and Rayleigh test respectively,
      which are non-trivial to vectorize and lower-value than the two
      replaced.

**Verification**: full encoding test suite. Check that the
`SpatialResultMixin` overrides correctly handle the directional case
(no `Environment`, just stored `bin_centers`).

---

## M3 — Binning hoisting and batch vectorization

**Goal**: per-population work happens once, not once per neuron.

For each task, record before/after on
`uv run python benchmarks/bench_encoding_backends.py` (population path).

- [x] **3.1 — REVERTED.** The proposed `np.searchsorted` snapshot lookup
      is *not* equivalent to the existing `np.interp`-based spike position
      mapping. With sparse sampling or fast movement, a spike between two
      trajectory frames falls in the previous bin under snapshot semantics
      but in the interpolated bin under the original (correct) semantics —
      a silent rate-map error. `bin_spike_train` keeps the per-spike
      `np.interp` lookup; `bin_spike_trains` calls it per neuron. Only
      occupancy is shared at the population level (it always was).
      A regression test in `tests/encoding/test_encoding_binning.py`
      (`TestSpikeInterpolationRegression`) guards against future
      re-introduction of the snapshot path.

- [x] **3.2** Hoist per-neuron work in
      [`bin_directional_spike_trains`](../../../src/neurospatial/encoding/_directional_binning.py#L328).
      `angle_unit` validation, `bin_size` validation,
      `headings_rad = np.radians(headings)`,
      `headings_wrapped = headings_rad % (2*pi)`, and
      `bin_edges = np.linspace(...)` are all population-level. Compute once
      before the per-neuron loop. Same pattern repeats in
      [`compute_directional_rates`](../../../src/neurospatial/encoding/directional.py#L1512)
      — wire it through.
      Baseline: TBD ms. After: TBD ms.

- [x] **3.3** Vectorize NumPy `batch_spatial_information` and
      `batch_sparsity` in
      [`_metrics.py batch_spatial_information line 191`](../../../src/neurospatial/encoding/_metrics.py#L191)
      and
      [`_metrics.py batch_sparsity line 402`](../../../src/neurospatial/encoding/_metrics.py#L402).
      Replace the per-neuron list-comp with one-pass: normalize `occupancy`,
      compute `mean = rates @ occ_prob`, `ratio = rates / mean[:, None]`,
      masked sum. The JAX path already `vmap`s — keep behavior parity.
      Baseline: TBD ms. After: TBD ms.

**Verification**: equality check (within `1e-10`) between batched output
and looped output on a fixed seed. Add as parametrized regression test.

---

## M4 — Backend symmetry

**Goal**: `_core_numpy.py` and `_core_jax.py` are byte-twin modules.
Prerequisite for M5 — see PLAN.md §M4.

- [x] **4.1** Add `spatial_information_single`, `spatial_information_batch`,
      `sparsity_single`, `sparsity_batch` to
      [`_core_numpy.py`](../../../src/neurospatial/encoding/_core_numpy.py)
      mirroring the JAX versions at
      [`_core_jax.py:343 (spatial_information_single)`](../../../src/neurospatial/encoding/_core_jax.py#L343),
      [`_core_jax.py:444 (spatial_information_batch)`](../../../src/neurospatial/encoding/_core_jax.py#L444),
      [`_core_jax.py:500 (sparsity_single)`](../../../src/neurospatial/encoding/_core_jax.py#L500),
      [`_core_jax.py:583 (sparsity_batch)`](../../../src/neurospatial/encoding/_core_jax.py#L583).
      Use NumPy semantics (`np.maximum`, `np.where`) but match the JAX
      output bit-for-bit on float64.

- [x] **4.2** Update
      [`_metrics.py spatial_information line 61`](../../../src/neurospatial/encoding/_metrics.py#L61),
      [`_metrics.py batch_spatial_information line 191`](../../../src/neurospatial/encoding/_metrics.py#L191),
      [`_metrics.py sparsity line 288`](../../../src/neurospatial/encoding/_metrics.py#L288),
      and
      [`_metrics.py batch_sparsity line 402`](../../../src/neurospatial/encoding/_metrics.py#L402)
      to dispatch to the correct backend kernel rather than running their
      own inline NumPy versions. After this, `_metrics.py` is a thin
      dispatch layer; the kernels live in `_core_*`.

**Verification**: parametrize existing metric tests across
`["numpy", "jax"]` and assert equality within 1e-10. Currently
`_metrics.py` has inline NumPy paths that have already drifted from
`_core_jax.py` in NaN clamping; regression tests must lock in the
post-merge behavior.

---

## M5 — JAX compilation and kernel caching

**Goal**: the JAX backend actually runs compiled. Gated on M4.

- [x] **5.1** Add `@jax.jit` (or `@functools.partial(jit, static_argnames=...)`)
      to pure functions in
      [`_core_jax.py`](../../../src/neurospatial/encoding/_core_jax.py):
      - [`smooth_rate_map_single` line 196](../../../src/neurospatial/encoding/_core_jax.py#L196)
      - [`smooth_rate_maps_batch` line 270](../../../src/neurospatial/encoding/_core_jax.py#L270)
      - [`spatial_information_single` line 343](../../../src/neurospatial/encoding/_core_jax.py#L343)
      - [`sparsity_single` line 500](../../../src/neurospatial/encoding/_core_jax.py#L500)
      Static args: `base` (int), `min_occupancy` (float), shape-determining
      ints. Verify with a `jax.make_jaxpr` smoke test that no recompile
      happens on second call with same shapes.
      Baseline: TBD ms. After: TBD ms.

- [x] **5.2** Cache the vmapped versions at module scope. Currently
      [`spatial_information_batch (jax.vmap rebuilt at line 494)`](../../../src/neurospatial/encoding/_core_jax.py#L494)
      and
      [`sparsity_batch (jax.vmap rebuilt at line 629)`](../../../src/neurospatial/encoding/_core_jax.py#L629)
      construct a fresh `jax.vmap(lambda ...)` on every call (re-traces).
      Pull the lambda out as a module-level `_jit`ed function and `vmap`
      it once.

- [x] **5.3** Add a `(bandwidth, env_id, n_bins)`-keyed cache for the
      gaussian-KDE kernel materialization. The dense `(n_bins, n_bins)`
      kernel is rebuilt at every call site:
      - [`_smoothing.py _gaussian_kde line 420`](../../../src/neurospatial/encoding/_smoothing.py#L420)
        (NumPy single, kernel built at line 447)
      - [`_smoothing.py _gaussian_kde_batch line 566`](../../../src/neurospatial/encoding/_smoothing.py#L566)
        (NumPy batch, kernel built at line 581)
      - [JAX gaussian KDE single (kernel built at line 738)](../../../src/neurospatial/encoding/_smoothing.py#L738)
      - [JAX gaussian KDE batch (kernel built at line 845)](../../../src/neurospatial/encoding/_smoothing.py#L845)
      Mirror `env.compute_kernel(..., cache=True)` for the diffusion path.
      For `n_bins ≈ 1000+` this is a dense (n_bins, n_bins) materialization
      plus exp per call; one-time cost on cache hit.
      Baseline: TBD ms. After: TBD ms.

**Verification**: `tests/benchmarks/test_encoding_backends.py` shows no
regression at population sizes 10, 100, 1000. JAX path numerics within
1e-6 of NumPy path on a fixed seed (M4 must already pass).

---

## M6 — Legacy surface removal or delegation

**Status: revised. Backwards compatibility is not required before release.
M6.1 added parity tests as xfail; remaining M6 work should remove stale
public surfaces or route runtime/docs through canonical result APIs.** See
PLAN.md §M6 for the gap analysis.

**Goal**: coherent user-facing encoding interfaces and ~2000 line reduction
where old implementations can be removed safely.

- [x] **6.1** Add parity test
      [`tests/encoding/test_legacy_delegation_parity.py`](../../../tests/encoding/test_legacy_delegation_parity.py).
      All four pairs mismatched at `rtol=1e-6` with relative differences
      of 50%+ — see PLAN.md §M6 for the per-pair gap. Tests are marked
      `xfail` so they document the gap and will flip to passing if the
      legacy/new pipelines are aligned later.

- [x] **6.2a** Route directional place fields through
      `compute_spatial_rate`, rename its occupancy threshold to
      `min_occupancy`, and update parity tests to compare against the
      canonical result API.

- [ ] **6.2** ~~Replace
      [`place.py compute_place_field`](../../../src/neurospatial/encoding/place.py)
      and its private helpers (`_interpolate_spike_positions`,
      `_binned_rate_map`, `_diffusion_kde`, `_gaussian_kde`, `_binned`)
      with a thin shim that calls `compute_spatial_rate` and adapts
      the return. Delete the helpers (~480 lines).~~ **Blocked by M6.1.**

- [ ] **6.3** ~~Replace
      [`head_direction.py compute_head_direction_tuning_curve`](../../../src/neurospatial/encoding/head_direction.py)
      with a delegator to `compute_directional_rate`. Keep the
      `HeadDirectionMetrics` dataclass and `head_direction_metrics`
      function as-is (they read from the rate map; no duplication).~~
      **Blocked by M6.1.**

- [ ] **6.4** ~~Replace
      [`object_vector.py compute_object_vector_tuning`](../../../src/neurospatial/encoding/object_vector.py)
      with a delegator to `compute_egocentric_rate`, and
      [`spatial_view.py compute_spatial_view_field`](../../../src/neurospatial/encoding/spatial_view.py)
      with a delegator to `compute_view_rate`. Adapt the legacy
      `*FieldResult` / `*Metrics` constructors to wrap the new result.~~
      **Blocked by M6.1.**

- [ ] **6.5** ~~Final pass: also dedupe
      [`_egocentric_binning._compute_egocentric_coords` line 199](../../../src/neurospatial/encoding/_egocentric_binning.py#L199)
      against
      [`ops/egocentric.compute_egocentric_distance` line 454](../../../src/neurospatial/ops/egocentric.py#L454)
      and
      [`ops/egocentric.compute_egocentric_bearing` line 380](../../../src/neurospatial/ops/egocentric.py#L380).
      The private helper hand-rolls Euclidean (line 242) + a geodesic branch
      via [`compute_distance_field`](../../../src/neurospatial/encoding/_egocentric_binning.py#L247)
      that duplicates the ops layer. Keep only the "select nearest object"
      step.~~ **Independent of M6.1's parity gap; could be done later as a
      standalone refactor inside `_egocentric_binning.py`.**

**Verification**: full test suite, including legacy module tests, passes
unchanged. Notebook smoke tests in `examples/11_place_field_analysis.ipynb`
and `examples/22_spatial_view_cells.ipynb` produce equivalent figures
(visual diff acceptable; numeric equivalence within 1e-6).

---

## Suggested batch sizes

Each milestone is small enough to land as a single PR if done end-to-end,
or split per task if reviewer bandwidth is limited:

| Milestone | Tasks | Est. PR size | Risk |
| --- | --- | --- | --- |
| M1 | 3 | ~300 LOC net | Low–medium (1.3 is API change) |
| M2 | 5 | ~400 LOC net | Medium |
| M3 | 3 | ~250 LOC net | Medium (perf) |
| M4 | 2 | ~400 LOC net | Medium (numerics) |
| M5 | 3 | ~150 LOC net | Medium-high (compile semantics) |
| M6 | 5 | ~−1500 LOC net | Highest (largest behavioral surface) |

Total: ~−1500 LOC, ~6 PRs. M1+M2 can run in parallel; M3+M4+M5 should
serialize because they touch overlapping perf-sensitive code (and M5
depends on M4 for backend numerical parity); M6 must follow M2 and M4.
