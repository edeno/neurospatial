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

- [ ] **1.1** Create `encoding/_validation.py` with shared helpers:
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

- [ ] **1.2** Wire shared validators into every public `compute_*` entry
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

- [ ] **1.3** **API change** — fix `compute_egocentric_rate(s)` argument
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

- [ ] **2.1** Generalize `SpatialResultMixin` in
      [`_base.py:158`](../../../src/neurospatial/encoding/_base.py#L158)
      to accept a `_bin_centers_source` class attribute (default
      `"env.bin_centers"`). Subclasses override to `"ego_env.bin_centers"`,
      `"bin_centers"` (directional), or env-less. Replace
      `hasattr(self, "firing_rates")` dispatch with explicit overrides
      in batch result classes.

- [ ] **2.2** Migrate `DirectionalRateResult` / `DirectionalRatesResult`
      to inherit `SpatialResultMixin`. Remove inline `peak_*` reimplementations
      starting at
      [`directional.py:349 (peak_firing_rate, single)`](../../../src/neurospatial/encoding/directional.py#L349)
      and
      [`directional.py:1139 (peak_firing_rates, batch)`](../../../src/neurospatial/encoding/directional.py#L1139).

- [ ] **2.3** Migrate `EgocentricRateResult(s)` and `ViewRateResult(s)` to
      inherit `SpatialResultMixin`. Remove inline `peak_firing_rates` block
      in
      [`egocentric.py:825`](../../../src/neurospatial/encoding/egocentric.py#L825)
      and the equivalent inline blocks in `view.py` (grep
      `def peak_firing_rate` to locate; line numbers verified at write time
      may have shifted).

- [ ] **2.4** Remove redundant fields `distance_range`, `n_distance_bins`,
      `n_direction_bins` from
      [`EgocentricRateResult`](../../../src/neurospatial/encoding/egocentric.py#L201)
      and
      [`EgocentricRatesResult`](../../../src/neurospatial/encoding/egocentric.py#L558).
      Convert each to `@property` reading from `ego_env`. Tests at
      `tests/encoding/test_compute_egocentric_rate.py:264-345` access these
      as attributes — `@property` keeps them working without construction
      changes.

- [ ] **2.5** Vectorize remaining per-neuron Python loops in
      `DirectionalRatesResult` (current line numbers from `grep "for i in
      range(n_neurons)"` on `directional.py`):
      - [`mean_vector_lengths`, loop at line 1068](../../../src/neurospatial/encoding/directional.py#L1068)
      - [`tuning_widths`, loop at line 1104](../../../src/neurospatial/encoding/directional.py#L1104)
      - [`detect_hd_cells`, loop at line 1202](../../../src/neurospatial/encoding/directional.py#L1202)
      All should follow the M0.3 pattern (compute arrays, mask, assign).

**Verification**: full encoding test suite. Check that the
`SpatialResultMixin` overrides correctly handle the directional case
(no `Environment`, just stored `bin_centers`).

---

## M3 — Binning hoisting and batch vectorization

**Goal**: per-population work happens once, not once per neuron.

For each task, record before/after on
`uv run python benchmarks/bench_encoding_backends.py` (population path).

- [ ] **3.1** Hoist trajectory binning in
      [`bin_spike_trains`](../../../src/neurospatial/encoding/_binning.py#L238).
      Currently each
      [`bin_spike_train`](../../../src/neurospatial/encoding/_binning.py#L41)
      call runs `np.interp` per dim and
      `env.bin_at(spike_positions)`. Precompute
      `trajectory_bins = env.bin_at(positions)` once outside the loop;
      per neuron, do `np.searchsorted(times, spike_times) - 1` and look up.
      Mirror the existing pattern in
      [`_view_binning._precompute_view_bins`](../../../src/neurospatial/encoding/_view_binning.py#L83).
      Baseline: TBD ms. After: TBD ms.

- [ ] **3.2** Hoist per-neuron work in
      [`bin_directional_spike_trains`](../../../src/neurospatial/encoding/_directional_binning.py#L328).
      `angle_unit` validation, `bin_size` validation,
      `headings_rad = np.radians(headings)`,
      `headings_wrapped = headings_rad % (2*pi)`, and
      `bin_edges = np.linspace(...)` are all population-level. Compute once
      before the per-neuron loop. Same pattern repeats in
      [`compute_directional_rates`](../../../src/neurospatial/encoding/directional.py#L1512)
      — wire it through.
      Baseline: TBD ms. After: TBD ms.

- [ ] **3.3** Vectorize NumPy `batch_spatial_information` and
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

- [ ] **4.1** Add `spatial_information_single`, `spatial_information_batch`,
      `sparsity_single`, `sparsity_batch` to
      [`_core_numpy.py`](../../../src/neurospatial/encoding/_core_numpy.py)
      mirroring the JAX versions at
      [`_core_jax.py:343 (spatial_information_single)`](../../../src/neurospatial/encoding/_core_jax.py#L343),
      [`_core_jax.py:444 (spatial_information_batch)`](../../../src/neurospatial/encoding/_core_jax.py#L444),
      [`_core_jax.py:500 (sparsity_single)`](../../../src/neurospatial/encoding/_core_jax.py#L500),
      [`_core_jax.py:583 (sparsity_batch)`](../../../src/neurospatial/encoding/_core_jax.py#L583).
      Use NumPy semantics (`np.maximum`, `np.where`) but match the JAX
      output bit-for-bit on float64.

- [ ] **4.2** Update
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

- [ ] **5.1** Add `@jax.jit` (or `@functools.partial(jit, static_argnames=...)`)
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

- [ ] **5.2** Cache the vmapped versions at module scope. Currently
      [`spatial_information_batch (jax.vmap rebuilt at line 494)`](../../../src/neurospatial/encoding/_core_jax.py#L494)
      and
      [`sparsity_batch (jax.vmap rebuilt at line 629)`](../../../src/neurospatial/encoding/_core_jax.py#L629)
      construct a fresh `jax.vmap(lambda ...)` on every call (re-traces).
      Pull the lambda out as a module-level `_jit`ed function and `vmap`
      it once.

- [ ] **5.3** Add a `(bandwidth, env_id, n_bins)`-keyed cache for the
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

## M6 — Legacy module delegation

**Goal**: ~2000 line reduction; legacy public surface kept.

- [ ] **6.1** Add parity test
      `tests/encoding/test_legacy_delegation_parity.py` that runs each
      legacy function and its new counterpart on a shared synthetic
      dataset and asserts firing-rate equivalence within 1e-8. Cover:
      - `compute_place_field` ↔ `compute_spatial_rate(...).firing_rate`
      - `compute_head_direction_tuning_curve` ↔ `compute_directional_rate(...).firing_rate`
      - `compute_object_vector_tuning` ↔ `compute_egocentric_rate(...)`
      - `compute_spatial_view_field` ↔ `compute_view_rate(...)`
      Run these tests before any shim work; failures here mean the new
      pipeline is missing a feature, not that the legacy code is wrong.

- [ ] **6.2** Replace
      [`place.py compute_place_field`](../../../src/neurospatial/encoding/place.py)
      and its private helpers (`_interpolate_spike_positions`,
      `_binned_rate_map`, `_diffusion_kde`, `_gaussian_kde`, `_binned`)
      with a thin shim that calls `compute_spatial_rate` and adapts
      the return. Delete the helpers (~480 lines).

- [ ] **6.3** Replace
      [`head_direction.py compute_head_direction_tuning_curve`](../../../src/neurospatial/encoding/head_direction.py)
      with a delegator to `compute_directional_rate`. Keep the
      `HeadDirectionMetrics` dataclass and `head_direction_metrics`
      function as-is (they read from the rate map; no duplication).

- [ ] **6.4** Replace
      [`object_vector.py compute_object_vector_tuning`](../../../src/neurospatial/encoding/object_vector.py)
      with a delegator to `compute_egocentric_rate`, and
      [`spatial_view.py compute_spatial_view_field`](../../../src/neurospatial/encoding/spatial_view.py)
      with a delegator to `compute_view_rate`. Adapt the legacy
      `*FieldResult` / `*Metrics` constructors to wrap the new result.

- [ ] **6.5** Final pass: also dedupe
      [`_egocentric_binning._compute_egocentric_coords` line 199](../../../src/neurospatial/encoding/_egocentric_binning.py#L199)
      against
      [`ops/egocentric.compute_egocentric_distance` line 454](../../../src/neurospatial/ops/egocentric.py#L454)
      and
      [`ops/egocentric.compute_egocentric_bearing` line 380](../../../src/neurospatial/ops/egocentric.py#L380).
      The private helper hand-rolls Euclidean (line 242) + a geodesic branch
      via [`compute_distance_field`](../../../src/neurospatial/encoding/_egocentric_binning.py#L247)
      that duplicates the ops layer. Keep only the "select nearest object"
      step.

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
