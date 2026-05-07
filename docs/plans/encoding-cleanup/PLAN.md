# Encoding Refactor — Post-Review Cleanup Plan

**Committed**: 2026-05-07
**Status**: Proposed
**Branch context**: `encoding-refactor-wip` vs `main`

## Source

This plan addresses findings from a three-agent `/simplify` review of the
encoding refactor (52K-line diff). The "low-risk wins" subset has already
been applied in commit ranges immediately preceding this plan:

- Removed 8 stale `"to be implemented in Tasks X.Y"` docstring references
- Cached `is_jax_available()` with `@functools.lru_cache(maxsize=1)`
- Vectorized `SpatialRatesResult.classify` (Python loop → boolean masks)

This plan tracks the remaining findings, ordered by risk × value.

## Goals

1. **Eliminate cross-module duplication** that has already started to drift
   (legacy place/HD/OV/SV modules vs new pipeline; sibling `Result` classes;
   four near-identical `compute_*` scaffolds).
2. **Unify validation and parameter handling** across the four public
   `compute_*` entry points so users get consistent error messages and so
   typos in stringly-typed kwargs (`smoothing_method`, `gaze_model`,
   `distance_metric`) fail at the boundary rather than deep inside.
3. **Realize the JAX backend's value** by adding `@jit`/`@partial(jit, ...)`
   decoration, caching `vmap`ped functions at module scope, and hoisting
   per-neuron work that should be shared across the population.
4. **Restore `_core_numpy.py` ↔ `_core_jax.py` symmetry** so the backend
   twins do not silently drift in numerics.

Out of scope for this plan: feature additions outside encoding cleanup.
API breaks are allowed when they make the pre-user public surface coherent.

## Non-goals

- No change to the public signatures of `compute_spatial_rate`,
  `compute_directional_rate`, `compute_egocentric_rate`, `compute_view_rate`
  or their plural variants — except where they are inconsistent with each
  other or with the documented argument order in CLAUDE.md (Task 2.1).
- No new dependencies.

## Risk and rollout

This is follow-on cleanup to a `-wip` refactor that has not yet merged.
Order milestones to land safest first so each milestone is independently
mergeable:

```text
M1  consistency           (3 tasks, low–medium risk) — validators + 1 API fix
M2  result-class reuse    (5 tasks, medium risk)     — touches public dataclasses
M3  binning hoisting      (3 tasks, medium risk)     — perf, behavior-preserving
M4  backend symmetry      (2 tasks, medium risk)     — restores numpy/jax parity
M5  jax compilation       (3 tasks, medium-high)     — perf, validates with bench
M6  legacy delegation     (5 tasks, highest risk)    — largest LOC reduction
```

M4 (symmetry) precedes M5 (compilation) because backend twins must be
numerically equivalent before optimizing one of them — see Milestone M4
notes below.

Each milestone gates on:

- Unit tests in the affected modules pass (no skips, no xfails introduced).
- Doctest baseline does not regress (the 5 pre-existing `_backend.py`
  doctest failures are platform-conditional and unchanged).
- For perf milestones (M3, M4): `tests/benchmarks/test_encoding_backends.py`
  shows no regression vs the pre-milestone baseline; record numbers in the
  task tracker.

## Milestone details

### M1 — Validation and API consistency

The four public `compute_*` entry points have drifted: only `compute_view_rate`
length-checks `positions`/`headings`, only `compute_directional_rate` calls
`.ravel()` on inputs, and `smoothing_method` is validated at runtime in some
paths but not others. Centralize the validators and call them from each entry.

Tasks 1.1 and 1.2 are behavior-preserving (validators that already existed in
some paths get applied to all paths; any new errors surface at the boundary
on inputs that would have failed deeper anyway).

Task 1.3 is **not** a validator change — it is a deliberate API fix.
`compute_egocentric_rate(s)` currently has `spike_times` first and `env` as
an optional kwarg-only at the end (see [egocentric.py line 950](../../../src/neurospatial/encoding/egocentric.py#L950)),
which contradicts the canonical argument order documented in
[CLAUDE.md line 262](../../../CLAUDE.md#L262). Bundling it into M1 keeps all
"compute_* entry-point cleanup" together; treat it as the breaking change in
the milestone and update callers (tests + examples) in the same PR.

### M2 — Result-class reuse

`SpatialResultMixin` provides `peak_locations()` / `peak_firing_rates()` and
is used by `SpatialRateResult(s)`. The other three `*Result` families
(`DirectionalRateResult(s)`, `EgocentricRateResult(s)`, `ViewRateResult(s)`)
re-roll equivalent logic inline. Generalize the mixin to accept a
`bin_centers` source attribute (`env.bin_centers`, `ego_env.bin_centers`, or
stored `bin_centers` for directional) and inherit it in all four families.

Also remove redundant fields from `EgocentricRateResult` /
`EgocentricRatesResult` (`distance_range`, `n_distance_bins`,
`n_direction_bins`) which are all reconstructible from `ego_env`. Tests
read these as attributes — convert them to `@property` rather than deleting,
so test code keeps working.

### M3 — Binning hoisting and batch vectorization

Three concrete wins, each replacing per-neuron work with shared work:

1. `bin_spike_trains` (spatial) currently re-runs `np.interp` per dim and
   `env.bin_at` per neuron. The trajectory→bin mapping is
   neuron-independent — precompute once.
2. `bin_directional_spike_trains` re-validates `angle_unit`/`bin_size`,
   re-wraps `headings` and re-builds `bin_edges` per neuron. Hoist the
   validation and edge construction.
3. `batch_spatial_information` (NumPy path) is a `for i in range(n_neurons)`
   list-comp. The JAX path correctly uses `vmap`. Vectorize the NumPy path
   to a single matmul + masked sum over `(n_neurons, n_bins)`.

The view and egocentric binning paths already have correct `_precompute_*`
helpers — use them as the template.

### M4 — Backend symmetry

`_core_jax.py` defines `spatial_information_single/_batch` and
`sparsity_single/_batch`. `_core_numpy.py` does not — the equivalents live
in `_metrics.py` with subtly different NaN/clamping behavior. Move the
NumPy implementations into `_core_numpy.py` mirroring the JAX module, and
have `_metrics.py` dispatch to whichever backend is active.

This is the structural prerequisite for M5: backend twins must be byte-for-
byte equivalent (within float tolerance) before optimizing one of them. If
the NumPy and JAX paths produce different numbers today, then M5's `@jit`
parity tests cannot tell whether a regression came from compilation or from
a pre-existing drift.

### M5 — JAX compilation and kernel caching

The JAX backend currently runs eager (zero `@jit` decorations) and rebuilds
hot intermediates on every call. Three changes:

1. Decorate the pure functions in `_core_jax.py` with `@jax.jit`, using
   `functools.partial(jit, static_argnames=...)` for kwargs that affect
   the graph (e.g. `base`, `min_occupancy`).
2. Cache the vmapped versions at module scope rather than constructing a
   fresh `jax.vmap(lambda ...)` on every call to
   `spatial_information_batch` / `sparsity_batch`.
3. Cache the dense gaussian_kde kernel keyed on `(bandwidth, env_id)`,
   mirroring the diffusion-KDE caching pattern that already exists in
   `env.compute_kernel(..., cache=True)`.

Each change has a measurable effect on the benchmark suite — record before/
after numbers in TASKS.md.

### M6 — Legacy surface removal or delegation

**Status: revised as of 2026-05-07.**

The original plan preserved backwards compatibility by rewriting the public
functions in `place.py`, `head_direction.py`, `object_vector.py`, and
`spatial_view.py` as thin shims over the new pipeline. Backwards
compatibility is no longer required before the first user-facing release, so
M6 should prefer removing or hiding stale public surfaces over preserving
numerically divergent behavior.

M6.1 (parity tests) was completed and is checked in at
[`tests/encoding/test_legacy_delegation_parity.py`](../../../tests/encoding/test_legacy_delegation_parity.py)
as `xfail`. Every pair fails with relative differences of 50%+:

- `compute_place_field` ("binned" / "gaussian_kde" / "diffusion_kde") --
  scale mismatch with `compute_spatial_rate`; values differ by an order
  of magnitude. Different occupancy normalization or smoothing-order
  convention.
- `compute_head_direction_tuning_curve` returns
  `(firing_rate, bin_centers)` rather than the new `Result` shape, and
  the bin-center convention may differ.
- `compute_object_vector_tuning` has a different signature (no `env`
  arg) and returns an `ObjectVectorMetrics` whose attribute layout does
  not include `firing_rate`.
- `compute_spatial_view_field` returns a `SpatialViewFieldResult` with
  no `firing_rate` attribute.

These are not floating-point drift; they are different algorithms. Because
there are no users yet, the preferred resolution is to update runtime call
sites and examples to the canonical result APIs, then either remove stale
exports or keep old modules only as internal/reference implementations with
tests that do not advertise them as supported.

The xfail tests stay in place so that any future alignment pass
automatically lights up — they will flip to passing the moment the gap
closes.

## Verification

For each milestone:

- `uv run pytest tests/encoding/ -q` passes with no new skips or xfails.
- `uv run ruff check . && uv run ruff format .` clean.
- `uv run mypy src/neurospatial/encoding/` shows no new errors (existing
  `_Buffer` / `Sized` warnings on `ArrayLike` predate this plan).
- For M3/M4: `uv run python benchmarks/bench_encoding_backends.py` records
  no regression on the population-rate hot path.

## Tracking

Task-level checkboxes live in [TASKS.md](TASKS.md) in this directory.
Update the file as work proceeds; record benchmark numbers inline next
to perf-sensitive tasks.
