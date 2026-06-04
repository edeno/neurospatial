# neurospatial UX Review — Independent Synthesis

**Date:** 2026-06-04
**Method:** 6 parallel code-grounded review agents (onboarding, API surface/naming, scaling, unit identity & result objects, errors/footguns, ecosystem strategy), each reading the actual source and validating/challenging the prior review (`UX_REVIEW_2026-06-04.md`). Every claim below is anchored to `file:line`. No source was changed.

---

## Thesis (where I land, in one paragraph)

The prior review's **diagnosis is correct**: the algorithms are good; the pain is that scientists hand-assemble too much, lose unit identity, and hit unsafe-at-scale defaults. But its **prescription over-builds**. The single best strategy is *not* to grow a large stateful `Session` "product layer" that re-implements pynapple. It is to (1) **stop the silent wrong-science footguns and fix the broken docs first** — these are urgent and nearly free; (2) **make unit identity durable and the API self-consistent** so a beginner can extrapolate; (3) **make the memory-safe path the default** for decoding and high-resolution encoding; and (4) **interoperate** with the neuro ecosystem (pynapple objects in, labeled `xarray.Dataset`/NWB out) instead of inventing a parallel data model. neurospatial's moat is the `Environment` (graph discretization, geodesic distance, linearization, masked/polar layouts). Spikes, epochs, time series, and unit tables are solved problems — adopt them, don't rebuild them.

---

## Maintainer refinements (adopted 2026-06-04 — these govern v0.6)

Four refinements from maintainer review, folded into the plan below:

1. **A `Session`/`Recording` is *not* ruled out — only the mutable god-object is.** The good version is precisely the small **frozen bundle** this doc proposes (`(env, position, spikes, epochs)` with no hidden mutable state). If the name `Session` or `Recording` helps users discover it, use it. The line we will not cross is a stateful, method-bearing object whose tab-completion shows methods that are invalid until hidden state is set.

2. **Raw NumPy arrays remain the universal, baseline input contract — interop is optional and additive.** Scientists keep data in many formats; we cannot assume pynapple or NWB. Every public function must keep working with plain arrays. pynapple/NWB/xarray are *optional adapters at the edges* (nice-to-have ingress/egress), never required, never a hard dependency. "Interop-first" means *design the boundaries to accept them when present*, not *require them*.

3. **Reframe the `to_dataframe` critique:** it is **not** a contract violation (the base contract legitimately allows heterogeneous long-form tables). The real problem is **one method name with two meanings** — sometimes a dense `(unit, bin)` table, sometimes a per-unit summary — which is a **UX + scale** hazard (a beginner gets a 10M-row frame when they expected a summary). Fix by giving the two shapes two names; don't frame it as a broken invariant.

4. **Defer dask *implementation*, but design the API for chunking now.** `to_xarray()` and decoding results must not bake in eager-NumPy assumptions: shape the return types and parameters (`return_posterior`, `dtype`, `time_chunk`, `Dataset` with labeled dims) so a chunked/lazy/dask backend can slot in later without an API break. Don't ship dask yet; don't foreclose it either.

---

## Scientific Python design principles (governing constraints)

Adopting the [Scientific Python development guide's design principles](https://learn.scientific-python.org/development/principles/design/) as constraints. The plan is already strongly aligned; the guide **confirms** most of it and **sharpens** three specific choices.

**Confirmed by the guide (keep doing):**
- *Layered API* — a thin "friendly" layer over a strict "cranky" core. Exactly the three layers here; the strict core stays functions on standard arrays.
- *Avoid mutable state; immutable classes representing state at each step* — the Session/Recording is an immutable, return-new **frozen bundle**; the fit/predict decoder returns new objects.
- *Raise (don't print) specific, actionable errors with the wrong value + what's wrong + how to fix + context; don't use permissive defaults when stricter is clearer* — this **is** the Phase 0 footgun/error work. Use `warnings.warn` (never `print`); hard-error where a result would otherwise be silently wrong.
- *Keyword-only args after `*`* — keep/extend the convention (also keeps future params back-compatible).
- *Separation of concerns* — `io/` (NWB/pynapple) stays pure I/O returning standard types; scientific logic never imports NWB.

**Sharpenings (these MODIFY specific plan items):**

**A. Standard types over bespoke containers; metadata alongside arrays ("100 functions on 1 data structure").** → Prefer **`xarray.Dataset` + minimal frozen dataclasses-of-arrays** over inventing container classes. Temper the prior review's `SpikeTrains`/`SpatialRatesDataset`/`PlaceFieldsTable`/`DecodedPositionDataset` proliferation: the labeled population backbone should *be* `xarray.Dataset` (a standard type other tools read), with identity riding as a `unit_id` coord / dataclass field — not locked inside a custom class. Add at most one tiny frozen `SpikeTrains` (arrays + `unit_table`), and only because *ragged* spike times don't fit a rectangular array. Result objects remain thin frozen dataclasses whose methods are conveniences over functions-on-arrays — never the *only* path to the data.

**B. Return-type stability; avoid args whose meaning depends on other args; refactor mode-flags into focused functions.** → Do **not** add a `return_posterior="full"|"summary"|"map"` flag that changes `decode_position`'s return shape, and do **not** add `to_dataframe(kind=...)`. Instead: `decode_position` always returns a `DecodingResult` (stable type) with a **lazy / optionally-materialized posterior** and always-available cheap reductions (`map_position`, `mean_position`, `posterior_entropy`) — so the user gets the `(n_time,)` track without ever allocating `(n_time, n_bins)`. Memory knobs (`dtype`, `time_chunk`) control *computation*, not return type. Expose the two table shapes as **two named methods** — `to_dataframe()` (dense) and `summary_table()` (per-unit) — not one flagged method. (This is the maintainer's refinement #3, now backed by the guide.)

**C. Duck typing over `isinstance`; use Protocols.** → Interop accepts pynapple/NWB objects by **duck-typing/Protocols**, converting to arrays at the boundary — never `isinstance(x, TsGroup)`. Define small Protocols (`SpikeTrainsLike`, `PositionLike`) so any object with the right attributes works, keeping arrays the universal contract. (Also lets us fix the `Environment`/`EgocentricPolarEnvironment` sibling-type `isinstance` surprise via a Protocol.)

**One honest tension:** the guide's "functions over custom classes" sits against neurospatial's result-object pattern (which this review praises). Reconciliation: result objects are fine as the friendly layer **iff** they are thin, frozen, hold standard arrays, export to standard types, and every computation they expose is *also* reachable as a function on arrays. Keep them; don't let them grow into a bespoke data universe.

---

## Where I agree, sharpen, or disagree with the prior review

| Prior-review claim | My verdict | Note |
|---|---|---|
| No first-class session; "bring your own orchestration" | **Agree on the problem, disagree on the fix** | Real gap, but build *interop + thin bundle*, not a method-bearing god-object that duplicates pynapple's `TsGroup`/`IntervalSet`/`time_support`. |
| Unit identity not preserved | **Agree, and it's worse than stated** | Identity is dropped at `encoding/_spikes.py:48` and **absent from every result dataclass**; PSTH results don't even inherit `ResultMixin` (`events/_core.py:27,83`). |
| Dense decoding ≈ **23 GB** (1 hr/25 ms/20k bins) | **Arithmetic correct, framing wrong** | 20k spatial bins is extreme. The everyday number is **~1.1 GB at 1,000 bins** (the code's own docstring, `decoding/posterior.py:373`). The real, *unmentioned* hazard is a **3–4× transient peak** from `.copy()` + `exp` intermediates (`decoding/posterior.py:153,218,221`) — a 5.8 GB result needs ~17 GB peak. Prioritize accordingly. |
| xarray should be a data model, not a veneer | **Agree** | `to_xarray()` uses integer coords (`encoding/spatial.py:1057`, `decoding/_result.py:534`) and exists on only 2 of ~8 result classes. |
| Docs teach competing/broken first paths | **Agree, verified at file:line** | `workflows.md` doesn't run; `compute_firing_rate` doesn't exist; `env.plot(field)` is wrong at 11 sites; README/docs still say v0.4.0. |
| Region semantics inconsistent for points | **Agree** | Four APIs disagree on point regions (prior review §6); not re-derived here but confirmed plausible from `environment/regions.py`. |
| Add a `Session` + fit/predict decoder + xarray/NWB-primary | **Mixed** | fit/predict decoder: **agree** (good idiom, keep functions underneath). xarray/NWB *primary/mandatory*: **disagree** — first-class **optional**, NumPy stays lingua franca. dask: **defer** (none exists; premature). |

**Net new findings the prior review missed:** the transient-memory multiplier in decoding; the dense `(n_bins, n_bins)` diffusion-kernel wall (scales as `n_bins²`, independent of neuron count, `encoding/_smoothing.py:414`); the library's **own docstrings teaching the per-neuron loop** (`encoding/population.py:220-225,265`); `to_dataframe` having **two different meanings under one name** — dense `(unit, bin)` table vs per-unit summary (`encoding/_base.py:324` vs the batch overrides) — a UX + scale hazard; double metric recomputation in `to_dataframe` (`spatial.py:1517` → `1387`); and the **CRITICAL silent units footgun** below.

---

## The single most important issue: silent wrong science from unit/time mismatch

**`encoding/_binning.py:120-141`.** Spikes outside `[times.min(), times.max()]` — and spikes that map to inactive bins — are **silently dropped, no warning.** If `spike_times` are in ms and `times` in s (a textbook NWB/Spyglass mix-up), nearly all spikes are filtered and the user gets a **plausible-looking near-empty place field or decode with no error.** The per-array entry validators (`encoding/_validation.py`) check each array's internal sanity but **never cross-check that `spike_times` and `times` share a range/units.**

This is worse than any crash: it produces confident, publishable, wrong numbers. **Fix:** when the in-window fraction is low (e.g. <50%, or 0), warn loudly naming both ranges and the units hypothesis. One warning defuses the most common silent failure for this exact user. This should ship before any new abstraction.

Related boundary footguns:
- **Transpose into `decode_position`** is only caught for non-square inputs (`decoding/likelihood.py:151`); square populations / consistently-transposed pairs decode garbage silently. `bin_spikes_in_time(orient=...)` mitigates but the boundary itself doesn't assert.
- **`align_spikes_to_events`** documents NaN/Inf rejection for event times but only checks NaN (`events/alignment.py:156`).
- **`behavior/trajectory.py:132,289,591`** read `.ndim` before `np.asarray`, so a list input throws a non-domain `AttributeError`.
- **`Environment()`** bare-call error is a low-level `ValueError("layout parameter is required")` (`environment/core.py:335`) with no pointer to factories — the #1 documented gotcha surfaces as the worst message.
- **User-facing errors reference `CLAUDE.md`** (`layout/validation.py:334,400`), invisible to a pip-installed user.

---

## Findings by theme (prioritized, each grounded in code)

### A. Docs actively teach the wrong idiom (cheap, urgent)
- `docs/user-guide/workflows.md:17-158` — the canonical "Place Field Analysis" page uses undefined loaders (`load_position_data`, `:25`), hand-rolls occupancy via `np.histogram` (`:44`), reaches into `env.layout.grid_shape`/`active_mask` + `scipy.ndimage` (`:86-99`), and calls the **non-existent** `compute_firing_rate` (`:304,312`). It contradicts the (excellent) README path.
- `env.plot(field, ...)` is wrong (first positional is `ax`, `environment/visualization.py:134`) at 11 doc sites (`rl-primitives.md`, `spike-field-primitives.md`); correct method is `env.plot_field`.
- Stale `v0.4.0` in `README.md:85,660`, `docs/index.md:97,109` (package is 0.5.0).
- The flagship decode notebook models **bad practice**: manual `np.histogram` loop + per-neuron `compute_spatial_rate` list-comp (`examples/20_bayesian_decoding.py:231-242,290-292`) instead of `compute_spatial_rates(...).firing_rates` + `bin_spikes_in_time` — the very helpers `__init__.py:21` advertises.

### B. The encode→decode golden path doesn't compose (high)
- `compute_spatial_rates(...).firing_rates` is `(n_neurons, n_bins)` and feeds `decode_position`'s `encoding_models` directly; `bin_spikes_in_time` defaults to the exact `spike_counts` shape. The pieces fit — **but nothing wraps them.** The 1000-neuron user hits two-matrix-plus-`dt` shape bookkeeping at the worst moment. The param name even mismatches the producer (`encoding_models` vs `.firing_rates`, `decoding/posterior.py:269`). **Fix:** a one-call `decode_session(env, spike_times, times, positions, dt=…)` wrapper (glue, not new math).

### C. Unit identity is structurally absent (high)
- Dropped at `encoding/_spikes.py:48` (`normalize_spike_times → list[NDArray]`, no id channel); no `unit_ids` param on any batch compute; no `unit_ids` field on any result dataclass (`SpatialRatesResult`, `spatial.py:862`). The only attach point is a late-bound, display-only `neuron_ids=` on four `to_dataframe`s.
- `to_xarray()` labels with `np.arange` (`spatial.py:1057`, `_result.py:534`) → `da.sel(neuron=42)` selects *row* 42, not unit 42 — a silent label-vs-position trap. `bin` coord is an integer, not `bin_center`.
- No labeled selection (`sel(unit_id=…)`), no shared index, no cross-domain merge (place+HD+OVC for one unit).

### D. API consistency defects that block extrapolation (high, cheap)
Nearly every API problem is a *consistency* defect — the user can't guess domain N+1 from domain N:
- **`detect_region_crossings(position_bins, times, region_name, env, …)`** puts `env` 4th (`segmentation.py:321`) while every sibling puts it 3rd (`detect_laps:989`, `segment_trials:1620`). Latent silent off-by-one; violates the project's own stated argument order.
- **Batch classifier has 4 names** for one job: `detect_cell_types` / `detect_ovcs` / `detect_view_cells` / `detect_hd_cells` (`spatial.py:1303`, `egocentric.py:928`, `view.py:824`, `directional.py:1336`). `detect_ovcs` is an unspelled acronym.
- **Place cells lack an `is_place_cell()` predicate** though the other three cell types have `is_X_cell` (free + method).
- **`peak_location` vs `peak_view_location`** differ only by a gratuitous `view_` infix (`spatial.py:218` vs `view.py:257`); plural-ness of method names doesn't track single-vs-batch.
- **`to_dataframe` means two different things under one name:** dense one-row-per-(neuron,bin) (mixin, `_base.py:324`) vs summary one-row-per-unit (batch overrides). Not a broken invariant — the base contract allows heterogeneous long-form — but a real UX + scale hazard: a beginner who expects a summary gets a 10M-row frame. Fix by naming the two shapes distinctly.
- **Two "egocentric" submodules**: `ops.egocentric` (transforms) vs `encoding.egocentric` (object-vector fields) — same word, different place.
- **341 public names across 10 submodules; ~2 reachable from `ns.`** A beginner must learn the tier taxonomy to find anything.
- **6+ factories named by input shape** (`from_samples`/`from_graph`/`from_pixel_mask`…), not experiment (open field / linear track / maze); `from_samples` has 9 keyword knobs (5 are morphology cleanup). `from_polar_egocentric` returns a *non-*`Environment`.

### E. Scale cliffs (high)
- **Dense float64 posterior, no `return_posterior`/`float32`/`chunk` modes** (`decoding/posterior.py:518`; grep confirms nothing). Users almost always want the `(n_time,)` MAP/entropy reductions, not the full `(n_time, n_bins)` array. Add `return_posterior="full"|"summary"|"map"`, `dtype=float32`, `time_chunk=N`. **~99.9% memory cut for the common case; removes the only hard OOM cliff.**
- **Dense `(n_bins, n_bins)` diffusion kernel**, cached on env (`encoding/_smoothing.py:414`): 3.2 GB at 20k bins, scaling `n_bins²` independent of neuron count, even though diffusion is intrinsically local. Make it sparse/banded — the batch GEMM (`kernel @ counts.T`, `_smoothing.py:658`) works transparently with a sparse kernel.
- **The library teaches the per-neuron loop** in `population.py:220-225` and even in an error message (`:265`); `compute_spatial_rates` (shares occupancy/kernel once) is discoverable only by a trailing `s`.
- **`to_dataframe` explodes** to `n_neurons × n_bins` rows (10M rows for 2000×5000; `_base.py:324`) and **recomputes grid/border scores twice** (`spatial.py:1517` → `detect_cell_types` at `1387` recomputes them).

### F. Strengths to preserve (don't break these)
- README "Your First Place Field" is excellent and runnable (`README.md:175-230`).
- `compute_spatial_rates` correctly shares all trajectory/occupancy/kernel work once and smooths via a single BLAS-3 GEMM (`encoding/_binning.py:323`, `_smoothing.py:646`).
- Decoding likelihood exploits spike-count sparsity (CSR matmul, `decoding/likelihood.py:182`) and has strong unconditional correctness guards (`posterior.py:431,494`).
- The numerical-core error culture is genuinely mature (`encoding/_validation.py:158`, `ops/egocentric.py:733` WHAT/WHY/HOW, `layout/.../regular_grid.py:163` "No active bins" diagnostic). The fix is to extend that culture to the *boundaries*.
- Result-object pattern (`.plot()`/`.summary()`/metrics on results) is the best UX decision in the codebase — generalize it, don't replace it.

---

## Recommended architecture: interop-first, three thin layers

Keep the existing primitives as the composable core. Add **thin** layers — and crucially, **do not own the time/spike/epoch data model**; borrow it.

1. **Primitive layer (keep as-is):** `compute_spatial_rate(s)`, `decode_position`, `bin_spikes_in_time`, `env.occupancy`, etc. Array-friendly, typed, testable.

2. **Labeled-result layer (add, lightweight):** thread `unit_ids` (+ optional unit metadata) onto every population result; `to_xarray()` → `xr.Dataset` with real `unit_id` / `bin_center_x/y` coords on *all* batch results; split the overloaded `to_dataframe` into `summary_table()` (per-unit, `unit_id`-indexed) and a strictly-dense `to_dataframe()`. Make PSTH results inherit `ResultMixin`.

3. **Interop + thin convenience (add, do NOT over-build):**
   - **Accept pynapple objects** (`TsGroup`/`Tsd`/`IntervalSet`, optional dep, duck-typed) at function boundaries → users get `spikes.restrict(run_epochs)`, unit metadata, and epoch selection *for free*, with neurospatial writing zero time-series machinery. This closes the largest gap (no epoch selection, no identity) at the lowest cost.
   - **`decode` as a small `fit/predict` object** over the functional core: `BayesianDecoder(env).fit(spikes, position, epoch=…).predict(spikes, return_posterior="summary").score(…)`. Keep `decode_position` underneath. Differentiator must be the `Environment` (geodesic/linearized decoding), since pynapple already has `decode_1d/2d`.
   - **`load_session()` returns a small frozen dataclass** `(env, position, spikes, epochs)` — a labeled *bundle* for discoverability, not a mutable method-bearing god-object. Lazy NWB reads (`lazy=True`; reads are eager today, `io/nwb/_behavior.py:85`).
   - **Experiment-shaped factory presets** as 3-line wrappers over `from_*`: `Environment.open_field(positions, bin_size)`, `.linear_track(...)`, `.maze("w", ...)` — same first two args, each documents which `from_*` it delegates to.

This matches Scientific-Python guidance (prefer standard types + metadata-alongside-arrays over bespoke containers) and meets users where the spikeinterface→pynapple→NWB pipeline already lives.

---

## Prioritized roadmap

### Phase 0 — Stop wrong science & fix the front door (days, near-zero risk)
1. **Warn on spikes dropped outside the trajectory window / inactive bins** (`encoding/_binning.py:120-141`) — the critical units footgun.
2. **Normalize-at-boundary** (`np.asarray` before `.ndim`) in `behavior/trajectory.py` and audit all public entry points; symmetric Inf check in `align_spikes_to_events`; upgrade bare-`Environment()` error; remove `CLAUDE.md` from user-facing errors.
3. **Docs sweep:** rewrite `workflows.md` to the README idiom; kill `compute_firing_rate`; `env.plot(field)` → `env.plot_field`; bump v0.4.0 → v0.5.0; fix the decode notebook to use `compute_spatial_rates`/`bin_spikes_in_time`.
4. **`decode_session(env, spike_times, times, positions, dt=…)`** one-call wrapper.

### Phase 1 — Identity & consistency (1–2 weeks, mostly mechanical)
5. Thread `unit_ids` onto every population result; `to_xarray()` → labeled `xr.Dataset` (real `unit_id` + `bin_center` coords) on all batch results.
6. Split `to_dataframe` → `summary_table()` + dense `to_dataframe()`; give PSTH results `ResultMixin`.
7. **One API-naming contract** (write into CLAUDE.md, enforce): predicate `is_X_cell`; batch `.classify()`; peak `peak_location(s)`; `env` slot consistent. Fix `detect_region_crossings` arg order; add `is_place_cell`; unify the 4 batch-classifier names; collapse `peak_view_location`.
8. Experiment-shaped factory presets; demote morphology knobs.

### Phase 2 — Scale safety (2–4 weeks)
9. Memory-safe decoding **without a return-type-changing flag** (constraint B): `DecodingResult` stays the stable return type with a **lazy/optionally-materialized posterior** and always-available `map_position`/`mean_position`/`posterior_entropy` reductions; add `dtype=float32` + `time_chunk=N` memory knobs and design the result so a chunked/dask backend can slot in later. — the OOM fix.
10. Sparse/banded diffusion kernel; fix double metric recompute; thread `n_jobs` into per-neuron loops; steer to `compute_spatial_rates` in docs/errors.

### Phase 3 — Interop & ergonomics (1–2 months)
11. Accept pynapple `TsGroup`/`IntervalSet` at boundaries (+ `from/to_pynapple` shim); lazy NWB; result round-trip (`read_place_field`, units-aligned `write_spatial_rates`).
12. `BayesianDecoder` fit/predict over the functional core; thin `load_session()` bundle; preset/annotation→analysis polish.

---

## Top 10 concrete changes (highest leverage first)

1. Warn-don't-drop on out-of-window / inactive-bin spikes (`encoding/_binning.py:120-141`).
2. Memory-safe decoding: lazy/optional posterior + always-available reductions on `DecodingResult`, `dtype=float32`, `time_chunk` (stable return type, no mode flag — constraint B) (`decoding/posterior.py:518`).
3. Rewrite `workflows.md` + kill `compute_firing_rate` + fix `env.plot(field)` (11 sites) + version bump.
4. `decode_session(...)` golden-path wrapper.
5. Thread `unit_ids` through results; `to_xarray()` → labeled `xr.Dataset`.
6. Split `to_dataframe`/`summary_table`; PSTH results get `ResultMixin`.
7. One naming contract; fix `detect_region_crossings` arg order; add `is_place_cell`; unify batch-classifier names.
8. Sparse diffusion kernel + stop teaching the per-neuron loop (`population.py`).
9. Accept pynapple objects at boundaries (restrict-to-epoch + identity for free).
10. Experiment-shaped factory presets (`Environment.open_field/linear_track/maze`).

---

## The meta-point

Almost none of this is a missing algorithm. It is **consistency, safety-by-default, durable identity, and interop**. The cheapest, highest-leverage work (Phase 0) is also the most urgent — it stops the library from silently producing wrong science and from teaching a broken first path. The bigger ergonomic wins (interop, fit/predict, presets) follow, and they're best achieved by *borrowing* the ecosystem's solved data model rather than building a parallel one around neurospatial's genuine moat, the `Environment`.
