# Changelog

## [Unreleased]

### Added — `method="glm"`: penalized-Poisson GAM estimator (spatial only)

`compute_spatial_rate` and `compute_spatial_rates` gain a fourth estimator,
`method="glm"`: a batched penalized-Poisson generalized additive model that uses
the cached finite-volume eigenbasis as its smoothness-penalty basis. Occupancy
enters as a **log-offset** (never a denominator) and the smoothness penalty `λ`
is chosen by REML, so the fit returns **finite firing rates everywhere** —
including the low-occupancy and unvisited bins where the ratio estimators
(`diffusion_kde` / `gaussian_kde` / `binned`) produce `NaN`.

- **New keyword-only params** on `compute_spatial_rate(s)`: `penalty`
  (fixed `λ ≥ 0`; `None` → choose by REML) and `rank` (requested basis rank;
  out-of-range values are clamped, not rejected, to the effective rank reported
  via `result.rank`). These are **mutually exclusive** with the ratio-method
  params `bandwidth` / `min_occupancy` / `fill_value`, which now default to
  `None` (resolving to their historical `5.0` / `0.0` / preserve-`NaN`
  behavior). Combining a param with the wrong `method` raises `ValueError`.
- **New result fields** on `SpatialRateResult` / `SpatialRatesResult`:
  `coefficients`, `penalty`, `penalty_weights`, `rank`, `deviance`, `converged`,
  `n_iter`, `reml_objective` (all `None` for the ratio methods). `bandwidth` is
  `None` for glm. `summary_table()` gains the GAM scalar columns for glm results.
- **Spatial only.** `compute_view_rate(s)`, `compute_egocentric_rate(s)`, and the
  directional encoders keep their ratio-only `method` set (no `"glm"`). NWB
  persistence of the GAM fields and glm decoding land in a later release; until
  then `write_spatial_rates` **rejects** a glm result (raises `NotImplementedError`)
  rather than persist a lossy record with the GAM diagnostics dropped.
- The default `method="diffusion_kde"` path is unchanged (byte-for-byte).

### Changed — BREAKING: `smoothing_method` renamed to `method` (estimator axis)

The smoothing/estimator keyword is now uniformly named `method` across every
smoothing encoder, result class, the decoder, and NWB metadata — one name, one
meaning. This is a **hard rename with no alias** (per project policy); callers
that named `smoothing_method=` or read `.smoothing_method` must update. Callers
that never touched it see identical behavior — the default stays
`"diffusion_kde"` (and `"binned"` for the egocentric encoders), and no numerics
change.

- **Renamed keyword `smoothing_method` → `method`** on `compute_spatial_rate`,
  `compute_spatial_rates`, `compute_view_rate`, `compute_view_rates`,
  `compute_egocentric_rate`, `compute_egocentric_rates`, `is_place_cell`,
  `compute_directional_place_fields`, `decode_session`,
  `decode_session_summary`, `BayesianDecoder`, and
  `simulation.validate_simulation`. Each encoder keeps its existing method value
  set and default.
- **Renamed field `smoothing_method` → `method`** on `SpatialRateResult`,
  `SpatialRatesResult`, `ViewRateResult`, and `ViewRatesResult`.
- **NWB:** `write_spatial_rates` now stores the estimator under the metadata key
  `"method"` (was `"smoothing_method"`); the spatial-rates schema version bumps
  to `2.0`. This is a clean break with no back-compatibility shim — spatial-rates
  tables written by earlier versions (schema `1.x`) no longer read.
- **`decode_session_summary`:** because `method` now names the smoothing
  estimator, it is no longer accepted as a per-block decode kwarg. The Poisson
  observation model (the only supported likelihood) is applied unconditionally.
- `method="glm"` (a penalized-Poisson GAM estimator, `compute_spatial_rate(s)`
  only) lands on this renamed axis — see the "Added — `method="glm"`" entry
  above.

### Added — `env.diffuse`: matrix-free diffusion smoothing (0.8.0, performance)

Smoothing no longer materializes the dense `(n_bins, n_bins)` diffusion kernel on
its hot paths. A new method

```python
env.diffuse(fields, bandwidth, *, mode="density", backend="numpy")
```

applies the finite-volume heat operator `H = exp(-t L)` **without building an
`(n, n)` matrix**, using a cached, per-component, bandwidth-aware **truncated
symmetric eigenbasis** of `S = M^{-1/2}(D − W) M^{-1/2}`. Time and memory scale
with `n_bins × rank` (`rank ~ measure(domain)/σ^d`), not `n_bins²`, so smoothing
now scales to large/fine grids. The eigenbasis depends only on geometry, so it is
built once and reused across every bandwidth, mode, and neuron (auto-invalidated
when the environment is mutated). `fields` may be 1-D `(n_bins,)` or a 2-D batch
`(n_bins, n_fields)`.

- **`env.diffuse` is a pure linear operator** — no output clip, no
  renormalization. The always-retained per-component null mode makes mass
  conservation exact under truncation (`mode="transition"` preserves `sum`;
  `mode="average"` preserves a constant field), so smoothing a **signed** field
  is preserved. Positivity, where required, is the consumer's job.
- **`env.smooth` now routes through `env.diffuse`** and gains a documented
  **approximation contract**: it reproduces the dense kernel to within a
  near-lossless truncation tolerance (dropped modes contribute ≤ `tol` (default
  `1e-6`) in the **M-weighted norm**; the raw per-bin error carries a
  volume-conditioning factor `κ(M) = sqrt(max vol / min vol)`, worst on polar
  `r→0` / skewed mesh). On a non-negative input the linear apply may leave
  tolerance-level negatives (bounded relative to the dense output) instead of the
  dense kernel's exact 0-floor — clip the result yourself if you need a strict
  floor.
- **`backend="jax"`** runs the apply **in JAX** (the cached NumPy eigenbasis is
  cast to `jnp`), so `jit` / `grad` / GPU work through the smoothing. The JAX
  `diffusion_kde` encoders no longer round-trip through NumPy.
- The `diffusion_kde`, `binned`, and diffuse-`resample_field` consumers route
  through `env.diffuse`; strict `> 0` support gates (`binned`, resample) derive
  their support from the **W-component structure** (exact, truncation-proof), not
  the smoothed denominator's sign, so truncation cannot emit a spurious `NaN`.
  `diffusion_kde` clips its own output `≥ 0` (decode nonnegativity).

**Breaking:** the `kernel=` parameter is **removed** from
`neurospatial.encoding._smoothing.smooth_rate_map` and `smooth_rate_maps_batch`.
Passing a precomputed dense kernel is obsolete — the cached eigenbasis now
provides the cross-neuron reuse the parameter existed for. There is no
backward-compat shim (per the project default); callers passing `kernel=` should
simply drop it. `env.compute_kernel` is **unchanged** (it still returns the dense
`(n, n)` matrix via the `expm` path, byte-identical, for callers that genuinely
need the matrix), as is `transitions(method="diffusion")`.

### Changed — diffusion smoothing bandwidth is now the true physical σ (breaking, correctness)

The `diffusion_kde` / diffusion-kernel `bandwidth` is now the **true physical
standard deviation (σ)** on every environment layout, independent of bin size or
resolution. Previously the effective smoothing width scaled with bin size (2D
place fields at `bin_size=1` oversmoothed by ~70% for the same `bandwidth`), so
place-field widths and spatial information were not comparable across
resolutions or with other tools. This is a **behavior-changing correctness fix**:
smoothed values change even on uniform grids — that difference *is* the
correction.

- **New finite-volume operator.** The kernel is now the finite-volume heat
  operator `H = exp(-t L)`, `t = σ²/2`, `L = M⁻¹(D − W)` with
  `W[i, j] = A[i, j] / d[i, j]` (`A` = shared-face measure, `d` = center
  distance) and `M` = per-bin cell volumes. On any K-orthogonal layout this has
  the continuum limit `−∇²`, so a point source smooths to physical σ exactly.
- **Impact on existing results.** For the same `bandwidth`, **2D place fields
  now smooth ~1.5–1.7× less**. To approximate the *old* amount of smoothing,
  scale `bandwidth` by roughly `√(bin_size)` (rough, mode-dependent — prefer
  re-choosing `bandwidth` as a physical σ in your units).
- **Correct orientation per consumer.** `compute_kernel(mode="transition")`
  returns `Hᵀ` (column-stochastic, mass-conserving smoothing of extensive
  quantities); `mode="density"` returns `H·M⁻¹` (count → density, integrates
  to 1 under bin volumes); `transitions(method="diffusion")` now returns the
  **row-stochastic** `H` via transpose (correct on non-uniform bin volumes —
  polar, mesh — where it no longer assumes symmetry).
- **Breaking: public `neurospatial.ops.compute_diffusion_kernels` signature.**
  The old Gaussian-weight form
  `compute_diffusion_kernels(graph, bandwidth_sigma, bin_sizes=..., mode=...)`
  is **replaced** by
  `compute_diffusion_kernels(graph, *, volumes, sigma, mode)`. The face measure
  is read from a per-edge `"A"` attribute (single source of truth, like
  `"distance"`); a missing `"A"` on an edge raises, and `A == 0` means no
  diffusion across that edge. `mode` now also accepts `"average"` (row-stochastic
  `H`) at the low level.
- **`min_occupancy` caveat.** `min_occupancy` is documented in seconds, but the
  KDE paths compare it to `occupancy_density = K @ occupancy`, which under the
  new density kernel is seconds-per-cell-volume on non-uniform grids — a
  pre-existing unit mismatch that this change neither fixes nor worsens for
  uniform grids, and that lies **outside** the grid-invariance guarantee.
  Tracked as a follow-up.
- **Out of scope (unchanged):** nonuniform-Cartesian custom `grid_edges` inherit
  the uniform-cell approximation and are excluded from the physical-σ guarantee;
  performance (dense `expm`) is unchanged.

### Added — `mode="average"` intensive-field smoother

- **New public `mode="average"`** on `Environment.compute_kernel` and
  `Environment.smooth` (and the low-level `compute_diffusion_kernels`): the
  row-stochastic heat operator `H` that **averages an intensive field** (a rate
  map or probability *density*). Unlike `density` (`H·M⁻¹`), it carries no
  cell-volume bias on non-uniform bin volumes (polar, mesh), and unlike
  `transition` it is the correct choice for a rate map rather than a total.
  Discrete probability *mass* (a posterior summing to 1) still uses `transition`.
  `env.smooth`'s **default stays `density`** (unchanged behavior for existing
  mode-less calls); `average` is recommended for intensive rate maps.
- **`smooth_rate_map(method="binned")`** and **`resample_field(method="diffuse")`**
  now smooth their intensive fields through the row-stochastic average
  (masked / valid-bin-normalized), removing the volume bias on non-uniform `M`.
  For `resample_field`, this also fixes a down-bias where covered bins adjacent
  to an uncovered region were pulled toward zero, and makes a source `NaN`
  interpolate from valid neighbours instead of propagating. Uniform-grid results
  are unchanged (the per-cell volume factor cancels in the rate/weight ratio).

### Corrected guidance

- **`method="binned"` is not a dense-kernel memory mitigation.**
  Earlier notes (through v0.6.0) recommended `binned` to avoid the dense
  `(n_bins, n_bins)` kernel, but `binned` smooths its rate map through the same
  diffusion kernel (via `env.smooth`), so **all** smoothing methods build a dense
  kernel. The only memory mitigation is reducing the bin count (a larger
  `bin_size`). Docstrings and the high-bin warnings are corrected accordingly.

### UX hardening — fail-loud errors, one count convention, docs & viewer polish (0.8.0)

A broad usability pass: silent failures now raise or warn with actionable
messages, a few APIs were made consistent, and the docs/viewers were polished.

#### Breaking changes

- **`add_positions` is now `add_positions(events, *, times, positions)`** —
  reordered to the canonical `(data, times, positions)` order and made
  **keyword-only**. A bare 1-D trajectory makes both arrays 1-D, so a positional
  swap was shape-indistinguishable and silently mis-interpolated; keyword-only
  removes the ambiguity. *Migration:*
  `add_positions(events, times=times, positions=positions)`.
- **Assembly functions take `(n_time_bins, n_neurons)`** — `detect_assemblies`,
  `assembly_activation`, `pairwise_correlations`, and `reactivation_strength`
  now use the same time-first count convention as `decode_position` and
  `bin_spikes_in_time`, so a binned count matrix feeds them directly.
  *Migration:* pass the default `bin_spikes_in_time(...)` output as-is, or
  transpose an existing `(n_neurons, n_time_bins)` matrix.
- **Behavior result `summary()` returns a `dict`** of scalar metrics (was a
  formatted string) on every `behavior` result class. *Migration:* use
  `str(result)` for the human-readable form.
- **Fail-loud instead of silent sentinels.** `bin_at` on a graph/track
  environment raises on wrong-dimension points (was all `-1`);
  `heading_from_velocity` raises when every sample is below `min_speed` (opt out
  with `allow_all_nan=True`); graph queries reject a multi-point coordinate batch.

#### Added

- **`units=` and `frame=` keyword args** on the factory methods (`from_samples`,
  `open_field`, `linear_track`, `maze`) set the metadata at construction:
  `Environment.from_samples(pos, bin_size=2.0, units="cm")`.
- **Graph queries accept coordinates**, not just bin indices — `neighbors`,
  `path_between`, and `reachable_from` map a coordinate via `bin_at` (integers
  are still treated as indices).
- **`ResultMixin` on every result class** — concise `__repr__`/`_repr_html_`
  (summary metrics, not array dumps) and a scalar `summary()` dict.
- `dir(neurospatial.ops)` now surfaces the lazily-exported ops (autocomplete).

#### Fixed

- **`min_occupancy` thresholds raw occupancy seconds, not smoothed density** —
  fixes silent all-zero place fields on the documented golden path; applied
  identically across `diffusion_kde` / `gaussian_kde` / `binned` (single and
  batch). `min_occupancy=0.0` (the default) is unchanged.
- **Grid allocation is preflighted** — a transposed `(2, N)` trajectory (read as
  N dimensions) raises in ~1 ms instead of OOMing in `histogramdd`/`meshgrid`; a
  likely-transposed array now *warns* rather than false-rejecting a valid
  low-sample N-D environment.
- Non-finite query rows map to the `-1` "outside" sentinel **per row** (one NaN
  no longer collapses the whole occupancy/rate map on graph/masked envs).
- `to_file` refuses to overwrite by default (`overwrite=True` to allow);
  `occupancy()` checks shape before monotonicity (clear swapped-argument error);
  `from_graph` validates the edge `distance` attribute.
- `SpatialRateResult.plot()` documents the real `colorbar`/`colorbar_label`
  kwargs and labels the colorbar "Firing Rate (Hz)"; batch `plot()` raises an
  actionable error when no unit index is given.
- **Interactive-viewer accessibility** — the standalone HTML player ignores
  global keyboard shortcuts when an interactive control has focus, adds
  `:focus-visible` styles, and stops announcing every autoplay frame to screen
  readers; the napari region dock and track builder gain a compact layout,
  visible labels, and accessible controls.

#### Documentation

- Docs version bumped to 0.8.0; dead links and the fork-clone step fixed; a
  value-first, runnable quickstart; grouped/collapsible nav; a new advanced
  architecture page; colorblind-safe `viridis` in examples; example-notebook
  links normalized with an internal link-check step in docs CI.

## [v0.6.0] - 2026-07-03

## What's Changed

### Features
- feat(io/nwb): SpatialRatesResult round-trip (write_spatial_rates/read_place_field) + lazy reads (92cb14a)
- feat(decoding): add immutable BayesianDecoder (fit/predict/predict_summary/score) over the decode core (5ec00da)
- feat(recording): add frozen Session bundle + load_session (from_arrays/from_nwb, with_environment/restrict) (5edf393)
- feat(behavior): add restrict/in_epochs/restrict_spike_trains (array or IntervalSet epochs) (9b940fa)
- feat(encoding): add SpikeTrains ragged-spike container (label access, filter, unit_table) (1eba7ca)
- feat(typing): add PositionLike/SpikeTrainsLike/EnvironmentLike Protocols + pynapple ingress (optional) (b35330c)
- feat(decoding): honor float32 end-to-end through decode_session(_summary) via a dtype knob (fd58811)
- feat(encoding): speed filtering masks both spikes and occupancy in encode path (c514061)
- feat(decoding): add memory-safe summary decode + dtype/time_chunk on decode_position (afc1d68)
- feat: Update documentation and code to reflect 'unit' terminology for v0.6 API consistency (a5520ab)
- feat: Add ux-reviewer agent for evaluating user experience in scientific software (b2d3cdb)
- feat: Update API for detect_region_crossings and enhance documentation (0c0488e)
- feat(environment): experiment-shaped factory presets (open_field, linear_track, maze) (928a921)
- feat(encoding,behavior,decoding): enforce naming contract (classify/label_cell_types/is_place_cell/peak_location; decode_position result arg; deprecation aliases) (e3c9c94)
- feat(results)!: split terminal verbs — dense to_dataframe + new summary_table; PSTH ResultMixin (breaking, D1) (ccc72a6)
- feat(results)!: to_xarray returns a labeled xr.Dataset (breaking, D1) (14fe4f3)
- feat(encoding,events): thread unit_ids through population results and compute functions (c3e29e9)
- feat(decoding): add decode_session() one-call encode->bin->decode golden path (b1fabc4)

### Bug Fixes
- Merge pull request #25 from edeno/phase-3-pynapple-validation (6d8f7e2)
- fix(io/pynapple): validate shapes in to_pynapple (ValueError, not raw AssertionError) (241ed10)
- Merge pull request #24 from edeno/phase-3-trajectory-extra (3d635b1)
- fix(decoding): drop nonexistent neurospatial[trajectory] hints (scikit-image is core) (51e1ac5)
- Merge pull request #23 from edeno/phase-3-nwb-extra-msgs (8f4ac4d)
- fix(io/nwb): ndx-pose/ndx-events ImportError hints name the real [nwb] extra (14a8d93)
- Merge pull request #22 from edeno/phase-3-bugfixes (e5c11d9)
- fix(interop): restrict speed under fit(epoch=); mixed-type unit_ids; correct nwb extra name (56d0b51)
- fix(io/nwb): persist+validate env on rates round-trip (no degenerate env); atomic write; occupancy integrity; lazy length checks + unit caching; add nwb CI job (fefe6c3)
- fix(decoding): score warns/raises on undecodable bins; validate BayesianDecoder config+fitted state; add is_fitted, warn_on_drop, score distance=; error provenance (ff606f3)
- fix(recording): raise on malformed NWB env (not silent None); add environment_name; warn on empty restrict; Position self-validates (822ebc4)
- fix(behavior): raise on ambiguous nested-list epochs; validate closed on empty epochs; doc NaN exclusion (a3d6ee2)
- fix(typing): extract TsGroup trains by unit-id index (not iterate-keys); narrow EnvironmentLike; fix pynapple tests (9fd8d86)
- fix(decoding): route log_poisson_likelihood/poisson_likelihood dt through shared validate_dt (9b939a5)
- fix(decoding): validate dt in decode_position_summary too; test decode_position(_summary) dt guards; unify dt message; fix examples README sync wording (917ad4b)
- fix(decoding): unify dt validation (reject str/bool) via shared helper; clean ValueError for unparseable dtype (726ed71)
- fix(decoding): clean error for non-numeric/bool dt; restrict sync prune to numbered examples (23501df)
- fix(decoding): validate dt (finite, >0) in decode_session(_summary) before grid math (2766022)
- fix(decoding): preserve float32 in result-object handoff; validate time_chunk as positive int (not bool/float/str) (20b0e63)
- fix(decoding): reject time_chunk=None in summary decoders (it defeats the never-materialize-full-posterior contract) (eeb3d80)
- fix(encoding): warn when interval filtering empties the rate map (max_gap/out-of-bounds, not just min_speed) (7536d53)
- fix(decoding): validate prior shape in decode_position_summary (was silently truncating over-long 2D priors) (8b18f59)
- fix(encoding): align spike counts with occupancy on the FULL interval mask (max_gap, out-of-bounds, speed) (87b2164)
- fix(encoding): raise on speed-without-min_speed; warn on all-excluded speed filter; tighten final-sample gate (d13c968)
- fix(decoding): robust scaling test, precise float32 summary reductions, quiet degenerate-row warning (0847aa7)
- fix: address Phase 1 PR review (linear_track NaN guard, unit_table validation, all-NaN peaks, decode_position error, maze kind, honest annotations, docs) (92067e7)
- fix(encoding,docs): add peak_locations() to batch results; correct CLAUDE.md peak/cell-type contract; dedup CHANGELOG (aee29bf)
- fix(environment): order-based (coordinate-independent) maze topology; no track_graph mutation (9d9cc69)
- fix(results): raise (not silently skip) on xarray coord shape mismatch; omit None units (eeac275)
- fix(decoding): validate times up front in decode_session (beginner-grade non-finite error) (ec427eb)
- fix(decoding): warn on out-of-window spike drop in decode_session (units footgun) (09b1dea)
- fix: address Phase 0 PR review findings (E1006 code, docs accuracy, test gaps) (3f779e7)
- fix(decoding): keep decode_session export in neurospatial.decoding only (e6b2030)
- fix(environment): distance_to raises RegionNotFoundError for missing region (KeyError-compatible) (85410c0)
- fix(errors): remove internal-doc (CLAUDE.md) references from user-facing errors (63c8062)
- fix(environment): correct factory examples + error code in bare-Environment() message (3eaa207)
- fix(environment): beginner-grade error for bare Environment() pointing to factories (80d6ae7)
- fix(events): reject Inf event_times in align_spikes_to_events (matches docstring) (a364db1)
- fix(behavior): normalize array-likes at trajectory public boundaries before .ndim (2d4b5f8)
- fix(encoding): address review findings on spike-drop warning (4b3dfa5)
- fix(encoding): warn when spikes are dropped outside trajectory window or inactive bins (0d04d01)

### Documentation
- docs(interop): add Interoperability guide (session-first-optional: pynapple/NWB/Session/SpikeTrains/BayesianDecoder) (7848436)
- docs(plan): add Phase 3 interop/ergonomics design RFC (locks Protocols, Session, BayesianDecoder) (a0dc512)
- docs(decoding): widen likelihood Raises docstrings to match new dt validation (65b0e2c)
- docs(changelog): sync hosted changelog with latest v0.6 fixes (dt/dtype validation, float32 result handoff, time_chunk) (bdf2afd)
- docs(animation): fix broken per-frame memmap rate-map example; clarify jupytext vs docs-mirror sync; soften index parity claim (06f5e91)
- docs(examples): add missing example-23 row to index; correct CI sync guidance; prune stale mirrors in sync script (ebf0397)
- docs: surface decode_session_summary/scale APIs (api index, workflow long-session callout, batch processing, snippet CI) (15dd728)
- docs(changelog): sync hosted changelog with v0.6 scale API (dtype end-to-end, time_chunk hybrid + None rejection, summary speed forwarding) (e3eb03c)
- docs(encoding): correct compute_spatial_rates dtype note for R8 (decode_session now honors float32 end-to-end); test blockwise prior validation (cc7c498)
- docs: teach compute_spatial_rates batch path in README/animation/overlays/metrics (not per-neuron loops) (fe83cc4)
- docs(changelog): smoothing kernels warn and proceed (no hard gate / allow_large) (4a099e2)
- docs(examples): re-sync stale bayesian-decoding mirror from source; add CI sync guard (961ea4c)
- docs(decoding): teach batch/decode_session path not the per-neuron loop; add Phase 2 entries to docs changelog (241a8d5)
- docs/fix(decoding): correct time_chunk memory claim; DecodingSummary map_estimate + post-init validation; test gaps (59b3a56)
- docs(encoding): clarify float32 dtype decode-memory scope; fix batch return annotations (4dbfbc9)
- docs(encoding): teach compute_spatial_rates batch path, not the per-neuron loop (952b861)
- docs(encoding): note joblib warning-swallowing in population_coverage; coerce label_cell_types score dtype (27be8cf)
- docs: v0.6 naming contract in CLAUDE.md + migration guide + breaking-change CHANGELOG (1.6, 1.7) (c91184a)
- docs(encoding): note is_place_cell (field-based) vs classify (info-threshold) divergence (review nit) (d725d9f)
- docs,fix(encoding,events): document unit_ids/unit_table/unit_id; guard ndim; honest unit_id scalar (2a4d1ac)
- docs: sync Phase 0 plan §0.7 to shipped decode_session; archive re-review (d5cb4eb)
- docs(reviews): archive Phase 0 independent review with resolution note (d296b58)
- docs,ci: bump installation.md to v0.5.0; execute nb-20 in notebook CI (0abba24)
- docs: align decode_session example with manual path + execute notebook (review fixes) (b0b1e55)
- docs: teach decode_session as the one-call decode golden path (720547f)
- docs: fix degenerate Workflow 1 example + review nits (Phase 0.6) (4bb7636)
- docs: teach one canonical beginner path; fix broken API snippets (Phase 0.6) (da3ba4c)
- fix(errors): remove internal-doc (CLAUDE.md) references from user-facing errors (63c8062)
- docs(plan): apply independent + maintainer review findings to v0.6 plan (8cf9c92)
- docs(plan): add v0.6 UX implementation plan + review synthesis (9ccf0f7)
- docs: update CHANGELOG.md for v0.5.0 (9552a27)

### Other Changes
- Merge pull request #27 from edeno/chore/release-0.6.0 (ce62eff)
- chore(release): 0.6.0 — bump version, cut CHANGELOG [0.6.0], sync docs; fix stale doc refs (aa3e80f)
- chore(reviews): strip trailing blank line at EOF (git diff --check hygiene) (b393fd3)
- Merge pull request #21 from edeno/phase-3-simplify (19beb30)
- refactor(interop): dedup pynapple adapters, collapse Session spike coercion, avoid double spike/interval normalization (35af797)
- refactor(encoding): store SpikeTrains.trains as an immutable tuple; fix lazy-export comment (229135e)
- test(io): cover to_pynapple TypeError guard for non-result + values=None (3d27e81)
- chore(deps): lock pynapple optional extra (quantities, tabulate) (dd06e7e)
- test(decoding): structural throughput guard for summary decoders (per-block decode count = ceil(n_time/time_chunk), independent of n_neurons) (b0da2c1)
- test(encoding): enforce batch throughput contract — occupancy/kernel computed once, not per-neuron (deterministic call-count) (2aa7c21)
- test(decoding): cover decode_position time_chunk<1 guard; route summary prior-shape check through shared helper (f0d40d7)
- refactor(smoothing): warn (don't hard-gate) on high-bin dense kernels; drop allow_large/_KERNEL_HARD_LIMIT_BINS (a3e74eb)
- refactor(decoding): make decode_session_summary block-alignment check unconditional (survive -O) (08568b9)
- test(encoding): guard population_coverage docstring arg order (55c6bee)
- refactor(environment): sync compute_kernel protocol signature with allow_large (63c6d09)
- refactor: Remove type ignore comments for clarity in various encoding functions (fe31432)
- refactor: Rename methods and update docstrings for clarity in encoding tests (64d30ab)
- feat(encoding,behavior,decoding): enforce naming contract (classify/label_cell_types/is_place_cell/peak_location; decode_position result arg; deprecation aliases) (e3c9c94)
- test(encoding): rename stale to_dataframe/neuron_id test names to summary_table/unit_id (review nit) (7e882c0)
- refactor(decoding): let encoder own the drop warning in decode_session None branch (891a5be)
- refactor(encoding): rename normalize_spike_times -> as_spike_trains (f4cf99a)
- refactor(decoding): polish decode_session per review (times guard, ArrayLike, tests) (9bd454d)
- test: let CLAUDE.md backstop raise on unreadable file (review nit) (5ae370c)
- fix(environment): beginner-grade error for bare Environment() pointing to factories (80d6ae7)
- test(events): strengthen finite-event_times assertion (review nit) (8cad979)
- test(behavior): add list-input normalization tests for traveled_path_length (b81d78b)
- chore(worktrees): ignore .claude/worktrees/ for per-task subagent worktrees (994e796)

**Full Changelog**: https://github.com/edeno/neurospatial/compare/v0.5.0...v0.6.0


All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0, minor releases may still include breaking changes;
these are called out under a dedicated **Breaking changes** heading.

## [Unreleased]

## [0.6.0] - 2026-07-03

### Changed

- The internal `_build_encoding_model` (shared by `decode_session` /
  `decode_session_summary` / `BayesianDecoder.fit`) gained an optional
  `context` parameter that names the caller in its up-front
  timestamp-validation error. `BayesianDecoder.fit` passes
  `context="BayesianDecoder.fit"`, so an `epoch` that selects too-few training
  samples now raises `"At least 2 samples required for BayesianDecoder.fit,
  got N"` instead of the misleading `decode_session` provenance. The default
  preserves every existing message.
- `decode_session` / `decode_session_summary` now accept `positions=None` when
  `encoding_models` is supplied (the passthrough decode never uses a position
  track). Previously this raised `"as_times_positions received a timestamp array
  but no positions"`; now the position track is normalized only when actually
  needed (the encode step, or a `PositionLike` `times`). Every existing caller
  that passes `positions` is byte-for-byte unchanged. This enables the
  fitted-model decode path used by `BayesianDecoder.predict` /
  `predict_summary`.

### Fixed

- The public likelihood functions `log_poisson_likelihood` and
  `poisson_likelihood` now **validate `dt` via the shared `validate_dt`
  helper**, consistent with the decode entry points. A non-numeric `dt`
  (including a numeric string like `"0.1"`), a `bool` (`dt=True`), and a
  non-finite `dt` (`nan`/`inf`) now raise a clear `ValueError` instead of the
  weaker `dt <= 0` guard (which leaked a raw `TypeError` on strings, silently
  accepted `dt=True` as `1`, and let `nan`/`inf` slip through). The error
  message changed from `"dt must be positive"` to `"dt must be a finite number
  > 0"`.
- `bin_spikes_in_time` now **validates `dt` consistently** with
  `decode_session` / `decode_session_summary` via a shared `validate_dt` helper:
  a non-numeric `dt` (including a numeric string like `"0.1"`) and a `bool`
  (`dt=True`) now raise a clear `ValueError` (`"dt must be a finite number >
  0, ..."`). Previously a numeric string leaked a raw `TypeError` from
  `"0.1" <= 0` and `dt=True` was silently accepted and used as a chunk size of
  `1`. `decode_position` routes through the same helper, so an invalid `dt`
  there also raises the clean message instead of a cryptic downstream error.
- An **unparseable `dtype`** (e.g. `dtype="bogus"`) now raises a clear
  `ValueError` naming `dtype` across the decode/encode entry points
  (`decode_position`, `decode_position_summary`, `compute_spatial_rates`,
  `decode_session`, `decode_session_summary`) instead of a raw NumPy
  `TypeError: data type 'bogus' not understood`.
- `decode_session` and `decode_session_summary` now **validate `dt`** (finite,
  `> 0`) up front with a clear `ValueError`, matching `bin_spikes_in_time`'s
  wording (`"dt must be finite and > 0, got ..."`). Previously the shared
  `_build_encoding_model` built the decode time grid directly, bypassing
  `bin_spikes_in_time`'s guard, so an invalid `dt` leaked a cryptic error:
  `dt=0` → `ZeroDivisionError`, `dt=NaN` → "cannot convert float NaN to
  integer", `dt<0` → a misleading "span smaller than one bin" message, and
  `dt=inf` → a similar cryptic failure.
- `decode_position` now **preserves `float32`** when handed a rate-result object
  (anything exposing `.firing_rates`). Previously the friendly object path
  promoted a `float32` `.firing_rates` to `float64`, silently losing part of the
  `dtype=np.float32` memory win that the raw-array path already delivered. The
  object path now matches the raw-array path byte-for-byte: `float32` stays
  `float32`, `float64` stays `float64`, an integer rate map is promoted to
  `float64`, and a `None` / dict / non-2-D `.firing_rates` still raises the same
  clear `ValueError`.
- `time_chunk` is now validated as a **positive integer (not `bool`)** across
  `normalize_to_posterior`, `decode_position`, `decode_position_summary`, and
  `decode_session_summary`, via a shared validator that raises a clear
  `ValueError` naming the value and its type. Previously a float (`1.5`) or
  string (`"2"`) leaked a raw `TypeError` from `range(...)`/comparison, and
  `True` was silently accepted as a chunk size of `1`.
- The summary decoders `decode_position_summary` and `decode_session_summary`
  now **reject `time_chunk=None`** (raising a clear `ValueError`). Previously a
  `None` value set the streaming block to the full session length, materializing
  the full `(n_time, n_bins)` posterior transiently and defeating the
  memory-safe "never materialize the full posterior" contract these functions
  promise. `time_chunk` must be a positive integer (default `1024`); use
  `decode_position` / `decode_session` if you want the full posterior.

### Added

- NWB `SpatialRatesResult` round-trip + lazy reads (`neurospatial.io.nwb`).
  `write_spatial_rates(nwbfile, result, *, name="spatial_rates", overwrite=False)`
  stores a population result on a **unit axis** — a `DynamicTable` (one row per
  unit) in `analysis/` with a `unit_id` column, a 2-D `firing_rate` column of
  shape `(n_units, n_bins)`, and one column per `unit_table` field; occupancy is
  stored once as a companion `TimeSeries`, `smoothing_method` / `bandwidth` /
  bin counts ride in the table description, and bin centers are shared via the
  existing `bin_centers` dataset. The full `Environment` is now **persisted**
  alongside the rates (via `write_environment` under the derived name
  `f"{name}_environment"`), so it round-trips with its **connectivity edges and
  geometry intact**. The write is **atomic**: all name collisions (the table,
  `f"{name}_occupancy"`, `f"{name}_environment"`) and shape validation are
  resolved before the first object is added, so a duplicate without `overwrite`
  raises before any mutation and `overwrite=True` cleans every companion; a
  later add failure rolls back the partial write. `firing_rates` / `occupancy`
  are defensively copied so the NWB containers never alias the live result, and a
  `unit_table` column named `unit_id` / `firing_rate` (reserved) raises a clear
  `ValueError`. `read_place_field(nwbfile, *, name="spatial_rates", env=None)` is
  the inverse, reconstructing a `SpatialRatesResult` with `firing_rates` /
  `occupancy` / `unit_ids` / `unit_table` / `smoothing_method` / `bandwidth` all
  preserved (**`unit_ids` and `unit_table` links survive** non-default ids and
  non-trivial tables). When `env=` is omitted it restores the **persisted
  environment with full connectivity** (`read_place_field(nwb).env.neighbors(i)`
  matches the original; there is no connectivity-less bin-centers fabrication),
  and it raises if neither an `env=` nor a persisted env is present. A mismatched
  or stale `env=` (its `n_bins` disagreeing with the stored rates) now raises a
  clear `ValueError` instead of silently attaching, the companion occupancy
  length is validated at read against the table's recorded `n_bins`, and a `name`
  pointing at a non-spatial-rates table raises a clear `ValueError`. Shape
  mismatches raise param-named `ValueError`s and duplicate names honor
  `overwrite`. `read_position` / `read_pose` / `read_units` gain a keyword-only
  `lazy=False`: `lazy=True` returns h5py-backed handles (per-unit spike-time
  handles for `read_units`) that materialize only when sliced / `np.asarray`-ed —
  valid only while the backing `NWBFile` is open — while the default eager path
  stays byte-for-byte unchanged. The lazy `read_position` / `read_pose` paths now
  **validate lengths** via the handles' `.shape[0]` (no materialization), so a
  positions/timestamps or per-bodypart mismatch raises exactly as the eager path
  (lazy pose no longer silently misaligns bodyparts of differing length), and the
  lazy per-unit spike-time handle **caches** its materialized array (the ragged
  slice is read and sorted at most once). A `test_nwb.yml` CI job installs the
  `nwb` extra and runs the NWB tests (`tests/nwb` + `tests/test_recording.py`),
  which the default `dev`-only job skips. `neurospatial` still never imports
  pynwb (loaded lazily in `io/nwb`).
- `BayesianDecoder` — an immutable (`frozen`) `fit`/`predict`/`predict_summary`/
  `score` wrapper over the decode core (`neurospatial.decoding`, also exported at
  the top level). `fit(spike_times, times, positions, *, speed=None,
  min_speed=None, epoch=None)` builds encoding models by reusing
  `decode_session`'s internal encoder and returns a **new** fitted decoder
  (never mutates the original); an optional `epoch` restricts the training data
  first (via `behavior.restrict` / `restrict_spike_trains`) for train/test
  splits. `predict` / `predict_summary` delegate to `decode_session` /
  `decode_session_summary` with the fitted models, so a fitted decoder's
  posterior is **byte-exact** with `decode_session` on the same inputs; `score`
  reports decode error (`"median_error"` / `"mean_error"`, lower is better) via
  `DecodingResult.error_against`. `score` gained a `distance=` option
  (`"euclidean"` default or `"geodesic"`, forwarded to `error_against`) so
  scoring can use the environment's graph distance, not only straight-line
  error, and now **does not silently drop undecodable bins**: it **warns**
  (naming the excluded fraction) when any decode time bin is undecodable
  (all-non-finite posterior row, which `nanmedian` / `nanmean` would otherwise
  ignore) and **raises** a clear `ValueError` — instead of returning `nan` — when
  *no* bin is decodable (naming the likely degenerate-encoding-model or
  seconds-vs-milliseconds unit-mismatch cause); the all-decodable path stays
  warning-free. Invalid `metric` / `distance` are validated **before** decoding
  so a typo does not cost a full decode. Construction now validates config and
  fitted state: `dt` (via the shared `validate_dt`) and `dtype` are checked at
  build time, and a directly-injected fitted decoder
  (`BayesianDecoder(env, encoding_models=..., unit_ids=...)`) is checked for
  fitted-state coupling (`unit_ids` present, 2-D models, and a bin/unit-count
  match against `env`) instead of detonating later inside the core. A new
  read-only `is_fitted` property lets callers branch without catching
  `RuntimeError`, and a `warn_on_drop` config field (default `True`) threads to
  both the `fit` encode step and the `predict` / `predict_summary` decode steps
  as a single knob to silence the spikes-out-of-window warnings. Accepts
  `SpikeTrainsLike` / `PositionLike` inputs, and decodes through the
  `Environment`, so geodesic / linearized-track / graph-based decoding works
  (unlike pynapple `decode_1d` / `decode_2d`). Unfitted `predict` /
  `predict_summary` / `score` raise a clear `RuntimeError`.
- `Session` — a frozen **discoverability bundle** (`neurospatial.recording`,
  also exported at the top level) grouping `env` / `position` / `spikes` /
  `epochs` / `metadata`. It is **not a god-object**: it exposes the raw arrays
  (`session.times` / `session.positions` / `session.spikes`) but carries **no**
  heavy analysis methods — compute stays functional, e.g.
  `compute_spatial_rates(session.env, session.spikes, session.times,
  session.positions)`. `position` uniformly exposes `.t` / `.values` (arrays are
  wrapped in a small internal `Position` holder; a pynapple `Tsd` / `TsdFrame`
  already conforms). Constructors: `Session.from_arrays(*, env=None, times,
  positions, spike_times, unit_ids=None, unit_table=None, epochs=None,
  metadata=None)` (accepts arrays or a `PositionLike`, and a list / 2-D array /
  `SpikeTrains` / pynapple `TsGroup` for spikes, threading `unit_ids` /
  `unit_table`) and `Session.from_nwb(path_or_file, *, environment_name=None,
  unit_ids=None, **read_kwargs)`. Frozen / immutable: `with_environment(env)` and
  `restrict(epochs)` return a **new** `Session` and never mutate the original.
  `restrict` slices the position and the spikes to the epochs and is
  **identity-preserving** — restriction trims spikes per unit (never drops
  units), so it rebuilds a `SpikeTrains` carrying the original `unit_ids` /
  `unit_table` unchanged, and records the epochs on the new session. A
  `restrict` that keeps **zero** position samples (epochs that miss the session
  entirely — often a seconds-vs-milliseconds unit mismatch) emits a
  `UserWarning` naming the likely cause and returns the (empty) session. A
  mismatched `position.t` / `position.values` length (also self-enforced by the
  internal `Position` holder, which additionally requires a 1-D `t`), a
  `position` that is not `PositionLike` (missing `.t` / `.values`), or a
  non-`EnvironmentLike` `env` raises a clear `ValueError`.
- `load_session(source, **kwargs)` — dispatches an NWB file path (`str` /
  `os.PathLike`) or an open pynwb `NWBFile` to `Session.from_nwb`; any other
  `source` raises a clear `TypeError` directing you to `Session.from_arrays`.
  `Session.from_nwb` builds the bundle via the existing lazy
  `neurospatial.io.nwb` readers (`read_units` / `read_position` /
  `read_environment`), so `neurospatial.recording` **never imports pynwb /
  pynapple** and `import neurospatial` stays cheap. The environment is read by
  **presence** (membership in `nwbfile.scratch`), not by catching an error: a
  genuinely-absent environment maps to `env=None`, while an environment that is
  **present but unreadable** (malformed / wrong schema) now **raises** instead of
  being silently swallowed to `None`. The new `environment_name=` selector reads
  the standard `spatial_environment` scratch entry by default and can select an
  environment written under a custom name. (The lazily-materialized `lazy=True`
  NWB read path is intentionally deferred to a later phase.)
- `restrict` / `in_epochs` / `restrict_spike_trains` — array-native epoch
  selection ("give me my running periods / trial N") in a new
  `neurospatial.behavior.epochs`. `epochs` accepts `(start, end)` scalars, two
  `(starts, ends)` 1-D arrays, an `(n, 2)` array, **or a pynapple `IntervalSet`**
  — the `IntervalSet` is **duck-typed** (`.start` / `.end`), so this stays
  array-first and **never imports pynapple**. The one genuinely ambiguous form
  — a length-2 pair of length-2 sequences (`[[0, 5], [10, 15]]`), which could
  mean two `(start, end)` interval rows **or** two parallel `(starts, ends)`
  arrays — **raises** `ValueError`; pass an `(n, 2)` NumPy array (interval rows)
  or explicit 1-D `start`/`end` arrays to disambiguate. Endpoints are **inclusive** by
  default (`closed="both"`, matching `behavior.segmentation` and pynapple; also
  `"left"` / `"right"` / `"neither"`). `restrict(times, *arrays, epochs=...)`
  slices `times` and any arrays **aligned to `times`** (e.g. `positions`) by the
  same in-epoch mask, order preserved (`t, pos = restrict(times, positions,
  epochs=run_epochs)`); with no extra arrays it restricts an event-time array by
  its own timestamps (`restrict(spike_train, epochs=...)`).
  `restrict_spike_trains(trains, epochs)` restricts **ragged** per-unit spikes,
  each train by its **own** timestamps, and accepts a plain sequence or a
  `SpikeTrains`. `restrict` is also exported at the top level (`neurospatial.restrict`).
- `SpikeTrains` — a frozen ragged-spike-train container (label access
  `st[unit_id]`, `filter("region == 'CA1'")`, optional per-unit `unit_table`),
  the one justified new container (ragged per-unit spike times genuinely don't
  fit a rectangular array). Exported from both `neurospatial` (top level) and
  `neurospatial.encoding`. It **duck-types as `SpikeTrainsLike`** — it is not a
  `Mapping`, exposes a non-callable `.index` property (the unit ids), and its
  `__iter__` yields the per-unit **train arrays** — so it flows through the
  Phase 3.1 spike-input adapter's iterate branch and into `compute_spatial_rates`
  (and the other batch encoders/decoders) with `unit_ids` preserved. Label
  access (`st[unit_id]`) and the adapter's positional iteration coexist because
  they use different dunders (`__getitem__` vs `__iter__` / `.index`). Frozen
  and immutable: `trains` is stored as a `tuple` so in-place mutation raises,
  `filter` returns a new container and never mutates the original; duplicate
  `unit_ids` raise `ValueError`.
- Input Protocol surface + pynapple ingress/egress (optional). A new
  `neurospatial/_typing.py` defines the structural Protocols that let
  third-party objects flow into the **array-first** scientific core without the
  core ever importing or `isinstance`-checking them:
  - `PositionLike` — a `.t` / `.values` time-series (pynapple `Tsd` / `TsdFrame`
    conform). `compute_spatial_rate` / `compute_spatial_rates` and
    `decode_session` / `decode_session_summary` now accept a `PositionLike` in
    the `times` slot (with `positions` omitted) and normalize it to plain
    `float64` arrays at the boundary via `as_times_positions`. The plain-array
    path is unchanged byte-for-byte.
  - `SpikeTrainsLike` — the accepted spike-input union, plus a real pynapple
    `TsGroup`. A `TsGroup` is a `collections.UserDict`, so **iterating it yields
    the unit-id keys, not the per-unit trains**; `SpikeTrainsLike` is therefore
    the *indexable-by-id* surface (`.index` of unit ids + `group[uid]` returning
    a per-unit `Ts` with `.t`). A new `encoding.as_spike_trains_with_ids`
    extracts trains by indexing each id (never by iterating), so a raw `TsGroup`
    flows correctly into `compute_spatial_rates` / `decode_session`; it surfaces
    the ids without changing `as_spike_trains`'s `list[NDArray]` contract, and
    when a group carries ids and the caller passes no `unit_ids=`, they now flow
    into `SpatialRatesResult.unit_ids` instead of being silently dropped.
  - `EnvironmentLike` — a public re-export of `EnvironmentProtocol`. The
    internal `isinstance(env, Environment)` check in `CompositeEnvironment` is
    replaced by a duck-typed `is_environment_like` check, fixing the surprise
    that the sibling `EgocentricPolarEnvironment` (not an `Environment`
    subclass) was rejected even though it is a legitimate environment.
- pynapple I/O shim behind a new optional `pynapple` extra:
  `neurospatial.io.from_pynapple` (`TsGroup` → `(trains, unit_ids)`;
  `Tsd` / `TsdFrame` → `(times, positions)`; `IntervalSet` → `(start, end)`) and
  `neurospatial.io.to_pynapple` (a decoded MAP track / `times`+`values` →
  `Tsd` / `TsdFrame`). `import pynapple` is lazy inside these functions, so the
  package and the array path import and run with pynapple absent; calling them
  without it raises a clear `ImportError` naming `neurospatial[pynapple]`. The
  scientific modules never import pynapple.
- Speed filtering on the encode path. `compute_spatial_rate` and
  `compute_spatial_rates` gain keyword-only `speed` / `min_speed` parameters
  (also forwarded by `decode_session` / `decode_session_summary`). When
  `min_speed` is set, low-speed periods are excluded using **one shared
  per-interval speed gate** applied to **both** the spike numerator and the
  occupancy denominator, so a `min_speed` knob can never silently bias firing
  rates by filtering only one side. The spike gate matches the occupancy gate
  exactly: occupancy keeps interval `k` iff `speed[k] >= min_speed`, and a
  spike at time `t` is kept iff
  `speed[searchsorted(times, t, "right") - 1] >= min_speed` — the same
  per-interval criterion, so identical intervals drop on both sides. When
  `speed` is omitted it is auto-derived with a **forward difference**
  (`speed[k] = ||positions[k+1] - positions[k]|| / (times[k+1] - times[k])`,
  with `speed[n-1] = speed[n-2]`) to match `env.occupancy`'s
  `time_allocation="start"` interval semantics; pass an explicit `speed` for
  geodesic / linearized-track environments. `min_speed=None` (the default)
  applies no filtering and leaves output byte-for-byte unchanged.
  - Passing `speed` **without** `min_speed` now raises `ValueError` (instead of
    silently ignoring `speed`), mirroring `env.occupancy`, which raises on
    `min_speed` without `speed`.
  - When the interval filter excludes **all** trajectory intervals — whether
    via `min_speed`, `max_gap`, or the out-of-bounds-start rule (e.g. a
    wrong-units threshold) — `compute_spatial_rate` / `compute_spatial_rates`
    now emit one `UserWarning` naming the active gate(s) and suggesting fixes
    (check units; pass `max_gap=None`), instead of silently returning an empty
    rate map. The warning fires once per call (not per neuron in the batch
    path) and is suppressed by `warn_on_drop=False`.
- `decode_session` and `decode_session_summary` gain a keyword-only `dtype`
  parameter (`np.float32` / `np.float64`, default `np.float64`). "Decode in
  this dtype": a single `decode_session(dtype=np.float32)` controls **both**
  the encoding-model working set and the posterior dtype, end-to-end — the
  functions no longer force-promote encoding models back to `float64`. On
  `decode_session_summary` it is now an explicit parameter rather than a
  `decode_kwargs` entry. Default `np.float64` leaves every existing caller
  byte-for-byte unchanged; any other dtype raises `ValueError`.

### Performance

- `decode_session(dtype=np.float32)` / `decode_session_summary(dtype=np.float32)`
  now actually halve the decode working set on the beginner golden path: the
  `float32` rate-map dtype is honored end-to-end (encoding-model working set +
  posterior) instead of being silently promoted back to `float64` inside the
  session helpers. Values match `float64` within `float32` tolerance (the rate
  computation is done in `float64` and only the result is cast, per
  `compute_spatial_rates`).

### Breaking changes

- `to_xarray()` now returns a labeled `xarray.Dataset` instead of an
  `xarray.DataArray` with integer coordinates. This is a clean break: there is
  no `DataArray` shim and no `to_dataset()` alias. Two distinct shapes are
  produced:
  - **Population rate results** (`SpatialRatesResult`, `DirectionalRatesResult`,
    `ViewRatesResult`, `EgocentricRatesResult`) return a `Dataset` with dims
    `("unit_id", "bin")`. `unit_id` is the index coordinate holding the *real*
    per-unit identity labels (`result.unit_ids`), so units are selected by
    label. The `bin` dimension carries non-index `bin_center_x` / `bin_center_y`
    (/ `bin_center_z`) coordinates for Cartesian environments, or
    `bin_center_distance` / `bin_center_angle` for the polar egocentric result.
    The rate matrix is the `firing_rate` data var; `occupancy` is a `("bin",)`
    data var. `attrs` carry `units`, `bandwidth` (where applicable), an `env`
    fingerprint, and `software_version`. Duplicate `unit_ids` now raise
    `ValueError` (label-based selection requires uniqueness).
  - **Decode results** (`DecodingResult`) return a `Dataset` with dims
    `("time", "bin")` (a posterior over space per time bin; no `unit_id` axis).
    The `posterior` data var holds the posterior, with the same `bin_center_*`
    coordinate logic and `units` / `env` / `software_version` attrs.

    Before → after:

    ```python
    # before (DataArray, integer coords)
    da = result.to_xarray()
    da.sel(neuron=0)
    # after (Dataset, real unit_id labels)
    ds = result.to_xarray()
    ds.sel(unit_id=result.unit_ids[0])
    ```

- The two terminal verbs now mean **one** thing on every result class.
  `to_dataframe()` on the batch (plural) encoding results — `SpatialRatesResult`,
  `DirectionalRatesResult`, `ViewRatesResult`, `EgocentricRatesResult` — is now
  **dense tidy**: one row per `(unit, bin)` (single-unit results: one row per
  `bin`), always carrying a `unit_id` column plus the bin-center coordinate
  columns (`bin_center_x`/`y`/`z` for Cartesian, `bin_center_distance`/`angle`
  for polar egocentric, `bin_center_angle` for directional), `firing_rate`, and
  `occupancy`. The **per-unit summary** that `to_dataframe()` used to return
  (one row per neuron with `peak_x`, `peak_rate`, `spatial_info`, `sparsity`,
  `grid_score`, `border_score`, `cell_type`, etc.) has moved to the new
  `summary_table()`, which is `unit_id`-indexed. This is a clean break: there is
  no mode flag and no transition shim. The `neuron_ids=` keyword on the old
  per-unit `to_dataframe()` is replaced by `unit_ids=` on `summary_table()`
  (defaulting to the result's own `unit_ids`).

    Before → after:

    ```python
    # before — to_dataframe() returned one row per neuron with metric columns
    df = result.to_dataframe()           # columns: neuron_id, peak_x, ...
    place = df[df["cell_type"] == "place"]

    # after — summary_table() is the per-unit summary; to_dataframe() is dense
    summary = result.summary_table()     # one row per unit, unit_id-indexed
    place = summary[summary["cell_type"] == "place"]
    dense = result.to_dataframe()         # one row per (unit, bin), carries unit_id
    ```

### Added

- Memory-safe summary decoding for long sessions. `decode_position_summary`
  is a new sibling of `decode_position` that streams over time, computing the
  posterior one time-block at a time and reducing each block to per-time
  scalars/vectors (`map_position` / `map_bin`, `mean_position`,
  `posterior_entropy`, `peak_prob`) without ever materializing the full
  `(n_time, n_bins)` posterior. It returns a new `DecodingSummary` frozen
  dataclass (alongside `DecodingResult`) carrying `ResultMixin` and the
  standard terminal verbs — `to_dataframe()` (one row per time bin),
  `summary()`, `plot()`, and `to_xarray()` (a track `Dataset` with a `time`
  dim and **no** `bin` posterior axis) — sharing accessor names and column
  conventions with `DecodingResult` so user code ports between the two.
  `decode_session_summary` is the matching one-call encode→bin→decode wrapper
  (sibling of `decode_session`). The summary reductions are bit-for-bit
  identical to reducing the full posterior for `map_position` / `map_bin`, and
  match to floating-point tolerance for `mean_position` / `posterior_entropy` /
  `peak_prob`.

- Experiment-shaped factory presets on `Environment` that speak experiment
  vocabulary and delegate to the existing `from_*` factories:
  - `Environment.open_field(positions, bin_size, ...)` — the only
    positions-based preset; delegates to `from_samples` with `fill_holes=True`
    flipped on (a sensible open-arena default that fills interior gaps).
  - `Environment.linear_track(*, endpoints=..., node_positions=..., bin_size)`
    — builds a 1D track graph (`is_linearized_track == True`) from an explicit
    topology (two endpoints for a straight track, or waypoints for a
    piecewise-linear track) and delegates to `from_graph`.
  - `Environment.maze(kind, *, track_graph=..., node_positions=..., bin_size)`
    — assembles the standard W / plus / T track-graph topology (or accepts a
    ready `networkx` graph) and delegates to `from_graph`.

  Track/maze presets require an explicit topology spec; raw positions cannot
  infer a linear/W/plus/T graph, and calling them without a topology raises a
  clear `ValueError`.

- `to_xarray()` on `DirectionalRatesResult`, `ViewRatesResult`, and
  `EgocentricRatesResult` (the directional/view/egocentric population results
  previously had no xarray export). Each returns the labeled `xr.Dataset`
  described under Breaking changes above.

- Durable unit identity on encoding/events results. Every population result
  (`SpatialRatesResult`, `DirectionalRatesResult`, `ViewRatesResult`,
  `EgocentricRatesResult`, `PopulationPeriEventResult`) now carries a
  `unit_ids: np.ndarray` field (plus an optional `unit_table: pd.DataFrame |
  None`), and every single-unit result (`SpatialRateResult`,
  `DirectionalRateResult`, `ViewRateResult`, `EgocentricRateResult`,
  `PeriEventResult`) carries a singular `unit_id`. Indexing or iterating a
  population result stamps the per-unit label onto the child
  (`rates[i].unit_id == rates.unit_ids[i]`, iteration preserves order and
  labels). The batch compute functions (`compute_spatial_rates`,
  `compute_directional_rates`, `compute_view_rates`,
  `compute_egocentric_rates`, `population_peri_event_histogram`) gained a
  keyword-only `unit_ids=` parameter that threads onto the result; a
  wrong-length value raises a clear `ValueError`. Fully back-compatible:
  `unit_ids` defaults to `np.arange(n_units)` and the new fields are
  `compare=False`, so existing callers and equality/hash behavior are
  unchanged.

- `summary_table()` — the per-unit summary terminal verb — on every batch
  encoding result (`SpatialRatesResult`, `DirectionalRatesResult`,
  `ViewRatesResult`, `EgocentricRatesResult`) and on `PopulationPeriEventResult`.
  Returns one row per unit, `unit_id`-indexed, with that result's scalar metric
  columns (peak location/rate, spatial info, grid/border score, cell type,
  preferred direction/distance, etc.). Accepts an optional `unit_ids=` to
  relabel the index.

- PSTH results now carry the uniform result surface. `PeriEventResult` and
  `PopulationPeriEventResult` inherit the canonical `ResultMixin` and implement
  the terminal verbs: `to_dataframe()` (dense — one row per time bin for the
  single-unit result, one row per `(unit, time-bin)` for the population result,
  always carrying `unit_id`), `summary()` (flat dict of headline scalars —
  peak rate/latency, baseline rate; population adds `mean_peak_rate` /
  `population_peak_latency`), `PopulationPeriEventResult.summary_table()` (one
  row per unit with `peak_rate` / `peak_latency` / `baseline_rate`), and
  `plot()` (delegates to `plot_peri_event_histogram`, returns the axis).

- `decode_session(env, spike_times, times, positions, *, dt, ...)` — one-call
  encode→bin→decode golden path in `neurospatial.decoding.session`.  Glues
  `compute_spatial_rates`, `bin_spikes_in_time`, and `decode_position` into a
  single function so beginners can decode position in ≤10 lines.  Exported from
  `neurospatial.decoding` (`from neurospatial.decoding import decode_session`).
  Accepts an optional `encoding_models=` array to bypass the encoding step
  entirely.
  Extra keyword arguments are forwarded verbatim to `decode_position`.

- `as_spike_trains` — public helper that coerces the various spike-input
  formats (1D array, NaN-padded 2D array, list of scalars, list of 1D arrays)
  into the canonical list of per-neuron spike-time arrays. Exported from
  `neurospatial.encoding` (and present in `__all__`); the implementation lives
  in `neurospatial.encoding._spikes`. It standardizes the container shape only
  — spike-time values are never shifted, rescaled, or aligned. (This is the
  previously-internal `normalize_spike_times`, renamed before any public
  release so the name reads as a structural conversion, not a value transform;
  no deprecated alias is kept since the public name was never released.)

- `classify(*, ...)` — a single-type boolean cell-type predicate (returns
  `NDArray[np.bool_]`) on every batch encoding result: `EgocentricRatesResult`
  (OVC, `min_info=0.3`), `ViewRatesResult` (view cell, `min_info=0.5`),
  `DirectionalRatesResult` (HD cell, `min_mvl`/`alpha`), and `SpatialRatesResult`
  (place cell, `min_spatial_info=0.5`). These replace the per-domain
  `detect_ovcs` / `detect_view_cells` / `detect_hd_cells` detectors (now
  deprecated aliases).

- `label_cell_types(...)` on `SpatialRatesResult` — the multi-class string
  labeler (`"place"`/`"grid"`/`"border"`/`"unclassified"`, returns
  `NDArray[np.str_]`). This is the renamed `detect_cell_types` and is kept
  deliberately SEPARATE from the boolean `classify` (different return type).

- `is_place_cell(...)` — a single-neuron place-cell predicate, both as a free
  function in `neurospatial.encoding.spatial` (exported from
  `neurospatial.encoding`) and as a `SpatialRateResult.is_place_cell()` method.
  Mirrors `is_spatial_view_cell` / `is_object_vector_cell` and agrees with
  `detect_place_fields` (returns `True` iff that detector finds ≥1 field).

- `decode_position(env, spike_counts, encoding_models, dt, ...)` now accepts a
  population rate result object (anything exposing a `firing_rates` attribute,
  e.g. `SpatialRatesResult`) directly in place of the raw `(n_neurons, n_bins)`
  array, removing the `np.stack([r.firing_rate ...])` glue between the encoding
  and decoding steps.

- New keyword-only parameter `warn_on_drop: bool = True` on
  `bin_spike_train`, `bin_spike_trains` (`encoding/_binning.py`),
  `compute_spatial_rate`, and `compute_spatial_rates` (`encoding/spatial.py`).
  Set to `False` to intentionally silence all spike-drop warnings (e.g. when
  the caller handles the diagnostic themselves).

### Performance

- `decode_session_summary` now **streams the time-binning** so the full
  `(n_time, n_neurons)` spike-count matrix is never materialized. It builds the
  encoding model once over the whole session (small, `(n_neurons, n_bins)`),
  then bins spikes block-by-block against a contiguous slice of the global time
  grid and decodes + reduces each block via the same shared inner-loop helper as
  `decode_position_summary`. Peak memory is now
  `O(time_chunk × max(n_neurons, n_bins))` plus the `(n_neurons, n_bins)`
  encoding model — **independent of session length** — meeting the
  1 hr / 25 ms / 5000-bin / <500 MB summary-decode DoD golden path (the dense
  `(144000, 1000)` count matrix alone would be ~1.15 GB). The result is
  byte-for-byte identical to the prior materialize-then-stream path; per-block
  counts are binned against the precomputed global edges so they match a single
  global histogram exactly, and boundary spikes are counted exactly once.
  `decode_session` (the full-posterior path) is unchanged.

- `decode_position` gains two keyword-only memory knobs with **no change to its
  return contract** (`.posterior` stays a fully-materialized `ndarray`):
  - `dtype=np.float32` stores and computes the posterior in single precision,
    halving stored and transient memory (parity with float64 to ~1e-6
    relative). Every `DecodingResult` method works unchanged on a float32
    posterior.
  - `time_chunk` is now **hybrid**. `time_chunk=None` (the default) keeps the
    full-matmul path **byte-for-byte unchanged** — the Poisson log-likelihood is
    computed once over the whole window and normalized at once (transient peak
    ~3× the stored posterior). An **explicit `time_chunk=k`** now computes the
    Poisson log-likelihood **blockwise directly into the preallocated
    posterior**, so the full-size log-likelihood and its working copy are never
    materialized — cutting the transient peak to **~1×** over the returned
    posterior (the posterior itself is unavoidably 1×, since `decode_position`
    returns the full dense array). The opt-in path is **tolerance-equal**, not
    byte-exact, to the full path: the per-block likelihood matmul is a different
    BLAS shape than the full matmul, so it differs by ~1e-15 (MAP/argmax
    identical; every row sums to 1). For a path that never holds even the full
    posterior, use `decode_position_summary`.

  For sessions where even the stored dense posterior is too large to hold, use
  the new `decode_position_summary` / `decode_session_summary` (see **Added**),
  which never materialize the full `(n_time, n_bins)` posterior.

- `compute_spatial_rates` gains a keyword-only `dtype` (`np.float32` /
  `np.float64`, default `np.float64`). `dtype=np.float32` halves the stored
  `(n_units, n_bins)` rate-map array. The rate computation (GEMM / division) is
  still performed in float64 and only the final result is cast, so float32
  values match the float64 default within float32 tolerance. `decode_session` /
  `decode_session_summary` now accept their own `dtype` parameter (default
  float64) that honors float32 end-to-end — the encoding-model working set and
  the posterior — so `decode_session(dtype=np.float32)` halves the decode
  working set on the golden path (see the `decode_session` / `decode_session_summary`
  `dtype` entry above). Default `np.float64` leaves every existing caller
  byte-for-byte unchanged; any other dtype raises `ValueError`.

- Documented the dense diffusion-kernel **O(n²) memory cost** and added a loud
  high-bin memory **warning**. The heat kernel `exp(-tL)` of a connected graph
  is dense by construction (every entry > 0), so it always costs
  `n_bins**2 * 8` bytes of float64 memory (≈ 3.2 GB at 20,000 bins).
  `compute_diffusion_kernels` and `env.compute_kernel` now emit a loud
  `UserWarning` (with a GB estimate) above 3,000 bins and then **proceed** —
  there is **no hard limit** and no `allow_large` opt-out. The warning names the
  size, the GB estimate, the dense O(n²) reason, and the fixes (reduce bins or
  use `smoothing_method="binned"`). No numerical results change — this is
  documentation plus a warn-and-proceed guard. The reliable scale wins this
  release remain float32 rate maps and the memory-safe summary decode (above). A
  faster, lower-peak `expm_multiply` / Chebyshev rewrite of the kernel is a
  **deferred stretch goal** and is intentionally **not** part of this release.

- Warned on the dense **Gaussian-KDE** high-bin path the same way as the
  diffusion kernel. `smoothing_method="gaussian_kde"` builds a dense
  `(n_bins, n_bins)` weight matrix (`exp(-d²/2σ²)`, every entry > 0), so it
  carries the same O(n_bins²) memory cost; it now emits a loud `UserWarning`
  (with a GB estimate) above the shared `_LARGE_KERNEL_THRESHOLD` (3,000 bins)
  and proceeds — no hard limit, no `allow_large`. One warn threshold is shared
  by both dense smoothing paths instead of a divergent copy. Also corrected
  stale `encoding/_smoothing.py` docs that wrongly described the diffusion
  kernel as "sparse" / `O(n_bins)` — it is dense `(n_bins, n_bins)`,
  O(n_bins²) per neuron, with a one-time O(n_bins³) matrix-exponential build.
  No numerical results change.

- `SpatialRatesResult.summary_table()` no longer double-computes grid and
  border scores. It previously ran the expensive per-neuron grid/border score
  computation **twice** per call (once for the `grid_score`/`border_score`
  columns and again inside `label_cell_types()`); it now computes each once and
  forwards them. `label_cell_types()` gains optional keyword-only
  `grid_scores=` / `border_scores=` parameters to accept precomputed score
  arrays (validated to be 1-D and length `n_neurons`); when omitted it
  recomputes as before, so existing callers are unchanged.

- `population_coverage()` gains a keyword-only `n_jobs` parameter (default `1`)
  to parallelize per-neuron place-field detection via joblib (`-1` uses all
  CPUs). `n_jobs=1` keeps the sequential path with no joblib overhead; results
  are byte-for-byte identical regardless of `n_jobs` (returned data identical;
  per-neuron exclusion warnings are not surfaced under `n_jobs != 1`).

### Changed

- **Default rate-map output changed for gappy / out-of-bounds data.** With the
  new `max_gap` default of `0.5 s`, `compute_spatial_rate` /
  `compute_spatial_rates` (and `decode_session` / `decode_session_summary`,
  which forward `max_gap`) now exclude spikes inside large tracking gaps and
  out-of-bounds excursions from the numerator so it matches the occupancy
  denominator. See the **Fixed** entry "Firing-rate numerator/denominator
  alignment on the FULL interval mask" for the full rationale and the
  `max_gap=None` opt-out.

- `detect_region_crossings` argument order changed to follow the
  behavioral-segmentation convention:
  `(position_bins, times, env, *, region_name, direction=...)`. The old
  positional order `(position_bins, times, region_name, env, ...)` is still
  accepted for one release (transitional dispatch emits a `DeprecationWarning`)
  and will be removed in 0.7.

- Docs/examples now teach `decode_session` as the one-call decode golden path;
  the manual 3-call path (`compute_spatial_rates` → `bin_spikes_in_time` →
  `decode_position`) is kept as an "Advanced: manual three-call path" section.
  `examples/20_bayesian_decoding` now leads with `decode_session`; a new
  "Workflow 2: Bayesian Decoding (one call)" section was added to
  `docs/user-guide/workflows.md` (with a `workflows_decode_session_golden_path`
  CI snippet in `docs/snippets.yml`); and `README.md` points to
  `from neurospatial.decoding import decode_session` as the position-decoding
  entry point.

- Docs (Phase 0.6 sweep): rewrote Workflow 1 in `docs/user-guide/workflows.md` to use the
  canonical `compute_spatial_rate(env, spike_times, times, positions, ...).firing_rate` idiom
  with `simulate_trajectory_ou` + `PlaceCellModel` fixtures, replacing the hand-rolled
  `np.histogram` / `scipy.ndimage.gaussian_filter` approach. Added a CI snippet entry
  (`workflows_place_field_canonical`) to `docs/snippets.yml`.

### Deprecated

All of the following emit a `DeprecationWarning` and are scheduled for removal
in 0.7. Each old name forwards to its replacement with unchanged behavior.

- `EgocentricRatesResult.detect_ovcs` → `EgocentricRatesResult.classify`.
- `ViewRatesResult.detect_view_cells` → `ViewRatesResult.classify`.
- `DirectionalRatesResult.detect_hd_cells` → `DirectionalRatesResult.classify`.
- `SpatialRatesResult.detect_cell_types` → `SpatialRatesResult.label_cell_types`
  (the multi-class string labeler; not folded into `classify`).
- `ViewRateResult.peak_view_location` → `ViewRateResult.peak_location`.
- `ViewRatesResult.peak_view_location` → `ViewRatesResult.peak_locations`.
- `detect_region_crossings` old positional order
  `(position_bins, times, region_name, env, ...)` → new order
  `(position_bins, times, env, *, region_name, ...)`.

### Fixed

- `decode_position_summary` now validates `prior` shape (was silently
  truncating over-long 2-D priors), matching `decode_position`. A 1-D prior
  must be `(n_bins,)` and a 2-D time-varying prior must be exactly
  `(n_time, n_bins)`; an over-long or short 2-D prior, a wrong-length 1-D
  prior, or a non-1-D/2-D prior now raises `ValueError` before streaming
  instead of silently slicing the prior to the decoded time range.

- **Firing-rate numerator/denominator alignment on the FULL interval mask.**
  A firing-rate map is `spike_counts (numerator) / occupancy (denominator)`
  per bin. `env.occupancy` drops a trajectory interval `k` for **three**
  reasons — `dt[k] > max_gap` (large tracking gap), `speed[k] < min_speed`
  (low speed), and `start_bin[k] < 0` (interval's start sample out of the
  active environment) — but the spike binner previously only filtered by the
  time window and (since the speed-filter work) by speed. A spike inside a
  dropped interval (e.g. a 1 s tracking gap, or an out-of-bounds excursion)
  was therefore **counted in the numerator** while occupancy **excluded that
  interval's time from the denominator**, inflating/biasing the rate. The
  spike numerator and the occupancy denominator now drop the **identical** set
  of intervals via one shared `interval_valid_mask` helper (the single source
  of truth that `env.occupancy` also consumes). `compute_spatial_rate` /
  `compute_spatial_rates` gain a keyword-only `max_gap: float | None = 0.5`
  (matching `env.occupancy`'s default, so occupancy behavior is unchanged)
  that gates **both** sides identically.
  - **Behavior change:** rate maps now differ for sessions that contain large
    tracking gaps (intervals longer than `max_gap`) or out-of-bounds samples —
    previously those rates were inflated. This is a correctness fix. Pass
    `max_gap=None` to disable gap gating on **both** sides (restoring the
    pre-fix, no-gap-gating behavior while keeping numerator and denominator
    aligned).
  - **No more silent empty rate maps.** When the interval filter excludes
    **every** trajectory interval (so occupancy is all-zero and the rate map is
    all-NaN/0), `compute_spatial_rate` / `compute_spatial_rates` now emit a
    single `UserWarning` naming the active gate(s) (`max_gap`, `min_speed`,
    and/or the out-of-bounds-start possibility) and suggesting fixes (check
    units; pass `max_gap=None`). Previously only the `min_speed` gate warned —
    a too-large `max_gap` (or a wrong-units one) and the out-of-bounds-start
    rule emptied the map **silently**. Gated by `warn_on_drop=True` (the
    default); fires once per call (not per neuron in the batch path).

- `decode_session` now warns loudly when most spikes fall outside the decode
  time window `[times.min(), times.max()]` instead of silently dropping them.
  Previously, when `encoding_models=` was passed (which skips
  `compute_spatial_rates`), spikes outside the window were silently discarded
  by `bin_spikes_in_time`'s `np.histogram`, so a milliseconds-vs-seconds unit
  mismatch produced an all-zero count matrix and a plausible-but-wrong
  posterior with no warning.  `decode_session` now owns a single
  `UserWarning` (naming the dropped count/total, percentage, decode window,
  spike range, and units hypothesis) that covers **both** the
  `encoding_models`-provided and `encoding_models=None` branches; the
  encoder's now-redundant duplicate is suppressed.  A new keyword-only
  `warn_on_drop: bool = True` parameter silences it for genuinely sparse
  sessions.

- The bare-`Environment()` error now uses a unique code `[E1006]` (was
  `[E1001]`, which collided with "No active bins found") and points at the
  correct docs host (`https://edeno.github.io/neurospatial/`).  Documented in
  `docs/errors.md`.

- `Environment.distance_to(region_name)` now raises `RegionNotFoundError`
  (from `neurospatial._exceptions`) instead of a bare `KeyError` when the
  named region is absent.  `RegionNotFoundError` subclasses `KeyError`, so all
  existing `except KeyError` blocks keep working without change.

- Removed stale `compute_firing_rate(...)` calls (that function does not exist publicly) from
  `docs/user-guide/workflows.md` (Workflow 3) and `docs/user-guide/spatial-analysis.md`;
  replaced with the correct `compute_spatial_rate(env, spike_times, times, positions, ...).firing_rate`
  API and argument order.

- Replaced 11 broken `env.plot(field, ax=...)` calls in `docs/user-guide/spike-field-primitives.md`
  and `docs/user-guide/rl-primitives.md` with `env.plot_field(field, ax=...)`. The `env.plot()`
  method's first positional argument is `ax`, not a field array, so passing a field there was silently
  broken.

- Updated `examples/20_bayesian_decoding.py` (and the jupytext-paired `.ipynb`) to use the batch
  `compute_spatial_rates(env, spike_times_list, times, positions, ...).firing_rates` for encoding
  models and the canonical `bin_spikes_in_time(spike_trains, dt, t_start, t_stop)` helper for
  time-binned spike counts; regenerated the notebook via `jupytext --sync`.

- Bumped stale version strings from `v0.4.0` to `v0.5.0` in `docs/index.md` (status line and
  BibTeX entry) and `README.md` (dependency table header and BibTeX entry).

- `GraphValidationError` messages in `layout/validation.py` and the wrapping
  `ValueError` in `environment/core.py` no longer reference the internal
  developer guide `CLAUDE.md`.  Error text now points to the public issue
  tracker (`https://github.com/edeno/neurospatial/issues`) instead.  A
  permanent repo-wide backstop test (`tests/test_no_internal_doc_refs.py`)
  asserts that no source file under `src/neurospatial/` contains this string,
  preventing the leak from regressing.

- `Environment.__init__` now raises a beginner-grade `ValueError` when called
  without arguments (i.e. bare `Environment()`).  The old message "layout
  parameter is required" gave no actionable guidance; the new message explains
  that `Environment` must be created through a factory method, shows a concrete
  correct example (`Environment.from_samples(data, bin_size=2.0)`), lists the
  other available factories (`from_polygon`, `from_graph`, `from_grid_mask`,
  `from_pixel_mask`), and links to the online docs — matching the style already
  used by `EnvironmentNotFittedError`.  The exception type remains `ValueError`
  so existing `except ValueError` callers are unaffected.

- `align_spikes_to_events` in `events/alignment.py` now rejects `event_times`
  containing `Inf` values with a descriptive `ValueError`, matching the
  existing `spike_times` Inf check and fulfilling the docstring promise.
  Previously an `Inf` event time silently produced an empty-or-wrong result.

- `behavior.trajectory` public functions (`compute_turn_angles`,
  `compute_step_lengths`, `mean_square_displacement`) and
  `behavior.navigation.traveled_path_length` now coerce `positions` (and
  `times` for MSD) with `np.asarray` at the public boundary before any
  `.ndim`/`.shape` access. Passing a plain Python list no longer raises a
  confusing `AttributeError`; valid list-of-lists inputs succeed, and
  malformed inputs raise a descriptive `TypeError` or `ValueError`.

- `bin_spike_train` and `bin_spike_trains` in `encoding/_binning.py` no
  longer silently drop spikes outside the trajectory time window or spikes
  that map to inactive environment bins.  When the dropped fraction exceeds
  50 % (or all spikes are dropped), a `UserWarning` is emitted naming the
  dropped count, total, both time ranges, and a units-hypothesis hint (e.g.
  spike_times in milliseconds vs. times in seconds).  Two separate messages
  cover the two drop causes (time-window and inactive-bin).  The batch path
  (`bin_spike_trains`, `compute_spatial_rates`) warns exactly **once per
  cause** in the main process — never from joblib worker processes where
  warnings are commonly swallowed.  Default behaviour for in-window spikes
  is byte-for-byte unchanged.

### Documentation

- `population_coverage` docstring example now shows the vectorized batch path
  (`compute_spatial_rates(...).firing_rates`) instead of the per-neuron
  `compute_spatial_rate` loop; the pre-existing arg-order bug
  (`population_coverage(firing_rates, env)`) has been corrected to
  `population_coverage(env, firing_rates)`.
- `population_coverage` shape-mismatch `ValueError` message now steers users to
  `compute_spatial_rates` (the batch function) rather than the slow
  `compute_spatial_rate` + `np.stack` recipe.
- `compute_spatial_rate` (singular) docstring now includes a short note pointing
  many-neuron users to `compute_spatial_rates` (plural), which shares occupancy
  and smoothing-kernel work across the whole population.

## [v0.5.0] - 2026-06-04

## What's Changed

### Features

### Bug Fixes
- fix(io,regions): atomic writes, NWB metadata round-trip, deep immutability (4307973)
- fix(animation): grid artist reuse, frame_times, figure/capture leaks (8f862d3)
- fix(decoding,stats,events): PSTH window, weighted Rayleigh, posterior NaN (301d2b0)
- fix(simulation): RNG independence, exact Poisson, mixed-mode kwargs (f00c048)
- fix(encoding): bearing wrap, occupancy masking, phase-precession optimizer (1525ffd)
- fix(environment,behavior,annotation): invariants, segmentation, leaks (d5c9f34)
- fix(ops,layout): correctness + safety fixes in core primitives (cefe030)
- docs(plan): mark test-suite remediation complete (9f18cc6)
- fix(encoding): recover egocentric distance tuning in binned rate (ef5090a)
- fix(simulation): place uniform field centers in the interior, not on the boundary (9872487)

### Documentation
- docs(plan): mark test-suite remediation complete (9f18cc6)
- docs: address review follow-up on the fix-all changeset (a2ca036)
- test: reword two docstrings to drop implementation-plan references (28105fe)
- docs: refresh UX-facing onboarding docs (4080f49)
- docs: update CHANGELOG.md for v0.4.0 (c03df84)

### Other Changes
- chore(release): 0.5.0 (#4) (d1afe9d)
- chore: remove completed test-suite remediation plan (8a7a04b)
- test(animation): drive visual regression through production renderer (8e5ed03)
- test: dedup shared test helpers into shared modules (review follow-up) (215c971)
- test: reword two docstrings to drop implementation-plan references (28105fe)
- test: tighten thresholds, strengthen sim recovery, hygiene (Phase 7) (fad0dcb)
- test: replace mocks with real-path tests where they hid integration (Phase 8) (ad3d105)
- test(ops,layout,environment): core-primitives behavioral coverage (Phase 5) (e23365d)
- test(nwb): add NWBHDF5IO disk round-trip tests for writers (Phase 6) (628acb2)
- test(animation): verify real cross-backend color-mapping parity (Phase 4) (b72dc61)
- test(decoding,stats,events): add closed-form + analytic-reference correctness tests (Phase 3) (d4f8da4)
- test(encoding): add end-to-end recovery + NumPy/JAX parity tests (Phase 2) (a8b216e)
- test(encoding): add object_vector_score tests; fix OVC convention doc + CLAUDE.md drift (d73cdc8)

**Full Changelog**: https://github.com/edeno/neurospatial/compare/v0.4.0...v0.5.0

## [0.4.0] - 2026-05-26

This release is the v0.4 UX cleanup: a wide-ranging consolidation of
public-API names, argument orders, return types, error semantics, and
example coverage. There are **no deprecations**; every change is a
clean delete-and-replace. Pin to `<0.4.0` if you need the old surface.

### Breaking changes

#### Parameter renames

- **`distance_metric` / `distance_type` / `use_geodesic` → `metric`** across
  all physical-distance APIs. Legal values are `{"euclidean", "geodesic"}`.
  Affects `Environment.distance_to`, `compute_egocentric_rate(s)`,
  `compute_egocentric_distance`, `compute_spatial_rate(s)`,
  `ObjectVectorCellModel`, `PlaceCellModel`, `BoundaryCellModel`, and the
  boundary / border modules.
- **`smoothing_sigma` / `kernel_bandwidth` → `bandwidth`** across smoothing
  APIs (`compute_spatial_rate(s)`, `compute_view_rate(s)`,
  `compute_egocentric_rate(s)`, `Environment.smooth`, KDE helpers).
- **`velocity_threshold` / `speed_threshold` / `threshold` → `min_speed`**
  across velocity-based behaviour segmentation
  (`segment_by_velocity`, `heading_from_velocity`, etc.).
- **Overlay `data=` → semantic name.** `PositionOverlay(data=...)` →
  `PositionOverlay(positions=...)`; `HeadDirectionOverlay(data=...)` →
  `HeadDirectionOverlay(headings=...)`.

#### Result-class field renames

- **`EgocentricRateResult.ego_env` → `env`**;
  **`ViewRateResult.view_occupancy` → `occupancy`**.
- **`PeriEventResult.firing_rate`** is now a cached attribute, not a method.
  Replace `result.firing_rate()` with `result.firing_rate`.
- **`DecodingResult.uncertainty` → `posterior_entropy`** (matches the free
  function in `decoding/estimates.py`).
- **Singular vs plural method/attribute normalization** on result classes
  (single-neuron results use singular methods; batch results use plural).
  Renames: `SpatialRateResult.peak_firing_rates()` → `peak_firing_rate()`,
  `SpatialRateResult.peak_locations()` → `peak_location()`,
  `ViewRateResult.peak_view_locations()` → `peak_view_location()`.
- **`is_X_cell` method names** normalized to match the free-function names
  (`is_object_vector_cell`, `is_spatial_view_cell`,
  `is_head_direction_cell`).

#### Argument-order canonicalization

- **Encoding functions are now `env`-first**, with the canonical order
  `(env, spike_times, times, positions, headings?, object_positions?, *, ...)`.
  Affects `compute_spatial_rate(s)`, `compute_egocentric_rate(s)`,
  `compute_view_rate(s)`, `detect_place_fields`, `is_spatial_view_cell`,
  `is_object_vector_cell`, `is_border_cell`, and friends.

  ```python
  # v0.3
  compute_spatial_rate(spike_times, times, positions, env)
  compute_egocentric_rate(spike_times, times, positions, headings,
                          object_positions, env=env)
  # v0.4
  compute_spatial_rate(env, spike_times, times, positions)
  compute_egocentric_rate(env, spike_times, times, positions,
                          headings, object_positions)
  ```

- **`compute_directional_rate` / `is_head_direction_cell`** keep the
  heading-domain-native `(spike_times, times, headings, *, ...)`
  signature — this is the documented exception to the env-first rule
  (heading is a circular angular variable, not a spatial position).
  See the function docstrings and `CLAUDE.md` "Canonical Argument
  Order".
- **Egocentric ops** `allocentric_to_egocentric` /
  `egocentric_to_allocentric` reorder to `(positions, headings, targets)`.
- **Behavioural segmentation** functions reordered to
  `(position_bins, times, env, *, ...)`.
- **`distance_to_reward`** in `events.regressors` reordered to
  `(env, times, positions, reward_times, ...)`.
- **`fit_isotonic_trajectory` / `fit_linear_trajectory`** reordered to
  `(env, posterior, times, *, ...)` with a standardized `method`
  keyword.
- **`*` keyword-only separator** added consistently across the public
  API. Numerical parameters and verbose flags become keyword-only.

#### Coordinate / convention changes

- **`Environment.is_1d` → `Environment.is_linearized_track`.** Same
  semantics (a 1-D graph track embedded in 2-D world coordinates); the
  new name resolves the historical "is this n_dims==1 or a 2-D track?"
  ambiguity. Serialized environment metadata uses the new key — pre-v0.4
  saved environments will need to be re-saved.
- **`GridProperties.peak_coords` is now `(x_offset, y_offset)`** instead
  of `(row_offset, col_offset)`. Swap `peak_coords[:, 0]` (was row) for
  `peak_coords[:, 1]` (now y) when reading the second component.
- **`simulate_trajectory_ou(speed_units=...)`** is now required (was
  defaulted). Speed defaults switch from m/s to cm/s. Mismatch between
  `speed_units` and `env.units` raises rather than silently rescaling.

#### Removed (no aliases, no deprecation)

- **`Environment.save` / `Environment.load`.** The pickle path is gone.
  Use `Environment.to_file` / `Environment.from_file` (JSON metadata
  plus npz arrays).
- **`Environment.mask_for_region`.** Use `Environment.region_mask`.
- **`from_image` / `from_mask` factory aliases.** Replaced by
  `from_pixel_mask(image_mask, pixel_size, ...)` and
  `from_grid_mask(active_mask, grid_edges, ...)`.
- **`path_efficiency` (float-returning).** Use `compute_path_efficiency`
  which returns a `PathEfficiencyResult`.
- **Cross-domain re-exports.** Each public symbol now has exactly one
  canonical import path; the top-level `neurospatial` namespace no
  longer re-exports symbols from `encoding`, `decoding`, etc.

### Added

- **`PlaceFieldsResult` dataclass.** `detect_place_fields` returns a
  frozen dataclass with `fields`, `excluded_reason`, and `n_excluded`
  fields. Still iterable / sized / indexable, so existing `for f in
  detect_place_fields(...)` and `len(...)` patterns keep working.
  Closes the "silent drop when mean rate too high" failure mode.
- **`BinSequenceWithRuns` dataclass + new method.**
  `Environment.bin_sequence` always returns an `ndarray`;
  `Environment.bin_sequence_with_runs` returns a dataclass with `bins`,
  `run_starts`, `run_lengths`.
- **`MSDResult` and friends.** Misc result-type cleanup in trajectory
  analysis: `MSDResult`, `SpatialAutocorrelationResult`,
  `PathEfficiencyResult`.
- **`Environment.is_polar` property and `coordinate_kind` attribute.**
  `from_polar_egocentric` sets `coordinate_kind="polar"`. Methods that
  assume Cartesian (`distance_to`, `distance_between`,
  `Environment.contains`, `apply_transform`, `bin_at` on `(x, y)`
  input) raise on polar environments with a clear error.
  `plot_field` switches axis labels and skips the equal-aspect call so
  egocentric polar firing fields still render correctly.
- **Custom exception classes.** `EnvironmentNotFittedError` (already
  existed) now has a free-function variant; added `RegionNotFoundError`,
  `RegionAlreadyExistsError`, and three more in `_exceptions.py`.
- **`Environment.from_pixel_mask` and `Environment.from_grid_mask`
  factories** (replacing `from_image` / `from_mask`).
- **`Environment._state_version` invalidation token.** Cached
  properties verify the version on access; subset / transform / rebin
  bump it, so stale caches are surfaced loudly instead of returning
  silently-wrong results.
- **`Environment.__str__` returns `info()`** for quick inspection.
- **Glossary page** at [docs/glossary.md](docs/glossary.md) defining 14
  core terms. Linked from `docs/getting-started/core-concepts.md` and
  the README.
- **`docs/api/index.md` expansion.** Structured sections for
  `encoding`, `decoding`, `behavior`, `events`, `ops.egocentric`,
  `ops.visibility`, `ops.basis`, `stats`, `animation`, `io.nwb`.
- **`docs/examples/index.md` rewrite.** Goal → notebook table plus
  full per-notebook entries with Time + Prerequisites.
- **Notebooks 24–27.** Object-vector cells, head-direction tuning,
  peri-event PSTH, and NWB loading round-trip.
- **README "Your First Place Field" front-door example.** Canonical
  pattern using `simulate_trajectory_ou`, `PlaceCellModel`,
  `generate_population_spikes`, and `compute_spatial_rate`.
- **CI doc-snippet test.** `scripts/test_doc_snippets.py` plus
  `.github/workflows/test_docs.yml` re-executes a curated manifest of
  doc snippets on every PR.
- **CI notebook regen test.** `.github/workflows/test_notebooks.yml`
  re-executes `11_place_field_analysis.ipynb` per PR to catch silent
  regressions in the example surface.
- **Shared example styling.** `examples/_style.py` Wong / Okabe-Ito
  palette and fixed figure sizes. Wired into every tutorial notebook
  that previously set matplotlib rcParams inline (01-08, 19, 20,
  22-27). Notebooks 09-18 and 21 had no rcParams blocks and
  intentionally keep matplotlib defaults.

### Changed

- **Silent failures replaced with loud failures.**
  - `subset()` round-trip now returns a `MaskedGrid` instead of a one-off
    `subset` layout kind, so the result is fully serializable.
  - `bin_at` vs `map_points_to_bins` standardize on `-1` for
    out-of-environment samples in trajectory contexts.
  - `detect_place_fields` returns a `PlaceFieldsResult` with
    `excluded_reason` set instead of silently returning `[]`.
  - `batch_grid_scores` / `batch_border_scores` use NaN as the explicit
    failure marker and warn once per batch.
  - Fitted-state checks at entry of `compute_spatial_rate(s)`,
    `compute_egocentric_rate(s)`, `compute_view_rate(s)`,
    `decode_position` raise immediately instead of failing deep in the
    call stack.
  - `spike_times` validation rejects unsorted / negative / non-finite
    values with diagnostic messages.
  - `decode_position(validate=True)` is the default; rejects negative
    spike counts and posteriors that don't sum to 1.
- **Canonical exception types** throughout. Manual "not fitted" checks
  migrated to `EnvironmentNotFittedError`; warning-and-overwrite paths
  in `Regions.__setitem__` now raise.
- **Errors carry units and stack context.** Length-mismatch errors
  from `_binning` include a `context` arg so messages say "in
  compute_spatial_rate: ..."; magnitude errors include the offending
  unit.
- **Warning hygiene.** `UserWarning` for data-quality, `RuntimeWarning`
  for numerical fallbacks, `stacklevel=2` everywhere.
- **Production `print()` calls** replaced with module-level
  `logger.info` / `logger.debug`.
- **`Environment.bin_attributes`, `edge_attributes`,
  `differential_operator`** converted from `@cached_property` to
  methods (`get_bin_attributes()`, etc.) so the cost is visible.
- **`Environment.units`** validated against a small registry (`{"cm",
  "m", "mm", "px", None}`) with a `UserWarning` for unknown values.
  Documented as advisory.
- **Heading convention** documented explicitly in every function that
  takes a `headings` argument (allocentric world-frame: 0 = East,
  +π/2 = North; egocentric for OVC tuning: 0 = ahead, +π/2 = left).
- **`events.__init__`** is now eager (was lazy).
- **Bandit-task notebook** prints the download URL and exits cleanly
  when `data/` is missing; CI no longer fails on the example.

### Fixed

- **`repr(env)` `name=None` bug** for empty-string names. Now uses
  `repr(self.name)` so empty strings are visible as `''`.
- **`Environment._state_version` cache invalidation** prevents
  stale-cache reads after mutating operations.
- **Polar environment misuse** is now an error instead of producing
  silently-wrong distances or transforms.

### Removed

- **`Environment.save` / `Environment.load`** (pickle). Replaced by
  `to_file` / `from_file`.
- **`Environment.mask_for_region`.** Use `region_mask`.
- **`from_image` / `from_mask` factory aliases.** Replaced by
  `from_pixel_mask` / `from_grid_mask`.
- **`path_efficiency` float-returning function.** Use
  `compute_path_efficiency`.
- **All cross-domain re-exports** from top-level `neurospatial`.

### Major feature additions (v0.3.x development cycle)

The following features were developed during the v0.3.x development
line and ship as part of v0.4.0. The names below reflect the final
v0.4 surface — many of these symbols were introduced under earlier
names that were renamed during the M2 consolidation pass (see
**Breaking changes** above for the v0.3 → v0.4 mapping).

#### Added (features)

- **Spatial View Cells**: Firing-rate fields indexed by gaze location
  - `compute_view_rate()` / `compute_view_rates()` — single / batch
  - `ViewRateResult` frozen dataclass (`firing_rate`, `occupancy`,
    `env`, plus `view_spatial_information()`, `is_spatial_view_cell()`,
    `sparsity()`, `selectivity()` methods)
  - `is_spatial_view_cell()` free-function classifier
  - `SpatialViewCellModel` simulation model with three gaze models
    (`fixed_distance`, `ray_cast`, `boundary`)

- **Visibility and Gaze**: Ray-casting visibility for view cells
  - `FieldOfView` frozen dataclass with species presets
    (`FieldOfView.rat()`, `FieldOfView.primate()`)
  - `ViewshedResult` frozen dataclass (visible bins + visibility
    fraction)
  - `compute_viewed_location()` — gaze-directed location projection
  - `compute_viewshed()` / `compute_viewshed_trajectory()` /
    `compute_view_field()` — observer-position visibility analysis
  - `visible_cues()` — line-of-sight check to cue / landmark positions
  - `visibility_occupancy()` — time each bin was visible

- **Object-Vector Cells**: Firing fields in egocentric polar coordinates
  - `compute_egocentric_rate()` / `compute_egocentric_rates()`
  - `EgocentricRateResult` / `EgocentricRatesResult` frozen dataclasses
    (`firing_rate`, `occupancy`, `env`, plus `preferred_distance()`,
    `preferred_direction()`, `egocentric_spatial_information()`,
    `is_object_vector_cell()` methods)
  - `object_vector_score()` — distance × direction selectivity metric
  - `is_object_vector_cell()` free-function classifier
  - `plot_object_vector_tuning()` — polar heatmap visualization
  - `ObjectVectorCellModel` simulation model with von Mises directional
    tuning
  - `ObjectVectorOverlay` for animation with object–animal vectors

- **Egocentric Reference Frames**: Foundation for object-vector and view cells
  - `EgocentricFrame` dataclass with `to_egocentric()` / `to_allocentric()`
  - `allocentric_to_egocentric()` / `egocentric_to_allocentric()` —
    batch frame transforms
  - `compute_egocentric_bearing()` — angle to targets relative to
    heading (egocentric convention: 0 = ahead, +π/2 = left)
  - `compute_egocentric_distance()` — Euclidean and geodesic distance
  - `heading_from_velocity()` / `heading_from_body_orientation()` —
    derive heading from tracking data
  - `Environment.from_polar_egocentric()` — egocentric polar coordinate
    environment with optional circular connectivity

- **3D Transform Support**: Full N-dimensional affine transformation capabilities
  - New `AffineND` class for N-dimensional affine transforms using (N+1)×(N+1) homogeneous matrices
  - `Affine3D` type alias for convenience (equivalent to `AffineND` with n_dims=3)
  - 3D factory functions: `translate_3d()`, `scale_3d()`, `from_rotation_matrix()`
  - Integration with `scipy.spatial.transform.Rotation` for 3D rotations
  - `estimate_transform()` now auto-detects dimensionality (2D or 3D) from input points
  - `apply_transform_to_environment()` supports N-dimensional environments with validation
  - 45 new tests for 3D transforms including scipy integration
  - Backward compatible: `Affine2D` unchanged for existing 2D workflows

- **N-D Probability Mapping**: `alignment.py` now accepts N×N rotation matrices
  - `map_probabilities_to_nearest_target_bin()` supports 3D environments
  - Updated `_transform_source_bin_centers()` validates rotation matrix dimensionality
  - For 2D: Use `get_2d_rotation_matrix(angle_degrees)`
  - For 3D: Use `scipy.spatial.transform.Rotation.as_matrix()`

#### Changed (internal)

- **Internal**: Refactored `environment.py` (5,335 lines) into modular package structure for improved maintainability
  - Split into 11 focused modules: `core.py` (1,023 lines), `factories.py` (630 lines), `queries.py` (897 lines), `trajectory.py` (1,222 lines), `transforms.py` (634 lines), `fields.py` (564 lines), `metrics.py` (469 lines), `regions.py` (398 lines), `serialization.py` (315 lines), `visualization.py` (211 lines), `decorators.py` (77 lines)
  - Implemented mixin pattern: Environment inherits from all functionality mixins
  - No breaking changes - `from neurospatial import Environment` continues to work
  - All 1,076 tests passing (100% success rate)
  - Improved code organization for easier contribution and maintenance
  - Largest module is trajectory.py at 1,222 lines (down from original analysis.py at 2,104 lines)

#### Documentation

- **3D Support**: Updated dimensionality support documentation to reflect 3D transforms availability
  - Updated `docs/dimensionality_support.md` with 3D transform examples and feature matrix
  - Updated `docs/user-guide/alignment.md` with comprehensive 3D transformation examples
  - Added complete 3D alignment workflow example
  - Updated compatibility matrix: ~75% of neurospatial now works in 3D (up from 70%)

## [0.2.0] - 2025-11-04

### Added

#### Environment Operations (Complete Feature Set)

**Core Analysis Operations (P0)**

- `Environment.occupancy()` - Compute time-in-bin from trajectory data with speed filtering, gap handling, and optional kernel smoothing
- `Environment.bin_sequence()` - Convert trajectories to bin sequences with run-length encoding
- `Environment.transitions()` - Compute empirical transition matrices with adjacency filtering and normalization
- `Environment.components()` - Find connected components in environment graph
- `Environment.reachable_from()` - Compute reachable bins via BFS or geodesic distance

**Smoothing & Resampling (P1)**

- `Environment.smooth()` - Apply diffusion kernel smoothing to arbitrary fields
- `Environment.rebin()` - Conservative grid coarsening with mass/mean aggregation (grid-only)
- `Environment.subset()` - Extract subregions by bins, regions, or polygons

**Interpolation & Field Utilities (P2)**

- `Environment.interpolate()` - Evaluate bin-valued fields at continuous points (nearest/linear modes)
- `Environment.occupancy()` linear mode - Ray-grid intersection for accurate boundary handling (grid-only)
- `field_ops.py` module:
  - `normalize_field()` - Normalize to probability distribution
  - `clamp()` - Bound field values
  - `combine_fields()` - Weighted combination (mean/max/min)
  - `divergence()` - KL/JS divergence and cosine distance

**Utilities & Polish (P3)**

- `Environment.region_membership()` - Vectorized bin-to-region containment checks
- `Environment.distance_to()` - Compute distances to target bins or regions (Euclidean/geodesic)
- `Environment.rings()` - K-hop neighborhoods via BFS layers
- `Environment.copy()` - Deep/shallow copying with cache invalidation
- `spatial.map_points_to_bins()` - Enhanced with `max_distance` and `max_distance_factor` thresholds for deterministic boundary decisions

**Diffusion Kernel Infrastructure**

- `kernels.py` module:
  - `compute_diffusion_kernels()` - Matrix-exponential heat kernel on graphs with volume correction
  - `Environment.compute_kernel()` - Convenience wrapper with caching
  - Support for both transition and density normalization modes

**Documentation**

- `docs/user-guide/spatial-analysis.md` - Comprehensive 1,400+ line guide covering all operations with scientific context
- `docs/examples/08_complete_workflow.ipynb` - Enhanced workflow notebook with movement/navigation analysis
- All methods have NumPy-style docstrings with working examples
- "Why This Matters" sections explaining scientific motivation for key operations

### Changed

- **GraphLayout**: Now supports 1D layouts correctly (conditional `angle_2d`, dynamic `dimension_ranges`)
- **KDTree operations**: Now deterministic by default using `tie_break="lowest_index"`
- **All environment operations**: Use `@check_fitted` decorator for consistent state enforcement
- **Input validation**: Comprehensive validation with diagnostic error messages across all operations
- **Caching**: Object identity-based caching for kernels and spatial queries

### Fixed

- GraphLayout `angle_2d` computation for 1D graphs (was unconditionally assuming 2D)
- GraphLayout `dimension_ranges` now correctly handles 1D case
- Disconnected graph handling in connectivity tests
- Hexagonal layout interpolation edge cases

### Testing

- **1067 tests passing** (up from 614 in v0.1.0)
- **0 skipped tests** (eliminated all 12 previous skips)
- Performance benchmarks: occupancy on 100k samples, large transition matrices, kernel computation
- Integration tests: end-to-end workflows, multi-layout compatibility
- Edge case coverage: empty environments, single bins, disconnected graphs

### Internal

- Systematic debugging skill used to eliminate all test skips
- Test-driven development for all features
- Code review and UX review completed
- Pre-commit hooks for code quality

## [0.1.0] - 2025-11-03

### Added

- **CompositeEnvironment API parity**: Added `bins_in_region()`, `mask_for_region()`, `shortest_path()`, `info()`, `save()`, and `load()` methods to CompositeEnvironment for full API compatibility with Environment class
- **KDTree-optimized spatial queries**: CompositeEnvironment.bin_at() now uses KDTree for O(M log N) performance instead of O(N×M) sequential queries (enabled by default via `use_kdtree_query=True`)
- **Structured logging infrastructure**: New `_logging.py` module with NullHandler by default, enabling optional logging for debugging and workflow tracing
- **Centralized numerical constants**: New `_constants.py` module consolidating all magic numbers (tolerances, KDTree parameters, epsilon values) for consistent behavior
- **Comprehensive type validation**: CompositeEnvironment constructor now validates input types with actionable error messages
- **Graph metadata validation**: Added `validate_connectivity_graph()` to enforce required node/edge attributes from layout engines
- **Dimensionality support documentation**: New `docs/dimensionality_support.md` clarifying 1D/2D/3D feature support with compatibility matrix

### Changed

- **Updated alignment module**: Now uses centralized constants (`IDW_MIN_DISTANCE`, `KDTREE_LEAF_SIZE`)
- **Updated regions module**: Uses `POINT_TOLERANCE` constant for consistent geometric comparisons
- **Enhanced error messages**: CompositeEnvironment now provides detailed diagnostics for dimension mismatches and type errors
- **Clarified 2D-only transforms**: Updated `transforms.py` docstring to explicitly state 2D-only status and suggest scipy for 3D

### Fixed

- Removed unused `type: ignore` comment in `regular_grid.py`
- Fixed potential `KeyError` in logging by renaming `name` parameter to `env_name` (avoids conflict with LogRecord reserved field)

### Documentation

- Added comprehensive dimensionality support guide (1D/2D/3D feature matrix)
- Updated CLAUDE.md with latest patterns and requirements
- Added 18 new tests for CompositeEnvironment type validation
- Added 23 new tests for Environment error path coverage
- Added 28 new tests for graph validation

### Internal

- Consolidated duplicate dimension inference code
- All 614 tests passing
- Ruff and mypy checks passing
- Test coverage: 78%

[Unreleased]: https://github.com/edeno/neurospatial/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/edeno/neurospatial/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/edeno/neurospatial/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/edeno/neurospatial/compare/v0.2.0...v0.4.0
[0.2.0]: https://github.com/edeno/neurospatial/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/edeno/neurospatial/releases/tag/v0.1.0
