# neurospatial — API & Design Review (for neuroscience users)

## Executive summary

neurospatial is a well-architected, idiomatically named library for spatial-coding neuroscience whose core design decisions are mostly right: `Environment` is the correct central abstraction, the factory-only construction pattern enforces a real fitted invariant rather than imposing ceremony, the cell-type/metric vocabulary maps one-to-one onto the field (place/grid/HD/border/object-vector/spatial-view cells; Skaggs information, grid score, MVL, with primary-literature citations), and the highest-value array contracts compose seamlessly (`compute_spatial_rates(...).firing_rates` is exactly `decode_position`'s `encoding_models` shape; `DecodingResult.posterior` feeds `animate_fields` directly). The single most damaging cross-cutting problem is that the busiest seam in a spike library — load spikes → bin them in time → decode/GLM/reactivation — has no public support: there is no `read_units`/`read_spikes` NWB reader and no public temporal spike-binning helper, forcing every multi-cell workflow into bare pynwb and hand-rolled `np.histogram`, and exposing a silent transpose footgun between `decode_position` (`n_time_bins, n_neurons`) and the assembly functions (`n_neurons, n_time_bins`). The second systemic problem is that the library's own recommended defaults and copy-paste examples produce crashes or silently-wrong results — `min_occupancy=0.5` injects NaN that `decode_position` rejects, the reactivation statistics are inert (EV==REV, strength ~1 by construction), and several QUICKSTART snippets raise on first run. Third, conceptual integrity frays where `Environment` is overloaded for polar/egocentric space via a hidden `coordinate_kind` flag that disables ~5 core methods, and where the result-object family splits into rich-mixin results versus bare dataclasses (`DirectionalPlaceFields` has no `.plot()`/`.compare()`). None of these is a hard blocker; together they mean a competent user reaches a real result, but typically after writing ~50% plumbing and hitting at least one confusing failure the docs led them into.

## User journeys

### NWB → place fields → Bayesian decode → animate posterior
**Feasibility: workable.** The two ends are smooth (`read_position`/`environment_from_position` in; `animate_fields(result.posterior, frame_times=result.times)` out), but the neuroscience middle forces three drops into raw pynwb and NumPy. Key snags: **no `read_units`/`read_spikes`** so step one uses bare `nwbfile.units` (`neurospatial.io.nwb`); **no public temporal spike-binner** so the user hand-rolls a per-neuron `np.histogram` loop for `decode_position`'s `(n_time_bins, n_neurons)` counts; and the **`min_occupancy=0.5` NaN trap** where the recommended encoding call produces models `decode_position(validate=True)` rejects, discoverable only via crash.

### Linear/W-track directional place fields
**Feasibility: workable.** Achievable via the real golden path (`segment_trials` → `goal_pair_direction_labels` → `compute_directional_place_fields`), but the connective tissue is hard to find. Key snags: the **segmentation→directional seam** needs a per-timepoint `direction_labels` object array (with magic string `"other"`) that none of `detect_laps`/`detect_runs_between_regions`/`segment_trials` returns directly — only `goal_pair_direction_labels` bridges, and only from `Trial`, so a user reaching for `detect_laps` dead-ends; **no first-class inbound/outbound concept** (must invent named end-regions); and **`DirectionalPlaceFields` is a bare dataclass** with no `.plot()`/`.compare()`/`.correlation()`, so the literal task verb "compare them" drops to NumPy.

### Object-vector cells (heading → bearing → field → classify)
**Feasibility: workable.** QUICKSTART maps almost 1:1 onto the four steps and the allocentric/egocentric convention story is coherent and well-documented. Key snags: **two functions named `is_object_vector_cell`** (free function uses score+peak; `EgocentricRateResult.is_object_vector_cell` uses spatial information) give different answers for the same cell (`encoding/egocentric.py:1869` vs `:388`); **`heading_from_velocity` dt is user-computed and unvalidated** so `dt<=0`/unsorted times silently yield NaN or 180°-flipped headings (`ops/egocentric.py:635`); and the **QUICKSTART OVC classify/plot snippet is mis-coded against a batch result** and crashes on paste (`QUICKSTART.md:459-468`).

### Simulate → encode → decode → reactivation
**Feasibility: workable (final stage effectively blocked).** Simulation (`open_field_session`) and encoding are the high point and compose with zero glue. Key snags: **reactivation statistics are silently meaningless** — `rev = ev` is hardcoded so the EV>REV control is inert, `reactivation_strength` is ~1 by double z-scoring, partial r is returned un-squared (`decoding/assemblies.py:959-972,1099,1115`); the **transposed spike-count axis** between `decode_position` and the assembly functions; **no public spike-binner**; and **`decoding_error` alignment** (decode grid → ground-truth positions via hand-rolled `searchsorted`) is left to the user.

### Event-locked firing (PSTH + GLM + relate to position)
**Feasibility: workable.** The PSTH half is excellent (`peri_event_histogram` → `PeriEventResult` with cached `.firing_rate`, one-line `plot_peri_event_histogram`). The GLM half compounds three frictions: **`window` means three different things** across sibling regressors (`time_to_nearest_event` `max_time`; `event_count_in_window` `(start,end)` tuple; `event_indicator` scalar half-width, keyword-only); the **documented `event_indicator` example passes a tuple** and raises on paste (`QUICKSTART.md:863-866`); **no public spike-binner** for the GLM `y` vector; and **`distance_to_reward` does no NaN validation** (`events/regressors.py:617-783`), silently corrupting `searchsorted` ordering on NaN reward times.

## Design axes

### Mental model & core abstractions
`Environment` is the right central object, the bin/node/graph model genuinely matches how neuroscientists think about discretized arenas and linearized tracks, and conceptual integrity is high within the Cartesian-spatial core. It frays at three seams a per-module review can't see: polar space wedged into `Environment`, `CompositeEnvironment` advertised-but-not-substitutable, and a split result-object family.

**Strengths**
- One frozen-invariant `Environment` assembled from 9 single-responsibility mixins; users see one noun, contributors get clean separation (`core.py:57-69`).
- Every layout (grid/hex/graph/masked/mesh/polygon) reduces to the same `(bin_centers, graph)` pair behind `LayoutEngine`; "active bins" is exactly the right domain primitive.
- Factory-only construction enforces a real invariant (`@check_fitted`, `_is_fitted` set only at end of setup) — no half-initialized escape hatch.
- `regions`/`encoding`/`decoding` are genuinely orthogonal minimal domains composing through `Environment` without circular deps.

**Weaknesses**
- [high] Polar egocentric space overloaded into `Environment` via `coordinate_kind` + `_check_cartesian` guard that disables `bin_at`/`distance_between`/euclidean `distance_to` at runtime — the central noun silently means two geometries (`core.py:920-967`, `factories.py:844-1004`).
- [high] `CompositeEnvironment` is explicitly NOT a superset (its own docstring admits surface drift: `save/load` vs `to_file/from_file`, cached properties vs methods) — a Liskov trap for `env: Environment | CompositeEnvironment` (`composite.py:1-33`).
- [medium] Result family split: `SpatialRateResult` et al. inherit a rich mixin, but `PlaceFieldsResult`/`DirectionalPlaceFields` are bare dataclasses (`spatial.py:87,1712` vs `_base.py:158`).
- [medium] Per-factory required-geometry rules (`bin_size` vs `pixel_size` vs neither) are an unwritten contract memorized from CLAUDE.md prose.
- [low] `is_linearized_track`/`is_polar`/`coordinate_kind` are three overlapping ways to ask "what kind of space is this."

### Onboarding & the golden path
The single-cell place-field path is genuinely strong (~4 lines, correct defaults including `min_occupancy=0.0`), with model first-run diagnostics (E1001) and 27 task-named example notebooks. Onboarding breaks down exactly when a task spans multiple domains, and the most damaging issues are the library's own docs leading users into crashes or silently-wrong numbers.

**Strengths**
- `compute_spatial_rate(env, spike_times, times, positions)` works with zero kwargs and safe defaults; result object gives answer + QC in one call.
- Factory-only + required `bin_size` eliminates the "unfitted object" footgun.
- E1001 "no active bins" is a model error with data range, params, causes, and remedies (`layout/engines/regular_grid.py:165+`).
- Sparse self-documenting top-level namespace; encode→decode and decode→animate handoffs "just work."

**Weaknesses**
- [high] Recommended `min_occupancy=0.5` produces NaN models that `decode_position` rejects — the library's own notebook `20_bayesian_decoding.py:246` silently patches with `np.nan_to_num`, proving the seam is known but unsurfaced.
- [high] No spike-train → temporal count helper; every multi-cell workflow hand-rolls `np.histogram`.
- [high] No `read_units`/`read_spikes`; the flagship NWB notebook (27) never loads a spike.
- [high] Multiple first-contact doc examples raise on paste (`PositionOverlay(data=)`, `event_indicator` tuple window, OVC batch-vs-single, `circular_basis_metrics` kwargs).
- [medium] Undiscoverable required setup: `frame_times`, hand-computed `dt`, heading, per-timepoint direction labels.
- [medium] Reactivation path runs end-to-end but returns scientifically meaningless numbers with no error.

### Cross-module API consistency & predictability
The library documents and largely follows a real convention set (env-first, data-before-metadata, keyword-only params, one angle convention) and the within-family result pattern is strong. The consistency problems are systemic but shallow — predictability traps at the seams between families, not deep rot.

**Strengths**
- Documented canonical argument order with principled, flagged exceptions (directional functions omitting `env` because heading is angular).
- Within-family result parity for encoding (`Spatial/View/Egocentric/Directional RateResult` share `SpatialResultMixin`; batch siblings add `.to_dataframe()` + classifier).
- Load-bearing array contracts correctly oriented and glue-free.
- Single coherent allocentric/egocentric angle convention, doctested.

**Weaknesses**
- [high] No global result-object parity: three incompatible "rich result" idioms plus a tier of method-less bare dataclasses (`PlaceFieldsResult`, `DirectionalPlaceFields`, `GridProperties`, assembly results, `PeriEventResult`).
- [high] `spike_counts` axis silently transposed between `decode_position` and `detect_assemblies`/`pairwise_correlations` (`assemblies.py:384`).
- [medium] RNG arg is `rng` in most places but `random_state` in `detect_assemblies` and all five `ops.basis` functions.
- [medium] Batch classifier named `.classify()` on `SpatialRatesResult` but `.detect_hd_cells()`/`.detect_ovcs()`/`.detect_view_cells()` on siblings.
- [medium] `window` means three things across sibling GLM regressors; two `is_object_vector_cell` functions compute different statistics.
- [medium] `env` placement / keyword-only separator violated in `decoding_error`, `compute_vte_session`, and several `ops` functions.
- [low] `.firing_rate` (attribute) vs `.peak_firing_rate()` (method); `compute_egocentric_rate` defaults to `binned` while `compute_spatial_rate` defaults to `diffusion_kde`.

### Discoverability & namespacing
The deliberate sparse-top-level + domain-submodule design is largely well-executed, with a top-level docstring that is a real navigation map and `io.nwb` demonstrating the correct lazy `__getattr__` pattern. The two serious gaps are concentrated where real users hit them.

**Strengths**
- Top-level docstring enumerates every submodule with one-line purpose + copy-pasteable import (`__init__.py:22-123`).
- Disciplined sparse `__all__`; domain submodules flat-re-export functions AND result classes.
- `goal_pair_direction_labels` correctly re-exported at `behavior` top level.
- `io.nwb` uses lazy `__getattr__` with explicit `__all__` — the pattern the top-level package is missing.

**Weaknesses**
- [high] No top-level `__getattr__`: after `import neurospatial as ns`, `ns.encoding`/`ns.decoding`/`ns.behavior`/etc. raise `AttributeError` — most domains invisible to autocomplete.
- [high] The central neural data type has no namespace entry point: no `read_units`, no public spike-binner.
- [medium] `is_object_vector_cell` same-name/different-math collision (free fn vs result method).
- [medium] Power functions submodule-only: `grid.spatial_autocorrelation_radial`, the `make_*` maze factories, `env_from_boundary_region`.
- [low] Sibling classifier method-name inconsistency; egocentric primitives live under `ops` not `encoding` where OVC users look.

### Domain fit & vocabulary
neurospatial speaks fluent spatial-coding neuroscience — cell types, metrics, frames, and citations are all named the way the field names them. Gaps appear at the edges: ripple/replay, raw spike-train statistics, and one terminology split.

**Strengths**
- Cell-type vocabulary matches the field one-to-one; simulation models mirror it exactly.
- Metrics named and cited as in primary literature (Skaggs 1993, Solstad 2008, Sargolini 2006, O'Keefe & Recce 1993) with stated units.
- Egocentric/allocentric frame distinction is first-class and correctly conventioned.
- Result accessors read like a methods section (`spatial_information()`, `preferred_direction()`, `mean_vector_length()`).

**Weaknesses**
- [high] No SWR/ripple detector or event vocabulary despite a replay-analysis decoding module — the hippocampal pipeline breaks at step one.
- [high] Reactivation/EV statistics mislabeled relative to the cited literature (REV==EV, control ignored).
- [medium] No raw spike-train toolkit (autocorrelogram, ISI, burst index, theta-phase) — `phase_precession` consumes phases the library can't compute.
- [medium] Same quantity is `mean_resultant_length` (stats) vs `mean_vector_length` (encoding).
- [medium] No first-class inbound/outbound running-direction primitive for linear tracks.
- [low] "heading" vs "head direction" used loosely; `spike_times`/`spike_trains`/`spike_counts` naming inconsistent across the pipeline.

### Ecosystem fit & interoperability
Plain-array-in / result-object-out is a genuinely low-friction contract with no proprietary Session object, and the shape-contract alignment across stage boundaries is the standout interop win. The hole is in the middle, exactly where spike-centric neuroscience lives.

**Strengths**
- Plain numpy in, result dataclasses out (with `.to_dataframe()` on heavy results); no mandatory wrapper to marshal through.
- Shape contracts compose across boundaries (`firing_rates` → `encoding_models`; `posterior` → `animate_fields`).
- Optional NWB dependency done right (lazy imports, extras); readers for position/HD/pose/events/intervals/trials.
- In-memory env interchange via `to_dict`/`from_dict` for multiprocessing/JSON.

**Weaknesses**
- [high] No NWB reader for spikes/units — the primary neural type is the one type with no reader.
- [high] No public spike-time → spike-count binner; users hand-roll `np.histogram` twice.
- [high] Transposed spike-count axis between `decode_position` and the assembly functions.
- [medium] Recommended encoding params produce NaN maps the recommended decode call rejects.
- [medium] xarray entirely unsupported despite outputs being textbook labeled `(time, bin)`/`(neuron, bin)` arrays (xarray is native in Spyglass decoding).
- [medium] NWB round-trip silently drops `coordinate_kind`, flipping polar envs to Cartesian (`io/nwb/_environment.py:230-243`); readers skip length/unit/frame checks (degrees-encoded HD read as radians).
- [low] `environment_from_position` discards timestamps; stale interop examples teach wrong constructors.

### Composability across modules
Well-composed at the endpoints, discontinuous in the middle. The recurring structural problem is that the library owns rich spatial binning but no temporal spike-count binning, even though three downstream consumers need it — this single gap appears in four of five journeys.

**Strengths**
- Encode→decode handoff shape-correct by construction (active-bin count matches even on masked grids).
- Decode→animation genuinely seamless.
- Batch encoding is a true single-call fan-out preserving both array and per-cell views.
- The directional seam HAS a correct bridge (`goal_pair_direction_labels`) with matching arg order; simulation→encoding and NWB-position→Environment compose with no glue.

**Weaknesses**
- [high] No public temporal spike-binner — the missing link between every spike-time source and every count-matrix consumer.
- [high] Transposed axis convention between `decode_position` and the assembly stack (silent garbage-in).
- [high] Recommended encoding params produce NaN that the recommended decode call rejects.
- [medium] Segmentation→directional seam returns objects, not the label array the consumer needs — and only `Trial` is bridged.
- [medium] `DirectionalPlaceFields` cannot do the task it exists for (no compare/plot/correlate).
- [medium] No spikes/units NWB reader; decode→error time-grid alignment left to the user.
- [low] Two `is_object_vector_cell` functions give different answers.

## Prioritized recommendations

### High

- **Add a public temporal spike-binning helper with an explicit `orient=` argument.** Export `bin_spikes_in_time(spike_trains, dt, t_start=None, t_stop=None, *, orient='time_x_neuron'|'neuron_x_time') -> (counts, bin_centers)` from a discoverable location (`neurospatial` or `neurospatial.decoding`/`.encoding`), owning the time-grid and `dt/2` bin-center construction. *Cross-cutting (composability, interop, onboarding, consistency, discoverability): repairs the single most-cited seam — present in 4 of 5 journeys — and the `orient=` flag simultaneously defuses the silent `decode_position` ↔ assembly transpose footgun by making the axis choice explicit.*

- **Add `read_units`/`read_spikes` to the NWB reader family and extend notebook 27 through a real analysis.** `read_units(nwbfile, *, unit_ids=None) -> (list[np.ndarray], np.ndarray)` mirroring `read_position`'s tuple contract, handling the ragged `units` DynamicTable; add it to `io/nwb` `_LAZY_IMPORTS`/`__all__`; extend `examples/27` to read_units → compute_spatial_rates → bin_spikes_in_time → decode_position → animate_fields. *Cross-cutting (interop, onboarding, discoverability, composability): spikes are the one neural type with no reader, so every spike pipeline starts in bare pynwb.*

- **Reconcile the `min_occupancy`/NaN-encoding-model collision.** Add `fill_value: float|None = 0.0` to `compute_spatial_rate(s)` so masked bins are zero-rate, OR have `decode_position` treat NaN encoding-model bins as zero-rate (excluded from the Poisson likelihood, with a one-time warning); stop shipping the silent `np.nan_to_num` in `examples/20` and change the documented `min_occupancy=0.5` recommendation. *Cross-cutting (onboarding, interop, composability): the library's recommended encode call currently crashes the recommended decode call.*

- **Fix the reactivation/EV statistics (or doc-flag them loudly until fixed).** Implement REV as the role-swapped (Kudrimoti) reversed explained variance so EV>REV is meaningful, set `ev = partial_corr**2` when a control is supplied, add a `reactivation_strength` path not pre-z-scored to ~1 (`assemblies.py:959-972,1099,1115`). *Cross-cutting (domain fit, onboarding, journeys): the API uses precise field terminology for quantities it does not compute, so users publish meaningless numbers with no error.*

- **Add a top-level `__getattr__` for lazy submodule access.** Mirror the proven `io/nwb` pattern: `__getattr__` returning `importlib.import_module(f'neurospatial.{name}')` for the known submodules, plus a `__dir__` listing them, so `ns.encoding.<TAB>` works without eager imports. *Discoverability: autocomplete is the dominant discovery path and currently reveals only a fraction of domains.*

- **Make every first-contact doc example runnable and gate it in CI.** Fix the crashing QUICKSTART/CLAUDE snippets (`PositionOverlay` `data=`→`positions=` + required `times=`; `BodypartOverlay` raw-list→`Skeleton`; `event_indicator` scalar `window`; OVC batch-vs-single methods; `circular_basis_metrics` kwargs), then execute QUICKSTART/example blocks in CI. *Cross-cutting (onboarding, interop): a paste-and-crash example is the most direct way to lose a new user's trust.*

### Medium

- **Define and enforce a single cross-module result-object contract.** A base `ResultMixin` guaranteeing `.to_dataframe()`, `.plot()`, `.summary()`; backfill the bare dataclasses first — give `DirectionalPlaceFields` `.plot()` (per-direction overlay), `.correlation(label_a, label_b)`/directionality index, `.to_dataframe()`; add `.summary()` to encoding/decoding results and `.plot()`/`.to_dataframe()` to behavior results. *Cross-cutting (mental model, consistency, composability, journeys): the split bites at the END of the analysis where "compare them"/"score it" dead-ends to NumPy.*

- **Add a Lap/Run → direction_labels bridge and a net-direction labeler.** `laps_to_direction_labels`/`runs_to_direction_labels` returning the per-timepoint `"other"`-excluded array, plus a first-class `running_direction_labels(position_bins, times, env) -> {'inbound','outbound','other'}` for linear tracks; cross-link from `detect_laps`/`detect_runs_between_regions`. *Cross-cutting (composability, domain fit, journeys): `goal_pair_direction_labels` bridges only `Trial`, so the lap-centric mental model dead-ends, and inbound/outbound has no named primitive.*

- **Resolve `Environment`'s polar overloading.** Promote `from_polar_egocentric`'s output to a distinct `EgocentricPolarEnvironment` (or clearly-marked wrapper) instead of a `coordinate_kind` flag, and fix the radians-as-edge-distance issue so geodesic ops are correct; persist `coordinate_kind` through NWB round-trip (`io/nwb/_environment.py`). *Cross-cutting (mental model, interop): a type that disables ~5 of its own methods via a hidden flag breaks the contract downstream functions rely on, and is the root cause of cm/rad unit-mixing bugs and the silent polar→Cartesian round-trip.*

- **Standardize naming: RNG arg, batch classifier, `is_object_vector_cell`.** Rename `random_state` → `rng` (deprecate alias) in `detect_assemblies` and the five `ops.basis` functions; adopt `detect_<celltype>s()` across all `…RatesResult` (alias `classify`); make the free `is_object_vector_cell` delegate to the result method so the two cannot disagree. *Cross-cutting (consistency, discoverability, journeys): restores reproducibility and classifier muscle-memory; removes a same-name/different-answer trap.*

- **Validate the hand-derived setup inputs that silently corrupt results.** Raise on `dt<=0`/non-finite/unsorted times in `heading_from_velocity`; add NaN/Inf validation to `distance_to_reward` (`events/regressors.py:617`); add a `validate_neural_inputs(...)` entry point since the `is_*_cell` classifiers swallow `ValueError` and return `False` on length mismatch. *Cross-cutting (onboarding, journeys): the inputs users hand-roll are exactly where silent corruption enters.*

- **Add `to_xarray()` to the array-shaped results and a decode-error alignment overload.** `DecodingResult.to_xarray()` with `('time','bin')` coords, `SpatialRatesResult.to_xarray()` with `('neuron','bin')`; and a `decoding_error` overload (or `DecodingResult.error_against(true_times, true_positions)`) that aligns internally via searchsorted/interpolation. *Cross-cutting (interop, composability, journeys): xarray is the de-facto container in modern decoding pipelines, and the decode→error alignment is recurring hand-rolled glue.*

- **Unify the events `window` contract and the rate-smoothing default.** Give `time_to_nearest_event`/`event_count_in_window`/`event_indicator` one `window` representation (recommend `(start, end)` everywhere) with consistent keyword-only placement; make `compute_egocentric_rate` default `smoothing_method='diffusion_kde'` to match `compute_spatial_rate` (or document the divergence prominently). *Consistency/journeys: side-by-side regressors meant to be column-stacked shouldn't each interpret `window` differently.*

- **Add SWR/ripple vocabulary (or explicitly scope it out).** `events.detect_ripples(lfp, sampling_rate, *, band=(150,250), ...) -> intervals` and a "ripple" path through `peri_event_histogram`; if out of scope, say so prominently in the replay docstrings and point to `ripple_detection`/`elephant`. *Domain fit: the decoding module is framed around replay, which is defined on ripple events the library can't produce.*

### Low

- **Restore documented argument-order conventions and re-export egocentric primitives.** Move `env` behind `*,` (or first) in `decoding_error`/`compute_vte_session`; add the `*` separator before algorithm params in the `ops.basis`/`combine_fields`/`compute_diffusion_kernels`/distance functions; re-export `heading_from_velocity`/`compute_egocentric_bearing`/`compute_egocentric_distance` from `neurospatial.encoding`. *Consistency/discoverability: isolated, mechanical violations of conventions the rest of the library honors.*

- **Give the factory family one introspectable geometry-spec contract and a `space_kind` accessor.** A `list_factories()`/`get_factory_parameters()` helper (mirroring `layout.get_layout_parameters`) and a single `env.space_kind` computed from the existing flags. *Mental model: construction is the first thing every user does, and "what kind of Environment is this" should be one question.*

- **Re-export submodule-only power functions and unify circular-concentration naming.** Surface `spatial_autocorrelation_radial` at `encoding`, the `make_*` maze factories at `simulation`, `env_from_boundary_region` at `annotation`; pick one canonical term for MRL/MVL with an explicit documented alias. *Discoverability/domain fit: low-cost index and vocabulary fixes.*

- **Add a theta-phase extraction helper and disambiguate head direction vs movement heading.** `encoding.theta_phase(lfp, sampling_rate, *, band=(6,10))` to feed `phase_precession`; a docstring note in `compute_directional_rate`/`is_head_direction_cell` that velocity-derived heading is movement direction, not head direction. *Domain fit: makes the cited O'Keefe & Recce path self-contained and prevents a common methodological mislabel.*

- **Have `environment_from_position` return `(env, positions, timestamps)`.** Removes a redundant second NWB read on the most common entry point. *Interop: small ergonomic tax on the busiest NWB on-ramp.*

- **Ship one worked end-to-end NWB → fields → decode → animate example covering the seams together.** Showing `read_units`, `bin_spikes_in_time`, NaN handling, and `frame_times = time_bins[:-1]+dt/2` in one runnable block. *Cross-cutting: every stage is documented in isolation; users get stuck at the joints.*

## What's working (keep)

- **`Environment` as the single central object built from focused mixins** — one frozen-invariant dataclass, ~40 cohesive methods, clean contributor separation (`core.py:57-69`).
- **Factory-only construction with required `bin_size`/`pixel_size` and `@check_fitted`** — enforces a real invariant and eliminates the "unfitted object" footgun. No bare `Environment()`.
- **The bin/node/graph model unifying grids, hexagons, tracks, polygons, and meshes** behind one `LayoutEngine`, with "active bins" as a domain-correct primitive.
- **The load-bearing array contracts**: `compute_spatial_rates(...).firing_rates` is exactly `decode_position`'s `encoding_models` (even on masked grids), and `DecodingResult.posterior` feeds `animate_fields` directly — these compose with zero glue and must not regress.
- **Within-family encoding result parity** (`Spatial/View/Egocentric/Directional RateResult` sharing `SpatialResultMixin`; batch siblings adding `.to_dataframe()` + classifier) — the model the other result classes should converge toward.
- **The single coherent, doctested allocentric/egocentric angle convention** (0=East allocentric; 0=ahead, +π/2=left egocentric), stated repeatedly and applied internally so users pass world-frame headings.
- **Field-idiomatic vocabulary and primary-literature citations** across cell types and metrics — the strongest reason a neuroscientist trusts the library.
- **The PSTH path** (`peri_event_histogram` → `PeriEventResult` with cached `.firing_rate`, one-line `plot_peri_event_histogram`) and `open_field_session` simulation entry point — genuinely smooth, glue-free starting points.
- **The E1001 "no active bins" diagnostic** and the sparse, self-documenting top-level namespace — models of good first-run guidance.
- **`io.nwb`'s lazy `__getattr__` + explicit `__all__` pattern** for the optional dependency — the correct pattern, and the template for the top-level fix.