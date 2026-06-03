# neurospatial — Unified Roadmap (Bugs × Design)

Merges the two independent reviews into one implementation-ordered plan:
- **Correctness** — [SUMMARY.md](SUMMARY.md) (43 criticals, 162 important across 13 modules)
- **API & design** — [DESIGN-REVIEW.md](DESIGN-REVIEW.md) (5 user journeys, 7 design axes)

Each item is tagged **[BUG]** (correctness — wrong/crashing results), **[DESIGN]** (ergonomics/capability gap), or **[BOTH]** (flagged independently by *both* reviews — highest confidence). `file:line` refs are exact; the long tail of correctness criticals is grouped by phase and cross-linked to SUMMARY.md rather than re-typed.

---

## Convergence — flagged by BOTH reviews (do these first)

Where an independent bug hunt and an independent design critique land on the *same code*, confidence is highest. These twelve are the backbone of the plan:

| # | Item | Where | Bug ref | Design ref |
|---|---|---|---|---|
| C1 | Reactivation/assembly stats are inert (REV≡EV, double z-scoring, control ignored) | `decoding/assemblies.py:959-972,1096-1115` | crit | High #4 |
| C2 | `min_occupancy` NaN encoding-models collide with `decode_position` | `encoding/spatial.py` + `decoding/posterior.py:365-399` | crit | High #3 |
| C3 | Two `is_object_vector_cell` with different criteria give different answers | `encoding/egocentric.py:1869-1979,388-462` | imp | Med |
| C4 | Polar space overloaded into `Environment` (`coordinate_kind` flag; radians-as-distance; NWB round-trip loss) | `environment/factories.py:980-1002`, `io/nwb/_environment.py:230-243` | crit | Med |
| C5 | `heading_from_velocity` dt unvalidated → silent NaN/180°-flip | `ops/egocentric.py:635-736` | crit | Med |
| C6 | `distance_to_reward` no NaN validation | `events/regressors.py:617-783` | crit | Med |
| C7 | Non-runnable QUICKSTART/docstring examples (crash on paste) | Theme 5 (many) | crit ×several | High #6 |
| C8 | RNG arg naming `random_state` vs `rng` | `decoding/assemblies.py:374`, `ops/basis.py` | imp | Med |
| C9 | `window` means 3 different things across events regressors | `events/regressors.py:211-352` | imp | Med |
| C10 | `is_*_cell` classifiers swallow `ValueError` → silent `False` | `encoding/directional.py:1884`, `phase_precession.py:399`, `egocentric.py:1955`, `view.py:1541` | imp | Med |
| C11 | Result-object parity: bare dataclasses lack `.plot()`/`.compare()`/`.to_dataframe()` | `encoding/spatial.py:1712` (`DirectionalPlaceFields`) | imp | Med |
| C12 | Keyword-only `*` / env-first arg-order drift | Theme 7 (many) | imp ×several | Low |

---

## Phased implementation plan

Ordered by scientific blast-radius, then by leverage. Check off as you go.

### Phase 0 — Stop shipping wrong science (correctness criticals, silent, no error)
The highest priority: these produce plausible-but-wrong published numbers.
- [ ] **[BOTH] C1** Reactivation stats: implement REV as role-swapped (Kudrimoti) reversed EV; `ev = partial_corr**2` with a control; un-double-z-score `reactivation_strength` — `decoding/assemblies.py:959-972,1096-1115`.
- [ ] **[BUG]** `occupancy(time_allocation='linear')` full-grid vs active-bin index mismatch (corrupts firing-rate denominators on `from_samples` grids) — `environment/trajectory.py:1116-1208`.
- [ ] **[BUG]** Simulation `PlaceCellModel`: anisotropic double-width-normalization + `width=0` NaN (corrupts the ground truth other tests depend on) — `simulation/models/place_cells.py:316-335,193-201`.
- [ ] **[BUG]** Rayleigh p-value uses firing rate (Hz) as sample size — `encoding/directional.py:649-662`; weighted-circular core (length/sign/NaN-cofilter), one fix clears several — `stats/circular.py:218-237,562-607`.
- [ ] **[BUG]** `-1` out-of-bounds bin wraps to last bin in region/Voronoi indexing — `behavior/segmentation.py:295,477-478,1313-1314`; `decisions.py:767`.
- [ ] **[BUG]** `events_to_intervals(match_by=)` Cartesian product on non-unique keys — `events/intervals.py:343-375`.
- [ ] **[BUG]** Remaining Theme-1 silent-corruption criticals: `resample_field` diffuse NaN (`ops/binning.py:785-804`), `credible_region` NaN-in-HPD (`decoding/estimates.py:342-369`), NaN headings→bin 0 (`encoding/_directional_binning.py:183-185`), `generate_poisson_spikes` non-ascending timestamps (`simulation/spikes.py:128-147`), `_rle_to_mask` bounds (`regions/io.py:212-250`), `MaskedGridLayout` dtype (`layout/engines/masked_grid.py:89-119`), alpha-shape non-polygon (`annotation/_boundary_inference.py:289-310`), napari int-dtype truncation (`animation/transforms.py:320-323`). → SUMMARY Theme 1.

### Phase 1 — Numerical-robustness guards (cheap, broad)
- [ ] **[BOTH] C5** `dt>0`/finite/sorted validation in `heading_from_velocity` — `ops/egocentric.py:635-736`; also behavior/simulation rate & MSD divisions — `behavior/navigation.py:1853`, `behavior/trajectory.py:617`.
- [ ] **[BOTH] C6** `distance_to_reward` NaN/Inf guards — `events/regressors.py:617-783`.
- [ ] **[BUG]** `timestamps_from_series` rate≤0 → NaN/Inf (`io/nwb/_adapters.py:57-70`); int32 overflow in `get_n_bins` (`layout/helpers/utils.py:121`).
- [ ] **[BUG]** Length-agreement checks in `read_position`/`read_head_direction`/`read_pose` — `io/nwb/_behavior.py`, `_pose.py`.

### Phase 2 — Public-API crashes + the missing spike seam ★
Where the bug list and the #1 design gap meet — the busiest workflow seam.
- [ ] **[DESIGN] Add `bin_spikes_in_time(spike_trains, dt, t_start=None, t_stop=None, *, orient='time_x_neuron'|'neuron_x_time')`** exported from `neurospatial`/`.decoding`. Repairs the seam in 4/5 journeys AND defuses the silent `decode_position`↔assembly transpose by making the axis explicit. *(DESIGN High #1.)*
- [ ] **[DESIGN] Add `read_units`/`read_spikes`** to `io/nwb` (mirror `read_position`'s tuple contract; handle ragged `units` table); add to `__all__`/`_LAZY_IMPORTS`. *(DESIGN High #2.)*
- [ ] **[BOTH] C2** Reconcile `min_occupancy`/NaN: `fill_value=0.0` on `compute_spatial_rate(s)` OR `decode_position` treats NaN bins as zero-rate (one-time warning); also add the missing `encoding_models` n_bins↔`env.n_bins` check — `decoding/posterior.py:365-399`. *(DESIGN High #3.)*
- [ ] **[BUG]** `to_file()` crashes on Graph/Polygon layouts (make `layout_parameters` JSON-safe) — `io/files.py:222`; `interpolate(mode='linear')` reshape crash on holed grids — `environment/fields.py:555-557`.

### Phase 3 — Statistical validity
- [ ] **[BUG]** Phase-precession slope-independent p-value — `encoding/phase_precession.py:318-352`.
- [ ] **[BUG]** `compute_shuffle_pvalue` NaN undercounting biases p→significance — `stats/shuffle.py:1000-1018`; `shuffle_cell_identity` neuron-count check — `stats/shuffle.py:306-313`.

### Phase 4 — Ergonomics & composability (design)
- [ ] **[BOTH] C11** Single result-object contract (`ResultMixin`: `.to_dataframe()`/`.plot()`/`.summary()`); backfill bare dataclasses, give `DirectionalPlaceFields` `.plot()`/`.correlation()` — `encoding/spatial.py:1712`. *(DESIGN Med.)*
- [ ] **[DESIGN]** Lap/Run → `direction_labels` bridge + first-class `running_direction_labels(...) → {inbound,outbound,other}`; cross-link from `detect_laps`/`detect_runs_between_regions`. *(DESIGN Med — fixes the linear-track journey dead-end.)*
- [ ] **[BOTH] C4** Promote `from_polar_egocentric` to a distinct `EgocentricPolarEnvironment` (retire the `coordinate_kind` flag), fix radians-as-edge-distance, persist `coordinate_kind` through NWB. *(DESIGN Med + Theme 4 criticals.)*
- [ ] **[BOTH] C3** Make free `is_object_vector_cell` delegate to the result method so they cannot disagree. *(DESIGN Med.)*
- [ ] **[DESIGN]** `__getattr__` lazy submodule access at top level → `ns.encoding.<TAB>` autocompletes (mirror `io/nwb`). *(DESIGN High #5.)*
- [ ] **[DESIGN]** `to_xarray()` on array-shaped results (`DecodingResult` `(time,bin)`, `SpatialRatesResult` `(neuron,bin)`); `DecodingResult.error_against(true_times, true_positions)` to internalize decode→error alignment. *(DESIGN Med.)*
- [ ] **[DESIGN]** `environment_from_position` returns `(env, positions, timestamps)` to avoid a redundant second NWB read. *(DESIGN Low.)*

### Phase 5 — Consistency / naming / conventions (batch once behavior is correct)
- [ ] **[BOTH] C8** `random_state`→`rng` (deprecate alias) in `detect_assemblies` + 5 `ops.basis` fns.
- [ ] **[BOTH] C9** One `window` representation (recommend `(start,end)`) across `time_to_nearest_event`/`event_count_in_window`/`event_indicator`; consistent keyword-only placement — `events/regressors.py`.
- [ ] **[BOTH] C12** Keyword-only `*` separators, env-first ordering, re-export egocentric primitives from `encoding` — SUMMARY Theme 7.
- [ ] **[BOTH] C10** `validate_neural_inputs(...)` + stop `is_*_cell` swallowing `ValueError` (validate outside the try) — SUMMARY Theme 1 (classifiers).
- [ ] **[DESIGN]** `detect_<celltype>s()` naming parity across `…RatesResult` (alias `classify`); `env.space_kind` accessor + `list_factories()`. *(DESIGN Med/Low.)*

### Phase 6 — Docs & examples runnable + CI-gated
- [ ] **[BOTH] C7** Fix crashing examples: `PositionOverlay data=`→`positions=` (+ required `times=`), `BodypartOverlay` raw-list→`Skeleton`, `event_indicator` scalar `window`, OVC batch-vs-single methods, `circular_basis_metrics` kwargs — SUMMARY Theme 5.
- [ ] **[BUG]** Stale module paths / renamed methods / rename-corrupted prose (`posterior_entropy`), See-Also targets — SUMMARY Theme 5.
- [ ] **[DESIGN]** **Execute QUICKSTART/example blocks in CI** so paste-and-crash can't recur; ship one worked end-to-end `NWB → read_units → bin_spikes_in_time → fields → decode → animate` example covering the seams together. *(DESIGN High #6 + Low.)*

### Phase 7 — Domain capability gaps (roadmap / optional)
- [ ] **[DESIGN]** `events.detect_ripples(lfp, sampling_rate, *, band=(150,250)) → intervals` (or explicitly scope out + point to `ripple_detection`). Replay framing currently depends on ripple events the library can't produce. *(DESIGN Med.)*
- [ ] **[DESIGN]** `encoding.theta_phase(lfp, sampling_rate, *, band=(6,10))` to make the phase-precession path self-contained; docstring note distinguishing movement-heading from head-direction. *(DESIGN Low.)*

### Phase 8 — Test backfill (locks in the fixes)
- [ ] **[BUG]** Holed-grid occupancy mass-conservation; IDW `map_probabilities`; polar + Graph/Polygon NWB round-trip; negative-weight circular stats; HD-cell recovery; REV semantics; `edge_order` staleness; sinusoidal trajectory. → SUMMARY Theme 6.

---

## Keep — good design that a refactor must not regress

From DESIGN-REVIEW.md "What's working":
- `Environment` as the single central object built from focused mixins; factory-only construction with required `bin_size`/`pixel_size` + `@check_fitted`.
- The bin/node/graph model unifying grids/hex/tracks/polygons/meshes behind one `LayoutEngine` with "active bins" as the primitive.
- **Load-bearing array contracts**: `compute_spatial_rates(...).firing_rates` *is* `decode_position`'s `encoding_models` (even on masked grids); `DecodingResult.posterior` feeds `animate_fields` directly — must not regress.
- Within-family encoding result parity (`SpatialResultMixin`) — the model others should converge toward (Phase 4).
- The single doctested allocentric/egocentric angle convention; field-idiomatic vocabulary with primary-literature citations.
- The PSTH path (`peri_event_histogram` → `PeriEventResult`) and `open_field_session` simulation — glue-free starting points.
- The `E1001` "no active bins" diagnostic — a model of good first-run guidance.

---

## Sources
- [SUMMARY.md](SUMMARY.md) — correctness (per-module bug review, 43 criticals) and its 5 batch reports.
- [DESIGN-REVIEW.md](DESIGN-REVIEW.md) — API/design (journeys + 7 axes + prioritized recs).
