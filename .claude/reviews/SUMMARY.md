# neurospatial — Consolidated Review (13 modules)

Synthesis of five module-review reports covering all 13 modules. This is a working fix-tracking document.

## Tally

| Batch (modules) | Critical | Important | Suggestion | Total |
|---|---|---|---|---|
| stats | 4 | 11 | 16 | 31 |
| encoding, decoding, ops | 9 | 36 | 35 | 80 |
| environment, behavior, events | 13 | 42 | 28 | 83 |
| io, layout, regions | 6 | 35 | 36 | 77 |
| animation, annotation, simulation | 11 | 38 | 27 | 76 |
| **TOTAL** | **43** | **162** | **142** | **347** |

> Counts are taken verbatim from each report's Executive Summary. Note: the `stats` Executive Summary tallies 4 criticals, but its `### Critical` section enumerates only 3 distinct bullets (the 4th — "weights misalign with angles after NaN removal" — is split across the two listed `rayleigh_test` NaN/weight criticals); the master list below therefore reflects 37 enumerated critical bullets across all reports against a headline total of 43. See the report-back note at the end.

---

## Critical issues (must fix)

Master list of every critical finding enumerated in the five reports' `### Critical` sections, grouped by cross-cutting theme.

### Theme 1 — Silent corruption over loud failure

- [ ] **[stats]** Weighted circular functions never validate `len(weights) == len(angles)`; length-1 weights silently broadcast to out-of-range R/z — `src/neurospatial/stats/circular.py:218-237`. Add a shared `_validate_weights(angles, weights)` (equal shape + non-negativity) used by every weighted entry point.
- [ ] **[stats]** `rayleigh_test` drops NaN angles but not paired weights, scrambling alignment / inflating `n_eff` — `src/neurospatial/stats/circular.py:562-594`. Have the validator co-filter weights with the same NaN mask and compute `n_eff` from filtered weights.
- [ ] **[stats]** `rayleigh_test` with all-zero weights raises bare `ZeroDivisionError` instead of NaN/clear error — `src/neurospatial/stats/circular.py:587-607`. Guard `n_eff <= 0` and return NaN / raise an actionable error.
- [ ] **[encoding]** NaN headings silently miscounted into direction bin 0, corrupting HD tuning curves — `src/neurospatial/encoding/_directional_binning.py:183-185`. Reject/mask non-finite headings out of both occupancy and spike counts.
- [ ] **[decoding]** `decode_position`/`log_poisson_likelihood` never check `encoding_models` bin count vs `env.n_bins` → silent wrong positions — `src/neurospatial/decoding/posterior.py:365-399`. Assert `encoding_models.shape[1] == env.n_bins` (and neuron-axis match) with a clear ValueError.
- [ ] **[decoding]** `credible_region` silently includes NaN bin as highest-density in HPD region — `src/neurospatial/decoding/estimates.py:342-369`. Detect NaN rows up front (raise/return empty) or mask NaN to -inf before argsort.
- [ ] **[ops]** `resample_field(method="diffuse")` propagates a single out-of-source NaN across the whole output, zeroing the field — `src/neurospatial/ops/binning.py:785-804`. Zero-fill outside-source NaNs before `apply_kernel`, re-impose NaN on genuinely-outside bins afterward.
- [ ] **[ops]** `heading_from_velocity` divides by dt with no validation; dt<=0/unsorted times silently give NaN or 180°-wrong headings — `src/neurospatial/ops/egocentric.py:635-736`. Add `if not np.isfinite(dt) or dt <= 0: raise ValueError(...)`.
- [ ] **[environment]** `occupancy(time_allocation='linear')` writes full-grid flat indices into the active-bin-indexed occupancy array, corrupting occupancy on masked grids — `src/neurospatial/environment/trajectory.py:1116-1208,1312-1359`. Map full-grid indices through `active_mask` before accumulating.
- [ ] **[behavior]** Out-of-bounds bins (-1) silently wrap to the last bin when indexing region masks/Voronoi labels — `src/neurospatial/behavior/segmentation.py:295,477-478,1313-1314`. Mask `(position_bins >= 0) & (position_bins < n_bins)` before gathering.
- [ ] **[events]** `distance_to_reward` silently produces wrong distances when `reward_times`/positions contain NaN — `src/neurospatial/events/regressors.py:617-783`. Add the NaN/Inf guards used by `time_to_nearest_event`.
- [ ] **[events]** `events_to_intervals(match_by=...)` silently produces a Cartesian product when key values are non-unique — `src/neurospatial/events/intervals.py:343-375`. Validate per-key uniqueness (or merge with `validate='one_to_one'`) and raise.
- [ ] **[io]** NWB round-trip silently drops `coordinate_kind`, flipping polar environments to Cartesian — `src/neurospatial/io/nwb/_environment.py:230-243,517-591`. Persist `coordinate_kind` in metadata JSON and restore on read.
- [ ] **[regions]** `_rle_to_mask` silently produces a wrong mask on out-of-bounds/negative RLE indices (docstring promises a ValueError that never fires) — `src/neurospatial/regions/io.py:212-250`. Validate each `(start, length)` and use `zip(..., strict=True)`.
- [ ] **[layout]** `MaskedGridLayout.build` does not validate `active_mask` dtype, allowing silent corruption from int/float masks — `src/neurospatial/layout/engines/masked_grid.py:89-119`. Raise `ValueError` if dtype is not `np.bool_`.
- [ ] **[animation]** Napari coordinate transform truncates float results to int when overlay coords are integer dtype, misplacing overlays — `src/neurospatial/animation/transforms.py:320-323`. Allocate `np.empty(coords.shape, dtype=np.float64)` (also lines 377/388).
- [ ] **[annotation]** Alpha-shape boundary inference silently returns a non-Polygon/empty geometry, producing a degenerate boundary — `src/neurospatial/annotation/_boundary_inference.py:289-310`. Validate result and raise `ValueError` when empty/non-polygonal.
- [ ] **[simulation]** `generate_poisson_spikes` silently produces wrong spike trains for non-uniform/non-ascending timestamps — `src/neurospatial/simulation/spikes.py:128-147`. Validate equal length, strictly increasing times, and uniform dt (or use per-bin `np.diff`).

### Theme 2 — Active-bin vs. full-grid index confusion on masked grids

- [ ] **[environment]** `occupancy(time_allocation='linear')` full-grid vs active-bin index mismatch — `src/neurospatial/environment/trajectory.py:1116-1208,1312-1359`. *(Also Theme 1.)*
- [ ] **[layout]** `MaskedGridLayout.build` dtype non-validation silently mis-sizes active bins — `src/neurospatial/layout/engines/masked_grid.py:89-119`. *(Also Theme 1.)*
- [ ] **[behavior]** `-1` bin wraparound to last bin in region/Voronoi indexing — `src/neurospatial/behavior/segmentation.py:295,477-478,1313-1314`. *(Also Theme 1.)*

### Theme 3 — Circular quantities handled linearly across the ±π wrap

- [ ] **[encoding]** NaN headings folded into direction bin 0 (failure of wrap/NaN handling on a circular variable) — `src/neurospatial/encoding/_directional_binning.py:183-185`. *(Also Theme 1.)*

*(No additional purely-critical findings sit only in this theme; the important-tier circular findings — HeadDirectionOverlay ±π interpolation, OVC angle-convention mixing, antipode -π — appear under Cross-cutting themes below.)*

### Theme 4 — Units / reference-frame / axis-order mixing

- [ ] **[environment]** Polar egocentric env stores raw radians as edge 'distance' (cm/rad unit mixing) — `src/neurospatial/environment/factories.py:980-1002`. Recompute edge distances as proper polar lengths (`r·Δθ`, `Δr`, diagonal) and fix `bin_sizes`, or block `metric='geodesic'` on polar envs.
- [ ] **[layout]** `ImageMaskLayout` axis-order mismatch: `bin_centers`/`grid_edges` are (y,x) but `dimension_ranges` is (x,y) — `src/neurospatial/layout/engines/image_mask.py:130-157`. Build `grid_edges`/`bin_centers` in (x,y) order to match other grid engines.

### Theme 5 — Documentation / example drift (non-runnable examples)

- [ ] **[animation]** QUICKSTART overlay examples use removed `PositionOverlay(data=...)` kwarg → TypeError — `.claude/QUICKSTART.md:674-693`. Replace `data=` with `positions=`.
- [ ] **[animation]** QUICKSTART `BodypartOverlay` passes a raw list as `skeleton` → AttributeError — `.claude/QUICKSTART.md:683-688`. Construct a `Skeleton` first.
- [ ] **[animation]** `animation_overlays.md` BodypartOverlay examples use removed `skeleton=[...]`/`skeleton_color`/`skeleton_width` — `docs/animation_overlays.md:105-126`. Build a `Skeleton(...)` and drop removed kwargs.

### Theme 6 — Untested validation / failure branches

- [ ] **[ops]** Inverse-distance-weighted mode of `map_probabilities` is completely untested — `src/neurospatial/ops/alignment.py:388-453`. Add tests for hand-computed splits, mass conservation, `n_neighbors` clamp, and `k_eff==1`.
- [ ] **[io]** No `to_file`/`from_file` round-trip test for Graph or Polygon layouts — both crash on save — `tests/io_tests/test_io.py:32-133`. Add a parametrized round-trip over all layout factories and make `layout_parameters` JSON-safe.
- [ ] **[annotation]** No regression test for `edge_order`/`edge_spacing` override staleness after `delete_node`/`delete_edge` → silently wrong linearization — `src/neurospatial/annotation/_track_state.py:205-249`. Reindex/clear overrides on delete and add tests.

### Theme 7 — API-convention drift

*(No criticals are purely API-convention; the convention-drift findings are all important-tier and listed under Cross-cutting themes below — e.g. `decoding` `random_state`-vs-`rng`, keyword-only `*` separators.)*

### Other (no single cross-cutting theme)

- [ ] **[encoding]** Rayleigh p-value uses firing rate (Hz) as the test's sample size, making p-values invalid and scale-dependent — `src/neurospatial/encoding/directional.py:649-662`. Plumb per-bin spike counts into the test and weight by counts. *(Borderline Theme 1 silent-invalid-result.)*
- [ ] **[decoding]** `reactivation_strength` structurally blind to reactivation magnitude due to double z-scoring — `src/neurospatial/decoding/assemblies.py:959-972`. Add a `z_score_output=False` path (or squared raw projections) and test that larger match activity yields strength > 1.
- [ ] **[decoding]** Reversed EV (REV) hardcoded equal to EV, making the EV>REV reactivation control meaningless — `src/neurospatial/decoding/assemblies.py:1096-1099`. Implement REV as the role-swapped partial correlation (Kudrimoti).
- [ ] **[decoding]** Explained variance ignores `control_correlations` even when provided — `src/neurospatial/decoding/assemblies.py:1096-1115`. Set `ev = partial_corr**2` when a control is supplied.
- [ ] **[simulation]** Anisotropic place field divides by width twice, producing a near-flat (wrong) Gaussian — `src/neurospatial/simulation/models/place_cells.py:316-335`. Keep anisotropic distance in sigma units; evaluate `exp(-0.5*d_sigma**2)` with unit divisor; add a 1-sigma `exp(-0.5)` regression test.
- [ ] **[simulation]** `PlaceCellModel.firing_rate` yields NaN at field center when width=0, silently producing zero spikes — `src/neurospatial/simulation/models/place_cells.py:193-201,321,334`. Reject non-positive width at construction.

---

## Cross-cutting themes

Each theme lists the most consequential critical AND important findings that fit it. Items marked *(repeat)* are cross-listed.

### 1. Silent corruption over loud failure

The dominant pattern across all 13 modules: invalid input (NaN/Inf, wrong dtype, non-unique keys, out-of-range indices) flows through to plausible-but-wrong scientific output with no error, instead of failing loudly. Includes NaN selected as argmax/max, broad `except Exception → degrade silently` fallbacks, and silent coercion.

- [stats] Weighted-circular NaN/length/zero-weight handling — `circular.py:218-237,562-594,587-607` *(repeat, critical)*
- [stats] `compute_shuffle_pvalue` undercounts NaN null scores, biasing p toward significance — `src/neurospatial/stats/shuffle.py:1000-1018`
- [stats] `shuffle_cell_identity` does not check neuron-count match — `src/neurospatial/stats/shuffle.py:306-313`
- [encoding] NaN headings → bin 0 — `_directional_binning.py:183-185` *(repeat, critical)*
- [encoding] `tuning_width` returns NaN when a NaN bin is adjacent to the half-max crossing — `directional.py:522-547`
- [encoding] `is_head_direction_cell`/`has_phase_precession`/`is_object_vector_cell`/`is_spatial_view_cell` swallow ValueError and return False — `directional.py:1884-1895`, `phase_precession.py:399-407`, `egocentric.py:1955-1979`, `view.py:1541-1555`
- [encoding] Geodesic field distance silently falls back to Euclidean on any exception — `_field_metrics.py:900-914`
- [decoding] `credible_region` includes NaN bin in HPD — `estimates.py:342-369` *(repeat, critical)*
- [decoding] `confusion_matrix` mis-handles NaN posterior rows — `metrics.py:338-348`
- [decoding] `log_poisson_likelihood`/`poisson_likelihood` silently accept 1D spike_counts and collapse the time axis — `likelihood.py:118-181`
- [decoding] `decode_position` stores mismatched-length `times` without validation — `posterior.py:394-399`
- [decoding] `decoding_error(metric='geodesic')` silently falls back to Euclidean for out-of-env positions — `metrics.py:162-172`
- [ops] `resample_field` diffuse NaN propagation — `binning.py:785-804` *(repeat, critical)*
- [ops] `heading_from_velocity` dt<=0 → silent NaN/180°-wrong — `egocentric.py:635-736` *(repeat, critical)*
- [ops] `map_probabilities` returns all-zero probability on any KDTree failure (broad except) — `alignment.py:374-382,424-432,578-586`
- [ops] `compute_egocentric_distance` does not validate 3D targets' time axis matches `n_time` — `egocentric.py:575-592`
- [ops] `_estimate_spectral_radius` broad-except → silent max-degree fallback — `basis.py:988-1002`
- [environment] `occupancy` linear full-grid index mismatch — `trajectory.py:1116-1208` *(repeat, critical)*
- [environment] `interpolate(mode='linear')` reshapes active-bin field by full grid_shape — `fields.py:555-557`
- [environment] NaN/non-finite timestamps produce a self-contradictory monotonicity error — `trajectory.py:228-235,541-548`
- [behavior] `-1` bin wraparound in region/Voronoi indexing — `segmentation.py:295,477-478,1313-1314` *(repeat, critical)*
- [behavior] `detect_boundary_crossings` indexes voronoi_labels with raw -1 bins — `decisions.py:767`
- [behavior] `compute_path_efficiency` treats +inf geodesic traveled_length as valid → reports 0.0 — `navigation.py:1569-1592`
- [behavior] Velocity/rate divides by dt with no guard against zero/negative time deltas — `segmentation.py:533,673-674`
- [events] `distance_to_reward` NaN corruption — `regressors.py:617-783` *(repeat, critical)*
- [events] `events_to_intervals(match_by=...)` Cartesian product on non-unique keys — `intervals.py:343-375` *(repeat, critical)*
- [events] `add_positions` silently returns all-NaN/wrong positions for degenerate trajectories — `detection.py:141-162`
- [io] NWB drops `coordinate_kind` — `_environment.py:230-243` *(repeat, critical)*
- [io] `timestamps_from_series` emits NaN/Inf timestamps when rate <= 0 — `_adapters.py:57-70`
- [io] `read_position`/`read_head_direction`/`read_pose` return data with no length-agreement check — `_behavior.py:85-88,298-299`, `_pose.py:82-96`
- [layout] `_points_to_regular_grid_bin_ind` returns all -1 on dimensionality mismatch instead of raising — `regular_grid.py:638-648`
- [layout] Silent int32 overflow in `get_n_bins` — `utils.py:121`
- [regions] `_rle_to_mask` out-of-bounds/negative RLE — `io.py:212-250` *(repeat, critical)*
- [regions] `region_center` raises IndexError on empty polygon instead of returning None — `core.py:589-591`
- [regions] Broad `except (ValueError, Exception)` in CVAT processors swallows errors, silently drops shapes — `io.py:435-442`
- [animation] Napari int-dtype overlay truncation — `transforms.py:320-323` *(repeat, critical)*
- [animation] `_validate_bounds` warning prints NaN data ranges (use nanmin/nanmax) — `overlays.py:3389-3390`
- [annotation] Alpha-shape non-Polygon degenerate boundary — `_boundary_inference.py:289-310` *(repeat, critical)*
- [annotation] `validate_region_overlap` crashes on invalid polygons (violates "warn, never raise") — `validation.py:262-266`
- [annotation] `shapes_to_regions` silently drops <3-vertex polygons; `annotate_track_graph` silently discards invalid `initial_edges` — `converters.py:112-114`, `track_graph.py:208-210`
- [simulation] `generate_poisson_spikes` non-ascending/non-uniform timestamps — `spikes.py:128-147` *(repeat, critical)*
- [simulation] `generate_poisson_spikes` silently swallows NaN/Inf firing rates — `spikes.py:141-147`
- [simulation] `PlaceCellModel.firing_rate` width=0 NaN — `place_cells.py:193-201` *(repeat, critical)*

### 2. Active-bin vs. full-grid index confusion on masked grids

Any environment with inactive bins (the common `from_samples` case) is mis-indexed or crashes because code mixes full-grid flat indices with active-bin-indexed arrays.

- [environment] `occupancy(time_allocation='linear')` — `trajectory.py:1116-1208,1312-1359` *(repeat, critical)*
- [environment] `interpolate(mode='linear')` reshapes by full grid_shape — `fields.py:555-557`
- [environment] `rebin()` infers diagonal connectivity from a single node's degree using a full-grid index against an active-bin graph — `transforms.py:258-276`
- [layout] `MaskedGridLayout.build` dtype non-validation — `masked_grid.py:89-119` *(repeat, critical)*
- [layout] `GraphLayout.linear_point_to_bin_ind` returns full-grid (gap-inclusive) indices, not active-bin — `engines/graph.py:365-395`
- [behavior] `-1` bin wraparound — `segmentation.py:295,477-478,1313-1314`; `decisions.py:767,904` *(repeat, critical)*
- [io] `read_environment` Hexagonal goes through KDTree fallback, not MaskedGrid reconstruction (docstring overstates fidelity) — `_environment.py:476-482`

### 3. Circular quantities handled linearly across the ±π wrap

Angular variables (headings, phases, bearings) are interpolated, binned, or differenced as if linear, breaking at the ±π discontinuity.

- [encoding] NaN headings folded into bin 0 (circular binning) — `_directional_binning.py:183-185` *(repeat, critical)*
- [animation] `HeadDirectionOverlay` linearly interpolates circular angles across ±π — `overlays.py:672-680,639,3856-3862`. Interpolate on the circle (unit vectors) or unwrap first.
- [stats] `plot_circular_basis_tuning` compass polar convention disagrees with math-convention phase — `circular.py:1680-1682` *(also Theme 4)*
- [simulation] Object-vector angle-convention mixing (egocentric `preferred_direction` vs allocentric `headings`) without cross-reference — `models/object_vector_cells.py:351-367`
- [ops] `_wrap_angle`/`compute_egocentric_bearing` returns -π at the antipode, contradicting documented `(-π, π]` — `egocentric.py:490`

### 4. Units / reference-frame / axis-order mixing

cm vs rad, deg vs rad, `coordinate_kind` loss, and (y,x) vs (x,y) axis order silently mixed.

- [environment] Polar egocentric env stores radians as edge distance (cm/rad) — `factories.py:980-1002` *(repeat, critical)*
- [layout] `ImageMaskLayout` (y,x) vs (x,y) axis-order mismatch — `image_mask.py:130-157` *(repeat, critical)*
- [io] NWB drops `coordinate_kind` (polar→Cartesian) — `_environment.py:230-243` *(repeat, critical)*
- [io] `read_head_direction` does not enforce allocentric (0=East) convention, ignores deg/rad unit attribute, and `.ravel()` corrupts vector-form data — `_behavior.py:242-301,297-301,298`
- [encoding] `gaussian_kde` on egocentric polar env mixes cm and radians in one scalar bandwidth — `egocentric.py:1098-1108`
- [decoding] Slope-to-speed conversion references nonexistent `env.bin_size` and conflates bin area with width; only valid for 1D tracks but documented "for regular grids" — `trajectory.py:135,424,419-424`
- [behavior] `time_efficiency` uses Euclidean optimal distance even when result is geodesic — `navigation.py:1357-1367,1581-1585`
- [behavior] `heading_direction_labels` default `min_speed=5.0` silently assumes cm/s with no unit docs — `navigation.py:1083-1104`
- [stats] `plot_circular_basis_tuning` compass vs math convention — `circular.py:1680-1682` *(repeat, Theme 3)*
- [regions] `point_tolerance` default 1e-8 is a float-equality epsilon applied to physical cm/pixel coords → membership all-False — `ops.py:192-359`
- [ops] `compute_viewshed` silently caps ray distance at a hardcoded 200 spatial units — `visibility.py:830`

### 5. Documentation / example drift

Non-runnable QUICKSTART/docstring examples, renamed kwargs, stale module paths. High volume, low blast radius — batch together.

- [animation] Removed `PositionOverlay(data=...)`, raw-list `skeleton=` examples — `.claude/QUICKSTART.md:674-693`, `docs/animation_overlays.md:105-126` *(repeat, critical ×3)*
- [behavior] `path_progress`/`cost_to_goal`/`compute_vte_session` wrong arg order/arg collisions → TypeError — `navigation.py:538-554,796-798`, `vte.py:44-56,638-649`, `.claude/QUICKSTART.md:267-273`
- [events] `event_indicator` examples pass a tuple `window` but param is a scalar half-width → TypeError — `events/__init__.py:58-64`, `.claude/QUICKSTART.md:863-866`
- [stats] QUICKSTART `circular_basis_metrics` example uses non-existent `var_sin`/`var_cos`/`cov_sin_cos` args → TypeError — `.claude/QUICKSTART.md:797-802`
- [stats] Stale module path `neurospatial.metrics.phase_precession` (should be `neurospatial.encoding.phase_precession`) — `circular.py:31-34`
- [encoding] QUICKSTART references non-existent `peak_view_locations()` (renamed singular) — `.claude/QUICKSTART.md:564,583`; API_REFERENCE lists `DirectionalRateResult.mrl()` which is `mean_vector_length()` — `.claude/API_REFERENCE.md:346`; OVC block calls single-neuron methods on a batch result — `.claude/QUICKSTART.md:459-468`
- [decoding] Global-rename doc rot: prose "uncertainty"/"entropy" replaced by identifier `posterior_entropy` — `trajectory.py:12,102,373,386,390,392`, `estimates.py:203,205`
- [ops] `graph_convolve` docstring imports/calls a non-existent `convolve`; `basis.py` examples call `env.bin_sequence` with reversed args and `env.plot()` instead of `plot_field()` — `graph.py:360-381`, `basis.py:55,69,87,419,1053`
- [environment] See-Also paths to non-existent modules (`neurospatial.differential`, `neurospatial.reference_frames`); `environments.md` uses `close=` instead of `close_gaps` — `core.py:1099`, `factories.py:944`, `docs/user-guide/environments.md:43`
- [behavior] User-guide imports from non-existent `neurospatial.segmentation` — `docs/user-guide/trajectory-and-behavioral-analysis.md:236,...,729`
- [io] `write_trials` points to non-existent `write_intervals()`; `read_environment` Hexagonal claim wrong; ADVANCED.md `events_name` kwarg wrong — `_events.py:608,780`, `_environment.py:476-482`, `.claude/ADVANCED.md:50`
- [regions] `plot_regions` docstring/example contradict signature; `ops.py` example uses `transform=` instead of `pixel_to_world=`; `core-concepts.md` uses `.type`; `PATTERNS.md` claims overwrite warns but it raises KeyError — `plot.py:44-66`, `ops.py:37`, `docs/getting-started/core-concepts.md:292`, `.claude/PATTERNS.md:225,247,250`
- [annotation] `from neurospatial.annotation import Role` ImportError (symbol is `RegionType`); `annotate_video` docstring states wrong `boundary_config` default — `docs/user-guide/video-annotation.md:263-271`, `core.py:100-102`
- [simulation] Wrong documented `baseline_rate` defaults across `simulate_session`/`open_field_session`; coverage docstrings describe old algorithms — `session.py:165-169,193,205,213`, `examples.py:44`

### 6. Untested validation / failure branches

Bugs survived because failure/validation branches and non-dense-grid paths have no coverage; tests exercise the happy path on dense grids only.

- [ops] IDW mode of `map_probabilities` completely untested — `alignment.py:388-453` *(repeat, critical)*
- [io] No Graph/Polygon round-trip test (both crash on save); files.py round-trip never tests active_mask/grid_edges/is_linearized_track; no polar `coordinate_kind` round-trip test — `tests/io_tests/test_io.py:32-133`, `_environment.py:230-243` *(repeat, critical)*
- [annotation] No edge_order/edge_spacing override-staleness test; mixed override case untested — `_track_state.py:205-249`, `tests/annotation/test_edge_order.py:209-262` *(repeat, critical)*
- [environment] Linear-allocation occupancy for an exiting trajectory only smoke-tested (`sum() >= 0`) — `tests/environment/test_linear_occupancy.py:279-295`
- [stats] No regression test for negative weights in weighted circular stats — `tests/stats/test_circular_metrics.py:233-285`
- [encoding] No end-to-end HD-cell recovery test for `compute_directional_rate` — `tests/encoding/test_encoding_directional.py:2740-2906`
- [decoding] `detect_assemblies` crashes (IndexError) when `n_time_bins < n_components ≤ n_neurons`; REV directional semantics never tested — `tests/decoding/test_assemblies.py:259-265,471-479`
- [behavior] `approach_rate` geodesic branch entirely untested — `navigation.py:1835-1851`
- [events] Duplicate `match_by` and `distance_to_reward` selected-reward magnitudes untested — `tests/events/test_intervals.py:335-358`, `tests/events/test_regressors.py:931-998`
- [layout] Public validators `validate_bin_size`/`validate_dimension_ranges` have zero coverage; N-D boundary-trimming untested — `validation.py:44-233`, `regular_grid.py:266-273`
- [ops] `visible_cues` never asserts numerical bearings; `normalize_field`/`combine_fields` validation paths untested — `tests/ops/test_visibility.py:605-632`, `tests/ops/test_ops_normalize.py:11-123`
- [animation] `field_to_rgb_for_napari` orientation never verified; reuse-artists video path never renders events — `tests/animation/test_rendering.py:104-161`, `_parallel.py:1066-1104`
- [simulation] `simulate_trajectory_sinusoidal` has zero behavioral coverage; velocity-autocorrelation test asserts on an ignored parameter — `tests/simulation/test_trajectory_sim.py:203-216,45-81`

### 7. API-convention drift

RNG naming (`random_state` vs `rng`), keyword-only `*` separator, env-first ordering, and result-class parity each violated in several places.

- [decoding] `detect_assemblies` uses `random_state` while everything else uses `rng` — `assemblies.py:374`
- [decoding] Same `{'map','expected'}` choice is `summary_method` here but `method` there; `decoding_error` places `env` 3rd positional (breaks env-first) — `metrics.py:233,29-35`
- [ops] `basis.py` five public functions lack keyword-only `*`; `combine_fields`/`compute_diffusion_kernels`/`distance` functions take algorithm params positionally — `basis.py:117-1010`, `normalize.py:161-165`, `smoothing.py:70-75`, `distance.py:55-59,410-414`
- [encoding] `is_object_vector_cell` free function vs result method use different criteria; `view_distance`/`gaze_model` order reversed across siblings; batch classifier names inconsistent (`classify` vs `detect_*`); mandatory-but-unused `env` args — `egocentric.py:1869-1979,388-462`, `view.py:867-869,1474-1476`, `spatial.py:1042`
- [environment] `EnvironmentProtocol.animate_fields`/`occupancy` stubs stale (removed `fps`, wrong `max_gap` default) — `_protocols.py:718-811,385`
- [behavior] `env` placement inconsistent (`compute_decision_analysis` env-first vs `compute_vte_session` env-third); `detect_goal_directed_runs` raises KeyError where siblings raise ValueError — `vte.py:593-604`, `segmentation.py:1749-1755`
- [events] `window` type/semantics inconsistent across three GLM regressors (scalar vs tuple vs `max_time`); keyword-only placement inconsistent — `regressors.py:211-215,347-352`
- [io] `description` parameter placement inconsistent across sibling NWB writers — `_events.py:211-217,419-422,1004-1008`
- [layout] `TriangularMeshLayout.build` breaks keyword-only convention and skips `@capture_build_params`; `ImageMaskLayout.build` param `bin_size` vs factory `pixel_size` — `triangular_mesh.py:77-89`, `image_mask.py:62`
- [annotation] `calibration` keyword-only placement inconsistent; `method`/`simplify_tolerance` before `*` — `io.py:13-19,72-75`, `_boundary_inference.py:74-82`, `converters.py:21-31`
- [simulation] Inconsistent `*` separator across sibling trajectory functions; `headings` placed after algorithm params in `generate_population_spikes` — `trajectory.py:62-76,460-469,598-608`, `spikes.py:173-182`
- [regions] `region_center` param `region_name` inconsistent with sibling `area(name)`; advertises a `| None` return it never produces — `core.py:542-591,562-591`

---

## Suggested fix order

Ordered by scientific blast-radius. Checkboxes track progress.

1. **Result-corrupting criticals with no error — fix first.** These silently produce wrong published numbers.
   - [ ] [decoding] Reactivation/assembly statistics cluster: REV==EV, control-period ignored, double z-scoring — `assemblies.py:959-972,1096-1099,1096-1115`. The entire documented reactivation-control workflow is currently meaningless.
   - [ ] [environment] `occupancy` linear active-bin/full-grid index mismatch — `trajectory.py:1116-1208`.
   - [ ] [behavior] `-1` bin wraparound in region/Voronoi indexing — `segmentation.py:295,477-478,1313-1314`; `decisions.py:767`.
   - [ ] [events] `match_by` Cartesian product + `distance_to_reward` NaN/sort — `intervals.py:343-375`, `regressors.py:617-783`.
   - [ ] [io] NWB `coordinate_kind` round-trip loss — `_environment.py:230-243`.
   - [ ] [layout]/[regions] `MaskedGridLayout` dtype + `_rle_to_mask` bounds — `masked_grid.py:89-119`, `regions/io.py:212-250`.
   - [ ] [simulation] PlaceCellModel anisotropic double-normalization + width=0 NaN — `place_cells.py:316-335,193-201`.
   - [ ] [encoding] Rayleigh sample-size = firing rate — `directional.py:649-662`; [stats] weighted-circular core fix — `circular.py:218-237,562-607` (one fix clears multiple criticals).

2. **Numerical-robustness guards (cheap, broad).**
   - [ ] dt>0 / finite-timestamp validation across behavior + simulation + ops — `navigation.py:1853-1857`, `trajectory.py:617-621`, `simulation/spikes.py:128-147`, `ops/egocentric.py:635-736`.
   - [ ] NaN handling in argmax/argsort/cumsum paths — `decoding/estimates.py:342-369`, `decoding/metrics.py:338-348`, `ops/binning.py:785-804`, `encoding/_directional_binning.py:183-185`.
   - [ ] int32 overflow + rate<=0 timestamps — `layout/helpers/utils.py:121`, `io/nwb/_adapters.py:57-70`.

3. **Crashes / round-trip failures on public API.**
   - [ ] Graph/Polygon `to_file` JSON crash + missing round-trip test — `io/files.py:222`, `tests/io_tests/test_io.py:32-133`.
   - [ ] `interpolate(mode='linear')` reshape crash on holed grids — `environment/fields.py:555-557`.
   - [ ] `estimate_transform` Kabsch reflection bug — `ops/transforms.py:1269-1276`.
   - [ ] Alpha-shape non-Polygon boundary — `annotation/_boundary_inference.py:289-310`; integer-dtype overlay truncation — `animation/transforms.py:320-323`.

4. **Statistical-validity.**
   - [ ] Phase-precession slope-independent p-value — `encoding/phase_precession.py:318-352`.
   - [ ] `compute_shuffle_pvalue` NaN undercounting — `stats/shuffle.py:1000-1018`.
   - [ ] Fisher-z averaging of cross-dim correlations — `decoding/metrics.py:524-529`; `trajectory_similarity` Pearson-on-bin-indices — `behavior/segmentation.py:1535-1558`.

5. **Doc / API / example drift in bulk.** Batch all of Theme 5 and Theme 7 once behavior is correct.
   - [ ] Overlay kwargs (`data=`→`positions=`, raw-list `skeleton=`) across `.claude/QUICKSTART.md`, `docs/animation_overlays.md`, PATTERNS/ADVANCED/TROUBLESHOOTING.
   - [ ] Behavior/events non-runnable examples (`path_progress`, `cost_to_goal`, `compute_vte_session`, `event_indicator` window).
   - [ ] Stale module paths, renamed methods (`peak_view_locations`, `mrl()`), rename-corrupted prose (`posterior_entropy`), See-Also targets.
   - [ ] API conventions: `random_state`→`rng`, keyword-only `*` separators, env-first ordering, result-class parity.

6. **Test-coverage backfill that locks in the fixes.**
   - [ ] Holed-grid occupancy mass-conservation — `tests/environment/test_linear_occupancy.py:279-295`.
   - [ ] IDW `map_probabilities` — `ops/alignment.py:388-453`.
   - [ ] Polar NWB round-trip + Graph/Polygon round-trip — `io_tests/test_io.py`.
   - [ ] Negative-weight circular stats, HD-cell recovery, REV semantics, edge_order staleness, sinusoidal trajectory.

---

## Per-module index

| Module | Crit / Imp / Sug | Report |
|---|---|---|
| stats | 4 / 11 / 16 | [report](module-review-stats.md) |
| encoding | 2 / 9 / 8 | [report](module-review-encoding-decoding-ops.md) |
| decoding | 5 / 13 / 8 | [report](module-review-encoding-decoding-ops.md) |
| ops | 3 / ~9 / ~15 | [report](module-review-encoding-decoding-ops.md) |
| environment | 2 / 8 / 9 | [report](module-review-environment-behavior-events.md) |
| behavior | 6 / ~18 / ~9 | [report](module-review-environment-behavior-events.md) |
| events | 2 / 11 / 7 | [report](module-review-environment-behavior-events.md) |
| io | 2 / 13 / 11 | [report](module-review-io-layout-regions.md) |
| layout | 2 / 13 / 11 | [report](module-review-io-layout-regions.md) |
| regions | 1 / 9 / 14 | [report](module-review-io-layout-regions.md) |
| animation | 4 / 12 / 8 | [report](module-review-animation-annotation-simulation.md) |
| annotation | 2 / 9 / 9 | [report](module-review-animation-annotation-simulation.md) |
| simulation | 3 / 16 / 14 | [report](module-review-animation-annotation-simulation.md) |

> Per-module important/suggestion splits are approximate where a report only published a per-batch total; batch-level critical counts (stats 4, enc/dec/ops 9, env/beh/events 13, io/layout/regions 6, anim/annot/sim 11) are exact from the Executive Summaries.
