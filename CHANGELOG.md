# Changelog

## Unreleased

### Added

- `compute_spatial_rate` and `compute_spatial_rates` gain a
  `fill_value: float | None = None` keyword. The default `None` preserves the
  existing behavior: bins masked by `min_occupancy` remain NaN, so existing
  callers see no change. Passing `fill_value=0.0` (the recommended decoding
  golden path) replaces those NaN bins with an explicit zero firing rate so the
  encoding model composes directly with `decode_position()` without manual
  `np.nan_to_num` scrubbing. `occupancy` is left untouched, so the masked-bin
  set is still recoverable via `result.occupancy < min_occupancy`.
- `decode_position` now tolerates NaN bins in `encoding_models` (e.g. a
  `fill_value=None` rate map carrying `min_occupancy` masks): each NaN
  `(neuron, bin)` is treated as a zero-rate observation and excluded from that
  neuron's contribution to the Poisson log-likelihood at that bin, and a single
  `UserWarning` is emitted per call instead of raising. A **partial-NaN bin**
  (NaN for some neurons but observed by at least one other) still decodes
  normally from its observing neurons — excluded, not penalized. A bin that is
  NaN for **every** neuron, however, carries no information: excluding all of
  its terms would leave a neutral log-likelihood and could let an
  uninformative bin spuriously win the MAP. Such all-NaN bins now get `-inf`
  log-likelihood (zero posterior mass) and can never be the argmax. This is
  reconciled with the `validate=True` NaN guard: the zero-rate exclusion runs
  first, so `validate=True` no longer rejects NaN *encoding-model* bins (it
  still rejects NaN/Inf in `spike_counts` and `prior`, and Inf/negative entries
  in `encoding_models`). The recommended path remains passing `fill_value=0.0`
  to the encoder so no NaN reaches the decoder.
- Array-shaped result objects now export to `xarray`. `DecodingResult.to_xarray()`
  returns a labeled `DataArray` with dims `("time", "bin")` (the `time` coord is
  `result.times`, or a positional integer index `np.arange(n_time)` when `times`
  is `None`), and `SpatialRatesResult.to_xarray()` returns a `DataArray` with
  dims `("neuron", "bin")`. `xarray` is imported lazily inside these methods and
  remains an **optional** dependency — install it with
  `pip install neurospatial[xarray]` (a new `xarray` extra) or `pip install
  xarray`. Calling `to_xarray()` without it raises an actionable `ImportError`.
- `DecodingResult.error_against(true_times, true_positions, *, metric="euclidean")`
  computes per-time-bin decode error against an independently sampled
  ground-truth position track, interpolating the ground truth onto the decode
  times so callers no longer hand-roll `searchsorted` alignment. It reuses the
  existing `decoding_error` core and supports `"euclidean"` and `"geodesic"`
  metrics.
- The top-level `neurospatial` package now exposes its analysis submodules
  (`encoding`, `decoding`, `behavior`, `events`, `ops`, `layout`, `regions`,
  `stats`, `simulation`, `annotation`, `animation`, `io`) via lazy (PEP 562)
  attribute access. Writing `import neurospatial as ns; ns.encoding` now
  resolves the submodule on first use and `dir(ns)` lists every domain
  alongside the eager exports, so autocomplete reveals the full API. The
  submodules are imported only when first accessed — importing the package no
  longer pulls them in eagerly — and unknown attributes still raise
  `AttributeError` so typos fail loudly. The eager top-level exports
  (`Environment`, `Region`, `Regions`, `CompositeEnvironment`,
  `bin_spikes_in_time`) are unchanged.
- Introduced a unified result-object surface so the verbs that end an analysis
  -- "summarize it", "plot it", "put it in a table" -- always have a home. A
  new `neurospatial._results.ResultMixin` base guarantees `to_dataframe()`
  (tidy/long form, so heterogeneous results `pandas.concat` cleanly),
  `summary()` (a flat dict of scalar headline metrics), and an optional
  `plot(ax=None, ...)` that returns the axis for multi-panel composition.
  `pandas` and `matplotlib` are imported lazily so neither becomes an
  import-time dependency. The existing `encoding._base.SpatialResultMixin` now
  extends `ResultMixin` (additive: all existing accessors such as
  `firing_rate`, `occupancy`, `peak_location()`, and `spatial_information()`
  are preserved) and gains default `summary()` / tidy `to_dataframe()`
  implementations. The previously bare `PlaceFieldsResult` and
  `DirectionalPlaceFields` dataclasses now carry the surface;
  `DirectionalPlaceFields` additionally gains `correlation(label_a, label_b)`,
  a per-bin `directionality_index(label_a, label_b)`, and a per-direction
  overlay `plot()`. `DecodingResult` gains `summary()` and now also extends
  `ResultMixin`.
- `encoding.theta_phase(lfp, sampling_rate, *, band=(6, 10))` extracts the
  instantaneous theta phase from a single LFP channel using a zero-phase
  Butterworth band-pass plus the Hilbert analytic-signal phase. The phase is
  returned in radians wrapped to `[0, 2*pi)`, the convention
  `encoding.phase_precession` consumes, so the output is drop-in for
  phase-precession analysis once sampled at spike times. Uses `scipy.signal`
  only — no new dependency. Exported from `neurospatial.encoding`.
- `encoding.phase_precession` and `encoding.has_phase_precession` gained a
  shuffle-based significance test via two new keyword-only parameters,
  `n_shuffles` (default `1000` / `200` respectively) and `rng`
  (`int | numpy.random.Generator | None`, for a deterministic result). The
  null is built by permuting the phase–position pairing and re-fitting the
  slope on each shuffle, so `PhasePrecessionResult.pval` is now this
  fitted-slope shuffle p-value (with `+1` smoothing, never exactly zero)
  rather than the slope-free circular-linear value; `correlation` is retained
  as a slope-independent descriptive effect size.
- Added `bin_spikes_in_time`, a public primitive that bins a sequence of
  per-neuron spike-time arrays onto a regular time grid (owning the bin
  edges and `dt / 2` bin centers) and returns an integer count matrix. Its
  `orient` argument makes the axis order explicit — `"time_x_neuron"` (the
  default, matching `decode_position`'s `(n_time_bins, n_neurons)` input) or
  `"neuron_x_time"` (matching the cell-assembly functions) — defusing the
  silent-transpose footgun between those two consumers. Exported from both
  `neurospatial.decoding` and the top-level `neurospatial` package.
- `neurospatial.io.nwb.read_units` reads per-neuron spike-time arrays from an
  NWB `units` table, returning a `(spike_trains, unit_ids)` tuple that mirrors
  the existing `read_position` contract. `unit_ids` selects a subset by the
  table's `id` values (not row indices); an unknown id raises a clear
  `ValueError` rather than silently returning the wrong unit. This closes the
  last gap in the NWB reader family, so multi-cell workflows no longer have to
  start in bare `pynwb`.
- `behavior.segmentation` gains three direction-label bridges so detected
  laps/runs can feed `compute_directional_place_fields` directly:
  `laps_to_direction_labels` (lap `direction` -> per-timepoint labels),
  `runs_to_direction_labels` (caller-supplied scalar or per-run labels), and
  `running_direction_labels` (first-class `{"inbound", "outbound", "other"}`
  labeler for linear / W-tracks). All three return the
  `NDArray[np.object_]` of shape `(n_times,)` with the `"other"` sentinel that
  `compute_directional_place_fields` consumes, closing the lap/run ->
  directional-place-field gap previously reachable only from a `Trial` via
  `goal_pair_direction_labels`. Exported from `neurospatial.behavior`.

### API changes

- Renamed the `random_state` argument to `rng` in `detect_assemblies`
  (`neurospatial.decoding`) and in `select_basis_centers`,
  `geodesic_rbf_basis`, `heat_kernel_wavelet_basis`, `chebyshev_filter_basis`,
  and `spatial_basis` (`neurospatial.ops.basis`), for consistency with the
  rest of the library. Both `int` seeds and `np.random.Generator` instances
  are still accepted. Pre-1.0: no alias is kept — update call sites to `rng=`.
- `fit_isotonic_trajectory` no longer takes a mandatory leading `env`; `env`
  is now an optional keyword-only argument (it was unused). Drop the `None`
  you previously passed as the first positional argument.
- `decoding_error`'s `env` is now keyword-only (was an optional 3rd positional).
- `confusion_matrix`'s `summary_method` argument was renamed to `method`,
  matching `fit_isotonic_trajectory`.
- Unified the time-window argument across the events GLM regressors:
  `event_indicator` and `event_count_in_window` now both take a keyword-only
  `window=(start, end)` tuple (relative seconds). `event_indicator` previously
  took a scalar symmetric half-width; rewrite `window=w` as `window=(-w, w)`.
  `event_count_in_window`'s `window` is now keyword-only. `time_to_nearest_event`
  keeps its distinct scalar `max_time` (it is a continuous regressor, not a
  windowing function).
- `is_object_vector_cell` (free function in `neurospatial.encoding.egocentric`)
  now delegates to `EgocentricRateResult.is_object_vector_cell`, using the
  egocentric-spatial-information criterion. It replaces the old
  `score_threshold`/`min_peak_rate` parameters with a single `min_info`
  (default 0.3, matching the result method), so the quick-check and the
  result-object classification can no longer disagree.
- Reordered the keyword-only `gaze_model`/`view_distance` parameters in
  `is_spatial_view_cell` to match `compute_view_rate` (`gaze_model` first).
  Keyword callers are unaffected.
- Renamed `SpatialRatesResult.classify` to `SpatialRatesResult.detect_cell_types`
  for naming parity with `EgocentricRatesResult.detect_ovcs` and
  `ViewRatesResult.detect_view_cells`.
- `compute_vte_session` is now env-first — `(env, positions, times, *, ...)` —
  matching `compute_decision_analysis`. Update positional callers.
- `detect_goal_directed_runs` now raises `ValueError` (was `KeyError`) when the
  requested region is absent, matching sibling behavior validators.
- `calibration` (and `method`/`simplify_tolerance` where present) are now
  keyword-only in `regions_from_labelme`, `regions_from_cvat`,
  `boundary_from_positions`, and `shapes_to_regions`.
- `Regions.region_center`'s parameter was renamed from `region_name` to `name`,
  matching `Regions.area`. Update keyword callers.
- The optional arguments of `simulate_trajectory_sinusoidal` and
  `simulate_trajectory_laps` (`neurospatial.simulation`) are now keyword-only.
  Only the leading required positionals remain positional — `(env, duration)`
  for `simulate_trajectory_sinusoidal` and `(env, n_laps)` for
  `simulate_trajectory_laps`. Positional callers such as
  `simulate_trajectory_sinusoidal(env, 10.0, 100.0)` must update to
  `simulate_trajectory_sinusoidal(env, 10.0, sampling_frequency=100.0)`.

### Breaking changes

- **Egocentric polar space is now a distinct type.**
  `Environment.from_polar_egocentric(...)` now returns a new
  `EgocentricPolarEnvironment` (in `neurospatial.environment.polar`) instead
  of an `Environment` carrying a hidden `coordinate_kind="polar"` flag.
  `EgocentricPolarEnvironment` is a *sibling* of `Environment` (both share a
  common `_BaseEnvironment` base), **not** a subclass — so
  `isinstance(polar_env, Environment)` is now `False`. The Cartesian-only
  methods that previously raised `ValueError` at runtime on a polar env now
  raise `NotImplementedError` (and are simply absent from the type's
  contract): `bin_at`, `contains`, `distance_between`,
  `distance_to(metric="euclidean")`, and `apply_transform`. Graph operations
  (`neighbors`, `path_between`, `reachable_from`,
  `distance_to(metric="geodesic")`, `smooth`) remain available. Consequences:
  the `Environment.coordinate_kind` attribute, the `Environment.is_polar`
  property, and the `Environment._check_cartesian` guard are removed. Along
  with the type change, polar geometry is now physically correct — connectivity
  edge `distance` weights use arc length `r·Δθ` for angular steps, `Δr` for
  radial steps, and `sqrt(Δr² + (r·Δθ)²)` for diagonals (previously the
  Euclidean norm of `(Δr, Δθ)` collapsed cm and radians), `bin_sizes` returns
  the true annular-sector area `0.5·(r₁²−r₀²)·Δθ`, and the egocentric
  `gaussian_kde` smoothing kernel uses the physical polar distance with a
  single length-unit bandwidth rather than mixing cm and radians. NWB and
  file (`to_file`/`from_file`, `to_dict`/`from_dict`) round-trips restore the
  `EgocentricPolarEnvironment` type (the on-disk `coordinate_kind` marker is
  retained for backward compatibility).
- `time_efficiency` lost its positional `goal` parameter and now takes a
  keyword-only `optimal_distance` (with `reference_speed` also keyword-only);
  external callers must update their call sites. This also fixes a latent
  bug: the function always used the Euclidean start-to-goal distance even for
  geodesic callers, so `optimal_distance` must now be supplied explicitly.
- `points_in_any_region` / `regions_containing_points` changed their default
  `point_tolerance` from ~1e-8 to 1.0 coordinate unit. Zero-area point
  regions need a real spatial tolerance to match nearby query points; pass
  `point_tolerance=1e-8` to restore the previous strict matching behavior.
- `ImageMaskLayout` / `Environment.from_pixel_mask`: the build key `bin_size`
  is renamed to `pixel_size` (the legacy `bin_size` is still accepted as a
  deprecated alias). The layout is now stored in consistent (x, y) order, so
  `grid_shape`, `bin_centers`, and `active_mask` follow (x, y); code relying
  on the previous (y, x) ordering must update.

### Documentation

- Documentation examples are now exercised in CI. Every paste-and-run example
  that had drifted from the current API was corrected against the live
  signatures — overlay construction (`PositionOverlay(positions=...)`,
  `BodypartOverlay(skeleton=Skeleton.from_edge_list(...))`), `path_progress` /
  `cost_to_goal` argument order, `circular_basis_metrics(..., cov_matrix=...)`,
  the singular `peak_view_location()` / `mean_vector_length()` result methods,
  the single-neuron object-vector-cell classification path, `graph_convolve`,
  `bin_sequence(times, positions)`, and `Environment.plot_field()`. Several
  stale module paths (`neurospatial.metrics.phase_precession`,
  `neurospatial.differential`, `neurospatial.reference_frames`,
  `neurospatial.segmentation`) and a nonexistent `write_intervals()` reference
  were repointed to their real homes, and rename-corrupted "uncertainty" /
  "entropy" prose in the decoding docstrings was restored.
- A new CI doctest gate runs `pytest --doctest-modules src/neurospatial/`, and
  the curated `docs/snippets.yml` harness gained executable entries for the
  overlay, object-vector classify, spatial-view classify, VTE, circular-basis,
  and event-regressor quickstart blocks so these examples can no longer drift
  out of sync without failing CI.
- The `decoding` trajectory (replay) module now documents that
  sharp-wave-ripple intervals come from the external `ripple_detection`
  package and that its returned `(start, end)` intervals feed
  `events.peri_event_histogram` directly. neurospatial intentionally does not
  implement ripple detection; `ripple_detection` is referenced as a
  recommended external tool, not a hard dependency.
- `encoding.compute_directional_rate`, `compute_directional_rates`, and
  `is_head_direction_cell` now note in their docstrings that they expect
  *head direction* and that a velocity-derived *movement heading* is a common
  mislabel for a "head direction cell".
- Dropped unimplemented `exponential_kernel` from the
  `events.regressors` module docstring.

### Bug fixes

- `decode_position` now raises a clear `ValueError` when `encoding_models` has
  **no finite bins** — every spatial bin is non-finite (`NaN` **or** `Inf`)
  across all neurons (e.g. `np.full((n_neurons, n_bins), np.nan)` or
  `np.full((n_neurons, n_bins), np.inf)`). Such a model carries zero
  information; previously the all-NaN-bin→`-inf` handling left the entire
  likelihood `-inf`, and `normalize_to_posterior(handle_degenerate="uniform")`
  then returned a confident-looking **uniform posterior over invalid
  positions**. The guard is now `np.isfinite`-based, so it catches all-`Inf`
  (and mixed `NaN`/`Inf`) models that slipped through the earlier `np.isnan`-only
  check when `validate=False` bypassed the `Inf` rejection. The check is
  column-wise (a bin is usable if finite for any neuron) and runs
  unconditionally, even with `validate=False`. Partial-NaN models with at least
  one finite bin still decode normally, and a legitimate no-spike time bin
  against a valid finite model still yields the correct flat (uniform)
  posterior.
- `DecodingResult.error_against` now requires `true_times` to be **strictly
  increasing** and rejects duplicates (`np.diff(true_times) <= 0`). Previously
  only descending values were rejected; duplicate timestamps (e.g.
  `[0.0, 0.0, 1.0]`) were allowed and `np.interp` resolved the tied x-values
  arbitrarily, silently mis-aligning the ground truth.
- Egocentric-polar connectivity now adds **diagonal** edges across the ±π seam
  when `connect_diagonal_neighbors=True` and `circular_angle=True`. Previously
  interior diagonals and same-ring seam (wrap) edges were added, but the
  diagonal edges crossing the seam between adjacent distance rings
  (`(r_i, last_angle)`↔`(r_{i+1}, first_angle)` and vice-versa) were missing, so
  the ±π boundary had fewer connections than every other angular step (an
  anisotropic seam). These seam diagonals now mirror the interior diagonal
  connectivity and carry the correct physical polar diagonal length
  `sqrt(Δr² + (r̄·Δθ)²)`. `circular_angle=False` still adds no seam edges
  (diagonal or same-ring), leaving the angular axis open.
- `DecodingResult.error_against` now validates that `times`, `true_times`, and
  `true_positions` are all finite up front, raising a clear `ValueError` on any
  NaN or Inf. Previously a non-finite decode time or ground-truth value passed
  silently through the `numpy.interp` alignment and produced a NaN error that
  looked like (but was not) a decode failure. The existing sorted-`true_times`
  check is unchanged.
- Egocentric-polar `gaussian_kde` smoothing now respects `circular_angle`.
  A previous fix unconditionally wrapped the angular difference into
  `[-pi, pi]`, but `Environment.from_polar_egocentric(..., circular_angle=False)`
  builds an **open** angular axis (no seam edges in the connectivity graph),
  where bins at `-pi` and `+pi` are genuinely far apart. Wrapping there leaked
  smoothing across a boundary the caller deliberately left open (the `-pi/+pi`
  seam pair got weight ~`0.7346`, the same as a true angular neighbor). The
  dense polar Gaussian kernel now wraps `Delta theta` **only** when the angular
  axis is circular and uses the raw angular difference otherwise. Circularity
  is derived from the connectivity graph (presence of seam edges), so it stays
  consistent with the graph and survives `to_file`/`from_file`, NWB round-trip,
  and `copy()`. For `circular_angle=True` the seam still receives a full
  angular-neighbor weight (unchanged); for `circular_angle=False` the seam now
  receives a tiny weight like any other far pair.
- Egocentric-polar circular **seam edges** now use the actual realized angular
  bin step instead of the requested `angle_bin_size`. When `angle_bin_size`
  does not evenly divide the angular range, `ceil` binning yields a slightly
  different realized step, and the regular angular edges already used that
  realized step; the seam (wrap) edge alone used the requested value, biasing
  seam geodesics. For example, `angle_bin_size=1.0` over `[-pi, pi]` realizes a
  step of ~`0.8976`, so a regular angular edge at radius `r` had distance
  ~`r*0.8976` while the seam edge had `r*1.0` (e.g. `4.488` vs `5.0` at `r=5`).
  The seam edge arc length is now `r * Delta_theta_actual`, matching the
  regular angular edges at the same radius.
- The weighted (count) Rayleigh test no longer rejects strongly-tuned cells
  whose spikes concentrate in only 1–2 angular bins. `rayleigh_test` validated
  the minimum sample size against the number of distinct angles, and
  `DirectionalRateResult.rayleigh_pvalue()` gated on the number of nonzero-count
  bins (`< 3 -> NaN`); a head-direction/object-vector cell with all spikes in
  two adjacent bins was therefore reported as NaN despite being genuinely,
  strongly tuned. For weighted input the effective sample size is now the total
  weight (`sum(counts)`), so 100 spikes in 2 bins returns a significant
  p-value (≈`8e-44`, matching `np.repeat` physical replication) instead of NaN.
  Genuinely-insufficient data is still rejected (`sum(weights) < 3`), and the
  count-weighting (not Hz), scale-invariance, and NaN co-filter behavior are
  unchanged.
- `EgocentricPolarEnvironment` now rejects **every** inherited method that
  assumes Cartesian `(x, y[, z])` coordinates or a Cartesian grid, raising
  `NotImplementedError` with a polar-specific message instead of silently
  returning geometric nonsense. Previously only five methods (`bin_at`,
  `contains`, `distance_between`, Euclidean `distance_to`, `apply_transform`)
  were overridden; notably `interpolate` bypassed the overridden `bin_at` and
  silently returned an array when given `(distance, angle)` points. The
  override set now also covers `interpolate`, `occupancy`, `bin_sequence`,
  `bin_sequence_with_runs`, `to_linear`, `linear_to_nd`, `rebin`, and
  `subset`. Graph operations (`neighbors`, `path_between`, `reachable_from`,
  `smooth`, `distance_to(metric="geodesic")`) remain valid and unchanged.
  Relatedly, `repr()` of any environment now reflects the concrete type
  (e.g. `EgocentricPolarEnvironment(...)`) instead of always printing
  `Environment(...)`.
- `detect_assemblies` now clamps `n_components` to the achievable factorization
  rank `min(n_neurons, n_time_bins)` (emitting a `UserWarning`) instead of
  crashing on short recordings. Previously, requesting more components than
  `n_time_bins` (with `n_time_bins < n_components <= n_neurons`) raised an
  opaque `IndexError` for `pca`/`ica` or a `ValueError` for `nmf` from deep
  inside the membership loop.
- Weighted circular statistics (`rayleigh_test`, `circular_mean`,
  `circular_variance`, `mean_resultant_length`) now validate that `weights`
  is the same length as `angles` (a length-1 array is rejected, not
  broadcast) and is non-negative; previously a mismatched or signed weight
  array could silently produce an out-of-range result. `rayleigh_test`
  additionally co-filters `weights` with the same mask used to drop NaN
  angles (keeping the two arrays aligned), and now raises a `ValueError`
  when the total weight (sum of weights) is zero instead of raising a bare
  `ZeroDivisionError` or returning a meaningless statistic.
- `explained_variance_reactivation` now returns `np.nan` for
  `explained_variance`, `reversed_ev`, and `partial_correlation` (with
  `n_pairs` set to the actual valid-pair count) when fewer than 3 valid
  neuron pairs survive NaN removal. Previously `np.corrcoef` returned NaN for
  0/1 pairs and `np.nan_to_num(..., nan=0.0)` coerced it to `0.0`, which read
  as a confident "no reactivation" rather than an undefined statistic. The
  existing `UserWarning` is unchanged.
- `compute_shuffle_pvalue` now drops non-finite null scores (NaN/Inf) with a
  warning and excludes them from `n` before computing the p-value;
  previously they were silently excluded from the count of extreme values
  while still inflating `n`, biasing the p-value toward significance. An
  all-non-finite null distribution now raises a `ValueError`. A non-finite
  `observed` score now raises a `ValueError` early instead of silently
  returning the floor p-value `1/(n+1)` (all comparisons against the null are
  `False` for a NaN observed).
- `load_labelme_json` now lets unexpected errors from polygon construction
  propagate instead of swallowing them: the `except` around `shapely.Polygon`
  is narrowed to `(TypeError, ValueError, ShapelyError)`, matching the sibling
  CVAT loaders. Malformed coordinate data is still warned-and-skipped; genuine
  bugs (e.g. a `KeyError` from a faulty `pixel_to_world` transform) now surface.
- `shuffle_cell_identity` now validates that the number of neurons in
  `spike_counts` (columns) matches the number of rows in `encoding_models`,
  raising a `ValueError` instead of silently yielding wrong decodes.
- `events_to_intervals(match_by=...)` now raises a `ValueError` on duplicate
  match keys instead of silently forming a Cartesian product
  (`DataFrame.merge(how="inner")` cross-joins repeated keys, producing more
  intervals than real start/stop pairs).
- `distance_to_reward` now raises a `ValueError` on non-finite
  `times`/`positions`/`reward_times`, and on unsorted `times` when inferring
  reward positions (no explicit `reward_positions`), instead of returning
  silently-wrong distances from corrupted interpolation.
- `add_positions` now raises a `ValueError` on degenerate trajectories (a
  single sample, all-identical trajectory times, or non-finite trajectory
  times) instead of returning all-NaN/Inf position columns.
- `Environment.bin_at`, `contains`, and `distance_between` now raise a
  `ValueError` when queried with points whose dimensionality differs from
  the environment (e.g. 3-D points against a 2-D environment), instead of
  warning and silently returning `-1`/`inf`. The legitimate
  outside-the-environment sentinel (in-dimension points that fall outside
  the active bins) is unchanged.

- `compute_egocentric_distance(metric="geodesic")` previously honored only
  the first timestep's targets when called with a time-varying target array
  of shape `(n_time, n_targets, 2)`. Now distances are computed
  per-timestep, with an internal cache so repeated targets remain cheap.
  Static-target callers (passing shape `(n_targets, 2)`) see no behavior
  change.
- `detect_runs_between_regions(min_speed=...)` no longer treats an
  off-environment (`-1`) or out-of-range bin in a run as a real position. The
  velocity filter indexed `env.bin_centers[run_bin_idx]` with raw run bins, so
  a `-1` sample silently wrapped to `bin_centers[-1]` (inflating the mean speed
  and keeping runs that should be filtered out), and an index `>= n_bins` would
  raise `IndexError`. Speed is now computed only over consecutive sample pairs
  whose endpoints both map to a valid bin; runs with no usable pair are not
  filtered on a spurious velocity. Fully on-environment runs are unaffected.
- Weighted circular statistics (`rayleigh_test`, `circular_mean`,
  `mean_resultant_length`, `circular_variance`) now reject non-finite
  `weights` (NaN/Inf) with a `ValueError` via `_validate_weights`. Previously
  only negative weights were rejected, so a NaN/Inf weight flowed into
  `sum(weights)` and silently turned the statistic into `nan` instead of
  raising on invalid count/frequency input. The non-negativity / equal-length
  checks and the NaN-*angle* co-filter (which drops NaN angles, not weights)
  are unchanged.
- Weighted `circular_mean`, `mean_resultant_length`, and `circular_variance`
  now apply the same NaN-*angle* co-filter that `rayleigh_test` already used:
  a NaN angle is dropped together with its paired weight, and the statistic is
  computed on the remainder. Previously a single NaN angle propagated through
  `cos`/`sin` and turned the weighted result into `nan` even when the other
  samples were valid. Non-finite *weights* still raise (unchanged); only NaN
  *angles* are dropped.
- `detect_runs_between_regions(min_speed=...)` now drops a run when there is no
  usable speed estimate (no consecutive on-environment bin pair from which to
  compute a velocity), instead of silently keeping it. A run whose speed cannot
  be validated must not pass a speed gate it never satisfied; e.g. a run slice
  like `[bin, -1, bin]` with `min_speed > 0` is now dropped. Runs with `min_speed
  is None` and fully on-environment runs are unaffected.

## [v0.4.0] - 2026-05-26

## What's Changed

### Features
- feat(M5.1): Environment._state_version + versioned_cached_property (9f4d375)
- feat(M3.4): add public exception hierarchy in neurospatial._exceptions (86315f3)
- feat(M3.1): EnvironmentNotFittedError supports free-function form (749c7f5)
- feat(environment): mark polar envs and refuse Cartesian-assuming methods (M1 1.3) (7377271)
- feat(encoding): surface batch metric failures via BatchScoresResult + warning (M1 1.5) (dea5404)
- feat(encoding): warn before silent interneuron exclusion in detect_place_fields (M1 1.4) (14291b3)
- feat(decoding): default decode_position to validate=True; add non-negative spike_counts and rate checks (M1 1.8) (8143baa)
- feat(encoding/decoding): validate env._is_fitted at compute_*_rate(s) and decode_position entry (M1 1.6) (83cc42b)
- feat(encoding): validate spike_times shape/sortedness at compute_*_rate(s) entry (M1 1.7) (ac30a17)
- feat(M8): expose gaze_offsets parameter in compute_view_rate(s) API (b3bc49b)
- feat(M8): implement backend-aware JAX metrics for result classes (5641618)
- feat(M7): add _field_metrics.py and extend _metrics.py (472cecb)
- feat(M6): add NumPy vs JAX performance benchmarks (Task 6.7) (4bbffaa)
- feat(M6): make metrics backend-aware and wire view/egocentric to JAX smoothing (b6de96c)
- feat(M6): implement real JAX compute path for spatial rate computation (991ae80)
- feat(M6): wire up JAX backend dispatch in compute functions (68f3e69)
- feat(M6): fix JAX array detection and add backend awareness tests (Task 6.5) (0d6d02e)
- feat(M6): add backend parameter to all compute functions (Task 6.4) (13f49d8)
- feat(M6): implement JAX metric computations in _core_jax.py (Task 6.2) (61c9f6d)
- feat(M6): implement JAX smoothing operations in _core_jax.py (Task 6.1) (4e3c024)
- feat(M5): verify comprehensive tests for egocentric encoding (Task 5.9) (4bdfef3)
- feat(M5): implement compute_egocentric_rates() function (Task 5.8) (9b721c4)
- feat(M5): implement compute_egocentric_rate() function (Task 5.7) (98a9250)
- feat(M5): implement egocentric binning layer (Task 5.6) (14128b8)
- feat(M5): implement EgocentricRatesResult.to_dataframe() (Task 5.5) (8073457)
- feat(M5): implement EgocentricRatesResult batch methods (Task 5.4) (2f591bd)
- feat(M5): implement EgocentricRateResult classification (Task 5.3) (fda734f)
- feat(M5): implement EgocentricRateResult convenience methods (Task 5.2) (837212d)
- feat(M5): implement EgocentricRateResult and EgocentricRatesResult classes (Task 5.1) (3c2e823)
- feat(M4): implement compute_view_rates() batch function (Task 4.8) (4e9f740)
- feat(M4): implement compute_view_rate() function (Task 4.7) (c4a25f8)
- feat(M4): implement view binning layer for spatial view encoding (Task 4.6) (6407637)
- feat(M4): implement ViewRatesResult.to_dataframe() (Task 4.5) (3a151ee)
- feat(M4): implement ViewRatesResult batch methods (Task 4.4) (cc22358)
- feat(M4): implement ViewRateResult.is_view_cell() classification (Task 4.3) (dfe8597)
- feat(M4): implement ViewRateResult convenience methods (Task 4.2) (ce2b7b5)
- feat(M4): add ViewRateResult and ViewRatesResult classes (Task 4.1) (08433dd)
- feat(M3): add comprehensive tests for directional encoding (Task 3.11) (87cadbf)
- feat(M3): implement compute_directional_rates() function (2af04fe)
- feat(M3): implement compute_directional_rate() function (212dc4a)
- feat(M3): implement directional binning layer (db988f9)
- feat(M3): implement DirectionalRatesResult.to_dataframe() (b21ba4f)
- feat(M3): implement DirectionalRatesResult batch methods (9617198)
- feat(M3): implement DirectionalRateResult classification methods (115e7e2)
- feat(M3): implement DirectionalRateResult tuning metrics (4dd95a2)
- feat(M3): implement DirectionalRateResult convenience methods (89489c0)
- feat(encoding): add DirectionalRateResult and DirectionalRatesResult classes (8fae7f0)
- feat(encoding): implement compute_spatial_rates() batch function (b89cc93)
- feat(encoding): implement compute_spatial_rate() function (b5c39af)
- feat(encoding): implement binning layer for spatial encoding (6a6f42d)
- feat(encoding): add SpatialRatesResult.to_dataframe() method (462116f)
- feat(encoding): add SpatialRatesResult batch metrics methods (e1a170d)
- feat(encoding): add SpatialRatesResult batch convenience methods (9df1e90)
- feat(encoding): add SpatialRateResult cell type metric methods (9db2b9e)
- feat(encoding): add SpatialRateResult convenience methods (72b1263)
- feat(encoding): add SpatialRateResult and SpatialRatesResult classes (3ec6ca8)
- feat(encoding): add batch_border_scores to _metrics.py (11e1ed8)
- feat(encoding): add batch_grid_scores to _metrics.py (b790877)
- feat(encoding): add _smoothing.py with shared smoothing implementations (9fcac0b)
- feat(encoding): add _metrics.py with spatial information and sparsity (b0872a5)
- feat(encoding): add _core_jax.py with stub functions (27ec643)
- feat(encoding): add _core_numpy.py with stub functions (99cf92b)
- feat(encoding): add _backend.py with backend selection (1c366af)
- feat(encoding): add _spikes.py with spike format normalization (b7356ef)
- feat(encoding): add _base.py with shared protocols and helpers (7424105)
- feat(M13): add comprehensive import test suite (009e91a)
- feat(M5): complete integration and documentation for behavioral metrics (57e3a52)
- feat(M4): implement VTE metrics module (a517953)
- feat(M3): implement spatial decision analysis module (be179d6)
- feat(M2): implement goal-directed navigation metrics module (bd56410)
- feat(M1): implement path efficiency metrics module (73b38e4)
- feat(M3.4): implement spatial view cell metrics module (0e077d5)
- feat(M3.3): implement spatial view field analysis (4bc2b53)
- feat(M3.2): implement spatial view cell simulation model (74fba03)
- feat(M3.1): implement visibility/gaze computation module (6dff4ec)
- feat(M2.6): complete object-vector cells integration and documentation (31a7cd0)
- feat(M2.4): implement ObjectVectorOverlay for animation (4be296e)
- feat(M2.3): implement ObjectVectorFieldResult and compute_object_vector_field (bd3e901)
- feat(M2.2): implement ObjectVectorMetrics and analysis functions (d7bd14f)
- feat(M2.1): implement ObjectVectorCellModel simulation (04c4623)
- feat(M1.3): implement egocentric polar environment factory (4d75906)
- feat(M1): implement egocentric reference frames module (26aa1d6)
- feat(M8): add top-level exports and documentation for events module (59955bc)
- feat(M7): implement NWB integration for events module (04c5d76)
- feat(M6): implement peri-event analysis functions (7f5ade7)
- feat(M4): implement interval utilities for events module (6da1016)
- feat(M3.4): implement add_positions() spatial event utility (af9eb4d)
- feat(M2.3): implement event_indicator() temporal GLM regressor (ebd5234)
- feat(M2.2): implement event_count_in_window() temporal GLM regressor (75f2d06)
- feat(M2.1): implement time_since_event() temporal GLM regressor (a5080cb)
- feat(M1): implement events module core infrastructure (02ff340)
- feat(basis): add maze-aware spatial basis functions for GLMs (e491364)
- feat(M6): add confidence bands to plot_circular_basis_tuning() (a209d4f)
- feat(M3): add plot_circular_basis_tuning() visualization for GLM tuning curves (faaba7a)
- feat(M2): add is_modulated() convenience function for GLM circular modulation (252a5dc)
- feat(M1): add circular basis functions for GLM design matrices (bb9ad19)
- feat(M4.7-M5): add property-based tests and complete integration (76b41a8)
- feat(M3.5): implement plot_head_direction_tuning for HD cell visualization (911b874)
- feat(M3.3-M3.4): implement HeadDirectionMetrics and HD cell classification (8480fb8)
- feat(M3.2): implement head_direction_tuning_curve for HD cell analysis (118820d)
- feat(M3.1): create head_direction.py module with setup (dc5c3d8)
- feat(M2.3): implement plot_phase_precession for phase precession visualization (31344ff)
- feat(M2.2): implement phase_precession for place cell analysis (dc271be)
- feat(M1.4): implement circular_circular_correlation for phase coherence analysis (4d0e103)
- feat(M1.3): implement circular_linear_correlation for phase-position analysis (9a51ee1)
- feat(M1.2): implement rayleigh_test for circular uniformity testing (d605fa0)
- feat(M1.1): add circular statistics module with internal helpers (d5eb130)
- feat(phase6.1): add automated pytest-benchmark suite for napari playback (417d089)
- feat(phase5.2): audit layer.data assignments and add test documentation (da10c43)
- feat(phase5.1): complete callback audit and document findings (e078fdc)
- feat(phase4.2): add rapid scrubbing debounce to PlaybackController (1c246cf)
- feat(phase3.2): add throttle parameters and xlim caching for time series (39aa50b)
- feat(phase3.1): add update_mode parameter to TimeSeriesOverlay (ee8febc)
- feat(phase2.2): add configurable cache_size and async prefetching for video (1284873)
- feat(phase2.1): add time-indexed image layer for in-memory video (3709383)
- feat(phase1.2): integrate PlaybackController into render_napari() (7d0fa42)
- feat(phase1): add PlaybackController class for frame-skipping playback (82e75d4)
- feat(phase0): complete baseline measurements for napari playback (adb270e)
- feat(phase0): add video/timeseries tracing to perfmon config (5d0c78b)
- feat(phase0): add napari playback benchmark script (34fc1d1)
- feat(M7): update demo scripts to use speed-based API (9288fcc)
- feat(animation): make frame_times required and add speed parameter (Task 3.1) (d128325)
- feat(animation): add playback speed constants and helper function (d986f2e)
- feat(animation): add HTML backend warning for TimeSeriesOverlay (8cae4a2)
- feat(animation): add time series support to widget backend (82981f6)
- feat(animation): add napari dock widget for time series overlays (bdab281)
- feat(animation): add TimeSeriesArtistManager for matplotlib rendering (302246d)
- feat(animation): add TimeSeriesOverlay for continuous variable visualization (4b239bb)
- feat(F24): complete M6 testing milestone - all EventOverlay tests verified (07e6392)
- feat(F24): implement HTML backend event overlay rendering (M4) (20d4ef6)
- feat(F24): implement matplotlib event overlay rendering (M3) (d37eb2f)
- feat(F24): implement napari backend for EventOverlay (M2) (f1c3d07)
- feat(F24): implement EventOverlay core data structures (M1) (afa5a43)
- feat(M5.2): implement DecodingResult visualization methods (fe1a97e)
- feat(M5.1): finalize public API exports (6298bbf)
- feat(decoding): implement significance testing functions (Milestone 4.5) (a66ea07)
- feat(decoding): implement Poisson surrogate generation (Milestone 4.4) (1fa6c19)
- feat(decoding): implement posterior shuffles (Milestone 4.3) (799095b)
- feat(decoding): implement cell identity shuffles (Milestone 4.2) (8caeb61)
- feat(decoding): implement core temporal shuffles (Milestone 4.1) (8d870cb)
- feat(decoding): implement detect_trajectory_radon (Milestone 3.4) (465e749)
- feat(decoding): implement fit_linear_trajectory (Milestone 3.3) (aa3d105)
- feat(decoding): implement fit_isotonic_trajectory (Milestone 3.2) (732c4e3)
- feat(decoding): implement trajectory result dataclasses (Milestone 3.1) (2dc27d3)
- feat(decoding): implement decoding_correlation (Milestone 2.3) (0aa884f)
- feat(decoding): implement confusion_matrix (Milestone 2.2) (364fa8e)
- feat(decoding): implement error metrics (Milestone 2.1) (3a01e7b)
- feat(decoding): implement estimate functions (Milestone 1.5) (7375e6d)
- feat(decoding): implement posterior normalization (Milestone 1.4) (4c9e27d)
- feat(decoding): implement likelihood functions (Milestone 1.3) (e645fbe)
- feat(decoding): implement DecodingResult container (Milestone 1.2) (3fa9ac3)
- feat(decoding): add decoding subpackage structure (Milestone 1.1) (9eacfe3)
- feat(M4): export helper functions from animation module (Task 4.3) (985f750)
- feat(M4): add large_session_napari_config helper (Task 4.2) (f2b2f17)
- feat(M4): add estimate_colormap_range_from_subset helper (Task 4.1) (6e97d70)
- feat(M3): disable multiscale pyramids in napari viewer (Task 3.3) (48d0d04)
- feat(M3): add large dataset warning and subsampling for colormap range (Task 3.2) (332ab7f)
- feat(M3): add streaming path to compute_global_colormap_range (Task 3.1) (e204029)
- feat(M2): add benchmark comparing LazyFieldRenderer vs Dask (Task 2.6) (3f86107)
- feat(M2): add array input support to lazy field renderers (Tasks 2.2-2.4) (a8b844b)
- feat(M2): add array input support to render_napari (Task 2.1) (1d5352c)
- feat(animation): preserve array format for napari backend (Milestone 1) (ea81920)
- feat(exports): add ScaleBarConfig to public API (M4) (91b075a)
- feat(animation): add scale_bar parameter to all backends (M3) (7a42cd7)
- feat(visualization): add scale bar support for static plots (M1-M2) (c226be5)
- feat(exports): add directional place fields to public API (Tasks 4.1-4.3) (44dec0e)
- feat(metrics): add directional_field_index function (Task 3.1) (653cdf7)
- feat(behavioral): add heading_direction_labels function (Task 2.2) (b1b9b97)
- feat(behavioral): add goal_pair_direction_labels function (Task 2.1) (42b4a69)
- feat(spike_field): add compute_directional_place_fields function (Task 1.3) (f7b0a6f)
- feat(spike_field): add _subset_spikes_by_time_mask helper (Task 1.2) (672e2f2)
- feat(spike_field): add DirectionalPlaceFields dataclass (Task 1.1) (a33f4af)
- feat(annotation): implement 1D preview for track graph (Task 4.2) (331a2bf)
- feat(annotation): implement edge order editing UI (Task 4.1) (e237764)
- feat(annotation): add Task 3.3 - module exports for track graph (7fada5b)
- feat(annotation): implement Task 3.2 - annotate_track_graph entry point (ab7bfb0)
- feat(annotation): implement Task 3.1 - TrackGraphResult in track_graph.py (a2a4815)
- feat(annotation): implement Task 2.3 - create control widget for track graph (04f3a34)
- feat(annotation): implement Task 2.2 - event handlers for track widget (465d105)
- feat(annotation): implement Task 2.1 - create layer setup for track graph (c61f0ea)
- feat(annotation): complete Task 1.4 - unit tests for state management (dd8810c)
- feat(annotation): implement graph building helpers for track annotation (8031b54)
- feat(annotation): implement TrackBuilderState for track graph annotation (55af604)
- feat(annotation): add TrackGraphMode type definition (2820f6b)
- feat(mazes): implement T-Maze (Milestone 2.2) (89e9259)
- feat(mazes): implement Linear Track maze (Milestone 2.1) (691d7e5)
- feat(mazes): implement Milestone 1 - foundation for simulation mazes (e55b701)
- feat(M4-M7): complete boundary seeding feature (3df9890)
- feat(M3): wire boundary seeding into annotate_video() (f307b65)
- feat(M2): add napari integration for boundary seeding (3f7c6f0)
- feat(annotation): add boundary inference from position data (M1) (09ad454)
- feat(behavioral): implement graph_turn_sequence for turn classification (b3cb24c)
- feat(behavioral): implement cost_to_goal with cost maps and terrain difficulty (8a27bc2)
- feat(behavioral): implement compute_trajectory_curvature with smoothing (1624063)
- feat(behavioral): implement time_to_goal with temporal countdown (d6a27b7)
- feat(behavioral): implement distance_to_region for scalar and dynamic targets (69a0289)
- feat(behavioral): implement path_progress with geodesic/euclidean metrics (e3ab9ea)
- feat(behavioral): implement trials_to_region_arrays helper (0de2c87)
- feat(api): export segmentation functions to public API (a738b5e)
- feat(M3): extend write_laps() with region columns (1a91141)
- feat(M2): add write_trials() and read_trials() NWB functions (91f54e5)
- feat(M1): add start_region field to Trial, rename outcome to end_region (3bf40e2)
- feat(nwb): add 1D GraphLayout round-trip support and preserve is_1d property (66c8c14)
- feat(nwb): implement read_environment() for NWB round-trip (M3.2.1) (81cc8f2)
- feat(nwb): implement write_environment() for saving Environment to NWB scratch (M3.1.1) (8b9420b)
- feat(nwb): implement write_region_crossings() for saving region crossing events (M2.2.2) (7f3d585)
- feat(nwb): implement write_laps() for saving lap events (M2.2.1) (99d6d61)
- feat(nwb): implement write_occupancy() for saving occupancy maps (M2.1.2) (77dc16c)
- feat(nwb): implement write_place_field() for saving spatial fields (M2.1.1) (73cab07)
- feat(nwb): implement environment_from_position() factory function (M1.4.4) (68a2d99)
- feat(nwb): implement head_direction_overlay_from_nwb() factory function (M1.4.3) (f0af9f0)
- feat(nwb): implement bodypart_overlay_from_nwb() factory function (M1.4.2) (78c80e3)
- feat(nwb): implement position_overlay_from_nwb() factory function (M1.4.1) (05a0e57)
- feat(nwb): implement read_intervals() function (M1.3.2) (73cf6e6)
- feat(nwb): implement read_events() function (M1.3.1) (152bf51)
- feat(nwb): implement read_pose() function (M1.2.1) (c3caab5)
- feat(nwb): implement read_head_direction() function (M1.1.3) (9ada46d)
- feat(nwb): implement read_position() function (M1.1.2) (10e3d8e)
- feat(nwb): add NWB integration module structure (M0.1) (5ac9f92)
- feat(annotation): improve widget UX and add validation (649994a)
- feat(annotation): export annotation API from main package (dd565eb)
- feat(annotation): add napari widget and core annotation entry point (43e7688)
- feat(annotation): add IO module for LabelMe/CVAT import (ff749d3)
- feat(annotation): implement converters module with TDD (6b7ac0f)
- feat(8.5): add VideoReaderProtocol for type safety (dd54f96)
- feat(6.3): add video overlay backend integration tests (9727d7f)
- feat(6.2): add video fixture utilities for testing (a0e87d8)
- feat(6.1): add calibration edge case tests and ill-conditioned landmark check (b304a75)
- feat(5.1): add calibrate_video() convenience function (1b4a38e)
- feat(4.3): add widget and HTML backend video overlay support (5a68b7d)
- feat(4.2): add video export backend support for video overlays (37773af)
- feat(4.1): add napari video layer rendering support (ba65888)
- feat(3.2): add pipeline integration for VideoOverlay (8f40fbf)
- feat(3.1): add VideoReader class with LRU caching (b79c257)
- feat(2.2): add VideoData internal container (59c7dca)
- feat(2.1): add VideoOverlay dataclass (6d1cfa5)
- feat(1.2): add VideoCalibration dataclass (a9edcfb)
- feat(1.1): add video calibration helpers (f189618)
- feat(I.2-I.6): complete M0 integration pre-requisites for VideoOverlay (34a1c93)
- feat(P6.2): add adjacency property to Skeleton for graph traversal (6908667)
- feat(P6.1): add skeleton edge canonicalization and deduplication (f4ea368)
- feat(P5.3): add DPI warning for video export (b1bb9d1)
- feat(P4.3): add optional JPEG image format support to widget backend (5a6834a)
- feat(P4.1): fix widget set_array optimization and add fallback diagnostics (d5aa081)
- feat(P3): complete Phase 3 - Overlay Conversion & Core Orchestration (f48a075)
- docs(P2.3): verify tracks color_by workaround is correct for napari 0.5.6 (ad6ca4b)
- feat(P2.2): add per-viewer transform fallback warning tracking (6ec55a9)
- feat(P1.2): add layout_type and is_grid_compatible properties to layouts (376708a)
- feat(P1.1): centralize coordinate transforms in shared module (dec6581)
- feat(P0.3): add benchmark scripts and record baseline metrics (70c0994)
- feat(P0.2): add benchmark dataset generators for animation performance testing (dd82407)
- feat(P0.1): add timing instrumentation for animation backends (d21a2b0)
- feat(M2): replace Shapes-based skeleton with precomputed Vectors layer (4fc1773)
- feat(napari): add _build_skeleton_vectors for precomputed skeleton rendering (7482c05)
- feat(M7.4): complete CLAUDE.md updates and address code review feedback (5ba933c)
- feat(M7.3): create comprehensive overlay example notebook (2675b56)
- feat(M7.2): add overlay examples to animate_fields() docstring (3f04a99)
- feat(M7.1): add comprehensive overlay API documentation (bb231c8)
- feat(M6.3): add backend capability matrix tests (b88e2ec)
- feat(animation): add visual regression tests for overlay rendering (M6.2) (4deec69)
- feat(animation): add comprehensive integration tests for overlay system (M6.1) (ae8d0fb)
- feat(animation): implement widget backend overlay rendering (M5.3) (b57db14)
- feat(animation): implement HTML overlay size guardrails (M5.2) (0b94364)
- feat(animation): implement HTML backend overlay rendering (M5.1) (17f2d89)
- feat(animation): implement video overlay optimization and performance tests (M4.3, M4.5) (fdeaf40)
- feat(animation): add pickle-ability validation for parallel video rendering (M4.2) (0693e5c)
- feat(animation): implement video backend overlay rendering (M4.1) (5775939)
- feat(animation): add napari performance benchmarks (M3.6) (f50ea35)
- feat(animation): implement napari overlay rendering (M3.1) (349f886)
- feat(animation): integrate overlay system into core dispatcher (M2.2) (99a7616)
- feat(animation): update animate_fields() signature for overlay support (dede6d4)
- feat(animation): implement conversion funnel for overlay data alignment (f1633a3)
- feat(animation): implement validation functions with WHAT/WHY/HOW error messages (6b9b37c)
- feat(animation): add multi-field viewer support for Napari (2fafc13)
- feat(napari): improve UX with larger slider and spacebar playback (3b07b7f)
- feat(napari): add interactive playback speed slider widget (95fb53f)
- feat(napari): add playback controls - start at frame 0 and configure FPS (b74ef30)
- feat(M6): integrate animate_fields() method into Environment class (58692db)
- feat(M5): implement Jupyter widget backend with interactive controls (6c0c2d8)
- feat(M4): implement napari GPU-accelerated backend with LRU caching (b9740cf)
- feat(M3): add -r and -threads ffmpeg flags from original gist (7f361cc)
- feat(M3): implement parallel video backend with ffmpeg (7b8af5d)
- feat(animation): implement core dispatcher and backend stubs (626679c)
- feat(animation): add optional dependencies for animation backends (111f052)
- feat(animation): implement HTML backend with embedded player (M2) (9c3fa5a)
- feat(animation): implement rendering utilities for multi-backend animation (d801f41)
- feat(M3.2): rename map_probabilities_to_nearest_target_bin() to map_probabilities() (a555454)
- feat(M2.1): rename shortest_path() to path_between() (83ad2d9)
- feat(M1.3): add env.apply_transform() method (5a92d4b)
- feat(M1.2): add env.region_mask() method (3aafe6e)
- feat(M1.1): add selective clearing to env.clear_cache() (5f0f931)
- feat(M4): implement EnvironmentNotFittedError custom exception (Tasks 4.6-4.8) (22cdf15)
- feat(M4): add PathLike type alias and comprehensive pathlib tests (Task 4.1) (c4647b9)
- feat(M2): implement scipy connected components (Task 2.8-2.12) - 6.16x speedup (3b7abfa)
- feat(M2): add comprehensive cache management system (Task 2.11) (90940b7)
- feat(M2): standardize scale parameter naming across transforms and alignment (Task 2.10) (d969611)
- feat(api): fix import inconsistencies and add comprehensive tests (56286f9)
- feat(docs): replace T-maze with 2D Shapely polygon in notebook 11 (a6ba6f3)
- feat(docs): add place field shift measurement examples to notebook 11 (28e2a57)
- feat(M3): implement grid_cell_session() with 18 tests (feecbfa)
- feat(M3): implement boundary_cell_session() with mixed cell types (03033b9)
- feat(M3): implement tmaze_alternation_session() with trial metadata (5e1d9d1)
- feat(M3): implement linear_track_session() convenience function (5f07f92)
- feat(M3): implement open_field_session() convenience function (9edc3c4)
- feat(M3): implement plot_session_summary() visualization (b947396)
- feat(M3): implement validate_simulation() validation helper (1908357)
- feat(M3): implement simulate_session() convenience function (e28045e)
- feat(M3): implement SimulationSession dataclass (017cd79)
- feat(M3): implement GridCellModel with hexagonal firing pattern (5504ff2)
- feat(M2): implement add_modulation() with parameter validation (5b90bc5)
- feat(M2): implement BoundaryCellModel with TDD (8558b68)
- feat(simulation): implement simulate_trajectory_laps (6024dcc)
- feat(simulation): complete Milestone 1 - core simulation framework (ad96365)
- feat(visualization): add plot_field() method for spatial field visualization (3fa792b)
- feat(spike_field): implement unified place field API with diffusion KDE default (82ad0ba)
- feat(M4.6): add behavioral segmentation example notebook (7547f9a)
- feat(M4.6): add trajectory analysis example notebook (2049764)
- feat(M4.4): implement trial segmentation for behavioral tasks (cb9740b)
- feat(M4.2): implement region-based segmentation (b4e482f)
- feat(M4.1): implement trajectory characterization metrics (4fcd3fa)
- feat(M3.3): implement border_score for boundary cell analysis (9f17b69)
- feat(M3.2): implement population-level place field metrics (3bed01b)
- feat(M3.1): implement place field detection and spatial metrics (9f1083e)
- feat(M2.2): implement convolve() spatial convolution primitive (7364cf7)
- feat(M2.1): implement neighbor_reduce() spatial signal processing primitive (136c004)
- feat(M1.3): implement divergence operator and rename KL divergence (a255207)
- feat(M1.2): implement gradient operator (1051dc7)
- feat(M1.1): implement differential operator infrastructure (4e2be9f)
- feat(M0.3): add comprehensive example notebook for Phase 0 primitives (d515e89)
- feat(M0.2): add reward field primitives for RL applications (ad87427)
- feat: add spike-to-field conversion and occupancy options (69e9106)
- feat(M11): complete comprehensive testing and verification (94c1a89)
- feat(M10): create package init and extract remaining methods (7b18695)
- feat(M9): create core module with mixin inheritance (519b40e)
- feat(M8): extract factory methods to mixin (237aa54)
- feat(M4): extract analysis methods to mixin (bf8baf7)
- feat(M6): extract serialization methods to mixin (bd3a18b)
- feat(M5): extract region methods to mixin (3938948)
- feat(M3): extract visualization methods to mixin (5581cfd)
- feat(M2): extract check_fitted decorator (ff34c31)
- feat(M1): complete Milestone 1 - Preparation (5a4c187)

### Bug Fixes
- test(encoding): act on four-agent PR review of d864b6d (cb25580)
- fix(M5.8): preserve None vs '' distinction in Environment.__repr__; add __str__ (9a1709c)
- fix(M4.5): set env.units in HeadDirectionCellModel doctest + numpy bool wrap (63ed531)
- fix(M4.5): add speed_units to remaining source-file doctests (1d7dc22)
- fix(M4): repair user-facing docs/examples broken by M4.5 + finish M4.1, M4.3 (ad1ef4d)
- fix(M3): finish four reviewer-flagged gaps (616c186)
- fix(M2.D): align remaining distance_to_reward call sites with current signature (14f6256)
- fix(M2.D): repoint Bayesian decoding examples + docs to canonical stats imports (1f01755)
- fix(M2): README CI snippet + batch wrappers reject bad API params (22736ac)
- fix(M2): drop dedup= from bin_sequence_with_runs; bins shape is per-run (0795927)
- fix(M2): bin_sequence dedup is gap-blind; tighten test naming + stale docs (408b8f8)
- fix(M2): tighten bin_sequence_with_runs gap semantics + batch_grid_scores irregular-env contract (7b5f54b)
- docs(M2): close M2.A/M2.B review-response findings (9 items) (f5ad3bf)
- fix(M1): match plain-ndarray copy=False semantics in BatchScoresResult (cd02b24)
- fix(M1): close remaining review-response findings (4 items) (6ead6a5)
- fix(M1): bundle review-response findings (10 items) (312b7a5)
- fix(environment): rewrite Environment.subset() to return a MaskedGrid for grid envs (M1 1.1) (3f93811)
- fix(environment): align Environment.occupancy with bin_sequence on out-of-env samples (M1 1.2) (2e3dfed)
- fix(decoding): reject zero-mass and array-like priors at validate=True boundary (M1 1.8 second follow-up) (c6745f2)
- fix(decoding): tighten decode_position validation per review (M1 1.8 follow-up) (85bf631)
- docs(examples): rewrite examples/README and place notebook 21 in canonical home (M0 0.6, 0.7) (76a236d)
- fix(encoding): enable JAX x64 in production; gate JAX tests on the optional extra (055ba08)
- fix(encoding): three more review findings (precision, validation, test unpack) (23614c0)
- fix(encoding): address review findings on M3, M4 and M2.5 (cf7ce0a)
- fix(encoding): make _backend.py doctests platform-independent (63cef8a)
- fix(M8): enforce monotonic time validation in single-neuron binning (91bf041)
- fix(M8): handle NaN in egocentric nearest-object selection (c50c947)
- fix(test): update segment_trials parameter validation test (8b71f9f)
- fix(M6): skip JAX dispatch for binned smoothing method (ea92811)
- fix(M6): add backend validation to compute_spatial_rate(s) (Task 6.4 follow-up) (5a1c547)
- fix(M6): add backend validation to compute_spatial_rate(s) (Task 6.4 follow-up) (322ad00)
- fix(encoding): add time validation parity and fix docs (95f1946)
- fix(encoding): complete remaining code review fixes (0cd005d)
- fix(encoding): correct view occupancy calculation and add validation (b0ce142)
- fix(M3): address code review findings for directional encoding (cbf1001)
- fix(M0): change head_direction.py angle_unit default from deg to rad (dcdeb91)
- fix(phase6.2): add QTimer for automatic debounce trailing-edge flush (89b85ac)
- fix(tests): add missing random seeds to prevent flaky tests (dec0588)
- fix(behavioral): fix test fixture and test assertion (a7259cd)
- fix(nwb): address code review feedback and add timestamps parameter (1762302)
- fix(annotation): deselect shape after naming using processEvents (631ecd4)
- fix(annotation): deselect shape after Enter with QTimer delay (79dbe56)
- fix(annotation): clear selection after applying name with Enter (76ec8fb)
- fix(annotation): address code and UX review feedback (c0d32a6)
- fix(8.1): add warning for VideoOverlay interp='linear' not implemented (d930e81)
- fix(4.1): add video frame update callback for scrubbing (bd1798c)
- fix(4.1): get VideoReader dimensions for affine transform (2048ff3)
- fix(4.1): use VideoReader subscript access in VideoData.get_frame() (7a4f287)
- fix(P5.2): control ffmpeg I/O to avoid buffer issues (33dc134)
- fix(P4.2): fix skeleton initialization IndexError and add overlay artist tests (349a12a)
- fix(P2.4): only throttle frame info updates during playback (0f19282)
- fix(napari): use line style for skeleton vectors (hide arrow heads) (3e3ec4e)
- fix(napari): use [position, direction] format for skeleton vectors (e6e87ca)
- fix(animation): map overlay coordinates to pixel indices for napari (d02406f)
- fix(animation): invert Y-axis for napari image coordinate system (439347c)
- fix(animation): transpose RGB image for napari (y, x) convention (6bf6dc6)
- fix(animation): correct napari add_tracks API usage for position overlays (37f697e)
- fix(tests): reduce memory safety test grid size to prevent CI timeout (dd6df73)
- fix(ci): sync mypy configuration between pre-commit and CI (970a358)
- fix(tests): prevent Qt crashes in napari tests with xdist_group (b6cd89a)
- fix(napari): use correct API for spacebar playback toggle (9a82171)
- fix(napari): fix deprecation warnings and notebook blocking issues (5fe723e)
- fix(napari): ensure only time slider visible, improve control docs (f57e9c6)
- fix(napari): remove contrast_limits for RGB images (c7fae05)
- fix(examples): add napari.run() to block and keep windows open (fc296f1)
- fix(napari): add ndim property and tuple indexing support to LazyFieldRenderer (9fe2606)
- fix(M2): remove incorrect type annotations from SubsetLayout (Task 2.8) (b4f9918)
- fix(regions): prevent metadata mutation with deep copy (f74e6e7)
- fix(hexagonal): improve numerical stability with MIN_HEX_RADIUS validation (d0ccbf4)
- fix(trajectory): improve numerical stability with named EPSILON constant (d320438)
- fix(security): prevent path traversal attacks in io.py (5a8cd24)
- fix(docs): simplify T-maze to clean T shape with uniform corridors (3a1588b)
- fix(docs): connect T-maze arms to form continuous horizontal bar (50c84ca)
- fix(docs): sync examples/11_place_field_analysis with Part 9 additions (00c1588)
- fix(viz): fix grey checkerboard artifacts in grid field visualizations (555beb7)
- fix(docs): fix f-string-missing-placeholders in 08_spike_field_basics.ipynb (12afc63)
- fix(docs): fix CI failures - formatting and broken links (3038cf0)
- fix: remove unused variables instead of prefixing with underscore (a178917)
- fix(M3.5): increase duration to 100s in notebook 08 for better coverage (bb1f93d)
- fix(M3.5): improve 2D exploration in 08_spike_field_basics.ipynb (a25d090)
- fix(examples): update notebooks 11 and 12 with corrected OU parameters (49fd1e0)
- fix(simulation): correct OU process sigma formula to match RatInABox (6edfda1)
- fix(M3.5): correct ground_truth access in 11_place_field_analysis.ipynb (2d2e75d)
- fix(visualization): address code review improvements for plot_field() (013a9db)
- fix: resolve linting errors in example scripts (8b3e5a0)
- fix(M4.6): correct trajectory/times length mismatch in behavioral segmentation notebook (edde6fc)
- fix(M3.1): remove internal min peak rate filter (ceea9aa)
- fix(M1.4): correct Laplacian verification to use weighted NetworkX comparison (4b7a312)

### Documentation
- docs(examples): finish M7.4 — wire _style.py into notebooks 01-23 (52e2044)
- docs: act on ux-reviewer findings on PR #2 (1c800eb)
- docs(encoding): fix dangling See-Also in object_vector_score (25285ec)
- docs(M6,M7): act on suggestion-tier PR review findings (abe6de9)
- docs(M6,M7): act on three-agent PR review of b30e9e1 (b430e2b)
- docs(M6): add notebooks 24-27 (OVC, HD, PSTH, NWB) (6757eff)
- docs(M6.11): add Prerequisites lines + convert bold notebook refs to MD links (1f5d7fc)
- docs(M6): land M6.5–M6.10 and M6.12 — docs infrastructure + bandit skip (9f99d94)
- docs(M5): correct differential-operator docstrings + composite parity claim (1609cc2)
- docs(M5): close reviewer-flagged gaps (0b848e3)
- docs(M5.4): add T-maze, plus-maze, and linear-track examples to from_graph (3258d56)
- docs(M4.3): document heading-convention on every public heading consumer (7a211da)
- docs(M2.C): add directional-exception note to compute_directional_rates (9d467af)
- docs(M2): fix border_score arg order in stale public docs (81b2a2d)
- docs(M2): close M2.A/M2.B review-response findings (9 items) (f5ad3bf)
- docs(M1): close out doc drift left by review-response bundle (62b837d)
- docs(plan): mark M1 1.1, 1.2, 1.3 complete in TASKS.md (1e58b0b)
- docs(plan): mark M1 1.4 and 1.5 complete in TASKS.md (1d21413)
- docs(plan): mark M1 1.6/1.7/1.8 complete in TASKS.md (6423a8b)
- docs/ci: address second-pass review (multi-line +SKIP, README placeholder snippet) (bf44732)
- docs/ci: address M0 code-review findings (H1-H3, I1, I3, I4, M2, M4) (5bedcab)
- docs(plan): mark M0 tasks complete in TASKS.md (0c523ee)
- ci: add curated doc-snippet smoke test (M0 0.10) (b9a4df8)
- docs/examples: regenerate notebook 15/20 outputs, fix QUICKSTART bin_at example (M0 0.3 follow-up, 0.8) (842e8c6)
- docs(examples): rewrite examples/README and place notebook 21 in canonical home (M0 0.6, 0.7) (76a236d)
- docs: fix broken first-run snippets in README, QUICKSTART, package docstring (M0 0.1-0.5) (a30dc4e)
- docs(plan): add v0.4 UX cleanup plan, tasks, and source review (98831fc)
- test: fix Python-3.10/11/12 docstring tests and Windows test failures (84e64a4)
- docs: align README, CLAUDE.md, and QUICKSTART with current encoding API (e0b5e9e)
- docs/tests: clean up post-deletion legacy references (2055987)
- docs(M6): add legacy parity tests as xfail; record M6 hold in plan (e452f45)
- docs: update SCRATCHPAD.md with Milestone 7 completion notes (19296b1)
- docs(M7): mark Task 7.6 complete (full test suite passes) (9446be9)
- docs(M7): update QUICKSTART.md with new encoding API (8ac2bc1)
- docs(M7): update documentation with new encoding API (730d9e7)
- docs(M7): update example notebooks to new encoding API (99e1f73)
- docs(M7): mark Task 7.2 migration subtasks as complete (ac4a22e)
- docs(M6): mark Task 6.6 complete - JAX-specific tests verified (5d058e7)
- docs(M6): skip Task 6.3 - JAX grid/border score not feasible (a57d103)
- docs(M4): verify comprehensive test coverage for view encoding (Task 4.9) (63e449f)
- docs: update TASKS.md and SCRATCHPAD.md for Task 4.2 completion (56592c7)
- docs(encoding): mark Task 2.10 and Milestone 2 as complete (8213a97)
- docs(M14): complete final verification for package reorganization (dae73e2)
- docs: add doctest skip annotations and fix docstring examples (59dfe73)
- docs(M12): mark milestone 12 complete (b617308)
- docs(M12): update remaining notebooks with new import paths (e008efc)
- docs(M12): update example notebooks with new import paths (9db19a7)
- docs(M11): mark milestone 11 complete (cddaa17)
- docs(M11): update import paths in documentation and docstrings (f07d097)
- docs: update TASKS.md and SCRATCHPAD.md for Milestone 10 (db0e194)
- docs(M2): mark all Milestone 2 tasks as complete (45e5f6e)
- docs(M2): update progress for transforms.py migration (7c791a3)
- docs(M3.6): complete spatial view cells integration and documentation (65c2001)
- docs(M3.5): verify all spatial view cell tests passing (f1f0d0b)
- docs(M1.5): add egocentric reference frames documentation (784ec6c)
- docs: add circular basis functions to QUICKSTART and API_REFERENCE (41878df)
- docs(M7): add basis functions to QUICKSTART and API_REFERENCE (24b76eb)
- docs(M5.6): enhance circular basis docs with multi-domain examples (72dcf0a)
- docs(profiling): add napari performance profiling guide (a99437a)
- docs(M8): update CLAUDE.md with speed-based animation API (bbb1adb)
- docs: update TASKS.md to reflect actual completion status from code inspection (224f369)
- docs(M4): update Environment.animate_fields() signature and docstring (e026aa0)
- docs: mark Milestone 3 tasks complete in TASKS.md (40a1fac)
- docs: update SCRATCHPAD with Milestone 1 & 2 completion status (2a9d9f6)
- docs(animation): add TimeSeriesOverlay documentation to CLAUDE.md (30e7c89)
- docs(decoding): add animate_fields guidance to plot() docstring (3abad65)
- docs(M6/M7): complete example scripts & documentation updates (72144d0)
- docs: mark Milestone 4 as complete in TASKS.md (f6906ba)
- docs(scale_bar): add scale bar examples to CLAUDE.md (M6) (1a4eec7)
- docs: add directional place fields example and CLAUDE.md reference (Tasks 6.1-6.2) (77239db)
- docs(Task 4.4): add docstrings and See Also sections (e33e5a5)
- docs(Task 4.3): add track graph annotation to CLAUDE.md (f8754cc)
- docs: mark M5.4 complete - all code quality checks pass (1807d2f)
- docs: update progress tracking for M5.3.1 completion (60b82df)
- docs(behavioral): update CLAUDE.md with v0.8.0 features (c109df1)
- docs: update SCRATCHPAD with review completion status (4232228)
- docs(behavioral): address code and UX review feedback (6b2dfc4)
- docs(behavioral): update TASKS.md - M4.2 complete (e4eda24)
- docs(behavioral): update TASKS.md - M4.1 complete (acd9ab1)
- docs: update TASKS.md - M3.2 complete (2ded57e)
- docs: update TASKS.md - M1.1 complete (e805291)
- docs(animation): fix griffe warning in OverlayProtocol docstring (9e6ebfe)
- docs(M5): update documentation for Trial.end_region field (fdb4e06)
- docs(nwb): complete M4.4 final verification - all milestones done (a9d76f8)
- docs(annotation): mark verification checklist complete and update scratchpad (b510d7b)
- docs(annotation): add annotation feature to CLAUDE.md (8b40c20)
- docs: update tracking files with final verification status (c18d666)
- docs: fix calibrate_from_landmarks parameter names in CLAUDE.md (a4622a0)
- docs(8.8): improve animate_fields docstring with multi-field mode (d9d726c)
- docs(8.6): document overlay JSON schema for HTML backend (6d2f7fc)
- docs(8.4): document coordinate conventions for video overlay (c7b70ea)
- docs(7.3): update animation guide with VideoOverlay documentation (aa3e71b)
- docs(7.2): create video overlay example notebook (4ce0981)
- docs(7.1): update CLAUDE.md with VideoOverlay documentation (f9388f3)
- docs(4.1): mark napari backend task complete (6392d7f)
- docs(UX): improve error messages and documentation consistency (37a9a8b)
- docs(P7.3): add backend selection guide and Skeleton class documentation (87ec8c6)
- docs(P7.1): verify unit tests already exist and pass (a103be1)
- docs(P5.4): complete video re-profiling - no performance regression (d880416)
- docs(P4.4): complete widget re-profiling - 1.8-2.2x speedup verified (27c5dad)
- docs(P2.5): re-profile napari - document 42-50x skeleton speedup (fc9681c)
- docs(P2.3): verify tracks color_by workaround is correct for napari 0.5.6 (ad6ca4b)
- docs(P1.3): verify no regressions after Phase 1 changes (340f486)
- docs: exclude research directory from build to fix CI (23120dd)
- docs(animation): fix broken links and add to navigation (86f3944)
- docs: mark code review complete in TASKS.md (f5d9269)
- docs: mark docstring review complete in TASKS.md (89cb31a)
- docs(animation): convert backend docstring examples to code-block format (c9c2da1)
- docs: add Session 25b docstring review progress (da0b3de)
- docs(animation): fix docstring examples to follow NumPy format (ee81505)
- docs: mark Error Message Review tasks complete in TASKS.md (623d663)
- docs(animation): add Session 25 error message review summary (d09cf0d)
- docs(animation): document subsample memory behavior investigation (dd2e611)
- docs(animation): update benchmark results with larger sample sizes (cd1ba2f)
- docs(animation): verify error message tests complete (824aa66)
- docs: add Session 19 notes for documentation enhancements (a5d901e)
- docs(napari): update render_napari docstring with multi-field features (02c6cfd)
- docs(animation): add multi-field viewer example and enhance documentation (d82b60a)
- docs(animation): update README with animation feature (6b4dc96)
- docs(animation): create comprehensive user guide (aa6e40e)
- docs(animation): update CLAUDE.md with animation documentation (57fe991)
- docs: add Milestone 7.5 (Enhanced Napari UX) to planning documents (dafdefc)
- docs(napari): correct playback control documentation (8a44a99)
- docs: mark milestones 1 and 2 complete and add HTML demo (fb93742)
- docs: update scratchpad with milestone 1 completion status (a68b975)
- docs: add notebook update steps to Milestone 3 tasks (5bb614d)
- docs: mark Milestone 2.2 as skipped (344d050)
- docs(M4): mark pathlib tasks 4.2-4.5 complete - already implemented (4dfa1b5)
- docs(M3): update task tracking - keep dual API decision (6e077db)
- docs(M3): clarify bin_at() vs map_points_to_bins() semantics (1aec606)
- docs(M2): update SCRATCHPAD with Tasks 2.8-2.12 completion (65d75bd)
- docs(M2): Task 2.7 complete - Connected Components investigation (09d574c)
- docs(M2): Document NetworkX incidence_matrix investigation (a6acf08)
- docs(M2): Task 2.4 complete - Laplacian investigation (KEEP CURRENT) (576c3ec)
- docs(M2): add scipy investigation report for Task 2.1 (b0dcfab)
- docs(M1): complete coverage audit for distance.py (Task 1.1) (aaf2d91)
- docs(M2): add comprehensive module-level docstring to __init__.py (Task 2.9) (92900c5)
- docs: mark Task 2.2 as completed in TASKS.md (22850fd)
- docs: mark Task 1.8 as completed in TASKS.md (500d2d9)
- docs: update task tracking for completed task 1.5 (93d1f71)
- docs: mark Task 1.4 as completed in TASKS.md (76c061b)
- docs: update task tracking for completed task 1.1 (3bbb7c5)
- fix(docs): sync examples/11_place_field_analysis with Part 9 additions (00c1588)
- docs: add learning objectives to notebooks 9-15 for pedagogical consistency (c27c53f)
- docs(M3.5): complete comprehensive pedagogical validation of notebook suite (99df3e0)
- docs(M3.5): complete validation of simulation notebooks and code quality (6879644)
- docs(M3.5): add simulation subpackage section to API reference (fec0bb0)
- docs(M3.5): add comprehensive Simulation section to README.md (fe81421)
- docs(M3.5): verify all documentation links work correctly (5c19ceb)
- docs(M3.5): update mkdocs config and examples index for all 15 notebooks (9cdf9db)
- docs(M3.5): sync all notebooks from examples/ to docs/examples/ (b5ca267)
- docs: document notebook 08 coverage investigation and 100s duration fix (007049b)
- docs: document OU process critical bug fix in SCRATCHPAD (455fc91)
- docs(M3.5): create comprehensive simulation workflows tutorial notebook (7565e7f)
- docs(M3.5): update 12_boundary_cell_analysis.ipynb to use simulation subpackage (fdcdee4)
- docs(M3.5): update 08_spike_field_basics.ipynb to use simulation subpackage (43eed0e)
- docs: update SCRATCHPAD.md with notebook execution verification (12e83d0)
- docs(M3.5): update 11_place_field_analysis.ipynb to use simulation subpackage (624304b)
- docs(M3): update SCRATCHPAD and TASKS for SimulationSession completion (b193f45)
- docs(M2): complete Milestone 2 documentation and validation (f19132b)
- docs: update SCRATCHPAD.md to reflect Milestone 1 completion (bac5f1d)
- docs: fix broken API reference links (faedcb6)
- docs(M4.6): add comprehensive trajectory and behavioral analysis user guide (c9b20e5)
- docs(M3.4): add boundary cell analysis notebook (c0a8858)
- docs(M3.4): add place field analysis notebook (9fb4040)
- docs(M3.4): add comprehensive neuroscience metrics documentation (ddec7bb)
- docs(M2.3): add comprehensive signal processing primitives documentation (0c21a1d)
- docs: document Laplacian verification bug fix in SCRATCHPAD (6ad54d0)
- docs(M1.4): add comprehensive differential operators documentation and examples (40e99df)
- docs(M0): mark all Milestone 0 success criteria complete (e5425b1)
- docs(M0.3): fix plotting errors and enable notebook execution (1fd85d6)
- docs(M0.3): add comprehensive user guides for Phase 0 primitives (2e609f2)
- docs: update README for v0.2.0 release (826d5ad)
- docs: update CHANGELOG.md for v0.2.0 (580b8eb)

### Other Changes
- test(animation): skip PlaybackController tests when no Qt binding present (c7c2808)
- chore(release): prep v0.4.0 (CHANGELOG, version bump, _style, smoke) (b30e9e1)
- test(encoding): act on four-agent PR review of d864b6d (cb25580)
- test(encoding): simplify backend dispatch tests (review fixes) (d864b6d)
- test(encoding): close three reviewer findings on JAX guards and plot ax assertions (cont.) (15cd7bd)
- test(encoding): JAX-skip the remaining unguarded test + assert array type + cover backend="auto" (019fab0)
- test(encoding): close three reviewer findings on JAX guards and plot ax assertions (5700c26)
- test: address remaining best-practice gaps (test-order, weak tests, properties) (80e95cf)
- test(environment,nwb,simulation): strip env-attr / setup-recheck / docstring-only tests (batch 21) (27729a6)
- test(animation,visualization,regions): trim weak/mocked/setup-recheck tests (batch 20) (8ead868)
- test(encoding): strip import/dataclass/dtype tautologies across 7 files (batch 19) (f77bf87)
- test(ops): strip meta-tests on module exports and dataclass field echoes (batch 18) (d543926)
- test(segmentation): strip parameter-order tautologies (batch 17) (600c4ec)
- test(behavior): strip dataclass / summary-string / one-line-method tests (batch 16) (99f19f3)
- test(events): strip setup re-checks and return-type tautologies (batch 15) (910382a)
- test(decoding): strip return-type tautologies and success-criteria scaffolding (batch 14) (084d062)
- test(decoding): strip 34 copy-paste shuffle boilerplate methods (batch 13) (479df6a)
- test(decoding,encoding): drop @cached_property tautologies and plot-axes-not-None duplicates (batch 12) (d2d9076)
- test(mazes): strip per-maze dataclass-trivia classes (batch 11) (88e7697)
- test(encoding): trim tautological imports and duplicate _to_numpy variants (batch 10) (04e3094)
- test(ops): trim import-shim and __all__-pinning tests (batch 9) (caebf93)
- test(stats): remove import tautologies and consolidate circular tests (batch 8) (0a0155d)
- test(animation): remove self-referential constant/formula tests (batch 7) (f1f40ab)
- test(encoding): collapse 24 backend-dispatch passthrough tests to one parametrized validator (batch 6) (4636244)
- test: collapse top-level import-test files (batch 5) (1efd3fd)
- test: delete five more duplicate TDD-scaffolding test files (batch 4) (9483217)
- test: delete duplicate, import-only, and print-only test files (batch 3) (6c8d9ad)
- test: collapse parameter/path/layout-kind permutations (batch 2) (da10542)
- test: remove tautological and cosmetic-formatting tests (batch 1) (e9360cc)
- docs(M5): correct differential-operator docstrings + composite parity claim (1609cc2)
- docs(M5): close reviewer-flagged gaps (0b848e3)
- chore(M5): tick M5 checkboxes in TASKS.md (dee6b3e)
- refactor(M5.9): remove Environment.save / Environment.load (d0cf6e3)
- refactor(M5.7): remove Environment.mask_for_region in favor of region_mask (dce5cfb)
- refactor(M5.6): convert bin_attributes / edge_attributes / differential_operator to methods (c48d0ee)
- refactor(M5.5): finish Regions API consolidation by raising in remove() (9074804)
- refactor(M5.3): rename from_image/from_mask -> from_pixel_mask/from_grid_mask (6ae8ee1)
- chore(M4): tick M4 checkboxes in TASKS.md (7d90e66)
- refactor(M4.4, M4.5): coordinate-convention safety part 2 (2d5ec4c)
- refactor(M4.1, M4.2): coordinate-convention safety part 1 (134c57d)
- chore(M3): tick M3 checkboxes in TASKS.md (c11d4fc)
- refactor(M3.5-M3.9): error/warning hygiene sweep (c1eea2d)
- refactor(M3.3): expand @check_fitted coverage on Environment methods (3d64e21)
- refactor(M3.2): migrate manual not-fitted checks to EnvironmentNotFittedError (65221a0)
- refactor(M2.C): finish 2.19 + 2.15; tick M2.C/M2.D in TASKS.md (88a6f7d)
- refactor(M2.D): convert events.__init__ lazy __getattr__ to eager imports (7fbbe94)
- refactor(M2.D): remove cross-domain re-exports for one canonical import path (c6360d3)
- refactor(M2.D): remove path_efficiency, keep compute_path_efficiency (8ad0d24)
- refactor(M2.C): force kwarg-only on settings parameters across the API (751a591)
- refactor(M2.C): is_object_vector_cell takes raw data like sister classifiers (f956804)
- refactor(M2.C): align argument order with canonical conventions (5fe5886)
- test(properties): loosen rate_map_coherence smooth-field bound (08e37df)
- fix(M2): README CI snippet + batch wrappers reject bad API params (22736ac)
- refactor(M2.B 2.13): misc result-type cleanup (7ceaecc)
- refactor(M2.B): result-class consolidation (renames + structured returns) (53a4c66)
- refactor(M2.A): unify M2 parameter names across the public API (0e7aea8)
- feat(environment): mark polar envs and refuse Cartesian-assuming methods (M1 1.3) (7377271)
- feat(decoding): default decode_position to validate=True; add non-negative spike_counts and rate checks (M1 1.8) (8143baa)
- docs/ci: address second-pass review (multi-line +SKIP, README placeholder snippet) (bf44732)
- docs/ci: address M0 code-review findings (H1-H3, I1, I3, I4, M2, M4) (5bedcab)
- test: fix Python-3.10/11/12 docstring tests and Windows test failures (84e64a4)
- test: gate optional-extra tests on their actual dependencies (26efb25)
- docs/tests: clean up post-deletion legacy references (2055987)
- test(encoding): make x64-import regression actually fail without the fix (927d819)
- refactor(M4): symmetric NumPy/JAX backends; _metrics.py is dispatch only (561cc8b)
- refactor(M2): inherit SpatialResultMixin and vectorize directional batch ops (0a028d6)
- refactor(M1): unify input validation and fix egocentric arg order (14be540)
- refactor(M7): update remaining test imports to new module locations (3ea4b43)
- refactor(M7): clean up place.py and update test imports (3a10db3)
- refactor(M7): update encoding/__init__.py exports for new locations (e1e9a64)
- refactor(M7): migrate object_vector.py utilities to egocentric.py (32cf53e)
- refactor(M7): migrate spatial_view.py utilities to view.py (2f64b0d)
- refactor(M7): migrate head_direction.py utilities to directional.py (1462788)
- refactor(M7): migrate DirectionalPlaceFields and detect_place_fields to spatial.py (16b2a1b)
- refactor(M6): consolidate JAX helpers and add backend validation (00c83f6)
- refactor(M10): consolidate metrics modules into encoding (4303194)
- refactor(M10): delete old files and consolidate encoding modules (21d7222)
- refactor(M9): reduce top-level exports to 5 core classes (4c87ccf)
- refactor(M8): consolidate animation/ module - move scale bar config (807dbe1)
- refactor(M7): add re-exports from stats to decoding module (cd90a47)
- refactor(M6): update internal imports for encoding modules (cf02f62)
- refactor(M6): create encoding/population.py with population metrics re-exports (f40493b)
- refactor(M6): create encoding/phase_precession.py with phase precession re-exports (ebf9fc2)
- refactor(M6): create encoding/spatial_view.py with spatial view cell re-exports (1fc8145)
- refactor(M6): create encoding/object_vector.py with object-vector cell re-exports (4b9d129)
- refactor(M6): create encoding/border.py with border cell re-exports (494b28b)
- refactor(M6): create encoding/head_direction.py with head direction re-exports (7b9a8f2)
- refactor(M6): create encoding/grid.py with grid cell re-exports (3b123c4)
- refactor(M6): create encoding/place.py with place cell re-exports (5dfcdcc)
- refactor(M5): move reward.py to behavior/reward.py (f887c20)
- refactor(M5): create behavior/decisions.py with decision_analysis and VTE (1bfb0c4)
- refactor(M5): create behavior/navigation.py with path and goal metrics (e2ea4e7)
- refactor(M5): move segmentation/ to behavior/segmentation.py (e4d94cc)
- refactor(M5): move trajectory functions to behavior/trajectory.py (355a337)
- refactor(M2): remove backward compatibility shims (246fb63)
- refactor(M4): create stats/surrogates.py with surrogate generation functions (e808d0f)
- refactor(M4): move shuffle functions to stats/shuffle.py (89a588a)
- refactor(M4): move circular statistics to stats/circular.py (ff00e6e)
- refactor(M3): move nwb/ → io/nwb/ (20a2ecc)
- refactor(M2): move basis.py → ops/basis.py (63616af)
- refactor(M2): move visibility.py → ops/visibility.py (285a6c2)
- refactor(M2): move reference_frames.py → ops/egocentric.py (75243fb)
- refactor(M2): move alignment.py → ops/alignment.py (95cd446)
- refactor(M2): move transforms.py → ops/transforms.py (d2514de)
- refactor(M2): move differential.py → ops/calculus.py (f5bd05b)
- refactor(M2): move primitives.py → ops/graph.py (f92ed86)
- refactor(M2): move kernels.py → ops/smoothing.py (9248e66)
- refactor(M2): move field_ops.py → ops/normalize.py (7776cda)
- refactor(M2): move distance.py → ops/distance.py (4ae7995)
- refactor(M2): move spatial.py → ops/binning.py (770fa0a)
- refactor(M1): create new package directory structure (d0a9415)
- chore: defer M3.5-M3.8 spatial event utilities (dcdd2a6)
- chore: defer M2.4 exponential_kernel and M3.1-M3.3 detection functions (bb01fde)
- refactor(M2.1): replace time_since_event with time_to_nearest_event (67ecc2d)
- chore: update TASKS.md with completed milestone checkboxes (1727d72)
- chore: update SCRATCHPAD with completion status (f42fba2)
- test: add behavioral tests verifying plot_circular_basis_tuning math (671b26e)
- refactor: extract phase_precession module from circular.py (2b892f6)
- test(animation): remove skip markers for Task 3.2 (now implemented) (24ca1e9)
- refactor(animation): add _validate_frame_times function (8af7335)
- test(animation): add final integration tests for TimeSeriesOverlay (d8ae692)
- test(animation): add performance and edge case tests for TimeSeriesOverlay (8bf771c)
- refactor(M2): remove dask renderer after benchmark showed poor performance (38e952a)
- refactor(scale_bar): DRY helper, validation, and edge case tests (5cdc999)
- chore(Task 4.5): complete final review and testing (3fac478)
- test(annotation): complete Task 3.4 - end-to-end tests for track graph (37b8b74)
- test(annotation): complete Task 2.4 - widget integration tests (4563fde)
- chore(tests): complete Milestones 6 and 8 (fb91de4)
- refactor(tests): remove duplicate fixtures from test files (c2e0a8c)
- refactor(tests): move animation fixtures to local conftest (c03dcbd)
- refactor(tests): make simple_3d_env fixture deterministic (888ae95)
- refactor(tests): migrate animation tests to local RNG (b2ed549)
- refactor(tests): migrate segmentation tests to local RNG (73af5ef)
- refactor(tests): migrate test_population.py to local RNG (9194837)
- refactor(tests): migrate test_trajectory.py to local RNG (d3e6365)
- refactor(tests): migrate test_grid_cells.py to local RNG (1a44f86)
- chore: update TASKS.md with test_behavioral.py migration status (0b2b7ac)
- refactor(tests): migrate test_behavioral.py to deterministic fixtures (e9a1948)
- refactor(tests): migrate test_transforms_new.py to local RNG (7587fb3)
- refactor(tests): migrate test_differential.py to local RNG (7d07063)
- refactor(tests): migrate test_io.py to local RNG (fc39766)
- refactor(tests): migrate test_trajectory_metrics.py to local RNG (b62681b)
- refactor(tests): migrate test_interpolate.py to local RNG (363a697)
- refactor(tests): migrate test_validation_new.py to local RNG (fd9029f)
- refactor(tests): migrate test_transforms_3d.py to local RNG (1402b57)
- refactor(tests): migrate test_spike_field.py to local RNG (ef0b160)
- refactor(tests): migrate test_place_fields.py to local RNG (96a16fd)
- refactor(tests): reduce from_samples calls in test_transitions.py (3f6aba6)
- refactor(tests): reduce from_samples calls in test_occupancy.py (3071612)
- refactor(tests): reduce from_samples calls in test_boundary_cells.py (96052b7)
- test(behavioral): add coverage improvement tests (2161653)
- test(behavioral): add comprehensive curvature tests for multiple turns (c426c5c)
- test(nwb): add comprehensive parametrized tests for all layout types (2a995aa)
- test(nwb): add round-trip tests for all remaining layout types (806a78a)
- test(nwb): add file-based round-trip tests for Environment (246181d)
- test(nwb): add linked timestamps test for pose reading (a182672)
- test(nwb): add skeleton round-trip tests (M1.2.2) (67e45d2)
- test(nwb): add core discovery utilities tests (M1.1.1) (abd6e5c)
- test(nwb): add test infrastructure with NWB fixtures (M0.3) (0cd49cc)
- chore(nwb): add optional NWB dependencies (M0.2) (8710f75)
- chore(annotation): remove non-working features from help text (ae93477)
- chore(annotation): remove non-working UX features and update shortcuts (0c0525e)
- test(8.6): add TypedDict JSON schema conformance tests (f0eac94)
- chore: update verification status and code review results (44d9d60)
- refactor(8.3): centralize pickle validation messages (2cfdc62)
- refactor(8.2): normalize OverlayData.regions to dict format (96f1bf4)
- test(6.5): add shared environment fixtures for video overlay testing (20cc64b)
- test(6.4): add video validation and napari-free operation tests (a922ef4)
- test(P5.1): add frame naming pattern tests (5ac437f)
- refactor: move benchmark_datasets to scripts/ directory (5b23c41)
- chore(M4): remove dead skeleton callback code (5c0af5d)
- chore(animation): remove legacy overlay_trajectory tests (542c154)
- test: remove backup file and mark slow tests to prevent CI timeouts (1cbcbeb)
- test(memory): remove redundant test_very_large_environment_only_warns (bfbadad)
- chore(animation): remove unnecessary type ignore comments (19cf867)
- test(animation): add comprehensive error message tests (99a44ac)
- test(animation): add memory profiling tests (c5bb173)
- test(animation): increase benchmark sample sizes for robust baselines (ded4c69)
- test(animation): add comprehensive performance benchmarks (939270c)
- test(animation): add Session 20 integration tests and Pillow dependency (c4652eb)
- test(animation): add end-to-end layout integration tests (fbfdd0e)
- refactor(animation): apply code review recommendations (4bfe66f)
- chore(M2): cleanup investigation files and update task tracking (d343ec3)
- test(M1): achieve 95% coverage for place_fields.py (Task 1.4 complete) (bb52c0e)
- test(M1): achieve 100% coverage for kernels.py (Task 1.3 complete) (1355731)
- chore(M1): achieve 100% coverage for differential.py via dead code removal (Task 1.2 complete) (d3fac33)
- test(M1): achieve 100% coverage for distance.py (Task 1.1 complete) (735697a)
- chore(M2): remove dead code from utils.py (Task 2.12) (1520a91)
- refactor(M2): extract helper functions from _create_regular_grid() (Task 2.7) (c22f8c3)
- test(M2): add property-based tests with Hypothesis (Task 2.4) (3ca45bd)
- test(perf): complete performance regression test suite (Task 2.3) (f7e555d)
- refactor(M2): extract generic graph connectivity helper (Task 2.1) (7edf6e0)
- test: add comprehensive 3D environment test coverage (8d338cd)
- fix(trajectory): improve numerical stability with named EPSILON constant (d320438)
- chore: re-execute notebook 08 with seed=137 to update embedded outputs (a32da4e)
- docs(M2): complete Milestone 2 documentation and validation (f19132b)
- refactor: replace fragile type name checks with _layout_type_tag (6153ebb)
- refactor: reduce nesting with guard clauses in spike_field (7805e1b)
- refactor: standardize parameter order - env first in all functions (576cc14)
- chore(M4.6): update progress tracking for completed documentation (0a80dd0)
- test(M4.6): add comprehensive integration tests for trajectory analysis workflows (85bafd5)

**Full Changelog**: https://github.com/edeno/neurospatial/compare/v0.2.0...v0.4.0

## [0.4.0] - 2026-05-21

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

## [v0.2.0] - 2025-11-04

## What's Changed

### Features

- feat(P3.15): implement deterministic KDTree with distance thresholds (7ef1109)
- feat(P3.14): implement Environment.copy() method (17a1d0e)
- feat(P3.13): implement distance utilities (distance_to and rings) (2e07dce)
- feat(P3.12): implement Environment.region_membership() for vectorized region containment (a02d926)
- feat(P2.11): implement linear time allocation for occupancy (d1e3690)
- feat(P2.10): implement field math utility functions (9f8e53b)
- feat(P2.9): implement Environment.interpolate() for field evaluation at points (160da81)
- feat(P1.8): implement Environment.subset() for bin selection and cropping (7269e86)
- feat(P1.7): implement Environment.rebin() for grid coarsening (d841d3d)
- feat(P1.6): implement Environment.smooth() for field smoothing (dc4ce39)
- feat(P0.4): implement connected components and reachability methods (7a1d75f)
- feat(P0.3): implement Environment.transitions() for empirical transition matrices (a9f3beb)
- feat(P0.2): implement Environment.bin_sequence() for trajectory-to-bin conversion (f0e008b)
- feat(P0.1): implement Environment.occupancy() for time-in-bin computation (28ccae0)
- feat(kernels): implement diffusion kernel infrastructure (Phase 1) (ad5f23f)
- feat(ci): add manual workflow dispatch to publish workflow (703ed86)

### Bug Fixes

- fix(lint): resolve ruff errors in example notebook (6b69b2e)
- fix(GraphLayout): support 1D graph layouts (26b8abc)
- fix(P2.11): combine nested if statements, apply ruff format (5bddcdf)
- fix(P0.3): add parameter validation for transitions() method (dad0f96)

### Documentation

- docs: mark all Environment Operations tasks complete (46e65ab)
- docs: add jupytext pairing and track all example notebooks (a341e36)
- docs: mark P3.15 (Deterministic KDTree) as complete (eda4f9b)
- docs: update CHANGELOG.md for v0.1.0 (84e8a46)

### Other Changes

- chore: remove completed project management files (66666e7)
- test: fix disconnected graph tests using systematic debugging (8dc6de6)
- refactor(test): remove untestable unfitted Environment check_fitted test (03b3722)
- refactor(test): remove untestable 1D graph region_membership test (a20bbbd)
- test: implement hexagonal layout interpolation test (233f501)
- chore: mark public API export task complete in TASKS.md (9fbb8eb)
- chore: sync notebooks with formatting changes from pre-commit (f8b7528)
- refactor(P0.3): add unified transitions() interface with model-based methods (d27a1e9)

**Full Changelog**: <https://github.com/edeno/neurospatial/compare/v0.1.0...v0.2.0>

All notable changes to the neurospatial project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
