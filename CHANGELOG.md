# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0, minor releases may still include breaking changes;
these are called out under a dedicated **Breaking changes** heading.

## [Unreleased]

### Added

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
  - When `min_speed` excludes **all** trajectory intervals (e.g. a wrong-units
    threshold), `compute_spatial_rate` / `compute_spatial_rates` now emit one
    `UserWarning` naming `min_speed` and the resolved speed range — instead of
    silently returning an empty rate map. The warning fires once per call
    (not per neuron in the batch path) and is suppressed by `warn_on_drop=False`.

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
  - `time_chunk=k` computes the exp/normalize in time-blocks into a single
    preallocated output array, materializing the full-size `ll_shifted`
    temporary one block at a time instead of all at once. This drops the
    transient memory peak from ~4× to ~3× the stored posterior — the full-size
    log-likelihood and its working copy still coexist with the output, so it is
    not a ~1× path (that is `decode_position_summary`, which computes the
    likelihood per block and never holds the full log-likelihood).
    `time_chunk=None` (the default) reproduces the previous behavior
    byte-for-byte.

  For sessions where even the stored dense posterior is too large to hold, use
  the new `decode_position_summary` / `decode_session_summary` (see **Added**),
  which never materialize the full `(n_time, n_bins)` posterior.

- `compute_spatial_rates` gains a keyword-only `dtype` (`np.float32` /
  `np.float64`, default `np.float64`). `dtype=np.float32` halves the stored
  `(n_units, n_bins)` rate-map array. The rate computation (GEMM / division) is
  still performed in float64 and only the final result is cast, so float32
  values match the float64 default within float32 tolerance. Note:
  `decode_session` currently re-materializes encoding models as float64
  internally, so passing a float32 `SpatialRatesResult.firing_rates` there does
  not by itself shrink the decode working set; the decode-side memory knobs are
  `decode_position(..., dtype=..., time_chunk=...)` and
  `decode_position_summary`. Default `np.float64` leaves every existing caller
  byte-for-byte unchanged; any other dtype raises `ValueError`.

- Documented the dense diffusion-kernel **O(n²) memory cost** and added a hard
  high-bin **safety gate**. The heat kernel `exp(-tL)` of a connected graph is
  dense by construction (every entry > 0), so it always costs
  `n_bins**2 * 8` bytes of float64 memory (≈ 3.2 GB at 20,000 bins).
  `compute_diffusion_kernels` and `env.compute_kernel` now **raise**
  `MemoryError` above 20,000 bins, turning a silent out-of-memory crash into an
  actionable error that names the size, the GB estimate, and the fixes (reduce
  bins, use `smoothing_method="binned"`, or pass `allow_large=True` to override
  if you have the RAM). The existing `UserWarning` above 3,000 bins is kept.
  No numerical results change — this is documentation plus a default-refuse
  gate with an explicit opt-out. The reliable scale wins this release remain
  float32 rate maps and the memory-safe summary decode (above). A faster,
  lower-peak `expm_multiply` / Chebyshev rewrite of the kernel is a **deferred
  stretch goal** and is intentionally **not** part of this release.

- Gated the dense **Gaussian-KDE** high-bin path the same way as the diffusion
  kernel. `smoothing_method="gaussian_kde"` builds a dense `(n_bins, n_bins)`
  weight matrix (`exp(-d²/2σ²)`, every entry > 0), so it carries the same
  O(n_bins²) memory cost; it now **raises** `MemoryError` above the shared
  `_KERNEL_HARD_LIMIT_BINS` ceiling (20,000 bins) before allocating the matrix,
  naming the size, the GB estimate, and the fixes (reduce bins, use
  `smoothing_method="binned"`, or `allow_large=True`). One ceiling constant is
  now shared by both dense smoothing paths instead of a divergent copy. Also
  corrected stale `encoding/_smoothing.py` docs that wrongly described the
  diffusion kernel as "sparse" / `O(n_bins)` — it is dense `(n_bins, n_bins)`,
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

## [0.5.0] - 2026-06-04

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

### Changed

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

### Fixed

- `gaussian_kde` smoothing could return a kernel computed for a *different*
  environment. The internal Gaussian-kernel cache keyed on `(id(env),
  bandwidth)` and validated only `n_bins`; since `id()` is unique only among
  live objects, an environment built after an earlier one was garbage-collected
  could reuse its address and silently receive the stale kernel when `n_bins`
  matched (e.g. two `from_polar_egocentric` envs differing only in
  `circular_angle` — the open-axis env would get the circular env's
  seam-wrapped kernel). The cache now also holds a weakref to the owning env and
  treats any `ref() is not env` (id reused or env collected) as a miss, so a
  recycled id can never serve another env's kernel.
- `DirectionalPlaceFields.to_dataframe()` no longer crashes on an empty result.
  When `compute_directional_place_fields` excludes every sample as `"other"`
  (so `labels == ()`), the method previously built an empty `frames` list and
  called `pd.concat([])`, raising `ValueError: No objects to concatenate`. It
  now returns an empty `DataFrame` carrying the documented column schema
  (`direction`, `bin`, `coord_0…`, `firing_rate`, `occupancy`).
- `DirectionalRateResult.rayleigh_pvalue` now restricts the weighted Rayleigh
  test to **occupied** heading bins (positive occupancy and finite firing
  rate). Previously a bin the animal never occupied (zero occupancy → `NaN`
  rate) could still contribute its raw spike counts to the test and drive
  spurious significance (e.g. 100 spikes assigned to unvisited bins yielded
  `p ≈ 3.7e-44`); such cases now return `NaN` (insufficient valid bins). A
  genuinely-visited cell concentrated in 1–2 occupied bins remains significant.
- `DecodingResult.error_against` now reports `NaN` error for **undecodable**
  time bins whose posterior row is entirely non-finite (all-`NaN`/`Inf`).
  Previously `np.argmax` picked bin 0 for such rows, producing a finite (wrong)
  error; finite posterior rows are unaffected.
- `decode_position` now treats **`Inf`** encoding-model bins like `NaN` bins in
  its per-bin exclusion path (`validate=False`): a partial-`Inf` model such as
  `rates=[inf, inf, 5]` now concentrates posterior mass on the single finite
  bin instead of warning and returning a uniform posterior. The detection was
  broadened from `np.isnan` to `~np.isfinite`. The `validate=True` Inf
  rejection, the `fill_value=0.0` golden path, and the existing all-`NaN`/
  partial-`NaN` semantics are unchanged.
- `SpatialResultMixin.summary` (shared encoding result summary) no longer
  raises on an **empty** result (0 neurons or 0 bins). Previously the peak
  reduction over a zero-size array raised `ValueError: zero-size array to
  reduction operation fmax which has no identity` (e.g.
  `compute_directional_rates([], ...).summary()`); it now returns a dict with
  `n_neurons=0` and a `NaN` `peak_firing_rate`.
- Anisotropic pixel-mask environments now round-trip through
  `Environment.to_dict()`/`from_dict()` (and `to_file`/`from_file`) again. An
  env built with `from_pixel_mask(mask, pixel_size=(3.0, 1.0))` serialized
  `pixel_size` as a list; deserialization rebuilt it as an ndarray, and the
  scalar `pixel_size <= 0` validation in `ImageMaskLayout.build` then raised
  `ValueError: The truth value of an array with more than one element is
  ambiguous`. The validation is now array-safe
  (`np.any(np.asarray(pixel_size, dtype=float) <= 0)`) and covers both scalar
  and per-axis pixel sizes.
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
- Masked-grid occupancy is now routed correctly. `_allocate_time_linear`
  translates full-grid ray-intersection indices to active-bin ids through a
  prebuilt inverse map, so on masked (holed) grids occupancy time lands in the
  right bins instead of being silently misrouted or dropped.
  `Environment.interpolate` scatters active values into a `NaN`-filled full
  grid before reshaping, so it works on holed grids (was a reshape crash) and
  returns `NaN` over holes; `rebin` infers diagonal-vs-orthogonal connectivity
  from the active graph's max degree rather than probing a possibly-inactive
  full-grid center node; and `occupancy` validates finite timestamps before the
  monotonicity check (clearer error, no self-contradictory message).
- `MaskedGridLayout` rejects a non-array / non-boolean `active_mask`, and
  `get_n_bins` is computed in `float64` with an overflow guard before casting to
  `int64`, so large spatial extents no longer overflow `int32`. `GraphLayout`
  linear-point lookups remap gap-inclusive full-grid indices to active-bin
  indices, preserving the `-1` off-track sentinel.
- Environment file/dict I/O round-trips non-trivial `layout_parameters`:
  `networkx` graphs are encoded node-link and `shapely` geometries as WKT (and
  decoded on load), fixing a `to_file` crash for graph/polygon layouts and
  restoring graph `edge_order` on read. NWB `read_environment` round-trips
  `coordinate_kind` (defaulting to `cartesian` for legacy files), the NWB rate
  adapter rejects non-finite/non-positive rates, `read_position`/`read_pose`
  validate data/timestamp length agreement, and `read_head_direction` converts
  degrees to radians (and `(n, 2)` vectors via `arctan2`) under the documented
  `0 = East` convention.
- Decoding correctness fixes: `explained_variance_reactivation` now computes the
  true Kudrimoti role-swapped reversed EV and a control-aware
  (partial-correlation) EV instead of the inert `rev = ev`;
  `reactivation_strength` is magnitude-sensitive (shared template-baseline
  normalization rather than a per-period double z-score that pinned it near 1);
  `credible_region` excludes non-finite bins from the HPD set (including the
  low-mass fallback); `confusion_matrix` drops non-finite posterior rows with a
  warning; and `log_poisson_likelihood` requires a 2-D
  `(n_time_bins, n_neurons)` `spike_counts` with a matching neuron axis,
  rejecting the silent time-axis collapse. `decode_position` also validates the
  `encoding_models` bin count against `env.n_bins` and the `spike_counts` time
  length even when `validate=False`.
- `ops` robustness: `resample_field` (diffuse) zero-fills out-of-source bins
  before the kernel and re-masks after, so a single `NaN` no longer poisons
  every reachable bin; `heading_from_velocity` rejects non-positive/non-finite
  `dt` and non-finite positions; `_wrap_angle` is now half-open `(-pi, pi]`
  (keeping `+pi` at the antipode); `estimate_transform` applies the Kabsch
  determinant sign-correction so the fit is a proper rotation with no
  reflection; and several broad `except Exception` guards (KDTree mappers,
  spectral-radius estimation) were narrowed so real errors surface.
- `simulation`: `PlaceCellModel` rejects a non-positive/non-finite `width` at
  construction and no longer double-normalizes anisotropic distances (the
  isotropic firing-rate path is byte-for-byte unchanged);
  `generate_poisson_spikes` validates `firing_rate`/`times` length agreement and
  finiteness, requires strictly-increasing `times`, and uses a per-bin `dt` so
  non-uniform sampling is supported.
- `annotation`: boundary inference raises a `ValueError` (suggesting
  `convex_hull`) when the alpha shape is not a non-empty polygon instead of
  returning a degenerate geometry; `TrackBuilderState` node/edge deletion
  invalidates stale edge-layout overrides so they can no longer corrupt
  linearization; and `validate_region_overlap` warns and records an issue on a
  shapely `GEOSException` instead of aborting the whole validation.
- `regions`: RLE mask decoding validates each run's start/length against the
  mask size (rejecting negatives) instead of writing out of bounds;
  `region_center` returns `None` for an empty polygon instead of raising; and
  the remaining broad `_process_cvat_box` catch-all was narrowed to
  `(TypeError, ValueError, ShapelyError)` so real bugs (e.g. a `KeyError`)
  propagate while malformed coordinates still warn-and-skip.
- `behavior`: off-environment / `-1` "no bin" indices now resolve to an explicit
  fill (`False` or `-1`) across segmentation (crossings, runs, trials) and
  decision analysis instead of wrapping to the last bin/region, and finite-time
  plus positive-`dt` guards were added before the velocity/rate divisions in
  `detect_runs_between_regions`, `segment_by_velocity`, `approach_rate`, and
  `mean_square_displacement`.
- `animation`: overlay coordinate transforms allocate `float64` outputs so
  integer-dtyped inputs no longer truncate sub-pixel positions;
  `HeadDirectionOverlay` interpolates headings via `(cos, sin)` → `arctan2` (the
  short way across the `+/-pi` wrap); bounds diagnostics use NaN-safe
  `nanmin`/`nanmax`; and event-overlay artists are rendered/updated/cleared in
  the artist-reuse path so events appear in reused-figure animations.
- `encoding.is_head_direction_cell` validates `angle_unit in ("rad", "deg")`
  up front, so a typo like `angle_unit="degrees"` raises a `ValueError` instead
  of being swallowed by the classifier's `except` into a silent "not an HD
  cell".

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
- `io.nwb.write_trials` no longer misdirects users to `write_region_crossings()`
  for storing trial *intervals*: that writer stores point events
  (`crossing_times`, `region_names`, `event_types`), not intervals. The
  `overwrite` docstring and the `NotImplementedError` message now point to
  adding a separate `pynwb.epoch.TimeIntervals` table (via
  `nwbfile.add_time_intervals`) with its own `start_time`/`stop_time` columns,
  readable back via `read_intervals`.

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

[Unreleased]: https://github.com/edeno/neurospatial/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/edeno/neurospatial/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/edeno/neurospatial/compare/v0.2.0...v0.4.0
[0.2.0]: https://github.com/edeno/neurospatial/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/edeno/neurospatial/releases/tag/v0.1.0
