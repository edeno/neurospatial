# API Reference

Complete API documentation for neurospatial, automatically generated from source code docstrings.

## Start with the root workflows

<!-- docs-test: run -->
```python
from neurospatial import (
    Environment,
    compute_spatial_rate,
    compute_spatial_rates,
    SpatialRateResult,
    SpatialRatesResult,
    decode_position,
    DecodingResult,
    peri_event_histogram,
    PeriEventResult,
)
```

These workflows load lazily. Use domain imports for specialized analyses.
The [shared-unit workflow](../user-guide/workflows.md#workflow-6-shared-units-for-decoding-and-population-statistics)
shows simulation and four-array encoding followed by separate decoding and
population-statistics branches.

## Core Modules

### [neurospatial.environment](neurospatial/environment/index.md)

The `Environment` class: bins, their connectivity graph and named regions,
with methods for spatial queries, occupancy, smoothing, plotting and saving.

**Key Classes:**

- `Environment`: Discretized spatial environment, built with factory
  classmethods such as `Environment.from_samples` and `Environment.from_polygon`
- `BinSequenceWithRuns`: Result of `Environment.bin_sequence_with_runs`
- `BinMapping`: Result of `Environment.bin_mapping_from`, the exact bin pairs
  between an environment and the one it was derived from
- `CartesianSystem`, `EgocentricPolarSystem`: What the bin coordinates mean

Every method is documented on the
[`Environment` reference page](neurospatial/environment/core.md). The
`_`-prefixed modules in this package hold private implementations.

### [Combining environments](../user-guide/composite-environments.md)

Build one ordinary environment from disjoint component environments.

**Key Classes:**

- `Environment.combine`: Combine disjoint environments with automatic navigation bridges

### [neurospatial.regions](neurospatial/regions/index.md)

Define and manage named regions of interest (ROIs).

**Key Classes:**

- `Region`: Immutable point or polygon region
- `Regions`: Container for managing multiple regions

### [neurospatial.annotation](neurospatial/annotation/index.md)

Draw arena boundaries, holes, named regions, walls and landmarks on a video
frame. See the [video annotation guide](../user-guide/video-annotation.md).

- `annotate_video(video_path, *, project_path=None, autosave_interval=30.0,
  segmentation_provider=None, ...)`: Open a reference frame in napari;
  `None` selects the default CPU outline provider.
- `annotate_project(project, *, project_path=None, autosave_interval=30.0,
  segmentation_provider=None)`: Resume a saved path or an `AnnotationProject`
  using its embedded reference frame.
- `AnnotationProject`, `AnnotationShape`: Portable editable drafts in pixel
  coordinates, with `AnnotationProject.save(path)` and `.load(path)`.
- `RegionGrowProvider`, `SegmentAnythingProvider`: CPU color segmentation
  and an optional adapter for a separately installed SAM 2 image predictor.
- `SegmentationPrompt`, `SegmentationProposal`, `SegmentationProvider`,
  `mask_to_polygons`: Viewer-independent point prompts, mask results,
  provider contract and exact pixel-cell polygon conversion.

### [neurospatial.layout](neurospatial/layout/index.md)

Layout engines for discretizing continuous space.

**Key Modules:**

- `layout.base`: `LayoutEngine` protocol definition
- `layout.engines.*`: Concrete layout implementations
- `layout.factories`: Factory functions for creating layouts

### [neurospatial.ops.alignment](neurospatial/ops/alignment.md)

Move probability between the bins of aligned environments.

**Key Functions:**

- `map_probabilities()`: Move probability mass (posteriors, occupancy) onto another environment's bins

### [neurospatial.ops.transforms](neurospatial/ops/transforms.md)

Affine transformations and video calibration.

**Key Classes:**

- `AffineND`: Composable affine transformations in any dimension

### [neurospatial.encoding](neurospatial/encoding/index.md)

Neural encoding analyses: how neurons represent space, direction,
egocentric targets, and gaze.

**Key Modules:**

- `encoding.spatial`: Place fields and spatial firing-rate maps
- `encoding.grid`: Grid cells (autocorrelation, gridness score)
- `encoding.directional`: Head-direction cells and tuning curves
- `encoding.border`: Border / boundary cell metrics
- `encoding.egocentric`: Object-vector cells and egocentric tuning
- `encoding.view`: Spatial-view cells and gaze-based fields
- `encoding.phase_precession`: Theta-phase precession
- `encoding.population`: Population-level metrics

**Key Functions:**

- `compute_spatial_rate()` / `compute_spatial_rates()`: Build firing-rate maps
- `compute_directional_rate()`: Head-direction tuning curve
- `compute_egocentric_rate()`: Object-vector tuning
- `compute_view_rate()`: Spatial-view tuning
- `compute_phase_precession()`: Theta-phase slope, offset and fit quality
- `detect_place_fields()`: Threshold-and-cluster on a rate map
- `spatial_information()`, `sparsity()`, `selectivity()`, `border_score()`,
  `grid_score()`: Classic place / boundary / grid metrics

### [neurospatial.decoding](neurospatial/decoding/index.md)

Bayesian decoding of position from spike counts.

**Key Functions:**

- `decode_position()`: Single-step or sequential Bayesian decoder
- `decode_session()`: End-to-end decode of a full session (encode +
  decode) returning a `DecodingResult`
- `decode_jointly()`: Decode several sets of rate maps (running directions,
  tracks) in one posterior; returns a `JointDecodingResult` with each set's
  probability per time bin
- `decode_position_summary()` / `decode_session_summary()`:
  Memory-safe, streaming decoders that return per-time summaries
  (MAP / mean / entropy / peak) without materializing the full
  `(n_time, n_bins)` posterior — for long sessions / thousands of units
- `decoding_error()`, `median_decoding_error()`: Accuracy metrics

**Key Classes:**

- `DecodingResult`: Posterior + helpers (MAP, mean, entropy)
- `DecodingSummary`: Per-time decode summary (MAP / mean / entropy /
  peak) from the memory-safe summary decoders

### [neurospatial.replay](neurospatial/replay/index.md)

Score decoded candidate replay events and test them against shuffles.
Detecting the candidate events (ripples, population bursts) is not included.

- `fit_isotonic_trajectory()`, `fit_linear_trajectory()`,
  `detect_trajectory_radon()`: Fit replay trajectories along a track
  (Michon et al. 2019; Karlsson & Frank 2009 and Wu et al. 2017; Davidson
  et al. 2009); slopes are in environment units per second
- `fit_circular_sequence()`: Fit the decoded path around a circular track,
  through the start of its linear position (Wikenheiser & Redish 2013)
- `weighted_correlation()`: Posterior-weighted correlation of time and
  position (Wu & Foster 2014)
- `trajectory_metrics()`, `template_fractions()`: Per-event metrics that
  criteria-based replay detection thresholds (jumps, spread, range, path
  length, dispersion, coverage, empty time bins) and the directional
  fraction of a joint decode
- `replay_segments()`: Cut an event into segments in which the decoded
  position moves smoothly, by the jump, gap, merge and duration rules of
  Pfeiffer & Foster, Igata et al., Carey et al., Widloski & Foster and
  Mallory et al.; keep the longest or all
- `replay_significance()`: Test any replay score against shuffles:
  column-cycle, unit-identity and pseudo-event (the default), spike-train
  and place-field circular shifts, time-bin permutation (the "time swap"),
  spike times redrawn uniformly in the event (`"spike_time_uniform"`,
  Stella et al. 2019) and the weighted column cycle
  (`"column_cycle_weighted"`, Farooq et al. 2019; options through
  `null_options=`); significant when every null run rejects the event
- `rank_order_correlation()`, `rank_order_significance()`: Spearman
  correlation of field order and firing order in an event, with a
  cell-order permutation test; needs no decoding
- `pairwise_order_score()`, `pairwise_order_significance()`: Firing order
  of every pair of spikes less than 80 ms apart against the cells'
  place-field order, summed (Gupta et al. 2010), with spike-time and
  cell-identity shuffles; cells with several fields take their best field;
  needs no decoding
- `detect_replay()`, `compare_replay_methods()`: Apply published replay
  methods (`neurospatial.replay.methods`: Davidson 2009, Karlsson & Frank
  2009, Pfeiffer & Foster 2013 and 2015, Wikenheiser & Redish 2013,
  Ólafsdóttir 2015, 2016 and 2017, Silva 2015, Ambrose 2016, Grosmark
  2016, Jadhav 2016, Tang 2017, Wu 2017, Yamamoto & Tonegawa 2017, Carey
  2019, Farooq 2019, Michon 2019, Muessig 2019, Shin 2019, Igata 2021,
  Bush 2022, Tirole 2022, Berners-Lee 2022, Widloski & Foster 2022,
  Harvey 2023, Liu 2023, Mallory 2025) to decoded events, including joint decodes over running
  directions, and compare their verdicts; `ReplaySegmentation` sets how
  criteria-based methods find the trajectory segment
- `null_events()`, `simulated_null_events()`, `decode_intervals()`: Events
  without replay (cell-identity, spike-shift, uniform-spike-time and
  rate-matched Poisson copies of real events,
  simulated random and stationary events) for false-positive rates
- `event_properties()`, `detection_rates()`, `method_agreement()`: Calibrate
  methods on a comparison table: rates by event property with confidence
  intervals (false positives on null data), and agreement between methods
- `matched_alpha()`, `apply_alpha()`: Each method's α that holds its
  false-positive rate on null data to a target (Takigawa et al. 2024), and
  a comparison table's verdicts at those α
- `detect_arm_replay()`, `ArmReplayMethod`: Single-arm and joint-arm replay
  on a multi-arm maze (Wu & Foster 2014; Berners-Lee et al. 2021)
- `detect_rank_order_replay()`, `RankOrderMethod`: Apply published
  rank-order tests (Foster & Wilson 2006, Diba & Buzsáki 2007) to an event's
  spikes; `compare_replay_methods(spike_times=...)` compares them with the
  decoded methods and runs Chenani 2019, which tests an epoch's events
  together
- `detect_pairwise_order_replay()`, `PairwiseOrderMethod`: Apply the
  pairwise-order test of Gupta et al. 2010 (`GUPTA_2010`) to an event's
  spikes on every template (such as each lap), forward and backward;
  `compare_replay_methods(spike_times=...)` runs it with the other methods
- `shuffle_cell_identity()`, `shuffle_spike_counts_circular()`,
  `shuffle_spike_counts_uniform()`, `shuffle_place_fields_circular()`,
  `shuffle_place_fields_circular_2d()`, `shuffle_posterior_circular()`,
  `shuffle_posterior_weighted_circular()`: Replay shuffles;
  `replay_significance(nulls=...)` runs them by name

### [neurospatial.reactivation](neurospatial/reactivation/index.md)

Population covariance and reactivation analyses take count matrices with the
same units in the template, match and control periods. Pass each period's
column labels (`unit_ids=` or `template_unit_ids=`, `match_unit_ids=`,
`control_unit_ids=`) to match cells by label; without labels the columns pair
by position:

- `detect_assemblies()`: Dimensions above the Marchenko-Pastur reference and
  fitted patterns; a selected dimension may have no thresholded core members
- `assembly_activation()`: Activation strength (co-firing of the pattern's
  cells) in each bin of a chosen period; descriptive rather than a
  calibrated bin-level significance test
- `pairwise_correlations()`: Upper-triangle neuron-pair correlations
- `explained_variance_reactivation()`: EV and controlled REV effect sizes;
  pass an explicit baseline control and avoid treating EV > REV as a p-value
- `reactivation_strength()`: Mean activation strength in the match period
  over the template period; an effect size rather than a significance verdict

See the [complete public workflow](../user-guide/workflows.md#workflow-6-shared-units-for-decoding-and-population-statistics).

### [neurospatial.behavior](neurospatial/behavior/index.md)

Behavioral analysis built on trajectory + environment.

**Key Modules:**

- `behavior.trajectory`: Speed, step lengths, cumulative distance, MSD,
  curvature, planar turning angles, planar curvature, movement angular
  velocity, home range
- `compute_speed(times, positions)`, `heading_from_velocity(times, positions)`,
  `heading_from_body_orientation(times, nose, tail)`: Per-sample speed and
  heading (radians, 0 = East) from tracking data
- `behavior.resampling`: Distance resampling of observed runs
- `behavior.navigation`: Path efficiency, goal-directed metrics,
  graph turn sequences
- `distance_to_region()`, `distance_to_boundary()`, `distance_to_reward()`:
  Per-sample distance to a goal, the environment's edge or the reward
  location (GLM covariates)
- `behavior.segmentation`: Laps, trials, region crossings,
  velocity-based segmentation
- `detect_region_visits()`, `region_transitions()`: Observed visits and
  entry-to-entry transitions with original sample indices and gap censoring
- `behavior.decisions` / `behavior.vte`: Decision-point analysis,
  vicarious trial-and-error
- `behavior.reward`: Reward-field construction

### [neurospatial.events](neurospatial/events/index.md)

Peri-event spike alignment and GLM regressors.

**Key Functions:**

- `peri_event_histogram()`: PSTH around discrete events
- `plot_peri_event_response()`: Matching raster and PSTH using the same retained
  events and half-open histogram bins
- `align_spikes_to_events()`: Per-trial spike rasters
- `align_continuous_to_events()`: Speed, heading or zone sampled around events
  only where observed, with per-kind means and category counts
- `time_to_nearest_event()`, `event_count_in_window()`, `event_indicator()`:
  Time-based GLM regressors

### [neurospatial.ops.egocentric](neurospatial/ops/egocentric.md)

Allocentric ↔ egocentric coordinate transforms.

**Key Functions:**

- `allocentric_to_egocentric()`, `egocentric_to_allocentric()`:
  Frame conversions
- `compute_egocentric_bearing()`, `compute_egocentric_distance()`:
  Relate animal to external targets

### [neurospatial.ops.feature_sampling](neurospatial/ops/feature_sampling.md)

Read bin-valued feature maps (floor texture, zones, heights) along the path.

**Key Functions:**

- `sample_spatial_features(env, fields, positions)`: Value of the bin that
  contains each position, NaN outside the environment
- `sample_egocentric_features(env, fields, positions, headings, ego_points)`:
  The same at points fixed in the animal's frame (ahead, left)
- `SpatialFeatureResult`: Sampled values with per-sample support and bins

### [neurospatial.ops.visibility](neurospatial/ops/visibility.md)

Visibility / gaze / viewshed computations.

**Key Classes:**

- `FieldOfView`: Symmetric / rat / primate FOV presets
- `ViewshedResult`: Visible bins + visibility fraction

**Key Functions:**

- `compute_viewshed()`, `compute_view_field()`,
  `compute_viewshed_trajectory()`: Viewshed from positions / headings
- `compute_viewed_location()`: Gaze-direction projection
- `visibility_occupancy()`, `visible_cues()`: View-time aggregates

### [neurospatial.ops.basis](neurospatial/ops/basis.md)

Spatial basis functions for regression / kernel methods.

**Key Functions:**

- `select_basis_centers()`: K-means / random / farthest-point placement
- `geodesic_rbf_basis()`, `heat_kernel_wavelet_basis()`,
  `chebyshev_filter_basis()`: Graph-aware kernels
- `spatial_basis()`, `plot_basis_functions()`: Unified spatial-basis API

### [neurospatial.stats](neurospatial/stats/index.md)

Circular statistics, surrogates, and shuffle controls.

**Key Modules:**

- `stats.circular`: Rayleigh test, circular mean / variance / R,
  circular-circular and circular-linear correlations, circular basis
- `stats.shuffle`: Temporal, trial and spike-train shuffles and Monte Carlo
  p-values; replay shuffles are in `neurospatial.replay`
- `stats.surrogates`: Poisson and inhomogeneous-Poisson surrogates,
  jittered-spike controls

### [neurospatial.animation](neurospatial/animation/index.md)

Spatial fields over time on the recording clock: napari, notebook widget
and fastplotlib viewers, and video, GIF and HTML exports.

**Key Function:**

- `Environment.animate_fields()`: One call that opens a view
  (`AnimationView`) or writes a file (`Path`)

**Key Classes:**

- `Animation`, `AnimationField`: Reusable scenes of named fields, each on its
  own clock, shared by every view and export
- `AnimationView`, `EventInspector`: Recording-time controls, and review of
  identified events
- `PositionOverlay`, `BodypartOverlay`, `HeadDirectionOverlay`,
  `EventOverlay`, `VideoOverlay`, `TimeSeriesOverlay`, `SpikeRasterOverlay`,
  `EpochOverlay`, `ObjectVectorOverlay`: Composable overlays for animations

### [neurospatial.io.nwb](neurospatial/io/nwb/index.md)

NWB (Neurodata Without Borders) read / write integration.

**Key Functions:**

- `read_environment()`, `read_position()`, `read_units()`, `read_head_direction()`, `read_pose()`,
  `read_events()`, `read_intervals()`, `read_trials()`: Read NWB
  components into neurospatial types
- `write_environment()`, `write_place_field()`,
  `write_occupancy()`, `write_events()`, `write_laps()`,
  `write_region_crossings()`, `write_trials()`: Persist neurospatial
  results back into an NWBFile
- `environment_from_position()`: Build an environment directly from
  the position channel of an NWB file

### [neurospatial.simulation](neurospatial/simulation/index.md) <span style="color: #4CAF50;">v0.2.0+</span>

Generate synthetic spatial data, neural activity, and spike trains for testing and validation.

**Key Modules:**

- `simulation.trajectory`: Trajectory generation (OU process, structured laps)
- `simulation.models`: Neural models (place, border, boundary-vector and grid cells)
- `simulation.spikes`: Spike generation (Poisson process, refractory periods)
- `simulation.session`: High-level session simulation API
- `simulation.validation`: Automated validation against ground truth
- `simulation.examples`: Pre-configured example sessions
- `simulation.replay`: `simulate_replay_event()`, a candidate replay event
  with a known path (sweep, stationary, jump or random), for comparing replay
  methods

**Key Classes:**

- `PlaceCellModel`: Gaussian place field model with ground truth
- `BorderCellModel`: Firing from the distance to the nearest wall
- `BoundaryVectorCellModel`: Boundary vector cell (walls at a preferred
  distance and direction)
- `GridCellModel`: Hexagonal grid cell model (2D only)
- `SimulationSession`: Complete simulation session dataclass
- `SimulationValidationResult`: Per-unit recovery of simulated place fields

**Key Functions:**

- `simulate_trajectory_ou()`: Ornstein-Uhlenbeck process for realistic exploration
- `simulate_trajectory_sinusoidal()`: Sinusoidal movement for 1D tracks
- `simulate_trajectory_laps()`: Structured lap-based trajectories
- `generate_poisson_spikes()`: Generate spikes from firing rates
- `generate_population_spikes()`: Generate spikes for neuron populations
- `simulate_session()`: One-call workflow for complete sessions
- `validate_simulation()`: Compare detected place fields to ground truth
  (returns a `SimulationValidationResult`)
- `open_field_session()`, `linear_track_session()`, etc.: Pre-configured examples

**See Also:**

- [Simulation Workflows Tutorial](../examples/05_simulation_and_validation.ipynb): Comprehensive examples and quick start guide

### Interoperability and data holders

Named data holders and optional adapters for the array-first API
(see the [Interoperability guide](../user-guide/interoperability.md)). The array
path never depends on these; `import neurospatial` never imports pynapple or
pynwb.

**Simulator holder** ([neurospatial.simulation.session](neurospatial/simulation/session.md)):

- `SimulationSession`: Immutable attributes `env`, `spike_times` (a tuple of
  read-only arrays), `unit_ids`, `times`, `positions`, `models`, integer-label
  `ground_truth`, and `metadata` (read-only mappings). Pass attributes
  explicitly to analyses.
- `validate_simulation(sim, *, unit_ids=...)` and
  `plot_session_summary(sim, *, unit_ids=...)` select simulation units by label.

**Spike-train container** ([neurospatial.encoding.spike_trains](neurospatial/encoding/spike_trains.md)):

- `SpikeTrains`: Frozen ragged per-unit spike trains with `unit_ids` / optional
  `unit_table`. Positional access (`st[i]`), label selection
  (`st.sel(unit_id=...)`), iteration, `.filter(query)`; flows directly into
  the batch `compute_*` / `decode_*` functions. Population results select units
  the same way: `rates[i]` by position, `rates.sel(unit_id=...)` by label.

**Epoch selection** ([neurospatial.behavior.epochs](neurospatial/behavior/epochs.md)):

- `restrict(times, *arrays, epochs=...)`: Slice `times` and time-aligned arrays
  to a set of epochs.
- `in_epochs(t, epochs)`: Boolean mask, `True` where `t` falls in any epoch.
- `restrict_spike_trains(spike_times, epochs)`: Mask ragged per-unit trains, each by
  its own timestamps.

**Bayesian decoder object** ([neurospatial.decoding.estimator](neurospatial/decoding/estimator.md)):

- `BayesianDecoder`: Frozen `fit(...)` → `predict(...)` / `predict_summary(...)`
  / `score(...)` wrapper over `decode_session` (byte-exact). Decodes through the
  `Environment`, so linearized-track / geodesic decoding works.
- `BayesianDecoder.from_rates(rates, dt=...)`: Fitted decoder from a
  `SpatialRatesResult`, preserving unit identity and training spike-window
  provenance. Prediction takes spikes and array timestamps; supplied labels
  align by identity, generated labels pair by position.
- `BayesianDecoder.from_models(env, encoding_models, unit_ids=..., spatial_support=..., dt=...)`:
  Fitted decoder from hand-built `(n_units, n_bins)` rate maps, such as a
  simulation's ground truth. The constructor takes configuration only; fitted
  decoders come from `fit`, `from_rates` or `from_models`.
- `fit(spike_times, times, positions, unit_ids=...)` requires tracking positions;
  explicit labels must match a labelled spike group's values and order.
  `decode_session` also requires positions and computes its own encoding maps.

**pynapple adapters** ([neurospatial.io.pynapple](neurospatial/io/pynapple.md)) — requires `neurospatial[pynapple]`:

- `from_pynapple(obj)`: `TsGroup` → `(trains, unit_ids)`, `Tsd` / `TsdFrame` →
  `(times, positions)`, `IntervalSet` → `(start, end)`.
- `to_pynapple(times, values=...)`: Decoded track / arrays → `Tsd` / `TsdFrame`.

**NWB adapters** ([neurospatial.io.nwb](neurospatial/io/nwb/index.md)) — requires `neurospatial[nwb]`:

- `read_position(...)` → `NWBPosition(times, positions, units)` and
  `read_head_direction(...)` → `NWBHeadDirection(times, headings)`.
- `read_units(nwbfile, *, unit_ids=None, lazy=False)` →
  `NWBUnits(spike_times, unit_ids, obs_intervals, spike_window)`. Unit labels
  are table IDs; `spike_window` intersects the selected units' acquisition
  coverage. Choose analysis epochs with `read_intervals(nwbfile, interval_name="epochs")`.
- Holders are frozen and cannot be tuple-unpacked. Position units describe
  converted values; missing declarations are `None`. `read_position` /
  `read_pose` / `read_units` accept `lazy=True` (handles valid only while the
  file is open; eager arrays remain usable after close).
- The [complete NWB recipe](../user-guide/interoperability.md#population-fields-decoding-and-a-truthful-overlay)
  carries physical units, labels, selected epochs and acquisition coverage
  through population fields, a fitted decoder, summary tables and overlays.
- `write_spatial_rates(nwbfile, result)` / `read_place_field(nwbfile, env=None)`:
  Round-trip a `SpatialRatesResult` (firing rates / occupancy / unit ids /
  `unit_table` + a persisted, connected environment).

## Layout Engines

Detailed documentation for each layout engine:

- [RegularGridLayout](neurospatial/layout/engines/regular_grid.md)
- [HexagonalLayout](neurospatial/layout/engines/hexagonal.md)
- [GraphLayout](neurospatial/layout/engines/graph.md)
- [MaskedGridLayout](neurospatial/layout/engines/masked_grid.md)
- [ShapelyPolygonLayout](neurospatial/layout/engines/shapely_polygon.md)
- [TriangularMeshLayout](neurospatial/layout/engines/triangular_mesh.md)
- [ImageMaskLayout](neurospatial/layout/engines/image_mask.md)

## Navigation

Use the sidebar to browse the complete API reference, or search for specific functions, classes, or methods.

## Docstring Format

All docstrings follow [NumPy docstring conventions](https://numpydoc.readthedocs.io/en/latest/format.html) for consistency with the scientific Python ecosystem.
