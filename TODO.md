# **1. Core Computation Engine**

### **1.1 Population-level computation**

* [ ] **Shared occupancy & binning:**

  * Compute occupancy once per session.
  * Precompute kernel maps (Gaussian, boxcar) once.
  * Population APIs that accept spike-times per cell and reuse shared structures.
  * Major speedup for large populations.

### **1.2 Performance improvements in existing code**

* [x] `_extract_connected_component_graph`

  * ~~Replace `frontier.pop(0)` with `collections.deque.popleft()` to avoid O(n²).~~
* [ ] Vectorization improvements:

  * Gaussian KDE loops in `spike_field.py`.
  * Layout helper loops in dense bins.
  * Only necessary for very large bin counts.

---

# **2. Core Single-Cell Analyses (Neuro-GIS primitives)**

### **2.1 Place field analysis**

* [x] `compute_place_field` rewrite to rely on shared occupancy.
* [ ] Rich result object:

  * `field_map`, `fields[]`, `skaggs_info`, `sparsity`, `stability`, `params`.
  * `.to_dict()` for logging / reproducibility.

### **2.2 Head direction system**

* [x] Circular occupancy (angular bins).
  * `head_direction_tuning_curve()` with configurable bin_size
* [x] HD tuning curves.
  * `head_direction_tuning_curve()` computes firing rate as function of head direction
* [x] Mean vector length, preferred direction.
  * `head_direction_metrics()` returns `HeadDirectionMetrics` with `preferred_direction`, `mean_vector_length`, `tuning_width`
* [x] Rayleigh test & circular statistics library.
  * `rayleigh_test()`, `circular_mean()`, `mean_resultant_length()` in `circular.py`
* [x] HD cell classification.
  * `is_head_direction_cell()`, `HeadDirectionMetrics.is_hd_cell`
* [x] HD tuning visualization.
  * `plot_head_direction_tuning()` with polar/linear projections

### **2.3 Phase precession**

* [x] Per-field phase–position analysis.
  * `phase_precession()` returns `PhasePrecessionResult` with slope, offset, correlation
* [x] Circular–linear correlation.
  * `circular_linear_correlation()` in `circular.py`
* [x] Field-restricted spike-phase plots.
  * `plot_phase_precession()` with doubled-axis visualization
* [x] Fast screening for many neurons.
  * `has_phase_precession()` boolean filter

### **2.4 Distance-to-goal**

* [x] Compute distance-to-goal over time (Euclidean or maze-graph).
  * `distance_to_region()` - geodesic distance to goal
  * `cost_to_goal()` - with terrain/learned cost weighting
* [ ] Correlate with firing, behavior, running speed.

---

# **3. Event + Spatio-Temporal Analysis Module**

### **3.1 Events module (beh. events in space–time)**

* [x] Standardized events representation for:
  * `validate_events_dataframe()` validates pandas DataFrame structure
  * `validate_spatial_columns()` checks for x, y columns
  * Supports: rewards, cues, licks, errors, choices, opto, zone entries/exits
* [x] IO utilities for events.
  * NWB: `read_events()`, `write_events()`, `write_region_crossings()`
  * Interval conversion: `intervals_to_events()`, `events_to_intervals()`
  * Filtering: `filter_by_intervals()`
* [ ] Event timeline visualization.

### **3.2 Spatio-temporal field estimation**

* [x] Event-triggered PSTHs.
  * `peri_event_histogram()` returns `PeriEventResult`
  * `population_peri_event_histogram()` returns `PopulationPeriEventResult`
  * `align_spikes_to_events()` for per-trial spike times
  * `align_events()` for event-to-event alignment
  * `plot_peri_event_histogram()` for visualization
* [ ] Event-triggered rate maps (reward-locked).
* [ ] Full spatio-temporal rate cubes:

  * $$\lambda(x,y,\tau)$$

### **3.3 Regressors for GLMs**

* [x] Time since reward, cue, zone entry.
  * `time_to_nearest_event()` - signed peri-event time
  * `event_count_in_window()` - count events in window
  * `event_indicator()` - binary presence indicator
* [ ] Distance to reward (Euclidean or cost-distance).
* [ ] Distance to walls, obstacles, region boundaries.
* [x] Head direction, speed, acceleration, behavioral phase.
  * `circular_basis()` for circular predictors (HD, phase, direction)
  * `circular_basis_metrics()` to interpret GLM coefficients
  * `is_modulated()` for quick significance checks
  * `plot_circular_basis_tuning()` for visualization
* [ ] "Reachability" features (speed-constrained).
* [x] Maze-aware spatial basis functions.
  * `spatial_basis()` for automatic parameter selection
  * `geodesic_rbf_basis()` for RBF using geodesic distances
  * `heat_kernel_wavelet_basis()` for diffusion-based multi-scale
  * `chebyshev_filter_basis()` for polynomial filters

---

# **4. Spatial Statistics (Neuro-GIS Statistics Layer)**

### **4.1 Point-process spatial analyses**

* [ ] Moran’s I on firing rate maps.
* [ ] Ripley’s K (inhomogeneous for occupancy).
* [ ] Baddeley’s inhomogeneous estimator.
* [ ] Pair-correlation function g(r).
* [ ] Cross-K (spikes vs reward events).
* [ ] Point-process co-clustering between units or unit-vs-events.
* [ ] Spatio-temporal K-functions.

### **4.2 Spatial field statistics**

* [ ] Variograms of rate maps (field smoothness, anisotropy).
* [ ] Local Moran’s I (LISA).
* [ ] Hotspot detection (Getis–Ord Gi*).
* [ ] Zonal statistics on fields:

  * per region mean firing, occupancy, reward frequency.

### **4.3 GLM-based residual spatial structure**

* [ ] Fit GLM to firing with covariates.
* [ ] Compute spatial autocorrelation of residuals to detect unmodeled structure.

---

# **5. Geospatial Tools**

### **5.1 Cost-distance maps**

* [x] Compute distance under movement constraints (walls, obstacles).
  * `distance_field()` - geodesic/euclidean distance from sources
  * `cost_to_goal()` - with `cost_map` and `terrain_difficulty` params
* [x] Graph-shortest-path distance, not Euclidean.
* [x] Cost maps for reward access, avoidance, hazard zones.

### **5.2 Visibility / line-of-sight**

* [ ] Viewshed analysis: what cues / boundaries are visible.
* [ ] Use for cue-based remapping analyses.

### **5.3 Maze graph (network model of environment)**

* [ ] Automatic extraction of maze skeleton graph.
* [ ] Graph distances, centrality, chokepoints.
* [ ] Path prediction vs actual trajectories.
* [ ] Useful for planning and replay analysis.

### **5.4 Patch metrics (landscape ecology)**

* [ ] Compute sizes, shapes, edge lengths of environment zones.
* [ ] Useful for understanding field structure vs environment.

---

# **6. Visualization + napari Integration**

### **6.1 Built-in plotting utilities**

* [ ] Place fields with field outlines.
* [ ] Stability correlation plots.
* [ ] Occupancy maps with spike overlays.
* [x] Dynamic field evolution over time.
  * `env.animate_fields()` with napari/video/html/widget backends
* [x] Rendering of regions, overlays, obstacles.
  * `show_regions=True`, `PositionOverlay`, `BodypartOverlay`, `HeadDirectionOverlay`
* [ ] Population comparison plots.

### **6.2 napari visualization**

* [x] Scale bar.
  * `ScaleBarConfig`, `scale_bar=True` for plots and animations
* [ ] Color bar.
* [ ] 3D support (3D tunnels, ramps, VR environments).
* [x] Dynamic spike raster overlayed on position heatmap.
  * `EventOverlay` / `SpikeOverlay` with decay_frames for visualization
* [ ] Show events timeline synced to time slider.
* [ ] add a small overview strip (full-session timeline) with a brushed window indicating the current animated window; the right-column plots then show “detail in context,” which aligns with his recommendations on focus+context views.
* [ ] Visualize continuous variables (speed, hd, acceleration).
* [x] Multi-body-part tracking (nose, tail base, ears).
  * `BodypartOverlay` with skeleton rendering
* [x] Multi-animal support.
  * Multiple `PositionOverlay` instances
* [ ] Sync multiple environments.
* [x] Track graph overlay.
  * `annotate_track_graph()` for 1D linearized environments
* [x] Integrate video (different sampling rates handled automatically).
  * `VideoOverlay` with `calibrate_video()`, auto temporal alignment
* [x] Mark events (reward, errors, choice) on video + environment.
  * `EventOverlay` supports trajectory mode and fixed-location events
* [ ] SAM based integration for quick region annotation.

---

# **7. UX / Data Preparation**

### **7.1 Easy annotation workflows**

* [x] Tools for defining environment boundaries + obstacles.
  * `annotate_video()` - interactive napari annotation
  * `boundary_from_positions()` - seed boundary from trajectory
* [ ] Annotation quality checks:

  * self-intersection, small polygons, missing holes.
* [ ] Dashboard for quick annotation and validation.

---

# **8. Session-Level Pipelines**

### **8.1 Flagship session analysis pipeline**

* [ ] Single-call `analyze_session()` wrapper:

  * occupancy
  * rate maps
  * place fields
  * head direction fields
  * event-triggered maps
  * statistics (Moran’s I, K, etc.)
  * results container
  * export
* [ ] YAML/JSON config-based pipelines.

### **8.2 Multi-session & population analysis**

* [ ] Remapping module:

  * cross-session alignment
  * population vector correlations
  * field overlap metrics
  * drift / stability
* [ ] Cross-animal alignment:

  * common coordinate reference frames
  * environment warping / transforms.

---

# **9. IO & Interoperability**

### **9.1 NWB integration**

* [x] NWB import/export for:
  * Full `nwb/` module with read/write functions
  * `read_position()`, `read_pose()`, `read_head_direction()`
  * `write_place_field()`, `write_occupancy()`, `write_trials()`
  * `write_environment()`, `read_environment()` (round-trip)
  * ~~spikes~~ (use pynwb directly)
  * ~~LFP~~ (use pynwb directly)
  * [x] continuous behavior
  * [x] events (`read_events()`, `write_events()`, `write_region_crossings()`, `read_intervals()`, `read_trials()`, `write_laps()`)
  * [x] regions (via Environment scratch storage)

### **9.2 Other IO**

* [x] LabelMe/CVAT → Regions.
  * `regions_from_labelme()`, `regions_from_cvat()`
* [ ] Standard "session bundle" export for downstream ML/GLM.

---

# **10. Scalability + Big Data**

### **10.1 Performance & scaling**

* [ ] Chunked / streaming analysis for long sessions.
* [ ] Lazy evaluation of fields (compute when needed).
* [x] Spatial indexing (R-tree or kd-tree).
  * KDTree caching in `map_points_to_bins()` and `bin_at()`
* [x] Efficient caching of:
  * `env.clear_cache()` for cache management
  * [x] binning (KDTree cached)
  * [x] kernel maps (`env.compute_kernel(..., cache=True)`)
  * [ ] occupancy
  * [ ] distance fields

### **10.2 Parallelism**

* [ ] Batch compute population metrics with joblib/dask.
* [ ] Parallelized K-functions and GLMs.

---

# 🎯 **Structured Roadmap View (Suggested)**

### **Phase 1: Foundations** ✅ (mostly complete)

* [ ] Population API (shared occupancy)
* [ ] Result objects
* [x] Basic visualization utilities
* [x] Events module
  * `peri_event_histogram()`, `population_peri_event_histogram()`
  * GLM regressors: `time_to_nearest_event()`, `event_count_in_window()`, `event_indicator()`
  * Interval utilities: `intervals_to_events()`, `events_to_intervals()`, `filter_by_intervals()`
  * Spatial utilities: `add_positions()`
  * NWB integration: `read_events()`, `write_events()`
* [x] NWB IO
* [x] Head direction system

### **Phase 2: Spatial statistics + GIS**

* [ ] K-functions, Moran's I, variograms
* [ ] Zonal stats
* [x] Cost-distance maps, maze graphs
* [ ] Visibility / viewshed
* [ ] Patch metrics

### **Phase 3: Spatio-temporal analytics**

* [x] Peri-event histograms (PSTHs)
  * `peri_event_histogram()`, `population_peri_event_histogram()`
  * `plot_peri_event_histogram()` visualization
* [ ] Event-triggered rate cubes (spatial + temporal)
* [x] GLM regressors (circular basis, spatial basis, temporal event regressors)
* [x] Phase precession module
* [ ] Dynamic place fields across time (estimation, not just visualization)

### **Phase 4: Pipelines + UX**

* [ ] `analyze_session()` pipeline
* [ ] Remapping / cross-animal alignment
* [ ] napari dashboards for annotation + inspection
* [x] Multi-animal and video integration

### **Phase 5: Scaling**

* [ ] Chunked computation
* [ ] Parallelism
* [ ] Large-population optimizations

### Other

* ~~We should be able to seed the environment polygon from position (convex hull?), user can adjust in napari.~~ ✅ DONE: `boundary_from_positions()` with alpha_shape/convex_hull, `annotate_video(initial_boundary=positions)`. Templates for common mazes (linear, T, plus, circular, open field with obstacles) still TODO.

* Consider what to do for dynamic environments (moving obstacles, changing boundaries) in future versions. Regions can also be dynamic (e.g., zones that appear/disappear, optostimulation, reward delivery).

* Consider integration with reinforcement learning environments (OpenAI Gym, DeepMind Lab) for simulated data analysis.

* Consider integration with pynapple.

* Consider what we could learn from video games (in terms of paradigms for spatial representation, navigation, and environment design). Also consider robotics literature for path planning and spatial mapping.

* ~~plot spikes at position in napari (dynamic raster overlay) - could be useful for QC. Color different units differently.~~ ✅ DONE: `EventOverlay` / `SpikeOverlay` with multi-unit support, auto-colors, and decay_frames.

* Simulation + benchmarking suite for replay/RL

Simulation recipes that include explicit reward structure and policy (e.g., T‑maze alternation with changing contingencies), plus utilities to benchmark decoders and replay/theta metrics against ground truth (are sequences truly prospective/retrospective relative to value?).

* egocentric spatial representations (transformations between allocentric and egocentric frames)
