# Package Reorganization Plan (v1.1)

**Goal**: Reorganize neurospatial into a clean, domain-centric structure following Raymond Hettinger's principle: "Flat is better than nested."

**Guiding Principles**:

1. No backward compatibility shims — clean break
2. No `api.py` facade — direct imports only
3. Three top-level domains: `encoding/`, `decoding/`, `behavior/`
4. Sparse top-level `__init__.py` — only core classes and essential verbs
5. Group by scientific domain, not by code function
6. **Dependencies flow inward** (Brandon Rhodes) — `ops/` and `stats/` have no internal deps

---

## Dependency Graph (Rhodes-Approved)

```
TIER 1 - Zero internal deps (Foundation)
layout/       ← no neurospatial imports
regions/      ← no neurospatial imports
stats/        ← no neurospatial imports (circular, shuffle, surrogates)

TIER 2 - Core
environment/  ← imports layout/, regions/ only

TIER 3 - Primitives
ops/          ← imports environment/ only (visibility.py); rest have zero deps
              (binning, distance, normalize, smoothing, graph, calculus,
               transforms, alignment, egocentric, basis all have zero internal deps)

TIER 4 - Domains
behavior/     ← imports ops/, stats/
events/       ← imports ops/, stats/
encoding/     ← imports ops/, stats/ (NO dependency on behavior/)
decoding/     ← imports ops/, stats/, encoding/

TIER 5 - Leaf nodes
animation/    ← imports all
simulation/   ← imports all
```

**Key insights**:

1. `stats/` belongs at TIER 1 (not TIER 3) — it has zero internal dependencies
2. By moving `reference_frames.py` → `egocentric.py` and `visibility.py` to `ops/`, we eliminate the `encoding/ → behavior/` dependency
3. Only `visibility.py` in ops/ imports Environment; all other ops/ files are dependency-free

---

## New Package Structure

```
neurospatial/
├── __init__.py              # Sparse: 5 exports (core classes only)
│
├── environment/             # UNCHANGED - Core Environment class
├── layout/                  # UNCHANGED - Layout engines
├── regions/                 # UNCHANGED - Region management
│
├── encoding/                # NEW - Neural encoding (how neurons represent space)
│   ├── __init__.py
│   ├── place.py             # Place cells
│   ├── grid.py              # Grid cells
│   ├── head_direction.py    # Head direction cells
│   ├── border.py            # Border/boundary cells
│   ├── object_vector.py     # Object-vector cells
│   ├── spatial_view.py      # Spatial view cells (SVC-specific only)
│   ├── phase_precession.py  # Theta phase precession
│   └── population.py        # Population-level metrics
│
├── decoding/                # EXPANDED - Neural decoding (read out from population)
│   ├── __init__.py
│   ├── bayesian.py          # Bayesian position decoding
│   ├── trajectory.py        # Trajectory detection (Radon, isotonic)
│   ├── assemblies.py        # Cell assembly detection
│   └── metrics.py           # Decoding quality metrics
│
├── behavior/                # NEW - Behavioral analysis (unified)
│   ├── __init__.py
│   ├── trajectory.py        # Step lengths, turn angles, MSD, home range
│   ├── segmentation.py      # Laps, trials, region crossings, runs
│   ├── navigation.py        # Path efficiency, goal-directed metrics
│   ├── decisions.py         # VTE, decision analysis, choice points
│   └── reward.py            # Reward fields
│
├── events/                  # UNCHANGED - Peri-event analysis
│   ├── __init__.py
│   ├── peri_event.py        # PSTH, alignment
│   ├── glm_regressors.py    # time_to_event, event_count, etc.
│   ├── intervals.py         # Interval utilities
│   └── validation.py        # DataFrame validation
│
├── simulation/              # UNCHANGED - Neural & trajectory simulation
│   ├── __init__.py
│   ├── models/              # PlaceCell, GridCell, etc.
│   ├── trajectory.py        # OU, laps, sinusoidal
│   ├── spikes.py            # Poisson spike generation
│   ├── session.py           # High-level session API
│   └── examples.py          # Pre-configured sessions
│
├── animation/               # KEEP - Napari viewer, video export, overlays
│   ├── __init__.py
│   ├── overlays.py          # Position, Spike, Event, HD overlays
│   ├── config.py            # ScaleBarConfig, colormaps (moved from visualization/)
│   ├── skeletons.py         # Skeleton definitions
│   └── backends/            # Napari, video backends
│
├── annotation/              # UNCHANGED - Video/image annotation
│
├── io/                      # NEW - All I/O (unified)
│   ├── __init__.py
│   ├── files.py             # to_file, from_file, to_dict, from_dict
│   └── nwb/                 # NWB integration (moved from top-level)
│
├── ops/                     # NEW - Low-level operations (power users)
│   ├── __init__.py
│   ├── binning.py           # map_points_to_bins, regions_to_mask (was spatial.py)
│   ├── distance.py          # distance_field, pairwise_distances
│   ├── normalize.py         # normalize_field, clamp, combine_fields (was field_ops.py)
│   ├── smoothing.py         # diffusion kernels, apply_kernel (was kernels.py)
│   ├── graph.py             # convolve, neighbor_reduce (was primitives.py)
│   ├── calculus.py          # gradient, divergence (was differential.py)
│   ├── transforms.py        # Affine2D, AffineND, calibration (merge calibration.py)
│   ├── alignment.py         # map_probabilities, apply_similarity_transform
│   ├── egocentric.py        # heading, allocentric↔egocentric (was reference_frames.py)
│   ├── visibility.py        # viewshed, gaze, line-of-sight
│   └── basis.py             # GLM spatial basis functions
│
├── stats/                   # NEW - Statistical methods
│   ├── __init__.py
│   ├── circular.py          # Circular statistics, Rayleigh test
│   ├── shuffle.py           # Shuffle controls, permutation tests
│   └── surrogates.py        # Surrogate data generation
│
├── _constants.py            # UNCHANGED - Private constants
└── _logging.py              # UNCHANGED - Private logging
```

---

## Detailed Module Contents

### Top-Level `__init__.py` (Sparse: 5 exports)

```python
# Core classes only (5)
from neurospatial.environment import Environment, EnvironmentNotFittedError
from neurospatial.regions import Region, Regions
from neurospatial.composite import CompositeEnvironment
```

Everything else requires explicit submodule import (Rhodes: "explicit is better than implicit"):

```python
# Domain imports
from neurospatial.encoding.place import compute_place_field
from neurospatial.decoding import decode_position
from neurospatial.events import peri_event_histogram
from neurospatial.ops.distance import distance_field
from neurospatial.behavior.segmentation import detect_laps

# Or import submodules
from neurospatial import encoding, decoding, behavior
from neurospatial.encoding import place, grid
```

---

### `encoding/` — Neural Encoding

How neurons represent space. Combines field computation + metrics + classification for each cell type.

#### `encoding/place.py`

**From**: `spike_field.py`, `metrics/place_fields.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `compute_place_field()` | spike_field.py | Place field estimation from spikes |
| `compute_directional_place_fields()` | spike_field.py | Direction-conditioned place fields |
| `spikes_to_field()` | spike_field.py | Convert spikes to firing rate |
| `DirectionalPlaceFields` | spike_field.py | Container for directional fields |
| `detect_place_fields()` | metrics/place_fields.py | Detect fields from rate map |
| `skaggs_information()` | metrics/place_fields.py | Spatial info (bits/spike) |
| `sparsity()` | metrics/place_fields.py | Spatial sparsity |
| `selectivity()` | metrics/place_fields.py | Place field selectivity |
| `field_centroid()` | metrics/place_fields.py | Field centroid |
| `field_size()` | metrics/place_fields.py | Field size |
| `field_stability()` | metrics/place_fields.py | Temporal stability |
| `field_shape_metrics()` | metrics/place_fields.py | Shape metrics |
| `field_shift_distance()` | metrics/place_fields.py | Shift between sessions |
| `in_out_field_ratio()` | metrics/place_fields.py | In/out firing ratio |
| `information_per_second()` | metrics/place_fields.py | Bits/second |
| `mutual_information()` | metrics/place_fields.py | MI between fields |
| `rate_map_coherence()` | metrics/place_fields.py | Spatial coherence |
| `spatial_coverage_single_cell()` | metrics/place_fields.py | Coverage fraction |
| `compute_field_emd()` | metrics/place_fields.py | Earth mover distance |

#### `encoding/grid.py`

**From**: `metrics/grid_cells.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `grid_score()` | metrics/grid_cells.py | Grid cell grid score |
| `spatial_autocorrelation()` | metrics/grid_cells.py | Autocorrelation of field |
| `grid_orientation()` | metrics/grid_cells.py | Grid orientation |
| `grid_scale()` | metrics/grid_cells.py | Grid spacing |
| `grid_properties()` | metrics/grid_cells.py | Comprehensive properties |
| `periodicity_score()` | metrics/grid_cells.py | Periodicity measure |
| `GridProperties` | metrics/grid_cells.py | Properties dataclass |

#### `encoding/head_direction.py`

**From**: `metrics/head_direction.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `head_direction_tuning_curve()` | metrics/head_direction.py | HD tuning curve |
| `head_direction_metrics()` | metrics/head_direction.py | Comprehensive HD metrics |
| `is_head_direction_cell()` | metrics/head_direction.py | HD cell classification |
| `plot_head_direction_tuning()` | metrics/head_direction.py | Polar plot |
| `HeadDirectionMetrics` | metrics/head_direction.py | Metrics dataclass |

**Re-exports from `stats/circular`** (for discoverability):

| Function/Class | Canonical Location | Description |
|---------------|-------------------|-------------|
| `rayleigh_test()` | stats/circular.py | Rayleigh test for uniformity |
| `mean_resultant_length()` | stats/circular.py | Mean resultant length |
| `circular_mean()` | stats/circular.py | Circular mean direction |

#### `encoding/border.py`

**From**: `metrics/boundary_cells.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `border_score()` | metrics/boundary_cells.py | Border score |
| `compute_region_coverage()` | metrics/boundary_cells.py | Region coverage stats |

#### `encoding/object_vector.py`

**From**: `object_vector_field.py`, `metrics/object_vector_cells.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `compute_object_vector_field()` | object_vector_field.py | OVC field computation |
| `ObjectVectorFieldResult` | object_vector_field.py | Result container |
| `object_vector_score()` | metrics/object_vector_cells.py | OVC score |
| `compute_object_vector_tuning()` | metrics/object_vector_cells.py | OVC tuning curve |
| `is_object_vector_cell()` | metrics/object_vector_cells.py | OVC classification |
| `plot_object_vector_tuning()` | metrics/object_vector_cells.py | OVC plot |
| `ObjectVectorMetrics` | metrics/object_vector_cells.py | Metrics dataclass |

#### `encoding/spatial_view.py`

**From**: `spatial_view_field.py`, `metrics/spatial_view_cells.py`

SVC-specific functions only. General visibility functions moved to `ops/visibility.py`.

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `compute_spatial_view_field()` | spatial_view_field.py | SVC field computation |
| `SpatialViewFieldResult` | spatial_view_field.py | Result container |
| `spatial_view_cell_metrics()` | metrics/spatial_view_cells.py | SVC metrics |
| `is_spatial_view_cell()` | metrics/spatial_view_cells.py | SVC classification |
| `SpatialViewMetrics` | metrics/spatial_view_cells.py | Metrics dataclass |

**Re-exports from `ops/visibility`** (for SVC workflow convenience):

| Function/Class | Canonical Location | Description |
|---------------|-------------------|-------------|
| `compute_viewed_location()` | ops/visibility.py | Gaze location |
| `compute_viewshed()` | ops/visibility.py | Full viewshed analysis |
| `visibility_occupancy()` | ops/visibility.py | Time viewing each bin |
| `FieldOfView` | ops/visibility.py | FOV specification |

#### `encoding/phase_precession.py`

**From**: `metrics/phase_precession.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `phase_precession()` | metrics/phase_precession.py | Phase precession analysis |
| `has_phase_precession()` | metrics/phase_precession.py | Significance test |
| `plot_phase_precession()` | metrics/phase_precession.py | Phase-position plot |
| `PhasePrecessionResult` | metrics/phase_precession.py | Result dataclass |

#### `encoding/population.py`

**From**: `metrics/population.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `population_vector_correlation()` | metrics/population.py | PVC metric |
| `population_coverage()` | metrics/population.py | Coverage by population |
| `count_place_cells()` | metrics/population.py | Count by threshold |
| `field_density_map()` | metrics/population.py | Field center density |
| `field_overlap()` | metrics/population.py | Pairwise field overlap |
| `plot_population_coverage()` | metrics/population.py | Coverage visualization |
| `PopulationCoverageResult` | metrics/population.py | Result dataclass |

---

### `decoding/` — Neural Decoding

Reading out position from population activity.

#### `decoding/__init__.py` (re-exports)

Main interface — kept flat for common operations.

#### `decoding/bayesian.py`

**From**: `decoding/` (existing)

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `decode_position()` | `decoding/__init__.py` | Bayesian decoding |
| `normalize_to_posterior()` | decoding/posterior.py | Log-likelihood → posterior |
| `log_poisson_likelihood()` | decoding/likelihood.py | Log-likelihood |
| `poisson_likelihood()` | decoding/likelihood.py | Likelihood |
| `DecodingResult` | decoding/_result.py | Result container |

#### `decoding/estimates.py`

**From**: `decoding/estimates.py` (existing)

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `map_estimate()` | decoding/estimates.py | MAP bin index |
| `map_position()` | decoding/estimates.py | MAP position |
| `mean_position()` | decoding/estimates.py | Posterior mean |
| `credible_region()` | decoding/estimates.py | HPD region |
| `entropy()` | decoding/estimates.py | Posterior entropy |

#### `decoding/metrics.py`

**From**: `decoding/metrics.py` (existing)

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `decoding_error()` | decoding/metrics.py | Error per timepoint |
| `median_decoding_error()` | decoding/metrics.py | Median error |
| `decoding_correlation()` | decoding/metrics.py | Correlation |
| `confusion_matrix()` | decoding/metrics.py | Confusion matrix |

#### `decoding/trajectory.py`

**From**: `decoding/trajectory.py` (existing)

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `detect_trajectory_radon()` | decoding/trajectory.py | Radon transform |
| `fit_isotonic_trajectory()` | decoding/trajectory.py | Monotonic fit |
| `fit_linear_trajectory()` | decoding/trajectory.py | Linear fit |
| `RadonDetectionResult` | decoding/trajectory.py | Radon result |
| `IsotonicFitResult` | decoding/trajectory.py | Isotonic result |
| `LinearFitResult` | decoding/trajectory.py | Linear result |

#### `decoding/assemblies.py`

**From**: `decoding/assemblies.py` (existing)

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `detect_assemblies()` | decoding/assemblies.py | Assembly detection |
| `assembly_activation()` | decoding/assemblies.py | Activation strength |
| `pairwise_correlations()` | decoding/assemblies.py | Correlation matrix |
| `explained_variance_reactivation()` | decoding/assemblies.py | EV analysis |
| `reactivation_strength()` | decoding/assemblies.py | Reactivation metric |
| `marchenko_pastur_threshold()` | decoding/assemblies.py | Random matrix theory significance |
| `AssemblyDetectionResult` | decoding/assemblies.py | Result container |
| `AssemblyPattern` | decoding/assemblies.py | Pattern dataclass |
| `ExplainedVarianceResult` | decoding/assemblies.py | EV results container |

#### `decoding/__init__.py` Re-exports

**Re-exports from `stats/shuffle`** (for discoverability in decoding workflows):

| Function/Class | Canonical Location | Description |
|---------------|-------------------|-------------|
| `shuffle_time_bins()` | stats/shuffle.py | Shuffle temporal order |
| `shuffle_cell_identity()` | stats/shuffle.py | Shuffle cell labels |
| `compute_shuffle_pvalue()` | stats/shuffle.py | P-value from null distribution |
| `ShuffleTestResult` | stats/shuffle.py | Result dataclass |

**Re-exports from `stats/surrogates`**:

| Function/Class | Canonical Location | Description |
|---------------|-------------------|-------------|
| `generate_poisson_surrogates()` | stats/surrogates.py | Poisson spike surrogates |

---

### `behavior/` — Behavioral Analysis

Movement, navigation, and decision-making.

#### `behavior/trajectory.py`

**From**: `metrics/trajectory.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `compute_step_lengths()` | metrics/trajectory.py | Step lengths |
| `compute_turn_angles()` | metrics/trajectory.py | Turn angles |
| `compute_home_range()` | metrics/trajectory.py | MCP home range |
| `mean_square_displacement()` | metrics/trajectory.py | MSD vs lag |
| `compute_trajectory_curvature()` | behavioral.py | Curvature |

#### `behavior/segmentation.py`

**From**: `segmentation/`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `detect_laps()` | segmentation/ | Lap detection |
| `detect_region_crossings()` | segmentation/ | Entry/exit events |
| `detect_runs_between_regions()` | segmentation/ | Source→target runs |
| `detect_goal_directed_runs()` | segmentation/ | Goal-directed segments |
| `segment_by_velocity()` | segmentation/ | Movement/rest |
| `segment_trials()` | segmentation/ | Behavioral trials |
| `trajectory_similarity()` | segmentation/ | Path similarity |
| `Trial` | segmentation/ | Trial dataclass |
| `Crossing` | segmentation/ | Crossing dataclass |
| `Lap` | segmentation/ | Lap dataclass |
| `Run` | segmentation/ | Run dataclass |

#### `behavior/navigation.py`

**From**: `behavioral.py`, `metrics/path_efficiency.py`, `metrics/goal_directed.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `path_progress()` | behavioral.py | Progress along path |
| `distance_to_region()` | behavioral.py | Distance to region |
| `cost_to_goal()` | behavioral.py | Cost-aware distance |
| `time_to_goal()` | behavioral.py | Time to reach goal |
| `trials_to_region_arrays()` | behavioral.py | Trials to arrays |
| `path_efficiency()` | metrics/path_efficiency.py | Path efficiency |
| `compute_path_efficiency()` | metrics/path_efficiency.py | Comprehensive analysis |
| `angular_efficiency()` | metrics/path_efficiency.py | Angular efficiency |
| `time_efficiency()` | metrics/path_efficiency.py | Time efficiency |
| `subgoal_efficiency()` | metrics/path_efficiency.py | Multi-goal efficiency |
| `shortest_path_length()` | metrics/path_efficiency.py | Shortest path |
| `traveled_path_length()` | metrics/path_efficiency.py | Traveled distance |
| `goal_bias()` | metrics/goal_directed.py | Goal direction bias |
| `goal_direction()` | metrics/goal_directed.py | Direction to goal |
| `goal_vector()` | metrics/goal_directed.py | Vector to goal |
| `approach_rate()` | metrics/goal_directed.py | Approach rate |
| `instantaneous_goal_alignment()` | metrics/goal_directed.py | Alignment |
| `compute_goal_directed_metrics()` | metrics/goal_directed.py | Comprehensive |
| `PathEfficiencyResult` | metrics/path_efficiency.py | Result dataclass |
| `SubgoalEfficiencyResult` | metrics/path_efficiency.py | Result dataclass |
| `GoalDirectedMetrics` | metrics/goal_directed.py | Metrics dataclass |

#### `behavior/decisions.py`

**From**: `metrics/decision_analysis.py`, `metrics/vte.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `compute_decision_analysis()` | metrics/decision_analysis.py | Decision analysis |
| `compute_pre_decision_metrics()` | metrics/decision_analysis.py | Pre-decision metrics |
| `decision_region_entry_time()` | metrics/decision_analysis.py | Entry time |
| `detect_boundary_crossings()` | metrics/decision_analysis.py | Boundary crossings |
| `distance_to_decision_boundary()` | metrics/decision_analysis.py | Distance to boundary |
| `extract_pre_decision_window()` | metrics/decision_analysis.py | Extract window |
| `geodesic_voronoi_labels()` | metrics/decision_analysis.py | Voronoi partition |
| `pre_decision_heading_stats()` | metrics/decision_analysis.py | Heading stats |
| `pre_decision_speed_stats()` | metrics/decision_analysis.py | Speed stats |
| `compute_vte_index()` | metrics/vte.py | VTE index |
| `compute_vte_trial()` | metrics/vte.py | Trial VTE |
| `compute_vte_session()` | metrics/vte.py | Session VTE |
| `classify_vte()` | metrics/vte.py | VTE classification |
| `head_sweep_from_positions()` | metrics/vte.py | Head sweep detection from positions |
| `head_sweep_magnitude()` | metrics/vte.py | Magnitude of head sweeps |
| `integrated_absolute_rotation()` | metrics/vte.py | Integrated absolute rotation (total rotation) |
| `normalize_vte_scores()` | metrics/vte.py | Normalize VTE scores |
| `DecisionAnalysisResult` | metrics/decision_analysis.py | Result dataclass |
| `DecisionBoundaryMetrics` | metrics/decision_analysis.py | Metrics dataclass |
| `PreDecisionMetrics` | metrics/decision_analysis.py | Metrics dataclass |
| `VTETrialResult` | metrics/vte.py | Result dataclass |
| `VTESessionResult` | metrics/vte.py | Result dataclass |

#### `behavior/reward.py`

**From**: `reward.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `goal_reward_field()` | reward.py | Point goal reward |
| `region_reward_field()` | reward.py | Region reward |

---

### `events/` — Peri-Event Analysis

PSTH, event alignment, GLM regressors.

#### `events/__init__.py`

Re-exports main interface.

#### `events/peri_event.py`

**From**: `events/` (existing)

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `peri_event_histogram()` | events/ | PSTH |
| `population_peri_event_histogram()` | events/ | Population PSTH |
| `align_spikes_to_events()` | events/ | Spike alignment |
| `align_events()` | events/ | Event alignment |
| `plot_peri_event_histogram()` | events/ | PSTH plot |
| `PeriEventResult` | events/ | Result dataclass |
| `PopulationPeriEventResult` | events/ | Result dataclass |

#### `events/glm_regressors.py`

**From**: `events/` (existing)

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `time_to_nearest_event()` | events/ | Time to event |
| `event_count_in_window()` | events/ | Count in window |
| `event_indicator()` | events/ | Binary indicator |
| `distance_to_reward()` | events/ | Distance to reward location |
| `distance_to_boundary()` | events/ | Distance to boundary |
| `add_positions()` | events/ | Add x,y to events |

#### `events/intervals.py`

**From**: `events/` (existing)

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `events_to_intervals()` | events/ | Events → intervals |
| `intervals_to_events()` | events/ | Intervals → events |
| `filter_by_intervals()` | events/ | Filter by intervals |

#### `events/validation.py`

**From**: `events/` (existing)

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `validate_events_dataframe()` | events/ | Validate DataFrame |
| `validate_spatial_columns()` | events/ | Check x,y columns |

---

### `simulation/` — UNCHANGED

Keep existing structure. Already well-organized.

#### `simulation/validation.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `validate_simulation()` | simulation/validation.py | Compare detected vs ground truth place fields |
| `plot_session_summary()` | simulation/validation.py | Visualize simulation summary |

---

### `animation/` — KEEP (Minor Consolidation)

Keep existing `animation/` structure. Move `visualization/scale_bar.py` here, delete empty `visualization/`.

#### `animation/` Contents (EXISTING)

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `NapariAnimator` | animation/ | Napari viewer (internal) |
| `VideoExporter` | animation/ | Video export (internal) |
| `subsample_frames()` | animation/ | Frame subsampling |
| `estimate_colormap_range_from_subset()` | animation/ | Auto vmin/vmax |
| `large_session_napari_config()` | animation/ | Large dataset config |
| `calibrate_video()` | animation/calibration.py | Calibrate video to environment coordinates |

#### `animation/overlays.py` (EXISTING)

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `PositionOverlay` | animation/overlays.py | Position trajectory |
| `EventOverlay` | animation/overlays.py | Discrete events |
| `SpikeOverlay` | animation/overlays.py | Spike events |
| `HeadDirectionOverlay` | animation/overlays.py | HD vector |
| `BodypartOverlay` | animation/overlays.py | Pose skeleton |
| `VideoOverlay` | animation/overlays.py | Video background |
| `TimeSeriesOverlay` | animation/overlays.py | Time series |
| `ObjectVectorOverlay` | animation/overlays.py | OVC overlay |
| `OverlayProtocol` | animation/overlays.py | Protocol |

#### `animation/config.py` (MOVED from visualization/)

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `ScaleBarConfig` | visualization/scale_bar.py | Scale bar config |

#### `animation/skeletons.py` (EXISTING)

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `Skeleton` | animation/ | Skeleton definition |
| `MOUSE_SKELETON` | animation/ | Mouse preset |
| `RAT_SKELETON` | animation/ | Rat preset |
| `SIMPLE_SKELETON` | animation/ | Simple skeleton |

**Note**: Domain-specific plotting functions stay in their domain modules:

- `plot_head_direction_tuning()` → `encoding/head_direction.py`
- `plot_object_vector_tuning()` → `encoding/object_vector.py`
- `plot_phase_precession()` → `encoding/phase_precession.py`
- `plot_population_coverage()` → `encoding/population.py`
- `plot_circular_basis_tuning()` → `stats/circular.py`
- `plot_basis_functions()` → `ops/basis.py`
- `plot_peri_event_histogram()` → `events/peri_event.py`
- `plot_session_summary()` → `simulation/`
- `plot_regions()` → `regions/`

---

### `annotation/` — UNCHANGED

Keep existing structure.

#### `annotation/converters.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `shapes_to_regions()` | annotation/converters.py | Convert napari shapes to Regions |
| `subtract_holes_from_boundary()` | annotation/converters.py | Remove hole polygons from boundary |
| `env_from_boundary_region()` | annotation/converters.py | Create Environment from boundary Region |

#### `annotation/validation.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `validate_annotations()` | annotation/validation.py | Validate all annotations |
| `validate_polygon_geometry()` | annotation/validation.py | Validate polygon geometry |
| `validate_region_within_boundary()` | annotation/validation.py | Check region inside boundary |
| `validate_region_overlap()` | annotation/validation.py | Check region overlap |
| `validate_environment_boundary()` | annotation/validation.py | Validate environment boundary |

---

### `io/` — I/O (Unified)

#### `io/__init__.py`

```python
from neurospatial.io.files import to_file, from_file, to_dict, from_dict
```

#### `io/files.py`

**From**: `io.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `to_file()` | io.py | Save to disk |
| `from_file()` | io.py | Load from disk |
| `to_dict()` | io.py | Serialize |
| `from_dict()` | io.py | Deserialize |

#### `io/nwb/`

**From**: `nwb/` (moved)

Keep existing structure within nwb/.

---

### `ops/` — Low-Level Operations

Power user utilities. Files renamed for discoverability:

#### File Rename Mapping

| Old Name | New Name | Rationale |
|----------|----------|-----------|
| `field_ops.py` | `normalize.py` | Primary operation; avoids confusion with neural "fields" |
| `spatial.py` | `binning.py` | Clearer — it's about point↔bin mapping |
| `kernels.py` | `smoothing.py` | Users search for "smoothing", not "kernels" |
| `primitives.py` | `graph.py` | These operate on connectivity graphs |
| `differential.py` | `calculus.py` | More discoverable |
| `reference_frames.py` | `egocentric.py` | Shorter, clearer |
| `calibration.py` | *(merged into transforms.py)* | Too small standalone |
| `alignment.py` | `alignment.py` | Unchanged — clear name |

#### `ops/normalize.py`

**From**: `field_ops.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `normalize_field()` | field_ops.py | Normalize to sum=1 |
| `clamp()` | field_ops.py | Clamp to range |
| `combine_fields()` | field_ops.py | Weighted combination |

#### `ops/binning.py`

**From**: `spatial.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `map_points_to_bins()` | spatial.py | Point-to-bin mapping |
| `regions_to_mask()` | spatial.py | Regions to mask |
| `resample_field()` | spatial.py | Resample field |
| `clear_kdtree_cache()` | spatial.py | Clear KD-tree cache |
| `TieBreakStrategy` | spatial.py | Tie-break enum |

#### `ops/smoothing.py`

**From**: `kernels.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `compute_diffusion_kernels()` | kernels.py | Diffusion kernels |
| `apply_kernel()` | kernels.py | Apply kernel |

#### `ops/graph.py`

**From**: `primitives.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `convolve()` | primitives.py | Graph convolution |
| `neighbor_reduce()` | primitives.py | Neighborhood reduce |

#### `ops/calculus.py`

**From**: `differential.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `gradient()` | differential.py | Spatial gradient |
| `divergence()` | differential.py | Spatial divergence |
| `compute_differential_operator()` | differential.py | Build sparse differential operator matrix |

#### `ops/transforms.py`

**From**: `transforms.py`, `calibration.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `Affine2D` | transforms.py | 2D affine |
| `AffineND` | transforms.py | ND affine |
| `SpatialTransform` | transforms.py | Transform protocol |
| `VideoCalibration` | transforms.py | Video calibration |
| `identity()` | transforms.py | Identity transform |
| `scale_2d()` | transforms.py | 2D scale |
| `translate()` | transforms.py | Translation |
| `flip_y()` | transforms.py | Flip Y transform |
| `identity_nd()` | transforms.py | ND identity |
| `translate_3d()` | transforms.py | 3D translation |
| `scale_3d()` | transforms.py | 3D scale |
| `from_rotation_matrix()` | transforms.py | Create affine from rotation matrix |
| `calibrate_from_scale_bar()` | transforms.py | Scale bar calibration |
| `calibrate_from_landmarks()` | transforms.py | Landmark calibration |
| `flip_y_data()` | transforms.py | Flip Y axis |
| `convert_to_cm()` | transforms.py | Convert to cm |
| `convert_to_pixels()` | transforms.py | Convert to pixels |
| `estimate_transform()` | transforms.py | Estimate affine |
| `apply_transform_to_environment()` | transforms.py | Transform env |
| `simple_scale()` | calibration.py | Simple scale calibration |

#### `ops/alignment.py`

**From**: `alignment.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `ProbabilityMappingParams` | alignment.py | Mapping config |
| `get_2d_rotation_matrix()` | alignment.py | Rotation matrix |
| `apply_similarity_transform()` | alignment.py | Apply similarity transform |
| `map_probabilities()` | alignment.py | Probability alignment |

#### `ops/egocentric.py`

**From**: `reference_frames.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `heading_from_velocity()` | reference_frames.py | Heading from movement |
| `heading_from_body_orientation()` | reference_frames.py | Heading from body pose |
| `allocentric_to_egocentric()` | reference_frames.py | Allocentric → egocentric |
| `egocentric_to_allocentric()` | reference_frames.py | Egocentric → allocentric |
| `compute_egocentric_bearing()` | reference_frames.py | Bearing to objects |
| `compute_egocentric_distance()` | reference_frames.py | Distance to objects |
| `EgocentricFrame` | reference_frames.py | Frame dataclass |

#### `ops/distance.py`

**From**: `distance.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `distance_field()` | distance.py | Multi-source distance |
| `pairwise_distances()` | distance.py | Pairwise distances |
| `neighbors_within()` | distance.py | Neighbors within range |
| `euclidean_distance_matrix()` | distance.py | Euclidean matrix |
| `geodesic_distance_matrix()` | distance.py | Geodesic matrix |
| `geodesic_distance_between_points()` | distance.py | Point-to-point geodesic |

#### `ops/visibility.py`

**From**: `visibility.py`

Visibility primitives and analysis. Per Brandon Rhodes' "dependencies flow inward"
principle, these live in `ops/` so both `encoding/` and `behavior/` can import them.

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `compute_viewed_location()` | visibility.py | Where is animal looking |
| `compute_viewshed()` | visibility.py | Full visibility analysis |
| `compute_view_field()` | visibility.py | Binary visibility mask |
| `compute_viewshed_trajectory()` | visibility.py | Viewshed along path |
| `visible_cues()` | visibility.py | Line-of-sight checks |
| `visibility_occupancy()` | visibility.py | Time each bin was visible |
| `FieldOfView` | visibility.py | FOV specification |
| `ViewshedResult` | visibility.py | Viewshed result |

#### `ops/basis.py`

**From**: `basis.py`

GLM spatial basis functions — used by encoding AND decoding.

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `spatial_basis()` | basis.py | Unified basis interface |
| `geodesic_rbf_basis()` | basis.py | Geodesic RBF |
| `heat_kernel_wavelet_basis()` | basis.py | Heat kernel wavelets |
| `chebyshev_filter_basis()` | basis.py | Chebyshev polynomials |
| `select_basis_centers()` | basis.py | Select basis centers |
| `plot_basis_functions()` | basis.py | Visualize basis |

---

### `stats/` — Statistical Methods

General-purpose statistical utilities used across encoding, decoding, and behavior.

**Design Pattern: Canonical Location + Re-exports**

Following Brandon Rhodes' "dependencies flow inward" principle:

- `stats/` modules contain the canonical implementations (stable, no domain imports)
- Domain modules (`encoding/`, `decoding/`) re-export relevant functions for discoverability
- Users can import from either location; implementation is in one place

```python
# stats/circular.py - canonical implementation
def rayleigh_test(...): ...

# encoding/head_direction.py - re-exports for convenience
from neurospatial.stats.circular import rayleigh_test  # re-export in __all__

# User can do either:
from neurospatial.stats.circular import rayleigh_test      # explicit (power user)
from neurospatial.encoding.head_direction import rayleigh_test  # convenient (HD workflow)
```

This avoids circular dependencies while providing multiple discovery paths.

#### `stats/circular.py`

**From**: `metrics/circular.py`, `metrics/circular_basis.py`

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `rayleigh_test()` | metrics/circular.py | Rayleigh test for uniformity |
| `circular_circular_correlation()` | metrics/circular.py | Circular-circular correlation |
| `circular_linear_correlation()` | metrics/circular.py | Circular-linear correlation |
| `phase_position_correlation()` | metrics/circular.py | Phase-position correlation |
| `circular_mean()` | metrics/circular.py (private → public) | Circular mean direction |
| `circular_variance()` | metrics/circular.py (private → public) | Circular variance |
| `mean_resultant_length()` | metrics/circular.py (private → public) | Mean resultant length |
| `circular_basis()` | metrics/circular_basis.py | Circular harmonics basis |
| `circular_basis_metrics()` | metrics/circular_basis.py | Basis fit metrics |
| `is_modulated()` | metrics/circular_basis.py | Modulation significance test |
| `plot_circular_basis_tuning()` | metrics/circular_basis.py | Visualize circular basis tuning |
| `CircularBasisResult` | metrics/circular_basis.py | Result dataclass |
| `wrap_angle()` | metrics/vte.py | Wrap angle to [-π, π] |

**Note**: `circular_mean()`, `circular_variance()`, and `mean_resultant_length()` are currently private
(`_function_name`) in `metrics/circular.py`. The reorganization makes them public exports.

#### `stats/shuffle.py`

**From**: `decoding/shuffle.py` (generalized)

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `shuffle_time_bins()` | decoding/shuffle.py | Shuffle temporal order |
| `shuffle_time_bins_coherent()` | decoding/shuffle.py | Coherent temporal shuffle |
| `shuffle_cell_identity()` | decoding/shuffle.py | Shuffle cell labels |
| `shuffle_circular()` | decoding/shuffle.py | Circular shift |
| `shuffle_place_fields_circular()` | decoding/shuffle.py | 1D circular place field shuffle |
| `shuffle_place_fields_circular_2d()` | decoding/shuffle.py | 2D circular place field shuffle |
| `shuffle_posterior_circular()` | decoding/shuffle.py | Posterior circular shuffle |
| `shuffle_posterior_weighted_circular()` | decoding/shuffle.py | Weighted circular posterior shuffle |
| `shuffle_trials()` | NEW | Shuffle trial labels |
| `shuffle_spikes_isi()` | NEW | Shuffle inter-spike intervals |
| `compute_shuffle_pvalue()` | decoding/shuffle.py | P-value from null distribution |
| `compute_shuffle_zscore()` | decoding/shuffle.py | Z-score from null distribution |
| `ShuffleTestResult` | decoding/shuffle.py | Result dataclass |

#### `stats/surrogates.py`

**From**: `decoding/shuffle.py` (extracted)

| Function/Class | Source | Description |
|---------------|--------|-------------|
| `generate_poisson_surrogates()` | decoding/shuffle.py | Homogeneous Poisson surrogates |
| `generate_inhomogeneous_poisson_surrogates()` | decoding/shuffle.py | Inhomogeneous Poisson surrogates |
| `generate_jittered_spikes()` | NEW | Temporal jitter surrogates |

---

### Private Utilities (Top-Level)

Keep at top level with underscore prefix — no need for `_internal/` directory.

- `_constants.py` — UNCHANGED
- `_logging.py` — UNCHANGED

---

## Files to Delete After Migration

These files become empty after moving their contents:

```
src/neurospatial/behavioral.py        → behavior/navigation.py
src/neurospatial/spatial.py           → ops/binning.py
src/neurospatial/distance.py          → ops/distance.py
src/neurospatial/field_ops.py         → ops/normalize.py
src/neurospatial/kernels.py           → ops/smoothing.py
src/neurospatial/primitives.py        → ops/graph.py
src/neurospatial/differential.py      → ops/calculus.py
src/neurospatial/transforms.py        → ops/transforms.py
src/neurospatial/alignment.py         → ops/alignment.py
src/neurospatial/calibration.py       → ops/transforms.py (merged)
src/neurospatial/basis.py             → ops/basis.py
src/neurospatial/reward.py            → behavior/reward.py
src/neurospatial/reference_frames.py  → ops/egocentric.py
src/neurospatial/spike_field.py       → encoding/place.py
src/neurospatial/object_vector_field.py → encoding/object_vector.py
src/neurospatial/spatial_view_field.py  → encoding/spatial_view.py
src/neurospatial/visibility.py        → ops/visibility.py
src/neurospatial/io.py                → io/files.py
src/neurospatial/visualization/       → animation/config.py (then delete visualization/)
src/neurospatial/nwb/                 → io/nwb/
src/neurospatial/metrics/             → distributed to encoding/, behavior/, stats/
src/neurospatial/segmentation/        → behavior/segmentation.py
```

---

## Implementation Order

### Phase 1: Create New Directory Structure

1. Create directories: `encoding/`, `behavior/`, `io/`, `ops/`, `stats/`
2. Create `__init__.py` files for each
3. Keep `_constants.py` and `_logging.py` at top level (UNCHANGED)

### Phase 2: Move Core Modules (No Dependencies on domain modules)

1. `ops/` — Move and rename low-level utilities first (no internal deps)
   - binning.py (was spatial.py), distance.py, normalize.py (was field_ops.py)
   - smoothing.py (was kernels.py), graph.py (was primitives.py), calculus.py (was differential.py)
   - transforms.py (merge with calibration.py), alignment.py
   - egocentric.py (was reference_frames.py)
   - visibility.py, basis.py
2. `io/` — Move file I/O and nwb/

### Phase 3: Move Behavioral Analysis

1. `behavior/trajectory.py` — From metrics/trajectory.py
2. `behavior/segmentation.py` — From segmentation/
3. `behavior/navigation.py` — From behavioral.py + metrics/path_efficiency.py + metrics/goal_directed.py
4. `behavior/decisions.py` — From metrics/decision_analysis.py + metrics/vte.py
5. `behavior/reward.py` — From reward.py

### Phase 4: Move Neural Encoding

1. `encoding/place.py` — From spike_field.py + metrics/place_fields.py
2. `encoding/grid.py` — From metrics/grid_cells.py
3. `encoding/head_direction.py` — From metrics/head_direction.py
4. `encoding/border.py` — From metrics/boundary_cells.py
5. `encoding/object_vector.py` — From object_vector_field.py + metrics/object_vector_cells.py
6. `encoding/spatial_view.py` — From spatial_view_field.py + metrics/spatial_view_cells.py (SVC-specific only)
7. `encoding/phase_precession.py` — From metrics/phase_precession.py
8. `encoding/population.py` — From metrics/population.py

### Phase 5: Move Statistics

1. `stats/circular.py` — From metrics/circular.py + metrics/circular_basis.py
2. `stats/shuffle.py` — From decoding/shuffle.py (generalized)
3. `stats/surrogates.py` — Extract from decoding/shuffle.py

### Phase 6: Reorganize Decoding

1. Remove shuffle.py (moved to stats/)
2. Keep remaining decoding/ structure
3. Ensure clean `__init__.py` exports

### Phase 7: Consolidate Animation

1. Move `visualization/scale_bar.py` → `animation/config.py`
2. Delete empty `visualization/` directory
3. Keep existing `animation/` structure

### Phase 8: Update Top-Level `__init__.py`

1. Reduce to ~15 core exports
2. Update docstring

### Phase 9: Update All Internal Imports

For each moved module:

1. **Find old imports**:

   ```bash
   grep -r "from neurospatial.spike_field import" src/ tests/
   grep -r "from neurospatial import.*skaggs_information" src/ tests/
   ```

2. **Update to new paths**:

   ```python
   # Old
   from neurospatial.spike_field import compute_place_field
   from neurospatial import skaggs_information

   # New
   from neurospatial.encoding.place import compute_place_field, skaggs_information
   ```

3. **Run tests after EACH file update**:

   ```bash
   uv run pytest tests/ -x -v
   ```

4. **Run mypy to catch missed imports**:

   ```bash
   uv run mypy src/neurospatial/
   ```

5. **Use ruff to fix import sorting**:

   ```bash
   uv run ruff check --fix .
   ```

**Key import mappings** (complete list in Files to Delete section):

- `neurospatial.spike_field` → `neurospatial.encoding.place`
- `neurospatial.visibility` → `neurospatial.ops.visibility`
- `neurospatial.reference_frames` → `neurospatial.ops.egocentric`
- `neurospatial.spatial` → `neurospatial.ops.binning`
- `neurospatial.field_ops` → `neurospatial.ops.normalize`
- `neurospatial.kernels` → `neurospatial.ops.smoothing`
- `neurospatial.primitives` → `neurospatial.ops.graph`
- `neurospatial.differential` → `neurospatial.ops.calculus`
- `neurospatial.metrics.place_fields` → `neurospatial.encoding.place`
- `neurospatial.metrics.circular` → `neurospatial.stats.circular`

### Phase 10: Delete Old Files

1. Remove empty/migrated files
2. Remove old directories

### Phase 11: Update Documentation

1. Update CLAUDE.md
2. Update all .claude/*.md files
3. Update mkdocs.yml
4. Update docstrings with new import paths

### Phase 12: Update Example Notebooks

1. Update imports in all `examples/*.ipynb` (~22 notebooks)
2. Update imports in all `docs/examples/*.ipynb` (mirrors of examples/)
3. Re-run notebooks to verify they execute without errors
4. Clear notebook outputs and re-execute for clean state

**Key notebooks requiring import updates:**

- `08_spike_field_basics.ipynb` — uses `spike_field` → `encoding.place`
- `11_place_field_analysis.ipynb` — uses `metrics.place_fields` → `encoding.place`
- `12_boundary_cell_analysis.ipynb` — uses `metrics.boundary_cells` → `encoding.border`
- `13_trajectory_analysis.ipynb` — uses `metrics.trajectory` → `behavior.trajectory`
- `14_behavioral_segmentation.ipynb` — uses `segmentation` → `behavior.segmentation`
- `20_bayesian_decoding.ipynb` — uses `decoding` (verify exports)
- `22_spatial_view_cells.ipynb` — uses `spatial_view_field` → `encoding.spatial_view`

**Verification:**

```bash
# Run all notebooks to check for import errors
for nb in examples/*.ipynb; do
    uv run jupyter nbconvert --to notebook --execute "$nb" --output /dev/null
done
```

---

## Testing Strategy

After each phase:

```bash
uv run pytest tests/ -x -v
uv run ruff check .
uv run mypy src/neurospatial/
uv run pytest --doctest-modules src/neurospatial/
```

Create `tests/test_imports.py` to verify:

1. All new import paths work
2. Core exports from top-level work
3. No circular imports
4. Docstrings have correct import examples

---

## Example Usage After Reorganization

```python
import neurospatial as ns
from neurospatial import Environment, Region

# Create environment
env = Environment.from_samples(positions, bin_size=2.0)

# Neural encoding
from neurospatial.encoding import place, grid, head_direction

rate_map = place.compute_place_field(env, spikes, times, positions)
info = place.skaggs_information(rate_map, occupancy)
fields = place.detect_place_fields(rate_map)

gs = grid.grid_score(rate_map)

hd_curve = head_direction.head_direction_tuning_curve(spikes, headings, times)

# Neural decoding (with re-exported shuffle functions)
from neurospatial.decoding import decode_position, shuffle_time_bins, compute_shuffle_pvalue

result = decode_position(env, spikes, rate_maps, times)
p_value = compute_shuffle_pvalue(result, n_shuffles=1000)  # re-exported from stats

# OR: Import from canonical location (power user)
from neurospatial.stats import shuffle, circular

p_value = shuffle.compute_shuffle_pvalue(result, n_shuffles=1000)

# HD cell analysis with re-exported circular stats
from neurospatial.encoding.head_direction import (
    head_direction_tuning_curve,
    rayleigh_test,  # re-exported from stats/circular
)
r_stat, p_rayleigh = rayleigh_test(hd_curve)

# Behavioral analysis
from neurospatial.behavior import segmentation, navigation, trajectory

laps = segmentation.detect_laps(env, positions, times)
efficiency = navigation.path_efficiency(env, positions, start, goal)
msd = trajectory.mean_square_displacement(positions, times)

# Events
from neurospatial.events import peri_event_histogram

psth = peri_event_histogram(spikes, events, window=(-1, 2))

# Low-level ops (power users)
from neurospatial.ops import distance, normalize

dist = distance.distance_field(env, goal_bins)
normalized = normalize.normalize_field(rate_map)

# Visualization
from neurospatial.animation import overlays

pos_overlay = overlays.PositionOverlay(data=positions, color="red")
env.animate_fields(rate_maps, frame_times=times, overlays=[pos_overlay])
```

---

## Resolved Design Decisions

1. **`CompositeEnvironment`** — Keep at top level (commonly used)

2. **Plotting functions** — Keep with domain modules, NO re-exports
   - Avoids circular dependencies (animation → encoding → animation)
   - Each domain module owns its plotting functions
   - Users import plots from the module that computes the metric

3. **`graph_turn_sequence()`, `heading_direction_labels()`** — Keep in `behavior/navigation.py`
   - They're about behavior semantics, not graph algorithms

4. **`stats/` functions** — Canonical location in `stats/`, re-exported to domain modules
   - `stats/circular.py` → re-exported in `encoding/head_direction.py`
   - `stats/shuffle.py` → re-exported in `decoding/__init__.py`
   - Follows "dependencies flow inward" principle (Brandon Rhodes)

5. **`reference_frames.py` → `egocentric.py`** — Move to `ops/` (CHANGED from behavior/)
   - Coordinate transforms are low-level primitives
   - Used by encoding (OVC, SVC) AND behavior
   - Moving to ops/ eliminates encoding → behavior dependency
   - Follows Rhodes' "dependencies flow inward" principle

6. **`visibility.py`** — Move entirely to `ops/visibility.py`
   - All visibility functions stay together (viewshed, gaze, occupancy)
   - `encoding/spatial_view.py` re-exports relevant functions for SVC workflow convenience
   - Keeps related functionality cohesive rather than splitting by abstract categorization

7. **`basis.py`** — Move to `ops/` (CHANGED from encoding/)
   - Used by encoding (GLM place fields) AND decoding (basis priors)
   - General-purpose spatial basis functions
   - `encoding/` can re-export if desired for discoverability

---

## Summary

| Metric | Before | After |
|--------|--------|-------|
| Top-level exports | 115 | 5 |
| Top-level modules | 22 .py files | 6 subpackages |
| Depth to cell types | 2 (neurospatial.metrics.place_fields) | 2 (neurospatial.encoding.place) |
| Domain grouping | By function (compute vs metrics) | By science (place, grid, HD) |
| Behavioral scatter | 3 locations | 1 location (behavior/) |
| Visualization scatter | 2 locations (animation/, visualization/) | 1 location (animation/) |
| Statistics scatter | 2 locations (metrics/, decoding/) | 1 location (stats/) |
| Primitives scatter | 8+ top-level files | 1 location (ops/) |

**Key wins:**

1. Clear domain organization: encoding/ vs decoding/ vs behavior/
2. All cell types in one place: encoding/
3. All behavior in one place: behavior/
4. Animation consolidated: animation/ (keep existing, absorb visualization/)
5. All statistics in one place: stats/
6. All primitives in one place: ops/ (transforms, visibility, basis, distance, etc.)
7. Sparse top-level for better autocomplete
8. Flat structure (Raymond Hettinger approved)
9. Clean dependency graph (Brandon Rhodes approved) — no circular imports
