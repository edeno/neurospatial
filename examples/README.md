# neurospatial Examples

This directory contains Jupyter notebook tutorials demonstrating the features and capabilities of the neurospatial package. Each notebook is a jupytext pair — a `.py` (percent format) and a `.ipynb` — so editing either keeps the pair in sync via jupytext, within `examples/`. A separate step, `docs/sync_notebooks.py`, then copies `examples/` → `docs/examples/` to produce the committed docs mirror.

## Getting Started

### Prerequisites

Install neurospatial:

```bash
# From the project root
uv sync --extra notebooks

# Or with pip
pip install -e ".[notebooks]"
```

### Running the Notebooks

From the project root directory:

```bash
# Launch Jupyter notebook
uv run jupyter notebook examples/

# Or with JupyterLab
uv run jupyter lab examples/
```

## Goal-Oriented Index

If you have a specific neuroscience question, jump directly to the relevant notebook.

| Goal | Notebook(s) |
| --- | --- |
| Build an environment from position data | [01](01_introduction_basics.ipynb), [03](03_morphological_operations.ipynb) |
| Pick the right layout (grid, hex, polygon, graph) | [02](02_layout_engines.ipynb), [05](05_track_linearization.ipynb) |
| Define spatial regions / zones of interest | [04](04_regions_of_interest.ipynb) |
| Compute occupancy and a spatial firing-rate map | [08](08_spike_field_basics.ipynb), [11](11_place_field_analysis.ipynb) |
| Detect place fields and compute place-cell metrics | [11](11_place_field_analysis.ipynb) |
| Direction-conditioned place fields | [21](21_directional_place_fields.ipynb) |
| Boundary / border cells | [12](12_boundary_cell_analysis.ipynb) |
| Spatial view cells | [22](22_spatial_view_cells.ipynb) |
| Object-vector cells | [24](24_object_vector_cells.ipynb) |
| Head-direction tuning | [25](25_head_direction_tuning.ipynb) |
| Peri-event histograms (PSTH) | [26](26_peri_event_psth.ipynb) |
| Load data from an NWB file | [27](27_loading_from_nwb.ipynb) |
| Trajectory metrics (step length, MSD, curvature) | [13](13_trajectory_analysis.ipynb) |
| Detect laps, trials, and region crossings | [14](14_behavioral_segmentation.ipynb) |
| Bayesian decoding from population spikes | [20](20_bayesian_decoding.ipynb) |
| Animate fields over time (napari / video / html / widget) | [16](16_field_animation.ipynb), [17](17_animation_with_overlays.ipynb) |
| Composite raw behavioral video with spatial fields | [18](18_video_overlay.ipynb) |
| Simulate trajectories and spike trains | [15](15_simulation_workflows.ipynb) |
| Path progression along a linearized track | [23](23_path_progression.ipynb) |
| Differential operators, signal processing on graphs | [09](09_differential_operators.ipynb), [10](10_signal_processing_primitives.ipynb) |
| Multi-room / composite environments | [06](06_composite_environments.ipynb), [07](07_advanced_operations.ipynb) |
| Real-data end-to-end pipeline | [19](19_real_data_bandit_task.ipynb) |

## Notebook Series

The tutorials are designed to be completed in order, each building on concepts from previous notebooks. Notebooks 09+ are domain-specific and can be skipped to based on your needs.

### Foundations (01-07)

#### 01. Introduction to neurospatial: The Basics

**File**: [01_introduction_basics.ipynb](01_introduction_basics.ipynb) · **Time**: 15-20 minutes
Topics: spatial discretization, creating an environment from position data, `bin_size` and active bins, basic spatial queries (`bin_at`, `contains`, `neighbors`), visualization with `plot()`, common pitfalls. **Start here** if you're new to neurospatial.

#### 02. Layout Engines

**File**: [02_layout_engines.ipynb](02_layout_engines.ipynb) · **Time**: 20-25 minutes · **Prerequisites**: 01
Topics: regular grids vs hexagonal tessellations, polygon-bounded environments, connectivity patterns, comparing layouts side-by-side.

#### 03. Morphological Operations

**File**: [03_morphological_operations.ipynb](03_morphological_operations.ipynb) · **Time**: 15-20 minutes · **Prerequisites**: 01
Topics: active bin inference, `dilate` / `fill_holes` / `close_gaps`, `bin_count_threshold`, fixing fragmented environments.

#### 04. Regions of Interest

**File**: [04_regions_of_interest.ipynb](04_regions_of_interest.ipynb) · **Time**: 15-20 minutes · **Prerequisites**: 01
Topics: point and polygon regions, `Regions` API, region operations (buffering, area, center), Morris water maze example, JSON serialization.

#### 05. Track Linearization

**File**: [05_track_linearization.ipynb](05_track_linearization.ipynb) · **Time**: 25-30 minutes · **Prerequisites**: 01-02
Topics: 1D linearization, graph-based environments, linear track and plus-maze examples, `edge_order` and `edge_spacing`, 2D ↔ 1D conversion.

#### 06. Composite Environments

**File**: [06_composite_environments.ipynb](06_composite_environments.ipynb) · **Time**: 20-25 minutes · **Prerequisites**: 01-02
Topics: merging multiple environments, automatic bridge inference, multi-room experiments, `max_mnn_distance`.

#### 07. Advanced Operations

**File**: [07_advanced_operations.ipynb](07_advanced_operations.ipynb) · **Time**: 20-25 minutes · **Prerequisites**: 01-02
Topics: shortest paths, geodesic vs Euclidean distances, 2D coordinate transforms, mapping probability distributions between environments, graph centrality.

### Spike-Field Basics (08)

#### 08. Spike-Field Basics: From Tracking Data to Place Fields

**File**: [08_spike_field_basics.ipynb](08_spike_field_basics.ipynb) · **Time**: 30-40 minutes · **Prerequisites**: 01, 03, 04
Topics: end-to-end pipeline from raw tracking data to place fields, occupancy-normalized firing rates, spatial information, place cell detection, multi-region analysis, publication-quality visualization.

### Graph & Signal Primitives (09-10)

#### 09. Differential Operators on Spatial Graphs

**File**: [09_differential_operators.ipynb](09_differential_operators.ipynb) · **Time**: 15-20 minutes · **Prerequisites**: 01-02
Topics: gradient and divergence on graphs, applications to RL, replay analysis, flow fields.

#### 10. Signal Processing Primitives

**File**: [10_signal_processing_primitives.ipynb](10_signal_processing_primitives.ipynb) · **Time**: 15-20 minutes · **Prerequisites**: 01-02
Topics: `neighbor_reduce()`, `graph_convolve()` for custom filtering and local aggregation on graph-based environments.

### Cell-Type Analyses (11-12, 21-22)

#### 11. Place Field Analysis

**File**: [11_place_field_analysis.ipynb](11_place_field_analysis.ipynb) · **Time**: 30-35 minutes · **Prerequisites**: 08
Topics: detecting place fields, place-field metrics (size, stability, centroid, EMD), spatial information, classification.

#### 12. Boundary Cell Analysis

**File**: [12_boundary_cell_analysis.ipynb](12_boundary_cell_analysis.ipynb) · **Time**: 15-20 minutes · **Prerequisites**: 08, 11
Topics: border cells (Solstad, Boccara et al. 2008), `border_score`, geodesic boundary detection.

#### 21. Directional Place Fields

**File**: [21_directional_place_fields.ipynb](21_directional_place_fields.ipynb) · **Time**: 20-25 minutes · **Prerequisites**: 08, 11
Topics: direction-conditioned firing-rate maps, splitting place fields by movement direction, comparing left-vs-right traversals on a linear track, quantifying directional tuning.

#### 22. Spatial View Cell Analysis

**File**: [22_spatial_view_cells.ipynb](22_spatial_view_cells.ipynb) · **Time**: 20-25 minutes · **Prerequisites**: 08, 11
Topics: cells that fire based on the *viewed* location (not the animal's position), simulating view cells, `compute_view_rate`, gaze models, classification.

### Behavior (13-14)

#### 13. Trajectory Analysis

**File**: [13_trajectory_analysis.ipynb](13_trajectory_analysis.ipynb) · **Time**: 10-15 minutes · **Prerequisites**: 01
Topics: step lengths, turn angles, mean-square displacement, home range, curvature.

#### 14. Behavioral Segmentation

**File**: [14_behavioral_segmentation.ipynb](14_behavioral_segmentation.ipynb) · **Time**: 15-20 minutes · **Prerequisites**: 01, 04
Topics: detecting laps, trials, region crossings, runs between regions; `segment_trials`, `detect_laps`, `detect_region_crossings`.

### Simulation (15)

#### 15. Simulation Workflows

**File**: [15_simulation_workflows.ipynb](15_simulation_workflows.ipynb) · **Time**: 30-40 minutes · **Prerequisites**: 01, 08
Topics: trajectory simulation (Ornstein-Uhlenbeck), neural cell models (place, grid, head-direction, boundary, OVC, view), spike-train generation.

### Visualization (16-18)

#### 16. Field Animation

**File**: [16_field_animation.ipynb](16_field_animation.ipynb) · **Time**: 20-25 minutes · **Prerequisites**: 08
Topics: four animation backends — napari, video (mp4), html, jupyter widget — when to use each, performance considerations.

#### 17. Animation with Overlays

**File**: [17_animation_with_overlays.ipynb](17_animation_with_overlays.ipynb) · **Time**: 20-25 minutes · **Prerequisites**: 16
Topics: position trails, bodypart pose tracking, head-direction arrows, multi-animal support, region overlays.

#### 18. Video Overlay

**File**: [18_video_overlay.ipynb](18_video_overlay.ipynb) · **Time**: 25-30 minutes · **Prerequisites**: 16-17
Topics: compositing raw behavioral video frames with spatial fields, scale-bar and landmark calibration, `VideoOverlay`.

### Real Data & Decoding (19-20)

#### 19. Real Data: Hippocampal Place Cells on a Bandit Task

**File**: [19_real_data_bandit_task.ipynb](19_real_data_bandit_task.ipynb) · **Time**: 45-60 minutes · **Prerequisites**: 08, 11, 14
Topics: full workflow on a real rat-hippocampus dataset (203 units, 24 minutes, bandit task with 3 reward patches). The dataset is not bundled with the repo; see the notebook's data-loading section for the download link.

#### 20. Bayesian Position Decoding

**File**: [20_bayesian_decoding.ipynb](20_bayesian_decoding.ipynb) · **Time**: 30-40 minutes · **Prerequisites**: 08, 11
Topics: encoding models from place fields, decoding position from population spikes, `DecodingResult` (posterior, MAP, mean, posterior entropy), credible regions, decoding error.

## Learning Path Recommendations

### Quick Start (≈90 min)

You want to see what neurospatial can do, quickly:

1. [01 Introduction Basics](01_introduction_basics.ipynb) (20 min)
2. [04 Regions](04_regions_of_interest.ipynb) (20 min)
3. [08 Spike-Field Basics](08_spike_field_basics.ipynb) (40 min) — skim for overview

### Comprehensive Foundations (≈3-4 hours)

For a thorough understanding of the discretization layer:

- Notebooks 01-08 in order.

### Specific Use Cases

- **Open-field rodent navigation**: 01, 03, 04, 08, 11, 19
- **Linear / plus / T-maze experiments**: 01, 02, 05, 14, 20
- **Multi-room / composite environments**: 01, 02, 06, 07
- **Custom arena shapes**: 01, 02, 04
- **Place / boundary / view cell screening**: 08, 11, 12, 22
- **Bayesian decoding pipeline**: 08, 11, 20
- **Animation and figures for talks / papers**: 16, 17, 18

## Getting Help

If you hit issues, open an issue on [GitHub](https://github.com/edeno/neurospatial/issues).

## Contributing

Found a typo or have a suggestion?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use neurospatial in your research, please cite:

```bibtex
@software{neurospatial2026,
  author = {Denovellis, Eric},
  title = {neurospatial: Spatial environment discretization for neuroscience},
  year = {2026},
  url = {https://github.com/edeno/neurospatial},
  version = {0.4.0}
}
```

## Additional Resources

- **Documentation**: <https://edeno.github.io/neurospatial/>
- **Main README**: [../README.md](../README.md)
- **Developer Guide**: [../CLAUDE.md](../CLAUDE.md)
- **GitHub Repository**: <https://github.com/edeno/neurospatial>
- **Issue Tracker**: <https://github.com/edeno/neurospatial/issues>

---

**Happy learning!**
