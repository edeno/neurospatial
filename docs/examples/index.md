# Examples

Real-world examples demonstrating neurospatial's capabilities through
interactive Jupyter notebooks. In the `examples/` directory each notebook
is a jupytext pair — a `.py` (percent format) and a `.ipynb` — so editing
either keeps the pair in sync via jupytext. A separate step,
`docs/sync_notebooks.py`, then copies `examples/` → `docs/examples/` to
produce the committed docs mirror shown here.

This page lists the example notebooks; see `examples/README.md` in the repo
for the full catalog and details.

## Goal → Notebook

If you have a specific neuroscience question, jump straight to the
relevant notebook.

| Goal | Notebook(s) |
| --- | --- |
| Build an environment from position data | [01](01_introduction_basics.ipynb), [03](03_morphological_operations.ipynb) |
| Pick the right layout (grid / hex / polygon / graph) | [02](02_layout_engines.ipynb), [05](05_track_linearization.ipynb) |
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
| Animate fields (napari / video / HTML / widget) | [16](16_field_animation.ipynb), [17](17_animation_with_overlays.ipynb) |
| Composite raw video with spatial fields | [18](18_video_overlay.ipynb) |
| Simulate trajectories and spike trains | [15](15_simulation_workflows.ipynb) |
| Path progression along a linearized track | [23](23_path_progression.ipynb) |
| Differential operators, signal processing on graphs | [09](09_differential_operators.ipynb), [10](10_signal_processing_primitives.ipynb) |
| Multi-room / composite environments | [06](06_composite_environments.ipynb), [07](07_advanced_operations.ipynb) |
| Real-data end-to-end pipeline | [19](19_real_data_bandit_task.ipynb) |

## Foundations (01-07)

### [01. Introduction & Basics](01_introduction_basics.ipynb)

**Time**: 15-20 min · **Prerequisites**: none

Spatial discretization, creating an environment from position data,
`bin_size` and active bins, basic spatial queries (`bin_at`,
`contains`, `neighbors`), visualization, common pitfalls. **Start
here.**

### [02. Layout Engines](02_layout_engines.ipynb)

**Time**: 20-25 min · **Prerequisites**: 01

Regular grids vs hexagonal tessellations, polygon-bounded
environments, connectivity patterns, side-by-side comparison.

### [03. Morphological Operations](03_morphological_operations.ipynb)

**Time**: 15-20 min · **Prerequisites**: 01

Active bin inference, `dilate` / `fill_holes` / `close_gaps`,
`bin_count_threshold`, fixing fragmented environments.

### [04. Regions of Interest](04_regions_of_interest.ipynb)

**Time**: 15-20 min · **Prerequisites**: 01

Creating point and polygon regions, region operations, using regions
in analysis, serialization.

### [05. Track Linearization](05_track_linearization.ipynb)

**Time**: 20-25 min · **Prerequisites**: 01, 02

Building 1D environments from track graphs, mapping 2D position to
linear coordinates, T-maze / W-maze / plus-maze examples.

### [06. Composite Environments](06_composite_environments.ipynb)

**Time**: 25-30 min · **Prerequisites**: 01, 02

Multi-room arenas, automatic bridge edges via mutual nearest neighbors,
querying across rooms, plus-maze with separate compartments.

### [07. Advanced Operations](07_advanced_operations.ipynb)

**Time**: 20-30 min · **Prerequisites**: 01, 06

Subsetting, applying transforms, custom layouts via `from_layout`.

## Spatial Coding (08-12)

### [08. Spike-Field Basics](08_spike_field_basics.ipynb)

**Time**: 25-35 min · **Prerequisites**: 01

Mapping spike times to bin indices, computing occupancy and rate maps
from scratch, smoothing methods (diffusion KDE, gaussian KDE, binned).

### [09. Differential Operators](09_differential_operators.ipynb)

**Time**: 25-35 min · **Prerequisites**: 02

Edge-oriented differential operator, gradient and divergence on graphs,
laplacian, signal-processing primitives.

### [10. Signal Processing Primitives](10_signal_processing_primitives.ipynb)

**Time**: 25-35 min · **Prerequisites**: 09

Graph kernels, spatial basis functions (geodesic RBF, heat kernel,
Chebyshev), smoothing with `compute_kernel` and `apply_kernel`.

### [11. Place Field Analysis](11_place_field_analysis.ipynb)

**Time**: 30-40 min · **Prerequisites**: 08

Build place fields from spike trains, detect place fields with
`detect_place_fields`, compute spatial information / sparsity /
selectivity, analyze field sizes and centroids.

### [12. Boundary Cell Analysis](12_boundary_cell_analysis.ipynb)

**Time**: 25-35 min · **Prerequisites**: 11

Distance-to-boundary fields, border score, classifying boundary cells.

## Behavior (13-15)

### [13. Trajectory Analysis](13_trajectory_analysis.ipynb)

**Time**: 25-35 min · **Prerequisites**: 01

Step lengths, turn angles, MSD, curvature, home range, persistence.

### [14. Behavioral Segmentation](14_behavioral_segmentation.ipynb)

**Time**: 25-35 min · **Prerequisites**: 04, 13

Lap detection, trial segmentation, region crossings, velocity-based
movement / rest segmentation.

### [15. Simulation Workflows](15_simulation_workflows.ipynb)

**Time**: 30-40 min · **Prerequisites**: 11

Generate synthetic trajectories and spike trains, ground-truth
validation, `simulate_session` API, pre-configured examples.

## Animation & Visualization (16-18)

### [16. Field Animation](16_field_animation.ipynb)

**Time**: 20-30 min · **Prerequisites**: 08

Animate spatial fields over time using napari / HTML / widget / video
backends.

### [17. Animation with Overlays](17_animation_with_overlays.ipynb)

**Time**: 25-35 min · **Prerequisites**: 16

Position, pose, head-direction, event, and spike overlays composed on
field animations.

### [18. Video Overlay](18_video_overlay.ipynb)

**Time**: 25-35 min · **Prerequisites**: 17

Composite raw behavioral video with spatial-field overlays for
publication / talks.

## Advanced Neural Coding (19-27)

### [19. Real-Data Bandit Task](19_real_data_bandit_task.ipynb)

**Time**: 40-60 min · **Prerequisites**: 05, 11, 13

End-to-end pipeline on a hippocampal recording from a plus-maze
bandit task. Loads the J16 dataset (see `data/README.md` for download
instructions); the notebook prints the download URL and exits cleanly
if the data files are missing.

### [20. Bayesian Decoding](20_bayesian_decoding.ipynb)

**Time**: 35-45 min · **Prerequisites**: 11, 15

Decode position from population spike counts, MAP / posterior-mean
estimates, decoding-error metrics, sequential decoders.

### [21. Directional Place Fields](21_directional_place_fields.ipynb)

**Time**: 25-35 min · **Prerequisites**: 11

Direction-conditioned place fields on linear tracks; building rate
maps separately per running direction.

### [22. Spatial View Cells](22_spatial_view_cells.ipynb)

**Time**: 30-40 min · **Prerequisites**: 11

Build firing-rate fields indexed by gaze direction rather than
position. Three gaze models (fixed-distance, ray-cast, boundary).
Classification via `is_spatial_view_cell`.

### [23. Path Progression](23_path_progression.ipynb)

**Time**: 25-35 min · **Prerequisites**: 05, 14

Path progression along linearized tracks; per-trial alignment for
sequential analyses.

### [24. Object-Vector Cells](24_object_vector_cells.ipynb)

**Time**: 30-40 min · **Prerequisites**: 11

Object-vector cells: firing tuned to (distance, egocentric direction)
to external landmarks. Uses `compute_egocentric_rate` and
`is_object_vector_cell` classification.

### [25. Head-Direction Tuning](25_head_direction_tuning.ipynb)

**Time**: 25-35 min · **Prerequisites**: 08

Build a circular tuning curve over head direction, Rayleigh test for
significant directionality, `is_head_direction_cell` classification,
polar plotting.

### [26. Peri-Event PSTH](26_peri_event_psth.ipynb)

**Time**: 25-35 min · **Prerequisites**: 11

Align spikes to discrete event times (reward arrivals, lap starts).
Compute peri-event histograms, plot rasters, build GLM regressors.

### [27. Loading from NWB](27_loading_from_nwb.ipynb)

**Time**: 25-35 min · **Prerequisites**: 01, 11

Read position, spikes, and environment metadata from a synthetic in-memory
NWB file. Build a `neurospatial` environment, recover place fields, write
derived results back into the NWB; the same API applies to downloaded DANDI
files.

---

!!! note "Editing notebooks"
    The notebooks displayed here are mirrored from the `examples/`
    directory and the mirror is committed to the repository.

    **To update notebooks in the documentation:**

    1. Edit notebooks in the `examples/` directory (repository root)
    2. Run `uv run python docs/sync_notebooks.py` to regenerate the
       mirror under `docs/examples/`
    3. Commit the regenerated `docs/examples/` files alongside your
       `examples/` changes. CI **verifies** the committed mirror is in
       sync (a `git diff --exit-code docs/examples` guard in
       `.github/workflows/docs.yml`) and **fails the build** if it is
       out of date — it does not auto-sync or commit for you.

    **Do not** edit the generated `.ipynb` / `.py` files in
    `docs/examples/` directly - they are overwritten by the sync script.
