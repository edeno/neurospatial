# neurospatial Examples

These 27 notebooks form a collection of standalone tutorials. Each creates or
loads its own inputs, so you can follow a learning path without carrying kernel
state between notebooks. Run each notebook from top to bottom in a fresh kernel.

## Start here

From a checkout of the repository:

```bash
uv sync --extra notebooks
uv run jupyter lab examples/
```

For every tutorial, including synthetic video and NWB examples:

```bash
uv sync --extra dev --extra notebooks --extra video --extra nwb
uv run jupyter lab examples/
```

Notebooks 19 and 23 download the same 75 MB
[J16 recording](https://doi.org/10.5281/zenodo.23020757) once and verify its
SHA-256 before reusing the cached file. To prepare for offline work, run
`uv run python data/load_bandit_data.py` first. See the
[data guide](https://github.com/edeno/neurospatial/blob/main/data/README.md)
for units, provenance and attribution.

Desktop viewers are opt-in with `RUN_NAPARI = True`. Optional video exports use
`RUN_VIDEO_EXPORT = True` and require the dependencies described in the tutorial.
Interactive widgets use `RUN_WIDGET = True` where provided.
The default notebook path does not open a desktop viewer. Tutorial 18 creates
its own short synthetic video; it does not require a private behavioral video.

## Choose a learning path

Start with [01](01_introduction_basics.ipynb), then follow the branch that fits
what you want to do. Prerequisites introduce concepts; each notebook still
constructs its own data.

- **First spatial rate map:** [01](01_introduction_basics.ipynb) →
  [08](08_spike_field_basics.ipynb) → [11](11_place_field_analysis.ipynb).
- **Maze behavior and real data:** [05](05_track_linearization.ipynb) →
  [14](14_behavioral_segmentation.ipynb) → [19](19_real_data_bandit_task.ipynb)
  and [23](23_path_progression.ipynb).
- **Population decoding:** [11](11_place_field_analysis.ipynb) →
  [15](15_simulation_workflows.ipynb) → [20](20_bayesian_decoding.ipynb).
- **Specialized tuning:** [21](21_directional_place_fields.ipynb),
  [22](22_spatial_view_cells.ipynb), [24](24_object_vector_cells.ipynb) and
  [25](25_head_direction_tuning.ipynb).
- **Events and recorded data:** [26](26_peri_event_psth.ipynb) and
  [27](27_loading_from_nwb.ipynb).
- **Spatial operators:** [09](09_differential_operators.ipynb) →
  [10](10_signal_processing_primitives.ipynb).
- **Animation and video:** [16](16_field_animation.ipynb) →
  [17](17_animation_with_overlays.ipynb) → [18](18_video_overlay.ipynb).

## Complete catalog

Times estimate reading and exploration, rather than just execution. The catalog
is generated from each notebook's title, estimated time and prerequisite links.

<!-- notebook-catalog:start -->

| Notebook | Estimated time | Start after |
| --- | --- | --- |
| [01. Introduction to neurospatial: The Basics](01_introduction_basics.ipynb) | 15-20 minutes | None |
| [02. Layout Engines: Choosing the Right Spatial Discretization](02_layout_engines.ipynb) | 20-25 minutes | [01](01_introduction_basics.ipynb) |
| [03. Morphological Operations: Handling Sparse and Patchy Data](03_morphological_operations.ipynb) | 20-30 minutes | [01](01_introduction_basics.ipynb), [02](02_layout_engines.ipynb) |
| [04. Regions of Interest: Defining and Managing Spatial Regions](04_regions_of_interest.ipynb) | 20-25 minutes | [01](01_introduction_basics.ipynb), [02](02_layout_engines.ipynb) |
| [05. Track Linearization: Working with 1D Environments](05_track_linearization.ipynb) | 25-30 minutes | [01](01_introduction_basics.ipynb), [02](02_layout_engines.ipynb), [04](04_regions_of_interest.ipynb) |
| [06. Composite Environments: Merging Multiple Spaces](06_composite_environments.ipynb) | 25-30 minutes | [02](02_layout_engines.ipynb), [04](04_regions_of_interest.ipynb) |
| [07. Advanced Operations: Paths, Distances, and Alignment](07_advanced_operations.ipynb) | 30-35 minutes | [02](02_layout_engines.ipynb), [04](04_regions_of_interest.ipynb) |
| [08. Spike Field and Reward Primitives](08_spike_field_basics.ipynb) | 20-25 minutes | [01](01_introduction_basics.ipynb) |
| [09. Differential Operators on Spatial Graphs](09_differential_operators.ipynb) | 15-20 minutes | [02](02_layout_engines.ipynb) |
| [10. Signal Processing Primitives](10_signal_processing_primitives.ipynb) | 15-20 minutes | [09](09_differential_operators.ipynb), [08](08_spike_field_basics.ipynb) |
| [11. Place Field Analysis](11_place_field_analysis.ipynb) | 30-35 minutes | [08](08_spike_field_basics.ipynb) |
| [12. Boundary Cell Analysis](12_boundary_cell_analysis.ipynb) | 15-20 minutes | [11](11_place_field_analysis.ipynb) |
| [13. Trajectory Analysis](13_trajectory_analysis.ipynb) | 10-15 minutes | [01](01_introduction_basics.ipynb) |
| [14. Behavioral Segmentation](14_behavioral_segmentation.ipynb) | 15-20 minutes | [04](04_regions_of_interest.ipynb), [13](13_trajectory_analysis.ipynb) |
| [15. Simulation Workflows](15_simulation_workflows.ipynb) | 30-40 minutes | [11](11_place_field_analysis.ipynb) |
| [16. Field Animation Examples](16_field_animation.ipynb) | 15-20 minutes | [08](08_spike_field_basics.ipynb) |
| [17. Animation Overlays](17_animation_with_overlays.ipynb) | 20-25 minutes | [16](16_field_animation.ipynb) |
| [18. Video Overlay](18_video_overlay.ipynb) | 15-20 minutes | [17](17_animation_with_overlays.ipynb) |
| [19. Real Data: Hippocampal Spatial Coding on a Bandit Task](19_real_data_bandit_task.ipynb) | 30-40 minutes | [05](05_track_linearization.ipynb), [11](11_place_field_analysis.ipynb), [13](13_trajectory_analysis.ipynb) |
| [20. Bayesian Position Decoding](20_bayesian_decoding.ipynb) | 35-45 minutes | [11](11_place_field_analysis.ipynb), [15](15_simulation_workflows.ipynb) |
| [21. Directional Place Fields](21_directional_place_fields.ipynb) | 20-25 minutes | [11](11_place_field_analysis.ipynb) |
| [22. Spatial View Cell Analysis](22_spatial_view_cells.ipynb) | 15-20 minutes | [11](11_place_field_analysis.ipynb) |
| [23. Path Progression on a Track Graph (Bandit Task Real Data)](23_path_progression.ipynb) | 25-35 minutes | [05](05_track_linearization.ipynb), [14](14_behavioral_segmentation.ipynb) |
| [24. Object-Vector Cell Analysis](24_object_vector_cells.ipynb) | 20-25 minutes | [11](11_place_field_analysis.ipynb) |
| [25. Head Direction Tuning](25_head_direction_tuning.ipynb) | 15-20 minutes | [08](08_spike_field_basics.ipynb) |
| [26. Peri-Event PSTH](26_peri_event_psth.ipynb) | 20-25 minutes | [11](11_place_field_analysis.ipynb) |
| [27. Loading from NWB](27_loading_from_nwb.ipynb) | 20-25 minutes | [01](01_introduction_basics.ipynb), [11](11_place_field_analysis.ipynb) |

<!-- notebook-catalog:end -->

## Conventions across the tutorials

- Position and distance values use the environment's declared units; times are
  seconds and angles are radians. Allocentric angles use zero East and positive
  angles toward North; egocentric angles use zero ahead and positive angles left.
- Missing observations remain distinct from zero activity. Inspect occupancy and
  spatial support before interpreting rate maps or passing models to a decoder.
- Field detection segments a map. Threshold-based cell-type screens are
  descriptive candidates; a significance claim requires a stated statistical test.
- Training-set decoding describes a fit. Evaluation on new trials or time blocks
  requires a separate training/evaluation split, as demonstrated in notebook 20.
- Simulated examples have known inputs and limitations. Position-derived visits
  in real data do not establish recorded task trials, rewards or intended goals.

## Editing and checking the collection

Each notebook has a Jupytext `.py` companion in `examples/`. After editing a
companion, synchronize its pair, then refresh the catalog and documentation mirror:

```bash
uv run jupytext --sync examples/11_place_field_analysis.py
uv run python docs/sync_notebooks.py
uv run pytest tests/docs/test_notebook_collection.py -n 0
```

The documentation includes this same catalog. Edit the source notebooks and
this README rather than their generated copies under `docs/examples/`.

To execute a notebook and refresh its saved figures:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 examples/11_place_field_analysis.ipynb
uv run python docs/sync_notebooks.py
```

The notebook CI job executes the full collection with optional desktop viewers
disabled. For package documentation and help, see the
[user guide](https://edeno.github.io/neurospatial/) and
[issue tracker](https://github.com/edeno/neurospatial/issues).
