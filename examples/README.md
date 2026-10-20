# Spatial neuroscience tutorials

These nine notebooks work through questions about occupancy, spatial firing,
maze behavior, position decoding, and responses to cues. They are written for
spatial neuroscientists who work with spike times and tracking data and know
basic Python and NumPy.

Choose the question that matches your analysis. Each notebook provides its own
recording, explains the analysis choices, and shows how to read the resulting
figures. A final section describes how to use your own recording. Start a fresh
kernel and run the cells in order; the notebooks do not share variables or
require files produced by another notebook.

## Start here

From a repository checkout:

```bash
uv sync --extra notebooks
uv run jupyter lab examples/
```

To run the complete collection, including video and NWB:

```bash
uv sync --extra dev --extra notebooks --extra video --extra nwb
uv run jupyter lab examples/
```

If you are new to neurospatial, start with
[occupancy](01_getting_started.ipynb) and
[firing-rate maps](02_spatial_rate_maps.ipynb). The background links in the table
introduce package concepts used later; each notebook can run on its own.

<!-- notebook-catalog:start -->

| Notebook | Estimated time | Useful background |
| --- | --- | --- |
| [01. Where did the animal spend time?](01_getting_started.ipynb) | 15-20 minutes | None |
| [02. Where does this neuron fire after accounting for occupancy?](02_spatial_rate_maps.ipynb) | 20-25 minutes | [01](01_getting_started.ipynb) |
| [03. When was the animal running, and which traversals were fully observed?](03_trajectories_and_behavior.ipynb) | 20-25 minutes | [01](01_getting_started.ipynb) |
| [04. Where do hippocampal neurons fire along this maze?](04_real_data_analysis.ipynb) | 25-30 minutes | [02](02_spatial_rate_maps.ipynb), [03](03_trajectories_and_behavior.ipynb) |
| [05. Can we recover a known place field from simulated spikes?](05_simulation_and_validation.ipynb) | 15-20 minutes | [02](02_spatial_rate_maps.ipynb) |
| [06. How accurately can population activity predict position?](06_position_decoding.ipynb) | 20-25 minutes | [02](02_spatial_rate_maps.ipynb), [05](05_simulation_and_validation.ipynb) |
| [07. How does firing change around a cue?](07_peri_event_analysis.ipynb) | 15-20 minutes | [02](02_spatial_rate_maps.ipynb) |
| [08. Inspect spatial maps alongside behavior and video](08_visualization.ipynb) | 15-20 minutes | [02](02_spatial_rate_maps.ipynb) |
| [09. Analyze an NWB recording and save the results](09_nwb_io.ipynb) | 15-20 minutes | [02](02_spatial_rate_maps.ipynb) |

<!-- notebook-catalog:end -->

## Analyze recorded hippocampal activity

[The real-data tutorial](04_real_data_analysis.ipynb) downloads the 75 MB
[J16 Zenodo recording](https://doi.org/10.5281/zenodo.23020757), verifies its
SHA-256 and reuses the cache on subsequent runs. It combines neural rate maps
and maze behavior in one analysis. To prepare for offline work, first run:

```bash
uv run python data/load_bandit_data.py
```

The [data guide](https://github.com/edeno/neurospatial/blob/main/data/README.md)
describes units, preprocessing, attribution and the CC BY 4.0 license.

## Explore focused recipes

The core tutorials teach complete analyses. More specialized choices have
focused recipes and explanations in the user guide:

- [Alternative layouts](https://edeno.github.io/neurospatial/user-guide/layout-engines/),
  [morphology](https://edeno.github.io/neurospatial/user-guide/environments/) and
  [composite environments](https://edeno.github.io/neurospatial/user-guide/composite-environments/).
- [Coordinate alignment](https://edeno.github.io/neurospatial/user-guide/alignment/),
  [differential operators](https://edeno.github.io/neurospatial/user-guide/differential-operators/) and
  [spatial signal processing](https://edeno.github.io/neurospatial/user-guide/signal-processing-primitives/).
- [Specialized tuning recipes](https://edeno.github.io/neurospatial/user-guide/neuroscience-metrics/#specialized-tuning-recipes)
  for boundary, grid, directional, head-direction, object-vector and spatial-view analyses.
- [Animation backends and large recordings](https://edeno.github.io/neurospatial/user-guide/animation/)
  and [video calibration](https://edeno.github.io/neurospatial/user-guide/video-annotation/).
- [Interactive annotation launcher](try_annotation.py): try CPU or optional SAM 2
  outlines on a sample arena, image, or video. See the
  [setup and controls](../docs/user-guide/video-annotation.md#try-the-sample-arena).

## Reading and adapting the analyses

- Positions and distances use the environment's declared units. Times are
  seconds and angles are radians. Allocentric angles use zero East and positive
  angles toward North; egocentric angles use zero ahead and positive angles left.
- Blank rate-map locations have too little recorded occupancy. A measured rate
  of zero means the animal spent time there while spikes were recorded.
- A place-field outline depends on the rate and area thresholds used to draw
  it. A claim about spatial tuning needs an appropriate statistical test.
- Decode position in a different part of the recording from the one used to
  estimate place fields. Choose parameters before examining test performance.
- Position can identify visits to maze arms. Use task logs to identify trials,
  reward delivery, and the choices available to the animal.

## Optional viewers and exports

Static figures and small HTML examples run by default. Desktop viewers,
interactive widgets and longer video exports require explicit opt-in flags
where provided (`RUN_NAPARI`, `RUN_WIDGET`, `RUN_VIDEO_EXPORT`). The visualization
tutorial creates its own short video; no private behavioral video is needed.

## Edit and verify

The `.py` companions are Jupytext sources. Edit the source, synchronize the
notebook, execute it to refresh figures, then update the documentation mirror:

```bash
uv run jupytext --sync examples/02_spatial_rate_maps.py
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 examples/02_spatial_rate_maps.ipynb
uv run python docs/sync_notebooks.py
uv run pytest tests/docs/test_notebook_collection.py -n 0
```

The documentation includes this catalog and mirrors the notebooks. Edit
`examples/`, not generated files under `docs/examples/`. CI checks source parity,
metadata, links and prerequisites, and executes all nine notebooks in fresh
kernels with optional desktop viewers disabled.
