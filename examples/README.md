# neurospatial tutorials

Nine standalone notebooks follow common spatial-neuroscience tasks. Each creates
or loads its own inputs. Start a fresh kernel and run from top to bottom; no
notebook depends on another notebook's variables or generated files.

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

Begin with [getting started](01_getting_started.ipynb), then
[spatial rate maps](02_spatial_rate_maps.ipynb). Choose the next task from the
catalog. Prerequisites introduce concepts; each tutorial supplies its own data.

<!-- notebook-catalog:start -->

| Notebook | Estimated time | Start after |
| --- | --- | --- |
| [01. Getting started: define an arena and measure occupancy](01_getting_started.ipynb) | 15-20 minutes | None |
| [02. Spatial rate maps: estimate firing with observation support](02_spatial_rate_maps.ipynb) | 20-25 minutes | [01](01_getting_started.ipynb) |
| [03. Trajectories and behavior: segment shuttle traversals](03_trajectories_and_behavior.ipynb) | 20-25 minutes | [01](01_getting_started.ipynb) |
| [04. Analyze a Hippocampal Recording on a Maze](04_real_data_analysis.ipynb) | 25-30 minutes | [02](02_spatial_rate_maps.ipynb), [03](03_trajectories_and_behavior.ipynb) |
| [05. Simulation and Validation](05_simulation_and_validation.ipynb) | 15-20 minutes | [02](02_spatial_rate_maps.ipynb) |
| [06. Position Decoding](06_position_decoding.ipynb) | 20-25 minutes | [02](02_spatial_rate_maps.ipynb), [05](05_simulation_and_validation.ipynb) |
| [07. Peri-event analysis](07_peri_event_analysis.ipynb) | 15-20 minutes | [02](02_spatial_rate_maps.ipynb) |
| [08. Visualization](08_visualization.ipynb) | 15-20 minutes | [02](02_spatial_rate_maps.ipynb) |
| [09. NWB I/O](09_nwb_io.ipynb) | 15-20 minutes | [02](02_spatial_rate_maps.ipynb) |

<!-- notebook-catalog:end -->

## Work with a real recording

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

## Scientific conventions

- Positions and distances use the environment's declared units. Times are
  seconds and angles are radians. Allocentric angles use zero East and positive
  angles toward North; egocentric angles use zero ahead and positive angles left.
- Missing observations remain distinct from zero activity. Inspect occupancy and
  spatial support before interpreting rate maps or fitting a decoder.
- Field detection segments a map. Threshold screens produce descriptive
  candidates; a significance claim needs a stated, appropriate statistical test.
- Decoding performance is evaluated on separate observations. Choose parameters
  using training/validation data before inspecting the test block.
- Visits inferred from position do not establish task trials, rewards or goals.

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
