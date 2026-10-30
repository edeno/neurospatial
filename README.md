# neurospatial

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**neurospatial** is a Python library for spatial neuroscience. It helps scientists
analyze how animals move through space and how neural activity represents it,
with tools for environments, behavior, neural encoding and decoding, simulation,
and visualization.

Our highest design priority is the scientist's experience: common analyses should
be easy to discover, behave as a scientist reasonably expects, and produce
scientifically correct, interpretable results. Clear assumptions, consistent
interfaces, and useful error messages are part of that experience.

The long-term goal is to encompass the operations of spatial neuroscience in a
coherent workflow, from recorded data and environment definition through analysis,
statistical evaluation, visualization, and export. Spatial discretization and
connectivity graphs provide a shared foundation for these operations. See the
[design principles](docs/contributing.md#design-principles) for how we evaluate
changes toward this goal.

## Key Features

### Core Capabilities

- **Multiple Layout Engines**: Choose from regular grids, hexagonal tessellations, masked regions, polygon-bounded areas, triangular meshes, and 1D linearized tracks
- **Automatic Bin Detection**: Infer active bins from data samples with morphological operations (dilation, closing, hole filling)
- **Connectivity Graphs**: Built-in NetworkX graphs with mandatory node/edge metadata for spatial queries
- **1D Linearization**: Transform complex 2D environments into 1D linearized coordinates for track-based analysis
- **Region Support**: Define and manage named regions of interest (ROIs) with immutable semantics
- **Environment Composition**: Merge multiple environments with automatic bridge inference

### What neurospatial does that others don't

- generalizes analyses to 2D/3D and arbitrary shapes
- Geodesic distance computations (distances are not just Euclidean, they respect environment topology)
- Spatial kernels that respect connectivity graphs (smoothing is not just Gaussian, it respects environment topology)
- Interactive and static visualization of environments and spatial fields
- Comprehensive simulation subpackage for generating synthetic trajectories, neural activity, and spikes with ground truth
- unified analyses (the field reimplements many common neuroscience spatial analyses), this is designed to be a one-stop shop for spatial environment discretization and analysis so that the field is using consistent methods
- python-native with no matlab dependencies
- gpu acceleration

### Spatial Analysis Operations

- **Trajectory Analysis**: Convert trajectories to bin sequences, compute empirical transition matrices with adjacency filtering
- **Occupancy Mapping**: Time-in-bin computation with speed filtering, gap handling, and optional kernel smoothing (including linear time allocation for accurate boundary handling)
- **Field Smoothing**: Diffusion kernel smoothing on graphs with volume correction for continuous fields
- **Interpolation**: Evaluate bin-valued fields at arbitrary points (nearest neighbor or bilinear/trilinear for grids)
- **Distance Fields**: Compute geodesic and Euclidean distances, k-hop neighborhoods, connected components
- **Field Utilities**: Normalize, clamp, combine fields; compute KL/JS divergence and cosine distance
- **Environment Operations**: Subset/crop environments by regions or polygons, rebin grids, copy with cache management

### Field Animation

- **Fields on the recording clock**: Play a decoded posterior or other
  time-varying field with the animal's position, pose, head direction, events,
  LFP, spike rasters, epochs and calibrated video, each on its own clock
- **Viewers**: napari desktop viewer, Jupyter notebook widget, and a
  fastplotlib GPU viewer that also works from a remote server
- **Exports**: MP4, WebM, MOV, AVI, GIF and self-contained HTML players, with
  parallel rendering for long clips
- **Side-by-side fields**: Compare a posterior with its likelihood, or early
  with late laps, on one clock
- **Long recordings**: Memory-mapped fields are read only when shown; review
  identified replay events and export selected windows

## Installation

### From PyPI

```bash
pip install neurospatial
```

Or with uv:

```bash
uv pip install neurospatial
```

### For Development

```bash
# Clone the repository
git clone https://github.com/edeno/neurospatial.git
cd neurospatial

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

**Note**: This project uses `uv` for package management. If you have `uv` installed, all commands should be prefixed with `uv run` (e.g., `uv run pytest`).

### Tested Dependency Versions

neurospatial v0.8.0 has been tested with the following dependency versions:

| Package | Tested Version |
|---------|---------------|
| Python | 3.13.13 |
| numpy | 2.3.4 |
| pandas | 2.3.3 |
| matplotlib | 3.10.7 |
| networkx | 3.5 |
| scipy | 1.16.3 |
| scikit-learn | 1.7.2 |
| scikit-image | 0.25.2 |
| shapely | 2.1.2 |
| alphashape | 1.3.1 |
| track-linearization | 2.4.0 |
| tqdm | 4.67.1 |

These versions match the current locked development environment. The supported
dependency ranges are defined in `pyproject.toml`, and CI exercises all supported
Python versions plus dedicated optional-dependency jobs.

### Optional Dependencies

For animation features, install optional dependencies:

```bash
pip install "neurospatial[animation]"    # napari viewer and notebook widget
pip install "neurospatial[video]"        # MP4, WebM, MOV, AVI and GIF export
pip install "neurospatial[fastplotlib]"  # GPU viewer (Python 3.11 or newer)
```

HTML players need no extra packages. Video export uses the ffmpeg binary that
the `video` extra installs (through `imageio-ffmpeg`); no system ffmpeg is
needed.

## Quickstart

Here's a minimal example showing how to create an environment from spatial data:

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

# Generate some 2D position data (e.g., from animal tracking)
# Shape: (n_samples, 2) for x, y coordinates in centimeters
position_data = np.array([
    [0.0, 0.0],
    [5.0, 5.0],
    [10.0, 10.0],
    [15.0, 5.0],
    [20.0, 0.0],
    # ... more positions
])

# Create an environment with 2 cm bins
env = Environment.from_samples(
    positions=position_data,
    bin_size=2.0,  # 2 cm bins
    name="OpenField"
)

# Query the environment
print(f"Environment has {env.n_bins} bins")
print(f"Dimensions: {env.n_dims}D")
print(f"Extent: {env.dimension_ranges}")

# Map a point to its bin
point = np.array([[10.5, 10.2]])
bin_idx = env.bin_at(point)
print(f"Point {point[0]} is in bin {bin_idx[0]}")

# Find neighbors of a bin
neighbors = env.neighbors(bins=bin_idx[0])
print(f"Bin {bin_idx[0]} has {len(neighbors)} neighbors")

# Visualize the environment
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
env.plot(ax=ax)
plt.show()
```

### Your First Place Field

A neuroscientist's first task is usually: "I have an animal moving
around and spikes from a neuron — show me where the cell fires." Here
is the end-to-end pipeline using simulated data so you can run it
right now without any setup:

<!-- docs-test: run -->
```python
import numpy as np
import matplotlib.pyplot as plt

from neurospatial import Environment
from neurospatial import compute_spatial_rate
from neurospatial.simulation import generate_population_spikes
from neurospatial.simulation.models import PlaceCellModel
from neurospatial.simulation.trajectory import simulate_trajectory_ou

# 1. Build a square open-field environment (60 × 60 cm, 2 cm bins).
xx, yy = np.meshgrid(np.linspace(0, 60, 31), np.linspace(0, 60, 31))
arena_corners = np.column_stack([xx.ravel(), yy.ravel()])
env = Environment.from_samples(arena_corners, bin_size=2.0)
env = env.with_metadata(units="cm")

# 2. Simulate ten minutes of foraging.
positions_t, times = simulate_trajectory_ou(
    env, duration=600.0, speed_units="cm", seed=42,
)

# 3. Build a place cell tuned to the middle of the arena.
cell = PlaceCellModel(env, center=np.array([30.0, 30.0]), width=8.0,
                      max_rate=15.0, seed=42)

# 4. Generate the spike train from the cell's firing model + trajectory.
spike_times = generate_population_spikes(
    [cell], times, positions_t, seed=42, show_progress=False,
)[0]

# 5. Recover the place field from spikes + trajectory.
result = compute_spatial_rate(
    env, spike_times, times, positions_t,
    bandwidth=5.0,
)

# Fast screen; field detection and cell classification are separate questions.
print("Place candidate:", result.is_place_cell(criterion="threshold"))

# 6. Plot.
fig, ax = plt.subplots()
result.plot(ax=ax)
ax.set_title("Place field recovered from simulated spikes")
plt.show()
```

The recovered field will be a Gaussian-like blob centered near `(30,
30)` — the same location we put the simulated cell. From here you can
swap in real spike times, change the trajectory, add more cells, or
detect place fields with [`detect_place_fields`](https://edeno.github.io/neurospatial/api/).
A field can also flag untuned low-count units. For a verdict that also
tests information against a circular-shift null, call
`is_place_cell(env, spike_times, times, positions_t, criterion="shuffle", rng=0)`
with the raw arrays (about 1000 map recomputes by default).
See [the spatial rate-map tutorial](https://github.com/edeno/neurospatial/blob/main/examples/02_spatial_rate_maps.ipynb)
for the full tutorial.

To start with a real recording, run
[the real-data tutorial](examples/04_real_data_analysis.ipynb): it downloads and verifies
the 75 MB [J16 Zenodo dataset](https://doi.org/10.5281/zenodo.23020757), then
analyzes its 203 hippocampal spike trains.
To download ahead of time, run `uv run python data/load_bandit_data.py` from the
repository root. See [the data guide](data/README.md) for units and attribution.

To go the other way — reconstruct position from population spikes — the one-call
path is
[`decode_session`](https://edeno.github.io/neurospatial/api/):
`from neurospatial.decoding import decode_session` runs the whole encode → bin →
decode pipeline and returns a `DecodingResult`. With precomputed count matrices
and population rate maps, use
`decode_position(env, spike_counts, encoding_models, dt=dt)`. Pauses in tracking
are never decoded: time bins are formed only within recorded stretches, and
`epochs=`/`spike_window=` restrict them further. See
[the position-decoding tutorial](https://github.com/edeno/neurospatial/blob/main/examples/06_position_decoding.ipynb)
for the full Bayesian decoding tutorial.

### Recording gaps and time windows

Gaps longer than `max_gap=0.5` seconds are detected from `times` and excluded automatically.
`epochs=` selects analysis windows; `spike_window=` covers ephys that started late or stopped early.
Without `spike_window`, spikes are assumed recorded whenever position was; `result.spike_window_assumed` says so.

<!-- docs-test: run -->
```python
run_epochs = [(float(times[0]), float(times[-1]))]
t_ephys_start, t_ephys_stop = run_epochs[0]
result = compute_spatial_rate(env, spike_times, times, positions_t,
    epochs=run_epochs, spike_window=(t_ephys_start, t_ephys_stop))
```

## Core Concepts

### Bins and Active Bins

neurospatial discretizes continuous space into **bins** (also called nodes). Each bin represents a region of space with a center coordinate and (optionally) a size.

**Active bins** are bins that contain actual data or are considered part of the environment. When creating an environment from data samples, neurospatial can automatically infer which bins should be active based on:

- Data occupancy (bins with enough samples)
- Morphological operations (filling gaps, connecting nearby regions)
- Explicit masks or polygons

This is essential for neuroscience applications where you want to focus on visited areas while excluding walls, obstacles, or unvisited regions.

### Connectivity Graphs

Each environment includes a **connectivity graph** (NetworkX `Graph`) defining which bins are neighbors. This graph powers spatial queries like:

- Finding shortest paths between locations
- Computing geodesic (manifold) distances
- Determining local neighborhoods

The connectivity graph includes mandatory metadata on all nodes and edges (positions, distances, vectors, indices) for robust spatial operations.

### Layout Engines

Layout engines define **how** space is discretized. Available engines include:

- **RegularGridLayout**: Standard rectangular/cuboid grids
- **HexagonalLayout**: Hexagonal tessellations (more uniform neighbor distances)
- **GraphLayout**: 1D linearized tracks for maze/track experiments
- **MaskedGridLayout**: Grids with arbitrary active/inactive regions
- **ImageMaskLayout**: Binary image-based layouts
- **ShapelyPolygonLayout**: Polygon-bounded grids
- **TriangularMeshLayout**: Triangular tessellations

You typically don't interact with layout engines directly; instead, use the `Environment` factory methods which select the appropriate engine for you.

## Common Use Cases

### 1. Analyzing Animal Position Data

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

# In a real analysis, replace these three lines with your own loader:
#   times = load_timestamps()   # shape (n_timepoints,) in seconds
#   position = load_tracking()  # shape (n_timepoints, 2)
#   speeds = load_speeds()      # shape (n_timepoints,)
# Here we synthesize a 1 Hz, 60 s random walk in a 100x100 cm arena so the
# block is runnable end-to-end.
rng = np.random.default_rng(0)
times = np.linspace(0.0, 60.0, 60)
position = np.cumsum(rng.normal(0, 3.0, size=(60, 2)), axis=0) + 50.0
speeds = np.linalg.norm(np.diff(position, axis=0, prepend=position[:1]), axis=1) / np.gradient(times)

# Create environment with 5 cm bins, auto-detect active areas
env = Environment.from_samples(
    positions=position,
    bin_size=5.0,  # cm
    infer_active_bins=True,
    dilate=True,  # Expand active region
    fill_holes=True,  # Fill small gaps
    name="Experiment1_OpenField"
)

# Treat this intentionally coarse 1 Hz example as continuous with max_gap=None.
# Compute occupancy with speed filtering
occupancy = env.occupancy(
    times=times,
    positions=position,
    speed=speeds,
    min_speed=2.5,  # cm/s - filter slow periods
    bandwidth=10.0,  # cm - smooth the occupancy map
    max_gap=None,
)

# Analyze movement patterns
transitions = env.transitions(times=times, positions=position, normalize=True, max_gap=None)
bin_sequence = env.bin_sequence(times=times, positions=position, dedup=True, max_gap=None)
```

### 2. Creating Masked Environments

```python
from shapely.geometry import Polygon

# Define a circular arena (80 cm diameter)
theta = np.linspace(0, 2*np.pi, 100)
boundary = np.column_stack([40 * np.cos(theta), 40 * np.sin(theta)])
polygon = Polygon(boundary)

# Create environment bounded by polygon
env = Environment.from_polygon(
    polygon=polygon,
    bin_size=2.5,  # cm
    name="CircularArena"
)
```

### 3. Linearizing Track Mazes

```python
import networkx as nx

# Define track graph (e.g., plus maze)
graph = nx.Graph()
graph.add_node(0, pos=(0.0, 0.0))    # center
graph.add_node(1, pos=(0.0, 50.0))   # north arm
graph.add_node(2, pos=(50.0, 0.0))   # east arm
graph.add_node(3, pos=(0.0, -50.0))  # south arm
graph.add_node(4, pos=(-50.0, 0.0))  # west arm

graph.add_edge(0, 1, edge_id=0, distance=50.0)
graph.add_edge(0, 2, edge_id=1, distance=50.0)
graph.add_edge(0, 3, edge_id=2, distance=50.0)
graph.add_edge(0, 4, edge_id=3, distance=50.0)

# Create 1D linearized environment
env = Environment.from_graph(
    graph=graph,
    edge_order=[(4, 0), (0, 1), (0, 2), (0, 3)],  # traversal order
    edge_spacing=0.0,  # no gaps between edges
    bin_size=2.0,  # cm
    track_width=10.0,  # measured corridor width, cm; positions farther out get bin -1
    name="PlusMaze"
)

# Convert 2D positions to 1D linearized coordinates
position_2d = np.array([[25.0, 0.0]])  # halfway down east arm
position_1d = env.to_linear(position_2d)

# And back
position_2d_reconstructed = env.linear_to_nd(position_1d)
```

### 4. Defining Regions of Interest

```python
from neurospatial import Regions

from shapely.geometry import Point

# Create environment
env = Environment.from_samples(position_data, bin_size=3.0)

# Define reward zones as circular regions (buffered points)
reward1_polygon = Point(10.0, 10.0).buffer(5.0)  # 5 cm radius circle
reward2_polygon = Point(30.0, 30.0).buffer(5.0)

region_builder = Regions(env.regions.values())
region_builder.add("RewardZone1", polygon=reward1_polygon)
region_builder.add("RewardZone2", polygon=reward2_polygon)

# Or add a point region
region_builder.add("StartLocation", point=(0.0, 0.0))
env = env.with_regions(region_builder)

# Access region information
print(f"Number of regions: {len(env.regions)}")
print(f"Region names: {list(env.regions)}")

# Get region statistics
area = env.regions["RewardZone1"].area
center = env.regions["RewardZone1"].center
print(f"RewardZone1 area: {area:.2f}, center: {center}")
```

## Simulation

neurospatial includes a comprehensive simulation subpackage for generating synthetic spatial data, neural activity, and spike trains. This is essential for testing analysis pipelines, validating algorithms against ground truth, and creating educational examples.

### Quick Example

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment
from neurospatial.simulation import (
    simulate_trajectory_ou,
    PlaceCellModel,
    generate_poisson_spikes,
)

# Sample positions used to infer the active region (e.g., a 100x100 cm arena).
# In real use, replace this with your own tracking data.
arena_data = np.random.default_rng(0).uniform(0, 100, size=(2000, 2))

# Create environment
env = Environment.from_samples(arena_data, bin_size=5.0)
env = env.with_metadata(units="cm")  # Required for trajectory simulation

# Generate realistic trajectory using Ornstein-Uhlenbeck process.
# speed_units must match env.units exactly (no auto-conversion).
positions, times = simulate_trajectory_ou(
    env,
    duration=600.0,  # seconds: ten minutes covers a 1 m arena
    speed_units="cm",
    speed_mean=8.0,  # cm/s
    speed_std=4.0,  # cm/s: the animal speeds up and slows down
    coherence_time=0.7,  # seconds over which the speed changes
    seed=42
)

# Create place cell with known ground truth
place_cell = PlaceCellModel(
    env,
    center=[50.0, 75.0],  # field center in cm
    width=10.0,  # field width (Gaussian std)
    max_rate=25.0  # peak firing rate in Hz
)

# Generate spikes
firing_rates = place_cell.firing_rate(times, positions)
# Seed spikes apart from the trajectory so spike draws do not track position.
spike_times = generate_poisson_spikes(firing_rates, times, seed=1)

# Validate with neurospatial analysis
from neurospatial import compute_spatial_rate
result = compute_spatial_rate(env, spike_times, times, positions)
detected_field = result.firing_rate

# Compare detected field to ground truth
true_center = place_cell.ground_truth['center']
print(f"True field center: {true_center}")
print(f"Detected peak: {result.peak_location()}")
```

### Available Features

- **Trajectory Simulation**
  - Ornstein-Uhlenbeck process for realistic exploration
  - Structured trajectories (laps, alternation tasks)
  - Boundary handling (reflect, periodic, stop)

- **Neural Models**
  - Place cells (Gaussian fields, direction-selective, speed-gated)
  - Boundary/border cells (distance-tuned)
  - Grid cells (hexagonal patterns)
  - All models expose `.ground_truth` for validation

- **Spike Generation**
  - Inhomogeneous Poisson process
  - Refractory period constraints
  - Population spike generation with progress tracking

- **High-Level API**
  - Pre-configured sessions: `open_field_session()`, `linear_track_session()`, etc.
  - Automated validation: `validate_simulation()` compares detected vs true parameters
  - One-call workflow: `simulate_session()` handles trajectory + models + spikes

### Learn More

Follow the **[place-field recovery tutorial](examples/05_simulation_and_validation.ipynb)**
to generate labeled spike trains, recover known fields, and interpret per-unit
recovery outcomes and missing metrics. The
[API reference](https://edeno.github.io/neurospatial/api/) covers additional
simulation models and customization.

## Animation

Watch how a spatial field changes over time on the recording clock: a decoded
posterior beside the animal's position, a field across laps, or a replay
event at a tenth of its speed.

### Quick Example

<!-- docs-test: skip Opens a napari window and writes video files; the animation guide's examples run in tests/docs. -->
```python
from neurospatial.animation import PositionOverlay
from neurospatial.decoding import decode_session

# Assumes you already have positions (n_samples, 2) in cm, times (n_samples,)
# in seconds, and spike_times (one array per unit), as in the place-field
# example above.
env = Environment.from_samples(positions, bin_size=2.5, units="cm")
result = decode_session(env, spike_times, times, positions, dt=0.025)

# One row per 25 ms decoding bin; the animal's position on its own clock
animal = PositionOverlay(positions, times=times, color="magenta")

# Explore interactively (returns a view; view.native is the napari viewer)
view = env.animate_fields(
    result.posterior, frame_times=result.times, overlays=[animal],
    backend="napari",
)

# Share a clip at half speed (returns the file path)
env.animate_fields(
    result.posterior, frame_times=result.times, overlays=[animal],
    save_path="decoding.mp4", speed=0.5,
)
```

A field row is held until the next one, up to `max_gap=0.5` s; across longer
gaps in the recording it is shown as missing rather than frozen. The
`backend` can be `"napari"`, `"widget"` (Jupyter), `"fastplotlib"` (GPU), or
a file: `.mp4`, `.webm`, `.mov`, `.avi`, `.gif` or `.html`. Without
`backend=`, a video or `.html` `save_path` writes that file; otherwise
Jupyter opens the notebook widget and other sessions open napari.

### Large-Scale Datasets

Keep a long posterior on disk as a memory-mapped array; rows are read only
when a frame needs them:

<!-- docs-test: skip Requires a saved posterior file and a display. -->
```python
import numpy as np

posterior = np.load("posterior.npy", mmap_mode="r")  # (n_bins_time, env.n_bins)
env.animate_fields(posterior, frame_times=decode_times, backend="napari")
```

### Learn More

- **[Animation User Guide](https://edeno.github.io/neurospatial/user-guide/animation/)**: Complete documentation with troubleshooting
- **[Visualization tutorial](examples/08_visualization.ipynb)**: Static previews, HTML overlays, calibrated video, and opt-in viewers and exports
- Common use cases: place field dynamics, theta sequences, remapping, population activity

## Documentation

- **[Documentation Home](https://edeno.github.io/neurospatial/)**: Complete documentation site
- **[Getting Started](https://edeno.github.io/neurospatial/getting-started/installation/)**: Installation and quickstart guide
- **[User Guide](https://edeno.github.io/neurospatial/user-guide/)**: Detailed feature documentation
- **[API Reference](https://edeno.github.io/neurospatial/api/)**: Auto-generated API documentation
- **[Examples](https://edeno.github.io/neurospatial/examples/)**: Jupyter notebooks with real-world use cases
- **[Glossary](https://edeno.github.io/neurospatial/glossary/)**: Spatial-coding vocabulary reference
- **[Contributing](https://edeno.github.io/neurospatial/contributing/)**: Guidelines for contributors
- **[CLAUDE.md](CLAUDE.md)**: Development guide for Claude Code users
- **[GitHub Issues](https://github.com/edeno/neurospatial/issues)**: Bug reports and feature requests

## Package Map

```text
src/neurospatial/
├── environment/    # Environments, trajectories, fields, queries, and plots
├── layout/         # Grid, hexagonal, graph, polygon, mask, and mesh layouts
├── encoding/       # Place, grid, border, directional, view, and egocentric coding
├── decoding/       # Bayesian decoding, posterior summaries, and decoding accuracy
├── replay/         # Replay trajectory scores, replay shuffles, and significance tests
├── reactivation/   # Cell assemblies and pairwise reactivation
├── behavior/       # Segmentation, navigation, decisions, rewards, and VTE
├── events/         # Peri-event alignment, detection, intervals, and regressors
├── stats/          # Circular statistics, temporal shuffles, p-values, and surrogates
├── simulation/     # Mazes, trajectories, neural models, spikes, and validation
├── io/             # Environment files plus NWB and pynapple interoperability
├── animation/      # Field animation: viewers, overlays, and video/HTML export
├── annotation/     # Interactive video and track-graph annotation
├── ops/            # Graph-aware distance, smoothing, transforms, and visibility
├── regions/        # Standalone region containers, operations, I/O, and plotting
└── composite.py    # Multi-environment composition

tests/                 # More than 9,000 automated tests across the package
examples/              # 9 task-oriented, paired Python/Jupyter tutorials
docs/                  # User guides, API reference, examples, and contributor docs
```

## Requirements

- Python 3.10+
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- networkx >= 3.0
- scipy >= 1.10.0
- scikit-learn >= 1.2.0
- shapely >= 2.0.0
- track-linearization >= 2.4.0

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`uv run pytest`)
5. Run code quality checks (`uv run ruff check . && uv run ruff format .`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.

## Citation

If you use neurospatial in your research, please cite:

```bibtex
@software{neurospatial2026,
  author = {Denovellis, Eric},
  title = {neurospatial: Spatial environment discretization for neuroscience},
  year = {2026},
  url = {https://github.com/edeno/neurospatial},
  version = {0.8.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [NetworkX](https://networkx.org/) for graph operations
- Uses [Shapely](https://shapely.readthedocs.io/) for geometric operations
- Leverages [track-linearization](https://github.com/Eden-Kramer-Lab/track_linearization) for 1D linearization

## Contact

### Eric Denovellis

- Email: <eric.denovellis@ucsf.edu>
- GitHub: [@edeno](https://github.com/edeno)
- Issues: [GitHub Issues](https://github.com/edeno/neurospatial/issues)

---

**Status**: Alpha - API may change. Contributions and feedback welcome!
