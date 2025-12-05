# neurospatial

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**neurospatial** is a Python library for discretizing continuous N-dimensional spatial environments into bins/nodes with connectivity graphs. It provides tools for spatial analysis, particularly for neuroscience applications involving place fields, position tracking, and spatial navigation.

Whether you're analyzing animal navigation data, modeling place cells, or working with any spatial discretization problem, neurospatial gives you flexible, powerful tools to represent and analyze spatial environments.

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

- **Multi-Backend Animation**: Visualize spatial fields over time with 4 specialized backends
  - **Napari**: GPU-accelerated interactive viewer with lazy loading (100K+ frames)
  - **Video**: Parallel MP4/WebM export with ffmpeg (unlimited frames)
  - **HTML**: Standalone player with instant scrubbing (up to 500 frames)
  - **Jupyter Widget**: Notebook integration with play/pause controls
- **Auto-Selection**: Intelligent backend selection based on file extension, dataset size, and environment
- **Large-Scale Support**: Memory-mapped arrays, LRU caching, frame subsampling for hour-long sessions
- **Trajectory Overlays**: Overlay animal trajectories on animated fields (Napari backend)

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

neurospatial v0.2.0 has been tested with the following dependency versions:

| Package | Tested Version |
|---------|---------------|
| Python | 3.13.5 |
| numpy | 2.3.4 |
| pandas | 2.3.3 |
| matplotlib | 3.10.7 |
| networkx | 3.5 |
| scipy | 1.16.3 |
| scikit-learn | 1.7.2 |
| shapely | 2.1.2 |
| track-linearization | 2.4.0 |

These versions represent the tested configuration. neurospatial likely works with a range of versions for each dependency, but these specific versions have full test coverage.

### Optional Dependencies

For animation features, install optional dependencies:

```bash
# Napari backend (GPU-accelerated interactive viewer)
pip install "napari[all]>=0.4.18,<0.6"

# Jupyter widget backend (notebook integration)
pip install "ipywidgets>=8.0,<9.0"

# Video backend (requires system ffmpeg installation)
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg

# Conda
conda install -c conda-forge ffmpeg
```

**Note**: HTML backend requires no additional dependencies. Video backend performance scales with CPU cores (use `n_workers` parameter for parallel rendering).

## Quickstart

Here's a minimal example showing how to create an environment from spatial data:

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
    data_samples=position_data,
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
neighbors = env.neighbors(bin_idx[0])
print(f"Bin {bin_idx[0]} has {len(neighbors)} neighbors")

# Visualize the environment
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
env.plot(ax=ax)
plt.show()
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

```python
# Load position tracking data
times = load_timestamps()  # Shape: (n_timepoints,) in seconds
position = load_tracking_data()  # Shape: (n_timepoints, 2)
speeds = load_speeds()  # Shape: (n_timepoints,)

# Create environment with 5 cm bins, auto-detect active areas
env = Environment.from_samples(
    data_samples=position,
    bin_size=5.0,  # cm
    infer_active_bins=True,
    dilate=True,  # Expand active region
    fill_holes=True,  # Fill small gaps
    name="Experiment1_OpenField"
)

# Compute occupancy with speed filtering
occupancy = env.occupancy(
    times=times,
    positions=position,
    speed=speeds,
    min_speed=2.5,  # cm/s - filter slow periods
    kernel_bandwidth=10.0  # cm - smooth the occupancy map
)

# Analyze movement patterns
transitions = env.transitions(times=times, positions=position, normalize=True)
bin_sequence = env.bin_sequence(times=times, positions=position, dedup=True)
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
from shapely.geometry import Point

# Create environment
env = Environment.from_samples(position_data, bin_size=3.0)

# Define reward zones as circular regions (buffered points)
reward1_polygon = Point(10.0, 10.0).buffer(5.0)  # 5 cm radius circle
reward2_polygon = Point(30.0, 30.0).buffer(5.0)

env.regions.add("RewardZone1", polygon=reward1_polygon)
env.regions.add("RewardZone2", polygon=reward2_polygon)

# Or add a point region
env.regions.add("StartLocation", point=(0.0, 0.0))

# Access region information
print(f"Number of regions: {len(env.regions)}")
print(f"Region names: {env.regions.list_names()}")

# Get region statistics
area = env.regions.area("RewardZone1")
center = env.regions.region_center("RewardZone1")
print(f"RewardZone1 area: {area:.2f}, center: {center}")
```

## Simulation

neurospatial includes a comprehensive simulation subpackage for generating synthetic spatial data, neural activity, and spike trains. This is essential for testing analysis pipelines, validating algorithms against ground truth, and creating educational examples.

### Quick Example

```python
from neurospatial import Environment
from neurospatial.simulation import (
    simulate_trajectory_ou,
    PlaceCellModel,
    generate_poisson_spikes,
)

# Create environment
env = Environment.from_samples(arena_data, bin_size=2.0)
env.units = "cm"  # Required for trajectory simulation

# Generate realistic trajectory using Ornstein-Uhlenbeck process
positions, times = simulate_trajectory_ou(
    env,
    duration=120.0,  # seconds
    speed_mean=0.08,  # m/s (8 cm/s)
    coherence_time=0.7,  # smoothness parameter
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
firing_rates = place_cell.firing_rate(positions, times)
spike_times = generate_poisson_spikes(firing_rates, times, seed=42)

# Validate with neurospatial analysis
from neurospatial import compute_place_field
detected_field = compute_place_field(env, spike_times, times, positions)

# Compare detected field to ground truth
true_center = place_cell.ground_truth['center']
print(f"True field center: {true_center}")
print(f"Detected peak: {env.bin_centers[detected_field.argmax()]}")
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

See the comprehensive tutorial: **[Simulation Workflows Notebook](examples/15_simulation_workflows.ipynb)** for complete examples including:

- Quick start with pre-configured sessions
- Low-level API for custom workflows
- All cell types (place, boundary, grid)
- Validation and visualization
- Performance tips and customization

## Animation

Visualize how spatial fields evolve over time with multi-backend animation support.

### Quick Example

```python
from neurospatial import Environment
from neurospatial.animation import subsample_frames

# Create environment and compute fields over time
env = Environment.from_samples(positions, bin_size=2.5)
fields = [compute_place_field(env, spikes[i], times, positions) for i in range(30)]

# Interactive Napari viewer (best for exploration)
env.animate_fields(fields, backend="napari")

# Video export with parallel rendering (best for presentations)
env.clear_cache()  # Required for parallel rendering
env.animate_fields(fields, save_path="animation.mp4", fps=30, n_workers=4)

# HTML standalone player (best for sharing)
env.animate_fields(fields, save_path="animation.html")

# Jupyter widget (best for notebooks)
env.animate_fields(fields, backend="widget")
```

### Backend Selection Guide

| Backend | Best For | Max Frames | Output |
|---------|----------|-----------|--------|
| **Napari** | Large datasets (100K+), interactive exploration | Unlimited* | Live viewer |
| **Video** | Presentations, publications | Unlimited | .mp4, .webm |
| **HTML** | Sharing, web embedding | 500 | .html |
| **Widget** | Jupyter notebooks | ~1000 | Interactive widget |

\* Limited only by disk space (lazy loading with LRU cache)

### Large-Scale Datasets

For sessions with 100K+ frames (e.g., 1-hour recording at 250 Hz):

```python
import numpy as np

# Use memory-mapped arrays (doesn't load into RAM)
fields = np.memmap('fields.dat', dtype='float32', mode='w+',
                   shape=(900_000, env.n_bins))

# Napari lazy-loads from disk (no data loading)
env.animate_fields(fields, backend="napari")

# Or subsample for video export (250 Hz → 30 fps)
subsampled = subsample_frames(fields, source_fps=250, target_fps=30)
env.clear_cache()
env.animate_fields(subsampled, save_path="replay.mp4", n_workers=4)
```

### Learn More

- **[Animation User Guide](https://edeno.github.io/neurospatial/user-guide/animation/)**: Complete documentation with troubleshooting
- **[Animation Examples Notebook](examples/16_field_animation.ipynb)**: Working examples for all backends
- Common use cases: place field dynamics, theta sequences, remapping, population activity

## Documentation

- **[Documentation Home](https://edeno.github.io/neurospatial/)**: Complete documentation site
- **[Getting Started](https://edeno.github.io/neurospatial/getting-started/installation/)**: Installation and quickstart guide
- **[User Guide](https://edeno.github.io/neurospatial/user-guide/)**: Detailed feature documentation
- **[API Reference](https://edeno.github.io/neurospatial/api/)**: Auto-generated API documentation
- **[Examples](https://edeno.github.io/neurospatial/examples/)**: Jupyter notebooks with real-world use cases
- **[Contributing](https://edeno.github.io/neurospatial/contributing/)**: Guidelines for contributors
- **[CLAUDE.md](CLAUDE.md)**: Development guide for Claude Code users
- **[GitHub Issues](https://github.com/edeno/neurospatial/issues)**: Bug reports and feature requests

## Project Structure

```text
neurospatial/
├── src/neurospatial/
│   ├── environment/            # Main Environment class (modular package)
│   │   ├── core.py            # Core dataclass with state and properties
│   │   ├── factories.py       # Factory classmethods (from_samples, from_graph, etc.)
│   │   ├── queries.py         # Spatial query methods
│   │   ├── trajectory.py      # Trajectory analysis (occupancy, transitions)
│   │   ├── transforms.py      # Rebin/subset operations
│   │   ├── fields.py          # Spatial field operations (smooth, interpolate)
│   │   ├── metrics.py         # Environment metrics and properties
│   │   ├── serialization.py   # Save/load methods
│   │   ├── regions.py         # Region operations
│   │   ├── visualization.py   # Plotting methods (includes animate_fields)
│   │   └── decorators.py      # check_fitted decorator
│   ├── animation/              # Field animation
│   │   ├── core.py            # Main dispatcher and subsample_frames utility
│   │   ├── rendering.py       # Rendering utilities (colormap, RGB conversion)
│   │   ├── _parallel.py       # Parallel frame rendering for video backend
│   │   └── backends/          # Backend implementations
│   │       ├── napari_backend.py   # GPU-accelerated interactive viewer
│   │       ├── video_backend.py    # Parallel MP4/WebM export with ffmpeg
│   │       ├── html_backend.py     # Standalone HTML player
│   │       └── widget_backend.py   # Jupyter widget integration
│   ├── composite.py            # CompositeEnvironment for multi-env merging
│   ├── alignment.py            # Probability distribution transforms
│   ├── transforms.py           # 2D affine transformations
│   ├── layout/
│   │   ├── base.py            # LayoutEngine protocol
│   │   ├── factories.py       # Layout factory functions
│   │   └── engines/           # Concrete layout implementations
│   └── regions/
│       ├── core.py            # Region and Regions classes
│       └── serialization.py   # JSON I/O for regions
└── tests/                      # Comprehensive test suite (1,185+ tests)
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
@software{neurospatial2025,
  author = {Denovellis, Eric},
  title = {neurospatial: Spatial environment discretization for neuroscience},
  year = {2025},
  url = {https://github.com/edeno/neurospatial},
  version = {0.3.0}
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
