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
- **Type-Safe Protocol Design**: Layout engines implement protocols, not inheritance, for maximum flexibility

### Spatial Analysis Operations (v0.2.0+)

- **Trajectory Analysis**: Convert trajectories to bin sequences, compute empirical transition matrices with adjacency filtering
- **Occupancy Mapping**: Time-in-bin computation with speed filtering, gap handling, and optional kernel smoothing (including linear time allocation for accurate boundary handling)
- **Field Smoothing**: Diffusion kernel smoothing on graphs with volume correction for continuous fields
- **Interpolation**: Evaluate bin-valued fields at arbitrary points (nearest neighbor or bilinear/trilinear for grids)
- **Distance Fields**: Compute geodesic and Euclidean distances, k-hop neighborhoods, connected components
- **Field Utilities**: Normalize, clamp, combine fields; compute KL/JS divergence and cosine distance
- **Environment Operations**: Subset/crop environments by regions or polygons, rebin grids, copy with cache management

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

# Compute occupancy with speed filtering (v0.2.0+)
occupancy = env.occupancy(
    times=times,
    positions=position,
    speed=speeds,
    min_speed=2.5,  # cm/s - filter slow periods
    kernel_bandwidth=10.0  # cm - smooth the occupancy map
)

# Analyze movement patterns (v0.2.0+)
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
│   │   ├── visualization.py   # Plotting methods
│   │   └── decorators.py      # check_fitted decorator
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
└── tests/                      # Comprehensive test suite (1,076 tests)
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
  version = {0.1.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [NetworkX](https://networkx.org/) for graph operations
- Uses [Shapely](https://shapely.readthedocs.io/) for geometric operations
- Leverages [track-linearization](https://github.com/Eden-Kramer-Lab/track_linearization) for 1D linearization
- Inspired by spatial analysis needs in systems neuroscience research

## Contact

### Eric Denovellis

- Email: <eric.denovellis@ucsf.edu>
- GitHub: [@edeno](https://github.com/edeno)
- Issues: [GitHub Issues](https://github.com/edeno/neurospatial/issues)

---

**Status**: Alpha - API may change. Contributions and feedback welcome!
