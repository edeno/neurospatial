# neurospatial

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://edeno.github.io/neurospatial/)

**neurospatial** is a Python library for discretizing continuous N-dimensional spatial environments into bins/nodes with connectivity graphs. It provides tools for spatial analysis, particularly for neuroscience applications involving place fields, position tracking, and spatial navigation.

Whether you're analyzing animal navigation data, modeling place cells, or working with any spatial discretization problem, neurospatial gives you flexible, powerful tools to represent and analyze spatial environments.

## Key Features

- **Multiple Layout Engines**: Choose from regular grids, hexagonal tessellations, masked regions, polygon-bounded areas, triangular meshes, and 1D linearized tracks
- **Automatic Bin Detection**: Infer active bins from data samples with morphological operations (dilation, closing, hole filling)
- **Connectivity Graphs**: Built-in NetworkX graphs with mandatory node/edge metadata for spatial queries
- **1D Linearization**: Transform complex 2D environments into 1D linearized coordinates for track-based analysis
- **Spatial Queries**: Point-to-bin mapping, neighbor finding, shortest paths, geodesic distances
- **Region Support**: Define and manage named regions of interest (ROIs) with immutable semantics
- **Environment Composition**: Merge multiple environments with automatic bridge inference
- **Alignment Tools**: Transform and map probability distributions between environments
- **Type-Safe Protocol Design**: Layout engines implement protocols, not inheritance, for maximum flexibility

## Quick Example

```python
import numpy as np
from neurospatial import Environment

# Generate some 2D position data (e.g., from animal tracking)
position_data = np.array([
    [0.0, 0.0],
    [5.0, 5.0],
    [10.0, 10.0],
    [15.0, 5.0],
    [20.0, 0.0],
])

# Create an environment with 2 cm bins
env = Environment.from_samples(
    data_samples=position_data,
    bin_size=2.0,
    name="OpenField"
)

# Query the environment
print(f"Environment has {env.n_bins} bins")
print(f"Dimensions: {env.n_dims}D")

# Map a point to its bin
point = np.array([[10.5, 10.2]])
bin_idx = env.bin_at(point)
print(f"Point {point[0]} is in bin {bin_idx[0]}")

# Find neighbors
neighbors = env.neighbors(bin_idx[0])
print(f"Bin {bin_idx[0]} has {len(neighbors)} neighbors")
```

## Installation

Install neurospatial using pip:

```bash
pip install neurospatial
```

For development installation with documentation tools:

```bash
git clone https://github.com/edeno/neurospatial.git
cd neurospatial
uv sync --extra docs
```

See the [Installation Guide](getting-started/installation.md) for more details.

## Documentation

- **[Getting Started](getting-started/index.md)**: Installation, quickstart, and core concepts
- **[User Guide](user-guide/index.md)**: Detailed guides on using neurospatial features
- **[API Reference](api/index.md)**: Complete API documentation
- **[Examples](examples/index.md)**: Jupyter notebooks with real-world use cases
- **[Contributing](contributing.md)**: Guidelines for contributors

## Use Cases

neurospatial is designed for researchers working with spatial data in neuroscience:

- **Place Cell Analysis**: Discretize environments for computing firing rate maps
- **Position Tracking**: Convert continuous trajectories into spatial bins
- **Maze Experiments**: Linearize complex track structures for 1D analysis
- **Spatial Navigation**: Compute geodesic distances and shortest paths
- **Multi-Environment Studies**: Align and compare spatial representations across sessions

## Project Status

**Status**: Alpha (v0.1.0) - API may change. Contributions and feedback welcome!

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

This project is licensed under the MIT License - see the [LICENSE](https://github.com/edeno/neurospatial/blob/main/LICENSE) file for details.

## Contact

**Eric Denovellis**

- Email: <eric.denovellis@ucsf.edu>
- GitHub: [@edeno](https://github.com/edeno)
- Issues: [GitHub Issues](https://github.com/edeno/neurospatial/issues)
