# Architecture & Internals

This page covers the internals you need when **extending** neurospatial —
writing a custom layout engine, or reading the raw graph metadata directly.
Most users never need this; the mental model in
[Core Concepts](../getting-started/core-concepts.md) is enough for day-to-day
analysis.

## Mandatory Graph Metadata

Every environment carries a **connectivity graph** (a NetworkX `Graph`) whose
nodes are bins and whose edges connect neighboring bins. neurospatial enforces
a set of **mandatory attributes** on that graph so that spatial queries
(shortest paths, geodesic distances, neighbor finding) are always well-defined.

**Node attributes:**

- `'pos'`: Tuple of N-D coordinates (the bin center)
- `'source_grid_flat_index'`: Flat index in the original grid
- `'original_grid_nd_index'`: N-D grid index tuple

**Edge attributes:**

- `'distance'`: Euclidean distance between bin centers
- `'vector'`: Displacement vector (as a tuple)
- `'edge_id'`: Unique integer edge identifier
- `'angle_2d'`: Angle in 2D (optional, present for 2D layouts)

You can read these attributes directly off `env.connectivity`:

```python
import numpy as np
from neurospatial import Environment

positions = np.random.default_rng(0).uniform(0, 50, size=(500, 2))
env = Environment.from_samples(positions, bin_size=5.0, units="cm")

G = env.connectivity
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Node attributes (mandatory)
node_0_attrs = G.nodes[0]
print(f"Node 0 position: {node_0_attrs['pos']}")
print(f"Node 0 grid index: {node_0_attrs['original_grid_nd_index']}")

# Edge attributes (mandatory) — pick any existing edge
u, v = next(iter(G.edges))
edge_attrs = G.edges[u, v]
print(f"Edge ({u},{v}) distance: {edge_attrs['distance']}")
print(f"Edge ({u},{v}) vector: {edge_attrs['vector']}")
```

These attributes are what enable geodesic distances that respect boundaries,
shortest-path queries, and the boundary-aware diffusion smoothing used by the
encoding functions.

## Protocol-Based Design

**Layout engines** define *how* continuous space is discretized into bins. They
are wired into `Environment` through a **protocol** (structural typing), not
inheritance — any object providing the required members works, so you can add a
custom discretization strategy without modifying core code.

```python
# Layout engines must provide:
# - bin_centers: NDArray of shape (n_bins, n_dims)
# - connectivity: nx.Graph with the mandatory attributes above
# - dimension_ranges: List of (min, max) tuples
# - is_linearized_track: bool (True for linearized layouts)
# - build(): Method to construct the layout
# - point_to_bin_index(): Map points to bins
# - bin_sizes(): Compute bin sizes
# - plot(): Visualize the layout
```

This design allows:

- Custom layout engines without modifying core code
- Static type checking against the protocol
- Maximum flexibility across grid, hexagonal, masked, polygon, mesh, and
  1D-linearized layouts

See the [layout engines guide](../user-guide/layout-engines.md) for the
built-in engines and their factory methods.
