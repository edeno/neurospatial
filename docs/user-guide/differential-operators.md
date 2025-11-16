# Differential Operators on Spatial Graphs

This guide covers the differential operator infrastructure for graph signal processing on spatial environments, enabling gradient-based analyses for reinforcement learning, replay analysis, and flow field computations.

## Overview

Neurospatial provides differential operators that extend classical calculus to graph-structured spatial environments:

- **`compute_differential_operator(env)`** - Builds the fundamental differential operator matrix **D**
- **`gradient(field, env)`** - Computes gradient of scalar fields (node values → edge values)
- **`divergence(edge_field, env)`** - Computes divergence of edge fields (edge values → node values)
- **Laplacian** - Composition of divergence and gradient: `div(grad(f))`

These operators respect the **graph connectivity** of your environment, making them suitable for irregular spatial structures, mazes, and track-based environments where Euclidean calculus doesn't apply.

## Why Differential Operators Matter

Differential operators provide a principled framework for analyzing spatial gradients and flows on graph-structured environments:

### Neuroscience Applications

1. **Successor Representation Analysis** - Compute flow fields from hippocampal replay sequences
2. **Value Gradient Estimation** - Analyze policy gradients in reinforcement learning tasks
3. **Place Field Dynamics** - Study spatial derivatives of firing rate fields
4. **Trajectory Flow Analysis** - Detect sources and sinks in spatial navigation patterns

### Mathematical Foundation

Classical calculus defines:
- **Gradient**: Rate of change of a scalar field (direction of steepest ascent)
- **Divergence**: Net outflow from a point in a vector field
- **Laplacian**: Second derivative operator (smoothness measure)

Graph signal processing extends these concepts to discrete, graph-structured domains (Shuman et al., 2013). Neurospatial implements the **weighted differential operator** framework following PyGSP conventions.

## The Differential Operator Matrix **D**

The fundamental object is the **differential operator matrix** **D**, which has shape `(n_bins, n_edges)`:

$$
D_{i,e} = \begin{cases}
-\sqrt{w_e} & \text{if bin } i \text{ is the source of edge } e \\
+\sqrt{w_e} & \text{if bin } i \text{ is the destination of edge } e \\
0 & \text{otherwise}
\end{cases}
$$

where $w_e$ is the edge weight (distance between bin centers).

### Why Square Root Weighting?

The square root weighting $\sqrt{w_e}$ ensures that the graph Laplacian $L = D \cdot D^T$ matches the standard NetworkX Laplacian matrix. This is the convention in graph signal processing (Shuman et al., 2013; Perraudin et al., 2014).

### Accessing the Differential Operator

The differential operator is cached on the `Environment` object for efficiency:

```python
from neurospatial import Environment
import numpy as np

# Create environment
env = Environment.from_samples(trajectory_data, bin_size=2.5)

# Access differential operator (automatically cached)
D = env.differential_operator  # Shape: (n_bins, n_edges)

print(f"Shape: {D.shape}")
print(f"Sparse format: {D.format}")  # CSC (compressed sparse column)

# Repeated access returns the same cached object (fast)
D2 = env.differential_operator
assert D is D2  # Same object in memory
```

**Performance**: The differential operator is computed once and cached. Subsequent accesses are ~50x faster (no recomputation).

## Gradient Operator

The **gradient** transforms a **scalar field** (values at nodes) into an **edge field** (values on edges):

$$
\nabla f = D^T f
$$

where $f$ is a field with shape `(n_bins,)` and the result has shape `(n_edges,)`.

### Physical Interpretation

For each edge connecting bins $i$ and $j$:
- **Positive gradient**: field increases from $i$ to $j$ (uphill direction)
- **Negative gradient**: field decreases from $i$ to $j$ (downhill direction)
- **Zero gradient**: field is constant along the edge

### Example: Distance Field Gradient

Compute the gradient of a distance field from a goal location:

```python
from neurospatial import Environment, distance_field, gradient
import numpy as np

# Create environment
env = Environment.from_samples(trajectory_data, bin_size=2.5)

# Define goal bin (e.g., reward location)
goal_bin = 42

# Compute distance field (geodesic distances from goal)
distances = distance_field(env.connectivity, sources=[goal_bin])

# Compute gradient (edge field pointing toward/away from goal)
grad_distances = gradient(distances, env)

print(f"Input shape (nodes): {distances.shape}")  # (n_bins,)
print(f"Output shape (edges): {grad_distances.shape}")  # (n_edges,)

# Edges with negative gradient point toward goal (downhill in distance)
# Edges with positive gradient point away from goal (uphill in distance)
```

**Use Case**: In reinforcement learning, the negative gradient of a value function gives the optimal policy direction.

### Example: Constant Field Has Zero Gradient

```python
# Constant field
constant_field = np.ones(env.n_bins)

# Gradient should be zero everywhere
grad_constant = gradient(constant_field, env)

print(f"Max gradient magnitude: {np.abs(grad_constant).max()}")  # ~0.0 (numerical precision)
```

## Divergence Operator

The **divergence** transforms an **edge field** (values on edges) into a **scalar field** (values at nodes):

$$
\text{div}(g) = D \cdot g
$$

where $g$ is an edge field with shape `(n_edges,)` and the result has shape `(n_bins,)`.

### Physical Interpretation

For each node (bin):
- **Positive divergence**: Net outflow (source) - more flow leaving than entering
- **Negative divergence**: Net inflow (sink) - more flow entering than leaving
- **Zero divergence**: Conservation - inflow equals outflow

### Example: Flow Field from Successor Representation

In reinforcement learning, the successor representation defines flow fields. The divergence identifies source and sink states:

```python
from neurospatial import divergence
import numpy as np

# Example: edge field representing transition probabilities
# (In practice, this would come from analyzing trajectory transitions)
edge_flow = np.random.randn(env.connectivity.number_of_edges())

# Compute divergence (net outflow from each bin)
div_flow = divergence(edge_flow, env)

print(f"Input shape (edges): {edge_flow.shape}")  # (n_edges,)
print(f"Output shape (nodes): {div_flow.shape}")  # (n_bins,)

# Identify source states (positive divergence)
sources = np.where(div_flow > 0)[0]
print(f"Source bins: {sources}")

# Identify sink states (negative divergence)
sinks = np.where(div_flow < 0)[0]
print(f"Sink bins: {sinks}")
```

**Use Case**: Analyzing hippocampal replay sequences to detect where trajectories typically start (sources) and end (sinks).

### Relationship to Gradient: The Laplacian

The **Laplacian operator** is the composition of divergence and gradient:

$$
L = D \cdot D^T
$$

Applied to a field $f$:

$$
Lf = \text{div}(\text{grad}(f))
$$

This measures the **smoothness** of a field - how much a field value differs from its neighbors.

```python
from neurospatial import gradient, divergence
import networkx as nx
import numpy as np

# Arbitrary scalar field
field = np.random.randn(env.n_bins)

# Compute Laplacian via composition
grad_field = gradient(field, env)
laplacian_field = divergence(grad_field, env)

# Compare with NetworkX Laplacian
L_nx = nx.laplacian_matrix(env.connectivity).toarray()
laplacian_field_nx = L_nx @ field

# Should match (within numerical precision)
print(f"Max difference: {np.abs(laplacian_field - laplacian_field_nx).max()}")  # ~0.0
```

**Use Case**: Laplacian smoothing - iteratively applying $f \leftarrow f - \alpha \cdot L f$ smooths a field while respecting graph structure.

## When to Use Differential Operators

### Use Differential Operators When:

1. **Analyzing RL Value Functions** - Compute policy gradients from learned value fields
2. **Studying Replay Trajectories** - Detect flow patterns in sequential reactivation
3. **Custom Smoothing** - Implement graph-aware diffusion processes
4. **Flow Field Analysis** - Identify sources, sinks, and circulation in spatial trajectories
5. **Computing Spatial Derivatives** - Measure rate of change of firing rate fields

### Use `env.smooth()` When:

- You just want **Gaussian smoothing** of a spatial field (standard place field analysis)
- You don't need explicit gradient or divergence information
- Performance matters (Gaussian smoothing is optimized with kernel caching)

### Use `distance_field()` When:

- You need **geodesic distances** from goal locations or boundaries
- You're computing reward fields or navigation metrics
- You don't need flow or gradient information

## Complete Example: Goal-Directed Flow Analysis

Combine distance fields, gradients, and divergence to analyze goal-directed navigation:

```python
from neurospatial import Environment, distance_field, gradient, divergence
import numpy as np
import matplotlib.pyplot as plt

# Create environment
env = Environment.from_samples(trajectory_data, bin_size=2.5)
env.units = "cm"

# Define goal location
goal_bin = env.bin_at(np.array([[50.0, 50.0]]))[0]

# Step 1: Compute distance field from goal
distances = distance_field(env.connectivity, sources=[goal_bin])

# Step 2: Compute gradient (direction to goal)
grad_distances = gradient(distances, env)

# Negative gradient points toward goal
goal_directed_flow = -grad_distances

# Step 3: Compute divergence (find sources and sinks)
div_flow = divergence(goal_directed_flow, env)

# Step 4: Analyze results
print(f"Goal bin divergence: {div_flow[goal_bin]:.3f}")  # Should be negative (sink)

# Bins far from goal act as sources (positive divergence)
sources = np.where(div_flow > 0.1)[0]
print(f"Number of source bins: {len(sources)}")

# Step 5: Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Distance field
ax = axes[0]
ax.scatter(env.bin_centers[:, 0], env.bin_centers[:, 1],
           c=distances, cmap='viridis', s=100)
ax.set_title('Distance to Goal')
ax.set_xlabel(f'X ({env.units})')
ax.set_ylabel(f'Y ({env.units})')
plt.colorbar(ax.collections[0], ax=ax, label='Distance')

# Gradient magnitude (steepness)
grad_magnitude = np.abs(grad_distances)
# Map edge values to nodes (average over incident edges)
grad_per_node = np.zeros(env.n_bins)
for i, (u, v) in enumerate(env.connectivity.edges()):
    grad_per_node[u] += grad_magnitude[i]
    grad_per_node[v] += grad_magnitude[i]
grad_per_node /= np.array([env.connectivity.degree(i) for i in range(env.n_bins)])

ax = axes[1]
ax.scatter(env.bin_centers[:, 0], env.bin_centers[:, 1],
           c=grad_per_node, cmap='hot', s=100)
ax.set_title('Gradient Magnitude')
ax.set_xlabel(f'X ({env.units})')
plt.colorbar(ax.collections[0], ax=ax, label='|∇ distance|')

# Divergence (sources and sinks)
ax = axes[2]
scatter = ax.scatter(env.bin_centers[:, 0], env.bin_centers[:, 1],
                      c=div_flow, cmap='RdBu_r', s=100, vmin=-0.5, vmax=0.5)
ax.scatter(env.bin_centers[goal_bin, 0], env.bin_centers[goal_bin, 1],
           marker='*', s=500, c='gold', edgecolors='black', linewidths=2,
           label='Goal')
ax.set_title('Flow Divergence')
ax.set_xlabel(f'X ({env.units})')
ax.legend()
plt.colorbar(scatter, ax=ax, label='div(flow)')

plt.tight_layout()
plt.show()
```

**Interpretation**:
- **Left panel**: Distance increases away from goal (blue = close, yellow = far)
- **Middle panel**: Gradient magnitude shows steepness (red = steep, dark = flat)
- **Right panel**: Divergence shows sources (red, flow originates) and sinks (blue, flow terminates)

## Mathematical Background

### Graph Signal Processing Theory

Classical calculus operators are defined on continuous domains. Graph signal processing extends these to discrete graphs (Shuman et al., 2013):

| Classical Calculus | Graph Signal Processing |
|-------------------|-------------------------|
| Scalar field $f(x)$ | Node field $f_i$, $i \in \mathcal{V}$ |
| Vector field $\mathbf{v}(x)$ | Edge field $g_e$, $e \in \mathcal{E}$ |
| Gradient $\nabla f$ | $D^T f$ (nodes → edges) |
| Divergence $\nabla \cdot \mathbf{v}$ | $D g$ (edges → nodes) |
| Laplacian $\nabla^2 f$ | $D D^T f = L f$ |

### Weighted vs. Unweighted Graphs

Neurospatial uses **weighted graphs** where edge weights represent spatial distances. The differential operator accounts for these weights using the square root scaling $\sqrt{w_e}$.

For **unweighted graphs** (uniform weights $w_e = 1$), the differential operator reduces to the **incidence matrix** used in algebraic graph theory.

### Sign Convention

The differential operator uses the convention:
- **Source node**: $-\sqrt{w_e}$ (negative coefficient)
- **Destination node**: $+\sqrt{w_e}$ (positive coefficient)

This ensures:
- Gradient measures change **along** edge direction
- Divergence is positive for **net outflow** (source)
- Laplacian $L = D D^T$ is positive semi-definite

## Advanced Topics

### Computing Laplacian Smoothing

Implement iterative Laplacian smoothing (heat diffusion):

```python
from neurospatial import gradient, divergence
import numpy as np

def laplacian_smooth(field, env, alpha=0.1, n_iterations=10):
    """Smooth field via heat diffusion on graph.

    Parameters
    ----------
    field : NDArray[np.float64]
        Initial field values (n_bins,)
    env : Environment
        Spatial environment
    alpha : float
        Step size (controls diffusion rate)
    n_iterations : int
        Number of iterations

    Returns
    -------
    smoothed : NDArray[np.float64]
        Smoothed field values
    """
    smoothed = field.copy()

    for _ in range(n_iterations):
        # Compute Laplacian: L(f) = div(grad(f))
        grad_field = gradient(smoothed, env)
        laplacian = divergence(grad_field, env)

        # Heat equation: f_new = f - alpha * L(f)
        smoothed = smoothed - alpha * laplacian

    return smoothed

# Example usage
noisy_field = np.random.randn(env.n_bins)
smooth_field = laplacian_smooth(noisy_field, env, alpha=0.05, n_iterations=20)
```

### Edge Field Visualization

Visualize edge fields by plotting values along edges:

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_edge_field(edge_field, env, ax=None, cmap='RdBu_r', vmin=None, vmax=None):
    """Plot edge field as colored lines between bin centers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Get edge endpoints
    for i, (u, v) in enumerate(env.connectivity.edges()):
        pos_u = env.bin_centers[u]
        pos_v = env.bin_centers[v]

        # Plot edge with color based on value
        value = edge_field[i]
        color = plt.cm.get_cmap(cmap)((value - (vmin or edge_field.min())) /
                                       ((vmax or edge_field.max()) - (vmin or edge_field.min())))
        ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]],
                color=color, linewidth=2, alpha=0.7)

    ax.set_aspect('equal')
    ax.set_xlabel(f'X ({env.units})')
    ax.set_ylabel(f'Y ({env.units})')

    return ax

# Example: visualize gradient field
distances = distance_field(env.connectivity, sources=[goal_bin])
grad_distances = gradient(distances, env)

plot_edge_field(grad_distances, env)
plt.title('Distance Gradient (Edge Field)')
plt.show()
```

## Comparison with Existing Functions

### `gradient()` vs. `env.smooth()`

| Feature | `gradient()` | `env.smooth()` |
|---------|-------------|---------------|
| Input | Scalar field (nodes) | Scalar field (nodes) |
| Output | Edge field (edges) | Scalar field (nodes) |
| Purpose | Compute spatial derivative | Reduce noise via Gaussian kernel |
| Use case | RL policy gradients, flow analysis | Place field smoothing |
| Kernel | N/A (differential operator) | Gaussian (distance-based) |

**Note**: `divergence()` was added in v0.3.0 as a graph signal processing operator for detecting sources and sinks in flow fields. For comparing probability distributions (KL divergence, JS divergence), use `scipy.stats.entropy()` or `scipy.spatial.distance.jensenshannon()`.

## References

1. **Shuman, D. I., Narang, S. K., Frossard, P., Ortega, A., & Vandergheynst, P. (2013).** The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains. *IEEE Signal Processing Magazine*, 30(3), 83-98.

2. **Perraudin, N., Paratte, J., Shuman, D., Kalofolias, V., Vandergheynst, P., & Hammond, D. K. (2014).** GSPBOX: A toolbox for signal processing on graphs. *arXiv preprint arXiv:1408.5781*.

3. **Stachenfeld, K. L., Botvinick, M. M., & Gershman, S. J. (2017).** The hippocampus as a predictive map. *Nature Neuroscience*, 20(11), 1643-1653. [Successor representations and flow fields]

4. **Pfeiffer, B. E., & Foster, D. J. (2013).** Hippocampal place-cell sequences depict future paths to remembered goals. *Nature*, 497(7447), 74-79. [Replay analysis applications]

## See Also

- [Spike-Field Primitives](spike-field-primitives.md) - Converting spike trains to firing rate fields
- [RL Primitives](rl-primitives.md) - Reward field generation for reinforcement learning
- [Spatial Analysis](spatial-analysis.md) - Core spatial query and analysis methods
- [Performance Guide](performance.md) - Optimization strategies and caching

## API Reference

- `neurospatial.compute_differential_operator(env)` - Build differential operator matrix
- `neurospatial.gradient(field, env)` - Compute gradient (scalar → edge field)
- `neurospatial.divergence(edge_field, env)` - Compute divergence (edge → scalar field)
- `env.differential_operator` - Cached differential operator property
- `neurospatial.distance_field(graph, sources)` - Multi-source geodesic distances
- `env.smooth(field, bandwidth)` - Gaussian smoothing on graph
