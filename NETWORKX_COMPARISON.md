# NetworkX vs Proposed Differential Operators

## Quick Answer

**Yes, NetworkX has Laplacian functions, but NO, it doesn't have gradient or proper graph signal processing operators.**

## What NetworkX Provides

### 1. Laplacian Matrix Functions

```python
import networkx as nx

# Multiple Laplacian variants
nx.laplacian_matrix(G)                    # L = D - A (unnormalized)
nx.normalized_laplacian_matrix(G)          # Normalized variant
nx.directed_laplacian_matrix(G)            # For directed graphs
nx.directed_combinatorial_laplacian_matrix(G)
```

**Returns**: The Laplacian matrix **L** directly as a sparse matrix.

For undirected graphs: `L = D - A`
- D = diagonal degree matrix
- A = adjacency matrix

### 2. Incidence Matrix

```python
B = nx.incidence_matrix(G, oriented=True)
```

**Returns**: Unweighted oriented incidence matrix **B** where:
- `B[node, edge] = -1` if node is source
- `B[node, edge] = +1` if node is target
- `B[node, edge] = 0` otherwise

**Key property**: `B @ B.T = L` (Laplacian)

## What NetworkX Does NOT Provide

### 1. Gradient Operator ❌

```python
# Does NOT exist in NetworkX
gradient = nx.gradient(field, G)  # AttributeError!
```

**What we need**: Transform scalar field on nodes → vector field on edges

### 2. Divergence Operator ❌

```python
# Does NOT exist in NetworkX
div = nx.divergence(edge_field, G)  # AttributeError!
```

**What we need**: Transform vector field on edges → scalar field on nodes

### 3. Weighted Differential Operator ❌

NetworkX's `incidence_matrix()` is **unweighted** (values are just -1, 0, +1).

For graph signal processing, we need the **weighted differential operator D**:

```
D[source, edge] = -√(edge_weight)
D[target, edge] = +√(edge_weight)
```

This is what PyGSP provides and what we benchmarked in `benchmark_differential_operator.py`.

### 4. Graph Signal Operations ❌

NetworkX does not provide:
- Gradient computation on irregular graphs
- Divergence computation
- Signal smoothing on graph domains
- Spatial autocorrelation on graphs
- Convolution with graph kernels

## Why NetworkX's Tools Aren't Sufficient

### Example: Computing Gradient of a Field

```python
import networkx as nx
import numpy as np

G = nx.Graph()
G.add_edge(0, 1, distance=2.0)
G.add_edge(1, 2, distance=3.0)

# Scalar field on nodes (e.g., firing rates at each bin)
field = np.array([1.0, 2.0, 3.0])

# NetworkX approach - UNWEIGHTED
B = nx.incidence_matrix(G, oriented=True)
gradient_unweighted = B.T @ field
# Result: [1.0, 1.0] - ignores edge weights!

# What we need - WEIGHTED differential operator
# D[source, edge] = -√w, D[target, edge] = +√w
# gradient_weighted = D.T @ field
# Result accounts for edge distances: [-√2, √3]
```

### Why This Matters for Neurospatial

In neuroscience applications:

1. **Edge weights represent physical distances** in the environment
2. **Gradient magnitude should reflect spatial rate of change** (spikes/cm)
3. **Spatial autocorrelation** requires proper distance weighting
4. **Place field analysis** needs weighted spatial derivatives

NetworkX's unweighted incidence matrix is designed for **graph topology**, not **spatial signal processing**.

## What PyGSP Provides (That NetworkX Doesn't)

PyGSP (Python Graph Signal Processing) implements:

```python
# PyGSP has these operations
G.grad(signal)           # Weighted gradient
G.div(edge_signal)       # Weighted divergence
G.D                      # Differential operator (weighted)
```

**Our investigation found**: PyGSP's implementation is the right approach for neurospatial.

## Summary Table

| Operation | NetworkX | PyGSP | Neurospatial (Proposed) |
|-----------|----------|-------|-------------------------|
| Laplacian matrix | ✅ `laplacian_matrix()` | ✅ `L` | ✅ Via layout engines |
| Incidence matrix | ✅ Unweighted `incidence_matrix()` | ✅ Weighted `D` | 🔶 Need to add |
| Gradient | ❌ | ✅ `grad()` | 🔶 Need to add |
| Divergence | ❌ | ✅ `div()` | 🔶 Need to add |
| Spatial autocorrelation | ❌ | ❌ | 🔶 Need to add (UNIQUE) |
| Convolution | ❌ | ✅ `filter()` | 🔶 Need to add |
| API fits neuroscience | ❌ | 🔶 Partial | ✅ Designed for neuro |

Legend:
- ✅ = Fully supported
- 🔶 = Needs implementation
- ❌ = Not available

## Conclusion

**NetworkX provides basic graph topology tools** (Laplacian, unweighted incidence matrix) but **does not provide graph signal processing operators** (weighted gradient, divergence, filtering).

**Our proposal is NOT redundant** - we need:
1. Weighted differential operator (like PyGSP)
2. Gradient/divergence operations for spatial fields
3. Spatial autocorrelation (unique to neurospatial)
4. Neuroscience-friendly API

NetworkX is great for what it does (graph algorithms, topology, traversal), but spatial signal processing on irregular grids requires specialized operators that neither NetworkX nor PyGSP fully provide in a neuroscience-appropriate API.

## References

- NetworkX Laplacian: https://networkx.org/documentation/stable/reference/generated/networkx.linalg.laplacianmatrix.laplacian_matrix.html
- PyGSP gradient: https://pygsp.readthedocs.io/en/stable/reference/graphs.html#pygsp.graphs.Graph.grad
- Our benchmark: `benchmark_differential_operator.py`
