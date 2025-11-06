# How Existing Packages Implement Differential Operators

## Comprehensive Analysis of Implementation Strategies

Based on investigation of PyGSP, LaPy, pcdiff, and related packages.

---

## 🔵 PyGSP Implementation (EPFL LTS2)

### **Core Data Structure: Incidence Matrix → Differential Operator**

PyGSP uses the **incidence matrix** approach to build differential operators.

### **`compute_differential_operator()` Implementation**

```python
def compute_differential_operator(self):
    """
    Construct differential operator D such that L = D @ D.T

    D has shape (n_vertices, n_edges)
    L is the graph Laplacian
    """
    sources, targets, weights = self.get_edge_list()

    n = self.n_edges
    rows = np.concatenate([sources, targets])  # Both directions
    columns = np.concatenate([np.arange(n), np.arange(n)])
    values = np.empty(2 * n)

    if self.lap_type == "combinatorial":
        # D[i,e] = -√(w_e) for source vertex
        # D[j,e] = +√(w_e) for target vertex
        values[:n] = -np.sqrt(weights)
        values[n:] = -values[:n]  # Negate for reverse direction

    elif self.lap_type == "normalized":
        # Normalize by vertex degrees
        values[:n] = -np.sqrt(weights / self.dw[sources])
        values[n:] = +np.sqrt(weights / self.dw[targets])

    if self.is_directed():
        values /= np.sqrt(2)

    # Build sparse matrix (CSC format for efficient column operations)
    self._D = sparse.csc_matrix(
        (values, (rows, columns)),
        shape=(self.n_vertices, self.n_edges)
    )
    self._D.eliminate_zeros()
```

### **Key Insights from PyGSP:**

1. **Incidence Matrix Foundation**:
   - Each edge gets a column
   - Each vertex gets a row
   - Entry (i, e): How vertex i participates in edge e

2. **Bidirectional Encoding**:
   - For undirected graphs: both endpoints encoded
   - For edge (i, j): D[i, e] = -√w, D[j, e] = +√w

3. **Two Laplacian Types**:
   - **Combinatorial**: `L = D - A` (unnormalized)
   - **Normalized**: `L = I - D^(-1/2) A D^(-1/2)`

4. **Sparse Matrix Format**:
   - CSC (Compressed Sparse Column) for efficient matrix-vector products
   - Only ~2 non-zero entries per edge (source + target)

### **Gradient Implementation**

```python
def grad(self, x):
    """
    Compute gradient of signal x on vertices.

    Parameters
    ----------
    x : array, shape (n_vertices,)
        Signal on vertices

    Returns
    -------
    y : array, shape (n_edges,)
        Gradient signal on edges
        y[e] = √(w_e) * (x[target] - x[source])
    """
    if self._D is None:
        self.compute_differential_operator()

    # Gradient = D.T @ x
    return self._D.T @ x
```

**Mathematical meaning**:
```
gradient[edge(i→j)] = √w_ij * (signal[j] - signal[i])
```

### **Divergence Implementation**

```python
def div(self, y):
    """
    Compute divergence of signal y on edges.

    Parameters
    ----------
    y : array, shape (n_edges,)
        Signal on edges

    Returns
    -------
    z : array, shape (n_vertices,)
        Divergence signal on vertices
        z[i] = sum over incident edges
    """
    if self._D is None:
        self.compute_differential_operator()

    # Divergence = D @ y
    return self._D @ y
```

**Mathematical meaning**:
```
divergence[vertex_i] = sum_{edges incident to i} weighted_edge_signal
```

### **Laplacian Relationship**

```python
# Laplacian = divergence of gradient
L = D @ D.T

# For a signal x:
Lx = D @ (D.T @ x) = divergence(gradient(x))
```

---

## 🔵 LaPy Implementation (Medical Imaging)

### **Approach: Finite Element Method (FEM) on Meshes**

LaPy uses **cotangent weights** for triangle meshes - a classical FEM approach.

### **Gradient on Triangle Meshes**

```python
def gradient(mesh, field):
    """
    Compute gradient of scalar field on triangle mesh.

    Uses barycentric coordinates and edge cotangent weights.
    Returns gradient vectors at triangle centers.
    """
    # For each triangle:
    # 1. Compute edge vectors
    # 2. Compute cotangent weights at opposite vertices
    # 3. Gradient = sum of cotangent-weighted perpendicular vectors

    # Gradient lives in tangent plane of each triangle
```

**Key difference from PyGSP**:
- PyGSP: Gradient on **graph edges** (1D manifold)
- LaPy: Gradient on **triangle faces** (2D manifold)

### **Cotangent Laplacian**

```python
# Laplacian matrix for vertex i:
L[i,j] = (cot(α_ij) + cot(β_ij)) / 2

# Where α_ij and β_ij are angles opposite to edge (i,j)
```

**Why cotangent?**
- Discretization of continuous Laplace-Beltrami operator
- Preserves geometric properties (angles, areas)
- Widely used in geometry processing

---

## 🔵 pcdiff Implementation (Point Clouds)

### **Approach: Moving Least Squares (MLS)**

```python
def compute_operators(points, k_neighbors=20):
    """
    Build gradient/divergence operators via local surface fitting.

    Algorithm:
    1. For each point, find k nearest neighbors
    2. Fit local tangent plane using PCA
    3. Estimate local coordinate frame
    4. Build gradient operator by differencing in local coordinates
    5. Build divergence as adjoint of gradient
    """
```

### **Matrix Structure**

```python
# Gradient operator: (2*N, N) sparse matrix
# For each point i:
#   grad[2*i, neighbors(i)]   = x-component weights
#   grad[2*i+1, neighbors(i)] = y-component weights

# Divergence operator: (N, 2*N) sparse matrix
# div = -grad.T (adjoint relationship)

# Laplacian: L = div @ grad
```

**Key insight**: Approximate smooth manifold locally, then apply calculus.

---

## 🔵 NetworkX Implementation (Graph Analysis)

### **Laplacian Only**

```python
import networkx as nx

def laplacian_matrix(G, nodelist=None, weight='weight'):
    """
    L = D - A

    Where:
    - D is diagonal degree matrix
    - A is adjacency matrix
    """
    A = nx.adjacency_matrix(G, nodelist, weight)
    n, m = A.shape
    diags = A.sum(axis=1).A.flatten()  # Degrees
    D = sparse.diags(diags, 0, shape=(n, m), format='csr')
    L = D - A
    return L
```

**What's missing**: No gradient or divergence operators (not in NetworkX's scope).

---

## 📊 **Comparison of Approaches**

| Package | Data Structure | Gradient | Divergence | Laplacian | Best For |
|---------|----------------|----------|------------|-----------|----------|
| **PyGSP** | Graph edges | ✅ (D.T) | ✅ (D) | ✅ (D@D.T) | Arbitrary graphs |
| **LaPy** | Triangle meshes | ✅ (FEM) | ✅ (FEM) | ✅ (Cotangent) | Smooth surfaces |
| **pcdiff** | Point clouds | ✅ (MLS) | ✅ (MLS) | ✅ (composition) | Irregular samples |
| **NetworkX** | Graph | ❌ | ❌ | ✅ (D-A) | Graph analysis |
| **neurospatial** | Bin graphs | **Needed** | **Needed** | ⚠️ (have via kernels) | Spatial discretization |

---

## 🎯 **Best Practices Learned**

### **1. Incidence Matrix is Foundation**

**Core idea**: Build differential operator D from graph structure

```python
# For each edge e = (i, j):
D[i, e] = -√w_e    # Source vertex
D[j, e] = +√w_e    # Target vertex

# Then:
# Gradient: D.T @ field_on_nodes → field_on_edges
# Divergence: D @ field_on_edges → field_on_nodes
# Laplacian: D @ D.T (composition)
```

### **2. Sparse Matrices are Essential**

```python
# Graph with N nodes, E edges:
# - D is (N, E) but only ~2E non-zeros
# - Sparsity ratio: 2E / (N*E) = 2/N ≈ 0.1% for large graphs

# Use scipy.sparse:
from scipy.sparse import csr_matrix, csc_matrix

# CSR (row) for row operations (div, Laplacian)
# CSC (column) for column operations (grad)
```

### **3. Two Normalization Schemes**

**Combinatorial Laplacian** (unnormalized):
```python
# Simple, interpretable
# But scale depends on graph size/density
L = D - A
```

**Normalized Laplacian**:
```python
# Scale-invariant
# Better for spectral methods
L_norm = I - D^(-1/2) @ A @ D^(-1/2)
```

### **4. Caching is Critical**

```python
class GraphOperators:
    def __init__(self, graph):
        self.graph = graph
        self._D = None  # Lazy computation

    @property
    def D(self):
        if self._D is None:
            self._D = self._compute_differential_operator()
        return self._D
```

### **5. Validation Strategy**

```python
def test_laplacian_identity():
    """Test L = D @ D.T"""
    G = create_test_graph()

    # Method 1: Build L directly
    L_direct = compute_laplacian(G)

    # Method 2: Via differential operator
    D = compute_differential_operator(G)
    L_from_D = D @ D.T

    assert np.allclose(L_direct.toarray(), L_from_D.toarray())
```

---

## 🔑 **Recommended Implementation for neurospatial**

### **Option 1: PyGSP-Style (Recommended)**

**Pros:**
- ✅ Works on arbitrary graphs (hexagonal, masked, irregular)
- ✅ Well-tested mathematical foundation
- ✅ Efficient sparse implementation
- ✅ Natural fit for neurospatial's graph connectivity

**Implementation sketch**:

```python
import numpy as np
from scipy import sparse

def compute_differential_operator(env):
    """
    Build differential operator for neurospatial Environment.

    Returns D: (n_bins, n_edges) sparse matrix
    """
    G = env.connectivity

    # Extract edges with attributes
    edges = list(G.edges(data=True))
    n_edges = len(edges)
    n_bins = env.n_bins

    # Build incidence matrix
    rows = []
    cols = []
    vals = []

    for edge_idx, (i, j, data) in enumerate(edges):
        weight = data.get('distance', 1.0)

        # Source vertex contribution
        rows.append(i)
        cols.append(edge_idx)
        vals.append(-np.sqrt(weight))

        # Target vertex contribution
        rows.append(j)
        cols.append(edge_idx)
        vals.append(+np.sqrt(weight))

    D = sparse.csc_matrix(
        (vals, (rows, cols)),
        shape=(n_bins, n_edges)
    )

    return D


def gradient(field, env):
    """
    Compute gradient of scalar field on graph.

    Parameters
    ----------
    field : array, shape (n_bins,)
        Scalar field on bins
    env : Environment
        Neurospatial environment

    Returns
    -------
    grad_field : array, shape (n_edges,)
        Gradient on edges
    """
    D = compute_differential_operator(env)
    return D.T @ field


def divergence(edge_field, env):
    """
    Compute divergence of edge field.

    Parameters
    ----------
    edge_field : array, shape (n_edges,)
        Field defined on edges
    env : Environment

    Returns
    -------
    div_field : array, shape (n_bins,)
        Divergence on bins
    """
    D = compute_differential_operator(env)
    return D @ edge_field


def laplacian_explicit(field, env):
    """
    Compute Laplacian = divergence of gradient.

    Parameters
    ----------
    field : array, shape (n_bins,)
    env : Environment

    Returns
    -------
    laplacian_field : array, shape (n_bins,)
    """
    D = compute_differential_operator(env)
    L = D @ D.T  # Laplacian matrix
    return L @ field
```

### **Integration with Existing neurospatial**

```python
# Already have:
env.connectivity  # NetworkX graph with edge attributes
env.n_bins       # Number of bins
env.bin_centers  # Positions (for visualization)

# Already have (implicit Laplacian):
from neurospatial import compute_diffusion_kernels
kernel = compute_diffusion_kernels(env.connectivity, bandwidth=5.0)
# This uses graph Laplacian internally via NetworkX

# New (explicit gradient/divergence):
from neurospatial.differential import gradient, divergence, laplacian

grad = gradient(firing_rate, env)
div_grad = divergence(grad, env)  # Should equal laplacian(firing_rate, env)
```

---

## 🧪 **Validation Tests to Implement**

### **Test 1: Laplacian Identity**

```python
def test_laplacian_from_differential_operator():
    """L = D @ D.T"""
    env = create_test_environment()
    field = np.random.rand(env.n_bins)

    # Method 1: Via NetworkX
    L_nx = nx.laplacian_matrix(env.connectivity)
    result1 = L_nx @ field

    # Method 2: Via differential operator
    D = compute_differential_operator(env)
    L = D @ D.T
    result2 = L @ field

    np.testing.assert_allclose(result1, result2)
```

### **Test 2: Divergence of Gradient**

```python
def test_divergence_of_gradient_equals_laplacian():
    """div(grad(f)) = Laplacian(f)"""
    env = create_test_environment()
    field = np.random.rand(env.n_bins)

    # Method 1: Composition
    grad_f = gradient(field, env)
    div_grad_f = divergence(grad_f, env)

    # Method 2: Direct Laplacian
    laplacian_f = laplacian_explicit(field, env)

    np.testing.assert_allclose(div_grad_f, laplacian_f)
```

### **Test 3: Smooth Signal has Small Gradient**

```python
def test_gradient_of_smooth_signal():
    """Constant signal → zero gradient"""
    env = create_test_environment()
    field = np.ones(env.n_bins) * 5.0  # Constant

    grad = gradient(field, env)

    # Gradient should be near zero
    np.testing.assert_allclose(grad, 0.0, atol=1e-10)
```

### **Test 4: Regular Grid Matches Finite Differences**

```python
def test_gradient_on_regular_grid():
    """On regular grid, should match finite differences"""
    env = create_regular_grid_environment(10, 10)

    # Create linear field: f(x,y) = x + 2*y
    field = env.bin_centers[:, 0] + 2 * env.bin_centers[:, 1]

    grad = gradient(field, env)

    # Gradient should be approximately (1, 2) everywhere
    # (actual values depend on edge orientation)
```

---

## 📝 **Implementation Roadmap**

### **Phase 1: Differential Operator (Week 1)**

- [ ] Implement `compute_differential_operator(env)`
- [ ] Cache in Environment class
- [ ] Unit tests: incidence matrix properties
- [ ] Validate against NetworkX Laplacian

### **Phase 2: Gradient & Divergence (Week 1-2)**

- [ ] Implement `gradient(field, env)`
- [ ] Implement `divergence(edge_field, env)`
- [ ] Unit tests: composition tests
- [ ] Validate: div(grad(f)) = Laplacian(f)

### **Phase 3: Integration (Week 2)**

- [ ] Add to `src/neurospatial/differential.py`
- [ ] Export in `__init__.py`
- [ ] Documentation with examples
- [ ] Tutorial notebook

### **Phase 4: Optimization (Week 3)**

- [ ] Profile performance
- [ ] Optimize sparse matrix operations
- [ ] Add caching strategy
- [ ] Benchmark against PyGSP

---

## 🎓 **Key Takeaways**

1. **PyGSP's approach is most suitable** for neurospatial (arbitrary graph structures)

2. **Incidence matrix → Differential operator → Gradient/Divergence** is the canonical pipeline

3. **Sparse matrices are non-negotiable** for performance on large graphs

4. **Validation through composition** (div∘grad = Laplacian) is essential

5. **Can implement cleanly** without external dependencies (just scipy.sparse)

6. **Integration point**: Builds on existing `env.connectivity` graph

7. **Complements existing**: Already have diffusion kernels (implicit Laplacian), now get explicit gradient/divergence

---

## 🔗 **References for Implementation**

1. **PyGSP Source Code**:
   - https://github.com/epfl-lts2/pygsp/blob/master/pygsp/graphs/difference.py

2. **Graph Signal Processing Theory**:
   - Shuman et al. "The Emerging Field of Signal Processing on Graphs" (2013)
   - https://arxiv.org/abs/1211.0053

3. **Discrete Differential Geometry**:
   - Crane, "Discrete Differential Geometry: An Applied Introduction"
   - http://www.cs.cmu.edu/~kmcrane/Projects/DDG/

4. **NetworkX Laplacian**:
   - https://networkx.org/documentation/stable/reference/linalg.html

---

This analysis provides a complete blueprint for implementing differential operators in neurospatial based on proven, battle-tested approaches from the graph signal processing community.
