# Differential Geometry Packages: Comparison & Positioning

## Existing Python Packages for Differential Geometry

### 🔵 **PyDEC** - Discrete Exterior Calculus
**Repository**: `hirani/pydec` (GitHub)
**Publication**: ACM TOMS 2012

**What it does:**
- Discrete exterior calculus on simplicial complexes
- Exterior derivative (coboundary operator)
- Hodge star operator
- Whitney forms (lowest-order finite elements)

**Scope**: General differential forms, algebraic topology, exterior calculus

**Strengths:**
- Rigorous mathematical framework
- Handles arbitrary simplicial complexes
- Topological correctness

**Limitations for neurospatial:**
- ❌ No spatial autocorrelation
- ❌ Not designed for neuroscience workflows
- ❌ Focus on topology, not spatial analysis
- ❌ Overkill for most neuroscience use cases

---

### 🔵 **PyGSP** - Graph Signal Processing
**Repository**: `epfl-lts2/pygsp` (GitHub)
**Institution**: EPFL LTS2 Laboratory

**What it does:**
```python
G.compute_differential_operator()  # Gradient operator D
G.grad(signal)                     # Gradient: D @ signal
G.div(signal)                      # Divergence
G.compute_laplacian()              # Graph Laplacian L = D^T @ D
```

**Scope**: Signal processing on graphs (spectral methods, filtering)

**Strengths:**
- ✅ Graph-based differential operators
- ✅ Gradient and divergence
- ✅ Spectral graph theory
- ✅ Efficient for large graphs
- ✅ Active maintenance (EPFL)

**Limitations for neurospatial:**
- ❌ **No spatial autocorrelation** (only spectral methods)
- ❌ No trajectory operations
- ❌ No RL primitives
- ❌ Signal processing focus, not spatial analysis
- ❌ No neuroscience-specific features

**Overlap**: **~20%**
- Gradient operator exists but different interface
- Laplacian computation (neurospatial already has via `compute_diffusion_kernels`)

---

### 🔵 **LaPy** - Mesh Differential Geometry
**Repository**: `Deep-MI/LaPy` (GitHub)

**What it does:**
- Gradient, divergence, Laplacian on triangle/tetrahedral meshes
- FEM solvers (Laplace, Poisson, Heat equations)
- Mean-curvature flow
- Geodesics, conformal mappings
- ShapeDNA (Laplace spectra)

**Scope**: Medical imaging, surface analysis, FEM

**Strengths:**
- ✅ Complete differential operators
- ✅ Fast (vectorized Python)
- ✅ Medical imaging focus

**Limitations for neurospatial:**
- ❌ **Triangle/tet meshes only** (not arbitrary graphs)
- ❌ No irregular bin layouts (hexagonal, masked grids)
- ❌ No spatial autocorrelation
- ❌ No trajectory/behavioral analysis
- ❌ No RL primitives

**Overlap**: **~15%**
- Both compute gradient/Laplacian, but on different structures

---

### 🔵 **pcdiff** - Point Cloud Differential Operators
**Repository**: `rubenwiersma/pointcloud-differential`
**Available**: PyPI (`pip install pcdiff`)

**What it does:**
```python
pcdiff.operators.gradient(points, values)
pcdiff.operators.divergence(points, vector_field)
pcdiff.operators.laplacian(points, values)
```

**Scope**: Point clouds, deep learning on 3D data

**Strengths:**
- ✅ Gradient, divergence, Laplacian
- ✅ Works on point clouds

**Limitations for neurospatial:**
- ❌ **Point clouds, not bin-based discretization**
- ❌ No spatial autocorrelation
- ❌ No trajectory operations
- ❌ No RL support
- ❌ 3D graphics/ML focus

**Overlap**: **~10%**
- Similar operators, but different data structure

---

### 🔵 **PyTorch Geometric** - Graph Neural Networks
**Repository**: `pyg-team/pytorch_geometric`

**What it does:**
```python
ChebConv(...)  # Chebyshev spectral graph convolution (uses Laplacian)
GCNConv(...)   # Graph convolutional network
LaplacianLambdaMax()  # Compute max eigenvalue
```

**Scope**: Graph neural networks, deep learning

**Strengths:**
- ✅ Graph Laplacian operations
- ✅ Efficient GPU computation
- ✅ Spectral convolutions

**Limitations for neurospatial:**
- ❌ **Deep learning focus**, not spatial analysis
- ❌ No differential operators as standalone tools
- ❌ No spatial autocorrelation
- ❌ No trajectory/behavioral primitives
- ❌ Requires PyTorch

**Overlap**: **<5%**
- Both use graph Laplacian, but for entirely different purposes

---

### 🔵 **libigl** - Geometry Processing
**C++ library with Python bindings**

**What it does:**
- Gradient operator on triangle meshes
- Cotangent Laplacian
- Mesh processing utilities

**Scope**: 3D geometry, graphics, mesh processing

**Limitations for neurospatial:**
- ❌ Triangle meshes only
- ❌ C++ dependency
- ❌ Graphics/CAD focus
- ❌ No spatial analysis primitives

**Overlap**: **<5%**

---

### 🔵 **NetworkX** - Graph Analysis
**What it has:**
```python
nx.laplacian_matrix(G)          # Laplacian matrix
nx.laplacian_spectrum(G)         # Eigenvalues
```

**What it's missing:**
- ❌ No gradient operator
- ❌ No divergence operator
- ❌ No differential geometry beyond Laplacian
- ❌ Graph analysis focus, not spatial operators

**Status**: Neurospatial already uses NetworkX for connectivity.

---

## 📊 **Summary Comparison**

| Package | Gradient | Divergence | Laplacian | Autocorr | Trajectories | RL | Neuroscience | Graph Arbitrary Layouts |
|---------|----------|------------|-----------|----------|--------------|----|--------------|----|
| **PyDEC** | ✅ (exterior) | ✅ (exterior) | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ (simplicial) |
| **PyGSP** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **LaPy** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ (medical) | ❌ (meshes only) |
| **pcdiff** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ (point clouds) |
| **PyG** | ⚠️ (implicit) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **libigl** | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ (meshes) |
| **NetworkX** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **neurospatial (proposed)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 🎯 **Key Findings**

### **1. PyGSP is the Closest**

**Overlap**: ~20%
- Has `grad()`, `div()`, Laplacian
- Graph-based (like neurospatial)
- Efficient implementation

**Why not just use PyGSP?**
- ❌ **No spatial autocorrelation** - ESSENTIAL for grid cells
- ❌ No trajectory operations (occupancy, transitions, paths)
- ❌ No RL primitives (Bellman, propagate, accumulate)
- ❌ Signal processing API, not spatial analysis API
- ❌ No integration with neurospatial's Environment/layout system

**Decision**: Could potentially use PyGSP's gradient implementation, but need to wrap it in neurospatial API.

---

### **2. No Package Handles Spatial Autocorrelation**

**Critical for neuroscience:**
```python
# Grid cell analysis - NO EXISTING PACKAGE DOES THIS!
autocorr = spatial_autocorrelation(firing_rate, env, max_lag=20)
grid_score = compute_hexagonal_score(autocorr)
```

**Current state**: Every neuroscience lab implements this themselves
- ❌ Only works on regular grids
- ❌ Breaks on hexagonal/triangular/masked layouts
- ❌ No graph-aware correlation

**Conclusion**: This is a **unique contribution** neurospatial would provide.

---

### **3. No Package Bridges Differential Geometry + Neuroscience**

| Need | Exists In | Missing |
|------|-----------|---------|
| Gradient on graphs | PyGSP, LaPy | ✅ Available |
| Spatial autocorrelation | **NONE** | ❌ Must implement |
| Trajectory primitives | **NONE** | ❌ Must implement |
| RL primitives | **NONE** | ❌ Must implement |
| Neuroscience workflows | opexebo (limited) | ⚠️ Only basic grids |

**Conclusion**: Neurospatial fills a **unique niche** by combining:
- Differential geometry (gradient, Laplacian)
- Spatial correlation (autocorr, xcorr)
- Trajectory analysis (accumulate_along_path)
- RL primitives (propagate, Bellman)
- All on arbitrary graph layouts

---

## 💡 **Implementation Strategy**

### **Option 1: Implement Everything (Recommended)**

**Pros:**
- ✅ Full control over API
- ✅ Integration with Environment class
- ✅ Can optimize for neurospatial's data structures
- ✅ No external dependencies

**Cons:**
- ⚠️ More implementation work
- ⚠️ Need to validate correctness

**Recommendation**: **Yes**, because:
- PyGSP's API doesn't match neurospatial's needs
- Need spatial autocorrelation anyway (no existing solution)
- Need trajectory/RL primitives anyway (no existing solution)
- Better integration with existing neurospatial features

---

### **Option 2: Wrap PyGSP for Gradient/Divergence**

**Possible:**
```python
# Internal implementation
def gradient(field, env):
    # Convert to PyGSP graph
    G_gsp = pygsp.graphs.Graph(env.connectivity)
    G_gsp.compute_differential_operator()
    grad = G_gsp.grad(field)
    return grad
```

**Pros:**
- ✅ Reuse validated implementation
- ✅ Less code to maintain

**Cons:**
- ❌ Extra dependency (pygsp)
- ❌ API mismatch (need adapter layer)
- ❌ Still need to implement autocorr, trajectories, RL ourselves
- ❌ Performance overhead (conversion)

**Recommendation**: **Maybe** for gradient/divergence only, but still need to implement:
- `spatial_autocorrelation` (unique)
- `neighbor_reduce` (unique)
- `accumulate_along_path` (unique)
- `propagate` (unique)
- `convolve` (custom kernels)

---

## 🔑 **Conclusion**

### **What Exists:**
✅ Differential geometry libraries (PyDEC, LaPy)
✅ Graph signal processing (PyGSP)
✅ Mesh operators (libigl, pcdiff)
✅ Deep learning on graphs (PyTorch Geometric)

### **What's Missing for Neurospatial:**
❌ **Spatial autocorrelation** on arbitrary graphs
❌ **Trajectory primitives** (accumulate_along_path)
❌ **RL primitives** (propagate, Bellman)
❌ **Integration** with neurospatial's Environment/layouts
❌ **Neuroscience workflows**

### **Overlap Assessment:**
- **PyGSP**: 20% overlap (gradient, divergence, Laplacian)
- **Others**: <15% overlap (wrong data structures or focus)

### **Recommendation:**

**Implement differential operators natively in neurospatial:**

**Tier 1 - Must Implement (No alternatives):**
1. ✅ `spatial_autocorrelation` - NO package has this for graphs
2. ✅ `neighbor_reduce` - Fundamental primitive, unique
3. ✅ `accumulate_along_path` - Trajectory primitive, unique
4. ✅ `propagate` - RL primitive, unique
5. ✅ `integrate` (weighted) - Need bin size awareness

**Tier 2 - Could Use PyGSP But Better Native:**
6. ⚠️ `gradient` - PyGSP has it, but API mismatch
7. ⚠️ `divergence_vector` - PyGSP has it, but API mismatch
8. ⚠️ `laplacian` - Already have via `compute_diffusion_kernels`

**Tier 3 - Extend from Tier 1:**
9. ✅ `convolve` (custom kernels) - Build from neighbor_reduce
10. ✅ `spatial_cross_correlation` - Similar to autocorr

---

## 📝 **Final Verdict**

**Are we reinventing the wheel?**

**NO for the core value proposition:**
- ❌ No package does spatial autocorrelation on graphs
- ❌ No package has trajectory/RL primitives
- ❌ No package integrates differential geometry + neuroscience

**PARTIALLY for differential operators:**
- ⚠️ PyGSP has gradient/divergence, but:
  - Different API (signal processing vs spatial analysis)
  - Extra dependency
  - Still need 80% of other functionality anyway

**Decision**: **Implement natively**
- Clean integration with neurospatial
- No external dependencies
- Can optimize for our use cases
- Need to implement spatial autocorr/trajectories/RL anyway

**Validation strategy**: Use PyGSP as reference implementation to validate correctness of our gradient/divergence.

---

## 🎓 **Positioning**

**neurospatial = Spatial discretization + Differential geometry + Neuroscience workflows**

Not competing with PyGSP (signal processing) or LaPy (medical meshes).

**Filling a gap**: Graph-based spatial operators specifically designed for neuroscience (place cells, grid cells, navigation, RL).

**Unique contributions:**
1. Spatial autocorrelation on arbitrary graphs ⭐
2. Trajectory primitives (accumulate, path operations) ⭐
3. RL primitives (propagate, Bellman) ⭐
4. Integration with Environment/layouts ⭐
5. Neuroscience-specific API ⭐

Plus standard differential operators (gradient, Laplacian) that happen to exist elsewhere but need custom implementation for our API.
