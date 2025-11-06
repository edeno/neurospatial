# Missing Fundamental Spatial Operators

## Viewing the Graph as a Discrete Spatial Manifold

Current primitives operate on **node fields** (values at bins). But we're missing fundamental differential/integral operators from vector calculus.

---

## 🔴 Tier 1: Essential Missing Operators

### **1. Gradient (Scalar Field → Vector Field)**

```python
grad_vectors = gradient(
    field,  # Scalar field on nodes
    env,
    method='central'  # or 'forward', 'backward'
)
# Returns: (n_bins, n_dims) - vector at each bin

# Or just magnitude:
grad_mag = gradient_magnitude(field, env)
# Returns: (n_bins,) - magnitude of gradient
```

**Why fundamental:**
- **Direction of steepest ascent** - policy gradients, navigation
- **Field boundaries** - where does place field end?
- **Flow fields** - direction of value increase
- Cannot use `np.gradient` on irregular graphs

**Mathematical definition (discrete):**
```
∇f[i] = Σⱼ∈N(i) (f[j] - f[i]) * edge_vector[i,j] / |edge|²
```

**Use cases:**
- Place field boundary detection
- Policy gradient estimation
- Flow/trajectory prediction
- Detecting field structure

---

### **2. Spatial Autocorrelation**

```python
autocorr = spatial_autocorrelation(
    field,
    env,
    max_lag=20,  # bins or distance
    method='pearson'  # or 'spearman'
)
# Returns: (2*max_lag+1, 2*max_lag+1) for 2D
#          Grid of correlations at different spatial lags
```

**Why ESSENTIAL:**
- **Grid cell analysis** - THE most requested feature
- Hexagonal symmetry detection
- Periodic structure
- Field regularity

**This is THE missing primitive for grid cell research!**

**Use cases:**
- Grid score computation
- Detecting periodic firing
- Field spacing analysis
- Spatial structure quantification

---

### **3. Spatial Cross-Correlation**

```python
xcorr = spatial_cross_correlation(
    field1,
    field2,
    env,
    max_lag=10
)
# Returns: correlation at different spatial offsets
```

**Why fundamental:**
- Field similarity at different alignments
- Template matching
- Multi-session comparison
- Remapping detection

**Cannot be done with scipy on graphs!**

---

### **4. Convolve (Kernel-Based Filtering)**

```python
convolved = convolve(
    field,
    kernel,  # Custom spatial kernel
    env,
    mode='same'  # or 'valid', 'full'
)
```

**Different from `neighbor_reduce`:**
- `neighbor_reduce`: applies same operation to all neighborhoods
- `convolve`: applies **custom kernel** that can vary by position/distance
- Kernels can be learned, asymmetric, multi-scale

**Why fundamental:**
- Feature extraction (like CNNs but on graphs)
- Multi-scale analysis
- Edge detection kernels
- Custom filtering

**Use cases:**
- Gabor filters on place fields
- Multi-resolution analysis
- Feature learning
- Graph neural network operations

---

## 🟡 Tier 2: Very Useful Operators

### **5. Divergence (Vector Field → Scalar Field)**

```python
div = divergence(
    vector_field,  # (n_bins, n_dims) vectors at each node
    env
)
# Returns: (n_bins,) - how much field spreads from each point
```

**Why useful:**
- Sources and sinks
- Flow conservation
- Detecting convergence/divergence points
- Complementary to gradient

**Mathematical definition:**
```
∇·F[i] = Σⱼ∈N(i) (F[j] - F[i]) · edge_vector[i,j]
```

---

### **6. Explicit Laplacian**

```python
laplacian_field = laplacian(field, env, normalized=False)
```

**Currently:** `field - neighbor_reduce(field, env, op='mean')`
**Better:** Explicit operator with normalization options

**Why useful:**
- Smoothness measure
- Diffusion operator
- Regularization
- Heat equation

---

### **7. Integrate (Proper Weighted Integration)**

```python
total = integrate(
    field,
    env,
    region=None,  # All bins or specific region
    method='trapezoid'  # or 'simpson'
)
```

**Different from `masked_reduce`:**
- Properly accounts for **bin sizes** (areas/volumes)
- Numerical integration methods
- Boundary handling

**Why important:**
- Normalizing probability distributions
- Total reward/cost
- Energy functionals
- Proper normalization

---

### **8. Edge Field Operations**

```python
# Create edge field from node field (gradient)
edge_field = node_to_edge(field, env, op='difference')

# Reduce edge field back to nodes
node_field = edge_to_node(edge_field, env, op='mean')

# Operate on edge field directly
edge_smoothed = edge_reduce(edge_field, env, op='mean')
```

**Why useful:**
- Many quantities live on edges (flow, differences)
- Transition analysis
- Flow conservation
- Edge-based learning

---

## 🟢 Tier 3: Advanced/Specialized Operators

### **9. Spectral Decomposition**

```python
eigenvalues, eigenvectors = spectral_decompose(
    env,
    n_components=10,
    normalized=True
)
# Get graph Laplacian eigenmodes
```

**Why powerful:**
- Graph Fourier transform
- Spectral clustering
- Low-dimensional representations
- Diffusion maps

---

### **10. Circulation (Loop Integrals)**

```python
circ = circulation(
    vector_field,
    loop_path,  # Closed path of bins
    env
)
```

**Why useful:**
- Conservative vs non-conservative fields
- Rotational structure
- Topological features

---

### **11. Flux (Boundary Integrals)**

```python
flux = compute_flux(
    vector_field,
    boundary_bins,  # Set of boundary bins
    env,
    outward=True
)
```

**Why useful:**
- Flow through boundaries
- Divergence theorem verification
- Barrier crossing analysis

---

## 🎯 Comparison: Current vs Needed

### **What we have:**

| Primitive | Type | Scope |
|-----------|------|-------|
| `neighbor_reduce` | Node → Node | Local aggregation |
| `accumulate_along_path` | Node sequence → Scalar | Path integral |
| `propagate` | Node → Node | Global diffusion |

### **What's missing:**

| Operator | Type | Why Essential |
|----------|------|---------------|
| **`gradient`** | Node → Edge/Vector | Differential operator |
| **`spatial_autocorrelation`** | Node → Grid | Grid cell analysis |
| **`convolve`** | Node → Node | Kernel filtering |
| **`divergence`** | Edge/Vector → Node | Dual of gradient |
| **`integrate`** | Node → Scalar | Weighted sum |

---

## 📐 Mathematical Structure

Think of it as **discrete differential geometry**:

### **Node Fields** (what we have):
- Scalar field: `f: bins → ℝ`
- Operations: `neighbor_reduce`, `propagate`, `accumulate_along_path`

### **Edge Fields** (missing):
- Vector field: `F: bins → ℝⁿ`
- Flow field: `flow: edges → ℝ`
- Operations: `gradient`, `divergence`, `circulation`

### **Operators Connecting Them:**

```
Node field (scalar)  -[gradient]→  Edge field (vector)
     ↑                                     |
     |                                     |
[Laplacian]                         [divergence]
     |                                     |
     ↓                                     ↓
Node field (scalar) ←[neighbor_reduce]- Node field (scalar)
```

This is the **fundamental exact sequence** of discrete calculus!

---

## 🎓 Priority Ranking

### **Implement ASAP** (completes the primitive set):

1. **`gradient`** - fundamental differential operator
2. **`spatial_autocorrelation`** - essential for grid cells
3. **`convolve`** - kernel-based filtering
4. **`integrate`** - proper weighted integration

These 4 + the existing 3 primitives = **complete spatial operator toolkit**

### **High value** (extend capabilities):

5. **`divergence`** - dual operator to gradient
6. **`spatial_cross_correlation`** - field comparison
7. **`laplacian`** (explicit) - currently implicit
8. **`edge_to_node` / `node_to_edge`** - type conversion

### **Advanced** (specialized use cases):

9. **`spectral_decompose`** - graph spectrum
10. **`circulation`** - topological
11. **`flux`** - boundary integrals

---

## 💡 Key Insights

### **1. Two Primitive Types:**

**Aggregation primitives** (have):
- `neighbor_reduce` - local aggregation
- `propagate` - global diffusion
- `accumulate_along_path` - path integrals

**Differential primitives** (missing):
- `gradient` - rate of change
- `divergence` - spreading
- `curl` (2D) - rotation
- `laplacian` - curvature

### **2. Field Types:**

**Scalar fields** (current focus):
- Place fields, value functions, occupancy

**Vector fields** (missing):
- Gradients, flows, velocities
- Policy directions, heading
- Displacement fields

### **3. Graph Structure:**

**Nodes** (bins) - where we have primitives
**Edges** - where gradients/flows live (missing!)
**Faces** (dual graph) - advanced topology

---

## 🔨 Proposed Implementation

### **Phase 1: Differential Operators** (Week 1)
- [ ] `gradient(field, env)` → vector field
- [ ] `gradient_magnitude(field, env)` → scalar field
- [ ] `divergence(vector_field, env)` → scalar field
- [ ] `laplacian(field, env)` → scalar field (explicit)

### **Phase 2: Correlation Operators** (Week 2)
- [ ] `spatial_autocorrelation(field, env, max_lag)`
- [ ] `spatial_cross_correlation(field1, field2, env, max_lag)`
- [ ] Tests with grid cell-like fields

### **Phase 3: Kernel Operators** (Week 3)
- [ ] `convolve(field, kernel, env)`
- [ ] `integrate(field, env, region)`
- [ ] Edge field support: `node_to_edge`, `edge_to_node`

### **Phase 4: Advanced** (Week 4+)
- [ ] `spectral_decompose(env, n_components)`
- [ ] `circulation(vector_field, loop, env)`
- [ ] Performance optimization

---

## 📊 What This Unlocks

### **With these operators:**

```python
# Grid cell analysis
autocorr = spatial_autocorrelation(firing_rate, env, max_lag=20)
grid_score = compute_hexagonal_score(autocorr)

# Place field boundaries
grad = gradient(firing_rate, env)
boundary = gradient_magnitude(firing_rate, env) > threshold

# Multi-scale filtering
scales = [1.0, 2.0, 4.0]
filtered = [convolve(field, gaussian_kernel(s), env) for s in scales]

# Proper normalization
posterior = firing_rate * prior
posterior = posterior / integrate(posterior, env)  # Sums to 1!

# Policy gradient
value_grad = gradient(V, env)
policy = softmax_direction(value_grad)

# Flow conservation
flow_field = compute_flow(trajectory, env)
div = divergence(flow_field, env)
# div should be 0 everywhere except sources/sinks

# Spectral analysis
eigenvals, eigenvecs = spectral_decompose(env, n_components=10)
low_freq_component = eigenvecs[:, 0]  # Smoothest mode
```

---

## 🎯 Bottom Line

**Current primitives are aggregation-focused.**
**Missing: differential/correlation operators.**

Adding these completes the toolkit:
- ✅ Aggregation: `neighbor_reduce`, `propagate`, `accumulate`
- ➕ Differential: `gradient`, `divergence`, `laplacian`
- ➕ Correlation: `spatial_autocorrelation`, `convolve`
- ➕ Integration: `integrate`

This transforms neurospatial from "spatial discretization + aggregation" to **"complete discrete differential geometry toolkit"** for neuroscience.
