# Spatial Operators: Existing vs Proposed

## Current State of neurospatial Operations

### ✅ **What We Have** (Strong Foundation)

#### **1. Field Operations** (`field_ops.py`)
```python
from neurospatial import normalize_field, clamp, combine_fields, divergence

# Statistical divergence (probability distributions)
divergence(p, q, kind='kl')    # KL, JS, cosine divergence
normalize_field(field)          # Sum to 1
clamp(field, lo, hi)            # Clip values
combine_fields([f1, f2], mode='mean')  # Merge fields
```

**Type**: Statistical/algebraic operations on scalar fields
**Note**: `divergence` is **KL/JS divergence**, NOT vector field divergence

#### **2. Smoothing Operations** (`kernels.py`, `environment/fields.py`)
```python
from neurospatial import compute_diffusion_kernels

# Graph Laplacian smoothing
kernel = compute_diffusion_kernels(graph, bandwidth_sigma=5.0, mode='transition')
smoothed = env.smooth(field, bandwidth=5.0)  # Uses diffusion kernels internally
```

**Type**: Diffusion-based smoothing (graph Laplacian)
**What it does**: `exp(-t * L)` where `L = D - W` is graph Laplacian

#### **3. Distance Operations** (`distance.py`, `environment/queries.py`)
```python
from neurospatial import distance_field, pairwise_distances

# Multi-source distance field
distances = distance_field(G, sources=[goal_bin])

# Pairwise distances
dist_matrix = pairwise_distances(G, [bin1, bin2, bin3])

# Environment methods
dist = env.distance_between(point1, point2, edge_weight='distance')
path = env.shortest_path(source_bin, target_bin)
distances = env.distance_to(target_bins, metric='geodesic')
reachable = env.reachable_from(source_bin, max_distance=20.0)
```

**Type**: Geodesic distances on graphs

#### **4. Graph Queries** (`environment/queries.py`)
```python
# Neighborhood
neighbors = env.neighbors(bin_idx)  # Direct neighbors

# Rings (BFS layers)
layers = env.rings(center_bin, hops=5)  # k-hop neighborhoods

# Connectivity
components = env.components()  # Connected components
```

**Type**: Graph traversal and topology

#### **5. Trajectory Operations** (`environment/trajectory.py`)
```python
# Occupancy
occ = env.occupancy(times, positions, kernel_bandwidth=5.0)

# Bin sequence
seq = env.bin_sequence(times, positions, dedup=True)

# Transition matrix
T = env.transitions(times=times, positions=positions, normalize=True)
# OR model-based:
T = env.transitions(method='diffusion', bandwidth=10.0)
```

**Type**: Trajectory analysis, Markov transitions

#### **6. Interpolation** (`environment/fields.py`)
```python
# Evaluate field at arbitrary points
values = env.interpolate(field, query_points, mode='linear')
```

**Type**: Point queries from field

---

## 🔴 **What's Missing** (Gaps in the Toolkit)

### **Gap 1: Differential Operators**

| Operator | Status | What's Missing |
|----------|--------|----------------|
| **Gradient** | ❌ Missing | Scalar field → Vector field |
| **Divergence** (vector) | ❌ Missing | Vector field → Scalar field |
| **Laplacian** (explicit) | ⚠️ Implicit | `field - smooth(field)` |
| **Curl** | ❌ Missing | 2D rotation field |

**Current workaround:**
```python
# Laplacian (implicit via smoothing)
smoothed = env.smooth(field, bandwidth=5.0)
laplacian = field - smoothed  # Approximate Laplacian

# Gradient - NO WORKAROUND! Cannot compute on graphs
```

**Problem**: You **cannot** compute gradients properly on irregular graphs without dedicated operator.

---

### **Gap 2: Correlation Operators**

| Operator | Status | What's Missing |
|----------|--------|----------------|
| **Spatial autocorrelation** | ❌ Missing | Field correlation at spatial lags |
| **Spatial cross-correlation** | ❌ Missing | Correlation between two fields |
| **Convolve** (custom kernels) | ⚠️ Partial | Only Gaussian via `smooth()` |

**Current workaround:**
```python
# Grid cell autocorrelation - NO GOOD WORKAROUND!
# Users manually implement on regular grids only
if hasattr(env.layout, 'grid_shape'):
    field_grid = field.reshape(env.layout.grid_shape)
    autocorr = scipy.signal.correlate2d(field_grid, field_grid)
    # BUT: breaks on irregular layouts, doesn't respect graph structure
```

**Problem**: **Cannot do grid cell analysis on irregular environments!**

---

### **Gap 3: Integration with Bin Sizes**

| Operator | Status | What's Missing |
|----------|--------|----------------|
| **integrate** | ⚠️ Partial | No weighted integration |
| **masked_reduce** | ❌ Missing | Reduce over masked regions |

**Current workaround:**
```python
# Normalize probability (WRONG on irregular layouts!)
posterior = posterior / posterior.sum()  # Ignores varying bin sizes!

# Should be:
# posterior = posterior / integrate(posterior, env)  # Weighted by bin_sizes
```

**Problem**: Incorrect normalization when bin sizes vary (hexagonal, masked, triangular layouts).

---

### **Gap 4: Aggregation Primitives**

| Operator | Status | What's Missing |
|----------|--------|----------------|
| **neighbor_reduce** | ❌ Missing | Generic neighborhood aggregation |
| **accumulate_along_path** | ❌ Missing | Path integrals with decay |
| **propagate** | ⚠️ Partial | `distance_field` is special case |

**Current state:**
- Have `distance_field` (specific case of propagate with no decay)
- Have `smooth` (specific case of neighbor_reduce with Gaussian weights)
- **Missing**: Generic primitives that compose into these

**Problem**: Users cannot build custom operations. Everything is hard-coded.

---

### **Gap 5: Edge Field Operations**

| Operator | Status | What's Missing |
|----------|--------|----------------|
| **node_to_edge** | ❌ Missing | Convert node field to edge field |
| **edge_to_node** | ❌ Missing | Convert edge field to node field |
| **edge_reduce** | ❌ Missing | Aggregate over edge neighborhoods |

**Problem**: Many quantities naturally live on edges (flow, differences), but no support.

---

## 📊 **Comparison: Current vs Complete**

### **Current Capabilities Matrix**

| Operation Type | What We Have | What's Missing |
|----------------|--------------|----------------|
| **Scalar field algebra** | ✅ normalize, clamp, combine | ⚠️ masked operations |
| **Statistical divergence** | ✅ KL, JS, cosine | - |
| **Smoothing** | ✅ Gaussian (Laplacian-based) | ⚠️ Custom kernels |
| **Distance** | ✅ Geodesic, Euclidean | - |
| **Graph queries** | ✅ neighbors, rings, components | ⚠️ Generic aggregation |
| **Trajectory** | ✅ occupancy, transitions | ⚠️ Path accumulation |
| **Interpolation** | ✅ nearest, linear | - |
| **Differential** | ❌ | **gradient, div, curl, Laplacian** |
| **Correlation** | ❌ | **autocorr, xcorr, convolve** |
| **Integration** | ❌ | **integrate (weighted)** |
| **Primitives** | ⚠️ Partial | **neighbor_reduce, accumulate, propagate** |

### **What Current Operations Build From**

```
Current operations are SPECIFIC implementations:
├─ smooth() = neighbor_reduce(field, op='mean', weights=gaussian)
├─ distance_field() = propagate(sources, 1.0, decay=1.0)
├─ neighbors() = graph.neighbors() [NetworkX]
└─ rings() = BFS layers [custom implementation]

Missing: GENERIC primitives that users can compose
```

---

## 🎯 **Proposed Organization**

### **Tier 1: Core Primitives** (Fundamental, everything builds from these)

```python
# AGGREGATION PRIMITIVES (new module: primitives.py)
neighbor_reduce(field, env, op='mean')  # Generic neighborhood aggregation
accumulate_along_path(field, path, discount=γ)  # Path integrals
propagate(sources, values, env, decay=γ)  # Value diffusion

# DIFFERENTIAL OPERATORS (new module: differential.py)
gradient(field, env)  # ∇f: scalar → vector
divergence_vector(vector_field, env)  # ∇·F: vector → scalar
laplacian(field, env)  # ∇²f: scalar → scalar

# CORRELATION OPERATORS (new module: correlation.py)
spatial_autocorrelation(field, env, max_lag)
spatial_cross_correlation(field1, field2, env, max_lag)
convolve(field, kernel, env)

# INTEGRATION (extend primitives.py)
integrate(field, env, region=None)  # Weighted integration
```

### **Tier 2: Extended Operations** (Built from primitives)

```python
# FIELD OPERATIONS (extend field_ops.py)
masked_reduce(field, mask, op='sum')
apply_field_mask(field, mask, fill_value=0.0)
piecewise_field(regions, values, overlap_policy='priority')

# EDGE OPERATIONS (new module: edge_ops.py)
node_to_edge(field, env, op='difference')
edge_to_node(edge_field, env, op='mean')
edge_reduce(edge_field, env, op='mean')
```

### **Tier 3: High-Level Algorithms** (Built from Tier 1+2)

```python
# RL OPERATIONS (new module: rl_ops.py)
bellman_backup(values, rewards, env, gamma=0.95)  # Uses neighbor_reduce
successor_representation(env, gamma=0.95)  # Uses propagate
advantage(Q_values, V_values, env)  # Uses neighbor_reduce

# FIELD ANALYSIS (new module: field_analysis.py)
detect_peaks(field, env, min_prominence)  # Uses gradient
detect_boundaries(field, env, threshold)  # Uses gradient_magnitude
grid_score(firing_rate, env)  # Uses spatial_autocorrelation
```

---

## 🔄 **How Proposed Operators Complement Existing**

### **Example 1: Smoothing**

**Current** (specific implementation):
```python
smoothed = env.smooth(field, bandwidth=5.0, mode='transition')
```

**With primitives** (composable):
```python
# Same result, but now user can customize:
smoothed = neighbor_reduce(field, env, op='mean', weights=gaussian_weights)

# Or custom kernel:
smoothed = convolve(field, custom_kernel, env)

# Or multi-scale:
scales = [2.0, 5.0, 10.0]
smoothed_multi = [neighbor_reduce(field, env, weights=kernel(s)) for s in scales]
```

**Benefit**: Flexibility without breaking existing API.

---

### **Example 2: Distance Field**

**Current** (specific implementation):
```python
distances = distance_field(G, sources=[goal_bin])
```

**With primitives** (extends capability):
```python
# Same result:
distances = propagate(sources=[goal_bin], initial_values=0.0, env=env, decay=1.0)

# But now can do discounted distances:
discounted_distances = propagate(goal_bin, 1.0, env, decay=0.95)

# Or multi-source with different values:
values = propagate([goal1, goal2], [10.0, 5.0], env, decay=0.9)
```

**Benefit**: Generalizes existing operation.

---

### **Example 3: Gradient (NEW capability)**

**Current** (NO WORKAROUND):
```python
# Cannot compute gradient on irregular graphs!
```

**With differential operators**:
```python
# Now possible:
grad = gradient(firing_rate, env)  # Returns (n_bins, n_dims) vectors

# Place field boundaries:
grad_mag = gradient_magnitude(firing_rate, env)
boundary_bins = grad_mag > threshold

# Policy from value function:
value_grad = gradient(V, env)
policy = softmax_direction(value_grad)
```

**Benefit**: Enables entirely new analyses.

---

### **Example 4: Grid Cell Analysis (NEW capability)**

**Current** (IMPOSSIBLE on irregular layouts):
```python
# Only works on regular grids:
if hasattr(env.layout, 'grid_shape'):
    field_grid = field.reshape(env.layout.grid_shape)
    autocorr = scipy.signal.correlate2d(field_grid, field_grid)
```

**With correlation operators**:
```python
# Works on ANY layout:
autocorr = spatial_autocorrelation(firing_rate, env, max_lag=20)
grid_score = compute_hexagonal_score(autocorr)

# Hexagonal, triangular, masked - all work!
```

**Benefit**: Grid cell analysis on any environment type.

---

## 📈 **Migration Path**

### **Phase 1: Add Primitives (No Breaking Changes)**

```python
# New modules, existing code unchanged
from neurospatial.primitives import neighbor_reduce, accumulate_along_path, propagate
from neurospatial.differential import gradient, divergence_vector, laplacian
from neurospatial.correlation import spatial_autocorrelation, convolve
```

- Existing functions (`smooth`, `distance_field`) work as before
- New primitives available for power users
- Documentation shows how existing ops relate to primitives

### **Phase 2: Extend Existing Modules**

```python
# Add to field_ops.py
from neurospatial import masked_reduce, integrate

# Add to Environment methods
grad_mag = env.gradient_magnitude(field)
autocorr = env.spatial_autocorrelation(field, max_lag=20)
```

- Convenience methods on Environment class
- Still uses primitives under the hood

### **Phase 3: High-Level Algorithms**

```python
# New rl_ops.py
from neurospatial.rl_ops import bellman_backup, successor_representation

# New field_analysis.py
from neurospatial.field_analysis import detect_peaks, grid_score
```

- Built from primitives
- Domain-specific utilities

---

## 🎓 **Design Philosophy**

### **Current State: Specific Implementations**
```
User Request → Specific Function
(cannot customize)
```

### **Proposed State: Composable Primitives**
```
User Request → Primitives → Compose → Custom Solution
(flexible, extensible)
```

**Example**:
```python
# Current: User wants custom smoothing → stuck
smoothed = env.smooth(field, bandwidth=5.0)  # Only Gaussian

# Proposed: User wants custom smoothing → compose primitives
my_kernel = create_custom_kernel(...)
smoothed = neighbor_reduce(field, env, weights=my_kernel)  # Any kernel!
```

---

## 📊 **Summary**

### **What We Have (Strong!)**
✅ Field algebra (normalize, clamp, combine)
✅ Statistical divergence (KL, JS, cosine)
✅ Diffusion smoothing (graph Laplacian)
✅ Distance operations (geodesic, Euclidean)
✅ Graph queries (neighbors, rings, components)
✅ Trajectory analysis (occupancy, transitions)
✅ Interpolation (nearest, linear)

### **Critical Gaps**
❌ **Differential operators** (gradient, vector divergence, curl)
❌ **Correlation operators** (autocorr, xcorr, custom kernels)
❌ **Generic primitives** (neighbor_reduce, accumulate, propagate)
❌ **Weighted integration** (proper normalization)
❌ **Edge field support** (many quantities live on edges)

### **Impact**
Current: **Spatial discretization + specific operations**
With primitives: **Complete spatial analysis toolkit**

**Use cases unlocked:**
- Grid cell analysis on ANY layout
- Place field boundary detection
- Policy gradients (RL)
- Custom smoothing kernels
- Multi-scale analysis
- Proper probability normalization
- Flow field analysis
- And many more...

---

## 🔑 **Bottom Line**

**neurospatial currently has:**
- Excellent foundation for spatial discretization
- Strong specific implementations (smooth, distance_field, etc.)
- Good graph-based operations

**Adding proposed operators:**
- **Complements** (doesn't replace) existing functionality
- **Generalizes** specific ops into composable primitives
- **Enables** new analyses (grid cells, gradients, custom kernels)
- **Maintains** backward compatibility

**Result**: Transforms neurospatial from "good spatial discretization tool" into **"complete discrete differential geometry toolkit for computational neuroscience"**.
