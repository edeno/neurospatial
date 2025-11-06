# Primitive Operations for neurospatial - Executive Summary

## The Problem

Users need to perform complex spatial analyses (behavioral, replay, RL), but must either:
1. **Implement from scratch** - error-prone, inconsistent across labs
2. **Use domain-specific libraries** - opexebo (grids only), traja (no graphs), pynapple (no spatial structure)
3. **Compose from numpy/scipy** - **fails on irregular graphs**, breaks at walls/barriers

**Key insight**: Existing tools assume regular grids or don't respect spatial connectivity.

---

## The Solution: Three Core Primitives

### 🔴 **1. `neighbor_reduce` - THE FUNDAMENTAL PRIMITIVE**

```python
result = neighbor_reduce(field, env, op='mean')
```

**What it does**: Aggregate field values over graph neighborhoods

**Why fundamental**: ALL local spatial operations are built from this:
- **Smoothing** = `neighbor_reduce(field, env, op='mean')`
- **Laplacian** = `field - neighbor_reduce(field, env, op='mean')`
- **Bellman backup** = `neighbor_reduce(values, env, op='sum', weights=T)`
- **Peak detection** = `field > neighbor_reduce(field, env, op='max')`

**Proof-of-concept results**:
```
Original field (5x5 grid):
[[ 0.  1.  2.  3.  4.]
 [ 5.  6.  7.  8.  9.]
 [10. 11. 12. 13. 14.]
 [15. 16. 17. 18. 19.]
 [20. 21. 22. 23. 24.]]

After neighbor_reduce (mean smoothing):
[[ 3.0   2.67  3.67  4.67  6.0 ]
 [ 5.33  6.0   7.0   8.0   8.67]
 [10.33 11.0  12.0  13.0  13.67]
 [15.33 16.0  17.0  18.0  18.67]
 [18.0  19.33 20.33 21.33 21.0 ]]
```

---

### 🟠 **2. `accumulate_along_path` - TRAJECTORY PRIMITIVE**

```python
returns = accumulate_along_path(
    rewards, trajectory,
    op='sum', discount=0.95, reverse=True
)
```

**What it does**: Accumulate values along bin sequences with optional decay

**Why fundamental**: ALL trajectory operations:
- **Discounted returns** (RL) = backward accumulation with discount
- **Path integrals** = forward accumulation
- **Trajectory likelihood** = accumulation of log-probabilities
- **Running max/min** = max/min accumulation

**Proof-of-concept results**:
```
Trajectory: [0, 6, 12, 18, 24]  # Diagonal path
Rewards: [0, 0, 0, 0, 10.0]      # Goal at end

Discounted returns (γ=0.9):
  Step 0: return = 6.56  (0.9^4 * 10)
  Step 1: return = 7.29  (0.9^3 * 10)
  Step 2: return = 8.10  (0.9^2 * 10)
  Step 3: return = 9.00  (0.9^1 * 10)
  Step 4: return = 10.00 (0.9^0 * 10)
```

---

### 🟡 **3. `propagate` - VALUE PROPAGATION**

```python
value_field = propagate(
    sources=goal_bin,
    initial_values=1.0,
    env=env,
    decay=0.95,
    max_steps=100
)
```

**What it does**: Spread values from sources through graph with decay

**Why fundamental**: ALL value propagation:
- **Value iteration** (RL) = iterative propagate
- **Successor representation** = propagate from each bin
- **Distance fields** = propagate with no decay
- **Diffusion** = propagate with exponential decay

**Proof-of-concept results**:
```
Value field (propagated from bottom-right corner with decay=0.8):
[[0.168  0.210  0.262  0.328  0.410]
 [0.210  0.262  0.328  0.410  0.512]
 [0.262  0.328  0.410  0.512  0.640]
 [0.328  0.410  0.512  0.640  0.800]
 [0.410  0.512  0.640  0.800  1.000]]

Gradient naturally emerges from connectivity structure!
```

---

## Composition: Complex Operations from Simple Primitives

### **Value Iteration** (RL algorithm)

```python
def bellman_backup(values, rewards, env, gamma=0.95):
    """One line using neighbor_reduce!"""
    return rewards + gamma * neighbor_reduce(values, env, op='mean')

# Run value iteration
V = np.zeros(25)
for i in range(20):
    V = bellman_backup(V, rewards, env, gamma=0.95)
```

**Result after 15 iterations**:
```
[[0.66   0.85   1.07   1.55   1.84 ]
 [0.85   0.95   1.47   2.00   2.69 ]
 [1.07   1.47   2.17   3.47   4.65 ]
 [1.55   2.00   3.47   5.80   9.16 ]
 [1.84   2.69   4.65   9.16  18.49 ]]

Perfect value gradient toward goal at (4,4)!
```

### **Successor Representation** (predictive representation)

```python
def successor_representation_column(env, source_bin, gamma=0.95):
    """One line using propagate!"""
    return propagate(source_bin, 1.0, env, decay=gamma, max_steps=None)
```

**Result for center bin** (expected discounted occupancy):
```
[[0.656  0.729  0.810  0.729  0.656]
 [0.729  0.810  0.900  0.810  0.729]
 [0.810  0.900  1.000  0.900  0.810]  ← center
 [0.729  0.810  0.900  0.810  0.729]
 [0.656  0.729  0.810  0.729  0.656]]

Radial decay from center, respecting graph structure!
```

---

## Why These Are "True Primitives"

### **1. Cannot be decomposed further without losing graph structure**

- ❌ `np.mean(field.reshape(5,5), axis=0)` - assumes regular grid
- ❌ `scipy.ndimage.convolve(field, kernel)` - assumes grid, ignores walls
- ✅ `neighbor_reduce(field, env, op='mean')` - works on ANY graph

### **2. Everything builds from these**

**From the 3 primitives** → **100+ operations**:

| Primitive | Enables |
|-----------|---------|
| `neighbor_reduce` | Smoothing, Laplacian, Bellman, local stats, max pooling, peak detection |
| `accumulate_along_path` | Returns, path cost, likelihood, compression, running stats |
| `propagate` | Value iteration, SR, diffusion, distance fields, influence maps |

**Composing 2 primitives** → **Higher-level algorithms**:
- `propagate` + `neighbor_reduce` = Policy evaluation
- `accumulate` + `neighbor_reduce` = TD learning
- All 3 together = Complete RL suite

### **3. Graph-aware by design**

Works on:
- ✅ Regular grids
- ✅ Hexagonal tessellations
- ✅ 1D linearized tracks
- ✅ Masked/irregular regions
- ✅ Arbitrary connectivity graphs

---

## What This Enables

### **Behavioral Analysis**
```python
# Trajectory smoothness
distances = pairwise_sequence_distances(trajectory, env)
smooth_score = 1.0 / np.mean(distances)

# Path efficiency
total_distance = accumulate_along_path(distances, trajectory, op='sum')[-1]
straight_distance = euclidean(start, end)
efficiency = straight_distance / total_distance
```

### **Replay Analysis**
```python
# Temporal compression
decoded_spatial_dist = pairwise_sequence_distances(decoded, env).sum()
actual_temporal_dist = time_window_duration
compression_ratio = decoded_spatial_dist / actual_temporal_dist  # ~10-20x

# Direction detection
forward_corr = sequence_correlation(decoded, forward_trajectory, env)
reverse_corr = sequence_correlation(decoded, reverse_trajectory, env)
is_reverse_replay = reverse_corr > forward_corr
```

### **RL Analysis**
```python
# Value function learning
V = np.zeros(n_bins)
for episode in episodes:
    V = bellman_backup(V, rewards, env, gamma=0.95)

# Advantage computation
advantages = Q_values - neighbor_reduce(V, env, op='mean', weights=policy)

# Successor representation
SR = np.column_stack([
    propagate(i, 1.0, env, decay=gamma)
    for i in range(n_bins)
])
predicted_values = SR @ rewards  # Instant value estimates!
```

---

## Comparison to Existing Tools

| Feature | neurospatial primitives | opexebo | traja | Minigrid |
|---------|------------------------|---------|-------|----------|
| Graph-aware operations | ✅ | ❌ | ❌ | ❌ |
| Irregular layouts | ✅ | ❌ | ❌ | ❌ |
| RL primitives | ✅ | ❌ | ❌ | ⚠️ (toy only) |
| Trajectory operations | ✅ | ⚠️ (limited) | ✅ | ❌ |
| Behavioral real data | ✅ | ✅ | ✅ | ❌ |
| Replay analysis | ✅ | ❌ | ❌ | ❌ |
| Works on all neurospatial layouts | ✅ | ❌ | ❌ | ❌ |

**Key differentiator**: Only neurospatial provides **graph-aware spatial primitives** that work across all layout types.

---

## Implementation Roadmap

### **Phase 1: Core Primitives** (Week 1)
- `neighbor_reduce` in `src/neurospatial/primitives.py`
- `accumulate_along_path` in `src/neurospatial/primitives.py`
- `propagate` in `src/neurospatial/primitives.py`
- Comprehensive tests
- Add to public API

### **Phase 2: Sequence Operations** (Week 2)
- `pairwise_sequence_distances` in `src/neurospatial/sequence_ops.py`
- `sequence_correlation` in `src/neurospatial/sequence_ops.py`
- `masked_reduce` in `src/neurospatial/primitives.py`

### **Phase 3: Higher-Level Functions** (Week 3)
- `bellman_backup` - built from `neighbor_reduce`
- `successor_representation` - built from `propagate`
- `local_statistic` - multi-hop `neighbor_reduce`

### **Phase 4: Integration & Docs** (Week 4)
- Integration with existing `Environment` methods
- Performance optimization (sparse matrices, caching)
- Tutorial notebooks
- API documentation

---

## Why This Matters

**Current state**: Every lab reimplements these operations
- ❌ Bugs in spatial calculations
- ❌ Inconsistent across papers
- ❌ Doesn't work with irregular environments
- ❌ Wasted effort

**With primitives**: One correct implementation
- ✅ Graph-aware by design
- ✅ Works on any neurospatial layout
- ✅ Composable into complex operations
- ✅ Tested, documented, maintained

**Bottom line**: These primitives are the **missing foundation** for behavioral/replay/RL analyses in neuroscience. They turn neurospatial from a discretization library into a complete spatial analysis framework.

---

## See Full Proposals

- **[PRIMITIVES_PROPOSAL.md](PRIMITIVES_PROPOSAL.md)** - Complete API design with all primitives
- **[primitives_poc.py](primitives_poc.py)** - Working proof-of-concept code
- **[PLAN.md](PLAN.md)** - Existing plan for trajectory/topology operations

## Next Steps

1. Review this proposal
2. Prioritize primitives (recommend starting with the 3 core ones)
3. Implement Phase 1
4. Iterate based on user feedback
