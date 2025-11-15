# Reward Fields in neurospatial: Implementation and Applications

## Overview

Reward fields in neurospatial are **spatial value functions** that assign scalar rewards to each location (bin) in the environment. They're essential for:

1. **Reinforcement Learning (RL)** - Defining task structure and reward shaping
2. **Neuroscience Analysis** - Modeling value representations in brain circuits
3. **Path Planning** - Creating cost/reward landscapes for navigation
4. **Experimental Design** - Defining goal zones and reward contingencies

neurospatial provides **two complementary functions** for creating reward fields:

- **`region_reward_field()`** - Region-based rewards (areas, zones)
- **`goal_reward_field()`** - Point-based rewards (specific locations)

---

## 1. Region-Based Rewards: `region_reward_field()`

### What It Does

Creates reward fields from **spatial regions** (areas) defined in your environment. Perfect for:
- Goal zones in behavioral tasks
- Reward wells in T-mazes
- Safe zones vs. danger zones
- Territory boundaries

### Algorithm

**File**: `src/neurospatial/reward.py:23-201`

```python
def region_reward_field(
    env,
    region_name,
    reward_value=1.0,
    decay="constant",
    bandwidth=None
)
```

**Implementation by decay type**:

#### Decay Type 1: `"constant"` (Binary Reward)

**Simplest case - sparse reward:**

```python
# Algorithm
region_mask = regions_to_mask(env, region_name)  # Boolean array
reward = np.where(region_mask, reward_value, 0.0)
```

**Result**:
- Bins **inside** region: `reward_value`
- Bins **outside** region: `0.0`
- **Hard boundary** - no gradient

**Use case**: Classic sparse reward RL (goal-reaching without hints)

---

#### Decay Type 2: `"linear"` (Boundary Decay)

**Linear gradient from region boundary:**

```python
# Algorithm
# 1. Find boundary bins (bins in region with neighbors outside)
boundary_bins = []
for bin_id in region_bin_ids:
    neighbors = env.connectivity.neighbors(bin_id)
    if any(not region_mask[n] for n in neighbors):
        boundary_bins.append(bin_id)

# 2. Compute distance from all bins to boundary using Dijkstra
distances = distance_field(env.connectivity, sources=boundary_bins)

# 3. Inside region: full reward
reward[region_mask] = reward_value

# 4. Outside region: linear decay
max_dist = distances[~region_mask].max()
reward[~region_mask] = reward_value * max(0, 1 - distances / max_dist)
```

**Result**:
- Inside region: `reward_value` (flat)
- Outside region: `reward_value * (1 - d/d_max)`
- **Linear gradient** away from boundary
- Reaches zero at furthest point

**Mathematical formula**:
```
reward(x) = {
    reward_value                    if x in region
    reward_value * (1 - d(x)/d_max) if x outside region
}
```

**Key insight**: Distance computed via **graph geodesic** (respects walls/boundaries), not Euclidean!

**Use case**: Reward shaping with clear boundaries but guidance toward goal

---

#### Decay Type 3: `"gaussian"` (Smooth Falloff)

**Smoothest decay - Gaussian kernel smoothing:**

```python
# Algorithm
# 1. Create indicator field (1 inside, 0 outside)
indicator = np.where(region_mask, 1.0, 0.0)

# 2. Smooth using graph diffusion or Gaussian kernel
smoothed = env.smooth(indicator, bandwidth)

# 3. CRITICAL: Rescale to preserve max reward IN REGION
max_in_region = smoothed[region_mask].max()
reward = (smoothed / max_in_region) * reward_value
```

**Result**:
- Inside region: Peak at `reward_value`
- Outside region: Smooth Gaussian decay
- **No sharp edges** - smooth everywhere

**Mathematical formula** (for graph diffusion):
```
reward = (K * indicator) / max(K * indicator)

where K = exp(-t * L)  # Heat kernel on graph
      L = graph Laplacian
      t = bandwidth²/(2*n_dims)  # Diffusion time
```

**Key feature**: **Boundary-aware smoothing** via graph Laplacian (doesn't leak through walls)

**Use case**: Smooth reward shaping, potential-based RL (Ng et al., 1999)

---

### Example Usage

```python
from neurospatial import Environment, region_reward_field
import numpy as np

# Create environment
positions = np.random.randn(1000, 2) * 50
env = Environment.from_samples(positions, bin_size=5.0)
env.units = "cm"

# Add goal region (circular zone)
from shapely.geometry import Point
goal_polygon = Point(0.0, 0.0).buffer(10.0)  # 10 cm radius
env.regions.add("goal", polygon=goal_polygon)

# Binary reward (sparse)
reward_sparse = region_reward_field(
    env, "goal",
    reward_value=100.0,
    decay="constant"
)
# Result: 100 inside goal, 0 everywhere else

# Linear reward shaping
reward_linear = region_reward_field(
    env, "goal",
    reward_value=100.0,
    decay="linear"
)
# Result: 100 inside goal, decays linearly outside

# Gaussian reward shaping
reward_smooth = region_reward_field(
    env, "goal",
    reward_value=100.0,
    decay="gaussian",
    bandwidth=15.0  # Smoothing scale (cm)
)
# Result: 100 at goal center, smooth Gaussian decay
```

---

## 2. Point-Based Rewards: `goal_reward_field()`

### What It Does

Creates reward fields from **specific goal locations** (bin indices). Perfect for:
- Goal states in RL
- Target locations in navigation
- Multi-goal tasks (e.g., alternation)
- Distance-based value functions

### Algorithm

**File**: `src/neurospatial/reward.py:204-335`

```python
def goal_reward_field(
    env,
    goal_bins,
    decay="exponential",
    scale=1.0,
    max_distance=None
)
```

**Implementation by decay type**:

#### Decay Type 1: `"exponential"` (Most Common)

**Exponential distance decay:**

```python
# Algorithm
# 1. Compute distance from all bins to nearest goal (Dijkstra)
distances = distance_field(env.connectivity, sources=goal_bins)

# 2. Exponential decay
reward = scale * np.exp(-distances / scale)
```

**Mathematical formula**:
```
reward(x) = scale * exp(-d(x) / scale)

where d(x) = geodesic distance to nearest goal
```

**Properties**:
- At goal: `reward = scale`
- **Never reaches zero** (asymptotic)
- **Scale controls both peak and decay rate**
  - Larger scale → slower decay, wider influence
  - Smaller scale → faster decay, local influence

**Use case**: Standard RL reward shaping (most common in literature)

---

#### Decay Type 2: `"linear"` (Cutoff at Distance)

**Linear decay with hard cutoff:**

```python
# Algorithm
distances = distance_field(env.connectivity, sources=goal_bins)
reward = scale * np.maximum(0, 1 - distances / max_distance)
```

**Mathematical formula**:
```
reward(x) = {
    scale * (1 - d(x)/d_max)  if d(x) < d_max
    0                          if d(x) >= d_max
}
```

**Properties**:
- At goal: `reward = scale`
- At `max_distance`: `reward = 0` (exactly)
- **Constant gradient** within range
- **Zero beyond** max_distance

**Use case**: Local reward shaping with explicit horizon

---

#### Decay Type 3: `"inverse"` (Inverse Distance)

**Inverse distance function:**

```python
# Algorithm
distances = distance_field(env.connectivity, sources=goal_bins)
reward = scale / (1 + distances)
```

**Mathematical formula**:
```
reward(x) = scale / (1 + d(x))
```

**Properties**:
- At goal (d=0): `reward = scale`
- **Never reaches zero** (global influence)
- Slower decay than exponential at long distances

**Use case**: Global value functions (caution: can bias policies)

---

### Multi-Goal Support

**When `goal_bins` contains multiple goals**, distance is computed to the **nearest goal**:

```python
# Create Voronoi-like partition
distances = distance_field(env.connectivity, sources=[goal1, goal2, goal3])
# distances[i] = min(d(i, goal1), d(i, goal2), d(i, goal3))

reward = scale * np.exp(-distances / scale)
```

**Result**: Each bin influenced by its **closest goal**

---

### Example Usage

```python
from neurospatial import Environment, goal_reward_field
import numpy as np

# Create environment
positions = np.random.randn(1000, 2) * 50
env = Environment.from_samples(positions, bin_size=5.0)
env.units = "cm"

# Find goal bin at origin
goal_bin = env.bin_at(np.array([[0.0, 0.0]]))[0]

# Exponential decay (most common)
reward_exp = goal_reward_field(
    env, goal_bins=goal_bin,
    decay="exponential",
    scale=10.0  # Peak reward and decay rate
)
# At goal: 10.0, decays exponentially

# Linear with cutoff
reward_linear = goal_reward_field(
    env, goal_bins=goal_bin,
    decay="linear",
    scale=1.0,
    max_distance=50.0  # Zero beyond 50 cm
)

# Multi-goal (e.g., T-maze left/right arms)
left_bin = env.bin_at(np.array([[-30.0, 30.0]]))[0]
right_bin = env.bin_at(np.array([[30.0, 30.0]]))[0]

reward_multi = goal_reward_field(
    env, goal_bins=[left_bin, right_bin],
    decay="exponential",
    scale=15.0
)
# Each bin rewarded by nearest goal
```

---

## 3. How They're Useful

### Use Case 1: Sparse Reward RL

**Problem**: Agent must reach goal with no intermediate feedback

```python
# Binary reward - only get feedback at goal
reward = region_reward_field(env, "goal", reward_value=1.0, decay="constant")

# For RL: sparse signal, hard to learn
# Advantage: No bias from reward shaping
```

---

### Use Case 2: Reward Shaping (Ng et al., 1999)

**Problem**: Sparse rewards are hard to learn - provide gradient hints

```python
# Smooth Gaussian shaping
reward = region_reward_field(
    env, "goal",
    reward_value=100.0,
    decay="gaussian",
    bandwidth=20.0
)

# For RL: Smooth gradients guide policy
# Advantage: Faster learning
# Caution: Can bias toward suboptimal paths
```

**Best practice**: Combine sparse + shaped:

```python
# Primary sparse reward
sparse = region_reward_field(env, "goal", reward_value=100.0, decay="constant")

# Weak shaping (10% weight)
shaping = goal_reward_field(env, goal_bin, decay="exponential", scale=10.0)

# Combined
reward = sparse + 0.1 * shaping
```

---

### Use Case 3: Value Function Initialization

**Problem**: Initialize value function for RL algorithm

```python
# Distance-based value estimate
V_init = goal_reward_field(
    env, goal_bins=goal_bin,
    decay="exponential",
    scale=gamma * R_max  # Discount factor × max reward
)

# Use as starting point for value iteration/Q-learning
```

---

### Use Case 4: Multi-Goal Navigation

**Problem**: Agent must visit multiple goals (e.g., foraging)

```python
# Define all food locations
food_bins = [bin1, bin2, bin3, bin4]

# Create value field (nearest food)
food_reward = goal_reward_field(
    env, goal_bins=food_bins,
    decay="exponential",
    scale=5.0
)

# Policy: Follow gradient toward nearest food
```

---

### Use Case 5: Neuroscience - Value Coding

**Problem**: Model value representations in brain regions (OFC, striatum)

```python
# Hypothesized value field in rat brain
value_field = goal_reward_field(
    env, goal_bins=reward_port_bin,
    decay="exponential",
    scale=10.0  # Fitted to neural data
)

# Compare with actual neural firing rates
correlation = np.corrcoef(value_field, neural_rates)[0, 1]
```

---

### Use Case 6: Experimental Design

**Problem**: Define task structure for behavioral experiment

```python
# T-maze: Left arm = 1 pellet, Right arm = 3 pellets
env.regions.add("left_arm", polygon=left_polygon)
env.regions.add("right_arm", polygon=right_polygon)

reward_left = region_reward_field(env, "left_arm", reward_value=1.0)
reward_right = region_reward_field(env, "right_arm", reward_value=3.0)

# Total reward landscape
reward_total = reward_left + reward_right

# Analyze: Does animal prefer high-reward arm?
```

---

## 4. Technical Implementation Details

### Graph-Based Distance Computation

**Both functions use `distance_field()` for geodesic distances:**

```python
# From src/neurospatial/distance.py
def distance_field(connectivity, sources):
    """Compute geodesic distances using Dijkstra's algorithm."""
    # Multi-source Dijkstra via NetworkX
    distances = nx.multi_source_dijkstra_path_length(
        connectivity,
        sources=sources,
        weight='distance'  # Edge weights from graph
    )

    # Convert to array
    dist_array = np.full(n_bins, np.inf)
    for node, dist in distances.items():
        dist_array[node] = dist

    return dist_array
```

**Why geodesic (not Euclidean)?**
- **Respects environment structure** (walls, boundaries)
- **Realistic path lengths** for navigation
- **Boundary-aware** (can't shortcut through obstacles)

**Example**: In a plus maze:
- Euclidean distance from North to South arm: ~50 cm (straight line)
- Geodesic distance: ~150 cm (must go through center)

---

### Smoothing via Graph Diffusion

**Gaussian decay uses graph-based smoothing:**

```python
# From env.smooth() method
def smooth(field, bandwidth):
    # Convert bandwidth to diffusion time
    sigma_bins = bandwidth / mean_bin_size
    diffusion_time = sigma_bins**2 / (2 * n_dims)

    # Graph Laplacian
    L = nx.laplacian_matrix(connectivity)

    # Heat equation: ∂u/∂t = -L*u
    # Solution: u(t) = exp(-t*L) * u(0)
    smoothed = expm_multiply(-diffusion_time * L, field)

    return smoothed
```

**Why graph diffusion?**
- **Doesn't leak through walls** (disconnected graph nodes)
- **Adapts to environment shape** (narrow corridors vs. open spaces)
- **Mathematically principled** (solves PDE on manifold)

---

## 5. Comparison: Region vs. Goal

| Feature | `region_reward_field` | `goal_reward_field` |
|---------|----------------------|-------------------|
| **Input** | Region name (string) | Bin index/indices |
| **Shape** | Area/zone | Point(s) |
| **Decay options** | constant, linear, gaussian | exponential, linear, inverse |
| **Multi-target** | Single region | Multiple goals |
| **Use case** | Zones, territories | Specific locations |
| **Computation** | Boundary detection + distance | Direct distance field |

---

## 6. Best Practices

### 1. **Match decay to task difficulty**

```python
# Easy task (goal visible, short distance)
reward = region_reward_field(env, "goal", decay="constant")  # Sparse OK

# Medium task (longer distance)
reward = goal_reward_field(env, goal_bin, decay="exponential", scale=20.0)

# Hard task (complex maze)
reward = region_reward_field(env, "goal", decay="gaussian", bandwidth=30.0)
```

### 2. **Validate against sparse baseline**

```python
# Always compare shaped reward to sparse reward
reward_sparse = region_reward_field(env, "goal", decay="constant")
reward_shaped = region_reward_field(env, "goal", decay="gaussian", bandwidth=15.0)

# Train two agents - ensure shaped doesn't bias policy
```

### 3. **Combine sparse + shaped**

```python
# Primary sparse reward (unbiased)
R_sparse = region_reward_field(env, "goal", reward_value=100.0, decay="constant")

# Weak shaping (guidance only)
R_shape = goal_reward_field(env, goal_bin, decay="exponential", scale=10.0)

# Combined (sparse dominates)
R_total = R_sparse + 0.1 * R_shape
```

### 4. **Use appropriate units**

```python
# Always set env.units first
env.units = "cm"

# Then bandwidth/scale have meaning
reward = region_reward_field(
    env, "goal",
    decay="gaussian",
    bandwidth=15.0  # 15 cm - has physical meaning
)
```

### 5. **Visualize before using**

```python
# Always plot reward fields to verify
import matplotlib.pyplot as plt

reward = goal_reward_field(env, goal_bin, decay="exponential", scale=20.0)

env.plot_field(reward, cmap='hot', title='Reward Field')
plt.show()

# Check: Does it match your intention?
```

---

## 7. Mathematical Foundations

### Potential-Based Reward Shaping (Ng et al., 1999)

**Theorem**: If shaped reward is:
```
R'(s, a, s') = R(s, a, s') + γΦ(s') - Φ(s)
```

where Φ is a potential function, then optimal policy is **invariant**.

**In neurospatial**:

```python
# Create potential function
Φ = goal_reward_field(env, goal_bin, decay="exponential", scale=γ*R_max)

# Shaped reward (implicit in state transitions)
# R'(s→s') = R(s→s') + γ*Φ[s'] - Φ[s]
```

**Guarantee**: Optimal policy unchanged (but learning faster)

---

### Graph Diffusion Theory

**Heat equation on graph**:
```
∂u/∂t = -L*u

where L = D - A (Laplacian)
      D = degree matrix
      A = adjacency matrix
```

**Solution**:
```
u(t) = exp(-t*L) * u(0)
```

**In neurospatial**:
```python
# Initial condition: indicator on region
u_0 = indicator_field

# Diffuse for time t
u_t = expm(-t * L) @ u_0

# This is what Gaussian decay computes
```

---

## 8. References

1. **Ng, A. Y., Harada, D., & Russell, S. (1999)**. *Policy invariance under reward transformations: Theory and application to reward shaping.* ICML.
   - Theory behind reward shaping

2. **Sutton, R. S., & Barto, A. G. (2018)**. *Reinforcement Learning: An Introduction (2nd ed.)*
   - RL fundamentals

3. **Botvinick, M., & Weinstein, A. (2014)**. *Model-based hierarchical reinforcement learning and human action control.* Phil. Trans. R. Soc. B.
   - RL in neuroscience

4. **Schultz, W. (2015)**. *Neuronal reward and decision signals: from theories to data.* Physiological Reviews.
   - Value coding in brain

---

## Summary

**Reward fields** in neurospatial provide:

1. **Flexible reward definition** - Regions or points, sparse or shaped
2. **Graph-aware computation** - Respects environment structure
3. **Multiple decay profiles** - Constant, linear, exponential, Gaussian, inverse
4. **RL integration** - Ready for value iteration, Q-learning, policy gradient
5. **Neuroscience modeling** - Compare with neural value representations

**Key insight**: By computing rewards on the **graph connectivity** rather than Euclidean space, reward fields naturally respect environment boundaries and provide realistic navigation gradients.
