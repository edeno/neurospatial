# Reward Field Primitives for Reinforcement Learning

This guide covers reward field generation for reinforcement learning (RL) and reward shaping in spatial navigation tasks.

## Overview

The neurospatial library provides two complementary functions for reward field generation:

- **`region_reward_field()`** - Generate rewards from named spatial regions
- **`goal_reward_field()`** - Generate distance-based rewards from goal bins

Both functions follow the neurospatial API convention: **environment comes first** in the parameter order. They use a **consistent parameter name** (`decay`) for specifying reward decay profiles.

## Why Reward Shaping?

Reward shaping provides additional guidance signals to help RL agents learn faster in sparse reward environments. The key trade-off:

- **Sparse rewards** (binary 0/1): Unbiased but slow learning
- **Shaped rewards** (gradients): Faster learning but can bias toward suboptimal policies

**Best practice:** Always validate shaped reward policies against sparse reward baselines to ensure you haven't inadvertently changed the optimal policy (Ng et al., 1999).

## Region-Based Rewards

### Basic Usage

Generate reward fields from named regions in your environment:

```python
import numpy as np
from neurospatial import Environment
from neurospatial import region_reward_field

# Create environment
positions = np.random.randn(1000, 2) * 50
env = Environment.from_samples(positions, bin_size=5.0)
env.units = "cm"

# Define goal region (circular, 10 cm radius around origin)
env.regions.add("goal", point=np.array([0.0, 0.0]))

# Binary reward (sparse RL)
reward_sparse = region_reward_field(env, "goal", decay="constant")

# Linear decay from boundary
reward_linear = region_reward_field(env, "goal", decay="linear")

# Smooth Gaussian falloff
reward_smooth = region_reward_field(
    env, "goal", decay="gaussian", bandwidth=10.0
)
```

### Decay Types for Region Rewards

#### 1. Constant (Binary Reward)

**Use case:** Sparse reward RL, no reward shaping

```python
reward = region_reward_field(env, "goal", reward_value=1.0, decay="constant")
```

Creates a binary reward field:
- `reward_value` inside the region
- `0.0` outside the region

This is the **unbiased** approach - the agent receives reward only when reaching the goal. No gradient information is provided.

**Pros:**
- Guaranteed to preserve optimal policy
- Clear success criterion
- Standard for benchmark tasks

**Cons:**
- Can be very slow to learn in large environments
- No guidance toward goal

#### 2. Linear Decay

**Use case:** Gradient-based reward shaping with clear boundaries

```python
reward = region_reward_field(env, "goal", reward_value=1.0, decay="linear")
```

Provides linear decay from the region boundary:
- Full `reward_value` inside the region
- Linear decay outside: `reward = reward_value * (1 - distance/max_distance)`
- Uses graph-based distance (respects walls and obstacles)

**Pros:**
- Provides gradient information
- Constant gradient magnitude (predictable)
- Decays to exactly zero at maximum distance

**Cons:**
- May inadvertently guide toward suboptimal paths (test against sparse baseline!)
- Normalizes by global maximum distance (far regions get near-zero rewards)

**Note:** The linear decay uses the **maximum distance** in the environment for normalization. This means bins very far from the region will have near-zero rewards. Consider using `decay="gaussian"` if you need more control over the decay rate.

#### 3. Gaussian Falloff

**Use case:** Smooth potential-based reward shaping

```python
reward = region_reward_field(
    env, "goal", reward_value=1.0, decay="gaussian", bandwidth=10.0
)
```

Provides smooth Gaussian falloff using spatial smoothing:
- Indicator field (1 inside, 0 outside) is smoothed with Gaussian kernel
- **Critical:** Rescaled so maximum **within the region** equals `reward_value`
- Bandwidth controls decay rate (larger = slower decay)

**Pros:**
- Smoothest gradients (good for gradient-based RL)
- Tunable decay rate via `bandwidth`
- Peak reward preserved in region (unlike naive smoothing)

**Cons:**
- Most likely to bias policies (use with caution!)
- Requires tuning bandwidth parameter
- Computationally more expensive (smoothing operation)

**Important:** The Gaussian rescaling uses the max **within the region**, not the global max. This ensures the intended reward magnitude is preserved. Naive smoothing would reduce the peak reward, weakening the goal signal.

```python
# CORRECT (implemented in neurospatial)
max_in_region = smoothed[region_mask].max()
reward = (smoothed / max_in_region) * reward_value

# WRONG (naive approach - peak is reduced)
# reward = smoothed * reward_value
```

### Complete Example: Region Rewards

```python
import numpy as np
import matplotlib.pyplot as plt
from neurospatial import Environment
from neurospatial import region_reward_field

# Create simple 2D environment
positions = np.random.randn(2000, 2) * 50
env = Environment.from_samples(positions, bin_size=5.0)
env.units = "cm"

# Define circular goal region at origin
env.regions.add("goal", point=np.array([0.0, 0.0]))

# Compare decay types
reward_sparse = region_reward_field(env, "goal", reward_value=10.0, decay="constant")
reward_linear = region_reward_field(env, "goal", reward_value=10.0, decay="linear")
reward_smooth = region_reward_field(env, "goal", reward_value=10.0,
                                     decay="gaussian", bandwidth=15.0)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

env.plot(reward_sparse, ax=axes[0], cmap='viridis')
axes[0].set_title('Constant (Sparse)')

env.plot(reward_linear, ax=axes[1], cmap='viridis')
axes[1].set_title('Linear Decay')

env.plot(reward_smooth, ax=axes[2], cmap='viridis')
axes[2].set_title('Gaussian Falloff')

plt.tight_layout()
plt.show()
```

## Distance-Based Rewards

### Basic Usage

Generate reward fields from goal bin locations:

```python
from neurospatial import goal_reward_field

# Select goal bin at origin
goal_bin = env.bin_at(np.array([[0.0, 0.0]]))[0]

# Exponential decay (most common in RL)
reward_exp = goal_reward_field(
    env, goal_bins=goal_bin, decay="exponential", scale=10.0
)

# Linear decay with cutoff
reward_lin = goal_reward_field(
    env, goal_bins=goal_bin, decay="linear", scale=10.0, max_distance=50.0
)

# Inverse distance
reward_inv = goal_reward_field(
    env, goal_bins=goal_bin, decay="inverse", scale=10.0
)
```

### Decay Functions for Goal Rewards

#### 1. Exponential Decay (Default)

**Use case:** Standard RL reward shaping

```python
reward = goal_reward_field(env, goal_bins, decay="exponential", scale=10.0)
```

Formula: `reward = scale * exp(-distance / scale)`

**Pros:**
- **Most common in RL literature** - well-studied properties
- Smooth gradients everywhere
- `scale` parameter controls both peak reward and decay rate
- Never exactly zero (global gradient information)

**Cons:**
- May guide agents through walls if not careful (uses graph distance in neurospatial, so this is handled)
- Requires tuning `scale` parameter

**Scale parameter:**
- Larger scale = slower decay, longer-range guidance
- Smaller scale = faster decay, more local reward
- At distance = scale, reward = scale * exp(-1) ≈ 0.37 * scale

**Validation:**
```python
# Must have scale > 0
goal_reward_field(env, goal_bins, decay="exponential", scale=0.0)
# Raises ValueError: "scale must be positive for exponential decay"
```

#### 2. Linear Decay

**Use case:** Strictly local reward shaping

```python
reward = goal_reward_field(
    env, goal_bins, decay="linear", scale=10.0, max_distance=50.0
)
```

Formula: `reward = scale * max(0, 1 - distance / max_distance)`

**Pros:**
- Reaches **exactly zero** at `max_distance`
- Constant gradient magnitude within range
- Easy to reason about (clear reward radius)

**Cons:**
- Requires specifying `max_distance` (extra hyperparameter)
- No gradient information beyond cutoff
- Abrupt transition at boundary

**Required parameter:**
```python
# max_distance is required
goal_reward_field(env, goal_bins, decay="linear", scale=10.0)
# Raises ValueError: "max_distance required for linear decay"
```

#### 3. Inverse Distance

**Use case:** Global reward gradients

```python
reward = goal_reward_field(env, goal_bins, decay="inverse", scale=10.0)
```

Formula: `reward = scale / (1 + distance)`

**Pros:**
- Simple formula
- Never reaches zero (global gradients)
- No hyperparameter tuning needed

**Cons:**
- **Most likely to bias policies** - use with extreme caution!
- Very slow decay (can dominate learning signal far from goal)
- Not commonly used in modern RL

**When to use:** Rarely. Consider exponential decay instead unless you have a specific reason to use inverse distance.

### Multiple Goals

All decay functions support multiple goal locations:

```python
# Define multiple goal bins
goal_bin_1 = env.bin_at(np.array([[20.0, 20.0]]))[0]
goal_bin_2 = env.bin_at(np.array([[-20.0, -20.0]]))[0]
goal_bins = [goal_bin_1, goal_bin_2]

# Reward is computed to NEAREST goal
reward = goal_reward_field(env, goal_bins, decay="exponential", scale=10.0)
```

This creates a **Voronoi-like partition** where each spatial bin is influenced by its closest goal. Useful for multi-goal tasks or hierarchical RL.

### Complete Example: Distance-Based Rewards

```python
import numpy as np
import matplotlib.pyplot as plt
from neurospatial import Environment, goal_reward_field

# Create environment
positions = np.random.randn(2000, 2) * 50
env = Environment.from_samples(positions, bin_size=5.0)
env.units = "cm"

# Select goal at origin
goal_bin = env.bin_at(np.array([[0.0, 0.0]]))[0]

# Compare decay functions
reward_exp = goal_reward_field(env, goal_bin, decay="exponential", scale=15.0)
reward_lin = goal_reward_field(env, goal_bin, decay="linear", scale=10.0, max_distance=50.0)
reward_inv = goal_reward_field(env, goal_bin, decay="inverse", scale=10.0)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

env.plot(reward_exp, ax=axes[0], cmap='viridis')
axes[0].set_title('Exponential Decay')

env.plot(reward_lin, ax=axes[1], cmap='viridis')
axes[1].set_title('Linear Decay')

env.plot(reward_inv, ax=axes[2], cmap='viridis')
axes[2].set_title('Inverse Distance')

plt.tight_layout()
plt.show()
```

## Consistent Parameter Naming: `decay`

**Important:** Both functions use the parameter name **`decay`** (not `falloff`, `kind`, or `method`) for consistency:

```python
# Region rewards
region_reward_field(env, "goal", decay="constant")  # ✓
region_reward_field(env, "goal", decay="linear")    # ✓
region_reward_field(env, "goal", decay="gaussian")  # ✓

# Goal rewards
goal_reward_field(env, goal_bins, decay="exponential")  # ✓
goal_reward_field(env, goal_bins, decay="linear")       # ✓
goal_reward_field(env, goal_bins, decay="inverse")      # ✓
```

This naming choice:
- Matches common RL terminology ("reward decay")
- Provides consistency across all reward field functions
- Makes code more readable and predictable

## Reward Shaping Strategies

### 1. Potential-Based Reward Shaping

The reward functions in neurospatial implement **potential-based reward shaping** (Ng et al., 1999), where:

```
shaped_reward(s, a, s') = reward(s, a, s') + γ * Φ(s') - Φ(s)
```

Where `Φ(s)` is a potential function (your reward field). This formulation **preserves optimal policies** under certain conditions.

**Usage in neurospatial:**

```python
# Potential function = distance-based reward field
phi = goal_reward_field(env, goal_bin, decay="exponential", scale=10.0)

# In your RL loop:
# shaped_reward = sparse_reward + gamma * phi[next_state] - phi[current_state]
```

### 2. Combining Region and Goal Rewards

You can combine multiple reward sources:

```python
# Region-based reward for primary goal
primary_reward = region_reward_field(env, "primary_goal", reward_value=100.0)

# Goal-based reward for subgoals
subgoal_bins = [bin1, bin2, bin3]
subgoal_reward = goal_reward_field(env, subgoal_bins, decay="exponential", scale=10.0)

# Combined reward field
total_reward = primary_reward + 0.1 * subgoal_reward  # Weight subgoals less
```

**Caution:** Combining multiple reward sources requires careful tuning. Always validate against sparse baseline!

### 3. Curriculum Learning

Use gradually decreasing reward shaping:

```python
# Early training: strong shaping
reward_early = goal_reward_field(env, goal_bin, decay="exponential", scale=20.0)

# Mid training: moderate shaping
reward_mid = goal_reward_field(env, goal_bin, decay="exponential", scale=10.0)

# Late training: minimal shaping or sparse
reward_late = region_reward_field(env, "goal", decay="constant")
```

This can help agents learn faster initially while converging to optimal policies later.

## Cautions About Reward Shaping

### When Shaping Can Hurt

From Ng et al. (1999):

> "Poorly designed reward shaping can cause agents to learn suboptimal policies that are difficult to correct."

**Example:** In a maze with a shortcut, linear decay might guide the agent around obstacles instead of through a hidden passage.

**Recommendation:**
1. Start with sparse (constant) rewards
2. Add shaping only if learning is too slow
3. Always validate final policy against sparse baseline
4. Prefer exponential decay (well-studied) over inverse distance

### Testing Your Reward Design

```python
# 1. Visualize reward field
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
env.plot(reward_field, ax=ax, cmap='viridis')
ax.set_title('Reward Field')
plt.show()

# 2. Check reward gradients point toward goal
# (Use gradient operators from neurospatial.differential when available)

# 3. Train with and without shaping
policy_sparse = train_agent(env, reward_sparse)
policy_shaped = train_agent(env, reward_shaped)

# 4. Compare performance
# (Shaped should learn faster, but converge to same or better policy)
```

## API Reference

For complete parameter descriptions and examples, see:

- [`region_reward_field()`](../api/reward.md#region_reward_field) - Region-based rewards
- [`goal_reward_field()`](../api/reward.md#goal_reward_field) - Distance-based rewards
- [`distance_field()`](../api/distance.md#distance_field) - Underlying distance computation

## Related Topics

- [Spike Field Primitives](spike-field-primitives.md) - Converting spike trains to spatial fields
- [Spatial Analysis](spatial-analysis.md) - Broader spatial analysis workflows
- [Differential Operators](differential-operators.md) - Computing reward gradients (coming in v0.3.0)

## References

1. **Ng, A. Y., Harada, D., & Russell, S. (1999)**. "Policy invariance under reward transformations: Theory and application to reward shaping." *ICML*.

2. **Sutton, R. S., & Barto, A. G. (2018)**. *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

3. **Mnih, V., et al. (2015)**. "Human-level control through deep reinforcement learning." *Nature*.
