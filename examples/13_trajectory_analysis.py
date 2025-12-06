# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: neurospatial
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Trajectory Analysis
#
# This notebook demonstrates trajectory characterization metrics in neurospatial.
#
# **Estimated time**: 10-15 minutes
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# - Compute turn angles to analyze direction changes in movement trajectories
# - Calculate step lengths to quantify distance traveled between positions
# - Estimate home range using quantile-based methods
# - Compute mean square displacement (MSD) to classify diffusion patterns
# - Interpret MSD slopes to distinguish Brownian, super-diffusive, and sub-diffusive motion
# - Apply trajectory metrics to behavioral characterization and quality control
#
# ## References
#
# - **Turn angles**: Pérez-Escudero et al. (2014). *idTracker*. Nature Methods.
# - **Home range**: Powell & Mitchell (2012). What is a home range? *Journal of Mammalogy*.
# - **MSD**: Einstein (1905). On the movement of small particles. *Annalen der Physik*.
# - **Ecology metrics**: Signer et al. (2019). Animal movement tools (amt). *Journal of Statistical Software*.

# %%
import matplotlib.pyplot as plt
import numpy as np

from neurospatial import Environment
from neurospatial.behavior.trajectory import (
    compute_home_range,
    compute_step_lengths,
    compute_turn_angles,
    mean_square_displacement,
)

# Set random seed for reproducibility
np.random.seed(42)

# %% [markdown]
# ## Part 1: Generate Synthetic Trajectory
#
# We'll create a realistic foraging trajectory with:
# - Random exploration phase (first 50%)
# - Directed movement to goal (last 50%)

# %%
# Generate realistic foraging trajectory in 2D arena
n_samples = 800
times = np.linspace(0, 100, n_samples)  # 100 seconds at 8 Hz

# Arena: 100x100 cm square
arena_size = 100.0
arena_center = arena_size / 2

# Phase 1: Random walk exploration (first 60% - 480 samples)
exploration_samples = int(n_samples * 0.6)
positions = np.zeros((n_samples, 2))
positions[0] = [arena_center, arena_center]  # Start at center

# Random walk with wall reflection
step_size = 2.5  # cm per step
boundary_margin = 5.0

for i in range(1, exploration_samples):
    # Random step
    angle = np.random.uniform(0, 2 * np.pi)
    step = step_size * np.array([np.cos(angle), np.sin(angle)])
    new_pos = positions[i - 1] + step

    # Reflect at boundaries
    for dim in range(2):
        if new_pos[dim] < boundary_margin:
            new_pos[dim] = boundary_margin + (boundary_margin - new_pos[dim])
        elif new_pos[dim] > (arena_size - boundary_margin):
            new_pos[dim] = (arena_size - boundary_margin) - (
                new_pos[dim] - (arena_size - boundary_margin)
            )

    positions[i] = new_pos

# Phase 2: Goal-directed movement to goal location (last 40%)
goal = np.array([85.0, 85.0])  # Top-right corner
directed_samples = n_samples - exploration_samples
current_pos = positions[exploration_samples - 1]

# Gradually move toward goal with some variability
for i in range(directed_samples):
    idx = exploration_samples + i
    progress = (i + 1) / directed_samples

    # Interpolate toward goal with noise
    target = current_pos + progress * (goal - current_pos)
    noise = (
        np.random.randn(2) * 2.0 * (1 - progress * 0.7)
    )  # Reduce noise as we approach
    positions[idx] = target + noise

    # Keep within bounds
    positions[idx] = np.clip(
        positions[idx], boundary_margin, arena_size - boundary_margin
    )

# Create environment from trajectory
env = Environment.from_samples(positions, bin_size=3.0)
env.units = "cm"
env.frame = "behavior_session_1"

print(f"Environment: {arena_size:.0f}x{arena_size:.0f} cm arena")
print(f"  {env.n_bins} bins, {env.n_dims}D")
print(f"Trajectory: {n_samples} samples over {times[-1]:.1f} seconds")
print(f"  Exploration phase: {exploration_samples} samples")
print(f"  Goal-directed phase: {directed_samples} samples")

# %% [markdown]
# Visualize the trajectory with both phases:

# %%
fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

# Plot trajectory with color gradient showing time
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, n_samples))
for i in range(n_samples - 1):
    ax.plot(
        positions[i : i + 2, 0],
        positions[i : i + 2, 1],
        color=colors[i],
        linewidth=2,
        alpha=0.6,
    )

# Mark start and goal
ax.scatter(
    positions[0, 0],
    positions[0, 1],
    c="green",
    s=200,
    marker="o",
    edgecolors="black",
    linewidths=2,
    label="Start",
    zorder=10,
)
ax.scatter(
    goal[0],
    goal[1],
    c="red",
    s=200,
    marker="*",
    edgecolors="black",
    linewidths=2,
    label="Goal",
    zorder=10,
)

# Mark transition point between exploration and directed
transition_idx = exploration_samples
ax.scatter(
    positions[transition_idx, 0],
    positions[transition_idx, 1],
    c="orange",
    s=150,
    marker="D",
    edgecolors="black",
    linewidths=2,
    label="Transition",
    zorder=10,
)

ax.set_xlabel("X position (cm)", fontsize=14, fontweight="bold")
ax.set_ylabel("Y position (cm)", fontsize=14, fontweight="bold")
ax.set_title(
    "Synthetic Trajectory: Exploration → Goal-Directed",
    fontsize=16,
    fontweight="bold",
)
ax.legend(fontsize=12, loc="upper left")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

plt.show()

# %% [markdown]
# ## Part 2: Compute Turn Angles
#
# **Turn angles** measure changes in movement direction between consecutive steps.
#
# - **Formula**: Angle between velocity vectors $\vec{v}_i$ and $\vec{v}_{i+1}$
# - **Range**: $[-\pi, \pi]$ radians
# - **Interpretation**:
#   - 0°: Straight ahead (ballistic movement)
#   - ±90°: Perpendicular turn
#   - ±180°: Reversal
#
# **Use cases**:
# - Identify search strategies (e.g., Lévy walks)
# - Detect behavioral state changes (exploration vs exploitation)
# - Quantify path tortuosity

# %% [markdown]
# ### Single Example: Visualize One Turn Angle
#
# Before computing all turn angles, let's visualize a single example to understand the calculation:

# %%
# Pick a section of trajectory to demonstrate (samples 100-103)
demo_idx = 100
demo_positions = positions[demo_idx : demo_idx + 3]

# Compute vectors
v1 = demo_positions[1] - demo_positions[0]  # First velocity vector
v2 = demo_positions[2] - demo_positions[1]  # Second velocity vector

# Compute turn angle manually
angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
# Wrap to [-π, π]
if angle > np.pi:
    angle -= 2 * np.pi
elif angle < -np.pi:
    angle += 2 * np.pi

# Visualize the turn
fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

# Plot the three positions
ax.plot(
    demo_positions[:, 0],
    demo_positions[:, 1],
    "ko-",
    markersize=12,
    linewidth=2,
    label="Path",
)

# Label positions
for i, pos in enumerate(demo_positions):
    ax.text(
        pos[0] + 1,
        pos[1] + 1,
        f"P{i + 1}",
        fontsize=16,
        fontweight="bold",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

# Draw velocity vectors
arrow_scale = 3.0
ax.arrow(
    demo_positions[0, 0],
    demo_positions[0, 1],
    v1[0] * arrow_scale,
    v1[1] * arrow_scale,
    head_width=1.5,
    head_length=1.0,
    fc="blue",
    ec="blue",
    linewidth=2,
    label="v₁",
)
ax.arrow(
    demo_positions[1, 0],
    demo_positions[1, 1],
    v2[0] * arrow_scale,
    v2[1] * arrow_scale,
    head_width=1.5,
    head_length=1.0,
    fc="orange",
    ec="orange",
    linewidth=2,
    label="v₂",
)

ax.set_xlabel("X position (cm)", fontsize=14, fontweight="bold")
ax.set_ylabel("Y position (cm)", fontsize=14, fontweight="bold")
ax.set_title(
    f"Single Turn Angle Example\nAngle = {np.degrees(angle):.1f}°",
    fontsize=16,
    fontweight="bold",
)
ax.legend(fontsize=12, loc="upper left")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

plt.show()

print("\n✓ Single Turn Angle Validation:")
print(f"  Position P1: {demo_positions[0]}")
print(f"  Position P2: {demo_positions[1]}")
print(f"  Position P3: {demo_positions[2]}")
print(f"  Velocity v1: {v1}")
print(f"  Velocity v2: {v2}")
print(f"  Turn angle: {np.degrees(angle):.2f}° ({angle:.3f} rad)")
print(
    f"  Interpretation: {'Left turn' if angle > 0 else 'Right turn' if angle < 0 else 'Straight'}"
)

# %% [markdown]
# ### Compute All Turn Angles
#
# Now that we understand how a single turn angle is calculated, let's compute all turn angles across the trajectory:

# %%
# Map trajectory to bins
trajectory_bins = env.bin_at(positions)

# Compute turn angles from continuous positions
turn_angles = compute_turn_angles(positions)

print(f"Computed {len(turn_angles)} turn angles")
print(
    f"Mean absolute turn angle: {np.abs(turn_angles).mean():.2f} rad ({np.degrees(np.abs(turn_angles).mean()):.1f}°)"
)
print(
    f"Median turn angle: {np.median(turn_angles):.2f} rad ({np.degrees(np.median(turn_angles)):.1f}°)"
)

# Validation checks
print("\n✓ Validation Checks:")
assert len(turn_angles) == len(positions) - 2, "ERROR: Unexpected number of turn angles"
print(f"  ✓ Correct number of turn angles: {len(turn_angles)} (n_positions - 2)")

assert np.all(np.abs(turn_angles) <= np.pi), "ERROR: Turn angles outside [-π, π]"
print(
    f"  ✓ All turn angles in valid range: [{np.degrees(-np.pi):.0f}°, {np.degrees(np.pi):.0f}°]"
)

# Check that our manual calculation matches
computed_angle = turn_angles[demo_idx]
if np.abs(computed_angle - angle) < 0.01:
    print(
        f"  ✓ Manual calculation matches compute_turn_angles(): {np.degrees(computed_angle):.1f}°"
    )
else:
    print(
        f"  ⚠ Warning: Manual calculation ({np.degrees(angle):.1f}°) differs from computed ({np.degrees(computed_angle):.1f}°)"
    )

# %% [markdown]
# Visualize turn angle distribution:

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")

# Histogram
axes[0].hist(
    np.degrees(turn_angles), bins=50, color="steelblue", edgecolor="black", alpha=0.7
)
axes[0].axvline(0, color="red", linestyle="--", linewidth=2, label="Straight ahead")
axes[0].set_xlabel("Turn angle (degrees)", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Frequency", fontsize=14, fontweight="bold")
axes[0].set_title("Turn Angle Distribution", fontsize=16, fontweight="bold")
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

# Time series (split by phase)
time_indices = np.arange(len(turn_angles))
explore_mask = time_indices < exploration_samples
directed_mask = time_indices >= exploration_samples

axes[1].plot(
    time_indices[explore_mask],
    np.degrees(turn_angles[explore_mask]),
    "o",
    color="steelblue",
    alpha=0.6,
    markersize=4,
    label="Exploration",
)
axes[1].plot(
    time_indices[directed_mask],
    np.degrees(turn_angles[directed_mask]),
    "o",
    color="orange",
    alpha=0.6,
    markersize=4,
    label="Goal-directed",
)
axes[1].axhline(0, color="red", linestyle="--", linewidth=2)
axes[1].set_xlabel("Time step", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Turn angle (degrees)", fontsize=14, fontweight="bold")
axes[1].set_title("Turn Angles Over Time", fontsize=16, fontweight="bold")
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.show()

# %% [markdown]
# **Observation**: Goal-directed phase shows smaller turn angles (straighter paths) compared to exploration phase.

# %% [markdown]
# ## Part 3: Compute Step Lengths
#
# **Step lengths** measure the distance traveled between consecutive positions.
#
# - **Units**: Graph geodesic distance (bin-to-bin on connectivity graph)
# - **Interpretation**:
#   - Short steps: Fine-scale exploration or slow movement
#   - Long steps: Ballistic movement or relocation
#
# **Use cases**:
# - Identify movement modes (Brownian vs Lévy flights)
# - Detect behavioral transitions
# - Compute travel distance and speed

# %%
# Compute step lengths using geodesic (graph) distances
# This respects the environment topology rather than using straight-line distance
step_lengths = compute_step_lengths(positions, distance_type="geodesic", env=env)

# Note: geodesic distances may be infinite for disconnected bins
finite_step_lengths = step_lengths[np.isfinite(step_lengths)]

# Compute cumulative distance (use finite values)
cumulative_distance = np.concatenate([[0], np.cumsum(finite_step_lengths)])

print(f"Computed {len(step_lengths)} step lengths")
print(
    f"Finite step lengths: {len(finite_step_lengths)} ({len(finite_step_lengths) / len(step_lengths):.1%})"
)
print(f"Mean step length: {finite_step_lengths.mean():.2f} cm")
print(f"Total distance traveled: {cumulative_distance[-1]:.1f} cm")
print(
    f"Straight-line displacement: {np.linalg.norm(positions[-1] - positions[0]):.1f} cm"
)
print(
    f"Path efficiency: {np.linalg.norm(positions[-1] - positions[0]) / cumulative_distance[-1]:.2%}"
)

# %% [markdown]
# Visualize step length distribution and time series:

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")

# Histogram (finite values only)
axes[0].hist(
    finite_step_lengths, bins=50, color="forestgreen", edgecolor="black", alpha=0.7
)
axes[0].axvline(
    finite_step_lengths.mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {finite_step_lengths.mean():.1f} cm",
)
axes[0].set_xlabel("Step length (cm)", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Frequency", fontsize=14, fontweight="bold")
axes[0].set_title("Step Length Distribution", fontsize=16, fontweight="bold")
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

# Cumulative distance
time_steps = np.arange(len(cumulative_distance))
explore_time = time_steps < exploration_samples

axes[1].plot(
    time_steps[explore_time],
    cumulative_distance[explore_time],
    color="steelblue",
    linewidth=3,
    label="Exploration phase",
)
axes[1].plot(
    time_steps[~explore_time],
    cumulative_distance[~explore_time],
    color="orange",
    linewidth=3,
    label="Goal-directed phase",
)
axes[1].set_xlabel("Time step", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Cumulative distance (cm)", fontsize=14, fontweight="bold")
axes[1].set_title("Distance Traveled Over Time", fontsize=16, fontweight="bold")
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.show()

# %% [markdown]
# **Observation**: Goal-directed phase shows steeper cumulative distance (faster movement toward goal).

# %% [markdown]
# ## Part 4: Compute Home Range
#
# **Home range** is the area where an animal spends a given percentage of its time.
#
# - **Standard percentiles**:
#   - 50%: Core area (intensive use)
#   - 95%: Full home range (typical ecology standard)
#   - 100%: Total area visited
#
# - **Computation**: Based on occupancy (time spent in each bin)
#
# **Use cases**:
# - Territory size estimation
# - Habitat preference analysis
# - Spatial memory assessment

# %%
# Compute home ranges at different percentiles
core_area_50 = compute_home_range(trajectory_bins, percentile=50.0)
home_range_95 = compute_home_range(trajectory_bins, percentile=95.0)
total_area = compute_home_range(trajectory_bins, percentile=100.0)

print(f"Core area (50%): {len(core_area_50)} bins")
print(f"Home range (95%): {len(home_range_95)} bins")
print(f"Total area visited: {len(total_area)} bins")

# %% [markdown]
# Visualize home range bins on environment:

# %%
# Create masks for visualization
core_mask = np.zeros(env.n_bins, dtype=bool)
core_mask[core_area_50] = True

home_mask = np.zeros(env.n_bins, dtype=bool)
home_mask[home_range_95] = True

total_mask = np.zeros(env.n_bins, dtype=bool)
total_mask[total_area] = True

# Plot home ranges
fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")

for ax, mask, title, _percentile in zip(
    axes,
    [core_mask, home_mask, total_mask],
    ["Core Area (50%)", "Home Range (95%)", "Total Area (100%)"],
    [50, 95, 100],
    strict=True,
):
    # Plot bins in range
    bins_in_range = np.where(mask)[0]
    ax.scatter(
        env.bin_centers[bins_in_range, 0],
        env.bin_centers[bins_in_range, 1],
        c="steelblue",
        s=100,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
    )

    # Overlay trajectory
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        color="gray",
        linewidth=1,
        alpha=0.4,
        zorder=0,
    )

    # Mark start and goal
    ax.scatter(
        positions[0, 0],
        positions[0, 1],
        c="green",
        s=150,
        marker="o",
        edgecolors="black",
        linewidths=2,
        label="Start",
        zorder=10,
    )
    ax.scatter(
        goal[0],
        goal[1],
        c="red",
        s=150,
        marker="*",
        edgecolors="black",
        linewidths=2,
        label="Goal",
        zorder=10,
    )

    ax.set_xlabel("X position (cm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Y position (cm)", fontsize=14, fontweight="bold")
    ax.set_title(f"{title}\n{len(bins_in_range)} bins", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

plt.show()

# %% [markdown]
# **Observation**: Core area (50%) concentrates around exploration region. Full home range (95%) includes path to goal.

# %% [markdown]
# ## Part 5: Mean Square Displacement (MSD)
#
# **MSD** characterizes diffusion properties of movement:
#
# $$\text{MSD}(\tau) = \langle [x(t + \tau) - x(t)]^2 \rangle$$
#
# **Power law relationship**: $\text{MSD}(\tau) \sim \tau^\alpha$
#
# **Diffusion classification** (from MSD exponent $\alpha$):
# - $\alpha < 1$: Subdiffusion (confined, territorial)
# - $\alpha = 1$: Normal diffusion (random walk, Brownian motion)
# - $\alpha > 1$: Superdiffusion (ballistic, directed movement)
# - $\alpha = 2$: Ballistic motion (constant velocity)
#
# **Use cases**:
# - Classify movement strategies
# - Detect confinement or barrier effects
# - Identify directed vs random search

# %%
# Compute MSD for different time lags using geodesic distances
tau_values, msd_values = mean_square_displacement(
    positions, times, distance_type="geodesic", env=env, max_tau=30.0
)

# Fit power law: MSD ~ tau^alpha
# Use log-log fit: log(MSD) = alpha * log(tau) + const
valid_idx = (tau_values > 0) & (msd_values > 0)
log_tau = np.log(tau_values[valid_idx])
log_msd = np.log(msd_values[valid_idx])

# Linear fit in log-log space
alpha, log_const = np.polyfit(log_tau, log_msd, 1)
msd_fit = np.exp(log_const) * tau_values[valid_idx] ** alpha

print(f"MSD exponent α = {alpha:.2f}")  # noqa: RUF001  # α is scientific notation
print("\nDiffusion classification:")
if alpha < 0.9:
    print("  → Subdiffusion (confined movement)")
elif alpha < 1.1:
    print("  → Normal diffusion (random walk)")
elif alpha < 1.9:
    print("  → Superdiffusion (directed movement)")
else:
    print("  → Ballistic motion (constant velocity)")

# %% [markdown]
# Visualize MSD curve with power law fit:

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")

# MSD vs tau (linear scale)
axes[0].plot(
    tau_values, msd_values, "o", color="steelblue", markersize=8, label="Observed MSD"
)
axes[0].plot(
    tau_values[valid_idx],
    msd_fit,
    "--",
    color="red",
    linewidth=2,
    label=f"Fit: MSD ~ τ^{alpha:.2f}",
)
axes[0].set_xlabel("Time lag τ (seconds)", fontsize=14, fontweight="bold")
axes[0].set_ylabel("MSD (cm²)", fontsize=14, fontweight="bold")
axes[0].set_title("Mean Square Displacement", fontsize=16, fontweight="bold")
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

# MSD vs tau (log-log scale)
axes[1].loglog(
    tau_values, msd_values, "o", color="steelblue", markersize=8, label="Observed MSD"
)
axes[1].loglog(
    tau_values[valid_idx],
    msd_fit,
    "--",
    color="red",
    linewidth=2,
    label=f"Slope = {alpha:.2f}",
)
axes[1].set_xlabel("Time lag τ (seconds)", fontsize=14, fontweight="bold")
axes[1].set_ylabel("MSD (cm²)", fontsize=14, fontweight="bold")
axes[1].set_title("MSD (Log-Log Scale)", fontsize=16, fontweight="bold")
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3, which="both")

plt.show()

# %% [markdown]
# **Observation**: MSD exponent α > 1 indicates superdiffusion (consistent with goal-directed movement in second phase).

# %% [markdown]
# ## Summary
#
# This notebook demonstrated four key trajectory metrics:
#
# 1. **Turn angles** - Quantified directional changes (straighter in goal-directed phase)
# 2. **Step lengths** - Measured movement distances (faster in goal-directed phase)
# 3. **Home range** - Identified core areas (50%, 95%, 100% occupancy)
# 4. **Mean square displacement** - Classified diffusion type (superdiffusion, α > 1)
#
# ### Key Takeaways
#
# - **Turn angles and step lengths** reveal local movement properties
# - **Home range** characterizes spatial coverage and territory
# - **MSD exponent** classifies overall movement strategy (random vs directed)
#
# ### Next Steps
#
# - **Behavioral segmentation**: Detect specific epochs (runs, laps, trials)
# - **Trajectory similarity**: Compare paths between sessions
# - **Integration with neural activity**: Correlate movement with place cell firing
#
# See `examples/15_behavioral_segmentation.ipynb` for advanced trajectory analysis workflows.
