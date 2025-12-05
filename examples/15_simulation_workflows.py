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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Simulation Workflows
#
# This notebook provides a comprehensive guide to the `neurospatial.simulation` subpackage for generating synthetic spatial data, neural activity, and spike trains.
#
# **Estimated time**: 30-40 minutes
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# - Generate complete simulation sessions using pre-configured functions
# - Understand the two-level API design (high-level convenience vs low-level control)
# - Simulate realistic trajectories using Ornstein-Uhlenbeck process
# - Create place cells, boundary cells, and grid cells with known ground truth
# - Generate spike trains from firing rates using inhomogeneous Poisson process
# - Validate detected spatial fields against ground truth parameters
# - Customize simulations for specific experimental designs
# - Apply simulation tools for algorithm testing and educational demonstrations
#
# **Contents:**
#
# 1. [Introduction](#1-Introduction)
# 2. [Quick Start with Pre-Configured Sessions](#2-Quick-Start-with-Pre-Configured-Sessions)
# 3. [Low-Level API: Building Blocks](#3-Low-Level-API-Building-Blocks)
# 4. [All Pre-Configured Examples](#4-All-Pre-Configured-Examples)
# 5. [Validation Workflow](#5-Validation-Workflow)
# 6. [Customization Examples](#6-Customization-Examples)
# 7. [Performance Tips](#7-Performance-Tips)

# %% [markdown]
# ## 1. Introduction
#
# The simulation subpackage provides tools for:
#
# - **Trajectory simulation**: Realistic animal movement patterns (OU process, laps, sinusoidal)
# - **Neural models**: Place cells, boundary cells, grid cells with biologically realistic tuning
# - **Spike generation**: Poisson spike trains with refractory periods and modulation
# - **Validation helpers**: Automated comparison of detected vs ground truth fields
# - **Pre-configured examples**: Ready-to-use datasets for testing and education
#
# ### Two API Levels
#
# **High-level (recommended for most users)**:
# - `simulate_session()` - One-call workflow for complete sessions
# - Pre-configured examples: `open_field_session()`, `linear_track_session()`, etc.
# - `validate_simulation()` - Automated validation against ground truth
#
# **Low-level (for fine-grained control)**:
# - `simulate_trajectory_ou()`, `simulate_trajectory_laps()` - Manual trajectory generation
# - `PlaceCellModel`, `BoundaryCellModel`, `GridCellModel` - Individual neural models
# - `generate_poisson_spikes()`, `generate_population_spikes()` - Manual spike generation

# %%
# Import all simulation functions
import time

import matplotlib.pyplot as plt
import numpy as np

from neurospatial import Environment
from neurospatial.simulation import (
    BoundaryCellModel,
    PlaceCellModel,
    boundary_cell_session,
    # Low-level spikes
    generate_poisson_spikes,
    generate_population_spikes,
    grid_cell_session,
    linear_track_session,
    # Pre-configured examples
    open_field_session,
    plot_session_summary,
    # High-level API
    simulate_session,
    simulate_trajectory_laps,
    # Low-level trajectory
    simulate_trajectory_ou,
    tmaze_alternation_session,
    validate_simulation,
)

# Set random seed for reproducibility
np.random.seed(42)

# %% [markdown]
# ## 2. Quick Start with Pre-Configured Sessions
#
# The fastest way to generate simulation data is using pre-configured session functions. These combine environment creation, trajectory generation, neural models, and spike generation in a single call.

# %%
# Generate a complete open field session in one line
session = open_field_session(
    duration=10.0,  # 10 seconds (short for demo)
    arena_size=100.0,  # 100 cm square arena
    bin_size=2.0,  # 2 cm spatial bins
    n_place_cells=20,  # 20 place cells
    seed=42,  # Reproducible
)

# Session is a dataclass with typed attributes
print(f"Environment: {session.env.n_bins} bins, {session.env.n_dims}D")
print(f"Trajectory: {len(session.times)} time points over {session.times[-1]:.1f}s")
print(f"Neural activity: {len(session.spike_trains)} cells")
print(f"Total spikes: {sum(len(spikes) for spikes in session.spike_trains)}")

# Access ground truth parameters
print("\nGround truth for first cell:")
print(f"  Center: {session.ground_truth['cell_0']['center']}")
print(f"  Width: {session.ground_truth['cell_0']['width']:.2f} cm")
print(f"  Max rate: {session.ground_truth['cell_0']['max_rate']:.1f} Hz")

# %% [markdown]
# ### Visualize the Session
#
# Use `plot_session_summary()` to get a comprehensive overview:

# %%
fig, axes = plot_session_summary(session, cell_ids=[0, 1, 2, 5, 10, 15])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Low-Level API: Building Blocks
#
# For fine-grained control, build simulations from individual components:
#
# 1. Create environment
# 2. Generate trajectory
# 3. Create neural models
# 4. Generate spikes

# %% [markdown]
# ### Step 1: Create Environment

# %%
# Create a 2D square arena using a simple grid
x = np.linspace(0, 100, 50)
y = np.linspace(0, 100, 50)
xx, yy = np.meshgrid(x, y)
arena_samples = np.column_stack([xx.ravel(), yy.ravel()])

env = Environment.from_samples(arena_samples, bin_size=2.0)
env.units = "cm"  # Required for trajectory simulation
env.name = "Custom Arena"

print(f"Created environment: {env.n_bins} bins, {env.n_dims}D")
print(f"Extent: {env.dimension_ranges}")

# %% [markdown]
# ### Step 2: Generate Trajectory
#
# Use Ornstein-Uhlenbeck process for realistic random exploration:

# %%
positions, times = simulate_trajectory_ou(
    env,
    duration=10.0,  # 10 seconds
    dt=0.01,  # 10 ms time step
    speed_mean=8.0,  # 8 cm/s mean speed
    speed_std=4.0,  # Speed variability
    coherence_time=0.7,  # Velocity correlation time (seconds)
    boundary_mode="reflect",  # Bounce off walls
    seed=42,
)

print(f"Generated trajectory: {len(times)} time points")
print(f"Duration: {times[-1]:.2f} seconds")
print(f"Position range: {positions.min(axis=0)} to {positions.max(axis=0)}")

# Plot trajectory
plt.figure(figsize=(8, 8))
plt.plot(positions[:, 0], positions[:, 1], "b-", alpha=0.5, linewidth=0.5)
plt.scatter(
    positions[0, 0],
    positions[0, 1],
    c="green",
    s=100,
    marker="o",
    label="Start",
    zorder=3,
)
plt.scatter(
    positions[-1, 0],
    positions[-1, 1],
    c="red",
    s=100,
    marker="X",
    label="End",
    zorder=3,
)
plt.xlabel("X position (cm)")
plt.ylabel("Y position (cm)")
plt.title("Simulated Trajectory (OU Process)")
plt.legend()
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ### Step 3: Create Neural Models
#
# Create place cells with known ground truth parameters:

# %%
# Create 5 place cells with specific field locations
place_cells = []
field_centers = [
    [25.0, 25.0],  # Bottom-left
    [75.0, 25.0],  # Bottom-right
    [50.0, 50.0],  # Center
    [25.0, 75.0],  # Top-left
    [75.0, 75.0],  # Top-right
]

for i, center in enumerate(field_centers):
    pc = PlaceCellModel(
        env,
        center=np.array(center),
        width=10.0,  # 10 cm field width
        max_rate=20.0 + i * 2.0,  # Vary peak rates slightly
        baseline_rate=0.1,
        distance_metric="euclidean",  # Fast
        seed=42 + i,
    )
    place_cells.append(pc)

print(f"Created {len(place_cells)} place cells")

# Visualize firing rate maps for the center cell
center_cell = place_cells[2]  # Cell at arena center

# Create grid of positions
x_test = np.linspace(0, 100, 50)
y_test = np.linspace(0, 100, 50)
xx_test, yy_test = np.meshgrid(x_test, y_test)
test_positions = np.column_stack([xx_test.ravel(), yy_test.ravel()])

# Compute firing rates
rates = center_cell.firing_rate(test_positions)
rate_map = rates.reshape(xx_test.shape)

# Plot
plt.figure(figsize=(8, 7))
plt.imshow(rate_map, extent=[0, 100, 0, 100], origin="lower", cmap="hot", aspect="auto")
plt.colorbar(label="Firing rate (Hz)")
plt.scatter(
    *center_cell.ground_truth["center"],
    c="cyan",
    s=200,
    marker="x",
    linewidths=3,
    label="Field center",
)
plt.xlabel("X position (cm)")
plt.ylabel("Y position (cm)")
plt.title("Place Cell Firing Rate Map (Ground Truth)")
plt.legend()
plt.show()

# %% [markdown]
# ### Step 4: Generate Spikes
#
# Generate spike trains from the place cell models:

# %%
# Generate spikes for all cells
spike_trains = generate_population_spikes(
    models=place_cells,
    positions=positions,
    times=times,
    refractory_period=0.002,  # 2 ms refractory period
    seed=42,
    show_progress=False,  # Disable progress bar for cleaner output
)

print(f"Generated spikes for {len(spike_trains)} cells")
for i, spikes in enumerate(spike_trains):
    mean_rate = len(spikes) / times[-1] if len(spikes) > 0 else 0.0
    print(f"  Cell {i}: {len(spikes)} spikes, mean rate = {mean_rate:.2f} Hz")

# Visualize raster plot
plt.figure(figsize=(12, 4))
for i, spikes in enumerate(spike_trains):
    plt.scatter(spikes, np.ones_like(spikes) * i, s=1, c="black", marker="|")
plt.xlabel("Time (s)")
plt.ylabel("Cell ID")
plt.title("Spike Raster Plot")
plt.ylim(-0.5, len(spike_trains) - 0.5)
plt.xlim(0, times[-1])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. All Pre-Configured Examples
#
# The simulation subpackage provides several pre-configured session types for common experimental paradigms.

# %% [markdown]
# ### 4.1 Open Field Session
#
# Standard 2D arena with place cells and random exploration:

# %%
open_field = open_field_session(
    duration=10.0,
    arena_size=100.0,
    bin_size=2.0,
    n_place_cells=15,
    seed=100,
)

print(f"Open field: {len(open_field.spike_trains)} cells, {open_field.env.n_bins} bins")

# %% [markdown]
# ### 4.2 Linear Track Session
#
# 1D linear track with lap-based trajectory:

# %%
linear_track = linear_track_session(
    duration=10.0,
    track_length=200.0,
    bin_size=1.0,
    n_place_cells=12,
    n_laps=5,
    seed=101,
)

print(
    f"Linear track: {len(linear_track.spike_trains)} cells, {linear_track.env.n_bins} bins"
)
print(f"  Track is 1D: {linear_track.env.is_1d}")

# %% [markdown]
# ### 4.3 T-Maze Alternation Session
#
# Graph-based T-maze with structured lap trajectories:

# %%
tmaze = tmaze_alternation_session(
    duration=10.0,
    n_trials=5,
    n_place_cells=15,
    seed=102,
)

print(f"T-maze: {len(tmaze.spike_trains)} cells, {tmaze.env.n_bins} bins")
print(f"  Trial choices: {tmaze.metadata['trial_choices']}")

# %% [markdown]
# ### 4.4 Boundary Cell Session
#
# Mixed population of boundary cells and place cells:

# %%
boundary_session = boundary_cell_session(
    duration=10.0,
    arena_shape="square",
    arena_size=100.0,
    n_boundary_cells=10,
    n_place_cells=10,
    seed=103,
)

print(f"Boundary session: {len(boundary_session.spike_trains)} cells")
print("  Cell types in ground truth:")
for i in range(min(3, len(boundary_session.spike_trains))):
    cell_type = boundary_session.ground_truth[f"cell_{i}"]["cell_type"]
    print(f"    Cell {i}: {cell_type}")

# %% [markdown]
# ### 4.5 Grid Cell Session
#
# Grid cells with hexagonal periodic firing patterns (2D only):

# %%
grid_session = grid_cell_session(
    duration=10.0,
    arena_size=150.0,
    grid_spacing=50.0,
    n_grid_cells=12,
    seed=104,
)

print(f"Grid session: {len(grid_session.spike_trains)} cells")
print(f"  Grid spacing: {grid_session.metadata['grid_spacing']} cm")

# %% [markdown]
# ## 5. Validation Workflow
#
# The simulation subpackage provides automated validation tools to compare detected spatial fields against ground truth.

# %% [markdown]
# ### Validate Simulation Against Ground Truth
#
# Use `validate_simulation()` to automatically compare detected place fields to true parameters:

# %%
# Validate the open field session
report = validate_simulation(
    session=open_field,
    smoothing_method="diffusion_kde",  # Use boundary-aware place field detection
    show_plots=False,  # Set to True to see diagnostic plots
)

# Print summary report
print(report["summary"])

# Check if validation passed
if report["passed"]:
    print("\n✓ Validation PASSED - Place field detection is working correctly!")
else:
    print("\n✗ Validation FAILED - Check detection parameters")

# %% [markdown]
# ### Examine Validation Metrics
#
# The validation report contains detailed metrics for each cell:

# %%
print("Validation metrics:")
print(f"  Center errors: {report['center_errors'][:5]} cm (first 5 cells)")
print(f"  Mean center error: {np.mean(report['center_errors']):.2f} cm")
print(f"  Mean correlation: {np.mean(report['correlations']):.3f}")

# Visualize error distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Center errors
axes[0].hist(report["center_errors"], bins=10, edgecolor="black", alpha=0.7)
axes[0].axvline(
    np.mean(report["center_errors"]),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {np.mean(report['center_errors']):.2f} cm",
)
axes[0].set_xlabel("Center error (cm)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Center Errors")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Correlations
axes[1].hist(report["correlations"], bins=10, edgecolor="black", alpha=0.7)
axes[1].axvline(
    np.mean(report["correlations"]),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {np.mean(report['correlations']):.3f}",
)
axes[1].set_xlabel("Rate map correlation")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of Rate Map Correlations")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Customization Examples
#
# The low-level API allows for sophisticated customization of neural models using condition functions and custom parameters.

# %% [markdown]
# ### 6.1 Direction-Selective Place Cell
#
# Create a place cell that only fires when the animal is moving in a specific direction:

# %%
# Create 1D environment for direction selectivity demo
track_data = np.linspace(0, 200, 200).reshape(-1, 1)
track_env = Environment.from_samples(track_data, bin_size=1.0)
track_env.units = "cm"

# Generate lap-based trajectory (back and forth)
lap_positions, lap_times = simulate_trajectory_laps(
    track_env,
    n_laps=5,
    speed_mean=15.0,
    speed_std=3.0,
    sampling_frequency=100.0,
    seed=200,
    return_metadata=False,
)


# Define direction-selective condition: fires only when moving rightward
def rightward_only(positions, times):
    """Condition function that returns True when velocity > 0."""
    velocity = np.gradient(positions[:, 0], times)
    return velocity > 0


# Create directional place cell at track center
pc_right = PlaceCellModel(
    track_env,
    center=np.array([100.0]),  # Middle of track
    width=15.0,
    max_rate=25.0,
    baseline_rate=0.1,
    condition=rightward_only,  # Only fires when moving right
    seed=200,
)

# Generate spikes
spikes_right = generate_poisson_spikes(
    firing_rate=pc_right.firing_rate(lap_positions, lap_times),
    times=lap_times,
    refractory_period=0.002,
    seed=200,
)

print(f"Direction-selective cell: {len(spikes_right)} spikes")

# Visualize: spikes should only occur during rightward runs
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Position over time
axes[0].plot(lap_times, lap_positions[:, 0], "b-", linewidth=0.5)
axes[0].set_ylabel("Position (cm)")
axes[0].set_title("Trajectory on Linear Track")
axes[0].grid(True, alpha=0.3)

# Spike times
axes[1].scatter(spikes_right, np.ones_like(spikes_right), s=10, c="red", marker="|")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Spikes")
axes[1].set_title("Direction-Selective Spikes (Rightward Only)")
axes[1].set_ylim([0.5, 1.5])
axes[1].set_yticks([])
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.2 Speed-Gated Place Cell
#
# Create a place cell that only fires when the animal is moving above a speed threshold:


# %%
# Define speed threshold condition
def high_speed_only(positions, times, threshold=10.0):
    """Condition function that returns True when speed > threshold."""
    velocity = np.gradient(positions, axis=0) / np.gradient(times)[:, np.newaxis]
    speed = np.linalg.norm(velocity, axis=1)
    return speed > threshold


# Create speed-gated place cell
pc_speed = PlaceCellModel(
    env,
    center=np.array([50.0, 50.0]),
    width=12.0,
    max_rate=30.0,
    baseline_rate=0.1,
    condition=lambda pos, t: high_speed_only(
        pos, t, threshold=10.0
    ),  # Only fires at high speed
    seed=201,
)

# Generate new trajectory for this demo
speed_positions, speed_times = simulate_trajectory_ou(
    env,
    duration=10.0,
    speed_mean=12.0,  # Higher mean speed
    speed_std=6.0,  # High variability
    coherence_time=0.5,
    seed=201,
)

# Generate spikes
spikes_speed = generate_poisson_spikes(
    firing_rate=pc_speed.firing_rate(speed_positions, speed_times),
    times=speed_times,
    refractory_period=0.002,
    seed=201,
)

# Compute actual speed for visualization
velocity = (
    np.gradient(speed_positions, axis=0) / np.gradient(speed_times)[:, np.newaxis]
)
speed = np.linalg.norm(velocity, axis=1)

print(f"Speed-gated cell: {len(spikes_speed)} spikes")
print(f"Mean speed: {np.mean(speed):.2f} cm/s")

# Visualize speed profile and spikes
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Speed over time
axes[0].plot(speed_times, speed, "b-", linewidth=1)
axes[0].axhline(10.0, color="red", linestyle="--", linewidth=2, label="Speed threshold")
axes[0].set_ylabel("Speed (cm/s)")
axes[0].set_title("Running Speed Over Time")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Spike times
axes[1].scatter(spikes_speed, np.ones_like(spikes_speed), s=10, c="red", marker="|")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Spikes")
axes[1].set_title("Speed-Gated Spikes (Speed > 10 cm/s)")
axes[1].set_ylim([0.5, 1.5])
axes[1].set_yticks([])
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.3 Custom Boundary Cell
#
# Create a boundary vector cell that responds to a specific wall:

# %%
# Create boundary cell tuned to south wall
bc_south = BoundaryCellModel(
    env,
    preferred_distance=10.0,  # Fires 10 cm from wall
    distance_tolerance=5.0,
    preferred_direction=-np.pi / 2,  # South (negative y)
    direction_tolerance=np.pi / 6,  # ±30 degrees
    max_rate=20.0,
    baseline_rate=0.1,
    distance_metric="geodesic",  # Use graph-based distance
)

# Generate spikes
bc_positions, bc_times = simulate_trajectory_ou(
    env,
    duration=10.0,
    speed_mean=8.0,
    coherence_time=0.7,
    seed=202,
)

spikes_boundary = generate_poisson_spikes(
    firing_rate=bc_south.firing_rate(bc_positions, bc_times),
    times=bc_times,
    refractory_period=0.002,
    seed=202,
)

print(f"Boundary vector cell: {len(spikes_boundary)} spikes")
print("Ground truth:")
print(f"  Preferred distance: {bc_south.ground_truth['preferred_distance']} cm")
print(
    f"  Preferred direction: {bc_south.ground_truth['preferred_direction']:.2f} rad ({np.degrees(bc_south.ground_truth['preferred_direction']):.0f}°)"
)

# %% [markdown]
# ## 7. Performance Tips
#
# Best practices for efficient simulations:

# %% [markdown]
# ### 7.1 Choose Appropriate Distance Metrics
#
# - **Euclidean** (fast): Use for open field environments without barriers
#   - ~10 ms for 6000 positions
# - **Geodesic** (slow but accurate): Use for complex environments with walls
#   - ~100x slower than Euclidean
#   - Precomputes distance field once in `__init__()`

# %%
# Example: Euclidean vs Geodesic performance

# Create test environment
test_env = Environment.from_samples(arena_samples, bin_size=2.0)
test_env.units = "cm"
test_positions = positions[:1000]  # Use 1000 positions for timing

# Euclidean distance (fast)
pc_euclidean = PlaceCellModel(
    test_env, center=np.array([50.0, 50.0]), distance_metric="euclidean"
)
start = time.time()
rates_euclidean = pc_euclidean.firing_rate(test_positions)
time_euclidean = time.time() - start

# Geodesic distance (slower)
pc_geodesic = PlaceCellModel(
    test_env, center=np.array([50.0, 50.0]), distance_metric="geodesic"
)
start = time.time()
rates_geodesic = pc_geodesic.firing_rate(test_positions)
time_geodesic = time.time() - start

print("Performance comparison (1000 positions):")
print(f"  Euclidean: {time_euclidean * 1000:.2f} ms")
print(f"  Geodesic: {time_geodesic * 1000:.2f} ms")
print(f"  Speedup: {time_geodesic / time_euclidean:.1f}x faster with Euclidean")

# %% [markdown]
# ### 7.2 Use Seeds for Reproducibility
#
# Always use seeds for reproducible simulations:

# %%
# Reproducible simulation
session1 = open_field_session(duration=5.0, n_place_cells=10, seed=999)
session2 = open_field_session(duration=5.0, n_place_cells=10, seed=999)

# Check reproducibility
spikes_match = all(
    np.allclose(s1, s2)
    for s1, s2 in zip(session1.spike_trains, session2.spike_trains, strict=True)
)
print(f"Identical sessions with same seed: {spikes_match}")

# %% [markdown]
# ### 7.3 Optimize Trajectory Duration and Resolution
#
# - **Duration**: Longer simulations provide better coverage but take more time
#   - 60-180s typical for open field
#   - 120-300s for grid cells (need more coverage)
# - **Time step (dt)**: Balance accuracy vs performance
#   - 0.01s (10 ms) is usually sufficient
#   - Smaller dt needed for fast movements or high frequencies

# %%
# Example: Different durations
durations = [5.0, 10.0, 30.0]

for dur in durations:
    sess = open_field_session(duration=dur, n_place_cells=10, seed=300)
    n_spikes = sum(len(st) for st in sess.spike_trains)
    print(
        f"Duration {dur:5.1f}s: {n_spikes:5d} total spikes, {len(sess.times):6d} time points"
    )

# %% [markdown]
# ### 7.4 Disable Progress Bars for Batch Processing

# %%
# When running many simulations, disable progress bars
sessions = []
for i in range(3):
    sess = simulate_session(
        env,
        duration=5.0,
        n_cells=10,
        cell_type="place",
        show_progress=False,  # Cleaner output
        seed=400 + i,
    )
    sessions.append(sess)

print(f"Generated {len(sessions)} sessions in batch")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
#
# 1. **Quick start**: Pre-configured sessions (`open_field_session()`, etc.)
# 2. **Low-level API**: Manual trajectory + models + spikes for fine control
# 3. **All examples**: Five pre-configured session types for different paradigms
# 4. **Validation**: Automated comparison to ground truth with `validate_simulation()`
# 5. **Customization**: Direction-selective, speed-gated, and custom boundary cells
# 6. **Performance**: Best practices for efficient simulations
#
# ### Key Takeaways
#
# - Use **high-level API** for most applications (faster, cleaner)
# - Use **low-level API** for custom models and fine-grained control
# - Always **set seeds** for reproducible results
# - Choose **Euclidean distance** for speed, **geodesic** for accuracy
# - Use **validation** to verify neurospatial's analysis functions
#
# ### Next Steps
#
# - See other notebooks for examples of analysis workflows
# - Consult API documentation for detailed parameter descriptions
# - Experiment with custom condition functions for specialized models
