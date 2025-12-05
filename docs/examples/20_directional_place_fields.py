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
# # Directional Place Fields
#
# This notebook demonstrates how to compute direction-conditioned place fields, enabling
# analysis of directional tuning in place cells.
#
# **Estimated time**: 20-25 minutes
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# - Understand when place cells have directional tuning
# - Use `goal_pair_direction_labels` for trialized tasks (T-maze, linear track)
# - Use `heading_direction_labels` for open field exploration
# - Compute directional place fields with `compute_directional_place_fields`
# - Quantify directionality using a simple index formula
# - Visualize outbound vs inbound place fields
#
# ## Background
#
# Place cells in the hippocampus fire when an animal is at a specific location, but many
# place cells also show **directional tuning** - they fire differently depending on which
# direction the animal is traveling. This is especially pronounced on linear tracks where
# outbound (away from home) and inbound (returning home) runs can produce different
# firing patterns.
#
# Directional place fields are important for:
# - Understanding spatial memory encoding
# - Analyzing path integration
# - Studying prospective and retrospective coding
# - Investigating navigation strategies

# %%
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

# Neurospatial imports
from neurospatial import (
    Environment,
    compute_directional_place_fields,
    compute_place_field,
    goal_pair_direction_labels,
    heading_direction_labels,
    segment_trials,
)

# Simulation imports
from neurospatial.simulation import (
    PlaceCellModel,
    generate_poisson_spikes,
    simulate_trajectory_ou,
)

# Set random seed for reproducibility
np.random.seed(42)

# %% [markdown]
# ---
#
# ## Part 1: Linear Track with Goal-Pair Direction Labels
#
# On a linear track, animals typically run back and forth between two ends (e.g., home
# and goal zones). We can label each time point based on which direction the animal
# is running using `goal_pair_direction_labels()`.
#
# ### Workflow:
# 1. Create a linear track environment with start/goal regions
# 2. Simulate a trajectory with back-and-forth runs
# 3. Segment into trials using `segment_trials()`
# 4. Generate direction labels with `goal_pair_direction_labels()`
# 5. Compute directional place fields
# 6. Compare outbound vs inbound fields

# %% [markdown]
# ### Create Linear Track Environment

# %%
# Create a 1D-like linear track (narrow 2D corridor)
track_length = 100.0  # cm
track_width = 10.0  # cm
bin_size = 5.0  # cm

# Create regular grid for the track
x = np.linspace(0, track_length, int(track_length / bin_size) + 1)
y = np.linspace(0, track_width, int(track_width / bin_size) + 1)
xx, yy = np.meshgrid(x, y)
track_data = np.column_stack([xx.ravel(), yy.ravel()])

# Create environment
track_env = Environment.from_samples(track_data, bin_size=bin_size)
track_env.units = "cm"
track_env.frame = "linear_track"

# Add regions at each end (as small rectangles covering first/last few bins)
# Using Shapely Polygon regions so segment_trials can properly identify entries
home_polygon = Polygon(
    [
        (0, 0),
        (15, 0),
        (15, track_width),
        (0, track_width),
    ]
)
goal_polygon = Polygon(
    [
        (track_length - 15, 0),
        (track_length, 0),
        (track_length, track_width),
        (track_length - 15, track_width),
    ]
)
track_env.regions.add("home", polygon=home_polygon)
track_env.regions.add("goal", polygon=goal_polygon)

print(f"Linear track: {track_env.n_bins} bins, {track_length} cm long")
print(f"Regions: {list(track_env.regions.keys())}")

# %% [markdown]
# ### Simulate Back-and-Forth Trajectory
#
# We'll simulate a trajectory where the animal runs from home to goal and back,
# mimicking typical linear track behavior.

# %%
# Generate trajectory on the track
duration = 300.0  # seconds
positions, times = simulate_trajectory_ou(
    track_env,
    duration=duration,
    dt=0.02,  # 50 Hz sampling
    speed_mean=15.0,  # 15 cm/s (typical rat running speed)
    speed_std=5.0,
    coherence_time=0.5,
    boundary_mode="reflect",  # Bounces off walls
    seed=42,
)

print(f"Generated {len(positions)} trajectory points over {duration:.0f}s")
print(
    f"Position range: x=[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}] cm"
)

# Visualize trajectory
fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
ax.scatter(positions[::10, 0], positions[::10, 1], c=times[::10], cmap="viridis", s=5)
ax.axvline(10, color="blue", linestyle="--", label="Home region")
ax.axvline(90, color="red", linestyle="--", label="Goal region")
ax.set_xlabel("X Position (cm)", fontsize=12)
ax.set_ylabel("Y Position (cm)", fontsize=12)
ax.set_title("Linear Track Trajectory (color = time)", fontsize=14, fontweight="bold")
ax.legend()
ax.set_aspect("equal")
plt.show()

# %% [markdown]
# ### Segment Trajectory into Trials

# %%
# Get bin indices for trajectory (dedup=False keeps one bin per timepoint)
trajectory_bins = track_env.bin_sequence(times, positions, dedup=False)

# Segment into trials: home -> goal and goal -> home
trials = segment_trials(
    trajectory_bins,
    times,
    track_env,
    start_region="home",
    end_regions=["goal"],
    min_duration=1.0,  # At least 1 second per trial
    max_duration=30.0,  # Maximum 30 seconds per trial
)

# Also get trials going the other direction
trials_reverse = segment_trials(
    trajectory_bins,
    times,
    track_env,
    start_region="goal",
    end_regions=["home"],
    min_duration=1.0,
    max_duration=30.0,
)

# Combine all trials
all_trials = trials + trials_reverse

print("\nSegmented trials:")
print(f"  Home -> Goal: {len(trials)} trials")
print(f"  Goal -> Home: {len(trials_reverse)} trials")
print(f"  Total: {len(all_trials)} trials")

# Show first few trials
for i, t in enumerate(all_trials[:5]):
    status = "success" if t.success else "timeout"
    print(f"  Trial {i + 1}: {t.start_region} -> {t.end_region} ({status})")

# %% [markdown]
# ### Generate Direction Labels

# %%
# Generate direction labels using goal_pair_direction_labels
direction_labels = goal_pair_direction_labels(times, all_trials)

# Count labels
unique_labels, counts = np.unique(direction_labels, return_counts=True)
print("\nDirection label distribution:")
for label, count in zip(unique_labels, counts, strict=True):
    pct = count / len(direction_labels) * 100
    print(f"  {label}: {count} samples ({pct:.1f}%)")

# %% [markdown]
# ### Create Directional Place Cell
#
# We'll create a place cell that fires differently depending on running direction.
# This mimics real directional place cells observed in hippocampus.

# %%
# Create a place cell at the center of the track
# It will have different field shapes for outbound vs inbound directions
field_center = np.array([50.0, track_width / 2])

# Outbound direction: field slightly ahead of center
pc_forward = PlaceCellModel(
    track_env,
    center=field_center + np.array([10.0, 0.0]),  # Shifted right
    width=12.0,
    max_rate=15.0,
    baseline_rate=0.5,
    distance_metric="euclidean",
    seed=42,
)

# Inbound direction: field slightly behind center
pc_reverse = PlaceCellModel(
    track_env,
    center=field_center - np.array([10.0, 0.0]),  # Shifted left
    width=12.0,
    max_rate=10.0,  # Lower rate
    baseline_rate=0.5,
    distance_metric="euclidean",
    seed=43,
)

# Generate spikes based on direction
# Outbound trials (home -> goal): use outbound field
# Inbound trials (goal -> home): use inbound field
forward_mask = direction_labels == "home\u2192goal"
reverse_mask = direction_labels == "goal\u2192home"

# Compute firing rates based on direction
firing_rates = np.zeros(len(times))
firing_rates[forward_mask] = pc_forward.firing_rate(
    positions[forward_mask], times[forward_mask]
)
firing_rates[reverse_mask] = pc_reverse.firing_rate(
    positions[reverse_mask], times[reverse_mask]
)

# Generate spikes from combined firing rate
spike_times = generate_poisson_spikes(
    firing_rates,
    times,
    refractory_period=0.002,
    seed=44,
)

print(f"Generated {len(spike_times)} spikes")
print(f"Mean firing rate: {len(spike_times) / times[-1]:.2f} Hz")

# %% [markdown]
# ### Compute Directional Place Fields

# %%
# Compute directional place fields
directional_fields = compute_directional_place_fields(
    track_env,
    spike_times,
    times,
    positions,
    direction_labels,
    smoothing_method="diffusion_kde",
    bandwidth=8.0,
)

print("\nDirectional Place Fields:")
print(f"  Labels: {directional_fields.labels}")
for label in directional_fields.labels:
    field = directional_fields.fields[label]
    print(f"  {label}: peak={np.nanmax(field):.2f} Hz, mean={np.nanmean(field):.2f} Hz")

# %% [markdown]
# ### Visualize Outbound vs Inbound Fields

# %%
# Get fields for each direction
outbound_field = directional_fields.fields["home\u2192goal"]
inbound_field = directional_fields.fields["goal\u2192home"]

# Also compute overall (non-directional) place field for comparison
overall_field = compute_place_field(
    track_env,
    spike_times,
    times,
    positions,
    smoothing_method="diffusion_kde",
    bandwidth=8.0,
)

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)

# Outbound field
track_env.plot_field(
    outbound_field,
    ax=axes[0],
    cmap="hot",
    colorbar_label="Firing Rate (Hz)",
)
axes[0].set_title("Outbound (Home \u2192 Goal)", fontsize=14, fontweight="bold")
axes[0].axvline(60, color="cyan", linestyle="--", alpha=0.7, label="True center")
axes[0].legend()

# Inbound field
track_env.plot_field(
    inbound_field,
    ax=axes[1],
    cmap="hot",
    colorbar_label="Firing Rate (Hz)",
)
axes[1].set_title("Inbound (Goal \u2192 Home)", fontsize=14, fontweight="bold")
axes[1].axvline(40, color="cyan", linestyle="--", alpha=0.7, label="True center")
axes[1].legend()

# Overall field
track_env.plot_field(
    overall_field,
    ax=axes[2],
    cmap="hot",
    colorbar_label="Firing Rate (Hz)",
)
axes[2].set_title("Overall (Both Directions)", fontsize=14, fontweight="bold")

plt.suptitle("Directional Place Fields on Linear Track", fontsize=16, fontweight="bold")
plt.show()

# %% [markdown]
# ### Quantify Directionality
#
# A simple directionality index computes per-bin preference: (outbound - inbound) / (outbound + inbound).
# - Values near +1: bin fires mainly during outbound runs
# - Values near -1: bin fires mainly during inbound runs
# - Values near 0: equal firing in both directions

# %%
# Compute directional index: (outbound - inbound) / (outbound + inbound + eps)
eps = 1e-9  # Prevent division by zero
dir_index = (outbound_field - inbound_field) / (outbound_field + inbound_field + eps)

print("\nDirectional Index Statistics:")
print(f"  Range: [{np.nanmin(dir_index):.2f}, {np.nanmax(dir_index):.2f}]")
print(f"  Mean: {np.nanmean(dir_index):.2f}")
print(f"  Std: {np.nanstd(dir_index):.2f}")

# Visualize directional index
fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
track_env.plot_field(
    dir_index,
    ax=ax,
    cmap="RdBu_r",  # Red = forward, Blue = reverse
    vmin=-1,
    vmax=1,
    colorbar_label="Directional Index",
)
ax.set_title(
    "Directional Index (Red=Outbound, Blue=Inbound)", fontsize=14, fontweight="bold"
)
plt.show()

# %% [markdown]
# The directional index map shows clear separation: the right side of the track (x > 50 cm)
# has positive values (outbound-preferring) while the left side has negative values
# (inbound-preferring). This matches our simulation where the outbound field was shifted
# right and the inbound field was shifted left.

# %% [markdown]
# ---
#
# ## Part 2: Open Field with Heading Direction Labels
#
# In open field environments, animals don't have predefined start/goal locations.
# Instead, we can label direction based on the animal's **heading** - the direction
# they are moving at each moment.
#
# ### Workflow:
# 1. Create an open field environment
# 2. Simulate random exploration
# 3. Generate heading-based direction labels with `heading_direction_labels()`
# 4. Compute directional place fields for each heading sector
# 5. Visualize direction-tuned firing

# %% [markdown]
# ### Create Open Field Environment

# %%
# Create a 2D open field arena
arena_size = 80.0  # cm
bin_size = 4.0  # cm

# Create regular grid
n_grid = 20
x = np.linspace(0, arena_size, n_grid)
y = np.linspace(0, arena_size, n_grid)
xx, yy = np.meshgrid(x, y)
arena_data = np.column_stack([xx.ravel(), yy.ravel()])

# Create environment
arena_env = Environment.from_samples(arena_data, bin_size=bin_size)
arena_env.units = "cm"
arena_env.frame = "open_field"

print(f"Open field: {arena_env.n_bins} bins, {arena_size}x{arena_size} cm")

# %% [markdown]
# ### Simulate Open Field Exploration

# %%
# Generate random exploration trajectory
arena_duration = 600.0  # 10 minutes
arena_positions, arena_times = simulate_trajectory_ou(
    arena_env,
    duration=arena_duration,
    dt=0.02,  # 50 Hz
    speed_mean=10.0,  # cm/s
    speed_std=3.0,
    coherence_time=0.8,  # Smooth trajectories
    boundary_mode="reflect",
    seed=100,
)

print(f"Generated {len(arena_positions)} trajectory points over {arena_duration:.0f}s")

# Visualize trajectory
fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
ax.scatter(
    arena_positions[::20, 0],
    arena_positions[::20, 1],
    c=arena_times[::20],
    cmap="viridis",
    s=5,
    alpha=0.5,
)
ax.set_xlabel("X Position (cm)", fontsize=12)
ax.set_ylabel("Y Position (cm)", fontsize=12)
ax.set_title("Open Field Exploration (color = time)", fontsize=14, fontweight="bold")
ax.set_aspect("equal")
plt.show()

# %% [markdown]
# ### Generate Heading Direction Labels
#
# `heading_direction_labels()` bins the animal's heading angle into sectors.
# By default, it uses 8 sectors (45 degrees each): 0-45, 45-90, ..., 315-360.

# %%
# Generate heading-based direction labels
# Using 4 directions for clearer visualization (N, E, S, W)
heading_labels = heading_direction_labels(
    arena_positions,
    arena_times,
    n_directions=4,  # 4 cardinal directions
    min_speed=0.5,  # Label as "stationary" if speed < 0.5 cm/s (low threshold)
)

# Count labels
unique_labels, counts = np.unique(heading_labels, return_counts=True)
print("\nHeading direction distribution:")
for label, count in zip(unique_labels, counts, strict=True):
    pct = count / len(heading_labels) * 100
    print(f"  {label}: {count} samples ({pct:.1f}%)")

# %% [markdown]
# ### Create Direction-Tuned Place Cell
#
# We'll create a place cell that fires preferentially when the animal is moving
# in a specific direction (eastward, i.e., +X direction).

# %%
# Create a place cell in the center of the arena
# It fires most strongly when moving eastward (0-90 degrees)
arena_field_center = np.array([40.0, 40.0])

# Base place cell model
pc_base = PlaceCellModel(
    arena_env,
    center=arena_field_center,
    width=15.0,
    max_rate=12.0,
    baseline_rate=0.2,
    distance_metric="euclidean",
    seed=200,
)

# Compute base firing rate
base_rate = pc_base.firing_rate(arena_positions, arena_times)

# Modulate by heading direction
# East (0-90 degrees) gets full rate, opposite direction gets reduced rate
# Compute heading angles
velocity = np.diff(arena_positions, axis=0) / np.diff(arena_times)[:, np.newaxis]
heading_angles = np.arctan2(velocity[:, 1], velocity[:, 0])
heading_angles = np.concatenate([[0], heading_angles])  # Pad first point

# East is 0 radians, modulate by cosine of heading
# cos(0) = 1 (east), cos(pi) = -1 (west)
direction_modulation = 0.5 + 0.5 * np.cos(heading_angles)  # Range [0, 1]

# Apply modulation to firing rate
modulated_rate = base_rate * direction_modulation

# Generate spikes
arena_spike_times = generate_poisson_spikes(
    modulated_rate,
    arena_times,
    refractory_period=0.002,
    seed=201,
)

print(f"Generated {len(arena_spike_times)} spikes")
print(f"Mean firing rate: {len(arena_spike_times) / arena_times[-1]:.2f} Hz")

# %% [markdown]
# ### Compute Heading-Based Directional Place Fields

# %%
# Compute directional place fields by heading
directional_fields_heading = compute_directional_place_fields(
    arena_env,
    arena_spike_times,
    arena_times,
    arena_positions,
    heading_labels,
    smoothing_method="diffusion_kde",
    bandwidth=10.0,
)

print("\nHeading-Based Directional Place Fields:")
print(f"  Directions: {directional_fields_heading.labels}")
for label in directional_fields_heading.labels:
    field = directional_fields_heading.fields[label]
    print(f"  {label}: peak={np.nanmax(field):.2f} Hz, mean={np.nanmean(field):.2f} Hz")

# %% [markdown]
# ### Visualize Heading-Tuned Fields

# %%
# Create 2x2 visualization of directional fields
fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)

# Define direction order for visualization (matching compass layout)
direction_order = [
    ("\u221290\u201390\u00b0", "North (Up)"),  # -90 to 90
    ("0\u201390\u00b0", "East (Right)"),  # 0 to 90
    ("90\u2013180\u00b0", "South (Down)"),  # 90 to 180
    ("\u2212180\u2013\u221290\u00b0", "West (Left)"),  # -180 to -90
]

# Map to actual labels in our data
label_map = {
    "\u221290\u20130\u00b0": ("East/North", (0, 0)),
    "0\u201390\u00b0": ("East/South", (0, 1)),
    "90\u2013180\u00b0": ("West/South", (1, 1)),
    "\u2212180\u2013\u221290\u00b0": ("West/North", (1, 0)),
}

# Get max rate for consistent color scaling (skip NaN-only fields like stationary)
all_rates = [
    directional_fields_heading.fields[lbl]
    for lbl in directional_fields_heading.labels
    if not np.all(np.isnan(directional_fields_heading.fields[lbl]))
]
vmax = max(np.nanmax(r) for r in all_rates) if all_rates else 1.0

for label in directional_fields_heading.labels:
    field = directional_fields_heading.fields[label]

    # Skip stationary or other labels that are all NaN
    if np.all(np.isnan(field)) or "stationary" in label.lower():
        continue

    # Determine subplot position based on label
    # Parse the label to get approximate direction
    if "0" in label and "90" in label and "-" not in label.replace("°", ""):
        ax = axes[0, 1]
        title = f"East ({label})"
    elif "90" in label and "180" in label:
        ax = axes[1, 0]
        title = f"South/West ({label})"
    elif "-180" in label:
        ax = axes[1, 1]
        title = f"West ({label})"
    else:
        ax = axes[0, 0]
        title = f"North ({label})"

    arena_env.plot_field(
        field,
        ax=ax,
        cmap="hot",
        vmax=vmax,
        colorbar_label="Firing Rate (Hz)",
    )
    ax.scatter(
        arena_field_center[0],
        arena_field_center[1],
        s=200,
        c="cyan",
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        zorder=10,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")

plt.suptitle(
    "Heading-Tuned Place Fields\n(cell fires more when moving East)",
    fontsize=16,
    fontweight="bold",
)
plt.show()

# %% [markdown]
# The visualization shows that the place cell fires most strongly when the animal
# is moving eastward (toward +X), with progressively lower rates as the heading
# rotates away from east.

# %% [markdown]
# ### Compare East vs West Directionality

# %%
# Get East and West fields for comparison
# Find the labels containing east and west directions
# East direction is 0-90 degrees, West is -180 to -90 degrees
east_label = None
west_label = None
for label in directional_fields_heading.labels:
    if "0" in label and "90" in label and "-" not in label.replace("°", ""):
        east_label = label  # "0–90°"
    elif "-180" in label:
        west_label = label  # "-180–-90°"

if east_label and west_label:
    east_field = directional_fields_heading.fields[east_label]
    west_field = directional_fields_heading.fields[west_label]

    # Compute directional index (east vs west): (east - west) / (east + west + eps)
    ew_index = (east_field - west_field) / (east_field + west_field + 1e-9)

    print("\nEast vs West Directional Index:")
    print(f"  Range: [{np.nanmin(ew_index):.2f}, {np.nanmax(ew_index):.2f}]")
    print(f"  Mean: {np.nanmean(ew_index):.2f}")
    print("  (Positive = East-preferring, Negative = West-preferring)")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    arena_env.plot_field(
        east_field, ax=axes[0], cmap="hot", colorbar_label="Firing Rate (Hz)"
    )
    axes[0].set_title(f"East ({east_label})", fontsize=13, fontweight="bold")

    arena_env.plot_field(
        west_field, ax=axes[1], cmap="hot", colorbar_label="Firing Rate (Hz)"
    )
    axes[1].set_title(f"West ({west_label})", fontsize=13, fontweight="bold")

    arena_env.plot_field(
        ew_index,
        ax=axes[2],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        colorbar_label="Directional Index",
    )
    axes[2].set_title("East vs West Index", fontsize=13, fontweight="bold")

    plt.suptitle("East vs West Directional Comparison", fontsize=16, fontweight="bold")
    plt.show()

# %% [markdown]
# ---
#
# ## Summary
#
# In this notebook, we demonstrated:
#
# 1. **Goal-Pair Direction Labels** (`goal_pair_direction_labels`):
#    - For trialized tasks (T-maze, linear track, Y-maze)
#    - Labels based on start/end regions of trials
#    - Format: "start\u2192end" (e.g., "home\u2192goal")
#
# 2. **Heading Direction Labels** (`heading_direction_labels`):
#    - For open field exploration
#    - Labels based on instantaneous heading angle
#    - Bins heading into N sectors (default: 8)
#    - Marks slow periods as "stationary"
#
# 3. **Directional Place Fields** (`compute_directional_place_fields`):
#    - Computes separate place field for each direction label
#    - Reuses `compute_place_field` internally
#    - Returns `DirectionalPlaceFields` dataclass with fields and labels
#
# 4. **Directional Index** (simple formula):
#    - Quantifies per-bin directionality: `(field_a - field_b) / (field_a + field_b + eps)`
#    - Range [-1, +1]: negative = B-preferring, positive = A-preferring
#
# ### Key Functions
#
# ```python
# from neurospatial import (
#     compute_directional_place_fields,
#     goal_pair_direction_labels,
#     heading_direction_labels,
# )
# ```
#
# ### Typical Workflow
#
# ```python
# # For trialized tasks
# trials = segment_trials(trajectory_bins, times, env, start_region="home", end_regions=["goal"])
# labels = goal_pair_direction_labels(times, trials)
# result = compute_directional_place_fields(env, spike_times, times, positions, labels)
#
# # For open fields
# labels = heading_direction_labels(positions, times, n_directions=8)
# result = compute_directional_place_fields(env, spike_times, times, positions, labels)
#
# # Quantify directionality (simple formula)
# outbound = result.fields["home→goal"]
# inbound = result.fields["goal→home"]
# index = (outbound - inbound) / (outbound + inbound + 1e-9)
# ```
#
# ### References
#
# - McNaughton et al. (1983): Directional place cells on linear tracks
# - Markus et al. (1995): Directional modulation in open fields
# - Battaglia et al. (2004): Local field potential and place cell directionality
# - Ainge et al. (2007): Hippocampal place cells encode prospective information

# %%
