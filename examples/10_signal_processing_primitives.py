# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Signal Processing Primitives
#
# **Estimated time**: 15-20 minutes
#
# **Prerequisites**: [09: differential operators](09_differential_operators.ipynb), [08: spike fields](08_spike_field_basics.ipynb).
#
# This notebook demonstrates the spatial signal processing primitives `neighbor_reduce()` and `graph_convolve()` for custom filtering and local aggregation on graph-based environments.
#
#
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# - Use `neighbor_reduce()` to perform local aggregation on spatial graphs
# - Analyze spatial coherence using neighborhood statistics
# - Apply custom convolution filters with `graph_convolve()` for spatial analysis
# - Distinguish neighborhood coverage from observed occupancy support
# - Construct balanced center-surround filters for spatial contrast
# - Assess local field variability for quality control
#
# **Topics covered**:
# - Local aggregation with `neighbor_reduce()`
# - Spatial coherence analysis
# - Custom convolution with `graph_convolve()`
# - Box filter for occupancy thresholding
# - Balanced center-surround contrast
# - Local field variability

# %%
import matplotlib.pyplot as plt
import numpy as np

from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rate
from neurospatial.ops.distance import geodesic_distance_matrix
from neurospatial.ops.graph import graph_convolve, neighbor_reduce
from neurospatial.simulation import generate_poisson_spikes

# Random seed for reproducibility
rng = np.random.default_rng(42)

# Set matplotlib style for presentations
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
    }
)

# %% [markdown]
# ## Part 1: Spatial Coherence with `neighbor_reduce()`
#
# Here coherence is the Pearson correlation of a map with neighboring rates.
# It is descriptive and depends on binning, support and prior smoothing.
# High coherence alone does not establish a place field or validate a cell.

# %%
# Generate 2D random walk in open field arena
sampling_rate = 20.0  # Hz
duration = 100.0  # seconds
n_samples = int(duration * sampling_rate)
times = np.arange(n_samples) / sampling_rate

# Arena size: 100x100 cm open field
arena_size = 100.0  # cm
arena_center = arena_size / 2

# Random walk parameters
step_size = 2.5  # cm per step
boundary_margin = 5.0  # cm from walls

# Initialize trajectory
positions = np.zeros((n_samples, 2))
positions[0] = [arena_center, arena_center]  # Start at center

# Generate random walk with wall reflection
for i in range(1, n_samples):
    # Random step direction
    angle = rng.uniform(0, 2 * np.pi)
    step = step_size * np.array([np.cos(angle), np.sin(angle)])

    # Propose new position
    new_pos = positions[i - 1] + step

    # Reflect at boundaries (with margin)
    for dim in range(2):
        if new_pos[dim] < boundary_margin:
            new_pos[dim] = boundary_margin + (boundary_margin - new_pos[dim])
        elif new_pos[dim] > (arena_size - boundary_margin):
            new_pos[dim] = (arena_size - boundary_margin) - (
                new_pos[dim] - (arena_size - boundary_margin)
            )

    positions[i] = new_pos

# Include the full known arena, including space the walk did not visit.
arena_edges = np.linspace(0.0, arena_size, 21)
env = Environment.from_grid_mask(np.ones((20, 20), bool), (arena_edges, arena_edges))
env.units = "cm"
print(f"Arena: {arena_size:.0f}x{arena_size:.0f} cm open field")

# Generate synthetic spike train with Gaussian place field
center = np.array([50.0, 50.0])
peak_rate = 10.0  # Hz
sigma = 15.0  # cm

# Compute instantaneous rate at each position
distances_to_center = np.linalg.norm(positions - center, axis=1)
instantaneous_rate = peak_rate * np.exp(-(distances_to_center**2) / (2 * sigma**2))

# An independent seed avoids correlating spike acceptance with the stimulus RNG.
spike_times = generate_poisson_spikes(instantaneous_rate, times, seed=7)

print(f"Environment: {env.n_bins} bins")
print(f"Generated {len(spike_times)} spikes")

# %%
# Compute firing rate field with diffusion KDE (recommended for visualization)
result = compute_spatial_rate(
    env, spike_times, times, positions, bandwidth=5.0, min_occupancy=0.5
)
firing_rate = result.firing_rate

# Compute neighbor average (for coherence)
neighbor_avg = neighbor_reduce(env, firing_rate, op="mean", include_self=False)

# Compute spatial coherence (Pearson correlation)
valid = result.spatial_support & np.isfinite(firing_rate) & np.isfinite(neighbor_avg)
coherence = np.corrcoef(firing_rate[valid], neighbor_avg[valid])[0, 1]

print(f"\nSpatial coherence: {coherence:.3f}")
print("Interpretation:")
print("  Correlation describes this smoothed map on jointly observed bins.")
print("  Smoothing itself increases coherence; this is not a cell-classification test.")

# %%
# Visualize: firing rate vs neighbor average
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# Firing rate
env.plot_field(
    firing_rate,
    ax=axes[0],
    cmap="hot",
    vmin=0,
    colorbar_label="Rate (Hz)",
)
axes[0].set_title("Firing Rate (Hz)", fontsize=14, weight="bold")

# Neighbor average
env.plot_field(
    neighbor_avg,
    ax=axes[1],
    cmap="hot",
    vmin=0,
    colorbar_label="Rate (Hz)",
)
axes[1].set_title("Neighbor Average (Hz)", fontsize=14, weight="bold")

# Scatter plot: firing rate vs neighbor average
axes[2].scatter(
    firing_rate[valid],
    neighbor_avg[valid],
    alpha=0.5,
    s=50,
    edgecolors="black",
    linewidths=0.5,
)
axes[2].set_xlabel("Firing Rate (Hz)", fontsize=12, weight="bold")
axes[2].set_ylabel("Neighbor Average (Hz)", fontsize=12, weight="bold")
axes[2].set_title(f"Spatial Coherence: {coherence:.3f}", fontsize=14, weight="bold")
axes[2].grid(True, alpha=0.3)

# Add diagonal reference line (if we have valid data)
if valid.sum() > 0:
    max_val = max(firing_rate[valid].max(), neighbor_avg[valid].max())
    axes[2].plot(
        [0, max_val], [0, max_val], "r--", linewidth=2, label="Equal rates (y=x)"
    )
    axes[2].legend()

plt.show()

# %% [markdown]
# The correlation describes neighboring map values, not tuning significance.
# Use identical smoothing and observation-support rules when comparing maps.

# %% [markdown]
# ## Part 2: Local Field Variability with `neighbor_reduce()`
#
# Compute local statistics to measure field consistency. High variability indicates place field edges or noisy regions.

# %%
# Compute local statistics using neighbor_reduce
neighbor_mean = neighbor_reduce(env, firing_rate, op="mean", include_self=False)
neighbor_std = neighbor_reduce(env, firing_rate, op="std", include_self=False)
neighbor_max = neighbor_reduce(env, firing_rate, op="max", include_self=False)

# Coefficient of variation (normalized variability)
epsilon = 1e-6
cv = neighbor_std / (neighbor_mean + epsilon)

# %%
# Visualize local statistics
fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

# Mean
env.plot_field(
    neighbor_mean,
    ax=axes[0, 0],
    cmap="hot",
    colorbar_label="Mean (Hz)",
)
axes[0, 0].set_title("Neighbor Mean (Hz)", fontsize=14, weight="bold")

# Std
env.plot_field(
    neighbor_std,
    ax=axes[0, 1],
    cmap="viridis",
    colorbar_label="Std (Hz)",
)
axes[0, 1].set_title("Neighbor Std (Hz)", fontsize=14, weight="bold")

# Max
env.plot_field(
    neighbor_max,
    ax=axes[1, 0],
    cmap="hot",
    colorbar_label="Max (Hz)",
)
axes[1, 0].set_title("Neighbor Max (Hz)", fontsize=14, weight="bold")

# Coefficient of variation
env.plot_field(
    cv,
    ax=axes[1, 1],
    cmap="coolwarm",
    vmin=0,
    vmax=2,
    colorbar_label="CV (std/mean)",
)
axes[1, 1].set_title("Coefficient of Variation", fontsize=14, weight="bold")

plt.show()

# %% [markdown]
# **Interpretation**:
# - **High CV (red)**: Variable firing rates among neighbors (place field edges)
# - **Low CV (blue)**: Consistent firing rates (place field center or background)

# %% [markdown]
# ## Part 3: Box Filter with `graph_convolve()`
#
# A box kernel averages the visited-bin indicator over a spatial neighborhood.
# This measures neighborhood coverage, not duration, uncertainty or reliability.

# %%
# Compute binary occupancy
occupancy = env.occupancy(times, positions, return_seconds=True)
binary_occupancy = (occupancy > 0).astype(float)


# Define box kernel (uniform within 10 cm)
def box_kernel(distances):
    """Uniform weight within radius, zero outside."""
    radius = 10.0  # cm
    return np.where(distances <= radius, 1.0, 0.0)


# Apply box filter
smoothed_occupancy = graph_convolve(env, binary_occupancy, box_kernel, normalize=True)

# Exploratory neighborhood-coverage screen; not an empirical support mask.
reliable_visited = smoothed_occupancy > 0.5

print(f"Binary occupancy: {binary_occupancy.sum():.0f} bins visited")
print(f"Neighborhood coverage >50%: {reliable_visited.sum():.0f} bins")
print(f"Observed time >=0.5 s: {(occupancy >= 0.5).sum()} bins (a separate criterion)")

# %%
# Visualize box filter effect
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# Binary occupancy
env.plot_field(
    binary_occupancy,
    ax=axes[0],
    cmap="gray",
    vmin=0,
    vmax=1,
    colorbar_label="Visited",
)
axes[0].set_title("Binary Occupancy", fontsize=14, weight="bold")

# Smoothed occupancy
env.plot_field(
    smoothed_occupancy,
    ax=axes[1],
    cmap="viridis",
    vmin=0,
    vmax=1,
    colorbar_label="Proportion",
)
axes[1].set_title("Box Filter (10 cm)", fontsize=14, weight="bold")

# Thresholded
env.plot_field(
    reliable_visited,
    ax=axes[2],
    cmap="gray",
    vmin=0,
    vmax=1,
    colorbar_label="Coverage >50%",
)
axes[2].set_title("Neighborhood Coverage >50%", fontsize=14, weight="bold")

plt.show()

# %% [markdown]
# A bin can have high neighborhood coverage without having been observed.
# Keep actual occupancy and `result.spatial_support` when selecting measured rates.

# %% [markdown]
# ## Part 4: Balanced Center-Surround Filtering with `graph_convolve()`
#
# Subtract broad and narrow Gaussian averages to measure local contrast.
# Both kernels must have equal total weight: subtracting unit-height Gaussians
# makes all off-center weights negative and does not remove a constant field.
# Normalize each row over observed source bins to respect this finite arena.

# %%
firing_rate_smooth = firing_rate
observed = np.isfinite(firing_rate_smooth)
distance_matrix = geodesic_distance_matrix(
    env.connectivity, env.n_bins, weight="distance"
)
center_weights = np.exp(-0.5 * (distance_matrix / 5.0) ** 2) * observed[None, :]
surround_weights = np.exp(-0.5 * (distance_matrix / 15.0) ** 2) * observed[None, :]
center_weights /= center_weights.sum(axis=1, keepdims=True)
surround_weights /= surround_weights.sum(axis=1, keepdims=True)
contrast_kernel = center_weights - surround_weights

# Balanced rows annihilate constant fields, including at the arena boundary.
np.testing.assert_allclose(contrast_kernel.sum(axis=1), 0.0, atol=1e-12)
np.testing.assert_allclose(
    graph_convolve(env, np.ones(env.n_bins), contrast_kernel, normalize=False),
    0.0,
    atol=1e-12,
)
edges = graph_convolve(env, firing_rate_smooth, contrast_kernel, normalize=False)
edges[~result.spatial_support] = np.nan
threshold_high = np.nanpercentile(edges, 95)
field_centers = np.isfinite(edges) & (edges > max(0.0, threshold_high))
print(f"Upper positive contrast tail: {field_centers.sum()} observed bins")

# %%
# Visualize edge detection
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# Original firing rate
env.plot_field(
    firing_rate_smooth,
    ax=axes[0],
    cmap="hot",
    vmin=0,
    colorbar_label="Rate (Hz)",
)
axes[0].set_title("Smoothed Firing Rate", fontsize=14, weight="bold")

# Edge response (Mexican hat)
# Use symmetric colormap centered at 0
vmax = np.nanmax(np.abs(edges))
env.plot_field(
    edges,
    ax=axes[1],
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax,
    colorbar_label="Response",
)
axes[1].set_title("Balanced Center-Surround\nContrast", fontsize=14, weight="bold")

# Detected field centers
env.plot_field(
    np.where(result.spatial_support, field_centers.astype(float), np.nan),
    ax=axes[2],
    cmap="gray",
    vmin=0,
    vmax=1,
    colorbar_label="High contrast",
)
axes[2].set_title("Exploratory Contrast Candidates", fontsize=14, weight="bold")

plt.show()

# %% [markdown]
# **Interpretation**:
# - **Positive (red)**: Narrow-neighborhood average exceeds the broad average
# - **Negative (blue)**: Narrow average is lower than its surroundings
# - **Near zero (white)**: Similar averages at both scales
#
# The upper-tail mask is an exploratory contrast screen, not detected biological
# place fields or their boundaries. NaN bins remain outside observation support.

# %% [markdown]
# ## Part 5: Comparison - `graph_convolve()` vs `env.smooth()`
#
# Compare custom Gaussian convolution with the built-in `env.smooth()` method.

# %%
# Create random field for comparison
random_field = rng.random(env.n_bins)

# Method 1: Custom Gaussian convolution
bandwidth = 7.5  # cm


def gaussian_kernel(distances):
    return np.exp(-(distances**2) / (2 * bandwidth**2))


result_convolve = graph_convolve(env, random_field, gaussian_kernel, normalize=True)

# Method 2: Built-in heat smoothing of an intensive field
result_smooth = env.smooth(random_field, bandwidth, mode="average")

# Compare
correlation = np.corrcoef(result_convolve, result_smooth)[0, 1]
rmse = np.sqrt(np.mean((result_convolve - result_smooth) ** 2))

print(f"Correlation between methods: {correlation:.6f}")
print(f"RMSE: {rmse:.6f}")
print("\nThe geodesic Gaussian and heat-diffusion kernels are different smoothers.")

# %%
# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# Original random field
env.plot_field(
    random_field,
    ax=axes[0],
    cmap="viridis",
    vmin=0,
    vmax=1,
)
axes[0].set_title("Original Random Field", fontsize=14, weight="bold")

# Custom convolution
env.plot_field(
    result_convolve,
    ax=axes[1],
    cmap="viridis",
    vmin=0,
    vmax=1,
)
axes[1].set_title("graph_convolve() Result", fontsize=14, weight="bold")

# Built-in smoothing
env.plot_field(
    result_smooth,
    ax=axes[2],
    cmap="viridis",
    vmin=0,
    vmax=1,
)
axes[2].set_title("env.smooth() Result", fontsize=14, weight="bold")

plt.show()

# %% [markdown]
# `env.smooth(..., mode="average")` applies heat diffusion to intensive fields.
# `graph_convolve` evaluates the supplied distance kernel, including our Gaussian
# of geodesic distance. They need not agree, especially near boundaries. Both
# preserve a constant field when used as normalized averages.

# %% [markdown]
# ## Summary
#
# This notebook demonstrated the spatial signal processing primitives:
#
# **`neighbor_reduce()`**:
# - Local aggregation (sum, mean, max, min, std)
# - Spatial coherence analysis (Muller & Kubie 1989)
# - Local field variability measurement
# - Simple, fast operations on graph neighborhoods
#
# **`graph_convolve()`**:
# - Custom kernel convolution (box, Mexican hat, etc.)
# - Local contrast with balanced differences of Gaussian averages
# - Neighborhood coverage with a box filter (separate from observed support)
# - Flexible filtering for problem-specific analyses
#
# **When to use which**:
# - `env.smooth(..., mode="average")` → Heat smoothing of intensive fields
# - `neighbor_reduce()` → Simple aggregation and local statistics
# - `graph_convolve()` → Custom kernels and advanced filtering
#
# **Key applications**:
# - Quality metrics (spatial coherence, local variability)
# - Preprocessing (occupancy thresholding)
# - Exploratory spatial contrast and feature screening
# - Custom analyses (directional smoothing, anisotropic filtering)

# %% [markdown]
# ## Next Steps
#
# Use [11: place-field analysis](11_place_field_analysis.ipynb) for explicit field
# segmentation and spatial-cell evidence beyond these descriptive filters.
