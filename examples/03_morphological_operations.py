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
# # Morphological Operations: Handling Sparse and Patchy Data
#
# **Estimated time**: 20-30 minutes
#
# **Prerequisites**: [01: basics](01_introduction_basics.ipynb), [02: layouts](02_layout_engines.ipynb).
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# - Understand why sparse data creates problems for spatial analysis
# - Use morphological operations (dilate, fill_holes, close_gaps) to improve connectivity
# - Control active bin inference with `bin_count_threshold`
# - Distinguish accessible geometry from observed spatial support
# - Make informed decisions about when to use morphological operations
#

# %% [markdown]
# ## The Problem: Real Data is Messy
#
# In notebook 01, we worked with clean, uniformly sampled position data. But real experimental data is often:
#
# - **Sparse**: Animals don't visit all locations equally
# - **Patchy**: Tracking can fail briefly, creating gaps
# - **Biased**: Animals prefer certain regions (edges, corners, goal locations)
# - **Noisy**: Position estimates can be inaccurate or have outliers
#
# These issues can fragment your environment into disconnected regions, making spatial analysis difficult or impossible.
#
# This notebook changes the spatial mask; it does not clean tracking artifacts,
# restore missing timestamps, or create observations. Fill only areas known to
# be accessible. Never bridge a real wall or fill a genuine obstacle.

# %% [markdown]
# ## Setup

# %%
import matplotlib.pyplot as plt
import numpy as np

from neurospatial import Environment

np.random.seed(42)
# Shared styling (Okabe-Ito palette, consistent figure / font sizes)
import sys
from pathlib import Path

_here = (
    str(Path(__file__).resolve().parent) if "__file__" in globals() else str(Path.cwd())
)
if _here not in sys.path:
    sys.path.insert(0, _here)
from _style import apply_style

apply_style(figsize=(12, 10), font_size=11)


# %% [markdown]
# ## Creating Realistic Sparse Data
#
# Let's simulate what real position tracking often looks like: an animal that prefers certain regions and avoids others.


# %%
def create_sparse_trajectory(n_samples=3000, seed=42):
    """
    Create sparse position data with realistic biases.

    Simulates an animal that:
    - Prefers to stay near walls (thigmotaxis)
    - Has occasional tracking failures
    - Makes quick runs across the center

    Parameters
    ----------
    n_samples : int
        Number of position samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    position : ndarray, shape (n_samples, 2)
        2D position data in cm.
    """
    np.random.seed(seed)
    position = np.zeros((n_samples, 2))
    position[0] = [10, 10]  # Start in corner

    # Create biased random walk
    for t in range(1, n_samples):
        # Strong preference for edges
        if np.random.rand() < 0.7:  # 70% of time, move along edges
            # Move along walls
            if (
                position[t - 1, 0] < 15 or position[t - 1, 0] > 85
            ):  # Near vertical walls
                step = [np.random.randn() * 0.5, np.random.randn() * 3.0]
            elif (
                position[t - 1, 1] < 15 or position[t - 1, 1] > 85
            ):  # Near horizontal walls
                step = [np.random.randn() * 3.0, np.random.randn() * 0.5]
            else:
                step = np.random.randn(2) * 2.0
        else:  # Occasional center crossing
            step = np.random.randn(2) * 5.0

        position[t] = position[t - 1] + step
        position[t] = np.clip(position[t], 5, 95)

    # Omit a few samples to make this spatial point cloud sparser.
    tracking_failures = np.random.choice(
        n_samples, size=int(n_samples * 0.05), replace=False
    )

    # This function returns geometry samples, not a timestamped trajectory.
    position = np.delete(position, tracking_failures, axis=0)

    return position


sparse_data = create_sparse_trajectory(n_samples=3000)
print(f"Generated {len(sparse_data)} position samples")
print(f"X range: [{sparse_data[:, 0].min():.1f}, {sparse_data[:, 0].max():.1f}] cm")
print(f"Y range: [{sparse_data[:, 1].min():.1f}, {sparse_data[:, 1].max():.1f}] cm")

# %%
# Visualize the sparse trajectory
fig, ax = plt.subplots(figsize=(10, 10))

scatter = ax.scatter(
    sparse_data[:, 0],
    sparse_data[:, 1],
    c=np.arange(len(sparse_data)),
    cmap="viridis",
    s=2,
    alpha=0.5,
)

ax.plot(sparse_data[0, 0], sparse_data[0, 1], "go", markersize=12, label="Start")
ax.plot(sparse_data[-1, 0], sparse_data[-1, 1], "ro", markersize=12, label="End")

ax.set_xlabel("X position (cm)")
ax.set_ylabel("Y position (cm)")
ax.set_title("Sparse Trajectory (wall-preferring animal)")
ax.set_aspect("equal")
ax.legend()
plt.colorbar(scatter, ax=ax, label="Time (samples)")
plt.tight_layout()
plt.show()

print("\nNotice: Animal spends most time near walls, rarely visits center")

# %% [markdown]
# ## Problem 1: Incomplete Sampled Geometry
#
# What happens when we create an environment from this sparse data using default settings?

# %%
# Create environment with default settings
env_default = Environment.from_samples(
    positions=sparse_data, bin_size=5.0, name="Default_Sparse", units="cm"
)

print(env_default.info())

# %%
# Visualize the sample-inferred mask
fig, ax = plt.subplots(figsize=(10, 10))
env_default.plot(ax=ax, show_connectivity=True)
ax.set_title(f"Sample-Inferred Mask\n{env_default.n_bins} active bins")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

print("\nThe active mask contains only sampled bins.")
print(
    "Unvisited areas may be accessible or may contain obstacles; tracking alone cannot decide."
)

# %% [markdown]
# ### Checking Connectivity
#
# Check connectivity rather than assuming sparse sampling disconnects the graph.
# This seeded example remains connected, despite holes in its active mask.

# %%
import networkx as nx

# Check if environment is connected
is_connected = nx.is_connected(env_default.connectivity)
print(f"Is environment connected? {is_connected}")

if not is_connected:
    # Find connected components
    components = list(nx.connected_components(env_default.connectivity))
    print(f"\nNumber of disconnected components: {len(components)}")
    print(f"Largest component size: {len(max(components, key=len))} bins")
    print(f"Smallest component size: {len(min(components, key=len))} bins")

    # Try to find a path from corner to corner (will fail if disconnected)
    try:
        corner1 = env_default.bin_at([[10, 10]])[0]
        corner2 = env_default.bin_at([[90, 90]])[0]
        path = env_default.path_between(corner1, corner2)
        print(f"\nPath from corner to corner: {len(path)} bins")
    except nx.NetworkXNoPath:
        print("\nNo path exists between opposite corners!")

# %% [markdown]
# ## Solution 1: Morphological Dilation
#
# **Dilation** expands active regions by adding neighboring bins. This fills small gaps and connects nearby fragments.
#
# Think of it like inflating the active region slightly.

# %%
# Create environment with dilation
env_dilated = Environment.from_samples(
    positions=sparse_data,
    bin_size=5.0,
    dilate=True,  # Enable dilation
    name="Dilated",
    units="cm",
)

print(env_dilated.info())
print(f"\nBins added by dilation: {env_dilated.n_bins - env_default.n_bins}")

# %%
# Compare default vs dilated
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

env_default.plot(ax=axes[0], show_connectivity=True)
axes[0].set_title(f"Default\n{env_default.n_bins} bins")
axes[0].set_aspect("equal")

env_dilated.plot(ax=axes[1], show_connectivity=True)
axes[1].set_title(f"With Dilation\n{env_dilated.n_bins} bins")
axes[1].set_aspect("equal")

plt.tight_layout()
plt.show()

# Check connectivity
is_connected = nx.is_connected(env_dilated.connectivity)
print(f"\nIs dilated environment connected? {is_connected}")

# %% [markdown]
# ## Solution 2: Filling Holes
#
# **Fill holes** fills in completely enclosed inactive regions. This is useful when an animal briefly crosses an area but doesn't sample it well enough to activate all bins.

# %%
# Create environment with hole filling
env_filled = Environment.from_samples(
    positions=sparse_data,
    bin_size=5.0,
    fill_holes=True,  # Enable hole filling
    name="Filled",
    units="cm",
)

print(env_filled.info())
print(f"\nBins added by fill_holes: {env_filled.n_bins - env_default.n_bins}")

# %% [markdown]
# ## Solution 3: Closing Gaps
#
# **Close gaps** performs dilation followed by erosion (shrinking). This connects nearby regions while preserving the overall shape better than dilation alone.
#
# Think of it as: expand -> connect -> shrink back to original size.

# %%
# Create environment with gap closing
env_closed = Environment.from_samples(
    positions=sparse_data,
    bin_size=5.0,
    close_gaps=True,  # Enable closing
    name="Closed",
    units="cm",
)

print(env_closed.info())
print(
    f"\nBins after close_gaps: {env_closed.n_bins} (compare to {env_default.n_bins} default)"
)

# %% [markdown]
# ## Combining Multiple Operations
#
# For very sparse data, you can combine multiple morphological operations:

# %%
# Combine all operations
env_combined = Environment.from_samples(
    positions=sparse_data,
    bin_size=5.0,
    dilate=True,
    fill_holes=True,
    close_gaps=True,
    name="Combined",
    units="cm",
)

print(env_combined.info())

# %%
# Compare all approaches
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes = axes.ravel()

envs = [
    (env_default, "Default (No operations)"),
    (env_dilated, "Dilate only"),
    (env_filled, "Fill holes only"),
    (env_combined, "All operations combined"),
]

for ax, (env, title) in zip(axes, envs, strict=False):
    env.plot(ax=ax, show_connectivity=True)

    # Check connectivity
    is_connected = nx.is_connected(env.connectivity)
    status = "✓ Connected" if is_connected else "✗ Disconnected"

    ax.set_title(f"{title}\n{env.n_bins} bins - {status}")
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Solution 4: Adjusting bin_count_threshold
#
# By default, a bin needs at least 1 sample to be considered active. You can adjust this threshold to:
#
# - **Threshold 0**: Include bins with at least one sample
# - **Threshold 4**: Include bins with at least five samples
#
# Regular-grid counts must be strictly greater than this parameter. Sample
# counts depend on tracking frequency; use occupancy in seconds for support.
#
# **Note**: `bin_count_threshold` is applied BEFORE morphological operations.

# %%
# Compare different thresholds
thresholds = [0, 4, 9]
env_thresholds = {}

for thresh in thresholds:
    env = Environment.from_samples(
        positions=sparse_data,
        bin_size=5.0,
        bin_count_threshold=thresh,
        name=f"Threshold_{thresh}",
        units="cm",
    )
    env_thresholds[thresh] = env
    print(f"Threshold = {thresh}: {env.n_bins} active bins")

# %%
# Visualize threshold effects
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, thresh in zip(axes, thresholds, strict=False):
    env_thresholds[thresh].plot(ax=ax, show_connectivity=True)
    ax.set_title(f"Threshold = {thresh}\n{env_thresholds[thresh].n_bins} bins")
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()

print("\nHigher thresholds remove bins with sparse sampling")
print("Use this to focus on well-sampled regions only")


# %% [markdown]
# ## Practical Example: Geometry Does Not Restore Missing Firing Rates
#
# Use independent 50 ms observations at the retained positions to illustrate
# occupancy normalization. These counts are a toy example, not a continuous
# spike recording reconstructed across the omitted tracking samples.


# %%
# Simulate neural spiking data correlated with position
def generate_place_cell_spikes(position, preferred_location, field_size=15.0):
    """
    Generate spikes from a place cell with a preferred location.

    Parameters
    ----------
    position : ndarray, shape (n_samples, 2)
        Position trajectory.
    preferred_location : array-like, shape (2,)
        Center of place field.
    field_size : float
        Spatial scale of place field.

    Returns
    -------
    spike_counts : ndarray, shape (n_samples,)
        Number of spikes at each timepoint.
    """
    distances = np.linalg.norm(position - preferred_location, axis=1)
    firing_rate = 20.0 * np.exp(
        -(distances**2) / (2 * field_size**2)
    )  # Gaussian place field

    # Poisson spiking
    dt = 0.05  # 50 ms bins
    spike_counts = np.random.poisson(firing_rate * dt)
    return spike_counts


# Create place cell centered in the arena
place_field_center = np.array([50.0, 50.0])
spikes = generate_place_cell_spikes(sparse_data, place_field_center, field_size=20.0)

print(f"Generated {spikes.sum()} total spikes")
print(f"Mean firing rate: {spikes.mean() / 0.05:.2f} Hz")

# %% [markdown]
# ### Problem: Computing Spatial Firing Rate Map
#
# Rates are defined only where observation time is positive. Observed silence
# is a valid zero rate; unsampled space has no empirical rate.

# %%
# Map spikes to bins
bin_indices = env_default.bin_at(sparse_data)

# Count spikes per bin
spike_counts_per_bin = np.bincount(
    bin_indices[bin_indices >= 0],
    weights=spikes[bin_indices >= 0],
    minlength=env_default.n_bins,
)

# Count occupancy per bin (in seconds)
occupancy_per_bin = (
    np.bincount(bin_indices[bin_indices >= 0], minlength=env_default.n_bins) * 0.05
)  # 50 ms per sample

# Compute firing rate (spikes / seconds)
# Leave unobserved bins NaN; zero spikes with positive occupancy remain zero.
firing_rate_map = np.divide(
    spike_counts_per_bin,
    occupancy_per_bin,
    where=occupancy_per_bin > 0,
    out=np.full_like(spike_counts_per_bin, np.nan, dtype=float),
)

print(
    f"Bins with valid firing rates: {np.sum(np.isfinite(firing_rate_map))} / {env_default.n_bins}"
)
print(f"Peak firing rate: {firing_rate_map.max():.2f} Hz")
print("\nUnvisited space has no measured firing rate; it is not observed silence.")

# %%
# Visualize problematic firing rate map
fig, ax = plt.subplots(figsize=(10, 10))

# Only plot bins with data
active_bins = np.isfinite(firing_rate_map)
scatter = ax.scatter(
    env_default.bin_centers[active_bins, 0],
    env_default.bin_centers[active_bins, 1],
    c=firing_rate_map[active_bins],
    s=300,
    cmap="hot",
    vmin=0,
    marker="s",
    edgecolors="black",
    linewidth=0.5,
)

# Mark true place field center
ax.plot(
    place_field_center[0],
    place_field_center[1],
    "c*",
    markersize=30,
    markeredgecolor="blue",
    markeredgewidth=2,
    label="True place field center",
)

ax.set_xlabel("X position (cm)")
ax.set_ylabel("Y position (cm)")
ax.set_title("Observed Rates on the Sample-Inferred Mask")
ax.set_aspect("equal")
ax.legend()
plt.colorbar(scatter, ax=ax, label="Firing Rate (Hz)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Expand the Geometry While Preserving Missing Support
#
# Suppose independent arena knowledge permits the added bins. Dilation and
# hole filling change the geometry, but keep their empirical rates missing.

# %%
# Create fixed environment
env_fixed = Environment.from_samples(
    positions=sparse_data,
    bin_size=5.0,
    dilate=True,
    fill_holes=True,
    name="Fixed",
    units="cm",
)

print(f"Fixed environment: {env_fixed.n_bins} bins")
print(f"Is connected: {nx.is_connected(env_fixed.connectivity)}")

# Recompute firing rate map
bin_indices_fixed = env_fixed.bin_at(sparse_data)

spike_counts_per_bin_fixed = np.bincount(
    bin_indices_fixed[bin_indices_fixed >= 0],
    weights=spikes[bin_indices_fixed >= 0],
    minlength=env_fixed.n_bins,
)

occupancy_per_bin_fixed = (
    np.bincount(bin_indices_fixed[bin_indices_fixed >= 0], minlength=env_fixed.n_bins)
    * 0.05
)

firing_rate_map_fixed = np.divide(
    spike_counts_per_bin_fixed,
    occupancy_per_bin_fixed,
    where=occupancy_per_bin_fixed > 0,
    out=np.full_like(spike_counts_per_bin_fixed, np.nan, dtype=float),
)

print(
    f"\nBins with valid firing rates: {np.sum(np.isfinite(firing_rate_map_fixed))} / {env_fixed.n_bins}"
)

# %%
# Visualize fixed firing rate map
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Before
active_bins = np.isfinite(firing_rate_map)
scatter1 = axes[0].scatter(
    env_default.bin_centers[active_bins, 0],
    env_default.bin_centers[active_bins, 1],
    c=firing_rate_map[active_bins],
    s=300,
    cmap="hot",
    vmin=0,
    vmax=np.nanmax(firing_rate_map),
    marker="s",
    edgecolors="black",
    linewidth=0.5,
)
axes[0].plot(
    place_field_center[0],
    place_field_center[1],
    "c*",
    markersize=30,
    markeredgecolor="blue",
    markeredgewidth=2,
)
axes[0].set_title("Before: Sample-Inferred Geometry")
axes[0].set_aspect("equal")
axes[0].set_xlabel("X position (cm)")
axes[0].set_ylabel("Y position (cm)")

# After: gray bins have geometry but no observations.
axes[1].scatter(
    env_fixed.bin_centers[:, 0],
    env_fixed.bin_centers[:, 1],
    c="lightgray",
    s=300,
    marker="s",
    label="No observations",
)
active_bins_fixed = np.isfinite(firing_rate_map_fixed)
scatter2 = axes[1].scatter(
    env_fixed.bin_centers[active_bins_fixed, 0],
    env_fixed.bin_centers[active_bins_fixed, 1],
    c=firing_rate_map_fixed[active_bins_fixed],
    s=300,
    cmap="hot",
    vmin=0,
    vmax=np.nanmax(firing_rate_map),
    marker="s",
    edgecolors="black",
    linewidth=0.5,
)
axes[1].plot(
    place_field_center[0],
    place_field_center[1],
    "c*",
    markersize=30,
    markeredgecolor="blue",
    markeredgewidth=2,
    label="True place field center",
)
axes[1].set_title("After: Added Geometry, Same Observed Rates")
axes[1].set_aspect("equal")
axes[1].set_xlabel("X position (cm)")
axes[1].set_ylabel("Y position (cm)")
axes[1].legend()

plt.colorbar(scatter2, ax=axes[1], label="Firing Rate (Hz)")
plt.tight_layout()
plt.show()

added_unobserved = occupancy_per_bin_fixed == 0
assert np.all(np.isnan(firing_rate_map_fixed[added_unobserved]))
matched_bins = env_fixed.bin_at(env_default.bin_centers)
np.testing.assert_allclose(firing_rate_map, firing_rate_map_fixed[matched_bins])
print(
    f"\nAdded {added_unobserved.sum()} bins without observations; their rates remain NaN."
)
print("Morphology changes geometry, not the evidence for a complete field.")

# %% [markdown]
# ## When to Use Each Operation
#
# Here's a decision guide:
#
# | Operation | Use When | Effect | Caution |
# |-----------|----------|--------|--------|
# | **dilate** | Sparse sampling, need connectivity | Expands active regions by 1 bin | Adds bins where animal never went |
# | **fill_holes** | Animal crosses regions quickly | Fills enclosed gaps | Only affects interior holes |
# | **close_gaps** | Need to connect nearby fragments | Connects then shrinks | Better shape preservation than dilate |
# | **bin_count_threshold** | Very sparse data or need robust estimates | Removes bins with few samples | May fragment more |
#
# **General recommendation:**
# 1. Start without morphological operations
# 2. Check if environment is connected (use `networkx.is_connected()`)
# 3. Compare any missing connections against known accessible arena geometry
# 4. Use a polygon/mask for real boundaries; apply morphology only when justified
# 5. Keep observation support separate from the expanded geometry

# %% [markdown]
# ## Common Pitfalls
#
# ### Pitfall 1: Over-dilation
#
# Too much dilation can create bins far from actual data:

# %%
# Multiple rounds of dilation
env_over_dilated = Environment.from_samples(
    positions=sparse_data, bin_size=5.0, dilate=True, name="OverDilated", units="cm"
)

# Apply dilation multiple times manually (not recommended!)
# This is just for demonstration
print("Warning: Don't dilate multiple times unless you have a good reason!")
print("Morphological operations should be used conservatively.")


# %% [markdown]
# ### Pitfall 2: Wrong operation order
#
# Operations are applied in this order:
# 1. bin_count_threshold (determines initial active bins)
# 2. dilate
# 3. fill_holes
# 4. close_gaps
#
# This order matters for the final result!

# %% [markdown]
# ### Pitfall 3: Forgetting to check connectivity
#
# Check connected components before interpreting paths. Disconnected spaces
# can be correct; paths between different components are unavailable.


# %%
def check_environment_quality(env):
    """
    Diagnostic function to check environment quality.

    Parameters
    ----------
    env : Environment
        Environment to check.
    """
    print(f"Environment: {env.name}")
    print(f"  Total bins: {env.n_bins}")
    print(f"  Total edges: {env.connectivity.number_of_edges()}")

    is_connected = nx.is_connected(env.connectivity)
    print(f"  Connected: {is_connected}")

    if not is_connected:
        components = list(nx.connected_components(env.connectivity))
        print(f"  WARNING: {len(components)} disconnected components!")
        print(f"  Largest component: {len(max(components, key=len))} bins")

    print()


# Check all our environments
check_environment_quality(env_default)
check_environment_quality(env_fixed)

# %% [markdown]
# ### Pitfall 4: Not considering the trade-off
#
# Morphological operations make connectivity better but:
# - Add bins where the animal never went
# - Can't compute occupancy-normalized rates for these bins
# - May artificially smooth spatial patterns
#
# **Best practice**: Justify geometry against the arena and retain an explicit
# occupancy/support mask. Smoothing into added space is a prediction, not an observation.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Real data is sparse** - Animals don't sample uniformly, tracking can fail
# 2. **Connectivity is an assumption** - Paths exist only within components
# 3. **Morphological operations change geometry**:
#    - `dilate=True` - Expand active regions
#    - `fill_holes=True` - Fill enclosed gaps
#    - `close_gaps=True` - Connect nearby fragments
# 4. **bin_count_threshold** - Control minimum samples needed per bin
# 5. **Always check connectivity** - Use `networkx.is_connected()` to verify
# 6. **Trade-offs exist** - More connectivity = more artificial bins
# 7. **Use conservatively** - Start minimal, add operations only if needed
#
# ## Next Steps
#
# In the next notebook ([04_regions_of_interest.ipynb](04_regions_of_interest.ipynb)), you'll learn:
# - How to define named regions (goal locations, reward zones)
# - Point vs polygon regions
# - Querying spatial relationships
# - Saving and loading region definitions
#
# ## Exercises (Optional)
#
# 1. Create extremely sparse data (remove 90% of samples) and find the minimal morphological operations needed
# 2. Compare firing rate maps with and without morphological operations
# 3. Compare how thresholds change observed support and connected components
# 4. Create a diagnostic function that recommends which operations to use
