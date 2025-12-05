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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bayesian Position Decoding
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# - Build encoding models (place fields) for a population of neurons
# - Decode spatial position from population spike counts using Bayesian methods
# - Access and interpret `DecodingResult` properties (posterior, MAP, mean, uncertainty)
# - Evaluate decoding accuracy with error metrics
# - Detect trajectory structure using isotonic/linear regression and Radon transform
# - Test significance of decoded sequences using shuffle-based methods
#
# **Estimated time: 25-30 minutes**

# %% [markdown]
# ---
#
# ## Setup
#
# First, let's import the necessary libraries:

# %%
import matplotlib.pyplot as plt
import numpy as np

from neurospatial import (
    Environment,
    compute_place_field,
    decode_position,
    decoding_error,
    median_decoding_error,
)
from neurospatial.decoding import (
    compute_shuffle_pvalue,
    confusion_matrix,
    decoding_correlation,
    fit_isotonic_trajectory,
    fit_linear_trajectory,
    shuffle_time_bins,
)
from neurospatial.simulation import (
    PlaceCellModel,
    generate_poisson_spikes,
    simulate_trajectory_ou,
)

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib for clear figures
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 14

# Colorblind-friendly palette
COLORS = {
    "blue": "#0173B2",
    "orange": "#DE8F05",
    "green": "#029E73",
    "red": "#CC78BC",
    "cyan": "#56B4E9",
}

# %% [markdown]
# ---
#
# ## Part 1: Generate Synthetic Data
#
# We'll create a 1D linear track environment with a population of place cells. This simplified setup makes it easy to visualize and understand decoding.

# %%
# Create a 1D linear track (100 cm)
track_length = 100.0  # cm
bin_size = 2.0  # cm per bin

# Create 1D positions along track
positions_1d = np.linspace(0, track_length, 51).reshape(-1, 1)
env = Environment.from_samples(positions_1d, bin_size=bin_size)
env.units = "cm"

print(f"Track length: {track_length} cm")
print(f"Number of spatial bins: {env.n_bins}")
print(f"Bin size: {bin_size} cm")

# %% [markdown]
# ### Generate Animal Trajectory
#
# Simulate a rat running back and forth on the linear track:

# %%
# Generate smooth trajectory on track using OU process
duration = 600.0  # 10 minutes
positions, times = simulate_trajectory_ou(
    env,
    duration=duration,
    dt=0.01,  # 10ms timestep
    speed_mean=15.0,  # cm/s
    speed_std=2.0,
    coherence_time=0.5,
    boundary_mode="reflect",
    seed=42,
)

print(f"Trajectory duration: {times[-1]:.1f} seconds")
print(f"Number of samples: {len(times)}")
print(f"Sampling rate: {1 / (times[1] - times[0]):.0f} Hz")
print(f"Position range: [{positions.min():.1f}, {positions.max():.1f}] cm")

# %%
# Visualize trajectory
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(times[:5000], positions[:5000, 0], color=COLORS["blue"], linewidth=0.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (cm)")
ax.set_title("First 50 seconds of trajectory", fontweight="bold")
ax.set_xlim(0, 50)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Create Place Cell Population
#
# Generate a population of place cells with fields distributed along the track:

# %%
# Create place cells with fields distributed along the track
n_neurons = 30
field_centers = np.linspace(5, track_length - 5, n_neurons)  # Spread across track
field_widths = np.random.uniform(8, 15, n_neurons)  # Variable field widths
peak_rates = np.random.uniform(10, 30, n_neurons)  # Variable peak firing rates

# Create PlaceCellModel for each neuron and generate spikes
spike_times_list = []
place_cells = []

for i in range(n_neurons):
    cell = PlaceCellModel(
        env,
        center=np.array([field_centers[i]]),
        width=field_widths[i],
        max_rate=peak_rates[i],
        baseline_rate=0.1,
    )
    place_cells.append(cell)

    # Generate spike train
    rates = cell.firing_rate(positions, times)
    spikes = generate_poisson_spikes(rates, times, refractory_period=0.002, seed=42 + i)
    spike_times_list.append(spikes)

print(f"Created {n_neurons} place cells")
print(f"Total spikes: {sum(len(s) for s in spike_times_list)}")
print(f"Mean spikes per neuron: {np.mean([len(s) for s in spike_times_list]):.0f}")

# %%
# Visualize a few place fields
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, ax in enumerate(axes):
    cell_idx = idx * (n_neurons // 3)
    field = compute_place_field(
        env,
        spike_times_list[cell_idx],
        times,
        positions,
        method="diffusion_kde",
        bandwidth=5.0,
    )

    # Plot field
    x_positions = env.bin_centers[:, 0]
    ax.bar(x_positions, field, width=bin_size * 0.8, color=COLORS["blue"], alpha=0.7)
    ax.axvline(
        field_centers[cell_idx],
        color=COLORS["red"],
        linestyle="--",
        linewidth=2,
        label="True center",
    )
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(f"Neuron {cell_idx + 1}", fontweight="bold")
    ax.legend()

plt.suptitle("Example Place Fields", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
#
# ## Part 2: Build Encoding Models
#
# Encoding models are place fields that describe how each neuron's firing rate varies with spatial position. These are the "tuning curves" we use for decoding.

# %%
# Compute place fields for all neurons (encoding models)
encoding_models = np.array(
    [
        compute_place_field(
            env,
            spike_times_list[i],
            times,
            positions,
            method="diffusion_kde",
            bandwidth=5.0,
        )
        for i in range(n_neurons)
    ]
)

print(f"Encoding models shape: {encoding_models.shape}")
print(f"  (n_neurons, n_bins) = ({n_neurons}, {env.n_bins})")
print(f"Max firing rate: {np.nanmax(encoding_models):.2f} Hz")
print(f"NaN values: {np.isnan(encoding_models).sum()}")

# Replace NaN with small baseline (bins without enough occupancy)
encoding_models = np.nan_to_num(encoding_models, nan=0.1)

# %%
# Visualize all encoding models as a heatmap
fig, ax = plt.subplots(figsize=(12, 6))

# Sort neurons by peak position for better visualization
peak_positions = np.argmax(encoding_models, axis=1)
sorted_idx = np.argsort(peak_positions)
sorted_models = encoding_models[sorted_idx]

im = ax.imshow(
    sorted_models,
    aspect="auto",
    cmap="hot",
    extent=[0, track_length, n_neurons, 0],
    vmin=0,
    vmax=np.percentile(sorted_models, 95),
)
ax.set_xlabel("Position (cm)")
ax.set_ylabel("Neuron (sorted by peak position)")
ax.set_title("Population Encoding Models (Place Fields)", fontweight="bold")
plt.colorbar(im, label="Firing rate (Hz)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
#
# ## Part 3: Bin Spikes for Decoding
#
# Bayesian decoding works on spike counts in discrete time bins. We need to convert spike times to spike counts:

# %%
# Time bin parameters
dt = 0.1  # 100 ms time bins (typical for spatial decoding)
time_bins = np.arange(0, times[-1], dt)
n_time_bins = len(time_bins) - 1

# Bin spikes for each neuron
spike_counts = np.zeros((n_time_bins, n_neurons), dtype=np.int64)
for i, spikes in enumerate(spike_times_list):
    spike_counts[:, i], _ = np.histogram(spikes, bins=time_bins)

# Get actual positions at each time bin center
time_bin_centers = time_bins[:-1] + dt / 2
actual_bin_indices = np.searchsorted(times, time_bin_centers) - 1
actual_bin_indices = np.clip(actual_bin_indices, 0, len(positions) - 1)
actual_positions = positions[actual_bin_indices]

print(f"Time bin width: {dt * 1000:.0f} ms")
print(f"Number of time bins: {n_time_bins}")
print(f"Spike counts shape: {spike_counts.shape}")
print(f"Total spikes in binned data: {spike_counts.sum()}")
print(f"Mean spikes per time bin: {spike_counts.sum(axis=1).mean():.2f}")

# %% [markdown]
# ---
#
# ## Part 4: Decode Position
#
# Now we can decode position using Bayesian inference. The `decode_position()` function computes the posterior probability distribution over positions for each time bin.

# %%
# Decode position from spike counts
result = decode_position(
    env,
    spike_counts,
    encoding_models,
    dt,
    prior=None,  # Uniform prior
    times=time_bin_centers,
)

print("Decoding complete!")
print(f"Result type: {type(result).__name__}")
print(f"Posterior shape: {result.posterior.shape}")
print(f"  (n_time_bins, n_bins) = ({result.n_time_bins}, {env.n_bins})")

# %% [markdown]
# ### DecodingResult Properties
#
# The `DecodingResult` container provides several useful properties (computed lazily):

# %%
# Access result properties
print("DecodingResult Properties:")
print(f"  posterior shape: {result.posterior.shape}")
print(f"  map_estimate shape: {result.map_estimate.shape} (bin indices)")
print(f"  map_position shape: {result.map_position.shape} (coordinates)")
print(f"  mean_position shape: {result.mean_position.shape} (coordinates)")
print(f"  uncertainty shape: {result.uncertainty.shape} (entropy in bits)")
print(f"  times shape: {result.times.shape if result.times is not None else 'None'}")

print(
    f"\nPosterior sum check (should be ~1.0): {result.posterior.sum(axis=1).mean():.6f}"
)
print(f"Mean uncertainty: {result.uncertainty.mean():.2f} bits")
print(f"Max uncertainty (uniform): {np.log2(env.n_bins):.2f} bits")

# %% [markdown]
# ### Visualize Decoding Results

# %%
# Plot posterior probability as heatmap (first 100 time bins)
fig, ax = plt.subplots(figsize=(14, 5))

n_show = 500  # Number of time bins to show
result.plot(ax=ax, show_map=True, colorbar=True)
ax.set_xlim(0, n_show * dt)

# Overlay actual position
ax.plot(
    time_bin_centers[:n_show],
    actual_positions[:n_show, 0],
    color=COLORS["cyan"],
    linewidth=2,
    linestyle="--",
    label="Actual position",
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (cm)")
ax.set_title(
    "Decoded Posterior with MAP Estimate (white) and Actual Position (cyan)",
    fontweight="bold",
)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# %%
# Compare decoded vs actual position
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time series comparison
ax = axes[0]
ax.plot(
    time_bin_centers[:n_show],
    actual_positions[:n_show, 0],
    color=COLORS["blue"],
    linewidth=1,
    alpha=0.7,
    label="Actual",
)
ax.plot(
    time_bin_centers[:n_show],
    result.map_position[:n_show, 0],
    color=COLORS["orange"],
    linewidth=1,
    alpha=0.7,
    label="Decoded (MAP)",
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (cm)")
ax.set_title("Decoded vs Actual Position Over Time", fontweight="bold")
ax.legend()

# Scatter plot
ax = axes[1]
ax.scatter(
    actual_positions[:, 0], result.map_position[:, 0], alpha=0.3, s=3, c=COLORS["blue"]
)
ax.plot(
    [0, track_length], [0, track_length], "k--", linewidth=2, label="Perfect decoding"
)
ax.set_xlabel("Actual position (cm)")
ax.set_ylabel("Decoded position (cm)")
ax.set_title("Decoded vs Actual Position", fontweight="bold")
ax.legend()
ax.set_aspect("equal")

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
#
# ## Part 5: Evaluate Decoding Accuracy
#
# Let's quantify how well the decoder performs using error metrics:

# %%
# Compute decoding error
errors = decoding_error(result.map_position, actual_positions)
median_err = median_decoding_error(result.map_position, actual_positions)

# Also compute error for mean position estimate
mean_errors = decoding_error(result.mean_position, actual_positions)
median_mean_err = median_decoding_error(result.mean_position, actual_positions)

print("Decoding Error Summary:")
print("\nMAP estimate:")
print(f"  Median error: {median_err:.2f} cm")
print(f"  Mean error: {np.nanmean(errors):.2f} cm")
print(f"  Std error: {np.nanstd(errors):.2f} cm")

print("\nMean position estimate:")
print(f"  Median error: {median_mean_err:.2f} cm")
print(f"  Mean error: {np.nanmean(mean_errors):.2f} cm")
print(f"  Std error: {np.nanstd(mean_errors):.2f} cm")

# %%
# Plot error distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram of errors
ax = axes[0]
ax.hist(errors, bins=50, color=COLORS["blue"], alpha=0.7, edgecolor="white")
ax.axvline(
    median_err,
    color=COLORS["red"],
    linewidth=2,
    linestyle="--",
    label=f"Median = {median_err:.2f} cm",
)
ax.set_xlabel("Decoding error (cm)")
ax.set_ylabel("Count")
ax.set_title("Error Distribution (MAP estimate)", fontweight="bold")
ax.legend()

# Error vs uncertainty
ax = axes[1]
ax.scatter(result.uncertainty, errors, alpha=0.3, s=5, c=COLORS["blue"])
ax.set_xlabel("Uncertainty (bits)")
ax.set_ylabel("Decoding error (cm)")
ax.set_title("Error vs Posterior Uncertainty", fontweight="bold")

plt.tight_layout()
plt.show()

# Compute correlation
corr = decoding_correlation(result.map_position, actual_positions)
print(f"\nDecoding correlation: {corr:.3f}")

# %% [markdown]
# ### Confusion Matrix
#
# A confusion matrix shows which positions are confused with each other:

# %%
# Compute confusion matrix
actual_bins = env.bin_at(actual_positions)
cm = confusion_matrix(env, result.posterior, actual_bins, method="map")

# Plot
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues", aspect="auto")
ax.set_xlabel("Decoded bin")
ax.set_ylabel("Actual bin")
ax.set_title("Confusion Matrix", fontweight="bold")
plt.colorbar(im, label="Count")
plt.tight_layout()
plt.show()

print(f"Diagonal (correct) proportion: {np.diag(cm).sum() / cm.sum():.1%}")

# %% [markdown]
# ---
#
# ## Part 6: Trajectory Analysis (for Replay Detection)
#
# For replay detection, we often want to detect whether decoded positions follow a coherent trajectory. Let's analyze a short segment that might resemble replay.

# %%
# Select a short segment (simulating a replay event)
start_idx = 100
end_idx = 150  # 50 time bins = 5 seconds

segment_posterior = result.posterior[start_idx:end_idx]
segment_times = time_bin_centers[start_idx:end_idx]
segment_actual = actual_positions[start_idx:end_idx, 0]

print(f"Segment: {segment_times[0]:.1f}s to {segment_times[-1]:.1f}s")
print(f"Duration: {segment_times[-1] - segment_times[0]:.1f}s")
print(f"Number of time bins: {len(segment_times)}")

# %% [markdown]
# ### Isotonic Regression
#
# Fit a monotonic (increasing or decreasing) trajectory to the posterior:

# %%
# Fit isotonic trajectory
iso_result = fit_isotonic_trajectory(
    segment_posterior,
    segment_times,
    method="expected",  # Use posterior mean
    increasing=None,  # Auto-detect direction
)

print("Isotonic Regression Results:")
print(f"  Direction: {iso_result.direction}")
print(f"  R-squared: {iso_result.r_squared:.4f}")
print(
    f"  Fitted positions range: [{iso_result.fitted_positions.min():.1f}, {iso_result.fitted_positions.max():.1f}] bins"
)

# %%
# Visualize isotonic fit
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Posterior heatmap with isotonic fit
ax = axes[0]
extent = [segment_times[0], segment_times[-1], 0, env.n_bins]
ax.imshow(
    segment_posterior.T, aspect="auto", origin="lower", cmap="viridis", extent=extent
)
ax.plot(
    segment_times,
    iso_result.fitted_positions,
    color="white",
    linewidth=2,
    label="Isotonic fit",
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (bin index)")
ax.set_title(f"Isotonic Fit (R² = {iso_result.r_squared:.4f})", fontweight="bold")
ax.legend()

# Residuals
ax = axes[1]
ax.bar(
    range(len(iso_result.residuals)),
    iso_result.residuals,
    color=COLORS["blue"],
    alpha=0.7,
)
ax.axhline(0, color="black", linestyle="-", linewidth=1)
ax.set_xlabel("Time bin")
ax.set_ylabel("Residual (bins)")
ax.set_title("Isotonic Regression Residuals", fontweight="bold")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Linear Regression with Uncertainty
#
# Fit a linear trajectory using Monte Carlo sampling to account for posterior uncertainty:

# %%
# Fit linear trajectory with sampling
lin_result = fit_linear_trajectory(
    env,
    segment_posterior,
    segment_times,
    method="sample",  # Monte Carlo sampling
    n_samples=1000,
    rng=42,
)

print("Linear Regression Results (sampling method):")
print(f"  Slope: {lin_result.slope:.2f} bins/s")
print(f"  Slope std: {lin_result.slope_std:.2f} bins/s")
print(f"  Intercept: {lin_result.intercept:.2f} bins")
print(f"  R-squared: {lin_result.r_squared:.4f}")

# Convert slope to cm/s
slope_cm_per_s = lin_result.slope * bin_size
print(f"\n  Speed: {slope_cm_per_s:.1f} cm/s")

# %% [markdown]
# ---
#
# ## Part 7: Shuffle-Based Significance Testing
#
# To determine if a decoded sequence is significant, we compare it to a null distribution generated by shuffling. This is essential for replay detection.

# %%
# Extract the segment of spike counts for shuffling
segment_spikes = spike_counts[start_idx:end_idx]

# Use R² from isotonic fit as our sequence score
observed_score = iso_result.r_squared

# Generate null distribution by shuffling time bins
n_shuffles = 500  # Use fewer for demo (typically 1000+)
null_scores = []

print(f"Running {n_shuffles} shuffles...")
for shuffled_spikes in shuffle_time_bins(segment_spikes, n_shuffles=n_shuffles, rng=42):
    # Decode shuffled spikes
    shuffled_result = decode_position(env, shuffled_spikes, encoding_models, dt)
    # Fit isotonic trajectory
    shuffled_fit = fit_isotonic_trajectory(
        shuffled_result.posterior, segment_times, method="expected"
    )
    null_scores.append(shuffled_fit.r_squared)

null_scores = np.array(null_scores)
print("Done!")

# %%
# Compute p-value
p_value = compute_shuffle_pvalue(observed_score, null_scores, tail="greater")

print("Significance Test Results:")
print(f"  Observed R²: {observed_score:.4f}")
print(f"  Null mean: {null_scores.mean():.4f}")
print(f"  Null std: {null_scores.std():.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant (p < 0.05): {'Yes' if p_value < 0.05 else 'No'}")

# %%
# Visualize null distribution
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    null_scores,
    bins=50,
    color=COLORS["blue"],
    alpha=0.7,
    edgecolor="white",
    label="Null distribution",
)
ax.axvline(
    observed_score,
    color=COLORS["red"],
    linewidth=3,
    linestyle="--",
    label=f"Observed (R² = {observed_score:.4f})",
)
ax.axvline(
    np.percentile(null_scores, 95),
    color=COLORS["orange"],
    linewidth=2,
    linestyle=":",
    label="95th percentile",
)

ax.set_xlabel("Isotonic R²")
ax.set_ylabel("Count")
ax.set_title(f"Shuffle Test: p = {p_value:.4f}", fontweight="bold")
ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
#
# ## Part 8: Export Results
#
# The `DecodingResult` can be exported to a pandas DataFrame for further analysis:

# %%
# Export to DataFrame
df = result.to_dataframe()

print(f"DataFrame shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print("\nFirst 5 rows:")
df.head()

# %% [markdown]
# ---
#
# ## Summary
#
# In this notebook, you learned:
#
# ### Building Encoding Models
# - Compute place fields for each neuron using `compute_place_field()`
# - Stack into encoding models array: shape `(n_neurons, n_bins)`
#
# ### Bayesian Decoding
# - Bin spikes into spike counts: shape `(n_time_bins, n_neurons)`
# - Decode with `decode_position()` to get posterior distribution
# - Access `DecodingResult` properties: `posterior`, `map_position`, `mean_position`, `uncertainty`
#
# ### Error Metrics
# - `decoding_error()` - Per-time-bin position error
# - `median_decoding_error()` - Summary statistic
# - `decoding_correlation()` - Weighted Pearson correlation
# - `confusion_matrix()` - Spatial confusion analysis
#
# ### Trajectory Analysis
# - `fit_isotonic_trajectory()` - Monotonic trajectory fitting
# - `fit_linear_trajectory()` - Linear fit with uncertainty estimation
# - R² measures trajectory coherence
#
# ### Significance Testing
# - `shuffle_time_bins()` - Generate null distribution
# - `compute_shuffle_pvalue()` - Monte Carlo p-value
# - Essential for determining if decoded sequences are significant
#
# ### Next Steps
# - Try different shuffle methods (`shuffle_cell_identity`, `shuffle_place_fields_circular`)
# - Explore the Radon transform for trajectory detection (`detect_trajectory_radon`)
# - Apply to real neural data
# - Experiment with custom priors

# %% [markdown]
# ---
#
# ## References
#
# 1. **Zhang, K., Ginzburg, I., McNaughton, B. L., & Sejnowski, T. J. (1998)**. "Interpreting neuronal population activity by reconstruction: Unified framework with application to hippocampal place cells." *Journal of Neurophysiology*.
#
# 2. **Davidson, T. J., Kloosterman, F., & Wilson, M. A. (2009)**. "Hippocampal replay of extended experience." *Neuron*.
#
# 3. **Karlsson, M. P., & Frank, L. M. (2009)**. "Awake replay of remote experiences in the hippocampus." *Nature Neuroscience*.
