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
# # Peri-Event PSTH
#
# Peri-stimulus / peri-event time histograms (PSTHs) are the workhorse
# visualization for event-locked neural responses. This notebook
# demonstrates:
#
# 1. Aligning spike trains to discrete event times
# 2. Computing a binned PSTH with standard error
# 3. Plotting per-trial rasters
# 4. Building GLM regressors from event times
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# - Align spikes to discrete events (rewards, lap starts, stimuli)
# - Compute peri-event histograms with ``peri_event_histogram``
# - Plot raster + PSTH summary panels
# - Build continuous time-to-event regressors for GLMs
#
# **Estimated time**: 20-25 minutes
#
# **Prerequisites**: [11_place_field_analysis.ipynb](11_place_field_analysis.ipynb)

# %% [markdown]
# ## Setup

# %%
import matplotlib.pyplot as plt
import numpy as np

from neurospatial.events import (
    align_spikes_to_events,
    peri_event_histogram,
    time_to_nearest_event,
)

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11

# %% [markdown]
# ## Part 1: Simulate Reward Events and an Event-Locked Neuron
#
# We construct a 600-second session with rewards delivered roughly every
# 20 seconds. The neuron fires at baseline ~2 Hz, with a transient
# response of ~30 Hz peaking ~150 ms after each reward.

# %%
rng = np.random.default_rng(42)
duration = 600.0
dt = 0.001  # 1 ms

# Reward times: jittered around 20s intervals
reward_centers = np.arange(20.0, duration, 20.0)
reward_times = reward_centers + rng.normal(0, 0.5, len(reward_centers))
print(f"Number of reward events: {len(reward_times)}")
print(f"First five reward times: {reward_times[:5]}")

# Build a time-varying firing rate: baseline + transient post-reward
times = np.arange(0, duration, dt)
baseline_rate = 2.0
peak_rate = 30.0
response_sigma = 0.08  # 80 ms width

firing_rate = np.full_like(times, baseline_rate)
for r_t in reward_times:
    # Gaussian bump peaking ~150ms after reward
    dt_to_event = times - (r_t + 0.15)
    firing_rate += peak_rate * np.exp(-0.5 * (dt_to_event / response_sigma) ** 2)

# Inhomogeneous Poisson sampling
spike_mask = rng.random(len(times)) < firing_rate * dt
spike_times = times[spike_mask]
print(
    f"Total spikes: {len(spike_times)} "
    f"(overall rate {len(spike_times) / duration:.2f} Hz)"
)

# %% [markdown]
# ## Part 2: Compute the PSTH
#
# ``peri_event_histogram`` returns a ``PeriEventResult`` with the binned
# spike count per bin (averaged across events), the SEM across events,
# and a cached ``firing_rate`` (count / bin_size).

# %%
window = (-0.5, 1.0)  # 500 ms before to 1 s after reward
bin_size = 0.025  # 25 ms

result = peri_event_histogram(
    spike_times,
    reward_times,
    window=window,
    bin_size=bin_size,
)

print(f"PSTH: {result.n_events} events x {len(result.bin_centers)} bins")
print(
    f"Peak firing rate: {result.firing_rate.max():.2f} Hz "
    f"at t = {result.bin_centers[result.firing_rate.argmax()]:.3f} s"
)
print(f"Baseline (window start): {result.firing_rate[:5].mean():.2f} Hz")

# %% [markdown]
# ## Part 3: Build the Raster Data
#
# ``align_spikes_to_events`` returns a per-trial list of spike times
# relative to each event - the natural shape for a raster plot.

# %%
aligned = align_spikes_to_events(spike_times, reward_times, window=window)
print(f"Aligned: {len(aligned)} trials")
print(f"Trial 0: {len(aligned[0])} spikes")
print(f"Trial 5: {len(aligned[5])} spikes")

# %% [markdown]
# ## Part 4: Plot Raster + PSTH
#
# The classic figure: raster on top (each row = one trial, dots = spike
# times), PSTH below (mean firing rate +/- SEM across trials).

# %%
fig, (ax_raster, ax_psth) = plt.subplots(
    2,
    1,
    figsize=(10, 9),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]},
)

# Raster
for trial_idx, trial_spikes in enumerate(aligned):
    ax_raster.vlines(
        trial_spikes,
        trial_idx - 0.4,
        trial_idx + 0.4,
        color="black",
        linewidth=0.8,
    )
ax_raster.axvline(
    0.0, color="tab:red", linewidth=1.5, linestyle="--", label="Reward onset"
)
ax_raster.set_ylabel("Trial")
ax_raster.set_ylim(-0.5, len(aligned) - 0.5)
ax_raster.set_title("Peri-Event Raster + PSTH")
ax_raster.legend(loc="upper right")

# PSTH (firing rate +/- SEM)
firing_rate_curve = result.firing_rate
sem_rate = np.asarray(result.sem) / bin_size  # SEM of counts -> Hz
ax_psth.plot(result.bin_centers, firing_rate_curve, "tab:blue", linewidth=2)
ax_psth.fill_between(
    result.bin_centers,
    firing_rate_curve - sem_rate,
    firing_rate_curve + sem_rate,
    color="tab:blue",
    alpha=0.3,
)
ax_psth.axvline(0.0, color="tab:red", linewidth=1.5, linestyle="--")
ax_psth.set_xlabel("Time relative to reward (s)")
ax_psth.set_ylabel("Firing rate (Hz)")
ax_psth.set_xlim(window)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 5: Per-Trial Variability
#
# Averaging across trials can hide trial-to-trial variability. The
# array of per-trial counts is useful for follow-up analyses (e.g.,
# regressing response amplitude against behaviour).

# %%
# Per-trial spike counts in the response window (0.05 - 0.35 s)
response_window = (0.05, 0.35)
per_trial_counts = np.array(
    [np.sum((t >= response_window[0]) & (t < response_window[1])) for t in aligned]
)
print(
    f"Per-trial response counts: mean={per_trial_counts.mean():.2f}, "
    f"std={per_trial_counts.std():.2f}, "
    f"range=[{per_trial_counts.min()}, {per_trial_counts.max()}]"
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(
    np.arange(len(per_trial_counts)),
    per_trial_counts,
    color="tab:blue",
    edgecolor="black",
    linewidth=0.5,
)
ax.axhline(
    per_trial_counts.mean(),
    color="tab:red",
    linestyle="--",
    label=f"Mean = {per_trial_counts.mean():.1f}",
)
ax.set_xlabel("Trial")
ax.set_ylabel(f"Spikes in [{response_window[0]:.2f}, {response_window[1]:.2f}] s")
ax.set_title("Per-trial response magnitude")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 6: GLM Regressor - Time to Nearest Event
#
# For GLM-style modelling, you typically want a continuous regressor
# rather than a categorical event indicator. ``time_to_nearest_event``
# returns the signed time difference from each sample to its closest
# event (negative before, positive after).

# %%
# Coarser sampling for visualization
sample_times = np.arange(0, duration, 0.05)  # 20 Hz
regressor = time_to_nearest_event(sample_times, reward_times, signed=True)
clipped = time_to_nearest_event(
    sample_times,
    reward_times,
    signed=True,
    max_time=2.0,
)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(sample_times[:600], regressor[:600], label="Unclipped", color="tab:blue")
ax.plot(
    sample_times[:600],
    clipped[:600],
    label="Clipped to ±2 s",
    color="tab:orange",
    linewidth=2,
    alpha=0.7,
)
for r_t in reward_times[:3]:
    ax.axvline(r_t, color="tab:red", linewidth=1.0, linestyle="--", alpha=0.5)
ax.axhline(0.0, color="gray", linewidth=0.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Time to nearest reward (s)")
ax.set_title("Continuous GLM regressor (first 30 s shown)")
ax.set_xlim(0, 30)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# %% [markdown]
# The clipped regressor saturates outside ±2 s of any reward, which is
# usually what you want in a GLM - the response is event-locked and
# you don't want far-from-event samples to dominate the design matrix.

# %% [markdown]
# ## Summary
#
# In this notebook, you learned:
#
# ### Key Concepts
# - PSTHs aggregate spike counts in time bins relative to discrete
#   events, then average across trials
# - Combining a raster (per-trial detail) with a PSTH (trial average)
#   is the canonical event-locked summary plot
# - SEM across trials is the natural uncertainty band for a PSTH
# - For GLMs, a continuous time-to-event regressor (optionally clipped)
#   captures event-locked structure without binning
#
# ### API
# - ``align_spikes_to_events`` returns a per-trial list of spike times
#   relative to each event (raw data for rasters)
# - ``peri_event_histogram`` returns a ``PeriEventResult`` with
#   ``bin_centers``, ``histogram`` (mean count), ``sem``, and a
#   cached ``firing_rate`` attribute
# - ``time_to_nearest_event`` returns a continuous signed-time
#   regressor with an optional clipping window
# - ``population_peri_event_histogram`` extends this to multiple units
#   simultaneously
#
# ### Next Steps
# - Use ``population_peri_event_histogram`` to compare PSTHs across many
#   units
# - Combine event regressors with spatial regressors in a GLM
#   (see ``encoding`` and ``ops.basis`` for spatial basis functions)
# - For lap-aligned PSTHs, see [14_behavioral_segmentation.ipynb](14_behavioral_segmentation.ipynb)
