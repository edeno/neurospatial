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
# # Head Direction Tuning
#
# Head direction (HD) cells fire when the animal is facing a particular
# allocentric direction, independent of position. They were first
# characterized in the rat postsubiculum (Taube, Muller & Ranck, 1990)
# and form the backbone of the spatial navigation system together with
# place cells, grid cells, and boundary cells.
#
# This notebook demonstrates:
#
# 1. Simulating an HD cell with known preferred direction
# 2. Computing a circular tuning curve over heading
# 3. Testing for significant directionality (Rayleigh test)
# 4. Classifying neurons as HD cells using mean vector length
# 5. Plotting on a polar axis
#
# **Heading convention** (allocentric, world frame):
# 0 = East, +π/2 = North, π = West, -π/2 = South.
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# - Simulate HD cells with configurable concentration and preferred direction
# - Compute directional rate maps with ``compute_directional_rate``
# - Interpret the mean vector length (MVL) and Rayleigh p-value
# - Classify neurons as HD cells using ``is_head_direction_cell``
# - Plot HD tuning curves on a polar axis
#
# **Estimated time**: 15-20 minutes
#
# **Prerequisites**: [08_spike_field_basics.ipynb](08_spike_field_basics.ipynb)

# %% [markdown]
# ## Setup

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neurospatial import Environment
from neurospatial.encoding import (
    compute_directional_rate,
    is_head_direction_cell,
)
from neurospatial.ops.egocentric import heading_from_velocity
from neurospatial.simulation import (
    HeadDirectionCellModel,
    PlaceCellModel,
    generate_poisson_spikes,
    simulate_trajectory_ou,
)

_here = (
    str(Path(__file__).resolve().parent) if "__file__" in globals() else str(Path.cwd())
)
if _here not in sys.path:
    sys.path.insert(0, _here)
from _style import apply_style  # noqa: E402

apply_style(figsize=(10, 8))

# %% [markdown]
# ## Part 1: Create Environment and Trajectory
#
# We need an environment to drive the trajectory simulator and produce
# a realistic distribution of headings.

# %%
xx, yy = np.meshgrid(np.linspace(0, 100, 41), np.linspace(0, 100, 41))
samples = np.column_stack([xx.ravel(), yy.ravel()])
env = Environment.from_samples(samples, bin_size=4.0)
env.units = "cm"

positions, times = simulate_trajectory_ou(
    env,
    duration=600.0,
    dt=0.02,
    speed_units="cm",
    speed_mean=15.0,
    speed_std=5.0,
    seed=42,
)
dt = float(times[1] - times[0])
headings = heading_from_velocity(positions, dt, min_speed=2.0, bandwidth=3.0)

print(f"Trajectory: {len(times)} samples, {times[-1]:.1f}s")
print(
    f"Heading samples: {np.sum(np.isfinite(headings))} valid "
    f"({100 * np.mean(np.isfinite(headings)):.1f}%)"
)

# %% [markdown]
# Sanity-check the heading distribution: with reflective walls and a
# long OU trajectory, all directions should be visited reasonably often.

# %%
valid_headings = headings[np.isfinite(headings)]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
bins = np.linspace(-np.pi, np.pi, 37)
counts, _ = np.histogram(valid_headings, bins=bins)
ax.bar(
    (bins[:-1] + bins[1:]) / 2,
    counts,
    width=np.diff(bins),
    bottom=0.0,
    edgecolor="black",
    linewidth=0.5,
)
ax.set_title("Heading occupancy (allocentric)\n0=East, +pi/2=North", pad=20)
plt.show()

# %% [markdown]
# ## Part 2: Simulate a Head-Direction Cell
#
# ``HeadDirectionCellModel`` uses a von Mises (circular Gaussian)
# distribution. The concentration parameter ``κ`` controls tuning
# sharpness; ``κ=4`` gives a half-width of about 30°.

# %%
preferred = np.pi / 2  # North
hd_model = HeadDirectionCellModel(
    preferred_direction=preferred,
    concentration=4.0,
    max_rate=40.0,
    baseline_rate=0.5,
)
hd_rates = hd_model.firing_rate(headings)
hd_spikes = generate_poisson_spikes(hd_rates, times, seed=42)

print(f"HD cell: {len(hd_spikes)} spikes")
print(f"Preferred direction: {np.degrees(preferred):.0f}° (North)")
print(f"Mean firing rate: {len(hd_spikes) / times[-1]:.2f} Hz")

# %% [markdown]
# ## Part 3: Simulate a Place Cell for Comparison
#
# A place cell should not show meaningful directional tuning (assuming
# uniform coverage of headings at its preferred location), so we use it
# as a negative control.

# %%
place_model = PlaceCellModel(
    env=env,
    center=np.array([50.0, 50.0]),
    width=10.0,
    max_rate=40.0,
    baseline_rate=0.5,
)
pc_rates = place_model.firing_rate(positions)
pc_spikes = generate_poisson_spikes(pc_rates, times, seed=43)

print(f"Place cell: {len(pc_spikes)} spikes")
print(f"Mean firing rate: {len(pc_spikes) / times[-1]:.2f} Hz")

# %% [markdown]
# ## Part 4: Compute Directional Rate Maps
#
# ``compute_directional_rate`` operates on the heading domain
# (a circular angular variable), so it does *not* take an
# ``Environment`` argument - it is the documented exception to the
# canonical ``(env, spike_times, times, positions, headings, ...)``
# argument order for encoding functions.

# %%
hd_result = compute_directional_rate(
    hd_spikes,
    times,
    headings,
    bin_size=np.deg2rad(6.0),
    bandwidth=np.deg2rad(10.0),
)
pc_result = compute_directional_rate(
    pc_spikes,
    times,
    headings,
    bin_size=np.deg2rad(6.0),
    bandwidth=np.deg2rad(10.0),
)

hd_rates_arr = np.asarray(hd_result.firing_rate)
hd_centers_arr = np.asarray(hd_result.bin_centers)
pc_rates_arr = np.asarray(pc_result.firing_rate)
pc_centers_arr = np.asarray(pc_result.bin_centers)

print(
    f"HD tuning peak: {np.nanmax(hd_rates_arr):.2f} Hz "
    f"at {np.degrees(hd_centers_arr[np.nanargmax(hd_rates_arr)]):.0f}°"
)
print(
    f"Place tuning peak: {np.nanmax(pc_rates_arr):.2f} Hz "
    f"at {np.degrees(pc_centers_arr[np.nanargmax(pc_rates_arr)]):.0f}°"
)

# %% [markdown]
# ## Part 5: Mean Vector Length and Rayleigh Test
#
# Two statistics determine whether a tuning curve is directionally
# selective:
#
# - **Mean vector length (MVL)** - in [0, 1]. Larger = sharper tuning.
#   MVL near 0 means firing is uniform across directions; MVL near 1
#   means firing is concentrated in a single direction.
# - **Rayleigh p-value** - probability that a uniform distribution
#   could have produced this much concentration by chance.
#
# An HD cell typically has MVL ≥ 0.4 and Rayleigh p < 0.05.

# %%
hd_mvl = hd_result.mean_vector_length()
pc_mvl = pc_result.mean_vector_length()
hd_pval = hd_result.rayleigh_pvalue()
pc_pval = pc_result.rayleigh_pvalue()
hd_pref = hd_result.preferred_direction()
pc_pref = pc_result.preferred_direction()

print("=" * 60)
print(f"{'Metric':<28} {'HD Cell':<15} {'Place Cell':<15}")
print("-" * 60)
print(f"{'Mean vector length':<28} {hd_mvl:<15.3f} {pc_mvl:<15.3f}")
print(f"{'Rayleigh p-value':<28} {hd_pval:<15.2e} {pc_pval:<15.2e}")
print(
    f"{'Preferred direction (deg)':<28} {np.degrees(hd_pref):<15.1f} "
    f"{np.degrees(pc_pref):<15.1f}"
)
print("=" * 60)
print(f"True preferred direction: {np.degrees(preferred):.0f}° (HD cell)")

# %% [markdown]
# ## Part 6: Classify with `is_head_direction_cell`
#
# ``is_head_direction_cell`` is the standalone classifier used for
# screening. It applies the standard MVL ≥ 0.4 and Rayleigh p < 0.05
# criteria.

# %%
hd_is_hd = is_head_direction_cell(hd_spikes, times, headings)
pc_is_hd = is_head_direction_cell(pc_spikes, times, headings)
print(f"HD cell classified as HD:    {hd_is_hd}")
print(f"Place cell classified as HD: {pc_is_hd}")

# %% [markdown]
# Equivalently, the result class has a method form that uses the same
# defaults but is cheaper if you already computed the rate map:

# %%
print(f"hd_result.is_head_direction_cell() = {hd_result.is_head_direction_cell()}")
print(f"pc_result.is_head_direction_cell() = {pc_result.is_head_direction_cell()}")

# %% [markdown]
# ## Part 7: Polar Tuning Curve Plot
#
# A polar plot is the natural representation for circular tuning. The
# radial axis is firing rate; the angular axis is heading.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"projection": "polar"})

for ax, result, label, true_dir in [
    (axes[0], hd_result, "HD cell", preferred),
    (axes[1], pc_result, "Place cell (control)", None),
]:
    rates = np.asarray(result.firing_rate)
    centers = np.asarray(result.bin_centers)
    # Close the loop for a continuous curve
    theta = np.concatenate([centers, centers[:1]])
    r = np.concatenate([rates, rates[:1]])
    ax.plot(theta, r, "tab:blue", linewidth=2)
    ax.fill(theta, r, "tab:blue", alpha=0.25)
    if true_dir is not None:
        ax.plot(
            [true_dir, true_dir],
            [0, np.nanmax(rates)],
            "tab:red",
            linewidth=2,
            linestyle="--",
            label="True preferred",
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    ax.set_title(
        f"{label}\nMVL = {result.mean_vector_length():.3f}, "
        f"p = {result.rayleigh_pvalue():.2e}",
        pad=20,
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# In this notebook, you learned:
#
# ### Key Concepts
# - **Head direction cells** fire when the animal faces a specific
#   allocentric direction, independent of position
# - **Mean vector length (MVL)** quantifies tuning sharpness in [0, 1]
# - **Rayleigh test** rejects the null hypothesis of uniform firing
#   across directions
# - HD tuning curves are circular; use polar plots for visualization
#
# ### API
# - ``HeadDirectionCellModel`` simulates a von Mises HD cell with
#   configurable concentration (sharpness) and preferred direction
# - ``compute_directional_rate`` builds a circular tuning curve from
#   spike times + headings (no ``env`` argument - heading is a
#   circular variable, not a spatial position)
# - ``DirectionalRateResult`` exposes ``mean_vector_length``,
#   ``rayleigh_pvalue``, ``preferred_direction``, and
#   ``is_head_direction_cell``
# - ``is_head_direction_cell`` is the standalone classifier with
#   default MVL ≥ 0.4 and α = 0.05
#
# ### Classification
# - HD cells typically have MVL ≥ 0.4 and Rayleigh p < 0.05
# - Place cells with uniform coverage of headings at their preferred
#   location should not pass these thresholds
# - Real recordings benefit from shuffling controls in addition to the
#   Rayleigh test
#
# ### Next Steps
# - Compute HD tuning across multiple cells with
#   ``compute_directional_rates`` (batch version)
# - Combine with [object-vector cells](24_object_vector_cells.ipynb) to
#   model conjunctive (HD x location) coding
# - For an analysis that crosses HD with allocentric position, see the
#   ``encoding.directional`` module
#
# ### References
# - Taube, J. S., Muller, R. U., & Ranck, J. B. (1990). Head-direction
#   cells recorded from the postsubiculum in freely moving rats. I.
#   Description and quantitative analysis. *Journal of Neuroscience*,
#   10(2), 420-435.
