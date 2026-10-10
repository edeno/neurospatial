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
# # Object-Vector Cell Analysis
#
# Allocentric object-vector cells (Høydal et al., 2019) encode distance
# and world-relative direction to objects. Egocentric bearing cells
# (Wang et al., 2018) encode direction relative to the animal's heading.
# This notebook first demonstrates an explicitly egocentric model, then
# compares both analysis frames for an allocentric model.
# References: https://doi.org/10.1038/s41586-019-1077-7 and
# https://doi.org/10.1126/science.aau4940.
#
# This notebook demonstrates:
#
# 1. Simulating an OVC with known tuning
# 2. Computing an egocentric rate map (distance x direction to object)
# 3. Comparing the egocentric field to the standard place field
# 4. Classifying neurons using the object-vector score
#
# **Key difference from place cells:**
# - **Place cell**: fires when animal is AT a location
# - **Allocentric object-vector cell**: tuned to distance and world direction
# - **Egocentric bearing cell**: tuned to distance and heading-relative bearing
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# - Simulate object-vector cells with known preferred distance and direction
# - Compute egocentric rate maps with ``compute_egocentric_rate``
# - Interpret tuning in (distance, egocentric direction) polar coordinates
# - Compute the object-vector score and classify candidate OVCs
# - Compare frame-specific tuning with an explicit place-cell control
#
# **Estimated time**: 20-25 minutes
#
# **Prerequisites**: [11_place_field_analysis.ipynb](../11_place_field_analysis/)

# %% [markdown]
# ## Setup

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neurospatial import Environment
from neurospatial.encoding import (
    compute_egocentric_rate,
    compute_spatial_rate,
    egocentric_object_vector_cell_significance,
    is_egocentric_object_vector_cell,
    object_vector_score,
    plot_object_vector_tuning,
)
from neurospatial.ops.egocentric import heading_from_velocity
from neurospatial.simulation import (
    ObjectVectorCellModel,
    PlaceCellModel,
    generate_poisson_spikes,
    simulate_trajectory_ou,
)

# Shared styling (Okabe-Ito palette, consistent figure / font sizes)
_here = (
    str(Path(__file__).resolve().parent) if "__file__" in globals() else str(Path.cwd())
)
if _here not in sys.path:
    sys.path.insert(0, _here)
from _style import apply_style

apply_style(figsize=(12, 10))

# %% [markdown]
# ## Part 1: Create Environment and Trajectory
#
# We build a square open field and simulate animal movement with
# the Ornstein-Uhlenbeck process, which produces biologically realistic
# exploration statistics (fitted to Sargolini et al. 2006).

# %%
# Create environment from a dense grid of sampled positions
xx, yy = np.meshgrid(np.linspace(0, 100, 41), np.linspace(0, 100, 41))
samples = np.column_stack([xx.ravel(), yy.ravel()])
env = Environment.from_samples(samples, bin_size=4.0)
env.units = "cm"
print(f"Environment: {env.n_bins} bins")

# %%
# Generate a long, smooth trajectory with realistic statistics
positions, times = simulate_trajectory_ou(
    env,
    duration=1200.0,
    dt=0.02,
    speed_units="cm",
    speed_mean=15.0,
    speed_std=5.0,
    seed=42,
)
dt = float(times[1] - times[0])

# Compute heading from velocity (radians, world frame: 0=East, +pi/2=North)
headings = heading_from_velocity(times, positions, min_speed=2.0, bandwidth=3.0)

print(f"Trajectory: {len(times)} samples, {times[-1]:.1f}s")
print(
    f"Position range: x=[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}], "
    f"y=[{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]"
)

# %% [markdown]
# ## Part 2: Place an Object in the Environment
#
# This egocentric model depends on object bearing relative to heading
# relative to the animal. We place a single object near the center of the
# arena so the animal experiences it from many distances and directions.

# %%
object_positions = np.array([[50.0, 50.0]])

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(
    positions[:, 0],
    positions[:, 1],
    "gray",
    alpha=0.3,
    linewidth=0.5,
    label="Trajectory",
)
ax.scatter(
    object_positions[:, 0],
    object_positions[:, 1],
    c="red",
    s=250,
    marker="X",
    zorder=5,
    label="Object",
    edgecolors="black",
    linewidths=2,
)
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_title("Arena with Object and Trajectory")
ax.set_aspect("equal")
ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# ## Part 3: Simulate an Object-Vector Cell
#
# The cell below fires when the object is at distance ~20 cm and at
# egocentric direction ~+π/2 (to the animal's **left**).
#
# **Egocentric direction convention:** 0 = ahead, +π/2 = left, -π/2 = right.

# %%
preferred_distance = 20.0
preferred_direction = np.pi / 2  # to the left

ovc_model = ObjectVectorCellModel(
    direction_frame="egocentric",
    env=env,
    object_positions=object_positions,
    preferred_distance=preferred_distance,
    distance_width=5.0,
    preferred_direction=preferred_direction,
    direction_kappa=4.0,  # ~30 deg half-width directional tuning
    max_rate=60.0,
    baseline_rate=0.05,
)

# firing_rate requires headings when preferred_direction is set
ovc_rates = ovc_model.firing_rate(positions, headings=headings)
ovc_spikes = generate_poisson_spikes(ovc_rates, times, seed=42)

print(f"Object-vector cell: {len(ovc_spikes)} spikes")
print(f"Mean firing rate: {len(ovc_spikes) / times[-1]:.2f} Hz")
print(f"Peak instantaneous rate: {ovc_rates.max():.2f} Hz")
print(
    f"Preferred (distance, direction): ({preferred_distance:.1f} cm, "
    f"{np.degrees(preferred_direction):.0f}°)"
)

# %% [markdown]
# ## Part 4: Simulate a Place Cell for Comparison
#
# A place cell fires when the animal is AT a fixed location. We use it
# as a negative control. As Part 9 shows, a structured place cell can
# still clear the one-shot ``is_egocentric_object_vector_cell`` info-only screen
# (its egocentric tuning carries information), but it fails the stricter
# manual score-plus-info check that captures true object-vector tuning.

# %%
place_model = PlaceCellModel(
    env=env,
    center=np.array([30.0, 30.0]),
    width=8.0,
    max_rate=40.0,
    baseline_rate=0.1,
)
pc_rates = place_model.firing_rate(positions)
pc_spikes = generate_poisson_spikes(pc_rates, times, seed=43)

print(f"Place cell: {len(pc_spikes)} spikes")
print(f"Mean firing rate: {len(pc_spikes) / times[-1]:.2f} Hz")

# %% [markdown]
# ## Part 5: Compute Egocentric Rate Maps
#
# ``compute_egocentric_rate`` builds a firing-rate map in polar
# (distance, egocentric direction) coordinates. The radial axis is the
# distance to the (nearest) object; the angular axis is the egocentric
# bearing.

# %%
ovc_result = compute_egocentric_rate(
    env,
    ovc_spikes,
    times,
    positions,
    headings,
    object_positions,
    distance_range=(0.0, 50.0),
    n_distance_bins=10,
    n_direction_bins=12,
    method="gaussian_kde",
    bandwidth=1.0,
    min_occupancy=1.0,
)
print("OVC egocentric rate computed:")
print(f"  Peak firing rate:    {np.nanmax(ovc_result.firing_rate):.2f} Hz")
print(
    f"  Preferred distance:  {ovc_result.preferred_distance():.1f} cm "
    f"(true: {preferred_distance:.1f} cm)"
)
print(
    f"  Preferred direction: {np.degrees(ovc_result.preferred_direction()):.0f}° "
    f"(true: {np.degrees(preferred_direction):.0f}°)"
)

# %%
pc_result = compute_egocentric_rate(
    env,
    pc_spikes,
    times,
    positions,
    headings,
    object_positions,
    distance_range=(0.0, 50.0),
    n_distance_bins=10,
    n_direction_bins=12,
    method="gaussian_kde",
    bandwidth=1.0,
    min_occupancy=1.0,
)
print("Place cell egocentric rate computed:")
print(f"  Peak firing rate: {np.nanmax(pc_result.firing_rate):.2f} Hz")

# %% [markdown]
# ## Part 6: Standard Place Fields for Both Cells
#
# We also compute the standard (allocentric) rate map for each cell so we
# can directly compare egocentric and allocentric tuning.

# %%
ovc_place_result = compute_spatial_rate(
    env,
    ovc_spikes,
    times,
    positions,
    method="diffusion_kde",
    bandwidth=8.0,
)
pc_place_result = compute_spatial_rate(
    env,
    pc_spikes,
    times,
    positions,
    method="diffusion_kde",
    bandwidth=8.0,
)

print(f"OVC place-field peak:        {np.nanmax(ovc_place_result.firing_rate):.2f} Hz")
print(f"Place cell place-field peak: {np.nanmax(pc_place_result.firing_rate):.2f} Hz")

# %% [markdown]
# ## Part 7: Visualize the Comparison
#
# - **Object-vector cell**: sharp peak in egocentric polar coordinates,
#   diffuse standard place field.
# - **Place cell**: sharp standard place field, diffuse in egocentric
#   coordinates.

# %%
fig = plt.figure(figsize=(13, 11))

ax = fig.add_subplot(2, 2, 1, projection="polar")
plot_object_vector_tuning(ovc_result, ax=ax, cmap="hot", add_colorbar=True)
ax.set_title(
    "Object-Vector Cell: Egocentric Field\n(distance x direction to object)",
    fontweight="bold",
    pad=15,
)

ax = fig.add_subplot(2, 2, 2)
env.plot_field(
    ovc_place_result.firing_rate, ax=ax, cmap="hot", colorbar_label="Firing rate (Hz)"
)
ax.scatter(
    object_positions[:, 0],
    object_positions[:, 1],
    c="cyan",
    s=180,
    marker="X",
    edgecolors="white",
    linewidths=2,
    zorder=5,
)
ax.set_title(
    "Object-Vector Cell: Place Field\n(binned by animal position)", fontweight="bold"
)
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")

ax = fig.add_subplot(2, 2, 3, projection="polar")
plot_object_vector_tuning(pc_result, ax=ax, cmap="hot", add_colorbar=True)
ax.set_title(
    "Place Cell: Egocentric Field\n(distance x direction to object)",
    fontweight="bold",
    pad=15,
)

ax = fig.add_subplot(2, 2, 4)
env.plot_field(
    pc_place_result.firing_rate, ax=ax, cmap="hot", colorbar_label="Firing rate (Hz)"
)
ax.scatter(
    [30],
    [30],
    c="cyan",
    s=180,
    marker="*",
    edgecolors="white",
    linewidths=2,
    zorder=5,
)
ax.set_title("Place Cell: Place Field\n(binned by animal position)", fontweight="bold")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")

fig.suptitle(
    "Object-Vector Cell vs Place Cell Comparison",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

print("\nObservations:")
print("- OVC: egocentric field shows clear peak at preferred distance/direction")
print("- OVC: place field is diffuse (fires from many positions)")
print("- Place cell: place field shows clear peak at preferred location")
print("- Place cell: egocentric field is diffuse")

# %% [markdown]
# ## Part 8: Object-Vector Score and Classification
#
# The **object-vector score** combines distance selectivity and
# direction selectivity into a single number in [0, 1]:
#
# - Distance selectivity = peak / mean (normalized)
# - Direction selectivity = mean resultant length over direction bins
#
# A higher score indicates stronger object-vector tuning. Note the
# one-shot ``is_egocentric_object_vector_cell`` classifier does *not* threshold this
# score directly: it classifies on egocentric spatial information against
# its default ``min_info=0.3`` bits/spike (see Part 9).

# %%
_shape = (ovc_result.n_distance_bins, ovc_result.n_direction_bins)
ovc_tuning = np.asarray(ovc_result.firing_rate).reshape(_shape)
pc_tuning = np.asarray(pc_result.firing_rate).reshape(_shape)

ovc_score = object_vector_score(ovc_tuning)
pc_score = object_vector_score(pc_tuning)

print("=" * 60)
print("OBJECT-VECTOR SCORE")
print("=" * 60)
print(f"  OVC score:        {ovc_score:.3f}")
print(f"  Place cell score: {pc_score:.3f}")

# Egocentric spatial information (bits/spike, Skaggs in polar coords)
ovc_egoc_info = ovc_result.spatial_information()
pc_egoc_info = pc_result.spatial_information()

# Allocentric spatial information (bits/spike, Skaggs over the arena)
ovc_alloc_info = ovc_place_result.spatial_information()
pc_alloc_info = pc_place_result.spatial_information()

print("\n" + "=" * 60)
print("SPATIAL INFORMATION (bits/spike)")
print("=" * 60)
print(f"{'Metric':<30} {'OVC':<12} {'Place Cell':<12}")
print("-" * 60)
print(f"{'Egocentric info':<30} {ovc_egoc_info:<12.3f} {pc_egoc_info:<12.3f}")
print(f"{'Allocentric info':<30} {ovc_alloc_info:<12.3f} {pc_alloc_info:<12.3f}")

# %% [markdown]
# ## Part 9: Two-Sided Classification
#
# Information screens identify candidate associations, not cell identity.
# Raw and smoothed estimators can give either ordering of information;
# raw binning is not universally lower or more conservative. Compare a free
# predicate and a result method using the same estimator and parameters.
# We show the library information screen, a manually chosen score+information
# screen, and a calibrated circular-shift verdict. The explicit place-cell
# control can have significant polar information too: a shuffle breaks
# alignment with behavior, not the distinction between place and object tuning.

# %%
# 1. Library information screen (binned, bandwidth=5 cm, min_info=0.3)
ovc_is_ovc = is_egocentric_object_vector_cell(
    env,
    ovc_spikes,
    times,
    positions,
    headings,
    object_positions,
    distance_range=(0.0, 50.0),
    n_distance_bins=10,
    n_direction_bins=12,
    criterion="threshold",
    method="binned",
    bandwidth=5.0,
)
pc_is_ovc = is_egocentric_object_vector_cell(
    env,
    pc_spikes,
    times,
    positions,
    headings,
    object_positions,
    distance_range=(0.0, 50.0),
    n_distance_bins=10,
    n_direction_bins=12,
    criterion="threshold",
    method="binned",
    bandwidth=5.0,
)
raw_ovc_result = compute_egocentric_rate(
    env,
    ovc_spikes,
    times,
    positions,
    headings,
    object_positions,
    method="binned",
    bandwidth=5.0,
)
raw_pc_result = compute_egocentric_rate(
    env,
    pc_spikes,
    times,
    positions,
    headings,
    object_positions,
    method="binned",
    bandwidth=5.0,
)
assert ovc_is_ovc == raw_ovc_result.is_object_vector_cell(min_info=0.3)
assert pc_is_ovc == raw_pc_result.is_object_vector_cell(min_info=0.3)
print(
    "Egocentric candidates (criterion=threshold, method=binned, bandwidth=5, min_info=0.3):"
)
print(f"  OVC -> {ovc_is_ovc}")
print(f"  Place cell -> {pc_is_ovc}")
if pc_is_ovc:
    print(
        "  (Note: the place cell clears the info-only screen; the stricter "
        "manual score+info check below rejects it.)"
    )

# 2. Manual screening on the smoothed tuning we computed in Part 5.
# Demo cutoffs are screening choices, not calibrated false-positive levels.
score_threshold = 0.1
info_threshold = 1.0
print(
    "\nManual screening (diffusion_kde, bandwidth=5, demo thresholds "
    f"score>{score_threshold}, info>{info_threshold} bits/spike):"
)
print(f"  {'Metric':<32} {'OVC':<10} {'Place':<10}")
print(f"  {'Object-vector score':<32} {ovc_score:<10.3f} {pc_score:<10.3f}")
print(
    f"  {'Egocentric info (bits/spike)':<32} {ovc_egoc_info:<10.3f} "
    f"{pc_egoc_info:<10.3f}"
)

ovc_passes = ovc_score > score_threshold and ovc_egoc_info > info_threshold
pc_passes = pc_score > score_threshold and pc_egoc_info > info_threshold
print(f"  OVC -> {ovc_passes}")
print(f"  Place cell -> {pc_passes}")

# %% [markdown]
# ## Part 10: Circular-Shift Significance
#
# Test egocentric spatial information with the same binned estimator as the
# library screen. Fifty shuffles keep this tutorial quick (minimum p=1/51);
# use more for a precise publication p-value. Shifts use the joined valid
# recording clock, preserve circular spike spacing there, and assume stable
# firing statistics. Neither significance nor a cutoff alone establishes
# object-vector identity; retain the place-cell control and compare frames.

# %%
shuffle_results = egocentric_object_vector_cell_significance(
    env,
    [ovc_spikes, pc_spikes],
    times,
    positions,
    headings,
    object_positions,
    unit_ids=["egocentric_model", "place_control"],
    method="binned",
    bandwidth=5.0,
    n_shuffles=50,
    min_shift=20.0,
    rng=0,
)
print(
    "Egocentric association (criterion=shuffle, method=binned, bandwidth=5, alpha=0.05):"
)
for label, test in shuffle_results.items():
    print(f"  {label}: p={test.p_value:.4f}, verdict={test.p_value < 0.05}")
    spikes = ovc_spikes if label == "egocentric_model" else pc_spikes
    assert is_egocentric_object_vector_cell(
        env,
        spikes,
        times,
        positions,
        headings,
        object_positions,
        criterion="shuffle",
        method="binned",
        bandwidth=5.0,
        n_shuffles=50,
        min_shift=20.0,
        rng=0,
        unit_id=label,
    ) == (test.p_value < 0.05)

# %% [markdown]
# ## Summary
#
# In this notebook, you learned:
#
# ### Key Concepts
# - **Allocentric object-vector cells** use world-relative direction;
#   **egocentric bearing cells** use direction relative to heading.
# - **Egocentric rate maps** (``compute_egocentric_rate``) index firing
#   by polar coordinates relative to the *nearest* object at each
#   timepoint
# - The animal's heading determines the egocentric reference frame:
#   0 = ahead, +π/2 = left, -π/2 = right
# - ``ObjectVectorCellModel`` exposes an ``object_selectivity``
#   parameter (``"nearest"`` / ``"any"`` / ``"specific"``) for
#   simulating cells that respond to a specific object or the maximum
#   response across all objects; the analysis function
#   ``compute_egocentric_rate`` always uses the nearest object
#
# ### API
# - ``ObjectVectorCellModel`` simulates a ground-truth OVC with
#   configurable distance / direction tuning
# - ``compute_egocentric_rate`` returns an ``ObjectVectorRateResult`` with
#   ``firing_rate``, ``occupancy``, and tuning summaries
#   (``preferred_distance``, ``preferred_direction``,
#   ``spatial_information``)
# - ``object_vector_score`` collapses a tuning curve into a single
#   selectivity score in [0, 1]
# - ``is_egocentric_object_vector_cell`` is a one-shot screening function that
#   classifies on egocentric spatial information (``min_info`` threshold)
# - ``plot_object_vector_tuning`` renders the egocentric rate map on a
#   polar axis
#
# ### Classification
# - The library default ``is_egocentric_object_vector_cell`` uses
#   ``min_info=0.3`` bits/spike (egocentric spatial information) on the
#   binned tuning with bandwidth=5 - a fast, biased candidate screen
# - Raw/smoothed information ordering depends on the data and estimator.
#   Report the estimator and criterion for every verdict.
# - Circular shifts calibrate association under a stable-clock null; a
#   significant place-cell control still does not establish object-vector identity.
# - Comparing egocentric vs allocentric spatial information (Part 8)
#   is a frame-specific check: the egocentric model should favor egocentric
#   tuning; the allocentric model below should favor allocentric tuning.
#
# ### Next Steps
# - Apply to real recordings with tracked head direction and known
#   object positions
# - Try ``metric="geodesic"`` for environments with obstacles
# - Combine with [spatial view cells](../22_spatial_view_cells/) for a
#   fuller picture of egocentric coding
#
# ### References
# - Hoydal, O. A., et al. (2019). Object-vector coding in the medial
#   entorhinal cortex. *Nature*, 568(7752), 400-404.

# %% [markdown]
# ## Both Reference Frames
#
# The demonstration above explicitly selects egocentric tuning. The simulator's
# default is allocentric: direction to the object is fixed in world coordinates,
# with 0 = East and +pi/2 = North. The animal-to-object vector reverses the
# object-to-animal vector used to describe a field location; add pi and wrap.
# Both maps carry their frame, use the same distance/window rules and provide
# candidate screens. The place-cell control above is retained; these screens
# do not establish biological identity.

# %%
from neurospatial.encoding import compute_object_vector_rate, is_object_vector_cell

allocentric_model = ObjectVectorCellModel(
    env=env,
    object_positions=object_positions,
    preferred_distance=20.0,
    distance_width=5.0,
    preferred_direction=np.pi,
    max_rate=60.0,
)
allocentric_spikes = generate_poisson_spikes(
    allocentric_model.firing_rate(positions),
    times,
    seed=43,
)
allocentric_result = compute_object_vector_rate(
    env,
    allocentric_spikes,
    times,
    positions,
    object_positions,
)
allocentric_ego_result = compute_egocentric_rate(
    env,
    allocentric_spikes,
    times,
    positions,
    headings,
    object_positions,
)
fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(12, 5))
for result, ax in zip([allocentric_result, allocentric_ego_result], axes, strict=True):
    plot_object_vector_tuning(result, ax=ax, add_colorbar=True)
    ax.set_title(
        f"{result.direction_frame}: {result.spatial_information():.3f} bits/spike"
    )
plt.tight_layout()
plt.show()
print(
    "Allocentric information-only candidate (default binned estimator):",
    is_object_vector_cell(
        env,
        allocentric_spikes,
        times,
        positions,
        object_positions,
    ),
)
print(
    "Recorded frames:",
    allocentric_result.direction_frame,
    allocentric_ego_result.direction_frame,
)
