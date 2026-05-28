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
# Object-vector cells (OVCs) fire when an object is at a specific
# **distance and direction** relative to the animal's heading -
# regardless of the animal's absolute position. They were first
# described in the medial entorhinal cortex (Hoydal et al., 2019).
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
# - **Object-vector cell**: fires when an OBJECT is at a specific
#   (distance, egocentric direction) relative to the animal
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# - Simulate object-vector cells with known preferred distance and direction
# - Compute egocentric rate maps with ``compute_egocentric_rate``
# - Interpret tuning in (distance, egocentric direction) polar coordinates
# - Compute the object-vector score and classify candidate OVCs
# - Distinguish OVCs from place cells using egocentric spatial information
#
# **Estimated time**: 20-25 minutes
#
# **Prerequisites**: [11_place_field_analysis.ipynb](11_place_field_analysis.ipynb)

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
    is_object_vector_cell,
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
from _style import apply_style  # noqa: E402

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
headings = heading_from_velocity(positions, dt, min_speed=2.0, bandwidth=3.0)

print(f"Trajectory: {len(times)} samples, {times[-1]:.1f}s")
print(
    f"Position range: x=[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}], "
    f"y=[{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]"
)

# %% [markdown]
# ## Part 2: Place an Object in the Environment
#
# An OVC's firing rate depends on the *egocentric* position of an object
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
# A place cell fires when the animal is AT a fixed location. We use
# this as a negative control - it should NOT pass the OVC classifier.

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
    smoothing_method="gaussian_kde",
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
    smoothing_method="gaussian_kde",
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
    smoothing_method="diffusion_kde",
    bandwidth=8.0,
)
pc_place_result = compute_spatial_rate(
    env,
    pc_spikes,
    times,
    positions,
    smoothing_method="diffusion_kde",
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
# A score above ~0.3 with peak rate above ~5 Hz typically passes the
# default OVC classifier.

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
ovc_egoc_info = ovc_result.egocentric_spatial_information()
pc_egoc_info = pc_result.egocentric_spatial_information()

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
# Two ways to classify a candidate OVC, with a caveat about which
# tuning curve each one uses:
#
# 1. **``is_object_vector_cell``** — the library's one-shot screener.
#    It internally calls ``compute_egocentric_rate`` with the default
#    smoothing (``method="binned"``, no bandwidth, no occupancy
#    threshold) and compares ``object_vector_score`` to its default
#    ``score_threshold=0.3``. This is fast but the raw-binned tuning
#    is noisy in egocentric polar coordinates, so the score is
#    typically lower than what you would get on a smoothed tuning.
# 2. **Manual two-criterion check** on the *smoothed* tuning we
#    already computed: ``object_vector_score`` plus
#    ``egocentric_spatial_information``. Smoothing buys a much cleaner
#    score but requires picking smoothing parameters explicitly, so
#    thresholds need to be tuned to your recording.
#
# We show both. They generally agree about whether a cell is an OVC,
# but the score values themselves differ — pick one workflow and stick
# with it. See the ``is_object_vector_cell`` docstring for guidance
# on choosing thresholds.

# %%
# 1. Library one-shot screener (raw-binned tuning, default thresholds)
ovc_is_ovc = is_object_vector_cell(
    env,
    ovc_spikes,
    times,
    positions,
    headings,
    object_positions,
    distance_range=(0.0, 50.0),
    n_distance_bins=10,
    n_direction_bins=12,
)
pc_is_ovc = is_object_vector_cell(
    env,
    pc_spikes,
    times,
    positions,
    headings,
    object_positions,
    distance_range=(0.0, 50.0),
    n_distance_bins=10,
    n_direction_bins=12,
)
print("is_object_vector_cell (library, default thresholds):")
print(f"  OVC -> {ovc_is_ovc}")
print(f"  Place cell -> {pc_is_ovc}")

# 2. Manual screening on the smoothed tuning we computed in Part 5.
# Thresholds chosen for this simulation - 0.1 is permissive given the
# small absolute scores typical of smoothed egocentric polar maps;
# 1.0 bits/spike is well above the library's 0.3-default but
# comfortably below the strong-OVC range (0.5-1.5+) the
# ``EgocentricRateResult.is_object_vector_cell`` docstring discusses.
# Tune both to your recording.
score_threshold = 0.1
info_threshold = 1.0
print(
    "\nManual screening (smoothed tuning, demo thresholds "
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
# ## Summary
#
# In this notebook, you learned:
#
# ### Key Concepts
# - **Object-vector cells** fire when an object is at a specific
#   (distance, egocentric direction) from the animal
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
# - ``compute_egocentric_rate`` returns an ``EgocentricRateResult`` with
#   ``firing_rate``, ``occupancy``, and tuning summaries
#   (``preferred_distance``, ``preferred_direction``,
#   ``egocentric_spatial_information``)
# - ``object_vector_score`` collapses a tuning curve into a single
#   selectivity score in [0, 1]
# - ``is_object_vector_cell`` is a one-shot screening function that
#   bundles score + peak-rate thresholds
# - ``plot_object_vector_tuning`` renders the egocentric rate map on a
#   polar axis
#
# ### Classification
# - The library default ``is_object_vector_cell`` uses
#   ``score_threshold=0.3`` and ``min_peak_rate=5`` Hz on the
#   *raw-binned* tuning - fast screen, conservative
# - Computing the score on a smoothed tuning (as in Part 5) gives
#   higher absolute scores but requires picking smoothing parameters
#   and thresholds appropriate to your recording
# - Comparing egocentric vs allocentric spatial information (Part 8)
#   is a useful sanity check: OVCs are higher in the egocentric frame
#
# ### Next Steps
# - Apply to real recordings with tracked head direction and known
#   object positions
# - Try ``metric="geodesic"`` for environments with obstacles
# - Combine with [spatial view cells](22_spatial_view_cells.ipynb) for a
#   fuller picture of egocentric coding
#
# ### References
# - Hoydal, O. A., et al. (2019). Object-vector coding in the medial
#   entorhinal cortex. *Nature*, 568(7752), 400-404.
