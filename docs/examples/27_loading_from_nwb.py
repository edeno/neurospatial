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
# # Loading from NWB
#
# **NWB (Neurodata Without Borders)** is the dominant interoperable
# file format for systems neuroscience. neurospatial reads position,
# pose, spikes, events, and intervals from NWB files - and writes
# derived results (environments, place fields, occupancy maps, laps,
# trials, region crossings) back into them.
#
# This notebook demonstrates the round-trip:
#
# 1. Build a synthetic NWB file with position + spike data
# 2. Read position from NWB, build a neurospatial ``Environment``
# 3. Compute a place field and write it back into the NWB file
# 4. Read the place field and environment back, confirm round-tripping
#
# We use a synthetic in-memory NWB file so the notebook runs without
# downloading anything. The same API works on real DANDI datasets - see
# the [io.nwb API reference](../../api/neurospatial/io/nwb/) for
# the full surface.
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# - Build a minimal NWB file with position data
# - Read NWB position with ``read_position``
# - Build an environment directly from NWB with ``environment_from_position``
# - Write derived results back with ``write_environment`` and ``write_place_field``
# - Read environments back with ``read_environment``
#
# **Estimated time**: 20-25 minutes
#
# **Prerequisites**:
# [01_introduction_basics.ipynb](../01_introduction_basics/),
# [11_place_field_analysis.ipynb](../11_place_field_analysis/).
# Requires the optional ``pynwb`` dependency
# (``uv add pynwb`` or install ``neurospatial[nwb]``).

# %% [markdown]
# ## Setup

# %%
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np

try:
    from pynwb import NWBFile  # type: ignore[import-not-found]
    from pynwb.behavior import Position, SpatialSeries  # type: ignore[import-not-found]
except ImportError as err:
    # pynwb ships as a required dependency, so this should not happen
    # in a normal install. If it does, surface a clear actionable error
    # rather than failing deep in the analysis cells. RuntimeError shows
    # cleanly in a Jupyter cell (no SystemExit traceback box).
    raise RuntimeError(
        "pynwb is not installed. Run: pip install pynwb (or uv add pynwb), "
        "then restart the kernel."
    ) from err

from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rate
from neurospatial.io.nwb import (
    environment_from_position,
    read_environment,
    read_position,
    write_environment,
    write_place_field,
)
from neurospatial.simulation import (
    PlaceCellModel,
    generate_poisson_spikes,
    simulate_trajectory_ou,
)

_here = (
    str(Path(__file__).resolve().parent) if "__file__" in globals() else str(Path.cwd())
)
if _here not in sys.path:
    sys.path.insert(0, _here)
from _style import apply_style

apply_style(figsize=(10, 8))

# %% [markdown]
# ## Part 1: Build a Synthetic NWB File
#
# In a real workflow you would do
# ``NWBHDF5IO("session.nwb", "r")`` and skip this section. Here we
# construct an in-memory NWB file with realistic position + spike data.

# %%
xx, yy = np.meshgrid(np.linspace(0, 100, 41), np.linspace(0, 100, 41))
samples = np.column_stack([xx.ravel(), yy.ravel()])
sim_env = Environment.from_samples(samples, bin_size=4.0)
sim_env.units = "cm"

positions, times = simulate_trajectory_ou(
    sim_env,
    duration=600.0,
    dt=0.02,
    speed_units="cm",
    speed_mean=15.0,
    seed=42,
)

cell = PlaceCellModel(
    sim_env,
    center=np.array([50.0, 50.0]),
    width=10.0,
    max_rate=30.0,
    baseline_rate=0.5,
    seed=42,
)
rates = cell.firing_rate(positions)
spike_times = generate_poisson_spikes(rates, times, seed=42)

print(f"Simulated session: {len(times)} samples, {len(spike_times)} spikes")

# %% [markdown]
# Wrap the trajectory in a minimal NWB file:

# %%
nwbfile = NWBFile(
    session_description="Synthetic place-cell session",
    identifier=str(uuid4()),
    session_start_time=datetime.now().astimezone(),
)

position_container = Position(name="Position")
position_container.add_spatial_series(
    SpatialSeries(
        name="position",
        description="Animal head position",
        data=positions,
        timestamps=times,
        reference_frame="Arena corner (0, 0)",
        unit="cm",
    )
)
behavior_module = nwbfile.create_processing_module(
    name="behavior",
    description="Behavioural processing",
)
behavior_module.add(position_container)

print(f"NWB file built: {nwbfile.identifier[:8]}...")
print(f"Processing modules: {list(nwbfile.processing.keys())}")

# %% [markdown]
# ## Part 2: Read Position from NWB
#
# ``read_position`` auto-discovers the Position container (priority:
# ``processing/behavior`` > ``processing/*`` > ``acquisition``). Pass
# ``processing_module`` and / or ``position_name`` to disambiguate when
# multiple series exist.

# %%
positions_nwb, times_nwb = read_position(nwbfile)
print(f"Read positions: shape={positions_nwb.shape}, dtype={positions_nwb.dtype}")
print(
    f"Read times:     shape={times_nwb.shape}, range=[{times_nwb[0]:.1f}, "
    f"{times_nwb[-1]:.1f}] s"
)
print(f"Round-trip match: {np.allclose(positions_nwb, positions)}")

# %% [markdown]
# ## Part 3: Build an Environment Directly from NWB
#
# ``environment_from_position`` reads the position channel and calls
# ``Environment.from_samples`` for you. ``units`` is auto-detected
# from the SpatialSeries.

# %%
env = environment_from_position(nwbfile, bin_size=4.0)
print(f"Environment: {env.n_bins} bins, units={env.units}")

# %% [markdown]
# ## Part 4: Compute a Place Field
#
# Standard ``compute_spatial_rate`` workflow using the NWB-derived
# environment and the spike times we already have.

# %%
result = compute_spatial_rate(
    env,
    spike_times,
    times_nwb,
    positions_nwb,
    smoothing_method="diffusion_kde",
    bandwidth=5.0,
)
print(f"Place field peak: {np.nanmax(result.firing_rate):.2f} Hz")
print(f"Spatial information: {result.spatial_information():.3f} bits/spike")

fig, ax = plt.subplots(figsize=(7, 7))
env.plot_field(result.firing_rate, ax=ax, cmap="hot", colorbar_label="Firing rate (Hz)")
ax.scatter(
    [50],
    [50],
    c="cyan",
    s=200,
    marker="*",
    edgecolors="white",
    linewidths=2,
    zorder=5,
    label="True centre",
)
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_title(
    f"Place field from NWB-derived environment\n"
    f"info = {result.spatial_information():.3f} bits/spike"
)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 5: Write Results Back into the NWB File
#
# Neurospatial provides ``write_environment`` (stores environment
# metadata, bin centres, and the edge list in scratch/) and
# ``write_place_field`` (stores the
# rate map aligned to bin centers in analysis/). Both attach to an
# in-memory NWBFile; persist with ``NWBHDF5IO(...).write(nwbfile)``.

# %%
write_environment(nwbfile, env, name="my_environment")
write_place_field(
    nwbfile,
    env,
    np.asarray(result.firing_rate),
    name="cell_001_place_field",
    description="Place field for simulated cell 001",
    unit="Hz",
)
print(f"Scratch entries: {list(nwbfile.scratch.keys())}")
analysis_module = nwbfile.processing["analysis"]
print(
    f"Analysis processing module entries: "
    f"{list(analysis_module.data_interfaces.keys())}"
)

# %% [markdown]
# ## Part 6: Round-Trip the Environment
#
# ``read_environment`` reconstructs the full ``Environment`` (active
# bins, connectivity graph, regions, layout) from the stored bin
# centres and edge list. Check the pieces we care about to confirm
# the round-trip is structurally faithful:

# %%
env_back = read_environment(nwbfile, name="my_environment")
print(f"Original env:   {env.n_bins} bins, units={env.units}")
print(f"Round-trip env: {env_back.n_bins} bins, units={env_back.units}")
print(f"bin_centers match: {np.allclose(env.bin_centers, env_back.bin_centers)}")
print(
    f"n_edges match:     {env.connectivity.number_of_edges() == env_back.connectivity.number_of_edges()}"
)
print(f"units match:       {env.units == env_back.units}")
# Note: regions and layout class are also preserved; see read_environment
# documentation for the full equivalence contract.

# %% [markdown]
# ## Summary
#
# In this notebook, you learned:
#
# ### NWB Round-Trip API
# - ``read_position`` reads a Position SpatialSeries from
#   ``processing/behavior`` (or anywhere in ``processing/*`` or
#   ``acquisition``)
# - ``environment_from_position`` is a one-liner that reads position
#   and calls ``Environment.from_samples``
# - ``read_environment`` reads back a previously-written environment,
#   reconstructing the layout, active bins, and connectivity graph
# - ``write_environment`` stores environment metadata, bin centres,
#   and the edge list in ``scratch/``
# - ``write_place_field`` stores a rate map aligned to bin centres in
#   ``analysis/``
#
# ### Persistence
# - These functions operate on an in-memory ``NWBFile``; use
#   ``NWBHDF5IO(path, mode).write(nwbfile)`` to flush changes to disk
#
# ### Other Readers/Writers
# - ``read_pose``: pose tracking (body parts)
# - ``read_head_direction``: ``CompassDirection`` data
# - ``read_events`` / ``read_intervals`` / ``read_trials``: discrete
#   events, epoch intervals, trial tables
# - ``write_laps`` / ``write_region_crossings`` / ``write_trials`` /
#   ``write_events``: behavioural segmentation outputs
# - ``write_occupancy``: occupancy maps
#
# ### Next Steps
# - For real DANDI data, point ``NWBHDF5IO`` at a downloaded ``.nwb``
#   file and the same API applies
# - For an end-to-end real-data example, see
#   [19_real_data_bandit_task.ipynb](../19_real_data_bandit_task/)
#
# ### References
# - [NWB website](https://www.nwb.org/)
# - [DANDI archive](https://dandiarchive.org/) - public NWB datasets
