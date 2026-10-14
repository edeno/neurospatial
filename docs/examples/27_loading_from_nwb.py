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
# Read position and labeled spikes, compute spatial rates with explicit recording
# support, and preserve the results in an NWB file. This synthetic example writes
# a real temporary `.nwb` file and reopens it, so no download or private data is needed.
#
# **Estimated time**: 20-25 minutes
#
# **Prerequisites**: [01: basics](01_introduction_basics.ipynb),
# [11: place fields](11_place_field_analysis.ipynb).
# This notebook requires the optional NWB extra: from the repository root use
# `uv sync --extra notebooks --extra nwb` and select that environment's kernel.
# The array-only analyses in the other notebooks do not require NWB packages.
#
# ## Setup

# %%
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.behavior import Position, SpatialSeries
except ImportError as error:
    raise ImportError(
        "Example 27 requires neurospatial[nwb]. In a checkout run "
        "`uv sync --extra notebooks --extra nwb`, then restart the kernel."
    ) from error

from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rates
from neurospatial.io.nwb import (
    environment_from_position,
    read_environment,
    read_place_field,
    read_position,
    read_units,
    write_environment,
    write_spatial_rates,
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
# ## 1. Construct position and spikes with known units and support
#
# Position and spikes share a clock in seconds. A 50 Hz trajectory represents
# ten minutes in a 100 × 100 cm arena. Use independent random streams for
# trajectory and spike generation; the complete simulation is reproducible.

# %%
xx, yy = np.meshgrid(np.linspace(0, 100, 41), np.linspace(0, 100, 41))
sim_env = Environment.from_samples(np.c_[xx.ravel(), yy.ravel()], bin_size=4.0)
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
    sim_env, center=[50.0, 50.0], width=10.0, max_rate=30.0, baseline_rate=0.5
)
spike_times = generate_poisson_spikes(cell.firing_rate(positions), times, seed=142)
recording_window = np.array([[times[0], times[-1]]])
# Keep the half-open interval used throughout the analysis.
spike_times = spike_times[(spike_times >= times[0]) & (spike_times < times[-1])]

nwbfile = NWBFile(
    session_description="Synthetic place-cell session",
    identifier="neurospatial-example-27",
    session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
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
nwbfile.create_processing_module("behavior", "Behavioral tracking").add(
    position_container
)
# Label 7 is explicitly chosen for this synthetic unit, not inferred from row order.
nwbfile.add_unit(id=7, spike_times=spike_times, obs_intervals=recording_window)
print(
    f"Synthetic recording: {len(times):,} position samples, {len(spike_times):,} spikes"
)

# %% [markdown]
# ## 2. Read arrays and preserve identity at the NWB boundary
#
# Readers return named holders. Eager arrays (the default) remain usable after
# closing a file; `lazy=True` handles must be used while their file is open.
# Specify channel/module names when multiple position series exist. Stored
# conversion and offset are applied by `read_position`; values are in `pos.units`.

# %%
pos = read_position(nwbfile, processing_module="behavior", position_name="position")
units = read_units(nwbfile)
np.testing.assert_array_equal(pos.positions, positions)
np.testing.assert_array_equal(units.unit_ids, [7])
np.testing.assert_array_equal(units.spike_times[0], spike_times)
print(f"Position units: {pos.units}; unit labels: {units.unit_ids.tolist()}")
print(f"Spike recording windows: {units.spike_window}")

# %% [markdown]
# ## 3. Estimate fields with explicit analysis choices
#
# `environment_from_position` constructs bins from the selected tracking channel.
# Check declared units before constructing an environment; missing metadata should
# be resolved from the experiment. Inferring active bins from visited samples is
# one environment choice; a known arena boundary is another.
#
# `spike_window` records acquisition coverage, and `epochs` selects analysis
# windows. They are separate concepts even though they match in this example.
# Bins without sufficient observed occupancy remain unsupported. Keep NaNs and
# the support mask instead of presenting unobserved space as a zero firing rate.

# %%
if pos.units is None:
    raise ValueError("Choose spatial units from the experiment before continuing.")
env = environment_from_position(nwbfile, bin_size=4.0, units=pos.units)
rates = compute_spatial_rates(
    env,
    units.spike_times,
    pos.times,
    pos.positions,
    unit_ids=units.unit_ids,
    epochs=recording_window,
    spike_window=units.spike_window,
    bandwidth=5.0,
    min_occupancy=0.05,
)
result = rates.sel(unit_id=7)
print(rates.summary_table())
print(f"Observed supported bins: {rates.spatial_support.sum()} / {env.n_bins}")
fig, ax = plt.subplots(figsize=(7, 7))
result.plot(ax=ax, cmap="hot")
ax.scatter(50, 50, c="cyan", s=150, marker="*", label="Simulated field center")
ax.set_title("Spatial rate estimated from NWB arrays")
ax.legend()
plt.show()

# %% [markdown]
# ## 4. Persist the full result and reopen the file
#
# `write_spatial_rates` stores maps, occupancy, unit labels, analysis settings,
# support, and a companion environment. `read_place_field` is its population-result
# reader despite its singular name. `write_place_field` is a lower-level array
# writer and does not preserve the complete result provenance.
#
# Writers modify the in-memory NWBFile; `NWBHDF5IO.write` persists it. This demo
# uses a fresh temporary directory on each run rather than overwriting recordings.
# To add results to your own file, choose an output copy or deliberate update
# workflow, and register ndx-pose/ndx-events before opening files containing those
# extensions. This minimal example uses only core NWB containers.

# %%
write_environment(nwbfile, env, name="analysis_environment")
write_spatial_rates(nwbfile, rates, name="spatial_rates")
temporary_data = tempfile.TemporaryDirectory(prefix="neurospatial-nwb-example-")
nwb_path = Path(temporary_data.name) / "synthetic_session.nwb"
with NWBHDF5IO(str(nwb_path), "w") as io:
    io.write(nwbfile)
with NWBHDF5IO(str(nwb_path), "r") as io:
    restored_file = io.read()
    restored_position = read_position(restored_file)
    restored_units = read_units(restored_file)
    env_back = read_environment(restored_file, name="analysis_environment")
    rates_back = read_place_field(restored_file, name="spatial_rates")

# All readers above were eager; comparison happens after the file is closed.
np.testing.assert_array_equal(restored_position.positions, pos.positions)
np.testing.assert_array_equal(restored_units.unit_ids, units.unit_ids)
np.testing.assert_allclose(rates_back.firing_rates, rates.firing_rates, equal_nan=True)
np.testing.assert_array_equal(rates_back.occupancy, rates.occupancy)
np.testing.assert_array_equal(rates_back.spatial_support, rates.spatial_support)
np.testing.assert_array_equal(rates_back.unit_ids, rates.unit_ids)
assert rates_back.analysis_settings == rates.analysis_settings
np.testing.assert_array_equal(env_back.bin_centers, env.bin_centers)
assert set(env_back.connectivity.edges) == set(env.connectivity.edges)
assert env_back.units == env.units
print(
    "Disk round trip preserved positions, labeled rates, occupancy, support and analysis settings."
)

# %% [markdown]
# Bin arrays and connectivity checks do not establish every geometric capability
# for every layout. Test point lookup and other operations you need after loading,
# especially for non-grid layouts. Here is an explicit outside-arena probe:

# %%
outside = np.array([[-100.0, -100.0]])
print(
    f"Outside point lookup: original={env.bin_at(outside).tolist()}, "
    f"reloaded={env_back.bin_at(outside).tolist()}"
)
# The temporary file is no longer needed: all returned arrays were eager.
temporary_data.cleanup()

# %% [markdown]
# ## Next steps
#
# - [19: real-data spatial coding](19_real_data_bandit_task.ipynb) uses the same array workflow.
# - [26: peri-event histograms](26_peri_event_psth.ipynb) demonstrates full-window event exposure.
# - NWB also supports `read_pose`, `read_head_direction`, `read_events`, `read_intervals`
#   and `read_trials`; install the corresponding extensions before opening those files.
# - [NWB](https://www.nwb.org/) and [DANDI](https://dandiarchive.org/) provide format/data resources.
