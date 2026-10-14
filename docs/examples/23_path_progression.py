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
# # Path Progression on a Track Graph (Bandit Task Real Data)
#
# This notebook demonstrates **well-to-well transition segmentation and path
# progression analysis** on a real hippocampal recording from the same
# bandit-task dataset used in `19_real_data_bandit_task.ipynb`.
#
# **Dataset**: J16 session, 2021-07-10 (triple-Y maze, 3 reward patches,
# 6 well endpoints, 24 minutes of position tracking at 500 Hz).
#
# **What you'll learn**
#
# - How to define endpoint regions from a track graph
# - How to infer endpoint-region visits from position tracking
# - How to compare progress between consecutive visits to different endpoints
# - How to retain observed arrival samples and respect recording gaps
# - How to summarize transitions separately for each ordered endpoint pair
#
# These are **position-defined transitions**, not recorded task trials. The
# archive does not provide reward-delivery events or task trial boundaries.
# The next visited endpoint defines the destination retrospectively; this
# analysis does not identify a cued goal, reward consumption, or intention.
#
# **Estimated time**: 25-35 minutes.
# **Prerequisites**: [05: track linearization](05_track_linearization.ipynb)
# and [14: behavioral segmentation](14_behavioral_segmentation.ipynb).
# Run from a checkout of this repository, which includes the data helper and
# example styling. From the repository root, launch
# `uv run --extra notebooks jupyter lab examples/` and open this notebook.
#
# **Pipeline**
#
# ```
# track graph + positions
#         |
#         v
#  Environment.from_graph (2D bins with a 1D track-coordinate mapping)
#         |
#         v
#  env.regions.buffer(...)  for each endpoint = well region
#         |
#         v
#  detect_region_crossings(direction="entry")  for each well
#         |
#         v
#  pair consecutive visits within one observed run -> transitions
#         |
#         v
#  path_progress(metric="geodesic")
# ```

# %%
import importlib.util
import itertools
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from neurospatial import Environment
from neurospatial.behavior import path_progress
from neurospatial.behavior.segmentation import detect_region_crossings


def _find_project_root(start: Path) -> Path:
    """Find the project root containing the real-data helper."""
    for path in (start, *start.parents):
        if (path / "data" / "load_bandit_data.py").exists():
            return path
    raise FileNotFoundError(f"Could not find data/load_bandit_data.py from {start}")


try:
    _start_path = Path(__file__).resolve().parent
except NameError:
    _start_path = Path.cwd().resolve()
_base_path = _find_project_root(_start_path)

_loader_spec = importlib.util.spec_from_file_location(
    "load_bandit_data", _base_path / "data" / "load_bandit_data.py"
)
if _loader_spec is None or _loader_spec.loader is None:
    raise ImportError("Could not load bandit data helper")
_loader_module = importlib.util.module_from_spec(_loader_spec)
_loader_spec.loader.exec_module(_loader_module)
load_neural_recording_from_files = _loader_module.load_neural_recording_from_files
fetch_bandit_data = _loader_module.fetch_bandit_data

# Shared styling (Okabe-Ito palette, consistent figure / font sizes)
import sys

_here = (
    str(Path(__file__).resolve().parent) if "__file__" in globals() else str(Path.cwd())
)
if _here not in sys.path:
    sys.path.insert(0, _here)
from _style import apply_style

apply_style(figsize=(12, 8), font_size=11)

# %% [markdown]
# ## 1. Load the bandit-task recording
#
# We only need the position trace and the track graph for this notebook;
# spike data is ignored.
# The first run downloads the same 75 MB, CC BY 4.0 archive as example 19:
# [Zenodo version 1.0](https://doi.org/10.5281/zenodo.23020757).
# Its SHA-256 is verified, and subsequent runs use the cached file.

# %%
data_file = fetch_bandit_data(_base_path / "data")
print(f"Verified dataset: {data_file.name}")
data = load_neural_recording_from_files(_base_path / "data", "j1620210710_02_r1")
position_info = data["position_info"]
track_graph = data["track_graph"]
linear_edge_order = data["linear_edge_order"]
linear_edge_spacing = data["linear_edge_spacing"]

times = position_info.index.values
positions = position_info[["head_position_x", "head_position_y"]].values

duration = times[-1] - times[0]
print(f"Recording: {len(times):,} samples, {duration / 60:.1f} min")
print(
    f"Track graph: {track_graph.number_of_nodes()} nodes, "
    f"{track_graph.number_of_edges()} edges"
)

# %% [markdown]
# ## 2. Build the environment from the track graph
#
# `Environment.from_graph` creates bins embedded in the original 2D track
# coordinates, plus a 1D linearization. `env.bin_at(positions)` assigns a
# finite 2D point to its nearest track-bin center. This assignment does not
# reject tracking points far from the track or resolve ambiguous branches
# using movement history. We use the archived, preprocessed tracking here;
# inspect tracking and endpoint placement before applying this recipe elsewhere.
# The 15 cm gaps in the linear coordinate are display spacing, not travel distance.

# %%
env = Environment.from_graph(
    graph=track_graph,
    edge_order=linear_edge_order,
    edge_spacing=linear_edge_spacing,
    bin_size=2.0,
    name="bandit_track",
)
env.units = "cm"

print(f"Environment: {env.n_bins} bins along the linearized track")
print(f"is_linearized_track: {env.is_linearized_track}")

# %% [markdown]
# ## 3. Define endpoint well-region proxies
#
# The maze has 6 track endpoints (nodes with degree 1). We use a circle of
# radius 8 cm around each endpoint as an **operational well-region proxy**.
# This radius is an analysis choice, not a measured well boundary: changing
# it changes entry times and visit counts. Confirm these proxies against
# experimental metadata or video when interpreting behavior.
#
# `detect_region_crossings` uses membership of track-bin centers in polygon
# regions; an entry is not an exact continuous crossing of the drawn circle.
# Point regions have no area for that rasterization. `path_progress` itself
# takes bin arrays and does not require regions or perform segmentation.

# %%
endpoint_nodes = sorted(n for n in track_graph.nodes if track_graph.degree(n) == 1)
well_radius_cm = 8.0
well_names: list[str] = []
for n in endpoint_nodes:
    name = f"well_{n}"
    env.regions.buffer(
        np.asarray(track_graph.nodes[n]["pos"], dtype=float),
        well_radius_cm,
        name,
    )
    well_names.append(name)

print(f"Defined {len(well_names)} wells (radius={well_radius_cm} cm):")
for w in well_names:
    bins = env.bins_in_region(w)
    print(f"  {w}: {len(bins)} track bins")

# Endpoint guide lines must use linearized coordinates, not the physical x
# component of env.bin_centers (which remains a 2D coordinate array).
well_linear_positions = {
    w: float(
        env.to_linear(np.asarray(track_graph.nodes[n]["pos"], dtype=float)[None, :])[0]
    )
    for n, w in zip(endpoint_nodes, well_names, strict=True)
}

# %% [markdown]
# Plot the maze with wells highlighted to confirm the regions land on the
# track endpoints.

# %%
fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)

# Track edges
for u, v in track_graph.edges():
    pu = np.array(track_graph.nodes[u]["pos"])
    pv = np.array(track_graph.nodes[v]["pos"])
    ax.plot([pu[0], pv[0]], [pu[1], pv[1]], "k-", lw=4, alpha=0.3, zorder=1)

# Trajectory (subsampled)
ax.plot(
    positions[::40, 0],
    positions[::40, 1],
    color="0.6",
    lw=0.3,
    alpha=0.4,
    zorder=2,
    label="trajectory",
)

# Well regions (drawn as circles at the endpoint node positions, matching
# the buffer radius used to build the region polygons).
for n, w in zip(endpoint_nodes, well_names, strict=True):
    cx, cy = track_graph.nodes[n]["pos"]
    ax.add_patch(
        Circle(
            (cx, cy),
            well_radius_cm,
            facecolor="tab:orange",
            edgecolor="black",
            lw=1.5,
            alpha=0.6,
            zorder=5,
        )
    )
    ax.annotate(
        w.replace("well_", "W"),
        (cx, cy),
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=6,
    )

ax.set_aspect("equal")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_title("Triple-Y bandit maze: trajectory and 6 well regions")
ax.legend(loc="upper right")
plt.show()

# %% [markdown]
# ## 4. Detect well-entry events
#
# `detect_region_crossings(..., direction="entry")` detects an observed
# outside-to-inside change in region membership. It returns the first inside
# sample, and does not invent an entry when a recording run begins inside a well.
#
# Split at invalid positions or timestamp gaps longer than 0.5 s before
# detecting or pairing entries. This explicit gap threshold defines what we
# treat as continuous observation; it does not recover missing behavior.
# Within each run, merge entries from every well chronologically.

# %%
position_bins = env.bin_at(positions)
MAX_GAP = 0.5  # seconds; matches the crossing detector's default
valid_sample = (position_bins >= 0) & np.isfinite(positions).all(axis=1)
run_starts = np.flatnonzero(
    valid_sample & np.r_[True, (~valid_sample[:-1]) | (np.diff(times) > MAX_GAP)]
)
run_stops = (
    np.flatnonzero(
        valid_sample & np.r_[(~valid_sample[1:]) | (np.diff(times) > MAX_GAP), True]
    )
    + 1
)
observed_runs = list(zip(run_starts, run_stops, strict=True))

visits = []
for run_id, (start, stop) in enumerate(observed_runs):
    for w in well_names:
        for crossing in detect_region_crossings(
            position_bins[start:stop],
            times[start:stop],
            env,
            region_name=w,
            direction="entry",
            max_gap=MAX_GAP,
        ):
            index = int(np.searchsorted(times, crossing.time))
            visits.append(
                {
                    "index": index,
                    "well": w,
                    "bin": int(crossing.bin_index),
                    "run_id": run_id,
                }
            )
visits.sort(key=lambda visit: visit["index"])

print(f"Observed runs: {len(observed_runs)} (gap threshold {MAX_GAP:g} s)")
print(f"Total well-entry events: {len(visits)}")
print("Visits per well:")
for w, n in Counter(visit["well"] for visit in visits).most_common():
    print(f"  {w}: {n}")

# %% [markdown]
# ## 5. Pair consecutive visits into entry-to-entry transitions
#
# A transition spans an observed entry into A through the next observed
# entry into a different endpoint B, **within the same observed run**.
# Its reference bins are the actual observed entry bins at both ends.
#
# Adjacent entries to the same endpoint are omitted rather than merged.
# For A, A, B, only the interval from the second A entry to B is retained.
# We also omit intervals crossing observation breaks; their destination
# cannot be inferred from the tracking we observed.
#
# This convention includes any dwell and excursions after the retained A
# entry. It measures entry-to-entry elapsed time, not departure-to-arrival
# travel time. A plateau near zero indicates little displacement from the
# start bin; tracking alone does not establish why the animal stayed there.

# %%
transitions = []
omitted_same_well = 0
omitted_run_break = 0
for first, second in itertools.pairwise(visits):
    if first["run_id"] != second["run_id"]:
        omitted_run_break += 1
        continue
    if first["well"] == second["well"]:
        omitted_same_well += 1
        continue
    transitions.append(
        {
            "start_index": first["index"],
            "stop_index": second["index"],
            "t0": times[first["index"]],
            "t1": times[second["index"]],
            "start": first["well"],
            "goal": second["well"],
            "start_bin": first["bin"],
            "goal_bin": second["bin"],
        }
    )

if not transitions:
    raise ValueError(
        "No between-endpoint transitions found; inspect tracking and well regions."
    )
durations = np.array([tr["t1"] - tr["t0"] for tr in transitions])
pair_counts = Counter((tr["start"], tr["goal"]) for tr in transitions)
print(f"Retained between-endpoint transitions: {len(transitions)}")
print(
    f"Omitted adjacent pairs: {omitted_same_well} same endpoint; "
    f"{omitted_run_break} across observation breaks"
)
print(
    f"Duration (s): min={durations.min():.2f}  "
    f"median={np.median(durations):.2f}  max={durations.max():.2f}"
)

# Transition matrix (first endpoint -> next visited endpoint)
labels = well_names
idx = {w: i for i, w in enumerate(labels)}
M = np.zeros((len(labels), len(labels)), dtype=int)
for tr in transitions:
    M[idx[tr["start"]], idx[tr["goal"]]] += 1

fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
im = ax.imshow(M, cmap="viridis")
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels([w.replace("well_", "W") for w in labels])
ax.set_yticklabels([w.replace("well_", "W") for w in labels])
ax.set_xlabel("next visited endpoint")
ax.set_ylabel("start well")
ax.set_title("Observed transition counts (first -> next endpoint)")
for i in range(len(labels)):
    for j in range(len(labels)):
        if M[i, j]:
            ax.text(
                j,
                i,
                M[i, j],
                ha="center",
                va="center",
                color="white" if M[i, j] < M.max() / 2 else "black",
                fontsize=10,
            )
plt.colorbar(im, ax=ax, label="n transitions")
plt.show()

# %% [markdown]
# ## 6. Compute path progression with observed endpoints
#
# `path_progress` computes
#
# `clip(d(start_bin, current_bin) / d(start_bin, destination_bin), 0, 1)`.
#
# Geodesic distance follows the track's connectivity rather than the artificial
# gaps in its linearized coordinate. On the shortest route from start to
# destination, this ratio describes progress along that route. On a branched
# maze, movement down another branch can also increase it, even to 1 after
# clipping. **A value of 1 alone does not establish arrival or efficient travel.**
# It is a normalized displacement from the start, not cumulative distance traveled
# or distance remaining to the destination. A decrease means the animal is closer
# to the start in graph distance, not necessarily that it retraced its exact path.
#
# Keep both entry samples for each transition, so its observed endpoint values
# are 0 and 1 by construction. Adjacent transitions share an entry timestamp:
# its value is 1 for the ending transition and 0 for the next one. Concatenating
# the individual sample lists lets one vectorized call preserve both values.
# For a single session-wide array, assign half-open intervals `[t0, t1)`; use
# the complete individual curves for endpoint plots and time normalization.

# %%
lengths = np.array([tr["stop_index"] - tr["start_index"] + 1 for tr in transitions])
sample_indices = np.concatenate(
    [np.arange(tr["start_index"], tr["stop_index"] + 1) for tr in transitions]
)
start_bins = np.repeat([tr["start_bin"] for tr in transitions], lengths)
goal_bins = np.repeat([tr["goal_bin"] for tr in transitions], lengths)
all_progress = path_progress(
    position_bins[sample_indices],
    env,
    start_bins=start_bins,
    goal_bins=goal_bins,
    metric="geodesic",
)
progress = np.full(len(times), np.nan)
offset = 0
for tr, length in zip(transitions, lengths, strict=True):
    tr["indices"] = sample_indices[offset : offset + length]
    tr["progress"] = all_progress[offset : offset + length]
    assert np.isclose(tr["progress"][0], 0.0)
    assert np.isclose(tr["progress"][-1], 1.0)
    # The ending timestamp may belong to the next transition in this array.
    progress[tr["start_index"] : tr["stop_index"]] = tr["progress"][:-1]
    offset += length

print(f"progress range: [{np.nanmin(all_progress):.3f}, {np.nanmax(all_progress):.3f}]")
print(
    f"Session samples in retained half-open intervals: "
    f"{np.isfinite(progress).sum():,} / {len(progress):,}"
)

# %% [markdown]
# ## 7. Linear position and path progression over the recording
#
# The two rows share seconds relative to the first tracking sample. The top
# row shows the archive's linear-position coordinate; abrupt changes can reflect
# its arrangement of track segments, not a physical jump or extra travel distance.
# Orange guide lines mark endpoint coordinates in that same linearization.
#
# Plot each progress curve separately so the reset at the next entry is not
# drawn as animal movement. Line breaks also preserve observation gaps and
# intervals omitted from the transition analysis. Subsampling is only for display.

# %%
linear_pos = position_info["linear_position"].to_numpy()
t_rel = times - times[0]
SUB = 17  # about 30 Hz for this 500 Hz recording


def plot_recording_window(axes, start, stop, *, step):
    """Draw only observed runs/retained transitions on a shared relative clock."""
    for first, last in observed_runs:
        indices = np.arange(first, last, step)
        selected = (t_rel[indices] >= start) & (t_rel[indices] <= stop)
        axes[0].plot(
            t_rel[indices][selected],
            linear_pos[indices][selected],
            color="steelblue",
            lw=0.6,
        )
    for well, coordinate in well_linear_positions.items():
        axes[0].axhline(coordinate, color="tab:orange", lw=0.6, alpha=0.5)
        axes[0].text(
            stop,
            coordinate,
            well.replace("well_", "W"),
            fontsize=8,
            va="center",
            ha="left",
            color="tab:orange",
        )
    for tr in transitions:
        # Keep the observed final entry even after display subsampling.
        local = np.unique(
            np.r_[np.arange(0, len(tr["indices"]), step), len(tr["indices"]) - 1]
        )
        indices = tr["indices"][local]
        selected = (t_rel[indices] >= start) & (t_rel[indices] <= stop)
        axes[1].plot(
            t_rel[indices][selected],
            tr["progress"][local][selected],
            color="tab:blue",
            lw=0.6,
        )
    axes[0].set_ylabel("linear position (cm)")
    axes[1].set_xlabel("time since first tracking sample (s)")
    axes[1].set_ylabel("path progress")
    axes[1].axhline(1.0, color="tab:gray", lw=0.5, ls="--")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlim(start, stop)


fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
plot_recording_window(axes, 0.0, float(t_rel[-1]), step=SUB)
axes[0].set_title("Linear position: endpoint guides use linearized coordinates")
axes[1].set_title("Entry-to-entry progress (separate curve for each transition)")
plt.show()

# %% [markdown]
# Zoom into the first three minutes with a retained transition. The separate
# curves show resets explicitly; the metric need not rise monotonically.

# %%
zoom_start = float(transitions[0]["t0"] - times[0])
zoom_end = min(zoom_start + 180.0, float(t_rel[-1]))
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, constrained_layout=True)
plot_recording_window(axes, zoom_start, zoom_end, step=1)
axes[0].set_title(f"Recording window: {zoom_start:.0f}-{zoom_end:.0f} s")
for tr in transitions:
    t0, t1 = tr["t0"] - times[0], tr["t1"] - times[0]
    if t0 < zoom_end and t1 > zoom_start:
        axes[1].axvspan(
            max(t0, zoom_start), min(t1, zoom_end), color="tab:blue", alpha=0.07
        )
plt.show()

# %% [markdown]
# ## 8. Individual transition curves
#
# Show the first nine retained transitions, including their observed arrivals.
# A plateau near zero indicates little graph displacement from the entry bin.
# It does not identify reward consumption, a pause in a recorded task trial,
# or the animal's intended destination.

# %%
N_SHOW = 9
fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey=True, constrained_layout=True)
for ax, tr in zip(axes.flat, transitions[:N_SHOW], strict=False):
    elapsed = times[tr["indices"]] - tr["t0"]
    ax.plot(elapsed, tr["progress"], color="tab:blue", lw=1.5)
    ax.scatter([elapsed[0], elapsed[-1]], [0, 1], color="tab:blue", s=20, zorder=3)
    ax.axhline(1.0, color="tab:gray", lw=0.5, ls="--")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{tr['start']} -> {tr['goal']} ({elapsed[-1]:.1f} s)", fontsize=10)
    ax.set_xlabel("time since first entry (s)")
    ax.set_ylabel("path progress")
for ax in axes.flat[min(N_SHOW, len(transitions)) :]:
    ax.set_visible(False)
fig.suptitle("First retained entry-to-entry transitions", fontsize=13)
plt.show()

# %% [markdown]
# ## 9. Duration and progress summaries by endpoint pair
#
# Entry-to-entry durations include time spent near the first endpoint. Report
# them for the retained cohort, without calling them pure travel times.
# For progress summaries, group by the **ordered** start/destination pair:
# routes can differ in distance, geometry and behavioral context.
#
# Resample each fully observed curve onto normalized elapsed time `[0, 1]`.
# Both endpoints are observed; `np.interp` does not extrapolate a missing arrival.
# Pairs spanning invalid samples or gaps were excluded before this step.
# Normalization gives each transition equal weight and removes absolute duration.
# The plots show the six most frequent pairs by count. Other pairs remain in
# the duration distribution. Bands are the between-transition interquartile
# range, **not confidence intervals**; these are repeated observations from one
# animal in one recording. No particular curve shape is assumed in advance.

# %%
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
ax.hist(durations, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
ax.axvline(
    np.median(durations),
    color="red",
    ls="--",
    label=f"median = {np.median(durations):.1f} s",
)
ax.set_xlabel("entry-to-entry duration (s)")
ax.set_ylabel("n transitions")
ax.set_title(f"Retained between-endpoint intervals (n={len(transitions)})")
ax.legend()
plt.show()

N_GRID = 100
t_norm = np.linspace(0.0, 1.0, N_GRID)
per_transition = np.full((len(transitions), N_GRID), np.nan)
for i, tr in enumerate(transitions):
    normalized_time = (times[tr["indices"]] - tr["t0"]) / (tr["t1"] - tr["t0"])
    # No filtering then interpolation across holes: a curve must be complete.
    if np.isfinite(tr["progress"]).all():
        per_transition[i] = np.interp(t_norm, normalized_time, tr["progress"])

most_frequent_pairs = pair_counts.most_common(6)
fig, axes = plt.subplots(
    2, 3, figsize=(13, 7), sharex=True, sharey=True, constrained_layout=True
)
for ax, (pair, count) in zip(axes.flat, most_frequent_pairs, strict=False):
    selected = np.array([(tr["start"], tr["goal"]) == pair for tr in transitions])
    curves = per_transition[selected]
    curves = curves[np.isfinite(curves).all(axis=1)]
    if not len(curves):
        ax.set_title(f"{pair[0]} -> {pair[1]}: no complete curves")
        continue
    lo, hi = np.percentile(curves, [25, 75], axis=0)
    ax.fill_between(t_norm, lo, hi, color="tab:blue", alpha=0.2, label="IQR")
    ax.plot(t_norm, curves.mean(axis=0), color="tab:blue", lw=2, label="mean")
    ax.set_title(f"{pair[0]} -> {pair[1]} (n={len(curves)}/{count})", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("normalized entry-to-entry time")
    ax.set_ylabel("path progress")
for ax in axes.flat[len(most_frequent_pairs) :]:
    ax.set_visible(False)
axes.flat[0].legend(loc="lower right")
fig.suptitle("Six most frequent ordered endpoint pairs; n=complete/retained")
plt.show()

# %% [markdown]
# ## 10. Spatial view of one transition
#
# Choose the transition whose duration is nearest the median **within the most
# frequent ordered endpoint pair**. This is an explicit display choice, not a
# classification of a clean or representative behavioral strategy.
#
# Color the 2D trajectory by progress and mark the two observed entry samples.
# Their values are 0 and 1 because those samples define the reference bins.
# Interpret intermediate values with the branch/clipping limitation in section 6.

# %%
demo_pair = most_frequent_pairs[0][0]
pair_transitions = [tr for tr in transitions if (tr["start"], tr["goal"]) == demo_pair]
pair_median = np.median([tr["t1"] - tr["t0"] for tr in pair_transitions])
demo_transition = min(
    pair_transitions, key=lambda tr: abs((tr["t1"] - tr["t0"]) - pair_median)
)
xy = positions[demo_transition["indices"]]
p = demo_transition["progress"]

fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
for u, v in track_graph.edges():
    pu = np.asarray(track_graph.nodes[u]["pos"])
    pv = np.asarray(track_graph.nodes[v]["pos"])
    ax.plot([pu[0], pv[0]], [pu[1], pv[1]], "k-", lw=4, alpha=0.3, zorder=1)
for n in endpoint_nodes:
    ax.add_patch(
        Circle(
            track_graph.nodes[n]["pos"],
            well_radius_cm,
            facecolor="tab:orange",
            edgecolor="black",
            alpha=0.25,
            zorder=2,
        )
    )
sc = ax.scatter(xy[:, 0], xy[:, 1], c=p, cmap="viridis", vmin=0, vmax=1, s=8, zorder=5)
ax.scatter(
    *xy[0],
    color="white",
    edgecolor="black",
    s=140,
    marker="o",
    zorder=10,
    label=f"first entry ({demo_transition['start']})",
)
ax.scatter(
    *xy[-1],
    color="black",
    edgecolor="white",
    s=140,
    marker="s",
    zorder=10,
    label=f"next entry ({demo_transition['goal']})",
)
ax.set_aspect("equal")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_title(
    f"{demo_transition['start']} -> {demo_transition['goal']} "
    f"({demo_transition['t1'] - demo_transition['t0']:.1f} s)"
)
plt.colorbar(sc, ax=ax, label="path progress")
ax.legend(loc="upper right")
plt.show()

# %% [markdown]
# ## Summary
#
# 1. Build an environment from the track graph, retaining its 2D bin coordinates.
# 2. Define endpoint-region proxies with an explicit radius and inspect their placement.
# 3. Detect entries and pair consecutive different endpoints within observed runs.
# 4. Compute `path_progress` with the observed entry bins as references, preserving
#    both endpoints for each individual curve.
# 5. Plot resets separately and summarize complete curves by ordered endpoint pair.
#
# The result describes **outcome-defined transitions in position tracking**.
# It does not establish task trial boundaries, delivered rewards, goals known to
# the animal, or efficient navigation. Those questions require corresponding
# experimental events and additional measures, such as path efficiency or
# distance remaining to the destination.
#
# ### See also
#
# - [19: spatial coding](19_real_data_bandit_task.ipynb) — rate maps on this dataset
# - [14: behavioral segmentation](14_behavioral_segmentation.ipynb) — explicit task regions
# - `path_progress` and `compute_path_efficiency` in `neurospatial.behavior`
