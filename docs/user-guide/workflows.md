# Complete Workflows

This page demonstrates end-to-end analysis workflows that integrate multiple neurospatial features.

## Place fields on outbound and inbound runs

Does a place cell fire at the same place when the animal runs out along a
linear track as when it runs back? To answer this, you build one rate map from
the outbound runs and one from the inbound runs, then compare them cell by
cell. The comparison means something only when the two maps describe the same
cells, the same places, and time when the animal was actually tracked. This
section goes from per-frame direction labels to rate maps for each direction,
a cell-by-cell comparison, and a decoder tested on laps it never saw.

### The recording

The example simulates a recording so that the right answers are known; the
last subsection shows what to replace with your own data. The animal runs 20
laps on a 100 cm track, 5 s out and 5 s back, tracked at 30 Hz. The tracker
loses the animal for 1 s at t = 100 s. Six cells have Gaussian place fields
(6 cm wide, 20 Hz peak, 0.5 Hz baseline):

- `c1` fires at 30 cm in both directions.
- `c2` fires at 70 cm on outbound runs and at 40 cm on inbound runs.
- `c3` fires at 50 cm on outbound runs and is silent on inbound runs.
- `c4`, `c5` and `c6` fire at 15, 55 and 85 cm in both directions.

The block builds the recording, turns the direction labels into time windows
with `labels_to_epochs`, and estimates one rate map per direction. The later
blocks in this section continue from it.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment
from neurospatial.behavior import labels_to_epochs
from neurospatial.encoding import compute_spatial_rates
from neurospatial.simulation import generate_poisson_spikes

# 200 s at 30 Hz: 20 laps, each 5 s out (5 -> 95 cm) and 5 s back.
rng = np.random.default_rng(1)
dt = 1 / 30
times = np.arange(0, 200, dt)
phase = (times % 10) / 10
x_true = 5 + 90 * (1 - np.abs(2 * phase - 1)) + rng.normal(0, 0.3, times.size)
positions = x_true[:, None].copy()
positions[3000:3030] = np.nan  # the tracker lost the animal for 1 s at t = 100 s
outbound = phase < 0.5
direction = np.where(outbound, "outbound", "inbound")  # one label per frame


def place_field(center):
    """Firing rate (Hz) of a 6 cm wide, 20 Hz field at each true position."""
    return 0.5 + 20 * np.exp(-0.5 * ((x_true - center) / 6) ** 2)


unit_ids = ["c1", "c2", "c3", "c4", "c5", "c6"]
true_rates = [
    place_field(30),
    np.where(outbound, place_field(70), place_field(40)),
    np.where(outbound, place_field(50), 0.0),
    place_field(15),
    place_field(55),
    place_field(85),
]
# Spikes come from the true position, so they continue through the dropout.
# Seed the spikes differently from the trajectory.
spike_times = [
    generate_poisson_spikes(rate, times, seed=10 + i)
    for i, rate in enumerate(true_rates)
]
env = Environment.from_samples(
    np.linspace(0, 100, 201)[:, None], bin_size=5.0, units="cm"
)

# Direction labels -> half-open [start, stop) windows for each direction,
# using only frames where the animal was tracked.
valid = np.isfinite(positions).all(axis=1)
conditions = labels_to_epochs(times, direction, valid=valid)

# One rate map per direction: the original arrays, restricted by epochs=.
maps = {
    label: compute_spatial_rates(
        env, spike_times, times, positions,
        unit_ids=unit_ids, epochs=conditions[label],
        bandwidth=5.0, min_occupancy=0.5,
    )
    for label in conditions
}

print({label: round(seconds, 2) for label, seconds in conditions.durations.items()})
print({label: len(conditions[label]) for label in conditions})
table = conditions.epochs_table()
print(table.head())
shortened = np.isclose(table["start"], 101.0)  # the run the dropout shortened
print(table.loc[shortened, ["label", "start", "stop", "duration"]])
assert abs(conditions.durations["outbound"] - 99.0) < 1e-6
assert abs(conditions.durations["inbound"] - 99.93) < 0.01
assert len(conditions["outbound"]) == len(conditions["inbound"]) == 20
assert [101.0, 105.0] in np.round(conditions["outbound"], 6).tolist()
```

`labels_to_epochs` turns one label per frame into time windows for each
label. The time between two frames, `[times[k], times[k + 1])`, gets the label
of the frame at its start. An untracked frame (`valid` False) is skipped as if
its row were deleted, so a time between frames is kept when the tracked
frames around it are no more than `max_gap` (1.0 s by default) apart, and carries the
label of the tracked frame that starts it. Kept stretches with the same label
join into one epoch; a tracking loss longer than `max_gap` or a change of
label ends it, so the same label after a long dropout starts a new epoch. The
table lists each epoch with its label, start, stop and duration.

Here the dropout starts exactly at the beginning of an outbound run
(t = 100 s), so it shortens that run to 101–105 s instead of splitting it;
the last printed row shows it. Each direction has 20 epochs. Outbound runs
total 99.0 s instead of 100 s. Inbound runs total 99.93 s: they lose the
1/30 s from the last tracked frame into the dropout, and the last 1/30 s of
the recording, because the final frame starts no interval. A dropout in the
middle of a run would split that run into two epochs.

The rate maps use the original `times`, `positions` and `spike_times` with
`epochs=conditions[label]`. `compute_spatial_rates` then counts spikes and
dwell time over exactly the same windows: a spike counts only if it falls in a
window where the animal was tracked in that direction. Slicing the arrays to
outbound frames yourself would leave the spikes to be filtered separately,
and would put the last outbound frame of one lap next to the first outbound
frame of the next, so whether the 5 s between them counts as dwell time would
depend on `max_gap`, not on the labels. Use the same environment, estimator
settings and `max_gap` for both directions; only the windows change.

### Check the behavior first

A difference between two maps can come from the cells or from how the animal
behaved: less time in a condition, places it did not visit, or a different
running speed. Check these before comparing the maps.

<!-- docs-test: run setup=workflows_outbound_inbound -->
```python
from neurospatial.behavior import compute_speed, in_epochs

speed = compute_speed(times, positions)  # cm/s; NaN where tracking was lost
coverage = {}
for label in conditions:
    inside = in_epochs(times, conditions[label])
    coverage[label] = {
        "duration_s": round(conditions.durations[label], 2),
        "occupied_bins": int(maps[label].spatial_support.sum()),
        "mean_speed_cm_s": round(float(np.nanmean(speed[inside])), 1),
    }
print(coverage)
print("bins in the environment:", env.n_bins)
assert coverage["outbound"]["occupied_bins"] == coverage["inbound"]["occupied_bins"] == 19
assert abs(coverage["outbound"]["mean_speed_cm_s"] - 17.4) < 0.2
assert abs(coverage["inbound"]["mean_speed_cm_s"] - 17.4) < 0.2
```

Both directions have about 99 s of tracked time, the same 19 of 21 bins
occupied for at least 0.5 s (the two end bins lie beyond the 5–95 cm the
animal ran), and the same mean speed of 17.4 cm/s. Sampling does not explain
the differences below. In a real recording, a direction with much less time,
fewer visited bins or a different speed needs that caveat in the
interpretation.

### Compare the maps cell by cell

`compare_spatial_rates` pairs cells by `unit_id`, never by row order, and
correlates each cell's two maps over the bins occupied for at least
`min_occupancy` in both directions where both rates are finite. It refuses
maps estimated on different environments.

<!-- docs-test: run setup=workflows_outbound_inbound -->
```python
import matplotlib.pyplot as plt
import pandas as pd
from neurospatial.encoding import compare_spatial_rates

comparison = compare_spatial_rates(maps["outbound"], maps["inbound"])
table = comparison.summary_table()
with pd.option_context("display.width", 120, "display.precision", 3):
    print(table[["correlation", "n_common_bins", "mean_rate_a", "mean_rate_b", "status"]])
assert (table["n_common_bins"] == 19).all()
assert abs(table.loc["c1", "correlation"] - 0.996) < 0.005
assert abs(table.loc["c2", "correlation"] - (-0.383)) < 0.01
assert np.isnan(table.loc["c3", "correlation"])
assert table.loc["c3", "status"] == "constant_rates_b"
assert table.loc["c3", "mean_rate_b"] == 0.0
assert (table.loc[["c4", "c5", "c6"], "correlation"] > 0.95).all()

c2_peaks = {label: maps[label].sel(unit_id="c2").peak_location() for label in maps}
print("c2 peak (cm):", {label: float(peak[0]) for label, peak in c2_peaks.items()})
assert c2_peaks["outbound"][0] == 70.0 and c2_peaks["inbound"][0] == 40.0

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, label in zip(axes[:2], ["outbound", "inbound"]):
    maps[label].sel(unit_id="c2").plot(ax=ax)
    ax.set_title(f"c2, {label} runs")
comparison.plot(unit_id="c2", ax=axes[2])
axes[2].set_title("c2, inbound minus outbound")
plt.tight_layout()
plt.show()
```

Condition A is the first argument (outbound) and B the second (inbound);
`mean_rate_a` and `mean_rate_b` are each cell's mean rate in Hz over the
time that direction spent in the common bins (rates weighted by occupancy).
Read the table one cell at a time:

- `c1` (r = 0.996) fires at the same place in both directions. `c4`, `c5`
  and `c6` are the same.
- `c2` (r = −0.383) has a field in both directions, but in a different place:
  its field moved with running direction. Its peak is at 70 cm on outbound
  runs and at 40 cm on inbound runs, as the figure shows; the difference map is
  positive where the inbound field is and negative where the outbound field
  was.
- `c3` has no correlation (NaN) with status `constant_rates_b`: it was silent
  on inbound runs, so its inbound map is flat and a correlation is undefined.
  NaN here means "cannot be measured", not "no similarity". Its mean inbound
  rate of 0 Hz says what changed.

All 19 occupied bins entered every cell's score. These are descriptive
numbers, not significance tests. To decide whether a correlation is lower
than chance, you need a null distribution, for example from shuffling
whole runs between directions, and a comparison of two halves of the same
direction to know how stable maps are without any change.

### Decode held-out laps

A decoder trained on outbound runs should place the animal correctly on other
outbound runs. On inbound runs it should do worse if some cells change their
fields with direction. Number the laps from 0, train on the even-numbered
laps and test on the odd-numbered laps, so that training and test data never come from the same lap.
`labels_to_epochs` with `epochs=` splits each direction's windows into the
two folds.

<!-- docs-test: run setup=workflows_outbound_inbound -->
```python
from neurospatial.decoding import BayesianDecoder

lap = (times // 10).astype(int)
fold = np.where(lap % 2 == 0, "train", "test")
out_folds = labels_to_epochs(times, fold, valid=valid, epochs=conditions["outbound"])
in_folds = labels_to_epochs(times, fold, valid=valid, epochs=conditions["inbound"])

decoder = BayesianDecoder(env, dt=0.25, bandwidth=5.0).fit(
    spike_times, times, positions, unit_ids=unit_ids, epochs=out_folds["train"],
)
within = decoder.predict(spike_times, times, epochs=out_folds["test"])
transfer = decoder.predict(spike_times, times, epochs=in_folds["test"])
within_error = within.error_against(times, positions)  # cm, one per time bin
transfer_error = transfer.error_against(times, positions)
assert np.isfinite(within_error).all() and np.isfinite(transfer_error).all()
for name, error in [("Held-out outbound laps", within_error),
                    ("Inbound laps (transfer)", transfer_error)]:
    print(f"{name}: median error {np.median(error):.1f} cm, "
          f"errors over 10 cm in {100 * np.mean(error > 10):.1f}% of time bins")
print("Training coverage:", int(decoder.spatial_support.sum()), "of", env.n_bins, "bins")
assert within.times.min() >= out_folds["test"][0, 0]
assert abs(np.median(within_error) - 5.0) < 0.5
assert abs(np.median(transfer_error) - 5.0) < 0.5
assert abs(100 * np.mean(within_error > 10) - 4.5) < 1.0
assert abs(100 * np.mean(transfer_error > 10) - 37.4) < 2.0
assert int(decoder.spatial_support.sum()) == 19
```

On held-out outbound laps the median error is 5.0 cm, one bin, and errors
larger than 10 cm (two bins) occur in 4.5% of time bins. On inbound laps the median
is also 5.0 cm, because the four cells that keep their fields still place the
animal correctly most of the time. The large errors tell the difference: they
occur in 37.4% of inbound time bins. Where `c2` and `c3` fire differently on
the way back, the outbound model is the wrong model for them. A median alone
would hide this, so report the spread of errors, not only their center.

The held-out outbound error says how well the decoder generalizes within a
condition; the inbound error says how much the condition changes the code.
Report error only on held-out laps; error on the training laps is
optimistic. The decoder learned 19 of 21 bins and can only place the animal
there.

Whole laps keep training and test data apart; splitting frame by frame would
put neighboring, nearly identical frames in both. The direction labels here
come from the animal's movement at that moment, so the same labels would be
available during the recording. Labels chosen by what happens later, such as
which arm the animal will enter next, describe the data well but use future
behavior.

### A comparison neurospatial refuses

Two maps with the same number of bins are not necessarily maps of the same
places. Here the second map is estimated on the same bins, but with the
environment's units declared as meters instead of centimeters:

<!-- docs-test: run -->
```python
from neurospatial import IncompatibleEnvironmentError
from neurospatial.encoding import compare_spatial_rates, compute_spatial_rates
from neurospatial.simulation import linear_track_session

session = linear_track_session(duration=30, n_place_cells=3, seed=3)
env_in_meters = session.env.with_metadata(units="m")
rates_cm = compute_spatial_rates(
    session.env, session.spike_times, session.times, session.positions,
    unit_ids=session.unit_ids,
)
rates_m = compute_spatial_rates(
    env_in_meters, session.spike_times, session.times, session.positions,
    unit_ids=session.unit_ids,
)
try:
    compare_spatial_rates(rates_cm, rates_m)
except IncompatibleEnvironmentError as error:
    message = str(error)
else:
    raise AssertionError("compare_spatial_rates accepted maps in different units")
print(message)
assert "differ in units or coordinate frames ('cm', None vs 'm', None)" in message
assert "a bin_mapping cannot pair bins in different units" in message
```

The error names what differs, the units of each environment, and how to fix
it: a `bin_mapping` pairs bins by place, so it cannot pair bins measured in
different units.

Units, coordinate frame, bin positions and sizes, and the connections between
bins must all match. Estimate every condition on one environment. If the maps
already exist on different bins, re-estimate them from the spikes and tracking
on one environment: a rate cannot be averaged onto coarser bins, because the
merged rate depends on how long the animal spent in each fine bin.

<!-- docs-test: run -->
```python
import numpy as np

# Two fine bins: 0 Hz over 100 s of occupancy and 100 Hz over 1 s.
spikes = 0.0 * 100 + 100.0 * 1  # 100 spikes
merged_rate = spikes / (100 + 1)  # one coarse bin: spikes / time
mean_of_rates = np.mean([0.0, 100.0])
print(f"From counts: {merged_rate:.2f} Hz; mean of the two rates: {mean_of_rates:.0f} Hz")
assert abs(merged_rate - 0.99) < 0.005 and mean_of_rates == 50.0
```

The coarse bin's rate is 0.99 Hz; averaging the two rates would report 50 Hz.
If only a door or barrier changed between conditions and the bins are the
same, pass `topology="may_differ"`; the result records that choice in
`analysis_settings`.

### Use your own recording

<!-- docs-test: skip uses the reader's own arrays -->
```python
conditions = labels_to_epochs(times, labels, valid=valid)
maps = {
    label: compute_spatial_rates(
        env, spike_times, times, positions, unit_ids=unit_ids,
        epochs=conditions[label], bandwidth=5.0, min_occupancy=0.5,
    )
    for label in conditions
}
comparison = compare_spatial_rates(maps["outbound"], maps["inbound"])
comparison.summary_table()
```

- Replace `times` (seconds), `positions` (`(n_frames, n_dims)`, with NaN
  where tracking was lost), `spike_times` (one array of spike times in
  seconds per unit) and `unit_ids` with your arrays. Take unit IDs from spike
  sorting; they must name the same cells in every condition.
- Labels come from your task: a trial table converted to one label per frame,
  `laps_to_direction_labels` on detected laps,
  `goal_pair_direction_labels` on segmented trials, or door-state intervals.
  Samples labeled `None` or `NaN` are left out; the direction-label
  functions give `None` to frames outside any lap or trial; pass
  `missing_label=` if your own labels use another value.
- Build `valid` from your tracking quality, for example
  `np.isfinite(positions).all(axis=1)` plus your likelihood threshold.
- When you join several recordings on one clock, pass `break_before`, True at
  the first frame of each later recording, to `labels_to_epochs` so no epoch
  spans the join, and to `compute_spatial_rates` so the interval across the
  join adds no occupancy or spikes.
- Use the same `max_gap`, environment and estimator settings for every
  condition. `compare_spatial_rates` warns when the estimator settings differ.
- Pass `spike_window=` to `compute_spatial_rates` when spike sorting and
  tracking cover different times.
- For a door or barrier change on the same bins, pass
  `topology="may_differ"` to `compare_spatial_rates`.

## Workflow 1: Place Field Analysis

A complete workflow for analyzing spatial firing patterns of neurons during navigation.

### Overview

**Goal**: Compute spatial firing rate maps from position tracking and spike data

**Steps**: Simulate trajectory → Create environment → Generate spikes → Compute place field → Visualize

### Complete Example

<!-- docs-test: run -->
```python
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rate
from neurospatial.simulation import (
    PlaceCellModel,
    generate_poisson_spikes,
    simulate_trajectory_ou,
)

# Step 1: Simulate an open-field session in a 100x100 cm arena.
# We seed the trajectory simulator with a polygon environment so the animal
# explores the full arena from the start (avoids a degenerate seed from
# sparse random points). Duration=300 s gives ~90% arena coverage at 15 cm/s
# while keeping CI runtime well under 60 s.
env_seed = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0)
env_seed = env_seed.with_metadata(units="cm")  # Required by simulate_trajectory_ou
positions, times = simulate_trajectory_ou(
    env_seed,
    duration=300.0,   # 5-minute session — full arena coverage
    dt=1 / 30.0,      # 30 Hz tracking
    speed_mean=15.0,  # cm/s
    seed=0,
    speed_units="cm",
)

# Step 2: Create a finer environment from the recorded trajectory
# (2.5 cm bins → ~1 600 active bins for a 100×100 cm open field)
env = Environment.from_samples(
    positions,
    bin_size=2.5,             # 2.5 cm bins for a 100×100 cm arena
    bin_count_threshold=5,
    dilate=True,
    fill_holes=True,
    name="OpenFieldSession1",
)
env = env.with_metadata(units="cm")

print(f"Created environment with {env.n_bins} active bins")
print(f"Spatial extent: {env.dimension_ranges}")
assert env.bin_at([50.0, 50.0]) != -1, "Place cell center must be inside env!"

# Step 3: Generate spike train for a simulated place cell
# (In real experiments, load your spike timestamps here.)
cell = PlaceCellModel(env, center=np.array([50.0, 50.0]), width=12.0, max_rate=20.0)
rates = cell.firing_rate(positions, times)
spike_times = generate_poisson_spikes(rates, times, seed=1)

print(f"Total spikes: {len(spike_times)}")

# Step 4: Compute the place field with the canonical one-liner
result = compute_spatial_rate(
    env,
    spike_times,
    times,
    positions,
    method="diffusion_kde",  # boundary-aware graph-based KDE
    bandwidth=5.0,                     # smoothing bandwidth in cm
    # min_occupancy thresholds raw time in each bin (seconds), for every
    # ratio estimator. The default 0.0 disables this additional mask.
    # bin_count_threshold above counts samples when constructing geometry;
    # it does not replace an occupancy threshold on the analyzed recording.
)
firing_rate = result.firing_rate

print(f"Peak firing rate: {np.nanmax(firing_rate):.2f} Hz")
print(f"Mean firing rate: {np.nanmean(firing_rate):.2f} Hz")

# Step 5: Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Trajectory overlaid on environment layout
ax1 = axes[0]
env.plot(ax=ax1)
ax1.plot(positions[:, 0], positions[:, 1], "r-", alpha=0.3, linewidth=0.5)
ax1.set_title("Trajectory")

# Plot 2: Occupancy map (seconds per bin)
ax2 = axes[1]
env.plot_field(result.occupancy, ax=ax2, cmap="viridis")
ax2.set_title("Occupancy (s)")

# Plot 3: Place field (smoothed firing rate)
ax3 = axes[2]
env.plot_field(firing_rate, ax=ax3, cmap="hot")
ax3.set_title("Place Field (Hz)")

plt.tight_layout()
plt.show()

print(f"Spatial information: {result.spatial_information():.3f} bits/spike")
```

### Key Considerations

**Bin Size Selection**:
- Too large: Lose spatial resolution
- Too small: Insufficient occupancy, noisy firing rates
- Rule of thumb: 2-5 cm for rat open field (100x100 cm arena)

**Occupancy Threshold** (`min_occupancy`):
- `diffusion_kde`, `gaussian_kde` and `binned` all threshold raw per-bin
  occupancy in **seconds**, after applying the analysis windows, tracking-gap
  rules and any speed filter. Recover this mask with
  `result.occupancy < min_occupancy`.
- The default `0.0` disables this mask. It does not guarantee a finite rate
  in every bin: some bins may have no occupancy within the estimator's reach.
- Masked rates remain `NaN` unless `fill_value` is supplied. A zero-filled map
  still needs its occupancy map to distinguish missing support from silence.
- `bin_count_threshold` counts tracking samples during environment creation;
  it is separate from time observed during the selected analysis.
- `method="glm"` uses its penalty/rank controls and rejects `min_occupancy`.
  See the [worked occupancy check](spike-field-primitives.md#checking-the-occupancy-mask).

**Smoothing**:
- `"diffusion_kde"` (default) is boundary-aware and works on any graph layout
- `"gaussian_kde"` gives comparable results in the interior of regular
  rectangular grids (`diffusion_kde` is boundary-aware, so they differ near
  boundaries)
- Increase `bandwidth` for noisier data or coarser bins

## Workflow 2: Bayesian Decoding (one call)

Reconstruct an animal's position from population spike trains in a single call.

### Overview

**Goal**: Decode position over time from a population of place cells

**Steps**: Simulate or load `(env, spike_times, times, positions)` → `decode_session(...)` → inspect `result.map_position` / plot

`decode_session` is the one-call golden path: it builds the encoding models
(place fields), bins the spikes onto a regular time grid, and runs the Bayesian
decoder for you. The whole encode → bin → decode pipeline fits in one line.

### Complete Example

<!-- docs-test: run -->
```python
import numpy as np
import matplotlib.pyplot as plt

from neurospatial import Environment
from neurospatial.decoding import decode_session, decoding_error
from neurospatial.simulation import (
    PlaceCellModel,
    generate_population_spikes,
    simulate_trajectory_ou,
)

# Step 1: Simulate a population of place cells on a 100 cm linear track.
# (In a real analysis, load your env, spike_times, times, and positions here.)
env = Environment.from_samples(
    np.linspace(0.0, 100.0, 51).reshape(-1, 1), bin_size=2.0
)
env = env.with_metadata(units="cm")
positions, times = simulate_trajectory_ou(
    env, duration=600.0, dt=0.02, speed_mean=15.0, seed=0, speed_units="cm"
)
cells = [
    PlaceCellModel(env, center=np.array([c]), width=10.0, max_rate=20.0, seed=i)
    for i, c in enumerate(np.linspace(5.0, 95.0, 25))
]
spike_times = generate_population_spikes(
    cells, times, positions, seed=0, show_progress=False
)

# Step 2: Decode position in a single call (encode -> bin -> decode).
result = decode_session(env, spike_times, times, positions, dt=0.1)

print(f"Decoded {result.posterior.shape[0]} time bins over {env.n_bins} bins")

# Step 3: Inspect in-sample error: these spikes also fitted the rate maps.
actual = np.interp(result.times, times, positions[:, 0]).reshape(-1, 1)
median_err = np.nanmedian(decoding_error(env, result.map_position, actual))
print(f"Median in-sample decoding error: {median_err:.1f} cm")

# Step 4: Plot decoded vs. actual position.
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(result.times, actual[:, 0], label="Actual", linewidth=1)
ax.plot(result.times, result.map_position[:, 0], label="Decoded (MAP)", linewidth=1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (cm)")
ax.set_title("Bayesian decoding with decode_session")
ax.legend()
plt.tight_layout()
plt.show()
```

`result` is a `DecodingResult` with `.posterior` (n_time, n_bins),
`.map_position` (n_time, n_dims), `.mean_position`, `.posterior_entropy`, and
`.times`. See [`decoding_error`](../api/neurospatial/decoding/index.md) and
related helpers for accuracy metrics.

This example fits the rate maps and decodes the **same recording**. Its error
is in-sample and does not measure generalization. For held-out evaluation,
fit maps using training epochs, then predict and score disjoint trials or time
blocks with `BayesianDecoder`. Select parameters using training/validation
data before evaluating the test block. The
[rate-map reuse example](#reusing-rate-maps-on-a-recording-with-gaps) shows the
separate training and prediction epochs. Posterior entropy describes spread
under the chosen model; it is not a calibrated confidence level.

### Overlaying actual position on a posterior

`DecodingResult.plot(show_map=True)` uses **spatial-bin indices** on its y-axis.
Actual positions and `result.map_position` carry physical coordinates, such as
centimeters. Pass `true_times` and `true_positions` to overlay the actual track;
the plot aligns timestamps and converts coordinates internally. Keep physical
coordinates for error metrics and position-versus-time plots.

On a uniform continuous decoder clock, image columns center on the returned
timestamps: image edges extend half a bin past the first and last center.
Two timestamps use their actual spacing; one timestamp uses a 1-second
display width centered on that timestamp. Explicit `extent=` overrides these
edges. Posterior values, MAP positions and timestamps are unchanged.

On a continuous decoder clock, the plot's x-axis uses seconds. Across recording
gaps it uses time-bin indices and marks the breaks with dashed lines. The MAP
and actual overlays follow the same display coordinates and break at gaps.
Plotting and `error_against` share gap-aware truth alignment: `max_gap` is in
seconds, missing positions break observation runs, and truth is unavailable
outside observed intervals. Neither interpolation across dropouts nor endpoint
extrapolation creates observations.

This exact four-row posterior uses 5 cm bins. It represents perfect decoding
for both clocks below, so the actual and MAP overlays must coincide. The
gapped example contains no decoder rows between 10.15 and 20.05 seconds.

<!-- docs-test: run -->
```python
import matplotlib.pyplot as plt
import numpy as np

from neurospatial import Environment
from neurospatial.decoding import DecodingResult, median_decoding_error

env = Environment.from_samples(
    np.linspace(0.0, 100.0, 21)[:, None], bin_size=5.0, units="cm"
)
actual = np.array([[20.0], [80.0], [40.0], [60.0]])
actual_bins = env.bin_at(actual)
posterior = np.zeros((len(actual), env.n_bins))
posterior[np.arange(len(actual)), actual_bins] = 1.0
clocks = {
    "Continuous": 10.05 + np.arange(4) * 0.1,
    "Gapped": np.array([10.05, 10.15, 20.05, 20.15]),
}
fig, axes = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
for ax, (name, decoder_times) in zip(axes, clocks.items(), strict=True):
    result = DecodingResult(posterior, env, decoder_times)
    result.plot(
        ax=ax, show_map=True, colorbar=True,
        true_times=decoder_times, true_positions=actual, max_gap=0.5,
    )
    np.testing.assert_allclose(
        result.error_against(decoder_times, actual, max_gap=0.5), 0.0
    )
    assert median_decoding_error(env, result.map_position, actual) == 0.0
    ax.set_title(name)
    ax.legend()
plt.show()
```

### Long sessions / thousands of units: stream the decode

`decode_session` materializes the full `(n_time, n_bins)` posterior. For long
recordings or large populations that array can be too big to hold in memory.
Use the memory-safe summary decoder instead:

<!-- docs-test: run setup=workflows_decode_session_summary_streaming -->
```python
from neurospatial.decoding import decode_session_summary

summary = decode_session_summary(env, spike_times, times, positions, dt=0.1)
df = summary.to_dataframe()  # per-time MAP / mean / entropy / peak
print(summary.summary())     # headline scalar metrics
print(summary.map_position)  # (n_time, n_dims) MAP estimate
```

`decode_session_summary` streams the decode in time chunks and returns a
`DecodingSummary` carrying per-time MAP / mean / entropy / peak via
`.to_dataframe()` and `.summary()` (plus `.map_position`) — **without ever
materializing the full posterior**. If you already have binned spike counts and
encoding models, `decode_position_summary(env, spike_counts, encoding_models, *,
dt, time_chunk=...)` is the array-level equivalent.

### When to reach for the manual path

`decode_session` covers the common case. When you need custom control — passing
your own `encoding_models`, reusing fitted place fields across sessions, or
inspecting the binned spike counts — use the manual three-call path
(`compute_spatial_rates` → `bin_spikes_in_time` → `decode_position`). That
walk-through, plus trajectory analysis and a time-order shuffle test on a
synthetic sequence, is covered in
[the decoding tutorial](../examples/06_position_decoding.ipynb).

### Reusing rate maps on a recording with gaps

`decode_session` requires tracking and computes its own maps. For maps already
computed on a training epoch, use `BayesianDecoder.from_rates(rates)`; prediction
takes spike times and a plain timestamp array. An explicit array handoff uses
`rates.firing_rates` with binned counts and `prior=rates.spatial_support * env.bin_sizes` to retain the training mask. This example decodes the second recording
run through both routes. It bins only that run and uses each decoder's returned
timestamps, so the 10–20 s pause is never represented by a decode bin.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment, compute_spatial_rates, decode_position
from neurospatial.decoding import BayesianDecoder, bin_spikes_in_time

times = np.r_[np.arange(300) / 30, 20 + np.arange(300) / 30]
phase = np.arange(600) / 30
positions = 10 + 5 * np.c_[np.sin(phase), np.cos(phase)]
env = Environment.from_samples(positions, bin_size=2.0)
spike_times = [times[::10], times[::15]]
recording_windows = np.array([[0.0, 10.0], [20.0, 30.0]])
train, test = (0.0, 10.0), (20.0, 30.0)
dt = 0.1

rates = compute_spatial_rates(
    env, spike_times, times, positions, epochs=train,
    spike_window=recording_windows, fill_value=0.0, unit_ids=[101, 202],
)
decoder = BayesianDecoder.from_rates(rates, dt=dt)
result = decoder.predict(
    spike_times, times, epochs=test, spike_window=recording_windows,
)

# Clip the analysis epoch to the observed run, then bin that epoch alone.
counts, centers = bin_spikes_in_time(
    spike_times, dt=dt, epochs=(test[0], min(test[1], times[-1])),
)
array_result = decode_position(env, counts, rates.firing_rates, dt=dt, times=centers,
    prior=rates.spatial_support * env.bin_sizes)
np.testing.assert_array_equal(result.times, array_result.times)
np.testing.assert_array_equal(result.posterior, array_result.posterior)
assert np.all((result.times >= 20.0) & (result.times < 30.0))
print("Decode bins:", len(result.times), "unit order:", rates.unit_ids.tolist())
```

Use `BayesianDecoder(env).fit(spike_times, times, positions, unit_ids=...)` when
the decoder should build the maps. A labelled spike group and explicit `unit_ids`
must agree in order and value. Prediction matches labels only when both inputs
carry caller-supplied identity; generated row numbers retain positional pairing.
The training spike window carried on the decoder is provenance. Supply the
prediction recording's observation windows explicitly to `predict`.

The count-array route has already applied its time selection upstream. For
multiple recording runs, bin each run independently and retain the corresponding
centers; a single start/stop grid would span the pauses.

### Recordings joined on one clock

When several recordings are concatenated with no pause on the clock, `max_gap`
cannot see where one ends and the next begins. Declare each join with
`break_before`, True at the first frame of each later recording. Rate maps,
decoders and `error_against` take the same array: the interval across the join
adds no occupancy or spikes, no decode bin spans it, and ground truth is not
interpolated across it.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment, compute_spatial_rates
from neurospatial.decoding import decode_session

# Two 10 s recordings joined on one clock with no pause between them.
times = np.arange(600) / 30
recording_id = np.repeat([0, 1], 300)
positions = 10 + 5 * np.c_[np.sin(times), np.cos(times)]
positions[300:] = positions[300:][::-1]  # the animal is put back elsewhere
env = Environment.from_samples(positions, bin_size=2.0)
spike_times = [times[::10], times[::15]]
new_recording = np.r_[False, recording_id[1:] != recording_id[:-1]]

rates = compute_spatial_rates(
    env, spike_times, times, positions, break_before=new_recording
)
joined = compute_spatial_rates(env, spike_times, times, positions)
print("Break at", rates.analysis_settings["break_times"], "s")
print(f"Occupancy removed: {joined.occupancy.sum() - rates.occupancy.sum():.4f} s")

dt = 0.1
result = decode_session(
    env, spike_times, times, positions, dt=dt, break_before=new_recording
)
start = times[new_recording][0]
first = result.times < start
print(f"Decode bins: {first.sum()} in recording 1, {(~first).sum()} in recording 2")
assert np.all(result.times[first] + dt / 2 <= times[299] + 1e-9)
assert np.all(result.times[~first] - dt / 2 >= start - 1e-9)
errors = result.error_against(times, positions, break_before=new_recording)
```

Only the 1/30 s step between the recordings is removed, and the result records
the break time in `analysis_settings["break_times"]`. Without `break_before`,
that step is an ordinary sampling interval: its time and spikes count toward
the bin where the first recording ended, the decode bin from 9.9 to 10.0 s
covers it, and a speed derived for `min_speed` reads the jump between the two
places as running. With your own
data, build the array from your recording or epoch IDs,
`np.r_[False, recording_id[1:] != recording_id[:-1]]`, and pass it to every
call. `BayesianDecoder.fit`, `predict`, `predict_summary` and `score` take it
too.

## Workflow 3: Region-Based Analysis

Analyzing behavior across experimentally-defined spatial zones.

### Overview

**Goal**: Compare neural activity and behavior across different regions of the environment

**Steps**: Define regions → Compute metrics per region → Statistical comparison

### Complete Example

```python
from neurospatial import Regions

from neurospatial import Environment
from shapely.geometry import Point
import numpy as np

# Create environment from position data
env = Environment.from_samples(position_data, bin_size=3.0)

# Define experimental regions
# Center zone (15 cm radius circle)
center_point = Point(50.0, 50.0)  # Arena center
region_builder = Regions(env.regions.values())
region_builder.add("Center", polygon=center_point.buffer(15.0))

# Corner zones (10x10 cm squares)
corners = {
    "TopLeft": [(0, 90), (10, 90), (10, 100), (0, 100)],
    "TopRight": [(90, 90), (100, 90), (100, 100), (90, 100)],
    "BottomLeft": [(0, 0), (10, 0), (10, 10), (0, 10)],
    "BottomRight": [(90, 0), (100, 0), (100, 10), (90, 10)],
}

for name, coords in corners.items():
    from shapely.geometry import Polygon
    region_builder.add(name, polygon=Polygon(coords))
env = env.with_regions(region_builder)

# Find which bins belong to each region
region_bins = {}
for region_name in env.regions.list_names():
    region_polygon = env.regions[region_name].polygon
    bins_in_region = []

    for bin_idx in range(env.n_bins):
        bin_point = Point(env.bin_centers[bin_idx])
        if region_polygon.contains(bin_point):
            bins_in_region.append(bin_idx)

    region_bins[region_name] = np.array(bins_in_region)
    print(f"{region_name}: {len(bins_in_region)} bins")

# Compute occupancy per region
position_bins = env.bin_at(position_data)
sampling_rate = 30.0  # Hz

region_occupancy = {}
for region_name, bins in region_bins.items():
    time_in_region = np.sum(np.isin(position_bins, bins)) / sampling_rate
    region_occupancy[region_name] = time_in_region
    print(f"Time in {region_name}: {time_in_region:.2f} seconds")

# Compute firing rate per region
spike_positions = interpolate_position(position_data, spike_times)
spike_bins = env.bin_at(spike_positions)

region_firing_rates = {}
for region_name, bins in region_bins.items():
    spikes_in_region = np.sum(np.isin(spike_bins, bins))
    time_in_region = region_occupancy[region_name]

    if time_in_region > 0.5:  # Require 0.5s minimum
        firing_rate = spikes_in_region / time_in_region
        region_firing_rates[region_name] = firing_rate
    else:
        region_firing_rates[region_name] = np.nan

    print(f"{region_name} firing rate: {firing_rate:.2f} Hz")

# Statistical comparison
# Example: Is firing rate higher in center vs. corners?
center_rate = region_firing_rates["Center"]
corner_rates = [region_firing_rates[name] for name in corners.keys()]
corner_rates = [r for r in corner_rates if not np.isnan(r)]

print(f"\nCenter: {center_rate:.2f} Hz")
print(f"Corners: {np.mean(corner_rates):.2f} ± {np.std(corner_rates):.2f} Hz")

# Visualize regions
fig, ax = plt.subplots(figsize=(8, 8))
env.plot(ax=ax)

# Color-code regions
colors = plt.cm.Set3(np.linspace(0, 1, len(env.regions)))
for idx, region_name in enumerate(env.regions.list_names()):
    region = env.regions[region_name]
    if region.polygon:
        x, y = region.polygon.exterior.xy
        ax.fill(x, y, alpha=0.3, color=colors[idx], label=region_name)

ax.legend()
ax.set_title('Experimental Regions')
plt.show()
```

## Workflow 4: Multi-Session Alignment

Comparing a cell's place field across recording sessions in the same box.

### Overview

**Goal**: Measure how stable a place field is from one day to the next.

**Steps**: Build one environment for the box → Compute each day's rate map on
it → Correlate the maps over bins visited on both days

Two rate maps can be compared bin by bin only when bin `i` is the same place
in both. `Environment.from_samples` places its grid from the samples it is
given, so two sessions built separately get slightly different bins even in
the same box. Build the environment once from the box's geometry (or from
both sessions' positions together) and compute every session's map on it.
If you have maps on different bins already, move one onto the other's bins
with `resample_field`, which copies each rate from the nearest bin. Rates stay
in Hz, and bins outside the source environment stay NaN. Do not use
`map_probabilities` for rate maps: it sums probability mass, so it changes the
rates.

If the camera moved between sessions, first put both sessions' positions in
one frame (see [Alignment and transforms](alignment.md)).

### Complete Example

<!-- docs-test: run -->
```python
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box

from neurospatial import Environment, compute_spatial_rate
from neurospatial.encoding import compare_spatial_rates
from neurospatial.ops import resample_field
from neurospatial.simulation import (
    PlaceCellModel,
    generate_poisson_spikes,
    simulate_trajectory_ou,
)

# The same 100 x 100 cm box on both days. Building the environment from the
# box, not from each day's samples, gives both days the same bins.
env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0, units="cm")
place_cell = PlaceCellModel(env, center=[30.0, 60.0], width=8.0, max_rate=15.0)


def record_session(seed):
    """One 10-minute session: times, positions and the place cell's spikes."""
    positions, times = simulate_trajectory_ou(
        env, duration=600.0, dt=1 / 30, speed_mean=15.0, speed_units="cm", seed=seed
    )
    # Seed the spikes differently from the trajectory.
    spikes = generate_poisson_spikes(
        place_cell.firing_rate(positions), times, seed=seed + 100
    )
    return times, positions, spikes


times1, positions1, spikes1 = record_session(seed=1)
times2, positions2, spikes2 = record_session(seed=2)

options = dict(method="diffusion_kde", bandwidth=5.0, min_occupancy=0.5)
day1 = compute_spatial_rate(env, spikes1, times1, positions1, **options)
day2 = compute_spatial_rate(env, spikes2, times2, positions2, **options)

# Correlate the maps over bins with at least 0.5 s of occupancy on both days;
# the others are NaN, which means "not measured", not 0 Hz.
day_to_day = compare_spatial_rates(day1, day2, min_common_bins=3)
stability = float(day_to_day.correlation[0])
print(
    f"Day-to-day map correlation: {stability:.2f} "
    f"over {day_to_day.n_common_bins[0]} bins"
)
assert stability > 0.9

# A day-2 map someone already computed on bins built from that day's samples:
# move it onto the shared bins before comparing.
env_day2 = Environment.from_samples(positions2, bin_size=5.0, units="cm")
day2_own_bins = compute_spatial_rate(env_day2, spikes2, times2, positions2, **options)
day2_on_shared_bins = resample_field(day2_own_bins.firing_rate, env_day2, env)

# The rates are copied, not rescaled: the peak is the same number of Hz.
assert np.isclose(
    np.nanmax(day2_on_shared_bins), np.nanmax(day2_own_bins.firing_rate)
)
both_resampled = np.isfinite(day1.firing_rate) & np.isfinite(day2_on_shared_bins)
resampled_stability = np.corrcoef(
    day1.firing_rate[both_resampled], day2_on_shared_bins[both_resampled]
)[0, 1]
print(f"Correlation after resampling: {resampled_stability:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
vmax = np.nanmax([day1.firing_rate, day2.firing_rate])
for ax, rate, title in [
    (axes[0], day1.firing_rate, "Day 1"),
    (axes[1], day2.firing_rate, "Day 2"),
    (axes[2], day2_on_shared_bins, "Day 2, resampled from its own bins"),
]:
    env.plot_field(rate, ax=ax, vmin=0, vmax=vmax, colorbar_label="Rate (Hz)")
    ax.set_title(title)
fig.suptitle(f"Map correlation r = {stability:.2f}")
plt.tight_layout()
plt.show()
```

The two days' maps put the field in the same place, and the correlation over
bins visited on both days is close to 1. Gray bins had less than 0.5 s of
occupancy on that day and carry no rate. A cell that remapped would give a
low or negative correlation. `compare_spatial_rates` refuses two maps on
different bins, so the resampled map is compared by hand over the bins finite
in both maps. Resampling the day-2 map from its own bins gives nearly the same
answer; the small difference comes from bins whose centers are not exactly
aligned.

## Workflow 5: Track Linearization

Analyzing maze experiments with branching structures.

### Overview

**Goal**: Convert 2D maze positions to 1D linearized coordinates for sequential analysis

**Steps**: Define track graph → Create 1D environment → Map positions → Analyze

See the complete example in [examples/03_trajectories_and_behavior.ipynb](../examples/03_trajectories_and_behavior.ipynb).

### Direction-specific fields on a graph track

Linearization supplies coordinates along track geometry. On a single-edge
track, a return traversal reuses the same coordinates and bins; it does not
automatically become a separate directional map. For a branched graph,
assignment also depends on the graph and projection settings.

Use explicit trial labels to condition firing on direction. This complete
60-second example plants a field at 60 cm outbound and 40 cm inbound, then
checks each recovered peak against its own ground truth within one 5 cm bin.
Replace the synthetic positions and spikes with your recorded arrays for an
experiment. The two panels share the same environment.

<!-- docs-test: run -->
```python
from neurospatial import Regions

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import Polygon

from neurospatial import Environment
from neurospatial.behavior import goal_pair_direction_labels, segment_trials
from neurospatial.encoding import compute_directional_place_fields
from neurospatial.simulation import generate_poisson_spikes

# A 100 cm track: graph coordinates describe geometry, not running direction.
graph = nx.Graph()
graph.add_node(0, pos=(0.0, 0.0))
graph.add_node(1, pos=(100.0, 0.0))
graph.add_edge(0, 1, distance=100.0)
env = Environment.from_graph(
    graph, edge_order=[(0, 1)], edge_spacing=0.0, bin_size=5.0, track_width=10.0
)
env = env.with_metadata(units="cm")
repeated = np.array([[25.0, 0.0], [50.0, 0.0], [75.0, 0.0], [50.0, 0.0], [25.0, 0.0]])
np.testing.assert_allclose(env.to_linear(repeated), repeated[:, 0])
repeated_bins = env.bin_at(repeated)
assert repeated_bins[0] == repeated_bins[-1] and repeated_bins[1] == repeated_bins[-2]

# Six out-and-back cycles in 60 seconds, with different planted field centers.
times = np.arange(0.0, 60.0, 0.05)
phase = (times % 10.0) / 10.0
x = 10.0 + 80.0 * (1.0 - np.abs(2.0 * phase - 1.0))
positions = np.column_stack([x, np.zeros_like(x)])
planted_center = np.where(phase < 0.5, 60.0, 40.0)
intensity = 0.5 + 25.0 * np.exp(-0.5 * ((x - planted_center) / 10.0) ** 2)
spikes = generate_poisson_spikes(intensity, times, seed=7)

# Explicit trials supply the direction labels; both maps use the same graph.
region_builder = Regions(env.regions.values())
region_builder.add("home", polygon=Polygon([(-1, -5), (15, -5), (15, 5), (-1, 5)]))
region_builder.add("goal", polygon=Polygon([(85, -5), (101, -5), (101, 5), (85, 5)]))
env = env.with_regions(region_builder)
position_bins = env.bin_sequence(times, positions, dedup=False)
outbound = segment_trials(position_bins, times, env, start_region="home", end_regions=["goal"])
inbound = segment_trials(position_bins, times, env, start_region="goal", end_regions=["home"])
labels = goal_pair_direction_labels(times, outbound + inbound)
fields = compute_directional_place_fields(env, spikes, times, positions, labels)

fig, axes = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
for ax, label, truth in zip(axes, ["home→goal", "goal→home"], [60.0, 40.0], strict=True):
    field = fields.firing_rates[label]
    recovered = env.bin_centers[np.nanargmax(field), 0]
    assert abs(recovered - truth) <= 5.0  # Recover each planted center within one bin.
    env.plot_field(field, ax=ax, colorbar_label="Firing rate (Hz)")
    ax.set_title(f"{label}: peak {recovered:.1f} cm")
    print(f"{label}: planted {truth:.1f} cm, recovered {recovered:.1f} cm")
plt.show()
```

## Workflow 6: Shared Units for Decoding and Population Statistics

Simulate a labeled population, estimate spatial maps from four raw arrays,
and use one unit selection for both downstream branches. The earlier period
is the baseline control, the middle period is the encoding/statistical
reference (template), and the last period is the match. These are synthetic
navigation periods, not a claim of sleep replay. All models here are place
cells; covariance patterns can reflect their shared spatial drive.

| Branch | Inputs | Output |
| --- | --- | --- |
| Position decoding | Template rate maps and match spike counts, in the same unit order | Posterior over spatial bins |
| Population statistics | Control, template and match count matrices, in the same unit order | Patterns, standardized activations and EV/controlled REV effect sizes |

A posterior is not an input to assembly or EV analysis. Remove constant units
once across all periods, then apply the same mask to spike trains, map rows,
count columns and unit IDs. The silent unit below demonstrates the common
nonzero-variance selection. Do not independently filter each period: equal
column counts alone would not ensure that they represent the same neurons.

<!-- docs-test: run -->
```python
import numpy as np
from shapely.geometry import box

from neurospatial import Environment, compute_spatial_rates, decode_position
from neurospatial.decoding import (
    assembly_activation,
    bin_spikes_in_time,
    detect_assemblies,
    explained_variance_reactivation,
    pairwise_correlations,
    reactivation_strength,
)
from neurospatial.simulation import (
    PlaceCellModel,
    generate_poisson_spikes,
    simulate_trajectory_ou,
)

env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0)
env = env.with_metadata(units="cm")
positions, times = simulate_trajectory_ou(
    env, duration=180.0, dt=1 / 30, speed_units="cm", seed=7
)
centers = np.array(
    [
        [30, 35],
        [32, 37],
        [34, 33],
        [70, 70],
        [72, 68],
        [68, 72],
        [50, 50],
        [53, 50],
        [50, 53],
    ]
)
trains = [
    generate_poisson_spikes(
        PlaceCellModel(
            env, center=center, width=12, max_rate=15, baseline_rate=0.5
        ).firing_rate(positions, times),
        times,
        seed=10 + i,
    )
    for i, center in enumerate(centers)
]
trains.append(np.empty(0, dtype=np.float64))
unit_ids = np.r_[np.arange(101, 110), 999]
dt = 0.1
windows = {"control": (0.0, 60.0), "template": (60.0, 120.0), "match": (120.0, 180.0)}
counts, centers_by_period = {}, {}
for period, (start, stop) in windows.items():
    counts[period], centers_by_period[period] = bin_spikes_in_time(
        trains, dt=dt, t_start=start, t_stop=stop
    )
# Select once across every period, preserving original unit order.
keep = np.logical_and.reduce([np.var(matrix, axis=0) > 0 for matrix in counts.values()])
selected_ids = unit_ids[keep]
selected_trains = [
    train for train, retained in zip(trains, keep, strict=True) if retained
]
counts = {period: matrix[:, keep] for period, matrix in counts.items()}
assert len(selected_ids) >= 3 and 999 not in selected_ids
# Four-array encoding on the template; metadata carries the selected IDs.
rates = compute_spatial_rates(
    env,
    selected_trains,
    times,
    positions,
    unit_ids=selected_ids,
    epochs=[windows["template"]],
    spike_window=[(0.0, 180.0)],
    bandwidth=5.0,
    fill_value=0.0,
)
np.testing.assert_array_equal(rates.unit_ids, selected_ids)
# Branch 1: maps + count columns in the same selected order.
decoded = decode_position(
    env, counts["match"], rates, dt=dt, times=centers_by_period["match"]
)
assert decoded.posterior.shape == (len(centers_by_period["match"]), env.n_bins)
np.testing.assert_allclose(decoded.posterior.sum(axis=1), 1.0)
# Branch 2: counts, not the decoded posterior.
assemblies = detect_assemblies(
    counts["template"], algorithm="pca", unit_ids=selected_ids, rng=0
)
print("Retained unit IDs:", selected_ids.tolist())
print("Dimensions above Marchenko-Pastur reference:", assemblies.n_significant)
for pattern in assemblies.patterns:
    print("Thresholded core member IDs:", pattern.member_unit_ids.tolist())
    activation = assembly_activation(counts["match"], pattern, unit_ids=selected_ids)
    assert len(activation) == len(centers_by_period["match"])
    strength = reactivation_strength(
        counts["template"],
        counts["match"],
        pattern,
        template_unit_ids=selected_ids,
        match_unit_ids=selected_ids,
    )
    print("Standardized activation mean/SD:", activation.mean(), activation.std())
    print("Activation magnitude ratio (effect size):", strength)
correlations = {
    period: pairwise_correlations(matrix) for period, matrix in counts.items()
}
reactivation = explained_variance_reactivation(
    correlations["template"],
    correlations["match"],
    control_correlations=correlations["control"],
    template_unit_ids=selected_ids,
    match_unit_ids=selected_ids,
    control_unit_ids=selected_ids,
)
print(
    "EV, controlled REV (effect sizes):",
    reactivation.explained_variance,
    reactivation.reversed_ev,
)
assert np.isfinite(reactivation.explained_variance) and np.isfinite(
    reactivation.reversed_ev
)
```

`n_significant` counts dimensions above a random-matrix reference, not neurons
with calibrated p-values. If no dimension passes, the algorithm may still
return an exploratory pattern. Pattern `member_indices` are positions in the
selected unit order; `member_unit_ids` gives the original labels, and passing
`unit_ids=` to `assembly_activation`, or `template_unit_ids=` and
`match_unit_ids=` to `reactivation_strength`, matches the count columns to the
pattern's weights by label. The same per-period labels given to
`explained_variance_reactivation` match each pair correlation to the same two
cells in every period. A pattern can have no core members at the default absolute-weight
z-score cutoff, even when its dimension passes the reference.

`assembly_activation` standardizes the projection within each period,
removing absolute projection scale; its output does not give a tail
probability. In contrast, `reactivation_strength` normalizes both count
matrices with the template's neuron means/standard deviations and projects
onto the same pattern without separate projection standardization. Its ratio
preserves relative magnitude on that shared template scale, but remains an
effect size rather than a calibrated tail probability.

Controlled EV is `r(template, match | control)^2`, while controlled REV
is `r(template, control | match)^2`, swapping the pre- and post-behavior
periods. EV > REV is an effect-size comparison,
not calibrated reactivation significance. Report control choice, preprocessing,
unit selection and a separately justified null when a significance claim is
needed.

## Workflow 7: One Event Cohort Across a Gapped Recording

A PSTH keeps events whose entire peri-event window fits both the spike
recording and the analysis epochs. Its ``event_indices`` identifies retained
rows in your original event input. Use those rows for the raster, positioned
events and regressors so every view describes the same cohort. Manually
constructed results use ``event_indices=None`` when provenance is unavailable.

<!-- docs-test: run -->
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neurospatial.events import (
    add_positions, peri_event_histogram, plot_peri_event_response,
    event_count_in_window, event_indicator,
)

times = np.r_[np.arange(100) / 10, 20 + np.arange(100) / 10]
positions = np.c_[times, np.zeros(len(times))]
spike_times = np.array([4.0, 5.0, 6.0, 24.0, 25.0, 26.0])
events = pd.DataFrame({
    "event_id": ["reward-a", "reward-b", "reward-c", "reward-d"],
    "timestamp": [5.0, 9.5, 25.0, 29.5],
})
spike_window = np.array([[0.0, 10.0], [20.0, 30.0]])
epochs = np.array([[1.0, 9.0], [21.0, 29.0]])
window = (-1.0, 1.0)
psth = peri_event_histogram(
    spike_times, events["timestamp"].to_numpy(), window=window, bin_size=0.2,
    epochs=epochs, spike_window=spike_window,
)
selected = events.iloc[psth.event_indices].copy()
events["retained"] = False
events.loc[selected.index, "retained"] = True
event_times = selected["timestamp"].to_numpy()

raster_ax, rate_ax = plot_peri_event_response(
    spike_times, events["timestamp"].to_numpy(), psth,
)
raster_ax.set_yticks(np.arange(len(event_times)), selected["event_id"])

regressors = pd.DataFrame({
    "timestamp": times,
    "event_count": event_count_in_window(times, event_times, window=window),
    "event_present": event_indicator(times, event_times, window=window),
})
positioned = add_positions(selected, times=times, positions=positions, epochs=epochs)
table = events.join(positioned[["x", "y"]])
assert psth.n_events == len(positioned) == 2
assert selected["event_id"].tolist() == ["reward-a", "reward-c"]
assert regressors["event_count"].dtype == np.int64
assert regressors["event_present"].dtype == np.bool_
print(table.to_string(index=False))

# Raster, PSTH and regressors all use half-open [start, stop) windows.
assert psth.histogram.sum() == 2.0  # Average of two spikes per event.
assert event_count_in_window(
    np.array([4.0, 6.0]), np.array([5.0]), window=window,
).tolist() == [0, 1]  # [3, 5) and [5, 7): the event at 5 counts once.
plt.close(raster_ax.figure)
```

`plot_peri_event_response` uses the PSTH's retained event indices and actual
complete-bin range for both panels. Spikes on the right edge and in a trailing
partial bin are excluded from both views. Supply the original event array and
the spike train belonging to the selected unit. The low-level
`align_spikes_to_events` uses the same half-open `[start, stop)` window as
the histogram. Event-count and indicator windows are half-open too, so
neighboring sample windows never count the same boundary event twice. Regressors
describe the selected events at each sample; restrict their sample rows to your
modeling epochs when fitting a model.

## Common Patterns

### Pattern: Handling Edge Cases

```python
# Always check for valid bins
bin_indices = env.bin_at(positions)
valid = bin_indices != -1  # -1 indicates point outside environment

# Use only valid data
valid_positions = positions[valid]
valid_bins = bin_indices[valid]

# Or handle invalid gracefully
firing_rate = np.full(env.n_bins, np.nan)
valid_occupancy = occupancy_time > min_threshold
firing_rate[valid_occupancy] = spike_counts[valid_occupancy] / occupancy_time[valid_occupancy]
```

### Pattern: Batch Processing

<!-- docs-test: run setup=workflows_batch_processing_compute_spatial_rates -->
```python
from neurospatial.encoding import compute_spatial_rates

# Process multiple units efficiently with the batch API.
spike_trains_by_unit_id = load_all_neurons()
unit_ids = list(spike_trains_by_unit_id.keys())
spike_times = list(spike_trains_by_unit_id.values())

result = compute_spatial_rates(
    env, spike_times, times, positions, unit_ids=unit_ids
)
firing_rate_maps = result.firing_rates  # Shape: (n_units, n_bins)
```

`compute_spatial_rates` handles spike-to-position assignment, occupancy
normalization, and smoothing for the whole population in one call, returning a
`SpatialRatesResult` whose `unit_ids` line up with `firing_rates`. Prefer it
over a manual per-neuron loop.

### Pattern: Progressive Refinement

```python
# Start with coarse binning for quick overview
env_coarse = Environment.from_samples(positions, bin_size=10.0)
# ... analyze ...

# Refine in regions of interest
env_fine = Environment.from_samples(
    positions,
    bin_size=2.0,
    infer_active_bins=True,
    dilate=True
)
# ... detailed analysis ...
```

## See Also

- [Environment API](../api/neurospatial/environment/index.md): Complete method documentation
- [Regions Guide](regions.md): Working with ROIs
- [Example Notebooks](../examples/index.md): Interactive tutorials
